# Attention Sinks in Pythia

This repo is a small set of experiments I ran to understand attention sinks in Pythia, where they appear, whether the first few positions behave differently, and what happens during inference when I remove or preserve those positions.

I used **EleutherAI Pythia-160M** for most of the experiments. I kept the model in FP32 because I was getting NaNs in some attention tensors with FP16.

The experiments are split into four notebooks.

---

## 1. `01_pythia_attention_sinks.ipynb`

This was the first experiment. The goal was to check whether Pythia actually shows attention-sink behavior and where it appears across layers and heads.

I used a 192-token prompt and extracted the full self-attention tensors from all 12 layers and 12 attention heads.

I looked at:

- global attention across all layers and heads
- cumulative attention received by each token position
- normalized attention received per eligible causal query
- sink strength for the first four positions
- sink strength by layer and head
- the strongest individual sink head

### Main result

The first few positions received much more attention than the middle of the sequence.

The average normalized attention was:

```text
First 4 positions: 0.113128
Middle region:     0.005537
Ratio:             20.43x
```

So the early positions received about **20x more normalized attention** than the middle positions.

The strongest sink head I found was:

```text
Layer 9
Head 11

Mean attention to first 4 positions: 0.9997
Mean attention to position 0:        0.2373
```

That head was almost completely focused on the first few positions.

For that head, the largest cumulative attention went to:

```text
Position 1: 93.289
Position 2: 49.405
Position 0: 48.138
```

This made it clear that the sink behavior was real, but it was not necessarily only position 0. Some heads strongly preferred positions 1 and 2 as well.

Another thing I noticed is that sink behavior is very head-specific. Averaging everything together hides some extremely strong heads.

---

## 2. `02_pythia_position0_vs_1_2_3_attention_sinks.ipynb`

After the first notebook, I wanted to separate the first four positions instead of treating them as one sink region.

So here I measured attention to positions:

```text
0
1
2
3
```

individually across every layer and attention head.

The first tokens in this experiment were:

```text
0  <|endoftext|>
1  newline
2  newline
3  "Art"
```

### Main findings

Position 0 was very strong in some layers, but position 1 also became dominant in several middle layers.

The dominant early position by layer was approximately:

```text
Layers 0-2   -> position 0
Layers 3-9   -> mostly position 1
Layer 10     -> position 0
```

The strongest position-0 head was:

```text
Layer 10
Head 2

Mean attention to position 0: 0.6179
```

For that head, cumulative attention to the first four positions was:

```text
Position 0: 108.7469
Position 1:  26.8356
Position 2:  12.1974
Position 3:   0.0000
```

This was useful because it showed that the first four positions are not interchangeable.

The first position can behave like a very strong sink, but other early positions can also become sinks depending on the layer/head.

One important observation from this notebook is that **position 3 basically did not behave like a sink in the strongest position-0 head**.

---

## 3. `03_pythia_attention_sink_inference_ablation.ipynb`

This notebook moved from visualization to inference.

Instead of only asking where attention goes, I tested what happens when I change which tokens the model is allowed to attend to.

I used WikiText-2 with:

```text
16 sequences
192 tokens per sequence
```

I compared these policies:

```text
full
full_minus_pos0
full_minus_pos0_to_3

recent_only_64

sink1_recent63
sink2_recent62
sink3_recent61
sink4_recent60
```

The streaming policies all use the same total budget of 64 tokens.

For example:

```text
0 sinks + 64 recent
1 sink  + 63 recent
2 sinks + 62 recent
3 sinks + 61 recent
4 sinks + 60 recent
```

This was important because otherwise a model with more sink tokens would also have more total context.

### Perplexity results

```text
Policy                  NLL       PPL
------------------------------------------------
full                    3.9109    49.94
full_minus_pos0         3.9376    51.30
full_minus_pos0_to_3    3.9974    54.45

recent_only_64          4.4828    88.48
sink1_recent63          4.0479    57.28
sink2_recent62          4.0463    57.18
sink3_recent61          4.0441    57.06
sink4_recent60          4.0426    56.97
```

This was probably the strongest result in the project.

Using only the latest 64 tokens increased perplexity from about:

```text
49.94 -> 88.48
```

But keeping only one early sink token and 63 recent tokens reduced it to:

```text
57.28
```

Keeping 2, 3, or 4 sink positions improved it slightly more:

```text
2 sinks: 57.18
3 sinks: 57.06
4 sinks: 56.97
```

So preserving a tiny number of early positions recovered a large amount of the performance lost by a pure sliding window.

### Removing sinks from full attention

Removing only position 0 increased PPL:

```text
49.94 -> 51.30
```

Removing positions 0-3 increased it more:

```text
49.94 -> 54.45
```

So the early region is doing something useful even when the model still has access to the rest of the sequence.

### Replacing the first four tokens

I also replaced the first four tokens with different content.

Results:

```text
Replacement       Full PPL    Sink4 + Recent60 PPL
---------------------------------------------------
original           49.94       56.97
repeat BOS/EOS     63.64       71.43
random tokens      96.94      111.46
copy middle        53.07       60.58
shuffle first 4    55.25       63.11
```

This was interesting because random replacement was very damaging.

So in this setup, the behavior is not purely "any token placed in an early position becomes a useful sink." Token identity/content still matters.

Copying normal tokens from the middle was much less damaging than inserting random tokens, but it still performed worse than the original prefix.

### Long-generation experiment

I also used a 360-token prompt with a 64-token attention budget and generated 150 new tokens.

The full model, recent-only model, and sink-preserving models produced visibly different trajectories.

The pure recent-window version became very repetitive.

The one-sink version stayed closer to the prompt for longer, although Pythia-160M with greedy decoding was still repetitive.

The 2-4 sink runs sometimes collapsed into repeated initials or repeated phrases.

I do not treat this generation output as the main metric because greedy decoding can diverge after one different token. The NLL/perplexity experiment above is a much cleaner comparison.

---

## 4. `04_attention_sinks_pythia.ipynb`

The last notebook looks at a different question:

**When do attention sinks appear during training, and do they become functionally important?**

I used multiple Pythia training checkpoints from step 0 to step 143000.

For each checkpoint I measured:

- language-model loss
- average attention to position 0
- maximum sink-head strength
- sink strength per layer

### Sink emergence during training

At the beginning, sink mass was basically the same as the uniform causal-attention baseline:

```text
Uniform baseline: 0.0201

Step 0:
loss      = 11.059
sink mass = 0.020
max head  = 0.022
```

The sink gradually appeared as training progressed.

Some checkpoints:

```text
Step      Loss     Sink mass    Max head
-----------------------------------------
0         11.059   0.020        0.022
512        6.581   0.014        0.023
2000       4.606   0.032        0.103
4000       4.218   0.072        0.305
8000       4.009   0.171        0.610
16000      3.862   0.221        0.665
32000      3.781   0.261        0.685
100000     3.728   0.291        0.737
143000     3.811   0.296        0.771
```

The strongest individual heads developed much stronger sink behavior than the global average.

### When individual layers crossed sink mass > 0.3

```text
Layer 0:  never
Layer 1:  never
Layer 2:  never
Layer 3:  never

Layer 4:  step 16000
Layer 5:  step 16000
Layer 6:  step 16000
Layer 7:  step 16000
Layer 8:  step 16000
Layer 9:  step 8000
Layer 10: step 16000
Layer 11: step 64000
```

Layer 9 developed the strong sink earliest.

This also shows that sink formation is mainly a middle/deeper-layer phenomenon in this model.

### Causal blocking experiment

I then used TransformerLens to directly block later tokens from attending to position 0.

I compared this against blocking position 8 as a control.

The largest effect happened in layer 4:

```text
Layer 4:
block sink position 0 -> dLoss +0.341 +/- 0.020
block control pos 8   -> dLoss +0.001 +/- 0.000
```

Other noticeable layers:

```text
Layer 5: +0.097
Layer 6: +0.074
Layer 7: +0.040
```

The early layers had much smaller effects.

This shows that a large attention value by itself is not enough to tell whether a sink is functionally important. Some layers matter much more than others when the sink is removed.

I also repeated the intervention over several training checkpoints:

```text
Step      Sink dLoss    Control dLoss
--------------------------------------
512       +0.008        +0.003
2000      +0.009        +0.010
8000      +0.022        +0.013
32000     +0.020        +0.015
143000    +0.016        +0.015
```

The causal effect is not a simple monotonic curve with training, so I would not claim that stronger sink mass automatically means a larger all-layer loss penalty.

---

# What I learned

The main things I got from these experiments:

1. **Attention sinks are clearly present in Pythia-160M.**

2. **They are highly head- and layer-specific.** Global averages hide some very strong sink heads.

3. **The first few positions are not equivalent.** Position 0 is very important in some heads, while position 1 dominates several layers.

4. **A pure sliding window hurts badly.** In my 64-token fixed-budget test, PPL went from about 49.94 with full attention to 88.48 with recent-only attention.

5. **Keeping a few sink positions recovers a large part of that loss.** With 4 sink tokens + 60 recent tokens, PPL dropped to 56.97.

6. **Token identity also matters.** Replacing the first four tokens with random vocabulary tokens made perplexity much worse.

7. **Sink behavior is learned during training.** Early checkpoints were close to uniform attention, while later checkpoints developed very strong sink heads.

8. **High sink attention and causal importance are related but not identical.** Blocking the sink in some layers, especially layer 4, caused a much larger loss increase than blocking an ordinary control position.

---

## Files

```text
01_pythia_attention_sinks.ipynb
    Basic attention-sink visualization and head/layer analysis.

02_pythia_position0_vs_1_2_3_attention_sinks.ipynb
    Separate analysis of positions 0, 1, 2 and 3.

03_pythia_attention_sink_inference_ablation_FIXED.ipynb
    Fixed-budget inference, sliding-window comparison, token replacement,
    perplexity and generation experiments.

04_attention_sinks_pythia.ipynb
    Sink emergence across training checkpoints and causal intervention.
```

---

## Setup

The notebooks use Pythia-160M, PyTorch, Hugging Face Transformers, and TransformerLens.

For the TransformerLens experiments I used compatible versions because the newer TransformerLens API changed the `HookedTransformer` interface.

```bash
pip install "transformer-lens==2.16.1" "transformers==4.51.3" datasets accelerate pandas matplotlib
```

Most experiments were run on GPU in FP32.

I intentionally used FP32 for the attention-analysis notebooks because FP16 produced NaNs in some attention tensors during earlier runs.

---

## Notes

These are exploratory experiments, not a full benchmark.

The generation examples are useful for seeing qualitative differences, but I trust the NLL/perplexity results more because greedy decoding can diverge after a very small logit change.

The next step would be to repeat the same fixed-budget experiments on larger models and longer evaluation sets, and then compare logical attention masking against actual KV-cache eviction.
