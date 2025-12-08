# Experiment 029 v2 Results: Spectral Bands + History Attention Ablation

## Executive Summary

**Spectral band decomposition is the key ingredient, not history attention alone.**

The AKIRA design (spectral bands + variable history) achieves the best perplexity, but the improvement comes primarily from spectral decomposition. History attention alone actually hurts performance on language modeling.

## Results

| Model | Loss | PPL | vs Baseline |
|-------|------|-----|-------------|
| 1. Baseline (Standard GPT-2) | 1.5637 | 4.78 | - |
| 2. + History (Uniform) | 1.5922 | 4.91 | **-2.89%** (worse) |
| 3. + Spectral Bands (No History) | 1.5388 | 4.66 | +2.45% |
| 4. + Spectral + Uniform History | 1.5236 | 4.59 | +3.93% |
| 5. + Spectral + Variable History (AKIRA) | 1.5210 | 4.58 | **+4.17%** |

## Key Findings

### 1. History Attention Alone Hurts Language Modeling

Adding uniform history attention to a standard transformer **decreases** performance by 2.89%. This confirms what we suspected: per-position history doesn't make sense for language where "position 5" is arbitrary across sequences.

### 2. Spectral Decomposition Helps

Causal spectral band decomposition improves perplexity by 2.45% even without history. Breaking the embedding into frequency bands and processing them separately provides useful inductive bias.

### 3. History + Spectral Works Better Than Either Alone

When combined with spectral bands, history attention becomes beneficial:
- Spectral alone: +2.45%
- Spectral + Uniform History: +3.93%
- Spectral + Variable History: +4.17%

**Why?** Spectral bands create meaningful "positions" (frequency bands) where per-band memory makes sense. Each band tracks its own temporal patterns.

### 4. Variable History Depths Provide Marginal Benefit

The Heisenberg-inspired variable depths (low freq = long history, high freq = short history) provide only 0.24% additional improvement over uniform history. The benefit is real but small.

## Interpretation

### Why Spectral Decomposition Helps

The causal spectral convolutions act as learnable frequency filters:
- Band 0 (kernel=9): Captures slow-changing patterns (topics, style)
- Band 1 (kernel=7): Captures medium patterns
- Band 2 (kernel=5): Captures local patterns
- Band 3 (kernel=3): Captures high-frequency details (syntax, word boundaries)

Processing these separately allows the model to handle different linguistic scales independently.

### Why History Only Helps With Spectral Bands

In raw token space, "position 5" has no consistent meaning across sequences. History of "what position 5 computed" is meaningless.

In spectral band space, each band represents a specific frequency component. History of "what this frequency band computed" has consistent meaning - you're tracking how slow vs fast patterns evolve.

### The Real Lesson

**The AKIRA architecture's value is in the spectral decomposition, not the history mechanism.**

For domains where position is physically meaningful (signals, video, sensors), history attention helps directly. For language, you need to first create meaningful positions via spectral decomposition.

## Comparison With Experiment 028

| Domain | History Only | With Spectral |
|--------|--------------|---------------|
| Language (this exp) | -2.89% (hurts) | +4.17% (helps) |
| Signal (Exp 028B) | +62% (helps directly) | N/A |

This confirms: per-position memory is natural for signals but requires preprocessing (spectral decomposition) to work for language.

## Configuration

```
embed_dim: 256
num_layers: 6
num_bands: 4
band_history_depths: [128, 64, 32, 16]  # Variable
max_history_uniform: 64
total_steps: 5000
Dataset: WikiText-2
```

## Training Times

| Model | Time |
|-------|------|
| Baseline | 9.7 min |
| + History | 15.5 min |
| + Spectral | 17.1 min |
| + Spectral + Uniform History | 24.5 min |
| + Spectral + Variable History | 24.3 min |

## Conclusions

1. **Spectral decomposition is valuable** for language modeling (+2.45% alone)
2. **History attention hurts** when applied to raw token positions (-2.89%)
3. **History + Spectral work together** better than either alone (+4.17%)
4. **Variable history depths** provide marginal additional benefit (+0.24%)
5. **The AKIRA architecture is validated** but the key insight is spectral decomposition, not history

## Next Steps

1. Test on longer sequences where history effects might be stronger
2. Test different band configurations (more/fewer bands)
3. Apply to streaming/dialogue tasks where temporal structure matters more
4. Test on actual signal processing tasks (where AKIRA should shine)

---

AKIRA Project - Experiment 029 v2 Results
Oscar Goldman - Shogu Research Group @ Datamutant.ai

Date: December 2024

Why history alone hurts language modeling:
"Position 5" means nothing consistent across different text sequences. Remembering what position 5 computed before is noise.
Why spectral + history works:
Spectral decomposition creates meaningful "positions" (frequency bands). Each band represents slow vs fast patterns. Remembering what this frequency band computed before IS meaningful - you're tracking how topics (slow) vs syntax (fast) evolve.
The key insight:
Signal domain:    Position = physical location     -> History helps directlyLanguage domain:  Position = arbitrary index       -> History hurtsLanguage + FFT:   Position = frequency band        -> History helps (positions now meaningful)
AKIRA architecture validated, but the credit goes to:
Spectral decomposition (primary benefit)
History attention (secondary, only works with spectral)
Variable depths (marginal additional benefit)
This explains why your original architecture was designed for spatio-temporal signals. For language, you need spectral decomposition to make positions meaningful first.

------------------------------------------------------------
1. Baseline (Standard GPT-2)
------------------------------------------------------------

Training 1. Baseline (Standard GPT-2)...
Parameters: 30,536,192

1. Baseline (Standard GPT-2):  10%|█         | 501/5000 [00:57<41:05,  1.82it/s]

  Step 500: eval_loss=1.8835, ppl=6.58, time=57.5s

1. Baseline (Standard GPT-2):  20%|██        | 1001/5000 [01:54<37:10,  1.79it/s]

  Step 1000: eval_loss=1.7679, ppl=5.86, time=114.5s

1. Baseline (Standard GPT-2):  30%|███       | 1501/5000 [02:52<32:29,  1.79it/s]

  Step 1500: eval_loss=1.7010, ppl=5.48, time=172.4s

1. Baseline (Standard GPT-2):  40%|████      | 2001/5000 [03:50<27:38,  1.81it/s]

  Step 2000: eval_loss=1.6607, ppl=5.26, time=230.6s

1. Baseline (Standard GPT-2):  50%|█████     | 2501/5000 [04:48<23:10,  1.80it/s]

  Step 2500: eval_loss=1.6227, ppl=5.07, time=288.7s

1. Baseline (Standard GPT-2):  60%|██████    | 3001/5000 [05:47<18:36,  1.79it/s]

  Step 3000: eval_loss=1.5953, ppl=4.93, time=347.0s

1. Baseline (Standard GPT-2):  70%|███████   | 3501/5000 [06:45<14:05,  1.77it/s]

  Step 3500: eval_loss=1.5793, ppl=4.85, time=405.2s

1. Baseline (Standard GPT-2):  80%|████████  | 4001/5000 [07:43<09:20,  1.78it/s]

  Step 4000: eval_loss=1.5697, ppl=4.81, time=463.3s

1. Baseline (Standard GPT-2):  90%|█████████ | 4501/5000 [08:41<04:42,  1.77it/s]

  Step 4500: eval_loss=1.5640, ppl=4.78, time=521.5s

1. Baseline (Standard GPT-2): 100%|██████████| 5000/5000 [09:39<00:00,  8.63it/s]

  Step 5000: eval_loss=1.5637, ppl=4.78, time=579.6s
  Training complete in 579.6s

------------------------------------------------------------
2. + History (Uniform)
------------------------------------------------------------

Training 2. + History (Uniform)...
Parameters: 32,118,272

2. + History (Uniform):  10%|█         | 501/5000 [01:33<48:28,  1.55it/s]  

  Step 500: eval_loss=1.9010, ppl=6.69, time=93.9s

2. + History (Uniform):  20%|██        | 1001/5000 [03:07<42:39,  1.56it/s]

  Step 1000: eval_loss=1.7680, ppl=5.86, time=187.2s

2. + History (Uniform):  30%|███       | 1501/5000 [04:39<34:08,  1.71it/s]

  Step 1500: eval_loss=1.6899, ppl=5.42, time=279.7s

2. + History (Uniform):  40%|████      | 2001/5000 [06:13<31:21,  1.59it/s]

  Step 2000: eval_loss=1.6690, ppl=5.31, time=373.1s

2. + History (Uniform):  50%|█████     | 2501/5000 [07:46<25:59,  1.60it/s]

  Step 2500: eval_loss=1.6391, ppl=5.15, time=466.3s

2. + History (Uniform):  60%|██████    | 3001/5000 [09:18<19:55,  1.67it/s]

  Step 3000: eval_loss=1.6132, ppl=5.02, time=558.3s

2. + History (Uniform):  70%|███████   | 3501/5000 [10:51<15:58,  1.56it/s]

  Step 3500: eval_loss=1.6034, ppl=4.97, time=651.5s

2. + History (Uniform):  80%|████████  | 4001/5000 [12:24<10:42,  1.55it/s]

  Step 4000: eval_loss=1.5986, ppl=4.95, time=744.7s

2. + History (Uniform):  90%|█████████ | 4501/5000 [13:56<05:01,  1.65it/s]

  Step 4500: eval_loss=1.5903, ppl=4.91, time=836.0s

2. + History (Uniform): 100%|██████████| 5000/5000 [15:29<00:00,  5.38it/s]

  Step 5000: eval_loss=1.5922, ppl=4.91, time=929.3s
  Training complete in 929.3s

------------------------------------------------------------
3. + Spectral Bands (No History)
------------------------------------------------------------

Training 3. + Spectral Bands (No History)...
Parameters: 32,328,704

3. + Spectral Bands (No History):  10%|█         | 501/5000 [01:43<1:13:37,  1.02it/s]

  Step 500: eval_loss=1.8711, ppl=6.50, time=103.6s

3. + Spectral Bands (No History):  20%|██        | 1001/5000 [03:26<1:09:45,  1.05s/it]

  Step 1000: eval_loss=1.7444, ppl=5.72, time=206.4s

3. + Spectral Bands (No History):  30%|███       | 1501/5000 [05:09<56:37,  1.03it/s]  

  Step 1500: eval_loss=1.6762, ppl=5.35, time=308.9s

3. + Spectral Bands (No History):  40%|████      | 2001/5000 [06:51<48:28,  1.03it/s]  

  Step 2000: eval_loss=1.6298, ppl=5.10, time=411.6s

3. + Spectral Bands (No History):  50%|█████     | 2501/5000 [08:34<43:39,  1.05s/it]

  Step 2500: eval_loss=1.5929, ppl=4.92, time=514.6s

3. + Spectral Bands (No History):  60%|██████    | 3000/5000 [10:17<43:26,  1.30s/it]

  Step 3000: eval_loss=1.5684, ppl=4.80, time=617.6s

3. + Spectral Bands (No History):  70%|███████   | 3501/5000 [12:00<24:19,  1.03it/s]

  Step 3500: eval_loss=1.5565, ppl=4.74, time=720.0s

3. + Spectral Bands (No History):  80%|████████  | 4001/5000 [13:42<17:22,  1.04s/it]

  Step 4000: eval_loss=1.5440, ppl=4.68, time=822.6s

3. + Spectral Bands (No History):  90%|█████████ | 4500/5000 [15:25<11:11,  1.34s/it]

  Step 4500: eval_loss=1.5393, ppl=4.66, time=925.6s

3. + Spectral Bands (No History): 100%|██████████| 5000/5000 [17:08<00:00,  4.86it/s]

  Step 5000: eval_loss=1.5388, ppl=4.66, time=1028.1s
  Training complete in 1028.2s

------------------------------------------------------------
4. + Spectral + Uniform History
------------------------------------------------------------

Training 4. + Spectral + Uniform History...
Parameters: 32,731,136

4. + Spectral + Uniform History:  10%|█         | 500/5000 [02:27<1:56:31,  1.55s/it]

  Step 500: eval_loss=1.8258, ppl=6.21, time=147.8s

4. + Spectral + Uniform History:  20%|██        | 1000/5000 [04:55<1:35:41,  1.44s/it]

  Step 1000: eval_loss=1.7260, ppl=5.62, time=295.7s

4. + Spectral + Uniform History:  30%|███       | 1500/5000 [07:22<1:29:14,  1.53s/it]

  Step 1500: eval_loss=1.6579, ppl=5.25, time=442.6s

4. + Spectral + Uniform History:  40%|████      | 2000/5000 [09:49<1:11:53,  1.44s/it]

  Step 2000: eval_loss=1.6126, ppl=5.02, time=589.7s

4. + Spectral + Uniform History:  50%|█████     | 2500/5000 [12:16<1:02:20,  1.50s/it]

  Step 2500: eval_loss=1.5833, ppl=4.87, time=736.8s

4. + Spectral + Uniform History:  60%|██████    | 3000/5000 [14:43<49:57,  1.50s/it]

  Step 3000: eval_loss=1.5546, ppl=4.73, time=883.3s

4. + Spectral + Uniform History:  70%|███████   | 3500/5000 [17:10<35:48,  1.43s/it]

  Step 3500: eval_loss=1.5397, ppl=4.66, time=1030.3s

4. + Spectral + Uniform History:  80%|████████  | 4000/5000 [19:37<25:51,  1.55s/it]

  Step 4000: eval_loss=1.5314, ppl=4.62, time=1177.3s

4. + Spectral + Uniform History:  90%|█████████ | 4500/5000 [22:03<11:53,  1.43s/it]

  Step 4500: eval_loss=1.5249, ppl=4.59, time=1323.7s

4. + Spectral + Uniform History: 100%|██████████| 5000/5000 [24:30<00:00,  3.40it/s]

  Step 5000: eval_loss=1.5236, ppl=4.59, time=1470.9s
  Training complete in 1470.9s

------------------------------------------------------------
5. + Spectral + Variable History (AKIRA)
------------------------------------------------------------

Training 5. + Spectral + Variable History (AKIRA)...
Parameters: 32,731,136

5. + Spectral + Variable History (AKIRA):  10%|█         | 500/5000 [02:25<1:49:41,  1.46s/it]

  Step 500: eval_loss=1.8300, ppl=6.23, time=145.7s

5. + Spectral + Variable History (AKIRA):  20%|██        | 1000/5000 [04:51<1:35:04,  1.43s/it]

  Step 1000: eval_loss=1.7227, ppl=5.60, time=291.5s

5. + Spectral + Variable History (AKIRA):  30%|███       | 1500/5000 [07:16<1:28:25,  1.52s/it]

  Step 1500: eval_loss=1.6532, ppl=5.22, time=436.7s

5. + Spectral + Variable History (AKIRA):  40%|████      | 2000/5000 [09:42<1:16:05,  1.52s/it]

  Step 2000: eval_loss=1.6114, ppl=5.01, time=582.3s

5. + Spectral + Variable History (AKIRA):  50%|█████     | 2500/5000 [12:07<1:00:08,  1.44s/it]

  Step 2500: eval_loss=1.5789, ppl=4.85, time=727.9s

5. + Spectral + Variable History (AKIRA):  60%|██████    | 3000/5000 [14:33<47:34,  1.43s/it]

  Step 3000: eval_loss=1.5550, ppl=4.74, time=873.2s

5. + Spectral + Variable History (AKIRA):  70%|███████   | 3500/5000 [17:00<38:42,  1.55s/it]

  Step 3500: eval_loss=1.5369, ppl=4.65, time=1020.3s

5. + Spectral + Variable History (AKIRA):  80%|████████  | 4000/5000 [19:25<24:03,  1.44s/it]

  Step 4000: eval_loss=1.5259, ppl=4.60, time=1165.7s

5. + Spectral + Variable History (AKIRA):  90%|█████████ | 4500/5000 [21:51<12:05,  1.45s/it]

  Step 4500: eval_loss=1.5219, ppl=4.58, time=1311.1s

5. + Spectral + Variable History (AKIRA): 100%|██████████| 5000/5000 [24:16<00:00,  3.43it/s]

  Step 5000: eval_loss=1.5210, ppl=4.58, time=1456.9s
  Training complete in 1456.9s

======================================================================
ABLATION RESULTS COMPARISON
======================================================================

Model                                               Loss        PPL      vs Base
-----------------------------------------------------------------------------
1. Baseline (Standard GPT-2)                      1.5637       4.78       +0.00%
2. + History (Uniform)                            1.5922       4.91       -2.89%
3. + Spectral Bands (No History)                  1.5388       4.66       +2.45%
4. + Spectral + Uniform History                   1.5236       4.59       +3.93%
5. + Spectral + Variable History (AKIRA)          1.5210       4.58       +4.17%

----------------------------------------------------------------------
>>> WINNER: 5. + Spectral + Variable History (AKIRA)
    Perplexity: 4.58
    Improvement over baseline: +4.17%

>>> AKIRA DESIGN VALIDATED
    Spectral bands + variable history depths wins!
----------------------------------------------------------------------

======================================================================
EXPERIMENT 029 v2 COMPLETE
======================================================================
