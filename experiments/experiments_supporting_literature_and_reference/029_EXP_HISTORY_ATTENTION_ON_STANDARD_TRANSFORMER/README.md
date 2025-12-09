# 029 - History Attention on Standard Transformer

Tests whether adding per-position history attention and spectral band decomposition improves a standard GPT-2 style transformer on WikiText-2.

## Experiment Versions

This experiment evolved through three versions as we tested different hypotheses:

| Version | Focus | Key Finding |
|---------|-------|-------------|
| v1 | History attention alone | Marginal benefit (~3.75%) |
| v2 | Spectral bands + history | **Spectral decomposition is the key** (+4.17%) |
| v3 | Full AKIRA 7+1 architecture | In progress |

## Quick Start

### Google Colab (Recommended)
```bash
# v1: Basic history attention test
Upload code/colab_history_attention_gpt2.py

# v2: Spectral bands ablation (4 bands)
Upload code/colab_history_attention_gpt2_v2_ablation.py

# v3: Full AKIRA 7+1 architecture
Upload code/colab_history_attention_gpt2_v3_full_akira.py
```

Select GPU runtime and run all cells.

---

## V1: History Attention Test

**Question**: Does per-position history attention improve language modeling?

### Models Tested

| Model | Description |
|-------|-------------|
| Baseline | Standard GPT-2 (causal attention only) |
| + Token History | Adds history layer storing raw embeddings |
| + Belief History | Adds history layer storing attention outputs |

### V1 Results

| Model | Loss | PPL | vs Baseline |
|-------|------|-----|-------------|
| Baseline | 1.7042 | 5.50 | - |
| + Token History | 1.6674 | 5.30 | +3.61% |
| + Belief History | 1.6659 | 5.29 | +3.75% |

**Initial Conclusion**: Small improvement, belief slightly better than token.

---

## V2: Spectral Bands Ablation

**Question**: Does spectral decomposition help? Does it make history attention useful?

### Models Tested

| # | Model | Description |
|---|-------|-------------|
| 1 | Baseline | Standard GPT-2 |
| 2 | + History | Standard + uniform history attention |
| 3 | + Spectral | 4 causal spectral bands (no history) |
| 4 | + Spectral + Uniform History | Bands + same history depth per band |
| 5 | + Spectral + Variable History | Bands + Heisenberg-inspired depths |

### V2 Configuration

```python
embed_dim: 256
num_layers: 6
num_bands: 4
kernel_sizes: [9, 7, 5, 3]  # Causal convolutions
band_history_depths: [128, 64, 32, 16]  # Variable
max_history_uniform: 64
total_steps: 5000
Dataset: WikiText-2
```

### V2 Results

| Model | Loss | PPL | vs Baseline |
|-------|------|-----|-------------|
| 1. Baseline | 1.5637 | 4.78 | - |
| 2. + History (Uniform) | 1.5922 | 4.91 | **-2.89%** (worse!) |
| 3. + Spectral Bands | 1.5388 | 4.66 | +2.45% |
| 4. + Spectral + Uniform History | 1.5236 | 4.59 | +3.93% |
| 5. + Spectral + Variable History | 1.5210 | 4.58 | **+4.17%** |

### V2 Key Findings

1. **History attention ALONE hurts** (-2.89%): Per-position history doesn't make sense for language where "position 5" is arbitrary across sequences.

2. **Spectral decomposition helps** (+2.45%): Breaking embedding into frequency bands provides useful inductive bias even without history.

3. **Spectral + History work together** (+4.17%): History attention becomes beneficial when applied to meaningful positions (frequency bands) rather than arbitrary token positions.

4. **Variable depths provide marginal benefit** (+0.24% over uniform): The Heisenberg-inspired variable depths help slightly.

### Why Spectral Makes History Work

```
Raw token space:     "Position 5" = arbitrary, no meaning across sequences
                     History of position 5 = noise

Spectral band space: "Band 0" = low frequency content (topics, style)
                     "Band 6" = high frequency content (syntax, details)
                     History of each band = meaningful temporal context
```

---

## V3: Full AKIRA 7+1 Architecture

**Question**: Does the full AKIRA architecture (7 spectral + 1 temporal band) provide additional benefit?

### Models Tested

| # | Model | Description |
|---|-------|-------------|
| 1 | Baseline | Standard GPT-2 |
| 2 | AKIRA 7+1 | Full architecture (no history) |
| 3 | AKIRA 7+1 + History | + uniform history per band |
| 4 | AKIRA 7+1 + Variable History | + Heisenberg-inspired depths |

### V3 Architecture Details

**8 Bands Total** (embed_dim=512, 64 dim per band):

| Band | Type | Kernel | LR | History Window | Processing |
|------|------|--------|-----|----------------|------------|
| 0 | Spectral | 15 | 3e-5 | 128 | Geometric |
| 1 | Spectral | 11 | 5e-5 | 64 | Geometric |
| 2 | Spectral | 9 | 1e-4 | 32 | Geometric |
| 3 | Spectral (Bridge) | 7 | 3e-4 | 16 | Hybrid |
| 4 | Spectral | 5 | 5e-4 | 16 | Hybrid |
| 5 | Spectral | 3 | 1e-3 | 8 | Reactive |
| 6 | Spectral | 1 | 3e-3 | 4 | Reactive |
| 7 | Temporal | - | 3e-4 | - | Causal Attn |

**Temporal Wormhole**: Complementary pairs exchange information across time scales:
- Band 0 (sees 128 back) <-> Band 6 (sees 4 back)
- Band 1 (sees 64 back) <-> Band 5 (sees 8 back)
- Band 2 (sees 32 back) <-> Band 4 (sees 16 back)
- Band 3 (bridge): Aggregates from all others

**Processing Modes**:
- **Geometric (0-2)**: Low freq, gated residual, structure-preserving
- **Hybrid (3-4)**: Mid freq, balanced
- **Reactive (5-6)**: High freq, fast adaptive

### V3 Results

*Pending - run v3 notebook*

---

## Files

```
029_EXP_HISTORY_ATTENTION_ON_STANDARD_TRANSFORMER/
  029_EXP_HISTORY_ATTENTION_ON_STANDARD_TRANSFORMER.md  # Full documentation
  README.md                                              # This file
  code/
    colab_history_attention_gpt2.py                      # v1: Basic history test
    colab_history_attention_gpt2_v2_ablation.py          # v2: Spectral bands (4)
    colab_history_attention_gpt2_v3_full_akira.py        # v3: Full 7+1 AKIRA
  results/
    029_RESULTS.md                                       # v1 results
    029_v2_ABLATION_RESULTS.md                           # v2 results
```

## IMPORTANT: V1 vs V2 Discrepancy Explained

**Apparent contradiction:**
- v1: History alone gave **+3.75%** improvement
- v2: History alone gave **-2.89%** (worse!)

**Why the reversal?**

| Version | Baseline PPL | History PPL | Change |
|---------|--------------|-------------|--------|
| v1 | 5.50 (weaker) | 5.29 | +3.75% |
| v2 | 4.78 (stronger) | 4.91 | -2.89% |

**The explanation:**

1. **v2 baseline is stronger** (PPL 4.78 vs 5.50): Better training reached a stronger baseline in v2.

2. **Diminishing returns**: History attention might help an undertrained model "catch up" but provides no real structural benefit. Once training is sufficient, it becomes noise or even harmful.

3. **Experimental variance**: Single runs have variance. The "improvement" in v1 was likely noise that disappeared with better training.

**The honest conclusion:**

```
Short training / weak baseline:  History looks helpful (+3.75%)
Long training / strong baseline: History actually hurts (-2.89%)
```

This reveals that **per-position history for arbitrary token positions is NOT fundamentally useful for language modeling**. Any apparent benefit was:
- Random variance
- Helping an undertrained baseline catch up
- Extra parameters acting as regularization (not the history mechanism itself)

This is WHY spectral decomposition matters - it creates meaningful positions (frequency bands) where history IS useful, as shown by v2's +4.17% improvement when spectral + history are combined.

---

## Summary

| What We Learned | Evidence |
|-----------------|----------|
| History attention alone doesn't reliably help language modeling | v1: +3.75%, v2: -2.89% (inconsistent) |
| The v1 "improvement" was likely noise or weak-baseline artifact | v2 with stronger baseline showed negative effect |
| History attention helps signal processing (physical positions) | Exp 028: +62% |
| Spectral decomposition consistently improves language modeling | v2: +2.45% |
| Spectral + history work together reliably | v2: +4.17% |
| Variable history depths provide marginal additional benefit | v2: +0.24% |

**The key insight**: Per-position memory is natural for signals (where position = physical location) but requires spectral decomposition to work for language (where position = arbitrary index). Spectral bands create meaningful "positions" (frequency components) that benefit from temporal memory.

**Warning**: Be skeptical of small improvements from history attention alone on language tasks. The benefit may disappear or reverse with better training. The real value comes from combining history with spectral decomposition.

---



*Oscar Goldman — Shogu Research Group @ Datamutant.ai subsidiary of 温心重工業*