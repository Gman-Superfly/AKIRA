# Experiment 026: AKIRA Band Architecture

**The core test of AKIRA's architectural hypothesis.**

---

## Quick Start

```bash
# Test the AKIRA model implementation
python code/models/akira_band_transformer.py
```

Expected output:
```
Creating AKIRA Band Transformer...
Model parameters: ~25M
Hidden dim: 512
Band dims: (128, 96, 80, 64, 64, 48, 32)
Band LRs: (1e-05, 3e-05, 0.0001, 0.0003, 0.001, 0.003, 0.01)
...
AKIRA Band Transformer test complete!
```

---

## What This Tests

GPT-2 shows emergent spectral organization (Layer 5 has low-freq AQ).
AKIRA proposes explicit spectral organization should be better.

| Aspect | GPT-2 (Emergent) | AKIRA (Designed) |
|--------|------------------|------------------|
| Band structure | None | 7 explicit bands |
| Learning rates | Uniform | Per-band (slow to fast) |
| Cross-band comm | Implicit (attention) | Wormhole attention |
| Interpretability | Low | High |

---

## Key Files

```
026_EXP_AKIRA_BAND_ARCHITECTURE/
├── 026_EXP_AKIRA_BAND_ARCHITECTURE.md  # Full experiment design
├── README.md                            # This file
└── code/
    └── models/
        └── akira_band_transformer.py    # Core implementation
```

---

## The Band Architecture

```
Band 0 (DC):      128 dims, LR=1e-5   <- Slowest, most stable
Band 1 (low):      96 dims, LR=3e-5
Band 2 (low-mid):  80 dims, LR=1e-4
Band 3 (mid):      64 dims, LR=3e-4   <- Base rate
Band 4 (mid-high): 64 dims, LR=1e-3
Band 5 (high):     48 dims, LR=3e-3
Band 6 (highest):  32 dims, LR=1e-2   <- Fastest, most adaptive
                  ___
                  512 total (same as standard transformer)
```

---

## Predictions

If AKIRA's theory is correct:

1. **More AQ**: ~8-12% vs ~3-4% in standard transformers
2. **Cleaner separation**: Low-freq bands have clear low-freq AQ
3. **Better universality**: ~50% overlap vs ~25%
4. **Per-band grokking**: Bands mature at different rates

---

## Hardware Requirements

- **A100 (40GB)**: Recommended for full experiment
- **Training time**: ~3 hours per model
- **Full experiment**: ~1 day (including ablations)

---

*Oscar Goldman — Shogu Research Group @ Datamutant.ai subsidiary of 温心重工業*