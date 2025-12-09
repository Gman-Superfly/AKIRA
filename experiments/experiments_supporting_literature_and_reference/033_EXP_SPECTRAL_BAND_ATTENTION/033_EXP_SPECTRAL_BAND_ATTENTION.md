# Experiment 033: Spectral Band Attention - Working Reference Implementation

## Status: REFERENCE IMPLEMENTATION

## Overview

NOTE THIS TEST DOES A LOT MORE IT COME FROM ARCHIVE, THERE ARE MANY FINCTIONS NOT YET EXPOSED BUT IF YOU LOOK AT CODE YOU CAN TEASE THEM OUT, BE CAREFULL TO NOT ADD TOO MANY WORMHOLE CONNECTIONS AND KEEP THE GRID SMALL FOR NOW UNLESS YOU HAVE A GIGACOMPUTER


This experiment contains a **working implementation** of the three-attention-type architecture that demonstrates excellent real-time prediction on 2D spatiotemporal fields. This code should be studied deeply as it implements key AKIRA concepts in a practical, validated form.

## Origin

This code was developed in the `spatio_temporal_attention` repository and has been validated to work well for next-frame prediction tasks. It serves as a reference for how the theoretical AKIRA concepts can be implemented effectively.

## Architecture Summary

### Three Attention Types

The system decomposes attention into three complementary mechanisms, each with a distinct role:

**1. Temporal Attention**
- Each spatial position `(i, j)` attends ONLY to its own history at `(i, j, t' < t)`
- Uses **Top-K selection** with exponential decay
- Mask: `i' = i AND j' = j AND t' < t`
- Purpose: **Object permanence** - "what was here before?"

**2. Neighbor Attention**
- 8-connected spatial neighborhood (3x3 kernel) across temporal window
- Uses padding for boundary handling
- Mask: `|i - i'| <= 1 AND |j - j'| <= 1 AND |t - t'| <= layer_range`
- Purpose: **Local physics** - diffusion, collision, wave propagation

**3. Wormhole Attention**
- Sparse non-local connections based on cosine similarity threshold
- Uses **separate bands for similarity matching vs value retrieval**
- Adaptive spatial pooling for large grids
- Purpose: **Teleportation** - connect distant similar features instantly

### Band Decomposition

Uses FFT-based frequency decomposition:
- `intensity`: Raw input features
- `low_freq`: Low frequency components (structure)
- `high_freq`: High frequency components (detail)

### Key Design Insight: Cross-Band Wormhole

The wormhole attention uses different bands for different purposes:
- `similarity_band = low_freq`: Find similar patterns by matching structure
- `value_band = intensity`: Retrieve actual values from matched locations

This separation allows the system to find structurally similar patterns across time while retrieving the appropriate intensity values.

## Key Differences from 032 Experiment

| Aspect | This Implementation | 032 Experiment |
|--------|---------------------|----------------|
| **Temporal attention** | Per-position Top-K | Global MHA on pooled features |
| **Neighbor attention** | Explicit 3x3 kernel | Not present |
| **Wormhole** | Similarity-gated sparse | Simplified cross-band mixing |
| **History handling** | Ring buffer per band | List appending |
| **Band decomposition** | FFT low/high freq | 7 radial masks |

## Why This Works

1. **Per-position temporal attention** - Each pixel tracks its own history, not a pooled summary
2. **Explicit neighbor attention** - Captures local physics that CNNs learn implicitly
3. **Wormhole with similarity gating** - Finds resonant patterns efficiently without O(N^2) attention
4. **Ring buffer history** - Proper temporal ordering with efficient memory use

## Files

```
033_EXP_SPECTRAL_BAND_ATTENTION/
├── 033_EXP_SPECTRAL_BAND_ATTENTION.md  # This document
├── code/
│   ├── __init__.py
│   ├── temporal_attention.py      # Per-position temporal attention with Top-K
│   ├── neighbor_attention.py      # 8-connected local spatial attention
│   ├── wormhole_attention.py      # Sparse similarity-gated non-local attention
│   ├── spectral_attention.py      # Coordinator for all three attention types
│   ├── configs.py                 # Configuration presets
│   ├── synthetic_2d.py            # Main experiment script
│   └── README.md                  # Quick start guide
└── results/
    └── (experiment outputs)
```

## Running the Experiment

```bash
cd AKIRA/experiments/experiments_supporting_literature_and_reference/033_EXP_SPECTRAL_BAND_ATTENTION/code

# Quick test (32x32 grid, fast config)
python synthetic_2d.py --config fast --grid-size 32 --steps 300

# With visualization
python synthetic_2d.py --config fast --steps 500 --viz-interval 5 --viz-dir viz_out

# Different patterns
python synthetic_2d.py --pattern-train blob --steps 500
python synthetic_2d.py --pattern-train interference --steps 500
python synthetic_2d.py --pattern-train switching --steps 500

# Larger grid
python synthetic_2d.py --config large --grid-size 64 --steps 500

# Log attention statistics
python synthetic_2d.py --config fast --steps 300 --log-stats-csv attention_stats.csv
```

## Available Patterns

- `blob`: Moving Gaussian blob (circular motion)
- `interference`: Interference pattern (wave superposition)
- `switching`: Alternates between blob and interference
- `double_slit`: Double-slit interference simulation
- `counter_rotate`: Two blobs orbiting in opposite directions
- `chirp`: Frequency sweep pattern
- `phase_jump`: Pattern with abrupt phase changes
- `noisy_motion`: Blob with position/velocity noise
- `bifurcation`: Blob that occasionally splits
- `wave_collision`: Opposing wavefronts creating standing nodes

## Configuration Options

| Config | baseDim | attnDim | timeDepth | Notes |
|--------|---------|---------|-----------|-------|
| `turbo` | 16 | 16 | 4 | Maximum speed |
| `fast` | 32 | 32 | 8 | Quick iteration |
| `default` | 64 | 64 | 16 | Balanced |
| `large` | 32 | 32 | 8 | For 64x64 grids |
| `xlarge` | 32 | 32 | 8 | For 128x128 grids |

## Research Questions

This implementation can help investigate:

1. **Per-position vs pooled temporal attention**: Does per-position attention provide better object tracking?

2. **Neighbor attention necessity**: Can we remove neighbor attention and rely on wormhole for local patterns?

3. **Wormhole threshold tuning**: What is the optimal similarity threshold for different pattern types?

4. **Cross-band routing**: How does using low_freq for similarity vs intensity for values affect performance?

5. **Top-K vs softmax**: Does hard Top-K selection outperform soft attention for temporal memory?

## Connection to AKIRA Theory

This implementation validates several AKIRA concepts:

- **Spectral decomposition**: FFT-based band separation (simplified from 7+1)
- **Three attention types**: Temporal, Neighbor (local), Wormhole (non-local)
- **Causal constraints**: All attention types respect the arrow of time
- **Sparse non-local connections**: Wormhole uses threshold gating, not full attention

## Limitations

What this implementation does NOT have:
- 7+1 band architecture (uses 3 bands)
- Differential learning rates per band
- Explicit belief/entropy tracking
- Homeostat integration
- Collapse detection

## Next Steps

1. Study the attention statistics output to understand dynamics
2. Compare blob vs interference pattern performance
3. Analyze wormhole activation patterns
4. Test on more complex multi-object scenarios
5. Integrate insights back into full AKIRA architecture

---

**Author**: Imported from spatio_temporal_attention repository OG
**Date**: December 2024

*Oscar Goldman, Shogu Research Group @ Datamutant.ai subsidiary of 温心重工業*