# Experiment 042: Spectral Band Additive Viewer

cd C:\Git\AKIRA_working_folder\AKIRA\experiments\experiments_supporting_literature_and_reference\042_EXP_SPECTRAL_BAND_ADDITIVE_VIEWER\code

# Install deps (if not already)
pip install fastapi uvicorn torch matplotlib numpy

# Start the web server
python server.py

## Overview

Real-time visualization of attention dynamics using additive wave representation. This experiment combines:

1. **Spectral Band Attention** from 033: Temporal, Neighbor, and Wormhole attention
2. **Additive Wave Visualization** from 034/code_003: Each position gets a wave whose frequency reflects its importance/rank, and we watch the harmonics sum as attention focuses

## What we visualize

- **Attention as additive waves**: Each attended position contributes a wave. When attention focuses, the waves constructively interfere (high amplitude). When attention spreads, waves destructively interfere (low amplitude).
- **Phase coherence**: Are the attended positions in phase? High coherence = agreement.
- **Head synchronization**: Do different attention heads point the same direction?
- **Belief collapse**: Watch entropy drop and coherence rise through layers in real-time.

## Running the experiment

```bash
cd AKIRA/experiments/experiments_supporting_literature_and_reference/042_EXP_SPECTRAL_BAND_ADDITIVE_VIEWER/code

# Run the web server
python server.py

# Open browser at http://localhost:8042
```

## Architecture

```
042_EXP_SPECTRAL_BAND_ADDITIVE_VIEWER/
├── README.md
└── code/
    ├── __init__.py
    ├── additive_wave_encoder.py   # Wave generation from attention weights
    ├── spectral_predictor.py      # SpectralBandAttention model
    ├── configs.py                 # Configuration presets
    ├── server.py                  # FastAPI web interface
    └── run.py                     # CLI entry point
```

## Key insight: Additive interference as belief visualization

When attention weights \( w_i \) are applied to waves \( \psi_i(t) \):

$$\Psi(t) = \sum_i w_i \cdot \psi_i(t)$$

The resulting superposition \( \Psi \) shows:
- **Constructive interference** when attended positions agree (in phase)
- **Destructive interference** when attended positions disagree (out of phase)
- **Amplitude** reflects total attention weight on in-phase components

This is a direct visualization of "belief concentration" vs "belief diffusion".

## Connection to AKIRA theory

- **Phase coherence** maps to the attention centroid magnitude (code_004)
- **Head synchronization** shows collective agreement across heads
- **Additive waves** make the abstract "belief collapse" visually intuitive

---

*Oscar Goldman, Shogu Research Group @ Datamutant.ai subsidiary of 温心重工業*

