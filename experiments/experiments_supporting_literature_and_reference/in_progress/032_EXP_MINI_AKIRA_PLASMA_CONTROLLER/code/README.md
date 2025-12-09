# Experiment 032: Mini AKIRA Plasma Controller - Code

## Overview

This directory contains the implementation for Experiment 032, which tests AKIRA's theory-aligned architecture on a real-time control task.

## File Structure

```
code/
  README.md                    # This file
  run_experiment.py            # Main experiment orchestration
  mini_plasma_env.py           # Toy plasma environment
  spectral_belief_machine.py   # 7+1 architecture (main model)
  baselines.py                 # Comparison models
  belief_tracker.py            # Entropy and collapse tracking
  homeostat.py                 # PSON-aligned controller
```

## Quick Start

```bash
# Quick test run (reduced data, fast)
python run_experiment.py --quick

# Full experiment
python run_experiment.py

# Run specific stage
python run_experiment.py --stage predictor
python run_experiment.py --stage control
python run_experiment.py --stage analysis

# Use GPU
python run_experiment.py --device cuda
```

## Module Descriptions

### `mini_plasma_env.py`
A simple 2D continuous field environment with:
- Diffusion (Laplacian smoothing)
- Advection (mild drift)
- Actuator control (6 localized Gaussian forces)
- Noise (stochastic perturbation)

The control task is to keep a blob centered at the target position.

### `spectral_belief_machine.py`
The main AKIRA architecture implementing:
- **7 spectral bands**: Log-spaced radial FFT masks from DC to Nyquist
- **1 temporal band**: Causal self-attention over history
- **Wormhole interconnects**: Cross-band communication (0<->6, 1<->5, 2<->4, 3->all, 7->all)
- **Belief tracking**: Entropy per band, collapse detection
- **Differential learning rates**: Band 0 (slowest) to Band 6 (fastest)

### `baselines.py`
Three comparison models:
1. **FlatBaseline**: Standard ConvNet, no spectral structure
2. **FourBandBaseline**: 4 spectral bands instead of 7
3. **SpectralOnlyBaseline**: 7 bands but no temporal attention

All baselines are parameter-matched for fair comparison.

### `belief_tracker.py`
Tracks belief dynamics:
- Entropy per band over time
- Entropy rate (dH/dt)
- Collapse detection (sharp entropy drops)
- Temperature parameter for phase transition control

### `homeostat.py`
PSON-aligned homeostatic controller:
- Orthogonal noise injection (exploration without fighting gradient)
- Setpoint maintenance
- Oscillation detection and damping
- Adaptive gain

## Hypotheses Tested

1. **H1**: 7+1 architecture outperforms baselines at matched parameters
2. **H2**: Belief entropy predicts control difficulty
3. **H3**: Collapse events correlate with control transitions
4. **H4**: Homeostat improves stability over Adam

## Expected Outputs

Results are saved to `../results/` with timestamps:
- `results_YYYYMMDD_HHMMSS.json`: Numerical metrics

Key metrics:
- Predictor MSE per model
- Control error (Adam vs Homeostat)
- Entropy-error correlation
- Collapse statistics

## Dependencies

- PyTorch >= 2.0
- NumPy

No external dependencies beyond standard PyTorch.

## Theory Alignment

This experiment directly tests claims from:
- `SPECTRAL_BELIEF_MACHINE.md`: Core architecture
- `THE_SEVEN_PLUS_ONE_ARCHITECTURE.md`: 7+1 justification
- `ORTHOGONALITY.md`: Wormhole pairs
- `TERMINOLOGY.md`: Collapse, entropy definitions

See `032_EXP_MINI_AKIRA_PLASMA_CONTROLLER.md` for full experimental design.

## Falsification Criteria

- **H1 falsified**: 7+1 performs equal/worse than flat baseline
- **H2 falsified**: No correlation between entropy and future error
- **H3 falsified**: Collapse events random w.r.t. control transitions
- **H4 falsified**: Homeostat performs equal/worse than Adam

---

*Oscar Goldman, Shogu Research Group @ Datamutant.ai subsidiary of 温心重工業*
