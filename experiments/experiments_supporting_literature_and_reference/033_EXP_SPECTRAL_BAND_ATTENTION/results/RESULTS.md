# Experiment 033: Spectral Band Attention Results

## Summary

The SpectralPredictor (FFT-based multi-band attention) significantly outperforms an MLP baseline on 2D spatiotemporal prediction tasks, achieving 26x better generalization on held-out data.

---

## Methodology

### Task: Next-Frame Prediction

Given the current frame at time t, predict the frame at time t+1.

**Data Generation:**
- Synthetic 2D patterns on a grid (32x32 or 64x64)
- Patterns: moving blob, interference, switching (alternates between blob/interference), concentric rings, etc.
- Deterministic sequences - no stochasticity in the dynamics

**Training Protocol:**
- Online learning: generate frames on-the-fly
- 500 training steps
- Adam optimizer, LR=0.001
- MSE loss between predicted and actual next frame
- Held-out evaluation: 101 steps starting at t=10,000 (never seen during training)

### Models Compared

**SpectralPredictor:**
- FFT-based spectral decomposition into 3 bands: intensity (raw), low_freq, high_freq
- Three attention mechanisms:
  - **Temporal Attention**: Each spatial position attends to its own history (top-K selection)
  - **Neighbor Attention**: 8-connected local spatial attention across temporal window
  - **Wormhole Attention**: Sparse non-local connections via similarity gating
- History buffer of past frames (time_depth=8)
- Differential band routing: temporal/neighbor use intensity, wormhole uses low_freq for similarity matching
- Output: fused attention features projected to next frame prediction

**MLPPredictor (Baseline):**
- Flattens history buffer into single vector
- 3-layer MLP: input -> hidden (512) -> hidden (512) -> output
- Same history depth as SpectralPredictor
- No spectral decomposition, no attention

### Causal Correctness

All attention mechanisms enforce causality:
- Temporal: only attends to past timesteps (t' < t)
- Neighbor: only attends to past frames in temporal window
- Wormhole: causal mask ensures no future access

History buffer is updated AFTER attention computation, ensuring the model only sees frames t-1, t-2, etc. when predicting t+1.

---

## Results

### Blob Pattern (32x32 grid, 500 steps)

| Metric | SpectralPredictor | MLPPredictor | Winner |
|--------|-------------------|--------------|--------|
| Final Loss | 0.0031 | 0.031 | Spectral (10x better) |
| Min Loss | 0.0020 | 0.018 | Spectral (9x better) |
| Last-window Mean | 0.0059 | 0.033 | Spectral (5.6x better) |
| Improvement % | 94.3% | 28.8% | Spectral |
| **Eval Mean Loss** | **0.011** | **0.296** | **Spectral (26x better)** |
| Steps/sec | 21.4 | 27.7 | MLP (29% faster) |

**Key Finding:** MLP achieves reasonable training loss (0.031) but completely fails to generalize (eval 0.296). SpectralPredictor generalizes well (eval 0.011 vs training 0.006).

### Concentric Pattern (64x64 grid, 3000 steps)

| Metric | SpectralPredictor |
|--------|-------------------|
| Final Loss | 0.016 |
| Min Loss | 0.0015 |
| Improvement % | 89.6% |
| Eval Mean Loss | 0.011 |

Two concentric rings expanding/contracting in opposite phase. Tests multi-scale radial dynamics.

### Switching Pattern (64x64 grid, 1500 steps)

| Metric | SpectralPredictor |
|--------|-------------------|
| Final Loss | 0.0027 |
| Min Loss | 0.0012 |
| Improvement % | 89.5% |
| Eval Mean Loss | 0.012 |

Alternates between blob and interference patterns every 101 steps. Tests regime change adaptation.

Note: High loss spikes occur at exact transition frames (unpredictable discontinuities). Mean loss excludes these edge cases.

---

## Key Observations

### 1. Spectral Attention Enables Generalization

The MLP memorizes the training sequence but fails on held-out data (26x worse than Spectral). SpectralPredictor learns the underlying dynamics and generalizes to unseen time offsets.

### 2. FFT Decomposition Provides Meaningful Inductive Bias

Separating low-frequency (structure) from high-frequency (detail) allows:
- Wormhole attention to match similar patterns by structure
- Temporal attention to track object persistence
- Neighbor attention to model local physics (diffusion, propagation)

### 3. Continued Learning vs Early Plateau

- SpectralPredictor: 94% improvement over training, continues learning
- MLP: 29% improvement, plateaus early

### 4. Speed vs Accuracy Trade-off

MLP is ~30% faster per step, but the accuracy gap is massive. SpectralPredictor's overhead is justified by 10-26x better performance.

---

## Conclusions

### Validated Claims

1. **Spectral decomposition helps spatiotemporal prediction.** FFT-based band separation provides useful structure for attention mechanisms.

2. **Multi-attention architecture works.** Combining temporal (per-position history), neighbor (local spatial), and wormhole (non-local similarity) attention produces better results than a flat MLP.

3. **The model learns dynamics, not memorizes.** Strong generalization to held-out data confirms the model captures underlying patterns.

### Limitations

1. **Synthetic data only.** Real-world spatiotemporal data may have different characteristics.

2. **Deterministic patterns.** Stochastic dynamics would be a harder test.

3. **No comparison to ConvLSTM/U-Net.** These standard baselines would provide better context for the results.

### Next Steps

- Compare against ConvLSTM and U-Net baselines
- Test on real video/physics data
- Ablation: which attention type contributes most?
- Longer prediction horizons (t+5, t+10)

---

## Reproduction

Run comparison:
```bash
cd code
python synthetic_2d.py --config fast --grid-size 32 --steps 500 --pattern-train blob --model compare
```

Run single pattern:
```bash
python synthetic_2d.py --config fast --grid-size 64 --steps 1500 --pattern-train switching --model spectral
```

Available patterns: `blob`, `interference`, `switching`, `double_slit`, `counter_rotate`, `chirp`, `phase_jump`, `noisy_motion`, `bifurcation`, `wave_collision`, `concentric`

---

*Oscar Goldman - Shogu Research Group @ Datamutant.ai*
