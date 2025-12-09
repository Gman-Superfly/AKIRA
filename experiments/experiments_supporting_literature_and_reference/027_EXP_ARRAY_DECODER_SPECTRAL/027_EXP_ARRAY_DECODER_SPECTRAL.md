# Experiment 027: Array Decoder with Spectral Bands

## Overview

This experiment validates the spectral band architecture using **array reconstruction** rather than token prediction. The goal is to verify that spectral bands properly separate frequency content and that per-position temporal memory enables accurate signal reconstruction.

## Motivation

Previous experiments (026) used token-based language modeling to test the spectral band architecture. However, token prediction is not the natural domain for this architecture. The spectral decomposition and Heisenberg-inspired temporal windows are fundamentally designed for **continuous signals** where:

1. Different frequency bands capture different time scales
2. Each position (sensor/sample) maintains its own temporal history
3. Reconstruction validates band separation quality

## Key Insight: Per-Position Temporal Memory

Unlike standard transformers where all positions attend to all positions, this architecture gives **each position its own attention mechanism**:

```
Position t sees ONLY: [position t at time T-1, T-2, ..., T-16]
NOT: [all positions at all times]
```

This models:
- Sensor arrays where each sensor has memory
- Video pixels tracking their own history  
- Signal samples with local temporal context

## Architecture

### Components

1. **Spectral Decomposer**: Signal -> 7 frequency bands
   - Band 0: DC/low frequency (slow patterns)
   - Band 6: High frequency (fast transients)
   - Uses learnable bandpass-like filters

2. **Per-Position Temporal Memory**: Fixed structure
   - Each position maintains 16-frame history
   - Attention weights: fixed exponential decay (not learned initially)
   - Structure enforces "each sensor sees only itself"

3. **Per-Band Processing**: 
   - Temporal attention (self-history per position)
   - Neighbor attention (adjacent positions)
   - Different parameters per band

4. **Decoder**: Learnable reconstruction
   - Takes attended band features
   - Outputs reconstructed signal
   - Trained with MSE + spectral loss

### Training Strategy

**Phase 1: Train Decoder Only (Fixed Attention)**
```
Input Signal -> FFT Decomposition (or learnable) -> Bands
Bands -> Fixed Temporal Attention (exp decay) -> Attended Features  
Attended Features -> [LEARNABLE DECODER] -> Reconstructed Signal

Loss = MSE(input, output) + spectral_loss
```

This validates:
- Can bands represent the signal?
- Is 16-frame temporal memory sufficient?
- Does the structure (per-position memory) work?

**Phase 2: Train Attention (Optional)**
```
Add learnable Q/K/V projections
Compare: learned attention vs fixed decay
```

## Synthetic Signal Design

### Signal Generation

Sum of sinusoids at known frequencies to test band separation:

```python
def generate_signal(T, frequencies, amplitudes, phases):
    t = torch.linspace(0, 1, T)
    signal = torch.zeros(T)
    for f, a, p in zip(frequencies, amplitudes, phases):
        signal += a * torch.sin(2 * pi * f * t + p)
    return signal
```

### Frequency Bands

| Band | Frequency Range | Expected Content |
|------|-----------------|------------------|
| 0 | 0-2 Hz | DC, slow drift |
| 1 | 2-8 Hz | Coarse structure |
| 2 | 8-16 Hz | Medium patterns |
| 3 | 16-32 Hz | Bridge/transition |
| 4 | 32-64 Hz | Fine structure |
| 5 | 64-128 Hz | Textures |
| 6 | 128+ Hz | High-freq edges |

### Test Cases

1. **Single frequency**: Pure sine at known band
2. **Multi-frequency**: Sum spanning multiple bands
3. **Chirp**: Frequency sweep (tests all bands)
4. **Transient**: Impulse + decay (tests temporal memory)

## Metrics

1. **Reconstruction MSE**: ||input - output||^2
2. **Per-Band Spectral Error**: Compare FFT magnitude per band
3. **Phase Coherence**: Correlation of phase across bands
4. **Temporal Memory Utilization**: How much history actually helps

## Expected Results

If bands properly separate frequencies:
- Single-frequency signals should activate ONE band strongly
- Multi-frequency signals should activate corresponding bands
- Reconstruction error should be low
- Per-band spectral error should show clean separation

If per-position temporal memory works:
- Transient signals should be reconstructed accurately
- Longer history should help low-freq bands more than high-freq

## Code Structure

```
027_EXP_ARRAY_DECODER_SPECTRAL/
  027_EXP_ARRAY_DECODER_SPECTRAL.md  (this document)
  README.md
  code/
    array_decoder_spectral.py        (main model)
    synthetic_signals.py             (signal generation)
    train_decoder.py                 (training loop)
    evaluate_bands.py                (band separation analysis)
```

## References

- Experiment 026: AKIRA Band Architecture (token-based)
- spatio_temporal_attention: TemporalAttention, NeighborAttention, WormholeAttention
- Heisenberg uncertainty: time-frequency tradeoff

## Status

- [ ] Create synthetic signal generator
- [ ] Implement array decoder with fixed attention
- [ ] Train on single-frequency signals
- [ ] Train on multi-frequency signals
- [ ] Analyze band separation
- [ ] Compare fixed vs learned attention

---

AKIRA Project - Experiment 027
*Oscar Goldman, Shogu Research Group @ Datamutant.ai subsidiary of 温心重工業*