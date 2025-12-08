# Experiment 022: Band Phase Locking

## Do Spectral Bands Lock to Rational Ratios?

**Tier:** ★★ CORE (Harmony and Coherence)  
**Status:** PLANNED  
**Depends On:** 003 (Spectral Band Dynamics), 020 (Cross-Band Flow)

---

## 1. Problem Statement

Coupled oscillators with slightly different natural frequencies will "phase lock" to rational frequency ratios when coupled. This is seen in:
- Pendulums on a shared platform
- Fireflies synchronizing flashes
- Jupiter's moons (1:2:4 orbital resonance)

**Do AKIRA's spectral bands exhibit similar phase locking?**

If bands are coupled through wormhole attention and shared gradients, they may find resonant relationships during training.

---

## 2. Hypothesis

```
THE PHASE LOCKING HYPOTHESIS:

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  Spectral bands PHASE LOCK during training to rational ratios.        │
│                                                                         │
│  MECHANISM:                                                             │
│  • Bands are coupled through:                                          │
│    - Wormhole attention (cross-band communication)                    │
│    - Shared loss (common optimization target)                         │
│    - Gradient flow (backpropagation across bands)                     │
│                                                                         │
│  • Coupling forces synchronization:                                    │
│    - Arbitrary relationships are unstable                             │
│    - Rational ratios are stable attractors                            │
│                                                                         │
│  PREDICTED RATIOS:                                                      │
│  • Band 0 : Band 6 = 1 : 64 (frequency ratio)                         │
│  • Symmetric pairs (0↔6, 1↔5, 2↔4) should show strongest lock        │
│  • Band 3 (bridge) mediates all connections                           │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Scientific Basis

### 3.1 Phase Locking in Physics

```
COUPLED OSCILLATOR PHASE LOCKING:

Two oscillators with frequencies f₁ and f₂:
• Uncoupled: drift in and out of phase
• Coupled: lock to rational ratio m:n

The Arnold tongue: region of frequency ratios where locking occurs
Wider for stronger coupling, narrower for weaker

EXAMPLES:
• Huygens' pendulums: anti-phase lock
• Circadian rhythms: 24-hour lock
• Laser mode locking: frequency comb
```

### 3.2 Orbital Resonances

```
JUPITER'S MOONS:

Io : Europa : Ganymede = 1 : 2 : 4

These are NOT arbitrary.
Over billions of years, gravity forced resonance.
Non-resonant orbits → ejected or crashed.

The solar system "plays in equal temperament."
```

### 3.3 Application to Spectral Bands

```
BAND COUPLING SOURCES:

1. WORMHOLE ATTENTION
   • Direct communication between bands
   • 0↔6, 1↔5, 2↔4 are explicit pairs
   • Band 3 connects to all

2. SHARED LOSS
   • All bands contribute to prediction
   • Common target creates coupling

3. GRADIENT FLOW
   • Backprop flows through all bands
   • Creates implicit coupling
```

### 3.4 AKIRA Theory Basis

**Relevant Theory Documents:**
- `foundations/HARMONY_AND_COHERENCE.md` — §5 (Phase Locking), §6 (Resonant Ratios)
- `architecture_theoretical/SPECTRAL_BELIEF_MACHINE.md` — §2 (Differential Timescales)
- `architecture_expanded/wormhole/WORMHOLE_ARCHITECTURE.md` — §2 (Symmetric Pairs)

**Key Concepts:**
- **Coupled oscillators:** Bands are coupled via wormholes, shared loss, gradient flow
- **Phase locking:** Frequencies synchronize to rational ratios (m:n) due to coupling
- **Resonant pairs:** Band 0↔6 (1:64), Band 1↔5 (1:16), Band 2↔4 (1:4)
- **Arnold tongues:** Regions in parameter space where phase locking occurs

**From HARMONY_AND_COHERENCE.md (§5.1):**
> "Phase locking occurs when coupled oscillators synchronize to rational frequency ratios. Jupiter's moons (Io:Europa:Ganymede = 1:2:4) locked via gravity over billions of years. AKIRA's spectral bands coupled via wormholes should exhibit similar locking during training."

**From SPECTRAL_BELIEF_MACHINE.md (§2.1):**
> "Differential learning rates across bands: 3000× ratio from Band 0 to Band 6. These set natural timescales. Coupling via wormholes may force rational relationships between these timescales (phase locking to resonant ratios)."

**From WORMHOLE_ARCHITECTURE.md (§2.3):**
> "Symmetric pairs (0↔6, 1↔5, 2↔4) are explicitly coupled. Band 3 serves as bridge connecting all bands. These structural couplings should produce phase locking: pairs lock to rational ratios, Band 3 mediates."

**This experiment validates:**
1. Whether **bands phase lock** to rational frequency ratios
2. Whether **symmetric pairs** show strongest locking
3. Whether **coupling strength** correlates with locking width (Arnold tongues)
4. Whether locking is **stable attractor** (persists after training)

**Falsification:** If bands drift with arbitrary phase relationships → no resonance → coupling too weak → wormholes don't create functional binding.

---

## 4. Apparatus

### 4.1 Required Measurements

```
MEASUREMENT REQUIREMENTS:

1. BAND ACTIVATION TIME SERIES
   • Per-band activation over sequence
   • Shape: (time, bands, features)

2. CROSS-CORRELATION
   • Correlation between band pairs
   • Shape: (bands, bands)

3. FREQUENCY ANALYSIS
   • FFT of band activation time series
   • Identify dominant frequencies per band

4. RATIO DETECTION
   • Compute frequency ratios
   • Check for rational relationships
```

### 4.2 Experimental Setup

```python
class PhaseLockingDetector:
    """Detects phase locking between spectral bands."""
    
    def __init__(self, num_bands: int = 8):
        self.num_bands = num_bands
    
    def extract_band_activations(self, model, data) -> Tensor:
        """Get time series of per-band activations."""
        # Run model, extract per-band attention entropy or activation
        activations = []
        for batch in data:
            band_acts = model.get_band_activations(batch)
            activations.append(band_acts)
        return torch.stack(activations)  # (time, bands)
    
    def compute_cross_correlation(self, activations: Tensor) -> Tensor:
        """Cross-correlation matrix between bands."""
        # Normalize
        centered = activations - activations.mean(dim=0)
        normalized = centered / (centered.std(dim=0) + 1e-8)
        
        # Cross-correlation
        return torch.matmul(normalized.T, normalized) / len(activations)
    
    def detect_frequency_ratios(self, activations: Tensor) -> Dict:
        """Find dominant frequencies and their ratios."""
        ratios = {}
        
        for i in range(self.num_bands):
            for j in range(i+1, self.num_bands):
                # FFT of each band
                fft_i = torch.fft.fft(activations[:, i])
                fft_j = torch.fft.fft(activations[:, j])
                
                # Find dominant frequency
                freq_i = torch.argmax(torch.abs(fft_i[1:len(fft_i)//2])) + 1
                freq_j = torch.argmax(torch.abs(fft_j[1:len(fft_j)//2])) + 1
                
                # Compute ratio
                ratio = freq_i.float() / freq_j.float()
                ratios[(i, j)] = self._simplify_ratio(ratio)
        
        return ratios
    
    def _simplify_ratio(self, ratio: float, max_denom: int = 8) -> Tuple[int, int]:
        """Find nearest simple rational approximation."""
        from fractions import Fraction
        frac = Fraction(ratio).limit_denominator(max_denom)
        return (frac.numerator, frac.denominator)
```

---

## 5. Method

### 5.1 Protocol

```
EXPERIMENTAL PROTOCOL:

1. TRAIN MODEL
   • Standard training on prediction task
   • Record per-band activations throughout

2. EXTRACT ACTIVATION TIME SERIES
   • For each band, extract activation over sequence
   • Long sequences preferred (capture low-freq patterns)

3. COMPUTE CROSS-CORRELATIONS
   • All band pairs
   • Trained vs random initialization

4. FREQUENCY ANALYSIS
   • FFT of each band's time series
   • Find dominant frequencies

5. RATIO DETECTION
   • Compute frequency ratios between band pairs
   • Check for rational relationships

6. COMPARE TO RANDOM
   • Random init should show no rational relationships
   • Training should induce structure
```

### 5.2 Controls

- **Random initialization**: No phase locking expected
- **Shuffled activations**: Destroy temporal structure, check if ratios persist
- **Different seeds**: Verify ratios are consistent across training runs

---

## 6. Predictions

### 6.1 If Hypothesis is Correct

```
EXPECTED RESULTS:

1. RATIONAL RATIOS EMERGE
   • Trained model: frequency ratios ≈ simple fractions
   • Common ratios: 1:2, 1:4, 2:3, etc.
   • Random init: arbitrary ratios

2. SYMMETRIC PAIRS STRONGEST
   • 0↔6: Highest correlation (identity ↔ position)
   • 1↔5: Strong correlation (shape ↔ texture)
   • 2↔4: Strong correlation (structure ↔ detail)

3. BAND 3 MEDIATES
   • High correlation with all other bands
   • Acts as coupling channel

4. EMERGENCE DURING TRAINING
   • Early: low correlation, arbitrary ratios
   • Late: high correlation, rational ratios
   • Transition may be sudden (phase locking)
```

### 6.2 Quantitative Predictions

| Band Pair | Expected Correlation | Expected Ratio |
|-----------|---------------------|----------------|
| 0 ↔ 6 | > 0.5 | 1:64 or 1:32 |
| 1 ↔ 5 | > 0.4 | 1:16 or 1:8 |
| 2 ↔ 4 | > 0.4 | 1:4 or 1:2 |
| 3 ↔ any | > 0.3 | Variable |
| Random | < 0.1 | Arbitrary |

---

## 7. Falsification

### 7.1 What Would Disprove the Hypothesis

```
FALSIFICATION CRITERIA:

1. NO RATIONAL RELATIONSHIPS
   • Frequency ratios are irrational/arbitrary
   • No difference from random initialization
   → Bands do NOT phase lock

2. NO CORRELATION STRUCTURE
   • Symmetric pairs not special
   • Random correlations
   → No coupling between bands

3. NO EMERGENCE DURING TRAINING
   • Correlations stable from start
   • Or never develop
   → Training doesn't induce structure
```

### 7.2 Alternative Interpretations

If falsified, possible alternatives:
- Bands are more independent than expected
- Coupling is too weak for phase locking
- The coupling mechanism is different

---

## 8. Results

*To be filled after experiment*

### 8.1 Cross-Correlation Matrix

| Band | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|------|---|---|---|---|---|---|---|---|
| 0 | 1.0 | | | | | | | |
| 1 | | 1.0 | | | | | | |
| ... | | | | | | | | |

### 8.2 Frequency Ratios

| Pair | Trained Ratio | Random Ratio | Is Rational? |
|------|---------------|--------------|--------------|
| 0↔6 | | | |
| 1↔5 | | | |
| 2↔4 | | | |

### 8.3 Emergence During Training

*Space for training dynamics plots*

---

## 9. Conclusion

*To be filled after experiment*

### 9.1 Summary

### 9.2 Implications

### 9.3 Next Steps

---

## References

- `foundations/HARMONY_AND_COHERENCE.md` — Phase locking and coherence
- `003_EXP_SPECTRAL_BAND_DYNAMICS.md` — Band-specific dynamics
- `020_EXP_CROSS_BAND_FLOW.md` — Cross-band communication
- Physics literature on coupled oscillators and phase locking

---

*"Just as Jupiter's moons found 1:2:4 resonance over billions of years, spectral bands should find rational relationships through training. What survives is what harmonizes."*

