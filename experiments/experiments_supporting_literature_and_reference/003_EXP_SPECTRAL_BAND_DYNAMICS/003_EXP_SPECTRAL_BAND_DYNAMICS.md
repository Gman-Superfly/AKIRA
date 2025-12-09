# EXPERIMENT 003: Spectral Band Dynamics

## Do Different Bands Have Different Dynamics?

---

## Status: PENDING

## Depends On: 001_EXP_ENTROPY_OBSERVATION, 002_EXP_COLLAPSE_DETECTION

---

## 1. Problem Statement

### 1.1 The Question

Experiments 001-002 establish entropy measurement and collapse detection. Now:

**Do the 7 spectral bands exhibit demonstrably different dynamics — different entropy levels, different collapse timing, different learning rates in practice?**

### 1.2 Why This Matters

```
THE SPECTRAL HIERARCHY CLAIM

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  We claim:                                                              │
│  • Band 0 (DC) handles existence — slow, stable, low entropy          │
│  • Bands 1-2 (Low) handle structure — slow adaptation                 │
│  • Bands 3-4 (Mid) handle features — medium adaptation                │
│  • Bands 5-6 (High) handle details — fast, volatile, high entropy     │
│                                                                         │
│  If true:                                                               │
│  • Entropy should differ systematically across bands                  │
│  • Low bands should collapse first (structure before details)         │
│  • High bands should change faster during training                    │
│                                                                         │
│  If false:                                                              │
│  • All bands behave the same                                          │
│  • Spectral decomposition is cosmetic, not functional                 │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1.3 What Success Looks Like

If this experiment succeeds:
- Bands show statistically different entropy distributions
- Collapse cascades from low to high frequencies
- Learning dynamics differ measurably per band
- The spectral hierarchy is functional, not cosmetic

---

## 2. Hypothesis

### 2.1 Primary Hypothesis

**H1: Entropy increases monotonically from low to high bands.**

```
Expected ordering: H(Band 0) < H(Band 1) < ... < H(Band 6)

Reason: Low frequencies encode stable structure (certain).
        High frequencies encode transient details (uncertain).
```

### 2.2 Secondary Hypotheses

**H2: Collapse cascades from low to high bands.**
- Low bands collapse first (structure commits)
- High bands collapse later (details fill in)
- There is measurable temporal ordering

**H3: Weight change rate increases from low to high bands.**
- Low band weights change slowly (stable manifold)
- High band weights change quickly (volatile)

**H4: Different bands specialize in different content.**
- Low bands activate for global features
- High bands activate for local features

### 2.3 Null Hypotheses

**H0a:** All bands have same entropy distribution
**H0b:** Collapse timing is random across bands
**H0c:** Weight change rate is uniform across bands
**H0d:** Bands do not specialize

---

## 3. Scientific Basis

### 3.1 Fourier Frequency Interpretation

**ESTABLISHED SCIENCE:**

In Fourier analysis:
- Low frequencies = smooth, global, slowly varying
- High frequencies = sharp, local, rapidly varying

```
Physical interpretation:
- DC component: mean value (existence)
- Low freq: overall shape (identity)
- High freq: edges, textures (details)

This is mathematics, not theory.
Reference: Bracewell, R.N. (2000). The Fourier Transform and Its Applications.
```

### 3.2 Scale-Space Theory

**ESTABLISHED SCIENCE:**

Scale-space theory in computer vision:
- Coarse scales (low freq): category, layout
- Fine scales (high freq): texture, edge

```
Objects are detected coarse-to-fine:
1. First detect "blob" (is something there?)
2. Then detect "shape" (what category?)
3. Then detect "details" (what specific instance?)

Reference: Lindeberg, T. (1994). Scale-Space Theory in Computer Vision.
```

### 3.3 Neural Hierarchy

**ESTABLISHED SCIENCE:**

Visual cortex processes hierarchically:
- V1: edge detectors (high frequency)
- V2: contours, textures
- V4: object parts
- IT: object identity (low frequency, invariant)

```
Information flows from detail to abstraction:
High freq → processed first → feeds into low freq representations

Reference: DiCarlo, J.J. et al. (2012). How Does the Brain Solve Visual Object Recognition?
```

### 3.4 Wavelets and Multi-Resolution

**ESTABLISHED SCIENCE:**

Wavelet decomposition:
- Each level captures different frequency range
- Levels have different statistics
- Sparse at high frequencies, dense at low

Reference: Mallat, S. (1999). A Wavelet Tour of Signal Processing.

### 3.5 AKIRA Theory Basis

**Relevant Theory Documents:**
- `architecture_theoretical/SPECTRAL_BELIEF_MACHINE.md` — §2 (Differential Timescales), §5 (Spectral Decomposition)
- `architecture_theoretical/ORTHOGONALITY.md` — §4 (Spectral Band Orthogonality)
- `CANONICAL_PARAMETERS.md` — Learning rates Specification A, temperature values

**Key Concepts:**
- **Band orthogonality:** Fourier basis ensures bands are independent (∫ Band_i × Band_j = 0 for i≠j)
- **Differential learning rates:** Band 0 = 0.00001, Band 6 = 0.03 (3000× ratio)
- **Entropy ordering:** Predicted H(B0) < H(B1) < ... < H(B6)
- **Collapse cascade:** Low bands should collapse before high bands

**From SPECTRAL_BELIEF_MACHINE.md (§2):**
> "Band 0: LR = 0.00001 (protected identity). Band 6: LR = 0.03 (volatile details). Ratio: 3000× reflects natural timescales at which meaning operates at different scales."

**From ORTHOGONALITY.md (§4.3):**
> "Parseval's theorem: |Signal|² = |Band_0|² + ... + |Band_6|². Energy in each band is independent, additive, conserved."

**This experiment validates:**
1. Whether **7 bands have distinct dynamics** (architecture is functional)
2. Whether **entropy ordering** matches frequency hierarchy
3. Whether **collapse cascades** from structure to details
4. Whether **differential LR** produces measurably different learning rates

**Falsification:** If all bands behave identically → spectral architecture reduces to standard transformer → added complexity unjustified.

---

## 4. Apparatus

### 4.1 Required Components

```
SPECTRAL ANALYSIS INFRASTRUCTURE

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  COMPONENT              FUNCTION                       STATUS          │
│  ─────────              ────────                       ──────          │
│                                                                         │
│  entropy_tracker.py     From Experiment 001            REQUIRED        │
│  collapse_detector.py   From Experiment 002            REQUIRED        │
│  band_analyzer.py       Per-band statistics            TO BUILD        │
│  cascade_detector.py    Detect temporal ordering       TO BUILD        │
│  weight_tracker.py      Track weight changes           TO BUILD        │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Per-Band Entropy Analysis

```python
def analyze_band_entropy(
    model: nn.Module,
    sequence: torch.Tensor,
    num_bands: int = 7
) -> Dict[int, EntropyStats]:
    """
    Compute entropy statistics for each spectral band.
    """
    band_entropies = {i: [] for i in range(num_bands)}
    
    for t in range(len(sequence)):
        # Forward pass with hook capture
        output, attention = model.forward_with_attention(sequence[t])
        
        # Extract per-band attention
        for band_idx in range(num_bands):
            band_attention = extract_band_attention(attention, band_idx)
            H = compute_entropy(band_attention)
            band_entropies[band_idx].append(H)
    
    # Compute statistics per band
    stats = {}
    for band_idx, entropies in band_entropies.items():
        stats[band_idx] = EntropyStats(
            mean=np.mean(entropies),
            std=np.std(entropies),
            min=np.min(entropies),
            max=np.max(entropies)
        )
    
    return stats
```

### 4.3 Collapse Cascade Detection

```python
def detect_cascade(
    band_collapse_times: Dict[int, List[float]]
) -> CascadeAnalysis:
    """
    Analyze whether collapses cascade from low to high bands.
    """
    # For each collapse event, record which bands collapsed when
    cascade_order = []
    
    for event_id, times in enumerate(zip(*band_collapse_times.values())):
        # times = (t_band0, t_band1, ..., t_band6)
        order = np.argsort(times)  # Which band collapsed first?
        cascade_order.append(order)
    
    # Compute statistics
    # Perfect cascade: order is always [0, 1, 2, 3, 4, 5, 6]
    perfect_cascades = sum(
        1 for order in cascade_order
        if list(order) == list(range(7))
    )
    
    # Correlation: collapse time vs band index
    all_times = []
    all_bands = []
    for band_idx, times in band_collapse_times.items():
        all_times.extend(times)
        all_bands.extend([band_idx] * len(times))
    
    correlation = np.corrcoef(all_bands, all_times)[0, 1]
    
    return CascadeAnalysis(
        perfect_cascade_fraction=perfect_cascades / len(cascade_order),
        band_time_correlation=correlation
    )
```

---

## 5. Methods

### 5.1 Experimental Protocol

```
PROTOCOL: SPECTRAL BAND DYNAMICS

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  PHASE A: ENTROPY DISTRIBUTION PER BAND                                │
│  ──────────────────────────────────────                                 │
│  • Run model on diverse test sequences                                │
│  • Compute entropy for each band at each timestep                    │
│  • Aggregate statistics per band                                      │
│  • Test for monotonic ordering (H increases with band)               │
│                                                                         │
│  PHASE B: COLLAPSE CASCADE ANALYSIS                                    │
│  ───────────────────────────────────                                    │
│  • Run on sequences designed to trigger collapse                      │
│  • Record collapse time for each band                                │
│  • Analyze temporal ordering                                          │
│  • Test for low-to-high cascade pattern                              │
│                                                                         │
│  PHASE C: WEIGHT DYNAMICS PER BAND                                     │
│  ─────────────────────────────────                                      │
│  • Train model while recording weight changes per band               │
│  • Compute change rate: ‖Δw‖ / ‖w‖ per band                        │
│  • Test for increasing rate with band index                          │
│                                                                         │
│  PHASE D: SPECIALIZATION ANALYSIS                                       │
│  ────────────────────────────────                                       │
│  • Present stimuli at different spatial frequencies                  │
│  • Measure which bands activate                                       │
│  • Test for frequency-band correspondence                            │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 5.2 Test Stimuli

```
STIMULI FOR BAND SPECIALIZATION

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  STIMULUS               SPATIAL FREQ    EXPECTED ACTIVE BANDS          │
│  ────────               ───────────     ────────────────────           │
│                                                                         │
│  Uniform field          DC              Band 0                         │
│  Large blob             Very low        Bands 0-1                      │
│  Medium blob            Low             Bands 1-2                      │
│  Small blob             Medium          Bands 3-4                      │
│  Thin line              High            Bands 4-5                      │
│  Fine texture           Very high       Bands 5-6                      │
│  Sharp edge             Broadband       All bands                      │
│                                                                         │
│  Key test: Does stimulus freq match responding band?                  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 5.3 Statistical Tests

```
STATISTICAL ANALYSIS

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  TEST                  PURPOSE                      THRESHOLD          │
│  ────                  ───────                      ─────────          │
│                                                                         │
│  PHASE A:                                                               │
│  Kruskal-Wallis        Bands differ?                p < 0.05           │
│  Jonckheere-Terpstra   Monotonic ordering?         p < 0.05           │
│  Post-hoc pairwise     Which pairs differ?         Bonferroni         │
│                                                                         │
│  PHASE B:                                                               │
│  Spearman correlation  Cascade ordering?           ρ > 0.5, p < 0.05 │
│  Permutation test      Order non-random?           p < 0.05           │
│                                                                         │
│  PHASE C:                                                               │
│  Linear regression     Rate vs band?               R² > 0.7, p < 0.05│
│                                                                         │
│  PHASE D:                                                               │
│  Chi-square            Freq-band association?     p < 0.05           │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Predictions

### 6.1 If Theory Is Correct

```
EXPECTED RESULTS IF THEORY HOLDS

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  PHASE A: Entropy per band                                              │
│  ─────────────────────────                                              │
│  Band 0: Mean H ≈ 0.5-1.0 bits (very focused)                         │
│  Band 6: Mean H ≈ 3.0-4.0 bits (very spread)                          │
│  Monotonic increase confirmed                                          │
│                                                                         │
│  PHASE B: Collapse cascade                                              │
│  ─────────────────────────                                              │
│  Order: Band 0 collapses first, Band 6 last                           │
│  Correlation (band vs collapse time): ρ > 0.7                         │
│  Perfect cascades: > 50%                                               │
│                                                                         │
│  PHASE C: Weight dynamics                                               │
│  ────────────────────────                                               │
│  Band 0 weight change rate: ~0.01%                                    │
│  Band 6 weight change rate: ~1-10%                                    │
│  Linear relationship: R² > 0.8                                        │
│                                                                         │
│  PHASE D: Specialization                                                │
│  ───────────────────────                                                │
│  Low-freq stimuli activate low bands                                  │
│  High-freq stimuli activate high bands                                │
│  Chi-square significant                                                │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 6.2 Falsification Criteria

```
WHAT WOULD FALSIFY THE HYPOTHESIS

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  IF WE SEE:                               THEN:                        │
│  ───────────                              ─────                        │
│                                                                         │
│  All bands same entropy                   Spectral structure cosmetic  │
│  (Kruskal-Wallis p > 0.05)               (H1 false)                    │
│                                                                         │
│  No monotonic ordering                    Bands not hierarchical       │
│  (Jonckheere p > 0.05)                   (H1 false)                    │
│                                                                         │
│  Random collapse order                    No cascade                    │
│  (permutation p > 0.05)                  (H2 false)                    │
│                                                                         │
│  Uniform weight change rate               Differential LR not working  │
│  (regression R² < 0.3)                   (H3 false)                    │
│                                                                         │
│  No freq-band correspondence              Bands don't specialize       │
│  (Chi-square p > 0.05)                   (H4 false)                    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 7. Results

### 7.1 Phase A: Entropy Distribution

```
[ TO BE FILLED AFTER EXPERIMENT ]

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  BAND     MEAN H    STD H     MIN H     MAX H                          │
│  ────     ──────    ─────     ─────     ─────                          │
│                                                                         │
│  0        _____     _____     _____     _____                          │
│  1        _____     _____     _____     _____                          │
│  2        _____     _____     _____     _____                          │
│  3        _____     _____     _____     _____                          │
│  4        _____     _____     _____     _____                          │
│  5        _____     _____     _____     _____                          │
│  6        _____     _____     _____     _____                          │
│                                                                         │
│  Kruskal-Wallis H: _____   p-value: _____                             │
│  Jonckheere-Terpstra J: _____   p-value: _____                        │
│                                                                         │
│  Monotonic ordering: CONFIRMED / NOT CONFIRMED                        │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

[INSERT BOX PLOT: Entropy by Band]
```

### 7.2 Phase B: Collapse Cascade

```
[ TO BE FILLED AFTER EXPERIMENT ]

Total collapse events analyzed: _____
Perfect cascades (0→6 order): _____  (_____%)
Spearman ρ (band vs time): _____   p-value: _____
Permutation test p-value: _____

Cascade pattern: CONFIRMED / NOT CONFIRMED

[INSERT: Collapse time heatmap by band]
```

### 7.3 Phase C: Weight Dynamics

```
[ TO BE FILLED AFTER EXPERIMENT ]

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  BAND     MEAN Δw/w     STD         EXPECTED                           │
│  ────     ──────────    ───         ────────                           │
│                                                                         │
│  0        _____         _____       ~0.01%                             │
│  1        _____         _____       ~0.03%                             │
│  2        _____         _____       ~0.1%                              │
│  3        _____         _____       ~0.3%                              │
│  4        _____         _____       ~1%                                │
│  5        _____         _____       ~3%                                │
│  6        _____         _____       ~10%                               │
│                                                                         │
│  Linear regression: slope = _____   R² = _____   p = _____            │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 7.4 Phase D: Specialization

```
[ TO BE FILLED AFTER EXPERIMENT ]

Stimulus-Band Activation Matrix:
[INSERT HEATMAP]

Chi-square statistic: _____
p-value: _____

Specialization: CONFIRMED / NOT CONFIRMED
```

---

## 8. Conclusion

### 8.1 Summary

```
[ TO BE FILLED AFTER EXPERIMENT ]

H1 (entropy ordering): SUPPORTED / NOT SUPPORTED
H2 (collapse cascade): SUPPORTED / NOT SUPPORTED
H3 (weight dynamics): SUPPORTED / NOT SUPPORTED
H4 (specialization): SUPPORTED / NOT SUPPORTED

Overall: Spectral hierarchy is FUNCTIONAL / COSMETIC
```

### 8.2 Next Steps

```
[ TO BE FILLED AFTER EXPERIMENT ]

If hierarchy confirmed:
- Proceed to 004_EXP_PHASE_TRANSITION_SHARPNESS.md
- Spectral structure is real, test BEC predictions

If hierarchy not confirmed:
- Investigate why bands behave uniformly
- Consider architectural modifications
```

---

## References

1. Bracewell, R.N. (2000). The Fourier Transform and Its Applications. McGraw-Hill.
2. Lindeberg, T. (1994). Scale-Space Theory in Computer Vision. Springer.
3. DiCarlo, J.J. et al. (2012). How Does the Brain Solve Visual Object Recognition? Neuron.
4. Mallat, S. (1999). A Wavelet Tour of Signal Processing. Academic Press.

---



*"If the bands are truly different, the measurements will show it. If they are cosmetic, the statistics will reveal uniformity. We let the data speak."*

*Oscar Goldman — Shogu Research Group @ Datamutant.ai subsidiary of 温心重工業*

