# EXPERIMENT 006: Heresy Detection

## Are Our Measurements Corrupted by Processing Artifacts?

---

## Status: PENDING

## Depends On: 001-005 (must have working measurement before testing for corruption)

---

## 1. Problem Statement

### 1.1 The Question

Before trusting any results, we must ask:

**Are our observations true knowledge, or heresy, patterns that resonate with the architecture rather than reality?**

### 1.2 Why This Matters

```
THE HERESY PROBLEM

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  From FALSE_PROPHETS.md:                                               │
│                                                                         │
│  HERESIES are processing artifacts that the model believes:           │
│  • Aliasing: High frequencies fold into low (Nyquist violation)       │
│  • Spectral leakage: FFT discontinuities create false frequencies    │
│  • Boundary effects: Edges treated specially due to processing       │
│                                                                         │
│  The model has no oracle for truth. It believes what it sees.         │
│  If we show it artifacts, it learns artifacts.                        │
│                                                                         │
│  THE INQUISITION:                                                       │
│  Experiments that distinguish true knowledge from heresy              │
│                                                                         │
│  Without this, all our "discoveries" might be artifacts.              │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1.3 What Success Looks Like

If this experiment succeeds:
- We can distinguish true patterns from artifacts
- Windowing reduces spectral leakage to measurable degree
- Aliasing is detectable and quantifiable
- We know which measurements are reliable

---

## 2. Hypothesis

### 2.1 Primary Hypothesis

**H1: Spectral leakage is present and measurable in unwindowed FFT.**

Without windowing, FFT produces artifacts at frequencies not present in the signal.

### 2.2 Secondary Hypotheses

**H2: Windowing (Hamming, Hanning, etc.) reduces spectral leakage.**

Proper windowing should reduce sidelobe levels by 20+ dB.

**H3: Aliasing is detectable when signal violates Nyquist.**

If input contains frequencies > N/2, aliased components appear.

**H4: Boundary effects create measurable artifacts at image edges.**

Edge pixels have systematically different statistics than center pixels.

**H5: Model beliefs correlate with artifacts when artifacts are strong.**

If we deliberately introduce artifacts, model "learns" them.

### 2.3 Null Hypotheses

**H0a:** No spectral leakage (FFT is perfect)
**H0b:** Windowing has no effect
**H0c:** Aliasing is not detectable
**H0d:** Edges behave like centers

---

## 3. Scientific Basis

### 3.1 Spectral Leakage

**ESTABLISHED SCIENCE:**

Discrete Fourier Transform assumes periodic signal. Non-periodic signals cause leakage:

```
DFT of non-periodic signal:
• Main lobe at true frequency
• Side lobes spreading to other frequencies
• "Leakage" = energy appearing at wrong frequencies

This is NOT an approximation, it is mathematical fact.

Reference: Oppenheim & Schafer (2009). Discrete-Time Signal Processing.
```

### 3.2 Windowing Functions

**ESTABLISHED SCIENCE:**

Window functions reduce leakage by tapering signal edges:

```
WINDOW          SIDELOBE LEVEL    MAIN LOBE WIDTH
──────          ──────────────    ───────────────
Rectangular     -13 dB            Narrow
Hanning         -31 dB            Wider
Hamming         -43 dB            Wider
Blackman        -58 dB            Widest

Trade-off: Better sidelobe suppression = wider main lobe

Reference: Harris, F.J. (1978). On the Use of Windows for Harmonic Analysis with the DFT.
```

### 3.3 Nyquist-Shannon Sampling Theorem

**ESTABLISHED SCIENCE:**

```
To faithfully represent frequency f:
Sample rate > 2f (Nyquist rate)

If violated:
• High frequency f aliases to lower frequency
• Appears as f_alias = |f - n × f_sample| for some n
• INDISTINGUISHABLE from true low frequency

This is a theorem, not an approximation.

Reference: Shannon, C.E. (1949). Communication in the Presence of Noise.
```

### 3.4 Boundary Effects in Images

**ESTABLISHED SCIENCE:**

Image edges are artificial boundaries:
- Real scenes extend beyond frame
- Processing creates edge artifacts (ringing, reflection)
- CNNs show different activation patterns at edges

Reference: Innamorati et al. (2020). Learning on the Edge.

### 3.5 AKIRA Theory Basis

**Relevant Theory Documents:**
- `foundations/FALSE_PROPHETS.md`, §2 (Heresy Catalog), §3 (Inquisition Protocol)
- `architecture_theoretical/SPECTRAL_BELIEF_MACHINE.md`, §5 (FFT and Windowing), §7 (Conservation Laws)
- `foundations/ORTHOGONALITY.md`, §6 (Nyquist and Aliasing)

**Key Concepts:**
- **Heresy:** Processing artifacts that model believes (aliasing, spectral leakage, boundary effects)
- **Inquisition:** Experiments distinguishing true patterns from artifacts
- **Spectral leakage:** FFT discontinuities create false frequencies (sidelobes)
- **Aliasing:** Frequencies above Nyquist fold back into observable range
- **Boundary effects:** Edge pixels treated differently due to processing constraints

**From FALSE_PROPHETS.md (§2.1):**
> "Heresies are patterns that resonate with architecture rather than reality. Model has no oracle for truth. If we show it artifacts, it learns artifacts. Three primary heresies: (1) Aliasing, high freq folds to low, (2) Spectral leakage, FFT edge discontinuities create false freq, (3) Boundary effects, edges processed differently."

**From SPECTRAL_BELIEF_MACHINE.md (§5.2):**
> "Windowing reduces spectral leakage. Hamming window: -43 dB sidelobes. Blackman: -58 dB but wider main lobe. Tradeoff: frequency resolution vs leakage suppression. Without windowing, all spectral measurements are contaminated."

**From ORTHOGONALITY.md (§6.1):**
> "Nyquist theorem: max representable frequency = sampling_rate / 2. Frequencies above Nyquist alias to lower frequencies. Band 6 captures [f_max/2, f_max]. Input containing freq > f_max produces aliasing artifacts in all bands."

**This experiment validates:**
1. Whether **spectral leakage is measurable** (quantify sidelobe levels)
2. Whether **windowing reduces leakage** (compare windowed vs unwindowed)
3. Whether **aliasing is detectable** (identify folded frequencies)
4. Whether **model learns artifacts** (correlation between artifact strength and belief)

**Falsification:** If no measurable artifacts OR model doesn't learn them → processing is clean → heresy detection unnecessary.

**Critical:** This is meta-experiment validating measurement infrastructure. Must succeed before trusting spectral analysis.

---

## 4. Apparatus

### 4.1 Required Components

```
HERESY DETECTION INFRASTRUCTURE

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  COMPONENT                  FUNCTION                    STATUS         │
│  ─────────                  ────────                    ──────         │
│                                                                         │
│  leakage_detector.py        Measure spectral leakage    TO BUILD       │
│  aliasing_detector.py       Detect aliased frequencies  TO BUILD       │
│  window_comparator.py       Compare window functions    TO BUILD       │
│  boundary_analyzer.py       Edge vs center statistics   TO BUILD       │
│  artifact_injector.py       Deliberately add artifacts  TO BUILD       │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Spectral Leakage Measurement

```python
def measure_spectral_leakage(
    signal: torch.Tensor,
    true_frequency: float,
    window: Optional[str] = None
) -> LeakageMetrics:
    """
    Measure spectral leakage for a pure sinusoid.
    
    True signal: single frequency f
    Ideal FFT: single peak at f
    Leakage: energy at frequencies ≠ f
    """
    if window:
        signal = apply_window(signal, window)
    
    # FFT
    spectrum = torch.fft.fft(signal)
    power = torch.abs(spectrum) ** 2
    
    # Find peak (should be at true_frequency)
    peak_idx = power.argmax()
    peak_power = power[peak_idx]
    
    # Leakage = power outside main lobe
    main_lobe_width = 3  # bins
    main_lobe_mask = torch.zeros_like(power, dtype=bool)
    main_lobe_mask[peak_idx - main_lobe_width : peak_idx + main_lobe_width + 1] = True
    
    leakage_power = power[~main_lobe_mask].sum()
    total_power = power.sum()
    
    leakage_ratio = leakage_power / total_power
    leakage_db = 10 * torch.log10(leakage_ratio + 1e-10)
    
    return LeakageMetrics(
        peak_frequency=peak_idx,
        true_frequency=true_frequency,
        leakage_ratio=leakage_ratio.item(),
        leakage_db=leakage_db.item()
    )
```

### 4.3 Aliasing Detection

```python
def detect_aliasing(
    signal: torch.Tensor,
    sample_rate: float,
    test_frequencies: List[float]
) -> AliasingResults:
    """
    Test for aliasing by injecting known high frequencies.
    """
    results = []
    nyquist = sample_rate / 2
    
    for f in test_frequencies:
        # Create test signal at frequency f
        t = torch.arange(len(signal)) / sample_rate
        test_signal = torch.sin(2 * np.pi * f * t)
        
        # FFT
        spectrum = torch.fft.fft(test_signal)
        power = torch.abs(spectrum) ** 2
        
        # Expected peak location
        if f <= nyquist:
            expected_bin = int(f / sample_rate * len(signal))
            is_aliased = False
        else:
            # Aliased frequency
            aliased_f = abs(f - sample_rate * round(f / sample_rate))
            expected_bin = int(aliased_f / sample_rate * len(signal))
            is_aliased = True
        
        # Find actual peak
        actual_bin = power.argmax().item()
        
        results.append(AliasingResult(
            input_frequency=f,
            expected_bin=expected_bin,
            actual_bin=actual_bin,
            is_aliased=is_aliased,
            detected_correctly=(actual_bin == expected_bin)
        ))
    
    return AliasingResults(results)
```

### 4.4 Boundary Effect Analysis

```python
def analyze_boundary_effects(
    image: torch.Tensor,
    model: nn.Module,
    border_width: int = 5
) -> BoundaryAnalysis:
    """
    Compare model behavior at edges vs. center.
    """
    H, W = image.shape[-2:]
    
    # Create masks
    edge_mask = torch.zeros(H, W, dtype=bool)
    edge_mask[:border_width, :] = True
    edge_mask[-border_width:, :] = True
    edge_mask[:, :border_width] = True
    edge_mask[:, -border_width:] = True
    
    center_mask = ~edge_mask
    
    # Get model predictions/attention
    output, attention = model.forward_with_attention(image)
    
    # Compare statistics
    edge_attention = attention[..., edge_mask].mean()
    center_attention = attention[..., center_mask].mean()
    
    edge_entropy = compute_entropy(attention[..., edge_mask])
    center_entropy = compute_entropy(attention[..., center_mask])
    
    edge_error = compute_error(output[..., edge_mask], ...)
    center_error = compute_error(output[..., center_mask], ...)
    
    return BoundaryAnalysis(
        edge_attention=edge_attention,
        center_attention=center_attention,
        edge_entropy=edge_entropy,
        center_entropy=center_entropy,
        edge_error=edge_error,
        center_error=center_error,
        edge_center_ratio=edge_error / (center_error + 1e-10)
    )
```

---

## 5. Methods

### 5.1 Experimental Protocol

```
PROTOCOL: HERESY DETECTION (THE INQUISITION)

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  PHASE A: SPECTRAL LEAKAGE TEST                                        │
│  ──────────────────────────────                                         │
│  • Create pure sinusoid at known frequency                            │
│  • FFT with no window, measure leakage                                │
│  • FFT with each window type, measure leakage                        │
│  • Compare: which window best suppresses leakage?                    │
│                                                                         │
│  PHASE B: ALIASING TEST                                                 │
│  ──────────────────────                                                 │
│  • Create signals at frequencies below Nyquist                        │
│  • Create signals at frequencies above Nyquist                        │
│  • FFT and find peaks                                                 │
│  • Verify aliasing occurs as predicted                               │
│                                                                         │
│  PHASE C: WINDOWING ABLATION                                           │
│  ───────────────────────────                                            │
│  • Train model with no windowing                                      │
│  • Train model with Hamming window                                    │
│  • Compare: attention patterns, prediction error, entropy            │
│  • Does windowing reduce heresy?                                     │
│                                                                         │
│  PHASE D: BOUNDARY EFFECT TEST                                          │
│  ─────────────────────────────                                          │
│  • Compare attention/error at edges vs. center                       │
│  • Test if edges are systematically different                        │
│  • Is the difference due to content or processing?                   │
│                                                                         │
│  PHASE E: ARTIFACT INJECTION TEST                                       │
│  ────────────────────────────────                                       │
│  • Deliberately add artifacts to training data                       │
│  • Train model                                                        │
│  • Test: does model believe artifacts are real?                      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 5.2 Window Functions to Test

```
WINDOW FUNCTIONS FOR COMPARISON

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  WINDOW         FORMULA                      EXPECTED LEAKAGE         │
│  ──────         ───────                      ────────────────         │
│                                                                         │
│  Rectangular    w[n] = 1                     -13 dB (worst)           │
│                                                                         │
│  Hanning        w[n] = 0.5(1 - cos(2πn/N))   -31 dB                   │
│                                                                         │
│  Hamming        w[n] = 0.54 - 0.46cos(2πn/N) -43 dB                   │
│                                                                         │
│  Blackman       w[n] = 0.42 - 0.5cos(2πn/N)  -58 dB (best)           │
│                        + 0.08cos(4πn/N)                               │
│                                                                         │
│  Gaussian       w[n] = exp(-n²/2σ²)          Depends on σ             │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 5.3 Statistical Tests

```
STATISTICAL ANALYSIS

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  TEST                PURPOSE                        THRESHOLD          │
│  ────                ───────                        ─────────          │
│                                                                         │
│  Leakage reduction   Window reduces leakage?        > 10 dB reduction │
│                                                                         │
│  Aliasing detection  Above-Nyquist folds correctly? Peak at alias freq│
│                                                                         │
│  Edge vs center      Edges different from center?   t-test p < 0.05   │
│  (t-test)                                                              │
│                                                                         │
│  Artifact learning   Model believes artifacts?      Error on artifacts│
│                                                      < error on real  │
│                                                                         │
│  Windowing effect    Windowing helps prediction?    Error reduction   │
│  on model                                            > 5%              │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Predictions

### 6.1 If Heresy Is Real Problem

```
EXPECTED RESULTS IF ARTIFACTS AFFECT MODEL

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  PHASE A: Spectral leakage                                              │
│  • Rectangular: -13 dB leakage (severe)                               │
│  • Hamming: -43 dB leakage (much better)                              │
│  • Leakage IS measurable and significant                              │
│                                                                         │
│  PHASE B: Aliasing                                                      │
│  • Above-Nyquist signals alias as predicted                          │
│  • Aliased peaks at expected frequencies                              │
│                                                                         │
│  PHASE C: Windowing ablation                                            │
│  • Windowed model has lower prediction error                          │
│  • Windowed model has cleaner spectral structure                     │
│  • Difference is measurable (> 5% error reduction)                   │
│                                                                         │
│  PHASE D: Boundary effects                                              │
│  • Edges have higher error than center                                │
│  • Edges have different entropy than center                          │
│  • Difference is statistically significant                           │
│                                                                         │
│  PHASE E: Artifact learning                                             │
│  • Model learns to predict artifacts                                  │
│  • "Believes" false patterns are real                                │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 6.2 Falsification Criteria

```
WHAT WOULD FALSIFY THE HYPOTHESES

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  IF WE SEE:                               THEN:                        │
│  ───────────                              ─────                        │
│                                                                         │
│  No measurable leakage                    FFT implementation perfect  │
│  (leakage < -80 dB)                       (unlikely, check test)      │
│                                                                         │
│  Windowing has no effect                  Leakage not important here  │
│  (error difference < 1%)                  (H2 false for this system)  │
│                                                                         │
│  Edges same as center                     Boundary effects minimal    │
│  (t-test p > 0.05)                        (H4 false)                   │
│                                                                         │
│  Model rejects artifacts                  Model has artifact immunity │
│  (higher error on artifacts)              (robust to heresy)          │
│                                                                         │
│  Note: Some of these would be GOOD news (system is robust).          │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 7. Results

### 7.1 Phase A: Spectral Leakage

```
[ TO BE FILLED AFTER EXPERIMENT ]

Test signal: sinusoid at frequency f = _____

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  WINDOW         LEAKAGE (dB)    EXPECTED (dB)    MATCH?               │
│  ──────         ────────────    ─────────────    ──────               │
│                                                                         │
│  Rectangular    _____           -13              _____                 │
│  Hanning        _____           -31              _____                 │
│  Hamming        _____           -43              _____                 │
│  Blackman       _____           -58              _____                 │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

[INSERT: Spectrum comparison plot]

Best window for this application: _____
```

### 7.2 Phase B: Aliasing Detection

```
[ TO BE FILLED AFTER EXPERIMENT ]

Sample rate: _____ Hz
Nyquist frequency: _____ Hz

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  INPUT FREQ    ABOVE NYQUIST?    EXPECTED PEAK    ACTUAL PEAK         │
│  ──────────    ──────────────    ─────────────    ───────────         │
│                                                                         │
│  _____         NO                _____            _____                │
│  _____         NO                _____            _____                │
│  _____         YES               _____ (aliased)  _____                │
│  _____         YES               _____ (aliased)  _____                │
│                                                                         │
│  Aliasing confirmed: YES / NO                                          │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 7.3 Phase C: Windowing Ablation

```
[ TO BE FILLED AFTER EXPERIMENT ]

Model trained WITHOUT windowing:
• Mean prediction error: _____
• Entropy statistics: _____

Model trained WITH Hamming window:
• Mean prediction error: _____
• Entropy statistics: _____

Error reduction from windowing: _____% 
Statistically significant? YES / NO (p = _____)

Windowing recommended: YES / NO
```

### 7.4 Phase D: Boundary Effects

```
[ TO BE FILLED AFTER EXPERIMENT ]

Border width tested: _____ pixels

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  METRIC          EDGE VALUE    CENTER VALUE    RATIO      p-VALUE     │
│  ──────          ──────────    ────────────    ─────      ───────     │
│                                                                         │
│  Mean attention  _____         _____           _____      _____        │
│  Mean entropy    _____         _____           _____      _____        │
│  Mean error      _____         _____           _____      _____        │
│                                                                         │
│  Edges significantly different from center: YES / NO                  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 7.5 Phase E: Artifact Learning

```
[ TO BE FILLED AFTER EXPERIMENT ]

Artifact type injected: _____
Fraction of training data with artifacts: _____%

Model prediction on artifact patterns:
• Error (lower = model believes artifact): _____

Model prediction on true patterns:
• Error: _____

Model believes artifacts are real: YES / NO
```

---

## 8. Conclusion

### 8.1 Summary

```
[ TO BE FILLED AFTER EXPERIMENT ]

H1 (leakage measurable): CONFIRMED / NOT CONFIRMED
H2 (windowing helps): CONFIRMED / NOT CONFIRMED
H3 (aliasing detectable): CONFIRMED / NOT CONFIRMED
H4 (boundary effects): CONFIRMED / NOT CONFIRMED
H5 (model learns artifacts): CONFIRMED / NOT CONFIRMED

Heresy is a real problem: YES / NO
Recommended mitigations: _____
```

### 8.2 Implications

```
[ TO BE FILLED AFTER EXPERIMENT ]

If heresy confirmed:
- Use windowing in all spectral operations
- Be cautious about edge predictions
- Monitor for aliasing in high-frequency inputs
- All previous experiments may need re-evaluation

If heresy minimal:
- System is robust to common artifacts
- Proceed with confidence to advanced experiments
```

---

## References

1. Oppenheim, A.V. & Schafer, R.W. (2009). Discrete-Time Signal Processing. Pearson.
2. Harris, F.J. (1978). On the Use of Windows for Harmonic Analysis with the DFT. Proceedings of the IEEE.
3. Shannon, C.E. (1949). Communication in the Presence of Noise. Proceedings of the IRE.
4. Innamorati, C. et al. (2020). Learning on the Edge: Investigating Boundary Predictors.

---



*"The inquisition is not persecution, it is protection. We seek heresy not to punish the model, but to protect ourselves from false beliefs. If our measurements are artifacts, we must know before we build theory upon them."*


*Oscar Goldman, Shogu Research Group @ Datamutant.ai subsidiary of 温心重工業*