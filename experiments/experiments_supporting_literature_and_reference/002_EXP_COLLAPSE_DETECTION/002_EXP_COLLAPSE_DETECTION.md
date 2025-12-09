# EXPERIMENT 002: Collapse Event Detection

## Can We Detect When Belief Collapses?

---

## Status: PENDING

## Depends On: 001_EXP_ENTROPY_OBSERVATION (must succeed first)

---

## 1. Problem Statement

### 1.1 The Question

Experiment 001 establishes that we can measure attention entropy. Now:

**Can we detect sudden drops in entropy that correspond to "collapse" events — moments when the model transitions from uncertainty to certainty?**

### 1.2 Why This Matters

```
THE COLLAPSE HYPOTHESIS

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  We claim:                                                              │
│  • Belief dynamics exhibit SUDDEN transitions, not gradual change     │
│  • These transitions are "collapse" — many hypotheses → one          │
│  • Collapse is analogous to:                                          │
│    - Phase transitions (liquid → solid)                               │
│    - Lightning (many leaders → one path)                              │
│    - BEC condensation (many states → one ground state)                │
│                                                                         │
│  If true:                                                               │
│  • Entropy should drop SUDDENLY at collapse (not gradually)           │
│  • Drop rate should exceed what gradual learning produces            │
│  • Collapse should correlate with prediction improvement             │
│                                                                         │
│  If false:                                                              │
│  • Entropy changes are always gradual                                 │
│  • "Collapse" is just a metaphor, not a physical phenomenon          │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1.3 What Success Looks Like

If this experiment succeeds:
- We can automatically detect collapse events
- Collapse is demonstrably SUDDEN (not gradual)
- Collapse events are sparse (not constant)
- Collapse correlates with prediction quality improvement

---

## 2. Hypothesis

### 2.1 Primary Hypothesis

**H1: Collapse events are detectable as sudden entropy drops.**

Specifically:
- Entropy change rate dH/dt has occasional large negative spikes
- These spikes are statistically distinguishable from noise
- Spikes are sparse (not every timestep)

### 2.2 Secondary Hypotheses

**H2: Collapse events are SUDDEN, not gradual.**
- Time to collapse is short (few timesteps)
- Entropy drop rate during collapse >> average drop rate

**H3: Collapse events correlate with prediction improvement.**
- Error decreases when collapse occurs
- Error decrease is larger than typical variation

### 2.3 Null Hypotheses

**H0a:** Entropy changes are always gradual (no spikes)
**H0b:** Large entropy drops are random noise (uncorrelated with prediction)
**H0c:** Entropy drops are constant, not sparse (no distinct events)

---

## 3. Scientific Basis

### 3.1 Phase Transitions

**ESTABLISHED SCIENCE:**

Phase transitions exhibit discontinuities in derivatives of free energy:

```
First-order transitions: Discontinuity in first derivative (latent heat)
Second-order transitions: Discontinuity in second derivative (specific heat)

Characteristic: SUDDEN change at critical point
               Not gradual approach

Reference: Landau & Lifshitz (1980). Statistical Physics.
```

### 3.2 Critical Phenomena

**ESTABLISHED SCIENCE:**

Near phase transitions, systems exhibit:
- Power-law behavior: quantity ~ |T - T_c|^β
- Diverging correlation length
- Critical slowing down followed by sudden transition

Reference: Stanley, H.E. (1971). Introduction to Phase Transitions and Critical Phenomena.

### 3.3 AKIRA Theory Basis

**Relevant Theory Documents:**
- `bec/BEC_CONDENSATION_INFORMATION.md` — §4 (Phase Transition Dynamics)
- `architecture_theoretical/SPECTRAL_BELIEF_MACHINE.md` — §6 (Belief State Tracking)
- `CANONICAL_PARAMETERS.md` — collapse_threshold = 0.3

**Key Concepts:**
- **Collapse detection:** `dH/dt < -collapse_threshold` where collapse_threshold = 0.3
- **Temperature:** Controls sharpness of transition via softmax temperature τ
- **Belief collapse:** Sudden transition from diffuse to concentrated belief state

**From BEC_CONDENSATION_INFORMATION.md:**
> "Below critical uncertainty U_c, belief condenses suddenly. This is not gradual — it's a phase transition with sharp onset and finite-time collapse."

**From SPECTRAL_BELIEF_MACHINE.md:**
> "Collapse is detected via entropy rate dH/dt. When this exceeds threshold (large negative value), belief has transitioned from uncertain to committed state."

**This experiment validates:**
1. Whether collapse events are **sudden** (phase transition) or gradual (smooth learning)
2. Whether collapse correlates with prediction improvement
3. Whether collapse events are **sparse** (distinct transitions) or continuous

**Falsification:** If entropy always decreases gradually → no phase transition → BEC analogy invalid.

Deep networks can exhibit sudden generalization:
- Training loss plateaus
- Test loss suddenly drops
- Transition is sharp, not gradual

```
Power, A. et al. (2022). Grokking: Generalization Beyond Overfitting.

Key finding: Generalization can be SUDDEN after long training.
This is consistent with a phase transition interpretation.
```

### 3.4 Collapse Detection in Time Series

**ESTABLISHED SCIENCE:**

Detecting sudden changes in time series:
- Change point detection (CUSUM, PELT, etc.)
- Spike detection (threshold + derivative)
- Bayesian online change point detection

Reference: Aminikhanghahi, S. & Cook, D.J. (2017). A Survey of Methods for Time Series Change Point Detection.

---

## 4. Apparatus

### 4.1 Required Components

```
COLLAPSE DETECTION INFRASTRUCTURE

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  COMPONENT              FUNCTION                       STATUS          │
│  ─────────              ────────                       ──────          │
│                                                                         │
│  entropy_tracker.py     From Experiment 001            REQUIRED        │
│  collapse_detector.py   Detect sudden drops            TO BUILD        │
│  spike_statistics.py    Statistical analysis           TO BUILD        │
│  temporal_alignment.py  Align collapse with events     TO BUILD        │
│                                                                         │
│  AKIRA model            Subject of measurement         EXISTS          │
│  Test sequences         Ambiguous inputs               TO PREPARE      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Collapse Detection Algorithm

```python
def detect_collapse_events(
    entropy_series: torch.Tensor,
    threshold_std: float = 2.0,
    min_drop: float = 0.5,
    window_size: int = 5
) -> List[CollapseEvent]:
    """
    Detect sudden entropy drops (collapse events).
    
    Args:
        entropy_series: Entropy values over time [T]
        threshold_std: How many std below mean dH/dt counts as collapse
        min_drop: Minimum absolute entropy drop to count
        window_size: Smoothing window for derivative estimation
    
    Returns:
        List of CollapseEvent(time, magnitude, duration)
    """
    # Compute entropy change rate
    dH = torch.diff(entropy_series)
    
    # Compute rolling statistics
    mean_dH = dH.mean()
    std_dH = dH.std()
    
    # Threshold for collapse detection
    collapse_threshold = mean_dH - threshold_std * std_dH
    
    # Find collapse events (large negative changes)
    is_collapse = (dH < collapse_threshold) & (dH < -min_drop)
    
    # Extract events
    events = []
    in_event = False
    event_start = None
    
    for t, is_col in enumerate(is_collapse):
        if is_col and not in_event:
            in_event = True
            event_start = t
        elif not is_col and in_event:
            in_event = False
            magnitude = entropy_series[event_start] - entropy_series[t]
            duration = t - event_start
            events.append(CollapseEvent(
                time=event_start,
                magnitude=magnitude,
                duration=duration
            ))
    
    return events
```

### 4.3 Suddenness Metric

```python
def measure_suddenness(
    entropy_series: torch.Tensor,
    collapse_event: CollapseEvent
) -> float:
    """
    Measure how sudden a collapse is.
    
    Suddenness = (entropy drop) / (time to drop)
    Higher = more sudden
    
    Compare to background rate to determine if significantly sudden.
    """
    magnitude = collapse_event.magnitude
    duration = collapse_event.duration
    
    if duration == 0:
        return float('inf')  # Instantaneous
    
    collapse_rate = magnitude / duration
    
    # Compare to background change rate
    background_rate = entropy_series.diff().abs().mean()
    
    suddenness_ratio = collapse_rate / (background_rate + 1e-10)
    
    return suddenness_ratio
```

---

## 5. Methods

### 5.1 Experimental Protocol

```
PROTOCOL: COLLAPSE DETECTION

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  STEP 1: SETUP                                                          │
│  ────────────────                                                       │
│  • Load trained AKIRA model                                            │
│  • Enable entropy tracking from Experiment 001                        │
│  • Prepare ambiguous test sequences                                   │
│                                                                         │
│  STEP 2: DATA COLLECTION                                                │
│  ───────────────────────                                                │
│  • Run model on test sequences                                        │
│  • Record entropy at every timestep                                   │
│  • Record prediction error at every timestep                         │
│  • Record attention weights for visualization                        │
│                                                                         │
│  STEP 3: COLLAPSE DETECTION                                             │
│  ──────────────────────────                                             │
│  • Apply collapse detection algorithm                                 │
│  • Record: time, magnitude, duration of each event                   │
│  • Compute suddenness metric for each event                          │
│                                                                         │
│  STEP 4: STATISTICAL ANALYSIS                                           │
│  ────────────────────────────                                           │
│  • Test if collapse events are sparse (not constant)                 │
│  • Test if collapses are sudden (vs gradual baseline)                │
│  • Test if collapses correlate with error reduction                 │
│                                                                         │
│  STEP 5: VISUALIZATION                                                  │
│  ─────────────────────                                                  │
│  • Plot entropy time series with collapse markers                    │
│  • Plot attention before/during/after collapse                       │
│  • Create animations of collapse events                               │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 5.2 Test Sequences Designed for Collapse

```
SEQUENCES DESIGNED TO INDUCE COLLAPSE

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  SEQUENCE              DESIGNED TO                  EXPECTED COLLAPSE  │
│  ────────              ───────────                  ─────────────────  │
│                                                                         │
│  Ambiguous direction   Model uncertain which way   When direction     │
│  (pause then move)     blob will move              becomes clear      │
│                                                                         │
│  Two blobs merging     Uncertainty about identity  When they merge    │
│                        after overlap                or separate       │
│                                                                         │
│  Appearing blob        Object existence uncertain  When blob appears  │
│  (fade in)             until visible                clearly           │
│                                                                         │
│  Reversing blob        Prediction of continuation  At reversal point  │
│  (bounce)              violated                                       │
│                                                                         │
│  Splitting blob        One → two, which to track?  When split occurs │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 5.3 Statistical Tests

```
STATISTICAL ANALYSIS

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  TEST                PURPOSE                        THRESHOLD           │
│  ────                ───────                        ─────────           │
│                                                                         │
│  Sparsity test       Collapses are rare events?    < 20% of timesteps │
│  (fraction of time                                                     │
│   in collapse)                                                         │
│                                                                         │
│  Suddenness test     Collapse rate >> background?  Ratio > 3          │
│  (collapse rate /                                                      │
│   background rate)                                                     │
│                                                                         │
│  Error correlation   Error drops at collapse?      r > 0.3, p < 0.05 │
│  (error change at                                                      │
│   collapse events)                                                     │
│                                                                         │
│  Timing test         Collapses align with events?  Chi-square p < 0.05│
│  (collapse times vs.                                                   │
│   known event times)                                                   │
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
│  OBSERVATION                              EXPECTED VALUE                │
│  ───────────                              ──────────────                │
│                                                                         │
│  Collapse events per sequence             1-5 distinct events          │
│  Collapse duration                        1-3 timesteps (short)        │
│  Suddenness ratio                         > 3× background              │
│  Fraction of time in collapse             < 20%                        │
│  Error reduction at collapse              > 30% improvement            │
│  Alignment with ambiguity resolution      > 70% match                  │
│                                                                         │
│  Qualitative signature:                                                 │
│  • Entropy builds up → sudden drop → entropy stable → builds again   │
│  • Not: gradual monotonic decrease                                    │
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
│  No distinct collapse events              Collapse is metaphor only    │
│  (entropy always changes gradually)       (H1 false)                   │
│                                                                         │
│  Suddenness ratio < 1.5                   Collapses are not sudden    │
│                                            (H2 false)                   │
│                                                                         │
│  Collapse frequent (> 50% of time)        Not sparse events            │
│                                            (collapse constant, not event)│
│                                                                         │
│  No error correlation                     Collapse is noise            │
│                                            (H3 false)                   │
│                                                                         │
│  ANY of these would falsify the collapse hypothesis.                  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 7. Results

### 7.1 Collapse Event Statistics

```
[ TO BE FILLED AFTER EXPERIMENT ]

Total sequences analyzed: _______
Total collapse events detected: _______
Events per sequence (mean ± std): _______

Sparsity:
  Fraction of timesteps in collapse: _______
  Verdict (< 20%?): _______
```

### 7.2 Suddenness Analysis

```
[ TO BE FILLED AFTER EXPERIMENT ]

Collapse rate (mean ± std): _______ bits/timestep
Background rate (mean ± std): _______ bits/timestep
Suddenness ratio (mean): _______
Verdict (ratio > 3?): _______

Distribution of suddenness ratios: [INSERT HISTOGRAM]
```

### 7.3 Error Correlation

```
[ TO BE FILLED AFTER EXPERIMENT ]

Error before collapse (mean): _______
Error after collapse (mean): _______
Error reduction: _______
Correlation (r): _______
p-value: _______
Verdict (significant?): _______
```

### 7.4 Timing Alignment

```
[ TO BE FILLED AFTER EXPERIMENT ]

Sequence-specific results:

Ambiguous direction:
  Expected collapse time: _______
  Observed collapse time: _______
  Alignment: _______

Two blobs merging:
  Expected collapse time: _______
  Observed collapse time: _______
  Alignment: _______

[etc. for each sequence]

Chi-square test p-value: _______
Verdict (collapses align with events?): _______
```

### 7.5 Visualizations

```
[ TO BE FILLED AFTER EXPERIMENT ]

Insert:
- Entropy time series with collapse markers
- Attention heatmaps before/during/after collapse
- Error time series with collapse markers
- Animation links
```

---

## 8. Conclusion

### 8.1 Summary

```
[ TO BE FILLED AFTER EXPERIMENT ]

H1 (collapse events detectable): SUPPORTED / NOT SUPPORTED
H2 (collapses are sudden): SUPPORTED / NOT SUPPORTED
H3 (collapses correlate with error): SUPPORTED / NOT SUPPORTED
```

### 8.2 Implications

```
[ TO BE FILLED AFTER EXPERIMENT ]

If supported:
- Collapse is a real phenomenon, not metaphor
- Phase transition interpretation is viable
- Proceed to Experiment 003 (Spectral Band Dynamics)

If not supported:
- "Collapse" may be gradual optimization
- BEC/phase transition analogy may not hold
- Simpler gradient-based explanation may suffice
```

### 8.3 Next Steps

```
[ TO BE FILLED AFTER EXPERIMENT ]

If successful, proceed to:
- 003_EXP_SPECTRAL_BAND_DYNAMICS.md
```

---

## References

1. Landau, L.D. & Lifshitz, E.M. (1980). Statistical Physics. Pergamon Press.
2. Stanley, H.E. (1971). Introduction to Phase Transitions and Critical Phenomena. Oxford.
3. Power, A. et al. (2022). Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets.
4. Aminikhanghahi, S. & Cook, D.J. (2017). A Survey of Methods for Time Series Change Point Detection. Knowledge and Information Systems.

---

*Oscar Goldman — Shogu Research Group @ Datamutant.ai*

*"If collapse is real, it must be detectable. If it is metaphor, we will find gradual change. The experiment will tell us which."*


