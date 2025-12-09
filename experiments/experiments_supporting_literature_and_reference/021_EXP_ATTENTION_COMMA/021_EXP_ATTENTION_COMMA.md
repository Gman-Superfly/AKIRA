# Experiment 021: Attention Comma

## Is Error Distributed Like Equal Temperament?

**Tier:** ★★ CORE (Harmony and Coherence)  
**Status:** PLANNED  
**Depends On:** 001-003 (Entropy, Collapse, Band Dynamics)

---

## 1. Problem Statement

The Pythagorean comma reveals a fundamental truth: when mathematical perfection is impossible, optimal systems distribute error evenly rather than concentrating it.

In music, 12 perfect fifths don't equal 7 octaves (off by ~23 cents). Equal temperament spreads this error across all intervals, making every fifth slightly flat but allowing play in any key.

**Does attention do the same thing?**

When multiple positions compete for attention, the system cannot satisfy all demands perfectly. How is this "attention comma" distributed?

---

## 2. Hypothesis

```
THE ATTENTION COMMA HYPOTHESIS:

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  Like equal temperament, attention DISTRIBUTES ERROR evenly            │
│  rather than concentrating it in one position.                         │
│                                                                         │
│  MECHANISM:                                                             │
│  • Softmax forces normalization (attention must sum to 1)              │
│  • Multiple positions want "high" attention (conflicting demands)     │
│  • Perfect satisfaction impossible (the "comma")                       │
│  • Error spread across all positions (not concentrated)               │
│                                                                         │
│  TEMPERATURE EFFECT:                                                    │
│  • High τ: More uniform distribution (more "equal temperament")       │
│  • Low τ: More peaked distribution (more "pure intervals")            │
│  • Very low τ: May fail to generalize (circle doesn't close)         │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Scientific Basis

### 3.1 The Pythagorean Comma

```
MUSICAL MATHEMATICS:

12 perfect fifths: (3/2)^12 ≈ 129.746
7 perfect octaves: 2^7 = 128
Difference: ~23.5 cents (the "comma")

EQUAL TEMPERAMENT SOLUTION:
• Spread error: 23.5 / 12 ≈ 2 cents per fifth
• Every fifth slightly flat
• Circle closes, can play in any key
```

### 3.2 Softmax as Error Distribution

```
ATTENTION NORMALIZATION:

softmax(s)_i = exp(s_i) / Σ_j exp(s_j)

This FORCES Σ attention = 1
Like forcing the circle of fifths to close
By distributing the "error" across all positions
```

### 3.3 Prior Art

- Equal temperament in music (Pythagorean comma distribution)
- Load balancing in distributed systems
- Regularization in machine learning (weight decay distributes magnitude)

### 3.4 AKIRA Theory Basis

**Relevant Theory Documents:**
- `foundations/HARMONY_AND_COHERENCE.md`, §3 (Equal Temperament Analogy), §4 (Attention Comma)
- `architecture_theoretical/SPECTRAL_BELIEF_MACHINE.md`, §4 (Attention Mechanism)
- `CANONICAL_PARAMETERS.md`, Temperature τ range

**Key Concepts:**
- **Attention comma:** Normalization constraint (Σaᵢⱼ = 1) creates "error" that must be distributed
- **Softmax as equal temperament:** Spreads normalization error evenly across positions
- **Temperature control:** Low τ = peaked (pure intervals), high τ = uniform (equal temperament)
- **Harmony via error distribution:** Like 12-TET music, making all positions "slightly wrong" enables all positions to work

**From HARMONY_AND_COHERENCE.md (§4.2):**
> "Pythagorean comma: 12 perfect fifths ≠ 7 octaves (off by 23 cents). Equal temperament distributes error: each fifth slightly flat, but circle closes. Attention does same: softmax forces Σa=1 (circle must close). Pre-softmax scores want more than total=1 (comma). Softmax distributes error across all positions (equal temperament)."

**From SPECTRAL_BELIEF_MACHINE.md (§4.1):**
> "Temperature τ controls distribution. τ → 0: one-hot (breaks on ties). τ → ∞: uniform (loses information). Optimal τ balances sharpness and robustness. Like musical temperament: too pure → can't modulate, too equal → loses character."

**This experiment validates:**
1. Whether **error is distributed evenly** (not concentrated)
2. Whether **temperature controls distribution** (τ sweep)
3. Whether **error variance has minimum** (optimal τ exists)
4. Whether analogy to **equal temperament** is quantitative (not just metaphor)

**Falsification:** If error concentrates in few positions OR uniform distribution always optimal → attention doesn't operate like equal temperament → harmony analogy invalid.

---

## 4. Apparatus

### 4.1 Required Measurements

```
MEASUREMENT REQUIREMENTS:

1. PRE-SOFTMAX SCORES
   • Raw similarity scores s_i before normalization
   • These represent "ideal" attention per position

2. POST-SOFTMAX ATTENTION
   • Actual attention weights after softmax
   • These represent "tempered" attention

3. "IDEAL" ATTENTION
   • What each position would want if considered alone
   • No normalization constraint

4. ERROR DISTRIBUTION
   • Difference: |ideal_i - actual_i|
   • Distribution of errors across positions
```

### 4.2 Experimental Setup

```python
class AttentionCommaAnalyzer:
    """Measures attention comma distribution."""
    
    def compute_ideal_attention(self, scores: Tensor) -> Tensor:
        """What each position 'wants' in isolation."""
        # Each position's ideal is to receive its own softmax
        # But considering only itself
        return torch.sigmoid(scores)  # Per-position ideal
    
    def compute_actual_attention(self, scores: Tensor, tau: float) -> Tensor:
        """What each position actually gets."""
        return F.softmax(scores / tau, dim=-1)
    
    def compute_comma(self, ideal: Tensor, actual: Tensor) -> Tensor:
        """The 'comma' = difference between ideal and actual."""
        return ideal - actual
    
    def analyze_distribution(self, comma: Tensor) -> Dict:
        """How is the comma distributed?"""
        return {
            "mean_absolute": comma.abs().mean(),
            "std": comma.std(),
            "max": comma.abs().max(),
            "gini": self._gini_coefficient(comma.abs()),
            "entropy": self._distribution_entropy(comma.abs()),
        }
```

---

## 5. Method

### 5.1 Protocol

```
EXPERIMENTAL PROTOCOL:

1. SELECT TEST SEQUENCES
   • Choose sequences with clear competing attention demands
   • Multiple "important" positions that conflict
   
2. MEASURE IDEAL ATTENTION
   • For each position, compute its "ideal" attention
   • What it would receive if others didn't compete
   
3. MEASURE ACTUAL ATTENTION
   • Run softmax with temperature τ
   • Observe actual distribution
   
4. COMPUTE COMMA
   • Per-position: comma_i = ideal_i - actual_i
   • This is the "error" each position absorbs
   
5. ANALYZE DISTRIBUTION
   • Is error concentrated or spread?
   • How does distribution change with temperature?
   
6. VARY TEMPERATURE
   • Repeat at τ = {0.1, 0.5, 1.0, 2.0, 5.0}
   • Observe effect on distribution
```

### 5.2 Controls

- **Random baseline**: Random scores should show no systematic pattern
- **Single-peak control**: One dominant position should show concentrated comma
- **Temperature sweep**: Multiple temperatures to verify effect

---

## 6. Predictions

### 6.1 If Hypothesis is Correct

```
EXPECTED RESULTS:

1. ERROR IS DISTRIBUTED
   • Gini coefficient < 0.3 (low concentration)
   • No single position absorbs all error
   • Distribution across positions is relatively uniform

2. TEMPERATURE CONTROLS SPREADING
   • High τ → more uniform distribution (Gini ↓)
   • Low τ → more peaked distribution (Gini ↑)
   • Very low τ → may break (one position takes all)

3. PATTERN SIMILAR TO EQUAL TEMPERAMENT
   • Error per position ∝ 1/N (N positions)
   • Total error = comma (constant)
   • Each position "slightly flat"

4. FUNCTIONAL BENEFIT
   • Distributed error → more robust attention
   • Like equal temperament → can "play in any key"
```

### 6.2 Quantitative Predictions

| Metric | Prediction | Significance |
|--------|------------|--------------|
| Gini coefficient | < 0.3 | Low concentration |
| Error CV | < 1.0 | Relatively uniform |
| Temperature correlation | r > 0.7 | Strong τ-distribution relationship |
| Max/mean ratio | < 3.0 | No extreme outliers |

---

## 7. Falsification

### 7.1 What Would Disprove the Hypothesis

```
FALSIFICATION CRITERIA:

1. ERROR IS CONCENTRATED
   • Gini > 0.7 (high concentration)
   • One or few positions absorb most error
   → Attention does NOT distribute like equal temperament

2. NO TEMPERATURE EFFECT
   • Distribution unchanged with τ
   → Temperature doesn't control "tempering"

3. SYSTEMATIC PATTERN
   • Error concentrated at specific positions (e.g., edges)
   • Not random variation, but structured concentration
   → Different mechanism than comma distribution
```

### 7.2 Alternative Interpretations

If falsified, possible alternatives:
- Attention uses a different optimization strategy
- Error is absorbed by network depth, not position distribution
- The comma analogy is incomplete

---

## 8. Results

*To be filled after experiment*

### 8.1 Measured Distributions

| Temperature | Gini | CV | Max/Mean | Entropy |
|-------------|------|-----|----------|---------|
| 0.1 | | | | |
| 0.5 | | | | |
| 1.0 | | | | |
| 2.0 | | | | |
| 5.0 | | | | |

### 8.2 Visualizations

*Space for histograms, distribution plots, temperature effects*

---

## 9. Conclusion

*To be filled after experiment*

### 9.1 Summary

### 9.2 Implications

### 9.3 Next Steps

---

## References

- `foundations/HARMONY_AND_COHERENCE.md`, The Pythagorean principle
- `001_EXP_ENTROPY_OBSERVATION.md`, Entropy measurement foundation
- `003_EXP_SPECTRAL_BAND_DYNAMICS.md`, Band-specific dynamics
- Music theory: Equal temperament and the Pythagorean comma

---

*"The circle of fifths doesn't close. Neither does the circle of attention. The solution in both cases is the same: spread the error, achieve coherence."*

*Oscar Goldman, Shogu Research Group @ Datamutant.ai subsidiary of 温心重工業*