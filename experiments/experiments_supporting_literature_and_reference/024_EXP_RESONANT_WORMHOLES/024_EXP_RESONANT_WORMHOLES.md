# Experiment 024: Resonant Wormholes

## Do Wormholes Prefer Complementary Bands?

**Tier:** ★ SUPPORTING  
**Status:** PLANNED  
**Depends On:** 012 (Wormhole Activation), 022 (Band Phase Locking)

---

## 1. Problem Statement

Wormhole attention connects positions across the spectral hierarchy. The architecture defines symmetric pairs:
- Band 0 ↔ Band 6 (Identity ↔ Position)
- Band 1 ↔ Band 5 (Shape ↔ Texture)
- Band 2 ↔ Band 4 (Structure ↔ Detail)

**Do wormholes actually prefer these complementary pairs?**

Like orbital resonances (1:2:4 for Jupiter's moons), do wormholes naturally find and strengthen resonant connections?

---

## 2. Hypothesis

```
THE RESONANT WORMHOLE HYPOTHESIS:

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  Wormhole attention PREFERS connections between complementary bands.  │
│                                                                         │
│  MECHANISM:                                                             │
│  • Complementary bands encode complementary information               │
│    - Low-freq: WHAT (identity, category)                              │
│    - High-freq: WHERE (position, detail)                              │
│                                                                         │
│  • These are natural partners:                                         │
│    - "I know WHAT, tell me WHERE"                                     │
│    - "I see WHERE, tell me WHAT"                                      │
│                                                                         │
│  • Like orbital resonances, complementary connections are STABLE:     │
│    - They reinforce each other                                        │
│    - Non-complementary connections are less useful                    │
│                                                                         │
│  PREDICTION:                                                            │
│  Wormhole activation is HIGHER for complementary pairs.              │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Scientific Basis

### 3.1 Orbital Resonances as Template

```
WHY RESONANCES FORM:

Jupiter's moons: 1:2:4
Saturn's rings: gaps at resonances
The solar system is NOT random.

MECHANISM:
• Arbitrary relationships are unstable
• Resonant relationships reinforce
• Over time, only resonances survive

Same principle should apply to attention:
• Arbitrary connections don't persist
• Complementary connections reinforce
• Training should strengthen resonances
```

### 3.2 Complementary Information

```
WHY PAIRS ARE COMPLEMENTARY:

Band 0 (DC):    WHAT exists          Band 6 (High): WHERE exactly
Band 1 (VLow):  WHAT shape           Band 5 (MHigh): WHERE roughly  
Band 2 (Low):   WHAT structure       Band 4 (Mid):   WHAT details

These are NATURAL PARTNERS:

• Knowing "a cat" (Band 0) → want to know "where" (Band 6)
• Seeing "edge at (x,y)" (Band 6) → want to know "what" (Band 0)

Non-complementary pairs (e.g., 0↔1) have redundant information.
```

### 3.3 The Pythagorean Analogy

```
RESONANCE = HARMONY:

In music: consonant intervals (octave, fifth) are stable
In orbits: resonant ratios (1:2, 2:3) are stable
In attention: complementary pairs should be stable

Non-resonant = dissonant = unstable = less used
```

---

## 4. Apparatus

### 4.1 Required Measurements

```
MEASUREMENT REQUIREMENTS:

1. WORMHOLE ACTIVATION RATES
   • For each band pair, how often do wormholes fire?
   • Coherence gate threshold = 0.5 (normalized entropy threshold)
   • Gate opens when attention has low entropy (high coherence)

2. ACTIVATION STRENGTH
   • When wormholes fire, how strong is the connection?
   • Average attention weight through wormhole
   • Coherence gate value (0-1 sigmoid output)

3. PAIR CATEGORIZATION
   • Complementary: 0↔6, 1↔5, 2↔4
   • Non-complementary: all other pairs

4. COMPARISON METRICS
   • Ratio of complementary to non-complementary activation
   • Statistical significance of difference
```

### 4.2 Experimental Setup

```python
class ResonantWormholeAnalyzer:
    """Analyzes wormhole activation patterns across band pairs."""
    
    def __init__(self, model, coherence_threshold: float = 0.5):
        """
        Args:
            model: SpectralBeliefMachine instance
            coherence_threshold: Normalized entropy threshold for coherence gate.
                Lower = gate opens for more coherent (low-entropy) attention.
                Typical range: 0.3-0.7
        """
        self.model = model
        self.coherence_threshold = coherence_threshold
        self.complementary_pairs = [(0, 6), (1, 5), (2, 4)]
    
    def measure_activation(self, data) -> Dict:
        """Measure wormhole activation for all band pairs."""
        activations = {(i, j): [] for i in range(7) for j in range(i+1, 7)}
        coherence_gates = {(i, j): [] for i in range(7) for j in range(i+1, 7)}
        
        for batch in data:
            # Get wormhole attention maps and gate values
            wormhole_data = self.model.get_wormhole_attention(batch, return_gates=True)
            
            for (i, j), data in wormhole_data.items():
                attn = data['attention']
                gate = data['coherence_gate']
                
                # Measure gate activation (how often gate is open > 0.5)
                active = (gate > 0.5).float().mean()
                activations[(i, j)].append(active)
                
                # Track gate values
                coherence_gates[(i, j)].append(gate.mean())
        
        return {
            'activations': {k: torch.tensor(v).mean() for k, v in activations.items()},
            'coherence_gates': {k: torch.tensor(v).mean() for k, v in coherence_gates.items()}
        }
    
    def compare_complementary_vs_other(
        self,
        activations: Dict
    ) -> Dict:
        """Compare complementary pairs to non-complementary."""
        comp_rates = []
        other_rates = []
        
        for (i, j), rate in activations.items():
            if (i, j) in self.complementary_pairs or (j, i) in self.complementary_pairs:
                comp_rates.append(rate)
            else:
                other_rates.append(rate)
        
        return {
            "complementary_mean": torch.tensor(comp_rates).mean(),
            "other_mean": torch.tensor(other_rates).mean(),
            "ratio": torch.tensor(comp_rates).mean() / torch.tensor(other_rates).mean(),
            "t_statistic": self._t_test(comp_rates, other_rates),
        }
    
    def measure_frequency_correlation(
        self,
        activations: Dict
    ) -> float:
        """Is activation correlated with frequency ratio?"""
        freq_ratios = []
        act_rates = []
        
        for (i, j), rate in activations.items():
            # Frequency ratio (higher band / lower band)
            freq_ratio = 2 ** (j - i)  # Logarithmic spacing
            freq_ratios.append(freq_ratio)
            act_rates.append(rate)
        
        # Correlation
        return torch.corrcoef(
            torch.stack([torch.tensor(freq_ratios), torch.tensor(act_rates)])
        )[0, 1]
```

---

## 5. Method

### 5.1 Protocol

```
EXPERIMENTAL PROTOCOL:

1. SELECT DATA
   • Diverse sequences requiring cross-band communication
   • Both simple and complex patterns

2. MEASURE BASELINE
   • Random initialization wormhole activation
   • Expect: no preference for any pair

3. MEASURE TRAINED MODEL
   • After training, measure all pair activations
   • Expect: complementary pairs higher

4. COMPUTE STATISTICS
   • Compare complementary vs non-complementary
   • Test for statistical significance

5. ANALYZE FREQUENCY CORRELATION
   • Is activation correlated with frequency ratio?
   • Expect: activation ∝ frequency separation

6. TRACK EMERGENCE
   • Measure activation patterns during training
   • When does preference emerge?
```

### 5.2 Controls

- **Random initialization**: No preference expected
- **Shuffled bands**: Destroy structure, check if preference persists
- **Different architectures**: Is preference universal or specific?

---

## 6. Predictions

### 6.1 If Hypothesis is Correct

```
EXPECTED RESULTS:

1. COMPLEMENTARY PAIRS ACTIVATE MORE
   • 0↔6, 1↔5, 2↔4 have higher activation rates
   • Ratio > 1.5 (complementary / other)

2. FREQUENCY RATIO CORRELATION
   • Activation correlates with frequency separation
   • Higher separation = more activation

3. EMERGENCE DURING TRAINING
   • Early: no preference
   • Late: clear preference
   • Transition may be sudden (phase locking)

4. BAND 3 IS SPECIAL
   • Connects to all bands
   • Acts as "bridge" between groups
```

### 6.2 Quantitative Predictions

| Metric | Prediction | Significance |
|--------|------------|--------------|
| Complementary/Other ratio | > 1.5 | Clear preference |
| Frequency correlation | r > 0.5 | Strong relationship |
| Statistical significance | p < 0.01 | Not random |
| Band 3 correlation | > 0.3 with all | Bridge function |

---

## 7. Falsification

### 7.1 What Would Disprove the Hypothesis

```
FALSIFICATION CRITERIA:

1. NO PREFERENCE
   • Complementary/Other ratio ≈ 1.0
   • All pairs equally active
   → Architecture defines pairs but model ignores

2. OPPOSITE PREFERENCE
   • Non-complementary pairs more active
   → Complementarity is wrong model

3. NO FREQUENCY CORRELATION
   • Activation independent of frequency ratio
   → Spectral structure irrelevant
```

### 7.2 Alternative Interpretations

If falsified, possible alternatives:
- Wormhole connections are content-driven, not structure-driven
- Complementarity is imposed by architecture, not learned
- Different pairing scheme is optimal

---

## 8. Results

*To be filled after experiment*

### 8.1 Activation Rates by Pair

| Pair | Type | Activation Rate | Strength |
|------|------|-----------------|----------|
| 0↔6 | Complementary | | |
| 1↔5 | Complementary | | |
| 2↔4 | Complementary | | |
| 0↔1 | Non-comp | | |
| ... | | | |

### 8.2 Statistical Comparison

| Metric | Value | p-value |
|--------|-------|---------|
| Comp/Other ratio | | |
| Frequency correlation | | |

### 8.3 Training Dynamics

*Space for emergence plots*

---

## 9. Conclusion

*To be filled after experiment*

### 9.1 Summary

### 9.2 Implications

### 9.3 Next Steps

---

## References

- `foundations/HARMONY_AND_COHERENCE.md`, Resonance and coherence
- `012_EXP_WORMHOLE_ACTIVATION.md`, Wormhole activation patterns
- `022_EXP_BAND_PHASE_LOCKING.md`, Band phase locking
- `wormhole/WORMHOLE_HYBRID.md`, Wormhole architecture

---

*"Jupiter's moons found 1:2:4. Saturn's rings have gaps at resonances. The solar system plays in equal temperament. So should attention."*

*Oscar Goldman, Shogu Research Group @ Datamutant.ai subsidiary of 温心重工業*