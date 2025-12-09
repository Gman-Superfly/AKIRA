# EXPERIMENT 001: Basic Entropy Observation

## Can We Observe Attention Entropy?

---

## Status: PENDING

---

## 1. Problem Statement

### 1.1 The Question

Before we can test any theory about belief dynamics, collapse, or phase transitions, we must answer a simpler question:

**Can we observe and compute attention entropy in real-time during inference?**

This is the foundational measurement. Without it, all subsequent experiments are impossible.

### 1.2 Why This Matters

```
THE FOUNDATION OF OBSERVABILITY

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  We claim:                                                              │
│  • Attention weights encode belief                                     │
│  • Entropy of attention = uncertainty of belief                       │
│  • Low entropy = committed, high entropy = uncertain                  │
│                                                                         │
│  But we have NOT YET VERIFIED that we can:                            │
│  • Extract attention weights during forward pass                      │
│  • Compute entropy efficiently                                         │
│  • Observe meaningful variation in entropy                            │
│  • Correlate entropy with prediction quality                          │
│                                                                         │
│  This experiment establishes the measurement capability.              │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1.3 What Success Looks Like

If this experiment succeeds:
- We have working entropy measurement infrastructure
- Entropy varies meaningfully (not constant)
- Entropy correlates with prediction error
- We can proceed to more complex experiments

If this experiment fails:
- We cannot observe what we claim to theorize about
- Theory must be revised or measurement approach changed

---

## 2. Hypothesis

### 2.1 Primary Hypothesis

**H1: Attention entropy is observable and varies meaningfully during inference.**

Specifically:
- Entropy can be computed from attention weights
- Entropy varies across positions (spatial variation)
- Entropy varies across time (temporal variation)
- Entropy varies across bands (spectral variation)

### 2.2 Secondary Hypothesis

**H2: Attention entropy correlates with prediction error.**

Specifically:
- High entropy positions should have higher prediction error
- Low entropy positions should have lower prediction error
- The correlation should be statistically significant

### 2.3 Null Hypotheses

**H0a:** Entropy is constant across positions/time/bands (no meaningful variation)
**H0b:** Entropy is uncorrelated with prediction error (entropy is noise)

---

## 3. Scientific Basis

### 3.1 Shannon Entropy

**ESTABLISHED SCIENCE:**

Shannon entropy measures uncertainty in a probability distribution:

```
H(p) = -Σᵢ pᵢ log₂(pᵢ)

Properties:
• H = 0 when p is deterministic (one pᵢ = 1, rest = 0)
• H = log₂(N) when p is uniform (all pᵢ = 1/N)
• Concave function, maximized at uniform distribution

Reference: Shannon, C.E. (1948). A Mathematical Theory of Communication.
```

### 3.2 AKIRA Theory Basis

**Relevant Theory Documents:**
- `architecture_theoretical/SPECTRAL_BELIEF_MACHINE.md`, §6 (Belief State Tracking)
- `CANONICAL_PARAMETERS.md`, Entropy definition, normalization
- `MATHEMATICAL_FOUNDATIONS_CHECK.md`, Entropy formula consistency

**Key Concepts:**
- **Entropy formula:** `H = -Σ pᵢ log(pᵢ)` (natural log in code, log₂ for bits)
- **Normalized entropy:** `h = H / log(n)` where n = number of elements
- **Range:** h ∈ [0, 1] where 0 = collapsed, 1 = uniform

**From SPECTRAL_BELIEF_MACHINE.md:**
> "Belief state is tracked explicitly via entropy monitoring. Each band maintains H(B_k) = -Σ p_i log p_i. This is not implicit, it's a first-class observable."

**This experiment validates:** Whether the theoretical entropy definition can be computed in practice and varies meaningfully.

### 3.2 Attention as Probability Distribution

**ESTABLISHED SCIENCE:**

Softmax attention produces a valid probability distribution:

```
Attention(Q, K, V) = softmax(QK^T / √d) × V

The softmax outputs:
• aᵢⱼ ∈ [0, 1] for all i, j
• Σⱼ aᵢⱼ = 1 for each row i

This IS a probability distribution.
Therefore Shannon entropy is well-defined.

Reference: Vaswani et al. (2017). Attention Is All You Need.
```

### 3.3 Entropy as Uncertainty

**ESTABLISHED SCIENCE:**

Entropy measures the "spread" of attention:

```
Low entropy:  Attention concentrated on few keys
              Model is "confident" about what to attend to
              
High entropy: Attention spread across many keys
              Model is "uncertain" about what to attend to

This interpretation is standard in information theory.
```

### 3.4 Prior Work

| Work | Finding |
|------|---------|
| Voita et al. (2019) | Attention heads specialize; some have consistently low entropy |
| Michel et al. (2019) | Pruning high-entropy heads often doesn't hurt performance |
| Clark et al. (2019) | Entropy varies across layers; later layers often more focused |

---

## 4. Apparatus

### 4.1 Required Components

```
MEASUREMENT INFRASTRUCTURE

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  COMPONENT              FUNCTION                       STATUS          │
│  ─────────              ────────                       ──────          │
│                                                                         │
│  entropy_tracker.py     Compute H from attention       TO BUILD        │
│  attention_hooks.py     Extract weights from layers    TO BUILD        │
│  visualization.py       Plot entropy maps              TO BUILD        │
│  correlation_test.py    Statistical analysis           TO BUILD        │
│                                                                         │
│  AKIRA model            Subject of measurement         EXISTS          │
│  Test dataset           Input sequences                EXISTS          │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Entropy Computation

```python
def compute_attention_entropy(attention_weights: torch.Tensor) -> torch.Tensor:
    """
    Compute Shannon entropy of attention distribution.
    
    Args:
        attention_weights: Tensor of shape [B, H, N, N] or [B, N, N]
                          Already softmaxed (rows sum to 1)
    
    Returns:
        entropy: Tensor of shape [B, H, N] or [B, N]
                 Entropy for each query position
    """
    # Clamp to avoid log(0)
    eps = 1e-10
    weights = attention_weights.clamp(min=eps)
    
    # Shannon entropy: H = -Σ p log p
    entropy = -torch.sum(weights * torch.log2(weights), dim=-1)
    
    return entropy
```

### 4.3 Hook Registration

```python
def register_attention_hooks(model):
    """
    Register forward hooks to capture attention weights.
    """
    attention_weights = {}
    
    def make_hook(name):
        def hook(module, input, output):
            # Assuming output is (attended_values, attention_weights)
            if isinstance(output, tuple) and len(output) >= 2:
                attention_weights[name] = output[1].detach()
        return hook
    
    for name, module in model.named_modules():
        if 'attention' in name.lower():
            module.register_forward_hook(make_hook(name))
    
    return attention_weights
```

---

## 5. Methods

### 5.1 Experimental Protocol

```
PROTOCOL: ENTROPY OBSERVATION

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  STEP 1: SETUP                                                          │
│  ────────────────                                                       │
│  • Load trained AKIRA model                                            │
│  • Register attention hooks on all attention layers                   │
│  • Prepare test sequences (moving blob, double slit, etc.)           │
│                                                                         │
│  STEP 2: FORWARD PASS                                                   │
│  ─────────────────────                                                  │
│  • Run forward pass on test sequence                                  │
│  • Capture attention weights from all layers/heads/bands             │
│  • Store predictions and ground truth                                 │
│                                                                         │
│  STEP 3: ENTROPY COMPUTATION                                            │
│  ───────────────────────────                                            │
│  • Compute entropy for each attention layer                          │
│  • Compute entropy per position (spatial map)                        │
│  • Compute entropy per band (spectral decomposition)                 │
│  • Compute entropy per time step (temporal evolution)                │
│                                                                         │
│  STEP 4: ERROR COMPUTATION                                              │
│  ─────────────────────────                                              │
│  • Compute MSE error per position                                    │
│  • Compute error maps                                                 │
│                                                                         │
│  STEP 5: CORRELATION ANALYSIS                                           │
│  ────────────────────────────                                           │
│  • Correlate entropy with error (Pearson, Spearman)                  │
│  • Statistical significance tests                                     │
│  • Visualize relationships                                            │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 5.2 Test Sequences

```
TEST SEQUENCES TO USE

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  SEQUENCE          WHY                           EXPECTED ENTROPY      │
│  ────────          ───                           ────────────────      │
│                                                                         │
│  Moving blob       Simple, predictable           Low (confident)       │
│  (constant vel)    Clear trajectory                                    │
│                                                                         │
│  Moving blob       Ambiguous direction           High at reversal      │
│  (reversal)        Uncertainty at turn                                 │
│                                                                         │
│  Two blobs         Which to track?               High when close       │
│  (crossing)        Multiple hypotheses                                 │
│                                                                         │
│  Random motion     Unpredictable                 High throughout       │
│                    Maximum uncertainty                                 │
│                                                                         │
│  Static image      Nothing changes               Should be stable      │
│                    Baseline measurement                                │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 5.3 Statistical Tests

```
STATISTICAL ANALYSIS

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  TEST                 PURPOSE                     THRESHOLD            │
│  ────                 ───────                     ─────────            │
│                                                                         │
│  Variance test        H varies meaningfully?      Var(H) > 0.01       │
│                                                                         │
│  Pearson correlation  Linear H-error relation?    |r| > 0.3, p < 0.05 │
│                                                                         │
│  Spearman correlation Monotonic H-error relation? |ρ| > 0.3, p < 0.05 │
│                                                                         │
│  ANOVA (across bands) Bands differ in H?          F-test p < 0.05     │
│                                                                         │
│  t-test (high vs low) High-H ≠ low-H error?      p < 0.05            │
│                       (positions above/below                          │
│                        median entropy)                                 │
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
│  Entropy variance                         > 0.1 (meaningful variation) │
│  Entropy range                            0.5 to 4.0 bits typically    │
│  Entropy-error correlation                r > 0.3 (positive)           │
│  Band entropy difference                  Low bands < High bands       │
│  Temporal entropy variation               High during ambiguity        │
│                                                                         │
│  Spatial pattern:                                                       │
│  • Low entropy at object center (confident about existence)           │
│  • High entropy at object edge (uncertain about boundary)             │
│  • High entropy ahead of motion (uncertain about future position)     │
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
│  Entropy variance < 0.01                  H is constant (H1 false)    │
│  Entropy-error correlation |r| < 0.1     H unrelated to error (H2 false)│
│  No band differences (ANOVA p > 0.05)    Spectral structure absent    │
│  Entropy higher at confident regions     Theory inverted or wrong     │
│                                                                         │
│  ANY of these would require theory revision.                          │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 7. Results

### 7.1 Entropy Variance

```
[ TO BE FILLED AFTER EXPERIMENT ]

Variance of entropy across all measurements: _______
Range (min, max): _______ to _______
Verdict on H0a (constant entropy): _______
```

### 7.2 Spatial Entropy Maps

```
[ TO BE FILLED AFTER EXPERIMENT ]

Insert entropy heatmaps for each test sequence:

Moving blob (constant): 
Moving blob (reversal):
Two blobs (crossing):
Random motion:
Static image:
```

### 7.3 Temporal Entropy Evolution

```
[ TO BE FILLED AFTER EXPERIMENT ]

Insert entropy vs. time plots:

Mean entropy over time: _______
Entropy at key events (reversal, crossing): _______
```

### 7.4 Band Entropy Comparison

```
[ TO BE FILLED AFTER EXPERIMENT ]

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  BAND              MEAN ENTROPY      STD        EXPECTED                │
│  ────              ────────────      ───        ────────                │
│                                                                         │
│  Band 0 (DC)       _______           _______    Low                    │
│  Band 1 (VeryLow)  _______           _______    Low                    │
│  Band 2 (Low)      _______           _______    Medium                 │
│  Band 3 (MidLow)   _______           _______    Medium                 │
│  Band 4 (Mid)      _______           _______    Medium                 │
│  Band 5 (MidHigh)  _______           _______    High                   │
│  Band 6 (High)     _______           _______    High                   │
│                                                                         │
│  ANOVA F-statistic: _______                                            │
│  ANOVA p-value: _______                                                │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 7.5 Entropy-Error Correlation

```
[ TO BE FILLED AFTER EXPERIMENT ]

Pearson r: _______
Pearson p-value: _______
Spearman ρ: _______
Spearman p-value: _______

Scatter plot (entropy vs. error): [INSERT]
```

### 7.6 High vs Low Entropy Error Comparison

```
[ TO BE FILLED AFTER EXPERIMENT ]

Mean error at high-entropy positions (H > median): _______
Mean error at low-entropy positions (H < median): _______
t-statistic: _______
p-value: _______
```

---

## 8. Conclusion

### 8.1 Summary

```
[ TO BE FILLED AFTER EXPERIMENT ]

Primary hypothesis H1 (entropy observable and varies): SUPPORTED / NOT SUPPORTED
Secondary hypothesis H2 (entropy correlates with error): SUPPORTED / NOT SUPPORTED
```

### 8.2 Implications

```
[ TO BE FILLED AFTER EXPERIMENT ]

If supported:
- Proceed to Experiment 002 (Collapse Detection)
- Entropy is valid measure of belief uncertainty
- Foundational measurement capability established

If not supported:
- Investigate why measurement fails
- Consider alternative uncertainty measures
- Theory revision may be needed
```

### 8.3 Limitations

```
[ TO BE FILLED AFTER EXPERIMENT ]

Known limitations of this experiment:
- 
- 
- 
```

### 8.4 Next Steps

```
[ TO BE FILLED AFTER EXPERIMENT ]

If successful, proceed to:
- 002_EXP_COLLAPSE_DETECTION.md
```

---

## References

1. Shannon, C.E. (1948). A Mathematical Theory of Communication. Bell System Technical Journal.
2. Vaswani, A. et al. (2017). Attention Is All You Need. NeurIPS.
3. Voita, E. et al. (2019). Analyzing Multi-Head Self-Attention. ACL.
4. Michel, P. et al. (2019). Are Sixteen Heads Really Better than One? NeurIPS.
5. Clark, K. et al. (2019). What Does BERT Look At? ACL.

---



*"Before we can study the lightning, we must prove we can see the sky. This experiment establishes that we can observe the basic quantity, entropy, upon which all subsequent theory depends."*


*Oscar Goldman, Shogu Research Group @ Datamutant.ai subsidiary of 温心重工業*