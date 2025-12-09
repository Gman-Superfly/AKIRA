# EXPERIMENT 013: Differential Learning Rate Validation

## Does the Learning Rate Hierarchy Work?



---

## Status: PLANNED

## Tier: ★ SUPPORTING

## Depends On: 003 (Spectral Band Dynamics)

---

## 1. Problem Statement

### 1.1 The Question

AKIRA uses differential learning rates across bands (see `CANONICAL_PARAMETERS.md` for specification):
- Band 0 (DC): LR = 0.00001 (very slow)
- Band 6 (High): LR = 0.03 (fast)
- Band 7 (Temporal): LR = 0.001 (medium)

Ratio: 3000× between Band 0 and Band 6

**Does this hierarchy actually improve learning, or is uniform LR just as good?**

### 1.2 Why This Matters

```
THE DIFFERENTIAL LR HYPOTHESIS

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  Theory predicts:                                                       │
│  • Low bands encode stable structure (should change slowly)           │
│  • High bands encode transient details (should adapt quickly)         │
│  • Differential LR respects this natural timescale                    │
│                                                                         │
│  If true:                                                               │
│  • Differential LR should learn faster                                │
│  • Should generalize better                                           │
│  • Should be more stable                                              │
│                                                                         │
│  If false:                                                              │
│  • Uniform LR is simpler and should be used                          │
│  • Spectral hierarchy is cosmetic, not functional                    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Hypothesis

### 2.1 Primary Hypothesis

**H1: Differential LR improves learning speed.**

Reaches target loss faster than uniform LR.

### 2.2 Secondary Hypotheses

**H2: Differential LR improves generalization.**

Test loss lower than uniform LR.

**H3: Differential LR improves stability.**

Less variance in training, fewer collapses.

**H4: Bands learn at their prescribed rates.**

Actual weight changes match LR ratios.

### 2.3 Null Hypothesis

**H0:** Uniform LR performs equally well.

---

## 3. Methods

### 3.1 Protocol

```
DIFFERENTIAL LR ABLATION PROTOCOL

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  STEP 1: Train with differential LR (baseline)                         │
│  • Standard AKIRA configuration                                        │
│  • Record: loss curves, weight changes per band                       │
│                                                                         │
│  STEP 2: Train with uniform LR (ablation)                              │
│  • Same total learning budget                                         │
│  • LR = geometric mean of differential rates                         │
│                                                                         │
│  STEP 3: Compare learning dynamics                                     │
│  • Speed to target loss                                               │
│  • Final train/test loss                                              │
│  • Stability (loss variance)                                          │
│                                                                         │
│  STEP 4: Verify actual learning rates                                  │
│  • Measure ‖Δw‖/‖w‖ per band                                        │
│  • Compare to prescribed LR ratio                                     │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Predictions

```
IF THEORY IS CORRECT:

• Differential LR: 20% faster convergence
• Differential LR: 5% better test loss
• Differential LR: 30% less loss variance
• Actual weight changes: within 2× of prescribed ratios
```

---

## 4. Results

```
[ TO BE FILLED AFTER EXPERIMENT ]

Convergence speed:
• Differential: _____ steps to target
• Uniform: _____ steps to target
• Speedup: _____×

Final test loss:
• Differential: _____
• Uniform: _____
• Improvement: _____%

Stability (loss variance):
• Differential: _____
• Uniform: _____

Weight change ratios match prescribed: YES / NO
```

---

## 5. Conclusion

```
[ TO BE FILLED AFTER EXPERIMENT ]

H1 (faster learning): SUPPORTED / NOT SUPPORTED
H2 (better generalization): SUPPORTED / NOT SUPPORTED
H3 (more stable): SUPPORTED / NOT SUPPORTED
H4 (rates match): SUPPORTED / NOT SUPPORTED

Differential LR is RECOMMENDED / NOT RECOMMENDED.
```

---

*Oscar Goldman, Shogu Research Group @ Datamutant.ai subsidiary of 温心重工業*

