# EXPERIMENT 010: Tickling Techniques

## Can We Probe the Manifold Cheaply?

**Oscar Goldman — Shogu Research Group @ Datamutant.ai**

---

## Status: PLANNED

## Tier: ★ SUPPORTING

## Depends On: 001-003 (Basic observability)

---

## 1. Problem Statement

### 1.1 The Question

We have "free information assets" — data computed during forward passes that we typically discard:

**Can we use these cheap probes to predict model behavior without full inference?**

### 1.2 Why This Matters

```
THE TICKLING HYPOTHESIS

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  FREE ASSETS (already computed):                                       │
│  • Attention entropy                                                   │
│  • Pre-softmax scores                                                  │
│  • Near-threshold connections                                          │
│  • Similarity matrix                                                   │
│                                                                         │
│  CHEAP PROBES:                                                          │
│  • Temperature sweep (2-3 forward passes)                             │
│  • First-token only (0.1 forward pass)                               │
│  • Threshold sweep (1 forward pass with variants)                    │
│                                                                         │
│  If these predict collapse destination:                                │
│  • Massive speedup for prompt optimization                            │
│  • Early stopping when confidence is high                            │
│  • Real-time belief monitoring                                        │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Hypothesis

### 2.1 Primary Hypothesis

**H1: Entropy predicts collapse destination.**

High-entropy zones are "leaders" that will compete in collapse.

### 2.2 Secondary Hypotheses

**H2: Temperature sweep reveals fragility.**
- If output changes with temperature, prediction is uncertain
- Stable across temperature = high confidence

**H3: Near-threshold count predicts alternatives.**
- Many near-threshold connections = multiple hypotheses
- Few = committed prediction

**H4: First-token entropy predicts full-sequence quality.**

### 2.3 Null Hypothesis

**H0:** Cheap probes are uncorrelated with full inference results.

---

## 3. Methods

### 3.1 Protocol

```
TICKLING EVALUATION PROTOCOL

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  STEP 1: Collect ground truth                                          │
│  • Run full inference on test set                                     │
│  • Record: final predictions, confidence, error                       │
│                                                                         │
│  STEP 2: Apply tickling probes                                         │
│  • Entropy mapping                                                     │
│  • Temperature sweep (T = 0.5, 1.0, 2.0)                             │
│  • Near-threshold count                                               │
│  • First-token analysis                                               │
│                                                                         │
│  STEP 3: Measure correlation                                           │
│  • Correlate each probe with ground truth                            │
│  • Measure cost savings vs accuracy                                  │
│                                                                         │
│  STEP 4: Build composite predictor                                     │
│  • Combine probes optimally                                           │
│  • Evaluate on held-out set                                          │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Predictions

```
IF THEORY IS CORRECT:

• Entropy-error correlation: r > 0.5
• Temperature stability predicts confidence: r > 0.4
• Near-threshold count predicts alternatives: r > 0.3
• First-token predicts sequence: r > 0.4

Cost savings: > 10× for same accuracy
```

---

## 4. Results

```
[ TO BE FILLED AFTER EXPERIMENT ]

Probe correlations with ground truth:
• Entropy: r = _____
• Temperature stability: r = _____
• Near-threshold: r = _____
• First-token: r = _____

Composite predictor accuracy: _____%
Cost savings: _____×
```

---

## 5. Conclusion

```
[ TO BE FILLED AFTER EXPERIMENT ]

H1 (entropy predicts): SUPPORTED / NOT SUPPORTED
H2 (temperature reveals fragility): SUPPORTED / NOT SUPPORTED
H3 (near-threshold predicts alternatives): SUPPORTED / NOT SUPPORTED
H4 (first-token predicts sequence): SUPPORTED / NOT SUPPORTED

Tickling is USEFUL / NOT USEFUL for practical speedup.
```

---

*Oscar Goldman — Shogu Research Group @ Datamutant.ai*


