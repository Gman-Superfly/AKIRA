# Experiment 000: Random vs Text Comparison (Aligned Methods)

**Date:** December 5, 2025

**Script:** `001_action_quanta_extraction_aligned.py`

---

## Configuration (Both Runs)

```python
MODEL_A = "gpt2"           # 124M params
MODEL_B = "gpt2-medium"    # 355M params
LAYERS_TO_ANALYZE = [3, 5, 7]
NUM_SAMPLES = 500
SEQ_LENGTH = 64
THRESHOLD_METHOD = "statistical"  # mean + 2*std
```

**Platform:** Google Colab T4 GPU

---

## Results Comparison

### Analysis 1: CKA (Centered Kernel Alignment)

| Layer | Random Tokens | Real Text | Difference |
|-------|---------------|-----------|------------|
| 3 | 0.7179 (HIGH) | 0.4560 (MODERATE) | **-0.26** |
| 5 | 0.6853 (MODERATE) | 0.8036 (HIGH) | **+0.12** |
| 7 | 0.5287 (MODERATE) | 0.7491 (HIGH) | **+0.22** |
| **Average** | **0.6440** | **0.6696** | **+0.03** |

**Pattern Inversion:**
```
RANDOM: Layer 3 (0.72) > Layer 5 (0.69) > Layer 7 (0.53)
        Early layers most similar, decreasing with depth

TEXT:   Layer 5 (0.80) > Layer 7 (0.75) > Layer 3 (0.46)
        Middle layers most similar, early layers LEAST similar
```

---

### Analysis 2: Optimal Transport (Neuron Matching)

| Layer | Random Tokens | Real Text | Difference |
|-------|---------------|-----------|------------|
| 3 | 17/100 (17%) | 41/100 (41%) | **+24** |
| 5 | 12/100 (12%) | 23/100 (23%) | **+11** |
| 7 | 3/100 (3%) | 13/100 (13%) | **+10** |

**Text finds MORE matched neuron pairs at every layer.**

Even though CKA for layer 3 is LOWER with text, OT finds MORE individual neuron matches.
This suggests specialized semantic features that correlate strongly but contribute less to overall similarity.

---

### Analysis 3: Procrustes (Aligned AQ Detection)

| Metric | Random Tokens | Real Text |
|--------|---------------|-----------|
| **Layer 3** | | |
| Mean correlation | 0.4598 | 0.6351 |
| Threshold | 0.6322 | 0.7985 |
| AQ candidates | 138 (4.5%) | 118 (3.8%) |
| **Layer 5** | | |
| Mean correlation | 0.4720 | 0.6012 |
| Threshold | 0.6488 | 0.7582 |
| AQ candidates | 111 (3.6%) | 103 (3.4%) |
| **Layer 7** | | |
| Mean correlation | 0.4522 | 0.5701 |
| Threshold | 0.6030 | 0.7275 |
| AQ candidates | 106 (3.5%) | 93 (3.0%) |
| **Total** | **355 (3.9%)** | **314 (3.4%)** |

**Key observation:** Text has HIGHER mean correlations but HIGHER thresholds, resulting in slightly fewer AQ candidates.

---

## Interpretation

### The Pattern Inversion is Significant

```
                    Random Tokens          Real Text
                    ─────────────          ─────────
Layer 3 (early):    HIGHEST (0.72)   →    LOWEST (0.46)
Layer 5 (middle):   MIDDLE (0.69)    →    HIGHEST (0.80)
Layer 7 (late):     LOWEST (0.53)    →    HIGH (0.75)
```

**What this means:**

1. **Random tokens activate STRUCTURAL universality**
   - Early layers process low-level patterns (tokenization, position)
   - These are most similar across models because they're fundamental
   - Late layers diverge because they have nothing meaningful to process

2. **Real text activates SEMANTIC universality**
   - Early layers diverge because they process surface features differently
   - Middle/late layers CONVERGE on semantic representations
   - Both models learn similar semantic features despite different architectures

### This Aligns with Transformer Theory

```
LAYER FUNCTION THEORY:
─────────────────────
Early layers (1-4):   Tokenization, position, basic syntax
Middle layers (5-8):  Semantic features, entity recognition, relationships
Late layers (9-12):   Task-specific, context integration, output preparation

OUR RESULTS:
───────────
Random: Early > Middle > Late  (structural features dominate)
Text:   Middle > Late > Early  (semantic features dominate)
```

### The Increased Neuron Matches with Text

Despite LOWER overall CKA in layer 3 with text:
- Random: 17 matched neurons
- Text: 41 matched neurons (+24)

**Interpretation:** With text, individual neurons specialize on semantic features. These features correlate strongly between models (creating more matches) but are sparser, contributing less to overall CKA similarity.

This suggests **semantic AQ are more discrete** while **structural AQ are more distributed**.

---

## Summary Table

| Metric | Random | Text | Winner |
|--------|--------|------|--------|
| CKA Layer 3 | 0.72 | 0.46 | Random |
| CKA Layer 5 | 0.69 | 0.80 | **Text** |
| CKA Layer 7 | 0.53 | 0.75 | **Text** |
| CKA Average | 0.64 | 0.67 | Text |
| OT Matches L3 | 17 | 41 | **Text** |
| OT Matches L5 | 12 | 23 | **Text** |
| OT Matches L7 | 3 | 13 | **Text** |
| Procrustes AQ | 355 | 314 | Random |

---

## Conclusions

### Evidence for Action Quanta

1. **Universal structure exists** - Both conditions show CKA > 0.5 and significant AQ candidates

2. **Two types of universality detected:**
   - STRUCTURAL (random): early-layer dominant, distributed
   - SEMANTIC (text): middle-layer dominant, more discrete

3. **Aligned methods reveal ~350 AQ candidates (3-4%)** with principled thresholds

4. **The pattern inversion is a key finding** - it shows the methods are sensitive to input type and detecting different phenomena

### Implications for AKIRA

- Action Quanta may have subtypes: structural vs semantic
- Middle layers (5-7) are most important for semantic AQ
- The ~300-350 AQ candidates identified represent approximately 3-4% of MLP neurons
- These AQ survive rigorous statistical thresholds and proper alignment

### Remaining Questions

1. ~~**Do the AQ indices overlap?**~~ ANSWERED - see below
2. **What do these AQ encode?** Interpretability analysis needed
3. **Does this pattern hold across architecture families?** Test GPT-2 vs Pythia vs Llama
4. **Are semantic AQ more actionable?** Test downstream task performance

---

## Overlap Analysis (CRITICAL FINDING)

**Do the random AQ and text AQ refer to the same neurons?**

```
OVERLAP ANALYSIS: Random vs Text AQ
======================================================================

Layer 3:
  Random AQ: 138
  Text AQ: 118
  OVERLAP: 28 (20.3% of random, 23.7% of text)

Layer 5:
  Random AQ: 111
  Text AQ: 103
  OVERLAP: 31 (27.9% of random, 30.1% of text)

Layer 7:
  Random AQ: 106
  Text AQ: 93
  OVERLAP: 19 (17.9% of random, 20.4% of text)
```

### Summary Table

| Layer | Random AQ | Text AQ | Overlap | % of Random | % of Text |
|-------|-----------|---------|---------|-------------|-----------|
| 3 | 138 | 118 | 28 | 20.3% | 23.7% |
| 5 | 111 | 103 | 31 | 27.9% | 30.1% |
| 7 | 106 | 93 | 19 | 17.9% | 20.4% |
| **Total** | **355** | **314** | **78** | **22.0%** | **24.8%** |

### Interpretation

**LOW OVERLAP (~20-25%) confirms two distinct AQ populations:**

```
TOTAL AQ LANDSCAPE
──────────────────
Structural-only AQ:  277 neurons (random but NOT text)
Semantic-only AQ:    236 neurons (text but NOT random)
Universal AQ:         78 neurons (BOTH random AND text)
──────────────────
Total unique:        591 neurons across both conditions
```

**The 78 Universal AQ are the strongest candidates:**
- They correlate across models with random noise (structural universality)
- They ALSO correlate across models with real text (semantic universality)
- They are truly INPUT-INVARIANT universal features
- These represent ~2.5% of MLP neurons (78 / ~3072)

**This confirms the hypothesis of AQ subtypes:**
- ~75% of detected AQ are condition-specific
- Only ~25% are truly universal across input types
- The "bedrock" AQ that survive both tests are the most fundamental

---

## Raw Data

### Random Token Results
```
CKA: [0.7179, 0.6853, 0.5287], avg=0.6440
OT High Matches: [17, 12, 3]
Procrustes AQ: [138, 111, 106], total=355
Thresholds: [0.6322, 0.6488, 0.6030]
```

### Real Text Results
```
CKA: [0.4560, 0.8036, 0.7491], avg=0.6696
OT High Matches: [41, 23, 13]
Procrustes AQ: [118, 103, 93], total=314
Thresholds: [0.7985, 0.7582, 0.7275]
```

---

*Oscar Goldman - Shogu Research Group @ Datamutant.ai*

*"Random tokens and real text reveal different aspects of universal structure. The pattern inversion - early-layer dominance with noise, middle-layer dominance with text - is exactly what theory predicts. This isn't noise; it's signal about how transformers organize information."*
