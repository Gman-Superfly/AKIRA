# Experiment 000 Results: Aligned Analysis (Random Tokens)

**Date:** December 5, 2025

**Run ID:** 002 - Aligned Methods with Random Token Input

**Script:** `001_action_quanta_extraction_aligned.py`

---

## Configuration

```python
MODEL_A = "gpt2"           # 124M params
MODEL_B = "gpt2-medium"    # 355M params
LAYERS_TO_ANALYZE = [3, 5, 7]
NUM_SAMPLES = 500
SEQ_LENGTH = 64
INPUT_TYPE = "random"
THRESHOLD_METHOD = "statistical"  # mean + 2*std
```

**Platform:** Google Colab T4 GPU

---

## Results Summary

### Analysis 1: CKA (Centered Kernel Alignment)

```
Layer 3: CKA = 0.7179 (HIGH similarity)
Layer 5: CKA = 0.6853 (MODERATE similarity)  
Layer 7: CKA = 0.5287 (MODERATE similarity)

Average CKA: 0.6440
```

| Layer | CKA Score | Interpretation |
|-------|-----------|----------------|
| 3 | 0.7179 | HIGH - 72% representational similarity |
| 5 | 0.6853 | MODERATE-HIGH - 69% similarity |
| 7 | 0.5287 | MODERATE - 53% similarity |

**Pattern:** Decreasing similarity with depth (0.72 → 0.69 → 0.53)

---

### Analysis 2: Optimal Transport (Neuron Matching)

```
Layer 3: Transport cost 0.6606, Mean correlation 0.3394, High matches: 17/100
Layer 5: Transport cost 0.7218, Mean correlation 0.2782, High matches: 12/100
Layer 7: Transport cost 0.7738, Mean correlation 0.2262, High matches: 3/100
```

| Layer | Transport Cost | Mean Corr | High-Corr Pairs (>0.5) |
|-------|----------------|-----------|------------------------|
| 3 | 0.6606 | 0.3394 | 17/100 (17%) |
| 5 | 0.7218 | 0.2782 | 12/100 (12%) |
| 7 | 0.7738 | 0.2262 | 3/100 (3%) |

**Pattern:** Same decreasing trend. Best neuron matches are in early layers.

---

### Analysis 3: Procrustes (Linear Mapping + Statistical Threshold)

```
Layer 3: Error 0.8941, Threshold 0.6322, AQ: 138 (4.5%)
Layer 5: Error 1.0027, Threshold 0.6488, AQ: 111 (3.6%)
Layer 7: Error 1.0228, Threshold 0.6030, AQ: 106 (3.5%)

Total AQ candidates: 355
Average: 3.9%
```

| Layer | Procrustes Error | Threshold (mean+2std) | AQ Candidates | Percentage |
|-------|------------------|----------------------|---------------|------------|
| 3 | 0.8941 | 0.6322 | 138 | 4.5% |
| 5 | 1.0027 | 0.6488 | 111 | 3.6% |
| 7 | 1.0228 | 0.6030 | 106 | 3.5% |
| **Total** | - | - | **355** | **3.9%** |

**Key:** The threshold is now computed from data (mean + 2*std ≈ 0.63), not arbitrary.

---

## Comparison: Naive vs Aligned Methods

| Metric | Naive Method (Run 001) | Aligned Method (Run 002) |
|--------|------------------------|--------------------------|
| Threshold | 0.3 (arbitrary) | ~0.63 (statistical) |
| Total AQ | 38 | 355 |
| Average % | 0.4% | 3.9% |
| Method | Index-by-index | Procrustes alignment |

**The aligned method finds 10x more AQ candidates despite using a 2x HIGHER threshold.**

This happens because:
1. Procrustes ALIGNS the representation spaces first
2. After alignment, many more neurons correlate
3. The 0.3 threshold on unaligned spaces was comparing mismatched neurons

---

## Interpretation

### What CKA Tells Us

**CKA = 0.64 average means:** GPT-2 and GPT-2-medium encode ~64% similar information in their representations, independent of how neurons are organized.

This is **strong evidence for universal structure**. The models learned similar representations despite:
- Different number of parameters (124M vs 355M)
- Different layer dimensions
- Independent training runs

### What Optimal Transport Tells Us

**17% of top neurons have high correlation matches in layer 3.** This means specific neurons in GPT-2 have clear counterparts in GPT-2-medium.

The decreasing pattern (17% → 12% → 3%) confirms:
- Early layers: universal, matching features
- Late layers: model-specific organization

### What Procrustes Tells Us

**355 AQ candidates (3.9%)** survive a statistically principled threshold after proper alignment.

This is the most rigorous estimate:
- Spaces are aligned via learned rotation
- Threshold is mean + 2*std (statistical outliers)
- Results are ~10x higher than naive method

### The Layer Pattern is Consistent Across All Methods

```
ALL THREE METHODS SHOW THE SAME PATTERN:

CKA:      Layer 3 (0.72) > Layer 5 (0.69) > Layer 7 (0.53)
OT:       Layer 3 (17%)  > Layer 5 (12%)  > Layer 7 (3%)
Procrust: Layer 3 (4.5%) > Layer 5 (3.6%) > Layer 7 (3.5%)

INTERPRETATION: Early layers are more universal, later layers diverge.
```

---

## Opinion and Analysis

### This is Meaningful Evidence for Action Quanta

The results are consistent and principled:

1. **CKA shows the models share substantial structure** (64% average)
   - This isn't random - meaningless representations would show CKA near 0
   - 0.64 is solidly in the "similar representations" range

2. **The universality decreases predictably with depth**
   - All three methods show the same pattern
   - This matches known transformer behavior
   - Early layers: tokenization, position, basic patterns (universal)
   - Late layers: task-specific, scale-dependent (divergent)

3. **355 AQ candidates with principled threshold**
   - The naive method was severely underestimating (38 vs 355)
   - Proper alignment reveals much more shared structure
   - The statistical threshold removes arbitrary choices

### What These 355 AQ Might Represent

Given this was RANDOM TOKEN input:
- Not semantic features (no meaning in input)
- Likely structural features:
  - Position encoding patterns
  - Token frequency responses
  - Basic attention patterns
  - Learned "default" activations

### Remaining Questions

1. **What happens with real text?** Would expect similar or higher CKA (semantic alignment)

2. **Do these 355 AQ overlap with semantic AQ?** Critical for identifying "true" universal features

3. **What do these neurons encode?** Need interpretability analysis

4. **Does the pattern hold across architecture families?** Test GPT-2 vs Pythia vs Llama

---

## Conclusion

**MODERATE-TO-STRONG evidence for universal Action Quanta.**

- CKA = 0.64 shows substantial shared representation
- Procrustes finds 355 AQ candidates (3.9%) with rigorous threshold
- Consistent decreasing pattern with layer depth across all methods
- The naive method's 0.4% was a severe underestimate due to misalignment

**Key insight:** Proper alignment is essential. Comparing neurons by index across models is methodologically flawed. The representations ARE similar (CKA proves this), but the neurons are organized differently.

---

## Next Steps

1. Run aligned analysis with TEXT input (compare structural vs semantic universality)
2. Cross-reference AQ indices between runs (find truly universal features)
3. Test on different architecture pairs (Pythia, Llama)
4. Interpret what the 355 AQ candidates encode

---

*Oscar Goldman - Shogu Research Group @ Datamutant.ai*

*"The naive method was asking 'does neuron 42 in model A match neuron 42 in model B?' - the wrong question. The aligned method asks 'do these models encode similar information?' - and the answer is yes, about 64% similar. That's substantial universal structure."*
