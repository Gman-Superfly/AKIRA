# Experiment 000 Results: Text Semantic Run

**Date:** December 5, 2025

**Run ID:** 002 - Real Text Semantic Universality Test

---

## Configuration

```python
MODEL_A = "gpt2"           # 124M params
MODEL_B = "gpt2-medium"    # 355M params
LAYERS_TO_ANALYZE = [3, 5, 7]
NUM_SAMPLES = 500
SEQ_LENGTH = 64
BATCH_SIZE = 16
AQ_THRESHOLD = 0.3         # Same conservative threshold as Run 001
INPUT_TYPE = "text"        # Real text samples (semantic test)
SEED = 42
```

**Platform:** Google Colab T4 GPU

---

## Results

```
EXPERIMENT 000 SUMMARY: ACTION QUANTA EXTRACTION
======================================================================
Layer 3: 2 AQ candidates (0.1%)
Layer 5: 7 AQ candidates (0.2%)
Layer 7: 5 AQ candidates (0.2%)

Average across layers: 0.2%
Total AQ candidates found: 14
```

### Per-Layer Breakdown

| Layer | Total Neurons | AQ Candidates | Percentage |
|-------|---------------|---------------|------------|
| 3     | ~3072         | 2             | 0.1%       |
| 5     | ~3072         | 7             | 0.2%       |
| 7     | ~3072         | 5             | 0.2%       |

---

## Comparison with Run 001 (Random Tokens)

| Metric | Random (001) | Text (002) | Difference |
|--------|--------------|------------|------------|
| Layer 3 AQ | 15 (0.5%) | 2 (0.1%) | -13 |
| Layer 5 AQ | 13 (0.4%) | 7 (0.2%) | -6 |
| Layer 7 AQ | 10 (0.3%) | 5 (0.2%) | -5 |
| **Total** | **38 (0.4%)** | **14 (0.2%)** | **-24** |

**Unexpected result:** Text found FEWER AQ candidates than random tokens.

---

## Interpretation

### Why Fewer AQ with Text?

This result contradicts the initial prediction that text would show MORE universality due to semantic structure. Possible explanations:

1. **Random tokens may inflate artificial alignment**
   - Meaningless input might trigger similar "fallback" patterns in both models
   - These shared error-handling patterns create spurious correlations
   - Real text engages actual language processing, which may differ more between models

2. **Text creates more activation variance**
   - Semantic content causes more diverse activation patterns
   - Higher variance → lower average correlations → fewer pass threshold
   - The 0.3 threshold may be too strict for text-based correlations

3. **Sample diversity matters**
   - The fallback text samples may not be diverse enough
   - 40 sentences across ~8 domains, combined for variety
   - Still less diverse than random token space

4. **Different layer dynamics**
   - Random: decreasing AQ with depth (15 → 13 → 10)
   - Text: increasing AQ with depth (2 → 7 → 5, peak at middle)
   - Suggests different processing patterns

### What This Might Mean for AKIRA

**If random tokens create MORE apparent universality than text:**

The "universal features" found with random tokens may be:
- Low-level structural features (position, basic patterns)
- Error-handling / fallback mechanisms
- NOT semantically meaningful Action Quanta

**The 14 text AQ may be more meaningful:**
- They survive a harder test (semantic variation)
- They might represent true semantic universality
- Worth investigating what they encode

---

## Key Question: Overlap Analysis

**Critical next step:** Do the 14 text AQ overlap with the 38 random AQ?

| Scenario | Interpretation |
|----------|----------------|
| High overlap | True universal features that appear regardless of input |
| Low overlap | Different types of universality (structural vs semantic) |
| Zero overlap | Random and text test different aspects entirely |

**To compute overlap:**
```python
# From results objects:
random_aq_indices = {
    3: results_random[3]['aq_indices'],
    5: results_random[5]['aq_indices'],
    7: results_random[7]['aq_indices']
}
text_aq_indices = {
    3: results_text[3]['aq_indices'],
    5: results_text[5]['aq_indices'],
    7: results_text[7]['aq_indices']
}

# Per layer overlap:
for layer in [3, 5, 7]:
    overlap = set(random_aq_indices[layer]) & set(text_aq_indices[layer])
    print(f"Layer {layer}: {len(overlap)} neurons in both tests")
```

---

## Threshold Sensitivity Note

**The 0.3 threshold remains arbitrary.** Results might differ significantly with:
- Lower threshold (0.2): Would find more AQ in both tests
- Statistical threshold (mean + 2*std): Would adapt to each distribution
- The aligned version (`001_action_quanta_extraction_aligned.py`) computes threshold from data

---

## Layer Pattern Analysis

```
Random tokens:  Layer 3 > Layer 5 > Layer 7  (decreasing)
Real text:      Layer 3 < Layer 5 > Layer 7  (middle peak)
```

**Interpretation:**
- Random: Early layers have most universal "structural" features
- Text: Middle layers (5) have most universal "semantic" features
- This aligns with known layer specialization in transformers:
  - Early layers: tokenization, basic patterns
  - Middle layers: syntactic/semantic features
  - Late layers: task-specific representations

---

## Conclusions

1. **Counter-intuitive result:** Random tokens showed MORE apparent universality than text
   
2. **Possible explanation:** Random tokens trigger shared "fallback" patterns that inflate correlations

3. **The 14 text AQ are potentially more meaningful** as true semantic universal features

4. **Overlap analysis needed** to determine if random and text AQ are the same or different features

5. **Aligned analysis recommended** - current method assumes neuron-by-index correspondence, which is naive

---

## Next Steps

1. **Run overlap analysis** between random and text AQ indices
2. **Run aligned version** (`001_action_quanta_extraction_aligned.py`) for rigorous comparison
3. **Lower threshold** to 0.2 and compare results
4. **Investigate the 14 text AQ** - what do they encode?

---

*Oscar Goldman - Shogu Research Group @ Datamutant.ai*

*"Finding fewer features with meaningful input than with noise suggests the noise test may be measuring something other than true universality. The semantic test may be the harder, more valid test."*
