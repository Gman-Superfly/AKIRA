# 037: Task-Relative AQ Extraction

## Quick Summary

**Question:** Does the same input produce different Action Quanta under different task framings?

**Hypothesis:** AQ are emergent from signal-task interaction, not intrinsic properties of signals.

**Method:** Present identical inputs under different task prompts, compare activations, attention patterns, and load-bearing features.

**Key Prediction:** Features load-bearing for Task A should NOT be load-bearing for Task B (ablation asymmetry >5x).

## Status

- **Tier:** CORE
- **Status:** PLANNED
- **Dependencies:** 000, 001

## Files

- `037_EXP_TASK_RELATIVE_AQ_EXTRACTION.md` - Full experiment specification
- `code/` - Implementation (to be added)
- `results/` - Results (to be filled after experiment)

## Quick Start

```python
# Pseudocode for experiment
extractor = TaskRelativeAQExtractor(model, tokenizer)

# Same input, different tasks
input_text = "A red ball on a blue table"

result_color = extractor.extract_under_task(input_text, "What color is the ball?")
result_location = extractor.extract_under_task(input_text, "Where is the ball?")

# Compare: should show divergent activations
similarity = cosine_similarity(result_color['activations'], result_location['activations'])
# Prediction: similarity < 0.7 in late layers
```

## Connection to Theory

This experiment tests the claim from `ACTION_QUANTA.md`:

> "AQ are defined relative to a TASK. Different tasks → Different AQ. Same data → Different decompositions."

If validated, this supports the pragmatist view that meaning = action-enablement, not intrinsic signal property.
