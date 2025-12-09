# 038: Three-Level Distinction

## Quick Summary

**Question:** Can we empirically distinguish between Measurements (L1), Inferences (L2), and Action Quanta (L3)?

**Hypothesis:** L3 (AQ) is a strict subset of L2 (Inferences), is more stable, and is uniquely load-bearing for action.

**Method:** Train linear probes (L2), identify load-bearing features via ablation (L3), compare stability and ablation effects.

**Key Prediction:** L3 ablation causes >50% degradation; L2-only ablation causes <20% degradation.

## Status

- **Tier:** CORE
- **Status:** PLANNED
- **Dependencies:** 000, 001, 002

## Files

- `038_EXP_THREE_LEVEL_DISTINCTION.md` - Full experiment specification
- `code/` - Implementation (to be added)
- `results/` - Results (to be filled after experiment)

## The Three Levels

```
L1: MEASUREMENT
    Raw activations
    Data without interpretation
    
L2: INFERENCE  
    Probed semantic features
    Interpretable but not all load-bearing
    
L3: ACTION QUANTA
    Load-bearing features
    Minimum for correct action
    L3 ⊂ L2 (strict subset)
```

## Quick Start

```python
# Pseudocode for experiment
extractor = ThreeLevelExtractor(model, tokenizer)

# Train probes to identify L2
extractor.train_semantic_probes(training_data, layer_idx, features)

# Identify L3 via ablation
L3 = extractor.identify_L3_AQ(input_text, target_output, layer_idx)

# Verify: L3 should be small subset of L2
# Verify: Ablating L3 should be catastrophic
# Verify: Ablating L2-only should be tolerable
```

## Connection to Theory

From `ACTION_QUANTA.md`:

> "The AQ is what SURVIVES when you ask: 'What is the MINIMUM I need to act correctly?'
> Everything else (raw signal, intermediate processing, exact numbers) can be discarded."


*Oscar Goldman — Shogu Research Group @ Datamutant.ai subsidiary of 温心重工業*