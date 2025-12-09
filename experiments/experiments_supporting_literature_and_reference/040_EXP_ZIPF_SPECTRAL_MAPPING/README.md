# 040: Zipf-Spectral Mapping

## Quick Summary

**Question:** Do token embeddings naturally organize by Zipf rank in spectral space?

**Hypothesis:** Common tokens have energy in low spectral bands; rare tokens have energy in high spectral bands.

**Method:** Decompose vocabulary embeddings via FFT, correlate spectral centroid with Zipf rank.

**Key Prediction:** Spearman correlation r > 0.5 between log(Zipf rank) and spectral centroid.

## Status

- **Tier:** SUPPORTING
- **Status:** PLANNED
- **Dependencies:** 003, 034

## Files

- `040_EXP_ZIPF_SPECTRAL_MAPPING.md` - Full experiment specification
- `code/` - Implementation (to be added)
- `results/` - Results (to be filled after experiment)

## The Zipf-Spectral Hypothesis

```
TOKEN FREQUENCY → SPECTRAL BAND

Common tokens ("the", "is", "a")
  → Low Zipf rank
  → Low information (expected, unsurprising)
  → Energy in LOW spectral bands (DC component)

Rare tokens ("quasar", "mitochondria")  
  → High Zipf rank
  → High information (specific, surprising)
  → Energy in HIGH spectral bands (detail)
```

## Quick Start

```python
# Pseudocode for experiment
analyzer = ZipfSpectralAnalyzer(model, tokenizer)

# Analyze vocabulary
results = analyzer.analyze_vocabulary(n_tokens=5000)

# Compute correlation
correlation = analyzer.compute_correlation(results)
# Prediction: spearman_r > 0.5
```

## Connection to Theory

From `THE_LANGUAGE_OF_INFORMATION.md`:

> "This creates a NATURAL spectral structure:
> Band 0 (DC): the, a, is, of - Low information, Structure
> Band 4-5: domain-specific - Med-High information, Content  
> Band 6 (HF): rare/technical - High information, Specifics"

*Oscar Goldman — Shogu Research Group @ Datamutant.ai subsidiary of 温心重工業*
