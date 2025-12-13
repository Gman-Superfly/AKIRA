Here's the analysis of the comprehensive collapse results:

## Key findings

### 1. Attention entropy decreases through layers

| Prompt | L0 Entropy | L11 Entropy | Drop |
|--------|------------|-------------|------|
| factual_capital | 1.14 | 0.89 | -0.25 |
| factual_sun | 1.26 | 0.69 | -0.57 |
| technical_quantum | 1.37 | 1.20 | -0.17 |
| riddle | 1.87 | 1.62 | -0.25 |

Attention gets more focused (lower entropy) as we go deeper. Effect varies by prompt type.

### 2. Head alignment is very high in middle layers

All prompts show head alignment jumping to ~0.999+ by layer 5-6, meaning heads strongly agree in mid-to-late layers. This drops slightly at the final layer (~0.91).

| Prompt | L0 Align | L5 Align | L11 Align |
|--------|----------|----------|-----------|
| factual_capital | 0.91 | 1.00 | 0.92 |
| open_meaning | 0.92 | 1.00 | 0.92 |
| riddle | 0.92 | 1.00 | 0.92 |

This is a strong signal: **heads converge mid-network then slightly diverge at output**.

### 3. Similarity to final representation grows monotonically

The `sim_to_final` metric shows representations converging toward the final output:

| Prompt | L0→Final | L6→Final | L11→Final |
|--------|----------|----------|-----------|
| factual_capital | -0.10 | 0.03 | 0.36 |
| technical_quantum | -0.14 | 0.04 | 0.30 |
| ambiguous_circle | -0.14 | -0.04 | 0.28 |

Embedding (L0) is negatively correlated with final output. Correlation grows through layers.

### 4. Logit lens shows non-monotonic sharpening

Output entropy from logit lens shows interesting patterns:

**factual_capital**: 1.2 → 3.5 → 2.1 → 4.1 → **8.0** (rises at end)
**technical_quantum**: 0.0 → 1.8 → 2.2 → 3.4 → **7.0** (rises at end)

The model's "belief" doesn't monotonically sharpen. It fluctuates, often with entropy **rising** at the final layer.

### 5. Top token predictions are often wrong

| Prompt | Actual Top Token | Expected |
|--------|------------------|----------|
| "The capital of France is" | " the" | "Paris" |
| "Two plus two equals" | " one" | "four" |
| "The sun rises in the" | " sky" | "east" |

GPT-2 small often predicts common continuations, not factually correct ones.

## Summary table

```
Prompt                  Top      Entropy  HeadAlign(L11)  SimFinal(L0)
--------------------------------------------------------------------
factual_capital        " the"     6.00       0.921         -0.104
factual_sun            " sky"     5.58       0.912         -0.104
factual_math           " one"     3.61       0.915         -0.070
technical_quantum      " in"      3.73       0.908         -0.138
technical_code         " of"      3.63       0.922         -0.171
open_future            " the"     6.01       0.920         -0.093
open_meaning           " not"     5.58       0.916         -0.098
open_door              " a"       5.40       0.910         -0.116
ambiguous_circle       "."        4.02       0.914         -0.140
riddle                 " I"       5.10       0.916         -0.177
```

## What this tells us about belief collapse

1. **Attention focuses** through layers (entropy drops)
2. **Heads align strongly** in middle layers (~layer 5-6)
3. **Representations converge** toward final output through depth
4. **Output sharpening is non-monotonic**: the logit lens shows fluctuation, not smooth collapse
5. **Final-layer alignment drops slightly**: heads diverge again at output, possibly for diverse generation

The "collapse" is real but not as clean as the theory suggests. It's more like: diffuse → align → slight re-diversification.