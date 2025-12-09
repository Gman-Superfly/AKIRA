# Experiment 029: History Attention on Standard Transformer

## Overview

This experiment tests whether adding **history attention** (per-position temporal memory) to a standard GPT-2 style transformer improves language modeling performance.

## Hypothesis

From Experiment 028, we learned that:
1. Belief history (storing attention outputs) outperforms token history (storing raw embeddings)
2. The architecture combines horizontal (within-sequence) and vertical (across-time) attention

This experiment tests whether this hybrid approach improves a standard transformer on real language data.

## Experimental Design

### Three Configurations

1. **Baseline**: Standard GPT-2 transformer (causal self-attention only)
2. **+ Token History**: Standard + history attention storing raw token embeddings
3. **+ Belief History**: Standard + history attention storing attention outputs

### Architecture Comparison

```
BASELINE (Standard GPT-2):
    Input -> [Self-Attention] -> [FFN] -> Output
    
+ HISTORY ATTENTION:
    Input -> [History Attention] -> [Self-Attention] -> [FFN] -> Output
              (vertical: across time)  (horizontal: within sequence)
```

### Dataset

**WikiText-2** (smaller, faster to train):
- Train: ~2M tokens
- Validation: ~200K tokens  
- Test: ~240K tokens
- Vocabulary: GPT-2 tokenizer (50257 tokens)

### Model Configuration

```python
vocab_size: 50257          # GPT-2 vocabulary
embed_dim: 256             # Small for speed
num_layers: 6              # Moderate depth
num_heads: 8               # Standard
max_seq_length: 256        # Context window
max_history: 64            # History buffer size
```

### Training

- Optimizer: AdamW
- Learning rate: 3e-4 with cosine annealing
- Batch size: 16
- Steps: 5000 (adjustable)
- Evaluation: Every 500 steps

## Expected Results

If history attention helps:
- Lower perplexity on validation set
- Belief history should outperform token history (from Exp 028)
- Improvement may be larger on repetitive/structured text

If history attention doesn't help:
- Standard attention already captures necessary patterns
- The overhead of history tracking isn't justified for this task

## Code Structure

```
029_EXP_HISTORY_ATTENTION_ON_STANDARD_TRANSFORMER/
  029_EXP_HISTORY_ATTENTION_ON_STANDARD_TRANSFORMER.md  (this document)
  README.md
  code/
    colab_history_attention_gpt2.py    (main Colab notebook)
```

## Running the Experiment

### On Google Colab

1. Upload `colab_history_attention_gpt2.py` to Colab
2. Select GPU runtime (recommended)
3. Run all cells
4. Results will be printed at the end

### Locally

```bash
python colab_history_attention_gpt2.py
```

## Metrics

1. **Perplexity** - Primary metric for language modeling
2. **Training loss** - Convergence speed
3. **Memory usage** - History buffer overhead
4. **Inference speed** - Tokens/second

## Connection to AKIRA Theory

This experiment validates whether the "belief propagation over time" concept from AKIRA improves standard language models:

- **Vertical attention**: Each position remembers what it computed before
- **Horizontal attention**: Standard within-sequence attention
- **Combined**: Richer temporal context for prediction

## Status

- [x] Run baseline (standard GPT-2)
- [x] Run + Token History
- [x] Run + Belief History
- [x] Compare results
- [x] Document findings

## Results Summary

| Model | Perplexity | Improvement |
|-------|------------|-------------|
| Baseline | 7.11 | - |
| + Token History | 6.70 | +5.75% |
| + Belief History | 6.63 | +6.74% |

**Conclusion**: History attention improves standard transformer. Belief history wins.

See `results/029_RESULTS.md` for full analysis.

---

AKIRA Project - Experiment 029
*Oscar Goldman, Shogu Research Group @ Datamutant.ai subsidiary of 温心重工業*