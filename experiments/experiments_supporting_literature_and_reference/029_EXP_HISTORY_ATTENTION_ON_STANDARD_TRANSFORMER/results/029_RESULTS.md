# Experiment 029 Results: History Attention on Standard Transformer

## Executive Summary

**History attention improves a standard GPT-2 style transformer.**

Adding per-position temporal memory to standard causal self-attention reduces perplexity by 6.74% on WikiText-2. Belief history (storing attention outputs) outperforms token history (storing raw embeddings), consistent with Experiment 028.

## Results

| Model | Loss | Perplexity | vs Baseline |
|-------|------|------------|-------------|
| Baseline (Standard GPT-2) | 1.9614 | 7.11 | - |
| + Token History | 1.9021 | 6.70 | +5.75% |
| + Belief History | 1.8916 | 6.63 | +6.74% |

### Key Findings

1. **History attention helps**: Both token and belief history improve over baseline
2. **Belief > Token**: Storing processed states outperforms raw embeddings (+1.05%)
3. **Consistent with Exp 028**: The belief propagation hypothesis holds on real data

## What This Means

### The Architecture Works

The hybrid attention design combines two complementary mechanisms:

```
Standard Transformer Layer:
    Input -> [Self-Attention] -> [FFN] -> Output
             horizontal only

With History Attention:
    Input -> [History Attn] -> [Self-Attention] -> [FFN] -> Output
             vertical          horizontal
             (across time)     (within sequence)
```

Each position maintains memory of what it computed before. This "vertical" attention adds temporal context that standard "horizontal" attention cannot capture.

### Why Belief History Wins

**Token history stores**: Raw observations (what came in)

**Belief history stores**: Processed understanding (what the model concluded)

The belief buffer accumulates the model's evolving interpretation, not just raw data. When attending to history, the model retrieves its previous conclusions rather than re-processing raw inputs from scratch.

This is analogous to human memory - we remember our understanding of events, not raw sensory data.

### Practical Implications

1. **Drop-in improvement**: History attention can be added to existing transformers
2. **Modest overhead**: Extra parameters come from history projections (~10-15% increase)
3. **Streaming benefit**: Particularly useful for dialogue, real-time, or long-context tasks

## Experimental Details

### Configuration

```
vocab_size: 50257 (GPT-2)
embed_dim: 256
num_layers: 6
num_heads: 8
max_seq_length: 256
max_history: 64
total_steps: 5000
batch_size: 16
learning_rate: 3e-4
```

### Dataset

WikiText-2 (small, standard benchmark):
- Train: ~36K sequences
- Validation: ~3.7K sequences
- Tokenizer: GPT-2

### Hardware

Google Colab (GPU runtime)

## Connection to AKIRA Theory

This experiment validates a core AKIRA principle: **belief propagation over time**.

From the theory:
- Attention outputs contain *processed, contextualized* states
- These represent the system's understanding, not raw observations
- Propagating beliefs across time creates richer temporal context

The 6.74% improvement on real language data supports this hypothesis.

## Limitations

1. **Small model**: 256-dim embedding, 6 layers
2. **Short training**: 5000 steps
3. **Single dataset**: WikiText-2 only
4. **Fixed history length**: 64 steps

Further experiments needed:
- Larger models (GPT-2 medium/large scale)
- More training steps
- Multiple datasets (code, dialogue, long documents)
- Variable history lengths
- Integration with pretrained models

## Conclusion

**History attention is a valid enhancement for standard transformers.**

The belief propagation mechanism from AKIRA theory transfers to practical language modeling, providing measurable improvements with modest additional complexity.

---

AKIRA Project - Experiment 029 Results
Oscar Goldman - Shogu Research Group @ Datamutant.ai

Date: December 2024
