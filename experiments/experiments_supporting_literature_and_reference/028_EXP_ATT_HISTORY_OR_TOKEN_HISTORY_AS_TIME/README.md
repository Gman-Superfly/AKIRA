# 028 - Attention History vs Token History as Temporal Memory

## Core Question

Should temporal memory store **raw observations** (tokens/samples) or **processed beliefs** (attention outputs)?

## The Insight: Belief Propagation Over Time

Attention outputs represent the system's **beliefs** - contextualized, integrated understanding that incorporates neighbor information, cross-band communication, and temporal processing.

Storing attention history enables **belief propagation** - each position maintains a trajectory of understanding, not just raw data.

## Two Complementary Experiments

### Experiment A: Token Domain
- Language modeling task
- Compare token history vs attention history
- Metrics: perplexity, convergence, belief stability

### Experiment B: Signal Domain  
- Array reconstruction task
- Compare signal history vs attention history
- Metrics: MSE, spectral accuracy, temporal coherence

## Quick Start

```bash
cd code
python exp_a_token_domain.py   # Token experiment
python exp_b_signal_domain.py  # Signal experiment
python analysis.py             # Compare results
```

## Key Files

- `028_EXP_ATT_HISTORY_OR_TOKEN_HISTORY_AS_TIME.md` - Full documentation
- `code/exp_a_token_domain.py` - Token/LM experiment
- `code/exp_b_signal_domain.py` - Signal/array experiment
- `code/belief_buffer.py` - Shared belief history implementation

## Connection to Conservation Laws

Belief propagation conserves probability mass - this connects to symmetry principles and conserved quantities in the AKIRA framework.

## Citation

If you use this experiment in your research, please cite it. This is ongoing work - we would appreciate knowing about your opinions and experiments.

Authors: Oscar Goldman - Shogu Research Group @ Datamutant.ai subsidiary of 温心重工業
