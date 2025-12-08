# Experiment 028: Attention History vs Token History as Temporal Memory

## Core Question

What should each position's temporal memory contain?

**Option A - Token History**: Raw input values at this position over time
```
Position t remembers: [input[t-1], input[t-2], ..., input[t-128]]
"What was the raw observation at my position?"
```

**Option B - Attention History**: Processed attention states over time
```
Position t remembers: [belief[t-1], belief[t-2], ..., belief[t-128]]
"What did I believe/compute at my position?"
```

## Hypothesis: Belief Propagation Over Time

Attention history represents **belief propagation** - a fundamental process where:

1. Each position maintains a belief state (attention output)
2. Beliefs are updated based on new evidence (current input)
3. Beliefs are influenced by neighbors and other bands
4. The history of beliefs captures the **trajectory of understanding**

This connects to:
- **Conservation laws**: Belief/probability mass is conserved during propagation
- **Symmetry**: Time-translation symmetry implies conserved quantities
- **Information geometry**: Belief updates follow geodesics on probability manifolds

## Why Attention History Might Be Superior

| Aspect | Token History | Attention History |
|--------|---------------|-------------------|
| Information content | Raw, unprocessed | Contextualized, integrated |
| Neighbor info | None | Already incorporated |
| Cross-band info | None | Already incorporated |
| Temporal smoothing | Noisy | Filtered by processing |
| Belief state | Not available | Directly accessible |
| Gradient flow | Through input only | Through computation graph |

**Key insight**: Attention history contains the **integrated context** - information from neighbors, other bands, and temporal processing already combined. It represents the system's "current understanding" not just "current observation."

## Experimental Design

### Experiment A: Token Domain (Language Modeling)

Small-scale LM comparing:
- Model A1: History buffer stores token embeddings
- Model A2: History buffer stores attention outputs

**Setup**:
- Small vocabulary subset or synthetic token sequences
- 6-layer transformer variant
- Compare perplexity, convergence speed, representation quality

**Metrics**:
- Perplexity (standard)
- Belief stability (variance of attention outputs over time)
- Information retention (how much past context is accessible)

### Experiment B: Signal Domain (Array Reconstruction)

Array decoder comparing:
- Model B1: History buffer stores raw signal samples
- Model B2: History buffer stores attention outputs

**Setup**:
- Synthetic signals (sum of sinusoids, chirps, transients)
- 7 spectral bands with differential temporal windows
- Compare reconstruction MSE, spectral accuracy

**Metrics**:
- Reconstruction MSE
- Per-band spectral error
- Belief trajectory smoothness
- Temporal coherence

## Architecture Details

### History Buffer Management

```python
class BeliefHistoryBuffer:
    """
    Maintains rolling buffer of attention outputs (beliefs).
    
    Unlike token history (which is just input), this stores
    the PROCESSED states - the system's understanding.
    """
    
    def __init__(self, max_history: int, dim: int):
        self.max_history = max_history
        self.buffer = None  # [T_history, positions, dim]
    
    def update(self, attention_output: torch.Tensor):
        """Add new belief state to buffer."""
        # Roll and append
        if self.buffer is None:
            self.buffer = attention_output.unsqueeze(0)
        else:
            self.buffer = torch.cat([
                self.buffer[-(self.max_history-1):],
                attention_output.unsqueeze(0)
            ], dim=0)
    
    def get_history(self, position: int) -> torch.Tensor:
        """Get belief history for a specific position."""
        return self.buffer[:, position, :]  # [T_history, dim]
```

### Differential Windows (Heisenberg)

Band-specific history lengths remain:
- Band 0 (low freq): 128 belief states back
- Band 6 (high freq): 4 belief states back

But now each "state" is a processed belief, not a raw token.

### Belief Propagation Update

```python
def belief_update(current_input, belief_history, neighbors):
    """
    Update belief based on:
    1. Current evidence (input)
    2. Prior beliefs (history)  
    3. Neighbor beliefs (spatial context)
    
    This is approximate Bayesian inference.
    """
    # Attend to belief history (what I thought before)
    prior = attend_to_history(belief_history)
    
    # Incorporate current evidence
    likelihood = process_current(current_input)
    
    # Combine with neighbor context
    context = attend_to_neighbors(neighbors)
    
    # Posterior belief (new attention output)
    posterior = combine(prior, likelihood, context)
    
    return posterior
```

## Conservation Laws Connection

From symmetry principles:
- **Time translation**: If system is invariant under time shift, something is conserved
- **Belief propagation**: Total "belief mass" (probability) is conserved
- **Information**: Total information should be conserved (not created/destroyed)

The attention history approach naturally preserves these because:
1. Beliefs are updated, not replaced
2. History provides continuity
3. The update rule can be designed to conserve quantities

## Expected Results

**If attention history is superior**:
- Faster convergence (beliefs provide better initialization)
- More stable training (smoother gradients through belief history)
- Better long-range coherence (past beliefs inform current processing)
- Lower final loss (richer temporal context)

**If token history is superior**:
- Simpler (no recurrence-like dependencies)
- Less memory (tokens smaller than attention outputs)
- Easier to train (standard attention patterns)

## Code Structure

```
028_EXP_ATT_HISTORY_OR_TOKEN_HISTORY_AS_TIME/
  028_EXP_ATT_HISTORY_OR_TOKEN_HISTORY_AS_TIME.md  (this document)
  README.md
  code/
    exp_a_token_domain.py      (LM comparison)
    exp_b_signal_domain.py     (Array decoder comparison)
    belief_buffer.py           (Shared belief history implementation)
    analysis.py                (Comparison metrics and plots)
```

## Status

- [ ] Implement belief history buffer
- [ ] Experiment A: Token domain comparison
- [ ] Experiment B: Signal domain comparison
- [ ] Analyze conservation properties
- [ ] Compare convergence and final metrics
- [ ] Visualize belief trajectories

## References

- Experiment 027: Array Decoder with Spectral Bands
- Experiment 026: AKIRA Band Architecture
- Pandora folder: Conservation and symmetry principles
- Belief propagation literature (Pearl, Yedidia)

---

AKIRA Project - Experiment 028
Oscar Goldman - Shogu Research Group @ Datamutant.ai
