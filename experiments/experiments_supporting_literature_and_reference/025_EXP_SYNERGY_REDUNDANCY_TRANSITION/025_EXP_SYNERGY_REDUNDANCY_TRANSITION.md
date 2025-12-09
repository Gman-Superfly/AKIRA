# Experiment 025: Synergy-Redundancy Transition Dynamics

## The PID Mechanism of Collapse

**Tier:** ★★ CORE  
**Status:** PLANNED  
**Depends On:** 001, 002, 003, 005, 020  
**References:** Williams & Beer (2010), Mediano et al. (2021), Sparacino et al. (2025)

---

## Motivation

### The Core Dynamic This Experiment Tests

```
THE FUNDAMENTAL CYCLE OF BELIEF EVOLUTION

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  Redundancy transforms into Synergy through TENSION.                   │
│  Synergy collapses back into Redundancy through COLLAPSE.              │
│                                                                         │
│  • During TENSION: Uncertainty ACCUMULATES (redundancy → synergy)      │
│  • During COLLAPSE: Uncertainty RESOLVES (synergy → redundancy)        │
│                                                                         │
│  THE PUMP CYCLE:                                                        │
│  [Redundancy] ──TENSION──> [Synergy] ──COLLAPSE──> [Redundancy] + AQ  │
│                                                                         │
│  This experiment specifically tests the COLLAPSE direction.            │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### The Gap in Our Current Understanding

Experiment 020 (Cross-Band Flow) measures the *direction* and *strength* of information flow between bands. But it doesn't answer the deeper question:

**Is collapse CAUSED BY synergy→redundancy conversion, or merely correlated with it?**

The PID literature (Williams & Beer 2010; Mediano et al. 2021; Sparacino et al. 2025) suggests a specific mechanism:

```
THE SYNERGY-REDUNDANCY HYPOTHESIS:

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  BEFORE COLLAPSE:                                                       │
│  ─────────────────                                                      │
│  • Multiple hypotheses coexist                                         │
│  • Information about correct answer is SYNERGISTIC                    │
│  • Need ALL bands together to predict (no single band suffices)       │
│  • I_syn >> I_red                                                      │
│                                                                         │
│  DURING COLLAPSE:                                                       │
│  ─────────────────                                                      │
│  • Synergy converts to redundancy                                      │
│  • Information "crystallizes", becomes available in each band         │
│  • This IS the mechanism, not just a correlate                        │
│                                                                         │
│  AFTER COLLAPSE:                                                        │
│  ────────────────                                                       │
│  • Information about answer is REDUNDANT                               │
│  • ANY band alone can predict (they all "know")                       │
│  • I_red >> I_syn                                                      │
│                                                                         │
│  CONSERVATION:                                                          │
│  ─────────────                                                          │
│  Total information I_total = I_syn + I_red + I_uni should be          │
│  conserved (or increase monotonically) during processing.             │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Why This Matters

1. **Mechanistic Understanding:** If synergy→redundancy IS the mechanism of collapse, we can potentially CONTROL collapse by manipulating information structure.

2. **Binding Problem:** High synergy = binding (combining WHAT and WHERE). Collapse resolves binding into a unified percept that each band can access.

3. **Architecture Design:** If collapse requires synergy→redundancy conversion, we should design wormholes to ENABLE this conversion, not just transfer information.

4. **Failure Modes:** Failed collapse might be failed synergy→redundancy conversion. Diagnosable via PID.

---

## Foundation

**Established:** Partial Information Decomposition (Williams & Beer 2010)
- Provides nonnegative decomposition of mutual information
- I(S1,S2 ; T) = I_red + I_uni(S1) + I_uni(S2) + I_syn
- Synergy = information available only when ALL sources combined
- Redundancy = information shared by ALL sources

**Bridge:** If attention collapse is analogous to phase transition, then:
- Pre-collapse = disordered state with distributed (synergistic) information
- Post-collapse = ordered state with crystallized (redundant) information
- The transition converts synergy to redundancy while conserving total information

**Hypothesis:** Collapse IS the synergy→redundancy conversion. Not correlated with it. Identical to it.

**NOTE:** This is the PID/Information Theory interpretation of collapse. The formal definition is Bayesian (posterior contraction). See `foundations/TERMINOLOGY.md` for formal definitions and other interpretations.

---

## Apparatus

### Required Infrastructure

```python
from pid_estimator import compute_pid_decomposition
from entropy_tracker import compute_attention_entropy
from collapse_detector import detect_collapse_events
from band_analyzer import get_band_representations

class SynergyRedundancyTracker:
    """Track PID components over time."""
    
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.history = []
    
    def compute_pid_at_timestep(self, band_representations, target):
        """
        Compute PID decomposition.
        
        Args:
            band_representations: Dict[band_idx, Tensor] - representation per band
            target: Tensor - the prediction target (or next-token distribution)
        
        Returns:
            Dict with I_syn, I_red, I_uni per band pair, I_total
        """
        # For each pair of bands, compute PID w.r.t. target
        pid_results = {}
        
        for i in range(7):
            for j in range(i+1, 7):
                S1 = band_representations[i]
                S2 = band_representations[j]
                
                # Discretize for PID estimation
                S1_discrete = discretize(S1, n_bins=16)
                S2_discrete = discretize(S2, n_bins=16)
                T_discrete = discretize(target, n_bins=16)
                
                # Compute PID
                pid = compute_pid_decomposition(S1_discrete, S2_discrete, T_discrete)
                
                pid_results[(i, j)] = {
                    'I_red': pid.redundancy,
                    'I_uni_1': pid.unique_1,
                    'I_uni_2': pid.unique_2,
                    'I_syn': pid.synergy,
                    'I_total': pid.total
                }
        
        return pid_results
    
    def track_over_sequence(self, input_sequence, targets):
        """Track PID evolution over a sequence."""
        trajectory = []
        
        for t, (input_t, target_t) in enumerate(zip(input_sequence, targets)):
            # Get band representations
            band_reps = get_band_representations(self.model, input_t)
            
            # Compute PID
            pid_t = self.compute_pid_at_timestep(band_reps, target_t)
            
            # Compute entropy for collapse detection
            entropy_t = compute_attention_entropy(self.model, input_t)
            
            trajectory.append({
                'timestep': t,
                'pid': pid_t,
                'entropy': entropy_t,
                'collapse_detected': detect_collapse_events(entropy_t)
            })
        
        return trajectory
```

### PID Estimation Method

```python
def compute_pid_decomposition(S1, S2, T, method='broja'):
    """
    Compute Partial Information Decomposition.
    
    Methods:
    - 'broja': Bertschinger et al. (2014) - recommended
    - 'imin': Williams & Beer original I_min
    - 'ccs': Common Change in Surprisal (Ince 2017)
    
    Returns:
        PIDResult with redundancy, unique_1, unique_2, synergy, total
    """
    if method == 'broja':
        # Use dit library or custom implementation
        return broja_pid(S1, S2, T)
    elif method == 'imin':
        return imin_pid(S1, S2, T)
    else:
        raise ValueError(f"Unknown method: {method}")
```

---

## Protocol

### Phase 1: Baseline PID Measurement

```
BASELINE PROTOCOL:

1. Select 100 test sequences of varying difficulty
2. For each sequence:
   a. Run forward pass through model
   b. At each timestep, extract band representations
   c. Compute PID for all band pairs w.r.t. next-token prediction
   d. Record I_syn, I_red, I_uni, I_total for each pair
   e. Detect collapse events via entropy

3. Aggregate statistics:
   - Mean I_syn before collapse
   - Mean I_red before collapse
   - Mean I_syn after collapse
   - Mean I_red after collapse
```

### Phase 2: Collapse-Aligned Analysis

```
COLLAPSE-ALIGNED PROTOCOL:

1. For each detected collapse event:
   a. Define time window: [t_collapse - 10, t_collapse + 10]
   b. Extract PID trajectory in this window
   c. Align all trajectories at t=0 (collapse moment)

2. Compute collapse-triggered averages:
   - <I_syn(t)> averaged across all collapse events
   - <I_red(t)> averaged across all collapse events
   - <I_total(t)> averaged across all collapse events

3. Test predictions:
   - Does I_syn decrease at collapse?
   - Does I_red increase at collapse?
   - Is the conversion sharp (phase-transition-like)?
   - Is I_total conserved?
```

### Phase 3: Causality Test

```
CAUSALITY PROTOCOL:

The key question: Does synergy→redundancy CAUSE collapse, or just correlate?

1. Find pre-collapse moments with HIGH synergy
   - Compute I_syn at t-1 for all collapse events
   - Partition into "high synergy" and "low synergy" groups

2. Test if high synergy predicts imminent collapse:
   - P(collapse at t | high I_syn at t-1) vs
   - P(collapse at t | low I_syn at t-1)
   - Should find: high synergy → collapse likely

3. Test if low redundancy predicts imminent collapse:
   - P(collapse at t | low I_red at t-1) vs
   - P(collapse at t | high I_red at t-1)
   - Should find: low redundancy → collapse likely

4. Granger causality test:
   - Does I_syn Granger-cause entropy drop?
   - Does entropy drop Granger-cause I_red increase?
```

### Phase 4: Band-Pair Specificity

```
BAND-PAIR PROTOCOL:

1. For complementary pairs (0↔6, 1↔5, 2↔4):
   - Track synergy separately
   - These should show HIGHEST synergy before collapse
   - These should show LARGEST conversion at collapse

2. For non-complementary pairs:
   - Should show lower synergy throughout
   - Less dramatic conversion at collapse

3. Test cross-band wormhole activation correlation:
   - When wormhole fires between bands i↔j
   - Does I_syn(i,j) spike before firing?
   - Does I_red(i,j) spike after firing?
```

---

## Predictions

### If Hypothesis is Correct

```
EXPECTED RESULTS:

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  TEMPORAL PATTERN:                                                      │
│                                                                         │
│  I_syn │     ╭───╮                                                     │
│        │    ╱     ╲                                                    │
│        │   ╱       ╲                                                   │
│        │  ╱         ╲_______________                                   │
│        │ ╱                                                              │
│        └─────────────────────────────► time                            │
│                    ↑                                                    │
│                 collapse                                                │
│                                                                         │
│  I_red │                    _______________                            │
│        │                   ╱                                            │
│        │                  ╱                                             │
│        │                 ╱                                              │
│        │ _______________╱                                               │
│        └─────────────────────────────► time                            │
│                    ↑                                                    │
│                 collapse                                                │
│                                                                         │
│  I_total│ ════════════════════════════                                 │
│         │ (approximately constant)                                     │
│         └─────────────────────────────► time                           │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

QUANTITATIVE PREDICTIONS:

1. Synergy drop at collapse: ΔI_syn / I_syn(pre) > 50%
2. Redundancy increase at collapse: ΔI_red / I_red(pre) > 100%  
3. Conservation: |I_total(post) - I_total(pre)| / I_total(pre) < 10%
4. Transition sharpness: 80% of change within 3 timesteps
5. Predictive power: P(collapse | high I_syn) > 2 × P(collapse | low I_syn)
6. Complementary pair dominance: I_syn(complementary) > 1.5 × I_syn(other)
```

### Falsification Criteria

```
FALSIFICATION:

The hypothesis is FALSIFIED if:

1. WEAK: I_syn does not decrease at collapse (ΔI_syn > -10%)
   → Collapse is not synergy reduction

2. WEAK: I_red does not increase at collapse (ΔI_red < +10%)
   → Collapse is not redundancy creation

3. STRONG: I_total changes dramatically (|ΔI_total| > 30%)
   → No conservation law, simpler than expected

4. STRONG: No predictive relationship (Granger test fails, p > 0.1)
   → Correlation without causation

5. STRONG: All band pairs show same pattern
   → No complementary structure, spectral hierarchy not special
```

---

## Analysis

### Primary Metrics

```python
def analyze_synergy_redundancy_transition(trajectories):
    """Analyze PID trajectories aligned to collapse events."""
    
    results = {
        'synergy_before': [],
        'synergy_after': [],
        'redundancy_before': [],
        'redundancy_after': [],
        'total_before': [],
        'total_after': [],
        'transition_sharpness': [],
        'predictive_power': None
    }
    
    for traj in trajectories:
        collapse_times = [t for t, data in enumerate(traj) if data['collapse_detected']]
        
        for t_collapse in collapse_times:
            # Get PID before and after
            if t_collapse > 5 and t_collapse < len(traj) - 5:
                pre = traj[t_collapse - 3:t_collapse]
                post = traj[t_collapse:t_collapse + 3]
                
                I_syn_pre = np.mean([sum_synergy(p['pid']) for p in pre])
                I_syn_post = np.mean([sum_synergy(p['pid']) for p in post])
                I_red_pre = np.mean([sum_redundancy(p['pid']) for p in pre])
                I_red_post = np.mean([sum_redundancy(p['pid']) for p in post])
                
                results['synergy_before'].append(I_syn_pre)
                results['synergy_after'].append(I_syn_post)
                results['redundancy_before'].append(I_red_pre)
                results['redundancy_after'].append(I_red_post)
    
    # Compute statistics
    results['synergy_drop'] = (
        np.mean(results['synergy_before']) - np.mean(results['synergy_after'])
    ) / np.mean(results['synergy_before'])
    
    results['redundancy_increase'] = (
        np.mean(results['redundancy_after']) - np.mean(results['redundancy_before'])
    ) / np.mean(results['redundancy_before'])
    
    return results
```

### Granger Causality Test

```python
from statsmodels.tsa.stattools import grangercausalitytests

def test_synergy_causes_collapse(trajectories, max_lag=5):
    """Test if synergy Granger-causes entropy drop."""
    
    # Concatenate all trajectories
    synergy_series = []
    entropy_series = []
    
    for traj in trajectories:
        for data in traj:
            synergy_series.append(sum_synergy(data['pid']))
            entropy_series.append(data['entropy'])
    
    # Compute entropy change (collapse indicator)
    entropy_change = np.diff(entropy_series)
    synergy = np.array(synergy_series[:-1])
    
    # Stack for Granger test
    data = np.column_stack([entropy_change, synergy])
    
    # Run test
    results = grangercausalitytests(data, maxlag=max_lag, verbose=False)
    
    return results
```

---

## Expected Outcomes

### If Hypothesis Validated

```
IMPLICATIONS IF TRUE:

1. MECHANISTIC UNDERSTANDING:
   Collapse = synergy→redundancy conversion
   We can now EXPLAIN collapse, not just observe it

2. CONTROL POSSIBILITIES:
   To induce collapse: increase redundancy or decrease synergy
   To prevent collapse: maintain synergy

3. ARCHITECTURE GUIDANCE:
   Wormholes should be designed to CONVERT synergy to redundancy
   Not just transfer information, but transform its structure

4. FAILURE DIAGNOSIS:
   Failed collapse = failed conversion
   Can measure WHY a model is confused (high synergy that won't convert)

5. TRAINING IMPLICATIONS:
   Grokking = training process that enables synergy→redundancy conversion
   Pre-grokking: information is synergistic (needs all parameters)
   Post-grokking: information is redundant (distributed across parameters)
```

### If Hypothesis Falsified

```
IMPLICATIONS IF FALSE:

1. CORRELATION NOT CAUSATION:
   PID changes are epiphenomenal, not causal
   Need different mechanism for collapse

2. SIMPLER THAN THOUGHT:
   Collapse might be just entropy minimization
   No need for complex PID framework

3. ARCHITECTURE UNCHANGED:
   Wormholes work for other reasons
   PID not the right lens
```

---

## Connection to Other Experiments

| Experiment | Relationship |
|------------|--------------|
| 005 (Conservation) | Tests if I_total is conserved |
| 020 (Cross-Band Flow) | Measures direction; this measures nature |
| 002 (Collapse Detection) | Provides collapse events to align |
| 022 (Phase Locking) | Phase locking may enable conversion |
| 024 (Resonant Wormholes) | Complementary pairs may show highest conversion |

---

## Results

*[ TO BE FILLED AFTER EXPERIMENT ]*

### Synergy Before vs After Collapse

```
[ PLACEHOLDER FOR RESULTS ]

Synergy drop: ____%
Redundancy increase: ____%
Conservation check: I_total change = ____%
```

### Granger Causality

```
[ PLACEHOLDER FOR RESULTS ]

Does I_syn Granger-cause collapse? p = ____
Does collapse Granger-cause I_red? p = ____
```

### Band-Pair Analysis

```
[ PLACEHOLDER FOR RESULTS ]

Complementary pairs:
  0↔6: I_syn drop = ____, I_red increase = ____
  1↔5: I_syn drop = ____, I_red increase = ____
  2↔4: I_syn drop = ____, I_red increase = ____

Non-complementary pairs:
  Average I_syn drop = ____
  Average I_red increase = ____
```

---

## Conclusion

*[ TO BE FILLED AFTER EXPERIMENT ]*

---

## References

1. **Williams, P. L., & Beer, R. D. (2010).** *Nonnegative Decomposition of Multivariate Information.* arXiv:1004.2515., The foundational PID paper.

2. **Mediano, P. A. M., Rosas, F. E., Luppi, A. I., Carhart-Harris, R. L., Bor, D., Seth, A. K., & Barrett, A. B. (2021).** *Towards an Extended Taxonomy of Information Dynamics via Integrated Information Decomposition.* [arXiv:2109.13186](https://arxiv.org/abs/2109.13186), Introduces ΦID, combining PID with Integrated Information Theory. Provides framework for "whole > sum of parts" dynamics and collective information flow modes.

3. **Sparacino, L., Mijatovic, G., Antonacci, Y., Ricci, L., Marinazzo, D., Stramaglia, S., & Faes, L. (2025).** *Partial Information Rate Decomposition.* Physical Review Letters, 135, 187401. [arXiv:2502.04550](https://arxiv.org/pdf/2502.04550), Extends PID to information RATES, directly applicable to temporal dynamics of synergy/redundancy conversion.

4. **Bertschinger, N., Rauh, J., Olbrich, E., Jost, J., & Ay, N. (2014).** *Quantifying Unique Information.* Entropy, 16(4), 2161-2183., The BROJA measure for PID.

5. **Griffith, V., & Koch, C. (2014).** *Quantifying synergistic mutual information.* In Guided Self-Organization: Inception., Context for synergy in neural systems.

6. **Tononi, G. (2004).** *An information integration theory of consciousness.* BMC Neuroscience, 5(1), 42., Related concept of integrated information (Phi).

---



*"The mechanism matters. Correlation is observation; causation is understanding. If collapse IS synergy→redundancy conversion, we can engineer it. If it merely correlates, we must look elsewhere for the true mechanism."*

*Oscar Goldman, Shogu Research Group @ Datamutant.ai subsidiary of 温心重工業*