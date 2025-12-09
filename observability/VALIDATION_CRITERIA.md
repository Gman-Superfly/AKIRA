# VALIDATION CRITERIA

## How We Know If The Theories Are Correct

---

## Philosophy of Validation

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  GOOD THEORIES MAKE PREDICTIONS.                                       │
│  PREDICTIONS CAN BE WRONG.                                             │
│  BEING WRONG IS INFORMATIVE.                                           │
│                                                                         │
│  We prefer:                                                             │
│  • Falsifiable predictions over vague claims                          │
│  • Quantitative tests over qualitative impressions                    │
│  • Surprising predictions over obvious ones                           │
│  • Multiple independent tests over single experiments                 │
│                                                                         │
│  VALIDATION ≠ CONFIRMATION                                             │
│  We're looking for ways the theory could FAIL.                        │
│  Survival of attempted falsification = stronger validation.           │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 1. The Lightning Model

### Theory

```
THEORY: Collapse Dynamics Follow Lightning Pattern

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  CLAIMS:                                                                │
│  1. Error wavefront shows BRANCHING (stepped leaders)                 │
│  2. Collapse is SUDDEN (exponential, not linear)                      │
│  3. Winner-take-all (one hypothesis dominates)                        │
│  4. Recovery phase follows collapse                                    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Predictions

```
PREDICTIONS (Falsifiable)

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  P1.1: BRANCHING                                                        │
│  Before collapse, entropy map should show MULTIPLE high-entropy       │
│  regions that progressively merge or die.                             │
│                                                                         │
│  TEST: Track number of entropy peaks over time.                       │
│  EXPECT: Count decreases, then sudden drop to 1.                     │
│  FALSIFIED IF: Count stays constant or decreases linearly.           │
│                                                                         │
│  ─────────────────────────────────────────────────────────────────     │
│                                                                         │
│  P1.2: SUDDENNESS                                                       │
│  Entropy decay should be better fit by exponential than linear.       │
│                                                                         │
│  TEST: Fit both models, compare R².                                   │
│  EXPECT: R²_exp > R²_linear significantly.                           │
│  FALSIFIED IF: R²_linear ≥ R²_exp or difference < 0.1.               │
│                                                                         │
│  ─────────────────────────────────────────────────────────────────     │
│                                                                         │
│  P1.3: WINNER-TAKE-ALL                                                  │
│  Post-collapse, one attention target should dominate (>80% weight).  │
│                                                                         │
│  TEST: Measure max attention weight post-collapse.                    │
│  EXPECT: max_weight > 0.8 in >90% of cases.                          │
│  FALSIFIED IF: Typical max_weight < 0.5 or bimodal distribution.     │
│                                                                         │
│  ─────────────────────────────────────────────────────────────────     │
│                                                                         │
│  P1.4: RECOVERY                                                         │
│  After collapse, entropy should increase again (recovery phase).      │
│                                                                         │
│  TEST: Track entropy after collapse detection.                        │
│  EXPECT: Entropy rises within 10-20 steps.                           │
│  FALSIFIED IF: Entropy stays low indefinitely.                       │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Validation Code

```python
def validate_lightning_model(entropy_history, attention_history):
    """Validate all lightning model predictions."""
    
    results = {}
    
    # P1.1: Branching
    peak_counts = [count_entropy_peaks(e) for e in entropy_history]
    results['branching'] = {
        'observed': peak_counts,
        'decreases_before_collapse': is_decreasing_then_drops(peak_counts),
        'validated': max(peak_counts) > 1 and min(peak_counts) == 1
    }
    
    # P1.2: Suddenness
    collapse_events = detect_collapses(entropy_history)
    for event in collapse_events:
        curve = event['entropy_curve']
        r2_exp = fit_exponential(curve)
        r2_lin = fit_linear(curve)
        results.setdefault('suddenness', []).append({
            'r2_exponential': r2_exp,
            'r2_linear': r2_lin,
            'validated': r2_exp > r2_lin + 0.1
        })
    
    # P1.3: Winner-take-all
    post_collapse_weights = [a[-1].max() for a in attention_history 
                             if detect_collapse_in(a)]
    results['winner_take_all'] = {
        'max_weights': post_collapse_weights,
        'mean_max_weight': np.mean(post_collapse_weights),
        'validated': np.mean(post_collapse_weights) > 0.8
    }
    
    # P1.4: Recovery
    recovery_entropies = [e[collapse_idx:collapse_idx+20] 
                         for e, collapse_idx in zip(entropy_history, 
                                                     detect_collapse_indices(entropy_history))]
    results['recovery'] = {
        'entropy_increases': [e[-1] > e[0] for e in recovery_entropies],
        'validated': np.mean([e[-1] > e[0] for e in recovery_entropies]) > 0.8
    }
    
    # Overall
    results['model_validated'] = all([
        results['branching']['validated'],
        all(r['validated'] for r in results.get('suddenness', [{'validated': False}])),
        results['winner_take_all']['validated'],
        results['recovery']['validated']
    ])
    
    return results
```

### Falsification Criteria

```
WHAT WOULD FALSIFY THE LIGHTNING MODEL?

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  STRONG FALSIFICATION:                                                  │
│  • No branching observed (always single peak)                         │
│  • Collapse is linear, not exponential (R²_lin > R²_exp)             │
│  • Post-collapse weights are bimodal (two winners)                   │
│                                                                         │
│  WEAK FALSIFICATION:                                                    │
│  • Branching rare (<30% of cases)                                     │
│  • Collapse rate varies widely (no consistent pattern)               │
│  • Recovery is slow or inconsistent                                   │
│                                                                         │
│  IF FALSIFIED:                                                          │
│  • Revise model (maybe collapse is gradual for some inputs?)         │
│  • Look for conditions where lightning model holds                   │
│  • Consider alternative models                                        │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Spectral Hierarchy

### Theory

```
THEORY: Information Has Frequency, Low-Freq First

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  CLAIMS:                                                                │
│  1. Low-frequency bands collapse BEFORE high-frequency bands          │
│  2. Low-frequency bands have LOWER entropy (more stable)              │
│  3. Learning rates should follow hierarchy (slow for low-freq)        │
│  4. Prompts have spectral structure (role = low-freq, style = high)  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Predictions

```
PREDICTIONS (Falsifiable)

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  P2.1: COLLAPSE ORDER                                                   │
│  Band 0 collapses first, Band 6 collapses last.                       │
│                                                                         │
│  TEST: Measure collapse time per band, compute rank correlation.      │
│  EXPECT: Kendall's τ > 0.5 (positive correlation with band index).   │
│  FALSIFIED IF: τ ≤ 0 or collapse order random.                       │
│                                                                         │
│  ─────────────────────────────────────────────────────────────────     │
│                                                                         │
│  P2.2: ENTROPY HIERARCHY                                                │
│  At any time, entropy(band_i) < entropy(band_j) for i < j.           │
│                                                                         │
│  TEST: Compute per-band entropy, check ordering.                      │
│  EXPECT: Ordering preserved >80% of steps.                           │
│  FALSIFIED IF: Ordering random or reversed.                          │
│                                                                         │
│  ─────────────────────────────────────────────────────────────────     │
│                                                                         │
│  P2.3: LEARNING RATE MATCHING                                          │
│  Optimal learning rates for low-freq bands < high-freq bands.        │
│                                                                         │
│  TEST: Grid search learning rates per band, find optimal.             │
│  EXPECT: Optimal LR increases with band index.                       │
│  FALSIFIED IF: Optimal LRs are uniform or reversed.                  │
│                                                                         │
│  ─────────────────────────────────────────────────────────────────     │
│                                                                         │
│  P2.4: PROMPT SPECTRAL STRUCTURE                                        │
│  Role changes affect low-freq bands more; style changes affect high. │
│                                                                         │
│  TEST: Measure band response to prompt perturbations.                 │
│  EXPECT: Role changes → large Band 0-2 changes.                      │
│          Style changes → large Band 4-6 changes.                     │
│  FALSIFIED IF: All bands respond equally to all perturbations.       │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Validation Code

```python
def validate_spectral_hierarchy(per_band_entropy_history, 
                                 per_band_collapse_times,
                                 per_band_optimal_lr):
    """Validate spectral hierarchy predictions."""
    
    results = {}
    
    # P2.1: Collapse order
    expected_order = list(range(7))
    actual_order = sorted(range(7), key=lambda b: per_band_collapse_times.get(b, float('inf')))
    tau, p_value = stats.kendalltau(actual_order, expected_order)
    results['collapse_order'] = {
        'expected': expected_order,
        'actual': actual_order,
        'kendall_tau': tau,
        'p_value': p_value,
        'validated': tau > 0.5 and p_value < 0.05
    }
    
    # P2.2: Entropy hierarchy
    entropy_ordering_correct = []
    for step_entropies in per_band_entropy_history:
        # Check if band i has lower entropy than band j for i < j
        ordering_correct = all(
            step_entropies[i] <= step_entropies[j] + 0.1  # Small tolerance
            for i in range(7) for j in range(i+1, 7)
        )
        entropy_ordering_correct.append(ordering_correct)
    
    results['entropy_hierarchy'] = {
        'ordering_preserved_fraction': np.mean(entropy_ordering_correct),
        'validated': np.mean(entropy_ordering_correct) > 0.7
    }
    
    # P2.3: Learning rate matching
    lr_order = sorted(range(7), key=lambda b: per_band_optimal_lr[b])
    lr_tau, lr_p = stats.kendalltau(lr_order, expected_order)
    results['learning_rate_matching'] = {
        'optimal_lrs': per_band_optimal_lr,
        'lr_order': lr_order,
        'kendall_tau': lr_tau,
        'validated': lr_tau > 0.5
    }
    
    # Overall
    results['hierarchy_validated'] = all([
        results['collapse_order']['validated'],
        results['entropy_hierarchy']['validated'],
        results['learning_rate_matching']['validated']
    ])
    
    return results


def validate_prompt_spectral_structure(model, base_prompt, role_perturbations, style_perturbations):
    """Validate that prompts have spectral structure."""
    
    base_bands = compute_prompt_band_activations(model, base_prompt)
    
    role_band_changes = []
    for perturbed in role_perturbations:
        pert_bands = compute_prompt_band_activations(model, perturbed)
        changes = {b: abs(pert_bands[b] - base_bands[b]) for b in range(7)}
        role_band_changes.append(changes)
    
    style_band_changes = []
    for perturbed in style_perturbations:
        pert_bands = compute_prompt_band_activations(model, perturbed)
        changes = {b: abs(pert_bands[b] - base_bands[b]) for b in range(7)}
        style_band_changes.append(changes)
    
    # Average changes
    avg_role = {b: np.mean([c[b] for c in role_band_changes]) for b in range(7)}
    avg_style = {b: np.mean([c[b] for c in style_band_changes]) for b in range(7)}
    
    # Role should affect low bands more
    role_low = np.mean([avg_role[b] for b in range(3)])
    role_high = np.mean([avg_role[b] for b in range(4, 7)])
    
    # Style should affect high bands more
    style_low = np.mean([avg_style[b] for b in range(3)])
    style_high = np.mean([avg_style[b] for b in range(4, 7)])
    
    return {
        'role_affects_low': role_low > role_high,
        'style_affects_high': style_high > style_low,
        'role_low_high_ratio': role_low / (role_high + 1e-10),
        'style_high_low_ratio': style_high / (style_low + 1e-10),
        'validated': role_low > role_high and style_high > style_low
    }
```

### Falsification Criteria

```
WHAT WOULD FALSIFY SPECTRAL HIERARCHY?

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  STRONG FALSIFICATION:                                                  │
│  • Collapse order is random (τ ≈ 0)                                   │
│  • High-freq bands collapse BEFORE low-freq (τ < 0)                  │
│  • Entropy shows no band-wise pattern                                 │
│                                                                         │
│  WEAK FALSIFICATION:                                                    │
│  • Hierarchy holds for some inputs but not others                    │
│  • Middle bands don't follow pattern                                  │
│  • Effect exists but is weak (τ < 0.5)                               │
│                                                                         │
│  IF FALSIFIED:                                                          │
│  • Spectral decomposition may still be useful but not hierarchical   │
│  • Consider input-dependent hierarchy                                 │
│  • Revise band boundaries                                             │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Compression and Interference

### Theory

```
THEORY: Compression via Interference

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  CLAIMS:                                                                │
│  1. Shared structure REINFORCES (constructive interference)           │
│  2. Unique details CANCEL (destructive interference)                  │
│  3. Optimal prompts are "load-bearing" (minimal but essential)        │
│  4. MDL correlates with conceptual primitivity                        │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Predictions

```
PREDICTIONS (Falsifiable)

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  P3.1: SHARED STRUCTURE REINFORCES                                     │
│  Averaging multiple prompts with same meaning → stronger signal.      │
│                                                                         │
│  TEST: Compare signal strength of averaged vs individual prompts.     │
│  EXPECT: |avg(embeddings)| > mean(|embeddings|) for shared meaning.  │
│  FALSIFIED IF: Averaging always weakens signal.                       │
│                                                                         │
│  ─────────────────────────────────────────────────────────────────     │
│                                                                         │
│  P3.2: UNIQUE DETAILS CANCEL                                           │
│  Averaging prompts with different meanings → weaker signal.           │
│                                                                         │
│  TEST: Compare signal of random prompt average vs single prompt.      │
│  EXPECT: |avg(random_prompts)| < mean(|prompts|).                    │
│  FALSIFIED IF: Random averaging doesn't reduce signal.               │
│                                                                         │
│  ─────────────────────────────────────────────────────────────────     │
│                                                                         │
│  P3.3: LOAD-BEARING PROMPTS                                            │
│  Removing tokens from optimal prompts degrades performance more       │
│  than removing tokens from verbose prompts.                           │
│                                                                         │
│  TEST: Measure performance drop per token removed.                    │
│  EXPECT: Drop/token is higher for short optimized prompts.          │
│  FALSIFIED IF: All prompts have same drop/token.                     │
│                                                                         │
│  ─────────────────────────────────────────────────────────────────     │
│                                                                         │
│  P3.4: MDL CORRELATION                                                  │
│  Simpler concepts have shorter minimum description lengths.           │
│                                                                         │
│  TEST: Estimate MDL for concepts, compare to human complexity.        │
│  EXPECT: Correlation > 0.5 between MDL and rated complexity.         │
│  FALSIFIED IF: No correlation or negative correlation.               │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Validation Code

```python
def validate_interference_model(model, prompts_by_meaning, random_prompts):
    """Validate interference and compression predictions."""
    
    results = {}
    
    # P3.1: Shared structure reinforces
    reinforcement_results = []
    for meaning, prompts in prompts_by_meaning.items():
        embeddings = [get_prompt_embedding(model, p) for p in prompts]
        
        avg_embedding = torch.stack(embeddings).mean(dim=0)
        avg_magnitude = avg_embedding.norm().item()
        individual_magnitudes = [e.norm().item() for e in embeddings]
        mean_individual = np.mean(individual_magnitudes)
        
        reinforcement_results.append({
            'meaning': meaning,
            'avg_magnitude': avg_magnitude,
            'mean_individual': mean_individual,
            'reinforced': avg_magnitude > mean_individual * 0.9  # Allow some tolerance
        })
    
    results['reinforcement'] = {
        'results': reinforcement_results,
        'reinforcement_rate': np.mean([r['reinforced'] for r in reinforcement_results]),
        'validated': np.mean([r['reinforced'] for r in reinforcement_results]) > 0.7
    }
    
    # P3.2: Unique details cancel
    random_embeddings = [get_prompt_embedding(model, p) for p in random_prompts]
    random_avg = torch.stack(random_embeddings).mean(dim=0)
    random_avg_mag = random_avg.norm().item()
    random_mean_individual = np.mean([e.norm().item() for e in random_embeddings])
    
    results['cancellation'] = {
        'avg_magnitude': random_avg_mag,
        'mean_individual': random_mean_individual,
        'ratio': random_avg_mag / random_mean_individual,
        'validated': random_avg_mag < random_mean_individual * 0.9
    }
    
    # Overall
    results['interference_validated'] = (
        results['reinforcement']['validated'] and 
        results['cancellation']['validated']
    )
    
    return results


def validate_load_bearing(model, optimized_prompts, verbose_prompts, evaluation_fn):
    """Validate that optimized prompts are load-bearing."""
    
    def compute_drop_per_token(prompt, eval_fn):
        base_score = eval_fn(model, prompt)
        tokens = tokenize(prompt)
        
        drops = []
        for i in range(len(tokens)):
            reduced = tokens[:i] + tokens[i+1:]
            reduced_prompt = detokenize(reduced)
            reduced_score = eval_fn(model, reduced_prompt)
            drops.append(base_score - reduced_score)
        
        return np.mean(drops), np.std(drops)
    
    optimized_drops = [compute_drop_per_token(p, evaluation_fn) for p in optimized_prompts]
    verbose_drops = [compute_drop_per_token(p, evaluation_fn) for p in verbose_prompts]
    
    mean_optimized = np.mean([d[0] for d in optimized_drops])
    mean_verbose = np.mean([d[0] for d in verbose_drops])
    
    return {
        'optimized_drop_per_token': mean_optimized,
        'verbose_drop_per_token': mean_verbose,
        'ratio': mean_optimized / (mean_verbose + 1e-10),
        'validated': mean_optimized > mean_verbose * 1.5
    }
```

### Falsification Criteria

```
WHAT WOULD FALSIFY COMPRESSION MODEL?

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  STRONG FALSIFICATION:                                                  │
│  • Averaging same-meaning prompts weakens signal                      │
│  • Random averaging doesn't cause cancellation                        │
│  • All prompts equally load-bearing                                   │
│                                                                         │
│  WEAK FALSIFICATION:                                                    │
│  • Effect exists but is small                                         │
│  • Only works for some prompt types                                   │
│  • MDL estimation is unreliable                                       │
│                                                                         │
│  IF FALSIFIED:                                                          │
│  • Prompts may not behave like waves                                  │
│  • Compression may happen differently                                 │
│  • Revise interference model                                          │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Tickling Effectiveness

### Theory

```
THEORY: Cheap Probes Reveal Structure

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  CLAIMS:                                                                │
│  1. Free assets (entropy, etc.) predict collapse destination          │
│  2. Cheap probes correlate with full inference results                │
│  3. Edge detection enables efficiency gains                           │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Predictions

```
PREDICTIONS (Falsifiable)

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  P4.1: LEADER PREDICTION                                                │
│  High-entropy positions should become collapse destinations.          │
│                                                                         │
│  TEST: Correlate pre-collapse entropy with collapse location.         │
│  EXPECT: Correlation > 0.5.                                           │
│  FALSIFIED IF: Correlation ≤ 0 or collapse is random.                │
│                                                                         │
│  ─────────────────────────────────────────────────────────────────     │
│                                                                         │
│  P4.2: CHEAP-FULL CORRELATION                                          │
│  Temperature probe divergence predicts prompt quality.                │
│                                                                         │
│  TEST: Correlate probe results with full evaluation.                  │
│  EXPECT: Correlation > 0.7.                                           │
│  FALSIFIED IF: Correlation < 0.3.                                     │
│                                                                         │
│  ─────────────────────────────────────────────────────────────────     │
│                                                                         │
│  P4.3: EDGE DETECTION SPEEDUP                                          │
│  Using edge detection to stop probing early saves compute.            │
│                                                                         │
│  TEST: Compare compute to reach same accuracy with/without edge.      │
│  EXPECT: >30% compute reduction with edge detection.                  │
│  FALSIFIED IF: No significant speedup.                                │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Validation Code

```python
def validate_tickling_effectiveness(model, test_cases):
    """Validate that tickling techniques work."""
    
    results = {}
    
    # P4.1: Leader prediction
    leader_predictions = []
    for case in test_cases:
        # Get pre-collapse entropy
        pre_entropy = compute_entropy_map(model, case['input'])
        high_entropy_positions = (pre_entropy > 0.7).nonzero()
        
        # Get actual collapse location
        output = model(case['input'])
        attention = extract_attention_weights(model, case['input'])
        collapse_position = attention.values()[0].argmax()
        
        # Check if high entropy predicted collapse
        predicted = collapse_position in high_entropy_positions
        leader_predictions.append(predicted)
    
    results['leader_prediction'] = {
        'accuracy': np.mean(leader_predictions),
        'validated': np.mean(leader_predictions) > 0.5
    }
    
    # P4.2: Cheap-full correlation
    cheap_scores = []
    full_scores = []
    
    for case in test_cases:
        # Cheap probe
        probe_result = temperature_probe(model, case['input'])
        cheap_score = 1 - probe_result['divergence'][0.1]['l2']  # Low divergence = good
        cheap_scores.append(cheap_score)
        
        # Full evaluation
        full_score = case['ground_truth_quality']
        full_scores.append(full_score)
    
    correlation = np.corrcoef(cheap_scores, full_scores)[0, 1]
    results['cheap_full_correlation'] = {
        'correlation': correlation,
        'validated': correlation > 0.5
    }
    
    # P4.3: Edge detection speedup
    # Compare compute with and without edge detection
    without_edge_compute = []
    with_edge_compute = []
    
    for case in test_cases:
        # Without edge: probe until fixed iterations
        compute_without = 100  # Fixed iterations
        
        # With edge: probe until edge detected
        compute_with = 0
        for i in range(100):
            entropy = compute_entropy(model, case['input'])
            compute_with += 1
            if 0.3 < entropy < 0.9:  # At edge
                break
        
        without_edge_compute.append(compute_without)
        with_edge_compute.append(compute_with)
    
    speedup = 1 - np.mean(with_edge_compute) / np.mean(without_edge_compute)
    results['edge_speedup'] = {
        'speedup': speedup,
        'validated': speedup > 0.3
    }
    
    results['tickling_validated'] = all([
        results['leader_prediction']['validated'],
        results['cheap_full_correlation']['validated'],
        results['edge_speedup']['validated']
    ])
    
    return results
```

---

## 5. Conservation and Phase Transitions

### Theory

```
THEORY: Learning Shows Physical Behavior

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  CLAIMS:                                                                │
│  1. Some quantity is conserved during learning                        │
│  2. Grokking is a phase transition                                    │
│  3. Critical phenomena observable (power laws, scaling)               │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Predictions

```
PREDICTIONS (Falsifiable)

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  P5.1: CONSERVATION                                                     │
│  Total "actionable information" stays constant during grokking.       │
│                                                                         │
│  TEST: Measure explicit info (in data) + implicit info (in weights). │
│  EXPECT: Sum stays constant ± 10%.                                    │
│  FALSIFIED IF: Sum changes by >50%.                                   │
│                                                                         │
│  ─────────────────────────────────────────────────────────────────     │
│                                                                         │
│  P5.2: PHASE TRANSITION SIGNATURES                                     │
│  Grokking shows: sudden change, bimodal distribution, divergent       │
│  susceptibility near transition.                                       │
│                                                                         │
│  TEST: Look for these signatures in loss/accuracy curves.             │
│  EXPECT: At least 2 of 3 signatures present.                         │
│  FALSIFIED IF: Transition is gradual with no signatures.             │
│                                                                         │
│  ─────────────────────────────────────────────────────────────────     │
│                                                                         │
│  P5.3: POWER LAWS                                                       │
│  Near transition, observables follow power laws: X ~ |t-tc|^α.       │
│                                                                         │
│  TEST: Fit power law near transition, measure goodness of fit.        │
│  EXPECT: R² > 0.8 for power law fit.                                 │
│  FALSIFIED IF: No power law regime observable.                       │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Validation Code

```python
def validate_conservation(training_history):
    """Validate conservation during learning."""
    
    # Measure explicit (data) and implicit (weights) information
    explicit_info = []
    implicit_info = []
    
    for step in training_history:
        # Explicit: how much model relies on memorization
        explicit = step['train_loss_memorization_component']
        
        # Implicit: how much model has learned generalizable patterns
        implicit = step['test_accuracy_generalization_component']
        
        explicit_info.append(explicit)
        implicit_info.append(implicit)
    
    total_info = [e + i for e, i in zip(explicit_info, implicit_info)]
    
    # Check if total stays constant
    variation = np.std(total_info) / np.mean(total_info)
    
    return {
        'explicit': explicit_info,
        'implicit': implicit_info,
        'total': total_info,
        'variation': variation,
        'validated': variation < 0.1
    }


def validate_phase_transition(loss_curve, accuracy_curve):
    """Validate grokking is a phase transition."""
    
    results = {}
    
    # Signature 1: Sudden change
    derivatives = np.diff(accuracy_curve)
    max_derivative = max(derivatives)
    mean_derivative = np.mean(derivatives)
    results['sudden_change'] = max_derivative > 5 * mean_derivative
    
    # Signature 2: Bimodal distribution
    # Before transition: low accuracy mode
    # After transition: high accuracy mode
    transition_idx = np.argmax(derivatives)
    before = accuracy_curve[:transition_idx]
    after = accuracy_curve[transition_idx:]
    
    bimodal_test = np.mean(after) - np.mean(before) > 0.5 * (max(accuracy_curve) - min(accuracy_curve))
    results['bimodal'] = bimodal_test
    
    # Signature 3: Divergent susceptibility
    # Fluctuations peak near transition
    window = 10
    fluctuations = [np.std(accuracy_curve[max(0,i-window):i+window]) 
                    for i in range(len(accuracy_curve))]
    susceptibility_peak = np.argmax(fluctuations)
    near_transition = abs(susceptibility_peak - transition_idx) < 20
    results['susceptibility'] = near_transition
    
    results['signatures_present'] = sum([
        results['sudden_change'],
        results['bimodal'],
        results['susceptibility']
    ])
    
    results['validated'] = results['signatures_present'] >= 2
    
    return results
```

---

## Summary: Validation Checklist

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│           V A L I D A T I O N   C H E C K L I S T                      │
│                                                                         │
│  ═══════════════════════════════════════════════════════════════════   │
│                                                                         │
│  LIGHTNING MODEL:                                                       │
│  [ ] Branching observed before collapse                               │
│  [ ] Collapse is exponential, not linear                              │
│  [ ] Winner-take-all post-collapse                                    │
│  [ ] Recovery phase follows collapse                                  │
│                                                                         │
│  SPECTRAL HIERARCHY:                                                    │
│  [ ] Low-freq bands collapse before high-freq                        │
│  [ ] Entropy follows band hierarchy                                   │
│  [ ] Optimal learning rates match hierarchy                          │
│  [ ] Prompts show spectral structure                                  │
│                                                                         │
│  COMPRESSION/INTERFERENCE:                                              │
│  [ ] Same-meaning averaging reinforces                                │
│  [ ] Random averaging cancels                                         │
│  [ ] Optimized prompts are load-bearing                               │
│  [ ] MDL correlates with complexity                                   │
│                                                                         │
│  TICKLING:                                                              │
│  [ ] Leaders predicted from entropy                                   │
│  [ ] Cheap probes correlate with full results                        │
│  [ ] Edge detection provides speedup                                  │
│                                                                         │
│  CONSERVATION:                                                          │
│  [ ] Total information approximately conserved                        │
│  [ ] Phase transition signatures present                              │
│  [ ] Power law behavior near transition                               │
│                                                                         │
│  ═══════════════════════════════════════════════════════════════════   │
│                                                                         │
│  OVERALL STATUS: ___/17 predictions validated                         │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

*Oscar Goldman, Shogu Research Group @ Datamutant.ai*

*"Good theories make predictions. Predictions can be wrong. Being wrong is informative. Design experiments to falsify, not confirm. Survival of falsification attempts is stronger than confirmation."*

