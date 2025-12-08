# Experiment 005: Conservation Laws - Supporting Literature

## Evidence for architecture experiment choices

This experiment asks whether there is a conserved quantity during belief dynamics — analogous to energy, normalization, or particle number in physical systems. The experiment proposes tracking "attention energy," "belief mass," and total entropy. However, these scalar measures miss a critical structure: **information can be conserved in total while changing form**. A system may conserve total information while converting *synergy* to *redundancy* during collapse. **Partial Information Decomposition (PID)** provides the mathematical framework to decompose conserved quantities into their constituent parts and track how they transform.

---

## Primary References

**Williams, P.L., & Beer, R.D. (2010).** *Nonnegative Decomposition of Multivariate Information.* arXiv:1004.2515.  
📄 [arXiv PDF](https://arxiv.org/pdf/1004.2515)

### Why This Paper Supports EXP_005

**Core Problem with Standard Conservation Measures:**

EXP_005 proposes testing whether quantities like total entropy H_total = Σᵢ H(aᵢ) are conserved during inference. But this misses a fundamental distinction:

```
SCENARIO A: Total information = 10 bits
  - Redundancy between bands: 8 bits
  - Synergy between bands: 2 bits
  → System is highly redundant (bands duplicate information)

SCENARIO B: Total information = 10 bits  
  - Redundancy between bands: 2 bits
  - Synergy between bands: 8 bits
  → System is highly synergistic (bands provide complementary info)

Both have the same total. Both would pass a "conservation test."
But they represent fundamentally different computational states.
```

**PID decomposes information into nonnegative atoms:**

| Atom | Symbol | Meaning in AKIRA |
|------|--------|------------------|
| **Redundancy** | R(S; A₁, A₂, ...) | Information that ANY band could provide alone about target S |
| **Unique** | U(S; Aᵢ) | Information ONLY band i provides about S |
| **Synergy** | Syn(S; A₁, A₂, ...) | Information that EMERGES only when bands combine |

**The key insight for EXP_005:**

Total mutual information I(S; A₁, A₂, ..., Aₙ) decomposes as:

```
I(S; All Bands) = Redundancy + Σ Unique_i + Synergy + (higher-order terms)
```

**This sum is conserved by definition** — it's the total information. What changes during dynamics is the *distribution* across atoms.

### The Conservation-Transformation Hypothesis

**Proposed addition to EXP_005 hypotheses:**

**H5: Collapse transforms synergy into redundancy while conserving total information.**

| Phase | Redundancy | Synergy | Interpretation |
|-------|------------|---------|----------------|
| Pre-collapse (uncertainty) | Low | High | Bands hold different hypotheses; need all bands to predict |
| During collapse | Increasing | Decreasing | Winner emerges; bands begin agreeing |
| Post-collapse (certainty) | High | Low | All bands encode the same committed belief |

**Why this makes physical sense:**

1. **Before collapse:** Each band maintains different plausible futures. Band 0 might favor "blob moving left," Band 6 might favor "blob moving right." Neither band alone can predict — you need to combine them (high synergy). The information is distributed across bands in complementary form.

2. **During collapse:** Evidence accumulates. One hypothesis gains support. Bands begin synchronizing their beliefs. The information that was spread synergistically starts concentrating.

3. **After collapse:** All bands agree on the winner. Any single band can now predict the target nearly as well as all bands combined (high redundancy). The system has committed.

**This is not a violation of conservation — it's a phase transition in information structure.**

### Detailed Implementation for EXP_005

**Augment the existing protocol with PID tracking:**

```python
from dit.pid import PID_BROJA  # or Williams-Beer I_min

def compute_pid_decomposition(
    target: torch.Tensor,           # Ground truth or prediction target
    band_representations: Dict[int, torch.Tensor]  # Per-band activations
) -> Dict[str, float]:
    """
    Decompose information about target across spectral bands.
    
    Returns:
        Dictionary with:
        - 'total_mi': I(Target; All Bands)
        - 'redundancy': Information any band could provide
        - 'synergy': Information only available when bands combine
        - 'unique_per_band': Dict[band_idx, unique_info]
    """
    # Discretize for PID computation (or use continuous estimators)
    # ... implementation details ...
    
    pid = PID_BROJA(target, list(band_representations.values()))
    
    return {
        'total_mi': pid.total_mi(),
        'redundancy': pid.redundancy(),
        'synergy': pid.synergy(),
        'unique_per_band': {i: pid.unique(i) for i in range(len(band_representations))},
    }


def track_pid_conservation(
    pid_history: List[Dict[str, float]]
) -> Dict[str, float]:
    """
    Verify that total MI is conserved while tracking atom transformation.
    """
    total_mis = [p['total_mi'] for p in pid_history]
    redundancies = [p['redundancy'] for p in pid_history]
    synergies = [p['synergy'] for p in pid_history]
    
    return {
        'total_mi_mean': np.mean(total_mis),
        'total_mi_std': np.std(total_mis),
        'total_mi_cv': np.std(total_mis) / np.mean(total_mis),  # Should be < 1%
        
        'redundancy_trend': np.polyfit(range(len(redundancies)), redundancies, 1)[0],
        'synergy_trend': np.polyfit(range(len(synergies)), synergies, 1)[0],
        
        # Key ratio: does synergy convert to redundancy?
        'syn_to_red_ratio_start': synergies[0] / (redundancies[0] + 1e-10),
        'syn_to_red_ratio_end': synergies[-1] / (redundancies[-1] + 1e-10),
    }
```

**New metrics for Phase C (Cross-Band Conservation):**

The existing protocol tests whether "total attention mass is conserved." PID refines this:

```
PHASE C (Original): Track Σ_bands M_b(t) — is total mass constant?

PHASE C (PID-augmented):
  C1. Track Total_MI(t) — should be constant (true conservation)
  C2. Track Redundancy(t) — should INCREASE during collapse
  C3. Track Synergy(t) — should DECREASE during collapse
  C4. Verify: ΔRedundancy ≈ -ΔSynergy (synergy converts, not destroyed)
```

### Predictions Refined by PID

| Original Hypothesis | PID Refinement | Specific Prediction |
|---------------------|----------------|---------------------|
| H2: Energy-like quantity conserved | Total_MI is the conserved quantity | Std(Total_MI) / Mean(Total_MI) < 1% |
| H3: Total belief mass conserved across bands | Mass conservation is necessary but not sufficient | Total_MI constant, but R/S ratio changes |
| (New) H5: Synergy→Redundancy conversion | Collapse is a phase transition in information structure | d(Redundancy)/dt > 0 during collapse, d(Synergy)/dt < 0 |

**Falsification criteria (PID-specific):**

| Observation | Interpretation |
|-------------|----------------|
| Total_MI varies > 5% | Conservation fails — information leaks or is injected |
| Redundancy and Synergy both increase | System is gaining information (external driving) |
| Redundancy and Synergy both decrease | System is losing information (dissipation) |
| Synergy→Redundancy conversion NOT observed | Collapse is not an information phase transition — alternative mechanism |

### Connection to Noether's Theorem (Section 3.1)

The experiment invokes Noether's theorem: every continuous symmetry corresponds to a conserved quantity. PID provides a refinement:

**If the system has a symmetry that conserves total information**, then:
- Total_MI is the Noether charge
- The decomposition into R, U, S represents different "species" of information
- Dynamics can convert between species while conserving the total

**Analogy to physics:**
- In particle physics, baryon number is conserved, but protons can convert to neutrons
- In AKIRA, Total_MI is conserved, but Synergy can convert to Redundancy

This is precisely analogous to **phase transitions in statistical mechanics**: total energy is conserved, but the system transitions between ordered and disordered states. Here, the order parameter is the Redundancy/Synergy ratio.

### Practical Considerations

**Computational cost:**

PID computation is exponential in the number of sources. With 7 bands, full PID is expensive. Practical approaches:

1. **Pairwise PID:** Compute PID for each band pair (i,j) with target. O(n²) instead of O(2ⁿ).

2. **Coarse-grained PID:** Group bands into Low (0-2), Mid (3), High (4-6), compute 3-source PID.

3. **Temporal subsampling:** Compute full PID at key timepoints (pre-collapse, collapse, post-collapse) rather than every timestep.

**Estimator choice:**

- **Williams-Beer I_min:** Original proposal, well-understood but can underestimate synergy
- **BROJA:** Optimization-based, handles continuous variables better
- **Griffith-Koch:** Alternative synergy measure, may be more sensitive

Recommend starting with I_min for interpretability, cross-validate with BROJA.

### Theoretical Connections

**To other AKIRA theory:**

1. **EXP 020 (Cross-Band Flow):** Conservation here explains WHERE information goes. EXP 020 explains HOW it flows. They're complementary — run together.

2. **Collapse Dynamics:** The synergy→redundancy conversion IS the collapse, viewed information-theoretically. High synergy = superposition (need all hypotheses). High redundancy = collapsed (any hypothesis suffices).

3. **Parseval's Theorem (H4):** Parseval guarantees energy conservation in spectral decomposition. PID adds: the *type* of information (R vs S) can change even as total energy is conserved.

4. **BEC Analogy:** In BEC, particles condense into ground state (redundant occupation of single mode). In AKIRA, information condenses from synergistic (distributed across bands) to redundant (concentrated in agreement).

---

## Additional References

**Griffith, V., & Koch, C. (2014).** *Quantifying Synergistic Mutual Information.* arXiv:1205.4265.  
Alternative synergy measure that may be more sensitive to the synergy→redundancy transition. Consider as secondary validation.

**Bertschinger, N. et al. (2014).** *Quantifying Unique Information.* Entropy, 16(4).  
Formal treatment of unique information in PID. Relevant if EXP_005 wants to track which bands carry unique information that others lack.

**Rosas, F. et al. (2020).** *Quantifying High-Order Interdependencies via Multivariate Extensions of the Mutual Information.* Physical Review E.  
Modern treatment including O-information (balance of redundancy and synergy). The sign of O-information indicates whether system is redundancy-dominated or synergy-dominated — could be a single summary statistic.

**Lizier, J. et al. (2012).** *Local Information Dynamics.* Information Sciences.  
Pointwise (local) information measures. Could identify WHEN in a sequence the synergy→redundancy conversion occurs, not just that it occurs on average.

---

## Summary: What PID Adds to EXP_005

| Without PID | With PID |
|-------------|----------|
| "Is total entropy conserved?" | "Is total information conserved, and in what form?" |
| Conservation = constant scalar | Conservation = constant total with changing composition |
| Collapse = entropy drop | Collapse = synergy→redundancy phase transition |
| Violation = bug or dissipation | Violation = information injection/leakage, diagnose by atom |

**Bottom line:** EXP_005 as written tests whether a scalar is constant. PID reveals that the more interesting question is whether information *transforms* during collapse while being conserved in total. This is the information-theoretic signature of a phase transition.

---

## Citation

Williams, P.L., & Beer, R.D. (2010). Nonnegative Decomposition of Multivariate Information. *arXiv preprint arXiv:1004.2515.*

---

*Oscar Goldman — Shogu Research Group @ Datamutant.ai*

