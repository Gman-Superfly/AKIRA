# Theory-Experiment Mapping

## Bidirectional Links Between Hypotheses and Experimental Validation

**Date:** December 2025  
**Status:** ACTIVE — Update as experiments complete  
**Purpose:** Ensure every hypothesis has experimental validation, trace reasoning from foundations to predictions

---

## Epistemological Framework

```
DERIVATION CHAIN: Foundations → Bridges → Hypotheses → Experiments

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  ESTABLISHED          THEORETICAL        TESTABLE          EXPERIMENTS │
│  FOUNDATIONS    →     BRIDGES      →     HYPOTHESES   →    TO RUN      │
│  ─────────────        ──────────         ──────────        ──────────  │
│  (well-tested)        (derivations)      (predictions)     (validation)│
│                                                                         │
│  Examples:                                                              │
│                                                                         │
│  Fourier/Parseval → Spectral decomp → Bands differ    → Exp 003       │
│  Shannon entropy  → Belief = dist.  → Entropy varies  → Exp 001       │
│  BEC mathematics  → Structural sim. → Phase transition→ Exp 004       │
│  PID (W&B 2010)   → Cross-band info → Syn→Red collapse→ Exp 020       │
│                                                                         │
│  Each hypothesis should trace back to foundations and forward to tests.│
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### The Core Dynamic All Experiments Test

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
│  Every experiment relates to some aspect of this cycle.               │
│  See foundations/TERMINOLOGY.md for formal definitions.               │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## How to Use This Document

**For theorists:** Find which experiments test your hypotheses  
**For experimentalists:** Find which hypothesis your experiment tests  
**For reviewers:** Check derivation chain from foundations to predictions

---

## Core Architecture Hypotheses → Experiments

### Hypothesis 1: Attention Entropy is Observable

**Foundation:** Shannon entropy (well-established information theory)  
**Bridge:** Attention weights sum to 1, forming a probability distribution  
**Theory:** `architecture_theoretical/SPECTRAL_BELIEF_MACHINE.md` §6  
**Hypothesis:** Attention weights encode belief as probability distribution with measurable entropy

**Tested By:**
- **Experiment 001** (Entropy Observation) — Can we compute entropy?
- **Experiment 003** (Band Dynamics) — Does entropy differ across bands?

**Falsification:** If entropy is constant or uncorrelated with prediction

---

### Hypothesis 2: Collapse is Sharp (Phase Transition Analogy)

**Foundation:** BEC phase transitions (well-established in physics)  
**Bridge:** Attention has same structural form as g|ψ|² self-interaction  
**Theory:** `bec/BEC_CONDENSATION_INFORMATION.md` §4  
**Hypothesis:** IF the BEC analogy holds, belief collapse should behave like a sudden phase transition, not gradual

**Tested By:**
- **Experiment 002** (Collapse Detection) — Are drops sudden?
- **Experiment 004** (Phase Transition Sharpness) — Critical exponents?

**Falsification:** If entropy decreases linearly (no sharp transition)

---

### Hypothesis 3: Spectral Bands Have Different Dynamics

**Theory:** `architecture_theoretical/SPECTRAL_BELIEF_MACHINE.md` §2 (Differential Timescales)  
**Hypothesis:** Low-freq bands are stable, high-freq bands adapt quickly

**Tested By:**
- **Experiment 003** (Band Dynamics) — Do bands differ in entropy/collapse?
- **Experiment 013** (Differential LR) — Does LR hierarchy help?

**Falsification:** If all bands behave identically

---

### Hypothesis 4: Wormholes Connect Complementary Pairs

**Theory:** `architecture_theoretical/ORTHOGONALITY.md` §8 (Wormhole Complementarity)  
**Hypothesis:** Pairs (0↔6, 1↔5, 2↔4) activate more than non-complementary

**Tested By:**
- **Experiment 012** (Wormhole Activation) — Activation patterns?
- **Experiment 024** (Resonant Wormholes) — Are pairs special?

**Falsification:** If non-complementary pairs activate equally

---

### Hypothesis 5: Conservation Laws Hold

**Foundation:** Parseval's theorem (established mathematics, MUST hold for correct FFT)  
**Bridge:** If implementation is correct, energy is conserved by mathematical necessity  
**Theory:** `architecture_theoretical/ORTHOGONALITY.md` §4.3 (Parseval's Theorem)  
**Hypothesis:** Energy conserved in spectral decomposition (this is verification, not hypothesis)

**Tested By:**
- **Experiment 005** (Conservation Laws) — Parseval holds?
- **Experiment 006** (Heresy Detection) — Violations = artifacts?

**Falsification:** If energy is not conserved → implementation bug, not theory failure

---

### Hypothesis 6: Differential Learning Rates Improve Learning

**Theory:** `architecture_theoretical/SPECTRAL_BELIEF_MACHINE.md` §2  
**Parameters:** `CANONICAL_PARAMETERS.md` (Specification A)  
**Hypothesis:** 3000× ratio between Band 0 and Band 6 improves convergence

**Tested By:**
- **Experiment 013** (Differential LR Validation)

**Falsification:** If uniform LR performs equally well

---

### Hypothesis 7: Action Quanta Emerge (Quasiparticle Analogy)

**Theory:** `bec/BEC_CONDENSATION_INFORMATION.md` §7  
**Hypothesis:** Stable, composite patterns (analogous to quasiparticles) emerge

**Tested By:**
- **Experiment 008** (Quasiparticle Dispersion) — Do atoms have dispersion?
- **Experiment 017** (MDL Atomic Truth) — Minimum description length?

**Falsification:** If no stable composite patterns exist

---

### Hypothesis 8: Temporal Band is Orthogonal to Spectral

**Theory:** `architecture_theoretical/PHASE_AND_TIME.md`  
**Hypothesis:** Time and frequency are orthogonal (Heisenberg)

**Tested By:**
- **Experiment 003** (Band Dynamics) — Band 7 behavior distinct?
- **Experiment 023** (Timeline Coherence) — Temporal vs spectral entropy?

**Falsification:** If Band 7 behaves like another spectral band

---

### Hypothesis 9: Error Propagates Like Wavefronts

**Theory:** `foundations/HARMONY_AND_COHERENCE.md`  
**Hypothesis:** Error spreads via manifold geometry, analogous to lightning

**Tested By:**
- **Experiment 007** (Wavefront Interference) — Visual propagation patterns?
- **Experiment 019** (Belief Geometry) — Manifold distance predicts spread?

**Falsification:** If error spreads randomly, not along manifold

---

### Hypothesis 10: Grokking Resembles Condensation

**Theory:** `bec/BEC_CONDENSATION_INFORMATION.md` §8  
**Hypothesis:** Sudden generalization shows signatures similar to belief condensation

**Tested By:**
- **Experiment 009** (Grokking as Condensation) — Entropy drop at grokking?

**Falsification:** If grokking shows no entropy signature

---

### Hypothesis 11: Collapse ≈ Synergy→Redundancy Conversion (PID Framework)

**Foundation:** Partial Information Decomposition (Williams & Beer 2010, established framework)  
**Bridge:** PID provides nonnegative decomposition of multivariate information into redundancy, unique, synergy  
**Theory:** `pomdp/REFERENCES.md` (Williams & Beer 2010), `DESIGN_DECISIONS.md` §5  
**Hypothesis:** Before collapse, bands hold synergistic information (need all to predict). After collapse, bands hold redundant information (any can predict). Total information conserved.

**Tested By:**
- **Experiment 005** (Conservation Laws) — Is total information conserved?
- **Experiment 020** (Cross-Band Flow) — Does synergy decrease and redundancy increase during collapse?

**Falsification:** If synergy and redundancy don't show inverse relationship during collapse

---

### Hypothesis 12: Duality Methods Enable Cheap Observability

**Foundation:** Fourier duality, convex duality (established mathematics)  
**Bridge:** Mathematical dualities swap what's easy/hard to compute; transforms are cheap (FFT is O(N log N))  
**Theory:** `observability/DUALITY_METHODS.md`, `foundations/DUALITY_AND_EFFICIENCY.md`  
**Hypothesis:** What is hard to observe in one domain becomes easy in the dual domain. The transform is cheap, so using both views costs little.

**Tested By:**
- **Experiment 006** (Heresy Detection) — Can we detect artifacts via dual view?
- **Experiment 010** (Tickling) — Does temperature sweep reveal uncertainty cheaply?

**Falsification:** If dual view doesn't reveal additional information, or if transform cost is prohibitive

---

### Hypothesis 13: Collapse IS Synergy→Redundancy Conversion (PID Mechanism)

**Foundation:** Partial Information Decomposition (Williams & Beer 2010), Integrated Information Decomposition/ΦID (Mediano et al. 2021), Partial Information Rate Decomposition (Sparacino et al. 2025)  
**Bridge:** PID decomposes information into synergistic (need all sources) and redundant (any source suffices). If collapse resolves uncertainty, the information structure should change from synergistic to redundant.  
**Theory:** `experiments/025_EXP_SYNERGY_REDUNDANCY_TRANSITION.md`  
**Hypothesis:** Collapse is not merely correlated with synergy→redundancy conversion; it IS that conversion. Synergy drop CAUSES collapse (Granger causality). Total information is conserved.

**Tested By:**
- **Experiment 025** (Synergy-Redundancy Transition) — Does synergy Granger-cause collapse? Is total information conserved?
- **Experiment 005** (Conservation Laws) — Is I_total = I_syn + I_red + I_uni constant?
- **Experiment 020** (Cross-Band Flow) — Do complementary pairs show highest conversion?

**Falsification:** If synergy does not drop at collapse, or redundancy does not increase, or total information changes dramatically, or Granger test fails

---

## Experiment → Theory Mapping

### Foundation Experiments (★★★ CRUCIAL)

| Experiment | Theory Documents | Key Parameters |
|------------|------------------|----------------|
| **001** Entropy Observation | `SPECTRAL_BELIEF_MACHINE.md` §6 | entropy formula |
| **002** Collapse Detection | `BEC_CONDENSATION_INFORMATION.md` §4 | collapse_threshold = 0.3 |
| **003** Band Dynamics | `SPECTRAL_BELIEF_MACHINE.md` §2, `ORTHOGONALITY.md` | per-band LR from `CANONICAL_PARAMETERS.md` |

---

### Core Experiments (★★ CORE)

| Experiment | Theory Documents | Key Parameters |
|------------|------------------|----------------|
| **004** Phase Transition | `BEC_CONDENSATION_INFORMATION.md` §4 | critical exponents |
| **005** Conservation Laws | `ORTHOGONALITY.md` §4.3 | Parseval's theorem |
| **006** Heresy Detection | `SPECTRAL_BELIEF_MACHINE.md` §7 | spectral leakage tolerance |
| **008** Quasiparticle Dispersion | `BEC_CONDENSATION_INFORMATION.md` §7 | dispersion relation |
| **009** Grokking | `BEC_CONDENSATION_INFORMATION.md` §8 | entropy collapse signature |

---

### Supporting Experiments (★ SUPPORTING)

| Experiment | Theory Documents | Key Parameters |
|------------|------------------|----------------|
| **007** Wavefront | `HARMONY_AND_COHERENCE.md` | manifold geometry |
| **010** Tickling | `observability/FREE_INFORMATION_ASSETS.md` | free information |
| **011** Prompt Spectral | `SPECTRAL_BELIEF_MACHINE.md` §1 | band decomposition |
| **012** Wormhole Activation | `ORTHOGONALITY.md` §8 | coherence_threshold = 0.5, top_k = 16 |
| **013** Differential LR | `CANONICAL_PARAMETERS.md`, `SPECTRAL_BELIEF_MACHINE.md` §2 | LR Spec A |
| **014** Critical Velocity | `BEC_CONDENSATION_INFORMATION.md` §9 | superfluidity analogy |
| **015** Attention Vortices | `BEC_CONDENSATION_INFORMATION.md` §10 | vortex detection |
| **018** Pump Cycle | `OLD_LADY_PARABLE.md` | belief oscillation |
| **019** Belief Geometry | `HARMONY_AND_COHERENCE.md` | manifold structure |
| **020** Cross-Band Flow | `ORTHOGONALITY.md` §8 | wormhole information flow |

---

### Exploratory Experiments (○ EXPLORATORY)

| Experiment | Theory Documents | Key Parameters |
|------------|------------------|----------------|
| **016** Cross-Model Manifold | General architecture | manifold universality |
| **017** MDL Atomic Truth | `OLD_LADY_PARABLE.md` | compression |
| **021** Attention Comma | Perceptual theory | punctuation in belief |
| **022** Band Phase Locking | `ORTHOGONALITY.md` §5 | phase coherence |
| **023** Timeline Coherence | `PHASE_AND_TIME.md` | temporal vs spectral |
| **024** Resonant Wormholes | `ORTHOGONALITY.md` §8 | complementary pairs |

---

## Parameter Lineage

### Learning Rates
- **Source:** `CANONICAL_PARAMETERS.md` (Specification A)
- **Theory:** `SPECTRAL_BELIEF_MACHINE.md` §2 (Differential Timescales)
- **Experiments:** 003, 013

### Wormhole Parameters
- **Source:** `CANONICAL_PARAMETERS.md`
- **Theory:** `architecture_base/attention/spectral_wormhole/SPECTRAL_WORMHOLE_ATTENTION.md`
- **Parameters:**
  - `coherence_threshold = 0.5` (normalized entropy gate)
  - `top_k = 16` (sparsity)
  - `gate_sharpness = 10.0` (sigmoid steepness)
- **Experiments:** 012, 020, 024

### Collapse Parameters
- **Source:** `CANONICAL_PARAMETERS.md`
- **Theory:** `BEC_CONDENSATION_INFORMATION.md` §4
- **Parameters:**
  - `collapse_threshold = 0.3` (dH/dt detection)
  - `temperature_init = 1.0` (belief temperature)
- **Experiments:** 002, 004

### Temperature Values (Per-Band)
- **Source:** `CANONICAL_PARAMETERS.md`
- **Theory:** `SPECTRAL_BELIEF_MACHINE.md` §6
- **Values:** `[0.5, 0.6, 0.7, 1.0, 0.8, 0.9, 1.0, 0.8]`
- **Experiments:** 002, 004, 022

---

## Dependency Graph

```
Theory Hierarchy:
SPECTRAL_BELIEF_MACHINE.md (root architecture)
├─→ ORTHOGONALITY.md (mathematical foundation)
├─→ THE_SEVEN_PLUS_ONE_ARCHITECTURE.md (band count justification)
├─→ PHASE_AND_TIME.md (temporal orthogonality)
├─→ BEC_CONDENSATION_INFORMATION.md (physics analogy)
└─→ CANONICAL_PARAMETERS.md (hyperparameters)

Experiment Dependency:
001 (Entropy)
└─→ 002 (Collapse)
    └─→ 003 (Band Dynamics)
        ├─→ 004 (Phase Transition)
        │   ├─→ 008 (Quasiparticles)
        │   └─→ 009 (Grokking)
        ├─→ 005 (Conservation)
        ├─→ 006 (Heresy)
        ├─→ 007 (Wavefront)
        │   ├─→ 012 (Wormholes) → 020
        │   ├─→ 016 (Cross-Model)
        │   └─→ 019 (Geometry)
        └─→ 013 (Differential LR)

017 (MDL) depends on ALL above
```

---

## Status Tracking

| Hypothesis | Experiments | Status | Last Updated |
|------------|-------------|--------|--------------|
| Entropy observable | 001, 003 | PENDING | — |
| Collapse is sharp | 002, 004 | PENDING | — |
| Bands differ | 003, 013 | PENDING | — |
| Wormholes selective | 012, 024 | PENDING | — |
| Conservation holds | 005, 006 | PENDING | — |
| Differential LR helps | 013 | PENDING | — |
| Info atoms emerge | 008, 017 | PENDING | — |
| Temporal orthogonal | 003, 023 | PENDING | — |
| Wavefront propagation | 007, 019 | PENDING | — |
| Grokking ≈ condensation | 009 | PENDING | — |
| Collapse ≈ synergy→redundancy (PID) | 005, 020 | PENDING | — |
| Duality methods work | 006, 010 | PENDING | — |
| Collapse IS syn→red (causal) | 025, 005, 020 | PENDING | — |

---

## How to Update

When an experiment completes:

1. **Update status** in table above
2. **Add results** to theory document
   ```markdown
   ### Experimental Evidence
   **Experiment 002 (Collapse Detection):** [PASSED/FAILED]
   - Result: [Summary]
   - Implication: [What this means for theory]
   ```
3. **Update experiment doc** with conclusion
4. **Update `000_EXPERIMENT_INDEX.md`** master status

---

## References

- `experiments/000_EXPERIMENT_INDEX.md` — Master experiment list
- `CANONICAL_PARAMETERS.md` — Parameter specifications
- `EXPERIMENT_THEORY_CROSSREF_CHECK.md` — Consistency audit
- `pomdp/REFERENCES.md` — Williams & Beer (2010) PID, external literature
- `observability/DUALITY_METHODS.md` — Seven duality methods for observability
- `foundations/DUALITY_AND_EFFICIENCY.md` — Transform-conserved-inversion pattern

---

*"Every hypothesis needs a test. Every test either supports or refutes a hypothesis. The links must be explicit. Until experiments run, these remain working hypotheses."*

