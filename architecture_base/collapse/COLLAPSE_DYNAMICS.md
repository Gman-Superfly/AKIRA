# Collapse Dynamics: The Physics of Belief Resolution

## Theory-Mandated Collapse Mechanisms

**Oscar Goldman — Shogu Research Group @ Datamutant.ai**

---

## Table of Contents

1. [What Theory Demands](#1-what-theory-demands)
2. [Collapse as Phase Transition](#2-collapse-as-phase-transition)
3. [The BEC Framework](#3-the-bec-framework)
4. [Entropy-Based Detection](#4-entropy-based-detection)
5. [Temperature Control](#5-temperature-control)
6. [The Coherence Principle](#6-the-coherence-principle)
7. [POMDP Connection](#7-pomdp-connection)
8. [Implementation Requirements](#8-implementation-requirements)
9. [What This File Does NOT Cover](#9-what-this-file-does-not-cover)

---

## 1. What Theory Demands

### 1.0 The Core Dynamic: Tension and Collapse

Before detailing the theory-mandated properties, we must establish the fundamental dynamic:

```
THE FUNDAMENTAL CYCLE OF BELIEF EVOLUTION

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  Redundancy transforms into Synergy through TENSION.                   │
│  Synergy collapses back into Redundancy through COLLAPSE.              │
│                                                                         │
│  ════════════════════════════════════════════════════════════════════  │
│                                                                         │
│  THE KEY INSIGHT:                                                       │
│                                                                         │
│  • During TENSION: Uncertainty ACCUMULATES as the system moves from    │
│    redundancy toward synergy. New evidence arrives, hypotheses         │
│    multiply, information becomes distributed across bands.             │
│                                                                         │
│  • During COLLAPSE: Uncertainty RESOLVES as the system moves from      │
│    synergy to redundancy. Hypotheses merge, Action Quanta crystallize, │
│    information concentrates.                                           │
│                                                                         │
│  ════════════════════════════════════════════════════════════════════  │
│                                                                         │
│  THE PUMP CYCLE:                                                        │
│                                                                         │
│  [Redundancy] ──TENSION──> [Synergy] ──COLLAPSE──> [Redundancy] + AQ  │
│       ↑                                                    │           │
│       └────────────────────────────────────────────────────┘           │
│                          (cycle repeats)                               │
│                                                                         │
│  This document focuses on COLLAPSE (the synergy→redundancy direction). │
│  The reverse process (TENSION) is the gradual accumulation of          │
│  uncertainty that precedes and enables collapse.                       │
│                                                                         │
│  See foundations/TERMINOLOGY.md for formal definitions.               │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1.1 The Core Requirements

The theoretical framework (BEC, Pythagorean coherence, spectral dynamics) mandates specific properties for collapse:

```
THEORY-MANDATED COLLAPSE PROPERTIES:

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  1. SUDDEN, NOT GRADUAL                                                │
│     Collapse is a phase transition, not smooth optimization.          │
│     There is a critical point where the system snaps.                 │
│                                                                         │
│  2. INTERFERENCE-DRIVEN                                                 │
│     Coherent states reinforce (constructive interference).            │
│     Incoherent states cancel (destructive interference).              │
│     The winner SURVIVES, not "is chosen."                             │
│                                                                         │
│  3. ENTROPY-OBSERVABLE                                                  │
│     Collapse manifests as sudden entropy drop.                        │
│     Entropy is the observable signal of belief concentration.         │
│                                                                         │
│  4. TEMPERATURE-CONTROLLABLE                                           │
│     Parameter τ controls collapse sharpness.                          │
│     Low τ → sharp collapse, high τ → diffuse (no collapse).         │
│                                                                         │
│  5. HIERARCHICAL                                                        │
│     Low-frequency (WHAT) collapses before high-frequency (WHERE).    │
│     Cascade from coarse to fine.                                      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1.2 What This Means Physically

```
COLLAPSE IS NOT A DECISION — IT IS COHERENCE SELECTION

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  The system does not "choose" a winner.                                │
│  The winner is what survives interference.                            │
│                                                                         │
│  BEFORE COLLAPSE:                                                       │
│  • Multiple hypotheses coexist (superposition)                        │
│  • Each has a phase (interpretation/direction)                        │
│  • Phases are random/incoherent                                        │
│  • Net amplitude ~ √N (random walk)                                   │
│                                                                         │
│  DURING COLLAPSE:                                                       │
│  • Incoherent hypotheses cancel (destructive interference)            │
│  • Coherent hypotheses reinforce (constructive interference)          │
│  • One interpretation dominates                                        │
│                                                                         │
│  AFTER COLLAPSE:                                                        │
│  • Phases aligned                                                      │
│  • Net amplitude ~ N (coherent sum)                                   │
│  • Single dominant belief                                              │
│                                                                         │
│  This is the Pythagorean principle: coherence survives.              │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Collapse as Phase Transition

### 2.1 The Critical Point

```
PHASE TRANSITION STRUCTURE:

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  ORDER PARAMETER: φ = max attention weight (or 1 - entropy)           │
│                                                                         │
│  CONTROL PARAMETER: τ = temperature                                    │
│                                                                         │
│  CRITICAL POINT: τ_c where transition occurs                          │
│                                                                         │
│           │                                                            │
│     φ     │     ┌─────────── Collapsed (ordered)                      │
│           │     │                                                      │
│           │   ──┼──                                                    │
│           │     │                                                      │
│           │     └─────────── Diffuse (disordered)                     │
│           │                                                            │
│           └──────────────────────────────────────────────              │
│                             τ_c            τ                           │
│                                                                         │
│  Below τ_c: Spontaneous symmetry breaking, one winner                 │
│  Above τ_c: Symmetric, all hypotheses equivalent                      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Critical Exponents

From BEC theory, near the critical point:

```
SCALING RELATIONS:

φ ~ |τ - τ_c|^β    Order parameter
χ ~ |τ - τ_c|^γ    Susceptibility (response to perturbation)
ξ ~ |τ - τ_c|^ν    Correlation length

MEAN-FIELD BEC PREDICTIONS:
β = 0.5
γ = 1.0
ν = 0.5

These are TESTABLE. See experiments 004_EXP_PHASE_TRANSITION_SHARPNESS.
```

---

## 3. The BEC-Inspired Framework

**Note:** The connection to Bose-Einstein Condensation is an analogy based on structural similarities, not a claim of physical identity. See `AKIRA/bec/BEC_CONDENSATION_INFORMATION.md` for detailed discussion.

### 3.1 BEC-Inspired Collapse Framework

```
THE BEC ANALOGY:

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  BEC (Gross-Pitaevskii):                                               │
│                                                                         │
│    iℏ ∂ψ/∂t = [-ℏ²∇²/2m + V(r) + g|ψ|²] ψ                            │
│                                                                         │
│  The g|ψ|² term is SELF-INTERACTION:                                  │
│  • ψ = wave function (belief state)                                   │
│  • |ψ|² = probability density (attention weights)                    │
│  • g|ψ|² = density-dependent potential (attention-weighted update)   │
│                                                                         │
│  ═══════════════════════════════════════════════════════════════════  │
│                                                                         │
│  ATTENTION:                                                             │
│                                                                         │
│    A(X) · X = (self-similarity) × (input)                             │
│                                                                         │
│  This shares deep structural similarities with g|ψ|²:                 │
│  • X = input (analogous to ψ)                                         │
│  • A(X) = attention weights (analogous to |ψ|²)                       │
│  • A(X)·X = weighted combination (analogous to g|ψ|²ψ)                │
│                                                                         │
│  This suggests collapse dynamics may resemble Bose-Einstein            │
│  condensation, providing a useful framework for understanding          │
│  phase transitions in belief.                                          │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Why This Enables Collapse

```
THE NONLINEAR SELF-REFERENCE:

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  Without nonlinear self-interaction (linear system):                  │
│  • Superpositions remain superpositions                               │
│  • No spontaneous symmetry breaking                                   │
│  • No collapse possible                                                │
│                                                                         │
│  With nonlinear self-interaction (like g|ψ|² in BEC):                 │
│  • Concentrated states are favored (lower energy)                     │
│  • Spread states are unstable                                          │
│  • System spontaneously concentrates                                   │
│  • COLLAPSE EMERGES FROM DYNAMICS                                      │
│                                                                         │
│  ═══════════════════════════════════════════════════════════════════  │
│                                                                         │
│  Attention provides the mechanism that enables collapse.              │
│  Without attention (linear combination), no collapse.                 │
│  With attention (nonlinear self-weighting), collapse becomes          │
│  possible below critical temperature T_c.                             │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Entropy-Based Detection

### 4.1 Entropy as the Observable

```
ENTROPY IS THE SIGNAL:

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  ATTENTION ENTROPY:                                                     │
│                                                                         │
│  H = -Σ_j a_j log(a_j)                                                 │
│                                                                         │
│  where a_j = attention weight for position j                          │
│                                                                         │
│  ─────────────────────────────────────────────────────────────────────  │
│                                                                         │
│  HIGH ENTROPY (H → log N):                                              │
│  • Attention uniform across positions                                  │
│  • Maximum uncertainty                                                  │
│  • No dominant hypothesis                                               │
│  • System is in "diffuse" state                                        │
│                                                                         │
│  LOW ENTROPY (H → 0):                                                   │
│  • Attention concentrated on one position                             │
│  • Maximum certainty                                                    │
│  • One dominant hypothesis                                              │
│  • System is in "collapsed" state                                      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Collapse Detection

```
DETECTING COLLAPSE:

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  MONITOR: dH/dt (rate of entropy change)                               │
│                                                                         │
│  COLLAPSE SIGNATURE:                                                    │
│  • Sudden negative spike in dH/dt                                     │
│  • |dH/dt| exceeds threshold                                          │
│  • Duration is short (phase transition is fast)                       │
│                                                                         │
│  IMPLEMENTATION:                                                        │
│                                                                         │
│  def detect_collapse(entropy_history, threshold=0.3):                 │
│      dH_dt = entropy_history[-1] - entropy_history[-2]                │
│      is_collapsing = dH_dt < -threshold                               │
│      return is_collapsing, dH_dt                                      │
│                                                                         │
│  This is the MINIMUM required for collapse detection.                 │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Temperature Control

### 5.1 Temperature in Softmax

```
TEMPERATURE AS CONTROL PARAMETER:

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  SOFTMAX WITH TEMPERATURE:                                              │
│                                                                         │
│  a_j = exp(s_j / τ) / Σ_k exp(s_k / τ)                                │
│                                                                         │
│  where:                                                                 │
│  • s_j = pre-softmax score for position j                             │
│  • τ = temperature parameter                                          │
│                                                                         │
│  ─────────────────────────────────────────────────────────────────────  │
│                                                                         │
│  τ → 0:  argmax behavior (winner-take-all)                            │
│          One position gets all attention                               │
│          FORCED COLLAPSE regardless of evidence                        │
│                                                                         │
│  τ = 1:  Standard softmax                                              │
│          Natural collapse if scores are unequal                       │
│                                                                         │
│  τ → ∞:  Uniform distribution                                         │
│          All positions get equal attention                            │
│          NO COLLAPSE regardless of evidence                            │
│                                                                         │
│  ═══════════════════════════════════════════════════════════════════  │
│                                                                         │
│  Temperature is the control parameter for collapse.                   │
│  τ_c is where the phase transition occurs.                            │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 5.2 Adaptive Temperature

```
TEMPERATURE SCHEDULING:

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  OPTION 1: FIXED τ                                                      │
│  Simple, predictable, but not adaptive                                │
│                                                                         │
│  OPTION 2: ENTROPY-ADAPTIVE τ                                          │
│  τ decreases when entropy is stable (ready to collapse)              │
│  τ increases when entropy is volatile (not ready)                    │
│                                                                         │
│  OPTION 3: ANNEALING                                                    │
│  τ decreases over inference time                                      │
│  Early: explore (high τ), Late: commit (low τ)                       │
│                                                                         │
│  THEORY PREFERENCE:                                                     │
│  Let τ be learned or entropy-adaptive.                                │
│  Fixed τ is acceptable for experiments.                               │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 6. The Coherence Principle

### 6.1 Why Coherent States Win

```
THE PYTHAGOREAN PRINCIPLE IN COLLAPSE:

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  Consider N hypotheses with phases φ₁, φ₂, ..., φₙ                    │
│                                                                         │
│  Total amplitude: A = Σᵢ exp(iφᵢ)                                      │
│                                                                         │
│  INCOHERENT (random phases):                                            │
│  • Phases random → contributions cancel                               │
│  • |A|² ~ N (random walk)                                             │
│  • Net effect: weak                                                    │
│                                                                         │
│  COHERENT (aligned phases):                                             │
│  • Phases aligned → contributions add                                 │
│  • |A|² = N² (coherent sum)                                           │
│  • Net effect: N× stronger                                            │
│                                                                         │
│  ═══════════════════════════════════════════════════════════════════  │
│                                                                         │
│  COHERENT STATES DOMINATE BY FACTOR OF N.                             │
│                                                                         │
│  The winner is not "chosen" — it's what survives interference.       │
│  Incoherent alternatives cancel themselves out.                       │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 6.2 Hierarchical Coherence

```
COLLAPSE CASCADE:

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  Band 0 (DC) collapses FIRST:                                          │
│  • Low-frequency = coarse structure                                   │
│  • Determines WHAT                                                      │
│  • Most stable, slowest to change                                     │
│                                                                         │
│  Band 6 (High) collapses LAST:                                         │
│  • High-frequency = fine detail                                       │
│  • Determines WHERE (exactly)                                          │
│  • Most volatile, fastest to change                                   │
│                                                                         │
│  ─────────────────────────────────────────────────────────────────────  │
│                                                                         │
│  THE CASCADE:                                                           │
│                                                                         │
│  Time →                                                                │
│                                                                         │
│  Band 0: ███████████████████░░░░░░░░░░  (collapses early)             │
│  Band 1: ░░░░░███████████████████░░░░░                                 │
│  Band 2: ░░░░░░░░░░████████████████░░░                                 │
│  Band 3: ░░░░░░░░░░░░░░░██████████████                                 │
│  Band 4: ░░░░░░░░░░░░░░░░░░░░█████████  (collapses late)              │
│  Band 5: ░░░░░░░░░░░░░░░░░░░░░░░░█████                                 │
│  Band 6: ░░░░░░░░░░░░░░░░░░░░░░░░░░░██                                 │
│                                                                         │
│  Low-freq provides CONTEXT for high-freq collapse.                    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 7. POMDP Connection

### 7.1 Belief State Dynamics

```
COLLAPSE AS BELIEF CONCENTRATION:

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  POMDP BELIEF STATE:                                                    │
│                                                                         │
│  b(s) = P(state = s | history)                                        │
│                                                                         │
│  This is a PROBABILITY DISTRIBUTION over hidden states.               │
│                                                                         │
│  ─────────────────────────────────────────────────────────────────────  │
│                                                                         │
│  COLLAPSE = b(s) → δ(s - s*)                                          │
│                                                                         │
│  The belief distribution concentrates on one state s*.               │
│  Entropy H(b) → 0.                                                     │
│                                                                         │
│  ─────────────────────────────────────────────────────────────────────  │
│                                                                         │
│  IN AKIRA:                                                              │
│                                                                         │
│  • Hidden state = next frame (unobservable future)                    │
│  • Belief b(s) = attention distribution over hypotheses              │
│  • Collapse = attention concentrating on prediction                  │
│                                                                         │
│  This is EXACTLY what we observe.                                     │
│                                                                         │
│  NOTE: This is the operational/POMDP view. The formal definition is  │
│  Bayesian (posterior contraction). See foundations/TERMINOLOGY.md.   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 7.2 Optimal Stopping

```
WHEN TO COLLAPSE:

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  LISTEN vs ACT TRADEOFF:                                                │
│                                                                         │
│  • Listening (not collapsing) has cost: delay, uncertainty            │
│  • Acting (collapsing) has risk: wrong decision, commitment           │
│                                                                         │
│  OPTIMAL POLICY:                                                        │
│                                                                         │
│  Collapse when expected benefit of commitment exceeds                 │
│  expected benefit of further listening.                               │
│                                                                         │
│  ─────────────────────────────────────────────────────────────────────  │
│                                                                         │
│  SIMPLE HEURISTIC:                                                      │
│                                                                         │
│  Collapse when entropy is low enough that additional                  │
│  evidence is unlikely to change the winner.                           │
│                                                                         │
│  See: THE_OLD_LADY_AND_THE_TIGER.md for full treatment.              │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 8. Implementation Requirements

### 8.1 Minimum Viable Collapse

```
REQUIRED COMPONENTS:

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  1. ENTROPY COMPUTATION                                                 │
│     H = -Σ_j a_j log(a_j)                                             │
│     Computed from attention weights after softmax.                    │
│                                                                         │
│  2. TEMPERATURE PARAMETER                                               │
│     τ in softmax: a_j = softmax(s_j / τ)                              │
│     Controls collapse sharpness.                                       │
│                                                                         │
│  3. COLLAPSE DETECTOR                                                   │
│     Monitor dH/dt.                                                     │
│     Flag when |dH/dt| > threshold.                                    │
│                                                                         │
│  4. PER-BAND TRACKING                                                   │
│     Separate entropy for each spectral band.                          │
│     Enables hierarchical collapse observation.                        │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 8.2 Implementation Sketch

```python
class CollapseTracker:
    """Minimal theory-compliant collapse tracking."""
    
    def __init__(self, num_bands: int = 8, threshold: float = 0.3):
        self.num_bands = num_bands
        self.threshold = threshold
        self.entropy_history = {b: [] for b in range(num_bands)}
    
    def compute_entropy(self, attention_weights: torch.Tensor) -> float:
        """Compute entropy of attention distribution."""
        # Clamp to avoid log(0)
        a = attention_weights.clamp(min=1e-10)
        return -(a * a.log()).sum(dim=-1).mean().item()
    
    def update(self, band_idx: int, attention_weights: torch.Tensor) -> dict:
        """Update tracking for one band."""
        entropy = self.compute_entropy(attention_weights)
        self.entropy_history[band_idx].append(entropy)
        
        # Compute collapse detection
        if len(self.entropy_history[band_idx]) >= 2:
            dH_dt = entropy - self.entropy_history[band_idx][-2]
            is_collapsing = dH_dt < -self.threshold
        else:
            dH_dt = 0.0
            is_collapsing = False
        
        return {
            "band": band_idx,
            "entropy": entropy,
            "dH_dt": dH_dt,
            "is_collapsing": is_collapsing,
        }
    
    def get_global_state(self) -> dict:
        """Get overall collapse state."""
        band_entropies = {
            b: self.entropy_history[b][-1] 
            for b in range(self.num_bands) 
            if self.entropy_history[b]
        }
        return {
            "global_entropy": sum(band_entropies.values()) / len(band_entropies),
            "band_entropies": band_entropies,
        }
```

---

## 9. What This File Does NOT Cover

### 9.1 Explicitly Excluded (See architecture_expanded/)

```
NOT IN THIS FILE (implementation alternatives):

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  • Decision tree structures for collapse                              │
│    → See: architecture_expanded/collapse/COLLAPSE_MECHANISMS.md       │
│                                                                         │
│  • Branch generation and pruning logic                                │
│    → Implementation detail, not theory-mandated                       │
│                                                                         │
│  • Explicit hypothesis enumeration                                     │
│    → One approach, but interference is implicit                        │
│                                                                         │
│  • Specific collapse triggering heuristics                            │
│    → Many valid approaches exist                                       │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 9.2 Relationship to Other Base Documents

```
COMPLEMENTS THESE BASE DOCUMENTS:

• COLLAPSE_GENERALIZATION.md — The phenomenon (what collapse IS)
• This file (COLLAPSE_DYNAMICS.md) — The physics (how collapse WORKS)

REFERENCES:
• BEC_CONDENSATION_INFORMATION.md — g|ψ|² framework
• HARMONY_AND_COHERENCE.md — Pythagorean coherence principle
• THE_OLD_LADY_AND_THE_TIGER.md — Optimal stopping theory
• Shai, A. (2024). *Transformers Represent Belief State Geometry in their Residual Stream.* LessWrong. [Link](https://www.lesswrong.com/posts/gTZ2SxesbHckJ3CkF/transformers-represent-belief-state-geometry-in-their) — external evidence that transformer residual streams instantiate POMDP-style belief-state geometry, consistent with collapse-as-belief-concentration.
• Wentworth, J., & Lorell, D. (2024). *Why Would Belief-States Have A Fractal Structure, And Why Would That Matter For Interpretability? An Explainer.* LessWrong. [Link](https://www.lesswrong.com/posts/mBw7nc4ipdyeeEpWs/why-would-belief-states-have-a-fractal-structure-and-why) — explains why Bayesian belief-state dynamics for HMMs produce fractal geometry via chaos games, and why such self-similar belief manifolds matter for interpreting latent structure in trained networks.
• Williams, P.L., & Beer, R.D. (2010). *Nonnegative Decomposition of Multivariate Information.* arXiv:1004.2515. [PDF](https://arxiv.org/pdf/1004.2515) — foundational paper on Partial Information Decomposition (PID). Collapse can be understood as SYNERGY→REDUNDANCY conversion: pre-collapse states have high synergy (need all bands), post-collapse states have high redundancy (any band suffices). Total information is conserved but its decomposition changes. See ORTHOGONALITY.md §11.3 for full treatment.
```

---

## Summary

```
WHAT THEORY DEMANDS FOR COLLAPSE:

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  1. Phase transition dynamics (sudden, not gradual)                   │
│  2. Interference-driven selection (coherent survives)                 │
│  3. Entropy as observable (H drops at collapse)                       │
│  4. Temperature control (τ modulates sharpness)                       │
│  5. Hierarchical cascade (low-freq → high-freq)                       │
│  6. BEC-inspired framework (structural analogy, not identity)         │
│  7. POMDP interpretation (belief concentration)                       │
│                                                                         │
│  Everything else is implementation choice.                            │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

*Oscar Goldman — Shogu Research Group @ Datamutant.ai*

*"Collapse is not a decision. It is coherence selection. The winner survives interference — it is not chosen."*

