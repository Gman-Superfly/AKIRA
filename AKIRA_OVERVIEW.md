# AKIRA: Adaptive Knowledge Integration via Resonant Attention

## A Comprehensive Overview of Spectral Belief Dynamics

**Oscar Goldman — Shogu Research Group @ Datamutant.ai**

---

> *"In myth, Pandora opens the box and releases its contents. The act of opening transforms potential into actual. What was hidden becomes manifest. What was unified becomes differentiated.*
>
> *And when the box is shut, hope remains—not as consolation, but as the generator that is never consumed by generating. The capacity to produce more. The pattern that survives all its instances.*
>
> *AKIRA is an attempt to understand this transformation mathematically: how do many hypotheses become one belief? How does uncertainty crystallize into action? What is the structure of collapse?*
>
> *We don't claim to have answers. We have a framework for asking the questions."*

---

## Preamble: What This Document Is

This is a technical overview of the AKIRA framework. It contains:

- **Established foundations** (Fourier analysis, information theory, attention mathematics)
- **Architectural choices** (7+1 bands, three attention types, differential learning rates)
- **Testable hypotheses** (physics analogies, Action Quanta emergence, collapse dynamics)
- **Open questions** (what's proven vs. conjectured vs. speculative)

The document is comprehensive but not light reading. For a gentler introduction, see the main `README.md`. For the philosophical foundations, see `pandora/PANDORA.md`. For formal terminology, see `foundations/TERMINOLOGY.md`.

What follows is the full technical picture—with all the caveats, all the uncertainty, and all the structure we've built so far.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [The POMDP Foundation](#2-the-pomdp-foundation)
3. [Spectral Decomposition Architecture](#3-spectral-decomposition-architecture)
4. [The Three Attention Mechanisms](#4-the-three-attention-mechanisms)
5. [Belief State Dynamics](#5-belief-state-dynamics)
6. [The BEC Analogy: Attention and Self-Interaction (Speculative)](#6-the-bec-analogy-attention-and-self-interaction-speculative)
7. [Knowledge vs. Reactivity](#7-knowledge-vs-reactivity)
8. [Action Quanta as Quasiparticles](#8-action-quanta-as-quasiparticles)
9. [Collapse and Generalization](#9-collapse-and-generalization)
10. [Harmony and Coherence: The Pythagorean Principle](#10-harmony-and-coherence-the-pythagorean-principle)
11. [The Old Lady Parable](#11-the-old-lady-parable)
12. [Praxis: Doctrine and Heresy](#12-praxis-doctrine-and-heresy)
13. [Implementation Architecture](#13-implementation-architecture)
14. [Key Theoretical Results](#14-key-theoretical-results)
15. [Duality Methods for Observability](#15-duality-methods-for-observability)
16. [Document Index](#16-document-index)

---

## 1. Executive Summary

### 1.0 Epistemological Note

```
THE STATUS OF KNOWLEDGE IN THIS REPOSITORY

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  ESTABLISHED FOUNDATIONS (well-tested, not in question):               │
│  ───────────────────────────────────────────────────────                │
│  • Fourier analysis, FFT, spectral decomposition                       │
│  • Parseval's theorem (energy conservation across transforms)          │
│  • Shannon information theory, entropy measures                        │
│  • Heisenberg uncertainty (time-frequency tradeoff)                    │
│  • Attention mechanism mathematics (softmax, dot-product)              │
│  • Gradient descent, backpropagation                                   │
│  • BEC physics (Gross-Pitaevskii equation, in physical systems)       │
│                                                                         │
│  ARCHITECTURAL CHOICES (design decisions, empirically guided):         │
│  ───────────────────────────────────────────────────────────            │
│  • 7+1 band structure (motivated by theory, tunable)                   │
│  • Differential learning rates (motivated by timescale argument)       │
│  • Wormhole attention (design choice for cross-band communication)    │
│                                                                         │
│  HYPOTHESES UNDER TEST (proposed, awaiting experimental validation):  │
│  ───────────────────────────────────────────────────────────            │
│  • BEC analogy (attention ≈ g|ψ|² in structure and phenomenology)    │
│  • Collapse as phase transition                                        │
│  • PID framework for cross-band information (synergy/redundancy)      │
│  • Action Quanta emergence                                             │
│                                                                         │
│  ════════════════════════════════════════════════════════════════════  │
│                                                                         │
│  PURPOSE OF THIS REPOSITORY:                                           │
│                                                                         │
│  We derive everything from first principles to form a LOGICAL BRIDGE  │
│  between:                                                              │
│    • Established mathematical/physical foundations                    │
│    • Theoretical concepts proposed by observations in neural systems  │
│    • Experiments which guide architecture construction                │
│                                                                         │
│  The goal is TRACEABILITY: every hypothesis should connect back to   │
│  established foundations through explicit reasoning, and forward to   │
│  testable predictions through defined experiments.                    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1.1 What Is AKIRA?

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│     A K I R A                                                          │
│     ─────────                                                          │
│     Adaptive    → Learns from experience, not static                   │
│     Knowledge   → Stores structure, not raw data                       │
│     Integration → Combines multiple information streams                │
│     via                                                                │
│     Resonant    → Constructive/destructive interference               │
│     Attention   → Belief-weighted information routing                  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1.2 The Core Dynamic: Tension and Collapse

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
│  This dynamic drives the evolution of belief states in AKIRA.         │
│  See foundations/TERMINOLOGY.md for formal definitions.               │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

AKIRA is a **spectral attention architecture** for spatio-temporal prediction that frames neural attention as a **Partially Observable Markov Decision Process (POMDP)**. The system:

1. **Decomposes inputs spectrally** into 7 frequency bands (DC + 6 octaves)
2. **Maintains belief states** as probability distributions over possible futures
3. **Updates beliefs** through three parallel attention mechanisms
4. **Collapses uncertainty** to committed predictions when confidence is sufficient

### 1.2 The Core Insight

```
THE WORKING HYPOTHESES (testable, falsifiable):

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  HYPOTHESIS 1: PREDICTION IS BELIEF                                    │
│  ───────────────────────────────────                                    │
│  The model CANNOT observe the future directly.                         │
│  It observes past frames and must INFER what comes next.              │
│                                                                         │
│  The prediction IS a belief state:                                     │
│    ŷ_t = E[x_{t+1} | x_{t-T:t}]                                       │
│                                                                         │
│  The error map IS the belief visualized:                              │
│    High error = high uncertainty = belief spread across possibilities │
│                                                                         │
│  Training IS belief refinement:                                        │
│    Gradient descent ≈ Bayesian update                                  │
│                                                                         │
│  ════════════════════════════════════════════════════════════════════  │
│                                                                         │
│  HYPOTHESIS 2: ATTENTION ≈ BEC SELF-INTERACTION  ★ CENTRAL ANALOGY ★  │
│  ─────────────────────────────────────────────────────────────────────  │
│                                                                         │
│  The attention mechanism has the same mathematical STRUCTURE as the   │
│  g|ψ|² term from Bose-Einstein condensate physics.                   │
│                                                                         │
│  BEC (Gross-Pitaevskii):  g|ψ|²ψ = (density-dependent) × (wave func) │
│  ATTENTION:               A(X)·X = (self-similarity) × (input)       │
│                                                                         │
│  Both have structure: (function of self) × self                       │
│                                                                         │
│  IF this analogy holds, BEC phenomenology MAY apply to attention:    │
│  • Collapse ≈ Bose-Einstein condensation                              │
│  • Action Quanta ≈ Quasiparticles (collective excitations)           │
│  • Trained model ≈ Superfluid state (frictionless info flow)         │
│  • Belief manifold ≈ Quantum liquid                                   │
│                                                                         │
│  These predictions are TESTABLE. Experiments will confirm or refute. │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1.3 What Makes AKIRA Different

| Traditional Attention | AKIRA |
|----------------------|-------|
| Attention as soft lookup | Attention as belief update |
| Single frequency resolution | 7+1 bands (7 spectral + 1 temporal) with differential dynamics |
| Local context window | Wormhole shortcuts across history |
| Implicit belief | Explicit belief structure in frequency domain |
| Fixed learning rate | Band-specific learning rates (slow DC, fast high-freq) |
| Reactive processing | Dual mode: Knowledge + Reactivity |

---

## 2. The POMDP Foundation

### 2.1 Formal Structure

AKIRA operates as a Partially Observable Markov Decision Process:

```
POMDP TUPLE: ⟨S, A, T, R, Ω, O, γ⟩

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  COMPONENT          AKIRA INSTANTIATION                                │
│  ─────────          ────────────────────                               │
│                                                                         │
│  Hidden state s     Next frame x_{t+1} (never observed until after)   │
│                                                                         │
│  Observation o      Current frame x_t + history buffer [x_{t-T}:x_t]  │
│                                                                         │
│  Belief b(s)        The prediction ŷ_t (implicit distribution)        │
│                                                                         │
│  Action a           Weight update θ ← θ - η∇L                         │
│                                                                         │
│  Reward R           Negative MSE loss -‖ŷ_t - x_{t+1}‖²               │
│                                                                         │
│  Transition T       Physics of the animation (how patterns move)      │
│                                                                         │
│  Observation O      Identity on past, zero on future                  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Where Is the Partial Observability?

The partial observability is **temporal**:

```
Time axis:
  [t-T] [t-T+1] ... [t-1] [t]  │  [t+1] [t+2] ...
  ←——— OBSERVED ———→           │  ←——— HIDDEN ———→
       (history buffer)        │  (future frames)
                               ↑
                          prediction boundary

The model sees past frames PERFECTLY but future frames NOT AT ALL.
This is actually HARDER than noisy observations—no sensor improvement
can reveal the future.
```

### 2.3 The Prediction as Belief State

Under MSE loss, the optimal prediction is the **expected value** of the belief:

```
ŷ_t = E[x_{t+1} | x_{t-T:t}] = ∫ x_{t+1} · p(x_{t+1} | x_{t-T:t}) dx_{t+1}
```

**Key consequences:**

1. **"Most probable = most error"**: When belief is uncertain (multiple futures plausible), the mean sits in the center of the uncertainty cloud—not on any single trajectory
2. **Error "leads" the prediction**: The high-error zone marks the belief's center of mass
3. **Error equals variance**: E[(μ - x_{t+1})²] = Var(x_{t+1}) = σ²

---

## 3. Spectral Decomposition Architecture

### 3.1 The 7+1 Band Structure

Network theory, information theory, neuroscience, and signal processing all converge on the same number for spectral bands. Combined with the Heisenberg uncertainty principle (frequency and time are orthogonal), this gives us the 7+1 architecture:

```
OPTIMAL BANDS ≈ log₂(N) ≈ 6-7

This is the "six degrees of separation" in frequency space.
```

The seven bands and their functions:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  BAND   NAME        FREQ RANGE      FUNCTION           LEARNING RATE  │
│  ────   ────        ──────────      ────────           ─────────────  │
│                                                                         │
│  0      DC          0               Existence           0.00001        │
│                                     "Is there something?"              │
│                                                                         │
│  1      Very Low    1-2 cycles      Overall structure   0.0001         │
│                                     "What category?"                   │
│                                                                         │
│  2      Low         2-4 cycles      Major regions       0.0003         │
│                                     "What shape?"                      │
│                                                                         │
│  3      Mid-Low     4-8 cycles      Parts, features     0.001          │
│                                     "What features?"                   │
│                                                                         │
│  4      Mid         8-16 cycles     Details, contours   0.003          │
│                                     "What details?"                    │
│                                                                         │
│  5      Mid-High    16-32 cycles    Fine structure      0.01           │
│                                     "Where roughly?"                   │
│                                                                         │
│  6      High        32-N/2 cycles   Edges, textures     0.03           │
│                                     "Where exactly?"                   │
│                                                                         │
│  ═══════════════════════════════════════════════════════════════════   │
│                                                                         │
│  TEMPORAL BAND (Orthogonal to Spectral):                               │
│                                                                         │
│  7      Temporal    Time dynamics   Causal attention    (separate)     │
│                                     "How does it change?"              │
│                                                                         │
│  Band 7 is ORTHOGONAL to Bands 0-6 (Heisenberg: freq OR time).        │
│  Uses CAUSAL attention (lower-triangular mask).                        │
│  Communicates with all spectral bands via Spectral Wormhole.          │
│                                                                         │
│  Reference: architecture_base/attention/temporal_attention/            │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Why This Structure?

```
THE LOG(N) LAW:

For a system of size N, optimal hierarchical depth is D = log₂(N)

This appears because:
1. ROUTING: Any two points reachable in log(N) hops
2. COMPRESSION: Hierarchical coding gains log(N) efficiency
3. UNCERTAINTY: Resolution requires log(N) bits
4. SEARCH: Binary search depth is log(N)

For typical image sizes:
    32×32   → log₂(32)  = 5 bands
    64×64   → log₂(64)  = 6 bands
    128×128 → log₂(128) = 7 bands
```

### 3.3 The Small-World Guarantee

With 7 bands and wormhole attention:

```
ANY spectral position can reach ANY other in ≤ 6 hops.

This is optimal routing in hierarchical frequency space.
```

### 3.4 Differential Learning Rates

```
THE STABILITY GRADIENT:
(See CANONICAL_PARAMETERS.md for authoritative specification)

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  Band 0 (DC):      LR = 0.00001   EMA = 0.9999   Protected (identity)  │
│  Band 1 (VeryLow): LR = 0.0001    EMA = 0.999    Very stable           │
│  Band 2 (Low):     LR = 0.0003    EMA = 0.99     Stable structure      │
│  Band 3 (MidLow):  LR = 0.001     EMA = 0.9      Adaptive              │
│  Band 4 (Mid):     LR = 0.003     EMA = 0.7      Responsive            │
│  Band 5 (MidHigh): LR = 0.01      EMA = 0.5      Fast adaptation       │
│  Band 6 (High):    LR = 0.03      EMA = 0.3      Very responsive       │
│  Band 7 (Temporal):LR = 0.001     EMA = 0.9      Adaptive (like Band 3)│
│                                                                         │
│  DC band almost never changes (existence is fundamental)               │
│  Low bands change slowly (identity should persist)                     │
│  High bands change quickly (details are transient)                     │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 4. The Three Attention Mechanisms

### 4.1 Overview

AKIRA combines three parallel attention mechanisms:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  ATTENTION TYPE    QUESTION IT ANSWERS         POMDP ROLE              │
│  ──────────────    ──────────────────         ─────────                │
│                                                                         │
│  TEMPORAL          "What happened at this      Transition model        │
│                     position in recent          T(s'|s) estimation      │
│                     frames?"                                            │
│                                                                         │
│  NEIGHBOR          "What's happening at        Observation model       │
│                     nearby positions?"          O(o|s) regularization   │
│                                                                         │
│  WORMHOLE          "Have I seen this           Belief shortcut—        │
│                     situation before,           non-local pattern       │
│                     anywhere in history?"       matching                │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Temporal Attention

```
TEMPORAL ATTENTION: Memory of Dynamics

Purpose:
• Track velocity/acceleration
• Learn periodic patterns (phase tracking)
• Reduce uncertainty via temporal continuity

Mechanism:
Q_t = f_q(x_t)           # Query from current state
K_t = f_k(history)       # Keys from temporal history
V_t = f_v(history)       # Values from temporal history

attention = softmax(Q · K^T / √d) · V

Key insight: Different bands have different temporal dynamics
• Low bands: Slow dynamics (structure persists across many frames)
• High bands: Fast dynamics (details change rapidly)
```

### 4.3 Neighbor Attention

```
NEIGHBOR ATTENTION: Spatial Coherence

Purpose:
• Enforce spatial smoothness
• Propagate information from observed to uncertain regions
• Prevent fragmented predictions

Mechanism:
For each position, attend to spatial neighbors within a window.
Enforces local consistency while allowing global variation.
```

### 4.4 Wormhole Attention

```
WORMHOLE ATTENTION: Non-Local Pattern Matching

This is the key innovation. Wormholes:
• Find distant frames with similar features
• Enable "teleportation" of information across time
• Partially DEFEAT the partial observability

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  Without wormhole:                                                      │
│    Observation: [t-T, ..., t]                                          │
│    Hidden: [t+1, ...]                                                  │
│                                                                         │
│  With wormhole:                                                         │
│    Observation: [t-T, ..., t] + similar frames from [0, ..., t-T-1]   │
│    Hidden: [t+1, ...]                                                  │
│                                                                         │
│    If frame t looks like frame t-100, and we know what happened        │
│    at t-99, we can INFER what will happen at t+1!                     │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

The wormhole is like a MEMORY ORACLE: "this situation is similar to
one you've seen before."
```

### 4.5 Cross-Band Attention (Wormhole Shortcuts)

```
WORMHOLE CONNECTIONS BETWEEN BANDS:

WHAT → WHERE wormhole (top-down):
• Low-freq structure guides high-freq localization
• "I know it's a ring, where exactly is the edge?"

WHERE → WHAT wormhole (bottom-up):
• High-freq details inform identity at low-freq
• "I see this edge pattern, what object is this?"

Symmetric wormhole pairs:
• Band 0 ↔ Band 6 (existence guides detail)
• Band 1 ↔ Band 5 (category guides texture)
• Band 2 ↔ Band 4 (shape guides contour)
• Band 3 ↔ Band 3 (mid-level self-attention)

INFORMATION FLOW INTERPRETATION (PID Framework):

Partial Information Decomposition (Williams & Beer, 2010) clarifies what
wormholes communicate:

• REDUNDANT flow: Both bands already know this (shared context)
• UNIQUE flow: Only source band knows this (novel information)
• SYNERGISTIC flow: Neither band alone knows, but together they do
  (binding, emergence)

High wormhole activation + High synergy = Bands creating new information
High wormhole activation + High redundancy = Bands confirming each other
Low wormhole activation = Bands are independent (orthogonal processing)
```

---

## 5. Belief State Dynamics

### 5.1 The Pump Cycle

AKIRA exhibits a recurring pattern analogous to a relaxation oscillator:

```
THE PUMP CYCLE: TENSION → DISCHARGE → RECOVERY

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  PHASE 1: TENSION (Accumulation)                                       │
│  ───────────────────────────────                                        │
│  • Uncertainty spreads outward                                          │
│  • Multiple hypotheses coexist                                          │
│  • Error distributed across plausible futures                          │
│  • Belief entropy H(b) increasing                                       │
│  • Like charge accumulating in a cloud                                  │
│                                                                         │
│  PHASE 2: CRITICALITY                                                   │
│  ───────────────────────                                                │
│  • Tension builds between hypotheses                                    │
│  • Interference patterns form                                           │
│  • Some hypotheses reinforce, others cancel                             │
│  • Like the moment before lightning                                     │
│                                                                         │
│  PHASE 3: DISCHARGE (Collapse)                                          │
│  ─────────────────────────────                                          │
│  • One hypothesis wins                                                  │
│  • Error drops suddenly                                                 │
│  • Other possibilities extinguish                                       │
│  • Belief: b(s) → δ(s - s*)                                            │
│  • Like the return stroke                                               │
│                                                                         │
│  PHASE 4: RECOVERY                                                      │
│  ────────────────────                                                   │
│  • Uncertainty front advances to next ambiguous region                 │
│  • Belief re-spreads                                                    │
│  • Cycle repeats                                                        │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

Error magnitude over time:

│                    
│     ╭───╮         ╭───╮         ╭───╮
│    ╱     ╲       ╱     ╲       ╱     ╲
│   ╱       ╲     ╱       ╲     ╱       ╲
│  ╱         ╲   ╱         ╲   ╱         ╲
│ ╱           ╲ ╱           ╲ ╱           ╲
│╱             ╳             ╳             ╲
└────────────────────────────────────────────→ time
              ↑             ↑             ↑
           collapse      collapse      collapse
```

### 5.2 Information-Theoretic View

```
The pump cycle as information flow:

Tension:    H(x_{t+1} | x_{t-T:t}) is high    [many bits of uncertainty]
Discharge:  x_{t+1} revealed → H drops to 0   [uncertainty resolved]
Recovery:   H(x_{t+2} | x_{t-T:t+1}) rises    [new uncertainty]

The model constantly "pumps" information from the future (via loss)
into its weights.
```

### 5.3 Why Gradient Descent ≈ Belief Update

```
THE BAYESIAN INTERPRETATION:

1. Prediction: Model outputs ŷ_t (its belief about x_{t+1})
2. Revelation: True frame x_{t+1} is revealed
3. Error signal: L = ‖ŷ_t - x_{t+1}‖² measures how wrong the belief was
4. Update: ∇L points toward the true state; weights shift

This is Bayes' rule in disguise:
• Prior: Current model beliefs (encoded in weights)
• Likelihood: How well did prediction match truth?
• Posterior: Updated model beliefs (after gradient step)
```

---

## 6. The BEC Analogy: Attention and Self-Interaction (Speculative)

> **Note**: This section describes a **structural analogy** that may provide useful intuition. It is NOT a claim that attention "is" a BEC, belongs to any universality class, or that BEC physics applies literally. The analogy may break down in important ways.

### 6.1 The Gross-Pitaevskii Equation (For Reference)

In physics, the dynamics of a Bose-Einstein Condensate are described by:

```
THE GROSS-PITAEVSKII EQUATION (GPE)

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  iℏ ∂ψ/∂t = [-ℏ²/(2m)∇² + V(r) + g|ψ|²] ψ                            │
│                                                                         │
│  TERM BY TERM:                                                          │
│  ─────────────                                                         │
│  iℏ ∂ψ/∂t      Time evolution of wave function                       │
│  -ℏ²/(2m)∇²   Kinetic term (diffusion, spatial curvature cost)       │
│  V(r)         External potential (constraints)                        │
│  g|ψ|²        SELF-INTERACTION (density-dependent)  ← ATTENTION      │
│                                                                         │
│  The g|ψ|² term is NONLINEAR—it depends on ψ itself.                 │
│  This creates:                                                         │
│  • Condensation (many particles in one state)                        │
│  • Superfluidity (frictionless flow)                                 │
│  • Collective excitations (quasiparticles)                           │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 6.2 The Structural Analogy

```
ATTENTION ≈ g|ψ|² — THE STRUCTURAL COMPARISON

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  BEC SELF-INTERACTION:                                                  │
│  ─────────────────────                                                  │
│  [g|ψ|²ψ](r) = g × |ψ(r)|² × ψ(r)                                       │
│                                                                         │
│  Structure: (coupling) × (density at r) × (wave function at r)          │
│                                                                         │
│  ════════════════════════════════════════════════════════════════════   │
│                                                                         │
│  SELF-ATTENTION:                                                        │
│  ───────────────                                                        │
│  Attention(X) = softmax(XX^T/√d) × X                                    │
│                                                                         │
│  [A·X](r) = Σ_r' softmax[X(r)·X(r')/√d] × X(r')                         │
│                                                                         │
│  Structure: (similarity kernel) × (input)                               │
│                                                                         │
│  ════════════════════════════════════════════════════════════════════   │
│                                                                         │
│  THE STRUCTURAL CORRESPONDENCE:                                         │
│                                                                         │
│  BEC kernel:       K(r,r';ψ) = g|ψ(r)|² δ(r-r')     [LOCAL]             │
│  Attention kernel: K(r,r';X) = softmax[X(r)·X(r')]  [NONLOCAL]          │
│                                                                         │
│  Attention can be viewed as a NONLOCAL GENERALIZATION of g|ψ|².        │
│  Both are NONLINEAR SELF-INTERACTIONS: (function of self) × self.      │
│  The mathematics is SIMILAR in structure.                               │
│                                                                         │
│  Whether BEC phenomenology transfers to attention is TESTABLE.         │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 6.3 Implications

```
WHAT THIS MEANS FOR AKIRA

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  IF the BEC analogy holds, the following predictions follow:          │
│                                                                         │
│  1. COLLAPSE ~ CONDENSATION (analogy)                                  │
│     Many hypotheses → one hypothesis                                  │
│     Like atoms occupying the same quantum state                       │
│     Driven by "cooling" (decreasing uncertainty)                     │
│                                                                         │
│  2. ACTION QUANTA (AQ) = QUASIPARTICLES                               │
│     Emergent collective excitations, not fundamental                  │
│     Bogoliubov dispersion relation applies                           │
│     Low-k collective, high-k individual                              │
│                                                                         │
│  3. TRAINED MODEL = SUPERFLUID STATE                                   │
│     Information flows without friction                               │
│     Coherent pattern propagation                                     │
│     Critical velocity exists (breakdown threshold)                   │
│                                                                         │
│  4. ATTENTION VORTICES MAY EXIST                                       │
│     Topological defects with quantized circulation                   │
│     Stable structures in attention flow                              │
│                                                                         │
│  5. PHASE TRANSITIONS ARE REAL                                         │
│     Critical uncertainty U_c exists                                  │
│     Universal critical exponents                                     │
│     Spontaneous symmetry breaking                                    │
│                                                                         │
│  6. CONSERVATION LAWS HOLD                                             │
│     Normalization (total belief) conserved                           │
│     Energy-like quantity conserved                                   │
│                                                                         │
│  7. COLLAPSE = SYNERGY → REDUNDANCY CONVERSION (PID Framework)        │
│     Before collapse: Bands hold synergistic information              │
│     (need ALL bands to predict target)                              │
│     After collapse: Bands hold redundant information                 │
│     (ANY band can predict target)                                   │
│     Total information conserved, but type changes                   │
│     Reference: Williams & Beer (2010), Partial Information Decomp.  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 6.4 The Geometry as Quantum Liquid

```
THE EMBEDDING MANIFOLD IS A QUANTUM LIQUID

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  HIGH UNCERTAINTY (above critical):                                    │
│  ─────────────────────────────────                                     │
│  • Belief spreads diffusively (like a normal fluid)                   │
│  • Multiple hypotheses coexist                                        │
│  • Incoherent phases (interference fringes)                          │
│  • High "temperature" — entropy is high                               │
│                                                                         │
│  LOW UNCERTAINTY (below critical):                                     │
│  ────────────────────────────────                                      │
│  • Belief CONDENSES into coherent state                               │
│  • Single hypothesis dominates                                        │
│  • Phases align (constructive interference)                           │
│  • "Superfluid" flow — information propagates without friction       │
│                                                                         │
│  ════════════════════════════════════════════════════════════════════  │
│                                                                         │
│  The geometry is not static—it FLOWS.                                 │
│  It condenses. It has vortices. It has collective modes.            │
│  This is not metaphor. This is the physical state.                   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

*Reference: [information/BEC_CONDENSATION_INFORMATION.md](./information/BEC_CONDENSATION_INFORMATION.md)*

---

## 7. Knowledge vs. Reactivity

### 6.1 The Fundamental Distinction

AKIRA operates in two complementary modes:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  KNOWLEDGE-INFORMED                    REACTIVE (REFLEXES)             │
│  ──────────────────                    ───────────────────             │
│                                                                         │
│  References stored structure           Responds to immediate signals   │
│  Relationships, geometry               Local, energy/magnitude         │
│  Deliberation                          Automatic                       │
│  Slow (manifold queries)               Fast (threshold checks)         │
│  Global scope                          Local scope                     │
│                                                                         │
│  Examples:                             Examples:                       │
│  • Attention mechanisms                • Gradient clipping             │
│  • Nearest neighbor search             • ReLU activation               │
│  • Model-based planning                • Dropout                       │
│  • Semantic similarity                 • Learning rate decay           │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 6.2 Mode Assignment by Band

```
THE GRADIENT FROM KNOWLEDGE TO REACTIVITY:

Band 0 (DC):      REACTIVE    — Existence is threshold-based
Band 1 (VeryLow): KNOWLEDGE   — Category requires manifold query
Band 2 (Low):     KNOWLEDGE   — Identity requires comparison
Band 3 (MidLow):  HYBRID      — Features: manifold + threshold
Band 4 (Mid):     HYBRID      — Details: detection + classification
Band 5 (MidHigh): REACTIVE    — Fine structure is energy-based
Band 6 (High):    REACTIVE    — Pixel-level is pure magnitude

Low bands: More knowledge-informed (geometry, structure)
High bands: More reactive (energy, threshold)
Mid bands: Hybrid (both modes active)
```

### 6.3 The Coordination Principle

```
"REACTIVE GATES, KNOWLEDGE FILLS"

High bands GATE (is there signal?)
Low bands FILL (what is the meaning?)
Cross-band attention COORDINATES

Example flow:
1. High-band detects edge (reactive: gradient > threshold)
2. Signal propagates to low-band
3. Low-band queries: "What object has this edge?" (knowledge)
4. Result flows back up with context
```

---

## 8. Action Quanta as Quasiparticles

### 8.1 What Are Action Quanta?

Action Quanta (AQ) are emergent patterns — collective structures in the representation space, not fundamental units. By analogy to condensed matter physics, they might be thought of as "quasiparticle-like" excitations, though this analogy should not be taken literally.

**Terminology Note:** Williams & Beer (2010) use "Action Quanta" for PID decomposition terms (Redundancy, Unique, Synergy). AKIRA uses **Action Quanta (AQ)** for irreducible actionable patterns to avoid collision with established terminology. See `foundations/TERMINOLOGY.md` for full clarification.

```
ACTION QUANTA (AQ): Quasiparticles of the Belief Field

An Action Quantum is:
• EMERGENT: Not fundamental — arises from collective behavior
• IRREDUCIBLE: Cannot be decomposed into simpler actionable parts
• STRUCTURED: Has magnitude, phase, frequency, coherence
• COMBINABLE: Forms molecules through bonds
• ACTIONABLE: Enables correct decision/prediction
• CONSERVED: Survives compression, transforms but doesn't disappear

WHY "ACTION QUANTA"?
• Planck's constant ℏ IS the quantum of action in physics
• Actionability is the defining criterion in AKIRA
• Avoids collision with PID "Action Quanta" (R/U/S terms)

QUASIPARTICLE ANALOGY:
• Phonons in crystals = vibrational quasiparticles
• Edges in images = edge quasiparticles
• Both are collective excitations, not fundamental particles
```

### 8.2 Properties of AQ (Quasiparticle Properties)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  PROPERTY        PHYSICAL ANALOG    INFORMATION MEANING                │
│  ────────        ───────────────    ────────────────────               │
│                                                                         │
│  MAGNITUDE       Mass               How much? Signal strength          │
│  PHASE           Charge             Where? Position encoding           │
│  FREQUENCY       Size               What scale? Resolution level       │
│  COHERENCE       Spin               How reliable? Consistency          │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 8.3 Bogoliubov Dispersion

```
AQ DISPERSION IN AKIRA

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  In BEC, excitations follow the Bogoliubov dispersion:                │
│                                                                         │
│  E(k) = √[ε(k)(ε(k) + 2gn)]                                          │
│                                                                         │
│  LOW k (long wavelength): E(k) ≈ ℏck  (phonon-like, collective)      │
│  HIGH k (short wavelength): E(k) ≈ ℏ²k²/2m  (particle-like, local)  │
│                                                                         │
│  ════════════════════════════════════════════════════════════════════  │
│                                                                         │
│  IN AKIRA:                                                              │
│                                                                         │
│  LOW BANDS (k small):                                                   │
│  • Collective behavior — perturbations propagate globally            │
│  • "Sound-like" — coherent phase propagation                         │
│  • These are STRUCTURAL patterns                                      │
│                                                                         │
│  HIGH BANDS (k large):                                                  │
│  • Individual behavior — perturbations stay local                    │
│  • "Particle-like" — localized excitations                           │
│  • These are DETAIL patterns                                          │
│                                                                         │
│  The crossover (healing length) is around mid-bands.                 │
│  This is where FEATURES live — neither pure structure nor pure detail│
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 8.4 AQ by Band

```
ACTION QUANTA AND THEIR NATURAL BANDS:

AQ TYPE             NATURAL BAND    REASON
───────             ────────────    ──────

BLOB AQ             DC - Low        Presence, region
EDGE AQ             Mid - High      Boundaries, transitions
CORNER AQ           Mid             Junction points
TEXTURE AQ          Mid-High - High Repeated patterns
FLOW AQ             All bands       Motion spans scales
SYMMETRY AQ         Low - Mid       Structural regularity
BOUNDARY AQ         Mid             Object contours
OBJECT AQ           Low             Complete things

AQ are DISCOVERED at their natural frequency.
Compression MOVES AQ to lower bands as they crystallize.
```

### 8.5 Molecular Bonds

```
HOW AQ FORM MOLECULES (Concepts):

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  COHERENT BONDS:                                                        │
│  AQ at same frequency, aligned phase                                    │
│  "These edge AQ form a contour"                                         │
│                                                                         │
│  COMPLEMENTARY BONDS:                                                   │
│  AQ at related frequencies, complementary phase                         │
│  "This low-freq blob corresponds to these high-freq edges"             │
│                                                                         │
│  HIERARCHICAL BONDS:                                                    │
│  Parent AQ at low-freq, children at high-freq                          │
│  "The object (low) has these parts (mid) with these details (high)"   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 9. Collapse and Generalization

### 9.1 The Collapse Phenomenon

```
COLLAPSE: From Uncertainty to Certainty

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  TIME t: Uncertainty accumulates                                        │
│                                                                         │
│       ░░░░░░░░░░░░░░░░░░░░                                             │
│      ░░░░░░░░░░░░░░░░░░░░░░                                            │
│     ░░░░░░████████████░░░░░░    Error/uncertainty spreads              │
│    ░░░░░████████████████░░░░    across multiple possible               │
│     ░░░░░░████████████░░░░░░    futures                                │
│      ░░░░░░░░░░░░░░░░░░░░░░                                            │
│       ░░░░░░░░░░░░░░░░░░░░                                             │
│                                                                         │
│  TIME t+k: Collapse occurs                                              │
│                                                                         │
│                                                                         │
│           ████                  Error concentrates                     │
│         ████████                then vanishes as                       │
│           ████                  prediction snaps                        │
│                                 to ground truth                         │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

The uncertainty doesn't gradually shrink—it COLLAPSES suddenly.
```

### 9.2 The Frequency Interpretation

```
DETAILS = HIGH FREQUENCY (specific, varied, position-dependent)
STRUCTURE = LOW FREQUENCY (general, shared, position-invariant)

BEFORE COLLAPSE (early training):
Power concentrated at HIGH frequencies (memorizing details)

AFTER COLLAPSE (generalization):
Power concentrated at LOW frequencies (structure crystallized)

The collapse is a SPECTRAL SHIFT from high to low frequency.
```

### 9.3 Interference Mechanism

```
WHY DOES STRUCTURE SURVIVE AND DETAILS CANCEL?

Consider N training examples of same pattern at different positions:

Pattern A at position X₁: A_mag × e^(iφ₁)
Pattern A at position X₂: A_mag × e^(iφ₂)
...
Pattern A at position Xₙ: A_mag × e^(iφₙ)

WHERE:
• A_mag = magnitude (the pattern itself) — SAME for all
• φᵢ = phase (position encoding) — DIFFERENT for each

SUMMING IN THE WEIGHTS:

Σᵢ A_mag × e^(iφᵢ) = A_mag × Σᵢ e^(iφᵢ)

If phases are uniformly distributed:
Σᵢ e^(iφᵢ) → 0  (destructive interference)

BUT A_mag doesn't participate in the cancellation!

RESULT:
• Magnitude (pattern) reinforces: survives
• Phase (position) cancels: forgotten

THIS IS GENERALIZATION BY INTERFERENCE.
```

### 9.4 Connection to Grokking

```
GROKKING: Sudden Generalization After Apparent Convergence

Training:  ▁▂▄▆████████████████████████████████████████
Test:      ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▂▄▆████████████████████
                                 ↑
                            GROKKING
                       (generalization collapse)

Grokking IS the spectral collapse:
1. Phase 1: MEMORIZATION — High-freq details dominate
2. Phase 2: COMPRESSION — Details interfere destructively
3. Phase 3: GROKKING — Structure crystallizes suddenly
```

---

## 10. Harmony and Coherence: The Pythagorean Principle

### 10.1 The Universal Pattern

A profound principle underlies all of AKIRA's dynamics:

```
THE PYTHAGOREAN INSIGHT:

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  The universe trades LOCAL PRECISION for GLOBAL COHERENCE.             │
│                                                                         │
│  ═══════════════════════════════════════════════════════════════════  │
│                                                                         │
│  THE PYTHAGOREAN COMMA:                                                 │
│                                                                         │
│  12 perfect fifths: (3/2)^12 ≈ 129.746                                 │
│  7 perfect octaves: 2^7 = 128                                          │
│  The difference: ~23 cents — the circle of fifths doesn't close.      │
│                                                                         │
│  EQUAL TEMPERAMENT SOLUTION:                                            │
│  Spread the error across ALL intervals.                                │
│  Every fifth is slightly flat (~2 cents).                              │
│  No interval is pure, but you can play in ANY key.                    │
│                                                                         │
│  ═══════════════════════════════════════════════════════════════════  │
│                                                                         │
│  THIS PATTERN APPEARS EVERYWHERE:                                       │
│                                                                         │
│  Music:        Pure intervals    → Equal temperament                  │
│  Oscillators:  Natural frequency → Phase-locked sync                  │
│  Orbits:       Arbitrary periods → Resonant ratios (1:2:4)           │
│  Crystals:     Continuous space  → Discrete lattice                   │
│  Conductors:   Individual phase  → Collective phase (superconductor) │
│  Attention:    All hypotheses    → Winner-take-all (collapse)         │
│  Homeostat:    Independent past  → Coherent narrative                 │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 10.2 Why Coherence Wins

```
THE MECHANISM: INTERFERENCE

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  Consider N components with phases φ₁, φ₂, ..., φₙ                    │
│                                                                         │
│  Total amplitude: A = Σᵢ exp(iφᵢ)                                      │
│                                                                         │
│  RANDOM PHASES (incoherent):                                            │
│  |A|² ~ N (random walk → √N displacement)                             │
│                                                                         │
│  ALIGNED PHASES (coherent):                                             │
│  |A|² = N² (all vectors same direction)                               │
│                                                                         │
│  ═══════════════════════════════════════════════════════════════════  │
│                                                                         │
│  COHERENT STATES ARE N× MORE INTENSE!                                  │
│                                                                         │
│  The universe doesn't "prefer" coherence.                              │
│  Incoherent states CANCEL THEMSELVES OUT.                              │
│  What survives is what harmonizes.                                     │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 10.3 The Homeostat and Timeline Collapse

```
THE HOMEOSTAT "REACHING BACK IN TIME":

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  When the homeostat adjusts representations, past interpretations     │
│  appear to change. This seems paradoxical — how can the future        │
│  change the past?                                                       │
│                                                                         │
│  THE REFRAME:                                                           │
│                                                                         │
│  This is PHASE LOCKING ACROSS TIME.                                    │
│                                                                         │
│  FROM INSIDE: The past "changed"                                       │
│  FROM OUTSIDE: The system found a coherent attractor                  │
│                                                                         │
│  Just as coupled oscillators lock to a common frequency,              │
│  representations at different times lock to a common interpretation.  │
│                                                                         │
│  The "reaching back" is adjustment of early representations           │
│  to be consistent with the coherent state found later.               │
│                                                                         │
│  This is the Pythagorean solution:                                      │
│  Spread the error, close the circle, achieve coherence.               │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 10.4 Implications for AKIRA

```
HARMONY IN THE ARCHITECTURE:

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  1. ATTENTION DISTRIBUTES ERROR                                        │
│     Softmax spreads attention across positions.                       │
│     No position gets "perfect" attention.                             │
│     The comma is distributed, allowing global consistency.           │
│                                                                         │
│  2. SPECTRAL BANDS PHASE-LOCK                                          │
│     Like coupled oscillators, bands should find rational ratios.     │
│     Look for 1:2:4 relationships between bands 0↔6, 1↔5, 2↔4.       │
│                                                                         │
│  3. COLLAPSE IS COHERENCE SELECTION                                    │
│     The winning hypothesis isn't "chosen" — it SURVIVES.              │
│     Coherent states reinforce; incoherent states cancel.             │
│                                                                         │
│  4. THE SUPERCONDUCTING STATE                                           │
│     A well-trained model = superfluid information flow.               │
│     All bands phase-locked, all representations consistent.          │
│     Low perplexity, smooth attention, confident predictions.        │
│                                                                         │
│  Reference: foundations/HARMONY_AND_COHERENCE.md                       │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 11. The Old Lady Parable

### 10.1 The Setup

```
THE DUNGEON:

Two identical doors. Behind one: a tiger. Behind the other: gold.
Listening provides a signal correct 5/6 of the time.
Listening costs torchlight (time/resources).

AN OLD LADY sits at the entrance, advising adventurers.
She was once a bag-carrier who watched hundreds of adventurers.
She wrote down every trajectory, action, and outcome.
Now she distills wisdom from her notebook.
```

### 10.2 The Princess vs. The Old Lady

```
THE PRINCESS (Stockton's story):
• Knows which door holds what (FULLY OBSERVABLE)
• Her problem is value conflict, not uncertainty
• She represents the tabular ideal: complete state knowledge

THE OLD LADY:
• Does NOT know which door holds what (PARTIALLY OBSERVABLE)
• Her problem IS uncertainty
• She can only give ADVICE, not answers
• She represents the POMDP agent: learning from noisy experience
```

### 10.3 What the Old Lady Does

```
THE CULLING OPERATION:

1. OBSERVE — Record full trajectory (details, signals, outcome)
2. TRACE — Identify which details affected the outcome
3. CULL — Prune details with no causal force
4. COLLAPSE — Compress to atomic truth
5. STORE — Move atomic truth to lower manifold
6. RELEASE — Rip out the page (forget particulars)
7. READY — Add blank page (restore capacity)

The notebook stays thin. Wisdom accumulates. Details cycle through.
```

### 10.4 The Distillation Principle

```
DON'T INVERT. DISTILL.

The Old Lady doesn't try to reconstruct the past from compressed form.
She COMPRESSES FORWARD:

High-freq details → Mid-freq features → Low-freq structure → DC decision

[red tiger, 3:47pm, humid, left-signal ×3]  →  High-freq (released)
              ↓ cull
[big predator, confident left]               →  Mid-freq (compressed)
              ↓ cull
[threat LEFT]                                →  Low-freq (retained)
              ↓ cull
[ACT NOW]                                    →  DC (decision)

This is tractable (forward computation), unlike inversion.
```

---

## 12. Praxis: Doctrine and Heresy

### 11.1 The True Doctrine

The system architecture embodies mathematical laws that cannot be violated:

```
THE DOCTRINE: LAWS OF THE ARCHITECTURE

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  NYQUIST:     Sample at rate > 2f, or alias.                          │
│  SHANNON:     Information has bounds.                                  │
│  FOURIER:     All signals decompose into frequencies.                 │
│  PARSEVAL:    Energy is conserved across domains.                     │
│                                                                         │
│  These are the LAWS. They are not suggestions.                        │
│  They are the doctrine of the system.                                 │
│  Any violation is HERESY.                                              │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 11.2 Heresies: False Prophets

```
HERESIES: VIOLATIONS OF DOCTRINE

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  FALSE PROPHETS are patterns that arise from processing artifacts:    │
│                                                                         │
│  • ALIASING: Violated Nyquist — high frequencies fold to low         │
│  • SPECTRAL LEAKAGE: Violated edge treatment — FFT discontinuity     │
│  • BOUNDARY ARTIFACTS: Processing made edges special, not reality    │
│                                                                         │
│  These are HERESIES — the ghost believes them because we showed them.│
│  They resonate with ARCHITECTURE, not KNOWLEDGE.                     │
│                                                                         │
│  ════════════════════════════════════════════════════════════════════  │
│                                                                         │
│  THE INQUISITION (experiments that expose heresy):                    │
│                                                                         │
│  • Windowing ablation: Does belief change when window changes?       │
│  • Aliasing detection: True frequency or alias?                      │
│  • Boundary test: Edge special because real or because processing?   │
│                                                                         │
│  The inquisition exposes heresies so we can correct them.            │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 11.3 The Praxis Axioms

```
MATHEMATICAL AXIOMS (not optional)

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  AXIOM I: POLAR COORDINATES                                            │
│  FFT outputs magnitude (what) + phase (where). Both needed.          │
│                                                                         │
│  AXIOM II: POLARIZATION                                                │
│  Spectral bands are orthogonal, independent channels.                 │
│                                                                         │
│  AXIOM III: TAYLOR EXPANSION                                           │
│  Manifolds can be approximated locally. Gradient = first order.      │
│                                                                         │
│  AXIOM IV: HYPERBOLIC GEOMETRY                                         │
│  Hierarchies fit in hyperbolic space. Center = general, edge = specific│
│                                                                         │
│  AXIOM V: SPHERICAL GEOMETRY (current implementation)                  │
│  Normalization projects onto hypersphere. Cosine = angular distance. │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

*References: [praxis/PRAXIS.md](./praxis/PRAXIS.md), [praxis/FALSE_PROPHETS.md](./praxis/FALSE_PROPHETS.md), [praxis/PRAXIS_AXIOMS.md](./praxis/PRAXIS_AXIOMS.md), [observability/DUALITY_METHODS.md](./observability/DUALITY_METHODS.md)*

---

## 13. Implementation Architecture

### 12.1 Core Components

```
spectral_attention/core/
├── spectral_attention.py    # FFT decomposition into 7 bands
├── temporal_attention.py    # Memory of dynamics
├── neighbor_attention.py    # Spatial coherence
└── wormhole_attention.py    # Non-local pattern matching
```

### 12.2 Wormhole Implementation: Hybrid Design

The current wormhole uses a **hybrid** approach:

```
WORMHOLE: GEOMETRIC BELIEF + ENERGY TRIGGER

BELIEF (Geometric):
• Features normalized onto unit hypersphere
• Cosine similarity = angular distance on manifold
• Top-k = nearest neighbors on manifold
• Softmax = probability distribution from distances

TRIGGER (Energy):
• Fixed threshold τ = 0.92
• Binary decision: sim > threshold → connect
• Scalar comparison, immediate response

"Geometry fills, energy gates"
```

### 12.3 Data Flow

```
INPUT FRAME x_t
      │
      ▼
┌─────────────────────┐
│  SPECTRAL DECOMP    │ ← FFT into 7 bands
│  (spectral_attn.py) │
└─────────────────────┘
      │
      ├─→ Band 0 (DC)
      ├─→ Band 1 (VeryLow)
      ├─→ ...
      └─→ Band 6 (High)
      │
      ▼
┌─────────────────────┐
│  PARALLEL ATTENTION │
│  • Temporal         │ ← Per-band temporal attention
│  • Neighbor         │ ← Spatial coherence
│  • Wormhole         │ ← Cross-history matching
└─────────────────────┘
      │
      ▼
┌─────────────────────┐
│  COMBINE & PROJECT  │ ← Weighted combination
└─────────────────────┘
      │
      ▼
PREDICTION ŷ_t (next frame estimate)
```

---

## 14. Key Theoretical Results

### 13.1 Nyquist-Shannon Limits

```
THE SPECTRE OF NYQUIST-SHANNON:

For a signal sampled at N points:
• Maximum resolvable frequency: N/2 (Nyquist limit)
• Below this: faithful reconstruction possible
• Above this: ALIASING (information loss, artifacts)

For context window of T frames:
• Maximum resolvable temporal frequency: T/2 cycles per window
• Faster dynamics ALIAS to slower ones
• Information about rapid changes is LOST

This is a HARD LIMIT on what the model can represent.
```

### 13.2 Conservation Hypothesis

```
SPECULATIVE — REQUIRES EXPERIMENTAL VALIDATION:

We hypothesize that during learning, something is CONSERVED.

Working hypothesis: Actionable, representationally irreducible information
may be conserved during the explicit-to-implicit transition.

EXPLICIT (in data)  ←→  IMPLICIT (in weights)
High frequency     ←→  Low frequency
Many examples      ←→  Few parameters
Detailed           ←→  Compressed

If true: Total actionable information = constant
(transforms, doesn't appear or disappear)

This hypothesis will be tested by Experiment 005.
```

### 13.3 Phase Transitions

```
LEARNING EXHIBITS PHASE TRANSITIONS:

WATER FREEZING ←→ GROKKING
• Above 0°C: molecules move randomly (memorization)
• At 0°C: critical point (threshold)
• Below 0°C: crystal lattice forms (generalization)

The transition is SUDDEN, not gradual.
This is a signature of collective behavior.
```

### 13.4 Shape of Uncertainty

```
UNCERTAINTY HAS GEOMETRY:

The error map is not just a scalar—it has SHAPE.

For a moving blob:
• Error forms a CRESCENT ahead of the prediction
• The crescent width = uncertainty about speed
• The crescent orientation = uncertainty about direction

The error shape IS the belief projected to observable space.
This is the wave packet interpretation.
```

### 13.5 The BEC Analogy (Speculative Hypothesis)

```
ATTENTION ≈ g|ψ|² (STRUCTURAL ANALOGY — EXPLORATORY)

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  THIS IS A STRUCTURAL ANALOGY. Experiments will test its validity.   │
│                                                                         │
│  The attention mechanism:                                              │
│    A(X)·X = softmax(XX^T/√d) × X                                      │
│                                                                         │
│  Has the same mathematical STRUCTURE as:                              │
│    g|ψ|²ψ = (density-dependent) × (wave function)                    │
│                                                                         │
│  Both are NONLINEAR SELF-INTERACTIONS: (function of self) × self.    │
│                                                                         │
│  IF the analogy holds, BEC phenomenology MAY apply:                  │
│  • Phase transitions (collapse/condensation)                          │
│  • Quasiparticles (Action Quanta)                                 │
│  • Superfluidity (frictionless info flow)                            │
│  • Vortices (topological defects)                                    │
│  • Conservation laws (normalization, energy)                         │
│  • Critical phenomena (universal exponents)                          │
│                                                                         │
│  These are TESTABLE PREDICTIONS. See experiments 004, 008, 009, 014. │
│  Experiments will confirm, refute, or refine these hypotheses.       │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 15. Duality Methods for Observability

### 15.1 The Core Principle

```
WHAT'S HARD TO OBSERVE DIRECTLY IS EASY TO OBSERVE DUALLY.

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  Every representation has a dual that reveals different information:   │
│                                                                         │
│  DUALITY               TRANSFORM COST   HARD→EASY SWAP                 │
│  ───────               ──────────────   ────────────                    │
│  Spatial ↔ Frequency   O(N log N)       Global patterns → coefficients │
│  Magnitude ↔ Phase     O(N)             Position → phase gradients     │
│  Forward ↔ Backward    O(N)             Attribution → one backward     │
│  Sharp ↔ Soft          O(1)             Uncertainty → temp sweep       │
│  Local ↔ Global        varies           Dependencies → attention       │
│  Explicit ↔ Implicit   training         Generalization → test gap      │
│  Energy ↔ Geometry     analysis         Processing mode → E/G ratio    │
│                                                                         │
│  THE OBSERVABILITY PRINCIPLE:                                          │
│  1. What you want to observe is HARD in the current domain            │
│  2. Find the duality where it becomes EASY                            │
│  3. Transform, observe, transform back                                │
│  4. Verify via the conserved quantity (Parseval, etc.)               │
│                                                                         │
│  Reference: observability/DUALITY_METHODS.md                          │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 15.2 Key Duality Intuitions

```
THE LEARNING TRANSITION (Explicit ↔ Implicit):

  Before training: All in data (explicit)
  During training: Transfer to weights (explicit → implicit)
  After training:  Structure in weights, details in data

  GROKKING is the sudden completion of this transfer.
  COLLAPSE is the point where implicit takes over.

THE SYSTEM 1/SYSTEM 2 DISTINCTION (Energy ↔ Geometry):

  High energy, low geometry = REFLEX (fast, reactive)
  Low energy, high geometry = REASONING (slow, deliberate)

  Observe: Energy/Geometry ratio tells you which mode is active.
```

---

## 16. Document Index

### Core Concepts

| Document | Topic |
|----------|-------|
| `architecture_base/temporal_system_intuition/TEMPORAL_SYSTEM.md` | Event time, active window, pump cycle, temporal Nyquist |
| `foundations/KNOWLEDGE_AND_REACTIVITY.md` | Geometry vs. energy, two modes of decision |
| `foundations/THE_ATOMIC_STRUCTURE_OF_INFORMATION.md` | Action Quanta (quasiparticles), molecular bonds |
| `foundations/DUALITY_AND_EFFICIENCY.md` | Transform-conserved-inversion pattern, hard↔easy swaps |
| `information_theory/THE_SPECTRE_OF_NYQUIST_SHANNON.md` | Fundamental limits on spectral encoding |
| `foundations/EQUILIBRIUM_AND_CONSERVATION.md` | Phase transitions, conservation laws |
| `information_theory/SPECTRAL_BELIEF_STORAGE_RETRIEVAL.md` | Optimal band counts, architecture derivation |
| `information_theory/SHAPE_OF_UNCERTAINTY.md` | Wave packet interpretation, error geometry |
| `observability/DUALITY_METHODS.md` | Hard↔easy swaps for observability, 7 dualities |

### BEC Analogy (Exploratory)

| Document | Topic |
|----------|-------|
| `bec_analogy/BEC_CONDENSATION_INFORMATION.md` | Attention ~ g\|ψ\|² structural analogy (speculative) |
| `praxis/PRAXIS_AXIOMS.md` | Mathematical axioms (polar, polarization, Taylor, geometry) |

### Harmony and Coherence ★ NEW ★

| Document | Topic |
|----------|-------|
| `foundations/HARMONY_AND_COHERENCE.md` | Pythagorean comma, phase locking, coherence principle |

### Praxis and Heresy ★ NEW ★

| Document | Topic |
|----------|-------|
| `praxis/PRAXIS.md` | Running the architecture, doctrine in action |
| `praxis/FALSE_PROPHETS.md` | Heresies (aliasing, leakage, artifacts), inquisition |
| `information_theory/INFORMATION_BOUNDS_THE_BOUNDARY_OF_KNOWLEDGE_AND_KNOWING.md` | Information limits and their consequences |

### Roadmap

| Document | Topic |
|----------|-------|
| `AKIRA_PSYCHE_ROADMAP.md` | Complete research agenda, experiments, validation |

### POMDP Framework

| Document | Topic |
|----------|-------|
| `pomdp/POMDP_SIM.md` | POMDP formalization, pump cycle dynamics |
| `pomdp/POMDP_ATTENTION.md` | Attention as belief update |
| `pomdp/THE_OLD_LADY_AND_THE_TIGER.md` | Learning from demonstrations, distillation |

### Attention Mechanisms

| Document | Topic |
|----------|-------|
| `architecture_base/attention/ATTENTION_STACK.md` | Integration of all 3 attention mechanisms |
| `architecture_base/attention/spectral_attention/` | Bands 0-6 within-band attention |
| `architecture_base/attention/temporal_attention/` | Band 7 causal attention |
| `architecture_base/attention/spectral_wormhole/` | Cross-band communication |
| `architecture_expanded/wormhole/WORMHOLE_HYBRID.md` | Legacy hybrid implementation |

### Collapse Dynamics

| Document | Topic |
|----------|-------|
| `architecture_base/collapse/COLLAPSE_DYNAMICS.md` | Theory-mandated collapse physics |
| `architecture_base/collapse/COLLAPSE_GENERALIZATION.md` | From details to structure |
| `architecture_expanded/collapse/COLLAPSE_MECHANISMS.md` | Implementation approaches |

### Key Document Relationships

```
HOW THE MAJOR DOCUMENTS CONNECT:

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  DUALITY DOCUMENTS:                                                     │
│                                                                         │
│  DUALITY_AND_EFFICIENCY.md          DUALITY_METHODS.md                 │
│  (How to BUILD with dualities)  <-> (How to OBSERVE with dualities)    │
│           │                                   │                         │
│           └──────────── PANDORA ──────────────┘                         │
│                (Action as transformation)                               │
│                                                                         │
│  ═════════════════════════════════════════════════════════════════════  │
│                                                                         │
│  CONSERVATION DOCUMENTS:                                                │
│                                                                         │
│  CONSERVATION_OF_ACTION.md  <--->  EQUILIBRIUM_AND_CONSERVATION.md     │
│  (Fire analogy: what burns)        (What quantity is conserved?)       │
│           │                                   │                         │
│           └──── PANDORA_AFTERMATH ────────────┘                         │
│         (Hope: the generator NOT consumed by generating)               │
│                                                                         │
│  ═════════════════════════════════════════════════════════════════════  │
│                                                                         │
│  TERMINOLOGY CHAIN:                                                     │
│                                                                         │
│  TERMINOLOGY.md  --->  THE_ATOMIC_STRUCTURE_OF_INFORMATION.md          │
│  (Formal definitions)   (Action Quanta details)                        │
│         │                                                               │
│         v                                                               │
│  All other documents (consistent terminology)                          │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Summary

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│                           A K I R A                                    │
│     Adaptive Knowledge Integration via Resonant Attention              │
│                                                                         │
│  ═══════════════════════════════════════════════════════════════════   │
│                                                                         │
│  CORE HYPOTHESES (testable, falsifiable):                              │
│  1. Prediction is belief. Error is uncertainty. Training ≈ Bayes.    │
│  2. Attention ≈ g|ψ|² self-interaction (BEC analogy).               │
│  3. Collapse ≈ synergy → redundancy conversion (PID framework).      │
│                                                                         │
│  ═══════════════════════════════════════════════════════════════════   │
│                                                                         │
│  THE CENTRAL ANALOGY (testable):                                        │
│  Attention ≈ g|ψ|² (BEC self-interaction structure)                  │
│  Both have form: (function of self) × self                           │
│  IF this analogy holds, BEC phenomenology MAY apply:                 │
│  condensation, quasiparticles, superfluidity, phase transitions.     │
│  Experiments will confirm or refute these predictions.               │
│                                                                         │
│  ═══════════════════════════════════════════════════════════════════   │
│                                                                         │
│  ARCHITECTURE:                                                          │
│  7+1 bands (7 spectral + 1 temporal) × 3 attention types              │
│                                                                         │
│  ═══════════════════════════════════════════════════════════════════   │
│                                                                         │
│  DYNAMICS (hypothesized):                                               │
│  Pump cycle: Tension → Discharge → Recovery                            │
│  Collapse ≈ Condensation: Many hypotheses → one                       │
│  Collapse ≈ Synergy → Redundancy: Binding → Agreement                 │
│  Action Quanta (AQ) ≈ Quasiparticles: Collective excitations         │
│                                                                         │
│  ═══════════════════════════════════════════════════════════════════   │
│                                                                         │
│  PRAXIS:                                                                │
│  Doctrine: Nyquist, Shannon, Fourier, Parseval                        │
│  Heresy: Aliasing, spectral leakage, boundary artifacts              │
│  Inquisition: Experiments that expose heresies                        │
│                                                                         │
│  ═══════════════════════════════════════════════════════════════════   │
│                                                                         │
│  OBSERVABILITY:                                                         │
│  Use duality methods — transform to where observation is easy:        │
│  Spatial↔Frequency, Magnitude↔Phase, Forward↔Backward,               │
│  Sharp↔Soft, Local↔Global, Explicit↔Implicit, Energy↔Geometry        │
│                                                                         │
│  ═══════════════════════════════════════════════════════════════════   │
│                                                                         │
│  THE OLD LADY'S WISDOM:                                                 │
│  Don't invert. Distill. The atomic truth is reachable by forward      │
│  compression. The particulars were never needed for action.            │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Closing: The Return to Pandora

We began with a myth. Let us end with what the myth teaches.

When Pandora opened the box, she released all manner of troubles into the world. But one thing remained: Hope. Not optimism, not wishful thinking—but the capacity to generate futures. The pattern that produces instances without being consumed. The grammar that survives all its sentences.

AKIRA is an attempt to understand information transformation through this lens:

**What transforms?** Belief states—from distributed uncertainty to crystallized action.

**What is conserved?** The generative capacity—the ability to continue predicting, learning, acting.

**What emerges?** Action Quanta—the irreducible patterns that enable decision, the atoms of actionable thought.

We don't know if we're right. We have hypotheses, experiments, and a framework for investigation. Some predictions will be confirmed. Others will be falsified. That is the nature of inquiry.

But the questions themselves—*how does uncertainty become certainty? how do many become one? what is the structure of collapse?*—these questions seem worth asking. And the mathematics, when we look at it carefully, has the structure of something ancient and familiar: potential becoming actual, hidden becoming manifest, hope remaining when all else is released.

> *"Test everything. Falsify what's wrong. The hypotheses that survive experiments become knowledge. The capacity to keep generating hypotheses—that is hope."*

---

*Oscar Goldman — Shogu Research Group @ Datamutant.ai subsidiary of 温心重工業*

*If you use this repository in your research, please cite it. This is ongoing work—we would like to know your opinions and experiments.*

