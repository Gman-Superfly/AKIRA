# AKIRA Terminology: Formal Definitions

**Oscar Goldman — Shogu Research Group @ Datamutant.ai**

---

## Purpose

This document establishes formal terminology for AKIRA, aligned with established information theory literature (PID, ΦID) while introducing AKIRA-specific concepts with precise definitions.

---

## Table of Contents

1. [Core Distinction: PID Atoms vs Action Quanta](#1-core-distinction-pid-atoms-vs-action-quanta)
2. [Partial Information Decomposition (PID) Terms](#2-partial-information-decomposition-pid-terms)
3. [Information Dynamics: Extending PID to Time](#3-information-dynamics-extending-pid-to-time)
4. [Circuit Complexity Framework](#4-circuit-complexity-framework)
5. [AKIRA-Specific Terms](#5-akira-specific-terms)
   - [Critical Distinction: Synergy vs Modification vs Collapse](#critical-distinction-synergy-vs-modification-vs-collapse)
6. [The Superposition-Crystallized Duality](#6-the-superposition-crystallized-duality)
7. [The Action Quanta Framework](#7-the-action-quanta-framework)
8. [Terminology Mapping](#8-terminology-mapping)

---

## 1. Core Distinction: PID Atoms vs Action Quanta

```
CRITICAL TERMINOLOGY DISTINCTION

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  "INFORMATION ATOMS" (PID Literature)                                  │
│  ─────────────────────────────────────                                  │
│  Williams & Beer (2010) use "information atoms" to refer to the        │
│  nonnegative decomposition terms in PID:                               │
│                                                                         │
│  • Redundancy (I_red) - information shared by all sources             │
│  • Unique (I_uni) - information from one source only                  │
│  • Synergy (I_syn) - information emerging from combination            │
│                                                                         │
│  These are DECOMPOSITION TERMS - how information about a target       │
│  is distributed across sources.                                        │
│                                                                         │
│  ════════════════════════════════════════════════════════════════════  │
│                                                                         │
│  "ACTION QUANTA" (AQ) - AKIRA Framework                                │
│  ───────────────────────────────────────                                │
│  AKIRA introduces Action Quanta to refer to the irreducible units     │
│  of actionable information - emergent patterns in the belief field:   │
│                                                                         │
│  TERMINOLOGY NOTE:                                                      │
│    Singular = Action Quantum (ONE irreducible action unit)              │
│    Plural   = Action Quanta (MULTIPLE irreducible action units)         │
│    Combined = Composed Abstraction (AQ action units bonded together)   │
│                                                                         │
│  Bonded states are the STRUCTURAL implementation of                    │
│  FUNCTIONAL abstractions.                                              │
│                                                                         │
│  • Magnitude - signal strength, "how much"                            │
│  • Phase - position encoding, "where/when"                            │
│  • Frequency - scale of pattern, "what resolution"                    │
│  • Coherence - internal consistency, "how organized"                  │
│                                                                         │
│  THE INDUCTION CHAIN:                                                  │
│  ────────────────────                                                  │
│  1. Superposition of belief (wave-like, distributed)                  │
│  2. Collapse (phase transition)                                       │
│  3. AQ emerge (crystallized belief)                                   │
│  4. AQ guide correct action (their function)                          │
│                                                                         │
│  AQ ARE:                                                               │
│    WHAT: Crystallized belief (form belief takes after collapse)       │
│    WHY:  To enable correct action (their purpose)                     │
│    HOW:  Through collapse from superposition (the process)            │
│                                                                         │
│  These are EMERGENT PATTERNS - irreducible units that enable action.  │
│                                                                         │
│  ════════════════════════════════════════════════════════════════════  │
│                                                                         │
│  WHY "ACTION QUANTA"?                                                  │
│                                                                         │
│  1. PHYSICS: Planck's constant ℏ is THE quantum of action             │
│     - Action S = ∫L dt has units [energy × time]                      │
│     - ℏ is the minimum indivisible unit of action                     │
│     - All quantization derives from action quantization               │
│                                                                         │
│  2. AKIRA: Actionability is the defining criterion                    │
   │     - AQ = minimum pattern that enables correct decision              │
   │     - Cannot be reduced further without losing actionability          │
   │     - Representationally irreducible (minimum representation needed)  │
   │                                                                         │
   │     THE STRUCTURE-FUNCTION RELATIONSHIP:                              │
   │                                                                         │
   │       AQ (pattern) → enables DISCRIMINATION → enables ACTION          │
   │                                                                         │
   │       STRUCTURAL: AQ = Minimum PATTERN (what it IS)                   │
   │       FUNCTIONAL: Discrimination = ATOMIC ABSTRACTION (what it DOES)  │
   │                                                                         │
   │       AQ is NOT the abstraction. AQ ENABLES the abstraction.          │
   │       Discrimination IS the functional abstraction at atomic level.   │
   │                                                                         │
   │     REPRESENTATIONAL IRREDUCIBILITY:                                  │
│     The representation cannot be simplified further                   │
│     without losing the ability to act correctly.                      │
│     (Distinct from Wolfram's computational irreducibility             │
│     which concerns prediction, not representation for action.)        │
│                                                                         │
│  3. DISTINCTION: Avoids collision with PID "information atoms"        │
│     - Different concept, needs different name                         │
│     - Aligns with BEC quasiparticle interpretation                   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Partial Information Decomposition (PID) Terms

**Source:** Williams, P.L., & Beer, R.D. (2010). *Nonnegative Decomposition of Multivariate Information.* arXiv:1004.2515.

```
PID DECOMPOSITION

Given sources S₁, S₂ providing information about target T:

I(S₁, S₂ ; T) = I_red + I_uni(S₁) + I_uni(S₂) + I_syn

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  REDUNDANCY (I_red)                                                    │
│  ──────────────────                                                     │
│  Information that BOTH sources provide about T.                        │
│  Either source alone suffices to convey this information.             │
│  Sources AGREE - you can ask either one.                              │
│                                                                         │
│  UNIQUE INFORMATION (I_uni)                                            │
│  ──────────────────────────                                             │
│  Information that ONLY one source provides about T.                   │
│  Source-specific expertise that cannot be obtained elsewhere.         │
│                                                                         │
│  SYNERGY (I_syn)                                                       │
│  ───────────────                                                        │
│  Information that NEITHER source alone provides, but TOGETHER they do.│
│  Emergent information from combination.                               │
│  Sources must COOPERATE - you need both.                              │
│                                                                         │
│  "INFORMATION ATOMS" in PID = {I_red, I_uni(S₁), I_uni(S₂), I_syn}   │
│  These are the nonnegative decomposition terms.                       │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

**In AKIRA's spectral bands:**
- **High synergy state:** Need ALL bands to predict target (bands must cooperate)
- **High redundancy state:** ANY band can predict target (bands agree)
- The transition between these states is what AKIRA calls "collapse" (see Section 4)

---

## 3. Information Dynamics: Extending PID to Time

PID decomposes information at a single moment. **Information Dynamics** extends this to temporal processes.

```
TWO COMPLEMENTARY FRAMEWORKS - SAME PHENOMENA

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  ΦID (Mediano et al. 2025)                                             │
│  ─────────────────────────                                              │
│  Information-theoretic formalization.                                  │
│  Extends PID to many-to-many temporal dynamics.                       │
│  Provides: TDMI, transfer entropy, active storage, collective modes.  │
│                                                                         │
│  LIZIER et al. (2013)                                                  │
│  ────────────────────                                                   │
│  Computational interpretation.                                         │
│  Interprets information dynamics as distributed computation.          │
│  Provides: Storage/Transfer/Modification triad.                       │
│                                                                         │
│  ════════════════════════════════════════════════════════════════════  │
│                                                                         │
│  ★ KEY INSIGHT (Lizier et al.) ★                                      │
│                                                                         │
│            INFORMATION MODIFICATION = SYNERGY                          │
│                                                                         │
│  This is the bridge between PID and computation:                      │
│  - Synergy (PID term) = information requiring combination             │
│  - Modification (computational term) = non-trivial processing         │
│  - They are THE SAME THING measured different ways                    │
│                                                                         │
│  When two information streams interact to produce an outcome          │
│  that NEITHER could produce alone → that IS synergy → that IS        │
│  modification.                                                         │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### The Three Operations (Lizier's Triad)

```
STORAGE, TRANSFER, MODIFICATION

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  STORAGE                                                               │
│  ───────                                                                │
│  What it means:                                                        │
│  Information persisting within the same variable across time.         │
│  Like memory - the past of X tells you about the future of X.        │
│  A pattern that "remembers itself."                                   │
│                                                                         │
│  ΦID term: Active information storage                                 │
│  CA analog: Blinkers (stationary oscillators that persist)            │
│  AKIRA: Manifold structure, low-band persistence, learned weights     │
│                                                                         │
│  ────────────────────────────────────────────────────────────────────  │
│                                                                         │
│  TRANSFER                                                              │
│  ────────                                                               │
│  What it means:                                                        │
│  Information flowing from one variable to another across time.        │
│  The past of X tells you about the future of Y.                      │
│  A pattern that "moves" or "communicates."                            │
│                                                                         │
│  ΦID term: Transfer entropy (Schreiber, 2000)                         │
│  CA analog: Gliders (particles that move across space)                │
│  AKIRA: Wormhole attention, cross-band communication                  │
│                                                                         │
│  ────────────────────────────────────────────────────────────────────  │
│                                                                         │
│  MODIFICATION                                                          │
│  ────────────                                                           │
│  What it means:                                                        │
│  Non-trivial interaction between two or more information streams.     │
│  Neither X nor Y alone predicts the outcome - you need BOTH.         │
│  A pattern that emerges from COMBINATION.                             │
│                                                                         │
│  = SYNERGY (this is Lizier's key insight!)                            │
│  Modification IS synergy observed in dynamics.                        │
│                                                                         │
│  CA analog: Particle COLLISIONS (gliders interact, create new state)  │
│  AKIRA: When bands combine synergistically (see "collapse" below)     │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Additional ΦID Concepts

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  TDMI (Time-Delayed Mutual Information)                               │
│  I(X_t ; X_{t+1}) - TOTAL information from past to future            │
│  Decomposes into storage + transfer + modification components         │
│                                                                         │
│  COLLECTIVE MODES                                                      │
│  Many-to-many interactions beyond pairwise (ΦID's extension of PID)  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Circuit Complexity Framework

**Source:** Mao, J., Lozano-Perez, T., Tenenbaum, J.B., & Kaelbling, L.P. (2023). *What Planning Problems Can A Relational Neural Network Solve?* ICLR 2024. arXiv:2312.03682v2.

This framework provides formal bounds on what problems a fixed-architecture neural network can solve. It is the theoretical basis for AKIRA's 7+1 band architecture.

```
CIRCUIT COMPLEXITY TERMINOLOGY

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  SOS WIDTH (Strong Optimally-Serializable Width)                       │
│  ───────────────────────────────────────────────                        │
│  WHAT: The maximum number of constraints that must be tracked          │
│        simultaneously to solve a problem optimally.                    │
│                                                                         │
│  Formal: For a planning problem, SOS width k means:                    │
│  • At each step, at most k preconditions must be maintained            │
│  • These preconditions can be serialized (achieved one-by-one)         │
│  • The problem admits a goal regression strategy with width k          │
│                                                                         │
│  Lower k = easier problem (fewer simultaneous constraints)             │
│  Higher k = harder problem (more things to track at once)              │
│                                                                         │
│  Examples:                                                              │
│  • Blocks World: k = 2 (track block + destination)                    │
│  • Logistics: k = 2 (track package + location)                        │
│  • Sokoban: k = unbounded (push creates irreversible constraints)     │
│                                                                         │
│  AKIRA APPLICATION:                                                     │
│  For visual prediction, k ≈ 2-3 (object + neighbors + temporal)       │
│                                                                         │
│  ════════════════════════════════════════════════════════════════════  │
│                                                                         │
│  PREDICATE ARITY (β)                                                   │
│  ───────────────────                                                    │
│  WHAT: The number of arguments in the relations used to describe       │
│        the problem.                                                     │
│                                                                         │
│  Examples:                                                              │
│  • Unary predicate (β = 1): IsRed(x), IsMoving(x)                     │
│  • Binary predicate (β = 2): On(x,y), Near(x,y), Occluding(x,y)       │
│  • Ternary predicate (β = 3): Between(x,y,z)                          │
│                                                                         │
│  Most visual relations are binary (β = 2): spatial relations,         │
│  contact, occlusion, support, etc.                                     │
│                                                                         │
│  ════════════════════════════════════════════════════════════════════  │
│                                                                         │
│  CIRCUIT BREADTH (B)                                                   │
│  ───────────────────                                                    │
│  WHAT: The number of parallel processing channels in the network.      │
│        How many things can be tracked simultaneously.                  │
│                                                                         │
│  KEY THEOREM (Mao et al. 2023):                                        │
│  ┌───────────────────────────────────────────────────────────────┐     │
│  │  Required Breadth = (k + 1) × β                               │     │
│  │                                                                │     │
│  │  Where:                                                        │     │
│  │    k = SOS width of the problem                               │     │
│  │    β = predicate arity                                        │     │
│  │                                                                │     │
│  │  For AKIRA (visual prediction, k≈3, β=2):                     │     │
│  │    B = (3+1) × 2 = 8 bands                                    │     │
│  │                                                                │     │
│  │  This formally justifies the 7+1 architecture.                │     │
│  └───────────────────────────────────────────────────────────────┘     │
│                                                                         │
│  AKIRA APPLICATION:                                                     │
│  7 spectral bands + 1 temporal band = 8 parallel channels              │
│  This is sufficient for problems with SOS width ≤ 3                    │
│                                                                         │
│  ════════════════════════════════════════════════════════════════════  │
│                                                                         │
│  CIRCUIT DEPTH (D)                                                     │
│  ─────────────────                                                      │
│  WHAT: The number of sequential processing steps.                      │
│        How many layers of computation.                                 │
│                                                                         │
│  Relationship to problem size:                                         │
│  • Constant depth: Problem size doesn't affect layers needed           │
│  • Logarithmic depth: Layers grow as log(n)                           │
│  • Linear depth: Layers grow as O(n)                                  │
│                                                                         │
│  AKIRA APPLICATION:                                                     │
│  Depth corresponds to number of SBM layers. For bounded-width          │
│  problems, bounded depth suffices.                                     │
│                                                                         │
│  ════════════════════════════════════════════════════════════════════  │
│                                                                         │
│  S-GRS (Serialized Goal Regression Search)                             │
│  ─────────────────────────────────────────                              │
│  WHAT: The optimal strategy for solving problems with bounded          │
│        SOS width. Achieve preconditions one-by-one, maintaining        │
│        already-achieved ones.                                          │
│                                                                         │
│  The serialization order matters:                                      │
│  1. Achieve precondition p₁                                           │
│  2. Achieve p₂ while maintaining p₁                                   │
│  3. Achieve p₃ while maintaining p₁, p₂                               │
│  ... and so on                                                         │
│                                                                         │
│  AKIRA APPLICATION:                                                     │
│  Collapse may follow S-GRS: Lower bands (structure) collapse first,    │
│  higher bands collapse while maintaining lower band consistency.       │
│  This is a testable hypothesis (see EVIDENCE_TO_COLLECT.md).          │
│                                                                         │
│  ════════════════════════════════════════════════════════════════════  │
│                                                                         │
│  PROBLEM CLASSES                                                        │
│  ───────────────                                                        │
│  Based on circuit requirements:                                        │
│                                                                         │
│  CLASS 1: Constant breadth, constant depth                             │
│  • Easy for fixed architecture                                         │
│  • Example: Single object tracking                                     │
│                                                                         │
│  CLASS 2: Constant breadth, unbounded depth                            │
│  • Tractable for fixed architecture (given enough layers)              │
│  • Example: Blocks World, Logistics                                    │
│                                                                         │
│  CLASS 3: Unbounded breadth                                            │
│  • Intractable for fixed architecture                                  │
│  • Example: Sokoban (creates arbitrary many constraints)               │
│                                                                         │
│  AKIRA is designed for Class 1 and Class 2 problems.                   │
│  Class 3 problems will cause systematic failure.                       │
│                                                                         │
│  ════════════════════════════════════════════════════════════════════  │
│                                                                         │
│  RelNN[D,B] (Relational Neural Network)                                │
│  ──────────────────────────────────────                                 │
│  WHAT: A neural network with depth D and breadth B that operates       │
│        on relational (graph-structured) inputs.                        │
│                                                                         │
│  Mao et al. prove that RelNN[D,B] can solve exactly those problems     │
│  whose SOS width k satisfies: B ≥ (k+1) × β                           │
│                                                                         │
│  AKIRA as RelNN:                                                        │
│  AKIRA's spectral bands form a RelNN where:                            │
│  • B = 8 (the bands)                                                   │
│  • D = number of SBM layers                                            │
│  • Input = spectral decomposition of visual signal                     │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Why This Matters for AKIRA

```
CIRCUIT COMPLEXITY IMPLICATIONS

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  1. FORMAL JUSTIFICATION FOR 7+1                                       │
│     ────────────────────────────                                        │
│     The number 8 is not arbitrary. For visual prediction tasks with    │
│     SOS width k ≈ 3 and binary relations (β = 2):                      │
│                                                                         │
│       Required breadth = (3+1) × 2 = 8 bands                           │
│                                                                         │
│     The +1 (temporal band) is separately justified by time-frequency   │
│     uncertainty. The 7 spectral bands provide the remaining breadth.   │
│                                                                         │
│  2. PREDICTABLE FAILURE MODES                                          │
│     ─────────────────────────                                           │
│     AKIRA will fail on problems with SOS width > 3.                    │
│     This is not a bug but a theoretical limitation.                    │
│     Failure will be systematic: constraint violations, not random.     │
│                                                                         │
│  3. COLLAPSE AS SERIALIZATION                                          │
│     ──────────────────────────                                          │
│     S-GRS suggests collapse should proceed band-by-band,               │
│     maintaining already-collapsed constraints.                         │
│     This is testable (see EVIDENCE_TO_COLLECT.md, hypothesis 2.1).    │
│                                                                         │
│  4. DOMAIN BOUNDARIES                                                   │
│     ─────────────────                                                   │
│     AKIRA is suited for:                                                │
│     • Local visual prediction (k ≈ 2-3)                                │
│     • Object tracking (k ≈ 2)                                          │
│     • Motion prediction (k ≈ 2-3)                                      │
│                                                                         │
│     AKIRA is NOT suited for:                                            │
│     • Global coordination problems (k unbounded)                       │
│     • Sokoban-style puzzles (k grows with problem)                    │
│     • Problems requiring arbitrary constraint tracking                 │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 5. AKIRA-Specific Terms

### Critical Distinction: Synergy vs Modification vs Collapse

```
THESE ARE NOT THE SAME THING - USE CORRECTLY

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  SYNERGY (PID term)                                                    │
│  ──────────────────                                                     │
│  WHAT: A TYPE of information                                           │
│  Definition: Information requiring combination of sources              │
│  Use when: Describing information decomposition, measuring I_syn       │
│                                                                         │
│  MODIFICATION (Lizier term)                                            │
│  ──────────────────────────                                             │
│  WHAT: A computational EVENT                                           │
│  Definition: Synergy observed in temporal dynamics                     │
│  Use when: Describing distributed computation, CA collisions          │
│  Relationship: Modification = Synergy (same thing, different view)    │
│                                                                         │
│  COLLAPSE (AKIRA term) - one direction of phase transition            │
│  ─────────────────────────────────────────────────────────              │
│  WHAT: A PROCESS (not a state, not an information type)               │
│  Direction: Synergy → Redundancy                                       │
│  What happens: Uncertainty resolves, AQ crystallize                   │
│  Character: Sudden, dramatic (like freezing)                          │
│                                                                         │
│  TENSION ACCUMULATION (AKIRA term) - the other direction                           │
│  ──────────────────────────────────────────                             │
│  WHAT: A PROCESS (the reverse of collapse)                            │
│  Direction: Redundancy → Synergy                                       │
│  What happens: New uncertainty accumulates, hypotheses multiply       │
│  Character: Gradual (uncertainty builds over time)                    │
│  Brief form: "tension" in diagrams                                    │
│                                                                         │
│  ════════════════════════════════════════════════════════════════════  │
│                                                                         │
│  THE PUMP CYCLE (both directions):                                     │
│                                                                         │
│       TENSION                    COLLAPSE                              │
│    (uncertainty builds)      (uncertainty resolves)                   │
│                                                                         │
│    Redundancy ──────────→ Synergy ──────────→ Redundancy              │
│         ↑                                          │                   │
│         └──────────────────────────────────────────┘                   │
│                      (cycle repeats)                                   │
│                                                                         │
│  ════════════════════════════════════════════════════════════════════  │
│                                                                         │
│  SUMMARY:                                                               │
│  • Synergy = what you HAVE (information type)                         │
│  • Modification = synergy in ACTION (computational event)             │
│  • Collapse = synergy CHANGING to redundancy (transition process)     │
│                                                                         │
│  Before collapse: High synergy (bands cooperate)                      │
│  During collapse: Modification events occur (synergy in action)       │
│  After collapse: High redundancy (bands agree), AQ crystallized       │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### AKIRA Terms

```
AKIRA TERMINOLOGY

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  ACTION QUANTA (AQ)                                                    │
│  ──────────────────                                                     │
│  The irreducible unit of actionable information.                      │
│  A structured pattern that enables correct decision/prediction.       │
│                                                                         │
│  Properties:                                                           │
│  • Magnitude: Signal strength, "how much"                             │
│  • Phase: Position encoding, "where/when"                             │
│  • Frequency: Scale of pattern, "what resolution"                     │
│  • Coherence: Internal consistency, "how organized"                   │
│                                                                         │
│  Defining criteria:                                                    │
│  • IRREDUCIBLE: Cannot decompose further without losing actionability│
│  • ACTIONABLE: Enables correct decision for the task                  │
│  • STRUCTURED: Has internal organization (not a point)                │
│  • COMBINABLE: Forms bonded states (concepts) via phase alignment     │
│  • CONSERVED: Persists through valid transformations                  │
│                                                                         │
│  Physical analogy:                                                     │
│  AQ are QUASIPARTICLES of the belief field.                           │
│  Like phonons in crystals or Bogoliubov excitations in BEC.          │
│  Emergent collective excitations, not fundamental particles.          │
│                                                                         │
│  ════════════════════════════════════════════════════════════════════  │
│                                                                         │
│  COLLAPSE (a PROCESS, not a state)                                    │
│  ─────────────────────────────────                                      │
│  The PROCESS of transitioning from uncertainty to certainty.          │
│  What happens: Synergy converts to Redundancy.                        │
│  Result: Multiple hypotheses → Single committed prediction.           │
│  During collapse: AQ crystallize (emerge as stable patterns).         │
│  NOTE: Collapse is an event/process that OCCURS, not a state you're in│
│                                                                         │
│  ════════════════════════════════════════════════════════════════════  │
│                                                                         │
│  FORMAL DEFINITION (Bayesian Statistics)                               │
│  ───────────────────────────────────────                                │
│                                                                         │
│  COLLAPSE = POSTERIOR CONTRACTION                                      │
│  ────────────────────────────────────                                   │
│  The posterior distribution contracts around the true state as data   │
│  increases. Covariance → 0, belief → Dirac delta.                     │
│                                                                         │
│  Mathematical form (POMDP):                                            │
│  b(s) → δ(s - s*)     Belief concentrates on one state                │
│  H(b) → 0             Entropy decreases to minimum                    │
│                                                                         │
│  This is established Bayesian statistics (van der Vaart & van Zanten).│
│  Contraction rates quantify how quickly this happens.                 │
│                                                                         │
│  ────────────────────────────────────────────────────────────────────  │
│                                                                         │
│  TENSION = PROPAGATION / BELIEF EXPANSION                              │
│  ────────────────────────────────────────────                           │
│  Covariance expands due to process noise or model dynamics.           │
│  Belief diffuses over possible future states.                         │
│                                                                         │
│  Mathematical form (POMDP):                                            │
│  δ(s - s*) → b(s)     Belief spreads across states                    │
│  H(b) → H_max         Entropy increases toward maximum                │
│                                                                         │
│  Also known as: covariance inflation, belief expansion (Kalman filter │
│  literature, ensemble methods).                                        │
│                                                                         │
│  ────────────────────────────────────────────────────────────────────  │
│                                                                         │
│  AKIRA SHORTHAND:                                                       │
│  • "Collapse" = Posterior contraction                                 │
│  • "Tension" = Propagation / belief expansion                         │
│                                                                         │
│  These terms capture the intuition while grounded in Bayesian theory. │
│                                                                         │
│  ════════════════════════════════════════════════════════════════════  │
│                                                                         │
│  INTERPRETATIONS (how collapse/tension manifest in different contexts)│
│  ─────────────────────────────────────────────────────────────────────  │
│                                                                         │
│  These are NOT separate definitions — they are how the single Bayesian│
│  phenomenon appears when viewed through different lenses:             │
│                                                                         │
│  PID (Information Theory):                                             │
│  • Collapse: Synergy → Redundancy (information becomes shared)        │
│  • Tension: Redundancy → Synergy (information becomes distributed)    │
│  See: experiments/025_EXP_SYNERGY_REDUNDANCY_TRANSITION.md            │
│                                                                         │
│  Operational (What We Observe):                                        │
│  • Collapse: Attention concentrating, entropy dropping                │
│  • Tension: Attention spreading, entropy rising                       │
│  See: architecture_base/collapse/COLLAPSE_DYNAMICS.md §4              │
│                                                                         │
│  Geometric (Manifold):                                                 │
│  • Collapse: Converging to a point (commitment)                       │
│  • Tension: Diverging from a point (exploration)                      │
│  See: praxis/PRAXIS_AXIOMS.md §5                                      │
│                                                                         │
│  ────────────────────────────────────────────────────────────────────  │
│                                                                         │
│  Use whichever interpretation fits your analysis context:             │
│  • PID for information-theoretic analysis                             │
│  • Operational for empirical measurement                              │
│  • Geometric for manifold/embedding analysis                          │
│                                                                         │
│  NOTE: BEC physics provides a useful ANALOGY for collapse dynamics.   │
│  See bec_analogy/BEC_CONDENSATION_INFORMATION.md for that framework. │
│                                                                         │
│  ════════════════════════════════════════════════════════════════════  │
│                                                                         │
│  BELIEF FIELD                                                          │
│  ────────────                                                           │
│  The manifold of possible predictions/beliefs.                        │
│  AQ are excitations of this field.                                    │
│  Attention = g|ψ|² interaction within the field.                     │
│                                                                         │
│  ════════════════════════════════════════════════════════════════════  │
│                                                                         │
│  SPECTRAL BANDS                                                        │
│  ──────────────                                                         │
│  7+1 frequency bands (7 spatial + 1 temporal).                        │
│  Different bands carry different PID atom types:                      │
│  • Low bands: More redundancy (stable structure)                      │
│  • High bands: More unique (specific details)                         │
│  • Cross-band: Synergy (requires combination)                         │
│                                                                         │
│  ════════════════════════════════════════════════════════════════════  │
│                                                                         │
│  WORMHOLE ATTENTION                                                    │
│  ──────────────────                                                     │
│  Cross-band communication mechanism.                                  │
│  Enables synergistic information flow.                                │
│  Realizes the SYNERGY between complementary bands.                    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 6. The Superposition-Crystallized Duality

**Source:** AKIRA framework, with parallel to quantum mechanical wave-particle duality.

This duality captures the two phases of AKIRA's belief cycle. Both framings are needed; they describe different phases of the same process.

**Terminology Note:** We use "crystallized" rather than "molecule" for the post-collapse state because:
- Crystallized emphasizes IRREDUCIBILITY (cannot be reduced further without losing actionability)
- Crystallization is a PHASE TRANSITION (matches collapse dynamics)
- Crystal structures are STABLE and LOCKED (matches post-collapse properties)
- The term "bonded state" is reserved for when multiple crystallized AQ combine (tentative)

```
SUPERPOSITION-CRYSTALLIZED DUALITY

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  SUPERPOSITION (Pre-Collapse Phase)                                    │
│  ──────────────────────────────────                                     │
│  WHAT: The state of belief BEFORE collapse.                            │
│        Multiple possibilities coexist, interfere, compete.             │
│                                                                         │
│  Characteristics:                                                       │
│  • Belief distributed across possibilities                             │
│  • Multiple AQ activated but not committed                             │
│  • Interference patterns visible in error/uncertainty maps            │
│  • High entropy, high synergy                                          │
│  • Phase relationships fluid (not locked)                              │
│                                                                         │
│  Physical analog: Wave function before measurement                     │
│  Mathematical form: Σᵢ cᵢ |ψᵢ⟩ (linear combination of states)        │
│                                                                         │
│  USE "SUPERPOSITION" WHEN DISCUSSING:                                  │
│  • Pre-collapse dynamics                                               │
│  • Error fringes and interference                                      │
│  • Uncertainty handling                                                │
│  • Belief distribution analysis                                        │
│  • Temperature/entropy concepts                                        │
│                                                                         │
│  ════════════════════════════════════════════════════════════════════  │
│                                                                         │
│  CRYSTALLIZED (Post-Collapse Phase)                                    │
│  ──────────────────────────────────                                     │
│  WHAT: The state of belief AFTER collapse.                             │
│        AQ have crystallized into stable, IRREDUCIBLE configurations.   │
│                                                                         │
│  Characteristics:                                                       │
│  • Belief concentrated in specific pattern                             │
│  • AQ crystallized into irreducible configuration                      │
│  • Fringes resolved to definite structure                             │
│  • Low entropy, high redundancy                                        │
│  • Phase relationships locked (coherent bonds)                         │
│  • IRREDUCIBLE: Cannot be decomposed further without losing action     │
│                                                                         │
│  Physical analog: Definite eigenstate after measurement                │
│  Mathematical form: |ψ_final⟩ (single state)                          │
│                                                                         │
│  USE "CRYSTALLIZED" WHEN DISCUSSING:                                   │
│  • Post-collapse structures                                            │
│  • Irreducible AQ configurations                                       │
│  • Stable representations                                              │
│  • Phase-locked patterns                                               │
│  • The output of collapse events                                       │
│                                                                         │
│  ════════════════════════════════════════════════════════════════════  │
│                                                                         │
│  WHY BOTH ARE NEEDED                                                   │
│  ───────────────────                                                    │
│  • Superposition alone: Cannot explain stable outputs                 │
│  • Crystallized alone: Cannot explain uncertainty handling            │
│  • Together: Complete picture of belief dynamics                      │
│                                                                         │
│  This is STRUCTURAL PARALLEL to wave-particle duality, not metaphor:  │
│  • Both involve superposition of basis states                         │
│  • Both collapse to definite configuration                            │
│  • Both preserve information (unitary/Parseval)                       │
│  • Both have interference before collapse                             │
│                                                                         │
│  ════════════════════════════════════════════════════════════════════  │
│                                                                         │
│  THE BELIEF CYCLE                                                      │
│  ────────────────                                                       │
│                                                                         │
│  Superposition → Collapse → Crystallized → (Prediction/Action)        │
│       ↑                                        │                       │
│       └────────── New input ──────────────────┘                       │
│                                                                         │
│  1. Input creates distributed activation (superposition)              │
│  2. Tension builds as patterns compete                                │
│  3. Collapse selects winning configuration                            │
│  4. AQ crystallize into stable, irreducible form                      │
│  5. Crystallized AQ enable prediction/action                          │
│  6. New input restarts cycle                                          │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 7. The Action Quanta Framework

### 5.1 What AQ Are

```
ACTION QUANTA: THE COMPLETE PICTURE

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  AQ are the PRODUCT of synergy→redundancy collapse.                   │
│                                                                         │
│  BEFORE COLLAPSE:                                                       │
│  ────────────────                                                       │
│  • Information is distributed (synergistic)                           │
│  • Need ALL bands to predict target                                   │
│  • Belief field is "hot" (many hypotheses)                            │
│  • No crystallized AQ - just potential                                │
│                                                                         │
│  DURING COLLAPSE:                                                       │
│  ───────────────                                                        │
│  • Synergy converts to redundancy                                     │
│  • Information modification occurs (Lizier's term)                    │
│  • Belief field "cools" (hypotheses merge)                            │
│  • AQ BEGIN to crystallize                                            │
│                                                                         │
│  AFTER COLLAPSE:                                                        │
│  ───────────────                                                        │
│  • Information is concentrated (redundant)                            │
│  • ANY band can predict target                                        │
│  • Belief field is "cold" (single hypothesis)                         │
│  • AQ are CRYSTALLIZED - stable, actionable patterns                  │
│                                                                         │
│  ════════════════════════════════════════════════════════════════════  │
│                                                                         │
│  THE PHYSICS PARALLEL:                                                  │
│                                                                         │
│  Physical action S = ∫L dt                                            │
│  Quantum of action = ℏ (Planck's constant)                            │
│                                                                         │
│  When S >> ℏ: Classical behavior (continuous, many paths)             │
│  When S ~ ℏ:  Quantum behavior (discrete, specific paths)             │
│                                                                         │
│  Information "action" = belief-weighted prediction                    │
│  Quantum of action = AQ (minimum actionable unit)                     │
│                                                                         │
│  When uncertainty >> AQ: Classical belief (spread, many hypotheses)   │
│  When uncertainty ~ AQ:  Quantum belief (collapsed, specific answer)  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 5.2 AQ Properties (Detailed)

```
AQ INTERNAL STRUCTURE

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│                        ACTION QUANTUM                                  │
│                                                                         │
│              ┌──────────────────────────────┐                          │
│              │                              │                          │
│              │   MAGNITUDE                  │ → How much signal        │
│              │   └─ Determines salience     │                          │
│              │   └─ Higher = more dominant  │                          │
│              │                              │                          │
│              │   PHASE                      │ → Where/when             │
│              │   └─ Position encoding       │                          │
│              │   └─ Determines interference │                          │
│              │                              │                          │
│              │   FREQUENCY                  │ → What scale             │
│              │   └─ Band location           │                          │
│              │   └─ Resolution level        │                          │
│              │                              │                          │
│              │   COHERENCE                  │ → How organized          │
│              │   └─ Internal consistency    │                          │
│              │   └─ Determines bondability  │                          │
│              │                              │                          │
│              │   + INTERNAL PATTERN         │ → The structure itself   │
│              │                              │                          │
│              └──────────────────────────────┘                          │
│                                                                         │
│  These properties determine:                                           │
│  • What action the AQ enables                                         │
│  • How it combines with other AQ (forms bonded states)                │
│  • How it transforms under operations                                 │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 5.3 AQ Bonds (Bonded State Formation)

```
HOW AQ FORM BONDED STATES (Concepts)

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  COHERENT BONDS (phase alignment)                                      │
│  ────────────────────────────────                                       │
│  AQ at same frequency, aligned phase → constructive interference      │
│  "These edge AQ form a contour"                                       │
│                                                                         │
│  COMPLEMENTARY BONDS (magnitude exchange)                              │
│  ─────────────────────────────────────────                              │
│  AQ at different frequencies filling each other's gaps                │
│  "Low-freq identity + high-freq position = located object"            │
│                                                                         │
│  HIERARCHICAL BONDS (frequency bridging)                               │
│  ────────────────────────────────────────                               │
│  Coarse AQ contextualizes fine AQ                                     │
│  "Face (low-freq) contains eye (high-freq)"                           │
│                                                                         │
│  BONDED STATES = Combined crystallized AQ configurations              │
│  "Ring" = Blob_AQ + Boundary_AQ + Edge_AQ (circular)                  │
│                                                                         │
│  Note: Each individual AQ is irreducible after crystallization.       │
│  The bonded state is the combination of multiple crystallized AQ.     │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 8. Terminology Mapping

### Complete Cross-Reference

| Concept | PID/ΦID Term | Dynamics Term | Circuit Complexity Term | AKIRA Usage |
|---------|--------------|---------------|-------------------------|-------------|
| Information requiring combination | Synergy (I_syn) | Modification | - | Use "synergy" |
| Information sources share | Redundancy (I_red) | - | - | Use "redundancy" |
| Information only one has | Unique (I_uni) | - | - | Use "unique" |
| Decomposition terms | Information atoms | - | - | Say "PID atoms" |
| Within-variable persistence | - | Storage | - | Manifold/weights |
| Cross-variable flow | - | Transfer | - | Wormhole attention |
| Synergy→Redundancy transition | - | Collision events | - | **Collapse** = Posterior contraction |
| Redundancy→Synergy transition | - | - | - | **Tension** = Propagation/belief expansion |
| Emergent actionable patterns | - | - | - | **Action Quanta (AQ)** |
| Simultaneous constraint limit | - | - | SOS Width (k) | Problem complexity measure |
| Parallel processing channels | - | - | Circuit Breadth (B) | 7+1 = 8 bands |
| Sequential processing steps | - | - | Circuit Depth (D) | Number of SBM layers |
| Relation argument count | - | - | Predicate Arity (β) | Usually β = 2 for visual |
| Optimal constraint ordering | - | - | S-GRS | Collapse serialization |
| Pre-collapse distributed state | - | - | - | **Superposition** |
| Post-collapse stable state | - | - | - | **Crystallized** (irreducible) |

### Quick Reference

```
WHEN TO USE WHICH TERM

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  ═══════════════════════════════════════════════════════════════════   │
│  INFORMATION THEORY TERMS (PID/ΦID)                                    │
│  ═══════════════════════════════════════════════════════════════════   │
│                                                                         │
│  Use "SYNERGY" when discussing:                                        │
│  • Information TYPE that requires combination                         │
│  • PID decomposition measurements (I_syn)                             │
│  • Cross-band information that needs cooperation                      │
│  • The STATE before collapse (high synergy state)                     │
│                                                                         │
│  Use "MODIFICATION" when discussing:                                   │
│  • Computational EVENTS where streams interact                        │
│  • Lizier's distributed computation framework                         │
│  • Cellular automata collisions                                       │
│  • Synergy happening in time (modification = synergy in dynamics)     │
│                                                                         │
│  Use "REDUNDANCY" when discussing:                                     │
│  • Information TYPE that sources share                                │
│  • The STATE after collapse (high redundancy state)                   │
│  • When bands AGREE (either can predict)                              │
│                                                                         │
│  ═══════════════════════════════════════════════════════════════════   │
│  CIRCUIT COMPLEXITY TERMS (Mao et al.)                                 │
│  ═══════════════════════════════════════════════════════════════════   │
│                                                                         │
│  Use "SOS WIDTH" (k) when discussing:                                  │
│  • Problem complexity (how many constraints to track)                 │
│  • Architecture requirements (what breadth is needed)                 │
│  • Tractability boundaries (which problems AKIRA can solve)           │
│  • The FORMAL measure of problem difficulty                           │
│                                                                         │
│  Use "CIRCUIT BREADTH" (B) when discussing:                            │
│  • Architecture capacity (how many parallel channels)                 │
│  • The 7+1 = 8 band count                                             │
│  • Required breadth = (k+1) × β theorem                              │
│                                                                         │
│  Use "PREDICATE ARITY" (β) when discussing:                           │
│  • Relation structure (how many arguments)                            │
│  • Visual relations are typically β = 2 (binary)                     │
│  • Input to breadth calculation                                       │
│                                                                         │
│  Use "S-GRS" when discussing:                                          │
│  • Optimal constraint ordering                                        │
│  • How collapse might serialize (hypothesis)                          │
│  • Goal regression strategies                                         │
│                                                                         │
│  ═══════════════════════════════════════════════════════════════════   │
│  AKIRA PROCESS TERMS                                                   │
│  ═══════════════════════════════════════════════════════════════════   │
│                                                                         │
│  Use "COLLAPSE" (= Posterior Contraction) when discussing:            │
│  • The PROCESS of uncertainty resolving to certainty                 │
│  • Bayesian posterior contracting around true state                  │
│  • Synergy→redundancy conversion (PID interpretation)               │
│  • AQ crystallization (what happens DURING collapse)                  │
│  • Covariance shrinking, entropy dropping                            │
│                                                                         │
│  Use "TENSION" (= Propagation/Belief Expansion) when discussing:      │
│  • The PROCESS of certainty dissolving to uncertainty                │
│  • Covariance expanding due to process noise/dynamics                │
│  • Redundancy→synergy conversion (PID interpretation)               │
│  • Uncertainty building up over time                                  │
│  • The phase before collapse (tension accumulates until collapse)    │
│                                                                         │
│  ═══════════════════════════════════════════════════════════════════   │
│  AKIRA STATE TERMS (Superposition-Crystallized Duality)               │
│  ═══════════════════════════════════════════════════════════════════   │
│                                                                         │
│  Use "SUPERPOSITION" when discussing:                                  │
│  • Pre-collapse belief state (distributed, uncertain)                │
│  • Multiple competing hypotheses                                      │
│  • Interference patterns in error maps                                │
│  • Wave-like behavior of belief                                       │
│                                                                         │
│  Use "CRYSTALLIZED" when discussing:                                   │
│  • Post-collapse state (concentrated, certain, IRREDUCIBLE)          │
│  • Stable AQ configurations that cannot be further reduced           │
│  • How AQ stabilize after collapse                                    │
│  • Particle-like behavior of belief                                   │
│                                                                         │
│  NOTE: Both framings describe the SAME belief - different phases.     │
│  Like wave-particle duality: both are always true, different aspects │
│  dominate at different times.                                         │
│                                                                         │
│  ═══════════════════════════════════════════════════════════════════   │
│  AKIRA OUTPUT TERMS                                                    │
│  ═══════════════════════════════════════════════════════════════════   │
│                                                                         │
│  Use "ACTION QUANTA (AQ)" when discussing:                            │
│  • Irreducible actionable patterns                                    │
│  • Pattern properties (magnitude, phase, frequency, coherence)        │
│  • Bonded states and concept formation                                │
│  • What EMERGES from collapse (crystallized patterns)                 │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Summary

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  AKIRA TERMINOLOGY FRAMEWORK                                           │
│                                                                         │
│  ═══════════════════════════════════════════════════════════════════   │
│                                                                         │
│  LAYER 1: PID (Williams & Beer 2010) - STATIC DECOMPOSITION           │
│  ─────────────────────────────────────────────────────────             │
│  • Redundancy, Unique, Synergy                                        │
│  • "Information atoms" = R/U/S decomposition terms                    │
│  • HOW information is distributed across sources                      │
│                                                                         │
│  LAYER 2: INFORMATION DYNAMICS (ΦID + Lizier) - TEMPORAL              │
│  ─────────────────────────────────────────────────────────             │
│  • Storage, Transfer, Modification                                    │
│  • KEY: Modification = Synergy                                        │
│  • HOW information flows and interacts over time                      │
│                                                                         │
│  LAYER 3: CIRCUIT COMPLEXITY (Mao et al. 2023) - TRACTABILITY         │
│  ─────────────────────────────────────────────────────────             │
│  • SOS Width (k), Circuit Breadth (B), Predicate Arity (β)           │
│  • KEY: Required Breadth = (k+1) × β                                 │
│  • WHAT problems are tractable for fixed architecture                │
│  • Formal justification: 8 bands for k ≤ 3, β = 2                    │
│                                                                         │
│  LAYER 4: AKIRA - EMERGENT STRUCTURE                                   │
│  ─────────────────────────────────────                                  │
│  • Action Quanta (AQ) = Irreducible actionable patterns              │
│  • Collapse = Synergy → Redundancy transition                        │
│  • Superposition/Crystallized duality = Pre/post collapse phases     │
│  • WHAT emerges as actionable from the dynamics                       │
│                                                                         │
│  ═══════════════════════════════════════════════════════════════════   │
│                                                                         │
│  THE KEY DISTINCTIONS:                                                  │
│                                                                         │
│  Synergy            = Information TYPE (requires combination)         │
│  Modification       = Synergy in ACTION (computational event)         │
│  SOS Width          = Problem COMPLEXITY (constraints to track)       │
│  Circuit Breadth    = Architecture CAPACITY (parallel channels)       │
│  Collapse           = The PROCESS where synergy→redundancy           │
│  Superposition      = The STATE before collapse (distributed)         │
│  Crystallized       = The STATE after collapse (IRREDUCIBLE)         │
│  Action Quanta (AQ) = What EMERGES from collapse (patterns)          │
│                                                                         │
│  ═══════════════════════════════════════════════════════════════════   │
│                                                                         │
│  FORMAL BAYESIAN GROUNDING:                                            │
│                                                                         │
│  COLLAPSE = POSTERIOR CONTRACTION (Bayesian statistics)               │
│  • Posterior contracts around true state as data increases           │
│  • b(s) → δ(s - s*), H(b) → 0, covariance → 0                       │
│                                                                         │
│  TENSION = PROPAGATION / BELIEF EXPANSION (Kalman filter)             │
│  • Covariance expands due to process noise or dynamics               │
│  • δ(s - s*) → b(s), H(b) → H_max, covariance grows                 │
│                                                                         │
│  AKIRA uses "collapse" and "tension" as intuitive shorthand.         │
│                                                                         │
│  ═══════════════════════════════════════════════════════════════════   │
│                                                                         │
│  SUPERPOSITION-CRYSTALLIZED DUALITY:                                   │
│                                                                         │
│  SUPERPOSITION (pre-collapse)     CRYSTALLIZED (post-collapse)        │
│  • Distributed belief             • Concentrated belief               │
│  • High synergy                   • High redundancy                   │
│  • Interference patterns          • Stable, IRREDUCIBLE structure    │
│  • Wave-like                      • Particle-like                     │
│  • Reducible                      • IRREDUCIBLE                       │
│                                                                         │
│  Both describe same belief - different phases of same cycle.          │
│  Structural parallel to wave-particle duality in physics.             │
│                                                                         │
│  ═══════════════════════════════════════════════════════════════════   │
│                                                                         │
│  The two processes (phase transitions):                               │
│  • TENSION:  Redundancy → Synergy (uncertainty accumulates)           │
│  • COLLAPSE: Synergy → Redundancy (uncertainty resolves, AQ emerge)   │
│                                                                         │
│  The pump cycle:                                                       │
│  [Redundancy] --TENSION--> [Synergy] --COLLAPSE--> [Redundancy] + AQ  │
│                                                                         │
│  ═══════════════════════════════════════════════════════════════════   │
│                                                                         │
│  ARCHITECTURE JUSTIFICATION (Circuit Complexity):                      │
│                                                                         │
│  For visual prediction with k ≈ 3, β = 2:                             │
│    Required Breadth = (k+1) × β = (3+1) × 2 = 8 bands                │
│                                                                         │
│  7 spectral bands + 1 temporal band = 8 parallel channels             │
│  This is DERIVED, not arbitrary.                                       │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## References

### Information Theory (PID/ΦID)

1. Williams, P.L., & Beer, R.D. (2010). *Nonnegative Decomposition of Multivariate Information.* arXiv:1004.2515.

2. Mediano, P.A.M., et al. (2025). *Toward a unified taxonomy of information dynamics via Integrated Information Decomposition.* PNAS 122(39).

3. Lizier, J.T., Flecker, B., & Williams, P.L. (2013). *Towards a Synergy-based Approach to Measuring Information Modification.* arXiv:1303.3440.

4. Griffith, V., & Koch, C. (2014). *Quantifying Synergistic Mutual Information.* arXiv:1205.4265.

5. Sparacino, L., et al. (2025). *Partial Information Rate Decomposition.* Physical Review Letters, 135, 187401.

### Circuit Complexity

6. Mao, J., Lozano-Perez, T., Tenenbaum, J.B., & Kaelbling, L.P. (2023). *What Planning Problems Can A Relational Neural Network Solve?* ICLR 2024. arXiv:2312.03682v2. https://arxiv.org/html/2312.03682v2 — Foundational work on SOS width and circuit complexity for planning; provides the theoretical basis for AKIRA's 7+1 architecture.

### Bayesian Statistics (Posterior Contraction)

6. van der Vaart, A.W., & van Zanten, J.H. (2008). *Rates of contraction of posterior distributions based on Gaussian process priors.* Annals of Statistics, 36(3), 1435-1463. — Foundational work on posterior contraction rates; establishes how quickly posteriors concentrate around truth.

7. Ghosal, S., & van der Vaart, A. (2017). *Fundamentals of Nonparametric Bayesian Inference.* Cambridge University Press. — Comprehensive treatment of posterior contraction in nonparametric settings.

### Recursive Estimation (Propagation/Belief Expansion)

8. Kalman, R.E. (1960). *A New Approach to Linear Filtering and Prediction Problems.* Journal of Basic Engineering, 82(1), 35-45. — Original Kalman filter; establishes prediction step where covariance expands.

9. Evensen, G. (2003). *The Ensemble Kalman Filter: theoretical formulation and practical implementation.* Ocean Dynamics, 53, 343-367. — Ensemble methods; covariance inflation to counteract underestimation of uncertainty.

---

## Note for Readers from Physics

For readers familiar with condensed matter physics: AQ properties have natural analogs to particle properties (magnitude~mass, phase~charge, frequency~size, coherence~spin). The BEC framework document `bec_analogy/BEC_CONDENSATION_INFORMATION.md` develops these connections. AQ can be understood as quasiparticles (collective excitations) of the belief field, analogous to phonons or Bogoliubov excitations. This analogy is optional - the CS-native definitions above are self-contained.

---

*This document establishes the formal terminology for AKIRA, integrating four theoretical layers:*

1. *Information theory (PID/ΦID) — static and temporal decomposition*
2. *Bayesian statistics — posterior contraction/propagation*
3. *Circuit complexity (Mao et al.) — tractability bounds*
4. *Emergent structure — Action Quanta and collapse dynamics*

*Key terminological distinctions:*
- *PID "information atoms" (decomposition terms) vs AKIRA "Action Quanta" (emergent patterns)*
- *"Superposition" (pre-collapse) vs "Crystallized" (post-collapse, IRREDUCIBLE)*
- *"SOS Width" (problem complexity) vs "Circuit Breadth" (architecture capacity)*
- *"Bonded state" (combined crystallized AQ) - tentative term for concept formation*

*The 7+1 band architecture is formally justified by circuit complexity: for visual prediction with SOS width k ≈ 3 and binary relations (β = 2), required breadth = (3+1) × 2 = 8 bands.*

*Oscar Goldman — Shogu Research Group @ Datamutant.ai*

