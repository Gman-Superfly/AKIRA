# Information Dynamics: Storage, Transfer, and Modification

## Document Purpose

This document explains **Information Dynamics** - how information flows, persists, and transforms over time. This extends PID (static decomposition) to temporal processes and is essential for understanding how AKIRA processes sequential data.

---

## Table of Contents

1. [From Static to Dynamic](#1-from-static-to-dynamic)
2. [The Three Operations](#2-the-three-operations)
3. [Storage Explained](#3-storage-explained)
4. [Transfer Explained](#4-transfer-explained)
5. [Modification Explained](#5-modification-explained)
6. [The Key Insight: Modification = Synergy](#6-the-key-insight-modification--synergy)
7. [Theory and Mathematical Foundation](#7-theory-and-mathematical-foundation)
8. [General Applications](#8-general-applications)
9. [Information Dynamics in AKIRA](#9-information-dynamics-in-akira)
10. [How This Informs AKIRA's Theoretical Foundations](#10-how-this-informs-akiras-theoretical-foundations)
11. [References](#11-references)

---

## 1. From Static to Dynamic

### 1.1 The Limitation of Static Analysis

PID (see `SYNERGY_REDUNDANCY.md`) tells us how information is distributed across sources at a single moment. But real systems evolve over time:

```
STATIC vs DYNAMIC
─────────────────

STATIC (PID):
  Time t: How is information about T distributed across S₁, S₂?
  
  Answer: Redundancy, Unique(1), Unique(2), Synergy
  
  Problem: Doesn't tell us what happens NEXT

DYNAMIC (Information Dynamics):
  Time t → t+1: How does information FLOW?
  
  Questions:
  • Does information PERSIST within a variable? (Storage)
  • Does information MOVE between variables? (Transfer)
  • Do information streams INTERACT? (Modification)
```

### 1.2 Why Dynamics Matter

```
COMPUTATION IS TEMPORAL
───────────────────────

Any computation involves:
  1. Reading inputs (information arrives)
  2. Processing (information transforms)
  3. Producing outputs (information leaves)

This happens OVER TIME.

A static snapshot misses:
  • Where information came FROM
  • Where information is GOING
  • How information CHANGES

Information Dynamics captures the temporal structure.
```

---

## 2. The Three Operations

Lizier et al. (2013) identified three fundamental information operations:

```
THE TRIAD OF INFORMATION OPERATIONS
───────────────────────────────────

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  STORAGE                                                               │
│  ───────                                                                │
│  Information persisting WITHIN a variable across time.                 │
│  The past of X tells you about the future of X.                       │
│  X "remembers" itself.                                                 │
│                                                                         │
│  Symbol: A(X) - Active Information Storage                            │
│                                                                         │
│  ════════════════════════════════════════════════════════════════════  │
│                                                                         │
│  TRANSFER                                                              │
│  ────────                                                               │
│  Information flowing FROM one variable TO another.                    │
│  The past of X tells you about the future of Y.                       │
│  X "communicates" to Y.                                                │
│                                                                         │
│  Symbol: T(X→Y) - Transfer Entropy                                    │
│                                                                         │
│  ════════════════════════════════════════════════════════════════════  │
│                                                                         │
│  MODIFICATION                                                          │
│  ────────────                                                           │
│  Information arising from INTERACTION of multiple streams.            │
│  Neither X alone nor Y alone predicts outcome.                        │
│  Information EMERGES from combination.                                │
│                                                                         │
│  Symbol: M(X,Y→Z) - Information Modification                          │
│                                                                         │
│  ★ KEY INSIGHT: Modification = Synergy observed in dynamics ★         │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Storage Explained

### 3.1 Definition

**Storage** is information that persists within a single variable over time. The past of X informs the future of X.

### 3.2 Intuitive Example: Memory

```
EXAMPLE: Memory as Storage
──────────────────────────

Variable X = Your knowledge of your name

Time t:   You know your name is "Alice"
Time t+1: You still know your name is "Alice"
Time t+2: You still know your name is "Alice"

The information "my name is Alice" PERSISTS.
It STORES within X.

Past X predicts future X (your name won't change).

This is Active Information Storage.
```

### 3.3 Cellular Automata Example: Blinkers

```
EXAMPLE: Blinker in Game of Life
────────────────────────────────

A "blinker" is a pattern that oscillates:

  Time t:      Time t+1:    Time t+2:
  
  . ■ .        . . .        . ■ .
  . ■ .   →    ■ ■ ■   →    . ■ .
  . ■ .        . . .        . ■ .

The pattern returns to itself.
Information about "I am a blinker" PERSISTS.
The cell's past predicts its future.

High storage = stable/periodic patterns
```

### 3.4 Key Properties of Storage

```
STORAGE PROPERTIES
──────────────────

1. WITHIN-VARIABLE
   Information stays in the same variable
   X_past → X_future (not X → Y)

2. SELF-PREDICTION
   Knowing past of X helps predict future of X
   Autocorrelation in time series

3. MEMORY-LIKE
   The system "remembers" its state
   State persistence

4. MEASURED BY
   Active Information Storage A(X)
   A(X) = I(X_past ; X_future)
```

---

## 4. Transfer Explained

### 4.1 Definition

**Transfer** is information that flows from one variable to another over time. The past of X informs the future of Y.

### 4.2 Intuitive Example: Conversation

```
EXAMPLE: Conversation as Transfer
─────────────────────────────────

Variable X = What person A says
Variable Y = What person B thinks

Time t:   A says "The meeting is at 3pm"
Time t+1: B now thinks "The meeting is at 3pm"

Information TRANSFERRED from X to Y.
Past of X (what A said) predicts future of Y (what B knows).

This is Transfer Entropy.
```

### 4.3 Cellular Automata Example: Gliders

```
EXAMPLE: Glider in Game of Life
───────────────────────────────

A "glider" is a pattern that moves:

  Time t:      Time t+1:    Time t+2:
  
  . ■ .        . . ■        . . .
  . . ■   →    ■ . ■   →    . . ■
  ■ ■ ■        . ■ ■        ■ . ■
                            . ■ ■
  
  (Pattern shifts diagonally)

Information about "live cell" MOVES from one location to another.
The past state of cell A predicts future state of cell B.

High transfer = information propagation across space
```

### 4.4 Key Properties of Transfer

```
TRANSFER PROPERTIES
───────────────────

1. BETWEEN-VARIABLES
   Information moves from one variable to another
   X_past → Y_future

2. CROSS-PREDICTION
   Knowing past of X helps predict future of Y
   Beyond what Y's own past tells you

3. COMMUNICATION-LIKE
   One variable "sends" to another
   Signal propagation

4. MEASURED BY
   Transfer Entropy T(X→Y)
   T(X→Y) = I(X_past ; Y_future | Y_past)
   
   Note the conditioning on Y_past:
   This measures what X adds BEYOND Y's own memory
```

### 4.5 Directionality

Transfer has a direction:

```
TRANSFER IS DIRECTED
────────────────────

T(X→Y) ≠ T(Y→X) in general

Example:
  X = Teacher speaking
  Y = Student listening
  
  T(X→Y) is HIGH (teacher → student)
  T(Y→X) is LOW (student rarely changes teacher's speech)

Transfer entropy can detect CAUSAL direction.
```

---

## 5. Modification Explained

### 5.1 Definition

**Modification** is information that arises from the interaction of multiple information streams. Neither stream alone produces the outcome; it emerges from their combination.

### 5.2 Intuitive Example: Collision

```
EXAMPLE: Billiard Ball Collision
────────────────────────────────

Variable X = Ball A trajectory
Variable Y = Ball B trajectory
Outcome Z = Where balls end up after collision

BEFORE collision:
  X alone: Ball A continues straight (no information about bounce)
  Y alone: Ball B continues straight (no information about bounce)

COLLISION:
  X and Y TOGETHER determine Z
  Neither alone predicts the post-collision trajectories
  The outcome EMERGES from their interaction

This is Information Modification.
```

### 5.3 Cellular Automata Example: Glider Collision

```
EXAMPLE: Glider Collision in Game of Life
─────────────────────────────────────────

Two gliders approach each other:

  Glider A →   ← Glider B

Each glider alone would continue its path.

COLLISION:
  The outcome depends on BOTH gliders
  Could produce: annihilation, new patterns, reflection
  
  The post-collision state emerges from INTERACTION.
  Neither glider alone determines it.

High modification = non-trivial computation happening
```

### 5.4 Key Properties of Modification

```
MODIFICATION PROPERTIES
───────────────────────

1. MULTI-STREAM INTERACTION
   Multiple information sources combine
   Neither alone determines outcome

2. EMERGENT INFORMATION
   Outcome has information not in any single source
   Only in combination

3. COMPUTATION-LIKE
   Non-trivial processing
   Actual "work" being done

4. = SYNERGY IN DYNAMICS
   This is Lizier's key insight!
   Modification IS synergy observed temporally
```

---

## 6. The Key Insight: Modification = Synergy

### 6.1 The Bridge Between PID and Dynamics

```
THE FUNDAMENTAL EQUIVALENCE
───────────────────────────

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  SYNERGY (PID, static):                                                │
│  Information that NEITHER source alone provides                        │
│  but TOGETHER they do.                                                 │
│                                                                         │
│  MODIFICATION (Dynamics, temporal):                                    │
│  Information that NEITHER stream alone produces                        │
│  but their INTERACTION does.                                           │
│                                                                         │
│  ════════════════════════════════════════════════════════════════════  │
│                                                                         │
│             THEY ARE THE SAME THING                                    │
│                                                                         │
│  Synergy = how information IS distributed (snapshot)                   │
│  Modification = synergy HAPPENING (over time)                          │
│                                                                         │
│  When two streams interact to produce an outcome                       │
│  that neither could alone → that IS synergy → that IS modification    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 6.2 Why This Matters

```
COMPUTATIONAL INTERPRETATION
────────────────────────────

Storage = MEMORY
  Information persisting in time
  State maintenance

Transfer = COMMUNICATION
  Information moving in space
  Signal propagation

Modification = COMPUTATION
  Information being PROCESSED
  Non-trivial transformation

ANY computation can be decomposed into these three.
Modification (= synergy) is where the "actual work" happens.
```

### 6.3 The XOR Example Revisited

```
XOR AS MODIFICATION
───────────────────

Time t:
  Stream X has bit "0"
  Stream Y has bit "1"

Time t+1:
  Outcome Z = X XOR Y = "1"

Analysis:
  - Storage: X and Y remember their bits
  - Transfer: Bits move to the XOR gate
  - Modification: XOR COMBINES them
  
  The "1" output is modification/synergy:
    X alone: "0" → doesn't tell you Z
    Y alone: "1" → doesn't tell you Z
    X and Y together: "0" XOR "1" = "1"

Modification IS the XOR operation itself.
```

---

## 7. Theory and Mathematical Foundation

### 7.1 Active Information Storage

```
ACTIVE INFORMATION STORAGE
──────────────────────────

A(X) = I(X^(k) ; X_n)

Where:
  X^(k) = Past k values of X: (X_{n-k}, ..., X_{n-1})
  X_n   = Current value of X

This measures: How much does X's past tell you about X's present?

In the limit:
  A(X) = I(X^(-) ; X_n)
  
Where X^(-) is the entire past of X.
```

### 7.2 Transfer Entropy

```
TRANSFER ENTROPY (Schreiber, 2000)
──────────────────────────────────

T(X→Y) = I(X^(k) ; Y_n | Y^(l))

Where:
  X^(k) = Past k values of X
  Y^(l) = Past l values of Y
  Y_n   = Current value of Y

This measures: How much does X's past tell you about Y's present,
               BEYOND what Y's own past tells you?

The conditioning on Y^(l) is crucial:
  - Without it: Just correlation
  - With it: Actual transfer (new information from X)
```

### 7.3 Information Modification

```
INFORMATION MODIFICATION (Lizier et al., 2013)
──────────────────────────────────────────────

M(X,Y→Z) = I(X^(-), Y^(-) ; Z) - I(X^(-) ; Z) - I(Y^(-) ; Z)

This measures: Information in Z that requires BOTH X and Y's pasts,
               beyond what each alone provides.

Equivalently:
  M(X,Y→Z) = Synergy component of I(X^(-), Y^(-) ; Z)

This is synergy observed in temporal dynamics.
```

### 7.4 The Decomposition

```
TOTAL INFORMATION DECOMPOSITION
───────────────────────────────

The total information from past to present decomposes:

I(X^(-), Y^(-) ; Z) = A(X→Z) + A(Y→Z) + T(X→Z) + T(Y→Z) + M(X,Y→Z) + ...

Where:
  A(X→Z) = Storage contribution from X
  T(X→Z) = Transfer from X to Z
  M(X,Y→Z) = Modification (synergy) between X and Y

This decomposes ALL information flow into:
  - What persisted (storage)
  - What moved (transfer)
  - What emerged from interaction (modification)
```

---

## 8. General Applications

### 8.1 Neuroscience: Neural Computation

```
NEURAL INFORMATION DYNAMICS
───────────────────────────

Setup:
  X = Activity of neuron/region A
  Y = Activity of neuron/region B
  Z = Activity of downstream region C

Analysis:
  Storage A(X→C): How much does A's persistent state contribute?
  Transfer T(X→C): How much new information comes from A?
  Modification M(X,Y→C): How much emerges from A-B interaction?

Findings:
  - Sensory cortex: High transfer (signal propagation)
  - Association cortex: High modification (integration)
  - Motor cortex: High storage (stable commands)
```

### 8.2 Cellular Automata: Distributed Computation

```
CA INFORMATION DYNAMICS
───────────────────────

In cellular automata:

STORAGE locations: Where patterns persist
  → Blinkers, still lifes, stable structures

TRANSFER conduits: Where information flows
  → Glider paths, signal propagation channels

MODIFICATION sites: Where computation happens
  → Glider collisions, logic gate implementations

Rule 110 (Turing-complete CA):
  Information dynamics reveals WHERE computation occurs
  Not all cells compute equally
```

### 8.3 Time Series Analysis

```
CAUSAL ANALYSIS
───────────────

Transfer entropy detects causal relationships:

Example: Climate data
  X = Sea surface temperature
  Y = Rainfall
  
  T(X→Y) > 0 suggests temperature influences rainfall
  T(Y→X) ≈ 0 suggests rainfall doesn't influence temperature
  
  Direction of transfer ≈ Direction of causation
```

---

## 9. Information Dynamics in AKIRA

### 9.1 Mapping to AKIRA Architecture

```
AKIRA'S INFORMATION OPERATIONS
──────────────────────────────

STORAGE in AKIRA:
────────────────
  WHERE: Within each spectral band over time
  WHAT: Persistent structure at each frequency
  HOW: Band processors maintain state
  
  Examples:
  - Low-band structure persists (stable shape)
  - Temporal band stores sequence history
  - Learned weights are crystallized storage

TRANSFER in AKIRA:
─────────────────
  WHERE: Wormhole attention between bands
  WHAT: Information flowing across frequencies
  HOW: Cross-band attention mechanisms
  
  Examples:
  - Low→High: Structure constrains details
  - High→Low: Details inform structure
  - Temporal→Spatial: Time informs space

MODIFICATION in AKIRA:
─────────────────────
  WHERE: Collapse events, band interactions
  WHAT: Synergistic combination producing predictions
  HOW: Multi-band integration before collapse
  
  Examples:
  - Structure + detail → full prediction
  - Multiple bands combining → certainty
  - Collapse itself is massive modification event
```

### 9.2 The Three Operations in AKIRA's Cycle

```
AKIRA PROCESSING CYCLE
──────────────────────

INPUT PHASE:
  Transfer: New information enters from sensors
  
PROCESSING PHASE:
  Storage: Bands maintain their representations
  Transfer: Wormholes exchange information between bands
  Modification: Bands interact, synergy builds
  
COLLAPSE PHASE:
  Modification: Major synergistic event
    - All bands combine
    - Synergy converts to redundancy
    - New information (the prediction) emerges
    - This is WHERE computation "happens"
  
OUTPUT PHASE:
  Transfer: Prediction flows out
  Storage: New state persists for next cycle
```

### 9.3 Collapse as Modification

```
COLLAPSE = MASSIVE MODIFICATION EVENT
─────────────────────────────────────

Before collapse:
  - Bands have DISTRIBUTED information (high synergy)
  - Each band alone cannot predict
  - Information is "spread" across bands

During collapse:
  - Bands INTERACT intensively
  - Synergy is "consumed" to produce prediction
  - MODIFICATION happens here
  
After collapse:
  - Prediction emerges (new information!)
  - This prediction wasn't in any single band
  - It EMERGED from their combination

Collapse IS modification IS synergy being converted to output.
```

---

## 10. How This Informs AKIRA's Theoretical Foundations

### 10.1 Why Three Mechanisms Exist

```
JUSTIFICATION FOR AKIRA'S ARCHITECTURE
──────────────────────────────────────

AKIRA has three mechanism types because information dynamics
requires three operation types:

STORAGE MECHANISMS:
  - Band processors with state
  - Temporal band memory
  - Learned parameters
  
  Why needed: Must maintain information across time.
  Without storage: No memory, no learning.

TRANSFER MECHANISMS:
  - Wormhole attention
  - Cross-band connections
  - Input/output pathways
  
  Why needed: Must move information across space/bands.
  Without transfer: Isolated bands, no integration.

MODIFICATION MECHANISMS:
  - Multi-head attention (combines queries)
  - Non-linear activations
  - Collapse dynamics
  
  Why needed: Must COMBINE information to compute.
  Without modification: No actual computation.

The architecture implements all three because
all three are necessary for computation.
```

### 10.2 Why Wormholes Matter

```
WORMHOLES ENABLE TRANSFER
─────────────────────────

Without wormholes:
  - Each band isolated
  - Information cannot flow between frequencies
  - No cross-scale integration

With wormholes:
  - Transfer entropy T(Band_i → Band_j) > 0
  - Information flows across scales
  - Cross-frequency patterns can form

Wormholes are the TRANSFER conduits of AKIRA.
They enable the communication necessary for
modification (computation) to occur.
```

### 10.3 Why Collapse is Central

```
COLLAPSE IS WHERE COMPUTATION HAPPENS
─────────────────────────────────────

In any system:
  Modification = Actual computation
  
In AKIRA:
  Collapse = Massive modification event
  
Therefore:
  Collapse = Where AKIRA actually computes

Before collapse: Information is distributed (stored, transferred)
During collapse: Information is PROCESSED (modified)
After collapse: New information exists (the prediction)

Understanding information dynamics reveals:
  - Collapse is not "metaphor for decision"
  - Collapse IS the computational event itself
  - It's where synergy becomes output
```

### 10.4 Observability

```
INFORMATION DYNAMICS IS OBSERVABLE
──────────────────────────────────

We can MEASURE:
  - Storage: Autocorrelation within bands
  - Transfer: Cross-correlation between bands
  - Modification: Synergy during collapse

This allows:
  - Verifying architecture works as intended
  - Diagnosing failures (where is information stuck?)
  - Comparing to theoretical predictions

See: AKIRA/observability/OBSERVABILITY_EMBEDDINGS.md
```

---

## 11. References

### 11.1 Original Information Dynamics Literature

1. Lizier, J.T., Prokopenko, M., & Zomaya, A.Y. (2012). *Local measures of information storage in complex distributed computation.* Information Sciences, 208, 39-54.
   - Defines active information storage
   - Local computation in cellular automata

2. Schreiber, T. (2000). *Measuring information transfer.* Physical Review Letters, 85(2), 461.
   - Defines transfer entropy
   - Foundational paper

3. Lizier, J.T., Flecker, B., & Williams, P.L. (2013). *Towards a Synergy-based Approach to Measuring Information Modification.* arXiv:1303.3440.
   - Connects modification to synergy
   - The key bridging paper

4. Wibral, M., Vicente, R., & Lindner, M. (2014). *Directed Information Measures in Neuroscience.* Springer.
   - Comprehensive textbook
   - Applications to neural data

### 11.2 Applications

5. Lizier, J.T. (2014). *JIDT: An Information-Theoretic Toolkit for Studying the Dynamics of Complex Systems.* Frontiers in Robotics and AI, 1, 11.
   - Software implementation
   - Practical computation guide

6. Mediano, P.A.M., et al. (2025). *Toward a unified taxonomy of information dynamics via Integrated Information Decomposition.* PNAS 122(39).
   - ΦID framework
   - Unifies PID and dynamics

### 11.3 AKIRA Internal Documents

7. `AKIRA/foundations/TERMINOLOGY.md`
   - Complete terminology reference
   - Section 3 covers Information Dynamics

8. `AKIRA/foundations/terminology_foundations/SYNERGY_REDUNDANCY.md`
   - Foundation for understanding modification
   - Synergy is the static view of modification

9. `AKIRA/architecture_theoretical/SPECTRAL_BELIEF_MACHINE.md`
   - Architecture implementing these operations
   - Wormholes, band processors, collapse

10. `AKIRA/observability/OBSERVABILITY_EMBEDDINGS.md`
    - How to observe information dynamics in AKIRA
    - Measurement and visualization

---

## Summary

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  INFORMATION DYNAMICS: KEY TAKEAWAYS                                   │
│                                                                         │
│  ═══════════════════════════════════════════════════════════════════   │
│                                                                         │
│  THREE FUNDAMENTAL OPERATIONS:                                         │
│                                                                         │
│  STORAGE = Information persisting within a variable                    │
│    X_past → X_future                                                   │
│    Memory, state persistence                                           │
│                                                                         │
│  TRANSFER = Information flowing between variables                      │
│    X_past → Y_future                                                   │
│    Communication, signal propagation                                   │
│                                                                         │
│  MODIFICATION = Information emerging from combination                  │
│    (X,Y)_past → Z_future where neither X nor Y alone suffices         │
│    Computation, synergy in action                                      │
│                                                                         │
│  ═══════════════════════════════════════════════════════════════════   │
│                                                                         │
│  THE KEY INSIGHT:                                                       │
│                                                                         │
│  MODIFICATION = SYNERGY                                                │
│                                                                         │
│  They are the same thing:                                              │
│  - Synergy = static view (information requiring combination)          │
│  - Modification = dynamic view (combination happening in time)        │
│                                                                         │
│  ═══════════════════════════════════════════════════════════════════   │
│                                                                         │
│  IN AKIRA:                                                              │
│                                                                         │
│  Storage = Band state persistence, learned weights                     │
│  Transfer = Wormhole attention, cross-band flow                        │
│  Modification = Collapse events, band interactions                     │
│                                                                         │
│  Collapse IS modification IS where computation happens.                │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

*This document is part of AKIRA's terminology foundations. For the complete terminology framework, see `AKIRA/foundations/TERMINOLOGY.md`. For related concepts, see `SYNERGY_REDUNDANCY.md`.*

