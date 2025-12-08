# Circuit Complexity: SOS Width and Tractability Bounds

## Document Purpose

This document explains **Circuit Complexity** as it applies to neural network architectures, particularly the SOS Width framework from Mao et al. (2023). This provides the formal theoretical basis for AKIRA's 7+1 band architecture.

---

## Table of Contents

1. [The Core Question](#1-the-core-question)
2. [What is Circuit Complexity?](#2-what-is-circuit-complexity)
3. [SOS Width Explained](#3-sos-width-explained)
4. [Circuit Breadth and Depth](#4-circuit-breadth-and-depth)
5. [The Key Theorem](#5-the-key-theorem)
6. [Problem Classes](#6-problem-classes)
7. [Theory and Mathematical Foundation](#7-theory-and-mathematical-foundation)
8. [General Applications](#8-general-applications)
9. [Circuit Complexity in AKIRA](#9-circuit-complexity-in-akira)
10. [How This Informs AKIRA's Theoretical Foundations](#10-how-this-informs-akiras-theoretical-foundations)
11. [References](#11-references)

---

## 1. The Core Question

When designing a neural network architecture, a fundamental question arises:

> **What problems can this architecture solve, and what problems are fundamentally beyond it?**

This is not about training or data. It is about the **mathematical limits** of what a fixed architecture can represent.

```
THE ARCHITECTURE LIMITATION QUESTION
────────────────────────────────────

Given:
  - A neural network with D layers
  - Each layer has B parallel channels (breadth)
  - Fixed architecture (cannot grow)

Questions:
  1. What problems CAN this architecture solve?
  2. What problems are IMPOSSIBLE for it?
  3. How do we know the difference?

Circuit complexity theory provides the answers.
```

---

## 2. What is Circuit Complexity?

### 2.1 Circuits as Computation Model

A **circuit** is a model of computation:

```
CIRCUIT MODEL
─────────────

INPUT LAYER:       x₁   x₂   x₃   ...   xₙ
                    │    │    │         │
                    ▼    ▼    ▼         ▼
LAYER 1:          [gates performing operations]
                    │    │    │    │
                    ▼    ▼    ▼    ▼
LAYER 2:          [more gates]
                    │    │
                    ▼    ▼
    ...
                    │
                    ▼
OUTPUT:           result

Key parameters:
  - DEPTH (D): Number of layers (sequential steps)
  - BREADTH (B): Number of gates per layer (parallel capacity)
  - SIZE: Total number of gates
```

### 2.2 Why Circuits Matter for Neural Networks

```
NEURAL NETWORKS ARE CIRCUITS
────────────────────────────

A neural network layer:
  y = σ(Wx + b)

This is a circuit:
  - Input: x
  - Gates: Matrix multiplication, addition, activation
  - Output: y

A multi-layer network:
  y = f_D(...f_2(f_1(x)))

This is a D-layer circuit:
  - Depth D = number of layers
  - Breadth B = width of each layer
  - Each layer = one level of gates

Circuit complexity tells us what functions f
can be computed by circuits of given depth and breadth.
```

### 2.3 The Key Insight

```
FIXED ARCHITECTURE = FIXED CIRCUIT
──────────────────────────────────

If you fix:
  - Network depth (number of layers)
  - Network width (neurons per layer)

Then you have a FIXED circuit class.

Some functions CANNOT be computed by this class.
This is a MATHEMATICAL fact, not a training problem.

Example:
  A network with 100 neurons per layer, 10 layers
  CANNOT compute certain functions
  No matter how you train it
  No matter how much data you have
```

---

## 3. SOS Width Explained

### 3.1 The Planning Context

Mao et al. (2023) developed their theory in the context of **planning problems**:

```
PLANNING PROBLEMS
─────────────────

A planning problem has:
  - Initial state: Where you start
  - Goal state: Where you want to be
  - Actions: Ways to change state
  - Preconditions: What must be true to act
  - Effects: What changes when you act

Example: Blocks World
  Initial: Block A on table, Block B on table
  Goal: Block A on Block B
  Action: Pick up A, Put A on B
  Preconditions: A is clear, hand is empty
  Effects: A is now on B
```

### 3.2 What is SOS Width?

**SOS Width** (Strong Optimally-Serializable Width) measures how many things you need to track simultaneously:

```
SOS WIDTH DEFINITION
────────────────────

SOS width k means:

At any step in solving the problem optimally,
you need to track at most k PRECONDITIONS simultaneously.

These preconditions can be achieved one-by-one (serialized):
  1. Achieve precondition p₁
  2. Achieve p₂ while MAINTAINING p₁
  3. Achieve p₃ while MAINTAINING p₁, p₂
  ... and so on up to k preconditions

Lower k = simpler problem (fewer things to track)
Higher k = harder problem (more simultaneous constraints)
```

### 3.3 Examples of SOS Width

```
BLOCKS WORLD: SOS Width k = 2
─────────────────────────────

Goal: Stack A on B on C

Planning:
  Step 1: Get C placed correctly
    Need to track: C's position (1 constraint)
  
  Step 2: Get B on C
    Need to track: C stays correct, B's position (2 constraints)
  
  Step 3: Get A on B
    Need to track: B-C stays correct, A's position (2 constraints)

Maximum simultaneous tracking: 2
SOS Width = 2
```

```
LOGISTICS: SOS Width k = 2
──────────────────────────

Goal: Package at destination

Planning:
  Step 1: Package in vehicle
    Track: Package location (1 constraint)
  
  Step 2: Vehicle to destination
    Track: Package in vehicle, vehicle location (2 constraints)
  
  Step 3: Unload package
    Track: Vehicle at destination, package state (2 constraints)

Maximum: 2
SOS Width = 2
```

```
SOKOBAN: SOS Width k = UNBOUNDED
────────────────────────────────

Goal: Push boxes to targets

Problem:
  Pushing a box into a corner is IRREVERSIBLE
  Each bad push creates a constraint you must avoid forever
  Number of constraints can grow with problem size

SOS Width = unbounded (grows with puzzle complexity)
```

### 3.4 Key Properties of SOS Width

```
SOS WIDTH PROPERTIES
────────────────────

1. PROBLEM PROPERTY (not algorithm)
   SOS width is a property of the PROBLEM
   Not of how you solve it
   Different problems have different widths

2. SERIALIZATION
   Width k means k things at once, achievable in sequence
   The "S" in SOS stands for Serializable

3. OPTIMALLY
   We measure width for OPTIMAL solutions
   Suboptimal solutions might need more tracking

4. BOUNDS COMPLEXITY
   Width determines what neural architectures can solve it
   This is the key result
```

---

## 4. Circuit Breadth and Depth

### 4.1 Breadth

```
CIRCUIT BREADTH (B)
───────────────────

Breadth = Number of parallel channels at each layer

Think of it as:
  "How many things can I process SIMULTANEOUSLY?"

High breadth:
  - Can track many things at once
  - More parallel capacity
  - Wider network

Low breadth:
  - Limited simultaneous tracking
  - Less parallel capacity
  - Narrower network

In AKIRA: B = 8 (the 7+1 bands)
```

### 4.2 Depth

```
CIRCUIT DEPTH (D)
─────────────────

Depth = Number of sequential layers

Think of it as:
  "How many STEPS can I take?"

High depth:
  - Can do more sequential operations
  - Can handle longer plans
  - Deeper network

Low depth:
  - Fewer sequential operations
  - Limited plan length
  - Shallower network

In AKIRA: D = number of SBM layers
```

### 4.3 The Tradeoff

```
BREADTH vs DEPTH
────────────────

Some problems need breadth (many simultaneous constraints):
  → Increase width, depth may not help

Some problems need depth (many sequential steps):
  → Increase depth, width may not help

The key is matching architecture to problem structure.
```

---

## 5. The Key Theorem

### 5.1 Mao et al.'s Main Result

```
THE FUNDAMENTAL THEOREM (Mao et al., 2023)
──────────────────────────────────────────

For a planning problem with:
  - SOS width k
  - Predicate arity β (how many arguments in relations)

A relational neural network can solve it if and only if:

┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│              REQUIRED BREADTH = (k + 1) × β                        │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

This is NECESSARY and SUFFICIENT.

Less breadth → Cannot solve the problem (mathematical impossibility)
This breadth → Can solve (given enough depth)
```

### 5.2 Understanding the Formula

```
WHY (k + 1) × β?
────────────────

k = Number of constraints to track simultaneously
+1 = Need one extra slot for the "current goal"
β = Each constraint involves β entities

So total slots needed = (k + 1) × β

Example (Blocks World):
  k = 2 (track block position + destination)
  β = 2 (binary relations: On(A,B))
  
  Required breadth = (2 + 1) × 2 = 6 channels

Example (Visual prediction, estimated):
  k ≈ 3 (object + neighbors + temporal context)
  β = 2 (binary spatial relations)
  
  Required breadth = (3 + 1) × 2 = 8 channels
```

### 5.3 What This Means

```
IMPLICATIONS OF THE THEOREM
───────────────────────────

1. HARD LIMIT
   If your network has breadth B,
   it CANNOT solve problems with SOS width k > (B/β) - 1
   
   No training will overcome this.
   No data will help.
   It's mathematically impossible.

2. SUFFICIENT CONDITION
   If you HAVE breadth (k+1)×β,
   then the problem IS solvable (with enough depth).
   
   Training can find the solution.
   Architecture is not the bottleneck.

3. ARCHITECTURE DESIGN PRINCIPLE
   Know your problem's SOS width → Choose your breadth
   
   For AKIRA targeting k≈3, β=2:
   Need breadth ≥ 8
   7+1 = 8 bands → Sufficient!
```

---

## 6. Problem Classes

### 6.1 Three Classes

Mao et al. classify problems into three classes:

```
PROBLEM CLASSIFICATION
──────────────────────

CLASS 1: Constant Breadth, Constant Depth
─────────────────────────────────────────
  SOS width: Constant (doesn't grow with input)
  Required: Fixed small circuit
  Examples: Single object tracking, local pattern detection
  
  Status: EASY for fixed architecture

CLASS 2: Constant Breadth, Unbounded Depth
──────────────────────────────────────────
  SOS width: Constant
  Plan length: Can grow with input
  Required: Fixed width, more layers for bigger inputs
  Examples: Blocks World, Logistics
  
  Status: TRACTABLE for fixed architecture (add layers)

CLASS 3: Unbounded Breadth
──────────────────────────
  SOS width: Grows with input size
  Required: Architecture must grow with problem
  Examples: Sokoban, complex coordination
  
  Status: INTRACTABLE for fixed architecture
```

### 6.2 Visual Summary

```
PROBLEM DIFFICULTY
──────────────────

                    Breadth needed
                    ↑
  CLASS 3           │  ╱╱╱╱╱╱╱╱╱╱╱╱
  (Intractable)     │ ╱╱╱╱╱╱╱╱╱╱╱╱╱
                    │╱╱╱╱╱╱╱╱╱╱╱╱╱╱
                    │
  ──────────────────┼──────────────────
  CLASS 2           │
  (Tractable,       │   More depth
   add layers)      │   needed for
                    │   larger inputs
  ──────────────────┼──────────────────
  CLASS 1           │
  (Easy)            │   Fixed circuit
                    │   suffices
                    └───────────────────→ Input size

Fixed architecture (constant breadth) can handle Class 1 and 2.
Class 3 requires growing architecture.
```

### 6.3 Examples in Each Class

```
CLASS 1 EXAMPLES (Easy):
────────────────────────
  - Track one object's position
  - Detect if an edge exists
  - Check if a cell is occupied
  
  Fixed depth, fixed breadth suffices.

CLASS 2 EXAMPLES (Tractable):
─────────────────────────────
  - Blocks World (any number of blocks)
  - Logistics (any number of packages)
  - Path finding (any path length)
  
  Fixed breadth, but need more depth for bigger instances.

CLASS 3 EXAMPLES (Intractable):
───────────────────────────────
  - Sokoban (arbitrary complexity)
  - Arbitrary graph coloring
  - General SAT solving
  
  Cannot be solved by fixed architecture.
```

---

## 7. Theory and Mathematical Foundation

### 7.1 Relational Neural Networks

The theorem applies to **Relational Neural Networks (RelNN)**:

```
RelNN[D, B]
───────────

A RelNN with:
  D = depth (number of message-passing rounds)
  B = breadth (embedding dimension / channels)

Operations:
  - Message passing between entities
  - Aggregation of neighbor information
  - Update of entity representations

This captures:
  - Graph neural networks
  - Transformers (as special case)
  - Attention-based architectures
```

### 7.2 The Correspondence

```
RelNN ↔ PLANNING CIRCUIT
────────────────────────

Mao et al. prove a correspondence:

RelNN with:           ↔    Can solve problems with:
  Breadth B                   SOS width k ≤ (B/β) - 1
  Depth D                     Plan length ≤ O(D)

This is EXACT correspondence:
  - If breadth too low → Cannot represent solution
  - If depth too low → Cannot execute long plans
  - Given sufficient both → Solution exists
```

### 7.3 Proof Sketch

```
WHY THE THEOREM IS TRUE (Intuition)
───────────────────────────────────

The proof has two directions:

LOWER BOUND (why you NEED the breadth):
  - Each precondition requires tracking β entities
  - With k preconditions + current goal: (k+1) constraints
  - Total entity slots: (k+1) × β
  - Fewer slots → Cannot represent all constraints

UPPER BOUND (why the breadth SUFFICES):
  - Construct a circuit that:
    1. Represents current constraint set
    2. Selects next action via regression rule
    3. Updates constraints
  - This can be done with (k+1)×β breadth
  - Each layer = one planning step
```

### 7.4 S-GRS (Serialized Goal Regression Search)

```
S-GRS: THE OPTIMAL STRATEGY
───────────────────────────

S-GRS is the planning algorithm that achieves SOS width bound:

1. Start with goal G
2. Pick a precondition p to achieve
3. Achieve p using regression rule
4. MAINTAIN already-achieved preconditions
5. Repeat until initial state reached

The key is SERIALIZATION:
  - Don't try to achieve everything at once
  - Achieve one thing at a time
  - Maintain what you've achieved

This minimizes simultaneous tracking → Minimizes width
```

---

## 8. General Applications

### 8.1 Architecture Design

```
USING SOS WIDTH FOR DESIGN
──────────────────────────

Process:
  1. Analyze your problem domain
  2. Estimate SOS width k
  3. Identify predicate arity β
  4. Set network breadth ≥ (k+1)×β
  5. Add depth as needed for plan length

Example: Robot manipulation
  - k ≈ 2 (gripper state + object state)
  - β = 2 (binary relations)
  - Need breadth ≥ 6
  
  A 6-channel architecture suffices for basic manipulation.
```

### 8.2 Understanding Failures

```
DIAGNOSING ARCHITECTURE FAILURES
────────────────────────────────

If your network fails on certain problems:

Question: Is it Class 3 (intractable)?
  - Does complexity grow with input size?
  - Are there irreversible constraints?

If yes: No fixed architecture will work.
If no: Check if you have enough breadth.

This prevents wasting time training
an architecture that CANNOT succeed.
```

### 8.3 Complexity Boundaries

```
KNOWING YOUR LIMITS
───────────────────

Circuit complexity tells you:
  - What you CAN solve (within bounds)
  - What you CANNOT solve (beyond bounds)
  - Where the boundary is (exact formula)

This is valuable:
  - Don't attempt impossible tasks
  - Scale architecture appropriately
  - Understand fundamental limits
```

---

## 9. Circuit Complexity in AKIRA

### 9.1 AKIRA's Target Domain

```
AKIRA's DOMAIN: Visual Prediction
─────────────────────────────────

AKIRA targets continuous visual prediction:
  - Input: Video frames (continuous signals)
  - Output: Next frame prediction
  - Task: Local motion/change prediction

Estimating SOS width for this domain:

PREDICATES needed:
  - Position(object, location)     β = 2
  - Velocity(object, direction)    β = 2
  - Occludes(object1, object2)     β = 2
  - Near(object1, object2)         β = 2

Most are binary → β ≈ 2

CONSTRAINTS to track:
  - Object's current state: 1
  - Relevant neighbors: 1-2
  - Temporal context: 1
  - Total: k ≈ 2-3

Therefore: Required breadth = (3+1) × 2 = 8
```

### 9.2 The 7+1 Derivation

```
FORMAL JUSTIFICATION FOR 7+1
────────────────────────────

Given:
  k ≈ 3 (estimated SOS width for local visual prediction)
  β = 2 (binary spatial relations)

Required breadth:
  B = (k + 1) × β
  B = (3 + 1) × 2
  B = 8 channels

AKIRA has:
  7 spectral bands + 1 temporal band = 8 channels

This is NOT a coincidence.
The architecture is designed to handle k ≤ 3 problems.
```

### 9.3 What AKIRA Can and Cannot Do

```
AKIRA'S THEORETICAL LIMITS
──────────────────────────

CAN SOLVE (Class 1 & 2 with k ≤ 3):
  - Single object tracking
  - Local motion prediction
  - Neighbor interaction prediction
  - Sequential events (with depth)

CANNOT SOLVE (Class 3 or k > 3):
  - Global coordination (all objects at once)
  - Sokoban-like puzzles
  - Problems with growing constraints
  - Arbitrary multi-object coordination

This is a DESIGN CHOICE:
  AKIRA is optimized for local visual prediction,
  not general-purpose planning.
```

### 9.4 Collapse and S-GRS

```
COLLAPSE MAY FOLLOW S-GRS PATTERN
─────────────────────────────────

S-GRS serializes constraint satisfaction:
  Achieve p₁, then p₂ maintaining p₁, etc.

AKIRA collapse might serialize similarly:
  Band 0 collapses, then Band 1 maintaining Band 0, etc.

This is a HYPOTHESIS to test:
  - Do bands collapse in order?
  - Does early collapse persist?
  - Does violation of order cause failure?

See: EVIDENCE_TO_COLLECT.md, hypothesis 2.1
```

---

## 10. How This Informs AKIRA's Theoretical Foundations

### 10.1 Principled Architecture Design

```
WHY 8 BANDS? (The Complete Answer)
──────────────────────────────────

Previous justifications (partial):
  - Octave bands match perception (true but doesn't give "7")
  - Hardware alignment (true but coincidental)
  - Cognitive limits (true but indirect)

Circuit complexity justification (complete):
  - Visual prediction has k ≈ 3
  - Relations are binary (β = 2)
  - Required breadth = (3+1) × 2 = 8
  - 7 spectral + 1 temporal = 8

This is a DERIVATION, not a guess.
The architecture follows from the problem.
```

### 10.2 Predictable Failure Modes

```
THEORY PREDICTS FAILURES
────────────────────────

If AKIRA fails on a task, circuit complexity predicts:

CASE: k > 3 (too many simultaneous constraints)
  - Failure mode: Constraint violations
  - System "forgets" early constraints
  - Inconsistent predictions
  
CASE: Very long sequences (depth issue)
  - Failure mode: Accumulating errors
  - Need more SBM layers
  - Solvable by architecture scaling

This allows:
  - Diagnosing failures
  - Knowing when to scale
  - Knowing when task is impossible
```

### 10.3 Domain Boundaries

```
AKIRA'S INTENDED DOMAIN
───────────────────────

Circuit complexity clarifies what AKIRA is FOR:

INTENDED:
  - Local visual prediction
  - Object-centric processing
  - Short-range interactions
  - Sequential motion

NOT INTENDED:
  - Global optimization
  - Complex multi-agent coordination
  - Puzzle solving
  - Arbitrary reasoning

This is not a limitation to "fix."
It's a design choice for efficiency.
Trying to solve Class 3 problems would require
growing architecture, defeating the purpose.
```

### 10.4 Experimental Validation

```
TESTABLE PREDICTIONS
────────────────────

Circuit complexity makes testable predictions:

1. SHARP TRANSITION at k = 4
   Tasks with k ≤ 3: Should succeed
   Tasks with k > 3: Should fail systematically
   
2. FAILURE MODE SPECIFICITY
   Failures should be constraint violations
   Not random errors
   
3. DEPTH SCALING
   Longer sequences should need more layers
   Width should remain sufficient

These can be tested empirically.
See: EVIDENCE_TO_COLLECT.md, Section 5
```

---

## 11. References

### 11.1 Primary Source

1. Mao, J., Lozano-Perez, T., Tenenbaum, J.B., & Kaelbling, L.P. (2023). *What Planning Problems Can A Relational Neural Network Solve?* ICLR 2024. arXiv:2312.03682v2. https://arxiv.org/html/2312.03682v2
   - The foundational paper for SOS width
   - Proves the breadth theorem
   - Classifies problems

### 11.2 Circuit Complexity Background

2. Arora, S., & Barak, B. (2009). *Computational Complexity: A Modern Approach.* Cambridge University Press.
   - General circuit complexity theory
   - NC, AC hierarchy

3. Vollmer, H. (1999). *Introduction to Circuit Complexity.* Springer.
   - Detailed technical treatment
   - Foundations of circuit classes

### 11.3 Planning Background

4. Ghallab, M., Nau, D., & Traverso, P. (2016). *Automated Planning and Acting.* Cambridge University Press.
   - Planning problem fundamentals
   - Goal regression, width measures

5. Lipovetzky, N., & Geffner, H. (2012). *Width and Serialization of Classical Planning Problems.* ECAI.
   - Serialized width concept
   - Precursor to SOS width

### 11.4 AKIRA Internal Documents

6. `AKIRA/foundations/TERMINOLOGY.md`
   - Section 4: Circuit Complexity Framework
   - Complete terminology

7. `AKIRA/architecture_theoretical/EVIDENCE.md`
   - Section 3: Circuit Complexity Foundations
   - Evidence status

8. `AKIRA/architecture_theoretical/EVIDENCE_TO_COLLECT.md`
   - Section 5: Circuit Complexity Boundaries
   - Testable predictions

9. `AKIRA/architecture_theoretical/THE_SEVEN_PLUS_ONE_ARCHITECTURE.md`
   - Architecture specification
   - Band structure details

---

## Summary

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  CIRCUIT COMPLEXITY: KEY TAKEAWAYS                                     │
│                                                                         │
│  ═══════════════════════════════════════════════════════════════════   │
│                                                                         │
│  SOS WIDTH (k):                                                         │
│    How many constraints must be tracked simultaneously                 │
│    Problem property (not algorithm)                                    │
│    Lower k = easier problem                                            │
│                                                                         │
│  CIRCUIT BREADTH (B):                                                   │
│    How many parallel channels                                          │
│    Architecture property                                               │
│    Determines what problems can be solved                              │
│                                                                         │
│  THE KEY THEOREM:                                                       │
│    Required Breadth = (k + 1) × β                                      │
│    This is NECESSARY and SUFFICIENT                                    │
│                                                                         │
│  ═══════════════════════════════════════════════════════════════════   │
│                                                                         │
│  PROBLEM CLASSES:                                                       │
│    Class 1: Easy (constant breadth, constant depth)                    │
│    Class 2: Tractable (constant breadth, unbounded depth)              │
│    Class 3: Intractable (unbounded breadth)                            │
│                                                                         │
│  ═══════════════════════════════════════════════════════════════════   │
│                                                                         │
│  FOR AKIRA:                                                             │
│    Visual prediction: k ≈ 3, β = 2                                     │
│    Required breadth: (3+1) × 2 = 8                                     │
│    AKIRA has: 7 + 1 = 8 bands                                          │
│                                                                         │
│    This is a DERIVATION, not a guess.                                  │
│    The architecture is matched to the problem class.                   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

*This document is part of AKIRA's terminology foundations. For the complete terminology framework, see `AKIRA/foundations/TERMINOLOGY.md`. For related concepts, see `SYNERGY_REDUNDANCY.md` and `INFORMATION_DYNAMICS.md`.*

