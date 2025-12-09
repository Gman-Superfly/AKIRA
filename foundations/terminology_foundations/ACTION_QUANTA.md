# Action Quanta: The Minimum Units of Actionable Information

## Document Purpose

This document explains **Action Quanta (AQ)** - AKIRA's term for the irreducible units of actionable information that emerge during collapse. We build this concept **inductively**, starting from the physics of action and arriving at the complete framework through step-by-step reasoning.

---

## Table of Contents

1. [Building from First Principles: What is Action?](#1-building-from-first-principles-what-is-action)
2. [The Inductive Path to Action Quanta](#2-the-inductive-path-to-action-quanta)
3. [Formal Definition of Action Quanta](#3-formal-definition-of-action-quanta)
4. [AQ Properties: The Four Dimensions](#4-aq-properties-the-four-dimensions)
5. [How AQ Emerge: The Crystallization Process](#5-how-aq-emerge-the-crystallization-process)
6. [The Three-Level Distinction: Measurement, Inference, AQ](#6-the-three-level-distinction-measurement-inference-aq)
7. [AQ vs PID Information Atoms](#7-aq-vs-pid-information-atoms)
8. [AQ Bonded States: How AQ Combine](#8-aq-bonded-states-how-aq-combine)
9. [Physical Parallels: Quasiparticles and BEC](#9-physical-parallels-quasiparticles-and-bec)
10. [AQ in AKIRA Architecture](#10-aq-in-akira-architecture)
11. [Experimental Predictions](#11-experimental-predictions)
12. [Summary](#12-summary)
13. [Note: Representational vs Computational Irreducibility](#13-note-representational-vs-computational-irreducibility)
14. [References](#14-references)
15. [Appendix A: Intuition and Working Notes](#appendix-a-intuition-and-working-notes)

---

## 1. Building from First Principles: What is Action?

### 1.1 Action in Physics

We begin with the physics concept that gives Action Quanta their name.

```
WHAT IS ACTION IN PHYSICS?
──────────────────────────

DEFINITION:
  Action S = ∫ L dt
  
  WHERE:
    L = Lagrangian (kinetic energy - potential energy)
    t = time
    ∫ = integral over the path

UNITS:
  [Action] = [Energy] × [Time] = Joule-seconds (J·s)
  
  Action measures "how much happened" over a trajectory.
  It combines WHAT happened (energy) with HOW LONG (time).

PHYSICAL MEANING:
  - Small action: Not much happened, or it happened briefly
  - Large action: A lot happened, or it happened for a long time
  - Action is the "cost" of a physical process
```

### 1.2 Planck's Constant: The Quantum of Action

```
THE FUNDAMENTAL QUANTUM
───────────────────────

PLANCK'S CONSTANT:
  ℏ = 1.054 × 10⁻³⁴ J·s (reduced Planck constant)
  
  THIS IS THE QUANTUM OF ACTION:
    - The smallest indivisible unit of action
    - ALL quantization in physics derives from this
    - You cannot have less than ℏ of action

IMPLICATIONS:
  When S >> ℏ: Classical behavior (continuous, deterministic)
  When S ~ ℏ:  Quantum behavior (discrete, probabilistic)

THE KEY INSIGHT:
  Action is not continuous - it comes in discrete units.
  The minimum unit is ℏ.
  This is WHY quantum mechanics exists.
```

### 1.3 The Fine-Structure Constant Contains ℏ

```
COUPLING AND THE ACTION QUANTUM
───────────────────────────────

The fine-structure constant:
  α = k_e × e² / (ℏ × c) ≈ 1/137

NOTICE: ℏ appears in the DENOMINATOR

This means:
  - Coupling strength is measured RELATIVE TO the action quantum
  - The quantum of action sets the SCALE for all interactions
  - Larger ℏ → weaker coupling (more action needed per interaction)

THE PATTERN:
  Fundamental physics = (interaction strength) / (action quantum)
  The action quantum is the REFERENCE for everything else.
```

### 1.4 Why This Matters for Information

```
FROM PHYSICAL TO INFORMATIONAL ACTION
─────────────────────────────────────

In physics:
  - Action = How much physically happened
  - Quantum of action = Minimum physical change
  - All physics is built from action quanta

QUESTION: Is there an analogous concept for INFORMATION?

OBSERVATION:
  - Information alone is insufficient for behavior
  - You need information that enables ACTION
  - There should be a minimum unit of "actionable information"

THIS IS WHAT WE SEEK TO DEFINE:
  The Action Quantum (AQ) = Minimum informational pattern
                           that enables correct action
```

---

## 2. The Inductive Path to Action Quanta

### 2.1 Step 1: Raw Data is Not Actionable

```
STEP 1: THE PROBLEM OF RAW DATA
───────────────────────────────

Consider: A camera produces 1 million pixels per frame.

Can you ACT on 1 million pixels directly?
  - Too much data
  - No meaning extracted
  - No decision possible

Raw data is NOT actionable.
Something must be extracted.
```

### 2.2 Step 2: Patterns Enable Action

```
STEP 2: PATTERNS ENABLE ACTION
──────────────────────────────

Consider: From 1 million pixels, you detect "there is an edge here."

Now can you act?
  - YES: You can decide "boundary detected"
  - The pattern "edge" ENABLES the decision
  - You couldn't decide from raw pixels alone

OBSERVATION:
  Patterns extracted from data enable action.
  Raw data does not.
```

### 2.3 Step 3: There is a Minimum Pattern

```
STEP 3: MINIMUM PATTERNS EXIST
──────────────────────────────

Consider: What is the MINIMUM pattern to detect an edge?

TOO LITTLE:
  - 1 pixel: Cannot tell if it's an edge
  - 2 pixels in same region: Still cannot tell
  
JUST ENOUGH:
  - Contrast between adjacent regions
  - Sufficient to discriminate "edge" from "no edge"
  
TOO MUCH:
  - Entire image: More than needed
  - Most information is irrelevant

OBSERVATION:
  There exists a MINIMUM pattern that enables the decision.
  Less than this → cannot act
  This exact pattern → can act
  More than this → redundant
```

### 2.4 Step 4: This Minimum is Task-Relative

```
STEP 4: TASK-RELATIVITY
───────────────────────

The same data has DIFFERENT minimum patterns for DIFFERENT tasks:

TASK: "Is there an edge?"
  Minimum pattern: Contrast gradient
  
TASK: "What color is this region?"
  Minimum pattern: Color distribution
  
TASK: "Is there a face?"
  Minimum pattern: Face feature configuration

OBSERVATION:
  The minimum actionable pattern depends on the TASK.
  Same data → Different AQ for different tasks.
  AQ are defined RELATIVE TO what you need to do.
```

### 2.5 Step 5: This Pattern Cannot Be Reduced Further

```
STEP 5: IRREDUCIBILITY
──────────────────────

KEY PROPERTY: The minimum pattern is IRREDUCIBLE for the task.

MEANING:
  - Cannot simplify further WITHOUT losing ability to act
  - Remove any part → Decision becomes impossible or wrong
  - This is the SMALLEST thing that works

ANALOGY TO PHYSICS:
  ℏ is irreducible: Cannot have less than ℏ of physical action
  AQ is irreducible: Cannot have less actionable information

This irreducibility is REPRESENTATIONAL:
  - The REPRESENTATION cannot be simplified
  - (Different from COMPUTATIONAL irreducibility, which concerns execution)
```

### 2.6 Step 6: Naming the Concept

```
STEP 6: WHY "ACTION QUANTA"?
────────────────────────────

We call these minimum actionable patterns "Action Quanta" because:

1. "ACTION" - They enable action, not just convey information
   - The defining criterion is ACTIONABILITY
   - Not all information is actionable
   - AQ are specifically what you need to ACT

2. "QUANTA" - They are discrete, minimum units
   - Plural of "quantum" = smallest indivisible unit
   - Echoes Planck's quantum of action (ℏ)
   - Emphasizes IRREDUCIBILITY

3. AVOIDING COLLISION
   - "Information atoms" is used in PID (different concept)
   - "Action Quanta" is distinct and physics-aligned

TERMINOLOGY:
  Singular: Action Quantum (one irreducible unit)
  Plural:   Action Quanta (multiple irreducible units)
  Combined: Bonded State (AQ combined together)
```

---

## 3. Formal Definition of Action Quanta

### 3.1 The Definition

```
ACTION QUANTUM: FORMAL DEFINITION
─────────────────────────────────

An Action Quantum (AQ) is:

  THE MINIMUM PATTERN THAT ENABLES CORRECT DECISION

Formally:
  - MINIMUM: Cannot be reduced further for this task
  - PATTERN: Has internal structure (magnitude, phase, freq, coherence)
  - ENABLES: Makes action possible (not just provides information)
  - CORRECT: Leads to appropriate response (task-aligned)
  - DECISION: Discriminates between action alternatives

IRREDUCIBILITY CRITERION:
  AQ are REPRESENTATIONALLY IRREDUCIBLE:
    The representation cannot be simplified further
    without losing the ability to act correctly.
```

### 3.2 The Structure-Function Relationship

```
STRUCTURE AND FUNCTION OF AQ
────────────────────────────

THE CHAIN:
  AQ (pattern) → enables DISCRIMINATION → enables ACTION

STRUCTURAL LEVEL:
  AQ = Minimum PATTERN (what it IS)
  Has properties: magnitude, phase, frequency, coherence

FUNCTIONAL LEVEL:
  Discrimination = What the AQ DOES
  Enables choosing between action alternatives

IMPORTANT DISTINCTION:
  - The AQ IS the pattern
  - The pattern ENABLES discrimination
  - Discrimination IS the atomic abstraction (functional)
  - Discrimination ENABLES action

  ┌─────────────────────────────────────────────────────────────┐
  │  LEVEL      STRUCTURAL           FUNCTIONAL        RESULT  │
  │  ─────      ──────────           ──────────        ──────  │
  │  Atomic     AQ (pattern)    →    Discrimination →  Action  │
  │                                  (atomic abstr.)           │
  │                                                            │
  │  Composed   Bonded state    →    Classification →  Complex │
  │             (pattern combo)      (composed abstr.) action  │
  └─────────────────────────────────────────────────────────────┘
```

### 3.3 Intuitive Example: Edge Detection

```
EXAMPLE: EDGE AQ
────────────────

TASK: Detect edges in an image

RAW DATA: Millions of pixel values
  → Too much, not actionable

SINGLE PIXEL: Not enough
  → Can't tell if it's an edge

THE EDGE PATTERN: The minimum that works
  → A specific pattern of contrast
  → Has magnitude (how strong the edge)
  → Has phase (where the edge is located)
  → Has orientation (which direction)

THIS EDGE PATTERN IS AN ACTION QUANTUM:
  - MINIMUM: Fewer pixels = can't detect edge
  - PATTERN: Not a point, has internal structure
  - ENABLES: Can now decide "edge here" vs "no edge"
  - ACTIONABLE: Can act on this information

Everything smaller → Not actionable
This specific pattern → Actionable
```

---

## 4. AQ Properties: The Four Dimensions

### 4.1 The Four Properties

Every Action Quantum has four properties that characterize it:

```
AQ INTERNAL STRUCTURE
─────────────────────

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  MAGNITUDE                                                             │
│  ─────────                                                              │
│  "How much" of this pattern is present                                 │
│  Signal strength, salience, importance                                 │
│  Higher magnitude = more dominant, more confident                     │
│                                                                         │
│  Physical analog: Amplitude of a wave                                 │
│                                                                         │
│  ════════════════════════════════════════════════════════════════════  │
│                                                                         │
│  PHASE                                                                 │
│  ─────                                                                  │
│  "Where/when" this pattern occurs                                     │
│  Position encoding, temporal location                                  │
│  Determines interference with other AQ                                │
│                                                                         │
│  Physical analog: Phase of a wave                                     │
│                                                                         │
│  ════════════════════════════════════════════════════════════════════  │
│                                                                         │
│  FREQUENCY                                                             │
│  ─────────                                                              │
│  "What scale" this pattern operates at                                │
│  Coarse structure vs fine detail                                      │
│  Which spectral band the AQ belongs to                                │
│                                                                         │
│  Physical analog: Frequency of a wave                                 │
│                                                                         │
│  ════════════════════════════════════════════════════════════════════  │
│                                                                         │
│  COHERENCE                                                             │
│  ─────────                                                              │
│  "How organized" this pattern is                                      │
│  Internal consistency, phase stability                                │
│  Determines ability to bond with other AQ                             │
│                                                                         │
│  Physical analog: Coherence length of a wave                          │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.2 What Each Property Does

```
PROPERTY FUNCTIONS
──────────────────

MAGNITUDE determines:
  - Which AQ dominate the output
  - Confidence in the pattern
  - Salience for attention mechanisms

PHASE determines:
  - Spatial position (where)
  - Temporal position (when)
  - Interference behavior (constructive/destructive)

FREQUENCY determines:
  - Scale of the pattern
  - Level of detail
  - Which band processes it

COHERENCE determines:
  - Stability of the pattern
  - Ability to bond with other AQ (see Section 8)
  - Reliability for action
```

### 4.3 The Pattern Itself

```
BEYOND PROPERTIES: THE PATTERN CONTENT
──────────────────────────────────────

The four properties are METADATA about the AQ.
The AQ itself IS the pattern.

EXAMPLE: An edge AQ
  Properties:
    Magnitude: 0.8 (strong edge)
    Phase: (45, 120) (at this spatial location)
    Frequency: Band 4 (fine detail scale)
    Coherence: 0.95 (very stable pattern)
  
  Pattern:
    "Horizontal contrast gradient"
    (The actual structure that the properties describe)

Properties = How to characterize the AQ
Pattern = What the AQ actually is
```

---

## 5. How AQ Emerge: The Crystallization Process

### 5.1 The Emergence Process

AQ emerge through a process we call **crystallization**:

```
AQ CRYSTALLIZATION
──────────────────

BEFORE COLLAPSE (Superposition):
┌────────────────────────────────────────────────────────────────────┐
│                                                                    │
│  AQ are POTENTIAL                                                 │
│                                                                    │
│  - Multiple patterns could form                                   │
│  - Magnitude uncertain                                            │
│  - Phase relationships fluid                                      │
│  - Coherence low (not locked)                                     │
│  - REDUCIBLE: Could decompose into other configurations          │
│                                                                    │
│  Analogy: Water molecules in liquid state                         │
│           Can arrange many ways                                   │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘

DURING COLLAPSE:
┌────────────────────────────────────────────────────────────────────┐
│                                                                    │
│  AQ are CRYSTALLIZING                                             │
│                                                                    │
│  - Certain patterns selected by competition                       │
│  - Magnitude settles to definite values                          │
│  - Phase relationships lock                                       │
│  - Coherence increases                                            │
│  - Becoming irreducible                                           │
│                                                                    │
│  Analogy: Water molecules arranging into ice crystal              │
│           One configuration being selected                        │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘

AFTER COLLAPSE (Crystallized):
┌────────────────────────────────────────────────────────────────────┐
│                                                                    │
│  AQ are CRYSTALLIZED                                              │
│                                                                    │
│  - Specific patterns exist                                        │
│  - Magnitude definite                                             │
│  - Phase relationships stable                                     │
│  - Coherence high (locked)                                        │
│  - IRREDUCIBLE: Cannot reduce without losing actionability       │
│                                                                    │
│  Analogy: Ice crystal with fixed structure                        │
│           Cannot remove one atom without breaking crystal         │
│           Stable, actionable, MINIMUM needed                      │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

### 5.2 Why "Crystallized"?

```
WHY THIS TERM IS CORRECT
────────────────────────

We choose "crystallized" because crystals share key properties with AQ:

1. FORMED THROUGH PHASE TRANSITION
   Crystal: Liquid → Solid at freezing point
   AQ: Superposition → Crystallized at collapse threshold
   
2. ORDERED AND STABLE
   Crystal: Atoms locked in lattice positions
   AQ: Patterns locked with definite properties
   
3. IRREDUCIBLE
   Crystal: Remove one atom → Structure breaks
   AQ: Remove any part → Lose actionability
   
This is the KEY property: IRREDUCIBILITY
A crystallized AQ is the minimum that works.
```

### 5.3 Selection Mechanism

```
WHICH PATTERNS BECOME AQ?
─────────────────────────

Not all potential patterns crystallize. Selection based on:

1. MAGNITUDE (Dominance)
   Strong patterns suppress weak ones
   High-magnitude patterns "win" competition

2. COHERENCE (Stability)
   Coherent patterns persist through processing
   Incoherent patterns dissolve

3. MUTUAL SUPPORT (Resonance)
   Patterns that reinforce each other survive
   Conflicting patterns annihilate via interference

4. TASK RELEVANCE (Utility)
   Patterns useful for prediction strengthen
   Irrelevant patterns are not reinforced

RESULT: The minimal set of actionable patterns survives.
```

### 5.4 Failure Modes

```
WHEN CRYSTALLIZATION FAILS
──────────────────────────

NO COLLAPSE (Stuck in superposition):
  - AQ never crystallize
  - System remains uncertain
  - No actionable output
  Cause: Insufficient evidence, conflicting signals

PREMATURE COLLAPSE (Dark attractor):
  - AQ crystallize too early
  - Wrong patterns selected
  - Confident but incorrect
  Cause: Threshold too low, misleading patterns

FRAGMENTATION:
  - Too many small AQ
  - Don't form coherent representation
  - Scattered, unusable output
  Cause: Lack of integration across scales
```

---

## 6. The Three-Level Distinction: Measurement, Inference, AQ

### 6.1 The Three Levels

A critical insight: Measurement, Inference, and AQ are THREE DIFFERENT THINGS.

```
THREE DISTINCT LEVELS
─────────────────────

┌───────────────┬────────────────────┬─────────────────────────────────┐
│ LEVEL         │ EXAMPLE            │ WHAT IT IS                      │
├───────────────┼────────────────────┼─────────────────────────────────┤
│ MEASUREMENT   │ +2000 Hz Doppler   │ Physical quantity from sensor   │
│               │                    │ (DATA)                          │
├───────────────┼────────────────────┼─────────────────────────────────┤
│ INFERENCE     │ "Approaching fast" │ Interpretation in context       │
│               │                    │ (MEANING)                       │
├───────────────┼────────────────────┼─────────────────────────────────┤
│ AQ            │ "CLOSING RAPIDLY"  │ Minimum PATTERN enabling        │
│               │                    │ discrimination (FOR ACTION)     │
└───────────────┴────────────────────┴─────────────────────────────────┘

MEASUREMENT ≠ AQ:
  "+2000 Hz" is data. The AQ is the pattern that makes it actionable.

INFERENCE ≠ AQ:
  "Approaching fast" is interpretation. 
  The AQ is the PATTERN that enables discrimination.

AQ = MINIMUM PATTERN FOR ACTION:
  What survives when you ask: "What is the MINIMUM I need to act correctly?"
```

### 6.2 Detailed Example: Radar Doppler

This example shows the COMPLETE chain from raw signal to crystallized AQ:

```
DETAILED EXAMPLE: DOPPLER AQ IN RADAR
─────────────────────────────────────

Context: A radar system detects a +2000 Hz Doppler shift.
Question: What is the AQ here? How does it emerge?


LEVEL 0: RAW SIGNAL (Before any processing)
───────────────────────────────────────────
What exists: RF waveform reflected from something
  - Billions of voltage samples
  - Contains: target return + noise + clutter + interference
  - NO meaning yet - just physical signal


LEVEL 1: BELIEF STATE (Superposition - High Synergy)
────────────────────────────────────────────────────
After initial processing, the system has a BELIEF STATE:
  
  The signal COULD be:
    - A target approaching at ~300 m/s (high positive Doppler)
    - A target receding at ~300 m/s (negative Doppler misread)
    - Sea clutter with spreading (broad Doppler spectrum)
    - Noise spike (random)
    - Jamming (artificial)
    - Multiple targets (ambiguous)
  
  SYNERGY: Information is DISTRIBUTED across possibilities
    - No single measurement resolves the ambiguity
    - Must COMBINE frequency, amplitude, timing, angle
    - Must integrate over time, space, context
    
  This is the SUPERPOSITION of belief:
    |belief⟩ = a|approaching⟩ + b|receding⟩ + c|clutter⟩ + d|noise⟩ + ...


LEVEL 2: COLLAPSE (Synergy → Redundancy)
─────────────────────────────────────────
What causes collapse?

  1. DOPPLER PROCESSING (FFT)
     - Transforms time-domain to frequency-domain
     - Reveals velocity spectrum
     - Concentrates energy at specific Doppler bin
     
  2. THRESHOLD DETECTION (CFAR)
     - Compares signal to local noise estimate
     - Eliminates sub-threshold alternatives
     - "This is definitely something, not noise"
     
  3. CONSISTENCY CHECKS
     - Does Doppler match expected target behavior?
     - Is it consistent with range rate?
     - Does it persist across pulses?
     
  4. CONTEXT (Top-down)
     - "We're looking for fast threats"
     - "Sea-skimmers have this signature"
     - Prior knowledge biases the collapse

  COLLAPSE RESULT:
    Synergy → Redundancy
    Multiple possibilities → Single interpretation
    |belief⟩ → |+2000 Hz Doppler, high confidence⟩


LEVEL 3: THE AQ CRYSTALLIZES
────────────────────────────
The AQ is NOT the number (+2000 Hz).
The AQ is NOT the words ("approaching very fast").

The AQ is the IRREDUCIBLE PATTERN that enables ACTION:

  AQ = { pattern that discriminates "URGENT" from "NOT URGENT" }

  This AQ enables:
    - Prioritization: "Handle this first"
    - Time estimation: "Seconds, not minutes"
    - Threat classification: "High kinetic threat"
    
  The AQ has the four properties:
    - MAGNITUDE: How fast (degree of urgency)
    - PHASE: Direction relative to us (toward/away)
    - FREQUENCY: Coarse (strategic) vs fine (tactical)
    - COHERENCE: Confidence in the measurement
    
  Without this AQ, you cannot correctly prioritize.
  WITH this AQ, you can act appropriately.
  
  That is why it is REPRESENTATIONALLY IRREDUCIBLE:
    - Cannot simplify further without losing actionability
    - "+2000 Hz" vs "+50 Hz" IS the difference between
      "ENGAGE NOW" and "continue tracking"
```

### 6.3 The Complete Chain

```
THE COMPLETE PROCESSING CHAIN
─────────────────────────────

  Raw signal (no meaning)
      │
      ▼
  Belief state (superposition, high synergy)
      │
      ▼
  Processing (collapse, synergy → redundancy)
      │
      ▼
  Measurement (+2000 Hz)
      │
      ▼
  Inference ("approaching very fast")
      │
      ▼
  AQ crystallizes: "CLOSING RAPIDLY" (actionable discrimination)
      │
      ▼
  AQ bonds with others → "ANTI-SHIP MISSILE INBOUND" (bonded state)
      │
      ▼
  Action program executes: "ENGAGE IMMEDIATELY"
```

---

## 7. AQ vs PID Information Atoms

### 7.1 The Critical Distinction

```
TWO DIFFERENT CONCEPTS
──────────────────────

PID INFORMATION ATOMS (Williams & Beer, 2010):
  What they are: Decomposition terms
  What they measure: How information is DISTRIBUTED
  Categories: Redundancy, Unique, Synergy
  Nature: Analytical (we decompose into them)
  Static: Snapshot at one moment

ACTION QUANTA (AKIRA):
  What they are: Emergent patterns
  What they measure: ACTIONABLE units
  Categories: By properties (magnitude, phase, frequency, coherence)
  Nature: Emergent (they crystallize from dynamics)
  Dynamic: Result from collapse process
```

### 7.2 How They Relate

```
THE RELATIONSHIP
────────────────

PID atoms describe the DISTRIBUTION of information.
AQ are the RESULT of processing that information.

BEFORE COLLAPSE:
  Information distributed as synergy + redundancy + unique
  AQ exist only as potential

AFTER COLLAPSE:
  Information concentrated (high redundancy)
  AQ crystallize as patterns

PID atoms → Analysis tool (how to decompose)
AQ → Product of computation (what emerges)

See: SYNERGY_REDUNDANCY.md for PID atoms in detail
```

### 7.3 Example Showing the Difference

```
EXAMPLE: Two Sensor Readings
────────────────────────────

S₁ = Temperature sensor
S₂ = Humidity sensor
T = Will it rain?

PID ANALYSIS (how information is distributed):
  I_red = 0.3 bits (both indicate general weather)
  I_uni(S₁) = 0.2 bits (temp-specific info)
  I_uni(S₂) = 0.4 bits (humidity-specific info)
  I_syn = 0.1 bits (interaction effect)
  
  This tells us HOW info is distributed across sources.

AQ ANALYSIS (what crystallizes for action):
  After processing:
  AQ₁ = "Weather state" pattern
    Magnitude: 0.9, Phase: now, Frequency: coarse, Coherence: 0.85
  
  This is the ACTIONABLE RESULT.
  "It will rain" - the crystallized prediction.

PID = Analysis of distribution (how information sits)
AQ = Result of processing (what emerges for action)
```

---

## 8. AQ Bonded States: How AQ Combine

### 8.1 AQ Can Combine

Individual AQ are atomic units. They can combine into **bonded states**:

```
AQ BONDED STATES
────────────────

SINGLE AQ: Simple pattern (crystallized, irreducible)
  - One edge
  - One color region
  - One motion vector

BONDED STATE: Multiple AQ combined
  - Multiple edges → Shape
  - Multiple colors → Object identity
  - Multiple motions → Action sequence

The bonded state is MORE than sum of parts:
  - A bound configuration of crystallized AQ
  - Represents a concept or composed abstraction
  - Has emergent properties
```

### 8.2 Types of Bonds

```
THREE BOND TYPES
────────────────

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  COHERENT BONDS (Phase Alignment)                                      │
│  ────────────────────────────────                                       │
│  AQ at SAME frequency, ALIGNED phase                                   │
│  Result: Constructive interference                                     │
│                                                                         │
│  Example: Multiple edges forming a contour                             │
│    Edge₁ (phase 0°) + Edge₂ (phase 0°) + Edge₃ (phase 0°)             │
│    = Coherent contour pattern                                          │
│                                                                         │
│  ════════════════════════════════════════════════════════════════════  │
│                                                                         │
│  COMPLEMENTARY BONDS (Magnitude Exchange)                              │
│  ─────────────────────────────────────────                              │
│  AQ at DIFFERENT frequencies, FILLING GAPS                             │
│  Result: Complete picture across scales                                │
│                                                                         │
│  Example: Low-freq identity + high-freq position                       │
│    "Face" (low-freq) + "At location X" (high-freq)                    │
│    = Located face                                                      │
│                                                                         │
│  ════════════════════════════════════════════════════════════════════  │
│                                                                         │
│  HIERARCHICAL BONDS (Frequency Bridging)                               │
│  ────────────────────────────────────────                               │
│  Coarse AQ CONTEXTUALIZES fine AQ                                      │
│  Result: Part-whole relationships                                      │
│                                                                         │
│  Example: Face contains eye                                            │
│    "Face" (low-freq, large) CONTAINS "Eye" (high-freq, detail)        │
│    = Structured face representation                                    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 8.3 Bonded State Examples

```
EXAMPLE BONDED STATES
─────────────────────

"RING" bonded state:
  Blob_AQ (Band 2, coarse shape)
  + Boundary_AQ (Band 4, edge around)
  + Curvature_AQ (Band 5, roundness)
  = Ring concept

"MOVING OBJECT" bonded state:
  Object_AQ (Band 3, shape)
  + Position_AQ (Band 5, location)
  + Velocity_AQ (Temporal band, motion)
  = Moving object concept

"ANTI-SHIP MISSILE INBOUND" bonded state:
  CLOSING_RAPIDLY_AQ
  + CLOSE_AQ
  + SEA_SKIMMING_AQ
  + SMALL_TARGET_AQ
  + ROCKET_MOTOR_AQ
  = Threat concept → Enables action: "ENGAGE IMMEDIATELY"
```

---

## 9. Physical Parallels: Quasiparticles and BEC

### 9.1 AQ as Quasiparticles

```
AQ AS QUASIPARTICLES
────────────────────

In condensed matter physics:
  - Quasiparticles are EMERGENT excitations
  - Not fundamental, but behave as particles
  - Examples: Phonons (sound), polarons (charge+lattice), magnons (spin)

Properties of quasiparticles:
  - Emerge from collective behavior of many particles
  - Have particle-like properties (effective mass, momentum)
  - Can interact, scatter, decay
  - Are REAL in their effects on the system

AQ are analogous:
  - Emerge from belief field dynamics
  - Have AQ properties (magnitude, phase, frequency, coherence)
  - Can interact (bond), transform
  - Are REAL for computation

AQ = Quasiparticles of the belief field
```

### 9.2 BEC Analogy (Suggestive, Not Proven)

```
BEC ANALOGY FOR INTUITION
─────────────────────────

NOTE: This section describes an ANALOGY that may aid intuition.
It is NOT a claim that attention "is" a BEC or belongs to any
universality class. The analogy may break down in important ways.

In Bose-Einstein Condensate (BEC):
  Above T_c: Particles distributed thermally (high entropy)
  Below T_c: Particles condense to ground state (low entropy)
  
In AKIRA (by loose analogy):
  High temperature: Attention distributed (high entropy)
  Low temperature: Attention concentrated (low entropy)
  
SUGGESTIVE PARALLEL (not equivalence):
  BEC condensation ↔ Attention concentration
  Quasiparticles ↔ Action Quanta (by analogy)
  
WHY THIS IS ONLY AN ANALOGY:
  - BEC requires infinite-system thermodynamic limits
  - Attention operates on finite sequences
  - No phase transition has been proven for attention
  - Critical exponents have not been measured
```

### 9.3 Structural Similarity (Analogy Only)

```
STRUCTURAL SIMILARITY
─────────────────────

NOTE: This describes a mathematical similarity in form, 
NOT an equivalence or claim of shared universality class.

BEC (physical system):
  State: wave function ψ
  Self-interaction: g|ψ|²ψ
  
Attention (computational system):
  State: representation X
  Self-interaction-like: softmax(XX^T)X

BOTH have form: (self-interaction) × state

This structural similarity:
  - May provide useful intuition
  - Does NOT prove shared physics
  - Does NOT establish universality class membership
  - Should be treated as a hypothesis to test, not a fact

The equations are analogous in form.
The systems are fundamentally different.
Whether the analogy is deep or superficial is unknown.
```

---

## 10. AQ in AKIRA Architecture

### 10.1 AQ Across Bands

```
BAND-SPECIFIC AQ
────────────────

Each band extracts AQ at different scales:

BAND 0 (Lowest frequency):
  AQ type: Global structure
  Example: "Scene is indoor/outdoor"
  Properties: High coherence, low spatial detail

BAND 1-2:
  AQ type: Major shapes
  Example: "Large blob at center"
  Properties: Object-level patterns

BAND 3-4:
  AQ type: Features
  Example: "Edge with this orientation"
  Properties: Part-level patterns

BAND 5-6 (Highest frequency):
  AQ type: Fine details
  Example: "Texture at this location"
  Properties: High detail, more position-specific

TEMPORAL BAND:
  AQ type: Motion/change
  Example: "Moving rightward"
  Properties: Velocity, acceleration, dynamics
```

### 10.2 Cross-Band AQ Bonding

```
HOW BANDS COMBINE INTO BONDED STATES
────────────────────────────────────

Wormhole attention enables cross-band bonding:

Band 0 ↔ Band 6:
  Coarse structure contextualizes fine detail
  "This detail belongs to this object"

Band 1 ↔ Band 5:
  Shape and position combine
  "Shape X is at location Y"

Band 3 ↔ Band 4:
  Adjacent scales integrate
  "These features form this part"

All bands → Temporal:
  Spatial patterns get motion
  "This pattern is moving"
```

### 10.3 AQ in the Processing Cycle

```
AQ LIFECYCLE IN AKIRA
─────────────────────

INPUT:
  Raw signal enters
  No AQ yet (just data)

DECOMPOSITION:
  Signal split into spectral bands
  Potential AQ patterns appear
  Magnitude/phase extracted per band

TENSION:
  Multiple potential AQ compete
  Phase relationships fluid
  High synergy between bands

COLLAPSE:
  Winning patterns selected
  Phase relationships lock
  AQ crystallize (become irreducible)
  Cross-band bonded states form

OUTPUT:
  Crystallized AQ represent prediction
  Bonded states represent concepts
  Ready for action/further processing
```

### 10.4 Connection to Attention Coupling Factor

```
AQ AND THE COUPLING FACTOR
──────────────────────────

From ATTENTION_COUPLING_FACTOR.md:

The attention coupling factor β = 1/√d_k determines:
  - How strongly patterns interact
  - How quickly crystallization occurs
  - The "sharpness" of AQ selection

HIGH β (strong coupling):
  - AQ crystallize quickly
  - Sharp selection (winner-take-all)
  - Risk: Premature commitment

LOW β (weak coupling):
  - AQ crystallize slowly
  - Soft selection (multiple patterns persist)
  - Risk: Never committing

The coupling factor is the "fine-structure constant" of AQ dynamics.
```

---

## 11. Experimental Predictions

### 11.1 Testable Predictions

```
TESTABLE AQ PREDICTIONS
───────────────────────

1. AQ SHOULD BE CONSISTENT
   Same input → Same AQ (after training)
   Test: Measure AQ variability across runs
   Prediction: Low variance for well-trained models

2. AQ SHOULD CORRELATE WITH TASK
   Task-relevant features → Strong AQ
   Irrelevant features → Weak AQ
   Test: Compare AQ to task requirements
   Prediction: AQ magnitude tracks relevance

3. AQ COUNT SHOULD BE BOUNDED
   ~7±2 AQ per bonded state (working memory limit?)
   Test: Count AQ in typical bonded states
   Prediction: Consistent with cognitive constraints

4. BONDED STATES SHOULD FORM SMALL-WORLD
   AQ connection graph: High clustering, short paths
   Test: Analyze AQ co-activation patterns
   Prediction: Small-world network topology
```

### 11.2 Evidence from 035 Experiments

```
EXPERIMENTAL EVIDENCE
─────────────────────

From 035A (AQ Excitation Detection):
  - AQ cluster by ACTION TYPE, not semantic content
  - Sentiment opposites cluster together (same discrimination needed)
  - Evidence score: 3/4 (strong)

From 035D (Bonded State Decomposition):
  - Component AQ detectable within bonded states
  - Four-bond ratio at best layer: 1.50x
  - Evidence score: 3/3 (strong)

From 035F (Compositional Bonding):
  - Component AQ detection: Cohen's d = 1.7-2.0
  - Cross-model replication (GPT-2, Pythia)
  - Confirms AQ are compositional

From 035G (Belief Crystallization):
  - Three regimes: Diffuse, Polarized, Crystallized
  - Hedging peaks at Ambiguous (37.4%)
  - Crystallization is not linear with evidence

See: 035_EXP_AQ_EXCITATION_FIELDS for full results
```

---

## 12. Summary

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  ACTION QUANTA: KEY TAKEAWAYS                                          │
│                                                                         │
│  ═══════════════════════════════════════════════════════════════════   │
│                                                                         │
│  THE INDUCTIVE PATH:                                                   │
│    1. Physical action has a quantum (ℏ)                               │
│    2. Raw data is not actionable                                       │
│    3. Patterns enable action                                           │
│    4. There exists a MINIMUM pattern for each task                    │
│    5. This minimum is IRREDUCIBLE (cannot reduce without losing action)│
│    6. We call this the Action Quantum (AQ)                            │
│                                                                         │
│  ═══════════════════════════════════════════════════════════════════   │
│                                                                         │
│  DEFINITION:                                                            │
│    The minimum pattern that enables correct decision                   │
│    IRREDUCIBLE for the task (cannot reduce without losing action)     │
│                                                                         │
│  PROPERTIES:                                                            │
│    Magnitude (how much)                                                │
│    Phase (where/when)                                                  │
│    Frequency (what scale)                                              │
│    Coherence (how organized)                                           │
│                                                                         │
│  ═══════════════════════════════════════════════════════════════════   │
│                                                                         │
│  EMERGENCE (Crystallization):                                           │
│    Before collapse: AQ are potential (reducible)                       │
│    During collapse: AQ crystallize (becoming irreducible)              │
│    After collapse: AQ are CRYSTALLIZED (irreducible, stable)          │
│                                                                         │
│  THREE-LEVEL DISTINCTION:                                               │
│    Measurement: Physical quantity (DATA)                               │
│    Inference: Interpretation (MEANING)                                 │
│    AQ: Minimum pattern for action (ACTIONABLE)                        │
│                                                                         │
│  ═══════════════════════════════════════════════════════════════════   │
│                                                                         │
│  BONDED STATES:                                                         │
│    Multiple crystallized AQ can combine                                │
│    Bond types: Coherent, Complementary, Hierarchical                  │
│    Bonded states represent composed abstractions                      │
│                                                                         │
│  ═══════════════════════════════════════════════════════════════════   │
│                                                                         │
│  IN AKIRA:                                                              │
│    Each band extracts scale-specific AQ                                │
│    Wormholes enable cross-band bonding                                 │
│    Collapse crystallizes AQ from potential to actual                   │
│    Crystallized AQ are what AKIRA computes                             │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 13. Note: Representational vs Computational Irreducibility

```
TWO LEVELS OF IRREDUCIBILITY
────────────────────────────

LEVEL 1: REPRESENTATIONAL IRREDUCIBILITY (AQ)
  - Individual Action Quanta are representationally irreducible
  - The REPRESENTATION cannot be simplified further
  - Minimum pattern needed for correct action
  - This is what crystallization produces

LEVEL 2: COMPUTATIONAL IRREDUCIBILITY (Programs from AQ)
  - Programs/behaviors CONSTRUCTED FROM AQ
  - The EXECUTION cannot be shortcut (Wolfram's sense)
  - Must run to know outcome

THE CONNECTION:
  1. AQ crystallize → representationally irreducible primitives
  2. AQ combine → bonded states (composed abstractions)
  3. Bonded states compose → behavioral sequences (programs)
  4. Those programs may exhibit computational irreducibility
     (cannot predict outcome without executing)

KEY INSIGHT:
  AQ are the BUILDING BLOCKS (representationally irreducible).
  What you BUILD from them (complex behaviors, plans, reasoning)
  could be computationally irreducible in Wolfram's sense.

  Representational irreducibility → what the units ARE
  Computational irreducibility → what the compositions DO
```

---

## 14. References

### 14.1 Physics Background

1. Planck, M. (1901). "On the Law of Distribution of Energy in the Normal Spectrum." Annalen der Physik.
   - The quantum of action

2. Pitaevskii, L., & Stringari, S. (2003). *Bose-Einstein Condensation.* Oxford University Press.
   - BEC theory, quasiparticles

3. Pines, D. (1999). *Elementary Excitations in Solids.* Perseus Books.
   - Quasiparticle physics

### 14.2 Information Theory

4. Williams, P.L., & Beer, R.D. (2010). *Nonnegative Decomposition of Multivariate Information.*
   - PID information atoms (different from AQ)

5. See `SYNERGY_REDUNDANCY.md` for PID framework.

### 14.3 Attention Mechanisms

6. Vaswani, A., et al. (2017). *Attention Is All You Need.* NeurIPS.
   - Self-attention structure

### 14.4 AKIRA Internal Documents

7. `ATTENTION_COUPLING_FACTOR.md` - The "fine-structure constant" of attention
8. `COHERENCE.md` - Phase alignment and interference
9. `COLLAPSE_TENSION.md` - The collapse mechanism
10. `SUPERPOSITION_WAVES_CRYSTALLIZATION_PARTICLES.md` - The duality
11. `TERMINOLOGY.md` - Complete framework
12. `035_EXP_AQ_EXCITATION_FIELDS/` - Experimental evidence

---

## Appendix A: Intuition and Working Notes

*This appendix collects less formal intuitions, quick-reference summaries, and working notes that may help build understanding. These are not rigorous definitions but rather aids to thinking about AQ.*

### A.1 Quick Reference Box

```
ACTION QUANTA: QUICK REFERENCE
──────────────────────────────

TERMINOLOGY:
  Singular = Action Quantum (ONE irreducible unit)
  Plural   = Action Quanta (MULTIPLE irreducible units)
  Combined = Bonded State (AQ bonded together)

  Bonded states are the STRUCTURAL implementation of
  FUNCTIONAL abstractions.

THE INDUCTION CHAIN (short form):
  1. Superposition of belief (wave-like, distributed)
  2. Collapse (phase transition)
  3. AQ emerge (crystallized belief)
  4. AQ guide correct action (their function)

AQ ARE:
  WHAT: Crystallized belief (form belief takes after collapse)
  WHY:  To enable correct action (their purpose)
  HOW:  Through collapse from superposition (the process)

IMPORTANT: AQ is NOT the abstraction.
           AQ ENABLES the abstraction (discrimination).
           Discrimination IS the functional abstraction.
```

### A.2 The Actionability Criterion

```
WHY "ACTION" IS THE DEFINING FEATURE
────────────────────────────────────

Information alone is not enough.
You need ACTIONABLE information.

Example: Predicting rain
  "The humidity is 73.4%" - Information (but what do you DO with it?)
  "It will rain" - Actionable information (bring umbrella!)
  
The actionable version:
  - Enables decision (bring umbrella or not?)
  - Is the minimum needed (don't need humidity to 5 decimal places)
  - Is structured (yes/no, not continuous probability)

AQ are defined by actionability, not information content.
This is why we don't call them "information atoms."
```

### A.3 AQ as Compression

```
AQ AS COMPRESSION
─────────────────

Raw input: Very high dimensional
  Example: 256×256×3 image = 196,608 values

AQ output: Low dimensional
  Example: 10-20 Action Quanta

Compression ratio: ~10,000:1

But this isn't lossy compression in the usual sense:
  - AQ preserve ACTIONABLE information
  - Irrelevant details discarded
  - Task-relevant structure kept

AQ = Task-relevant compression

This explains why neural networks can work:
They learn to extract AQ, not to memorize data.
```

### A.4 Attention as AQ Selection

```
ATTENTION SELECTS AQ
────────────────────

What does attention actually do?

Standard view (ML textbooks):
  "Attention weights important tokens"

AQ view (AKIRA):
  "Attention selects which patterns crystallize"

The mechanism:
  High attention → Pattern survives, becomes AQ
  Low attention → Pattern dissolves, does not crystallize

Attention IS the selection mechanism for AQ.

This is why attention patterns are interpretable:
They show you WHAT the model considers actionable.
```

### A.5 AQ and Generalization

```
AQ AND GENERALIZATION
─────────────────────

Why do neural networks generalize?

Standard view:
  "Learn smooth functions that interpolate"

AQ view:
  "Learn to extract the SAME AQ from similar inputs"
  
  Similar inputs → Same AQ → Same output
  
  Generalization = Consistent AQ extraction

A well-trained network extracts consistent AQ
across variations in the input.
That's what generalization IS.

This explains why:
  - Overfitting = learning input-specific patterns, not true AQ
  - Regularization = forcing AQ to be simpler (more general)
  - Data augmentation = forcing same AQ across variations
```

### A.6 Key Insight from the Radar Example

```
KEY INSIGHT FROM RADAR EXAMPLE
──────────────────────────────

The radar example (Section 6.2) makes explicit:

1. MEASUREMENT ≠ AQ
   "+2000 Hz" is data. 
   The AQ is the pattern that makes it actionable.
   You could express the same AQ from different measurements.
   
2. INFERENCE ≠ AQ
   "Approaching fast" is interpretation (linguistic). 
   The AQ is the PATTERN that enables discrimination.
   The words are not the AQ; the pattern is.
   
3. AQ = MINIMUM PATTERN FOR ACTION
   What survives is the minimum PATTERN you NEED to act correctly.
   
   STRUCTURAL: Pattern (has magnitude, phase, frequency, coherence)
   FUNCTIONAL: Enables discrimination (distinguishes action alternatives)
   
4. AQ ENABLES BONDING
   "CLOSING RAPIDLY" alone is one AQ.
   Combined with "CLOSE!", "SEA-SKIMMING", "SMALL", "ROCKET MOTOR"...
   ...forms the bonded state: "ANTI-SHIP MISSILE INBOUND"
   
5. BONDED STATE ENABLES COMPLEX ACTION
   The bonded state is itself a composed abstraction.
   It enables the action program: "ENGAGE IMMEDIATELY"

See: RADAR_ARRAY.md for full treatment of radar parallels.
```

### A.7 Observing AQ in Practice

```
HOW TO DETECT AQ
────────────────

AQ are observable through several methods:

1. MAGNITUDE MAPS
   High magnitude = Strong AQ
   Plot magnitude across bands/positions
   Strong peaks indicate crystallized AQ

2. PHASE COHERENCE
   Locked phase = Crystallized AQ
   Measure phase stability over time/layers
   High coherence = stable AQ

3. CROSS-BAND CORRELATION
   Correlated bands = Bonded states forming
   Track inter-band relationships
   High correlation = AQ bonding

4. ENTROPY MONITORING
   Low entropy = Collapsed state
   AQ present when output entropy drops
   High entropy = still in superposition

5. ATTENTION PATTERN ANALYSIS
   Focused attention = AQ selection happening
   Diffuse attention = no clear AQ yet

See: AKIRA/observability/OBSERVABILITY_EMBEDDINGS.md
```

### A.8 The Physics Parallel in Plain Language

```
WHY THE PHYSICS PARALLEL MATTERS
────────────────────────────────

In physics:
  ℏ (Planck's constant) is THE quantum of action.
  All of physics is built on this irreducible unit.
  You cannot have "half an ℏ" of action.
  
  The fine-structure constant α ≈ 1/137 measures
  electromagnetic coupling RELATIVE TO ℏ.

In AKIRA:
  AQ are our quantum of informational action.
  All computation is built from these irreducible units.
  You cannot have "half an AQ" for a given task.
  
  The attention coupling factor β = 1/√d_k measures
  attention strength RELATIVE TO information dimension.

This parallel is not decoration:
  - It guides intuition (what to expect)
  - It suggests experiments (what to measure)
  - It provides mathematical structure (how to formalize)

The math is analogous. The physics is different.
But the STRUCTURE is the same.
```

### A.9 Common Misunderstandings

```
WHAT AQ ARE NOT
───────────────

1. AQ are NOT neurons or activations
   AQ are PATTERNS that emerge from many neurons.
   A single neuron is not an AQ.

2. AQ are NOT tokens or words
   AQ are task-relative patterns, not linguistic units.
   The same word may correspond to different AQ in different contexts.

3. AQ are NOT fixed features
   AQ depend on the task. Change the task, change the AQ.
   "Edge" is only an AQ if the task requires edge detection.

4. AQ are NOT learned explicitly
   We don't train a network "to produce AQ."
   AQ emerge from training on tasks.
   They are what crystallizes, not what we specify.

5. AQ are NOT always present
   If the system is stuck in superposition, AQ don't crystallize.
   Failed collapse = no AQ = no action possible.
```

### A.10 Why This Framework?

```
WHY BOTHER WITH AQ?
───────────────────

The AQ framework provides:

1. A UNIT OF ANALYSIS
   Instead of "what did the network compute?"
   We ask "what AQ crystallized?"
   This is more tractable.

2. A BRIDGE TO PHYSICS
   Quasiparticle intuitions apply.
   Phase transition intuitions apply.
   We can borrow tools and concepts.

3. TESTABLE PREDICTIONS
   AQ should be consistent (same input → same AQ)
   AQ should be task-relevant (magnitude tracks utility)
   AQ should bond predictably (coherent phases combine)
   
   These are empirically checkable.

4. DESIGN GUIDANCE
   If we understand AQ, we can:
   - Design architectures that crystallize AQ cleanly
   - Diagnose failures (stuck in superposition, premature collapse)
   - Tune parameters (coupling factor, threshold)

The goal: Make neural computation understandable
         in terms of discrete, actionable units.
```

---

*Oscar Goldman — Shogu Research Group @ Datamutant.ai*

*"Just as Planck's constant ℏ is the quantum of physical action - the smallest indivisible unit of 'something happening' in physics - the Action Quantum is the quantum of informational action: the smallest indivisible pattern that enables correct decision. Both represent irreducibility: you cannot have less and still have anything meaningful."*

---

*If you use this framework in your research, please cite it. This is ongoing work - we would like to know your opinions and experiments.*

*Authors: Oscar Goldman - Shogu Research Group @ Datamutant.ai subsidiary of Wenshin Heavy Industries*
