# The Role and Effects of Coherence

## How Phase Alignment Determines What Survives

**Oscar Goldman — Shogu Research Group @ Datamutant.ai**

---

## Table of Contents

1. [What This Document Explains](#1-what-this-document-explains)
2. [The Starting Point: Waves Combine](#2-the-starting-point-waves-combine)
3. [The First Principle: Interference](#3-the-first-principle-interference)
4. [Coherence Defined](#4-coherence-defined)
5. [Coherence as an AQ Property](#5-coherence-as-an-aq-property)
6. [Coherence Determines Bonding](#6-coherence-determines-bonding)
7. [Attention as Coherence Selection](#7-attention-as-coherence-selection)
8. [Why Coherent States Win](#8-why-coherent-states-win)
9. [The Collapse Process: Three Complementary Levels](#9-the-collapse-process-three-complementary-levels)
   - 9.1 AKIRA: How to Build It Physically
   - 9.2 PID: What to Measure
   - 9.3 Computational Mechanics: Abstract Description
   - 9.4 Physical Principle: Why Interference Works
   - 9.5 The Three Levels and Their Mappings
   - 9.6 Collapse as Coherence Selection
   - 9.7 Crystallization Locks Phase
   - 9.8 Coherence in the Pump Cycle
10. [Coherence vs Synergy: Critical Distinction](#10-coherence-vs-synergy-critical-distinction)
11. [Summary](#11-summary)
12. [Related Documents](#12-related-documents)

---

## 1. What This Document Explains

This document builds the concept of coherence from first principles, showing:

- What coherence IS (phase alignment — AKIRA term for physical property)
- Why it matters (interference selects phase-aligned patterns)
- How it functions in AKIRA (determines AQ bondability)
- The collapse process at three complementary levels (Section 9):
  - **PID** — DEFINES WHAT to measure
  - **Comp Mech** — DESCRIBES HOW abstractly, COMPOSES, PROVES
  - **AKIRA** — TRANSLATES to physical, SPECIFIES HOW TO BUILD
- How coherence differs from synergy (AKIRA term vs PID term)

**Key insight:** The three levels are COMPLEMENTARY:
- PID tells you WHAT you're computing
- Comp Mech lets you PROVE it works
- AKIRA tells you HOW TO BUILD it

Wave mechanics terms (phase, coherence, interference) ARE AKIRA terms — they describe what engineers actually build.

---

## 2. The Starting Point: Waves Combine

### 2.1 The Physical Fact

When two waves meet, they combine. This is not a choice or a design decision. It is what waves do.

```
TWO WAVES MEETING

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  Wave A:    ∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿                                       │
│                                                                         │
│  Wave B:    ∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿                                       │
│                                                                         │
│  They meet → They combine → Result depends on their relationship       │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

The result of combination depends on one thing: the **phase relationship** between the waves.

### 2.2 Phase

Phase is where a wave is in its cycle at a given moment.

```
PHASE IS POSITION IN CYCLE

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  A wave oscillates:  peak → zero → trough → zero → peak → ...          │
│                                                                         │
│  Phase = 0°:    At peak                                                │
│  Phase = 90°:   At zero (going down)                                   │
│  Phase = 180°:  At trough                                              │
│  Phase = 270°:  At zero (going up)                                     │
│                                                                         │
│  Two waves can have the SAME frequency but DIFFERENT phases.           │
│  This difference determines everything about their combination.        │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. The First Principle: Interference

### 3.1 Constructive Interference

When two waves are **in phase** (peaks align with peaks, troughs with troughs), they reinforce each other:

```
CONSTRUCTIVE INTERFERENCE: Phases Aligned

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  Wave A (phase = 0°):      ╱╲    ╱╲    ╱╲    ╱╲                        │
│                           ╱  ╲  ╱  ╲  ╱  ╲  ╱  ╲                       │
│                          ╱    ╲╱    ╲╱    ╲╱    ╲                      │
│                                                                         │
│  Wave B (phase = 0°):      ╱╲    ╱╲    ╱╲    ╱╲                        │
│                           ╱  ╲  ╱  ╲  ╱  ╲  ╱  ╲                       │
│                          ╱    ╲╱    ╲╱    ╲╱    ╲                      │
│                                                                         │
│  Result (A + B):          ╱╲    ╱╲    ╱╲    ╱╲                         │
│                          ╱  ╲  ╱  ╲  ╱  ╲  ╱  ╲                        │
│                         ╱    ╲╱    ╲╱    ╲╱    ╲   ← DOUBLE amplitude │
│                                                                         │
│  Peaks add to peaks. Troughs add to troughs.                           │
│  The combined wave is STRONGER.                                        │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Destructive Interference

When two waves are **out of phase** (peaks align with troughs), they cancel each other:

```
DESTRUCTIVE INTERFERENCE: Phases Opposed

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  Wave A (phase = 0°):      ╱╲    ╱╲    ╱╲    ╱╲                        │
│                           ╱  ╲  ╱  ╲  ╱  ╲  ╱  ╲                       │
│                          ╱    ╲╱    ╲╱    ╲╱    ╲                      │
│                                                                         │
│  Wave B (phase = 180°):    ╲╱    ╲╱    ╲╱    ╲╱                        │
│                           ╲  ╱  ╲  ╱  ╲  ╱  ╲  ╱                       │
│                            ╲╱    ╲╱    ╲╱    ╲╱                        │
│                                                                         │
│  Result (A + B):         ─────────────────────────  ← ZERO amplitude   │
│                                                                         │
│  Peaks cancel troughs. Troughs cancel peaks.                           │
│  The combined wave is NOTHING.                                         │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.3 The Inescapable Consequence

This is not theory. This is what happens when waves combine:

```
THE INTERFERENCE PRINCIPLE

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  Aligned phases     →  Constructive interference  →  Signal SURVIVES  │
│                                                                         │
│  Opposed phases     →  Destructive interference   →  Signal CANCELS   │
│                                                                         │
│  ═══════════════════════════════════════════════════════════════════   │
│                                                                         │
│  This is the foundation of everything that follows.                    │
│                                                                         │
│  The universe does not "choose" which patterns survive.                │
│  Interference SELECTS patterns based on phase alignment.               │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Coherence Defined

### 4.1 Definition

**Coherence** is the degree to which components maintain a consistent phase relationship.

```
COHERENCE DEFINED

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  COHERENT:                                                              │
│  ──────────                                                             │
│  Components have a FIXED, CONSISTENT phase relationship.               │
│  When they combine, they produce PREDICTABLE interference.             │
│  The relationship persists over time.                                  │
│                                                                         │
│  Example: Laser light                                                   │
│    All photons have the same phase.                                    │
│    They interfere constructively.                                      │
│    Result: Intense, directed beam.                                     │
│                                                                         │
│  ─────────────────────────────────────────────────────────────────────  │
│                                                                         │
│  INCOHERENT:                                                            │
│  ────────────                                                           │
│  Components have RANDOM, CHANGING phase relationships.                 │
│  When they combine, interference is unpredictable.                     │
│  Some add, some cancel — net effect is weak.                           │
│                                                                         │
│  Example: Light bulb                                                    │
│    Photons have random phases.                                         │
│    They interfere randomly.                                            │
│    Result: Diffuse, scattered light.                                   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.2 The Quantitative Difference

The difference between coherent and incoherent combination is dramatic:

```
COHERENT vs INCOHERENT: The Numbers

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  Consider N components, each with amplitude 1.                         │
│                                                                         │
│  INCOHERENT (random phases):                                            │
│  ───────────────────────────                                            │
│  Phases are random.                                                     │
│  Combination is a random walk in the complex plane.                    │
│                                                                         │
│  Total amplitude: |A| ~ √N                                              │
│  Total intensity: |A|² ~ N                                              │
│                                                                         │
│  COHERENT (aligned phases):                                             │
│  ──────────────────────────                                             │
│  All phases are the same.                                               │
│  Combination is direct addition.                                        │
│                                                                         │
│  Total amplitude: |A| = N                                               │
│  Total intensity: |A|² = N²                                             │
│                                                                         │
│  ═══════════════════════════════════════════════════════════════════   │
│                                                                         │
│  RATIO: Coherent / Incoherent = N² / N = N                             │
│                                                                         │
│  For N = 100 components:                                                │
│    Coherent is 100× more intense than incoherent.                      │
│                                                                         │
│  For N = 1,000,000 components:                                          │
│    Coherent is 1,000,000× more intense than incoherent.                │
│                                                                         │
│  COHERENT STATES DOMINATE BY A FACTOR OF N.                            │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.3 The Key Insight

Coherence is not about being "better" or "preferred." It is about survival under interference:

```
THE KEY INSIGHT

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  The universe does not "prefer" coherence.                             │
│                                                                         │
│  It is that INCOHERENT STATES CANCEL THEMSELVES OUT.                   │
│                                                                         │
│  What we observe is what survives interference.                        │
│  What survives interference is what is coherent.                       │
│                                                                         │
│  ═══════════════════════════════════════════════════════════════════   │
│                                                                         │
│  COHERENCE IS NOT IMPOSED — IT EMERGES FROM SURVIVAL.                  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Coherence as an AQ Property

### 5.1 The Four Properties of Action Quanta

Action Quanta (AQ) have four properties. Coherence is one of them:

```
AQ PROPERTIES

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  Property        Meaning                    Physics Analog              │
│  ────────        ───────                    ──────────────              │
│                                                                         │
│  MAGNITUDE       How much signal            Mass                        │
│                  Determines salience                                    │
│                  Higher = more dominant                                 │
│                                                                         │
│  PHASE           Where/when in cycle        Charge                      │
│                  Determines interference                                │
│                  Encodes position, timing                               │
│                                                                         │
│  FREQUENCY       What scale                 Size                        │
│                  Determines spectral band                               │
│                  Low = coarse, High = fine                             │
│                                                                         │
│  COHERENCE       How internally organized   Spin                        │
│                  Determines bondability                                 │
│                  High = precise, Low = fuzzy                           │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 5.2 Internal Coherence of an AQ

An individual AQ has internal coherence: the degree to which its own components are phase-aligned.

```
INTERNAL COHERENCE

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  HIGH INTERNAL COHERENCE:                                               │
│  ─────────────────────────                                              │
│  The components within the AQ are phase-aligned.                       │
│  The pattern has sharp boundaries, clear structure.                    │
│                                                                         │
│  Example: An edge AQ                                                    │
│    All frequency components align at the edge location.                │
│    Result: Sharp, well-defined edge.                                   │
│                                                                         │
│  ─────────────────────────────────────────────────────────────────────  │
│                                                                         │
│  LOW INTERNAL COHERENCE:                                                │
│  ────────────────────────                                               │
│  The components within the AQ have random phases.                      │
│  The pattern is fuzzy, textured, not sharply defined.                 │
│                                                                         │
│  Example: A texture AQ                                                  │
│    Frequency components have varied phases.                            │
│    Result: Distributed, statistical pattern.                           │
│                                                                         │
│  ═══════════════════════════════════════════════════════════════════   │
│                                                                         │
│  INTERNAL COHERENCE DETERMINES:                                         │
│                                                                         │
│  • How "sharp" the AQ pattern is                                       │
│  • How reliably it can be detected                                     │
│  • How precisely it can bond with other AQ                             │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 5.3 Coherence Determines Bondability

The coherence of an AQ determines how well it can combine with other AQ:

```
COHERENCE AND BONDABILITY

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  HIGH COHERENCE AQ:                                                     │
│  ──────────────────                                                     │
│  • Clear phase signature                                               │
│  • Can align precisely with other AQ                                   │
│  • Forms strong, stable bonds                                          │
│  • Like a puzzle piece with sharp edges                                │
│                                                                         │
│  LOW COHERENCE AQ:                                                      │
│  ─────────────────                                                      │
│  • Diffuse phase signature                                             │
│  • Alignment is approximate, not exact                                 │
│  • Forms weak, loose bonds                                             │
│  • Like a puzzle piece with fuzzy edges                                │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Coherence Determines Bonding

### 6.1 How AQ Combine

When two AQ meet, they combine according to their phase relationship. This is the bonding mechanism:

```
AQ BONDING MECHANISM

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  AQ₁ has phase φ₁                                                       │
│  AQ₂ has phase φ₂                                                       │
│                                                                         │
│  They meet and combine.                                                 │
│                                                                         │
│  IF phases are aligned (φ₁ ≈ φ₂):                                       │
│    → Constructive interference                                         │
│    → Combined signal is STRONG                                         │
│    → Bond FORMS                                                         │
│                                                                         │
│  IF phases are opposed (φ₁ ≈ φ₂ + 180°):                                │
│    → Destructive interference                                          │
│    → Combined signal is WEAK or ZERO                                   │
│    → Bond does NOT form (or is very weak)                              │
│                                                                         │
│  IF phases are random:                                                  │
│    → Partial interference                                              │
│    → Combined signal is moderate                                       │
│    → Bond is weak, unstable                                            │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 6.2 Valid vs Invalid Combinations

Phase coherence acts as a selection mechanism, determining which combinations are valid:

```
COHERENCE AS SELECTION

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  COHERENT COMBINATION (Valid):                                          │
│  ──────────────────────────────                                         │
│  • Phases align                                                         │
│  • Constructive interference                                           │
│  • Strong combined signal                                              │
│  • Stable bonded state                                                 │
│  • The combination SURVIVES                                            │
│                                                                         │
│  Example:                                                               │
│    Edge AQ (orientation 45°) + Edge AQ (orientation 45°)               │
│    → Phases align → Strong contour emerges                             │
│                                                                         │
│  ─────────────────────────────────────────────────────────────────────  │
│                                                                         │
│  INCOHERENT COMBINATION (Invalid):                                      │
│  ──────────────────────────────────                                     │
│  • Phases misaligned                                                   │
│  • Destructive interference                                            │
│  • Weak or zero combined signal                                        │
│  • No stable bonded state                                              │
│  • The combination CANCELS                                             │
│                                                                         │
│  Example:                                                               │
│    "High altitude" AQ + "Underground" AQ                               │
│    → Phases conflict → Combination rejected                            │
│                                                                         │
│  ═══════════════════════════════════════════════════════════════════   │
│                                                                         │
│  PHASE COHERENCE ENFORCES "GRAMMATICALITY"                             │
│                                                                         │
│  Valid combinations bond.                                               │
│  Invalid combinations are rejected.                                    │
│  The system allows only coherent structures.                           │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 6.3 Types of Coherent Bonds

Different types of coherence enable different types of bonds:

```
TYPES OF COHERENT BONDS

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  TEMPORAL COHERENCE:                                                    │
│  ───────────────────                                                    │
│  AQ from the same moment in time.                                      │
│  They describe the same state of the world.                            │
│                                                                         │
│  Example: Range AQ + Doppler AQ + Angle AQ from one radar pulse        │
│  They cohere because they're from the same observation.                │
│                                                                         │
│  ─────────────────────────────────────────────────────────────────────  │
│                                                                         │
│  SPATIAL COHERENCE:                                                     │
│  ──────────────────                                                     │
│  AQ that describe consistent spatial relationships.                    │
│  They make sense together geometrically.                               │
│                                                                         │
│  Example: "Above" AQ + "Fast" AQ + "Large" AQ = plausible aircraft     │
│  Example: "Above" AQ + "Stationary" AQ + "Small" AQ = implausible     │
│                                                                         │
│  ─────────────────────────────────────────────────────────────────────  │
│                                                                         │
│  SEMANTIC COHERENCE:                                                    │
│  ───────────────────                                                    │
│  AQ that describe consistent meaning.                                  │
│  They don't contradict each other.                                     │
│                                                                         │
│  Example: "Cat" AQ + "Furry" AQ + "Meowing" AQ = coherent             │
│  Example: "Cat" AQ + "Feathered" AQ + "Flying" AQ = incoherent        │
│                                                                         │
│  ─────────────────────────────────────────────────────────────────────  │
│                                                                         │
│  TRACK COHERENCE:                                                       │
│  ────────────────                                                       │
│  AQ from successive observations that are consistent over time.        │
│  The trajectory makes physical sense.                                  │
│                                                                         │
│  Example: Position at t=1 + Position at t=2 must be reachable          │
│  If Doppler doesn't match position change → track breaks              │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 7. Attention as Coherence Selection

### 7.1 Attention Measures Phase Alignment

In neural networks, attention implements coherence selection:

```
ATTENTION IS COHERENCE SELECTION

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  ATTENTION MECHANISM:                                                   │
│                                                                         │
│  weights = softmax(QK^T / √d)                                          │
│                                                                         │
│  Q = Query (what we're looking for)                                    │
│  K = Key (what each position offers)                                   │
│                                                                         │
│  QK^T = Dot product of query and key                                   │
│                                                                         │
│  ═══════════════════════════════════════════════════════════════════   │
│                                                                         │
│  DOT PRODUCT MEASURES PHASE ALIGNMENT:                                  │
│                                                                         │
│  High QK^T:                                                             │
│    Q and K point in the same direction                                 │
│    → Vectors are ALIGNED                                               │
│    → "Phases" match                                                    │
│    → High attention weight                                             │
│                                                                         │
│  Low QK^T:                                                              │
│    Q and K point in different directions                               │
│    → Vectors are ORTHOGONAL or OPPOSED                                 │
│    → "Phases" don't match                                              │
│    → Low attention weight                                              │
│                                                                         │
│  ═══════════════════════════════════════════════════════════════════   │
│                                                                         │
│  SOFTMAX IMPLEMENTS SELECTION:                                          │
│                                                                         │
│  softmax amplifies differences:                                         │
│    High similarities → even higher weights                             │
│    Low similarities → near-zero weights                                │
│                                                                         │
│  Result: COHERENT positions are selected.                              │
│          INCOHERENT positions are suppressed.                          │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 7.2 Temperature Controls Selectivity

The temperature parameter controls how strictly coherence is enforced:

```
TEMPERATURE AND COHERENCE STRICTNESS

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  weights = softmax(QK^T / τ)                                           │
│                                                                         │
│  τ = temperature                                                        │
│                                                                         │
│  LOW TEMPERATURE (τ small):                                             │
│  ──────────────────────────                                             │
│  • Sharp attention (winner-take-all)                                   │
│  • Only the MOST coherent position selected                            │
│  • Strict enforcement of phase alignment                               │
│  • Like pure intervals in music                                        │
│                                                                         │
│  HIGH TEMPERATURE (τ large):                                            │
│  ───────────────────────────                                            │
│  • Soft attention (distributed)                                        │
│  • Multiple positions contribute                                       │
│  • Loose enforcement of phase alignment                                │
│  • Like equal temperament — spread the imprecision                     │
│                                                                         │
│  ═══════════════════════════════════════════════════════════════════   │
│                                                                         │
│  TRADE-OFF:                                                             │
│                                                                         │
│  Low τ:  Precise but may miss globally coherent solutions             │
│  High τ: Flexible but may include incoherent contributions            │
│                                                                         │
│  The optimal τ balances local precision and global coherence.          │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 8. Why Coherent States Win

### 8.1 The Survival Argument

Coherent states dominate not because they are "chosen" but because incoherent states cancel:

```
THE SURVIVAL ARGUMENT

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  STEP 1: Multiple hypotheses exist (superposition)                     │
│  ─────────────────────────────────────────────────                      │
│  Before collapse, many interpretations coexist.                        │
│  Each has its own phase.                                               │
│  The system is in superposition.                                       │
│                                                                         │
│  STEP 2: Hypotheses interact (interference)                            │
│  ──────────────────────────────────────────                             │
│  When combined (via attention, computation), hypotheses interfere.     │
│  Phase relationships determine interference type.                      │
│                                                                         │
│  STEP 3: Incoherent hypotheses cancel                                  │
│  ────────────────────────────────────                                   │
│  Hypotheses with random phases:                                        │
│    Some add, some cancel.                                              │
│    Net contribution ~ √N (weak).                                       │
│                                                                         │
│  STEP 4: Coherent hypotheses reinforce                                 │
│  ─────────────────────────────────────                                  │
│  Hypotheses with aligned phases:                                       │
│    All add together.                                                   │
│    Net contribution ~ N (strong).                                      │
│                                                                         │
│  STEP 5: Coherent hypothesis dominates                                 │
│  ─────────────────────────────────────                                  │
│  The coherent interpretation has N times more influence.               │
│  It becomes the output.                                                │
│  This is collapse.                                                     │
│                                                                         │
│  ═══════════════════════════════════════════════════════════════════   │
│                                                                         │
│  THE COHERENT STATE WINS BECAUSE IT SURVIVES INTERFERENCE.             │
│  The system does not "choose." Interference SELECTS.                   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 8.2 The Harmony Principle

This is the same principle that governs music:

```
THE HARMONY PRINCIPLE

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  In music:                                                              │
│  ──────────                                                             │
│  Harmonious intervals have simple frequency ratios.                    │
│  Simple ratios → phases align periodically → constructive.             │
│  Dissonant intervals have complex ratios.                              │
│  Complex ratios → phases rarely align → neutral or destructive.       │
│                                                                         │
│  In AKIRA:                                                              │
│  ──────────                                                             │
│  Coherent combinations have aligned phases.                            │
│  Aligned phases → constructive interference → stable bonds.            │
│  Incoherent combinations have random phases.                           │
│  Random phases → cancellation → no stable bonds.                       │
│                                                                         │
│  ═══════════════════════════════════════════════════════════════════   │
│                                                                         │
│  HARMONY is what survives interference.                                │
│  DISSONANCE is what cancels under interference.                        │
│                                                                         │
│  This is why we call it "coherence" — it is the same phenomenon       │
│  whether in light, sound, or belief.                                   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 9. The Collapse Process: Three Complementary Levels

This section describes the collapse process at three complementary levels.

**For the full formulation of the three levels, see `COMPUTATIONAL_MECHANICS_EQUIVALENCE.md` Section 1.3.**

```
THE THREE COMPLEMENTARY LEVELS (Summary)

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  PID           DEFINES WHAT to measure                                 │
│                                                                         │
│  Comp Mech     DESCRIBES HOW abstractly, COMPOSES, PROVES              │
│                                                                         │
│  AKIRA         TRANSLATES to physical, SPECIFIES HOW TO BUILD          │
│                                                                         │
│  ═══════════════════════════════════════════════════════════════════   │
│                                                                         │
│  THESE ARE COMPLEMENTARY:                                               │
│                                                                         │
│  PID tells you WHAT you're computing.                                  │
│  Comp Mech lets you PROVE it works.                                    │
│  AKIRA tells you HOW TO BUILD it.                                      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

The following subsections describe the collapse process from each level's perspective.

### 9.1 AKIRA: How to Build It Physically

AKIRA specifies the physical mechanisms. Wave mechanics terms (phase, coherence, interference) are AKIRA vocabulary - they describe what you physically build:

```
AKIRA: PHYSICAL SPECIFICATION OF COLLAPSE

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  AKIRA VOCABULARY includes:                                            │
│    Superposition, collapse, crystallized, Action Quantum               │
│    Magnitude, phase, frequency, coherence (AQ properties)              │
│    Interference (constructive/destructive), phase alignment            │
│    Spectral bands, thresholding                                        │
│                                                                         │
│  These are PHYSICAL terms - they describe what you build.              │
│                                                                         │
│  ═══════════════════════════════════════════════════════════════════   │
│                                                                         │
│  SUPERPOSITION STATE:                                                   │
│  ────────────────────                                                   │
│  Multiple interpretations coexist across spectral bands.              │
│  Each has amplitude Aᵢ and phase φᵢ.                                   │
│  Belief is distributed. No single pattern dominates.                  │
│  The representation is REDUCIBLE (could decompose many ways).         │
│                                                                         │
│  COLLAPSE (the physical transition):                                   │
│  ───────────────────────────────────                                    │
│  When components combine: Total = Σᵢ Aᵢ · exp(iφᵢ)                    │
│                                                                         │
│  • Random phases → contributions cancel (destructive interference)    │
│  • Aligned phases → contributions reinforce (constructive)            │
│                                                                         │
│  Incoherent patterns cancel themselves.                               │
│  Coherent patterns survive and reinforce.                             │
│  Belief concentrates.                                                  │
│                                                                         │
│  CRYSTALLIZED STATE:                                                    │
│  ───────────────────                                                    │
│  Single dominant pattern emerges.                                     │
│  Phase relationships are LOCKED.                                      │
│  The representation is IRREDUCIBLE.                                   │
│                                                                         │
│  ACTION QUANTUM:                                                        │
│  ───────────────                                                        │
│  The crystallized pattern IS an Action Quantum.                       │
│  Properties: [Magnitude] [Phase] [Frequency] [Coherence]              │
│  Definition: Minimum pattern enabling correct action.                 │
│  Key property: IRREDUCIBLE — cannot decompose without losing          │
│                actionability.                                          │
│                                                                         │
│  ═══════════════════════════════════════════════════════════════════   │
│                                                                         │
│  WHY WAVE MECHANICS TERMS?                                              │
│  ─────────────────────────                                              │
│  These are not borrowed from quantum mechanics.                        │
│  Wave interference is a REAL PHYSICAL PHENOMENON.                     │
│  AKIRA uses these terms because they describe what you build:         │
│    - Spectral decomposition (FFT)                                     │
│    - Phase relationships (complex numbers)                            │
│    - Interference patterns (superposition principle)                  │
│    - Thresholding (CFAR, softmax)                                     │
│                                                                         │
│  Engineers already know how to build these systems.                   │
│  AKIRA tells them what these systems are computing.                   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 9.2 Information Theory (PID): What to Measure

PID DEFINES WHAT to measure. These are the quantities, not the mechanisms:

```
PID: SYNERGY TO REDUNDANCY TRANSITION

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  PID DEFINES THE QUANTITIES:                                           │
│  ───────────────────────────                                            │
│  I_syn (synergy): Information requiring COMBINATION of sources        │
│  I_red (redundancy): Information SHARED by sources                    │
│  I_uni (unique): Information from only one source                     │
│                                                                         │
│  These have PRECISE mathematical definitions (bits).                  │
│  They are NOT general English words.                                   │
│                                                                         │
│  ═══════════════════════════════════════════════════════════════════   │
│                                                                         │
│  BEFORE COLLAPSE (High I_syn):                                          │
│  ─────────────────────────────                                          │
│  Band 0 alone cannot predict target.                                  │
│  Band 1 alone cannot predict target.                                  │
│  BOTH together can predict target.                                    │
│  → Predictive information is SYNERGISTIC (requires combination).      │
│                                                                         │
│  AFTER COLLAPSE (High I_red):                                           │
│  ──────────────────────────────                                         │
│  Band 0 alone CAN predict target.                                     │
│  Band 1 alone CAN predict target.                                     │
│  Either suffices.                                                      │
│  → Predictive information is REDUNDANT (shared by sources).           │
│                                                                         │
│  THE TRANSITION:                                                        │
│  ───────────────                                                        │
│  I_syn → I_red (synergy converts to redundancy)                       │
│  This is MEASURABLE: Compute I_syn, I_red before and after.          │
│                                                                         │
│  ═══════════════════════════════════════════════════════════════════   │
│                                                                         │
│  WHY PID TERMS MATTER:                                                  │
│  ─────────────────────                                                  │
│  "Synergy" is not a synonym for "coherence" or "distributed."         │
│  It is a SPECIFIC quantity: information requiring combination.        │
│                                                                         │
│  We use PID terms when we want to MEASURE information distribution.   │
│  We use AKIRA terms when we want to BUILD the physical system.        │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 9.3 Computational Mechanics: Describe, Compose, Prove

Computational mechanics DESCRIBES HOW abstractly, shows how components COMPOSE, and enables PROOFS via formal symbolic logic:

```
COMPUTATIONAL MECHANICS: BELIEF SYNCHRONIZATION TO CAUSAL STATE

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  COMPUTATIONAL MECHANICS:                                               │
│  ────────────────────────                                               │
│  DESCRIBES HOW abstractly (the mathematical process)                  │
│  COMPOSES how components combine (ε-machine composition)              │
│  PROVES properties via formal symbolic logic (convergence, minimality)│
│                                                                         │
│  ═══════════════════════════════════════════════════════════════════   │
│                                                                         │
│  INITIAL STATE:                                                         │
│  ──────────────                                                         │
│  Belief state b(s) is distributed over possible states s.             │
│  The agent is uncertain which causal state ξ it is in.                │
│                                                                         │
│  SYNCHRONIZATION:                                                       │
│  ────────────────                                                       │
│  As observations accumulate, belief contracts.                        │
│  b(s) → δ(s - s*) (Dirac delta at true state)                        │
│                                                                         │
│  CAUSAL STATE ξ:                                                        │
│  ───────────────                                                        │
│  Definition: ξ = [h]_~ = equivalence class of histories               │
│              where h ~ h' iff P(future|h) = P(future|h')              │
│                                                                         │
│  Properties (PROVABLE):                                                 │
│  • Minimal sufficient statistic for prediction                        │
│  • Deterministic update: ξ' = f(ξ, observation) (unifilarity)        │
│  • Composition: ε-machines can be chained and composed                │
│                                                                         │
│  FINAL STATE:                                                           │
│  ────────────                                                           │
│  Agent has synchronized to causal state ξ.                            │
│  Prediction is now possible: P(future) = P(future|ξ)                 │
│                                                                         │
│  ═══════════════════════════════════════════════════════════════════   │
│                                                                         │
│  WHY COMP MECH MATTERS:                                                 │
│  ──────────────────────                                                 │
│  You can PROVE that:                                                   │
│    • Causal states are minimal (no smaller representation works)      │
│    • Synchronization converges (belief reaches the true state)        │
│    • Composition preserves properties (combined systems work)         │
│                                                                         │
│  These proofs justify AKIRA's physical mechanisms.                    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 9.4 Physical Principle: Why Interference Selection Works

AKIRA uses real physical interference, not quantum mechanics. The math is the same because wave interference is universal:

```
THE PHYSICAL PRINCIPLE (not QED - real waves)

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  WAVE INTERFERENCE IS REAL PHYSICS:                                    │
│  ──────────────────────────────────                                     │
│  This is not quantum mechanics. This is classical wave physics.       │
│  Sound waves, water waves, electromagnetic waves all do this.         │
│  AKIRA systems use real interference (FFT, spectral processing).      │
│                                                                         │
│  ═══════════════════════════════════════════════════════════════════   │
│                                                                         │
│  THE UNIVERSAL PRINCIPLE:                                               │
│  ────────────────────────                                               │
│  When wave components combine:                                         │
│    Total = Σᵢ Aᵢ · exp(iφᵢ)                                            │
│                                                                         │
│  Random phases (incoherent):                                           │
│    Contributions point in random directions → cancel                  │
│    Net amplitude ~ √N (random walk)                                   │
│                                                                         │
│  Aligned phases (coherent):                                            │
│    Contributions point same direction → reinforce                     │
│    Net amplitude ~ N (coherent sum)                                   │
│                                                                         │
│  Coherent patterns dominate by factor of √N.                          │
│  Incoherent patterns cancel themselves.                               │
│                                                                         │
│  ═══════════════════════════════════════════════════════════════════   │
│                                                                         │
│  FEYNMAN PATH INTEGRAL (for physics readers):                          │
│  ────────────────────────────────────────────                           │
│  QED uses the same math: paths with random phases cancel.             │
│  The principle is the same. The application domain differs.           │
│  AKIRA is classical wave physics, not quantum mechanics.              │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 9.5 Cross-Level Mappings

For the full three-level formulation, see `COMPUTATIONAL_MECHANICS_EQUIVALENCE.md` Section 1.3.

The key mappings for the collapse process:

```
CROSS-LEVEL MAPPINGS FOR COLLAPSE

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  NOTE: Wave mechanics terms (phase, coherence, interference)           │
│  ARE AKIRA terms. AKIRA is the physical implementation language.      │
│  These describe what you build: FFT, spectral processing, thresholds. │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

ENDPOINT MAPPING (what collapse produces):

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  AKIRA                  COMP MECH              PID                      │
│  ─────                  ─────────              ───                      │
│  Action Quantum    ≅    Causal state ξ         (not a PID term)        │
│  (crystallized,         (equiv. class of                               │
│   irreducible)          histories)                                     │
│                                                                         │
│  Both defined as: Minimal sufficient statistic for prediction/action  │
│  See: COMPUTATIONAL_MECHANICS_EQUIVALENCE.md for formal argument      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

PROCESS MAPPING (the transition):

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  AKIRA                  COMP MECH              PID                      │
│  ─────                  ─────────              ───                      │
│  Collapse          ↔    Synchronization   ↔    I_syn → I_red           │
│  (superposition→        (b → δ)                (synergy converts       │
│   crystallized)                                 to redundancy)         │
│                                                                         │
│  Three descriptions of the SAME transition.                           │
│  Use the appropriate level for your purpose.                          │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

WHEN TO USE WHICH:

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  BUILDING A SYSTEM → Use AKIRA                                         │
│    "Implement collapse via CFAR thresholding"                         │
│    "Phase-aligned patterns reinforce"                                 │
│    "AQ has properties: magnitude, phase, frequency, coherence"        │
│                                                                         │
│  MEASURING INFORMATION → Use PID                                        │
│    "Compute I_syn and I_red before and after"                         │
│    "The transition shows synergy converting to redundancy"            │
│    NOTE: Synergy is NOT a synonym for "distributed" or "uncertain"   │
│                                                                         │
│  PROVING PROPERTIES → Use Comp Mech                                    │
│    "Causal states are minimal sufficient statistics"                  │
│    "Unifilar update guarantees consistency"                           │
│    "Belief synchronization converges"                                 │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

### 9.6 Collapse is Coherence Selection (Detailed View)

The collapse from superposition to crystallized state is the process of coherence selection:

```
COLLAPSE AS COHERENCE SELECTION

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  BEFORE COLLAPSE (Superposition):                                       │
│  ─────────────────────────────────                                      │
│  • Multiple hypotheses coexist                                         │
│  • Phases are varied                                                   │
│  • High synergy (information distributed)                              │
│  • No dominant interpretation                                          │
│                                                                         │
│  DURING COLLAPSE:                                                       │
│  ────────────────                                                       │
│  • Hypotheses interfere                                                │
│  • Coherent components reinforce                                       │
│  • Incoherent components cancel                                        │
│  • Dominant pattern emerges                                            │
│                                                                         │
│  AFTER COLLAPSE (Crystallized):                                         │
│  ───────────────────────────────                                        │
│  • Single coherent interpretation                                      │
│  • Phases are locked                                                   │
│  • High redundancy (information concentrated)                          │
│  • Stable, actionable AQ                                               │
│                                                                         │
│  ═══════════════════════════════════════════════════════════════════   │
│                                                                         │
│  COLLAPSE IS NOT "CHOOSING" — IT IS LETTING INTERFERENCE SELECT.       │
│                                                                         │
│  The crystallized state is what remains after incoherent              │
│  alternatives have cancelled themselves.                               │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 9.7 Crystallization Locks Phase

After collapse, the crystallized AQ has locked phase relationships:

```
CRYSTALLIZATION LOCKS PHASE

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  SUPERPOSITION STATE:                                                   │
│  ────────────────────                                                   │
│  • Phases are fluid                                                    │
│  • Relationships can change                                            │
│  • System is reducible                                                 │
│                                                                         │
│  CRYSTALLIZED STATE:                                                    │
│  ───────────────────                                                    │
│  • Phases are LOCKED                                                   │
│  • Relationships are fixed                                             │
│  • System is IRREDUCIBLE                                               │
│                                                                         │
│  This is why crystallized AQ are stable:                               │
│    The phase relationships that survived are now permanent.            │
│    The coherent configuration persists.                                │
│    This IS the learned representation.                                 │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 9.8 Coherence in the Pump Cycle (Cross-Framework)

The pump cycle is defined in PID terms (I_syn, I_red). Coherence (AKIRA) plays a role during the transitions:

```
COHERENCE IN THE PUMP CYCLE (Cross-Framework View)

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  PID VIEW (the pump cycle):                                            │
│  ──────────────────────────                                             │
│  [I_red high] ──TENSION──► [I_syn high] ──COLLAPSE──► [I_red high] + AQ│
│                                                                         │
│  NOTE: Redundancy (I_red) and Synergy (I_syn) are PID terms.          │
│  The cycle describes how information DISTRIBUTION changes.             │
│                                                                         │
│  ─────────────────────────────────────────────────────────────────────  │
│                                                                         │
│  WAVE MECHANICS VIEW (what happens to phases):                         │
│  ─────────────────────────────────────────────                          │
│                                                                         │
│  During TENSION:                                                        │
│  • New input creates phase variation                                   │
│  • Multiple phase configurations emerge                                │
│  • No dominant coherent pattern                                        │
│                                                                         │
│  During COLLAPSE:                                                       │
│  • Interference occurs                                                 │
│  • Incoherent configurations cancel                                    │
│  • Coherent configurations reinforce                                   │
│  • Surviving pattern has locked phases (high coherence)               │
│                                                                         │
│  ─────────────────────────────────────────────────────────────────────  │
│                                                                         │
│  HOW THEY RELATE:                                                       │
│  ────────────────                                                       │
│  Coherence (wave mechanics) is the MECHANISM during collapse.         │
│  Synergy→Redundancy (PID) is what we MEASURE before/after.            │
│                                                                         │
│  These are different descriptions, not synonyms.                       │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 10. Coherence vs Synergy: Critical Distinction

### 10.1 Two Different Frameworks

Coherence and synergy are related but NOT the same concept. They come from different theoretical frameworks:

```
COHERENCE vs SYNERGY: THE DISTINCTION

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  SYNERGY (Information Theory - PID)                                    │
│  ───────────────────────────────────                                    │
│                                                                         │
│  Framework: Partial Information Decomposition                          │
│  Unit: Bits of information                                             │
│                                                                         │
│  Definition:                                                            │
│    Information that NEITHER source alone provides,                     │
│    but TOGETHER they do.                                               │
│                                                                         │
│  Measures: How information is DISTRIBUTED across sources.              │
│                                                                         │
│  High synergy = Information requires combination (uncertainty)         │
│  Low synergy = Information from single source (certainty)              │
│                                                                         │
│  ─────────────────────────────────────────────────────────────────────  │
│                                                                         │
│  COHERENCE (Wave Mechanics)                                             │
│  ──────────────────────────                                             │
│                                                                         │
│  Framework: Wave interference, phase relationships                     │
│  Unit: Degrees (phase alignment)                                       │
│                                                                         │
│  Definition:                                                            │
│    The degree to which components maintain consistent                  │
│    PHASE RELATIONSHIPS.                                                 │
│                                                                         │
│  Measures: How ALIGNED phases are within/between patterns.             │
│                                                                         │
│  High coherence = Phases aligned (constructive interference)           │
│  Low coherence = Phases random (destructive interference)              │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 10.2 Different Questions

```
WHAT EACH CONCEPT ANSWERS

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  SYNERGY answers:                                                       │
│  "How is information distributed across bands/sources?"                │
│  "Do I need multiple bands together to predict the target?"            │
│  "Is the system in a pre-collapse (high synergy) or                    │
│   post-collapse (high redundancy) state?"                              │
│                                                                         │
│  COHERENCE answers:                                                     │
│  "Are the phase relationships consistent within this pattern?"         │
│  "Can these AQ bond together?"                                         │
│  "Will this combination survive interference?"                         │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 10.3 How They Relate in Collapse Dynamics

The two concepts work together but describe different aspects of collapse:

```
SYNERGY AND COHERENCE IN COLLAPSE

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  THE PUMP CYCLE (Information-theoretic view):                          │
│                                                                         │
│  [Redundancy] ──TENSION──> [Synergy] ──COLLAPSE──> [Redundancy] + AQ  │
│                                                                         │
│  Synergy describes the STATE:                                          │
│  • High synergy = pre-collapse (information distributed)              │
│  • High redundancy = post-collapse (information concentrated)         │
│                                                                         │
│  ─────────────────────────────────────────────────────────────────────  │
│                                                                         │
│  THE SELECTION MECHANISM (Wave-mechanical view):                       │
│                                                                         │
│  [Hypotheses] ──INTERFERENCE──> [Coherent survive] ──> [AQ crystallize]│
│                                                                         │
│  Coherence is the MECHANISM:                                           │
│  • Coherent patterns reinforce (constructive interference)            │
│  • Incoherent patterns cancel (destructive interference)              │
│  • What survives has high coherence                                    │
│                                                                         │
│  ═══════════════════════════════════════════════════════════════════   │
│                                                                         │
│  RELATIONSHIP:                                                          │
│                                                                         │
│  Coherence is the MECHANISM that enables certain patterns to survive. │
│  Synergy → Redundancy is the INFORMATIONAL CONSEQUENCE of survival.   │
│                                                                         │
│  Coherence operates DURING collapse (the selection process).          │
│  Synergy/Redundancy measures BEFORE/AFTER collapse (the state).       │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 10.4 In AKIRA's Architecture

```
WHERE EACH CONCEPT APPEARS

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  SYNERGY appears in:                                                    │
│  ─────────────────────                                                  │
│  • Measuring belief state (high synergy = uncertain)                   │
│  • Cross-band information flow (wormholes enable synergy)              │
│  • Collapse detection (synergy drops below threshold)                  │
│  • PID analysis of band contributions                                  │
│                                                                         │
│  COHERENCE appears in:                                                  │
│  ─────────────────────────                                              │
│  • AQ property (one of four: magnitude, phase, frequency, COHERENCE)  │
│  • Bonding validity (do these AQ combine properly?)                   │
│  • Attention mechanism (dot product = phase alignment measure)        │
│  • Pattern stability (high coherence = stable representation)         │
│                                                                         │
│  ─────────────────────────────────────────────────────────────────────  │
│                                                                         │
│  EXAMPLE:                                                               │
│                                                                         │
│  Before collapse:                                                       │
│  • HIGH SYNERGY (bands disagree, need combination to predict)         │
│  • Variable COHERENCE (some hypotheses coherent, some not)            │
│                                                                         │
│  During collapse:                                                       │
│  • COHERENT hypotheses reinforce each other                           │
│  • INCOHERENT hypotheses cancel                                        │
│  • Selection by interference                                           │
│                                                                         │
│  After collapse:                                                        │
│  • HIGH REDUNDANCY (bands agree, any can predict)                     │
│  • HIGH COHERENCE (surviving AQ have locked phase relationships)      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 10.5 Do Not Conflate

```
COMMON CONFUSION TO AVOID

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  WRONG: "High coherence means low synergy"                             │
│                                                                         │
│  RIGHT: Coherence (phase alignment) and synergy (information          │
│         distribution) are measured differently.                        │
│                                                                         │
│         A pattern can have high internal coherence but still          │
│         require combination with other patterns (high synergy).       │
│                                                                         │
│  ─────────────────────────────────────────────────────────────────────  │
│                                                                         │
│  WRONG: "Collapse is when coherence increases"                         │
│                                                                         │
│  RIGHT: Collapse is when SYNERGY converts to REDUNDANCY.               │
│         Coherence is the mechanism that determines which patterns     │
│         survive this transition.                                       │
│                                                                         │
│         Post-collapse AQ have locked coherence (stable),               │
│         but the defining change is the information distribution.      │
│                                                                         │
│  ─────────────────────────────────────────────────────────────────────  │
│                                                                         │
│  USE THE RIGHT TERM FOR THE RIGHT CONCEPT:                             │
│                                                                         │
│  "High I_syn" (PID) = Information requires combination of sources     │
│  "High coherence" (Wave) = Phases are aligned                         │
│  "Collapse" (AKIRA) = Superposition → Crystallized transition         │
│  "I_syn → I_red" (PID) = What collapse produces informationally       │
│  "AQ bonded" (AKIRA) = Phase-coherent AQ combination formed           │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 11. Summary

```
THE ROLE AND EFFECTS OF COHERENCE (DISTINCT FROM SYNERGY)

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  WHAT COHERENCE IS:                                                     │
│  ──────────────────                                                     │
│  The degree to which components maintain consistent phase              │
│  relationships.                                                        │
│                                                                         │
│  ─────────────────────────────────────────────────────────────────────  │
│                                                                         │
│  WHY IT MATTERS:                                                        │
│  ───────────────                                                        │
│  Coherent components interfere constructively (intensity ~ N²).        │
│  Incoherent components interfere destructively (intensity ~ N).        │
│  Coherent states dominate by a factor of N.                            │
│                                                                         │
│  ─────────────────────────────────────────────────────────────────────  │
│                                                                         │
│  ITS ROLE IN AQ:                                                        │
│  ───────────────                                                        │
│  Coherence is one of four AQ properties.                               │
│  It determines internal organization and bondability.                  │
│  High coherence → sharp patterns, precise bonds.                       │
│  Low coherence → fuzzy patterns, loose bonds.                          │
│                                                                         │
│  ─────────────────────────────────────────────────────────────────────  │
│                                                                         │
│  ITS ROLE IN BONDING:                                                   │
│  ────────────────────                                                   │
│  Phase coherence determines valid combinations.                        │
│  Coherent combinations bond (constructive interference).               │
│  Incoherent combinations are rejected (destructive interference).      │
│  This enforces "grammaticality" — only coherent structures survive.   │
│                                                                         │
│  ─────────────────────────────────────────────────────────────────────  │
│                                                                         │
│  ITS ROLE IN ATTENTION:                                                 │
│  ──────────────────────                                                 │
│  Attention (QK^T) measures phase alignment.                            │
│  Softmax amplifies coherent contributions, suppresses incoherent.      │
│  Temperature controls how strictly coherence is enforced.              │
│                                                                         │
│  ─────────────────────────────────────────────────────────────────────  │
│                                                                         │
│  ITS ROLE IN COLLAPSE (AKIRA term):                                    │
│  ──────────────────────────────────                                     │
│  During collapse (superposition → crystallized):                       │
│    • Incoherent patterns cancel (wave mechanics)                       │
│    • Coherent patterns reinforce (wave mechanics)                      │
│  The crystallized AQ is what survives interference.                   │
│  Phase relationships lock. The AQ is now IRREDUCIBLE.                 │
│                                                                         │
│  ═══════════════════════════════════════════════════════════════════   │
│                                                                         │
│  THE CORE PRINCIPLE:                                                    │
│                                                                         │
│  Coherence is not imposed — it emerges from survival.                  │
│  What survives interference is what harmonizes.                        │
│  What harmonizes is what is coherent.                                  │
│                                                                         │
│  ═══════════════════════════════════════════════════════════════════   │
│                                                                         │
│  DISTINCTION FROM SYNERGY (See Section 10):                            │
│                                                                         │
│  COHERENCE = Phase alignment (mechanism of selection)                  │
│  SYNERGY = Information distribution across sources (PID measure)       │
│                                                                         │
│  Coherence determines WHAT survives collapse.                          │
│  Synergy → Redundancy measures the RESULT of collapse.                 │
│  These are complementary, not interchangeable concepts.                │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 12. Related Documents

### 12.1 Foundational

| Document | Location | Relevance |
|----------|----------|-----------|
| `SYNERGY_REDUNDANCY.md` | `terminology_foundations/` | Synergy/redundancy (contrast with coherence) |
| `HARMONY_AND_COHERENCE.md` | `foundations/` | Philosophical treatment, Pythagorean comma |
| `TERMINOLOGY.md` | `foundations/` | Coherence as AQ property (Section 5) |
| `ACTION_QUANTA.md` | `terminology_foundations/` | Full AQ specification |
| `COLLAPSE_TENSION.md` | `terminology_foundations/` | Collapse dynamics (uses both concepts) |
| `COMPUTATIONAL_MECHANICS_EQUIVALENCE.md` | `foundations/parallels/` | AQ ≅ ξ (causal state equivalence) |

### 12.2 Physical Parallels

| Document | Location | Relevance |
|----------|----------|-----------|
| `RADAR_ARRAY.md` | `foundations/parallels/` | Temporal, spatial, track coherence |
| `COMPUTATIONAL_MECHANICS_EQUIVALENCE.md` | `foundations/parallels/` | Phase alignment = coherence checking |

### 12.3 Architecture

| Document | Location | Relevance |
|----------|----------|-----------|
| `SPECTRAL_BELIEF_MACHINE.md` | `architecture_theoretical/` | Cross-band coherence |
| `WORMHOLE_HYBRID.md` | `architecture/wormhole/` | Phase alignment in wormholes |

---

## References

### External

1. **Interference (physics)**: Any introductory physics textbook covers wave interference.

2. **Coherence (physics)**: Born, M., & Wolf, E. (1999). *Principles of Optics*. Cambridge University Press. Chapter 10.

3. **Phase locking**: Pikovsky, A., Rosenblum, M., & Kurths, J. (2001). *Synchronization: A Universal Concept in Nonlinear Sciences*. Cambridge University Press.

4. **Path integral / QED selection**: Feynman, R.P., & Hibbs, A.R. (1965). *Quantum Mechanics and Path Integrals*. McGraw-Hill. (Classical path emerges from cancellation of non-classical paths.)

5. **Computational mechanics / Causal states**: Shalizi, C.R., & Crutchfield, J.P. (2001). *Computational mechanics: Pattern and prediction, structure and simplicity*. Journal of Statistical Physics, 104(3-4), 817-879.

6. **Attention mechanism**: Vaswani, A., et al. (2017). *Attention Is All You Need*. NeurIPS.

### Internal

- `SYNERGY_REDUNDANCY.md` - Synergy definition (distinct from coherence)
- `TERMINOLOGY.md` - Coherence definition in AQ context
- `HARMONY_AND_COHERENCE.md` - Extended philosophical treatment
- `RADAR_ARRAY.md` - Coherence in physical sensing systems
- `COLLAPSE_TENSION.md` - Collapse dynamics (synergy → redundancy transition)
- `COMPUTATIONAL_MECHANICS_EQUIVALENCE.md` - AQ ≅ ξ equivalence (causal states)

---

*This document establishes coherence as the phase-alignment property that determines which patterns survive interference. The argument builds from physical first principles (wave interference) to AKIRA-specific applications (AQ bonding, attention selection, collapse dynamics). Coherence is not imposed by the system but emerges from the survival of patterns under interference. Note: Coherence (phase alignment) is distinct from synergy (information distribution). Both concepts are needed to fully describe collapse dynamics — see Section 10 for the precise distinction.*

*Oscar Goldman — Shogu Research Group @ Datamutant.ai*
