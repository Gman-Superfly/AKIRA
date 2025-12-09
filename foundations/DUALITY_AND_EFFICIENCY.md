# Duality and Computational Efficiency

## How Mathematical Dualities Inform AKIRA's Architecture

---

## Table of Contents

1. [Introduction: The Duality Principle](#1-introduction)
2. [The Three Properties of Productive Dualities](#2-three-properties)
3. [Duality 1: Spatial ↔ Frequency](#3-spatial-frequency)
4. [Duality 2: Forward ↔ Backward (Causal ↔ Non-causal)](#4-forward-backward)
5. [Duality 3: Sharp ↔ Soft (Semiring Interpolation)](#5-sharp-soft)
6. [Duality 4: Local ↔ Global](#6-local-global)
7. [Duality 5: Redundancy ↔ Synergy (PID)](#7-redundancy-synergy)
8. [Duality 6: Magnitude ↔ Phase](#8-magnitude-phase)
9. [Summary: Architecture Decisions from Dualities](#9-summary)
10. [References](#10-references)

---

## 1. Introduction: The Duality Principle

### 1.0 The Core Dynamic: Tension and Collapse

The most important duality in AKIRA is the tension-collapse cycle:

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
│  This IS a duality: Redundancy ↔ Synergy (see Section 7)              │
│  See foundations/TERMINOLOGY.md for formal definitions.               │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1.1 The Core Pattern

```
THE PRODUCTIVE DUALITY PATTERN:

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  Every productive duality has THREE properties:                        │
│                                                                         │
│  1. TRANSFORM:           A mapping T between Domain A and Domain B     │
│                          with efficient implementation (often O(n log n))│
│                                                                         │
│  2. CONSERVED QUANTITY:  Something preserved under T                   │
│                          (energy, information, optimality value)       │
│                                                                         │
│  3. COMPLEXITY INVERSION: What's hard in A is easy in B, and vice versa│
│                          (the hard↔easy swap)                          │
│                                                                         │
│  ═══════════════════════════════════════════════════════════════════   │
│                                                                         │
│  The architecture insight:                                              │
│                                                                         │
│  Identify what's HARD in your problem.                                 │
│  Find a duality where it becomes EASY.                                 │
│  Transform, solve, transform back.                                     │
│  Pay only the transform cost.                                          │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Why This Matters for AKIRA

```
AKIRA EXPLOITS DUALITIES SYSTEMATICALLY:

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  PROBLEM: Attention over N positions is O(N²)                          │
│                                                                         │
│  DUALITY SOLUTION:                                                      │
│  • Transform to frequency domain (O(N log N))                          │
│  • Attention patterns that are shift-invariant become DIAGONAL         │
│  • Per-band attention is O(N/7) per band = O(N) total                  │
│  • Transform back (O(N log N))                                          │
│                                                                         │
│  NET: O(N²) → O(N log N)                                               │
│                                                                         │
│  This is not a hack. It follows from the duality principle:            │
│  • Transform: FFT                                                       │
│  • Conserved: Energy (Parseval)                                        │
│  • Complexity inversion: Global convolution → local multiplication     │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. The Three Properties of Productive Dualities

### 2.1 Property 1: The Transform

**ESTABLISHED MATHEMATICS:**

```
THE TRANSFORM REQUIREMENT:

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  A productive duality requires an EFFICIENT transform T:               │
│                                                                         │
│  T: Domain A → Domain B                                                │
│  T⁻¹: Domain B → Domain A                                              │
│                                                                         │
│  EFFICIENCY CRITERION:                                                  │
│  Cost(T) + Cost(operation in B) + Cost(T⁻¹) < Cost(operation in A)    │
│                                                                         │
│  If the transform is too expensive, the duality is useless.           │
│                                                                         │
│  ─────────────────────────────────────────────────────────────────────  │
│                                                                         │
│  EXAMPLES OF EFFICIENT TRANSFORMS:                                      │
│                                                                         │
│  Transform        Cost          What It Enables                        │
│  ─────────        ────          ─────────────────                       │
│  FFT              O(N log N)    Convolution → Multiplication           │
│  Backpropagation  O(N)          All gradients from one backward pass   │
│  Matrix inverse   O(N³)         Solve linear systems (sometimes worth it)│
│  Eigendecomp      O(N³)         Diagonalize operators (batch problems) │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Property 2: The Conserved Quantity

**ESTABLISHED MATHEMATICS:**

```
THE CONSERVATION REQUIREMENT:

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  Every productive duality CONSERVES something under the transform:     │
│                                                                         │
│  Duality              Conserved Quantity           Formula             │
│  ───────              ──────────────────           ───────             │
│  Fourier              Energy                       Σ|x|² = Σ|X|²       │
│  Primal-Dual LP       Optimal value                P* = D*             │
│  Max-Flow/Min-Cut     Capacity                     Flow = Cut          │
│  Forward/Backward     Inner products               ⟨∂L/∂x, v⟩ preserved │
│  Legendre             Convex structure             f** = f             │
│                                                                         │
│  ─────────────────────────────────────────────────────────────────────  │
│                                                                         │
│  WHY CONSERVATION MATTERS:                                             │
│                                                                         │
│  1. VALIDATION: If conservation fails, there's a bug                   │
│  2. INVARIANTS: Conserved quantities are testable                      │
│  3. CERTIFICATES: Conservation proves correctness                      │
│  4. SEMANTICS: What's conserved tells you what the computation means  │
│                                                                         │
│  AKIRA USES: Parseval (energy), PID (total information)               │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.3 Property 3: The Complexity Inversion

**THE KEY ARCHITECTURAL INSIGHT:**

```
THE COMPLEXITY INVERSION:

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  The POINT of a duality is that HARD ↔ EASY swaps:                    │
│                                                                         │
│  Duality              Hard in A              Easy in B                 │
│  ───────              ─────────              ─────────                  │
│  Fourier              Global convolution     Pointwise multiply        │
│  LP Duality           Constraints            Objectives                │
│  Graph Spectral       Combinatorial cut      Eigenvalue threshold      │
│  Forward/Backward     All gradients          One backward pass         │
│  Kalman               Full sequence estimate Forward+backward filter   │
│                                                                         │
│  ─────────────────────────────────────────────────────────────────────  │
│                                                                         │
│  THE DUALITY DESIGN ALGORITHM:                                          │
│                                                                         │
│  1. Identify what's HARD in your problem                               │
│  2. Find a duality where it becomes EASY                               │
│  3. Check that the transform is CHEAPER than the original problem     │
│  4. Verify the CONSERVATION law holds (for correctness)               │
│  5. Implement: Transform → Solve in easy domain → Transform back      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Duality 1: Spatial ↔ Frequency

### 3.1 The Duality

```
SPATIAL ↔ FREQUENCY DUALITY:

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  TRANSFORM: Fast Fourier Transform (FFT)                               │
│  COST: O(N log N)                                                       │
│  CONSERVED: Energy (Parseval: Σ|x|² = (1/N)Σ|X|²)                     │
│                                                                         │
│  COMPLEXITY INVERSION:                                                  │
│  ─────────────────────                                                  │
│  Spatial Domain                   Frequency Domain                     │
│  ─────────────                    ────────────────                     │
│  Convolution: O(N²)               Multiplication: O(N)                 │
│  Global patterns: hard            Global patterns: one coefficient     │
│  Filter design: iterative         Filter design: direct                │
│  Scale analysis: multi-pass       Scale analysis: direct lookup        │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.2 AKIRA Architecture Decision

```
ARCHITECTURE DECISION: SPECTRAL-FIRST PROCESSING

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  PROBLEM: Attention needs to capture patterns at multiple scales       │
│                                                                         │
│  SPATIAL APPROACH (what standard transformers do):                     │
│  • Full O(N²) attention over all positions                             │
│  • Hope the network learns to separate scales                          │
│  • Multi-head attention adds parameters, not structure                 │
│                                                                         │
│  SPECTRAL APPROACH (what AKIRA does):                                  │
│  • FFT decomposes signal into 7 frequency bands: O(N log N)            │
│  • Each band is processed independently: O(N/7) per band               │
│  • Scale separation is GUARANTEED by mathematics                       │
│  • Total: O(N log N) + O(N) = O(N log N)                               │
│                                                                         │
│  ═══════════════════════════════════════════════════════════════════   │
│                                                                         │
│  WHY THIS WORKS:                                                        │
│                                                                         │
│  In frequency domain:                                                   │
│  • Band 0 (DC): Identity, existence — what IS here                     │
│  • Band 6 (High): Position, edges — where EXACTLY                      │
│  • Middle bands: Structure at various scales                           │
│                                                                         │
│  Each band sees its natural information directly.                      │
│  No need to learn frequency decomposition — FFT gives it for free.    │
│                                                                         │
│  CONSERVATION CHECK: Parseval guarantees energy is preserved.          │
│  If Σ|band|² ≠ |signal|², there's a bug.                              │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.3 Implementation Consequence

```
CONCRETE IMPLEMENTATION:

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  INPUT: Embedding E ∈ ℝ^(B × T × D)                                    │
│                                                                         │
│  STEP 1: FFT along embedding dimension (O(D log D))                    │
│  STEP 2: Split into 7 logarithmically-spaced bands                     │
│  STEP 3: Per-band attention (7 × O((T × D/7)²) = O(T² × D²/7))        │
│  STEP 4: IFFT to reconstruct (O(D log D))                              │
│                                                                         │
│  vs. STANDARD ATTENTION: O(T² × D²)                                    │
│                                                                         │
│  SAVINGS: Factor of ~7 in attention computation                        │
│           Plus: bands are semantically meaningful                      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Duality 2: Forward ↔ Backward (Causal ↔ Non-causal)

### 4.1 The Duality

```
FORWARD ↔ BACKWARD DUALITY:

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  TRANSFORM: Time reversal / Adjoint operation                          │
│  COST: O(N) (one additional pass)                                       │
│  CONSERVED: Information (no bits lost)                                  │
│                                                                         │
│  COMPLEXITY INVERSION:                                                  │
│  ─────────────────────                                                  │
│  Forward (Causal)                 Backward (Non-causal)                │
│  ───────────────                  ─────────────────────                 │
│  P(x_t | past): online            P(x_t | all): batch                  │
│  Streaming: yes                   Streaming: no                        │
│  Respects causality: yes          Uses future: yes                     │
│  One hypothesis at a time         Full posterior                       │
│                                                                         │
│  EXAMPLES:                                                              │
│  • Kalman filter (forward) / RTS smoother (backward)                  │
│  • Viterbi (forward) / Backward pass in HMM training                  │
│  • Attention over past / Attention over all positions                 │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.2 AKIRA Architecture Decision

```
ARCHITECTURE DECISION: SEPARATE TEMPORAL AND SPECTRAL ATTENTION

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  PROBLEM: Time and space have different information access rules       │
│                                                                         │
│  • TIME is causal: cannot see future (arrow of time)                   │
│  • SPACE is non-causal: can see all positions (no spatial arrow)      │
│                                                                         │
│  UNIFIED APPROACH (what some architectures do):                        │
│  • Same attention mechanism for both                                   │
│  • Either violate causality or waste capacity with masks              │
│                                                                         │
│  AKIRA APPROACH (the 7+1 principle):                                   │
│  • Spectral Bands 0-6: NON-CAUSAL attention (see all spatial positions)│
│  • Temporal Band 7: CAUSAL attention (lower-triangular mask)          │
│  • These are ORTHOGONAL by Heisenberg uncertainty                      │
│                                                                         │
│  ═══════════════════════════════════════════════════════════════════   │
│                                                                         │
│  WHY THIS WORKS:                                                        │
│                                                                         │
│  Spectral bands ask: "What patterns exist NOW across all space?"      │
│  Temporal band asks: "How do patterns relate to the PAST?"            │
│                                                                         │
│  These are different questions requiring different information access. │
│  The duality says: use forward (causal) for time, backward (full)     │
│  for space. Mixing them loses efficiency or correctness.              │
│                                                                         │
│  HEISENBERG CONNECTION:                                                 │
│  Δf × Δt ≥ constant                                                    │
│  Spectral bands: good Δf, poor Δt                                      │
│  Temporal band: good Δt, poor Δf                                       │
│  They're orthogonal — combine them for complete information.          │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Duality 3: Sharp ↔ Soft (Semiring Interpolation)

### 5.1 The Duality

```
SHARP ↔ SOFT DUALITY (Semiring Interpolation):

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  TRANSFORM: Temperature parameter τ in softmax(x/τ)                    │
│  COST: O(1) (just divide by τ)                                          │
│  CONSERVED: Distribution structure (same ordering of scores)           │
│                                                                         │
│  COMPLEXITY INVERSION:                                                  │
│  ─────────────────────                                                  │
│  Sharp (τ→0, max-product)         Soft (τ→∞, sum-product)             │
│  ────────────────────────         ───────────────────────              │
│  Winner-take-all                  Maintains all hypotheses             │
│  Committed belief                 Uncertain belief                     │
│  MAP estimate                     Full posterior                       │
│  Fast decisions                   Exploration                          │
│                                                                         │
│  SEMIRING VIEW:                                                         │
│  τ→0: (max, ×) semiring — Viterbi, find best path                     │
│  τ=1: (+, ×) semiring — Forward-Backward, compute marginals           │
│  Same algorithm structure, different algebraic operations.             │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 5.2 AKIRA Architecture Decision

```
ARCHITECTURE DECISION: PER-BAND LEARNABLE TEMPERATURE

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  PROBLEM: Different bands should commit at different rates             │
│                                                                         │
│  • Low-freq (identity): Should be STABLE, commit when confident       │
│  • High-freq (position): Should be ADAPTIVE, stay flexible            │
│                                                                         │
│  FIXED TEMPERATURE (what standard attention does):                     │
│  • Same τ=1 everywhere                                                 │
│  • Identity and position treated the same                             │
│  • No principled commitment mechanism                                  │
│                                                                         │
│  AKIRA APPROACH:                                                        │
│  • Per-band learnable temperature                                      │
│  • Band 0: τ ≈ 0.5 (more decisive, identity should commit)            │
│  • Band 6: τ ≈ 1.0 (more exploratory, position stays flexible)        │
│  • The temperature IS the semiring interpolation parameter            │
│                                                                         │
│  ═══════════════════════════════════════════════════════════════════   │
│                                                                         │
│  COLLAPSE AS SEMIRING TRANSITION:                                       │
│                                                                         │
│  Pre-collapse: High τ → sum-product → maintain distribution           │
│  During collapse: τ drops → interpolates toward max-product           │
│  Post-collapse: Low τ → max-product → committed belief                │
│                                                                         │
│  The collapse IS the continuous transition from (+,×) to (max,×).     │
│  Temperature controls where on this spectrum the system operates.     │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Duality 4: Local ↔ Global

### 6.1 The Duality

```
LOCAL ↔ GLOBAL DUALITY:

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  TRANSFORM: Aggregation / Decomposition                                │
│  COST: Varies (attention O(N²), convolution O(N), spectral O(N log N))│
│  CONSERVED: Total information capacity                                  │
│                                                                         │
│  COMPLEXITY INVERSION:                                                  │
│  ─────────────────────                                                  │
│  Local Processing                 Global Processing                    │
│  ────────────────                 ─────────────────                    │
│  Texture, edges                   Identity, structure                  │
│  Fast (O(N))                      Slower (O(N²) naive)                 │
│  Limited context                  Full context                         │
│  High-frequency details           Low-frequency patterns               │
│                                                                         │
│  THE TRADEOFF:                                                          │
│  Local is fast but misses global structure.                           │
│  Global is slow but sees relationships.                                │
│  Wormholes: selective global (top-k) at local cost.                   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 6.2 AKIRA Architecture Decision

```
ARCHITECTURE DECISION: SPARSE WORMHOLE ATTENTION

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  PROBLEM: Need global context without O(N²) cost                       │
│                                                                         │
│  DENSE GLOBAL (what full attention does):                              │
│  • Every position attends to every other                              │
│  • O(N²) per layer                                                     │
│  • Most connections are uninformative                                  │
│                                                                         │
│  PURE LOCAL (what convolutions do):                                    │
│  • Only nearby positions interact                                      │
│  • O(N) per layer                                                       │
│  • Misses long-range dependencies                                      │
│                                                                         │
│  AKIRA APPROACH (Sparse Wormhole):                                     │
│  • Per-band attention is local (within frequency band)                │
│  • Wormhole attention is SPARSE global (top-k connections)            │
│  • Only complementary bands communicate (0↔6, 1↔5, 2↔4)              │
│  • Cost: O(N × k) where k << N                                         │
│                                                                         │
│  ═══════════════════════════════════════════════════════════════════   │
│                                                                         │
│  WHY COMPLEMENTARY PAIRS:                                               │
│                                                                         │
│  Adjacent bands (e.g., 0-1) have HIGH REDUNDANCY:                      │
│  • Similar frequency content → similar information                    │
│  • Connecting them adds little                                         │
│                                                                         │
│  Complementary bands (e.g., 0-6) have HIGH SYNERGY:                    │
│  • Opposite frequency content → complementary information             │
│  • Connecting them enables predictions neither can make alone         │
│                                                                         │
│  (This is formalized by Partial Information Decomposition — see §7)   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 7. Duality 5: Redundancy ↔ Synergy (PID)

### 7.1 The Duality

```
REDUNDANCY ↔ SYNERGY DUALITY (Partial Information Decomposition):

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  TRANSFORM: PID decomposition of mutual information                    │
│  COST: O(exponential in sources) — use pairwise approximations        │
│  CONSERVED: Total mutual information I(Target; All Sources)            │
│                                                                         │
│  COMPLEXITY INVERSION:                                                  │
│  ─────────────────────                                                  │
│  Redundancy                       Synergy                              │
│  ──────────                       ───────                              │
│  Same info in multiple sources    Emergent info from combination      │
│  Any source alone suffices        Need ALL sources together           │
│  Post-collapse state              Pre-collapse state                   │
│  Low uncertainty                  High uncertainty                     │
│                                                                         │
│  FORMULA:                                                               │
│  I(Target; S1, S2) = Redundancy + Unique(S1) + Unique(S2) + Synergy   │
│                                                                         │
│  Reference: Williams & Beer (2010). arXiv:1004.2515                    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 7.2 AKIRA Architecture Decision

```
ARCHITECTURE DECISION: COLLAPSE AS SYNERGY→REDUNDANCY CONVERSION

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  PROBLEM: What IS collapse, information-theoretically?                 │
│                                                                         │
│  PRE-COLLAPSE STATE:                                                    │
│  • Bands hold different hypotheses                                     │
│  • Need ALL bands to predict target (HIGH SYNERGY)                    │
│  • Any single band is insufficient (LOW REDUNDANCY)                   │
│                                                                         │
│  POST-COLLAPSE STATE:                                                   │
│  • All bands agree on winner                                           │
│  • ANY band can predict target (HIGH REDUNDANCY)                      │
│  • Combining bands adds nothing (LOW SYNERGY)                         │
│                                                                         │
│  ═══════════════════════════════════════════════════════════════════   │
│                                                                         │
│  THE INSIGHT:                                                           │
│                                                                         │
│  Collapse = Synergy → Redundancy conversion (PID perspective)         │
│  Total information I(Target; All Bands) is CONSERVED                   │
│  But its DECOMPOSITION changes                                          │
│  (See TERMINOLOGY.md for formal Bayesian definition)                  │
│                                                                         │
│  This is a PHASE TRANSITION in information structure.                  │
│  Like water→ice: same molecules, different arrangement.               │
│  Like synergy→redundancy: same total info, different distribution.   │
│                                                                         │
│  ARCHITECTURAL CONSEQUENCE:                                             │
│  Wormholes REALIZE synergy — they let bands combine to predict.       │
│  Without wormholes, synergy exists but is UNREALIZED.                 │
│  Collapse detection: track R/S ratio over time.                       │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 8. Duality 6: Magnitude ↔ Phase

### 8.1 The Duality

```
MAGNITUDE ↔ PHASE DUALITY:

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  TRANSFORM: Polar decomposition of complex FFT coefficients            │
│  COST: O(N) (just compute |z| and arg(z))                              │
│  CONSERVED: Complex structure (magnitude × e^(i×phase) = original)    │
│                                                                         │
│  COMPLEXITY INVERSION:                                                  │
│  ─────────────────────                                                  │
│  Magnitude                        Phase                                │
│  ─────────                        ─────                                │
│  How much energy at each freq     Where patterns are aligned          │
│  Texture, identity (WHAT)         Position, structure (WHERE)         │
│  Robust to translation            Sensitive to translation            │
│  Determines power spectrum        Determines spatial layout           │
│                                                                         │
│  CLASSIC RESULT:                                                        │
│  Swap phases between two images → result looks like phase donor       │
│  Phase carries structure; magnitude carries texture.                  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 8.2 AKIRA Architecture Decision

```
ARCHITECTURE DECISION: WHAT-PATH USES MAGNITUDE, WHERE-PATH USES PHASE

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  PROBLEM: WHAT and WHERE are different computational questions         │
│                                                                         │
│  WHAT (identity):                                                       │
│  • "Is this a cat?"                                                    │
│  • Translation-invariant (cat is cat regardless of position)          │
│  • Dominated by magnitude (texture, frequency content)                │
│  • Low-frequency bands (0, 1, 2)                                       │
│                                                                         │
│  WHERE (position):                                                      │
│  • "Where is the edge?"                                                │
│  • Translation-sensitive (position matters!)                          │
│  • Dominated by phase (alignment, structure)                          │
│  • High-frequency bands (4, 5, 6)                                      │
│                                                                         │
│  ═══════════════════════════════════════════════════════════════════   │
│                                                                         │
│  WORMHOLE AS WHAT↔WHERE BRIDGE:                                        │
│                                                                         │
│  Band 0 (low-freq, magnitude-dominant) knows WHAT but not WHERE       │
│  Band 6 (high-freq, phase-dominant) knows WHERE but not WHAT          │
│                                                                         │
│  Wormhole 0→6: "I know WHAT this is, WHERE exactly?"                  │
│  Wormhole 6→0: "I see something HERE, WHAT is it?"                    │
│                                                                         │
│  This is the Heisenberg tradeoff expressed at the band level.         │
│  The wormhole connects the two sides of the duality.                  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 9. Summary: Architecture Decisions from Dualities

### 9.1 The Complete Picture

```
AKIRA ARCHITECTURE: DUALITY-DERIVED DECISIONS

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  DUALITY               TRANSFORM         DECISION                      │
│  ───────               ─────────         ────────                      │
│                                                                         │
│  Spatial↔Frequency     FFT O(N log N)    Spectral-first processing    │
│                                          7 logarithmic bands           │
│                                                                         │
│  Causal↔Non-causal     Mask structure    7 spatial + 1 temporal band  │
│                                          Spectral non-causal, time causal│
│                                                                         │
│  Sharp↔Soft            Temperature τ     Per-band learnable τ         │
│                                          Collapse = semiring transition │
│                                                                         │
│  Local↔Global          Top-k sparse      Sparse wormhole attention    │
│                                          Complementary band pairs only │
│                                                                         │
│  Redundancy↔Synergy    PID decomp        High-synergy pairs connected │
│                                          Collapse = syn→red conversion │
│                                                                         │
│  Magnitude↔Phase       Polar coords      Low bands: WHAT (magnitude)  │
│                                          High bands: WHERE (phase)     │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 9.2 Conservation Laws as Validation

```
CONSERVATION LAWS FOR TESTING:

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  Every duality implies a conservation law. Use them for validation:   │
│                                                                         │
│  CONSERVATION             TEST                    IF FAILS             │
│  ────────────             ────                    ────────             │
│  Parseval (energy)        Σ|band|² = |signal|²   FFT bug              │
│  PID (total MI)           I_total constant        Information leak     │
│  Attention normalization  Σ weights = 1           Softmax bug          │
│  Gradient flow            ⟨∇L, v⟩ preserved      Autodiff bug         │
│                                                                         │
│  These are not optional. They're mathematical guarantees.             │
│  If they fail, the implementation is wrong.                           │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 9.3 The Duality Design Principle

```
THE DUALITY DESIGN PRINCIPLE:

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  1. IDENTIFY what's computationally HARD in your problem               │
│                                                                         │
│  2. FIND a duality where it becomes EASY                               │
│     (consult the catalog: Fourier, LP, spectral, etc.)                │
│                                                                         │
│  3. VERIFY the transform is CHEAPER than the original problem          │
│     (Transform cost + Easy domain cost < Hard domain cost)            │
│                                                                         │
│  4. IDENTIFY the CONSERVED QUANTITY                                    │
│     (This is your validation criterion)                               │
│                                                                         │
│  5. IMPLEMENT: Transform → Solve in easy domain → Transform back       │
│                                                                         │
│  6. TEST: Check conservation law holds                                 │
│                                                                         │
│  ═══════════════════════════════════════════════════════════════════   │
│                                                                         │
│  Every architectural decision in AKIRA follows this pattern.          │
│  It's not ad-hoc. It's duality-driven.                                │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 10. References

### Internal Documents

| Document | Location | Relevance |
|----------|----------|-----------|
| SPECTRAL_BELIEF_MACHINE | `architecture_theoretical/` | Complete architecture spec |
| ORTHOGONALITY | `architecture_theoretical/` | §11.3: PID and synergy |
| SPECTRAL_WORMHOLE_ATTENTION | `architecture_base/attention/` | §3.3: Complementary pairs |
| COLLAPSE_DYNAMICS | `architecture_base/collapse/` | Temperature, phase transition |
| KNOWLEDGE_AND_REACTIVITY | `foundations/` | Energy vs geometry duality |
| EQUILIBRIUM_AND_CONSERVATION | `foundations/` | Conservation principles |
| DUALITY_METHODS | `observability/` | Companion doc: dualities for OBSERVABILITY (hard/easy swaps) |
| PANDORA | `pandora/` | Action as transformation between dual forms |

### External References

1. **Cooley, J. W., & Tukey, J. W. (1965).** "An algorithm for the machine calculation of complex Fourier series." *Mathematics of Computation*, 19(90), 297-301. — FFT algorithm.

2. **Parseval, M. A. (1806).** "Mémoire sur les séries et sur l'intégration complète..." — Energy conservation under Fourier.

3. **Williams, P.L., & Beer, R.D. (2010).** "Nonnegative Decomposition of Multivariate Information." *arXiv:1004.2515*. — Partial Information Decomposition.

4. **Heisenberg, W. (1927).** "Über den anschaulichen Inhalt der quantentheoretischen Kinematik und Mechanik." *Zeitschrift für Physik*, 43(3-4), 172-198. — Uncertainty principle.

5. **Rockafellar, R.T. (1970).** *Convex Analysis*. Princeton University Press. — Legendre/Fenchel duality.

6. **Griewank, A., & Walther, A. (2008).** *Evaluating Derivatives: Principles and Techniques of Algorithmic Differentiation*. SIAM. — Forward/backward mode autodiff.

---

*Oscar Goldman — Shogu Research Group @ Datamutant.ai*

*"Every productive duality has a transform, a conserved quantity, and a complexity inversion. The architecture follows from applying this principle systematically. It's not magic — it's mathematics which sometimes feels like magic."*

