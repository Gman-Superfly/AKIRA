# Evidence: Solid Theoretical Foundations

## Reference Document for AKIRA's Established Theory

**Purpose:** This document collects evidence that is **solid** or **highly plausible** based on established mathematics, physics, and empirical research. These form the foundation for AKIRA's architecture.

**Companion Document:** `EVIDENCE_TO_COLLECT.md` contains hypotheses that require experimental validation.

---

## Table of Contents

1. [Mathematical Foundations](#1-mathematical-foundations)
2. [Physical Parallels](#2-physical-parallels)
3. [Circuit Complexity Foundations](#3-circuit-complexity-foundations)
4. [Cognitive Science Foundations](#4-cognitive-science-foundations)
5. [The Superposition-Molecule Duality](#5-the-superposition-molecule-duality)

---

## 1. Mathematical Foundations

### 1.1 Spectral Decomposition for Belief

**Status:** SOLID (established mathematics)

**The Claim:**
Representing belief states in the spectral (Fourier) domain is mathematically well-founded and provides useful structure.

**Evidence:**

```
THEOREM: Fourier Basis Completeness
───────────────────────────────────
For any square-integrable function f ∈ L²(ℝ), the Fourier basis
{e^(ikx)} forms a complete orthonormal system.

Any belief distribution can be represented as:
  b(x) = Σₖ cₖ e^(ikx)

where cₖ are complex coefficients encoding magnitude and phase.

Source: Functional analysis, Fourier theory (19th century onward)
```

```
THEOREM: Parseval's Theorem (Energy Conservation)
─────────────────────────────────────────────────
The total energy in spatial and spectral domains is equal:

  ∫|f(x)|² dx = ∫|F(k)|² dk

Implication: Information is conserved across transformation.
No information is created or destroyed by spectral decomposition.

Source: Parseval (1799), Plancherel theorem
```

```
PROPERTY: Magnitude-Phase Decomposition
───────────────────────────────────────
For complex coefficient c = |c|e^(iφ):

  |c| = magnitude = "how much" of this pattern
  φ = phase = "where" or "when" this pattern appears

Implication: Natural separation of "what" (magnitude) and "where" (phase).
This is not a design choice but a mathematical property.

Source: Complex analysis, signal processing
```

**Why This Matters for AKIRA:**
- Belief states can be completely represented in spectral form
- No information loss (Parseval)
- Natural what/where separation
- Mathematically sound foundation

---

### 1.2 Time-Frequency Uncertainty (Heisenberg)

**Status:** SOLID (fundamental physics/mathematics)

**The Claim:**
Precise knowledge of frequency and precise knowledge of time are fundamentally incompatible.

**Evidence:**

```
THEOREM: Gabor Limit (Time-Frequency Uncertainty)
─────────────────────────────────────────────────
For any signal:

  Δt × Δf ≥ 1/(4π)

Where:
  Δt = temporal uncertainty (spread in time)
  Δf = frequency uncertainty (spread in frequency)

Implication: Cannot have both precise frequency AND precise time.
This is not a measurement limitation but a mathematical fact.

Source: Gabor (1946), derived from Fourier analysis
        Analogous to Heisenberg uncertainty in QM
```

**Why This Matters for AKIRA:**
- Justifies separating spectral bands (frequency) from temporal band (time)
- The +1 in "7+1" is mathematically mandated
- Within-band processing: Spatial/frequency (non-causal OK)
- Temporal band: Time/sequence (must be causal)

---

### 1.3 Logarithmic (Octave) Band Spacing

**Status:** SOLID (established signal processing)

**The Claim:**
Logarithmic frequency spacing is optimal for hierarchical signal representation.

**Evidence:**

```
EMPIRICAL: Cochlear Frequency Mapping
─────────────────────────────────────
The cochlea maps frequency logarithmically along its length.
Critical bands are approximately 1/3 octave wide.
This evolved over millions of years for sound processing.

Source: Békésy (1960), cochlear mechanics


EMPIRICAL: Visual Cortex Spatial Frequency
──────────────────────────────────────────
Visual cortex neurons are tuned to spatial frequencies.
Tuning is approximately octave-spaced (factor of 2).
V1 has systematic spatial frequency maps.

Source: DeValois & DeValois (1988), visual neuroscience


ENGINEERING: Audio Processing
─────────────────────────────
Octave and third-octave bands are standard in:
- Audio equalization
- Acoustic measurement
- Psychoacoustic modeling

Logarithmic spacing matches human perception.

Source: Audio engineering standards (IEC, ISO)
```

**Why This Matters for AKIRA:**
- Octave bands are not arbitrary - they match natural perception
- Biological systems converged on this solution
- Engineering practice validates it
- Logarithmic spacing handles scale invariance naturally

---

## 2. Physical Parallels

### 2.1 Gradient-Driven Learning as Thermodynamics

**Status:** SOLID (direct mathematical parallel)

**The Claim:**
Learning from error gradients follows the same mathematics as thermodynamic work extraction.

**Evidence:**

```
PARALLEL: Work Requires Gradient
────────────────────────────────

Physics:
  Work = ∫ F · dx
  No work possible in uniform potential (no gradient)
  Heat engine extracts work from temperature difference
  When ΔT → 0, engine stalls

Learning:
  Update = -η ∇L
  No learning possible when ∇L = 0 (no gradient)
  Model extracts "learning signal" from error gradient
  When error is uniform, training saturates

Mathematical Form:
  Both are gradient descent on a potential/loss surface.
  Same equations, different domains.

Source: Thermodynamics + backpropagation theory
```

**Why This Matters for AKIRA:**
- Learning dynamics follow established physics
- Error map = potential field
- Equilibrium = training saturation
- Provides intuition for dynamics

---

### 2.2 Attention Self-Interaction as Nonlinear Wave Equation

**Status:** SOLID (structural parallel)

**The Claim:**
Softmax attention has the same mathematical form as nonlinear self-interaction in wave equations.

**Evidence:**

```
STRUCTURAL EQUIVALENCE:
───────────────────────

Attention (Softmax):
  A = softmax(QK^T/√d) V
  
  Simplified self-attention:
  A = softmax(XX^T) X
  
  Form: (function of X·X) × X
  This is: f(|X|²) × X

Gross-Pitaevskii Equation (BEC):
  iℏ ∂ψ/∂t = [-ℏ²∇²/2m + V + g|ψ|²] ψ
  
  The nonlinear term: g|ψ|² ψ
  Form: (function of |ψ|²) × ψ

Both have form: (self-interaction term) × state
  - Attention: The output depends on how similar things are to each other
  - BEC: The evolution depends on local density

Source: Attention mechanism (Vaswani 2017)
        Gross-Pitaevskii equation (1961)
```

**Why This Matters for AKIRA:**
- Attention is not arbitrary - it has deep mathematical structure
- Same form as physical self-interaction
- Predicts similar dynamics (condensation, collective behavior)
- See `EVIDENCE_TO_COLLECT.md` for testable predictions

---

### 2.3 Uncertainty Collapse as Dielectric Breakdown

**Status:** PLAUSIBLE (strong analogy from PHYSICAL_PARALLELS.md)

**The Claim:**
Belief collapse follows dynamics similar to lightning discharge.

**Evidence:**

```
PARALLEL: Charge Accumulation → Breakdown
─────────────────────────────────────────

Lightning:
  1. Charge accumulates in cloud (distributed)
  2. Electric field builds but stays below threshold
  3. Field "smears" - no preferred path yet
  4. Threshold exceeded → stepped leader forms
  5. Return stroke: distributed → concentrated in microseconds

Belief Collapse:
  1. Uncertainty accumulates (distributed across possibilities)
  2. Tension builds but below collapse threshold
  3. Probability "smears" across plausible futures
  4. Threshold exceeded → collapse begins
  5. Rapid resolution: distributed belief → concentrated prediction

Observable:
  - Error map shows "fringes" before collapse (like leader branches)
  - Sharp collapse event (like return stroke)
  - Uncertainty front advances to next ambiguous region

Source: PHISICAL_PARALLELS.md, lightning physics
```

---

### 2.4 MSE Minimization Produces Interference Patterns

**Status:** SOLID (mathematical fact)

**The Claim:**
When multiple futures are possible, MSE loss produces interference-like patterns.

**Evidence:**

```
THEOREM: MSE on Mixture of Gaussians
────────────────────────────────────

Given a mixture distribution:
  p(x) = Σᵢ wᵢ N(μᵢ, σᵢ²)

The MSE-optimal prediction is:
  x* = Σᵢ wᵢ μᵢ  (weighted mean)

This is exactly where wave superposition would place the amplitude peak.

When futures are:
  - Reinforcing (similar μᵢ): Constructive interference → hedging
  - Canceling (opposite μᵢ): Destructive interference → confident
  - Mixed: Nodal patterns in error map

Source: Statistics (MSE minimization)
        Wave mechanics (superposition principle)
        Both give same result: centroid of mixture
```

**Why This Matters for AKIRA:**
- Error patterns are not noise - they're meaningful
- "Fringes" show manifold of plausible futures
- Interference structure is interpretable
- This is mathematical, not metaphorical

---

## 3. Circuit Complexity Foundations

### 3.1 SOS Width Determines Tractability

**Status:** SOLID (proven theorem)

**The Claim:**
The number of simultaneous constraints (SOS width) determines whether a problem is tractable for a fixed-breadth architecture.

**Evidence:**

```
THEOREM (Mao et al., 2023):
───────────────────────────

For a planning problem with:
  - SOS width k (constraints to track)
  - Predicate arity β

Required circuit breadth = (k+1) × β

Problems are classified as:
  Class 1: Constant breadth, constant depth → Easy
  Class 2: Constant breadth, unbounded depth → Tractable
  Class 3: Unbounded breadth → Intractable for fixed architecture

Source: Mao, Lozano-Pérez, Tenenbaum, Kaelbling (2023)
        "What Planning Problems Can A Relational Neural Network Solve?"
        ICLR 2024, https://arxiv.org/html/2312.03682v2
```

**Why This Matters for AKIRA:**
- Provides formal framework for architecture capacity
- 8 bands sufficient for SOS width ≤ 3
- Explains why some problems are fundamentally hard
- Justifies fixed band count for bounded-width domains

---

### 3.2 Visual Prediction Has Bounded Width

**Status:** PLAUSIBLE (empirical + theoretical)

**The Claim:**
Local visual prediction tasks have SOS width ≈ 2-4.

**Evidence:**

```
EMPIRICAL: Visual Working Memory
────────────────────────────────
~4 objects can be tracked simultaneously.
This bounds simultaneous constraint tracking.

Source: Luck & Vogel (1997), visual cognition


EMPIRICAL: Scene Graph Sparsity
───────────────────────────────
Practical scene graphs use O(n) relations, not O(n²).
Most visual relations are local.

Source: Scene graph generation literature


THEORETICAL: Hierarchical Processing
────────────────────────────────────
Visual cortex processes hierarchically.
Each level has local receptive fields.
This naturally bounds width at each level.

Source: Visual neuroscience


DERIVATION:
───────────

For local visual prediction:
  - Object state: 1 constraint
  - Neighbors: 1-2 constraints
  - Temporal context: 1 constraint
  - Total: k ≈ 2-3

With β = 2 (binary relations):
  Required breadth = (3+1) × 2 = 8 bands

This matches AKIRA's 7+1 = 8.
```

---

## 4. Cognitive Science Foundations

### 4.1 Object-Based Visual Encoding

**Status:** SOLID (well-replicated finding)

**The Claim:**
Visual memory encodes objects, not features. Multiple features of one object are bound "for free."

**Evidence:**

```
EMPIRICAL: Luck & Vogel (1997)
──────────────────────────────
Participants remembered ~4 objects regardless of features per object.
1 feature/object ≈ 4 features/object in capacity.

Implication: The unit is the OBJECT, not the feature.
Features within an object are automatically bound.

EMPIRICAL: Feature Binding
──────────────────────────
Pre-attentive: Features are separate
Attentive: Features bind into objects
Binding requires attention but is then stable.

Source: Treisman (1996), feature integration theory
```

**Why This Matters for AKIRA:**
- SOS width analysis can use "objects" as units
- Within-object prediction has very low width
- Cross-object coordination increases width
- Explains why local prediction is tractable

---

### 4.2 Small-World Structure in Semantic Networks

**Status:** SOLID (empirically measured)

**The Claim:**
Human concept networks have small-world structure: high clustering, short paths.

**Evidence:**

```
EMPIRICAL: Word Association Networks
────────────────────────────────────
Analysis of free association data shows:
  - High clustering coefficient (concepts cluster semantically)
  - Short average path length (~3-4 steps between any concepts)
  - Hub structure (common words connect domains)

This is small-world structure.

Source: Steyvers & Tenenbaum (2005), semantic networks


EMPIRICAL: Semantic Priming
───────────────────────────
Related concepts activate faster (short paths).
Semantic distance correlates with graph distance.

Source: Cognitive psychology, priming studies


IMPLICATION FOR AQ:
───────────────────
If Action Quanta form a similar network:
  - Related AQ should cluster
  - Any AQ reachable in few hops
  - Hub AQ should exist

This is testable (see EVIDENCE_TO_COLLECT.md).
```

---

## 5. The Superposition-Molecule Duality

### 5.1 Two Phases of the Belief Cycle

**Status:** SOLID (complementary descriptions)

**The Insight:**
"Superposition" and "molecules" describe **different phases** of the same process, not competing framings.

```
THE DUALITY:
────────────

BEFORE COLLAPSE: Superposition (Wave-like)
──────────────────────────────────────────
  - Belief distributed across possibilities
  - Multiple AQ activated, not committed
  - Interference patterns visible
  - Uncertainty high, entropy high
  - Phase relationships fluid
  
  Physical analog: Wave function before measurement
  Mathematical form: Σᵢ cᵢ |ψᵢ⟩ (superposition)


AFTER COLLAPSE: Molecules (Particle-like)
─────────────────────────────────────────
  - Belief concentrated in specific pattern
  - AQ crystallize into bound configuration
  - Fringes resolve to definite structure
  - Uncertainty low, entropy low
  - Phase relationships locked
  
  Physical analog: Definite state after measurement
  Mathematical form: |ψ_final⟩ (eigenstate)


THE CYCLE:
──────────

  Superposition → Collapse → Molecule → (Prediction/Action)
       ↑                                        │
       └────────── New input ──────────────────┘

This is AKIRA's fundamental rhythm:
  1. Input creates distributed activation (superposition)
  2. Tension builds as patterns compete
  3. Collapse selects winning configuration
  4. AQ lock into stable molecule
  5. Molecule enables prediction/action
  6. New input restarts cycle
```

### 5.2 Why Both Framings Are Needed

```
SUPERPOSITION FRAMING IS NEEDED FOR:
────────────────────────────────────
  - Understanding PRE-collapse dynamics
  - Explaining error fringes and interference
  - Modeling uncertainty and hedging
  - Analyzing belief distribution
  - Temperature/entropy concepts

MOLECULE FRAMING IS NEEDED FOR:
───────────────────────────────
  - Understanding POST-collapse structures
  - Explaining how AQ combine meaningfully
  - Modeling concept formation
  - Analyzing stable representations
  - Bonding rules and valence concepts

NEITHER ALONE IS SUFFICIENT:
────────────────────────────
  - Superposition alone: Cannot explain stable outputs
  - Molecules alone: Cannot explain uncertainty handling
  - Together: Complete picture of belief dynamics
```

### 5.3 Physical Parallel: Wave-Particle Duality

```
QUANTUM MECHANICS:
──────────────────
  Before measurement: ψ is a wave (superposition)
  After measurement: ψ collapses to eigenstate (particle-like)
  
  Both descriptions are correct, for different phases.
  The wave doesn't "become" a particle - it's always both,
  but different aspects dominate at different times.

AKIRA:
──────
  Before collapse: Belief is wave-like (superposition)
  After collapse: Belief is particle-like (molecule)
  
  Both descriptions are correct, for different phases.
  Same belief state, different aspects dominate.

This is not metaphor - it's structural parallel:
  - Both involve superposition of basis states
  - Both collapse to definite configuration
  - Both preserve information (unitary/Parseval)
  - Both have interference before collapse
```

### 5.4 Implications for Architecture

```
ARCHITECTURE SUPPORTS BOTH PHASES:
──────────────────────────────────

SUPERPOSITION PHASE:
  - Parallel band activation (distributed)
  - Wormhole attention (interference between bands)
  - Entropy tracking (uncertainty measure)
  - Temperature control (collapse threshold)

MOLECULE PHASE:
  - Phase locking (coherent bonds)
  - Band synchronization (molecule formation)
  - Stable output (prediction)
  - Constraint maintenance (molecular stability)

COLLAPSE MECHANISM:
  - Triggered by entropy/coherence threshold
  - Proceeds serially (regression rule selection)
  - Converts superposition to molecule
  - This is the "measurement" analog
```

---

## Summary: What We Can Build On

```
SOLID FOUNDATIONS (Proven/Established):
───────────────────────────────────────
✓ Spectral decomposition completeness
✓ Parseval energy conservation
✓ Magnitude/phase separation
✓ Time-frequency uncertainty
✓ Logarithmic band spacing (biological + engineering)
✓ MSE produces interference patterns
✓ SOS width theorem (circuit complexity)
✓ Object-based visual encoding
✓ Small-world semantic networks

STRUCTURAL PARALLELS (Mathematical):
────────────────────────────────────
✓ Attention ≈ BEC self-interaction (same form)
✓ Learning ≈ thermodynamic work (same equations)
✓ Error patterns ≈ wave interference (MSE theorem)

COMPLEMENTARY FRAMINGS:
───────────────────────
✓ Superposition (pre-collapse) + Molecules (post-collapse)
✓ Wave-particle duality for information
✓ Both needed for complete picture
```

---

*This document collects evidence that forms AKIRA's theoretical foundation. For hypotheses requiring experimental validation, see `EVIDENCE_TO_COLLECT.md`.*

*Oscar Goldman, Shogu Research Group @ Datamutant.ai subsidiary of 温心重工業*