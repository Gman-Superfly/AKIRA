# Wormhole Attention: Explicit Belief with Geometric Trigger

A proposed alternative to the current wormhole implementation, based on the Neuro-Symbolic Homeostat framework. Uses explicit belief representation (precision/curvature) with geometric triggering mechanisms.

---

## Table of Contents

1. [Overview](#1-overview)
2. [The Homeostat Insight](#2-the-homeostat-insight)
3. [Explicit Belief = Precision](#3-explicit-belief--precision)
4. [Geometric Triggers](#4-geometric-triggers)
5. [The Complete Design](#5-the-complete-design)
6. [Comparison: Implicit vs Explicit](#6-comparison-implicit-vs-explicit)
7. [Implementation Sketch](#7-implementation-sketch)
8. [When to Use Which](#8-when-to-use-which)

---

## 1. Overview

### 1.1 The Proposed Design

```
WORMHOLE EXPLICIT: EXPLICIT BELIEF + GEOMETRIC TRIGGER

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  BELIEF REPRESENTATION: EXPLICIT (Precision/Curvature)         │
│  ─────────────────────────────────────────────────────          │
│                                                                 │
│  • Maintain explicit precision Λ (inverse variance)            │
│  • Precision IS the geometry of belief                         │
│  • High precision = concentrated belief (certain)              │
│  • Low precision = spread belief (uncertain)                   │
│  • Can compute entropy: H ∝ -log|Λ|                           │
│  • Can regularize directly                                      │
│                                                                 │
│  ════════════════════════════════════════════════════════════  │
│                                                                 │
│  TRIGGER MECHANISM: GEOMETRIC                                   │
│  ────────────────────────────                                   │
│                                                                 │
│  • Trigger based on curvature/precision relationships          │
│  • Gershgorin margins for stability                            │
│  • Relative position in belief landscape                       │
│  • Downstream benefit (non-local geometry)                     │
│  • Null-space structure (where is uncertainty?)               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 The Core Insight from Homeostat

```
FROM THE NEURO-SYMBOLIC HOMEOSTAT PAPER:

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  "Precision (inverse variance) is stiffness."                 │
│                                                                 │
│  Precision Λ = ∂²F/∂x² = curvature of energy surface          │
│                                                                 │
│  High Λ → Sharp valley → Confident belief                     │
│  Low Λ → Flat region → Uncertain belief                       │
│                                                                 │
│  ════════════════════════════════════════════════════════════  │
│                                                                 │
│  THE GEOMETRIC PRINCIPLE:                                       │
│                                                                 │
│  Don't ask: "Is this value high enough?" (energy)             │
│  Ask: "What is the SHAPE of belief around this point?"        │
│       "Is this connection STABLE given the full structure?"   │
│       "Does this help DOWNSTREAM?" (non-local geometry)       │
│                                                                 │
│  This is fundamentally geometric.                              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. The Homeostat Insight

### 2.1 Four Lenses on the Same Object

```
THE HOMEOSTAT VIEWS BELIEF THROUGH FOUR LENSES:

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  1. PHYSICS (Energy Minimization)                               │
│     ─────────────────────────────                               │
│     Belief = position in energy surface                         │
│     Update = relax toward ground state                         │
│     Precision = curvature (stiffness)                          │
│                                                                 │
│  2. CONTROL THEORY (H∞ Robustness)                             │
│     ──────────────────────────────                              │
│     Small-gain constraints: loop gains < 1                     │
│     Gershgorin bounds ensure contraction                       │
│     Trigger = stability margin check                           │
│                                                                 │
│  3. STATISTICS (Gaussian Graphical Models)                     │
│     ─────────────────────────────────────                       │
│     Precision matrix J = inverse covariance                    │
│     Couplings = off-diagonal J entries                         │
│     Belief update = message passing (GaBP)                     │
│                                                                 │
│  4. INFORMATION THEORY (Channel Capacity)                      │
│     ─────────────────────────────────────                       │
│     Precision = signal-to-noise ratio                          │
│     Low precision = noisy channel                              │
│     Adapt update strength to SNR                               │
│                                                                 │
│  ════════════════════════════════════════════════════════════  │
│                                                                 │
│  ALL FOUR ARE GEOMETRIC:                                        │
│  They describe the SHAPE and STRUCTURE of belief,              │
│  not just scalar magnitudes.                                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Precision-Scaled Updates

```
THE HOMEOSTAT UPDATE RULE:

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  STANDARD GRADIENT DESCENT:                                     │
│                                                                 │
│     Δx = -α · ∇F                                               │
│                                                                 │
│  Same step size everywhere.                                    │
│  Ignores geometry of the landscape.                            │
│                                                                 │
│  ────────────────────────────────────────────────────────────  │
│                                                                 │
│  PRECISION-SCALED UPDATE (Homeostat):                          │
│                                                                 │
│     Δx_i = -∇F_i / (Λ_ii + ε)                                 │
│                                                                 │
│  Step size INVERSELY proportional to precision.                │
│  High precision → small step (already certain)                │
│  Low precision → large step (explore more)                    │
│                                                                 │
│  THIS IS GEOMETRIC:                                             │
│  The update respects the curvature structure.                  │
│  It's Newton-like, not gradient descent.                       │
│                                                                 │
│  ────────────────────────────────────────────────────────────  │
│                                                                 │
│  FOR WORMHOLE:                                                  │
│                                                                 │
│  Connection strength update:                                    │
│  Δw_ij = -∂F/∂w_ij / (Λ_ij + ε)                               │
│                                                                 │
│  Connections with high precision (confident) update slowly.   │
│  Connections with low precision (uncertain) update quickly.   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2.3 The Wormhole Effect in Homeostat

```
NON-LOCAL CREDIT ASSIGNMENT:

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  FROM THE PAPER (Section 2.3):                                  │
│                                                                 │
│  F_gate = -w · η_gate · Δ_benefit                              │
│                                                                 │
│  The gradient w.r.t. the gate:                                 │
│                                                                 │
│  ∂F/∂η_gate = -w · Δ_benefit                                  │
│                                                                 │
│  ════════════════════════════════════════════════════════════  │
│                                                                 │
│  KEY INSIGHT:                                                   │
│                                                                 │
│  The gradient is INDEPENDENT of current gate value!           │
│  Even a CLOSED gate receives gradient from downstream.        │
│                                                                 │
│  This is GEOMETRIC because:                                     │
│  • It uses the RELATIONSHIP between gate and benefit          │
│  • It doesn't just threshold the gate value                   │
│  • It provides non-local credit assignment                    │
│                                                                 │
│  ────────────────────────────────────────────────────────────  │
│                                                                 │
│  FOR WORMHOLE:                                                  │
│                                                                 │
│  A potential connection that is currently below threshold     │
│  can still receive gradient if it would HELP downstream.      │
│                                                                 │
│  This breaks the "zero gradient deadlock" of energy triggers. │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Explicit Belief = Precision

### 3.1 What is Precision?

```
PRECISION AS EXPLICIT BELIEF:

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  DEFINITION:                                                    │
│                                                                 │
│  Precision Λ = Σ⁻¹ (inverse covariance)                       │
│              = -∂²log p(x)/∂x² (Fisher information)           │
│              = ∂²F/∂x² (Hessian of energy)                    │
│                                                                 │
│  For a Gaussian belief N(μ, Σ):                                │
│  • Mean μ = most likely state                                  │
│  • Precision Λ = Σ⁻¹ = confidence in that belief              │
│                                                                 │
│  ════════════════════════════════════════════════════════════  │
│                                                                 │
│  GEOMETRIC INTERPRETATION:                                      │
│                                                                 │
│  High precision:                                                │
│                                                                 │
│     ╭───╮                                                      │
│     │███│   Sharp peak, narrow valley                          │
│     ╰───╯   "I'm very confident about this"                   │
│                                                                 │
│  Low precision:                                                 │
│                                                                 │
│   ╭───────────╮                                                │
│   │░░░░░░░░░░░│   Flat region, wide valley                     │
│   ╰───────────╯   "I'm uncertain, many states plausible"      │
│                                                                 │
│  The SHAPE of the belief is captured by precision.            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Precision for Wormhole Connections

```
APPLYING PRECISION TO WORMHOLE:

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  FOR EACH POTENTIAL CONNECTION (query_i, key_j):               │
│                                                                 │
│  Compute connection precision:                                  │
│                                                                 │
│     Λ_ij = ∂²F/∂w_ij² = curvature at this connection          │
│                                                                 │
│  ────────────────────────────────────────────────────────────  │
│                                                                 │
│  WHAT CONTRIBUTES TO Λ_ij?                                      │
│                                                                 │
│  1. SIMILARITY CURVATURE                                        │
│     How sharply does similarity change around this pair?      │
│     Sharp peak in similarity = high precision                  │
│                                                                 │
│  2. ATTENTION CURVATURE                                         │
│     ∂²(softmax)/∂w² near this connection                      │
│     Dominant connection = high precision                       │
│                                                                 │
│  3. DOWNSTREAM CURVATURE                                        │
│     How does this connection affect output energy?            │
│     Strong causal link = high precision                        │
│                                                                 │
│  ────────────────────────────────────────────────────────────  │
│                                                                 │
│  EXPLICIT BELIEF MEANS:                                         │
│                                                                 │
│  We COMPUTE and STORE Λ_ij for each connection.               │
│  This is the explicit probability/confidence we have.         │
│  We can inspect it, regularize it, threshold it.              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 3.3 Entropy from Precision

```
COMPUTING ENTROPY EXPLICITLY:

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  FOR GAUSSIAN BELIEF:                                           │
│                                                                 │
│  H(p) = ½ log|2πeΣ| = ½ log|2πe| - ½ log|Λ|                  │
│                                                                 │
│  Entropy ∝ -log(precision)                                     │
│                                                                 │
│  High precision → Low entropy → Concentrated belief           │
│  Low precision → High entropy → Spread belief                 │
│                                                                 │
│  ════════════════════════════════════════════════════════════  │
│                                                                 │
│  FOR ATTENTION WEIGHTS:                                         │
│                                                                 │
│  Given explicit precision Λ, we can compute:                   │
│                                                                 │
│  entropy = -Σ_j w_ij log(w_ij)                                │
│                                                                 │
│  OR approximate via precision:                                  │
│                                                                 │
│  effective_entropy ≈ 1 / mean(Λ_ij for j in connections)     │
│                                                                 │
│  ────────────────────────────────────────────────────────────  │
│                                                                 │
│  WHY THIS MATTERS:                                              │
│                                                                 │
│  • Can trigger collapse when entropy < threshold              │
│  • Can regularize to prevent overconfidence                   │
│  • Can encourage exploration when entropy too low             │
│  • Have explicit access to belief shape                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. Geometric Triggers

### 4.1 The Gershgorin Trigger

```
STABILITY-BASED GEOMETRIC TRIGGER:

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  FROM HOMEOSTAT (Section 4.2):                                  │
│                                                                 │
│  Small-Gain Projector enforces contraction via Gershgorin.    │
│                                                                 │
│  For each row i, compute margin:                               │
│                                                                 │
│     m_i = a_ii - Σ_{j≠i} |a_ij|                               │
│                                                                 │
│  If m_i > ε: row is stable (safe to use these connections)   │
│  If m_i < ε: row is unstable (need to scale down)            │
│                                                                 │
│  ════════════════════════════════════════════════════════════  │
│                                                                 │
│  FOR WORMHOLE:                                                  │
│                                                                 │
│  The "matrix" is the attention/connection structure.           │
│  Diagonal = self-connection strength                           │
│  Off-diagonal = wormhole connection strengths                 │
│                                                                 │
│  GEOMETRIC TRIGGER:                                             │
│                                                                 │
│     if gershgorin_margin(query_i) > stability_threshold:      │
│         allow_connections(query_i)                             │
│     else:                                                       │
│         scale_down_connections(query_i)                        │
│                                                                 │
│  This considers the WHOLE ROW, not just individual values.    │
│  It's about the RELATIONSHIP between connections.             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 The Precision-Relative Trigger

```
CURVATURE-BASED GEOMETRIC TRIGGER:

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  INSTEAD OF:                                                    │
│                                                                 │
│     if similarity > 0.92:  # Fixed threshold (energy)         │
│         connect()                                               │
│                                                                 │
│  DO:                                                            │
│                                                                 │
│     if precision[i,j] > mean(precision[i,:]) * ratio:         │
│         connect()  # This connection is sharper than average  │
│                                                                 │
│  ════════════════════════════════════════════════════════════  │
│                                                                 │
│  THE GEOMETRIC PRINCIPLE:                                       │
│                                                                 │
│  A connection should fire when it's RELATIVELY confident,     │
│  not when it crosses an absolute threshold.                    │
│                                                                 │
│  EXAMPLE:                                                       │
│                                                                 │
│  Case 1: All similarities around 0.5                           │
│     Best connection at 0.6 → SHOULD connect (it's best)       │
│     Energy trigger (> 0.92) → blocks it                       │
│     Geometric trigger → allows it (0.6 >> 0.5 mean)           │
│                                                                 │
│  Case 2: All similarities around 0.93                          │
│     Best connection at 0.94 → maybe connect                   │
│     Energy trigger → allows all (all > 0.92)                  │
│     Geometric trigger → only allows 0.94 (barely above mean) │
│                                                                 │
│  The geometric trigger ADAPTS to the local landscape.         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 4.3 The Downstream Benefit Trigger

```
NON-LOCAL GEOMETRIC TRIGGER:

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  FROM HOMEOSTAT (Section 2.3):                                  │
│                                                                 │
│  Gate receives gradient: ∂F/∂η_gate = -w · Δ_benefit          │
│                                                                 │
│  Even closed gates feel the pull of downstream benefit.       │
│                                                                 │
│  ════════════════════════════════════════════════════════════  │
│                                                                 │
│  FOR WORMHOLE:                                                  │
│                                                                 │
│  Compute benefit of each potential connection:                 │
│                                                                 │
│     benefit_ij = reduction_in_prediction_error_if_connected   │
│                                                                 │
│  GEOMETRIC TRIGGER:                                             │
│                                                                 │
│     if benefit_ij > benefit_threshold:                         │
│         strengthen_connection(i, j)                            │
│     # Even if similarity is below 0.92!                        │
│                                                                 │
│  ────────────────────────────────────────────────────────────  │
│                                                                 │
│  WHY THIS IS GEOMETRIC:                                         │
│                                                                 │
│  • It uses the RELATIONSHIP between connection and output     │
│  • It's non-local (considers downstream effects)              │
│  • It breaks the "zero gradient deadlock"                     │
│  • A low-similarity but high-benefit connection can open     │
│                                                                 │
│  This is like Equilibrium Propagation's "nudge" —             │
│  credit assignment without backprop through inactive paths.   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 4.4 The Null-Space Trigger (PSON)

```
EXPLORATION-BASED GEOMETRIC TRIGGER:

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  FROM HOMEOSTAT (Section 2.2):                                  │
│                                                                 │
│  PSON: Precision-Scaled Orthogonal Noise                       │
│                                                                 │
│  ξ_injection ∝ Λ⁻¹ · proj_{∇F⊥}(noise)                        │
│                                                                 │
│  Noise is:                                                      │
│  1. ORTHOGONAL to gradient (doesn't fight descent)            │
│  2. SCALED by inverse precision (explore uncertain dims)      │
│                                                                 │
│  ════════════════════════════════════════════════════════════  │
│                                                                 │
│  FOR WORMHOLE:                                                  │
│                                                                 │
│  Low-precision connections → more exploration noise           │
│  High-precision connections → stable, less noise              │
│                                                                 │
│  GEOMETRIC TRIGGER FOR EXPLORATION:                             │
│                                                                 │
│     if precision_ij < exploration_threshold:                   │
│         add_pson_noise(connection_ij)                          │
│         # Allow random activation to discover new paths       │
│                                                                 │
│  ────────────────────────────────────────────────────────────  │
│                                                                 │
│  THE GEOMETRY:                                                  │
│                                                                 │
│  The null-space of precision IS the set of uncertain beliefs. │
│  PSON explores THIS SPACE specifically.                        │
│  It's not random noise everywhere —                           │
│  it's TARGETED exploration of geometric uncertainty.          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 5. The Complete Design

### 5.1 Explicit Wormhole Architecture

```
WORMHOLE_EXPLICIT ARCHITECTURE:

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │                     PRECISION LAYER                       │ │
│  │                                                           │ │
│  │  For each (query_i, key_j) pair:                         │ │
│  │                                                           │ │
│  │  Λ_ij = compute_connection_precision(q_i, k_j)           │ │
│  │                                                           │ │
│  │  Sources:                                                 │ │
│  │  • Similarity curvature: ∂²sim/∂features²               │ │
│  │  • Attention curvature: ∂²softmax/∂scores²              │ │
│  │  • Coupling curvature: from energy function             │ │
│  │                                                           │ │
│  │  EXPLICIT: Λ is stored, inspectable, regularizable      │ │
│  │                                                           │ │
│  └───────────────────────────────┬───────────────────────────┘ │
│                                  │                              │
│                                  ▼                              │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │                   GEOMETRIC TRIGGERS                      │ │
│  │                                                           │ │
│  │  1. GERSHGORIN STABILITY:                                │ │
│  │     margin_i = Λ_ii - Σ_j |coupling_ij|                 │ │
│  │     if margin_i < ε: scale_down(row_i)                  │ │
│  │                                                           │ │
│  │  2. PRECISION-RELATIVE:                                   │ │
│  │     if Λ_ij > α * mean(Λ_i,:): allow(connection_ij)    │ │
│  │                                                           │ │
│  │  3. DOWNSTREAM BENEFIT:                                   │ │
│  │     benefit_ij = -∂F_output/∂w_ij                       │ │
│  │     if benefit_ij > β: strengthen(connection_ij)        │ │
│  │                                                           │ │
│  │  4. PSON EXPLORATION:                                     │ │
│  │     if Λ_ij < γ: explore_null_space(connection_ij)      │ │
│  │                                                           │ │
│  │  ALL triggers use RELATIONSHIPS, not fixed thresholds   │ │
│  │                                                           │ │
│  └───────────────────────────────┬───────────────────────────┘ │
│                                  │                              │
│                                  ▼                              │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │                 PRECISION-SCALED UPDATE                   │ │
│  │                                                           │ │
│  │  Δw_ij = -∂F/∂w_ij / (Λ_ij + ε)                         │ │
│  │                                                           │ │
│  │  • High precision: small update (already confident)     │ │
│  │  • Low precision: large update (need to learn more)     │ │
│  │                                                           │ │
│  │  This respects the geometric structure of belief.       │ │
│  │                                                           │ │
│  └───────────────────────────────────────────────────────────┘ │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 The Connection Decision Flow

```
GEOMETRIC DECISION FLOW FOR EACH CONNECTION:

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  FOR connection (query_i, key_j):                              │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ STEP 1: Compute precision Λ_ij                          │   │
│  │         (explicit curvature at this connection)         │   │
│  └────────────────────────┬────────────────────────────────┘   │
│                           │                                     │
│                           ▼                                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ STEP 2: Check Gershgorin stability                      │   │
│  │         margin_i = Λ_ii - Σ_j |off_diag_ij|            │   │
│  │         if margin_i < ε: SCALE DOWN                     │   │
│  └────────────────────────┬────────────────────────────────┘   │
│                           │                                     │
│                           ▼                                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ STEP 3: Check relative precision                        │   │
│  │         if Λ_ij > α * mean(Λ_i,:): ALLOW               │   │
│  │         (this connection is sharper than average)       │   │
│  └────────────────────────┬────────────────────────────────┘   │
│                           │                                     │
│                           ▼                                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ STEP 4: Check downstream benefit                        │   │
│  │         if benefit_ij > β: BOOST                        │   │
│  │         (even if precision is low, benefit is high)    │   │
│  └────────────────────────┬────────────────────────────────┘   │
│                           │                                     │
│                           ▼                                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ STEP 5: Apply PSON if uncertain                         │   │
│  │         if Λ_ij < γ: ADD exploration noise             │   │
│  │         (explore null-space of uncertain connections)   │   │
│  └────────────────────────┬────────────────────────────────┘   │
│                           │                                     │
│                           ▼                                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ STEP 6: Precision-scaled update                         │   │
│  │         Δw_ij = -grad_ij / (Λ_ij + ε)                  │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 6. Comparison: Implicit vs Explicit

### 6.1 Side-by-Side

```
┌───────────────────────────────────────────────────────────────────────────┐
│                                                                           │
│  ASPECT            IMPLICIT (Current)         EXPLICIT (Proposed)        │
│  ──────            ─────────────────          ───────────────────        │
│                                                                           │
│  Belief            Encoded in similarities    Stored as precision Λ     │
│  storage           Not explicitly stored      Explicitly maintained      │
│                                                                           │
│  Trigger           sim > 0.92 (scalar)        Multiple geometric:        │
│  type              Energy threshold           • Gershgorin margin        │
│                                               • Relative precision       │
│                                               • Downstream benefit       │
│                                               • Null-space exploration  │
│                                                                           │
│  Adaptivity        Fixed threshold            Adapts to local geometry  │
│                    Same for all contexts      Context-dependent          │
│                                                                           │
│  Entropy           Not computed               H ∝ -log|Λ| available     │
│                                                                           │
│  Regularization    Can't directly regulate    Can constrain Λ directly  │
│                                                                           │
│  Update rule       α · gradient               gradient / precision       │
│                    (ignores curvature)        (respects curvature)       │
│                                                                           │
│  Exploration       Random noise or none       PSON in null-space         │
│                                               (targeted exploration)     │
│                                                                           │
│  Non-local         None                       Downstream benefit         │
│  credit                                       (wormhole effect)          │
│                                                                           │
│  Memory            O(K) per query             O(K) + O(K) for Λ         │
│                    (similarities only)        (similarities + precision)│
│                                                                           │
│  Complexity        Simple                     More complex               │
│                                                                           │
└───────────────────────────────────────────────────────────────────────────┘
```

### 6.2 When Each Shines

```
IMPLICIT + ENERGY TRIGGER WORKS WELL WHEN:

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  • Task is simple, threshold is well-tuned                     │
│  • Distribution of similarities is consistent                  │
│  • Memory is very constrained                                  │
│  • Don't need adaptive behavior                                │
│  • Speed is critical, simplicity valued                        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

EXPLICIT + GEOMETRIC TRIGGER WORKS WELL WHEN:

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  • Task varies in difficulty                                   │
│  • Need adaptive thresholds                                    │
│  • Want to regularize belief explicitly                        │
│  • Need entropy-based collapse decisions                       │
│  • Want exploration in uncertain regions                       │
│  • Need non-local credit assignment                            │
│  • Have memory for precision tracking                          │
│  • Value interpretability of belief state                      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 7. Implementation Sketch

### 7.1 Core Classes

```python
class ExplicitWormholeAttention(nn.Module):
    """
    Wormhole attention with explicit precision and geometric triggers.
    
    Based on Neuro-Symbolic Homeostat framework.
    """
    
    def __init__(
        self,
        feature_dim: int,
        attn_dim: int,
        # Geometric trigger parameters
        gershgorin_margin: float = 0.1,
        precision_ratio: float = 1.5,  # α: how much above mean to trigger
        benefit_threshold: float = 0.0,  # β: min benefit to boost
        exploration_precision: float = 0.5,  # γ: below this, explore
        # PSON parameters
        pson_scale: float = 0.1,
        # Other
        max_connections: int = 16,
        device: torch.device = torch.device('cpu')
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.attn_dim = attn_dim
        
        # Geometric trigger thresholds
        self.gershgorin_margin = gershgorin_margin
        self.precision_ratio = precision_ratio
        self.benefit_threshold = benefit_threshold
        self.exploration_precision = exploration_precision
        self.pson_scale = pson_scale
        self.max_connections = max_connections
        
        # Projections
        self.W_q = nn.Linear(feature_dim, attn_dim, bias=False)
        self.W_k = nn.Linear(feature_dim, attn_dim, bias=False)
        self.W_v = nn.Linear(feature_dim, attn_dim, bias=False)
        self.W_o = nn.Linear(attn_dim, feature_dim, bias=False)
        
        # Precision estimation layer
        self.precision_estimator = nn.Linear(attn_dim * 2, 1)
        
        self.to(device)
```

### 7.2 Precision Computation

```python
def compute_connection_precision(
    self,
    Q: torch.Tensor,  # [N, D]
    K: torch.Tensor,  # [M, D]
    similarity: torch.Tensor  # [N, M]
) -> torch.Tensor:
    """
    Compute explicit precision for each connection.
    
    Returns:
        precision: [N, M] - curvature at each connection
    """
    # Method 1: Similarity curvature (approximation)
    # High similarity + low variance = high precision
    sim_mean = similarity.mean(dim=1, keepdim=True)
    sim_std = similarity.std(dim=1, keepdim=True) + 1e-8
    precision = (similarity - sim_mean) / sim_std  # z-score
    precision = torch.sigmoid(precision)  # [0, 1]
    
    # Method 2: Learn precision from Q-K features
    # QK_pairs = torch.cat([Q.unsqueeze(1).expand(-1, M, -1),
    #                       K.unsqueeze(0).expand(N, -1, -1)], dim=-1)
    # precision = self.precision_estimator(QK_pairs).squeeze(-1)
    # precision = F.softplus(precision)  # Ensure positive
    
    return precision
```

### 7.3 Geometric Triggers

```python
def apply_geometric_triggers(
    self,
    similarity: torch.Tensor,  # [N, M]
    precision: torch.Tensor,   # [N, M]
    downstream_benefit: Optional[torch.Tensor] = None  # [N, M]
) -> Tuple[torch.Tensor, Dict]:
    """
    Apply geometric trigger conditions.
    
    Returns:
        mask: [N, M] - which connections to allow
        stats: diagnostic information
    """
    N, M = similarity.shape
    
    # Initialize mask (all potentially allowed)
    mask = torch.ones(N, M, dtype=torch.bool, device=similarity.device)
    
    # 1. GERSHGORIN STABILITY CHECK
    # Treat precision as diagonal, similarity as off-diagonal
    diagonal = precision.diagonal() if N == M else precision.mean(dim=1)
    row_sums = similarity.abs().sum(dim=1)
    gershgorin_margins = diagonal - row_sums
    
    unstable_rows = gershgorin_margins < self.gershgorin_margin
    # Scale down unstable rows
    scale_factors = torch.where(
        unstable_rows,
        (diagonal - self.gershgorin_margin) / (row_sums + 1e-8),
        torch.ones_like(diagonal)
    ).clamp(0, 1)
    
    # 2. PRECISION-RELATIVE CHECK
    precision_mean = precision.mean(dim=1, keepdim=True)
    precision_threshold = precision_mean * self.precision_ratio
    relative_mask = precision > precision_threshold
    mask &= relative_mask
    
    # 3. DOWNSTREAM BENEFIT CHECK
    if downstream_benefit is not None:
        benefit_mask = downstream_benefit > self.benefit_threshold
        # Benefit can OVERRIDE low precision
        mask |= benefit_mask
    
    # 4. PSON EXPLORATION FLAG (not a mask, but stats)
    explore_mask = precision < self.exploration_precision
    
    stats = {
        'gershgorin_margins': gershgorin_margins,
        'unstable_rows': unstable_rows.sum().item(),
        'scale_factors': scale_factors,
        'precision_mean': precision_mean.mean().item(),
        'connections_allowed': mask.sum().item(),
        'explore_connections': explore_mask.sum().item(),
    }
    
    return mask, stats
```

### 7.4 PSON Exploration

```python
def apply_pson(
    self,
    connections: torch.Tensor,  # Current connection values
    precision: torch.Tensor,    # Precision at each connection
    gradient: torch.Tensor      # Gradient direction
) -> torch.Tensor:
    """
    Apply Precision-Scaled Orthogonal Noise.
    
    Explores null-space of high-precision connections.
    """
    # Generate random noise
    noise = torch.randn_like(connections)
    
    # Project orthogonal to gradient
    grad_norm = gradient / (gradient.norm(dim=-1, keepdim=True) + 1e-8)
    parallel = (noise * grad_norm).sum(dim=-1, keepdim=True) * grad_norm
    orthogonal = noise - parallel
    
    # Scale by inverse precision (explore uncertain more)
    inverse_precision = 1.0 / (precision + 1e-8)
    scaled_noise = orthogonal * inverse_precision * self.pson_scale
    
    return connections + scaled_noise
```

### 7.5 Forward Pass

```python
def forward(
    self,
    query_features: torch.Tensor,   # [H, W, D]
    history_buffer: torch.Tensor,   # [T, H, W, D]
    return_precision: bool = False
) -> Tuple[torch.Tensor, Dict]:
    """
    Forward with explicit precision and geometric triggers.
    """
    H, W, D = query_features.shape
    T = history_buffer.shape[0]
    
    # Flatten and project
    Q = self.W_q(query_features).reshape(H*W, -1)
    K = self.W_k(history_buffer).reshape(T*H*W, -1)
    V = self.W_v(history_buffer).reshape(T*H*W, -1)
    
    # Compute similarity (geometric)
    Q_norm = F.normalize(Q, dim=-1)
    K_norm = F.normalize(K, dim=-1)
    similarity = Q_norm @ K_norm.T  # [H*W, T*H*W]
    
    # Top-k selection
    topk_sim, topk_idx = torch.topk(similarity, self.max_connections, dim=1)
    
    # EXPLICIT: Compute precision
    precision = self.compute_connection_precision(Q, K[topk_idx], topk_sim)
    
    # GEOMETRIC TRIGGERS
    mask, trigger_stats = self.apply_geometric_triggers(
        topk_sim, precision, downstream_benefit=None
    )
    
    # Compute attention with masked connections
    scores = topk_sim / math.sqrt(self.attn_dim)
    scores = scores.masked_fill(~mask, float('-inf'))
    attn_weights = F.softmax(scores, dim=-1)
    
    # Gather values and aggregate
    V_selected = V[topk_idx]  # [H*W, K, D]
    output = torch.bmm(attn_weights.unsqueeze(1), V_selected).squeeze(1)
    output = self.W_o(output).reshape(H, W, D)
    
    # Compute entropy from explicit precision
    entropy = -(attn_weights * (attn_weights + 1e-8).log()).sum(dim=-1).mean()
    
    stats = {
        **trigger_stats,
        'entropy': entropy.item(),
        'precision_mean': precision.mean().item(),
        'precision_std': precision.std().item(),
    }
    
    if return_precision:
        stats['precision'] = precision
    
    return output, stats
```

---

## 8. When to Use Which

### 8.1 Decision Guide

```
CHOOSING BETWEEN IMPLICIT AND EXPLICIT WORMHOLE:

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  USE HYBRID (WORMHOLE_HYBRID) WHEN:                             │
│                                                                 │
│  ✓ Fast inference is critical                                  │
│  ✓ Memory is very constrained                                  │
│  ✓ Task is well-understood, threshold tuned                   │
│  ✓ Similarity distribution is stable                          │
│  ✓ Don't need adaptive behavior                                │
│  ✓ Simplicity preferred over flexibility                      │
│                                                                 │
│  ────────────────────────────────────────────────────────────  │
│                                                                 │
│  USE TRUE EXPLICIT (This document) WHEN:                        │
│                                                                 │
│  ✓ Task difficulty varies                                      │
│  ✓ Need adaptive thresholds                                    │
│  ✓ Want entropy-based collapse                                 │
│  ✓ Need exploration in uncertain regions                      │
│  ✓ Want to regularize belief                                   │
│  ✓ Need interpretable belief state                            │
│  ✓ Can afford extra memory for precision                      │
│  ✓ Building a "System-2" reasoning layer                      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 8.2 Hybrid Approach

```
COMBINING BOTH:

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  IMPLICIT for fast path:                                        │
│  • Use for most connections                                    │
│  • Energy threshold as fast filter                             │
│                                                                 │
│  EXPLICIT for uncertain cases:                                  │
│  • When similarity is near threshold                           │
│  • Compute precision only for ambiguous connections           │
│  • Apply geometric triggers selectively                        │
│                                                                 │
│  ────────────────────────────────────────────────────────────  │
│                                                                 │
│  if similarity > high_threshold:                               │
│      connect()  # Fast path, clearly good                      │
│  elif similarity < low_threshold:                              │
│      reject()   # Fast path, clearly bad                       │
│  else:                                                          │
│      # Ambiguous zone: use explicit precision                  │
│      precision = compute_precision(...)                        │
│      if geometric_trigger(precision):                          │
│          connect()                                              │
│                                                                 │
│  This gets speed AND adaptivity.                               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Summary

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│            W O R M H O L E   E X P L I C I T                   │
│                                                                 │
│  ═══════════════════════════════════════════════════════════   │
│                                                                 │
│  BELIEF: EXPLICIT (Precision/Curvature)                        │
│                                                                 │
│  • Maintain explicit precision Λ_ij for each connection       │
│  • Precision = inverse variance = confidence                  │
│  • Can compute entropy: H ∝ -log|Λ|                           │
│  • Can regularize, inspect, constrain                          │
│                                                                 │
│  ═══════════════════════════════════════════════════════════   │
│                                                                 │
│  TRIGGER: GEOMETRIC                                             │
│                                                                 │
│  1. Gershgorin stability (relationship-based)                  │
│  2. Precision-relative (above average = allow)                │
│  3. Downstream benefit (non-local credit)                     │
│  4. PSON exploration (null-space probing)                     │
│                                                                 │
│  All triggers use STRUCTURE, not scalar thresholds.           │
│                                                                 │
│  ═══════════════════════════════════════════════════════════   │
│                                                                 │
│  THE HOMEOSTAT PRINCIPLES:                                      │
│                                                                 │
│  • "Precision is stiffness"                                    │
│  • Update ∝ gradient / precision                              │
│  • Explore null-space, not everywhere                          │
│  • Stability via Gershgorin bounds                             │
│  • Non-local credit via downstream benefit                     │
│                                                                 │
│  ═══════════════════════════════════════════════════════════   │
│                                                                 │
│  WHEN TO USE:                                                   │
│                                                                 │
│  • Variable difficulty tasks                                   │
│  • Adaptive thresholds needed                                  │
│  • Entropy-based collapse                                       │
│  • Targeted exploration                                         │
│  • Interpretable belief states                                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

**References:**

- `The_Neuro_Symbolic_Homeostat_Paper_V1.md` - Source framework for precision/geometric approach
- `WORMHOLE_IMPLICIT.md` - True Implicit (energy + energy) - pure reflex
- `WORMHOLE_COMPETITION.md` - Competition (energy + geometry) - winner selection
- `WORMHOLE_HYBRID.md` - Hybrid (geometry + energy) - current wormhole
- `WORMHOLE_GEO_GEO.md` - GEO+GEO (geometry + geometry) - entropy-adaptive
- `POMDP_ATTENTION.md` - POMDP framework (belief as probability distribution)
- `KNOWLEDGE_AND_REACTIVITY.md` - Geometry vs energy distinction

---

## Related Concepts

```
THE FOUR WORMHOLE APPROACHES (COMPLETE QUADRANT):

┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│                         │    ENERGY TRIGGER      │   GEOMETRY TRIGGER      │
│                         │    (absolute)          │   (relative)            │
│  ───────────────────────┼────────────────────────┼─────────────────────────│
│                         │                        │                         │
│  ENERGY BELIEF          │  IMPLICIT              │  COMPETITION            │
│  (raw activation)       │  WORMHOLE_IMPLICIT.md  │  WORMHOLE_COMPETITION.md│
│                         │  (pure reflex)         │  (winner selection)     │
│  ───────────────────────┼────────────────────────┼─────────────────────────│
│                         │                        │                         │
│  GEOMETRY BELIEF        │  HYBRID                │  GEO+GEO                │
│  (normalized)           │  WORMHOLE_HYBRID.md    │  WORMHOLE_GEO_GEO.md    │
│                         │  (current wormhole)    │  (entropy-adaptive)     │
│  ───────────────────────┼────────────────────────┼─────────────────────────│
│                         │                        │                         │
│  EXPLICIT BELIEF        │  (Explicit+Energy)     │  TRUE EXPLICIT          │
│  (stored precision)     │                        │  (This document)        │
│                         │                        │  (Homeostat)            │
│                         │                        │                         │
└─────────────────────────────────────────────────────────────────────────────┘

SIMPLEST ───────────────────────────────────────────────────────── RICHEST
FASTEST ────────────────────────────────────────────────────────── ADAPTIVE

When to use:
• IMPLICIT: Maximum speed, binary decisions, pure reflex
• COMPETITION: Winner-take-all with magnitude influence
• HYBRID: Good balance, general attention, known threshold
• GEO+GEO: Entropy-adaptive, relative thresholds, best match focus
• HOMEOSTAT: Adaptive, interpretable, variable difficulty (meta-layer)
```

---

*This document describes an alternative wormhole implementation with explicit precision-based belief and geometric triggering, based on the Neuro-Symbolic Homeostat framework. See WORMHOLE_HYBRID.md for the current implementation with energy-based triggering.*

---

*Oscar Goldman, Shogu Research Group @ Datamutant.ai subsidiary of 温心重工業*