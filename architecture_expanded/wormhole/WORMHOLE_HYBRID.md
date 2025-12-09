# Wormhole Attention: Hybrid (Geometric Belief + Energy Trigger)

**⚠️ NOTE: This is a SIMPLIFIED architecture, not the theory-aligned version.**

Analysis of a **hybrid wormhole attention** implementation, which uses **geometric belief** (cosine similarity on hypersphere) with a **reactive energy trigger** (fixed similarity threshold = 0.92). 

**This is NOT the canonical theory-aligned implementation.** The theory-aligned version uses coherence gating with entropy-based thresholds. See:
- `architecture_base/attention/spectral_wormhole/SPECTRAL_WORMHOLE_ATTENTION.md` for the canonical implementation

This hybrid approach:
- ✓ Simpler to implement (no entropy computation)
- ✓ Faster (one scalar comparison vs entropy calculation)
- ✗ Not adaptive (fixed threshold)
- ✗ Ignores belief distribution structure

This document is retained for:
1. Comparison with theory-aligned approach
2. Stepping stone toward full implementation
3. Understanding trade-offs

---

## Table of Contents

1. [Overview](#1-overview)
2. [The Geometric Belief Structure](#2-the-geometric-belief-structure)
3. [The Energy Trigger](#3-the-energy-trigger)
4. [Step-by-Step Walkthrough](#4-step-by-step-walkthrough)
5. [What This Means for POMDP](#5-what-this-means-for-pomdp)
6. [Strengths and Limitations](#6-strengths-and-limitations)
7. [Code Reference](#7-code-reference)

---

## 1. Overview

### 1.1 The Current Design: HYBRID

```
WORMHOLE ATTENTION: GEOMETRIC BELIEF + ENERGY TRIGGER = HYBRID

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  BELIEF REPRESENTATION: GEOMETRIC (Implicit)                   │
│  ───────────────────────────────────────────                    │
│                                                                 │
│  • Features normalized onto unit hypersphere                   │
│  • Cosine similarity = angular distance on manifold            │
│  • Attention weights via softmax (probability distribution)   │
│  • Belief IS the geometry of positions on sphere              │
│                                                                 │
│  ════════════════════════════════════════════════════════════  │
│                                                                 │
│  TRIGGER MECHANISM: ENERGY (Reactive)                          │
│  ────────────────────────────────────                           │
│                                                                 │
│  • Fixed similarity threshold (τ = 0.92)                       │
│  • Binary decision: sim > threshold → connect                 │
│  • Scalar comparison, immediate response                       │
│  • No geometric reasoning in the gate                          │
│                                                                 │
│  ════════════════════════════════════════════════════════════  │
│                                                                 │
│  THIS IS HYBRID BECAUSE:                                        │
│                                                                 │
│  • Belief uses GEOMETRY (hypersphere, cosine, softmax)        │
│  • Trigger uses ENERGY (scalar threshold comparison)          │
│  • Two different paradigms combined                            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 The Core Insight

```
THE HYBRID COMBINATION:

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  GEOMETRIC BELIEF determines WHAT to connect to:               │
│  • Compute similarities (geometric - cosine on sphere)        │
│  • Find nearest neighbors (geometric - manifold distance)     │
│  • Weight by relevance (geometric - softmax distribution)     │
│                                                                 │
│  ENERGY TRIGGER determines WHETHER to connect:                 │
│  • Compare similarity to threshold (scalar magnitude)         │
│  • Binary gate: yes/no (reactive)                              │
│  • Fixed threshold, not adaptive (energy-based)                │
│                                                                 │
│  ════════════════════════════════════════════════════════════  │
│                                                                 │
│  "Geometry fills, energy gates"                                 │
│                                                                 │
│  The geometric belief finds the best candidates.               │
│  The energy trigger decides if they're good enough.            │
│                                                                 │
│  ════════════════════════════════════════════════════════════  │
│                                                                 │
│  THE FOUR QUADRANTS:                                            │
│                                                                 │
│                    │ ENERGY Trigger │ GEOMETRY Trigger         │
│  ──────────────────┼────────────────┼──────────────────        │
│  IMPLICIT Belief   │ True Implicit  │ (not current)            │
│  (raw activations) │ (pure reflex)  │                          │
│  ──────────────────┼────────────────┼──────────────────        │
│  GEOMETRIC Belief  │ HYBRID ←       │ (alternative)            │
│  (normalized)      │ (THIS DOC)     │                          │
│  ──────────────────┼────────────────┼──────────────────        │
│  EXPLICIT Belief   │ Explicit+Energy│ TRUE EXPLICIT            │
│  (stored precision)│                │ (Homeostat)              │
│                                                                 │
│  HYBRID = Geometry fills, Energy gates (current wormhole)     │
│  EXPLICIT = Geometry fills, Geometry gates (WORMHOLE_EXPLICIT)│
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. The Geometric Belief Structure

### 2.1 Where Belief Lives

```
BELIEF IS DISTRIBUTED ACROSS SEVERAL GEOMETRIC STRUCTURES:

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  1. NORMALIZED FEATURE VECTORS (Hypersphere)                   │
│  ───────────────────────────────────────────                    │
│                                                                 │
│     query_norm = F.normalize(query_flat, p=2, dim=1)           │
│     key_norm = F.normalize(key_flat, p=2, dim=1)               │
│                                                                 │
│     These place features on a unit hypersphere.                │
│     Position on sphere = belief about identity.                │
│     THIS IS GEOMETRIC - manifold structure.                    │
│                                                                 │
│  ────────────────────────────────────────────────────────────  │
│                                                                 │
│  2. SIMILARITY MATRIX (Angular Distance)                        │
│  ───────────────────────────────────────                        │
│                                                                 │
│     sim = query_norm @ key_norm.T                              │
│                                                                 │
│     Cosine similarity = dot product of unit vectors.           │
│     Values in [-1, 1], higher = closer on sphere.             │
│     THIS IS GEOMETRIC - distance on manifold.                  │
│                                                                 │
│  ────────────────────────────────────────────────────────────  │
│                                                                 │
│  3. TOP-K SELECTION (Nearest Neighbors)                         │
│  ──────────────────────────────────────                         │
│                                                                 │
│     topk_sim, topk_indices = torch.topk(sim, K)                │
│                                                                 │
│     Select K nearest neighbors on manifold.                    │
│     Sparse belief: Keep only K closest hypotheses.            │
│     THIS IS GEOMETRIC - neighborhood on manifold.              │
│                                                                 │
│  ────────────────────────────────────────────────────────────  │
│                                                                 │
│  4. ATTENTION WEIGHTS (Probability Distribution)               │
│  ───────────────────────────────────────────────                │
│                                                                 │
│     scores = Q @ K.T / sqrt(d)                                 │
│     attn_weights = softmax(scores / tau)                       │
│                                                                 │
│     Softmax converts distances to probabilities.               │
│     This IS b(s) - belief distribution over keys.             │
│     THIS IS GEOMETRIC - exponential of distances.              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 The Geometry of Belief

```
BELIEF AS MANIFOLD STRUCTURE:

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  The unit hypersphere is the belief manifold:                  │
│                                                                 │
│                    ╭────────────╮                              │
│                 ╱                  ╲                           │
│               ╱   ● k₁              ╲                          │
│              │       ╲               │                          │
│              │        ╲ sim=0.95     │                          │
│              │         ╲             │                          │
│              │          ● q          │   Unit sphere           │
│              │         ╱             │                          │
│              │        ╱ sim=0.87     │                          │
│              │       ╱               │                          │
│               ╲   ● k₂              ╱                           │
│                 ╲                  ╱                            │
│                    ╰────────────╯                              │
│                                                                 │
│  • q = query (current state)                                   │
│  • k₁, k₂ = keys (past states)                                │
│  • Angular distance = belief distance                          │
│  • Closer on sphere = higher belief in match                  │
│                                                                 │
│  ════════════════════════════════════════════════════════════  │
│                                                                 │
│  WHY THIS IS GEOMETRIC (not just "implicit"):                  │
│                                                                 │
│  • Features are NORMALIZED (placed on manifold)               │
│  • Similarity is ANGULAR (respects manifold structure)        │
│  • Softmax is EXPONENTIAL of distance (geodesic-aware)        │
│  • Structure IS belief, not just raw activations              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2.3 Properties of Geometric Belief

```
CHARACTERISTICS OF THE GEOMETRIC BELIEF:

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  ADVANTAGES:                                                    │
│                                                                 │
│  • Normalized: All features comparable (unit sphere)          │
│  • Geometric distance: Cosine is meaningful similarity        │
│  • Natural sparsity: Top-k prunes on manifold                 │
│  • Differentiable: Gradients flow through geometry            │
│  • Probabilistic: Softmax gives valid distribution            │
│                                                                 │
│  ────────────────────────────────────────────────────────────  │
│                                                                 │
│  WHAT MAKES IT IMPLICIT (not explicit):                        │
│                                                                 │
│  • No precision/curvature stored                              │
│  • No covariance structure maintained                          │
│  • Belief computed on-the-fly, not persisted                  │
│  • Can't directly regularize the distribution                 │
│                                                                 │
│  ────────────────────────────────────────────────────────────  │
│                                                                 │
│  WHAT WE CAN MEASURE:                                           │
│                                                                 │
│  • num_connections: How many passed threshold                  │
│  • mean_similarity: Average strength of connections           │
│  • sparsity: Fraction of possible connections made            │
│                                                                 │
│  These are ENERGY scalars extracted from geometry.            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. The Energy Trigger

### 3.1 How the Trigger Works

```
THE THRESHOLD GATE (ENERGY-BASED):

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  DEFINITION:                                                    │
│                                                                 │
│     mask = topk_sim > self.threshold                           │
│                                                                 │
│  WHERE:                                                         │
│  • topk_sim: Similarity scores (scalars in [0, 1])            │
│  • threshold: Fixed value (default 0.92)                       │
│  • mask: Boolean tensor (connect / don't connect)             │
│                                                                 │
│  ════════════════════════════════════════════════════════════  │
│                                                                 │
│  THIS IS ENERGY-BASED (REACTIVE) BECAUSE:                      │
│                                                                 │
│  1. SCALAR COMPARISON                                           │
│     • Compares magnitude to fixed threshold                    │
│     • No relationship to other candidates                     │
│     • Pure magnitude check                                     │
│                                                                 │
│  2. BINARY OUTPUT                                               │
│     • Yes/no decision per connection                           │
│     • No gradation, no soft gating                             │
│     • Immediate, automatic                                      │
│                                                                 │
│  3. FIXED THRESHOLD                                             │
│     • τ = 0.92 hardcoded                                       │
│     • Not adaptive to context                                  │
│     • Not based on belief distribution shape                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Why This Creates a Hybrid

```
THE MISMATCH THAT MAKES IT HYBRID:

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  BELIEF COMPUTATION:                                            │
│  ───────────────────                                            │
│                                                                 │
│  Uses GEOMETRY:                                                 │
│  • Normalized features (manifold)                              │
│  • Cosine similarity (angular distance)                        │
│  • Top-k (neighborhood selection)                              │
│  • Softmax (probability from distances)                        │
│                                                                 │
│  ════════════════════════════════════════════════════════════  │
│                                                                 │
│  TRIGGER DECISION:                                              │
│  ─────────────────                                              │
│                                                                 │
│  Uses ENERGY:                                                   │
│  • Fixed scalar threshold (0.92)                               │
│  • Magnitude comparison                                         │
│  • Independent per connection                                  │
│  • Ignores distribution structure                              │
│                                                                 │
│  ════════════════════════════════════════════════════════════  │
│                                                                 │
│  THE DISCONNECT:                                                │
│                                                                 │
│  The trigger DOESN'T USE the geometric information:            │
│  • Doesn't know if belief is concentrated or spread           │
│  • Doesn't compare candidates to each other                   │
│  • Doesn't adapt to distribution shape                        │
│  • Same threshold for clear and ambiguous cases               │
│                                                                 │
│  This is why a GEOMETRIC trigger (WORMHOLE_EXPLICIT.md)       │
│  would be more coherent with the geometric belief.            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 3.3 The Threshold Value

```
THE SIGNIFICANCE OF τ = 0.92:

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  COSINE SIMILARITY SCALE:                                       │
│                                                                 │
│  1.0 ────── Identical (same direction)                         │
│  0.92 ───── THRESHOLD (very similar)                           │
│  0.8 ────── Similar                                            │
│  0.5 ────── Moderately similar                                 │
│  0.0 ────── Orthogonal (uncorrelated)                          │
│  -1.0 ───── Opposite                                           │
│                                                                 │
│  ════════════════════════════════════════════════════════════  │
│                                                                 │
│  WHY 0.92?                                                      │
│                                                                 │
│  • High threshold = sparse connections                         │
│  • Only very similar patterns connect                          │
│  • Prevents noise from creating spurious links                 │
│  • Empirically tuned for this task                             │
│                                                                 │
│  ════════════════════════════════════════════════════════════  │
│                                                                 │
│  THE PROBLEM WITH FIXED THRESHOLD:                             │
│                                                                 │
│  • Not adaptive to difficulty                                  │
│  • Same threshold for easy and hard examples                  │
│  • Doesn't know if belief is concentrated or spread           │
│  • Can't adjust based on confidence                            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. Step-by-Step Walkthrough

### 4.1 Phase 1: Feature Preparation (GEOMETRIC)

```
STEP 1: NORMALIZE FEATURES ONTO HYPERSPHERE

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  INPUT:                                                         │
│  • query_features: [H, W, D] - current frame                   │
│  • history_buffer: [T, H, W, D] - past frames                  │
│                                                                 │
│  PROCESS:                                                       │
│  query_flat = query_features.reshape(H*W, D)                   │
│  key_flat = history_buffer.reshape(T*H*W, D)                   │
│                                                                 │
│  query_norm = F.normalize(query_flat, p=2, dim=1)  # GEOMETRIC │
│  key_norm = F.normalize(key_flat, p=2, dim=1)      # GEOMETRIC │
│                                                                 │
│  RESULT:                                                        │
│  • All vectors on unit hypersphere (manifold)                 │
│  • Dot product = cosine similarity = angular distance         │
│  • Geometry encodes belief structure                           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 Phase 2: Similarity Computation (GEOMETRIC)

```
STEP 2: COMPUTE DISTANCES ON MANIFOLD

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  CHUNKED COMPUTATION (memory efficient):                       │
│                                                                 │
│  for i in range(0, T*H*W, chunk_size):                        │
│      end = min(i + chunk_size, T*H*W)                         │
│      sim_chunk = query_norm @ key_norm[i:end].T  # GEOMETRIC  │
│                                                                 │
│      # Top-k per chunk (nearest neighbors on manifold)        │
│      vals, inds = torch.topk(sim_chunk, k_local, dim=1)       │
│                                                                 │
│  # Merge and get global top-k                                  │
│  merged_vals = torch.cat(all_topk_vals, dim=1)                │
│  topk_sim, topk_indices = torch.topk(merged_vals, K)          │
│                                                                 │
│  This is GEOMETRIC: finding nearest neighbors on hypersphere. │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 4.3 Phase 3: Energy Trigger (ENERGY)

```
STEP 3: APPLY THRESHOLD GATE

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  THE ENERGY-BASED DECISION:                                     │
│                                                                 │
│  mask = topk_sim > self.threshold  # threshold = 0.92         │
│                                                                 │
│  EXAMPLE:                                                       │
│                                                                 │
│  topk_sim = [0.95, 0.93, 0.91, 0.88, 0.85, ...]              │
│  threshold = 0.92                                               │
│                                                                 │
│  mask = [True, True, False, False, False, ...]                │
│               ↑                                                 │
│            ENERGY GATE (scalar comparison)                     │
│                                                                 │
│  ────────────────────────────────────────────────────────────  │
│                                                                 │
│  THIS IS WHERE THE HYBRID HAPPENS:                              │
│                                                                 │
│  • Previous steps used GEOMETRY                                │
│  • This step uses ENERGY (scalar threshold)                   │
│  • The approach switches here                                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 4.4 Phase 4: Attention Computation (GEOMETRIC)

```
STEP 4: COMPUTE ATTENTION OVER SURVIVORS

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  PROJECT AND COMPUTE SCORES:                                    │
│                                                                 │
│  Q = W_q(query_features)     # [H*W, attn_dim]                │
│  K = W_k(gathered_history)   # [H*W, K, attn_dim]             │
│  V = W_v(gathered_history)   # [H*W, K, attn_dim]             │
│                                                                 │
│  scores = Q @ K.T / sqrt(d)  # [H*W, K]                       │
│  scores = scores.masked_fill(~mask, -inf)  # Energy gate      │
│                                                                 │
│  attn_weights = softmax(scores / tau)  # GEOMETRIC            │
│                                                                 │
│  Back to GEOMETRY: softmax creates probability distribution   │
│  based on distances in attention space.                        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 4.5 Phase 5: Output

```
STEP 5: WEIGHTED AGGREGATION

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  AGGREGATE VALUES:                                              │
│                                                                 │
│  output = attn_weights @ V  # [H*W, attn_dim]                 │
│  output = W_o(output)       # [H*W, D]                        │
│  output = output.reshape(H, W, D)                              │
│                                                                 │
│  ────────────────────────────────────────────────────────────  │
│                                                                 │
│  EXTRACT STATISTICS (Energy scalars from geometry):           │
│                                                                 │
│  stats = {                                                      │
│      'num_connections': mask.sum(),                            │
│      'sparsity': num / total,                                  │
│      'mean_similarity': topk_sim[mask].mean(),                │
│  }                                                              │
│                                                                 │
│  These summarize the geometric belief as energy scalars.      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 5. What This Means for POMDP

### 5.1 POMDP Interpretation

```
HOW THE HYBRID MAPS TO POMDP:

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  POMDP COMPONENT          HYBRID WORMHOLE                      │
│  ────────────────          ───────────────                      │
│                                                                 │
│  State space S            All T*H*W key positions              │
│                                                                 │
│  Belief state b(s)        GEOMETRIC:                           │
│                           • Positions on hypersphere           │
│                           • Similarity values                  │
│                           • Attention weights (softmax)        │
│                                                                 │
│  Belief update            GEOMETRIC:                           │
│                           • top-k selection (nearest)          │
│                           • softmax (exp of distances)         │
│                                                                 │
│  Collapse trigger         ENERGY:                               │
│                           • threshold > 0.92                   │
│                           • scalar comparison                  │
│                                                                 │
│  Collapse execution       GEOMETRIC:                           │
│                           • weighted aggregation               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 The Mismatch

```
THE DISCONNECT BETWEEN BELIEF AND TRIGGER:

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  BELIEF IS GEOMETRIC:                                           │
│  • Computed from manifold positions                            │
│  • Represents relationships between hypotheses                 │
│  • Has structure (distribution shape, entropy, modes)         │
│                                                                 │
│  BUT TRIGGER IS ENERGY:                                         │
│  • Uses fixed scalar threshold                                 │
│  • Ignores belief distribution structure                       │
│  • Doesn't know if belief is concentrated or spread           │
│                                                                 │
│  ════════════════════════════════════════════════════════════  │
│                                                                 │
│  CONSEQUENCE:                                                   │
│                                                                 │
│  Example 1: Clear case                                          │
│  • Best match: sim = 0.99, Second: sim = 0.70                 │
│  • Belief is CONCENTRATED                                       │
│  • Should confidently connect to best                          │
│                                                                 │
│  Example 2: Ambiguous case                                      │
│  • Best match: sim = 0.93, Second: sim = 0.92                 │
│  • Belief is SPREAD                                             │
│  • Should be cautious about connecting                         │
│                                                                 │
│  BUT: Both treated the same (just check > 0.92)               │
│                                                                 │
│  A GEOMETRIC trigger would distinguish these cases.           │
│  See: WORMHOLE_EXPLICIT.md                                     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 6. Strengths and Limitations

### 6.1 Strengths

```
ADVANTAGES OF HYBRID (GEOMETRIC BELIEF + ENERGY TRIGGER):

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  1. BEST OF BOTH WORLDS (partially)                            │
│     • Rich geometric belief representation                     │
│     • Fast energy-based gating                                 │
│                                                                 │
│  2. MEMORY EFFICIENT                                            │
│     • No explicit precision/covariance stored                 │
│     • Belief computed on-the-fly                              │
│                                                                 │
│  3. COMPUTATIONALLY FAST                                        │
│     • Energy trigger is O(1) per connection                    │
│     • Simple scalar comparison                                  │
│                                                                 │
│  4. NATURALLY SPARSE                                            │
│     • Top-k + threshold = double sparsity                     │
│                                                                 │
│  5. DIFFERENTIABLE                                              │
│     • Gradients flow through geometric operations             │
│                                                                 │
│  6. SIMPLE TO IMPLEMENT                                         │
│     • Clear code path                                          │
│     • Easy to debug                                             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 6.2 Limitations

```
DISADVANTAGES OF HYBRID APPROACH:

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  1. APPROACH MISMATCH                                          │
│     • Belief is geometric, trigger is energy                  │
│     • Trigger doesn't use belief structure                    │
│     • Potential for incoherent decisions                       │
│                                                                 │
│  2. NO ADAPTIVE COLLAPSE                                        │
│     • Can't trigger based on belief certainty                  │
│     • Fixed threshold regardless of context                    │
│                                                                 │
│  3. BELIEF-BLIND GATING                                         │
│     • Trigger ignores distribution shape                       │
│     • Can't distinguish clear vs ambiguous cases              │
│                                                                 │
│  4. HARD TO REGULARIZE                                          │
│     • Can't directly constrain belief entropy                  │
│     • Limited control over belief dynamics                     │
│                                                                 │
│  5. LIMITED INTROSPECTION                                       │
│     • No explicit precision/curvature                          │
│     • Statistics are energy scalars                            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 7. Code Reference

### 7.1 Key Code Sections

```python
# FILE: spectral_attention/core/wormhole_attention.py

# GEOMETRIC BELIEF: Normalize onto hypersphere
query_norm = F.normalize(query_flat, p=2, dim=1)  # Line 243
key_norm = F.normalize(key_flat, p=2, dim=1)      # Line 244

# GEOMETRIC BELIEF: Compute angular distance
sim = torch.matmul(query_norm, key_norm.T)        # Line 264

# GEOMETRIC BELIEF: Top-k nearest neighbors
topk_sim, topk_indices = torch.topk(merged_vals, K_neighbors, dim=1)  # Line 276

# ENERGY TRIGGER: Fixed threshold gate
mask = topk_sim > self.threshold  # Line 296 (threshold = 0.92)

# GEOMETRIC BELIEF: Softmax attention weights
attn_weights = F.softmax(scores / tau, dim=-1)  # Line 316

# GEOMETRIC BELIEF: Weighted aggregation
output = torch.bmm(attn_weights.unsqueeze(1), V_selected)  # Line 325
```

### 7.2 The Threshold Parameter

```python
# Constructor default
threshold: float = 0.92  # Line 28

# Usage in forward (ENERGY TRIGGER)
mask = topk_sim > self.threshold  # Line 296
```

---

## Summary

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│              W O R M H O L E   H Y B R I D                     │
│                                                                 │
│  ═══════════════════════════════════════════════════════════   │
│                                                                 │
│  BELIEF: GEOMETRIC                                              │
│                                                                 │
│  • Normalized features on hypersphere                          │
│  • Cosine similarity = angular distance                        │
│  • Top-k = nearest neighbors on manifold                       │
│  • Softmax = probability from distances                        │
│                                                                 │
│  ═══════════════════════════════════════════════════════════   │
│                                                                 │
│  TRIGGER: ENERGY                                                │
│                                                                 │
│  • Fixed threshold τ = 0.92                                    │
│  • Binary scalar comparison                                    │
│  • Ignores belief distribution structure                       │
│  • Fast, simple, not adaptive                                  │
│                                                                 │
│  ═══════════════════════════════════════════════════════════   │
│                                                                 │
│  WHY IT'S HYBRID:                                               │
│                                                                 │
│  "Geometry fills, energy gates"                                 │
│                                                                 │
│  Geometric belief finds candidates.                            │
│  Energy trigger accepts/rejects them.                          │
│  Two paradigms combined.                                        │
│                                                                 │
│  ═══════════════════════════════════════════════════════════   │
│                                                                 │
│  ALTERNATIVE: WORMHOLE_EXPLICIT.md                              │
│                                                                 │
│  Fully geometric: belief AND trigger both geometric.          │
│  Uses precision-based Gershgorin/relative triggers.           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

**References:**

- `spectral_attention/core/wormhole_attention.py` - Implementation
- `WORMHOLE_IMPLICIT.md` - True Implicit (energy + energy) - pure reflex
- `WORMHOLE_COMPETITION.md` - Competition (energy + geometry) - winner selection
- `WORMHOLE_GEO_GEO.md` - GEO+GEO (geometry + geometry) - entropy-adaptive
- `WH_HOMEOSTAT_STORED_EXPLICIT_PRECISION.md` - Homeostat (meta-layer with precision)
- `POMDP_ATTENTION.md` - POMDP framework (belief as probability distribution)
- `KNOWLEDGE_AND_REACTIVITY.md` - Geometry vs energy distinction
- `The_Neuro_Symbolic_Homeostat_Paper_V1.md` - Source for explicit/geometric approach

---

## Related Concepts

```
DOCUMENT RELATIONSHIPS:

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  POMDP_ATTENTION.md                                            │
│  ─────────────────                                              │
│  Establishes: Attention = Bayesian belief update               │
│  Defines: Implicit vs Explicit, Geometry vs Energy             │
│                                                                 │
│       ↓                                                         │
│  ┌─────────────────────┬─────────────────────┐                 │
│  │                     │                     │                 │
│  ▼                     ▼                     │                 │
│  IMPLICIT    COMPETITION   HYBRID       GEO+GEO      HOMEOSTAT │
│  (Simplest)  (Winner)      (This doc)   (Adaptive)   (Meta)    │
│  Energy+E    Energy+G      Geo+E        Geo+G        Explicit  │
│  "reflex"    "strongest"   "threshold"  "best match" "precision"│
│                                              │                 │
│  ────────────────────────────────────────────┘                 │
│                                                                 │
│  KNOWLEDGE_AND_REACTIVITY.md                                   │
│  ───────────────────────────                                    │
│  Defines: Geometry = relational, structure                     │
│           Energy = scalar, magnitude                           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

*This document describes the current wormhole implementation as a HYBRID approach: geometric belief computation with energy-based triggering. See WORMHOLE_EXPLICIT.md for a fully geometric alternative based on the Homeostat framework.*

---

*Oscar Goldman, Shogu Research Group @ Datamutant.ai subsidiary of 温心重工業*