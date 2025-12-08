# Wormhole Attention: GEO+GEO (Geometry Belief + Geometry Trigger)

The fully geometric quadrant within the base 2×2: normalized similarity (geometry belief) with relative/adaptive triggering (geometry trigger). This is entropy-adaptive attention that asks "Is this the best match?" rather than "Is this good enough?"

---

## Table of Contents

1. [Overview](#1-overview)
2. [The Quadrant Position](#2-the-quadrant-position)
3. [Geometry Belief](#3-geometry-belief)
4. [Geometry Trigger](#4-geometry-trigger)
5. [Entropy-Adaptive Behavior](#5-entropy-adaptive-behavior)
6. [Comparison to Other Quadrants](#6-comparison-to-other-quadrants)
7. [Implementation Sketch](#7-implementation-sketch)
8. [Relationship to Homeostat](#8-relationship-to-homeostat)

---

## 1. Overview

### 1.1 The GEO+GEO Design

```
GEO+GEO: GEOMETRY BELIEF + GEOMETRY TRIGGER

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  BELIEF REPRESENTATION: GEOMETRY (Normalized Similarity)       │
│  ────────────────────────────────────────────────────           │
│                                                                 │
│  • Features normalized onto unit hypersphere                   │
│  • Cosine similarity = angular distance on manifold            │
│  • Bounded [-1, 1], direction matters, magnitude removed      │
│  • Softmax creates probability distribution                    │
│                                                                 │
│  ════════════════════════════════════════════════════════════  │
│                                                                 │
│  TRIGGER MECHANISM: GEOMETRY (Relative/Adaptive)               │
│  ───────────────────────────────────────────────                │
│                                                                 │
│  • Compare to distribution, not fixed threshold               │
│  • "Is this the BEST match?" not "Is this good enough?"       │
│  • Entropy-based decisions                                      │
│  • Adaptive to local similarity landscape                      │
│                                                                 │
│  ════════════════════════════════════════════════════════════  │
│                                                                 │
│  FULLY GEOMETRIC:                                               │
│                                                                 │
│  Both belief and trigger respect manifold structure.          │
│  No absolute thresholds, no raw magnitudes.                   │
│  Everything is relational.                                     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 Core Insight

```
THE KEY DIFFERENCE FROM HYBRID:

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  HYBRID (Geometry + Energy):                                    │
│  ────────────────────────────                                   │
│                                                                 │
│     if cosine_similarity > 0.92:  # Absolute threshold        │
│         connect()                                               │
│                                                                 │
│  Problem: Fixed threshold ignores context.                     │
│  • All similarities at 0.5? Best at 0.6? → Blocked            │
│  • All similarities at 0.95? → All allowed (no selection)    │
│                                                                 │
│  ════════════════════════════════════════════════════════════  │
│                                                                 │
│  GEO+GEO (Geometry + Geometry):                                │
│  ──────────────────────────────                                 │
│                                                                 │
│     if similarity > mean(similarities) + σ:  # Relative       │
│         connect()  # Above local average                       │
│                                                                 │
│  OR:                                                            │
│                                                                 │
│     if entropy(attention_weights) < threshold:  # Concentrated│
│         connect_to_winner()  # Belief has collapsed           │
│                                                                 │
│  Adapts to the local similarity landscape.                     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. The Quadrant Position

### 2.1 The Complete 2×2 Base Quadrant

```
THE BASE QUADRANT (before Homeostat meta-layer):

┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│                         │    ENERGY TRIGGER      │   GEOMETRY TRIGGER      │
│                         │    (absolute)          │   (relative)            │
│  ───────────────────────┼────────────────────────┼─────────────────────────│
│                         │                        │                         │
│  ENERGY BELIEF          │  IMPLICIT              │  COMPETITION            │
│  (raw activation)       │  WORMHOLE_IMPLICIT.md  │  WORMHOLE_COMPETITION.md│
│                         │  "good enough?"        │  "strongest?"           │
│  ───────────────────────┼────────────────────────┼─────────────────────────│
│                         │                        │                         │
│  GEOMETRY BELIEF        │  HYBRID                │  GEO+GEO ←              │
│  (normalized)           │  WORMHOLE_HYBRID.md    │  (This document)        │
│                         │  "similar enough?"     │  "best match?"          │
│                         │                        │                         │
└─────────────────────────────────────────────────────────────────────────────┘

                    HOMEOSTAT sits ABOVE as meta-controller
                    (explicit precision, adapts thresholds)
```

### 2.2 What Makes GEO+GEO Distinct

```
THE FOUR BASE MECHANISMS:

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  IMPLICIT: "Is activation > threshold?"                        │
│  ──────────────────────────────────────                         │
│  • Raw magnitude, absolute threshold                           │
│  • Pure reflex, fastest                                        │
│  • No distribution awareness                                   │
│                                                                 │
│  COMPETITION: "Is this the strongest activation?"              │
│  ─────────────────────────────────────────────────              │
│  • Raw magnitude, relative selection                           │
│  • Competition, spectral peaks                                 │
│  • Winner-take-all without normalization                       │
│                                                                 │
│  HYBRID: "Is similarity > 0.92?"                               │
│  ───────────────────────────────                                │
│  • Normalized (direction), absolute threshold                  │
│  • Current wormhole, good balance                             │
│  • Fixed threshold regardless of context                       │
│                                                                 │
│  GEO+GEO: "Is this the best match?" ← THIS DOCUMENT           │
│  ────────────────────────────────────                           │
│  • Normalized (direction), relative selection                  │
│  • Entropy-adaptive, context-aware                             │
│  • Fully geometric, no absolute thresholds                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Geometry Belief

### 3.1 Normalized Similarity

```
GEOMETRY BELIEF = COSINE SIMILARITY ON HYPERSPHERE:

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  COMPUTATION:                                                   │
│                                                                 │
│  query_norm = F.normalize(query, dim=-1)  # Unit sphere        │
│  key_norm = F.normalize(key, dim=-1)      # Unit sphere        │
│  similarity = query_norm @ key_norm.T     # Cosine ∈ [-1, 1]  │
│                                                                 │
│  ════════════════════════════════════════════════════════════  │
│                                                                 │
│  WHAT NORMALIZATION DOES:                                       │
│                                                                 │
│  • Removes magnitude: ||v|| = 1 for all vectors               │
│  • Preserves direction: angle between vectors                 │
│  • Bounds similarity: always in [-1, 1]                       │
│  • Makes comparison fair: scale-invariant                      │
│                                                                 │
│  ════════════════════════════════════════════════════════════  │
│                                                                 │
│  THE MANIFOLD STRUCTURE:                                        │
│                                                                 │
│                    ╭────────────╮                              │
│                 ╱                  ╲                           │
│               ╱   ● k₁              ╲                          │
│              │       ╲ θ            │                          │
│              │        ╲             │   Unit hypersphere       │
│              │         ● q          │                          │
│              │        ╱             │                          │
│              │       ╱              │                          │
│               ╲   ● k₂              ╱                           │
│                 ╲                  ╱                            │
│                    ╰────────────╯                              │
│                                                                 │
│  cos(θ) = q · k = similarity                                  │
│  Angular distance IS the belief metric.                        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Softmax Distribution

```
FROM SIMILARITIES TO PROBABILITIES:

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  ATTENTION WEIGHTS:                                             │
│                                                                 │
│  scores = similarity / sqrt(d)                                 │
│  weights = softmax(scores / τ)  # Temperature τ               │
│                                                                 │
│  This creates a probability distribution over keys.           │
│  weights[j] = P(attend to key j | query)                      │
│                                                                 │
│  ════════════════════════════════════════════════════════════  │
│                                                                 │
│  TEMPERATURE CONTROLS SHARPNESS:                                │
│                                                                 │
│  τ → 0: Approaches argmax (winner-take-all)                   │
│  τ = 1: Standard softmax                                       │
│  τ → ∞: Approaches uniform (all equal)                        │
│                                                                 │
│  ════════════════════════════════════════════════════════════  │
│                                                                 │
│  THIS IS GEOMETRIC:                                             │
│                                                                 │
│  • Softmax is exponential of distances on manifold            │
│  • The distribution respects angular geometry                 │
│  • Closer on sphere = higher probability                      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. Geometry Trigger

### 4.1 Relative Threshold

```
INSTEAD OF ABSOLUTE, USE RELATIVE:

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  ABSOLUTE (HYBRID):                                             │
│  ──────────────────                                             │
│                                                                 │
│     mask = similarity > 0.92  # Fixed threshold               │
│                                                                 │
│  Problems:                                                      │
│  • If all similarities are 0.5, nothing passes               │
│  • If all similarities are 0.95, everything passes           │
│  • No adaptation to difficulty                                 │
│                                                                 │
│  ════════════════════════════════════════════════════════════  │
│                                                                 │
│  RELATIVE (GEO+GEO):                                           │
│  ───────────────────                                            │
│                                                                 │
│  Option 1: Above mean                                          │
│     mean_sim = similarity.mean(dim=-1, keepdim=True)          │
│     mask = similarity > mean_sim                               │
│                                                                 │
│  Option 2: Above mean + σ                                      │
│     std_sim = similarity.std(dim=-1, keepdim=True)            │
│     mask = similarity > (mean_sim + std_sim)                  │
│                                                                 │
│  Option 3: Top percentile                                      │
│     threshold = torch.quantile(similarity, 0.9, dim=-1)       │
│     mask = similarity >= threshold                             │
│                                                                 │
│  All adapt to the local similarity distribution.              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 Entropy-Based Triggering

```
USE ENTROPY TO DETECT BELIEF CONCENTRATION:

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  ENTROPY OF ATTENTION WEIGHTS:                                  │
│                                                                 │
│  H = -Σ_j w_j log(w_j)                                        │
│                                                                 │
│  High entropy: Spread belief (uncertain)                       │
│  Low entropy: Concentrated belief (confident)                  │
│                                                                 │
│  ════════════════════════════════════════════════════════════  │
│                                                                 │
│  ENTROPY-BASED TRIGGER:                                         │
│                                                                 │
│  entropy = -(weights * weights.log()).sum(dim=-1)             │
│  max_entropy = log(num_keys)  # Uniform distribution          │
│  normalized_entropy = entropy / max_entropy  # [0, 1]         │
│                                                                 │
│  if normalized_entropy < 0.5:  # Belief concentrated          │
│      # Connect to high-weight keys                             │
│      mask = weights > (1.0 / num_keys)  # Above uniform       │
│  else:                                                          │
│      # Belief spread, maybe explore or wait                   │
│      mask = weights > weights.max() * 0.9  # Near-winners    │
│                                                                 │
│  ════════════════════════════════════════════════════════════  │
│                                                                 │
│  THIS IS GEOMETRIC:                                             │
│                                                                 │
│  Entropy measures the SHAPE of the distribution.              │
│  It's a property of the geometry, not a scalar threshold.    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 4.3 Distribution-Aware Selection

```
MULTIPLE GEOMETRIC TRIGGER STRATEGIES:

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  1. RELATIVE THRESHOLD (Above local average)                   │
│  ────────────────────────────────────────────                   │
│                                                                 │
│     mask = similarity > similarity.mean(dim=-1, keepdim=True) │
│                                                                 │
│     Adapts to each query's similarity distribution.           │
│     Always selects ~half the keys.                             │
│                                                                 │
│  ────────────────────────────────────────────────────────────  │
│                                                                 │
│  2. GAP-BASED (Distance to runner-up)                          │
│  ─────────────────────────────────────                          │
│                                                                 │
│     sorted_sim = similarity.sort(descending=True)             │
│     gap = sorted_sim[0] - sorted_sim[1]  # Winner margin      │
│     if gap > gap_threshold:                                    │
│         mask = similarity == sorted_sim[0]  # Clear winner    │
│                                                                 │
│     Triggers when there's a clear best match.                 │
│                                                                 │
│  ────────────────────────────────────────────────────────────  │
│                                                                 │
│  3. ENTROPY-ADAPTIVE (Concentration-aware)                     │
│  ──────────────────────────────────────────                     │
│                                                                 │
│     weights = softmax(similarity / τ)                         │
│     H = entropy(weights)                                       │
│     if H < low_entropy_threshold:                             │
│         mask = weights > median(weights)                      │
│     else:                                                       │
│         mask = torch.ones_like(weights, dtype=bool)  # Keep all│
│                                                                 │
│     Only selects when belief is concentrated.                 │
│                                                                 │
│  ────────────────────────────────────────────────────────────  │
│                                                                 │
│  4. SPECTRAL GAP (Eigenvalue-style)                            │
│  ───────────────────────────────────                            │
│                                                                 │
│     Treat similarity as adjacency matrix.                      │
│     Large spectral gap → clear cluster structure.             │
│     Use gap to decide connectivity.                           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 5. Entropy-Adaptive Behavior

### 5.1 The Connection to Collapse

```
ENTROPY AND BELIEF COLLAPSE:

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  FROM POMDP/COLLAPSE DOCUMENTS:                                │
│                                                                 │
│  Collapse = Belief concentrating on single hypothesis         │
│           = Entropy decreasing                                  │
│           = Uncertainty resolving                              │
│                                                                 │
│  ════════════════════════════════════════════════════════════  │
│                                                                 │
│  GEO+GEO USES ENTROPY FOR TRIGGERING:                          │
│                                                                 │
│  High entropy (spread):                                         │
│  • Multiple hypotheses plausible                               │
│  • Don't commit yet                                            │
│  • Keep options open OR explore                               │
│                                                                 │
│  Low entropy (concentrated):                                    │
│  • Single hypothesis dominates                                 │
│  • Safe to commit                                               │
│  • Connect to winner                                           │
│                                                                 │
│  ════════════════════════════════════════════════════════════  │
│                                                                 │
│  THIS IS COLLAPSE-AWARE:                                        │
│                                                                 │
│  The trigger KNOWS the belief state shape.                     │
│  It waits for natural collapse, or forces it adaptively.     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 Adaptive Temperature

```
TEMPERATURE AS GEOMETRIC CONTROL:

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  Instead of fixed τ, adapt based on entropy:                  │
│                                                                 │
│  current_H = entropy(softmax(scores / τ))                     │
│  target_H = some_target_entropy  # Design choice              │
│                                                                 │
│  if current_H > target_H:                                      │
│      τ = τ * 0.9  # Cool down, sharpen distribution           │
│  else:                                                          │
│      τ = τ * 1.1  # Heat up, spread distribution              │
│                                                                 │
│  ════════════════════════════════════════════════════════════  │
│                                                                 │
│  EFFECT:                                                        │
│                                                                 │
│  • Maintains desired belief concentration                      │
│  • Automatically sharpens when uncertain                      │
│  • Automatically spreads when overconfident                   │
│  • Homeostatic entropy regulation                              │
│                                                                 │
│  ════════════════════════════════════════════════════════════  │
│                                                                 │
│  THIS IS GEOMETRIC:                                             │
│                                                                 │
│  Temperature controls the curvature of softmax.               │
│  Adapting τ = adapting the geometry of belief.               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 6. Comparison to Other Quadrants

### 6.1 Side-by-Side

```
┌───────────────────────────────────────────────────────────────────────────────┐
│                                                                               │
│  ASPECT          IMPLICIT     COMPETITION   HYBRID        GEO+GEO            │
│  ──────          ────────     ───────────   ──────        ───────            │
│                                                                               │
│  Belief          Raw act      Raw act       Cosine        Cosine             │
│                  (unbounded)  (unbounded)   (bounded)     (bounded)          │
│                                                                               │
│  Trigger         act > τ      argmax        sim > 0.92    sim > mean         │
│                  (absolute)   (relative)    (absolute)    (relative)         │
│                                                                               │
│  Threshold       Fixed        None (best)   Fixed         Adaptive           │
│                                                                               │
│  Entropy-aware   No           No            No            Yes                │
│                                                                               │
│  Distribution-   No           Yes           No            Yes                │
│  aware                        (winner)                    (shape)            │
│                                                                               │
│  Normalization   No           No            Yes           Yes                │
│                                                                               │
│  Magnitude       Matters      Influences    Ignored       Ignored            │
│                                                                               │
│  Adaptivity      None         To scale      None          Full               │
│                                                                               │
└───────────────────────────────────────────────────────────────────────────────┘
```

### 6.2 When to Use GEO+GEO

```
GEO+GEO IS APPROPRIATE FOR:

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  1. VARIABLE DIFFICULTY TASKS                                   │
│     ─────────────────────────                                   │
│     • Sometimes queries have clear matches                     │
│     • Sometimes matches are ambiguous                          │
│     • Need adaptive thresholds                                 │
│                                                                 │
│  2. COLLAPSE-AWARE ATTENTION                                    │
│     ─────────────────────────                                   │
│     • Want to detect when belief has concentrated             │
│     • Want to trigger on entropy, not magnitude               │
│                                                                 │
│  3. RELATIVE SELECTION                                          │
│     ──────────────────                                          │
│     • "Find best match" regardless of absolute similarity    │
│     • Even if best is only 0.5, still select it              │
│                                                                 │
│  4. SCALE-INVARIANT ATTENTION                                   │
│     ─────────────────────────                                   │
│     • Input magnitude shouldn't affect selection              │
│     • Direction (pattern) is what matters                     │
│                                                                 │
│  5. SOFT GATING                                                 │
│     ───────────                                                 │
│     • Want gradual inclusion, not binary                      │
│     • Temperature-controlled sharpness                         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 7. Implementation Sketch

### 7.1 Core Implementation

```python
class GeoGeoWormhole(nn.Module):
    """
    GEO+GEO Wormhole: Geometry belief + Geometry trigger.
    
    Normalized similarity with entropy-adaptive relative triggering.
    """
    
    def __init__(
        self,
        feature_dim: int,
        attn_dim: int = 64,
        max_connections: int = 16,
        # Geometric trigger parameters
        relative_threshold: float = 1.0,  # Multiplier on mean
        entropy_threshold: float = 0.5,   # Normalized entropy trigger
        temperature: float = 1.0,
        adaptive_temperature: bool = True,
        target_entropy: float = 0.3,  # Target normalized entropy
        device: torch.device = torch.device('cpu')
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.attn_dim = attn_dim
        self.max_connections = max_connections
        
        self.relative_threshold = relative_threshold
        self.entropy_threshold = entropy_threshold
        self.temperature = temperature
        self.adaptive_temperature = adaptive_temperature
        self.target_entropy = target_entropy
        
        self.W_q = nn.Linear(feature_dim, attn_dim, bias=False)
        self.W_k = nn.Linear(feature_dim, attn_dim, bias=False)
        self.W_v = nn.Linear(feature_dim, attn_dim, bias=False)
        self.W_o = nn.Linear(attn_dim, feature_dim, bias=False)
        
        self.to(device)
    
    def compute_entropy(self, weights: torch.Tensor) -> torch.Tensor:
        """Compute normalized entropy of attention weights."""
        # weights: [N, K]
        eps = 1e-8
        H = -(weights * (weights + eps).log()).sum(dim=-1)  # [N]
        max_H = math.log(weights.shape[-1])  # Max entropy = uniform
        return H / max_H  # Normalized to [0, 1]
    
    def geometry_trigger(
        self,
        similarity: torch.Tensor,  # [N, K]
        weights: torch.Tensor,     # [N, K]
    ) -> torch.Tensor:
        """
        Apply geometry trigger based on distribution shape.
        """
        N, K = similarity.shape
        
        # Compute entropy
        entropy = self.compute_entropy(weights)  # [N]
        
        # Relative threshold: above mean * factor
        mean_sim = similarity.mean(dim=-1, keepdim=True)  # [N, 1]
        relative_mask = similarity > (mean_sim * self.relative_threshold)
        
        # Entropy-based modulation
        # Low entropy → trust the distribution, select high-weight
        # High entropy → be cautious, maybe select more
        concentrated = entropy < self.entropy_threshold  # [N]
        
        # For concentrated beliefs, use weight-based selection
        weight_mask = weights > (1.0 / K)  # Above uniform
        
        # Combine: relative AND (concentrated → weight_based)
        mask = relative_mask & (
            concentrated.unsqueeze(-1) | weight_mask
        )
        
        return mask
    
    def forward(
        self,
        query_features: torch.Tensor,   # [H, W, D]
        history_buffer: torch.Tensor,   # [T, H, W, D]
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Forward with geometry belief + geometry trigger.
        """
        H, W, D = query_features.shape
        T = history_buffer.shape[0]
        
        # Project
        Q = self.W_q(query_features).reshape(H*W, -1)
        K = self.W_k(history_buffer).reshape(T*H*W, -1)
        V = self.W_v(history_buffer).reshape(T*H*W, -1)
        
        # GEOMETRY BELIEF: Normalize onto hypersphere
        Q_norm = F.normalize(Q, dim=-1)
        K_norm = F.normalize(K, dim=-1)
        
        # Cosine similarity
        similarity = Q_norm @ K_norm.T  # [HW, THW]
        
        # Top-k pre-selection
        topk_sim, topk_idx = torch.topk(
            similarity, self.max_connections, dim=-1
        )  # [HW, K]
        
        # Compute attention weights
        scores = topk_sim / math.sqrt(self.attn_dim)
        weights = F.softmax(scores / self.temperature, dim=-1)  # [HW, K]
        
        # GEOMETRY TRIGGER
        mask = self.geometry_trigger(topk_sim, weights)
        
        # Apply mask
        masked_scores = scores.masked_fill(~mask, float('-inf'))
        masked_weights = F.softmax(masked_scores / self.temperature, dim=-1)
        
        # Gather and aggregate
        V_selected = V[topk_idx]  # [HW, K, D]
        output = torch.bmm(masked_weights.unsqueeze(1), V_selected).squeeze(1)
        output = self.W_o(output).reshape(H, W, D)
        
        # Compute stats
        entropy = self.compute_entropy(weights)
        
        # Adaptive temperature
        if self.adaptive_temperature:
            mean_entropy = entropy.mean().item()
            if mean_entropy > self.target_entropy:
                self.temperature *= 0.99  # Cool down
            else:
                self.temperature *= 1.01  # Heat up
            self.temperature = max(0.1, min(10.0, self.temperature))
        
        stats = {
            'entropy_mean': entropy.mean().item(),
            'entropy_std': entropy.std().item(),
            'temperature': self.temperature,
            'connections_per_query': mask.sum(dim=-1).float().mean().item(),
            'similarity_mean': topk_sim.mean().item(),
        }
        
        return output, stats
```

---

## 8. Relationship to Homeostat

### 8.1 GEO+GEO vs Homeostat

```
HOW THEY DIFFER:

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  GEO+GEO (This document):                                      │
│  ────────────────────────                                       │
│                                                                 │
│  • Lives in the BASE QUADRANT                                  │
│  • Uses normalized similarity (geometry belief)                │
│  • Uses entropy/relative thresholds (geometry trigger)        │
│  • No explicit precision storage                               │
│  • Self-contained mechanism                                    │
│                                                                 │
│  ════════════════════════════════════════════════════════════  │
│                                                                 │
│  HOMEOSTAT (WH_HOMEOSTAT_...):                                 │
│  ─────────────────────────────                                  │
│                                                                 │
│  • Sits ABOVE the base quadrant as META-LAYER                 │
│  • Stores EXPLICIT precision (curvature Λ)                    │
│  • Provides Gershgorin stability, PSON exploration            │
│  • Can MODULATE any base mechanism                             │
│  • Adds precision-scaled updates                               │
│                                                                 │
│  ════════════════════════════════════════════════════════════  │
│                                                                 │
│  RELATIONSHIP:                                                  │
│                                                                 │
│  GEO+GEO can be used WITHOUT Homeostat                        │
│  (entropy-adaptive, no precision tracking)                     │
│                                                                 │
│  OR                                                             │
│                                                                 │
│  GEO+GEO can be ENHANCED BY Homeostat                         │
│  (Homeostat adjusts entropy threshold, temperature, etc.)     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 8.2 The Complete Architecture

```
THE FULL PICTURE:

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│                    H O M E O S T A T                           │
│              (Explicit Precision Λ, Meta-control)              │
│                         │                                       │
│     ┌───────────────────┼───────────────────┐                  │
│     │    Modulates:     │                   │                  │
│     │    • Thresholds   │                   │                  │
│     │    • Temperature  │                   │                  │
│     │    • Exploration  │                   │                  │
│     ▼                   ▼                   ▼                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │            │ ENERGY Trigger │ GEOMETRY Trigger           │  │
│  │────────────┼────────────────┼────────────────            │  │
│  │ ENERGY     │ IMPLICIT       │ COMPETITION                │  │
│  │ Belief     │ (reflex)       │ (winner)                   │  │
│  │────────────┼────────────────┼────────────────            │  │
│  │ GEOMETRY   │ HYBRID         │ GEO+GEO ←                  │  │
│  │ Belief     │ (current)      │ (This document)            │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
│               BASE QUADRANT (4 mechanisms)                     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Summary

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│                    G E O + G E O                               │
│            (Geometry Belief + Geometry Trigger)                │
│                                                                 │
│  ═══════════════════════════════════════════════════════════   │
│                                                                 │
│  BELIEF: GEOMETRY (Normalized)                                  │
│                                                                 │
│  • Cosine similarity on unit hypersphere                       │
│  • Bounded [-1, 1], direction matters                          │
│  • Softmax creates probability distribution                    │
│  • Scale-invariant, fair comparison                            │
│                                                                 │
│  ═══════════════════════════════════════════════════════════   │
│                                                                 │
│  TRIGGER: GEOMETRY (Relative/Adaptive)                          │
│                                                                 │
│  • Above-mean relative threshold                               │
│  • Entropy-based concentration detection                       │
│  • Gap-based winner detection                                  │
│  • Adaptive temperature control                                │
│                                                                 │
│  ═══════════════════════════════════════════════════════════   │
│                                                                 │
│  THE FULLY GEOMETRIC APPROACH:                                  │
│                                                                 │
│  "Is this the best match?"                                     │
│                                                                 │
│  No absolute thresholds, no raw magnitudes.                   │
│  Everything is relational and distribution-aware.             │
│  Collapse-aware through entropy monitoring.                   │
│                                                                 │
│  ═══════════════════════════════════════════════════════════   │
│                                                                 │
│  USE WHEN:                                                      │
│                                                                 │
│  • Variable difficulty, need adaptive thresholds              │
│  • Want entropy/collapse-aware decisions                       │
│  • Need relative selection (best, not good-enough)            │
│  • Don't want to track explicit precision                     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

**References:**

- `WORMHOLE_IMPLICIT.md` - True Implicit (energy + energy) - pure reflex
- `WORMHOLE_COMPETITION.md` - Competition (energy + geometry) - winner selection
- `WORMHOLE_HYBRID.md` - Hybrid (geometry + energy) - current wormhole
- `WH_HOMEOSTAT_STORED_EXPLICIT_PRECISION.md` - Homeostat (meta-layer with explicit precision)
- `POMDP_ATTENTION.md` - POMDP framework
- `COLLAPSE_GENERALIZATION.md` - Collapse and entropy dynamics

---

*This document describes the GEO+GEO quadrant: fully geometric attention where both belief (normalized similarity) and trigger (relative/entropy-based) respect manifold structure. It completes the base 2×2 quadrant alongside IMPLICIT, EXPLICIT, and HYBRID.*

---

*Oscar Goldman - Shogu Research Group @ Datamutant.ai*

