# Wormhole Attention: Competition (Energy Belief + Geometry Trigger)

Energy-based belief (raw activations, no normalization) combined with geometric/relational triggering. Magnitude influences the race, but relative position determines the winner.

---

## Table of Contents

1. [Overview](#1-overview)
2. [The Quadrant Position](#2-the-quadrant-position)
3. [Energy Belief](#3-energy-belief)
4. [Geometry Trigger](#4-geometry-trigger)
5. [Why This Combination Works](#5-why-this-combination-works)
6. [When This Makes Sense](#6-when-this-makes-sense)
7. [Implementation Sketch](#7-implementation-sketch)
8. [Comparison to Other Approaches](#8-comparison-to-other-approaches)

---

## 1. Overview

### 1.1 The Competition Design

```
WORMHOLE COMPETITION: ENERGY BELIEF + GEOMETRY TRIGGER

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  BELIEF REPRESENTATION: ENERGY (Raw Activations)               │
│  ───────────────────────────────────────────────                │
│                                                                 │
│  • NO normalization (not on hypersphere)                       │
│  • Raw dot product (magnitude matters)                         │
│  • Activation level IS belief                                  │
│  • No probability distribution via softmax                     │
│                                                                 │
│  ════════════════════════════════════════════════════════════  │
│                                                                 │
│  TRIGGER MECHANISM: GEOMETRY (Relational)                      │
│  ─────────────────────────────────────────                      │
│                                                                 │
│  • Compares activations to each other                          │
│  • "Is this the BEST?" not "Is this good enough?"             │
│  • Relative decisions: argmax, above-mean, top-k              │
│  • Considers distribution shape, not just magnitude            │
│                                                                 │
│  ════════════════════════════════════════════════════════════  │
│                                                                 │
│  THE COMPETITION:                                               │
│                                                                 │
│  • Magnitude influences who enters the race                   │
│  • Relative comparison picks the winner                       │
│  • "Energy races, geometry judges"                             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 Core Insight

```
THE KEY DIFFERENCE FROM OTHER QUADRANTS:

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  IMPLICIT (Energy + Energy):                                   │
│  "Is activation > threshold?" → Binary, absolute               │
│                                                                 │
│  COMPETITION (Energy + Geometry):   ← THIS DOCUMENT            │
│  "Is this the strongest activation?" → Winner, relative        │
│                                                                 │
│  HYBRID (Geometry + Energy):                                    │
│  "Is similarity > 0.92?" → Bounded, absolute                   │
│                                                                 │
│  GEO+GEO (Geometry + Geometry):                                │
│  "Is similarity above average?" → Bounded, relative            │
│                                                                 │
│  HOMEOSTAT (Explicit Precision + Geometry):                    │
│  "Is precision above average?" → Curvature, relative           │
│                                                                 │
│  ════════════════════════════════════════════════════════════  │
│                                                                 │
│  COMPETITION combines:                                          │
│  • Raw magnitude (can be arbitrarily large)                   │
│  • Relative comparison (pick best regardless of scale)        │
│                                                                 │
│  This is like:                                                  │
│  "I don't care how loud the signals are in absolute terms,    │
│   just tell me which one is loudest"                           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. The Quadrant Position

### 2.1 The Complete 2×2 Quadrant

```
THE BELIEF × TRIGGER QUADRANT:

┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│                         │    ENERGY TRIGGER      │   GEOMETRY TRIGGER      │
│                         │    (scalar threshold)  │   (relational/adaptive) │
│  ───────────────────────┼────────────────────────┼─────────────────────────│
│                         │                        │                         │
│  ENERGY BELIEF          │  IMPLICIT              │  COMPETITION ←          │
│  (raw activations,      │  ─────────             │  ───────────            │
│   no normalization)     │                        │  (This document)        │
│                         │  • act > threshold     │  • argmax(activation)   │
│                         │  • Binary gate         │  • Winner-take-all      │
│                         │  • Pure reflex         │  • Relative comparison  │
│                         │                        │                         │
│                         │  WORMHOLE_IMPLICIT.md  │  WORMHOLE_COMPETITION.md│
│                         │                        │                         │
│  ───────────────────────┼────────────────────────┼─────────────────────────│
│                         │                        │                         │
│  GEOMETRY BELIEF        │  HYBRID                │  GEO+GEO                │
│  (normalized features,  │  ──────                │  ───────                │
│   cosine similarity)    │                        │                         │
│                         │  • sim > 0.92          │  • sim > mean(sim)      │
│                         │  • Bounded [−1, 1]     │  • Relative threshold   │
│                         │  • Softmax weights     │  • Entropy-adaptive     │
│                         │                        │                         │
│                         │  WORMHOLE_HYBRID.md    │  WORMHOLE_GEO_GEO.md    │
│                         │                        │                         │
└─────────────────────────────────────────────────────────────────────────────┘

                    HOMEOSTAT sits ABOVE as meta-layer
                    (adds stored precision Λ to any mechanism)
```

### 2.2 The Explicitness Hierarchy

```
BELIEF EXPLICITNESS (LEAST → MOST):

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  LEAST EXPLICIT ────────────────────────────────── MOST EXPLICIT│
│                                                                 │
│  IMPLICIT     COMPETITION    HYBRID      GEO+GEO     HOMEOSTAT │
│  (E+E)        (E+G)          (G+E)       (G+G)       (Λ+G)     │
│     │              │             │           │           │      │
│     │              │             │           │           │      │
│     ▼              ▼             ▼           ▼           ▼      │
│  0% geo        50% geo       50% geo     100% geo   100% geo   │
│  0% explicit   (only trg)    (only bel)  (both)     + precision│
│                                                                 │
│  ════════════════════════════════════════════════════════════  │
│                                                                 │
│  NOTE: COMPETITION uses ENERGY belief (most implicit),         │
│  but has a GEOMETRY trigger (relational comparison).           │
│                                                                 │
│  This makes it LESS explicit overall than HYBRID or GEO+GEO,  │
│  despite having a relational trigger.                          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Energy Belief

### 3.1 Raw Activations (No Normalization)

```
ENERGY BELIEF = RAW DOT PRODUCT:

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  COMPUTATION:                                                   │
│                                                                 │
│  activation = query @ key.T  # Raw dot product                │
│                                                                 │
│  NO normalization:                                              │
│  • Query and key vectors can have any magnitude               │
│  • Result can be arbitrarily large or small                   │
│  • Not bounded to [-1, 1] like cosine                         │
│                                                                 │
│  ════════════════════════════════════════════════════════════  │
│                                                                 │
│  WHAT MAGNITUDE ENCODES:                                        │
│                                                                 │
│  activation = |query| · |key| · cos(θ)                        │
│                                                                 │
│  High activation can mean:                                      │
│  • Similar direction (high cos θ)          ← similarity       │
│  • Large query magnitude                    ← query "energy"  │
│  • Large key magnitude                      ← key "energy"    │
│  • All of the above                                            │
│                                                                 │
│  We don't separate these factors.                              │
│  That's why it's "energy" not "geometry".                     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Properties of Energy Belief

```
CHARACTERISTICS:

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  ADVANTAGES:                                                    │
│                                                                 │
│  • Fast: No normalization computation                          │
│  • Simple: Just matrix multiply                                │
│  • Magnitude-sensitive: Strong signals dominate               │
│  • Unbounded: Can express arbitrarily strong activation       │
│                                                                 │
│  ────────────────────────────────────────────────────────────  │
│                                                                 │
│  DISADVANTAGES:                                                 │
│                                                                 │
│  • Scale-dependent: Same pattern at different scales ≠        │
│  • Magnitude can swamp direction                               │
│  • Not comparable across different query magnitudes           │
│  • Numerical instability for large values                      │
│                                                                 │
│  ────────────────────────────────────────────────────────────  │
│                                                                 │
│  WHEN MAGNITUDE MATTERS:                                        │
│                                                                 │
│  • Feature magnitude encodes importance                        │
│  • Strong activations should dominate                          │
│  • Scale is meaningful, not just direction                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. Geometry Trigger

### 4.1 Relational Decisions

```
GEOMETRY TRIGGER = COMPARE TO OTHERS:

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  INSTEAD OF:                                                    │
│                                                                 │
│     if activation > threshold:  # Absolute (Energy trigger)   │
│         connect()                                               │
│                                                                 │
│  DO:                                                            │
│                                                                 │
│     if activation == max(all_activations):  # Relative        │
│         connect()  # Winner-take-all                           │
│                                                                 │
│  OR:                                                            │
│                                                                 │
│     if activation > mean(all_activations):  # Above average   │
│         connect()                                               │
│                                                                 │
│  OR:                                                            │
│                                                                 │
│     if activation in top_k(all_activations):  # Top-k         │
│         connect()                                               │
│                                                                 │
│  ════════════════════════════════════════════════════════════  │
│                                                                 │
│  THE KEY DIFFERENCE:                                            │
│                                                                 │
│  Energy trigger: "Is this signal strong enough?"              │
│  Geometry trigger: "Is this signal the strongest?"            │
│                                                                 │
│  The first uses absolute magnitude.                            │
│  The second uses relative position in the distribution.       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 Types of Geometry Triggers (for Energy Belief)

```
GEOMETRIC TRIGGER OPTIONS:

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  1. ARGMAX (Winner-Take-All)                                   │
│  ───────────────────────────                                    │
│                                                                 │
│     winner = activation.argmax(dim=-1)                         │
│     output = values[winner]                                    │
│                                                                 │
│     • Exactly one winner per query                            │
│     • Most selective                                           │
│     • Hard routing                                              │
│                                                                 │
│  ────────────────────────────────────────────────────────────  │
│                                                                 │
│  2. TOP-K (k Winners)                                           │
│  ────────────────────                                           │
│                                                                 │
│     topk_vals, topk_idx = torch.topk(activation, k)           │
│     output = values[topk_idx].mean()  # or weighted            │
│                                                                 │
│     • k winners per query                                      │
│     • Moderate selectivity                                     │
│     • Allows hedging                                            │
│                                                                 │
│  ────────────────────────────────────────────────────────────  │
│                                                                 │
│  3. ABOVE-MEAN (Adaptive Threshold)                            │
│  ──────────────────────────────────                             │
│                                                                 │
│     mean_act = activation.mean(dim=-1, keepdim=True)          │
│     mask = activation > mean_act                               │
│     output = (mask * values).sum() / mask.sum()               │
│                                                                 │
│     • Variable number of winners                               │
│     • Adapts to activation distribution                        │
│     • Relative threshold                                        │
│                                                                 │
│  ────────────────────────────────────────────────────────────  │
│                                                                 │
│  4. PERCENTILE (Top p%)                                         │
│  ──────────────────────                                         │
│                                                                 │
│     threshold = torch.quantile(activation, 0.9)  # Top 10%   │
│     mask = activation > threshold                              │
│                                                                 │
│     • Fixed fraction of winners                                │
│     • Robust to outliers                                       │
│     • Scale-invariant selection                                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 5. Why This Combination Works

### 5.1 The Tension and Resolution

```
THE COMPETITION LOGIC:

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  ENERGY BELIEF says:                                            │
│  "Magnitude matters. Strong signals are different from weak." │
│                                                                 │
│  GEOMETRY TRIGGER says:                                         │
│  "Relative position matters. Compare to others."              │
│                                                                 │
│  ════════════════════════════════════════════════════════════  │
│                                                                 │
│  THE RESOLUTION:                                                │
│                                                                 │
│  Sometimes you want BOTH:                                       │
│  • Let magnitude influence the race (energy belief)           │
│  • But pick winner(s) regardless of absolute scale (geo trig) │
│                                                                 │
│  Example: "Which expert should handle this, given their       │
│           current activation levels?"                          │
│                                                                 │
│  The expert with highest activation wins,                      │
│  but that activation reflects both match AND confidence.      │
│                                                                 │
│  ════════════════════════════════════════════════════════════  │
│                                                                 │
│  ANALOGY: A race where runners have different speeds (energy) │
│  but only ranking matters (geometry) - not absolute time.    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 When Magnitude Should Influence Relative Choice

```
SCENARIOS WHERE COMPETITION MAKES SENSE:

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  1. MIXTURE OF EXPERTS WITH CONFIDENCE                         │
│  ──────────────────────────────────────                         │
│                                                                 │
│     Each expert has activation = match × confidence            │
│     Higher confidence experts should be preferred              │
│     But we still want to pick the "best" not "good enough"    │
│                                                                 │
│     energy_activation = expert_match * expert_confidence      │
│     winner = argmax(energy_activation)  # Geometry trigger    │
│                                                                 │
│  ────────────────────────────────────────────────────────────  │
│                                                                 │
│  2. ATTENTION WITH IMPORTANCE WEIGHTING                        │
│  ───────────────────────────────────────                        │
│                                                                 │
│     Some keys are inherently more important (magnitude)       │
│     But we still want relative selection                       │
│                                                                 │
│     activation = query @ (importance * key).T                 │
│     top_k_winners = topk(activation, k)                        │
│                                                                 │
│  ────────────────────────────────────────────────────────────  │
│                                                                 │
│  3. SPARSE ACTIVATION WITH COMPETITION                         │
│  ──────────────────────────────────────                         │
│                                                                 │
│     Neurons compete, but magnitude reflects relevance         │
│     Winner-take-all with magnitude influence                  │
│                                                                 │
│     activation = raw_response  # Energy belief                │
│     winner = argmax(activation)  # Competition (geometry)     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 6. When This Makes Sense

### 6.1 Use Cases

```
APPROPRIATE APPLICATIONS:

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  1. HARD ROUTING WITH MAGNITUDE INFLUENCE                      │
│     ─────────────────────────────────────                       │
│     • Route to exactly one destination                         │
│     • But magnitude should bias the choice                    │
│     • Example: MoE with load-aware routing                    │
│                                                                 │
│  2. WINNER-TAKE-ALL COMPETITION                                │
│     ───────────────────────────────                             │
│     • Only strongest signal passes                             │
│     • Magnitude encodes signal strength                        │
│     • Example: Competitive learning                            │
│                                                                 │
│  3. SPARSE ATTENTION WITHOUT SOFTMAX                           │
│     ────────────────────────────────                            │
│     • Select top-k connections                                 │
│     • No probability distribution needed                       │
│     • Magnitude influences selection ranking                   │
│                                                                 │
│  4. FEATURE SELECTION WITH IMPORTANCE                          │
│     ─────────────────────────────────                           │
│     • Select most active features                              │
│     • Activation level = importance                            │
│     • Pick top-k most important                                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 6.2 When NOT to Use

```
INAPPROPRIATE APPLICATIONS:

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  1. WHEN SCALE SHOULD BE IGNORED                               │
│     ────────────────────────────                                │
│     • If you want pure similarity, use normalized (HYBRID)    │
│     • If magnitude is noise, don't let it influence           │
│                                                                 │
│  2. WHEN ABSOLUTE THRESHOLD IS KNOWN                           │
│     ─────────────────────────────────                           │
│     • If you know "activation > 5 means match"                │
│     • Use IMPLICIT (Energy + Energy)                          │
│                                                                 │
│  3. WHEN YOU NEED PROBABILITIES                                 │
│     ───────────────────────────                                 │
│     • Argmax gives no uncertainty                              │
│     • For soft attention, use HYBRID or GEO+GEO               │
│                                                                 │
│  4. WHEN INTERPRETABILITY MATTERS                              │
│     ─────────────────────────────                               │
│     • Raw activations are hard to interpret                   │
│     • For explicit belief tracking, use HOMEOSTAT             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 7. Implementation Sketch

### 7.1 Core Implementation

```python
class CompetitionWormhole(nn.Module):
    """
    Competition Wormhole: Energy belief + Geometry trigger.
    
    Raw activations (no normalization) with relative selection.
    "Energy races, geometry judges"
    """
    
    def __init__(
        self,
        feature_dim: int,
        trigger_type: str = 'topk',  # 'argmax', 'topk', 'above_mean', 'percentile'
        k: int = 4,  # for topk
        percentile: float = 0.9,  # for percentile
        device: torch.device = torch.device('cpu')
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.trigger_type = trigger_type
        self.k = k
        self.percentile = percentile
        
        # Projections (no normalization will be applied)
        self.W_q = nn.Linear(feature_dim, feature_dim, bias=False)
        self.W_k = nn.Linear(feature_dim, feature_dim, bias=False)
        self.W_v = nn.Linear(feature_dim, feature_dim, bias=False)
        
        self.to(device)
    
    def forward(
        self,
        query: torch.Tensor,   # [N, D]
        keys: torch.Tensor,    # [M, D]
        values: torch.Tensor,  # [M, D]
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass with energy belief + geometry trigger.
        """
        # Project (NO normalization - energy belief)
        Q = self.W_q(query)   # [N, D]
        K = self.W_k(keys)    # [M, D]
        V = self.W_v(values)  # [M, D]
        
        # Raw dot product (ENERGY BELIEF)
        activation = Q @ K.T  # [N, M] - unbounded
        
        # Apply GEOMETRY TRIGGER
        output, mask = self._apply_geometry_trigger(activation, V)
        
        stats = {
            'activation_mean': activation.mean().item(),
            'activation_max': activation.max().item(),
            'activation_min': activation.min().item(),
            'connections_per_query': mask.sum(dim=-1).float().mean().item(),
        }
        
        return output, stats
    
    def _apply_geometry_trigger(
        self,
        activation: torch.Tensor,  # [N, M]
        values: torch.Tensor,      # [M, D]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply geometry trigger to select connections.
        """
        N, M = activation.shape
        D = values.shape[-1]
        
        if self.trigger_type == 'argmax':
            # Winner-take-all
            winner_idx = activation.argmax(dim=-1)  # [N]
            output = values[winner_idx]  # [N, D]
            mask = F.one_hot(winner_idx, M).bool()  # [N, M]
            
        elif self.trigger_type == 'topk':
            # Top-k winners
            topk_vals, topk_idx = torch.topk(activation, self.k, dim=-1)
            V_selected = values[topk_idx]  # [N, k, D]
            output = V_selected.mean(dim=1)  # [N, D]
            mask = torch.zeros(N, M, dtype=torch.bool, device=activation.device)
            mask.scatter_(1, topk_idx, True)
            
        elif self.trigger_type == 'above_mean':
            # Above mean (adaptive threshold)
            mean_act = activation.mean(dim=-1, keepdim=True)  # [N, 1]
            mask = activation > mean_act  # [N, M]
            masked_act = activation * mask.float()
            weights = masked_act / (masked_act.sum(dim=-1, keepdim=True) + 1e-8)
            output = (weights.unsqueeze(-1) * values.unsqueeze(0)).sum(dim=1)
            
        elif self.trigger_type == 'percentile':
            # Top percentile
            threshold = torch.quantile(activation, self.percentile, dim=-1, keepdim=True)
            mask = activation >= threshold  # [N, M]
            masked_act = activation * mask.float()
            weights = masked_act / (masked_act.sum(dim=-1, keepdim=True) + 1e-8)
            output = (weights.unsqueeze(-1) * values.unsqueeze(0)).sum(dim=1)
            
        else:
            raise ValueError(f"Unknown trigger type: {self.trigger_type}")
        
        return output, mask
```

---

## 8. Comparison to Other Approaches

### 8.1 Side-by-Side

```
┌───────────────────────────────────────────────────────────────────────────────┐
│                                                                               │
│  ASPECT          IMPLICIT    COMPETITION   HYBRID       GEO+GEO    HOMEOSTAT│
│  ──────          ────────    ───────────   ──────       ───────    ─────────│
│                                                                               │
│  Belief          Energy      Energy        Geometry     Geometry   Precision│
│                  (unbounded) (unbounded)   (bounded)    (bounded)  (Λ)      │
│                                                                               │
│  Trigger         Energy      Geometry      Energy       Geometry   Geometry │
│                  (absolute)  (relative)    (absolute)   (relative) (relative)│
│                                                                               │
│  Normalize       No          No            Yes          Yes        Yes       │
│                                                                               │
│  Selection       Binary      Winner/top-k  Threshold    Adaptive   Adaptive │
│                                                                               │
│  Adaptivity      None        To scale      None         Entropy    Full     │
│                                                                               │
│  Magnitude       Matters     Influences    Ignored      Ignored    Curvature│
│                  (threshold) (competition) (direction)  (direction)(precision)│
│                                                                               │
│  Speed           Fastest     Fast          Medium       Medium     Slower   │
│                                                                               │
│  Use case        Hard gate   Hard route    General      Collapse-  Adaptive │
│                              with mag.     attention    aware      reasoning│
│                                                                               │
└───────────────────────────────────────────────────────────────────────────────┘
```

### 8.2 The Spectrum

```
THE FIVE MECHANISMS ON A SPECTRUM:

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  LESS EXPLICIT ─────────────────────────────── MORE EXPLICIT   │
│                                                                 │
│  IMPLICIT    COMPETITION    HYBRID      GEO+GEO     HOMEOSTAT │
│  │           │              │           │           │          │
│  │ E+E       │ E+G          │ G+E       │ G+G       │ Λ+G      │
│  │           │              │           │           │          │
│  │ 0% geo    │ 50% geo      │ 50% geo   │ 100% geo  │ 100%+Λ  │
│  │           │              │           │           │          │
│  ▼           ▼              ▼           ▼           ▼          │
│                                                                 │
│  "Strong    "Strongest?"   "Similar    "Best       "Precise   │
│   enough?"                  enough?"    match?"     enough?"   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Summary

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│          W O R M H O L E   C O M P E T I T I O N               │
│              (Energy Belief + Geometry Trigger)                │
│                                                                 │
│  ═══════════════════════════════════════════════════════════   │
│                                                                 │
│  BELIEF: ENERGY                                                 │
│                                                                 │
│  • Raw dot product (no normalization)                          │
│  • Magnitude matters                                            │
│  • Unbounded activations                                        │
│  • Fast, simple computation                                     │
│                                                                 │
│  ═══════════════════════════════════════════════════════════   │
│                                                                 │
│  TRIGGER: GEOMETRY                                              │
│                                                                 │
│  • Relative comparison (argmax, topk, above-mean)             │
│  • "Which is strongest?" not "Is this strong enough?"         │
│  • Distribution-aware selection                                │
│  • Adaptive to activation scale                                │
│                                                                 │
│  ═══════════════════════════════════════════════════════════   │
│                                                                 │
│  THE COMPETITION MOTTO:                                         │
│                                                                 │
│  "Energy races, geometry judges"                               │
│                                                                 │
│  Magnitude influences who's in the race.                       │
│  Relative position determines the winner.                      │
│  No fixed threshold, no normalization.                         │
│                                                                 │
│  ═══════════════════════════════════════════════════════════   │
│                                                                 │
│  USE WHEN:                                                      │
│                                                                 │
│  • Want hard routing with magnitude influence                  │
│  • Need winner-take-all or top-k selection                    │
│  • Magnitude encodes meaningful information                    │
│  • Want relative selection without normalization overhead     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

**References:**

- `WORMHOLE_IMPLICIT.md` - Implicit (energy + energy) - pure reflex
- `WORMHOLE_HYBRID.md` - Hybrid (geometry + energy) - current wormhole
- `WORMHOLE_GEO_GEO.md` - GEO+GEO (geometry + geometry) - entropy-adaptive
- `WH_HOMEOSTAT_STORED_EXPLICIT_PRECISION.md` - Homeostat (meta-layer with precision)
- `POMDP_ATTENTION.md` - POMDP framework
- `KNOWLEDGE_AND_REACTIVITY.md` - Geometry vs energy distinction

---

## Related Concepts

```
THE CONSISTENT QUADRANT:

┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│                         │    ENERGY TRIGGER      │   GEOMETRY TRIGGER      │
│                         │    (absolute)          │   (relative)            │
│  ───────────────────────┼────────────────────────┼─────────────────────────│
│                         │                        │                         │
│  ENERGY BELIEF          │  IMPLICIT              │  COMPETITION ←          │
│  (raw activation)       │  (pure reflex)         │  (This document)        │
│                         │                        │  "energy races"         │
│  ───────────────────────┼────────────────────────┼─────────────────────────│
│                         │                        │                         │
│  GEOMETRY BELIEF        │  HYBRID                │  GEO+GEO                │
│  (normalized)           │  (current wormhole)    │  (entropy-adaptive)     │
│                         │  "geo fills, E gates"  │  "best match"           │
│                         │                        │                         │
└─────────────────────────────────────────────────────────────────────────────┘

                    HOMEOSTAT sits ABOVE as meta-layer
                    (adds stored precision Λ)
```

---

*This document describes the Competition quadrant: energy belief (raw activations) combined with geometry trigger (relative comparison). It's the "magnitude-influenced winner selection" approach.*

---

*Oscar Goldman, Shogu Research Group @ Datamutant.ai subsidiary of 温心重工業*