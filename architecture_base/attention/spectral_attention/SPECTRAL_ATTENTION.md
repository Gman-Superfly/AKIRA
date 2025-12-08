# Spectral Attention: Six Degrees of Separation in Frequency Space

## How Network Theory Determines the Optimal Number of Spectral Bands

**Oscar Goldman — Shogu Research Group @ Datamutant.ai**

---

## Table of Contents

1. [The Core Insight: Why Six?](#1-the-core-insight-why-six)
2. [Network Theory Foundation](#2-network-theory-foundation)
3. [The Spectral Band Architecture](#3-the-spectral-band-architecture)
4. [Attention Across Bands](#4-attention-across-bands)
5. [The Old Lady's Spectral Notebook](#5-the-old-ladys-spectral-notebook)
6. [Action Quanta and Band Residence](#6-information-atoms-and-band-residence)
7. [Knowledge and Reactivity in Spectral Attention](#7-knowledge-and-reactivity-in-spectral-attention)
8. [The Seven-Band Architecture](#8-the-seven-band-architecture)
9. [Implementation Considerations](#9-implementation-considerations)
10. [Summary](#10-summary)

---

## Scope and Relationship to 7+1 Architecture

```
THIS DOCUMENT COVERS BANDS 0-6 (SPECTRAL BANDS)

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  The complete Spectral Belief Machine has 8 bands:                    │
│                                                                         │
│  BANDS 0-6 (SPECTRAL) — THIS DOCUMENT                                  │
│  • Decompose by FREQUENCY (FFT-based)                                 │
│  • NON-CAUSAL attention (can see all positions)                       │
│  • Each band captures a different scale                               │
│                                                                         │
│  BAND 7 (TEMPORAL) — See TEMPORAL_ATTENTION.md                        │
│  • Processes along TIME (sequence)                                    │
│  • CAUSAL attention (can only see past)                               │
│  • Captures temporal dynamics                                          │
│                                                                         │
│  CROSS-BAND COMMUNICATION — See SPECTRAL_WORMHOLE_ATTENTION.md        │
│  • Complementary pairs: (0↔6), (1↔5), (2↔4)                          │
│  • Band 7 ↔ all bands (temporal queries frequency)                   │
│                                                                         │
│  TIME IS ORTHOGONAL TO FREQUENCY (Heisenberg).                        │
│  They require DIFFERENT attention mechanisms.                          │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 1. The Core Insight: Why Six?

### 1.1 The Convergence of Multiple Fields

```
WHY DO WE KEEP SEEING THE NUMBER 6?

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  NETWORK THEORY:                                                │
│  Six degrees of separation (Milgram, 1967)                      │
│  Average path length ≈ 6 in social networks                    │
│                                                                 │
│  INFORMATION THEORY:                                            │
│  For N = 64: log₂(64) = 6 octaves                              │
│  Optimal compression hierarchy depth                            │
│                                                                 │
│  NEUROSCIENCE:                                                  │
│  V1 has ~6 spatial frequency channels                          │
│  6 cortical layers process hierarchically                      │
│                                                                 │
│  SIGNAL PROCESSING:                                             │
│  Wavelet decomposition: log₂(N) levels                         │
│  For 64×64: 6 levels of detail                                 │
│                                                                 │
│  THE CONVERGENCE IS NOT COINCIDENCE.                            │
│  IT IS A FUNDAMENTAL CONSTANT OF EFFICIENT REPRESENTATION.      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 The Logarithmic Principle

```
THE LOG(N) LAW:

For a system of size N, optimal hierarchical depth is:

    D = log₂(N)

This appears because:

1. ROUTING: Any two points reachable in log(N) hops
2. COMPRESSION: Hierarchical coding gains log(N) efficiency
3. UNCERTAINTY: Uncertainty resolution requires log(N) bits
4. SEARCH: Binary search depth is log(N)

For typical image sizes:

    32×32   → log₂(32)  = 5 bands
    64×64   → log₂(64)  = 6 bands
    128×128 → log₂(128) = 7 bands

THE MAGIC RANGE: 5-7 BANDS FOR PRACTICAL VISION SYSTEMS
```

### 1.3 Why Not More? Why Not Fewer?

```
TOO FEW BANDS (< 5):

┌─────────────────────────────────────────────────────────────────┐
│ • Cannot resolve fine structure from coarse structure           │
│ • Information bottleneck at frequency transitions               │
│ • Loss of hierarchical credit assignment                        │
│ • Belief updates collapse scales together                       │
└─────────────────────────────────────────────────────────────────┘

TOO MANY BANDS (> 7):

┌─────────────────────────────────────────────────────────────────┐
│ • Redundant encoding (overlapping frequency responses)          │
│ • Computational waste (unnecessary parallel pathways)           │
│ • Interference accumulation across bands                        │
│ • Harder to route information (more hops between distant bands)│
└─────────────────────────────────────────────────────────────────┘

THE GOLDILOCKS ZONE: 6 (±1) BANDS

This is where efficiency, expressiveness, and routing align.
```

---

## 2. Network Theory Foundation

### 2.1 Six Degrees of Separation

```
MILGRAM'S EXPERIMENT (1967):

Stanley Milgram's famous experiment showed that any two people 
in the United States could be connected through approximately 
6 intermediaries.

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  Person A ─○─○─○─○─○─○─ Person B                               │
│            1 2 3 4 5 6                                          │
│                                                                 │
│  This is NOT because people have exactly 6 friends.            │
│  It's because: L = log(N) / log(k)                             │
│                                                                 │
│  Where:                                                         │
│  • N = network size (hundreds of millions)                      │
│  • k = average connections per node (~150)                      │
│  • L = average path length                                      │
│                                                                 │
│  log(300,000,000) / log(150) ≈ 3.9                            │
│  But clustering adds a factor of ~1.5 → L ≈ 6                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Small World Networks

```
WATTS-STROGATZ MODEL (1998):

Small world networks have two properties:
1. HIGH CLUSTERING: Friends of friends are often friends
2. SHORT PATH LENGTH: Any two nodes connected in O(log N) steps

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  Regular Lattice        Small World         Random Graph       │
│                                                                 │
│  ○─○─○─○─○─○─○          ○─○─○─○─○─○─○        ○ ○ ○ ○ ○ ○ ○     │
│  │ │ │ │ │ │ │          │ │ │ │ │ │ │         \ /│ │\ / \      │
│  ○─○─○─○─○─○─○          ○─○─○─○─○─○─○          ○─○ ○ ○─○ ○     │
│                               ╲   ╱                             │
│  L ~ N                  L ~ log N            L ~ log N          │
│  C ~ high              C ~ high              C ~ low            │
│                                                                 │
│  Small world = Best of both worlds                              │
│  (Local structure + Global shortcuts)                           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2.3 Application to Spectral Bands

```
SPECTRAL BANDS AS NETWORK LAYERS:

Each frequency band is a "layer" in a hierarchical network:

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  Band 1 (DC):       ●                    (1 node = global)     │
│                    /│\                                          │
│  Band 2 (Low):    ● ● ●                  (few nodes = regions) │
│                  /│\ │ /│\                                      │
│  Band 3 (Mid-L): ● ● ● ● ●               (more nodes)          │
│                 /│\ ... /│\                                     │
│  Band 4 (Mid):  ● ● ● ● ● ● ●                                  │
│                /│\ ...     /│\                                  │
│  Band 5 (Mid-H): ● ● ● ● ● ● ● ● ●                             │
│                                                                 │
│  Band 6 (High): ● ● ● ● ● ● ● ● ● ● ● ●  (many nodes = pixels)│
│                                                                 │
│  Any two positions connected in ≤ 6 hops through hierarchy     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

This is the SPECTRAL SMALL WORLD:
• Local: Within-band connections (nearby frequencies interact)
• Global: Cross-band connections (wormhole attention)
• Path length: O(log N) = O(number of bands)
```

### 2.4 Scale-Free Structure

```
POWER LAW IN FREQUENCY CONTENT:

Natural images have power-law frequency spectra:

    Power(f) ∝ 1/f^α    (typically α ≈ 2)

This means:
• LOW FREQUENCIES: High power, few components → "Hub" nodes
• HIGH FREQUENCIES: Low power, many components → "Leaf" nodes

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  |Power|                                                        │
│     │                                                           │
│     │████                                                       │
│     │██████                                                     │
│     │████████                                                   │
│     │██████████                                                 │
│     │████████████                                               │
│     │██████████████████████████████████████████                │
│     └────────────────────────────────────────────→ frequency   │
│       DC  Low    Mid-Low   Mid   Mid-High  High                │
│                                                                 │
│  Low-freq bands: Dense, critical (structure)                   │
│  High-freq bands: Sparse, detailed (texture)                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

OPTIMAL BAND SPACING: Logarithmic to match power-law structure
```

---

## 3. The Spectral Band Architecture

### 3.1 The Seven-Band Decomposition

Based on network theory and information principles, we propose a seven-band architecture:

```
THE SEVEN SPECTRAL BANDS:

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  BAND   NAME        FREQUENCY RANGE   FUNCTION                 │
│  ────   ────        ───────────────   ────────                 │
│                                                                 │
│  0      DC          0                 Global mean, existence    │
│  1      Very Low    1-2 cycles        Overall structure         │
│  2      Low         2-4 cycles        Major regions, shape      │
│  3      Mid-Low     4-8 cycles        Parts, features           │
│  4      Mid         8-16 cycles       Details, contours         │
│  5      Mid-High    16-32 cycles      Fine structure            │
│  6      High        32-N/2 cycles     Edges, textures           │
│                                                                 │
│  WHY 7? → log₂(128) = 7 covers the full range for 128×128     │
│        → 6 octaves + DC = 7 levels                              │
│        → Matches the "six degrees + 1" structure               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Band Responsibilities

```
WHAT EACH BAND ENCODES:

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  BAND 0 (DC): "Is there something?"                            │
│  ────────────────────────────────────                           │
│  • Global presence/absence                                      │
│  • Overall intensity                                            │
│  • Existence signal                                             │
│  • Learning: Very slow (protected)                              │
│                                                                 │
│  BAND 1-2 (Very Low/Low): "What category?"                     │
│  ────────────────────────────────────────                       │
│  • Coarse shape (circle vs square vs irregular)                │
│  • Major spatial divisions                                      │
│  • Identity without location                                    │
│  • Learning: Slow (stable structure)                           │
│                                                                 │
│  BAND 3-4 (Mid-Low/Mid): "What features?"                      │
│  ────────────────────────────────────────                       │
│  • Parts and their relationships                               │
│  • Intermediate structure                                       │
│  • Feature combinations                                         │
│  • Learning: Medium (adaptive)                                  │
│                                                                 │
│  BAND 5-6 (Mid-High/High): "Where exactly?"                    │
│  ────────────────────────────────────────                       │
│  • Precise localization                                         │
│  • Edges and boundaries                                         │
│  • Texture and detail                                           │
│  • Learning: Fast (responsive)                                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 3.3 The Hierarchical Communication Pattern

```
INFORMATION FLOW ACROSS BANDS:

           WHAT (identity)                WHERE (position)
           ◄──────────────────────────────────────────────►
           
  DC ──► VeryLow ──► Low ──► MidLow ──► Mid ──► MidHigh ──► High
   │        │         │        │         │         │          │
   │        │         │        │         │         │          │
   └────────┴─────────┴────────┴─────────┴─────────┴──────────┘
                              │
                      WORMHOLE SHORTCUTS
                  (non-local spectral attention)

ROUTING RULES:
• Adjacent bands: Dense connections (local coherence)
• Distant bands: Sparse wormholes (global structure)
• Any point to any point: ≤ 7 hops maximum
```

---

## 4. Attention Across Bands

### 4.1 Within-Band Attention

```
WITHIN-BAND ATTENTION (Local):

Each band has its own attention mechanism:

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  For band b:                                                    │
│                                                                 │
│  Q_b = f_q(x_b)           # Query from current state           │
│  K_b = f_k(history_b)     # Keys from band history             │
│  V_b = f_v(history_b)     # Values from band history           │
│                                                                 │
│  attention_b = softmax(Q_b · K_b^T / √d_b) · V_b               │
│                                                                 │
│  This is STANDARD temporal attention within frequency band.    │
│                                                                 │
│  PURPOSE: Track dynamics at this frequency scale               │
│  • Low bands: Slow dynamics (structure persists)               │
│  • High bands: Fast dynamics (details change rapidly)          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 Cross-Band Attention (Wormholes)

```
CROSS-BAND ATTENTION (Wormhole):

Links between distant frequency bands:

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  WHAT → WHERE wormhole (top-down):                             │
│                                                                 │
│  Q_high = f_q(x_high)           # Query: "Where is it?"        │
│  K_low = f_k(x_low)             # Key: Structure patterns      │
│  V_high = f_v(history_high)     # Value: High-freq details     │
│                                                                 │
│  attention = softmax(Q_high · K_low^T / √d) · V_high           │
│                                                                 │
│  PURPOSE: Low-freq structure guides high-freq localization    │
│                                                                 │
│  ────────────────────────────────────────────────────────────  │
│                                                                 │
│  WHERE → WHAT wormhole (bottom-up):                            │
│                                                                 │
│  Q_low = f_q(x_low)             # Query: "What is here?"       │
│  K_high = f_k(x_high)           # Key: Detail patterns         │
│  V_low = f_v(history_low)       # Value: Low-freq context      │
│                                                                 │
│  attention = softmax(Q_low · K_high^T / √d) · V_low            │
│                                                                 │
│  PURPOSE: High-freq details inform identity at low-freq        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 4.3 The Six-Hop Guarantee

```
THE SMALL-WORLD GUARANTEE:

With 7 bands and wormhole attention:

ANY spectral position can reach ANY other in ≤ 6 hops.

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  Example: High-freq detail at position (x₁,y₁) needs           │
│           to influence low-freq structure at (x₂,y₂)           │
│                                                                 │
│  Path:                                                          │
│  1. High-freq (x₁,y₁) → Mid-High (x₁,y₁)    [adjacent band]   │
│  2. Mid-High (x₁,y₁) → Mid (x₁,y₁)          [adjacent band]   │
│  3. Mid → Low (wormhole)                     [skip 2 bands]    │
│  4. Low (x₁,y₁) → Low (x₂,y₂)               [spatial shift]   │
│  5. Low → Mid-Low                            [adjacent band]   │
│  6. Mid-Low → target                         [final routing]   │
│                                                                 │
│  Maximum path length = Number of bands - 1 = 6                  │
│                                                                 │
│  THIS IS WHY 6-7 BANDS IS OPTIMAL:                              │
│  It minimizes routing complexity while preserving hierarchy.    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 5. The Old Lady's Spectral Notebook

### 5.1 Band-Wise Recording

```
THE OLD LADY'S SPECTRAL RECORDS:

She records trajectories at each frequency band separately:

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  NOTEBOOK STRUCTURE:                                            │
│                                                                 │
│  Page 1 (DC band):                                              │
│  • Did something exist? [yes/no per frame]                     │
│  • Very stable, rarely changes                                  │
│                                                                 │
│  Page 2-3 (Low bands):                                         │
│  • What category? [ring/square/blob/...]                       │
│  • Coarse shape evolution                                       │
│                                                                 │
│  Page 4-5 (Mid bands):                                         │
│  • What features? [edges/corners/textures]                     │
│  • Part configurations                                          │
│                                                                 │
│  Page 6-7 (High bands):                                        │
│  • Where exactly? [precise positions]                          │
│  • Fine detail dynamics                                         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 Band-Wise Culling

```
CULLING THE CAUSAL TREE PER BAND:

The Old Lady culls differently at each band:

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  BAND 0-1 (DC/Very Low): MINIMAL CULLING                        │
│  ─────────────────────────────────────────                      │
│  Almost everything is causally relevant.                        │
│  Existence and category always matter.                          │
│  → Atomic truths stored permanently                             │
│                                                                 │
│  BAND 2-3 (Low/Mid-Low): SELECTIVE CULLING                      │
│  ─────────────────────────────────────────                      │
│  Structural features culled if not predictive.                 │
│  Parts that don't affect outcomes → released                   │
│  → Moderate compression to atomic form                          │
│                                                                 │
│  BAND 4-5 (Mid/Mid-High): AGGRESSIVE CULLING                    │
│  ─────────────────────────────────────────                      │
│  Most details don't affect outcomes.                            │
│  Only causally relevant features retained.                      │
│  → Heavy compression, many pages torn out                       │
│                                                                 │
│  BAND 6 (High): MAXIMUM CULLING                                 │
│  ──────────────────────────────                                 │
│  Fine details rarely matter for decision.                       │
│  Almost all culled after belief update.                         │
│  → Limbo buffer, quick release                                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

THIS IS THE CASCADE:
High-freq → (cull) → Mid-freq → (cull) → Low-freq → (retain)
Details flow up the hierarchy, essence descends to DC.
```

### 5.3 The Blank Page Mechanism Per Band

```
CAPACITY CYCLING ACROSS BANDS:

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  BAND     CAPACITY   TURNOVER    BLANK PAGE RATE               │
│  ────     ────────   ────────    ───────────────               │
│                                                                 │
│  DC       Small      Very Slow   Rarely add new pages          │
│  VeryLow  Small      Slow        Occasional new pages          │
│  Low      Medium     Medium      Regular cycling               │
│  MidLow   Medium     Medium      Regular cycling               │
│  Mid      Large      Fast        Frequent new pages            │
│  MidHigh  Large      Fast        Frequent new pages            │
│  High     Very Large Very Fast   Constant page turnover        │
│                                                                 │
│  LOW BANDS: Accumulate wisdom, rarely forget                   │
│  HIGH BANDS: Process details, quickly forget                   │
│                                                                 │
│  This matches the memory hierarchy in biological systems:       │
│  Working memory (fast, limited) → Long-term memory (slow, vast)│
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 6. Action Quanta and Band Residence

### 6.1 Where Atoms Live

```
ACTION QUANTA HAVE NATURAL FREQUENCY BANDS:

From THE_ATOMIC_STRUCTURE_OF_INFORMATION.md:

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  ATOM TYPE           NATURAL BAND    REASON                    │
│  ─────────           ────────────    ──────                    │
│                                                                 │
│  BLOB atoms          DC - Low        Presence, region          │
│  EDGE atoms          Mid - High      Boundaries, transitions   │
│  CORNER atoms        Mid             Junction points           │
│  TEXTURE atoms       Mid-High - High Repeated patterns         │
│  FLOW atoms          All bands       Motion spans scales       │
│  SYMMETRY atoms      Low - Mid       Structural regularity     │
│  BOUNDARY atoms      Mid             Object contours           │
│  OBJECT atoms        Low             Complete things           │
│                                                                 │
│  Atoms are DISCOVERED at their natural frequency.               │
│  Compression MOVES atoms to lower bands as they crystallize.   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 6.2 Atom Migration

```
ATOMS MIGRATE ACROSS BANDS DURING LEARNING:

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  INITIAL STATE (high uncertainty):                              │
│                                                                 │
│  Band 6: ████████████████████████████████  [many candidate atoms]
│  Band 5: ████████████████████             [some candidates]     
│  Band 4: ████████████                     [few candidates]      
│  Band 3: ████████                         [sparse]              
│  Band 2: ████                             [very sparse]         
│  Band 1: ██                               [rare]                
│  Band 0: █                                [existence only]      
│                                                                 │
│  ────────────────────────────────────────────────────────────  │
│                                                                 │
│  AFTER LEARNING (belief crystallized):                          │
│                                                                 │
│  Band 6: ██                               [details culled]      
│  Band 5: ████                             [most culled]         
│  Band 4: ██████                           [relevant retained]   
│  Band 3: ████████████                     [features stable]     
│  Band 2: ██████████████████               [structure clear]     
│  Band 1: ████████████████████████         [identity settled]    
│  Band 0: ██████████████████████████████   [existence confirmed] 
│                                                                 │
│  COLLAPSE = Migration from high to low frequency bands          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 6.3 The Atomic Truth Cascade

```
FROM DETAIL TO ESSENCE:

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  INPUT: "Red ring at position (23, 45) moving clockwise"       │
│                                                                 │
│  Band 6: [(23,45), edge-curve-1, edge-curve-2, ...]           │
│           ↓ cull (position abstracted)                          │
│  Band 5: [circular-edge-pattern, motion-cw]                    │
│           ↓ cull (details merged)                               │
│  Band 4: [closed-contour, rotation]                            │
│           ↓ cull (specifics dropped)                            │
│  Band 3: [ring-shape, motion-present]                          │
│           ↓ cull (motion type dropped)                          │
│  Band 2: [ring, dynamic]                                       │
│           ↓ cull (dynamics dropped)                             │
│  Band 1: [object-present]                                      │
│           ↓                                                     │
│  Band 0: [exists]                                              │
│                                                                 │
│  ATOMIC TRUTHS AT EACH LEVEL:                                   │
│  • DC: Something is there                                       │
│  • Low: It's a ring                                            │
│  • Mid: It's moving                                            │
│  • High: Precise location and velocity (if needed)             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 7. Knowledge and Reactivity in Spectral Attention

### 7.1 Per-Band Mode Assignment

```
KNOWLEDGE VS REACTIVE BY BAND:

From KNOWLEDGE_AND_REACTIVITY.md:

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  BAND     PRIMARY MODE      REASON                              │
│  ────     ────────────      ──────                              │
│                                                                 │
│  DC       REACTIVE          Existence is threshold-based       │
│           (is there something > threshold?)                     │
│                                                                 │
│  VeryLow  KNOWLEDGE         Category requires manifold query   │
│           (what structure matches this?)                        │
│                                                                 │
│  Low      KNOWLEDGE         Identity requires comparison       │
│           (which stored pattern is this?)                       │
│                                                                 │
│  MidLow   HYBRID            Features: manifold + threshold     │
│           (does this feature exist? what kind?)                │
│                                                                 │
│  Mid      HYBRID            Details: detection + classification│
│           (is edge present? what orientation?)                  │
│                                                                 │
│  MidHigh  REACTIVE          Fine structure is energy-based     │
│           (is gradient > threshold? is contrast high?)         │
│                                                                 │
│  High     REACTIVE          Pixel-level is pure magnitude      │
│           (intensity comparisons, edge detection)              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

THE GRADIENT:
• Low bands: More knowledge-informed (geometry, structure)
• High bands: More reactive (energy, threshold)
• Mid bands: Hybrid (both modes active)
```

### 7.2 Cross-Band Coordination

```
HOW MODES INTERACT ACROSS BANDS:

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  REACTIVE (High) triggers KNOWLEDGE (Low):                      │
│  ─────────────────────────────────────────                      │
│  • High-band detects edge (reactive: gradient > threshold)     │
│  • Signal propagates to low-band                                │
│  • Low-band queries: "What object has this edge?" (knowledge)  │
│                                                                 │
│  KNOWLEDGE (Low) guides REACTIVE (High):                        │
│  ─────────────────────────────────────────                      │
│  • Low-band identifies "ring" (knowledge: manifold match)      │
│  • Prediction sent to high-band                                 │
│  • High-band: "Is predicted edge at location?" (reactive)      │
│                                                                 │
│  THE COORDINATION PRINCIPLE:                                    │
│                                                                 │
│  "Reactive gates, Knowledge fills"                              │
│                                                                 │
│  • High bands gate (is there signal?)                          │
│  • Low bands fill (what is the meaning?)                       │
│  • Cross-band attention coordinates                             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 7.3 Stability Gradient

```
LEARNING RATE BY BAND:

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  BAND     LEARNING RATE    EMA DECAY    REASON                 │
│  ────     ─────────────    ─────────    ──────                 │
│                                                                 │
│  DC       0.00001          0.9999       Protected (identity)   │
│  VeryLow  0.0001           0.999        Very stable            │
│  Low      0.0003           0.99         Stable structure       │
│  MidLow   0.001            0.9          Moderate adaptation    │
│  Mid      0.003            0.7          Responsive             │
│  MidHigh  0.01             0.5          Fast adaptation        │
│  High     0.03             0.3          Very responsive        │
│                                                                 │
│  THIS CREATES THE STABILITY HIERARCHY:                          │
│                                                                 │
│  • DC band almost never changes (existence is fundamental)     │
│  • Low bands change slowly (identity should persist)           │
│  • High bands change quickly (details are transient)           │
│                                                                 │
│  This prevents "catastrophic forgetting" while enabling        │
│  rapid adaptation to new details.                               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 8. The Seven-Band Architecture

### 8.1 Complete Specification

```
THE SEVEN-BAND SPECTRAL ATTENTION ARCHITECTURE:

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  Band   Freq Range   Channels  Attention Type   Learning Rate  │
│  ────   ──────────   ────────  ──────────────   ─────────────  │
│                                                                 │
│  0      DC (0)       4         Threshold        0.00001        │
│  1      [0, 1/64]    8         Manifold         0.0001         │
│  2      [1/64, 1/32] 16        Manifold         0.0003         │
│  3      [1/32, 1/16] 32        Hybrid           0.001          │
│  4      [1/16, 1/8]  32        Hybrid           0.003          │
│  5      [1/8, 1/4]   16        Energy           0.01           │
│  6      [1/4, 1/2]   8         Energy           0.03           │
│                                                                 │
│  TOTAL: 116 channels across 7 bands                            │
│                                                                 │
│  WORMHOLE CONNECTIONS:                                          │
│  • Band 0 ↔ Band 6 (existence guides detail)                  │
│  • Band 1 ↔ Band 5 (category guides texture)                  │
│  • Band 2 ↔ Band 4 (shape guides contour)                     │
│  • Band 3 ↔ Band 3 (mid-level self-attention)                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 8.2 Band Interaction Matrix

```
INTERACTION STRENGTHS:

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│         Band 0  1    2    3    4    5    6                     │
│  Band 0   -    1.0  0.5  0.2  0.1  0.2  0.5   (wormhole to 6) │
│  Band 1  1.0   -    1.0  0.5  0.2  0.5  0.2   (wormhole to 5) │
│  Band 2  0.5  1.0   -    1.0  0.5  0.2  0.1                    │
│  Band 3  0.2  0.5  1.0   -    1.0  0.5  0.2   (self-attention)│
│  Band 4  0.1  0.2  0.5  1.0   -    1.0  0.5   (wormhole to 2) │
│  Band 5  0.2  0.5  0.2  0.5  1.0   -    1.0   (wormhole to 1) │
│  Band 6  0.5  0.2  0.1  0.2  0.5  1.0   -     (wormhole to 0) │
│                                                                 │
│  1.0 = Adjacent band (dense connection)                        │
│  0.5 = Wormhole target (sparse, direct)                        │
│  0.2 = Secondary routing (through intermediaries)              │
│  0.1 = Weak (rarely needed)                                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

THE SYMMETRY: Wormholes connect opposite ends of the spectrum
• DC ↔ High: Existence grounds detail, detail confirms existence
• Low ↔ MidHigh: Structure guides texture, texture refines structure
```

### 8.3 Why Not Just 6?

```
THE CASE FOR 7 BANDS:

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  6 BANDS (log₂(64)):                                           │
│  • Sufficient for 64×64 images                                 │
│  • Minimal small-world structure                               │
│  • Works but may be tight at boundaries                        │
│                                                                 │
│  7 BANDS (log₂(128) or log₂(64) + 1):                         │
│  • Handles up to 128×128 naturally                             │
│  • Extra band for "limbo" / staging area                       │
│  • Cleaner separation of mid-range frequencies                 │
│  • Matches 6 degrees + self (7 hops including origin)          │
│                                                                 │
│  THE DECISION:                                                  │
│                                                                 │
│  • For 64×64: 6 bands is sufficient                            │
│  • For 128×128 or variable size: 7 bands preferred             │
│  • For production systems: 7 bands with band 3-4 merged if     │
│    computational budget is tight                                │
│                                                                 │
│  DEFAULT RECOMMENDATION: 7 bands for flexibility               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 9. Implementation Considerations

### 9.1 Band Decomposition

```python
class SpectralBandDecomposition(nn.Module):
    """
    Decompose input into 7 frequency bands using octave-spaced filters.
    """
    
    def __init__(self, image_size: int = 64, num_bands: int = 7):
        super().__init__()
        self.image_size = image_size
        self.num_bands = num_bands
        
        # Frequency boundaries (normalized to Nyquist)
        self.band_edges = self._compute_band_edges()
        
        # Per-band channel counts (more at mid, less at extremes)
        self.channels_per_band = [4, 8, 16, 32, 32, 16, 8]
        
        # Learning rates per band (slow at low, fast at high)
        self.learning_rates = [0.00001, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03]
        
    def _compute_band_edges(self) -> list:
        """Octave-spaced frequency boundaries."""
        # Band 0: DC only
        # Bands 1-6: Octave spacing from 1/N to 0.5
        edges = [0]  # DC lower bound
        for i in range(self.num_bands - 1):
            edges.append(1.0 / (2 ** (self.num_bands - 1 - i)))
        edges.append(0.5)  # Nyquist
        return edges
    
    def forward(self, x: torch.Tensor) -> dict:
        """
        Decompose input into frequency bands.
        
        Returns:
            Dict mapping band index to band content
        """
        # FFT
        x_fft = torch.fft.fft2(x)
        x_fft_shifted = torch.fft.fftshift(x_fft)
        
        bands = {}
        for band_idx in range(self.num_bands):
            # Create frequency mask for this band
            mask = self._create_band_mask(band_idx)
            
            # Apply mask and inverse FFT
            band_fft = x_fft_shifted * mask
            band_spatial = torch.fft.ifft2(torch.fft.ifftshift(band_fft))
            
            bands[band_idx] = band_spatial.real
            
        return bands
```

### 9.2 Per-Band Attention

```python
class PerBandAttention(nn.Module):
    """
    Attention mechanism with band-specific parameters.
    """
    
    def __init__(self, band_idx: int, channels: int, history_len: int = 8):
        super().__init__()
        self.band_idx = band_idx
        self.channels = channels
        
        # Mode: reactive for high bands, knowledge for low bands
        self.mode = self._determine_mode()
        
        if self.mode in ['knowledge', 'hybrid']:
            # Full attention for knowledge-informed bands
            self.query = nn.Linear(channels, channels)
            self.key = nn.Linear(channels, channels)
            self.value = nn.Linear(channels, channels)
            self.history = nn.Parameter(torch.zeros(history_len, channels))
            
        if self.mode in ['reactive', 'hybrid']:
            # Threshold-based gating for reactive bands
            self.threshold = nn.Parameter(torch.tensor(0.1))
            self.gate = nn.Sequential(
                nn.Linear(channels, channels // 4),
                nn.ReLU(),
                nn.Linear(channels // 4, 1),
                nn.Sigmoid()
            )
    
    def _determine_mode(self) -> str:
        """Assign mode based on band position."""
        if self.band_idx <= 1:
            return 'knowledge'
        elif self.band_idx >= 5:
            return 'reactive'
        else:
            return 'hybrid'
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply band-appropriate attention."""
        
        if self.mode == 'reactive':
            # Pure energy-based gating
            energy = x.norm(dim=-1, keepdim=True)
            gate = (energy > self.threshold).float()
            return x * gate
            
        elif self.mode == 'knowledge':
            # Full manifold-based attention
            q = self.query(x)
            k = self.key(self.history)
            v = self.value(self.history)
            
            scores = torch.matmul(q, k.transpose(-2, -1)) / (self.channels ** 0.5)
            weights = torch.softmax(scores, dim=-1)
            return torch.matmul(weights, v)
            
        else:  # hybrid
            # Both modes active
            # Reactive gating
            gate = self.gate(x)
            
            # Knowledge attention
            q = self.query(x)
            k = self.key(self.history)
            v = self.value(self.history)
            
            scores = torch.matmul(q, k.transpose(-2, -1)) / (self.channels ** 0.5)
            weights = torch.softmax(scores, dim=-1)
            attended = torch.matmul(weights, v)
            
            # Combine: gate modulates attention output
            return attended * gate
```

### 9.3 Wormhole Connections

```python
class WormholeAttention(nn.Module):
    """
    Cross-band attention for non-local spectral routing.
    """
    
    def __init__(self, source_band: int, target_band: int, channels: int):
        super().__init__()
        self.source_band = source_band
        self.target_band = target_band
        
        # Wormholes connect distant bands
        self.distance = abs(target_band - source_band)
        
        # Projection layers
        self.query_proj = nn.Linear(channels, channels)
        self.key_proj = nn.Linear(channels, channels)
        self.value_proj = nn.Linear(channels, channels)
        
        # Sparsity: more distant = sparser connection
        self.top_k = max(1, 8 // self.distance)
        
    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Route information from source band to target band.
        
        Args:
            source: Features from source frequency band
            target: Features from target frequency band
            
        Returns:
            Updated target features with wormhole information
        """
        q = self.query_proj(target)
        k = self.key_proj(source)
        v = self.value_proj(source)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / (q.size(-1) ** 0.5)
        
        # Sparse top-k selection (wormholes are sparse)
        top_scores, top_indices = scores.topk(self.top_k, dim=-1)
        
        # Gather selected values
        sparse_weights = torch.softmax(top_scores, dim=-1)
        selected_values = torch.gather(
            v.unsqueeze(1).expand(-1, q.size(1), -1, -1),
            2,
            top_indices.unsqueeze(-1).expand(-1, -1, -1, v.size(-1))
        )
        
        # Weighted sum
        wormhole_info = (sparse_weights.unsqueeze(-1) * selected_values).sum(dim=2)
        
        return target + wormhole_info  # Residual connection
```

---

## 10. Summary

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│     S P E C T R A L   A T T E N T I O N                        │
│     SIX DEGREES OF SEPARATION IN FREQUENCY SPACE               │
│                                                                 │
│  ═══════════════════════════════════════════════════════════   │
│                                                                 │
│  THE CORE INSIGHT:                                              │
│                                                                 │
│  Network theory, information theory, neuroscience, and         │
│  signal processing all converge on the same number:            │
│                                                                 │
│        OPTIMAL BANDS ≈ log₂(N) ≈ 6-7                          │
│                                                                 │
│  This is the "six degrees of separation" in frequency space.  │
│                                                                 │
│  ═══════════════════════════════════════════════════════════   │
│                                                                 │
│  THE ARCHITECTURE:                                              │
│                                                                 │
│  • 7 frequency bands (DC + 6 octaves)                          │
│  • Within-band temporal attention                               │
│  • Cross-band wormhole attention                                │
│  • Differential learning rates (slow DC, fast high-freq)       │
│  • Mode gradient (reactive high, knowledge low)                │
│                                                                 │
│  ═══════════════════════════════════════════════════════════   │
│                                                                 │
│  THE SMALL-WORLD GUARANTEE:                                     │
│                                                                 │
│  Any spectral position can reach any other in ≤ 6 hops.       │
│  Wormholes provide shortcuts between distant frequency bands.  │
│  This is optimal routing in hierarchical space.                │
│                                                                 │
│  ═══════════════════════════════════════════════════════════   │
│                                                                 │
│  THE OLD LADY CONNECTION:                                       │
│                                                                 │
│  • High bands: Process details, quick turnover (blank pages)  │
│  • Low bands: Accumulate wisdom, slow change (atomic truths)  │
│  • Culling cascade: Details → Features → Structure → Essence  │
│  • The notebook is spectral; wisdom flows to DC.               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## References

- Milgram, S. (1967). "The Small World Problem." *Psychology Today*.
- Watts, D. J., & Strogatz, S. H. (1998). "Collective dynamics of small-world networks." *Nature*.
- Barabási, A.-L., & Albert, R. (1999). "Emergence of scaling in random networks." *Science*.
- SPECTRAL_BELIEF_STORAGE_RETRIEVAL.md — Optimal band counts from multiple fields
- THE_ATOMIC_STRUCTURE_OF_INFORMATION.md — Action Quanta across frequency bands
- KNOWLEDGE_AND_REACTIVITY.md — Mode assignment framework
- THE_OLD_LADY_AND_THE_TIGER.md — Culling and compression across scales

---

*Oscar Goldman — Shogu Research Group @ Datamutant.ai subsidiary of 温心重工業*

*"Six degrees separate any two people. Six bands separate any two frequencies. The small world is spectral."*

