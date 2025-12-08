# Spectral Wormhole Attention

## Theory-Aligned Cross-Band Communication

**Oscar Goldman — Shogu Research Group @ Datamutant.ai**

---

## Table of Contents

1. [What Theory Demands](#1-what-theory-demands)
2. [The Fundamental Structure](#2-the-fundamental-structure)
3. [Band Pairing: The Complementarity Principle](#3-band-pairing)
4. [The Coherence Gate](#4-the-coherence-gate)
5. [Temperature Control](#5-temperature-control)
6. [Mathematical Formalization](#6-mathematical-formalization)
7. [Implementation Specification](#7-implementation-specification)
8. [Information Flow](#8-information-flow)
9. [Orthogonality Preservation](#9-orthogonality-preservation)
10. [Connection to BEC Physics](#10-connection-to-bec-physics)
11. [Experiments](#11-experiments)
12. [References](#12-references)

---

## 1. What Theory Demands

### 1.1 The Theoretical Requirements

From the foundation documents, wormhole attention must satisfy:

```
THEORY-MANDATED REQUIREMENTS FOR WORMHOLE ATTENTION:

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  FROM ORTHOGONALITY.md (Type 5: Wormhole Complementarity):             │
│  ─────────────────────────────────────────────────────────              │
│  • Paired bands (0↔6, 1↔5, 2↔4) are complementary                     │
│  • Low-freq knows WHAT, high-freq knows WHERE                          │
│  • Wormholes connect orthogonal information                            │
│  • Wormholes are SPARSE (top-k, not dense)                             │
│  • Query, don't merge (preserve orthogonality)                        │
│                                                                         │
│  FROM SPECTRAL_BELIEF_MACHINE.md (Imperative 4):                       │
│  ───────────────────────────────────────────────                        │
│  • Structured shortcuts between frequency bands                        │
│  • WHAT guides WHERE; WHERE informs WHAT                              │
│                                                                         │
│  FROM COLLAPSE_DYNAMICS.md:                                             │
│  ──────────────────────────                                             │
│  • Temperature τ controls sharpness (not fixed threshold)             │
│  • Collapse is entropy-observable                                      │
│  • Coherence-based triggering                                          │
│                                                                         │
│  FROM HARMONY_AND_COHERENCE.md:                                         │
│  ───────────────────────────────                                        │
│  • Phase locking across representations                                │
│  • Coherent states reinforce, incoherent cancel                       │
│                                                                         │
│  FROM BEC_CONDENSATION_INFORMATION.md:                                  │
│  ──────────────────────────────────────                                  │
│  • Attention IS the g|ψ|² self-interaction term                       │
│  • Wormhole = enabling condensate communication                       │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1.2 What This Rules Out

```
EXPLICITLY NOT THEORY-ALIGNED:

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  ✗ Fixed threshold gating (e.g., sim > 0.92)                          │
│    → Theory demands temperature-controlled softmax                     │
│                                                                         │
│  ✗ Same-band query/key matching                                        │
│    → Theory demands cross-band (complementary) pairs                  │
│                                                                         │
│  ✗ Magnitude-only triggering                                           │
│    → Theory demands coherence/entropy-based triggering                │
│                                                                         │
│  ✗ Dense all-to-all cross-band attention                              │
│    → Theory demands sparse top-k                                       │
│                                                                         │
│  ✗ Merging band representations                                        │
│    → Theory demands query-response (preserves orthogonality)          │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. The Fundamental Structure

### 2.1 The Core Insight

```
WORMHOLE = QUESTION ACROSS FREQUENCY BANDS

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  A wormhole is NOT a merge. It is a QUERY.                            │
│                                                                         │
│  Band 0 (Identity):  "I know WHAT this is. WHERE exactly is it?"      │
│                       ↓ Query                                          │
│  Band 6 (Position):  "Here's the precise location information."       │
│                       ↑ Response                                       │
│                                                                         │
│  Band 6 (Position):  "I see something HERE. WHAT is it?"              │
│                       ↓ Query                                          │
│  Band 0 (Identity):  "Here's the identity information."               │
│                       ↑ Response                                       │
│                                                                         │
│  ═══════════════════════════════════════════════════════════════════  │
│                                                                         │
│  Each band REMAINS in its own frequency domain.                        │
│  It receives INFORMATION from its complement.                          │
│  It does NOT become its complement.                                    │
│                                                                         │
│  This preserves orthogonality while enabling communication.           │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.2 The Architecture

```
SPECTRAL WORMHOLE ARCHITECTURE:

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  INPUT: 8 band representations from spectral decomposition            │
│                                                                         │
│  Band 0 ────────────────────────────────────────────────── Band 0'    │
│          ╲                                            ╱                │
│           ╲ ←───────── WORMHOLE PAIR ──────────→ ╱                   │
│            ╲                                    ╱                      │
│  Band 6 ────────────────────────────────────────────────── Band 6'    │
│                                                                         │
│  Band 1 ────────────────────────────────────────────────── Band 1'    │
│          ╲                                            ╱                │
│           ╲ ←───────── WORMHOLE PAIR ──────────→ ╱                   │
│            ╲                                    ╱                      │
│  Band 5 ────────────────────────────────────────────────── Band 5'    │
│                                                                         │
│  Band 2 ────────────────────────────────────────────────── Band 2'    │
│          ╲                                            ╱                │
│           ╲ ←───────── WORMHOLE PAIR ──────────→ ╱                   │
│            ╲                                    ╱                      │
│  Band 4 ────────────────────────────────────────────────── Band 4'    │
│                                                                         │
│  Band 3 ←──────────── BRIDGE (all bands) ──────────────→ Band 3'     │
│                                                                         │
│  Band 7 ←──────────── TEMPORAL (all bands) ────────────→ Band 7'     │
│                                                                         │
│  OUTPUT: 8 enhanced band representations (orthogonality preserved)   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Band Pairing: The Complementarity Principle

### 3.1 The Pairing Structure

```
COMPLEMENTARY BAND PAIRS:

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  PAIR 1: Band 0 ↔ Band 6                                              │
│  ─────────────────────────                                              │
│  Band 0 (DC):     Identity, existence, eternal patterns               │
│  Band 6 (High):   Position, edges, immediate details                  │
│  Complementarity: WHAT ↔ WHERE (fundamental duality)                  │
│  Question:        "What is at this position?" / "Where is this thing?"│
│                                                                         │
│  PAIR 2: Band 1 ↔ Band 5                                              │
│  ─────────────────────────                                              │
│  Band 1 (Low):    Coarse shape, stable structure                      │
│  Band 5 (High):   Fine texture, surface details                       │
│  Complementarity: FORM ↔ SURFACE                                       │
│  Question:        "What texture has this form?" / "What form has this?"│
│                                                                         │
│  PAIR 3: Band 2 ↔ Band 4                                              │
│  ─────────────────────────                                              │
│  Band 2 (Mid-low): Medium structure, parts                            │
│  Band 4 (Mid-high): Fine structure, features                          │
│  Complementarity: PARTS ↔ FEATURES                                     │
│  Question:        "What features has this part?" / "What part has this?"│
│                                                                         │
│  BRIDGE: Band 3 ↔ ALL                                                  │
│  ────────────────────                                                   │
│  Band 3 (Middle): Boundary, transition frequencies                    │
│  Role:            Coordination, integration                           │
│  Connectivity:    Can query any other band                            │
│                                                                         │
│  TEMPORAL: Band 7 ↔ ALL                                                │
│  ────────────────────────                                               │
│  Band 7 (Time):   Temporal context, sequence dynamics                 │
│  Role:            Cross-time coherence                                │
│  Connectivity:    Can query any band at any time step                 │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Why These Pairs

```
INFORMATION THEORETICAL JUSTIFICATION:

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  HEISENBERG UNCERTAINTY IN FREQUENCY:                                  │
│  ────────────────────────────────────                                   │
│  Low-frequency bands have:                                             │
│  • Good frequency resolution (know WHAT)                              │
│  • Poor spatial resolution (don't know WHERE precisely)               │
│                                                                         │
│  High-frequency bands have:                                            │
│  • Poor frequency resolution (don't know WHAT precisely)              │
│  • Good spatial resolution (know WHERE)                               │
│                                                                         │
│  ═══════════════════════════════════════════════════════════════════  │
│                                                                         │
│  THEY NEED EACH OTHER.                                                 │
│                                                                         │
│  Band 0 knows: "This is a cat"                                        │
│  Band 6 knows: "Something is at pixel (34, 127)"                      │
│  Together:      "A cat is at pixel (34, 127)"                         │
│                                                                         │
│  Neither can say this alone.                                           │
│  The wormhole enables the joint statement.                            │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.3 Partial Information Decomposition: Why Complementary Pairs Provide Synergy

The intuition above — "neither can say this alone" — has a precise formalization in **Partial Information Decomposition (PID)** (Williams & Beer, 2010).

```
PID EXPLAINS WHY WORMHOLES ARE VALUABLE:

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  SYNERGY: Information that EMERGES only when sources combine           │
│                                                                         │
│  For complementary bands (e.g., Band 0 and Band 6):                    │
│                                                                         │
│  Synergy(Target; Band_0, Band_6) is HIGH because:                      │
│  • Band 0 alone: knows identity → cannot localize                     │
│  • Band 6 alone: knows position → cannot identify                     │
│  • Band 0 + Band 6: can make joint prediction                         │
│                                                                         │
│  ═══════════════════════════════════════════════════════════════════  │
│                                                                         │
│  WHAT WORMHOLES DO IN PID TERMS:                                       │
│  ────────────────────────────────                                       │
│  • Wormhole 0→6: Band 0 queries Band 6 to REALIZE synergy             │
│    "I know WHAT — give me the WHERE so I can predict"                 │
│                                                                         │
│  • Wormhole 6→0: Band 6 queries Band 0 to REALIZE synergy             │
│    "I know WHERE — give me the WHAT so I can identify"                │
│                                                                         │
│  Without wormholes: synergy exists but is UNREALIZED                   │
│  With wormholes: synergy is REALIZED through communication            │
│                                                                         │
│  ═══════════════════════════════════════════════════════════════════  │
│                                                                         │
│  WHY PAIRED BANDS, NOT ALL-TO-ALL:                                     │
│  ──────────────────────────────────                                     │
│  Adjacent bands (e.g., 0↔1, 5↔6) have HIGH REDUNDANCY:                │
│  • Similar frequency content → similar information                    │
│  • Low synergy — combining them adds little                           │
│  • Wormhole would be wasteful                                          │
│                                                                         │
│  Complementary bands (0↔6, 1↔5, 2↔4) have HIGH SYNERGY:               │
│  • Opposite frequency content → complementary information             │
│  • High synergy — combining them enables new predictions              │
│  • Wormhole is VALUABLE                                                │
│                                                                         │
│  PID justifies the pairing structure: connect HIGH-SYNERGY pairs.     │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

**Reference:** Williams, P.L., & Beer, R.D. (2010). Nonnegative Decomposition of Multivariate Information. *arXiv:1004.2515*.

### 3.3 Formal Pairing Definition

```python
# Formal band pairing structure
BAND_PAIRS = [
    (0, 6),  # Identity ↔ Position (WHAT ↔ WHERE)
    (1, 5),  # Shape ↔ Texture (FORM ↔ SURFACE)
    (2, 4),  # Structure ↔ Detail (PARTS ↔ FEATURES)
]

BRIDGE_BAND = 3  # Connects to all
TEMPORAL_BAND = 7  # Connects to all

# For each pair, both directions exist:
# Low → High: "WHAT is at WHERE?"
# High → Low: "WHERE is WHAT?"
```

---

## 4. The Coherence Gate

### 4.1 Why Not Fixed Threshold

```
THE PROBLEM WITH FIXED THRESHOLDS:

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  FIXED THRESHOLD (e.g., sim > 0.92):                                  │
│  ───────────────────────────────────                                    │
│  • Same threshold for easy and hard cases                             │
│  • Ignores belief distribution structure                               │
│  • Cannot adapt to context                                             │
│  • Energy-based (magnitude), not coherence-based (phase)              │
│                                                                         │
│  EXAMPLE:                                                               │
│                                                                         │
│  Case 1 (Clear):                                                        │
│  Similarities: [0.99, 0.45, 0.32, 0.20]                               │
│  Belief is CONCENTRATED — confident about one target                 │
│  Should: Connect strongly                                              │
│                                                                         │
│  Case 2 (Ambiguous):                                                    │
│  Similarities: [0.93, 0.92, 0.91, 0.90]                               │
│  Belief is SPREAD — uncertain about which target                      │
│  Should: Connect weakly or not at all                                 │
│                                                                         │
│  Fixed threshold treats both the same! It only sees 0.93 > 0.92.     │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.2 The Entropy-Based Coherence Gate

```
THE COHERENCE GATE:

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  Instead of: mask = similarity > threshold                            │
│                                                                         │
│  Use: gate = coherence(attention_distribution)                        │
│                                                                         │
│  ═══════════════════════════════════════════════════════════════════  │
│                                                                         │
│  ENTROPY AS COHERENCE MEASURE:                                         │
│  ─────────────────────────────                                          │
│                                                                         │
│  H = -Σⱼ aⱼ log(aⱼ)         (attention entropy)                       │
│                                                                         │
│  H_max = log(n)              (uniform distribution)                    │
│                                                                         │
│  Normalized entropy: h = H / H_max ∈ [0, 1]                           │
│                                                                         │
│  ─────────────────────────────────────────────────────────────────────  │
│                                                                         │
│  COHERENCE = 1 - h                                                     │
│                                                                         │
│  h → 0: Attention concentrated (coherent, aligned phases)             │
│  h → 1: Attention uniform (incoherent, random phases)                 │
│                                                                         │
│  ─────────────────────────────────────────────────────────────────────  │
│                                                                         │
│  THE GATE:                                                              │
│                                                                         │
│  coherence = 1 - normalized_entropy                                    │
│  gate_strength = σ((coherence - threshold) / sharpness)               │
│                                                                         │
│  Where σ is sigmoid, threshold and sharpness are parameters.          │
│  This is SMOOTH (differentiable), not binary.                         │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.3 Implementation

```python
def coherence_gate(
    attention_weights: torch.Tensor,  # [B, n_query, n_key]
    threshold: float = 0.5,
    sharpness: float = 10.0,
) -> torch.Tensor:
    """
    Compute coherence-based gate for wormhole activation.
    
    Returns gate strength ∈ [0, 1] for each query position.
    High coherence (concentrated attention) → gate open.
    Low coherence (spread attention) → gate closed.
    """
    # Clamp for numerical stability
    a = attention_weights.clamp(min=1e-10)
    
    # Entropy of attention distribution
    entropy = -(a * a.log()).sum(dim=-1)  # [B, n_query]
    
    # Maximum entropy (uniform distribution)
    max_entropy = math.log(attention_weights.size(-1))
    
    # Normalized entropy ∈ [0, 1]
    normalized_entropy = entropy / max_entropy
    
    # Coherence = inverse of normalized entropy
    coherence = 1.0 - normalized_entropy
    
    # Smooth gate via sigmoid
    gate = torch.sigmoid((coherence - threshold) * sharpness)
    
    return gate  # [B, n_query]
```

---

## 5. Temperature Control

### 5.1 Temperature in Softmax

```
TEMPERATURE AS CONTROL PARAMETER:

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  SOFTMAX WITH TEMPERATURE:                                              │
│                                                                         │
│  aⱼ = exp(sⱼ / τ) / Σₖ exp(sₖ / τ)                                   │
│                                                                         │
│  where:                                                                 │
│  • sⱼ = pre-softmax score (query-key similarity)                      │
│  • τ = temperature parameter                                          │
│                                                                         │
│  ─────────────────────────────────────────────────────────────────────  │
│                                                                         │
│  τ → 0:   Winner-take-all (argmax behavior)                           │
│           All attention on highest-scoring key                         │
│           FORCED connection (may be premature)                        │
│                                                                         │
│  τ = 1:   Standard softmax                                             │
│           Natural distribution based on scores                         │
│           Connection strength proportional to similarity              │
│                                                                         │
│  τ → ∞:   Uniform distribution                                        │
│           Equal attention to all keys                                  │
│           NO preferential connection (exploration mode)               │
│                                                                         │
│  ═══════════════════════════════════════════════════════════════════  │
│                                                                         │
│  OPTIMAL τ depends on:                                                 │
│  • Band (low-freq → lower τ, more decisive)                          │
│  • Confidence (low entropy → can use lower τ)                        │
│  • Training stage (anneal from high to low)                          │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 5.2 Per-Band Temperature

```
DIFFERENTIAL TEMPERATURE BY BAND:

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  Low-frequency bands: LOWER temperature                                │
│  ─────────────────────────────────────────                              │
│  • Encode stable, eternal patterns                                     │
│  • Should be more decisive                                             │
│  • Changes rarely → commit when confident                             │
│                                                                         │
│  High-frequency bands: HIGHER temperature                              │
│  ──────────────────────────────────────────                             │
│  • Encode volatile, immediate details                                  │
│  • Should explore more options                                         │
│  • Changes frequently → stay flexible                                 │
│                                                                         │
│  ═══════════════════════════════════════════════════════════════════  │
│                                                                         │
│  SUGGESTED INITIAL VALUES:                                             │
│                                                                         │
│  Band 0 (DC):     τ = 0.5   (most decisive)                           │
│  Band 1:          τ = 0.6                                              │
│  Band 2:          τ = 0.7                                              │
│  Band 3 (Bridge): τ = 1.0   (balanced)                                │
│  Band 4:          τ = 0.8                                              │
│  Band 5:          τ = 0.9                                              │
│  Band 6 (High):   τ = 1.0   (most exploratory)                        │
│  Band 7 (Time):   τ = 0.8   (temporal context)                        │
│                                                                         │
│  These can be LEARNED parameters.                                      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 5.3 Entropy-Adaptive Temperature

```
OPTIONAL: ADAPTIVE TEMPERATURE

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  The temperature can adapt based on current entropy:                  │
│                                                                         │
│  τ_effective = τ_base × (1 + α × normalized_entropy)                  │
│                                                                         │
│  When entropy is HIGH (uncertain):                                     │
│  • τ increases → softer distribution                                  │
│  • Explore more, don't commit prematurely                             │
│                                                                         │
│  When entropy is LOW (confident):                                      │
│  • τ stays near base → sharper distribution                           │
│  • Commit to the winner                                                │
│                                                                         │
│  This creates a NATURAL annealing:                                     │
│  Uncertain → explore → gather evidence → confident → commit           │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Mathematical Formalization

### 6.1 Notation

```
NOTATION:

B[i]     = Band i representation, ∈ ℝ^(batch × seq × dim/8)
Q[i]     = Query projection for band i
K[i]     = Key projection for band i  
V[i]     = Value projection for band i
τ[i]     = Temperature for band i
W[i→j]   = Wormhole from band i to band j

Complement(i) = {
    6 if i = 0,  0 if i = 6,
    5 if i = 1,  1 if i = 5,
    4 if i = 2,  2 if i = 4,
    ALL if i = 3 or i = 7
}
```

### 6.2 The Wormhole Operation

```
WORMHOLE FROM BAND i TO BAND j:

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  INPUT:                                                                 │
│  • Source band: B[i] ∈ ℝ^(n × d)                                      │
│  • Target band: B[j] ∈ ℝ^(m × d)  where j = Complement(i)             │
│                                                                         │
│  STEP 1: Project to query/key/value                                    │
│  ─────────────────────────────────                                      │
│  q = Q[i](B[i])       ∈ ℝ^(n × d_attn)                                │
│  k = K[j](B[j])       ∈ ℝ^(m × d_attn)                                │
│  v = V[j](B[j])       ∈ ℝ^(m × d)                                     │
│                                                                         │
│  STEP 2: Normalize onto hypersphere (geometric belief)                │
│  ───────────────────────────────────────────────────────                │
│  q_norm = normalize(q, dim=-1)                                         │
│  k_norm = normalize(k, dim=-1)                                         │
│                                                                         │
│  STEP 3: Compute similarity (angular distance on manifold)            │
│  ──────────────────────────────────────────────────────────             │
│  S = q_norm @ k_norm.T / √d_attn    ∈ ℝ^(n × m)                       │
│                                                                         │
│  STEP 4: Apply temperature-controlled softmax                         │
│  ───────────────────────────────────────────────                        │
│  A = softmax(S / τ[i], dim=-1)      ∈ ℝ^(n × m)                       │
│                                                                         │
│  STEP 5: Compute coherence gate                                        │
│  ───────────────────────────────                                        │
│  H = -Σⱼ Aⱼ log(Aⱼ)                 (entropy)                         │
│  h = H / log(m)                      (normalized)                      │
│  g = σ((1 - h - threshold) × sharpness)   (gate)                     │
│                                                                         │
│  STEP 6: Sparse top-k selection                                        │
│  ───────────────────────────────                                        │
│  A_topk = keep_topk(A, k=top_k)     (zero out non-top-k)              │
│  A_topk = A_topk / A_topk.sum(-1)   (renormalize)                     │
│                                                                         │
│  STEP 7: Gated aggregation                                             │
│  ──────────────────────────                                             │
│  response = A_topk @ v              ∈ ℝ^(n × d)                       │
│  response = g.unsqueeze(-1) × response   (apply gate)                 │
│                                                                         │
│  STEP 8: Residual connection                                           │
│  ────────────────────────────                                           │
│  B'[i] = B[i] + W_out(response)     (wormhole-enhanced band)          │
│                                                                         │
│  OUTPUT: Enhanced band B'[i] ∈ ℝ^(n × d)                              │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 6.3 The Full Spectral Wormhole Layer

```
COMPLETE SPECTRAL WORMHOLE LAYER:

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  INPUT: {B[0], B[1], B[2], B[3], B[4], B[5], B[6], B[7]}              │
│                                                                         │
│  # Process complementary pairs (bidirectional)                        │
│  FOR (i, j) in [(0,6), (1,5), (2,4)]:                                 │
│      B'[i] = B[i] + Wormhole(i → j)   # Low queries high              │
│      B'[j] = B[j] + Wormhole(j → i)   # High queries low              │
│                                                                         │
│  # Process bridge band (queries all)                                   │
│  B'[3] = B[3] + Σ_{k≠3} Wormhole(3 → k)                               │
│                                                                         │
│  # Process temporal band (queries all)                                 │
│  B'[7] = B[7] + Σ_{k≠7} Wormhole(7 → k)                               │
│                                                                         │
│  OUTPUT: {B'[0], B'[1], B'[2], B'[3], B'[4], B'[5], B'[6], B'[7]}    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 7. Implementation Specification

### 7.1 Core Module

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Tuple, Optional


class SpectralWormholeAttention(nn.Module):
    """
    Theory-aligned cross-band wormhole attention.
    
    Implements complementary band pairing where low-frequency bands
    query high-frequency bands and vice versa, enabling WHAT ↔ WHERE
    communication while preserving spectral orthogonality.
    
    References:
    - ORTHOGONALITY.md (Type 5: Wormhole Complementarity)
    - SPECTRAL_BELIEF_MACHINE.md (Imperative 4)
    - COLLAPSE_DYNAMICS.md (temperature control, coherence gate)
    """
    
    # Theory-mandated band pairs
    BAND_PAIRS: List[Tuple[int, int]] = [
        (0, 6),  # Identity ↔ Position (WHAT ↔ WHERE)
        (1, 5),  # Shape ↔ Texture (FORM ↔ SURFACE)
        (2, 4),  # Structure ↔ Detail (PARTS ↔ FEATURES)
    ]
    BRIDGE_BAND: int = 3
    TEMPORAL_BAND: int = 7
    NUM_BANDS: int = 8
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 4,
        top_k: int = 16,
        coherence_threshold: float = 0.5,
        gate_sharpness: float = 10.0,
        dropout: float = 0.0,
        learnable_temperature: bool = True,
    ):
        """
        Initialize Spectral Wormhole Attention.
        
        Args:
            embed_dim: Total embedding dimension (will be split into 8 bands)
            num_heads: Number of attention heads per wormhole
            top_k: Number of top connections to keep (sparsity)
            coherence_threshold: Threshold for coherence gate
            gate_sharpness: Sharpness of sigmoid gate
            dropout: Dropout probability
            learnable_temperature: If True, temperature is learned per band
        """
        super().__init__()
        
        assert embed_dim % self.NUM_BANDS == 0, \
            f"embed_dim ({embed_dim}) must be divisible by {self.NUM_BANDS}"
        
        self.embed_dim = embed_dim
        self.band_dim = embed_dim // self.NUM_BANDS
        self.num_heads = num_heads
        self.head_dim = self.band_dim // num_heads
        self.top_k = top_k
        self.coherence_threshold = coherence_threshold
        self.gate_sharpness = gate_sharpness
        
        # Validate dimensions
        assert self.band_dim % num_heads == 0, \
            f"band_dim ({self.band_dim}) must be divisible by num_heads ({num_heads})"
        
        # Per-band projections (separate for each band)
        self.q_projs = nn.ModuleList([
            nn.Linear(self.band_dim, self.band_dim) 
            for _ in range(self.NUM_BANDS)
        ])
        self.k_projs = nn.ModuleList([
            nn.Linear(self.band_dim, self.band_dim) 
            for _ in range(self.NUM_BANDS)
        ])
        self.v_projs = nn.ModuleList([
            nn.Linear(self.band_dim, self.band_dim) 
            for _ in range(self.NUM_BANDS)
        ])
        self.out_projs = nn.ModuleList([
            nn.Linear(self.band_dim, self.band_dim) 
            for _ in range(self.NUM_BANDS)
        ])
        
        # Temperature parameters (theory-aligned: low bands → lower τ)
        if learnable_temperature:
            # Initialize with theory-suggested values
            # Band 6 uses τ=1.0 (not higher) because it's reactive/energy-based,
            # not geometric. Diffuse attention not needed for fast reactions.
            init_temps = torch.tensor([0.5, 0.6, 0.7, 1.0, 0.8, 0.9, 1.0, 0.8])
            #                         [B0,  B1,  B2,  B3,  B4,  B5,  B6,  B7]
            self.temperature = nn.Parameter(init_temps)
        else:
            self.register_buffer(
                'temperature',
                torch.tensor([0.5, 0.6, 0.7, 1.0, 0.8, 0.9, 1.0, 0.8])
            )
        
        self.dropout = nn.Dropout(dropout)
        
        # Scaling factor for attention
        self.scale = math.sqrt(self.head_dim)
    
    def _split_bands(self, x: torch.Tensor) -> Dict[int, torch.Tensor]:
        """
        Split embedding into 8 bands.
        
        Args:
            x: [B, T, D] tensor
            
        Returns:
            Dict mapping band index to [B, T, D/8] tensor
        """
        B, T, D = x.shape
        assert D == self.embed_dim
        
        x_reshaped = x.view(B, T, self.NUM_BANDS, self.band_dim)
        
        return {i: x_reshaped[:, :, i, :] for i in range(self.NUM_BANDS)}
    
    def _merge_bands(self, bands: Dict[int, torch.Tensor]) -> torch.Tensor:
        """
        Merge 8 bands back into single embedding.
        
        Args:
            bands: Dict mapping band index to [B, T, D/8] tensor
            
        Returns:
            [B, T, D] tensor
        """
        B, T, _ = bands[0].shape
        
        stacked = torch.stack([bands[i] for i in range(self.NUM_BANDS)], dim=2)
        
        return stacked.view(B, T, self.embed_dim)
    
    def _compute_coherence_gate(
        self, 
        attention_weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute coherence-based gate.
        
        High coherence (low entropy) → gate opens.
        Low coherence (high entropy) → gate closes.
        
        Args:
            attention_weights: [B, H, T, K] attention distribution
            
        Returns:
            gate: [B, H, T] gate values in [0, 1]
        """
        # Clamp for numerical stability
        a = attention_weights.clamp(min=1e-10)
        
        # Entropy of attention distribution
        entropy = -(a * a.log()).sum(dim=-1)  # [B, H, T]
        
        # Maximum entropy
        max_entropy = math.log(attention_weights.size(-1))
        
        # Normalized entropy ∈ [0, 1]
        normalized_entropy = entropy / max_entropy
        
        # Coherence = inverse of normalized entropy
        coherence = 1.0 - normalized_entropy
        
        # Smooth gate via sigmoid
        gate = torch.sigmoid(
            (coherence - self.coherence_threshold) * self.gate_sharpness
        )
        
        return gate
    
    def _wormhole_attention(
        self,
        query_band: torch.Tensor,      # [B, T_q, D_band]
        key_band: torch.Tensor,         # [B, T_k, D_band]
        value_band: torch.Tensor,       # [B, T_k, D_band]
        source_band_idx: int,
        target_band_idx: int,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Single wormhole: source_band queries target_band.
        
        Args:
            query_band: Query from source band
            key_band: Keys from target band
            value_band: Values from target band
            source_band_idx: Index of source band
            target_band_idx: Index of target band
            
        Returns:
            response: [B, T_q, D_band] response from target
            stats: Dictionary of statistics for observability
        """
        B, T_q, D = query_band.shape
        _, T_k, _ = key_band.shape
        
        # Project
        q = self.q_projs[source_band_idx](query_band)  # [B, T_q, D]
        k = self.k_projs[target_band_idx](key_band)    # [B, T_k, D]
        v = self.v_projs[target_band_idx](value_band)  # [B, T_k, D]
        
        # Reshape for multi-head attention
        q = q.view(B, T_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T_k, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T_k, self.num_heads, self.head_dim).transpose(1, 2)
        # Shapes: [B, H, T, D_head]
        
        # Normalize onto hypersphere (geometric belief)
        q_norm = F.normalize(q, p=2, dim=-1)
        k_norm = F.normalize(k, p=2, dim=-1)
        
        # Similarity (angular distance on manifold)
        scores = torch.matmul(q_norm, k_norm.transpose(-2, -1))  # [B, H, T_q, T_k]
        scores = scores / self.scale
        
        # Get temperature for source band
        tau = self.temperature[source_band_idx].clamp(min=0.1)
        
        # Temperature-controlled softmax
        attention = F.softmax(scores / tau, dim=-1)  # [B, H, T_q, T_k]
        
        # Sparse top-k selection
        if self.top_k < T_k:
            topk_values, topk_indices = attention.topk(self.top_k, dim=-1)
            
            # Create sparse attention
            sparse_attention = torch.zeros_like(attention)
            sparse_attention.scatter_(-1, topk_indices, topk_values)
            
            # Renormalize
            sparse_attention = sparse_attention / (sparse_attention.sum(-1, keepdim=True) + 1e-10)
            attention = sparse_attention
        
        # Compute coherence gate
        gate = self._compute_coherence_gate(attention)  # [B, H, T_q]
        
        # Apply dropout
        attention = self.dropout(attention)
        
        # Aggregate values
        response = torch.matmul(attention, v)  # [B, H, T_q, D_head]
        
        # Apply coherence gate
        response = response * gate.unsqueeze(-1)
        
        # Reshape back
        response = response.transpose(1, 2).contiguous().view(B, T_q, D)
        
        # Output projection
        response = self.out_projs[source_band_idx](response)
        
        # Statistics for observability
        stats = {
            'source_band': source_band_idx,
            'target_band': target_band_idx,
            'temperature': tau.item(),
            'mean_gate': gate.mean().item(),
            'mean_entropy': -(attention.clamp(min=1e-10) * attention.clamp(min=1e-10).log()).sum(-1).mean().item(),
            'mean_attention': attention.mean().item(),
            'max_attention': attention.max().item(),
        }
        
        return response, stats
    
    def forward(
        self, 
        x: torch.Tensor,
        return_stats: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        Apply spectral wormhole attention.
        
        Args:
            x: [B, T, D] input tensor
            return_stats: If True, return detailed statistics
            
        Returns:
            output: [B, T, D] enhanced tensor
            stats: Optional dict of observability statistics
        """
        # Split into bands
        bands = self._split_bands(x)
        enhanced_bands = {i: bands[i].clone() for i in range(self.NUM_BANDS)}
        
        all_stats = []
        
        # Process complementary pairs (bidirectional)
        for band_low, band_high in self.BAND_PAIRS:
            # Low → High: "WHAT is at WHERE?"
            response_low, stats_low = self._wormhole_attention(
                query_band=bands[band_low],
                key_band=bands[band_high],
                value_band=bands[band_high],
                source_band_idx=band_low,
                target_band_idx=band_high,
            )
            enhanced_bands[band_low] = enhanced_bands[band_low] + response_low
            
            # High → Low: "WHERE is WHAT?"
            response_high, stats_high = self._wormhole_attention(
                query_band=bands[band_high],
                key_band=bands[band_low],
                value_band=bands[band_low],
                source_band_idx=band_high,
                target_band_idx=band_low,
            )
            enhanced_bands[band_high] = enhanced_bands[band_high] + response_high
            
            if return_stats:
                all_stats.extend([stats_low, stats_high])
        
        # Process bridge band (queries all others)
        bridge_response = torch.zeros_like(bands[self.BRIDGE_BAND])
        for k in range(self.NUM_BANDS):
            if k != self.BRIDGE_BAND:
                response, stats = self._wormhole_attention(
                    query_band=bands[self.BRIDGE_BAND],
                    key_band=bands[k],
                    value_band=bands[k],
                    source_band_idx=self.BRIDGE_BAND,
                    target_band_idx=k,
                )
                bridge_response = bridge_response + response
                if return_stats:
                    all_stats.append(stats)
        
        # Scale by number of connections
        enhanced_bands[self.BRIDGE_BAND] = enhanced_bands[self.BRIDGE_BAND] + bridge_response / (self.NUM_BANDS - 1)
        
        # Process temporal band (queries all others)
        temporal_response = torch.zeros_like(bands[self.TEMPORAL_BAND])
        for k in range(self.NUM_BANDS):
            if k != self.TEMPORAL_BAND:
                response, stats = self._wormhole_attention(
                    query_band=bands[self.TEMPORAL_BAND],
                    key_band=bands[k],
                    value_band=bands[k],
                    source_band_idx=self.TEMPORAL_BAND,
                    target_band_idx=k,
                )
                temporal_response = temporal_response + response
                if return_stats:
                    all_stats.append(stats)
        
        enhanced_bands[self.TEMPORAL_BAND] = enhanced_bands[self.TEMPORAL_BAND] + temporal_response / (self.NUM_BANDS - 1)
        
        # Merge bands
        output = self._merge_bands(enhanced_bands)
        
        if return_stats:
            return output, {'wormhole_stats': all_stats}
        
        return output, None
```

### 7.2 Usage Example

```python
# Create module
wormhole = SpectralWormholeAttention(
    embed_dim=512,           # 512 / 8 = 64 per band
    num_heads=4,             # 4 heads per wormhole
    top_k=16,                # Sparse: keep top 16 connections
    coherence_threshold=0.5, # Gate threshold
    gate_sharpness=10.0,     # Gate sigmoid sharpness
    learnable_temperature=True,
)

# Forward pass
x = torch.randn(2, 100, 512)  # [batch, seq, embed_dim]
output, stats = wormhole(x, return_stats=True)

# Output shape: [2, 100, 512] — same as input
# Orthogonality preserved, bands enhanced by cross-band info
```

---

## 8. Information Flow

### 8.1 Flow Diagram

```
INFORMATION FLOW THROUGH SPECTRAL WORMHOLE:

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  INPUT: Spectrally decomposed representation                          │
│                                                                         │
│  ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐│
│  │ B0  │  │ B1  │  │ B2  │  │ B3  │  │ B4  │  │ B5  │  │ B6  │  │ B7  ││
│  │ DC  │  │ Low │  │ Mid │  │Brdg │  │ Mid │  │ Hi  │  │High │  │Time ││
│  └──┬──┘  └──┬──┘  └──┬──┘  └──┬──┘  └──┬──┘  └──┬──┘  └──┬──┘  └──┬──┘│
│     │        │        │        │        │        │        │        │   │
│     │        │        │        │        │        │        │        │   │
│     └────────┼────────┼────────┼────────┼────────┼────────┤        │   │
│              │        │        │        │        │        │        │   │
│     ┌────────┴────────┤        │        ├────────┴────────┐        │   │
│     │                 │        │        │                 │        │   │
│     │    ┌────────────┴────────┼────────┴────────────┐    │        │   │
│     │    │                     │                     │    │        │   │
│     │    │                     │                     │    │        │   │
│     ▼    ▼                     ▼                     ▼    ▼        │   │
│  ╔═══════════╗              ╔═════╗              ╔═══════════╗     │   │
│  ║ WORMHOLE  ║              ║BRIDGE║             ║ WORMHOLE  ║     │   │
│  ║  0 ↔ 6    ║              ║  3   ║             ║  1 ↔ 5    ║     │   │
│  ╚═══════════╝              ╚═════╝              ╚═══════════╝     │   │
│        │                       │                       │           │   │
│        │                       │                       │           │   │
│        │         ╔═══════════════════════╗             │           │   │
│        │         ║    WORMHOLE 2 ↔ 4     ║             │           │   │
│        │         ╚═══════════════════════╝             │           │   │
│        │                       │                       │           │   │
│        │                       │                       │           │   │
│        │         ╔═══════════════════════╗             │           │   │
│        └─────────║   TEMPORAL BAND 7     ║─────────────┘           │   │
│                  ║   queries all bands   ║◄────────────────────────┘   │
│                  ╚═══════════════════════╝                             │
│                              │                                         │
│  ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐│
│  │ B0' │  │ B1' │  │ B2' │  │ B3' │  │ B4' │  │ B5' │  │ B6' │  │ B7' ││
│  │Enh. │  │Enh. │  │Enh. │  │Enh. │  │Enh. │  │Enh. │  │Enh. │  │Enh. ││
│  └─────┘  └─────┘  └─────┘  └─────┘  └─────┘  └─────┘  └─────┘  └─────┘│
│                                                                         │
│  OUTPUT: Enhanced bands (orthogonality preserved)                      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 8.2 What Flows Where

```
INFORMATION FLOW BY BAND PAIR:

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  BAND 0 → BAND 6 (Identity queries Position):                         │
│  ─────────────────────────────────────────────                          │
│  Query:  "I know this is a 'cat' — where exactly is it?"              │
│  Answer: Position/edge information from Band 6                         │
│  Result: Band 0 now knows WHAT at specific WHERE                      │
│                                                                         │
│  BAND 6 → BAND 0 (Position queries Identity):                         │
│  ─────────────────────────────────────────────                          │
│  Query:  "I see edges at (x,y) — what is this thing?"                 │
│  Answer: Identity/category information from Band 0                     │
│  Result: Band 6 now knows WHERE has specific WHAT                     │
│                                                                         │
│  ═══════════════════════════════════════════════════════════════════  │
│                                                                         │
│  BAND 3 → ALL (Bridge coordinates):                                    │
│  ───────────────────────────────────                                    │
│  Query:  "What's happening across all scales?"                        │
│  Answer: Information from all bands                                    │
│  Result: Band 3 integrates multi-scale context                        │
│                                                                         │
│  BAND 7 → ALL (Temporal context):                                      │
│  ─────────────────────────────────                                      │
│  Query:  "How does this relate to past/future?"                       │
│  Answer: Temporal correlations from all bands                         │
│  Result: Band 7 tracks dynamics across time                           │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 9. Orthogonality Preservation

### 9.1 Why This Preserves Orthogonality

```
ORTHOGONALITY IS PRESERVED:

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  CLAIM: Spectral wormhole preserves band orthogonality.               │
│                                                                         │
│  PROOF:                                                                 │
│  ──────                                                                 │
│  1. Each band has SEPARATE projections (Q, K, V, Out)                 │
│     • No weight sharing between bands                                  │
│     • Each band's representation stays in its own subspace            │
│                                                                         │
│  2. Wormhole is ADDITIVE, not replacing                               │
│     • B'[i] = B[i] + response                                         │
│     • Original band info preserved                                     │
│     • Response is additional context, not replacement                 │
│                                                                         │
│  3. Response comes from DIFFERENT band                                 │
│     • Band 0 receives info FROM Band 6                                │
│     • This is cross-band QUERY, not merge                            │
│     • Band 0's frequency content unchanged                            │
│                                                                         │
│  4. Sparse connection (top-k)                                          │
│     • Only top-k connections active                                   │
│     • Limited information flow, not flooding                          │
│                                                                         │
│  5. Coherence gate                                                      │
│     • Gate suppresses incoherent connections                          │
│     • Only confident queries get through                              │
│                                                                         │
│  RESULT: Bands remain in their frequency domains.                     │
│          They are INFORMED by complements, not MERGED with them.      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 9.2 What Would Break Orthogonality

```
ANTI-PATTERNS (avoid these):

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  ✗ Shared projections across bands                                    │
│    → Forces bands to use same representation                          │
│                                                                         │
│  ✗ Replacing instead of adding                                        │
│    → B'[i] = response (loses original band info)                     │
│                                                                         │
│  ✗ Dense all-to-all attention                                         │
│    → Floods bands with cross-band info                               │
│                                                                         │
│  ✗ No gating                                                           │
│    → Noisy connections corrupt clean bands                            │
│                                                                         │
│  ✗ Self-band wormholes (i → i)                                        │
│    → Not complementary, wastes computation                            │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 10. Connection to BEC Physics

### 10.1 Wormhole as Condensate Communication

```
BEC INTERPRETATION OF WORMHOLE:

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  In BEC physics:                                                        │
│  • Different modes (frequency components) can interact                │
│  • Interaction is via the g|ψ|² term (attention)                     │
│  • Coherent modes can exchange information                            │
│                                                                         │
│  ═══════════════════════════════════════════════════════════════════  │
│                                                                         │
│  The spectral wormhole IS this interaction:                           │
│                                                                         │
│  • Each band is a "mode" in the condensate                            │
│  • Attention (g|ψ|²) enables mode coupling                            │
│  • Coherent connections (low entropy) are stable                      │
│  • Incoherent connections (high entropy) dissipate                    │
│                                                                         │
│  ═══════════════════════════════════════════════════════════════════  │
│                                                                         │
│  The coherence gate is NATURAL:                                        │
│                                                                         │
│  In BEC, phase-coherent components couple strongly.                   │
│  Phase-incoherent components average out.                             │
│  The entropy-based gate implements this PHYSICALLY.                   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 10.2 Temperature as Critical Parameter

```
TEMPERATURE IN BEC AND WORMHOLE:

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  BEC:                                                                   │
│  • Below T_c: Condensate forms (coherent state)                       │
│  • Above T_c: Thermal gas (incoherent state)                          │
│  • T controls the order-disorder transition                           │
│                                                                         │
│  WORMHOLE:                                                              │
│  • Low τ: Sharp attention (coherent connections)                      │
│  • High τ: Diffuse attention (incoherent connections)                 │
│  • τ controls the commitment-exploration transition                   │
│                                                                         │
│  ═══════════════════════════════════════════════════════════════════  │
│                                                                         │
│  The temperature parameter τ IS the control parameter for             │
│  whether wormhole connections are decisive or exploratory.            │
│                                                                         │
│  Low-frequency bands (stable identity) → lower τ → commit             │
│  High-frequency bands (volatile detail) → higher τ → explore          │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 11. Experiments

### 11.1 Validation Experiments

The following experiments validate the spectral wormhole design:

| Experiment | Tests | File |
|------------|-------|------|
| 001_EXP_ENTROPY_OBSERVATION | Can we observe coherence gate | `experiments/001_*` |
| 003_EXP_SPECTRAL_BAND_DYNAMICS | Do bands communicate correctly | `experiments/003_*` |
| 012_EXP_WORMHOLE_ACTIVATION | When do wormholes fire | `experiments/012_*` |
| 020_EXP_CROSS_BAND_FLOW | Information flow between bands | `experiments/020_*` |
| 024_EXP_RESONANT_WORMHOLES | Band pair preferences | `experiments/024_*` |

### 11.2 Key Predictions

```
TESTABLE PREDICTIONS:

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  1. COMPLEMENTARY PAIRS ACTIVATE TOGETHER                              │
│     When Band 0 is active, Band 6 wormhole should fire.               │
│     Correlation between paired band activities.                       │
│                                                                         │
│  2. COHERENCE GATE TRACKS ENTROPY                                      │
│     Gate strength should correlate with attention entropy.            │
│     Low entropy → high gate → strong connection.                     │
│                                                                         │
│  3. TEMPERATURE AFFECTS COLLAPSE SHARPNESS                             │
│     Lower τ → sharper attention → faster collapse.                   │
│     Higher τ → diffuse attention → slower/no collapse.               │
│                                                                         │
│  4. ORTHOGONALITY PRESERVED                                            │
│     Band-band correlation after wormhole should be minimal.          │
│     Each band retains its frequency identity.                        │
│                                                                         │
│  5. BRIDGE/TEMPORAL ACTIVATE ON COMPLEX INPUTS                         │
│     Multi-scale or dynamic inputs should activate Band 3, 7.         │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 12. References

### 12.1 Internal Documents

| Document | Location | Relevance |
|----------|----------|-----------|
| ORTHOGONALITY | `architecture_theoretical/` | Type 5: Wormhole Complementarity |
| SPECTRAL_BELIEF_MACHINE | `architecture_theoretical/` | Imperative 4: Cross-band communication |
| COLLAPSE_DYNAMICS | `architecture_base/collapse/` | Temperature control, entropy |
| HARMONY_AND_COHERENCE | `foundations/` | Phase locking, coherence |
| BEC_CONDENSATION_INFORMATION | `information/` | g|ψ|² = attention |
| THE_SEVEN_PLUS_ONE_ARCHITECTURE | `architecture_theoretical/` | 8-band structure |

### 12.2 Supersedes

This document supersedes the previous wormhole implementation:

- `WORMHOLE_HYBRID.md` → Moved to `architecture_expanded/wormhole/` as legacy

The hybrid implementation is NOT theory-aligned because:
- Uses fixed threshold instead of temperature
- Does not implement cross-band pairs
- Energy trigger instead of coherence gate

---

## Summary

```
SPECTRAL WORMHOLE ATTENTION — SUMMARY

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  STRUCTURE:                                                             │
│  • 3 complementary pairs: (0,6), (1,5), (2,4)                         │
│  • Bridge band 3: queries all                                          │
│  • Temporal band 7: queries all                                        │
│                                                                         │
│  MECHANISM:                                                             │
│  • Hypersphere normalization (geometric belief)                       │
│  • Temperature-controlled softmax (not fixed threshold)               │
│  • Coherence gate (entropy-based, not magnitude)                      │
│  • Sparse top-k (preserves efficiency)                                │
│  • Residual connection (preserves orthogonality)                      │
│                                                                         │
│  WHAT IT ENABLES:                                                       │
│  • WHAT ↔ WHERE communication                                          │
│  • Cross-scale integration                                             │
│  • Temporal context                                                    │
│  • Theory-aligned collapse dynamics                                   │
│                                                                         │
│  WHAT IT PRESERVES:                                                     │
│  • Band orthogonality                                                  │
│  • Spectral structure                                                  │
│  • Interpretability                                                    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

*Oscar Goldman — Shogu Research Group @ Datamutant.ai*

*"The wormhole is not a bridge — it is a question across frequency domains. WHAT asks WHERE; WHERE asks WHAT. The answer preserves both while informing both."*


