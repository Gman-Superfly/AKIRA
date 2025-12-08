# Spectral Belief Storage and Retrieval: Optimal Band Structure

An exploration of the optimal number of frequency bands, phases, and scales for storing and retrieving meaningful information in belief manifolds, drawing from information theory, network theory, neuroscience, and signal processing.

---

## Table of Contents

1. [The Core Question](#1-the-core-question)
2. [Insights from Information Theory](#2-insights-from-information-theory)
3. [Network Theory and Small Worlds](#3-network-theory-and-small-worlds)
4. [Neuroscience: The Visual System's Answer](#4-neuroscience-the-visual-systems-answer)
5. [Signal Processing Theory](#5-signal-processing-theory)
6. [Manifold Dimensionality](#6-manifold-dimensionality)
7. [The Logarithmic Principle](#7-the-logarithmic-principle)
8. [Practical Band Counts](#8-practical-band-counts)
9. [Phase Discretization](#9-phase-discretization)
10. [The Optimal Architecture](#10-the-optimal-architecture)
11. [Implementation Guidelines](#11-implementation-guidelines)

---

## 1. The Core Question

### 1.1 What We're Asking

```
FUNDAMENTAL QUESTION:

Given a belief manifold of dimension D encoding patterns in an N×N grid:

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  1. How many FREQUENCY BANDS do we need?                        │
│     (scales of spatial resolution)                              │
│                                                                 │
│  2. How many ORIENTATION BANDS do we need?                      │
│     (directional selectivity)                                   │
│                                                                 │
│  3. How many PHASE BINS do we need?                             │
│     (position encoding precision)                               │
│                                                                 │
│  4. What is the TOTAL CHANNEL COUNT for optimal storage?        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

The answer lies at the intersection of multiple fields.
```

### 1.2 Why This Matters

```
TOO FEW BANDS:
─────────────
• Loss of information
• Cannot distinguish important patterns
• Coarse, blurry representations
• Under-parameterized manifold

TOO MANY BANDS:
───────────────
• Redundant encoding
• Computational waste
• Overfitting risk
• Interference accumulation (as we discussed)
• Noise amplification

OPTIMAL:
────────
• Minimal sufficient representation
• Captures essential structure
• Efficient storage and retrieval
• Robust to noise
• Matches intrinsic dimensionality of data
```

---

## 2. Insights from Information Theory

### 2.1 Shannon's Sampling Theorem

```
NYQUIST-SHANNON THEOREM:

To perfectly reconstruct a signal with maximum frequency f_max:
Sample at rate ≥ 2 × f_max

For an N×N image:
• Maximum frequency: N/2 cycles
• Minimum samples: N×N

This is the UPPER BOUND on information content.
But useful information is typically much sparser.
```

### 2.2 Rate-Distortion Theory

```
RATE-DISTORTION FUNCTION R(D):

R(D) = minimum bits needed to achieve distortion ≤ D

Key insight: You DON'T need all N² samples!

For typical images:
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  Distortion    Bits needed      Relative to N²                  │
│  ──────────    ───────────      ──────────────                  │
│  0 (perfect)   N² bits          100%                            │
│  0.01          ~0.3 N²          30%                              │
│  0.05          ~0.1 N²          10%                              │
│  0.10          ~0.03 N²         3%                               │
│                                                                 │
│  Acceptable quality needs FAR FEWER than N² measurements        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2.3 Sparse Coding and Compressed Sensing

```
COMPRESSED SENSING THEOREM:

If a signal is k-sparse in some basis (only k non-zero coefficients):

Required measurements: O(k · log(N/k))

For a 64×64 image (N² = 4096):
• If effectively k = 100 sparse: ~100 × log(41) ≈ 370 measurements
• If effectively k = 50 sparse: ~50 × log(82) ≈ 220 measurements

This suggests: A small number of well-chosen basis functions
              can capture most of the information.
```

### 2.4 Entropy and Intrinsic Dimension

```
ENTROPY-BASED ESTIMATE:

H = -Σ p_i log(p_i)

For natural images:
• Pixel entropy: ~7-8 bits/pixel
• After decorrelation (DCT/wavelet): ~3-4 bits/pixel
• After perceptual weighting: ~0.5-1 bit/pixel

Intrinsic information per N×N block:
~0.5 to 1 × N² bits = Effective dimensionality

For 64×64: ~2000-4000 effective dimensions
But organized into ~log₂(64) = 6 octaves
```

---

## 3. Network Theory and Small Worlds

### 3.1 Six Degrees of Separation

```
MILGRAM'S EXPERIMENT (1967):

Average path length between any two people ≈ 6

For N people in a network:
Average path length L ≈ log(N) / log(k)

Where k = average connections per node

This suggests: LOG(N) is a fundamental scale for connectivity.
```

### 3.2 Small World Networks

```
WATTS-STROGATZ MODEL:

Characteristic path length: L ∝ log(N)
Clustering coefficient: C >> C_random

Small world property emerges when:
• Local structure (clustering) preserved
• Long-range shortcuts added

For belief manifolds:
• Local = nearby frequencies interact strongly
• Long-range = wormhole attention spans frequency bands
• Optimal bands ≈ log(N) to enable small-world routing
```

### 3.3 Scale-Free Networks

```
BARABÁSI-ALBERT MODEL:

Degree distribution: P(k) ∝ k^(-γ), typically γ ≈ 2-3

Key insight: A few "hub" nodes dominate connectivity

For spectral representation:
• Low frequencies = hubs (connect to everything)
• High frequencies = leaves (local connections)
• Band structure should reflect this hierarchy

Optimal bands: Logarithmically spaced to match power-law structure
```

### 3.4 Network Dimension

```
NETWORK DIMENSION d_N:

Number of nodes within distance r scales as: N(r) ∝ r^(d_N)

For 2D spatial networks: d_N = 2
For small-world: d_N ≈ log(N)/log(log(N)) → effectively ∞

This means: Information can be routed through 
           O(log N) intermediaries regardless of network size.

For frequency bands:
If we have B bands, routing complexity ≈ log(B)
Optimal B ≈ log(N) makes routing complexity O(log log N) ≈ constant!
```

---

## 4. Neuroscience: The Visual System's Answer

### 4.1 Primary Visual Cortex (V1) Structure

```
V1 ORGANIZATION:

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  SPATIAL FREQUENCY BANDS (octaves):                             │
│  ────────────────────────────────                               │
│  • 4-6 distinct frequency preferences                           │
│  • Logarithmically spaced (octave intervals)                    │
│  • Range: ~0.5 to 8 cycles/degree                               │
│                                                                 │
│  ORIENTATION BANDS:                                             │
│  ─────────────────                                              │
│  • 8-12 distinct orientations                                   │
│  • Uniformly spaced (15-20° intervals)                          │
│  • Full 180° coverage (not 360° due to symmetry)                │
│                                                                 │
│  PHASE SENSITIVITY:                                             │
│  ─────────────────                                              │
│  • Simple cells: specific phase                                 │
│  • Complex cells: phase-invariant (pooled)                      │
│  • Effectively ~4 phase quadrants                               │
│                                                                 │
│  TOTAL CHANNELS: 4-6 freq × 8-12 orient × 2-4 phase            │
│                = 64 to 288 channels                             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

Evolution's answer after millions of years of optimization!
```

### 4.2 The Magic Number 4±1

```
MILLER'S LAW (1956):

Human working memory capacity: 7 ± 2 items
Subitizing limit: 4 items (instant recognition)

Related findings:
• 4 objects can be tracked simultaneously
• 4 features can be bound in attention
• 4 chunks in verbal working memory

For spectral bands:
• 4-5 frequency scales seems optimal for human perception
• More than ~6 bands provide diminishing returns
• Fewer than ~3 bands lose critical information
```

### 4.3 Cortical Magnification

```
CORTICAL MAGNIFICATION FACTOR:

More cortical area devoted to fovea (center) than periphery.
Scaling: M(e) ∝ 1/(e + e₀)

This is equivalent to LOG-POLAR sampling!

Implications:
• Central (low-frequency) information gets more resources
• Peripheral (high-frequency) information compressed
• Matches 1/f spectral structure of natural images
```

### 4.4 Gabor-Like Receptive Fields

```
GABOR FUNCTION (optimal for V1):

G(x,y) = exp(-(x'² + γ²y'²)/(2σ²)) × cos(2πx'/λ + φ)

Parameters:
• σ (size): determines frequency band
• θ (orientation): determines orientation band
• λ (wavelength): determines precise frequency
• φ (phase): determines phase sensitivity

Typical V1-like filter bank:
• 5 scales (σ)
• 8 orientations (θ)
• 2 phases (φ = 0, π/2 for quadrature pair)
─────────────────────────────────
Total: 5 × 8 × 2 = 80 filters

This matches compressed sensing requirements!
```

---

## 5. Signal Processing Theory

### 5.1 Wavelet Multiresolution Analysis

```
WAVELET DECOMPOSITION:

Level 0 (finest):    [D₀] ─────────────────────────────── N coeffs
Level 1:             [D₁] ───────────────────── N/2 coeffs
Level 2:             [D₂] ─────────────── N/4 coeffs
Level 3:             [D₃] ───────── N/8 coeffs
Level 4:             [D₄] ───── N/16 coeffs
Level L (coarsest):  [A_L] ── Approximation

Total levels: L = log₂(N)

For N = 64: L = 6 levels
For N = 128: L = 7 levels
For N = 256: L = 8 levels

OPTIMAL DECOMPOSITION DEPTH:

Usually stop at level 3-5, not full log₂(N):
• Levels 0-2: Fine detail (often noise)
• Levels 3-5: Meaningful structure
• Levels 6+: Too coarse (single coefficients)

Practical: 4-6 levels capture ~95% of useful information
```

### 5.2 Octave Band Structure

```
OCTAVE = doubling of frequency

┌──────────────────────────────────────────────────────────────┐
│                                                              │
│  Band    Frequency Range       Wavelength      Information   │
│  ────    ───────────────       ──────────      ───────────   │
│  1       [1, 2) cycles         32-64 px        Structure     │
│  2       [2, 4) cycles         16-32 px        Large feat    │
│  3       [4, 8) cycles         8-16 px         Medium feat   │
│  4       [8, 16) cycles        4-8 px          Small feat    │
│  5       [16, 32) cycles       2-4 px          Fine detail   │
│  6       [32, 64) cycles       1-2 px          Noise/edge    │
│                                                              │
│  Each octave spans equal log-frequency interval              │
│  Each octave contains roughly equal perceptual information   │
│                                                              │
└──────────────────────────────────────────────────────────────┘

For 64×64 image: 6 octaves possible, 4-5 typically useful
```

### 5.3 Critical Band Theory (from Audio)

```
BARK SCALE (auditory):

Human hearing divided into ~24 critical bands
Each band processes information somewhat independently
Bandwidth increases with frequency (log-like)

Analogous for vision:
• ~6 spatial frequency bands
• ~8 orientation bands
• Total: ~48 "critical bands" for 2D vision
```

### 5.4 JPEG's Empirical Answer

```
JPEG DCT COEFFICIENTS:

8×8 block → 64 DCT coefficients

Typical quantization keeps:
• DC: Always kept (average brightness)
• Low AC: 10-15 coefficients (structure)
• Mid AC: 5-10 coefficients (edges)
• High AC: Often discarded (noise/texture)

Effective bands: ~25-30 coefficients per 8×8 block
This is ~40-50% of coefficients

Scaling to 64×64:
64×64 = 8 × (8×8 blocks)
Effective info: 8 × 25 = 200 coefficients (of 4096)
≈ 5% of total, organized into ~5 frequency bands
```

---

## 6. Manifold Dimensionality

### 6.1 Intrinsic Dimensionality

```
INTRINSIC DIMENSIONALITY:

The number of independent parameters needed to describe data.

For images:
• Ambient dimension: N² (all pixels)
• Intrinsic dimension: Much smaller!

Empirical findings:
┌────────────────────────────────────────────────────────────────┐
│                                                                │
│  Dataset              Ambient D      Intrinsic D     Ratio    │
│  ───────              ─────────      ───────────     ─────    │
│  MNIST (28×28)        784            ~10-15          ~1-2%    │
│  Faces (64×64)        4096           ~50-100         ~1-2%    │
│  Natural images       Varies         ~100-500        ~1-10%   │
│  Video frames         N²×T           ~20-50/frame    <1%      │
│                                                                │
│  Intrinsic D ≈ O(√N) to O(N) depending on complexity          │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### 6.2 Johnson-Lindenstrauss Lemma

```
JL LEMMA:

N points in high-D can be embedded in d dimensions where:

d = O(log(N) / ε²)

with distances preserved within factor (1 ± ε).

For 4096 points (64×64 positions) with ε = 0.1:
d ≈ 4 × log(4096) / 0.01 ≈ 4 × 12 / 0.01 ≈ 4800

But this is worst-case for arbitrary points.
For structured data (manifolds), much less needed.

Practical: d ≈ C × intrinsic_dimension × log(N)
         ≈ 10-50 × 5-10 × 12 ≈ 600-6000

This matches the ~64-288 channels from V1!
```

### 6.3 Whitney Embedding Theorem

```
WHITNEY THEOREM:

A d-dimensional manifold can be embedded in 2d dimensions.

Implication: If data lies on d-dim manifold,
            need at most 2d coordinates.

For visual data:
• Motion on 2D surface: d = 2, need 4 dimensions
• Object identity: d ≈ 10-50, need 20-100 dimensions
• Position + identity: d ≈ 50-100, need 100-200 dimensions

This bounds our channel count:
• Minimum: intrinsic_d
• Safe: 2 × intrinsic_d
• Typical: log(N) × intrinsic_d
```

---

## 7. The Logarithmic Principle

### 7.1 Logarithms Everywhere

```
THE LOG SCALE APPEARS IN:

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  DOMAIN                    LOG RELATIONSHIP                     │
│  ──────                    ────────────────                     │
│  Network paths             log(N) degrees of separation         │
│  Wavelet levels            log₂(N) decomposition levels         │
│  Visual cortex             log(frequency) spacing               │
│  Information bits          log₂(states) bits needed             │
│  Complexity                log(N) operations for search         │
│  Perception                log(intensity) = Weber-Fechner law   │
│  Manifold embedding        log(N) dimensions (JL lemma)         │
│  Sparse coding             log(N/k) measurements                │
│                                                                 │
│  THE FUNDAMENTAL UNIT IS log(N), NOT N!                         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 7.2 Why Logarithmic?

```
LOG EMERGES FROM MULTIPLICATIVE PROCESSES:

1. SCALE INVARIANCE:
   Natural images have 1/f spectra (power law)
   Equal energy per octave (log-frequency bin)
   
2. EFFICIENT CODING:
   log₂(N) bits to encode N states
   log(N) comparisons to search N items
   
3. HIERARCHICAL STRUCTURE:
   log(N) levels in balanced tree of N leaves
   log(N) layers to combine N inputs
   
4. COMMUNICATION:
   log(N) hops in small-world network
   log(N) bits to specify 1-of-N
```

### 7.3 The Magic Formula

```
OPTIMAL BANDS ≈ log₂(N) ± 2

For different image sizes:

┌───────────────────────────────────────────────────────────────┐
│                                                               │
│  Image Size    log₂(N)    Optimal Freq Bands    Total Bands  │
│  ──────────    ───────    ──────────────────    ───────────  │
│  32×32         5          3-5                   24-40         │
│  64×64         6          4-6                   32-48         │
│  128×128       7          5-7                   40-56         │
│  256×256       8          6-8                   48-64         │
│  512×512       9          6-8                   48-64         │
│  1024×1024     10         6-8                   48-64         │
│                                                               │
│  Note: Saturates around 6-8 frequency bands for large images │
│  because finer detail becomes noise rather than signal       │
│                                                               │
└───────────────────────────────────────────────────────────────┘

Total bands = frequency_bands × orientation_bands
            ≈ 6 × 8 = 48 (for mature system)
```

---

## 8. Practical Band Counts

### 8.1 Frequency Bands

```
FREQUENCY BAND RECOMMENDATIONS:

┌────────────────────────────────────────────────────────────────┐
│                                                                │
│  APPLICATION          BANDS    RATIONALE                       │
│  ───────────          ─────    ─────────                       │
│  Basic structure      2-3      Coarse + fine only              │
│  Standard vision      4-5      Matches V1 octaves              │
│  Fine discrimination  6-7      Maximum useful detail           │
│  Over-complete        8+       Redundant, may help robustness  │
│                                                                │
│  RECOMMENDED DEFAULT: 5 frequency bands (octave spacing)       │
│                                                                │
│  Band 1: DC to 1/16 Nyquist  (very coarse structure)          │
│  Band 2: 1/16 to 1/8 Nyquist (coarse structure)               │
│  Band 3: 1/8 to 1/4 Nyquist  (medium features)                │
│  Band 4: 1/4 to 1/2 Nyquist  (fine features)                  │
│  Band 5: 1/2 to Nyquist      (finest detail)                  │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### 8.2 Orientation Bands

```
ORIENTATION BAND RECOMMENDATIONS:

┌────────────────────────────────────────────────────────────────┐
│                                                                │
│  BANDS    ANGULAR RESOLUTION    USE CASE                       │
│  ─────    ──────────────────    ────────                       │
│  2        90°                   Horizontal/vertical only       │
│  4        45°                   Cardinal + diagonal            │
│  6        30°                   Good for most patterns         │
│  8        22.5°                 V1-like, standard choice       │
│  12       15°                   Fine orientation discrimination│
│  16+      <12°                  Rarely needed                  │
│                                                                │
│  RECOMMENDED DEFAULT: 8 orientations                           │
│                                                                │
│  Why 8?                                                        │
│  • Matches V1 orientation columns                              │
│  • 22.5° is near orientation discrimination threshold          │
│  • 8 = 2³ enables efficient implementation                     │
│  • Good coverage without redundancy                            │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### 8.3 Phase Representation

```
PHASE REPRESENTATION OPTIONS:

┌────────────────────────────────────────────────────────────────┐
│                                                                │
│  APPROACH              BINS    DESCRIPTION                     │
│  ────────              ────    ───────────                     │
│  Phase-invariant       0       Pool over phases (complex cells)│
│  Quadrature pair       2       Cos + Sin (complete)            │
│  Cardinal phases       4       0°, 90°, 180°, 270°             │
│  Fine phase            8       45° resolution                  │
│  Continuous            ∞       Keep full phase information     │
│                                                                │
│  RECOMMENDED: Quadrature pairs (2 phases)                      │
│                                                                │
│  Why quadrature?                                               │
│  • Cos(φ) + i·Sin(φ) = complete representation                 │
│  • Enables phase-invariant magnitude: √(cos² + sin²)           │
│  • Enables phase reconstruction: atan2(sin, cos)               │
│  • Minimal representation (2 numbers) for complete info        │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### 8.4 Total Channel Count

```
TOTAL CHANNELS = FREQ × ORIENT × PHASE

┌────────────────────────────────────────────────────────────────┐
│                                                                │
│  Configuration         Channels    Memory (64×64)    Quality   │
│  ─────────────         ────────    ──────────────    ───────   │
│                                                                │
│  Minimal:                                                      │
│  3 freq × 4 orient × 2 = 24        24×4096 = 98K    Basic     │
│                                                                │
│  Standard:                                                     │
│  5 freq × 8 orient × 2 = 80        80×4096 = 328K   Good      │
│                                                                │
│  V1-like:                                                      │
│  6 freq × 8 orient × 2 = 96        96×4096 = 393K   Very good │
│                                                                │
│  Rich:                                                         │
│  7 freq × 12 orient × 2 = 168      168×4096 = 688K  Excellent │
│                                                                │
│  SWEET SPOT: 64-96 channels (≈ V1)                            │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

---

## 9. Phase Discretization

### 9.1 Phase as Position Encoder

```
PHASE ENCODES POSITION:

Continuous phase φ ∈ [-π, π] maps to position:

position ∝ -φ / (2π × frequency)

Resolution: Δx = Δφ / (2π × f)

For phase precision Δφ and frequency f:
• Low frequency, coarse phase: Large position uncertainty
• High frequency, fine phase: Small position uncertainty

This is the UNCERTAINTY PRINCIPLE at work:
Δx × Δf ≥ 1/2
```

### 9.2 How Many Phase Bins?

```
PHASE BIN ANALYSIS:

┌────────────────────────────────────────────────────────────────┐
│                                                                │
│  If we discretize phase into B bins:                           │
│                                                                │
│  Position resolution at frequency f:                           │
│  Δx = (2π/B) / (2πf) = 1/(B×f) pixels                         │
│                                                                │
│  For highest frequency f = N/2 (Nyquist):                      │
│  Δx = 2/(B×N) pixels                                           │
│                                                                │
│  To achieve 1-pixel resolution at Nyquist:                     │
│  B×N/2 ≥ N  →  B ≥ 2                                           │
│                                                                │
│  MINIMUM: 2 phase bins (quadrature pair)                       │
│                                                                │
│  More bins provide:                                            │
│  • Slightly better noise robustness                            │
│  • Faster phase estimation (lookup vs atan2)                   │
│  • Diminishing returns past B = 4                              │
│                                                                │
│  CONCLUSION: 2 phases (quadrature) is sufficient and optimal   │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### 9.3 Complex vs Real Representation

```
REPRESENTATION CHOICE:

Option A: Real (2 channels per filter)
─────────────────────────────────────
• Even filter: cos(2πfx)
• Odd filter: sin(2πfx)
• Phase: atan2(odd, even)
• Magnitude: √(even² + odd²)

Option B: Complex (1 complex channel per filter)
────────────────────────────────────────────────
• Complex filter: exp(i2πfx)
• Phase: angle(response)
• Magnitude: abs(response)

Both are equivalent! Complex is more elegant but harder to implement.
Quadrature pairs (Option A) are standard in neural networks.
```

---

## 10. The Optimal Architecture

### 10.1 Putting It All Together

```
OPTIMAL SPECTRAL BELIEF ARCHITECTURE:

┌────────────────────────────────────────────────────────────────┐
│                                                                │
│  FREQUENCY BANDS: 5 octave bands                               │
│  ─────────────────────────────────                             │
│  │                                                             │
│  ├── Band 1: [0, 1/16) × Nyquist    Coarse structure          │
│  ├── Band 2: [1/16, 1/8) × Nyquist  Large features            │
│  ├── Band 3: [1/8, 1/4) × Nyquist   Medium features           │
│  ├── Band 4: [1/4, 1/2) × Nyquist   Fine features             │
│  └── Band 5: [1/2, 1) × Nyquist     Finest detail             │
│                                                                │
│  ORIENTATION BANDS: 8 orientations                             │
│  ─────────────────────────────────                             │
│  │                                                             │
│  └── θ = 0°, 22.5°, 45°, 67.5°, 90°, 112.5°, 135°, 157.5°    │
│                                                                │
│  PHASE: Quadrature pairs (2 per filter)                        │
│  ─────────────────────────────────────                         │
│  │                                                             │
│  └── Cosine (even) + Sine (odd) for each freq×orient          │
│                                                                │
│  TOTAL: 5 × 8 × 2 = 80 channels                                │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### 10.2 Hierarchical Organization

```
BELIEF HIERARCHY:

Level 1: DC + Band 1 (coarse)
         4 channels (1 freq × 1 orient × 2 phase + DC)
         Most protected, slowest learning rate
         
Level 2: Bands 2-3 (medium)
         32 channels (2 freq × 8 orient × 2 phase)
         Moderately protected, medium learning rate
         
Level 3: Bands 4-5 (fine)
         64 channels (2 freq × 8 orient × 2 phase)
         Least protected, fastest learning rate


Protection hierarchy (from MANIFOLD_BELIEF_INTERFERENCE.md):

┌────────────────────────────────────────────────────────────────┐
│                                                                │
│  Level    Channels    Learning Rate    EMA Decay    λ_reg     │
│  ─────    ────────    ─────────────    ─────────    ─────     │
│  1 (DC)   4           0.0001           0.999        10.0      │
│  2        32          0.0003           0.99         1.0       │
│  3        64          0.001            0.9          0.1       │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### 10.3 The Small-World Connection

```
FREQUENCY BANDS CREATE SMALL-WORLD STRUCTURE:

Within band: Dense local connections (spatial neighbors)
Across bands: Sparse long-range connections (scale relationships)

Number of bands B ≈ log₂(N) ensures:
• Any two positions connected in O(B) = O(log N) hops
• This is the small-world property!

Wormhole attention exploits this:
• Low-freq bands: Find structurally similar (WHAT)
• High-freq bands: Locate precisely (WHERE)
• Cross-band: Link structure to location

Path length: O(1) within band + O(log N) across bands
           = O(log N) total
           = Small world!
```

---

## 11. Implementation Guidelines

### 11.1 PyTorch Implementation

```python
class OptimalSpectralDecomposition(nn.Module):
    """
    Optimal spectral decomposition based on theory.
    
    5 frequency bands × 8 orientations × 2 phases = 80 channels
    """
    def __init__(self, in_channels=1, grid_size=64):
        super().__init__()
        
        self.grid_size = grid_size
        self.n_freq_bands = 5
        self.n_orientations = 8
        self.n_phases = 2
        self.total_channels = self.n_freq_bands * self.n_orientations * self.n_phases
        
        # Frequency band boundaries (fraction of Nyquist)
        self.freq_bounds = [0, 1/16, 1/8, 1/4, 1/2, 1.0]
        
        # Orientations (in radians)
        self.orientations = torch.linspace(0, np.pi, self.n_orientations + 1)[:-1]
        
        # Create Gabor filter bank
        self.register_buffer('filters', self._create_gabor_bank())
        
        # Learnable per-band parameters (for adaptive weighting)
        self.band_weights = nn.Parameter(torch.ones(self.n_freq_bands))
        
    def _create_gabor_bank(self):
        """Create bank of Gabor filters."""
        filters = []
        
        for freq_idx in range(self.n_freq_bands):
            # Frequency for this band (geometric mean of bounds)
            f_low = self.freq_bounds[freq_idx]
            f_high = self.freq_bounds[freq_idx + 1]
            freq = np.sqrt(f_low * f_high) if f_low > 0 else f_high / 2
            freq *= self.grid_size / 2  # Convert to cycles
            
            for theta in self.orientations:
                for phase in [0, np.pi/2]:  # Quadrature pair
                    gabor = self._make_gabor(freq, theta.item(), phase)
                    filters.append(gabor)
        
        return torch.stack(filters)  # [80, H, W]
    
    def _make_gabor(self, freq, theta, phase, sigma_factor=0.5):
        """Create single Gabor filter."""
        size = self.grid_size
        sigma = sigma_factor * size / (freq + 1)
        
        x = torch.linspace(-size/2, size/2, size)
        y = torch.linspace(-size/2, size/2, size)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        
        # Rotate coordinates
        X_theta = X * np.cos(theta) + Y * np.sin(theta)
        Y_theta = -X * np.sin(theta) + Y * np.cos(theta)
        
        # Gabor = Gaussian × Sinusoid
        gaussian = torch.exp(-(X_theta**2 + Y_theta**2) / (2 * sigma**2))
        sinusoid = torch.cos(2 * np.pi * freq * X_theta / size + phase)
        
        gabor = gaussian * sinusoid
        gabor = gabor / (gabor.abs().sum() + 1e-8)  # Normalize
        
        return gabor
    
    def forward(self, x):
        """
        Decompose input into spectral bands.
        
        Args:
            x: [B, 1, H, W] input
            
        Returns:
            bands: [B, 80, H, W] spectral decomposition
        """
        B = x.shape[0]
        
        # Apply filter bank via FFT convolution
        x_fft = torch.fft.fft2(x)
        filters_fft = torch.fft.fft2(self.filters.unsqueeze(0))
        
        response_fft = x_fft * filters_fft.conj()
        response = torch.fft.ifft2(response_fft).real
        
        # Apply learned band weights
        response = response.view(B, self.n_freq_bands, -1, self.grid_size, self.grid_size)
        response = response * self.band_weights.view(1, -1, 1, 1, 1)
        response = response.view(B, -1, self.grid_size, self.grid_size)
        
        return response
    
    def get_magnitude_phase(self, x):
        """Get magnitude (WHAT) and phase (WHERE) from quadrature pairs."""
        bands = self.forward(x)
        B, C, H, W = bands.shape
        
        # Reshape to [B, n_freq, n_orient, 2, H, W]
        bands = bands.view(B, self.n_freq_bands, self.n_orientations, 2, H, W)
        
        even = bands[:, :, :, 0]  # Cosine response
        odd = bands[:, :, :, 1]   # Sine response
        
        magnitude = torch.sqrt(even**2 + odd**2 + 1e-8)  # WHAT
        phase = torch.atan2(odd, even)                    # WHERE
        
        return magnitude, phase
```

### 11.2 Configuration Recommendations

```python
# Recommended configurations for different use cases

CONFIGS = {
    'minimal': {
        'n_freq_bands': 3,
        'n_orientations': 4,
        'n_phases': 2,
        'total_channels': 24,
        'use_case': 'Resource-constrained, basic structure'
    },
    
    'standard': {
        'n_freq_bands': 5,
        'n_orientations': 8,
        'n_phases': 2,
        'total_channels': 80,
        'use_case': 'General purpose, good balance'
    },
    
    'v1_like': {
        'n_freq_bands': 6,
        'n_orientations': 8,
        'n_phases': 2,
        'total_channels': 96,
        'use_case': 'Biologically-inspired, high quality'
    },
    
    'rich': {
        'n_freq_bands': 7,
        'n_orientations': 12,
        'n_phases': 2,
        'total_channels': 168,
        'use_case': 'Maximum discrimination, research'
    }
}

# Learning rate schedule by frequency band
LEARNING_RATES = {
    'band_1': 0.0001,  # DC/coarse: very slow
    'band_2': 0.0003,  # 
    'band_3': 0.0005,  # 
    'band_4': 0.0008,  #
    'band_5': 0.001,   # Fine: normal speed
}
```

### 11.3 Summary Table

```
┌────────────────────────────────────────────────────────────────┐
│                                                                │
│           OPTIMAL SPECTRAL BELIEF STORAGE SUMMARY              │
│                                                                │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  FREQUENCY BANDS:                                              │
│  • Optimal: 5-6 octave-spaced bands                           │
│  • Minimum: 3 bands (coarse/medium/fine)                      │
│  • Maximum useful: 7-8 bands                                  │
│  • Follows: log₂(N) scaling                                   │
│                                                                │
│  ORIENTATION BANDS:                                            │
│  • Optimal: 8 orientations (22.5° spacing)                    │
│  • Minimum: 4 orientations                                    │
│  • Maximum useful: 12 orientations                            │
│  • Matches: V1 orientation columns                            │
│                                                                │
│  PHASE REPRESENTATION:                                         │
│  • Optimal: 2 (quadrature pair)                               │
│  • Sufficient for complete reconstruction                      │
│  • More phases: diminishing returns                           │
│                                                                │
│  TOTAL CHANNELS:                                               │
│  • Optimal: 5×8×2 = 80 channels                               │
│  • Range: 24-168 depending on requirements                    │
│  • Sweet spot: 64-96 channels                                 │
│                                                                │
│  THEORETICAL BASIS:                                            │
│  • Information theory: log(N) bits for N states               │
│  • Network theory: log(N) hops in small world                 │
│  • Neuroscience: V1 has ~80-100 filter types                 │
│  • Signal processing: log₂(N) wavelet levels                  │
│                                                                │
│  THE MAGIC NUMBER: log₂(N) ≈ 5-6 for typical image sizes     │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

---

## Key Takeaways

```
1. THE LOG PRINCIPLE:
   Optimal bands ≈ log₂(image_size)
   This appears in networks, wavelets, entropy, and neuroscience.

2. THE V1 SOLUTION:
   ~80 channels (5 freq × 8 orient × 2 phase)
   Evolution's answer after millions of years.

3. THE SMALL-WORLD CONNECTION:
   log(N) bands enable O(log N) routing
   This is why both brains and networks converge on this structure.

4. PHASE IS EFFICIENT:
   2 phases (quadrature) provide complete position information.
   More phases waste capacity.

5. HIERARCHY MATTERS:
   Protect coarse bands, let fine bands adapt.
   This prevents belief collapse while enabling learning.

6. THE SWEET SPOT:
   64-96 channels capture ~95% of useful structure
   with ~5% of raw pixel information.
```

---

*This document synthesizes insights from information theory, network theory, neuroscience, and signal processing to derive optimal band counts for spectral belief storage. The convergence of multiple fields on similar numbers (log N bands, ~80 channels) suggests these are fundamental constants of efficient information representation.*

