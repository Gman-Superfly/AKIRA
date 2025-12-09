# Hierarchical Spectral Manifolds: Architecture and Theory

A principled architecture where separate manifolds encode different frequency bands, connected by spectral attention, with learned routing and high observability.

---

## Table of Contents

1. [Motivation: Why Hierarchical Manifolds?](#1-motivation-why-hierarchical-manifolds)
2. [Theoretical Foundation](#2-theoretical-foundation)
3. [Evidence From Multiple Domains](#3-evidence-from-multiple-domains)
4. [Architecture Overview](#4-architecture-overview)
5. [Learned Routing Mechanism](#5-learned-routing-mechanism)
6. [Per-Manifold Learning Dynamics](#6-per-manifold-learning-dynamics)
7. [Cross-Manifold Spectral Attention](#7-cross-manifold-spectral-attention)
8. [Observability and Monitoring](#8-observability-and-monitoring)
9. [Training Strategy](#9-training-strategy)
10. [Strengths and Limitations](#10-strengths-and-limitations)
11. [Implementation](#11-implementation)

---

## 1. Motivation: Why Hierarchical Manifolds?

### 1.1 The Problem with Single Manifolds

```
CURRENT ARCHITECTURE: One manifold encodes everything

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  SINGLE MANIFOLD (all frequencies mixed)                        │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Low-freq   Mid-freq   High-freq                        │   │
│  │     ●──────────●──────────●                             │   │
│  │     │          │          │                             │   │
│  │     └──────────┴──────────┘                             │   │
│  │           All compete for same weights                   │   │
│  │           All subject to same learning rate              │   │
│  │           Interference accumulates across all            │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  PROBLEMS:                                                      │
│  • Low-freq (stable structure) corrupted by high-freq noise    │
│  • High-freq (dynamic edges) slowed by low-freq inertia        │
│  • Interference crosses scale boundaries                        │
│  • No natural protection hierarchy                              │
│  • Hard to debug: which scale is failing?                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 The Proposed Solution

```
HIERARCHICAL MANIFOLDS: Separate manifold per frequency band

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  MANIFOLD 0 (Coarse)    MANIFOLD 1 (Mid)     MANIFOLD 2 (Fine) │
│  ┌─────────────────┐    ┌─────────────────┐  ┌─────────────────┐│
│  │                 │    │                 │  │                 ││
│  │   Structure     │    │   Features      │  │   Edges         ││
│  │   Identity      │◄──►│   Relationships │◄─►│   Positions     ││
│  │                 │    │                 │  │                 ││
│  │   lr: 0.0001    │    │   lr: 0.0005    │  │   lr: 0.001     ││
│  │   λ: 1.0        │    │   λ: 0.3        │  │   λ: 0.1        ││
│  │                 │    │                 │  │                 ││
│  └─────────────────┘    └─────────────────┘  └─────────────────┘│
│          ▲                      ▲                    ▲          │
│          │                      │                    │          │
│          └──────── Spectral Attention ───────────────┘          │
│                    (learned routing)                            │
│                                                                 │
│  BENEFITS:                                                      │
│  • Each manifold optimized for its scale                        │
│  • Interference contained within bands                          │
│  • Natural protection hierarchy (coarse = slow, fine = fast)   │
│  • High visibility: monitor each manifold independently         │
│  • Learned routing: system decides what belongs where           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 1.3 Core Design Principles

```
PRINCIPLE 1: FREQUENCY SEPARATION
Each manifold specializes in a distinct frequency band.
No competition for representational capacity across scales.

PRINCIPLE 2: LEARNED ROUTING
The system learns which information goes to which manifold.
Not hardcoded, but discovered through training.

PRINCIPLE 3: DIFFERENTIAL LEARNING
Each manifold has its own learning rate and regularization.
Coarse = stable (slow), Fine = adaptive (fast).

PRINCIPLE 4: SPECTRAL COMMUNICATION
Manifolds communicate via spectral attention.
Low-freq for matching (WHAT), high-freq for transfer (WHERE).

PRINCIPLE 5: OBSERVABILITY
Every manifold's state is interpretable and monitorable.
We can see what each scale is learning.
```

---

## 2. Theoretical Foundation

### 2.1 The Uncertainty Principle Argument

```
HEISENBERG-GABOR UNCERTAINTY:

Δx · Δf ≥ 1/(4π)

A single representation CANNOT be simultaneously:
• Localized in space (small Δx)
• Localized in frequency (small Δf)

IMPLICATION FOR MANIFOLDS:

A single manifold trying to encode all frequencies faces:
• Low-freq needs large Δx (spread spatial receptive field)
• High-freq needs small Δx (localized receptive field)

These requirements CONFLICT in a single manifold!

SOLUTION: SEPARATE MANIFOLDS
• Coarse manifold: Large Δx, small Δf (low frequencies)
• Fine manifold: Small Δx, large Δf (high frequencies)

Each manifold can OPTIMIZE its uncertainty trade-off.
```

### 2.2 Information-Theoretic Argument

```
RATE-DISTORTION THEORY:

Different frequency bands have different information densities:

┌────────────────────────────────────────────────────────────────┐
│                                                                │
│  Band          Energy    Info Density    Bits Needed          │
│  ────          ──────    ────────────    ───────────          │
│  DC/Coarse     60%       Low             Few (< 10%)          │
│  Mid-Low       25%       Medium          Moderate (20-30%)    │
│  Mid-High      10%       High            Significant (30-40%) │
│  Fine          5%        Very High       Most (30-40%)        │
│                                                                │
│  High frequencies: Low energy but HIGH information density     │
│  Low frequencies: High energy but LOW information density      │
│                                                                │
└────────────────────────────────────────────────────────────────┘

IMPLICATION:
Different bands need different representational strategies.
A single manifold cannot optimally encode all bands.
```

### 2.3 Dynamical Systems Argument

```
LEARNING DYNAMICS:

Each frequency band has different optimal learning timescales:

Coarse (structure):
• Changes slowly (objects don't change identity often)
• Should have HIGH inertia (slow learning)
• Risk if fast: Structural collapse, identity confusion

Fine (edges):
• Changes rapidly (edges move every frame)
• Should have LOW inertia (fast learning)
• Risk if slow: Cannot track motion, blurry predictions

MIXING THESE IN ONE MANIFOLD:
• If lr is low: Fine features lag behind
• If lr is high: Coarse features become unstable
• No single lr is optimal for all

SEPARATE MANIFOLDS ALLOW:
lr_coarse << lr_mid << lr_fine

Each band learns at its natural timescale.
```

---

## 3. Evidence From Multiple Domains

### 3.1 Neuroscience Evidence

```
VISUAL CORTEX ORGANIZATION:

Primary Visual Cortex (V1):
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  SPATIAL FREQUENCY COLUMNS:                                     │
│  • Distinct populations of neurons per frequency band           │
│  • 4-6 octave bands (0.5 to 16 cpd)                            │
│  • Neurons in each band have matched receptive field sizes      │
│                                                                 │
│  HIERARCHICAL STREAMS:                                          │
│  • Ventral stream: Coarse → object identity (WHAT)             │
│  • Dorsal stream: Fine → spatial location (WHERE)              │
│                                                                 │
│  KEY FINDING:                                                   │
│  The cortex DOES separate frequencies into distinct processing  │
│  streams. This is not one unified manifold but a hierarchy.     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

Source: Hubel & Wiesel, De Valois, Movshon et al.
```

### 3.2 Signal Processing Evidence

```
MULTI-RESOLUTION ANALYSIS:

Wavelet transforms are optimal precisely because they:
• Decompose signals into frequency bands
• Process each band with matched resolution
• Reconstruct by combining band outputs

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  JPEG 2000 uses wavelets because:                               │
│  • Different frequencies need different quantization            │
│  • Coarse bands: Preserve exactly (critical for perception)     │
│  • Fine bands: Can be quantized heavily (less visible)          │
│                                                                 │
│  This IS a hierarchical manifold for image compression!         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 3.3 Network Theory Evidence

```
SMALL-WORLD AND SCALE-FREE NETWORKS:

Efficient networks have HIERARCHICAL structure:

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  Hub nodes (few): Connect to many, carry global information     │
│  └── Analogous to: Coarse manifold (structure, identity)       │
│                                                                 │
│  Leaf nodes (many): Connect locally, carry specific information │
│  └── Analogous to: Fine manifold (edges, positions)            │
│                                                                 │
│  Optimal networks are NOT uniform:                              │
│  They have HIERARCHY matching information hierarchy.            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

Source: Barabási, Watts & Strogatz
```

### 3.4 Machine Learning Evidence

```
SUCCESSFUL MULTI-SCALE ARCHITECTURES:

U-Net (2015):
• Separate encoder/decoder paths per scale
• Skip connections between scales
• State-of-the-art for segmentation

Feature Pyramid Networks (2017):
• Multi-scale feature extraction
• Lateral connections between scales
• State-of-the-art for detection

Transformers with hierarchical attention (2021+):
• Different attention patterns per scale
• Swin Transformer, Multiscale ViT
• State-of-the-art for many tasks

COMMON PATTERN:
Successful architectures separate processing by scale.
Our proposal makes this EXPLICIT and PRINCIPLED.
```

---

## 4. Architecture Overview

### 4.1 High-Level Structure

```
COMPLETE ARCHITECTURE:

                         ┌─────────────────────────────────────┐
                         │           INPUT [T, N, N]           │
                         │        (context window)             │
                         └────────────────┬────────────────────┘
                                          │
                                          ▼
                         ┌─────────────────────────────────────┐
                         │     SPECTRAL DECOMPOSITION          │
                         │  (FFT / Wavelet / Learned filters)  │
                         └────────────────┬────────────────────┘
                                          │
                    ┌─────────────────────┼─────────────────────┐
                    │                     │                     │
                    ▼                     ▼                     ▼
         ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
         │   MANIFOLD 0     │  │   MANIFOLD 1     │  │   MANIFOLD 2     │
         │   (Coarse)       │  │   (Mid)          │  │   (Fine)         │
         │                  │  │                  │  │                  │
         │ • DC + low freq  │  │ • Mid frequencies│  │ • High freq      │
         │ • Structure      │  │ • Features       │  │ • Edges          │
         │ • lr: 0.0001     │  │ • lr: 0.0005     │  │ • lr: 0.001      │
         │ • dim: 32        │  │ • dim: 64        │  │ • dim: 128       │
         └────────┬─────────┘  └────────┬─────────┘  └────────┬─────────┘
                  │                     │                     │
                  │     ┌───────────────┴───────────────┐     │
                  │     │                               │     │
                  ▼     ▼                               ▼     ▼
         ┌─────────────────────────────────────────────────────────────┐
         │              CROSS-MANIFOLD SPECTRAL ATTENTION              │
         │                                                             │
         │  • Learned routing gates: which info goes where             │
         │  • Wormhole connections: match structure across scales      │
         │  • Hierarchical context: coarse guides fine                 │
         │                                                             │
         └─────────────────────────────┬───────────────────────────────┘
                                       │
                    ┌──────────────────┼──────────────────┐
                    │                  │                  │
                    ▼                  ▼                  ▼
         ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
         │ Coarse Prediction│  │  Mid Prediction  │  │  Fine Prediction │
         └────────┬─────────┘  └────────┬─────────┘  └────────┬─────────┘
                  │                     │                     │
                  └─────────────────────┴─────────────────────┘
                                        │
                                        ▼
                         ┌─────────────────────────────────────┐
                         │       SPECTRAL SYNTHESIS            │
                         │   (combine frequency bands)         │
                         └────────────────┬────────────────────┘
                                          │
                                          ▼
                         ┌─────────────────────────────────────┐
                         │          OUTPUT [N, N]              │
                         │        (prediction)                 │
                         └─────────────────────────────────────┘
```

### 4.2 Manifold Count Decision

```
HOW MANY MANIFOLDS?

Option A: 2 manifolds (WHAT/WHERE)
────────────────────────────────────
Pros: Simplest, clearest separation
Cons: May miss mid-range structure

Manifold 0: DC to 0.2 Nyquist (structure)
Manifold 1: 0.2 to 1.0 Nyquist (detail)


Option B: 3 manifolds (Coarse/Mid/Fine)
──────────────────────────────────────
Pros: Good balance, matches perception
Cons: Still somewhat coarse

Manifold 0: DC to 0.1 Nyquist
Manifold 1: 0.1 to 0.3 Nyquist
Manifold 2: 0.3 to 1.0 Nyquist


Option C: 4 manifolds (matches log₂(N)/2)
──────────────────────────────────────────
Pros: Fine-grained control
Cons: More complexity, more parameters

Manifold 0: DC to 1/16 Nyquist
Manifold 1: 1/16 to 1/4 Nyquist
Manifold 2: 1/4 to 1/2 Nyquist
Manifold 3: 1/2 to 1.0 Nyquist


RECOMMENDATION: Start with 3 manifolds.
Can increase to 4 if needed, reduce to 2 if sufficient.
```

### 4.3 Capacity Allocation

```
CAPACITY FOLLOWS INFORMATION DENSITY:

┌────────────────────────────────────────────────────────────────┐
│                                                                │
│  Manifold    Freq Range    % Params    Dimension    Notes      │
│  ────────    ──────────    ────────    ─────────    ─────      │
│                                                                │
│  Coarse      DC - 0.1      15%         32 dim       Protected  │
│  Mid         0.1 - 0.3     30%         64 dim       Moderate   │
│  Fine        0.3 - 1.0     55%         128 dim      Flexible   │
│                                                                │
│  Total dimension: 224 (matches ~80-100 optimal channels +      │
│                   overhead for routing and attention)          │
│                                                                │
└────────────────────────────────────────────────────────────────┘

RATIONALE:
• Fine manifold is LARGER because:
  - Higher information density
  - More diverse edge patterns
  - Faster adaptation needed

• Coarse manifold is SMALLER because:
  - Lower information density
  - Fewer structural patterns
  - Stability more important than capacity
```

---

## 5. Learned Routing Mechanism

### 5.1 Why Learned Routing?

```
HARDCODED ROUTING (fixed frequency bands):
─────────────────────────────────────────
Pros: Simple, predictable
Cons: May not match task structure
      Optimal bands differ per dataset
      No adaptation

LEARNED ROUTING (soft gating):
──────────────────────────────
Pros: Adapts to data
      Discovers optimal band boundaries
      Can handle frequency-mixed signals
Cons: Need to prevent collapse
      Harder to interpret (but we add monitoring)

WE CHOOSE LEARNED ROUTING with strong initialization.
```

### 5.2 Routing Gate Architecture

```
LEARNED ROUTING GATES:

For each spatial position (x, y) and frequency band f:

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  Input: Spectral features at position (x, y)                    │
│                                                                 │
│  Routing network:                                               │
│                                                                 │
│  features → Linear(D, M) → Softmax → gate weights [g₀, g₁, g₂] │
│                                                                 │
│  Where:                                                         │
│  • D = feature dimension                                        │
│  • M = number of manifolds                                      │
│  • g_i = probability of routing to manifold i                   │
│  • Σ g_i = 1 (soft routing, or hard with Gumbel-softmax)       │
│                                                                 │
│  Routed input to manifold i:                                    │
│  input_i = g_i × features                                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 5.3 Preventing Routing Collapse

```
COLLAPSE RISK:
Routing could collapse to always choosing one manifold.

PREVENTION STRATEGIES:

1. INITIALIZATION BIAS:
   Initialize routing weights to match expected band boundaries.
   
   g_coarse biased toward low frequencies
   g_fine biased toward high frequencies

2. ENTROPY REGULARIZATION:
   L_entropy = -λ × Σ g_i × log(g_i)
   
   Encourages routing to spread across manifolds.

3. USAGE REGULARIZATION:
   L_usage = λ × Var(usage per manifold)
   
   Penalizes imbalanced manifold usage.

4. AUXILIARY FREQUENCY LOSS:
   L_freq = Σ_i |mean_freq(manifold_i) - target_freq_i|
   
   Encourages each manifold to contain expected frequencies.
```

### 5.4 Soft vs Hard Routing

```
SOFT ROUTING:
input_i = g_i × features  (weighted sum)

Pros: Differentiable, gradients flow everywhere
Cons: All manifolds always active, less efficient

HARD ROUTING (Gumbel-softmax):
input_i = one_hot(argmax(g + noise)) × features

Pros: Sparse, efficient at inference
Cons: Gradient estimation needed, may be unstable

RECOMMENDATION:
Train with SOFT routing + entropy regularization.
Optionally convert to HARD routing for deployment.
```

---

## 6. Per-Manifold Learning Dynamics

### 6.1 Learning Rate Hierarchy

```
LEARNING RATE SCHEDULE:

┌────────────────────────────────────────────────────────────────┐
│                                                                │
│  Manifold     Base LR       Decay         Effective LR        │
│  ────────     ───────       ─────         ────────────        │
│                                                                │
│  Coarse       0.0001        0.999^step    Very slow           │
│  Mid          0.0005        0.995^step    Moderate            │
│  Fine         0.001         0.99^step     Fast                │
│  Routing      0.0003        0.997^step    Moderate (shared)   │
│  Attention    0.0005        0.995^step    Moderate            │
│                                                                │
└────────────────────────────────────────────────────────────────┘

RATIONALE:
• Coarse: Changes should be rare and deliberate
• Fine: Needs to adapt quickly to new edge patterns
• Routing: Should stabilize after initial learning
```

### 6.2 Regularization Hierarchy

```
REGULARIZATION PER MANIFOLD:

┌────────────────────────────────────────────────────────────────┐
│                                                                │
│  Manifold    Weight Decay    Spectral Reg    Dropout          │
│  ────────    ────────────    ────────────    ───────          │
│                                                                │
│  Coarse      0.01            1.0             0.0              │
│  Mid         0.001           0.3             0.1              │
│  Fine        0.0001          0.1             0.2              │
│                                                                │
└────────────────────────────────────────────────────────────────┘

RATIONALE:
• Coarse: Heavy regularization keeps structure stable
• Fine: Light regularization allows flexibility
• Spectral reg: Penalizes high-freq components in wrong manifold
```

### 6.3 EMA Per Manifold

```
EXPONENTIAL MOVING AVERAGE:

Each manifold maintains its own EMA of weights:

ema_coarse: decay = 0.9999  (almost frozen)
ema_mid:    decay = 0.999   (very stable)
ema_fine:   decay = 0.99    (moderately stable)

For inference, use EMA weights (smoother predictions).
For training, use current weights (fresh gradients).

This provides DOUBLE PROTECTION for coarse manifold:
• Slow learning rate (small updates)
• High EMA decay (average out noise)
```

---

## 7. Cross-Manifold Spectral Attention

### 7.1 Purpose of Cross-Manifold Attention

```
WHY MANIFOLDS NEED TO COMMUNICATE:

Coarse manifold knows:  "There's a ring-like structure"
Mid manifold knows:     "It's moving to the right"
Fine manifold knows:    "Edges at specific pixels"

Without communication:
• Each manifold makes independent predictions
• No coordination → inconsistent outputs
• Fine manifold might predict edges where coarse sees nothing

With cross-manifold attention:
• Coarse provides structural context to fine
• Fine provides precise locations to coarse
• Predictions are coherent across scales
```

### 7.2 Attention Mechanism

```
CROSS-MANIFOLD SPECTRAL ATTENTION:

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  For manifold i attending to manifold j:                        │
│                                                                 │
│  Q_i = Linear(manifold_i_features)     # Query from i          │
│  K_j = Linear(manifold_j_features)     # Key from j            │
│  V_j = Linear(manifold_j_features)     # Value from j          │
│                                                                 │
│  # Spectral gating: only match on structure, not position      │
│  K_j_low = LowPassFilter(K_j)          # Structure only        │
│  Q_i_low = LowPassFilter(Q_i)                                   │
│                                                                 │
│  # Attention weights (WHAT matching)                            │
│  attn = softmax(Q_i_low @ K_j_low.T / sqrt(d))                 │
│                                                                 │
│  # Value includes full frequency content (WHERE transfer)      │
│  output_i = attn @ V_j                                          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

The attention uses LOW-FREQ for matching (position-invariant)
but transfers FULL CONTENT (including high-freq position info).
```

### 7.3 Hierarchical vs Full Attention

```
OPTION A: HIERARCHICAL ONLY

Coarse ──► Mid ──► Fine   (top-down context)
Fine ──► Mid ──► Coarse   (bottom-up feedback)

Complexity: O(M) attention operations
Pros: Efficient, respects hierarchy
Cons: May miss cross-scale relationships

OPTION B: FULL CROSS-ATTENTION

All manifolds attend to all others.

Complexity: O(M²) attention operations
Pros: Rich inter-scale communication
Cons: More compute, risk of interference

OPTION C: STAR TOPOLOGY (recommended)

Mid manifold is the hub:
Coarse ◄──► Mid ◄──► Fine

Complexity: O(M) attention operations
Pros: Efficient, mid serves as translator
Cons: Mid manifold becomes bottleneck

RECOMMENDATION: Start with HIERARCHICAL, try STAR if needed.
```

### 7.4 Wormhole Extension

```
WORMHOLE ATTENTION WITHIN HIERARCHY:

Standard wormhole: Connect similar structures across space/time
Extended wormhole: Connect similar structures across SCALES

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  If coarse manifold detects "ring-like structure" at time t-5  │
│  And current input has "ring-like structure" at time t         │
│                                                                 │
│  Wormhole can:                                                  │
│  1. Match coarse_t-5 to coarse_t (temporal wormhole)           │
│  2. Transfer fine_t-5 to fine_t (precision transfer)           │
│                                                                 │
│  Result: Current prediction uses past precision information    │
│          guided by structure matching across time.              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 8. Observability and Monitoring

### 8.1 High-Visibility Architecture

```
THIS ARCHITECTURE IS DESIGNED FOR OBSERVABILITY:

Every component has interpretable outputs:

┌────────────────────────────────────────────────────────────────┐
│                                                                │
│  OBSERVABLE                  INTERPRETATION                   │
│  ──────────                  ──────────────                   │
│                                                                │
│  Routing gates g_i           Which manifold handles which     │
│                              frequencies                       │
│                                                                │
│  Manifold activations        What each scale is encoding      │
│                                                                │
│  Per-manifold loss           Which scale is struggling        │
│                                                                │
│  Cross-attention weights     How scales are communicating     │
│                                                                │
│  Per-manifold gradients      Learning dynamics per scale      │
│                                                                │
│  Manifold EMA divergence     Stability per scale              │
│                                                                │
│  Frequency content per       Is routing working correctly?    │
│  manifold                                                      │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### 8.2 Monitoring Dashboard

```
REAL-TIME MONITORING:

┌─────────────────────────────────────────────────────────────────┐
│                       MANIFOLD MONITOR                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  STEP: 1000  |  TOTAL LOSS: 0.0234  |  TIME/STEP: 45ms         │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ COARSE MANIFOLD                                          │   │
│  │ Loss: 0.0012 ▃▃▃▃▃▂▂▂▂▁▁▁  Stable                       │   │
│  │ LR:   0.0001                                             │   │
│  │ Grad: 0.0002 (small, as expected)                        │   │
│  │ EMA:  0.0001 divergence                                  │   │
│  │ Freq: 94% low, 5% mid, 1% high ✓                        │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ MID MANIFOLD                                             │   │
│  │ Loss: 0.0089 ▆▆▅▅▄▄▃▃▃▂▂▂  Learning                     │   │
│  │ LR:   0.0005                                             │   │
│  │ Grad: 0.0012 (moderate)                                  │   │
│  │ EMA:  0.0008 divergence                                  │   │
│  │ Freq: 12% low, 76% mid, 12% high ✓                      │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ FINE MANIFOLD                                            │   │
│  │ Loss: 0.0133 ▇▇▆▆▅▅▅▄▄▃▃▃  Active learning              │   │
│  │ LR:   0.0010                                             │   │
│  │ Grad: 0.0034 (larger, as expected)                       │   │
│  │ EMA:  0.0021 divergence                                  │   │
│  │ Freq: 3% low, 15% mid, 82% high ✓                       │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ROUTING HEALTH:                                                │
│  Entropy: 0.89 (healthy, not collapsed)                        │
│  Balance: [0.28, 0.35, 0.37] (reasonable distribution)         │
│                                                                 │
│  ATTENTION STATS:                                               │
│  Coarse→Mid: 0.45 average weight                               │
│  Mid→Fine:   0.52 average weight                               │
│  Cross-scale connections: 234 active                           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 8.3 Alert Conditions

```
AUTOMATIC ALERTS:

┌────────────────────────────────────────────────────────────────┐
│                                                                │
│  CONDITION                          ACTION                     │
│  ─────────                          ──────                     │
│                                                                │
│  Coarse manifold loss increasing    Reduce LR, check data     │
│                                                                │
│  Routing entropy < 0.3              Increase entropy reg       │
│                                                                │
│  One manifold has >80% of traffic   Check routing init         │
│                                                                │
│  Fine manifold gradients exploding  Clip gradients, reduce LR  │
│                                                                │
│  Coarse manifold EMA diverging      Something is wrong,        │
│                                     stop and investigate       │
│                                                                │
│  Cross-attention weights uniform    Attention not learning,    │
│                                     check temperature          │
│                                                                │
│  Frequency content misrouted        Increase spectral reg      │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

---

## 9. Training Strategy

### 9.1 Initialization

```
INITIALIZATION STRATEGY:

1. MANIFOLD WEIGHTS:
   Initialize each manifold with frequency-appropriate filters.
   
   Coarse: Large receptive fields (7×7 or larger)
   Mid: Medium receptive fields (5×5)
   Fine: Small receptive fields (3×3)

2. ROUTING GATES:
   Initialize to match expected frequency bands.
   
   g_coarse biased for freq < 0.1 Nyquist
   g_mid biased for 0.1 < freq < 0.3 Nyquist
   g_fine biased for freq > 0.3 Nyquist

3. CROSS-ATTENTION:
   Initialize with identity-like weights.
   Each manifold initially attends most to itself.

4. LEARNING RATES:
   Start at target values (no warmup within manifolds).
   But use warmup for routing (0 → target over 1000 steps).
```

### 9.2 Training Phases

```
PHASED TRAINING:

PHASE 1: INDEPENDENT LEARNING (steps 0-5000)
─────────────────────────────────────────────
• Train each manifold on its frequency band independently
• Routing is fixed (hardcoded by frequency)
• No cross-manifold attention
• Goal: Each manifold learns its specialty

PHASE 2: ROUTING LEARNING (steps 5000-15000)
────────────────────────────────────────────
• Enable learned routing
• Routing weights start learning
• Still no cross-manifold attention
• Goal: Discover optimal band boundaries

PHASE 3: INTEGRATION (steps 15000-30000)
────────────────────────────────────────
• Enable cross-manifold attention
• Full system training
• All components learn together
• Goal: Learn to coordinate across scales

PHASE 4: REFINEMENT (steps 30000+)
──────────────────────────────────
• Reduce learning rates
• Focus on fine-tuning
• Monitor for collapse
• Goal: Polish the learned representations
```

### 9.3 Loss Function

```
TOTAL LOSS:

L_total = L_reconstruction 
        + Σ_i λ_i × L_manifold_i
        + λ_route × L_routing
        + λ_spectral × L_spectral
        + λ_entropy × L_entropy

Where:

L_reconstruction: Main prediction loss (MSE or L1)
L_manifold_i:     Per-manifold auxiliary loss
L_routing:        Routing regularization
L_spectral:       Frequency band enforcement
L_entropy:        Routing entropy (prevents collapse)

Per-manifold losses:
L_manifold_coarse: λ = 1.0 (high weight, must be correct)
L_manifold_mid:    λ = 0.5
L_manifold_fine:   λ = 0.3 (lower weight, can err more)
```

---

## 10. Strengths and Limitations

### 10.1 Strengths

```
ARCHITECTURAL STRENGTHS:

1. PRINCIPLED DESIGN
   Based on information theory, neuroscience, signal processing.
   Not ad-hoc but grounded in fundamentals.

2. NATURAL PROTECTION
   Coarse manifold isolated from fine interference.
   Each scale can fail independently without cascading.

3. INTERPRETABLE
   Can see what each manifold encodes.
   Can monitor health of each scale.
   Debugging is per-manifold, not whole system.

4. FLEXIBLE LEARNING
   Each manifold learns at its natural timescale.
   No compromise between stability and adaptability.

5. EFFICIENT
   Only process relevant frequencies per manifold.
   Sparse routing reduces unnecessary computation.

6. SCALABLE
   Can add more manifolds for finer granularity.
   Can reduce to 2 manifolds for simpler tasks.

7. OBSERVABLE
   Every component has interpretable outputs.
   Can catch problems early via monitoring.
```

### 10.2 Limitations

```
LIMITATIONS AND CHALLENGES:

1. COMPLEXITY
   More components than single-manifold design.
   More hyperparameters to tune.
   More things that can go wrong.

2. TRAINING OVERHEAD
   Phased training takes longer.
   Routing learning adds complexity.
   Need to monitor multiple manifolds.

3. ROUTING RISK
   Routing could collapse to single manifold.
   Mitigation needed (entropy reg, usage reg).
   Need to monitor routing health.

4. CROSS-MANIFOLD COORDINATION
   Manifolds could learn conflicting representations.
   Attention needs to learn translation.
   Risk of inconsistent predictions.

5. COMPUTATIONAL COST
   Multiple manifolds = more parameters.
   Cross-attention = more compute.
   May not be worth it for simple tasks.

6. BAND BOUNDARY ARTIFACTS
   Hard band boundaries could create artifacts.
   Learned routing mitigates but doesn't eliminate.
   Need smooth transitions between bands.
```

### 10.3 When to Use This Architecture

```
USE HIERARCHICAL MANIFOLDS WHEN:

✓ Task has multi-scale structure
✓ Stability of structure is critical
✓ Need to adapt quickly to details
✓ Interpretability is important
✓ Have compute budget for complexity
✓ Training time is available

USE SINGLE MANIFOLD WHEN:

✓ Task is simple / single-scale
✓ Compute is limited
✓ Training time is limited
✓ Interpretability not critical
✓ Good results already with simple model
```

---

## 11. Implementation

### 11.1 Core Classes

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralManifold(nn.Module):
    """Single manifold for one frequency band."""
    
    def __init__(
        self,
        band_idx: int,
        freq_range: tuple,  # (low, high) as fraction of Nyquist
        dim: int,
        kernel_size: int,
        lr_scale: float,
    ):
        super().__init__()
        self.band_idx = band_idx
        self.freq_range = freq_range
        self.dim = dim
        self.lr_scale = lr_scale
        
        # Encoder for this band
        self.encoder = nn.Sequential(
            nn.Conv2d(1, dim // 2, kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
            nn.Conv2d(dim // 2, dim, kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
        )
        
        # Temporal attention within this band
        self.temporal_attn = nn.MultiheadAttention(dim, num_heads=4)
        
        # Decoder for this band
        self.decoder = nn.Sequential(
            nn.Conv2d(dim, dim // 2, kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
            nn.Conv2d(dim // 2, 1, kernel_size, padding=kernel_size // 2),
        )
        
        # EMA for stability
        self.register_buffer('ema_weight', None)
        self.ema_decay = 0.999 - 0.009 * band_idx  # Coarse = 0.999, Fine = 0.99
    
    def update_ema(self):
        if self.ema_weight is None:
            self.ema_weight = {k: v.clone() for k, v in self.state_dict().items()}
        else:
            for k, v in self.state_dict().items():
                self.ema_weight[k] = self.ema_decay * self.ema_weight[k] + (1 - self.ema_decay) * v
    
    def forward(self, x, history=None):
        """
        Args:
            x: [B, 1, H, W] current frame (band-filtered)
            history: [B, T, 1, H, W] past frames (band-filtered)
        """
        B, _, H, W = x.shape
        
        # Encode current
        z = self.encoder(x)  # [B, dim, H, W]
        
        # Temporal attention if history provided
        if history is not None:
            T = history.shape[1]
            # Encode history
            hist_z = torch.stack([self.encoder(history[:, t]) for t in range(T)], dim=1)
            # [B, T, dim, H, W]
            
            # Reshape for attention
            z_flat = z.view(B, self.dim, -1).permute(2, 0, 1)  # [H*W, B, dim]
            hist_flat = hist_z.view(B, T, self.dim, -1).permute(3, 1, 0, 2)
            hist_flat = hist_flat.reshape(-1, B, self.dim)  # [H*W*T, B, dim]
            
            # Attention
            z_attended, _ = self.temporal_attn(z_flat, hist_flat, hist_flat)
            z = z_attended.permute(1, 2, 0).view(B, self.dim, H, W)
        
        # Decode
        out = self.decoder(z)
        
        return out, z  # Return both prediction and features


class LearnedRouter(nn.Module):
    """Learns to route input to appropriate manifolds."""
    
    def __init__(self, n_manifolds: int, dim: int):
        super().__init__()
        self.n_manifolds = n_manifolds
        
        # Routing network
        self.router = nn.Sequential(
            nn.Conv2d(1, dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(dim, n_manifolds, 1),
        )
        
        # Temperature for softmax
        self.temperature = nn.Parameter(torch.ones(1))
    
    def forward(self, x, hard=False):
        """
        Args:
            x: [B, 1, H, W] input
            hard: If True, use hard routing (Gumbel-softmax)
        
        Returns:
            gates: [B, n_manifolds, H, W] routing weights
        """
        logits = self.router(x)  # [B, n_manifolds, H, W]
        
        if hard and self.training:
            # Gumbel-softmax for hard routing during training
            gates = F.gumbel_softmax(logits, tau=self.temperature, hard=True, dim=1)
        else:
            gates = F.softmax(logits / self.temperature, dim=1)
        
        return gates


class CrossManifoldAttention(nn.Module):
    """Attention between manifolds."""
    
    def __init__(self, dims: list, n_heads: int = 4):
        super().__init__()
        self.n_manifolds = len(dims)
        
        # Project all manifolds to common dimension
        max_dim = max(dims)
        self.projections = nn.ModuleList([
            nn.Conv2d(d, max_dim, 1) for d in dims
        ])
        
        # Cross-attention
        self.cross_attn = nn.MultiheadAttention(max_dim, n_heads)
        
        # Project back
        self.unprojections = nn.ModuleList([
            nn.Conv2d(max_dim, d, 1) for d in dims
        ])
    
    def forward(self, manifold_features: list):
        """
        Args:
            manifold_features: List of [B, dim_i, H, W] per manifold
        
        Returns:
            attended: List of [B, dim_i, H, W] per manifold
        """
        B, _, H, W = manifold_features[0].shape
        
        # Project to common space
        projected = [self.projections[i](f) for i, f in enumerate(manifold_features)]
        
        # Concatenate all features for cross-attention
        all_features = torch.cat([
            p.view(B, -1, H * W).permute(2, 0, 1)  # [H*W, B, dim]
            for p in projected
        ], dim=0)  # [M*H*W, B, dim]
        
        # Self-attention across all manifolds and positions
        attended, _ = self.cross_attn(all_features, all_features, all_features)
        
        # Split back into manifolds
        chunk_size = H * W
        chunks = attended.split(chunk_size, dim=0)
        
        # Unproject back to original dimensions
        results = []
        for i, chunk in enumerate(chunks):
            reshaped = chunk.permute(1, 2, 0).view(B, -1, H, W)
            results.append(self.unprojections[i](reshaped))
        
        return results


class HierarchicalSpectralManifolds(nn.Module):
    """Complete hierarchical manifold architecture."""
    
    def __init__(
        self,
        n_manifolds: int = 3,
        base_dim: int = 32,
        grid_size: int = 64,
    ):
        super().__init__()
        self.n_manifolds = n_manifolds
        self.grid_size = grid_size
        
        # Frequency band boundaries
        self.freq_bounds = self._compute_freq_bounds(n_manifolds)
        
        # Create manifolds
        self.manifolds = nn.ModuleList([
            SpectralManifold(
                band_idx=i,
                freq_range=(self.freq_bounds[i], self.freq_bounds[i + 1]),
                dim=base_dim * (2 ** i),  # 32, 64, 128
                kernel_size=7 - 2 * i,     # 7, 5, 3
                lr_scale=0.1 ** (n_manifolds - i - 1),  # 0.01, 0.1, 1.0
            )
            for i in range(n_manifolds)
        ])
        
        # Learned router
        self.router = LearnedRouter(n_manifolds, dim=32)
        
        # Cross-manifold attention
        dims = [base_dim * (2 ** i) for i in range(n_manifolds)]
        self.cross_attention = CrossManifoldAttention(dims)
        
        # Spectral analysis/synthesis
        self.register_buffer('freq_masks', self._create_freq_masks())
    
    def _compute_freq_bounds(self, n):
        # Logarithmically spaced frequency bounds
        bounds = [0.0]
        for i in range(1, n):
            bounds.append(0.1 * (3.0 ** (i - 1)))  # 0.1, 0.3, 0.9
        bounds.append(1.0)
        return bounds
    
    def _create_freq_masks(self):
        H = W = self.grid_size
        cy, cx = H // 2, W // 2
        
        y = torch.arange(H).float() - cy
        x = torch.arange(W).float() - cx
        Y, X = torch.meshgrid(y, x, indexing='ij')
        
        freq_dist = torch.sqrt(X ** 2 + Y ** 2) / min(cx, cy)
        freq_dist = torch.fft.fftshift(freq_dist)
        
        masks = []
        for i in range(self.n_manifolds):
            low, high = self.freq_bounds[i], self.freq_bounds[i + 1]
            # Smooth mask
            mask = torch.sigmoid(20 * (freq_dist - low)) * torch.sigmoid(20 * (high - freq_dist))
            masks.append(mask)
        
        return torch.stack(masks)  # [n_manifolds, H, W]
    
    def decompose(self, x):
        """Decompose input into frequency bands."""
        x_fft = torch.fft.fft2(x)
        
        bands = []
        for i in range(self.n_manifolds):
            band_fft = x_fft * self.freq_masks[i].unsqueeze(0).unsqueeze(0)
            band = torch.fft.ifft2(band_fft).real
            bands.append(band)
        
        return bands
    
    def synthesize(self, bands):
        """Combine frequency bands into output."""
        return sum(bands)
    
    def forward(self, x, history=None):
        """
        Args:
            x: [B, 1, H, W] current frame
            history: [B, T, 1, H, W] past frames
        
        Returns:
            output: [B, 1, H, W] prediction
            info: dict with monitoring information
        """
        B = x.shape[0]
        
        # Decompose into frequency bands
        bands = self.decompose(x)
        hist_bands = None
        if history is not None:
            hist_bands = [self.decompose(history[:, t]) for t in range(history.shape[1])]
            # Reorganize: list of T lists of M bands → list of M tensors of [B, T, 1, H, W]
            hist_bands = [
                torch.stack([hist_bands[t][m] for t in range(len(hist_bands))], dim=1)
                for m in range(self.n_manifolds)
            ]
        
        # Learned routing gates
        gates = self.router(x)  # [B, n_manifolds, H, W]
        
        # Process each band through its manifold
        manifold_outputs = []
        manifold_features = []
        for i, manifold in enumerate(self.manifolds):
            # Apply routing gate
            routed_input = bands[i] * gates[:, i:i+1]
            routed_history = None
            if hist_bands is not None:
                routed_history = hist_bands[i] * gates[:, i:i+1].unsqueeze(1)
            
            # Process through manifold
            out, features = manifold(routed_input, routed_history)
            manifold_outputs.append(out)
            manifold_features.append(features)
        
        # Cross-manifold attention
        attended_features = self.cross_attention(manifold_features)
        
        # Final predictions from attended features
        final_outputs = []
        for i, manifold in enumerate(self.manifolds):
            final_out = manifold.decoder(attended_features[i])
            final_outputs.append(final_out)
        
        # Synthesize
        output = self.synthesize(final_outputs)
        
        # Monitoring info
        info = {
            'gates': gates.detach(),
            'manifold_outputs': [o.detach() for o in manifold_outputs],
            'routing_entropy': self._compute_routing_entropy(gates),
            'routing_balance': gates.mean(dim=(0, 2, 3)).detach(),
            'freq_content': [self._compute_freq_content(o) for o in manifold_outputs],
        }
        
        return output, info
    
    def _compute_routing_entropy(self, gates):
        # Entropy of routing distribution
        eps = 1e-8
        return -(gates * torch.log(gates + eps)).sum(dim=1).mean()
    
    def _compute_freq_content(self, x):
        # Compute frequency distribution of output
        x_fft = torch.fft.fft2(x)
        power = torch.abs(x_fft) ** 2
        
        low = (power * self.freq_masks[0]).sum() / power.sum()
        mid = (power * self.freq_masks[1]).sum() / power.sum() if self.n_manifolds > 2 else 0
        high = (power * self.freq_masks[-1]).sum() / power.sum()
        
        return {'low': low.item(), 'mid': mid.item() if mid else 0, 'high': high.item()}
    
    def get_param_groups(self, base_lr=0.001):
        """Get parameter groups with per-manifold learning rates."""
        groups = []
        
        for i, manifold in enumerate(self.manifolds):
            groups.append({
                'params': manifold.parameters(),
                'lr': base_lr * manifold.lr_scale,
                'name': f'manifold_{i}',
            })
        
        groups.append({
            'params': self.router.parameters(),
            'lr': base_lr * 0.5,
            'name': 'router',
        })
        
        groups.append({
            'params': self.cross_attention.parameters(),
            'lr': base_lr * 0.5,
            'name': 'cross_attention',
        })
        
        return groups
```

### 11.2 Training Loop

```python
def train_hierarchical_manifolds(
    model: HierarchicalSpectralManifolds,
    dataloader,
    n_epochs: int = 100,
    base_lr: float = 0.001,
):
    # Optimizer with per-manifold learning rates
    param_groups = model.get_param_groups(base_lr)
    optimizer = torch.optim.AdamW(param_groups)
    
    # Losses
    mse_loss = nn.MSELoss()
    
    # Monitoring
    history = {
        'total_loss': [],
        'manifold_losses': [[] for _ in range(model.n_manifolds)],
        'routing_entropy': [],
        'routing_balance': [],
    }
    
    for epoch in range(n_epochs):
        for batch_idx, (context, target) in enumerate(dataloader):
            # Forward pass
            x = context[:, -1]  # Current frame
            hist = context[:, :-1]  # History
            
            output, info = model(x, hist)
            
            # Main loss
            loss_main = mse_loss(output, target)
            
            # Per-manifold losses
            target_bands = model.decompose(target)
            loss_manifolds = [
                mse_loss(info['manifold_outputs'][i], target_bands[i])
                for i in range(model.n_manifolds)
            ]
            
            # Routing regularization
            lambda_entropy = 0.1
            loss_entropy = -lambda_entropy * info['routing_entropy']
            
            # Total loss (weighted by manifold importance)
            manifold_weights = [1.0, 0.5, 0.3]  # Coarse most important
            loss_total = loss_main + sum(
                w * l for w, l in zip(manifold_weights, loss_manifolds)
            ) + loss_entropy
            
            # Backward pass
            optimizer.zero_grad()
            loss_total.backward()
            
            # Gradient clipping per manifold
            for i, manifold in enumerate(model.manifolds):
                max_norm = 1.0 / manifold.lr_scale  # Stricter for slow-learning
                nn.utils.clip_grad_norm_(manifold.parameters(), max_norm)
            
            optimizer.step()
            
            # Update EMAs
            for manifold in model.manifolds:
                manifold.update_ema()
            
            # Logging
            history['total_loss'].append(loss_total.item())
            for i, l in enumerate(loss_manifolds):
                history['manifold_losses'][i].append(l.item())
            history['routing_entropy'].append(info['routing_entropy'].item())
            history['routing_balance'].append(info['routing_balance'].tolist())
            
            # Print monitoring info
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}")
                print(f"  Total loss: {loss_total.item():.4f}")
                for i, l in enumerate(loss_manifolds):
                    print(f"  Manifold {i} loss: {l.item():.4f}")
                print(f"  Routing entropy: {info['routing_entropy'].item():.4f}")
                print(f"  Routing balance: {info['routing_balance'].tolist()}")
    
    return history
```

---

## Summary

```
┌────────────────────────────────────────────────────────────────┐
│                                                                │
│  HIERARCHICAL SPECTRAL MANIFOLDS                               │
│                                                                │
│  CORE IDEA:                                                    │
│  Separate manifolds for different frequency bands,             │
│  connected by spectral attention, with learned routing.        │
│                                                                │
│  WHY:                                                          │
│  • Uncertainty principle: Can't optimize all scales at once    │
│  • Information theory: Different scales have different needs   │
│  • Neuroscience: Brain separates processing by scale           │
│  • Dynamics: Different scales need different learning rates    │
│                                                                │
│  HOW:                                                          │
│  • 3 manifolds: Coarse (structure), Mid (features), Fine (edges)│
│  • Learned routing gates decide what goes where                │
│  • Cross-manifold attention for coordination                   │
│  • Per-manifold learning rates and regularization              │
│                                                                │
│  BENEFITS:                                                     │
│  • Natural protection hierarchy                                │
│  • Contained interference                                      │
│  • High observability                                          │
│  • Interpretable failures                                      │
│  • Flexible learning dynamics                                  │
│                                                                │
│  CHALLENGES:                                                   │
│  • More complexity                                             │
│  • Routing collapse risk                                       │
│  • Training coordination                                       │
│  • Computational cost                                          │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

---

*This document describes a principled architecture for hierarchical spectral manifolds. The design is grounded in information theory, neuroscience, and signal processing. The key innovation is combining frequency-separated manifolds with learned routing and spectral attention for cross-scale communication.*

*Oscar Goldman, Shogu Research Group @ Datamutant.ai subsidiary of 温心重工業*