# The Spectre of Nyquist-Shannon: Fundamental Limits of Spectral Information

On the hard limits of spectral encoding, the relationship between sampling theory and belief manifolds, and the interaction between active latent space and learned weights during inference.

---

## Table of Contents

1. [The Nyquist-Shannon Theorem](#1-the-nyquist-shannon-theorem)
2. [What Can Be Encoded Spectrally](#2-what-can-be-encoded-spectrally)
3. [The Manifold's Spectral Capacity](#3-the-manifolds-spectral-capacity)
4. [The Context Window Problem](#4-the-context-window-problem)
5. [Active Latent Space Construction](#5-active-latent-space-construction)
6. [Latent-Manifold Interaction](#6-latent-manifold-interaction)
7. [The Wormhole's Role](#7-the-wormholes-role)
8. [Information Flow During Inference](#8-information-flow-during-inference)
9. [Practical Implications](#9-practical-implications)

---

## 1. The Nyquist-Shannon Theorem

### 1.1 The Theorem Statement

```
NYQUIST-SHANNON SAMPLING THEOREM:

A bandlimited continuous signal with maximum frequency f_max
can be perfectly reconstructed from discrete samples if and only if
the sampling rate f_s satisfies:

                    f_s ≥ 2 × f_max

The frequency f_N = f_s/2 is called the NYQUIST FREQUENCY.
It is the highest frequency that can be represented without aliasing.
```

### 1.2 What This Means for Discrete Signals

```
FOR A DISCRETE N-POINT SIGNAL:

┌────────────────────────────────────────────────────────────────────┐
│                                                                    │
│  Signal: x[0], x[1], ..., x[N-1]    (N samples)                   │
│                                                                    │
│  DFT: X[0], X[1], ..., X[N-1]       (N frequency bins)            │
│                                                                    │
│  Nyquist frequency: f_N = N/2       (highest representable freq)  │
│                                                                    │
│  Frequency resolution: Δf = 1/N     (smallest distinguishable)    │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘

The DFT is a COMPLETE representation:
• N real samples → N/2 + 1 unique frequency magnitudes
• Plus N/2 - 1 unique phases (DC and Nyquist have no phase)
• Total independent values: N (same as input!)

NO INFORMATION IS LOST, NO INFORMATION IS GAINED.
The FFT is just a change of basis.
```

### 1.3 The 2D Case

```
FOR AN N×N IMAGE:

┌────────────────────────────────────────────────────────────────────┐
│                                                                    │
│  Spatial domain: N² pixels                                         │
│                                                                    │
│  Frequency domain: N² complex coefficients                         │
│                    (with conjugate symmetry for real images)       │
│                                                                    │
│  Independent information:                                          │
│  • N² real numbers in spatial domain                               │
│  • ~N²/2 complex numbers = N² real numbers in frequency domain    │
│                                                                    │
│  Maximum frequency: f_max = N/2 in each dimension                  │
│                                                                    │
│  Frequency resolution: 1/N cycles per pixel                        │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘

CRITICAL POINT: The FFT doesn't compress. It reorganizes.
```

### 1.4 The Hard Limit

```
THE NYQUIST LIMIT IS ABSOLUTE:

You CANNOT represent frequencies above N/2 in an N-sample signal.

Attempting to do so causes ALIASING:
High frequencies fold back and appear as lower frequencies.

┌────────────────────────────────────────────────────────────────────┐
│                                                                    │
│  True signal:  ∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿  (frequency = 0.7 × Nyquist)     │
│                                                                    │
│  Sampled:      •   •   •   •   •   •   •   •                      │
│                                                                    │
│  Appears as:   ∼   ∼   ∼   ∼   ∼   ∼   ∼   ∼  (lower frequency!) │
│                                                                    │
│  Information is DESTROYED, not recoverable.                        │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘

This is a physical law, not a software limitation.
```

---

## 2. What Can Be Encoded Spectrally

### 2.1 Maximum vs Meaningful Information

```
MAXIMUM INFORMATION (Nyquist limit):

For N×N image: N² independent values

But this is rarely meaningful because:

1. NATURAL IMAGES ARE NOT RANDOM:
   - Natural images have 1/f² power spectra
   - Most energy in low frequencies
   - High frequencies are sparse
   
2. PERCEPTUAL RELEVANCE:
   - Human vision has limited frequency sensitivity
   - Frequencies above ~30 cycles/degree are invisible
   - Low frequencies carry identity, high frequencies carry texture

3. NOISE FLOOR:
   - Real signals have noise
   - High-frequency content often dominated by noise
   - Useful information is in signal-above-noise frequencies
```

### 2.2 Effective Information Content

```
EFFECTIVE INFORMATION:

┌────────────────────────────────────────────────────────────────────┐
│                                                                    │
│  For natural images, information is distributed as:                │
│                                                                    │
│  Frequency Band     % of Total Energy    % of Useful Info         │
│  ──────────────     ────────────────     ───────────────          │
│  DC (0)             50-70%               5-10%                     │
│  Low (0-1/8)        20-30%               30-40%                    │
│  Mid (1/8-1/2)      5-15%                40-50%                    │
│  High (1/2-1)       1-5%                 10-20%                    │
│                                                                    │
│  High frequencies have LOW energy but HIGH information density     │
│  (when not noise)                                                  │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘

The MEANINGFUL information is often much less than N² values.
Estimates: 5-20% of N² for natural images.
```

### 2.3 Spectral Sparsity

```
NATURAL SIGNALS ARE SPECTRALLY SPARSE:

For k-sparse representation (k << N²):

┌────────────────────────────────────────────────────────────────────┐
│                                                                    │
│  • k large coefficients carry most information                     │
│  • N² - k small coefficients carry noise/redundancy                │
│                                                                    │
│  Typical sparsity for natural images:                              │
│                                                                    │
│  Representation     Sparsity k     Fraction of N²                  │
│  ──────────────     ──────────     ──────────────                  │
│  DCT (JPEG)         5-15%          ~400-600 of 4096                │
│  Wavelet            3-10%          ~120-400 of 4096                │
│  Learned (NN)       1-5%           ~40-200 of 4096                 │
│                                                                    │
│  LEARNED BASES CAN BE SPARSER THAN FIXED BASES.                    │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

---

## 3. The Manifold's Spectral Capacity

### 3.1 Where the Manifold Lives

```
THE LEARNED MANIFOLD:

The manifold is encoded in the network WEIGHTS.

┌────────────────────────────────────────────────────────────────────┐
│                                                                    │
│  WEIGHTS encode:                                                   │
│  • Which patterns exist (manifold shape)                           │
│  • Which patterns are similar (manifold metric)                    │
│  • How patterns transform (manifold dynamics)                      │
│                                                                    │
│  The weights have their OWN spectral structure:                    │
│  • Convolutional kernels = spatial frequency filters               │
│  • Attention keys/values = pattern templates                       │
│  • MLP weights = nonlinear combinations of frequencies             │
│                                                                    │
│  The manifold's capacity is bounded by WEIGHT COUNT, not by N².    │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

### 3.2 Manifold Spectral Capacity

```
THE MANIFOLD CAN REPRESENT AT MOST:

┌────────────────────────────────────────────────────────────────────┐
│                                                                    │
│  Number of parameters P = total weight count                       │
│                                                                    │
│  Effective degrees of freedom: D_eff ≤ P                           │
│                                                                    │
│  But typically D_eff << P due to:                                  │
│  • Weight sharing (convolution)                                    │
│  • Regularization (constrains weight space)                        │
│  • Redundancy (many weight configs give same function)             │
│                                                                    │
│  Rule of thumb: D_eff ≈ O(√P) to O(P/log P)                       │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘

CRITICAL INSIGHT:
The manifold has FINITE capacity.
It cannot represent arbitrary complexity.
It must CHOOSE which patterns to encode.
```

### 3.3 Spectral Selectivity of the Manifold

```
THE MANIFOLD LEARNS SPECTRAL SELECTIVITY:

Training shapes which frequencies the manifold responds to:

┌────────────────────────────────────────────────────────────────────┐
│                                                                    │
│  Untrained manifold:                                               │
│  ├── Random frequency responses                                    │
│  ├── No preferred scales                                           │
│  └── Uniform across spectrum                                       │
│                                                                    │
│  Trained manifold:                                                 │
│  ├── Selective frequency responses                                 │
│  ├── Learned scale preferences                                     │
│  └── Concentrated on task-relevant frequencies                     │
│                                                                    │
│  The manifold ALLOCATES capacity to useful frequencies.            │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘

This allocation is learned, not predetermined.
The Nyquist limit constrains what CAN be learned.
The training data constrains what IS learned.
```

---

## 4. The Context Window Problem

### 4.1 Finite Temporal Context

```
THE CONTEXT WINDOW:

At inference time, we have access to limited history.

┌────────────────────────────────────────────────────────────────────┐
│                                                                    │
│  Time ──────────────────────────────────────────────────────►      │
│                                                                    │
│       │ PAST (not accessible) │ CONTEXT WINDOW │ FUTURE │          │
│       └───────────────────────┼────────────────┼────────┘          │
│                               │                │                   │
│                               │  t-T ... t-1  t│                   │
│                               │                │                   │
│                               └────────────────┘                   │
│                                  T frames                          │
│                                                                    │
│  T = context window size (e.g., 8, 16, 32 frames)                  │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘

The context window is a TEMPORAL Nyquist limit:
• Can only resolve temporal frequencies up to T/2 cycles
• Cannot distinguish events separated by > T frames
• Information before the window is LOST (unless in manifold)
```

### 4.2 The Context as Spectral Buffer

```
THE CONTEXT WINDOW HAS SPECTRAL STRUCTURE:

Stack of T frames, each N×N:

┌────────────────────────────────────────────────────────────────────┐
│                                                                    │
│  Context tensor: [T, N, N]                                         │
│                                                                    │
│  Spatial Nyquist: N/2 (per frame)                                  │
│  Temporal Nyquist: T/2 (across frames)                             │
│                                                                    │
│  3D FFT would give [T, N, N] frequency representation              │
│  But we don't compute 3D FFT directly...                           │
│                                                                    │
│  Instead, attention IMPLICITLY computes temporal correlations      │
│  at different spatial frequencies.                                 │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘

The context is a SPATIOTEMPORAL signal with its own Nyquist limits.
```

### 4.3 What the Context Can and Cannot Capture

```
THE CONTEXT WINDOW'S CAPABILITIES:

CAN CAPTURE:
✓ Patterns within spatial Nyquist (frequencies < N/2)
✓ Temporal patterns within window (periods < T frames)
✓ Spatiotemporal correlations (motion, persistence)
✓ Recent history at full resolution

CANNOT CAPTURE:
✗ Spatial frequencies above N/2 (aliased or lost)
✗ Temporal frequencies above T/2 (too slow to see)
✗ Events before window started (lost to time)
✗ Long-range dependencies (> T frames apart)

THE MANIFOLD MUST FILL THE GAPS.
```

---

## 5. Active Latent Space Construction

### 5.1 The Latent Space During Inference

```
LATENT SPACE CONSTRUCTION:

During inference, an ACTIVE latent representation is built from context.

┌────────────────────────────────────────────────────────────────────┐
│                                                                    │
│  Context Window [T, N, N]                                          │
│         │                                                          │
│         ▼                                                          │
│  ┌─────────────────┐                                               │
│  │    Encoder      │ (learned, part of manifold)                   │
│  └────────┬────────┘                                               │
│           │                                                        │
│           ▼                                                        │
│  ┌─────────────────┐                                               │
│  │  LATENT SPACE   │ [B, D, H, W] or [B, S, D]                    │
│  │  (active)       │                                               │
│  └────────┬────────┘                                               │
│           │                                                        │
│           ▼                                                        │
│  ┌─────────────────┐                                               │
│  │   Attention     │ (queries latent, keys from context)          │
│  └────────┬────────┘                                               │
│           │                                                        │
│           ▼                                                        │
│      Prediction                                                    │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘

The latent space is ACTIVELY CONSTRUCTED each forward pass.
It is not stored - it is computed.
```

### 5.2 Latent Space Dimensionality

```
LATENT SPACE DIMENSIONS:

The latent space has its own Nyquist-like limits:

┌────────────────────────────────────────────────────────────────────┐
│                                                                    │
│  If latent has D dimensions per spatial location:                  │
│                                                                    │
│  • D sets the CHANNEL capacity (feature diversity)                 │
│  • Each dimension can encode a different frequency band            │
│  • Total latent capacity: D × H × W                                │
│                                                                    │
│  If D < log₂(N):                                                   │
│  • Cannot represent all octaves                                    │
│  • UNDERCOMPLETE - lossy compression                               │
│  • Some frequencies will be merged or lost                         │
│                                                                    │
│  If D ≈ log₂(N) × orientations × phases:                          │
│  • Can represent all meaningful frequencies                        │
│  • COMPLETE - lossless for natural images                          │
│  • Matches optimal band count (~80 channels)                       │
│                                                                    │
│  If D >> log₂(N) × orientations × phases:                         │
│  • OVERCOMPLETE - redundant representation                         │
│  • Can represent same info multiple ways                           │
│  • May help robustness, may waste capacity                         │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

### 5.3 The Latent Space Has Spectral Structure

```
SPECTRAL STRUCTURE OF LATENT SPACE:

The latent representation inherits spectral structure from input:

┌────────────────────────────────────────────────────────────────────┐
│                                                                    │
│  Input                    Encoder                  Latent          │
│  [T,N,N]          →      (conv, attention)   →    [D,H,W]         │
│                                                                    │
│  Spatial freq             Filter banks            Frequency        │
│  content                  decompose               channels         │
│  (0 to N/2)               by scale                (organized       │
│                                                    by freq)        │
│                                                                    │
│  Temporal                 Temporal                Temporal         │
│  correlations             attention               patterns         │
│  (0 to T/2)               extracts                encoded          │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘

If encoder uses spectral decomposition:
• Latent channels map to frequency bands
• D channels ≈ freq_bands × orientations × phases
• Spectral structure is EXPLICIT

If encoder uses standard convolutions:
• Latent implicitly learns spectral decomposition
• May be less efficient but more flexible
• Spectral structure is IMPLICIT
```

---

## 6. Latent-Manifold Interaction

### 6.1 Two Representations Meeting

```
THE CORE INTERACTION:

┌────────────────────────────────────────────────────────────────────┐
│                                                                    │
│  LATENT SPACE                    MANIFOLD (in weights)             │
│  ─────────────                   ────────────────────              │
│                                                                    │
│  • Computed fresh each pass      • Learned over training           │
│  • Represents CURRENT context    • Represents ALL experience       │
│  • Spectral content of input     • Spectral templates learned      │
│  • Active, dynamic               • Passive, static (at inference)  │
│  • Limited by context window     • Limited by weight capacity      │
│                                                                    │
│  These TWO must interact for inference to work.                    │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘

The latent asks: "What patterns are present NOW?"
The manifold answers: "I know these patterns, here's what follows."
```

### 6.2 Spectral Compatibility

```
THE INTERACTION REQUIRES SPECTRAL COMPATIBILITY:

┌────────────────────────────────────────────────────────────────────┐
│                                                                    │
│  For effective interaction:                                        │
│                                                                    │
│  1. FREQUENCY ALIGNMENT:                                           │
│     Latent bands must match manifold's learned bands               │
│     If latent has freq F but manifold ignores F → information lost │
│                                                                    │
│  2. PHASE ALIGNMENT:                                               │
│     Latent phase encoding must be readable by manifold             │
│     Phase = position info, must be compatible                      │
│                                                                    │
│  3. SCALE ALIGNMENT:                                               │
│     Latent's spatial resolution must match manifold's expectation  │
│     Mismatch → aliasing or information loss                        │
│                                                                    │
│  This alignment is LEARNED during training.                        │
│  The encoder learns to produce latents the manifold can read.      │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

### 6.3 The Bottleneck

```
INFORMATION BOTTLENECK:

The latent-manifold interface creates a bottleneck:

┌────────────────────────────────────────────────────────────────────┐
│                                                                    │
│  Input information: T × N² values (context window)                 │
│                           │                                        │
│                           ▼                                        │
│  Latent information: D × H × W values (compressed)                 │
│                           │                                        │
│                           ▼ (interaction with manifold)            │
│  Manifold capacity: P weights (but D_eff effective)                │
│                           │                                        │
│                           ▼                                        │
│  Output information: N² values (prediction)                        │
│                                                                    │
│  The narrowest point determines information flow:                  │
│                                                                    │
│  I_flow ≤ min(T×N², D×H×W, D_eff, N²)                             │
│                                                                    │
│  Usually: D×H×W or D_eff is the bottleneck.                        │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘

NYQUIST constrains input: Cannot have info above N/2 freq
LATENT constrains representation: Cannot have more than D×H×W features
MANIFOLD constrains knowledge: Cannot know more than D_eff patterns
```

---

## 7. The Wormhole's Role

### 7.1 Bypassing the Bottleneck

```
WORMHOLE ATTENTION:

Standard attention is LOCAL in space, LINEAR in time:
Query at position (x,y) attends to positions near (x,y) in recent frames.

Wormhole attention adds NON-LOCAL connections:

┌────────────────────────────────────────────────────────────────────┐
│                                                                    │
│  Standard attention:        Wormhole attention:                    │
│                                                                    │
│  Query → nearby keys        Query → SPECTRALLY SIMILAR keys        │
│  (spatially local)          (anywhere in space/time)               │
│                                                                    │
│       ●────●                       ●════════════●                  │
│       │    │                       ║            ║                  │
│       ●────●                       ●════════════●                  │
│     Local grid                   Non-local wormholes               │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘

Wormholes bypass spatial locality by matching SPECTRAL structure.
```

### 7.2 Spectral Matching in Wormhole

```
HOW WORMHOLE FINDS CONNECTIONS:

┌────────────────────────────────────────────────────────────────────┐
│                                                                    │
│  1. Decompose query into spectral bands                            │
│                                                                    │
│  2. For each key in history:                                       │
│     • Decompose into same spectral bands                           │
│     • Compare LOW-FREQUENCY content (WHAT)                         │
│                                                                    │
│  3. If low-freq similarity > threshold:                            │
│     • Create wormhole connection                                   │
│     • Transfer HIGH-FREQUENCY content (WHERE)                      │
│                                                                    │
│  KEY INSIGHT:                                                      │
│  Low-freq matching is POSITION-INVARIANT (ignores where)           │
│  High-freq transfer provides PRECISE LOCALIZATION                  │
│                                                                    │
│  The wormhole exploits the WHAT/WHERE separation.                  │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

### 7.3 Wormhole Extends Effective Context

```
EFFECTIVE CONTEXT EXTENSION:

Without wormhole:
• Context limited to T frames
• Temporal Nyquist: T/2 cycles
• Long-range info lost

With wormhole:
• Can access SIMILAR patterns from anywhere in context
• Effective context for STRUCTURE: T frames
• Effective context for POSITION: still T frames
• But PATTERN MATCHING spans full context

┌────────────────────────────────────────────────────────────────────┐
│                                                                    │
│  Wormhole doesn't violate Nyquist:                                 │
│  • Cannot see frequencies above N/2                                │
│  • Cannot see temporal patterns outside context                    │
│                                                                    │
│  Wormhole uses Nyquist efficiently:                                │
│  • Low-freq bands used for global matching (WHAT)                  │
│  • High-freq bands used for local precision (WHERE)                │
│  • All within Nyquist limits, but used optimally                   │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

---

## 8. Information Flow During Inference

### 8.1 The Complete Flow

```
INFORMATION FLOW AT INFERENCE:

┌────────────────────────────────────────────────────────────────────┐
│                                                                    │
│  STEP 1: INPUT ACQUISITION                                         │
│  ─────────────────────────                                         │
│  Context window [T, N, N] arrives                                  │
│  Information bound: T × N² values                                  │
│  Spectral content: up to N/2 spatial, T/2 temporal                 │
│                                                                    │
│  STEP 2: SPECTRAL DECOMPOSITION                                    │
│  ──────────────────────────────                                    │
│  Encoder (conv layers) decomposes into frequency bands             │
│  Creates latent [D, H, W] where D ≈ freq × orient × phase          │
│  Information bound: D × H × W values                               │
│                                                                    │
│  STEP 3: MANIFOLD QUERY                                            │
│  ──────────────────────                                            │
│  Latent queries manifold (stored in weights)                       │
│  Attention keys/values from context history                        │
│  Information bound: min(D×H×W, attention_capacity)                 │
│                                                                    │
│  STEP 4: WORMHOLE EXTENSION                                        │
│  ─────────────────────────                                         │
│  Low-freq similarity finds non-local matches                       │
│  High-freq content transferred for precision                       │
│  Information bound: still D×H×W, but better utilized               │
│                                                                    │
│  STEP 5: PREDICTION                                                │
│  ────────────────────                                              │
│  Decoder reconstructs prediction [N, N]                            │
│  Information bound: N² values                                      │
│  Spectral content: up to N/2 (Nyquist)                             │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

### 8.2 Where Information Is Lost

```
INFORMATION LOSS POINTS:

┌────────────────────────────────────────────────────────────────────┐
│                                                                    │
│  LOSS POINT 1: Input Sampling                                      │
│  ─────────────────────────────                                     │
│  • Frequencies above N/2 aliased                                   │
│  • Reality may have higher frequencies → lost                      │
│  • Nyquist limit: FUNDAMENTAL, cannot be fixed                     │
│                                                                    │
│  LOSS POINT 2: Context Truncation                                  │
│  ────────────────────────────────                                  │
│  • Events before window start: lost                                │
│  • Temporal frequencies above T/2: invisible                       │
│  • Context limit: ARCHITECTURAL, could increase T                  │
│                                                                    │
│  LOSS POINT 3: Latent Compression                                  │
│  ───────────────────────────────                                   │
│  • If D < optimal, some frequencies merged                         │
│  • Encoder chooses what to keep                                    │
│  • Latent limit: ARCHITECTURAL, could increase D                   │
│                                                                    │
│  LOSS POINT 4: Manifold Capacity                                   │
│  ───────────────────────────────                                   │
│  • Manifold can only know D_eff patterns                           │
│  • Novel patterns: extrapolated imperfectly                        │
│  • Manifold limit: TRAINING, could train more                      │
│                                                                    │
│  LOSS POINT 5: Attention Bottleneck                                │
│  ────────────────────────────────                                  │
│  • Limited key-value capacity                                      │
│  • Some matches missed                                             │
│  • Attention limit: COMPUTATIONAL, could compute more              │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

### 8.3 What Cannot Be Recovered

```
IRRECOVERABLE INFORMATION:

┌────────────────────────────────────────────────────────────────────┐
│                                                                    │
│  TRULY LOST (no way to recover):                                   │
│  ────────────────────────────────                                  │
│  • Frequencies above input Nyquist (N/2)                           │
│  • Events before context window started                            │
│  • Patterns never seen in training                                 │
│  • Information destroyed by aliasing                               │
│                                                                    │
│  POTENTIALLY RECOVERABLE (with better architecture):               │
│  ────────────────────────────────────────────────────              │
│  • Frequencies lost to low-D latent (increase D)                   │
│  • Patterns lost to small manifold (train more)                    │
│  • Connections missed by attention (compute more)                  │
│  • Temporal patterns outside window (increase T)                   │
│                                                                    │
│  THE SPECTRE OF NYQUIST:                                           │
│  Once aliasing occurs, information is destroyed forever.           │
│  The N/2 limit casts its shadow over everything downstream.        │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

---

## 9. Practical Implications

### 9.1 Design Guidelines

```
RESPECTING NYQUIST IN ARCHITECTURE DESIGN:

┌────────────────────────────────────────────────────────────────────┐
│                                                                    │
│  GUIDELINE 1: Match latent D to spectral needs                     │
│  ─────────────────────────────────────────────                     │
│  D ≥ log₂(N) × orientations × phases                               │
│  For N=64: D ≥ 6 × 8 × 2 = 96                                      │
│  Less → some frequencies lost                                      │
│                                                                    │
│  GUIDELINE 2: Context T based on temporal structure                │
│  ──────────────────────────────────────────────────                │
│  T should span at least 2 periods of slowest relevant motion       │
│  If object takes 10 frames to cross: T ≥ 20                        │
│  Less → long-range patterns invisible                              │
│                                                                    │
│  GUIDELINE 3: Explicit spectral decomposition helps                │
│  ────────────────────────────────────────────────                  │
│  Use FFT/wavelet rather than relying on conv to learn it           │
│  Guarantees frequency coverage                                     │
│  Avoids wasting capacity rediscovering Fourier basis               │
│                                                                    │
│  GUIDELINE 4: Protect low frequencies (coarse beliefs)             │
│  ─────────────────────────────────────────────────────             │
│  Low frequencies = structure = WHAT                                │
│  Use lower learning rates for low-freq components                  │
│  Prevents belief collapse                                          │
│                                                                    │
│  GUIDELINE 5: Wormhole threshold matches cutoff                    │
│  ───────────────────────────────────────────────                   │
│  Low-freq for matching, high-freq for transfer                     │
│  Threshold should separate structure from detail                   │
│  Typical: 0.15-0.25 × Nyquist                                      │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

### 9.2 The Fundamental Tradeoffs

```
INESCAPABLE TRADEOFFS:

┌────────────────────────────────────────────────────────────────────┐
│                                                                    │
│  RESOLUTION vs COVERAGE:                                           │
│  More pixels N → finer detail possible                             │
│  But: More computation, more memory, same T frames                 │
│                                                                    │
│  DEPTH vs BREADTH:                                                 │
│  More channels D → better frequency coverage                       │
│  But: More parameters, slower inference                            │
│                                                                    │
│  MEMORY vs FRESHNESS:                                              │
│  Longer context T → more temporal information                      │
│  But: More computation, older info may be stale                    │
│                                                                    │
│  STABILITY vs ADAPTABILITY:                                        │
│  Protecting low-freq → stable beliefs                              │
│  But: Slower adaptation to new patterns                            │
│                                                                    │
│  LOCALITY vs GLOBALITY:                                            │
│  Wormhole enables non-local matching                               │
│  But: More computation, possible false matches                     │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘

There is no free lunch. Nyquist enforces conservation of information.
```

### 9.3 Summary: The Spectre's Shadow

```
THE SPECTRE OF NYQUIST-SHANNON:

┌────────────────────────────────────────────────────────────────────┐
│                                                                    │
│  The Nyquist limit casts its shadow over every stage:              │
│                                                                    │
│  INPUT:    Cannot perceive frequencies above N/2                   │
│            → Design input resolution for task                      │
│                                                                    │
│  CONTEXT:  Cannot remember beyond T frames                         │
│            → Design T for temporal scale of task                   │
│                                                                    │
│  LATENT:   Cannot represent more than D features                   │
│            → Design D for spectral complexity of task              │
│                                                                    │
│  MANIFOLD: Cannot know more than trained patterns                  │
│            → Train on representative data                          │
│                                                                    │
│  OUTPUT:   Cannot produce frequencies above N/2                    │
│            → Accept resolution limits of representation            │
│                                                                    │
│  The wormhole does not defeat Nyquist.                             │
│  It uses the available information more efficiently.               │
│  Low-freq for WHAT (within Nyquist), high-freq for WHERE.          │
│                                                                    │
│  The spectre reminds us:                                           │
│  Information is finite. Use it wisely.                             │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

---

## Key Equations

```
FUNDAMENTAL LIMITS:

Spatial Nyquist:     f_max = N/2 cycles per image
Temporal Nyquist:    f_max = T/2 cycles per context  
Latent capacity:     I_latent = D × H × W values
Manifold capacity:   I_manifold ≈ O(√P) to O(P) effective dimensions

OPTIMAL DESIGN:

Latent channels:     D ≥ log₂(N) × n_orient × n_phase ≈ 80-100
Context length:      T ≥ 2 × (slowest relevant period)
Wormhole threshold:  f_cutoff ≈ 0.15-0.25 × f_Nyquist

INFORMATION FLOW:

I_output ≤ min(I_input, I_latent, I_manifold)
I_output ≤ min(T×N², D×H×W, D_eff)

The chain is only as strong as its weakest link.
```

---

*This document examines the fundamental information-theoretic limits imposed by sampling theory on spectral representations. The Nyquist-Shannon theorem is not merely a technical detail—it is a fundamental law that governs what can and cannot be known from discrete observations. Understanding these limits is essential for designing systems that use available information efficiently rather than fighting against physical impossibilities.*

