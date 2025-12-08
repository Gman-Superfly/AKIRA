# Observability Embeddings: A Laboratory for Spectral Attention

## Visualizing Belief Dynamics, Error Propagation, and System State

**Oscar Goldman — Shogu Research Group @ Datamutant.ai**

---

## Table of Contents

1. [Introduction: The Observability Laboratory](#1-introduction-the-observability-laboratory)
2. [What We Embed](#2-what-we-embed)
3. [Dimensionality Reduction Methods](#3-dimensionality-reduction-methods)
4. [Trajectory Tracking](#4-trajectory-tracking)
5. [Belief Cloud Visualization](#5-belief-cloud-visualization)
6. [Cross-Component Analysis](#6-cross-component-analysis)
7. [Experimental Protocols](#7-experimental-protocols)
8. [Implementation Notes](#8-implementation-notes)

---

## 1. Introduction: The Observability Laboratory

### 1.1 Purpose

```
THE GOAL:

We are building a LABORATORY to study every part of the AKIRA system.

Not just the outputs—the INTERNALS:
• How belief forms and collapses
• How error propagates through the manifold
• How spectral bands interact
• How attention heads coordinate
• How gradients flow and stabilize
• How the pump cycle manifests in latent space

This is not debugging. This is SCIENCE.
We observe. We measure. We understand.
```

### 1.2 The Core Insight

```
THE WAVEFRONT ERROR IS IN THE MANIFOLD

The error we see in 2D output space is a PROJECTION of a
high-dimensional belief structure in the embedding manifold.

To understand the system, we must:
1. OBSERVE the manifold directly
2. TRACK how structures evolve
3. MEASURE the dynamics of collapse
4. CORRELATE across system components

If we can see the wavefront error collapse in embedding space,
we prove that belief dynamics are REAL, not metaphor.
```

### 1.3 What This Proves

```
SCIENTIFIC CLAIMS TO VALIDATE:

1. BELIEF IS GEOMETRIC
   Error has shape in the manifold, not just magnitude.
   Prediction is belief. Uncertainty has structure.

2. THE PUMP CYCLE IS OBSERVABLE
   Tension → Discharge → Recovery manifests as
   expansion → interference → collapse in embedding space.

3. SPECTRAL BANDS HAVE DIFFERENT DYNAMICS
   Low-freq: slow, smooth embedding evolution
   High-freq: fast, sharp embedding changes

4. INFORMATION IS CONSERVED
   Error moves through the manifold, doesn't appear/disappear.
   Total belief volume approximately constant.

5. COLLAPSE IS A PHASE TRANSITION
   Sudden, discontinuous change in embedding structure.
   Measurable as entropy drop, cluster formation.
```

---

## 2. What We Embed

### 2.1 Attention Head Representations

```
LATENT REPRESENTATIONS FROM EACH ATTENTION HEAD

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  COMPONENT              EMBEDDING                  DIMENSIONS           │
│  ─────────              ─────────                  ──────────           │
│                                                                         │
│  TEMPORAL ATTENTION                                                     │
│  ├─ Query vectors       Q_t[band, position]       [7, H×W, D_attn]     │
│  ├─ Key vectors         K_t[band, position, t]    [7, H×W, T, D_attn]  │
│  ├─ Value vectors       V_t[band, position, t]    [7, H×W, T, D_attn]  │
│  ├─ Attention weights   A_t[band, position, t]    [7, H×W, T]          │
│  └─ Output              O_t[band, position]       [7, H×W, D_out]      │
│                                                                         │
│  NEIGHBOR ATTENTION                                                     │
│  ├─ Query vectors       Q_n[band, position]       [7, H×W, D_attn]     │
│  ├─ Key vectors         K_n[band, pos, neighbor]  [7, H×W, N_nb, D]    │
│  ├─ Value vectors       V_n[band, pos, neighbor]  [7, H×W, N_nb, D]    │
│  ├─ Attention weights   A_n[band, pos, neighbor]  [7, H×W, N_nb]       │
│  └─ Output              O_n[band, position]       [7, H×W, D_out]      │
│                                                                         │
│  WORMHOLE ATTENTION                                                     │
│  ├─ Query vectors       Q_w[position]             [H×W, D_attn]        │
│  ├─ Key vectors         K_w[history_position]     [T×H×W, D_attn]      │
│  ├─ Similarity matrix   S_w[query, key]           [H×W, T×H×W]         │
│  ├─ Top-k indices       I_w[position, k]          [H×W, K]             │
│  ├─ Top-k similarities  Sim_w[position, k]        [H×W, K]             │
│  ├─ Gate mask           G_w[position, k]          [H×W, K] (boolean)   │
│  ├─ Attention weights   A_w[position, k]          [H×W, K]             │
│  └─ Output              O_w[position]             [H×W, D_out]         │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Error Signals

```
ERROR SIGNAL AT EACH SPATIAL POSITION

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  ERROR TYPE             EMBEDDING                  DIMENSIONS           │
│  ──────────             ─────────                  ──────────           │
│                                                                         │
│  SPATIAL ERROR                                                          │
│  ├─ Per-pixel error     E_pix[x, y]               [H, W]               │
│  ├─ Error gradient      ∇E_pix[x, y]              [H, W, 2]            │
│  ├─ Error Laplacian     ∇²E_pix[x, y]             [H, W]               │
│  └─ Error moments       μ, σ², skew, kurt         [4]                  │
│                                                                         │
│  SPECTRAL ERROR                                                         │
│  ├─ Per-band error      E_band[band, x, y]        [7, H, W]            │
│  ├─ Band-wise magnitude |E_band|[band]            [7]                  │
│  ├─ Cross-band corr     Corr[band_i, band_j]      [7, 7]               │
│  └─ Spectral centroid   f_c                       [1]                  │
│                                                                         │
│  TEMPORAL ERROR                                                         │
│  ├─ Error history       E_hist[t, x, y]           [T_err, H, W]        │
│  ├─ Error velocity      dE/dt[x, y]               [H, W]               │
│  ├─ Error acceleration  d²E/dt²[x, y]             [H, W]               │
│  └─ Error persistence   τ_err[x, y]               [H, W]               │
│                                                                         │
│  FREQUENCY-DOMAIN ERROR                                                 │
│  ├─ Error FFT           FFT(E)[u, v]              [H, W] (complex)     │
│  ├─ Error power spec    |FFT(E)|²[u, v]           [H, W]               │
│  ├─ Error phase         ∠FFT(E)[u, v]             [H, W]               │
│  └─ Radial power        P(r)                      [N/2]                │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.3 Spectral Band Activations

```
SPECTRAL BAND ACTIVATIONS

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  BAND                   EMBEDDING                  DIMENSIONS           │
│  ────                   ─────────                  ──────────           │
│                                                                         │
│  PER-BAND STATE                                                         │
│  ├─ Band content        X_band[band, x, y]        [7, H, W]            │
│  ├─ Band magnitude      |X_band|[band]            [7]                  │
│  ├─ Band phase (avg)    ∠X_band[band]             [7]                  │
│  ├─ Band entropy        H_band[band]              [7]                  │
│  └─ Band sparsity       S_band[band]              [7]                  │
│                                                                         │
│  CROSS-BAND DYNAMICS                                                    │
│  ├─ Band correlation    Corr[i, j]                [7, 7]               │
│  ├─ Band coherence      Coh[i, j, f]              [7, 7, F]            │
│  ├─ Energy flow         Flow[i → j]               [7, 7]               │
│  └─ Phase coupling      Phase_sync[i, j]          [7, 7]               │
│                                                                         │
│  TEMPORAL EVOLUTION                                                     │
│  ├─ Band history        X_band_hist[t, band]      [T, 7, H, W]         │
│  ├─ Band velocity       dX_band/dt[band]          [7, H, W]            │
│  └─ Band EMA            EMA_band[band]            [7, H, W]            │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.4 Gradient and Weight Dynamics

```
GRADIENT AND WEIGHT OBSERVABLES

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  GRADIENTS                                                              │
│  ├─ Per-layer gradient      ∇W[layer]             [L, D_layer]         │
│  ├─ Gradient norm           ‖∇W‖[layer]           [L]                  │
│  ├─ Gradient direction      ∇W/‖∇W‖[layer]        [L, D_layer]         │
│  ├─ Gradient alignment      cos(∇W_t, ∇W_{t-1})   [L]                  │
│  ├─ Per-band gradient       ∇W_band[band, layer]  [7, L, D]            │
│  └─ Gradient SNR            signal/noise ratio    [L]                  │
│                                                                         │
│  WEIGHTS                                                                │
│  ├─ Weight magnitude        ‖W‖[layer]            [L]                  │
│  ├─ Weight spectrum         SVD(W)[layer]         [L, min(D)]          │
│  ├─ Effective rank          rank_eff[layer]       [L]                  │
│  ├─ Weight change           ΔW[layer]             [L, D_layer]         │
│  └─ Weight velocity         dW/dt[layer]          [L, D_layer]         │
│                                                                         │
│  FISHER INFORMATION                                                     │
│  ├─ Fisher diagonal         F_diag[param]         [N_params]           │
│  ├─ Natural gradient        F⁻¹∇L                 [N_params]           │
│  └─ Information geometry    metric tensor         [D, D]               │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.5 Belief State Observables

```
BELIEF STATE OBSERVABLES

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  BELIEF DISTRIBUTION                                                    │
│  ├─ Prediction mean         ŷ[x, y]               [H, W]               │
│  ├─ Prediction variance     Var(ŷ)[x, y]          [H, W] (if ensemble) │
│  ├─ Belief entropy          H(b)[x, y]            [H, W]               │
│  ├─ Belief concentration    max(b)[x, y]          [H, W]               │
│  └─ Belief modes            modes[x, y, k]        [H, W, K]            │
│                                                                         │
│  ATTENTION AS BELIEF                                                    │
│  ├─ Attention entropy       H(A)[head, pos]       [N_heads, H×W]       │
│  ├─ Attention sharpness     max(A)/mean(A)        [N_heads, H×W]       │
│  ├─ Attention spread        effective support     [N_heads, H×W]       │
│  └─ Cross-head agreement    correlation           [N_heads, N_heads]   │
│                                                                         │
│  PUMP CYCLE INDICATORS                                                  │
│  ├─ Tension index           T = H(b) - H_baseline [1]                  │
│  ├─ Discharge trigger       ∂H/∂t < threshold     [1] (boolean)        │
│  ├─ Recovery rate           dH/dt after collapse  [1]                  │
│  └─ Cycle phase             φ ∈ {tension, crit, discharge, recovery}   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.6 Wormhole-Specific Observables

```
WORMHOLE ATTENTION OBSERVABLES

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  CONNECTION STRUCTURE                                                   │
│  ├─ Active connections      N_active[position]    [H×W]                │
│  ├─ Connection sparsity     N_active / N_possible [1]                  │
│  ├─ Connection graph        adjacency matrix      [H×W, T×H×W]         │
│  ├─ Temporal reach          avg time distance     [H×W]                │
│  └─ Spatial reach           avg spatial distance  [H×W]                │
│                                                                         │
│  SIMILARITY LANDSCAPE                                                   │
│  ├─ Similarity histogram    hist(sim)             [N_bins]             │
│  ├─ Threshold margin        sim - threshold       [H×W, K]             │
│  ├─ Near-threshold count    |margin| < ε          [1]                  │
│  └─ Similarity evolution    sim_history           [T, H×W, K]          │
│                                                                         │
│  INFORMATION TELEPORTATION                                              │
│  ├─ Source positions        where connections from [H×W, K, 3]         │
│  ├─ Information origin      temporal origin       [H×W, K]             │
│  ├─ Pattern recurrence      same pattern matches  [patterns]           │
│  └─ Teleportation distance  hops saved            [H×W]                │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.7 History Buffer State

```
HISTORY BUFFER OBSERVABLES

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  BUFFER STATE                                                           │
│  ├─ Buffer contents         X_hist[t, band, x, y] [T, 7, H, W]         │
│  ├─ Buffer age              age[t]                [T]                  │
│  ├─ Buffer utilization      which slots filled    [T] (boolean)        │
│  └─ Buffer diversity        variance across t     [7, H, W]            │
│                                                                         │
│  TEMPORAL PATTERNS                                                      │
│  ├─ Autocorrelation         R(τ)[band, x, y]      [T, 7, H, W]         │
│  ├─ Temporal frequency      dominant freq         [7, H, W]            │
│  ├─ Phase velocity          dφ/dt[band, x, y]     [7, H, W]            │
│  └─ Periodicity score       how periodic          [7, H, W]            │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.8 Meta-Observables

```
META-LEVEL OBSERVABLES (Observing the Observation)

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  SYSTEM-WIDE METRICS                                                    │
│  ├─ Total loss              L                     [1]                  │
│  ├─ Per-band loss           L_band[band]          [7]                  │
│  ├─ Loss velocity           dL/dt                 [1]                  │
│  ├─ Loss acceleration       d²L/dt²               [1]                  │
│  └─ Loss spectrum           FFT(L_history)        [T_loss]             │
│                                                                         │
│  INFORMATION FLOW                                                       │
│  ├─ Mutual info (bands)     I(band_i; band_j)     [7, 7]               │
│  ├─ Transfer entropy        TE(i → j)             [N, N]               │
│  ├─ Information bottleneck  IB compression        [layers]             │
│  └─ Effective connectivity  causal influence      [components]         │
│                                                                         │
│  COMPLEXITY MEASURES                                                    │
│  ├─ Kolmogorov complexity   (approximated)        [1]                  │
│  ├─ Neural complexity       integration measure   [1]                  │
│  ├─ Edge of chaos           Lyapunov exponent     [1]                  │
│  └─ Criticality index       power-law exponent    [1]                  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Dimensionality Reduction Methods

### 3.1 Custom Algorithm (Shogu)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  SHOGU DIMENSIONALITY REDUCTION                                        │
│  ─────────────────────────────                                          │
│                                                                         │
│  [PLACEHOLDER FOR CUSTOM ALGORITHM]                                     │
│                                                                         │
│  Properties to document:                                                │
│  • Name: _______________                                                │
│  • Core principle: _______________                                      │
│  • Computational complexity: O(___)                                     │
│  • Preserves: _______________                                           │
│  • Streaming/online capable: Yes / No                                   │
│  • GPU accelerated: Yes / No                                            │
│  • Parameters: _______________                                          │
│                                                                         │
│  Unique advantages over standard methods:                               │
│  • _______________                                                      │
│  • _______________                                                      │
│  • _______________                                                      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Industry Standard Methods

```
STANDARD DIMENSIONALITY REDUCTION METHODS

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  METHOD          TYPE          PRESERVES           USE CASE             │
│  ──────          ────          ─────────           ────────             │
│                                                                         │
│  PCA             Linear        Global variance     Fast overview,       │
│                                                    real-time streaming  │
│                                                                         │
│  t-SNE           Non-linear    Local structure     Cluster discovery,   │
│                                                    offline analysis     │
│                                                                         │
│  UMAP            Non-linear    Local + global      Best general,        │
│                                                    faster than t-SNE    │
│                                                                         │
│  MDS             Non-linear    Pairwise distances  Distance analysis    │
│                                                                         │
│  Isomap          Non-linear    Geodesic distances  Manifold structure   │
│                                                                         │
│  LLE             Non-linear    Local geometry      Curved manifolds     │
│                                                                         │
│  Autoencoders    Learned       Task-specific       Custom preservation  │
│                                                                         │
│  Random Proj.    Linear        Distances (approx)  Ultra-fast, JL lemma │
│                                                                         │
│  Spectral        Non-linear    Graph structure     Network analysis     │
│                                                                         │
│  TriMAP          Non-linear    Triplet constraints Better than t-SNE    │
│                                                                         │
│  PHATE           Non-linear    Diffusion geometry  Trajectory data      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.3 Real-Time vs. Offline

```
STREAMING REQUIREMENTS

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  REAL-TIME (< 50ms per frame):                                         │
│  ├─ PCA (incremental)                                                  │
│  ├─ Random Projection                                                   │
│  ├─ Shogu (if designed for streaming)                                  │
│  └─ Pre-trained Autoencoder                                            │
│                                                                         │
│  NEAR-REAL-TIME (< 500ms per frame):                                   │
│  ├─ UMAP (with pre-computed graph)                                     │
│  └─ Parametric UMAP/t-SNE                                              │
│                                                                         │
│  OFFLINE (batch processing):                                            │
│  ├─ t-SNE (full)                                                       │
│  ├─ UMAP (full)                                                        │
│  ├─ Isomap                                                              │
│  └─ LLE                                                                 │
│                                                                         │
│  HYBRID STRATEGY:                                                       │
│  • Real-time: Fast method (PCA, Shogu)                                 │
│  • Periodic refresh: Better method (UMAP)                              │
│  • Post-hoc analysis: Best method (t-SNE with tuning)                  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.4 Method Selection by Observable Type

```
RECOMMENDED METHODS BY OBSERVABLE

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  OBSERVABLE                  RECOMMENDED           REASON               │
│  ──────────                  ───────────           ──────               │
│                                                                         │
│  Attention weights           UMAP                  Local structure      │
│  Error trajectories          PHATE                 Trajectory-optimized │
│  Belief cloud samples        t-SNE                 Cluster separation   │
│  Gradient directions         PCA                   Linear subspace      │
│  Weight spectra              PCA                   Variance explanation │
│  Wormhole connections        Spectral embedding    Graph structure      │
│  Cross-band dynamics         Shogu + UMAP         Custom + validation  │
│  Real-time monitoring        PCA / Shogu          Speed                 │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Trajectory Tracking

### 4.1 Prediction Embedding Trajectory

```
PREDICTION EMBEDDING TRAJECTORY

Track how the prediction moves through embedding space over time.

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  WHAT TO TRACK:                                                         │
│  • Prediction vector ŷ_t at each timestep                              │
│  • Movement: Δŷ = ŷ_t - ŷ_{t-1}                                        │
│  • Velocity in embedding space                                          │
│  • Acceleration (change in velocity)                                    │
│  • Path length (cumulative movement)                                    │
│                                                                         │
│  VISUALIZATION:                                                         │
│                                                                         │
│     t=0 ●                                                              │
│          ╲                                                             │
│           ● t=1                                                        │
│            │                                                           │
│            ● t=2                                                       │
│           ╱                                                            │
│     t=3 ●                                                              │
│          ╲                                                             │
│           ● t=4                                                        │
│                                                                         │
│  Color by: time, error magnitude, confidence, pump cycle phase         │
│                                                                         │
│  WHAT TO LOOK FOR:                                                      │
│  • Smooth vs. jerky trajectories                                        │
│  • Loops (returning to similar states)                                  │
│  • Convergence (settling into attractor)                                │
│  • Bifurcations (splitting paths)                                       │
│  • Collapse events (sudden jumps)                                       │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Error Embedding Trajectory

```
ERROR EMBEDDING TRAJECTORY

Track how the error structure moves through embedding space.

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  WHAT TO TRACK:                                                         │
│  • Error map E_t[x,y] flattened and embedded                           │
│  • Error spectrum (FFT of error map)                                    │
│  • Error centroid (where error is concentrated)                         │
│  • Error shape descriptors (moments, contours)                          │
│                                                                         │
│  THE WAVEFRONT TRACKING:                                                │
│                                                                         │
│  The wavefront error should show:                                       │
│  1. EXPANSION during tension phase                                      │
│  2. INTERFERENCE patterns during criticality                            │
│  3. CONTRACTION during discharge                                        │
│  4. RE-EXPANSION during recovery                                        │
│                                                                         │
│  ┌───────────────────────────────────────────────────────────────┐     │
│  │ Tension:     ○ → ◎ → ◉ → ☉         (expanding)               │     │
│  │ Criticality: ☉ → ☼ → ✱ → ✴         (interference)            │     │
│  │ Discharge:   ✴ → ✳ → ● → •         (contracting)             │     │
│  │ Recovery:    • → ○ → ◎ → ...       (re-expanding)            │     │
│  └───────────────────────────────────────────────────────────────┘     │
│                                                                         │
│  MEASUREMENTS:                                                          │
│  • Effective radius of error cloud                                      │
│  • Entropy of error distribution                                        │
│  • Number of error modes (clusters)                                     │
│  • Error velocity (how fast wavefront moves)                           │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.3 Ground Truth Trajectory

```
GROUND TRUTH EMBEDDING TRAJECTORY

Track the actual next frame for comparison.

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  WHAT TO TRACK:                                                         │
│  • Ground truth x_{t+1} embedded                                        │
│  • Distance from prediction: d(ŷ_t, x_{t+1})                           │
│  • Lag between prediction and truth                                    │
│                                                                         │
│  VISUALIZATION:                                                         │
│                                                                         │
│     Prediction trajectory: ●──●──●──●──●                               │
│     Ground truth trajectory: ○──○──○──○──○                             │
│     Error vectors: ●→○  ●→○  ●→○  ●→○  ●→○                            │
│                                                                         │
│  The gap between trajectories IS the belief error.                     │
│  Watch it expand (tension) and collapse (discharge).                   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.4 Combined Trajectory View

```
MULTI-TRAJECTORY VISUALIZATION

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  OVERLAY:                                                               │
│  • Prediction trajectory (solid line)                                   │
│  • Ground truth trajectory (dashed line)                                │
│  • Error cloud (shaded region between them)                            │
│  • Attention focus (arrows showing what model attends to)              │
│  • Collapse events (markers)                                            │
│                                                                         │
│  ANIMATION:                                                             │
│  • Time flows along trajectory                                          │
│  • Error cloud breathes (expands/contracts)                             │
│  • Collapse events flash                                                │
│  • Wormhole connections appear/disappear                                │
│                                                                         │
│  SYNCHRONIZED VIEWS:                                                    │
│  • 2D output space (original video + prediction + error)               │
│  • Embedding space (trajectories)                                       │
│  • Spectral bands (per-band dynamics)                                   │
│  • Metrics panel (loss, entropy, etc.)                                  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Belief Cloud Visualization

### 5.1 What Is the Belief Cloud?

```
THE BELIEF CLOUD: Multiple Samples from the Belief Distribution

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  The model's prediction ŷ is the MEAN of its belief distribution.     │
│  But the belief has SHAPE—it's not just a point.                       │
│                                                                         │
│  To visualize the full belief, we need SAMPLES:                        │
│                                                                         │
│  METHOD 1: DROPOUT SAMPLING                                             │
│  • Run inference with dropout enabled                                   │
│  • Each forward pass gives different prediction                        │
│  • Collect N samples: {ŷ₁, ŷ₂, ..., ŷₙ}                               │
│  • These samples approximate the belief distribution                   │
│                                                                         │
│  METHOD 2: ENSEMBLE                                                     │
│  • Train multiple models                                                │
│  • Each model gives different prediction                               │
│  • Ensemble predictions sample belief space                            │
│                                                                         │
│  METHOD 3: LATENT PERTURBATION                                          │
│  • Add noise to latent representations                                 │
│  • Decode to get varied predictions                                    │
│  • Sample the sensitivity of the output                                │
│                                                                         │
│  METHOD 4: ATTENTION PERTURBATION                                       │
│  • Perturb attention weights                                            │
│  • See how prediction changes                                          │
│  • Map attention → output sensitivity                                  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 5.2 Belief Cloud Dynamics

```
BELIEF CLOUD DURING PUMP CYCLE

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  TENSION:                                                               │
│                    ○   ○                                               │
│                 ○    ○    ○                                            │
│              ○    ○    ○    ○         Cloud EXPANDS                    │
│                 ○    ○    ○           Samples spread apart             │
│                    ○   ○              Belief is uncertain              │
│                                                                         │
│  CRITICALITY:                                                           │
│                    ○  ○                                                │
│                 ○ ○  ○ ○                                               │
│              ○○    ○○    ○○           Cloud STRUCTURES                 │
│                 ○ ○  ○ ○              Modes form                       │
│                    ○  ○               Interference visible             │
│                                                                         │
│  DISCHARGE:                                                             │
│                                                                         │
│                                                                         │
│                    ○○○                Cloud COLLAPSES                   │
│                    ○●○                Samples converge                  │
│                    ○○○                One mode wins                     │
│                                                                         │
│  RECOVERY:                                                              │
│                     ○                                                  │
│                    ○○                 Cloud begins EXPANDING            │
│                   ○○○○                New uncertainty forms            │
│                    ○○                                                  │
│                     ○                                                  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 5.3 Measuring Cloud Properties

```
BELIEF CLOUD METRICS

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  SIZE METRICS:                                                          │
│  • Covariance: Cov(samples)                                             │
│  • Determinant: |Cov| (volume)                                         │
│  • Trace: Tr(Cov) (total variance)                                     │
│  • Effective radius: √(Tr(Cov)/D)                                      │
│                                                                         │
│  SHAPE METRICS:                                                         │
│  • Eigenvalues: λ₁, λ₂, ... (principal axes)                           │
│  • Anisotropy: λ_max / λ_min                                           │
│  • Entropy: H(p) where p fit to samples                                │
│  • Number of modes: cluster count                                      │
│                                                                         │
│  DYNAMICS METRICS:                                                      │
│  • Cloud velocity: d(centroid)/dt                                      │
│  • Cloud expansion rate: d(volume)/dt                                  │
│  • Mode splitting: when 1 mode → 2 modes                               │
│  • Collapse rate: how fast volume → 0                                  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Cross-Component Analysis

### 6.1 Correlation Matrices

```
CROSS-COMPONENT CORRELATIONS

Track how different parts of the system co-vary:

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  SPECTRAL BAND CORRELATIONS:                                            │
│  Corr[band_i, band_j] — how bands move together                        │
│                                                                         │
│  ATTENTION HEAD CORRELATIONS:                                           │
│  Corr[head_i, head_j] — how attention mechanisms agree                 │
│                                                                         │
│  ERROR-ATTENTION CORRELATIONS:                                          │
│  Corr[error, attention] — does attention predict error?                │
│                                                                         │
│  ERROR-GRADIENT CORRELATIONS:                                           │
│  Corr[error, gradient] — how error drives learning                     │
│                                                                         │
│  TEMPORAL CROSS-CORRELATIONS:                                           │
│  Corr[X_t, Y_{t+τ}] — lagged relationships                            │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 6.2 Causal Analysis

```
CAUSAL RELATIONSHIPS BETWEEN COMPONENTS

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  GRANGER CAUSALITY:                                                     │
│  Does X_t help predict Y_{t+1} beyond Y_t alone?                       │
│                                                                         │
│  TRANSFER ENTROPY:                                                      │
│  TE(X → Y) = H(Y_t | Y_{t-1}) - H(Y_t | Y_{t-1}, X_{t-1})             │
│  Information flow from X to Y                                          │
│                                                                         │
│  INTERVENTION ANALYSIS:                                                 │
│  Perturb component X, measure effect on Y                              │
│  Causal effect = E[Y | do(X)] - E[Y]                                   │
│                                                                         │
│  QUESTIONS TO ANSWER:                                                   │
│  • Does wormhole attention cause collapse?                             │
│  • Does low-freq guide high-freq or vice versa?                        │
│  • Does error drive attention or attention drive error?                │
│  • What causes the pump cycle?                                         │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 7. Experimental Protocols

### 7.1 Baseline Experiments

```
BASELINE EXPERIMENTS

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  EXPERIMENT B1: Random Weights                                          │
│  • Initialize randomly, don't train                                    │
│  • Establish baseline embedding structure                              │
│  • What structure exists before learning?                              │
│                                                                         │
│  EXPERIMENT B2: Ablation                                                │
│  • Remove temporal attention → measure change                          │
│  • Remove neighbor attention → measure change                          │
│  • Remove wormhole attention → measure change                          │
│  • Remove spectral decomposition → measure change                      │
│                                                                         │
│  EXPERIMENT B3: Fixed Patterns                                          │
│  • Simple blob → observe embedding dynamics                            │
│  • Rotating pattern → observe periodic structure                       │
│  • Bifurcation → observe splitting in embedding                        │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 7.2 Pump Cycle Experiments

```
PUMP CYCLE EXPERIMENTS

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  EXPERIMENT P1: Collapse Detection                                      │
│  • Monitor entropy of belief cloud                                      │
│  • Detect sudden drops (collapse events)                               │
│  • Correlate with error reduction                                      │
│                                                                         │
│  EXPERIMENT P2: Cycle Period                                            │
│  • Measure time between collapses                                       │
│  • Is it regular? Adaptive? Pattern-dependent?                         │
│  • Can we control it?                                                  │
│                                                                         │
│  EXPERIMENT P3: Cycle Manipulation                                      │
│  • Increase wormhole threshold → delay collapse                        │
│  • Decrease threshold → accelerate collapse                            │
│  • Add noise → prolong tension phase                                   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 7.3 Wavefront Experiments

```
WAVEFRONT EXPERIMENTS

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  EXPERIMENT W1: Wavefront Tracking                                      │
│  • Embed error map at each timestep                                    │
│  • Track trajectory through embedding space                            │
│  • Measure expansion/contraction dynamics                              │
│                                                                         │
│  EXPERIMENT W2: Wavefront Shape                                         │
│  • What shape is the error cloud?                                      │
│  • How does shape correlate with pattern type?                         │
│  • Does shape predict what model doesn't know?                         │
│                                                                         │
│  EXPERIMENT W3: Wavefront Collapse                                      │
│  • Observe collapse in embedding space                                  │
│  • Is it gradual or sudden?                                            │
│  • What triggers it?                                                   │
│                                                                         │
│  EXPERIMENT W4: Wavefront Conservation                                  │
│  • Does error volume stay constant?                                    │
│  • When error collapses here, does it appear elsewhere?               │
│  • Test conservation hypothesis                                        │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 8. Implementation Notes

### 8.1 Hooks and Extraction

```python
"""
HOOKS FOR EMBEDDING EXTRACTION

Register hooks on model layers to capture intermediate representations.
"""

class EmbeddingExtractor:
    """
    Extract embeddings from all layers of the AKIRA model.
    """
    
    def __init__(self, model):
        self.model = model
        self.embeddings = {}
        self.hooks = []
        
    def register_hooks(self):
        """Register forward hooks on all relevant layers."""
        
        # Temporal attention
        self.hooks.append(
            self.model.temporal_attention.register_forward_hook(
                self._make_hook('temporal')
            )
        )
        
        # Neighbor attention
        self.hooks.append(
            self.model.neighbor_attention.register_forward_hook(
                self._make_hook('neighbor')
            )
        )
        
        # Wormhole attention
        self.hooks.append(
            self.model.wormhole_attention.register_forward_hook(
                self._make_hook('wormhole')
            )
        )
        
        # Per-band representations
        for band in range(7):
            self.hooks.append(
                self.model.band_processors[band].register_forward_hook(
                    self._make_hook(f'band_{band}')
                )
            )
    
    def _make_hook(self, name):
        def hook(module, input, output):
            self.embeddings[name] = {
                'input': input,
                'output': output,
                'timestamp': time.time()
            }
        return hook
    
    def get_embeddings(self):
        return self.embeddings.copy()
    
    def clear(self):
        self.embeddings = {}
```

### 8.2 Real-Time Streaming

```python
"""
REAL-TIME EMBEDDING STREAMING

Stream embeddings to visualization frontend.
"""

class EmbeddingStreamer:
    """
    Stream embeddings for real-time visualization.
    """
    
    def __init__(self, extractor, reducer, websocket=None):
        self.extractor = extractor
        self.reducer = reducer  # Dimensionality reduction
        self.websocket = websocket
        self.history = []
        
    async def stream_step(self, frame_idx):
        """Process one frame and stream embeddings."""
        
        # Get raw embeddings
        embeddings = self.extractor.get_embeddings()
        
        # Reduce dimensions (fast method for real-time)
        reduced = {}
        for key, emb in embeddings.items():
            reduced[key] = self.reducer.transform(
                emb['output'].flatten().cpu().numpy()
            )
        
        # Store in history
        self.history.append({
            'frame': frame_idx,
            'embeddings': reduced,
            'timestamp': time.time()
        })
        
        # Stream to frontend
        if self.websocket:
            await self.websocket.send_json({
                'type': 'embedding_update',
                'frame': frame_idx,
                'data': reduced
            })
        
        return reduced
```

### 8.3 Visualization Dashboard

```
DASHBOARD LAYOUT

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  ┌─────────────────────────┬─────────────────────────┐                 │
│  │                         │                         │                 │
│  │   ORIGINAL FRAME        │   PREDICTION            │                 │
│  │                         │                         │                 │
│  │   [video frame]         │   [model prediction]    │                 │
│  │                         │                         │                 │
│  ├─────────────────────────┼─────────────────────────┤                 │
│  │                         │                         │                 │
│  │   ERROR MAP             │   SPECTRAL BANDS        │                 │
│  │                         │                         │                 │
│  │   [error heatmap]       │   [7 band previews]     │                 │
│  │                         │                         │                 │
│  └─────────────────────────┴─────────────────────────┘                 │
│                                                                         │
│  ┌─────────────────────────────────────────────────────┐               │
│  │                                                     │               │
│  │   EMBEDDING SPACE (3D)                              │               │
│  │                                                     │               │
│  │   [prediction trajectory]                           │               │
│  │   [ground truth trajectory]                         │               │
│  │   [belief cloud]                                    │               │
│  │   [error wavefront]                                 │               │
│  │                                                     │               │
│  └─────────────────────────────────────────────────────┘               │
│                                                                         │
│  ┌─────────────────────────────────────────────────────┐               │
│  │                                                     │               │
│  │   METRICS TIMELINE                                  │               │
│  │                                                     │               │
│  │   [loss] [entropy] [sparsity] [collapse events]    │               │
│  │                                                     │               │
│  └─────────────────────────────────────────────────────┘               │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Summary

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│     O B S E R V A B I L I T Y   E M B E D D I N G S                   │
│                                                                         │
│  ═══════════════════════════════════════════════════════════════════   │
│                                                                         │
│  WE EMBED:                                                              │
│  • Attention head representations (Q, K, V, weights)                   │
│  • Error signals (spatial, spectral, temporal, frequency-domain)       │
│  • Spectral band activations (all 7 bands)                             │
│  • Gradients and weight dynamics                                        │
│  • Belief state observables                                             │
│  • Wormhole connection structure                                        │
│  • History buffer state                                                 │
│  • Meta-observables (loss, information flow, complexity)               │
│                                                                         │
│  ═══════════════════════════════════════════════════════════════════   │
│                                                                         │
│  WE REDUCE WITH:                                                        │
│  • Shogu (custom, streaming-capable)                                   │
│  • PCA (fast, linear)                                                  │
│  • UMAP (local + global structure)                                     │
│  • t-SNE (cluster discovery)                                           │
│  • PHATE (trajectory-optimized)                                        │
│                                                                         │
│  ═══════════════════════════════════════════════════════════════════   │
│                                                                         │
│  WE TRACK:                                                              │
│  • Prediction embedding trajectory                                      │
│  • Error embedding trajectory (wavefront)                              │
│  • Ground truth trajectory                                              │
│  • Belief cloud (multiple samples)                                     │
│  • Collapse events                                                      │
│  • Pump cycle phases                                                    │
│                                                                         │
│  ═══════════════════════════════════════════════════════════════════   │
│                                                                         │
│  WE PROVE:                                                              │
│  • Belief has geometric structure                                       │
│  • The pump cycle is real and observable                               │
│  • Collapse is a phase transition                                       │
│  • Information is conserved (moves, doesn't disappear)                 │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

*Oscar Goldman — Shogu Research Group @ Datamutant.ai*

*"To understand the system, we must SEE the manifold. The wavefront error is in there. We must collapse it and view it. In real-time, if possible. This is not debugging—this is science."*

