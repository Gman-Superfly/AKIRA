# EVENTS: Operational Entities of the AKIRA System

## A Catalog of Moving Parts, Values, Triggers, Gates, and Interactions

---

## Table of Contents

1. [Entity Taxonomy](#1-entity-taxonomy)
2. [Core Computational Entities](#2-core-computational-entities)
3. [State Entities](#3-state-entities)
4. [Scalar Values](#4-scalar-values)
5. [Vector Entities](#5-vector-entities)
6. [Matrix/Tensor Entities](#6-matrixtensor-entities)
7. [Gates and Triggers](#7-gates-and-triggers)
8. [Manifolds and Spaces](#8-manifolds-and-spaces)
9. [Transform Operations](#9-transform-operations)
10. [Monitoring Entities](#10-monitoring-entities)
11. [Complex Nested Operations](#11-complex-nested-operations)
12. [Entity Interactions](#12-entity-interactions)
13. [Event Streams](#13-event-streams)

---

## 1. Entity Taxonomy

```
ENTITY CLASSIFICATION

Every operational part of AKIRA can be classified:

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  CATEGORY        DESCRIPTION                   EXAMPLES                │
│  ────────        ───────────                   ────────                │
│                                                                         │
│  COMPUTATIONAL   Operations that transform     FFT, Attention, Softmax │
│                  data from one form to another                         │
│                                                                         │
│  STATE           Persistent values that        History buffer, Weights │
│                  change over time                                       │
│                                                                         │
│  SCALAR          Single numerical values       Loss, threshold, LR     │
│                                                                         │
│  VECTOR          Ordered sequences of values   Q, K, V, gradients      │
│                                                                         │
│  MATRIX/TENSOR   Multi-dimensional arrays      Attention weights, FFT  │
│                                                                         │
│  GATE            Binary or soft switches       Threshold, ReLU, mask   │
│                                                                         │
│  TRIGGER         Conditions that cause events  Collapse, update        │
│                                                                         │
│  MANIFOLD        Geometric structures          Hypersphere, belief     │
│                                                                         │
│  TRANSFORM       Coordinate changes            FFT, normalize, project │
│                                                                         │
│  MONITOR         Observation/measurement       Entropy tracker, loss   │
│                                                                         │
│  COMPOSITE       Nested operations             MAP algo, Fisher info   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Core Computational Entities

### 2.1 Spectral Decomposition Engine

```
ENTITY: SpectralDecomposer

┌─────────────────────────────────────────────────────────────────────────┐
│ TYPE: Computational                                                     │
│ ID: SPEC_DECOMP                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│ INPUTS:                                                                 │
│   • frame: [H, W, C], input image frame                                │
│                                                                         │
│ OUTPUTS:                                                                │
│   • bands: [7, H, W], spectral band decomposition                      │
│   • magnitude: [H, W], FFT magnitude                                   │
│   • phase: [H, W], FFT phase                                           │
│                                                                         │
│ INTERNAL OPERATIONS:                                                    │
│   • FFT_2D: Forward Fourier transform                                  │
│   • SHIFT: Center the DC component                                     │
│   • BAND_FILTER: Apply octave band masks                               │
│   • IFFT_2D: Inverse transform per band                                │
│                                                                         │
│ TRIGGERS:                                                               │
│   • ON_FRAME_ARRIVAL: Process new frame                                │
│                                                                         │
│ EMITS:                                                                  │
│   • BANDS_READY: When decomposition complete                           │
│   • SPECTRAL_STATS: Band-wise statistics                               │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Temporal Attention

```
ENTITY: TemporalAttention

┌─────────────────────────────────────────────────────────────────────────┐
│ TYPE: Computational                                                     │
│ ID: TEMP_ATTN                                                           │
├─────────────────────────────────────────────────────────────────────────┤
│ INPUTS:                                                                 │
│   • query_features: [H, W, D], current frame features                  │
│   • history_buffer: [T, H, W, D], past frame features                  │
│                                                                         │
│ OUTPUTS:                                                                │
│   • attended_features: [H, W, D]                                       │
│   • attention_weights: [H, W, T]                                       │
│                                                                         │
│ INTERNAL ENTITIES:                                                      │
│   • Q_PROJ: Query projection matrix [D, D_attn]                        │
│   • K_PROJ: Key projection matrix [D, D_attn]                          │
│   • V_PROJ: Value projection matrix [D, D_attn]                        │
│   • SOFTMAX: Attention normalization                                   │
│   • SCALE: 1/√D_attn scaling factor                                    │
│                                                                         │
│ STATE:                                                                  │
│   • history_buffer: Sliding window of past features                    │
│   • EMA_weights: Smoothed attention pattern                            │
│                                                                         │
│ EVENTS:                                                                 │
│   • ATTENTION_COMPUTED: Weights available                              │
│   • HISTORY_UPDATED: Buffer shifted                                    │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.3 Neighbor Attention

```
ENTITY: NeighborAttention

┌─────────────────────────────────────────────────────────────────────────┐
│ TYPE: Computational                                                     │
│ ID: NEIGH_ATTN                                                          │
├─────────────────────────────────────────────────────────────────────────┤
│ INPUTS:                                                                 │
│   • features: [H, W, D], current features                              │
│   • neighborhood_size: int, spatial window                             │
│                                                                         │
│ OUTPUTS:                                                                │
│   • smoothed_features: [H, W, D]                                       │
│   • spatial_weights: [H, W, N_neighbors]                               │
│                                                                         │
│ INTERNAL ENTITIES:                                                      │
│   • UNFOLD: Extract local patches                                      │
│   • Q_PROJ, K_PROJ, V_PROJ: Projections                               │
│   • SPATIAL_BIAS: Learnable position bias                              │
│                                                                         │
│ EVENTS:                                                                 │
│   • SPATIAL_COHERENCE_COMPUTED                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.4 Wormhole Attention

```
ENTITY: WormholeAttention

┌─────────────────────────────────────────────────────────────────────────┐
│ TYPE: Computational                                                     │
│ ID: WORM_ATTN                                                           │
├─────────────────────────────────────────────────────────────────────────┤
│ INPUTS:                                                                 │
│   • query_features: [H, W, D]                                          │
│   • full_history: [T, H, W, D], extended history                       │
│                                                                         │
│ OUTPUTS:                                                                │
│   • wormhole_features: [H, W, D]                                       │
│   • connection_indices: [H*W, K], which history positions              │
│   • connection_similarities: [H*W, K]                                  │
│   • connection_mask: [H*W, K], boolean gate                            │
│                                                                         │
│ INTERNAL ENTITIES:                                                      │
│   • NORMALIZER: L2 normalize to hypersphere                            │
│   • SIM_COMPUTER: Cosine similarity matrix                             │
│   • TOP_K_SELECTOR: Find K nearest neighbors                           │
│   • THRESHOLD_GATE: sim > τ decision                                   │
│   • SOFTMAX: Normalize surviving connections                           │
│                                                                         │
│ PARAMETERS:                                                             │
│   • threshold: τ = 0.92 (default)                                      │
│   • K_neighbors: Top-K to consider                                     │
│   • temperature: Softmax temperature                                   │
│                                                                         │
│ EVENTS:                                                                 │
│   • WORMHOLES_OPENED: Connections established                          │
│   • WORMHOLES_CLOSED: Connections below threshold                      │
│   • TELEPORTATION_COMPLETE: Information transferred                    │
│                                                                         │
│ STATISTICS:                                                             │
│   • num_active_connections                                             │
│   • mean_similarity                                                    │
│   • sparsity_ratio                                                     │
│   • temporal_reach_distribution                                        │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. State Entities

### 3.1 History Buffer

```
ENTITY: HistoryBuffer

┌─────────────────────────────────────────────────────────────────────────┐
│ TYPE: State                                                             │
│ ID: HIST_BUF                                                            │
├─────────────────────────────────────────────────────────────────────────┤
│ SHAPE: [T, 7, H, W, D], time × bands × spatial × features             │
│                                                                         │
│ OPERATIONS:                                                             │
│   • PUSH: Add new frame, shift old                                     │
│   • POP: Remove oldest frame                                           │
│   • QUERY: Access specific timestep                                    │
│   • SLICE: Get temporal window                                         │
│                                                                         │
│ PROPERTIES:                                                             │
│   • capacity: Maximum frames stored                                    │
│   • fill_level: Current occupancy                                      │
│   • age: Timestamps of stored frames                                   │
│                                                                         │
│ EVENTS:                                                                 │
│   • BUFFER_FULL: Capacity reached                                      │
│   • FRAME_EVICTED: Oldest frame removed                                │
│   • FRAME_ADDED: New frame stored                                      │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Weight Manifold

```
ENTITY: WeightManifold

┌─────────────────────────────────────────────────────────────────────────┐
│ TYPE: State                                                             │
│ ID: WEIGHT_MAN                                                          │
├─────────────────────────────────────────────────────────────────────────┤
│ COMPONENTS:                                                             │
│   • temporal_weights: Parameters of temporal attention                 │
│   • neighbor_weights: Parameters of neighbor attention                 │
│   • wormhole_weights: Parameters of wormhole attention                 │
│   • band_weights: Per-band processing parameters                       │
│   • projection_weights: Input/output projections                       │
│                                                                         │
│ PER-BAND LEARNING RATES:                                                │
│   • band_0_lr: 0.00001 (DC - protected)                               │
│   • band_1_lr: 0.0001                                                  │
│   • band_2_lr: 0.0003                                                  │
│   • band_3_lr: 0.001                                                   │
│   • band_4_lr: 0.003                                                   │
│   • band_5_lr: 0.01                                                    │
│   • band_6_lr: 0.03 (High - responsive)                                │
│                                                                         │
│ EVENTS:                                                                 │
│   • WEIGHTS_UPDATED: After gradient step                               │
│   • GRADIENT_COMPUTED: Before update                                   │
│   • DIVERGENCE_DETECTED: Instability                                   │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.3 Belief State

```
ENTITY: BeliefState

┌─────────────────────────────────────────────────────────────────────────┐
│ TYPE: State (Implicit)                                                  │
│ ID: BELIEF                                                              │
├─────────────────────────────────────────────────────────────────────────┤
│ REPRESENTATION:                                                         │
│   The belief state is IMPLICIT, encoded in:                            │
│   • attention_weights: Where model attends                             │
│   • prediction: Mean of belief distribution                            │
│   • activation_patterns: Internal representations                      │
│                                                                         │
│ OBSERVABLES:                                                            │
│   • entropy: H(attention weights)                                      │
│   • concentration: max(attention) / mean(attention)                    │
│   • effective_support: Number of significant hypotheses                │
│   • confidence: 1 - entropy / max_entropy                              │
│                                                                         │
│ DYNAMICS:                                                               │
│   • SPREADING: Entropy increasing (tension)                            │
│   • FOCUSING: Entropy decreasing (approaching collapse)                │
│   • COLLAPSED: Entropy near zero (committed)                           │
│   • RECOVERING: Entropy rising again (new uncertainty)                 │
│                                                                         │
│ EVENTS:                                                                 │
│   • BELIEF_SPREADING: Uncertainty growing                              │
│   • BELIEF_FOCUSING: Uncertainty shrinking                             │
│   • BELIEF_COLLAPSED: Commitment made                                  │
│   • BELIEF_BIFURCATED: Split into modes                                │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Scalar Values

```
SCALAR ENTITIES

┌─────────────────────────────────────────────────────────────────────────┐
│ ID                  TYPE        RANGE       DESCRIPTION                │
├─────────────────────────────────────────────────────────────────────────┤
│ LOSS                Dynamic     [0, ∞)      MSE prediction error       │
│ LOSS_VELOCITY       Dynamic     (-∞, ∞)     dL/dt                      │
│ LOSS_ACCEL          Dynamic     (-∞, ∞)     d²L/dt²                    │
│                                                                         │
│ LR_GLOBAL           Param       (0, 1)      Base learning rate         │
│ LR_BAND[0-6]        Param       (0, 1)      Per-band learning rate     │
│                                                                         │
│ THRESHOLD_WORM      Param       [0, 1]      Wormhole similarity gate   │
│ TEMPERATURE         Param       (0, ∞)      Softmax temperature        │
│ TOP_K               Param       [1, N]      Number of neighbors        │
│                                                                         │
│ ENTROPY_BELIEF      Monitor     [0, log(N)] Belief uncertainty         │
│ ENTROPY_ATTN        Monitor     [0, log(T)] Attention uncertainty      │
│ SPARSITY            Monitor     [0, 1]      Fraction of active conn    │
│                                                                         │
│ CYCLE_PHASE         Monitor     {0,1,2,3}   Pump cycle stage           │
│ COLLAPSE_TRIGGER    Monitor     {0,1}       Collapse detected          │
│                                                                         │
│ GRAD_NORM           Monitor     [0, ∞)      Gradient magnitude         │
│ WEIGHT_NORM         Monitor     [0, ∞)      Weight magnitude           │
│ EFFECTIVE_RANK      Monitor     [1, D]      Matrix rank                │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Vector Entities

```
VECTOR ENTITIES

┌─────────────────────────────────────────────────────────────────────────┐
│ ID                  SHAPE       DESCRIPTION                            │
├─────────────────────────────────────────────────────────────────────────┤
│ QUERY[pos]          [D]         Query vector at position               │
│ KEY[pos, t]         [D]         Key vector at position, time           │
│ VALUE[pos, t]       [D]         Value vector at position, time         │
│                                                                         │
│ GRADIENT[layer]     [D_layer]   Gradient vector per layer              │
│ GRAD_DIRECTION      [D]         Normalized gradient                    │
│                                                                         │
│ FEATURE[pos]        [D]         Feature vector at position             │
│ PREDICTION[pos]     [C]         Predicted pixel values                 │
│ ERROR[pos]          [C]         Prediction - ground truth              │
│                                                                         │
│ POS_ENCODING[pos]   [D]         Position embedding                     │
│ BAND_EMBEDDING[b]   [D]         Band identity embedding                │
│                                                                         │
│ ATTN_WEIGHTS[pos]   [T] or [K]  Attention distribution                 │
│ SIMILARITY[pos]     [K]         Top-K similarities                     │
│                                                                         │
│ SPECTRUM[freq]      [N/2]       Radial power spectrum                  │
│ BAND_POWER          [7]         Power per spectral band                │
│ BAND_ENTROPY        [7]         Entropy per spectral band              │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Matrix/Tensor Entities

```
MATRIX AND TENSOR ENTITIES

┌─────────────────────────────────────────────────────────────────────────┐
│ ID                  SHAPE           DESCRIPTION                        │
├─────────────────────────────────────────────────────────────────────────┤
│ FFT_RESULT          [H, W] complex  Frequency domain representation    │
│ BAND_TENSOR         [7, H, W]       Spectral band decomposition        │
│                                                                         │
│ ATTN_MATRIX_T       [H*W, T]        Temporal attention weights         │
│ ATTN_MATRIX_N       [H*W, N_nb]     Neighbor attention weights         │
│ ATTN_MATRIX_W       [H*W, K]        Wormhole attention weights         │
│                                                                         │
│ SIMILARITY_FULL     [H*W, T*H*W]    Full similarity matrix             │
│ CONNECTION_GRAPH    [H*W, K, 3]     Wormhole connection structure      │
│                                                                         │
│ W_Q, W_K, W_V       [D, D_attn]     Projection matrices                │
│ W_O                 [D_attn, D]     Output projection                  │
│                                                                         │
│ COVARIANCE          [D, D]          Feature covariance                 │
│ FISHER_INFO         [N_p, N_p]      Fisher information matrix          │
│ HESSIAN             [N_p, N_p]      Loss Hessian                       │
│                                                                         │
│ CORRELATION_BANDS   [7, 7]          Cross-band correlation             │
│ TRANSFER_ENTROPY    [N, N]          Causal influence matrix            │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 7. Gates and Triggers

### 7.1 Binary Gates

```
BINARY GATE ENTITIES

┌─────────────────────────────────────────────────────────────────────────┐
│ ID                  CONDITION           OUTPUT      DESCRIPTION        │
├─────────────────────────────────────────────────────────────────────────┤
│ WORMHOLE_GATE       sim > threshold     {0, 1}      Allow connection   │
│ RELU_GATE           x > 0               {0, 1}      Activation gate    │
│ DROPOUT_GATE        random > p          {0, 1}      Regularization     │
│ GRADIENT_CLIP       |g| > max           {0, 1}      Stability gate     │
│                                                                         │
│ COLLAPSE_TRIGGER    ∂H/∂t < -threshold  {0, 1}      Detect collapse    │
│ DIVERGENCE_TRIGGER  loss > threshold    {0, 1}      Detect instability │
│ SATURATION_TRIGGER  |act| > threshold   {0, 1}      Detect saturation  │
│                                                                         │
│ HISTORY_FULL        fill_level = max    {0, 1}      Buffer capacity    │
│ CONFIDENCE_HIGH     entropy < τ         {0, 1}      Ready to commit    │
└─────────────────────────────────────────────────────────────────────────┘
```

### 7.2 Soft Gates

```
SOFT GATE ENTITIES

┌─────────────────────────────────────────────────────────────────────────┐
│ ID                  FUNCTION            OUTPUT      DESCRIPTION        │
├─────────────────────────────────────────────────────────────────────────┤
│ SOFTMAX_GATE        exp(x)/Σexp(x)      [0, 1]      Attention weights  │
│ SIGMOID_GATE        1/(1+exp(-x))       [0, 1]      Smooth binary      │
│ TANH_GATE           tanh(x)             [-1, 1]     Bounded linear     │
│ GELU_GATE           x·Φ(x)              smooth      Smooth ReLU        │
│                                                                         │
│ EMA_GATE            α·x + (1-α)·prev    smooth      Temporal smoothing │
│ TEMPERATURE_GATE    softmax(x/τ)        [0, 1]      Sharpness control  │
│                                                                         │
│ BAND_WEIGHT_GATE    learned per band    [0, 1]      Band importance    │
│ SPATIAL_MASK_GATE   position-dependent  [0, 1]      Spatial attention  │
└─────────────────────────────────────────────────────────────────────────┘
```

### 7.3 Event Triggers

```
EVENT TRIGGER ENTITIES

┌─────────────────────────────────────────────────────────────────────────┐
│ ID                  CONDITION               ACTION                     │
├─────────────────────────────────────────────────────────────────────────┤
│ FRAME_ARRIVAL       New input received      Start forward pass         │
│ FORWARD_COMPLETE    All attention computed  Compute prediction         │
│ LOSS_COMPUTED       Prediction vs GT        Trigger backward           │
│ BACKWARD_COMPLETE   Gradients ready         Update weights             │
│                                                                         │
│ COLLAPSE_EVENT      Entropy drop detected   Log, visualize             │
│ CYCLE_TRANSITION    Phase changed           Update monitors            │
│                                                                         │
│ DIVERGENCE_EVENT    Loss exploding          Emergency intervention     │
│ CONVERGENCE_EVENT   Loss stable             Reduce LR                  │
│                                                                         │
│ WORMHOLE_ACTIVATED  Connection opened       Track teleportation        │
│ WORMHOLE_CLOSED     Below threshold         Track rejection            │
│                                                                         │
│ BUFFER_EVICTION     Oldest frame removed    Update statistics          │
│ MODE_SWITCH         Knowledge ↔ Reactive    Adjust processing          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 8. Manifolds and Spaces

```
MANIFOLD ENTITIES

┌─────────────────────────────────────────────────────────────────────────┐
│ ID                  GEOMETRY            DESCRIPTION                    │
├─────────────────────────────────────────────────────────────────────────┤
│ HYPERSPHERE         S^{D-1}             Normalized feature space       │
│                     (unit sphere)       Used by wormhole attention     │
│                                                                         │
│ WEIGHT_SPACE        R^{N_params}        Full parameter space           │
│                     (Euclidean)         Where gradients flow           │
│                                                                         │
│ BELIEF_MANIFOLD     Simplex Δ^{K-1}     Probability distributions      │
│                     (probability)       Attention weights live here    │
│                                                                         │
│ LATENT_SPACE        R^D                 Internal representation        │
│                     (learned metric)    Where features live            │
│                                                                         │
│ FREQUENCY_DOMAIN    C^{H×W}             Fourier coefficients           │
│                     (complex plane)     Spectral representation        │
│                                                                         │
│ LOSS_LANDSCAPE      R^{N_params} → R    Energy surface                 │
│                     (scalar field)      Gradient descent terrain       │
│                                                                         │
│ INFORMATION_GEOM    Fisher metric       Natural gradient space         │
│                     (Riemannian)        Optimal parameter updates      │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 9. Transform Operations

```
TRANSFORM OPERATION ENTITIES

┌─────────────────────────────────────────────────────────────────────────┐
│ ID                  INPUT → OUTPUT          PROPERTIES                 │
├─────────────────────────────────────────────────────────────────────────┤
│ FFT_2D              R^{H×W} → C^{H×W}       Invertible, O(N log N)    │
│ IFFT_2D             C^{H×W} → R^{H×W}       Inverse of FFT            │
│                                                                         │
│ L2_NORMALIZE        R^D → S^{D-1}           Project to hypersphere     │
│ SOFTMAX             R^K → Δ^{K-1}           Project to simplex         │
│ LOG_SOFTMAX         R^K → R^K               Numerically stable         │
│                                                                         │
│ LINEAR_PROJ         R^{D_in} → R^{D_out}    Learned transformation     │
│ MATMUL              R^{m×n}×R^{n×p} → R^{m×p}  Core operation         │
│                                                                         │
│ RESHAPE             R^{...} → R^{...}       Change tensor shape        │
│ TRANSPOSE           R^{m×n} → R^{n×m}       Swap dimensions            │
│ UNFOLD              R^{H×W×D} → patches     Extract local windows      │
│                                                                         │
│ TOP_K               R^N → R^K               Select largest K           │
│ ARGSORT             R^N → indices           Rank elements              │
│ GATHER              tensor × indices → sub  Select by index            │
│                                                                         │
│ BAYES_UPDATE        prior × likelihood → post  Belief update          │
│ EMA_UPDATE          x, prev, α → smoothed   Exponential averaging      │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 10. Monitoring Entities

### 10.1 Gradient Monitors

```
ENTITY: GradientMonitor

┌─────────────────────────────────────────────────────────────────────────┐
│ TYPE: Monitor                                                           │
│ ID: GRAD_MON                                                            │
├─────────────────────────────────────────────────────────────────────────┤
│ OBSERVES:                                                               │
│   • Per-layer gradient norms: ‖∇W_l‖                                   │
│   • Gradient directions: ∇W/‖∇W‖                                       │
│   • Gradient alignment: cos(∇W_t, ∇W_{t-1})                            │
│   • Per-band gradients: ∇W for each spectral band                      │
│   • Gradient SNR: signal/noise in gradients                            │
│                                                                         │
│ COMPUTES:                                                               │
│   • gradient_health: Composite stability metric                        │
│   • vanishing_score: How much gradient vanishes                        │
│   • exploding_score: How much gradient explodes                        │
│                                                                         │
│ TRIGGERS:                                                               │
│   • GRADIENT_VANISHING: norm below threshold                           │
│   • GRADIENT_EXPLODING: norm above threshold                           │
│   • GRADIENT_ALIGNED: alignment above threshold                        │
└─────────────────────────────────────────────────────────────────────────┘
```

### 10.2 Entropy Tracker

```
ENTITY: EntropyTracker

┌─────────────────────────────────────────────────────────────────────────┐
│ TYPE: Monitor                                                           │
│ ID: ENTROPY_TRACK                                                       │
├─────────────────────────────────────────────────────────────────────────┤
│ OBSERVES:                                                               │
│   • Attention entropy: H(attention_weights)                            │
│   • Belief entropy: H(implicit belief)                                 │
│   • Spectral entropy: H(band power distribution)                       │
│   • Spatial entropy: H(error spatial distribution)                     │
│                                                                         │
│ COMPUTES:                                                               │
│   • entropy_velocity: dH/dt                                            │
│   • entropy_acceleration: d²H/dt²                                      │
│   • entropy_history: H over time                                       │
│                                                                         │
│ TRIGGERS:                                                               │
│   • ENTROPY_DROPPING: Belief focusing                                  │
│   • ENTROPY_RISING: Belief spreading                                   │
│   • ENTROPY_COLLAPSED: Near minimum                                    │
│   • ENTROPY_SATURATED: Near maximum                                    │
└─────────────────────────────────────────────────────────────────────────┘
```

### 10.3 Collapse Detector

```
ENTITY: CollapseDetector

┌─────────────────────────────────────────────────────────────────────────┐
│ TYPE: Monitor                                                           │
│ ID: COLLAPSE_DET                                                        │
├─────────────────────────────────────────────────────────────────────────┤
│ OBSERVES:                                                               │
│   • Entropy trajectory                                                  │
│   • Loss trajectory                                                     │
│   • Attention sharpness                                                 │
│   • Prediction stability                                                │
│                                                                         │
│ DETECTION CRITERIA:                                                     │
│   • Entropy drop: dH/dt < -threshold                                   │
│   • Loss drop: dL/dt < -threshold                                      │
│   • Attention spike: max(A) jumps                                      │
│   • Prediction jump: ‖ŷ_t - ŷ_{t-1}‖ > threshold                      │
│                                                                         │
│ OUTPUTS:                                                                │
│   • collapse_detected: boolean                                         │
│   • collapse_time: when it happened                                    │
│   • collapse_location: where in space                                  │
│   • collapse_magnitude: how strong                                     │
│                                                                         │
│ EVENTS:                                                                 │
│   • COLLAPSE_IMMINENT: Approaching threshold                           │
│   • COLLAPSE_OCCURRED: Threshold crossed                               │
│   • POST_COLLAPSE: Recovery phase entered                              │
└─────────────────────────────────────────────────────────────────────────┘
```

### 10.4 Pump Cycle Monitor

```
ENTITY: PumpCycleMonitor

┌─────────────────────────────────────────────────────────────────────────┐
│ TYPE: Monitor                                                           │
│ ID: PUMP_MON                                                            │
├─────────────────────────────────────────────────────────────────────────┤
│ PHASE STATES:                                                           │
│   • TENSION: Entropy rising, error spreading                           │
│   • CRITICALITY: Entropy peaked, interference forming                  │
│   • DISCHARGE: Entropy dropping, collapse occurring                    │
│   • RECOVERY: Entropy rising again, new cycle starting                 │
│                                                                         │
│ TRANSITIONS:                                                            │
│   TENSION → CRITICALITY: dH/dt → 0                                     │
│   CRITICALITY → DISCHARGE: dH/dt < -threshold                          │
│   DISCHARGE → RECOVERY: H reaches minimum                              │
│   RECOVERY → TENSION: dH/dt → 0 (rising)                               │
│                                                                         │
│ OUTPUTS:                                                                │
│   • current_phase: {TENSION, CRITICAL, DISCHARGE, RECOVERY}            │
│   • phase_duration: How long in current phase                          │
│   • cycle_count: Number of complete cycles                             │
│   • cycle_period: Average time per cycle                               │
│                                                                         │
│ EVENTS:                                                                 │
│   • PHASE_TRANSITION: Phase changed                                    │
│   • CYCLE_COMPLETE: Full cycle finished                                │
│   • ANOMALOUS_CYCLE: Unexpected dynamics                               │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 11. Complex Nested Operations

### 11.1 MAP Algorithm Components

```
ENTITY: MAPInference

┌─────────────────────────────────────────────────────────────────────────┐
│ TYPE: Composite                                                         │
│ ID: MAP_INFER                                                           │
├─────────────────────────────────────────────────────────────────────────┤
│ DESCRIPTION:                                                            │
│   Maximum A Posteriori inference over latent space.                    │
│   Finds the most likely hidden state given observations.               │
│                                                                         │
│ SUB-ENTITIES:                                                           │
│   • PRIOR_COMPUTER: p(z) from learned manifold                        │
│   • LIKELIHOOD_COMPUTER: p(x|z) from observation model                │
│   • POSTERIOR_COMPUTER: p(z|x) ∝ p(x|z)p(z)                          │
│   • MODE_FINDER: argmax_z p(z|x)                                      │
│                                                                         │
│ MONITORS:                                                               │
│   • posterior_entropy: Uncertainty in MAP                              │
│   • mode_stability: Does MAP change smoothly?                          │
│   • likelihood_score: p(x|z_MAP)                                      │
└─────────────────────────────────────────────────────────────────────────┘
```

### 11.2 Fisher Information Computer

```
ENTITY: FisherInformationComputer

┌─────────────────────────────────────────────────────────────────────────┐
│ TYPE: Composite                                                         │
│ ID: FISHER_COMP                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│ COMPUTES:                                                               │
│   F_ij = E[(∂log p/∂θ_i)(∂log p/∂θ_j)]                                │
│                                                                         │
│ SUB-OPERATIONS:                                                         │
│   • SCORE_FUNCTION: ∂log p(x|θ)/∂θ                                    │
│   • OUTER_PRODUCT: score ⊗ score                                       │
│   • EXPECTATION: Average over data                                     │
│                                                                         │
│ OUTPUTS:                                                                │
│   • fisher_matrix: [N_params, N_params]                                │
│   • fisher_diagonal: [N_params] (approximation)                        │
│   • natural_gradient: F^{-1} ∇L                                        │
│                                                                         │
│ USES:                                                                   │
│   • Natural gradient descent                                            │
│   • Measuring parameter importance                                      │
│   • Information geometry analysis                                       │
└─────────────────────────────────────────────────────────────────────────┘
```

### 11.3 Causal Tree Culler

```
ENTITY: CausalTreeCuller

┌─────────────────────────────────────────────────────────────────────────┐
│ TYPE: Composite                                                         │
│ ID: CAUSAL_CULL                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│ DESCRIPTION:                                                            │
│   The "Old Lady" operation, distill trajectories to atomic truths.    │
│   Prune details with no causal force on outcomes.                      │
│                                                                         │
│ SUB-OPERATIONS:                                                         │
│   • TRAJECTORY_TRACER: Follow causal lineages                          │
│   • OUTCOME_CORRELATOR: Which details affect outcome?                  │
│   • DETAIL_PRUNER: Remove non-causal details                          │
│   • TRUTH_COMPRESSOR: Collapse to atomic form                         │
│   • MANIFOLD_STORER: Save in low-freq manifold                        │
│                                                                         │
│ EVENTS:                                                                 │
│   • DETAIL_CULLED: Non-causal detail removed                           │
│   • TRUTH_EXTRACTED: Atomic truth identified                           │
│   • PAGE_TORN: Full trajectory released                                │
│   • BLANK_PAGE_ADDED: Capacity restored                                │
└─────────────────────────────────────────────────────────────────────────┘
```

### 11.4 Belief Reconstructor

```
ENTITY: BeliefReconstructor

┌─────────────────────────────────────────────────────────────────────────┐
│ TYPE: Composite                                                         │
│ ID: BELIEF_RECON                                                        │
├─────────────────────────────────────────────────────────────────────────┤
│ DESCRIPTION:                                                            │
│   Reconstruct explicit belief state from implicit representations.     │
│   Turns attention weights into probability distributions.              │
│                                                                         │
│ SUB-OPERATIONS:                                                         │
│   • ATTENTION_AGGREGATOR: Combine attention from all heads             │
│   • ENTROPY_COMPUTER: Compute belief entropy                           │
│   • MODE_FINDER: Find belief modes                                     │
│   • CONFIDENCE_COMPUTER: How concentrated is belief?                   │
│   • DISTRIBUTION_FITTER: Fit parametric distribution                   │
│                                                                         │
│ OUTPUTS:                                                                │
│   • belief_mean: Expected value                                        │
│   • belief_variance: Uncertainty                                       │
│   • belief_modes: Local maxima                                         │
│   • belief_entropy: Information content                                │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 12. Entity Interactions

### 12.1 Interaction Graph

```
ENTITY INTERACTION GRAPH

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│                      ┌──────────────────┐                              │
│                      │   INPUT FRAME    │                              │
│                      └────────┬─────────┘                              │
│                               │                                        │
│                               ▼                                        │
│                      ┌──────────────────┐                              │
│                      │ SPEC_DECOMPOSER  │                              │
│                      └────────┬─────────┘                              │
│                               │                                        │
│            ┌──────────────────┼──────────────────┐                     │
│            │                  │                  │                     │
│            ▼                  ▼                  ▼                     │
│     ┌────────────┐    ┌────────────┐    ┌────────────┐                │
│     │ TEMP_ATTN  │    │ NEIGH_ATTN │    │ WORM_ATTN  │                │
│     └─────┬──────┘    └─────┬──────┘    └─────┬──────┘                │
│           │                 │                 │                        │
│           │    ┌────────────┴────────────┐    │                        │
│           │    │                         │    │                        │
│           ▼    ▼                         ▼    ▼                        │
│           ┌──────────────────────────────────┐                         │
│           │          COMBINER                │                         │
│           └──────────────┬───────────────────┘                         │
│                          │                                             │
│                          ▼                                             │
│           ┌──────────────────────────────────┐                         │
│           │         PREDICTION               │                         │
│           └──────────────┬───────────────────┘                         │
│                          │                                             │
│              ┌───────────┴───────────┐                                 │
│              │                       │                                 │
│              ▼                       ▼                                 │
│     ┌──────────────┐        ┌──────────────┐                          │
│     │    LOSS      │        │   OUTPUT     │                          │
│     └──────┬───────┘        └──────────────┘                          │
│            │                                                           │
│            ▼                                                           │
│     ┌──────────────┐                                                   │
│     │  GRADIENTS   │                                                   │
│     └──────┬───────┘                                                   │
│            │                                                           │
│            ▼                                                           │
│     ┌──────────────┐                                                   │
│     │ WEIGHT_UPD   │                                                   │
│     └──────────────┘                                                   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 12.2 Event Flow

```
EVENT FLOW SEQUENCE

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  TIME  EVENT                    TRIGGERS                               │
│  ────  ─────                    ────────                               │
│                                                                         │
│  t+0   FRAME_ARRIVAL            → SPEC_DECOMP starts                   │
│  t+1   BANDS_READY              → TEMP_ATTN, NEIGH_ATTN, WORM_ATTN    │
│  t+2   HISTORY_UPDATED          → Temporal context available           │
│  t+3   WORMHOLES_COMPUTED       → Non-local connections ready          │
│  t+4   ATTENTION_COMBINED       → Features ready for prediction        │
│  t+5   PREDICTION_MADE          → Output available                     │
│  t+6   LOSS_COMPUTED            → Gradient computation starts          │
│  t+7   GRADIENTS_READY          → Weight update possible               │
│  t+8   WEIGHTS_UPDATED          → Model modified                       │
│                                                                         │
│  PARALLEL MONITORING:                                                   │
│  t+*   ENTROPY_TRACKED          → Continuous                           │
│  t+*   COLLAPSE_MONITORED       → Watching for events                  │
│  t+*   PUMP_PHASE_UPDATED       → Track cycle                          │
│                                                                         │
│  CONDITIONAL EVENTS:                                                    │
│  ?     COLLAPSE_DETECTED        → Log, visualize, analyze              │
│  ?     PHASE_TRANSITION         → Update statistics                    │
│  ?     DIVERGENCE_DETECTED      → Emergency response                   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 13. Event Streams

### 13.1 Observable Event Streams

```
EVENT STREAM DEFINITIONS

┌─────────────────────────────────────────────────────────────────────────┐
│ STREAM ID             CONTENT                  RATE                    │
├─────────────────────────────────────────────────────────────────────────┤
│ FRAME_STREAM          Input frames             1 per frame             │
│ PREDICTION_STREAM     Model predictions        1 per frame             │
│ ERROR_STREAM          Error maps               1 per frame             │
│                                                                         │
│ ATTENTION_STREAM      All attention weights    3 per frame             │
│ WORMHOLE_STREAM       Connection events        Variable                │
│ GRADIENT_STREAM       Gradient statistics      1 per step              │
│                                                                         │
│ ENTROPY_STREAM        Entropy values           1 per frame             │
│ COLLAPSE_STREAM       Collapse events          Sparse                  │
│ PHASE_STREAM          Pump cycle phase         1 per frame             │
│                                                                         │
│ EMBEDDING_STREAM      Reduced embeddings       1 per frame             │
│ BELIEF_STREAM         Belief cloud samples     N per frame             │
│                                                                         │
│ METRIC_STREAM         All scalar metrics       1 per frame             │
│ DIAGNOSTIC_STREAM     System health            Periodic                │
└─────────────────────────────────────────────────────────────────────────┘
```

### 13.2 Stream Subscription API

```python
"""
EVENT STREAM SUBSCRIPTION API
"""

class EventBus:
    """Central event bus for AKIRA observability."""
    
    def __init__(self):
        self.subscribers = defaultdict(list)
        self.event_history = defaultdict(list)
        
    def subscribe(self, stream_id: str, callback: Callable):
        """Subscribe to an event stream."""
        self.subscribers[stream_id].append(callback)
        
    def publish(self, stream_id: str, event: dict):
        """Publish an event to a stream."""
        event['timestamp'] = time.time()
        event['stream'] = stream_id
        
        # Store in history
        self.event_history[stream_id].append(event)
        
        # Notify subscribers
        for callback in self.subscribers[stream_id]:
            callback(event)
            
    def query_history(self, stream_id: str, 
                      start_time: float = None,
                      end_time: float = None) -> list:
        """Query historical events."""
        events = self.event_history[stream_id]
        
        if start_time:
            events = [e for e in events if e['timestamp'] >= start_time]
        if end_time:
            events = [e for e in events if e['timestamp'] <= end_time]
            
        return events


# Example usage
event_bus = EventBus()

# Subscribe to collapse events
event_bus.subscribe('COLLAPSE_STREAM', lambda e: 
    print(f"Collapse at t={e['timestamp']}: magnitude={e['magnitude']}")
)

# Subscribe to pump cycle phase changes
event_bus.subscribe('PHASE_STREAM', lambda e:
    update_visualization(e['phase'])
)
```

---

## Summary

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│                    E V E N T S   C A T A L O G                          │
│                                                                         │
│  ═══════════════════════════════════════════════════════════════════    │
│                                                                         │
│  COMPUTATIONAL ENTITIES:                                                │
│  • SpectralDecomposer, TemporalAttention, NeighborAttention             │
│  • WormholeAttention (with gates and triggers)                          │
│                                                                         │
│  STATE ENTITIES:                                                        │
│  • HistoryBuffer, WeightManifold, BeliefState                           │
│                                                                         │
│  SCALARS, VECTORS, MATRICES:                                            │
│  • Loss, gradients, attention weights, spectral bands                   │
│                                                                         │
│  GATES AND TRIGGERS:                                                    │
│  • Binary (threshold, ReLU) and Soft (softmax, sigmoid) etc.            │
│  • Event triggers (collapse, phase, divergence)                         │
│                                                                         │
│  MANIFOLDS:                                                             │
│  • Hypersphere, belief manifold, weight space, loss landscape           │
│                                                                         │
│  MONITORS:                                                              │
│  • GradientMonitor, EntropyTracker, CollapseDetector, PumpCycleMonitor  │
│                                                                         │
│  COMPOSITE OPERATIONS:                                                  │
│  • MAPInference, FisherInformation, CausalTreeCuller, BeliefReconstructor │
│                                                                         │
│  EVENT STREAMS:                                                         │
│  • Frame, prediction, error, attention, entropy, collapse, embedding    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

*Oscar Goldman, Shogu Research Group @ Datamutant.ai*

*"Every moving part is an entity. Every entity can be observed. Every observation reveals structure. This is the laboratory."*

