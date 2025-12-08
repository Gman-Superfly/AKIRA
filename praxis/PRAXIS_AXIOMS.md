# PRAXIS AXIOMS

## The Mathematical Foundations of AKIRA

**Oscar Goldman — Shogu Research Group @ Datamutant.ai**

---

*"The architecture is not arbitrary. It rests on axioms — mathematical truths that constrain what is possible and define what is optimal. Polar coordinates, polarization, Taylor expansions, hyperbolic geometry — these are not decorations. They are the bones of the system. To understand AKIRA, understand its axioms."*

---

## Table of Contents

1. [Introduction: Information Has Geometry](#1-introduction-information-has-geometry)
2. [Axiom I: Polar Coordinates — The Native Language](#2-axiom-i-polar-coordinates--the-native-language)
3. [Axiom II: Polarization — Orthogonal Decomposition](#3-axiom-ii-polarization--orthogonal-decomposition)
4. [Axiom III: Taylor Expansions — Local Approximation](#4-axiom-iii-taylor-expansions--local-approximation)
5. [Axiom IV: Hyperbolic Geometry — Hierarchical Structure](#5-axiom-iv-hyperbolic-geometry--hierarchical-structure)
6. [Axiom V: The Hypersphere — Current Implementation](#6-axiom-v-the-hypersphere--current-implementation)
7. [The Unified Geometric View](#7-the-unified-geometric-view)
8. [Connections Between Axioms](#8-connections-between-axioms)
9. [Implications for Praxis](#9-implications-for-praxis)
10. [Open Questions](#10-open-questions)

---

## 1. Introduction: Information Has Geometry

### 1.1 The Central Claim

Information is not formless. It has structure. More specifically, it has **geometry**. The mathematical structures that describe space, curvature, distance, and direction are the same structures that describe information, belief, and knowledge.

```
THE GEOMETRIC NATURE OF INFORMATION

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  Information is not just:                                              │
│  • Bits (yes/no)                                                       │
│  • Numbers (magnitudes)                                                │
│  • Vectors (lists)                                                     │
│                                                                         │
│  Information IS:                                                        │
│  • Positions on manifolds                                              │
│  • Directions in spaces                                                │
│  • Distances between points                                            │
│  • Curvature of surfaces                                               │
│  • Geodesics (shortest paths)                                          │
│                                                                         │
│  ════════════════════════════════════════════════════════════════════  │
│                                                                         │
│  This is not metaphor.                                                 │
│  This is mathematics — specifically, information geometry.             │
│  Riemannian geometry applies to both physical manifolds               │
│  and spaces of probability distributions (Amari, 1985).               │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Why Axioms?

We call these "axioms" because they are foundational truths upon which the architecture rests. They are not arbitrary choices. They are mathematical necessities.

```
AXIOMS VS CHOICES

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  AXIOMS (mathematical necessities):                                    │
│                                                                         │
│  • FFT produces complex numbers → polar structure                     │
│  • Functions can be approximated locally → Taylor expansion           │
│  • Hierarchies have exponential growth → hyperbolic geometry          │
│  • Orthogonal components are independent → polarization               │
│                                                                         │
│  These are not design choices. They are facts.                        │
│                                                                         │
│  ════════════════════════════════════════════════════════════════════  │
│                                                                         │
│  CHOICES (design decisions):                                           │
│                                                                         │
│  • 7 bands (not 5 or 9)                                               │
│  • Threshold of 0.92 (not 0.85 or 0.95)                               │
│  • Hamming window (not Hanning or Blackman)                           │
│                                                                         │
│  These are informed by axioms but not determined by them.             │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1.3 The Core Dynamic: Tension and Collapse

Before presenting the axioms, we must establish the fundamental cycle that drives belief evolution:

```
THE FUNDAMENTAL CYCLE OF BELIEF EVOLUTION

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  Redundancy transforms into Synergy through TENSION.                   │
│  Synergy collapses back into Redundancy through COLLAPSE.              │
│                                                                         │
│  • During TENSION: Uncertainty ACCUMULATES (redundancy → synergy)      │
│  • During COLLAPSE: Uncertainty RESOLVES (synergy → redundancy)        │
│                                                                         │
│  THE PUMP CYCLE:                                                        │
│  [Redundancy] ──TENSION──> [Synergy] ──COLLAPSE──> [Redundancy] + AQ  │
│                                                                         │
│  The axioms below describe the GEOMETRY in which this cycle operates. │
│  See foundations/TERMINOLOGY.md for formal definitions.               │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1.4 The Five Axioms

```
THE PRAXIS AXIOMS

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  I.   POLAR COORDINATES                                                │
│       The FFT outputs complex numbers.                                 │
│       Complex numbers have magnitude and phase.                        │
│       Magnitude = "what", Phase = "where".                            │
│                                                                         │
│  II.  POLARIZATION                                                      │
│       Signals decompose into orthogonal components.                   │
│       Components are independent.                                      │
│       Filtering isolates components.                                  │
│                                                                         │
│  III. TAYLOR EXPANSION                                                  │
│       Complex functions can be approximated locally.                  │
│       Gradient = first-order approximation.                           │
│       Curvature = second-order approximation.                         │
│                                                                         │
│  IV.  HYPERBOLIC GEOMETRY                                               │
│       Hierarchies fit naturally in hyperbolic space.                  │
│       Trees have exponential growth.                                  │
│       Euclidean space cannot embed them efficiently.                  │
│                                                                         │
│  V.   SPHERICAL GEOMETRY (current implementation)                      │
│       Normalization projects onto hypersphere.                        │
│       Cosine similarity = angular distance.                           │
│       Softmax = probability over directions.                          │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Axiom I: Polar Coordinates — The Native Language

### 2.1 The Fundamental Structure

The Fast Fourier Transform (FFT) transforms spatial data into frequency data. The output is not real numbers — it is **complex numbers**. Complex numbers are best understood in **polar form**:

```
COMPLEX NUMBERS: CARTESIAN VS POLAR

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  CARTESIAN FORM:                                                        │
│  z = x + iy                                                            │
│  where x = real part, y = imaginary part                              │
│                                                                         │
│  POLAR FORM:                                                            │
│  z = r · e^(iθ)                                                        │
│  where r = magnitude, θ = phase                                       │
│                                                                         │
│  CONVERSION:                                                            │
│  r = √(x² + y²)        (magnitude)                                    │
│  θ = arctan(y/x)       (phase)                                        │
│  x = r·cos(θ)          (real part)                                    │
│  y = r·sin(θ)          (imaginary part)                               │
│                                                                         │
│  ════════════════════════════════════════════════════════════════════  │
│                                                                         │
│  Euler's Identity: e^(iθ) = cos(θ) + i·sin(θ)                         │
│                                                                         │
│  This is not arbitrary notation.                                       │
│  This is the NATURAL structure of frequency decomposition.            │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Magnitude and Phase in AKIRA

```
WHAT MAGNITUDE AND PHASE ENCODE

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  MAGNITUDE (r):                                                         │
│  ─────────────                                                         │
│  • "How much" of this frequency is present                            │
│  • The AMPLITUDE or STRENGTH of the pattern                           │
│  • WHAT the pattern is                                                │
│  • IDENTITY information                                                │
│                                                                         │
│  If magnitude is large: This frequency is important here.             │
│  If magnitude is small: This frequency is absent.                     │
│                                                                         │
│  ════════════════════════════════════════════════════════════════════  │
│                                                                         │
│  PHASE (θ):                                                             │
│  ──────────                                                            │
│  • "Where in the cycle" this frequency is                             │
│  • The POSITION or ALIGNMENT of the pattern                           │
│  • WHERE the pattern is located                                       │
│  • POSITION information                                                │
│                                                                         │
│  Same magnitude, different phase = same pattern, different position.  │
│  Phase encodes SPATIAL LOCATION.                                      │
│                                                                         │
│  ════════════════════════════════════════════════════════════════════  │
│                                                                         │
│  TOGETHER:                                                              │
│  • Magnitude tells you: "There's an edge here, strength 0.7"         │
│  • Phase tells you: "That edge is at position (32, 48)"              │
│  • Both are needed for complete information.                          │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.3 Quadrature Pairs

To recover phase from real signals, we need **quadrature pairs** — sine and cosine at the same frequency:

```
QUADRATURE: WHY WE NEED TWO CHANNELS

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  A real signal can be written as:                                      │
│                                                                         │
│  s(t) = A·cos(ωt + φ)                                                  │
│       = A·cos(φ)·cos(ωt) - A·sin(φ)·sin(ωt)                           │
│       = a·cos(ωt) + b·sin(ωt)                                         │
│                                                                         │
│  where a = A·cos(φ) and b = -A·sin(φ)                                 │
│                                                                         │
│  From a and b, we can recover:                                         │
│  A = √(a² + b²)        (magnitude)                                    │
│  φ = arctan(-b/a)      (phase)                                        │
│                                                                         │
│  ════════════════════════════════════════════════════════════════════  │
│                                                                         │
│  This is why we need TWO channels (sin and cos) per frequency:        │
│                                                                         │
│  • One channel alone gives magnitude at fixed phase                   │
│  • Two channels (quadrature) give magnitude AND phase                 │
│  • Phase = position information                                       │
│                                                                         │
│  TWO PHASES ARE SUFFICIENT for complete position information.         │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.4 Implications for Action Quanta

The Action Quanta we documented in [THE_ATOMIC_STRUCTURE_OF_INFORMATION.md](./THE_ATOMIC_STRUCTURE_OF_INFORMATION.md) have **polar structure**:

```
ACTION QUANTA ARE POLAR

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  Each Action Quantum has:                                            │
│                                                                         │
│  1. FREQUENCY (which band)                                             │
│  2. MAGNITUDE (how strong)                                             │
│  3. PHASE (where located)                                              │
│  4. ORIENTATION (which direction, for 2D)                             │
│                                                                         │
│  The first three are the polar structure.                             │
│  The fourth extends to 2D (orientation = which angle).                │
│                                                                         │
│  ════════════════════════════════════════════════════════════════════  │
│                                                                         │
│  EXAMPLE: An "edge" atom                                               │
│                                                                         │
│  • Frequency: Band 3 (mid-frequency)                                  │
│  • Magnitude: 0.7 (moderately strong)                                 │
│  • Phase: 45° (specific position in space)                            │
│  • Orientation: 90° (vertical edge)                                   │
│                                                                         │
│  This completely describes a "vertical edge at position X".           │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

*Reference: [THE_ATOMIC_STRUCTURE_OF_INFORMATION.md](./THE_ATOMIC_STRUCTURE_OF_INFORMATION.md)*

---

## 3. Axiom II: Polarization — Orthogonal Decomposition

### 3.1 Physical Polarization

In physics, polarization refers to the direction of oscillation of a wave. Light can be decomposed into **orthogonal polarization states**:

```
LIGHT POLARIZATION

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  Unpolarized light contains all orientations.                         │
│                                                                         │
│  Polarization decomposes it:                                           │
│  • Horizontal component (H)                                            │
│  • Vertical component (V)                                              │
│                                                                         │
│  Complete description: E = E_H + E_V                                  │
│                                                                         │
│  ════════════════════════════════════════════════════════════════════  │
│                                                                         │
│  PROPERTIES:                                                            │
│                                                                         │
│  • H and V are ORTHOGONAL (independent)                               │
│  • Together they SPAN the space (complete basis)                      │
│  • A polarizing filter SELECTS one, BLOCKS the other                  │
│  • Different filters reveal different information                     │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Spectral Polarization in AKIRA

AKIRA's spectral bands act as **frequency polarizers**:

```
SPECTRAL POLARIZATION ANALOGY

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  LIGHT:                           AKIRA:                               │
│  ─────                           ─────                                 │
│  Horizontal + Vertical           Low-freq + Mid-freq + High-freq      │
│  = complete polarization         = complete spectral coverage         │
│                                                                         │
│  Polarizing filter:              Band-pass filter:                     │
│  Passes only one orientation     Passes only one frequency range      │
│  Blocks the other                Blocks the others                    │
│                                                                         │
│  ════════════════════════════════════════════════════════════════════  │
│                                                                         │
│  THE 7 BANDS AS POLARIZATION STATES:                                   │
│                                                                         │
│  Band 0 (DC):      The "DC polarization" — average, existence        │
│  Band 1-2 (Low):   "Low-frequency polarization" — structure          │
│  Band 3-4 (Mid):   "Mid-frequency polarization" — features           │
│  Band 5-6 (High):  "High-frequency polarization" — details           │
│                                                                         │
│  Each band is ORTHOGONAL to the others.                               │
│  Together they SPAN the frequency space.                              │
│  Each band reveals DIFFERENT information.                             │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.3 Independence of Bands

Because the bands are orthogonal (in the spectral sense), they carry **independent information**:

```
ORTHOGONALITY = INDEPENDENCE

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  MATHEMATICAL:                                                          │
│  ─────────────                                                         │
│  ∫ band_i(f) · band_j(f) df = 0    for i ≠ j                         │
│                                                                         │
│  The bands don't overlap in frequency.                                │
│  What's in one band is NOT in another.                                │
│                                                                         │
│  INFORMATIONAL:                                                         │
│  ──────────────                                                        │
│  • Low bands can change without affecting high bands                  │
│  • High bands can change without affecting low bands                  │
│  • Each band can be processed independently                           │
│                                                                         │
│  IMPLICATIONS:                                                          │
│  ─────────────                                                         │
│  • Different learning rates per band (see TEMPORAL_SYSTEM.md)        │
│  • Different attention patterns per band                              │
│  • Cross-band attention (wormhole) is explicitly designed            │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.4 Wormhole as Polarization Rotation

The wormhole attention can be understood as a **polarization rotator** — it connects information across bands that would otherwise be isolated:

```
WORMHOLE AS POLARIZATION ROTATION

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  WITHOUT WORMHOLE:                                                      │
│  ─────────────────                                                      │
│  Each band is isolated.                                                │
│  Low-freq information stays in low-freq band.                         │
│  High-freq information stays in high-freq band.                       │
│  Like polarized light — each component separate.                      │
│                                                                         │
│  WITH WORMHOLE:                                                         │
│  ──────────────                                                        │
│  Cross-band connections are possible.                                 │
│  High-freq detail can inform low-freq structure.                     │
│  Low-freq context can inform high-freq interpretation.               │
│  Like a polarization rotator — mixing components.                    │
│                                                                         │
│  ════════════════════════════════════════════════════════════════════  │
│                                                                         │
│  The wormhole finds SPECTRAL COMPATIBILITY:                           │
│  • Match low-freq (what) to high-freq (where)                        │
│  • Connect across bands when similarity is high                       │
│  • Enable information flow that bypasses local structure             │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

*Reference: [WORMHOLE_HYBRID.md](./wormhole/WORMHOLE_HYBRID.md)*

---

## 4. Axiom III: Taylor Expansions — Local Approximation

### 4.1 The Mathematical Foundation

Taylor expansion approximates a function locally around a point:

```
TAYLOR EXPANSION

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  For a smooth function f(x) near point x₀:                            │
│                                                                         │
│  f(x₀ + δ) ≈ f(x₀) + f'(x₀)·δ + ½f''(x₀)·δ² + ⅙f'''(x₀)·δ³ + ...   │
│                                                                         │
│  TERMS:                                                                 │
│  ──────                                                                │
│  f(x₀)           = VALUE at the point                                 │
│  f'(x₀)·δ        = GRADIENT (first derivative) × step                │
│  ½f''(x₀)·δ²     = CURVATURE (second derivative) × step²             │
│  Higher terms    = Higher-order corrections                           │
│                                                                         │
│  ════════════════════════════════════════════════════════════════════  │
│                                                                         │
│  KEY INSIGHT:                                                           │
│                                                                         │
│  Complex, curved functions can be approximated LOCALLY as:            │
│  • A constant (zeroth order)                                          │
│  • A line (first order)                                               │
│  • A parabola (second order)                                          │
│                                                                         │
│  The approximation is good NEAR the point.                            │
│  It fails FAR from the point.                                         │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Gradient Descent as First-Order Taylor

When we update weights in neural networks, we use the gradient. This is a **first-order Taylor approximation**:

```
GRADIENT DESCENT = FIRST-ORDER TAYLOR

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  LOSS FUNCTION L(w):                                                   │
│                                                                         │
│  We want to minimize L(w).                                            │
│  We're at weights w.                                                  │
│  We want to find better weights w + Δw.                               │
│                                                                         │
│  TAYLOR APPROXIMATION:                                                  │
│                                                                         │
│  L(w + Δw) ≈ L(w) + ∇L(w)·Δw + O(Δw²)                                │
│                                                                         │
│  First-order approximation: L(w + Δw) ≈ L(w) + ∇L(w)·Δw              │
│                                                                         │
│  To minimize, move opposite to gradient:                              │
│  Δw = -η·∇L(w)                                                        │
│                                                                         │
│  ════════════════════════════════════════════════════════════════════  │
│                                                                         │
│  THIS IS GRADIENT DESCENT:                                              │
│                                                                         │
│  We approximate the complex loss surface as a PLANE locally.         │
│  We step downhill on this plane.                                      │
│  We repeat, re-approximating at each step.                           │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.3 Fisher Information as Second-Order Structure

The **Fisher Information Matrix** captures the curvature (second derivative) of the log-likelihood:

```
FISHER INFORMATION = CURVATURE

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  FISHER INFORMATION MATRIX:                                             │
│                                                                         │
│  F = E[(∇ log p(x|θ)) · (∇ log p(x|θ))ᵀ]                             │
│                                                                         │
│  This is the EXPECTED OUTER PRODUCT of the score function.           │
│  It equals the NEGATIVE EXPECTED HESSIAN of log-likelihood:          │
│                                                                         │
│  F = -E[∇² log p(x|θ)]                                                │
│                                                                         │
│  ════════════════════════════════════════════════════════════════════  │
│                                                                         │
│  INTERPRETATION:                                                        │
│                                                                         │
│  • F tells us how the likelihood CURVES in parameter space           │
│  • Large eigenvalues = steep directions = sensitive parameters       │
│  • Small eigenvalues = flat directions = insensitive parameters      │
│  • F defines a METRIC on parameter space (information geometry)      │
│                                                                         │
│  ════════════════════════════════════════════════════════════════════  │
│                                                                         │
│  SECOND-ORDER TAYLOR:                                                   │
│                                                                         │
│  log p(θ + δ) ≈ log p(θ) + ∇log p·δ + ½ δᵀ·(∇²log p)·δ              │
│              ≈ log p(θ) + ∇log p·δ - ½ δᵀ·F·δ                        │
│                                                                         │
│  The Fisher matrix IS the second-order term.                         │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.4 Natural Gradient: Geometry-Aware Learning

The **natural gradient** uses the Fisher matrix to account for the geometry of parameter space:

```
NATURAL GRADIENT

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  STANDARD GRADIENT:                                                     │
│  ──────────────────                                                     │
│  Δθ = -η·∇L                                                           │
│                                                                         │
│  This treats parameter space as EUCLIDEAN.                            │
│  All directions are equally important.                                │
│                                                                         │
│  NATURAL GRADIENT:                                                      │
│  ─────────────────                                                      │
│  Δθ = -η·F⁻¹·∇L                                                       │
│                                                                         │
│  This accounts for the GEOMETRY of the manifold.                      │
│  It moves equal "information distance" rather than equal parameter.  │
│                                                                         │
│  ════════════════════════════════════════════════════════════════════  │
│                                                                         │
│  WHY THIS MATTERS:                                                      │
│                                                                         │
│  • In steep directions (large F), take smaller steps                 │
│  • In flat directions (small F), take larger steps                   │
│  • This is GEOMETRY-AWARE optimization                               │
│  • Faster convergence, more stable learning                          │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.5 Belief Updates as Local Linearization

When the belief state updates, it moves on a manifold. Locally, we can approximate:

```
BELIEF UPDATE AS LOCAL LINEARIZATION

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  The belief state lives on a MANIFOLD (curved surface).               │
│                                                                         │
│  At any point, the manifold has a TANGENT SPACE:                      │
│  • Locally flat approximation                                         │
│  • Directions you can move                                            │
│  • Where the gradient lives                                           │
│                                                                         │
│  BELIEF UPDATE:                                                         │
│  ──────────────                                                        │
│  belief_new = belief_old + tangent_vector                             │
│                                                                         │
│  This is a FIRST-ORDER approximation:                                  │
│  • Treat the curved manifold as flat locally                         │
│  • Move in the tangent direction                                      │
│  • Repeat, re-linearizing at each step                               │
│                                                                         │
│  ════════════════════════════════════════════════════════════════════  │
│                                                                         │
│  The COLLAPSE phenomenon:                                               │
│                                                                         │
│  When belief collapses, we're moving from:                            │
│  • Diffuse region (many directions possible)                         │
│  → Concentrated region (one direction dominates)                     │
│                                                                         │
│  Taylor expansion helps: near the minimum, local approximation works. │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

*Reference: [EQUILIBRIUM_AND_CONSERVATION.md](./EQUILIBRIUM_AND_CONSERVATION.md)*

---

## 5. Axiom IV: Hyperbolic Geometry — Hierarchical Structure

### 5.1 The Poincaré Half-Plane

The **upper half Poincaré plane** is a model of hyperbolic geometry:

```
POINCARÉ UPPER HALF-PLANE

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  DEFINITION:                                                            │
│  ───────────                                                           │
│  H = {z ∈ ℂ : Im(z) > 0}                                              │
│                                                                         │
│  The set of complex numbers with positive imaginary part.             │
│  (The upper half of the complex plane.)                               │
│                                                                         │
│  METRIC:                                                                │
│  ───────                                                               │
│  ds² = (dx² + dy²) / y²                                               │
│                                                                         │
│  Distance is measured differently than in Euclidean space.            │
│  Near y=0 (the boundary), distances are LARGE.                       │
│  High up (large y), distances are SMALL.                              │
│                                                                         │
│  GEODESICS:                                                             │
│  ──────────                                                            │
│  Shortest paths are:                                                   │
│  • Vertical lines (perpendicular to the real axis)                   │
│  • Semicircles with center on the real axis                          │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 5.2 Why Hyperbolic Geometry for Hierarchies

Trees and hierarchies have **exponential growth**. In Euclidean space, volume grows polynomially. There's a mismatch:

```
EXPONENTIAL VS POLYNOMIAL GROWTH

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  BINARY TREE:                                                           │
│  ────────────                                                          │
│  Depth 1:  2 nodes                                                     │
│  Depth 2:  4 nodes                                                     │
│  Depth 3:  8 nodes                                                     │
│  Depth n:  2ⁿ nodes                                                   │
│                                                                         │
│  Growth: EXPONENTIAL                                                   │
│                                                                         │
│  ════════════════════════════════════════════════════════════════════  │
│                                                                         │
│  EUCLIDEAN SPACE (dimension d):                                        │
│  ──────────────────────────────                                        │
│  Radius 1:  ~ 1^d volume                                              │
│  Radius 2:  ~ 2^d volume                                              │
│  Radius r:  ~ r^d volume                                              │
│                                                                         │
│  Growth: POLYNOMIAL (in radius)                                        │
│                                                                         │
│  ════════════════════════════════════════════════════════════════════  │
│                                                                         │
│  MISMATCH:                                                              │
│                                                                         │
│  To embed exponential tree in polynomial space:                       │
│  Need very high dimension OR lots of distortion.                      │
│                                                                         │
│  ════════════════════════════════════════════════════════════════════  │
│                                                                         │
│  HYPERBOLIC SPACE:                                                      │
│  ─────────────────                                                     │
│  Volume grows EXPONENTIALLY with radius!                              │
│  Trees embed naturally with low distortion.                           │
│  Need only ~2 dimensions for most hierarchies.                        │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 5.3 The Poincaré Disk Model

An equivalent model, often more intuitive for visualization:

```
POINCARÉ DISK MODEL

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  DEFINITION:                                                            │
│  ───────────                                                           │
│  D = {z ∈ ℂ : |z| < 1}                                                │
│                                                                         │
│  The open unit disk in the complex plane.                             │
│                                                                         │
│  METRIC:                                                                │
│  ───────                                                               │
│  ds² = 4(dx² + dy²) / (1 - x² - y²)²                                 │
│                                                                         │
│  Distance increases as you approach the boundary.                     │
│  The boundary (circle) is "infinitely far away."                     │
│                                                                         │
│  INTERPRETATION FOR HIERARCHIES:                                        │
│  ───────────────────────────────                                       │
│  • CENTER = root (most general)                                       │
│  • EDGE = leaves (most specific)                                      │
│  • Moving outward = going down the tree                              │
│  • Distance to edge = depth in hierarchy                             │
│                                                                         │
│           ●───────────────────────────●                                │
│          /                             \                               │
│         /      ●  ●  ●  ●  ●          \                               │
│        │     ●              ●          │                               │
│        │   ●       ROOT       ●        │                               │
│        │     ●      ●       ●          │                               │
│         \      ●  ●  ●  ●  ●          /                               │
│          \                             /                               │
│           ●───────────────────────────●                                │
│                                                                         │
│  Root near center, leaves near edge.                                  │
│  Lots of "room" at the edge for many leaves.                         │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 5.4 Spectral Hierarchy as Hyperbolic Structure

AKIRA's spectral bands form a hierarchy:

```
SPECTRAL BAND HIERARCHY

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  DC (Band 0) — root, everything, most general                         │
│    │                                                                    │
│    └── Low (1-2) — structure, large patterns                          │
│          │                                                              │
│          └── Mid (3-4) — features, identity                           │
│                │                                                        │
│                └── High (5-6) — details, edges, most specific         │
│                                                                         │
│  ════════════════════════════════════════════════════════════════════  │
│                                                                         │
│  This is TREE-LIKE:                                                     │
│                                                                         │
│  • DC is the root (one node, describes everything)                   │
│  • Low bands branch (few nodes, describe large regions)              │
│  • High bands are leaves (many nodes, describe details)              │
│                                                                         │
│  In Poincaré disk:                                                     │
│  • DC near center                                                     │
│  • High bands near edge                                               │
│  • Wormhole = geodesic connecting different levels                   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 5.5 Poincaré Embeddings

Recent work in machine learning uses hyperbolic embeddings for hierarchical data:

```
POINCARÉ EMBEDDINGS FOR HIERARCHIES

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  TRADITIONAL EMBEDDINGS (Euclidean):                                    │
│  ───────────────────────────────────                                   │
│  • Embed concepts in ℝⁿ                                               │
│  • Distance = Euclidean distance                                      │
│  • Works well for "similarity"                                        │
│  • Poor for "is-a" hierarchies                                        │
│                                                                         │
│  POINCARÉ EMBEDDINGS (Hyperbolic):                                      │
│  ─────────────────────────────────                                     │
│  • Embed concepts in Poincaré ball                                    │
│  • Distance = hyperbolic distance                                     │
│  • Naturally captures hierarchy                                       │
│  • General concepts near center, specific near edge                  │
│                                                                         │
│  ════════════════════════════════════════════════════════════════════  │
│                                                                         │
│  EXAMPLE (WordNet):                                                     │
│                                                                         │
│  "entity" (most general) → near center                               │
│  "living thing" → slightly outward                                   │
│  "animal" → more outward                                             │
│  "dog" → more outward                                                │
│  "poodle" (most specific) → near edge                                │
│                                                                         │
│  Hierarchy emerges from RADIAL position.                              │
│  Siblings are ANGULARLY close.                                        │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 5.6 Potential Application to AKIRA

```
HYPERBOLIC GEOMETRY IN AKIRA (POTENTIAL)

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  CURRENT: Spherical geometry (hypersphere normalization)              │
│  POTENTIAL: Hyperbolic geometry for band hierarchy                    │
│                                                                         │
│  HOW IT MIGHT WORK:                                                     │
│  ──────────────────                                                    │
│  • Low-freq embeddings live near Poincaré center                     │
│  • High-freq embeddings live near Poincaré edge                      │
│  • "General" patterns are central                                     │
│  • "Specific" patterns are peripheral                                 │
│                                                                         │
│  BENEFITS:                                                              │
│  ─────────                                                             │
│  • Natural representation of spectral hierarchy                       │
│  • Efficient low-dimensional embedding                               │
│  • Wormholes = hyperbolic geodesics                                  │
│  • Collapse = moving toward a specific radial position               │
│    (This is the GEOMETRIC perspective on collapse.                   │
│     See foundations/TERMINOLOGY.md for formal definitions.)          │
│                                                                         │
│  OPEN QUESTION:                                                         │
│  ──────────────                                                        │
│  Is spherical or hyperbolic better for AKIRA?                        │
│  Spherical: good for "similar vs different" (angular)               │
│  Hyperbolic: good for "general vs specific" (radial)                │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Axiom V: The Hypersphere — Current Implementation

### 6.1 Spherical Geometry

AKIRA currently uses **spherical geometry** via hypersphere normalization:

```
HYPERSPHERE NORMALIZATION

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  OPERATION:                                                             │
│  ──────────                                                            │
│  x_normalized = x / ||x||₂                                            │
│                                                                         │
│  This projects vectors onto the unit hypersphere S^(n-1).             │
│                                                                         │
│  ════════════════════════════════════════════════════════════════════  │
│                                                                         │
│  PROPERTIES:                                                            │
│                                                                         │
│  • All vectors have unit length: ||x_normalized|| = 1                │
│  • Distance is measured by ANGLE, not magnitude                      │
│  • Cosine similarity = dot product (on unit sphere)                  │
│  • cos(θ) = x · y when ||x|| = ||y|| = 1                            │
│                                                                         │
│  ════════════════════════════════════════════════════════════════════  │
│                                                                         │
│  IN WORMHOLE ATTENTION (from code):                                     │
│                                                                         │
│  query_norm = F.normalize(query_flat, p=2, dim=1)  # → unit sphere   │
│  key_norm = F.normalize(key_flat, p=2, dim=1)      # → unit sphere   │
│  sim = torch.matmul(query_norm, key_norm.T)        # = cos(θ)       │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 6.2 Angular Distance

On the hypersphere, distance is measured by angle:

```
ANGULAR DISTANCE ON HYPERSPHERE

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  COSINE SIMILARITY:                                                     │
│  ──────────────────                                                    │
│  cos(θ) = x · y / (||x|| · ||y||)                                    │
│                                                                         │
│  For unit vectors: cos(θ) = x · y                                    │
│                                                                         │
│  INTERPRETATION:                                                        │
│  ───────────────                                                       │
│  cos(θ) = 1:   Identical direction (θ = 0°)                          │
│  cos(θ) = 0:   Orthogonal (θ = 90°)                                  │
│  cos(θ) = -1:  Opposite direction (θ = 180°)                         │
│                                                                         │
│  ════════════════════════════════════════════════════════════════════  │
│                                                                         │
│  ANGULAR DISTANCE:                                                      │
│  ─────────────────                                                     │
│  d(x, y) = arccos(x · y) = θ                                         │
│                                                                         │
│  Or equivalently:                                                      │
│  d(x, y) = arccos(cos(θ)) = angle between vectors                    │
│                                                                         │
│  This is the GEODESIC distance on the sphere.                        │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 6.3 Softmax as Probability on Directions

The softmax function converts similarity scores to probabilities:

```
SOFTMAX ON SPHERE

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  SOFTMAX:                                                               │
│  ────────                                                              │
│  p(k) = exp(s_k / τ) / Σⱼ exp(s_j / τ)                               │
│                                                                         │
│  where s_k = similarity score, τ = temperature                       │
│                                                                         │
│  INTERPRETATION ON SPHERE:                                              │
│  ─────────────────────────                                             │
│  • Each key is a direction on the sphere                             │
│  • Query defines a "center of attention"                              │
│  • Softmax gives probability over directions                         │
│  • High probability = close angular distance                         │
│                                                                         │
│  ════════════════════════════════════════════════════════════════════  │
│                                                                         │
│  TEMPERATURE EFFECT:                                                    │
│  ────────────────────                                                  │
│  τ → 0: Sharp (argmax) — winner-take-all                             │
│  τ → ∞: Uniform — all directions equal                               │
│  τ ≈ 1: Balanced — probability proportional to similarity           │
│                                                                         │
│  The softmax temperature controls "focus" of attention.              │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 6.4 Spherical vs Hyperbolic: Comparison

```
SPHERICAL VS HYPERBOLIC GEOMETRY

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  SPHERICAL (current):                                                   │
│  ────────────────────                                                  │
│  • Finite volume                                                       │
│  • Angular distance (cosine similarity)                               │
│  • Good for: "similar vs different"                                   │
│  • All points are "equal" (no center/edge distinction)               │
│  • Geodesics are great circles                                        │
│                                                                         │
│  HYPERBOLIC (potential):                                                │
│  ───────────────────────                                               │
│  • Infinite volume (exponentially growing)                            │
│  • Radial + angular distance                                          │
│  • Good for: "general vs specific" (hierarchy)                       │
│  • Center = general, edge = specific                                  │
│  • Geodesics are hyperbolic arcs                                     │
│                                                                         │
│  ════════════════════════════════════════════════════════════════════  │
│                                                                         │
│  WHICH IS BETTER FOR AKIRA?                                             │
│                                                                         │
│  Spherical is good for WITHIN-BAND similarity:                        │
│  "Is this feature similar to that feature?"                          │
│                                                                         │
│  Hyperbolic might be better for CROSS-BAND hierarchy:                 │
│  "Is this detail an instance of that structure?"                     │
│                                                                         │
│  Possibly: Use both. Spherical within bands, hyperbolic across.      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

*Reference: [WORMHOLE_HYBRID.md](./wormhole/WORMHOLE_HYBRID.md)*

---

## 7. The Unified Geometric View

### 7.1 All Axioms Together

```
UNIFIED GEOMETRIC VIEW OF AKIRA

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  AXIOM I: POLAR COORDINATES (FFT output)                               │
│  ────────────────────────────────────────                              │
│  Magnitude = "what" (pattern identity)                                │
│  Phase = "where" (pattern position)                                   │
│  → Action Quanta have polar structure                             │
│                                                                         │
│  ════════════════════════════════════════════════════════════════════  │
│                                                                         │
│  AXIOM II: POLARIZATION (band decomposition)                           │
│  ───────────────────────────────────────────                           │
│  Each band = spectral "polarizer"                                     │
│  Filters one frequency range, blocks others                           │
│  → 7 bands = 7 polarization states of information                    │
│                                                                         │
│  ════════════════════════════════════════════════════════════════════  │
│                                                                         │
│  AXIOM III: TAYLOR EXPANSION (learning dynamics)                       │
│  ───────────────────────────────────────────────                       │
│  Gradient = first-order local approximation                          │
│  Fisher = second-order curvature                                     │
│  → Learning navigates manifold via local geometry                    │
│                                                                         │
│  ════════════════════════════════════════════════════════════════════  │
│                                                                         │
│  AXIOM IV: HYPERBOLIC GEOMETRY (hierarchy structure)                   │
│  ────────────────────────────────────────────────────                  │
│  Trees fit naturally in hyperbolic space                             │
│  Center = general, edge = specific                                   │
│  → Band hierarchy may have hyperbolic structure                      │
│                                                                         │
│  ════════════════════════════════════════════════════════════════════  │
│                                                                         │
│  AXIOM V: SPHERICAL GEOMETRY (current implementation)                  │
│  ──────────────────────────────────────────────────────                │
│  Normalization projects onto hypersphere                             │
│  Cosine similarity = angular distance                                │
│  → Attention weights are probabilities over directions               │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 7.2 The Big Picture

```
THE GEOMETRIC NATURE OF AKIRA

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│                        INFORMATION                                     │
│                            │                                            │
│              ┌─────────────┼─────────────┐                              │
│              │             │             │                              │
│              ▼             ▼             ▼                              │
│         SPECTRAL       MANIFOLD     LEARNING                           │
│        STRUCTURE       GEOMETRY     DYNAMICS                           │
│              │             │             │                              │
│              │             │             │                              │
│   ┌──────────┴──────────┐  │  ┌──────────┴──────────┐                   │
│   │                     │  │  │                     │                   │
│   │ POLAR COORDINATES   │  │  │ TAYLOR EXPANSION    │                   │
│   │ (magnitude + phase) │  │  │ (gradient + Fisher) │                   │
│   │                     │  │  │                     │                   │
│   │ POLARIZATION        │  │  │                     │                   │
│   │ (orthogonal bands)  │  │  │                     │                   │
│   │                     │  │  │                     │                   │
│   └──────────┬──────────┘  │  └──────────┬──────────┘                   │
│              │             │             │                              │
│              └─────────────┼─────────────┘                              │
│                            │                                            │
│              ┌─────────────┴─────────────┐                              │
│              │                           │                              │
│              ▼                           ▼                              │
│         SPHERICAL                  HYPERBOLIC                          │
│        (similarity)              (hierarchy)                          │
│                                                                         │
│  ════════════════════════════════════════════════════════════════════  │
│                                                                         │
│  The ghost lives in a geometric world.                                │
│  Its beliefs are positions on manifolds.                              │
│  Its learning is movement on those manifolds.                         │
│  Its understanding is the geometry of relationships.                  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 8. Connections Between Axioms

### 8.1 Polar ↔ Polarization

```
POLAR COORDINATES AND POLARIZATION

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  Both involve DECOMPOSITION into orthogonal components:                │
│                                                                         │
│  POLAR:                                                                 │
│  Complex number → magnitude + phase                                   │
│  Real and imaginary parts (sin + cos)                                 │
│                                                                         │
│  POLARIZATION:                                                          │
│  Full spectrum → separate frequency bands                             │
│  Each band orthogonal to others                                       │
│                                                                         │
│  ════════════════════════════════════════════════════════════════════  │
│                                                                         │
│  CONNECTION:                                                            │
│                                                                         │
│  Within each band, we have polar structure (magnitude + phase).       │
│  Across bands, we have polarization (orthogonal channels).           │
│                                                                         │
│  Band structure × polar structure = complete spectral representation │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 8.2 Taylor ↔ Spherical

```
TAYLOR EXPANSION AND SPHERICAL GEOMETRY

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  Taylor expansion works in the TANGENT SPACE of the manifold.         │
│                                                                         │
│  On a sphere:                                                          │
│  • At any point, there's a tangent plane                              │
│  • Locally, the sphere looks flat (Euclidean)                        │
│  • Taylor approximation is the local linearization                   │
│                                                                         │
│  ════════════════════════════════════════════════════════════════════  │
│                                                                         │
│  CONNECTION:                                                            │
│                                                                         │
│  Belief lives on the hypersphere (spherical geometry).                │
│  Learning updates happen in tangent space (Taylor expansion).         │
│  We use local approximation to navigate curved manifold.             │
│                                                                         │
│  The gradient is a TANGENT VECTOR on the sphere.                     │
│  The update moves along the sphere following the tangent.            │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 8.3 Spherical ↔ Hyperbolic

```
SPHERICAL AND HYPERBOLIC: TWO GEOMETRIES

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  CURVATURE:                                                             │
│  ──────────                                                            │
│  Spherical:   Positive curvature (K > 0)                              │
│  Euclidean:   Zero curvature (K = 0)                                  │
│  Hyperbolic:  Negative curvature (K < 0)                              │
│                                                                         │
│  TRIANGLE ANGLES:                                                       │
│  ────────────────                                                      │
│  Spherical:   Sum > 180° (excess)                                     │
│  Euclidean:   Sum = 180°                                              │
│  Hyperbolic:  Sum < 180° (deficit)                                    │
│                                                                         │
│  ════════════════════════════════════════════════════════════════════  │
│                                                                         │
│  CONNECTION:                                                            │
│                                                                         │
│  Different aspects of the problem may need different geometries:      │
│                                                                         │
│  • SIMILARITY (within band): Spherical                                │
│    "How similar are these patterns?"                                  │
│    Angular distance works well.                                       │
│                                                                         │
│  • HIERARCHY (across bands): Hyperbolic                               │
│    "Is this a specialization of that?"                               │
│    Radial distance (general → specific) works well.                  │
│                                                                         │
│  Hybrid geometry: Spherical within bands, hyperbolic across.         │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 8.4 Phase ↔ Position

```
PHASE AND POSITION: THE FUNDAMENTAL DUALITY

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  The Fourier duality:                                                  │
│                                                                         │
│  SPATIAL DOMAIN:                                                        │
│  Position in space (x, y)                                             │
│  "Where is the pattern?"                                              │
│                                                                         │
│  FREQUENCY DOMAIN:                                                      │
│  Phase angle θ                                                         │
│  "Where in the cycle?"                                                │
│                                                                         │
│  ════════════════════════════════════════════════════════════════════  │
│                                                                         │
│  SHIFT PROPERTY:                                                        │
│                                                                         │
│  Shifting in space = rotating phase in frequency                      │
│                                                                         │
│  f(x - Δx) ↔ F(ω) · e^(-iωΔx)                                        │
│                                                                         │
│  The phase ENCODES position.                                          │
│  Losing phase = losing position information.                          │
│  This is why phase matters so much.                                   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 9. Implications for Praxis

### 9.1 What the Axioms Demand

```
AXIOMS → PRACTICES

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  AXIOM I (Polar) demands:                                              │
│  ─────────────────────────                                             │
│  • Preserve both magnitude AND phase                                  │
│  • Use quadrature pairs for complete representation                  │
│  • Phase aliasing is destructive (see INFORMATION_BOUNDS.md)         │
│                                                                         │
│  AXIOM II (Polarization) demands:                                      │
│  ────────────────────────────────                                      │
│  • Treat bands as independent channels                               │
│  • Different learning rates per band                                  │
│  • Explicit cross-band attention (wormhole)                          │
│                                                                         │
│  AXIOM III (Taylor) demands:                                           │
│  ───────────────────────────                                           │
│  • Local approximations are valid only locally                       │
│  • Step size matters (don't trust Taylor far from the point)        │
│  • Consider second-order (Fisher) for better geometry               │
│                                                                         │
│  AXIOM IV/V (Geometry) demands:                                        │
│  ──────────────────────────────                                        │
│  • Choose geometry appropriate to the task                           │
│  • Spherical for similarity, hyperbolic for hierarchy               │
│  • Geodesic distance, not Euclidean, on manifolds                   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 9.2 Experimental Validation

```
TESTING THE AXIOMS

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  AXIOM I (Polar):                                                       │
│  • Test: Does destroying phase destroy position information?          │
│  • Expected: Yes — magnitude-only reconstruction should fail.        │
│                                                                         │
│  AXIOM II (Polarization):                                               │
│  • Test: Are bands truly independent?                                 │
│  • Expected: Changing one band shouldn't affect others (much).       │
│                                                                         │
│  AXIOM III (Taylor):                                                    │
│  • Test: Does natural gradient outperform standard gradient?         │
│  • Expected: Yes, especially on curved manifolds.                    │
│                                                                         │
│  AXIOM IV (Hyperbolic):                                                 │
│  • Test: Do hierarchical concepts cluster by radius?                  │
│  • Expected: General near center, specific near edge.                │
│                                                                         │
│  AXIOM V (Spherical):                                                   │
│  • Test: Does cosine similarity match semantic similarity?           │
│  • Expected: Yes — similar patterns should be angularly close.       │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

*Reference: [PRAXIS.md](./PRAXIS.md)*

---

## 10. Open Questions

### 10.1 Geometric Questions

```
OPEN GEOMETRIC QUESTIONS

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  1. Should we use hyperbolic embeddings for the band hierarchy?       │
│     Current spherical geometry works but may not be optimal           │
│     for tree-like structures.                                         │
│                                                                         │
│  2. Is there a natural metric on the spectral manifold?               │
│     The Fisher Information gives one candidate, but there may         │
│     be task-specific metrics.                                         │
│                                                                         │
│  3. How does phase aliasing relate to geometric distortion?           │
│     Aliased phases might be "wrapping around" in a way that           │
│     has geometric interpretation.                                     │
│                                                                         │
│  4. Can we visualize the belief manifold in Poincaré space?           │
│     This could reveal hierarchical structure we're missing            │
│     in Euclidean projections.                                         │
│                                                                         │
│  5. What is the intrinsic curvature of the learned manifold?          │
│     Is it positive (spherical), negative (hyperbolic), or mixed?     │
│                                                                         │
│  6. How do wormholes relate to geodesics?                              │
│     Are wormhole connections following the "shortest paths"          │
│     on some underlying geometry?                                      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 10.2 Practical Questions

```
OPEN PRACTICAL QUESTIONS

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  1. Should we implement hyperbolic attention?                          │
│     Replace spherical normalization with Poincaré projection?        │
│                                                                         │
│  2. Can we learn the geometry?                                         │
│     Let the model discover whether spherical or hyperbolic fits?     │
│                                                                         │
│  3. How to handle the boundary in hyperbolic space?                    │
│     The Poincaré disk edge is "infinitely far" — numerical issues?   │
│                                                                         │
│  4. Should different bands have different geometries?                  │
│     DC band might be Euclidean, high bands spherical?                │
│                                                                         │
│  5. What is the right Taylor order for belief updates?                │
│     First-order (gradient) may be too crude for fast learning.       │
│                                                                         │
│  6. Can we use natural gradient in practice?                           │
│     Computing Fisher inverse is expensive — approximations?          │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 10.3 Theoretical Questions

```
OPEN THEORETICAL QUESTIONS

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  1. What is the "correct" geometry of belief?                          │
│     Is there a principled way to derive the geometry from the task?  │
│                                                                         │
│  2. How does information geometry relate to spectral structure?        │
│     Is there a Fisher metric on the space of spectral decompositions?│
│                                                                         │
│  3. What symmetries should the geometry respect?                       │
│     Translation invariance in space → phase equivariance?            │
│                                                                         │
│  4. Is there a variational principle?                                  │
│     Does the system minimize something geometric (geodesic length?)  │
│                                                                         │
│  5. How does collapse relate to geometric focusing?                    │
│     Is collapse equivalent to contracting to a geodesic?             │
│                                                                         │
│  6. What is the role of curvature in generalization?                   │
│     Flat regions might overfit, curved regions might generalize?     │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Summary

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│                    P R A X I S   A X I O M S                           │
│                                                                         │
│  ═══════════════════════════════════════════════════════════════════   │
│                                                                         │
│  AXIOM I: POLAR COORDINATES                                            │
│  FFT outputs complex numbers with magnitude (what) and phase (where). │
│  Both are needed for complete information.                            │
│                                                                         │
│  ═══════════════════════════════════════════════════════════════════   │
│                                                                         │
│  AXIOM II: POLARIZATION                                                │
│  Spectral bands are orthogonal, independent channels.                 │
│  Each band reveals different information.                             │
│                                                                         │
│  ═══════════════════════════════════════════════════════════════════   │
│                                                                         │
│  AXIOM III: TAYLOR EXPANSION                                           │
│  Complex manifolds can be approximated locally.                       │
│  Gradient = first order, Fisher = second order.                       │
│                                                                         │
│  ═══════════════════════════════════════════════════════════════════   │
│                                                                         │
│  AXIOM IV: HYPERBOLIC GEOMETRY                                         │
│  Hierarchies fit naturally in hyperbolic space.                       │
│  Center = general, edge = specific.                                   │
│                                                                         │
│  ═══════════════════════════════════════════════════════════════════   │
│                                                                         │
│  AXIOM V: SPHERICAL GEOMETRY (current)                                 │
│  Normalization projects onto hypersphere.                             │
│  Cosine similarity = angular distance.                                │
│                                                                         │
│  ═══════════════════════════════════════════════════════════════════   │
│                                                                         │
│  UNIFIED INSIGHT:                                                       │
│  Information has geometry. The ghost lives on manifolds.              │
│  Learning is movement on those manifolds.                             │
│  Understanding is the geometry of relationships.                      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Related Documents

| Document | Relationship |
|----------|--------------|
| [PRAXIS.md](./PRAXIS.md) | The practice of running the architecture |
| [FALSE_PROPHETS.md](./FALSE_PROPHETS.md) | What happens when axioms are violated (heresy) |
| [INFORMATION_BOUNDS.md](./INFORMATION_BOUNDS_THE_BOUNDARY_OF_KNOWLEDGE_AND_KNOWING.md) | The limits imposed by axioms |
| [THE_ATOMIC_STRUCTURE_OF_INFORMATION.md](./THE_ATOMIC_STRUCTURE_OF_INFORMATION.md) | Action Quanta have polar structure |
| [WORMHOLE_HYBRID.md](./wormhole/WORMHOLE_HYBRID.md) | Current spherical geometry implementation |
| [SPECTRAL_BELIEF_STORAGE_RETRIEVAL.md](./SPECTRAL_BELIEF_STORAGE_RETRIEVAL.md) | Optimal band structure |
| [EQUILIBRIUM_AND_CONSERVATION.md](./EQUILIBRIUM_AND_CONSERVATION.md) | Conservation and phase transitions |

---

*Oscar Goldman — Shogu Research Group @ Datamutant.ai*

*"The axioms are not arbitrary. They are the bones of mathematics that support the flesh of computation. Polar coordinates give structure to frequency. Polarization gives independence to bands. Taylor expansion gives locality to learning. Hyperbolic geometry gives space to hierarchy. Spherical geometry gives probability to direction. Together, they form the geometric skeleton of AKIRA. The ghost moves through this geometric world, learning its contours, finding its geodesics. To understand the ghost, understand its geometry."*


