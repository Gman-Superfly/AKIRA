# Manifold Magnitude-Phase Architecture: Storing WHAT and WHERE Separately

A detailed exploration of how to properly store magnitude (WHAT) and phase (WHERE) information in hierarchical manifolds, respecting their mathematical inseparability while leveraging their functional differences.

---

## Table of Contents

1. [The Fundamental Question](#1-the-fundamental-question)
2. [Magnitude and Phase: Mathematical Relationship](#2-magnitude-and-phase-mathematical-relationship)
3. [What Each Component Encodes](#3-what-each-component-encodes)
4. [The Storage Dilemma](#4-the-storage-dilemma)
5. [Analysis of Storage Options](#5-analysis-of-storage-options)
6. [The Memory Theory Perspective](#6-the-memory-theory-perspective)
7. [Phase Decomposition: Absolute vs Relative](#7-phase-decomposition-absolute-vs-relative)
8. [Recommended Architecture](#8-recommended-architecture)
9. [Learning Dynamics](#9-learning-dynamics)
10. [Implementation Details](#10-implementation-details)
11. [Reconstruction and Synthesis](#11-reconstruction-and-synthesis)
12. [Observability and Debugging](#12-observability-and-debugging)
13. [Summary and Guidelines](#13-summary-and-guidelines)

---

## 1. The Fundamental Question

### 1.1 The Problem Statement

```
We have established:
• Frequency-band manifolds for different scales (coarse → fine)
• Magnitude encodes WHAT (structure, identity)
• Phase encodes WHERE (position, alignment)

THE QUESTION:
Within each frequency-band manifold, how should we handle magnitude and phase?

OPTIONS:
A) Store them together (as complex numbers)
B) Separate manifolds for each
C) Magnitude in manifolds, phase in attention
D) Some hybrid approach

This document explores this question step by step.
```

### 1.2 Why This Matters

```
THE STAKES:

If we get this wrong:
• Magnitude patterns could be corrupted by phase learning
• Position information could be over-fitted (memorizing specific locations)
• Interference between WHAT and WHERE could cause collapse
• Reconstruction quality could suffer

If we get this right:
• Clean separation of concerns
• Each component learns at appropriate rate
• Natural protection of structural knowledge
• Flexible position handling
• High-quality reconstruction
```

---

## 2. Magnitude and Phase: Mathematical Relationship

### 2.1 The Complex Representation

```
FOURIER TRANSFORM OUTPUT:

For a 2D signal f(x, y), the Fourier transform gives:

F(u, v) = ∫∫ f(x, y) × e^(-2πi(ux + vy)) dx dy

This is a COMPLEX number at each frequency (u, v):

F(u, v) = Re(F) + i × Im(F)

        = |F(u,v)| × e^(i × φ(u,v))
          ────────   ──────────────
          magnitude     phase

WHERE:
• |F(u,v)| = √(Re² + Im²)     ← Magnitude (always positive)
• φ(u,v) = atan2(Im, Re)      ← Phase (angle, -π to π)
```

### 2.2 Mathematical Inseparability

```
YOU CANNOT HAVE ONE WITHOUT THE OTHER:

To compute magnitude:
|F| = √(Re² + Im²)

You need both Re and Im, which also give you phase:
φ = atan2(Im, Re)

TO RECONSTRUCT THE SIGNAL:
You need BOTH magnitude AND phase:

f(x,y) = ∫∫ |F(u,v)| × e^(i × φ(u,v)) × e^(2πi(ux + vy)) du dv

Without magnitude: No strength information
Without phase: No position information

THEY ARE INSEPARABLE FOR RECONSTRUCTION.
```

### 2.3 Visual Demonstration

```
CLASSIC DEMONSTRATION:

Take two images: A face and a house

Image A (face):  Magnitude_A, Phase_A
Image B (house): Magnitude_B, Phase_B

Now swap:
Magnitude_A + Phase_B → Looks like HOUSE (follows phase)
Magnitude_B + Phase_A → Looks like FACE (follows phase)

CONCLUSION:
Phase carries MORE perceptual information about spatial structure.
But magnitude carries information about WHAT is present.
```

---

## 3. What Each Component Encodes

### 3.1 Magnitude Spectrum

```
MAGNITUDE |F(u,v)| ENCODES:

1. PRESENCE OF PATTERNS
   • High magnitude at frequency f means "pattern with period 1/f exists"
   • Low magnitude means "this frequency is absent"

2. STRENGTH OF PATTERNS
   • Magnitude value indicates how dominant the pattern is
   • Higher = more prominent in the image

3. SCALE STRUCTURE
   • Distribution across frequencies shows coarse vs fine content
   • Natural images: 1/f falloff (more energy at low frequencies)

4. ROTATIONAL CONTENT
   • Distribution across angles shows orientation of patterns
   • Horizontal edges → energy along vertical frequency axis

CRITICAL PROPERTY:
Magnitude is (theoretically) SHIFT-INVARIANT.
Moving an object doesn't change |F|, only φ.

This makes magnitude ideal for WHAT (identity regardless of position).
```

### 3.2 Phase Spectrum

```
PHASE φ(u,v) ENCODES:

1. ABSOLUTE POSITION
   • Shifting image by (Δx, Δy) changes phase by -2π(u×Δx + v×Δy)
   • Phase gradient encodes position directly

2. ALIGNMENT OF COMPONENTS
   • How different frequency components line up
   • Creates edges, textures, features at specific locations

3. TEMPORAL TIMING (for video)
   • Phase shifts over time as objects move
   • Velocity encoded in phase change rate

4. STRUCTURAL COHERENCE
   • Related components have consistent phase relationships
   • Edges: components in phase at the edge location
   • This coherence is STRUCTURAL, not just positional

CRITICAL PROPERTY:
Phase changes linearly with position.
φ_new = φ_old - 2π(u×Δx + v×Δy)

This makes phase ideal for WHERE (precise location).
```

### 3.3 The Information Distribution

```
INFORMATION CONTENT ANALYSIS:

┌────────────────────────────────────────────────────────────────┐
│                                                                │
│  MAGNITUDE SPECTRUM                                            │
│  ─────────────────                                             │
│                                                                │
│  Contains:                                                     │
│  • ~5-10% of perceptual information                           │
│  • Pattern vocabulary (what shapes exist)                      │
│  • Energy distribution (relative importance of scales)         │
│  • Texture statistics                                          │
│                                                                │
│  Changes:                                                      │
│  • Slowly (objects don't change identity)                     │
│  • Only when content changes (new object appears)              │
│                                                                │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  PHASE SPECTRUM                                                │
│  ──────────────                                                │
│                                                                │
│  Contains:                                                     │
│  • ~90-95% of perceptual information                          │
│  • Precise spatial structure                                   │
│  • Edge locations                                              │
│  • Object positions                                            │
│                                                                │
│  Changes:                                                      │
│  • Rapidly (objects move)                                     │
│  • Every frame in video                                        │
│                                                                │
└────────────────────────────────────────────────────────────────┘

PARADOX:
Phase carries most information but is most transient.
Magnitude carries less but is more stable.

IMPLICATION FOR STORAGE:
Stable information → Store in slow-learning manifold
Transient information → Compute/update dynamically
```

---

## 4. The Storage Dilemma

### 4.1 The Core Tension

```
MANIFOLD LEARNING OBJECTIVE:

We want manifolds to learn GENERALIZABLE KNOWLEDGE.
Knowledge that applies across many instances.

MAGNITUDE: Generalizable
─────────────────────────
• "Ring patterns have this magnitude structure"
• Applies to rings at ANY position
• Should be learned and stored

PHASE: Partially Generalizable
──────────────────────────────
• Absolute position: NOT generalizable (specific to this instance)
• Phase coherence: IS generalizable (edges have coherent phase)

THE DILEMMA:
If we store phase directly, we overfit to specific positions.
If we ignore phase, we lose structural coherence patterns.

WE NEED TO SEPARATE:
• Absolute phase (transient, don't learn)
• Relative phase / coherence (structural, do learn)
```

### 4.2 The Learning Rate Problem

```
OPTIMAL LEARNING RATES:

For magnitude:
• Changes are rare (new patterns)
• Should be protected from noise
• Optimal lr: VERY LOW (0.0001)

For absolute phase:
• Changes every frame
• Must track motion in real-time
• Optimal lr: VERY HIGH (0.01) or COMPUTED (not learned)

For phase coherence:
• Structural property of patterns
• Moderately stable
• Optimal lr: MODERATE (0.001)

IN A SINGLE MANIFOLD:
One learning rate cannot serve all three.
Fast lr → magnitude corrupted
Slow lr → phase can't track motion
```

### 4.3 The 2×2 Problem

```
FULL DECOMPOSITION:

We have TWO axes:
• Frequency: Coarse ↔ Fine
• Representation: Magnitude ↔ Phase

This creates a 2×2 space:

                    Coarse          Fine
                    ──────          ────
Magnitude           Coarse-Mag      Fine-Mag
                    (structure)     (texture)
                    lr: 0.0001      lr: 0.0003

Phase               Coarse-Phase    Fine-Phase
                    (rough pos)     (precise pos)
                    lr: 0.001       lr: 0.01

PLUS cross-terms:
• Phase coherence within coarse
• Phase coherence within fine
• Phase coherence across scales (edges span scales)

THIS IS COMPLEX.
We need a principled way to handle it.
```

---

## 5. Analysis of Storage Options

### 5.1 Option A: Joint Storage (Complex Numbers)

```
ARCHITECTURE:

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  MANIFOLD (per frequency band)                                  │
│                                                                 │
│  Input: Complex tensor z = mag × e^(i×phase)                   │
│  Storage: Complex weights                                       │
│  Output: Complex tensor                                         │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Conv2d_complex(z) = W_real × z_real - W_imag × z_imag  │   │
│  │                    + i(W_real × z_imag + W_imag × z_real)│   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

PROS:
✓ Mathematically correct
✓ Preserves magnitude-phase relationship
✓ Natural reconstruction
✓ One manifold per band

CONS:
✗ Single learning rate for both mag and phase
✗ Cannot protect magnitude while allowing phase flexibility
✗ Complex-valued networks are less mature
✗ Harder to interpret

VERDICT: Mathematically elegant but practically problematic.
```

### 5.2 Option B: Fully Separate Manifolds

```
ARCHITECTURE:

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  MAGNITUDE MANIFOLD              PHASE MANIFOLD                 │
│  (per frequency band)            (per frequency band)           │
│                                                                 │
│  ┌─────────────────────┐         ┌─────────────────────┐       │
│  │                     │         │                     │       │
│  │  Input: |F(u,v)|    │         │  Input: φ(u,v)      │       │
│  │  lr: 0.0001         │         │  lr: 0.01           │       │
│  │  Heavy regularization│        │  Light regularization│      │
│  │                     │         │                     │       │
│  │  Learns: patterns   │         │  Learns: positions  │       │
│  │                     │         │                     │       │
│  └─────────────────────┘         └─────────────────────┘       │
│           │                                │                    │
│           └────────────┬───────────────────┘                   │
│                        ▼                                        │
│              ┌──────────────────┐                              │
│              │  RECOMBINATION   │                              │
│              │  z = mag × e^iφ  │                              │
│              └──────────────────┘                              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

PROS:
✓ Clean separation
✓ Different learning rates
✓ Easy to interpret

CONS:
✗ Doubles manifold count (2M manifolds for M bands)
✗ Phase manifold will overfit to positions
✗ Loses phase coherence (relative phase is structural)
✗ Recombination may be unstable

VERDICT: Too simplistic. Treating all phase as position loses structure.
```

### 5.3 Option C: Magnitude in Manifolds, Phase in Attention

```
ARCHITECTURE:

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  MANIFOLDS (WHAT pathway)                                       │
│  Store only magnitude patterns                                  │
│                                                                 │
│  ┌───────────┐  ┌───────────┐  ┌───────────┐                  │
│  │ Coarse    │  │ Mid       │  │ Fine      │                  │
│  │ |F| only  │  │ |F| only  │  │ |F| only  │                  │
│  │ lr: 0.0001│  │ lr: 0.0003│  │ lr: 0.001 │                  │
│  └───────────┘  └───────────┘  └───────────┘                  │
│        │              │              │                         │
│        └──────────────┼──────────────┘                         │
│                       ▼                                         │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                                                         │   │
│  │  ATTENTION LAYER (WHERE pathway)                        │   │
│  │                                                         │   │
│  │  Query: Magnitude features (WHAT am I looking for)     │   │
│  │  Key:   Magnitude features (WHAT is here)              │   │
│  │  Value: Full complex values (transfer WHERE info)      │   │
│  │                                                         │   │
│  │  Phase flows through attention, not stored in weights   │   │
│  │                                                         │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

PROS:
✓ Magnitude protected in slow-learning manifolds
✓ Phase handled dynamically (not overfitted)
✓ Natural division: manifold = WHAT, attention = WHERE
✓ Efficient (fewer manifolds)

CONS:
✗ Loses phase coherence learning
✗ Cannot learn structural phase patterns
✗ Relies entirely on attention for position

VERDICT: Good direction but loses structural phase information.
```

### 5.4 Option D: Hybrid Architecture (Recommended)

```
ARCHITECTURE:

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  KEY INSIGHT: Decompose phase into two components              │
│                                                                 │
│  φ(u,v) = φ_absolute(u,v) + φ_relative(u,v)                    │
│           ────────────────   ─────────────────                  │
│           Position info      Structural info                    │
│           (transient)        (learnable)                        │
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │                                                           │ │
│  │  MANIFOLD LAYER                                           │ │
│  │  ───────────────                                          │ │
│  │                                                           │ │
│  │  Stores:                                                  │ │
│  │  • Magnitude patterns (WHAT)                              │ │
│  │  • Phase coherence patterns (structural WHERE)            │ │
│  │                                                           │ │
│  │  Does NOT store:                                          │ │
│  │  • Absolute phase (position)                              │ │
│  │                                                           │ │
│  │  Learning rates:                                          │ │
│  │  • Magnitude: 0.0001 (very slow)                         │ │
│  │  • Coherence: 0.001 (moderate)                           │ │
│  │                                                           │ │
│  └───────────────────────────────────────────────────────────┘ │
│                              │                                  │
│                              ▼                                  │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │                                                           │ │
│  │  ATTENTION LAYER                                          │ │
│  │  ───────────────                                          │ │
│  │                                                           │ │
│  │  Uses:                                                    │ │
│  │  • Magnitude for matching (position-invariant)           │ │
│  │  • Transfers full content including phase                 │ │
│  │                                                           │ │
│  └───────────────────────────────────────────────────────────┘ │
│                              │                                  │
│                              ▼                                  │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │                                                           │ │
│  │  PHASE INSTANTIATION LAYER                                │ │
│  │  ─────────────────────────                                │ │
│  │                                                           │ │
│  │  • Very fast learning (or no learning)                   │ │
│  │  • Adjusts absolute phase for current position           │ │
│  │  • "Instantiate pattern at location X"                   │ │
│  │                                                           │ │
│  └───────────────────────────────────────────────────────────┘ │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

PROS:
✓ Separates learnable from transient phase
✓ Protects magnitude with slow learning
✓ Captures structural phase coherence
✓ Handles absolute position dynamically
✓ Matches cognitive theory (patterns vs instances)

CONS:
✗ More complex architecture
✗ Need to define phase coherence extraction
✗ More components to tune

VERDICT: Most principled approach. Worth the complexity.
```

---

## 6. The Memory Theory Perspective

### 6.1 Cognitive Parallels

```
HUMAN MEMORY SYSTEMS:

┌────────────────────────────────────────────────────────────────┐
│                                                                │
│  LONG-TERM MEMORY                                              │
│  ────────────────                                              │
│                                                                │
│  • Stores patterns, concepts, categories                       │
│  • Very stable, slow to update                                 │
│  • "What a ring looks like" (magnitude patterns)              │
│  • "How edges are formed" (phase coherence)                   │
│                                                                │
│  ───────────────────────────────────────────────────────────── │
│                                                                │
│  WORKING MEMORY                                                │
│  ──────────────                                                │
│                                                                │
│  • Stores current instances, positions                         │
│  • Very dynamic, constantly updated                            │
│  • "The ring is currently at (23, 45)" (absolute phase)       │
│  • "The ring is moving right" (phase change rate)             │
│                                                                │
└────────────────────────────────────────────────────────────────┘

MAPPING TO ARCHITECTURE:

Long-term memory → Manifold weights (slow learning)
Working memory → Attention activations (dynamic)

Magnitude patterns → Long-term memory
Phase coherence → Long-term memory
Absolute phase → Working memory
```

### 6.2 Episodic vs Semantic Memory

```
EPISODIC MEMORY:
• Specific instances: "I saw a ring at position (23, 45) at time t"
• Contains: Absolute phase (position), time stamp
• Should NOT be in manifold (overfitting to instances)

SEMANTIC MEMORY:
• General knowledge: "Rings have circular magnitude patterns"
• Contains: Magnitude patterns, coherence patterns
• SHOULD be in manifold (generalizes across instances)

ARCHITECTURE IMPLICATION:

Manifold = Semantic memory
• Stores generalizable patterns
• Magnitude: What shapes exist
• Coherence: How components align within shapes

Attention = Episodic retrieval
• Finds relevant past instances
• Uses semantic features (magnitude) for matching
• Retrieves full content (including transient phase)
```

### 6.3 The Binding Problem

```
THE BINDING PROBLEM:

How do we associate "ring pattern" with "position (23, 45)"?

If they're in separate systems:
• Magnitude manifold knows: "There's a ring"
• Position system knows: "Something is at (23, 45)"
• How do we know it's the RING at (23, 45)?

SOLUTION: Attention as binding mechanism

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  QUERY: "Ring pattern" (from magnitude manifold)               │
│  KEY: Locations with ring-like magnitude                       │
│  VALUE: Full content at those locations (including phase)     │
│                                                                 │
│  ATTENTION BINDS:                                               │
│  • WHAT (ring) to WHERE (position with matching magnitude)     │
│                                                                 │
│  Result: "Ring pattern instantiated at position (23, 45)"      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 7. Phase Decomposition: Absolute vs Relative

### 7.1 The Key Insight

```
PHASE CAN BE DECOMPOSED:

φ(u, v) = φ_position(u, v) + φ_structure(u, v)

WHERE:
• φ_position: Linear component determined by object position
  φ_position(u, v) = -2π(u×x₀ + v×y₀) for object at (x₀, y₀)

• φ_structure: Non-linear component encoding internal structure
  The coherent phase relationships WITHIN an object

EXTRACTING φ_structure:

1. Estimate position from phase gradient
2. Subtract position contribution
3. What remains is structural phase

φ_structure = φ - φ_position
```

### 7.2 Mathematical Details

```
PHASE GRADIENT AND POSITION:

The phase gradient tells us about position:

∂φ/∂u ≈ -2π × x₀  (horizontal position)
∂φ/∂v ≈ -2π × y₀  (vertical position)

But this is approximate because:
• Multiple objects → multiple contributions
• Noise → gradient noise
• Boundary effects → distortions

PHASE COHERENCE (what we want to learn):

For an edge at location (x₀, y₀):
• All frequency components are IN PHASE at that location
• This creates constructive interference = visible edge

For a smooth region:
• Components have random phase relationship
• Destructive interference = no sharp features

THE COHERENCE PATTERN IS STRUCTURAL.
It defines WHERE features appear RELATIVE to the object center.
```

### 7.3 Computing Phase Coherence

```
PHASE COHERENCE COMPUTATION:

Method 1: Phase Gradient Residual
──────────────────────────────────
1. Compute phase gradient: ∇φ = [∂φ/∂u, ∂φ/∂v]
2. Fit linear model: φ_linear = a×u + b×v + c
3. Residual: φ_coherence = φ - φ_linear
4. The residual captures structural phase

Method 2: Local Phase Differences
─────────────────────────────────
For each frequency (u,v):
  coherence(u,v) = Σ cos(φ(u,v) - φ(u',v'))
  
  where (u',v') are neighboring frequencies
  
  High coherence = consistent phase relationship

Method 3: Phase Congruency
──────────────────────────
Popular in edge detection:
  PC = Σ w_f × cos(φ_f - φ̄) / Σ w_f
  
  High phase congruency = edge location

WHAT TO STORE IN MANIFOLD:

NOT: Raw phase φ(u,v) (contains position)
YES: Phase coherence patterns (position-independent)
     • What coherence patterns indicate edges
     • What coherence patterns indicate textures
     • How coherence varies with scale
```

### 7.4 Practical Extraction

```python
def extract_phase_components(fft_complex):
    """
    Decompose phase into absolute (position) and relative (structure).
    
    Args:
        fft_complex: Complex tensor [B, 1, H, W]
    
    Returns:
        magnitude: |F| - the WHAT
        phase_absolute: Position-encoding phase
        phase_structure: Structure-encoding phase (coherence)
    """
    # Get magnitude and phase
    magnitude = torch.abs(fft_complex)
    phase = torch.angle(fft_complex)
    
    # Compute phase gradient (finite differences)
    phase_unwrapped = unwrap_phase_2d(phase)  # Handle wrapping
    
    grad_u = phase_unwrapped[:, :, :, 1:] - phase_unwrapped[:, :, :, :-1]
    grad_v = phase_unwrapped[:, :, 1:, :] - phase_unwrapped[:, :, :-1, :]
    
    # Estimate position from mean gradient
    # (weighted by magnitude - strong components matter more)
    weights = magnitude[:, :, :, 1:] + 1e-8
    x_est = (grad_u * weights).sum() / weights.sum() / (-2 * np.pi)
    
    weights = magnitude[:, :, 1:, :] + 1e-8
    y_est = (grad_v * weights).sum() / weights.sum() / (-2 * np.pi)
    
    # Construct absolute phase (linear ramp from position)
    H, W = phase.shape[-2:]
    u = torch.fft.fftfreq(W, device=phase.device)
    v = torch.fft.fftfreq(H, device=phase.device)
    V, U = torch.meshgrid(v, u, indexing='ij')
    
    phase_absolute = -2 * np.pi * (U * x_est + V * y_est)
    
    # Structure phase is the residual
    phase_structure = phase - phase_absolute
    
    # Wrap to [-π, π]
    phase_structure = torch.atan2(
        torch.sin(phase_structure), 
        torch.cos(phase_structure)
    )
    
    return magnitude, phase_absolute, phase_structure
```

---

## 8. Recommended Architecture

### 8.1 Complete Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│                    MAGNITUDE-PHASE MANIFOLD ARCHITECTURE                │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  INPUT: x [B, T, H, W]  (temporal context)                             │
│                                                                         │
│                              │                                          │
│                              ▼                                          │
│         ┌────────────────────────────────────────────┐                 │
│         │           SPECTRAL ANALYSIS                 │                 │
│         │                                             │                 │
│         │  For each frame and frequency band:         │                 │
│         │  • FFT                                      │                 │
│         │  • Extract magnitude |F|                    │                 │
│         │  • Extract phase φ                          │                 │
│         │  • Decompose: φ_abs + φ_struct              │                 │
│         │                                             │                 │
│         └─────────────────┬───────────────────────────┘                 │
│                           │                                             │
│           ┌───────────────┼───────────────┐                            │
│           │               │               │                            │
│           ▼               ▼               ▼                            │
│                                                                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                 │
│  │   COARSE     │  │    MID       │  │    FINE      │                 │
│  │   MANIFOLD   │  │   MANIFOLD   │  │   MANIFOLD   │                 │
│  └──────────────┘  └──────────────┘  └──────────────┘                 │
│         │                 │                 │                          │
│         │  Each manifold contains:          │                          │
│         │                                   │                          │
│         ▼                                   ▼                          │
│  ┌──────────────────────────────────────────────────────────────────┐ │
│  │                                                                  │ │
│  │  ┌────────────────────────┐    ┌─────────────────────────────┐  │ │
│  │  │  MAGNITUDE ENCODER     │    │  COHERENCE ENCODER          │  │ │
│  │  │  ────────────────────  │    │  ─────────────────────────  │  │ │
│  │  │                        │    │                             │  │ │
│  │  │  Input: |F|            │    │  Input: φ_structure         │  │ │
│  │  │  lr: 0.0001 (frozen)   │    │  lr: 0.001 (moderate)       │  │ │
│  │  │  λ_reg: 1.0 (heavy)    │    │  λ_reg: 0.3 (moderate)      │  │ │
│  │  │                        │    │                             │  │ │
│  │  │  Learns: patterns      │    │  Learns: edge coherence     │  │ │
│  │  │  Output: mag_features  │    │  Output: coh_features       │  │ │
│  │  │                        │    │                             │  │ │
│  │  └───────────┬────────────┘    └──────────────┬──────────────┘  │ │
│  │              │                                │                  │ │
│  │              └───────────┬────────────────────┘                  │ │
│  │                          │                                       │ │
│  │                          ▼                                       │ │
│  │              ┌────────────────────────┐                         │ │
│  │              │  PATTERN COMBINER      │                         │ │
│  │              │  ────────────────────  │                         │ │
│  │              │                        │                         │ │
│  │              │  pattern = concat(     │                         │ │
│  │              │    mag_features,       │                         │ │
│  │              │    coh_features        │                         │ │
│  │              │  )                     │                         │ │
│  │              │                        │                         │ │
│  │              └───────────┬────────────┘                         │ │
│  │                          │                                       │ │
│  └──────────────────────────┼───────────────────────────────────────┘ │
│                             │                                         │
│           ┌─────────────────┼─────────────────┐                      │
│           │                 │                 │                      │
│           ▼                 ▼                 ▼                      │
│  ┌──────────────────────────────────────────────────────────────────┐ │
│  │                                                                  │ │
│  │                  CROSS-MANIFOLD ATTENTION                        │ │
│  │                  ────────────────────────                        │ │
│  │                                                                  │ │
│  │  Query: pattern features (WHAT am I predicting)                 │ │
│  │  Key:   pattern features from history (WHAT was there)          │ │
│  │  Value: full content including φ_absolute (WHERE it was)        │ │
│  │                                                                  │ │
│  │  Uses magnitude for MATCHING (position-invariant)               │ │
│  │  Transfers full information (position-specific)                  │ │
│  │                                                                  │ │
│  └───────────────────────────┬──────────────────────────────────────┘ │
│                              │                                        │
│                              ▼                                        │
│  ┌──────────────────────────────────────────────────────────────────┐ │
│  │                                                                  │ │
│  │                  PHASE INSTANTIATION                             │ │
│  │                  ───────────────────                             │ │
│  │                                                                  │ │
│  │  Input: attended patterns + attended phase                       │ │
│  │  lr: 0.01 (fast) or computed (no learning)                      │ │
│  │                                                                  │ │
│  │  Adjusts phase for predicted position:                          │ │
│  │  φ_output = φ_attended + Δφ_predicted                           │ │
│  │                                                                  │ │
│  │  "Put this pattern at the predicted location"                   │ │
│  │                                                                  │ │
│  └───────────────────────────┬──────────────────────────────────────┘ │
│                              │                                        │
│                              ▼                                        │
│  ┌──────────────────────────────────────────────────────────────────┐ │
│  │                                                                  │ │
│  │                  SPECTRAL SYNTHESIS                              │ │
│  │                  ──────────────────                              │ │
│  │                                                                  │ │
│  │  For each frequency band:                                        │ │
│  │  F_output = |F|_predicted × exp(i × φ_output)                   │ │
│  │                                                                  │ │
│  │  Combine bands:                                                  │ │
│  │  x_output = IFFT(Σ F_band)                                      │ │
│  │                                                                  │ │
│  └───────────────────────────┬──────────────────────────────────────┘ │
│                              │                                        │
│                              ▼                                        │
│                                                                         │
│  OUTPUT: x_predicted [B, H, W]                                         │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 8.2 Per-Band Manifold Structure

```
INSIDE EACH FREQUENCY-BAND MANIFOLD:

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  FREQUENCY BAND i MANIFOLD                                      │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                                                         │   │
│  │  MAGNITUDE ENCODER (the WHAT pathway)                   │   │
│  │  ─────────────────────────────────────                  │   │
│  │                                                         │   │
│  │  Architecture:                                          │   │
│  │  |F| → Conv(k=5) → ReLU → Conv(k=3) → ReLU → features  │   │
│  │                                                         │   │
│  │  Properties:                                            │   │
│  │  • Input: Magnitude spectrum |F(u,v)|                  │   │
│  │  • Output: Feature vector representing pattern type    │   │
│  │  • Learning rate: 0.0001 × (1 + band_idx × 0.5)       │   │
│  │  • Weight decay: 0.01                                  │   │
│  │  • EMA decay: 0.9999 - 0.001 × band_idx               │   │
│  │                                                         │   │
│  │  What it learns:                                        │   │
│  │  • "Ring patterns have energy at these frequencies"    │   │
│  │  • "Edges have this magnitude structure"               │   │
│  │  • "Textures have this spectral signature"             │   │
│  │                                                         │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                                                         │   │
│  │  COHERENCE ENCODER (the structural WHERE pathway)       │   │
│  │  ───────────────────────────────────────────────────    │   │
│  │                                                         │   │
│  │  Architecture:                                          │   │
│  │  φ_struct → Conv(k=3) → sin/cos → Conv(k=3) → features │   │
│  │                                                         │   │
│  │  Note: Use sin/cos encoding because phase wraps        │   │
│  │                                                         │   │
│  │  Properties:                                            │   │
│  │  • Input: Structural phase φ_structure                 │   │
│  │  • Output: Feature vector representing coherence       │   │
│  │  • Learning rate: 0.001 × (1 + band_idx × 0.5)        │   │
│  │  • Weight decay: 0.003                                 │   │
│  │  • EMA decay: 0.999 - 0.005 × band_idx                │   │
│  │                                                         │   │
│  │  What it learns:                                        │   │
│  │  • "Edges have all frequencies in phase"               │   │
│  │  • "Textures have random phase relationships"          │   │
│  │  • "Object boundaries show phase discontinuities"      │   │
│  │                                                         │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                                                         │   │
│  │  TEMPORAL ATTENTION (within this band)                  │   │
│  │  ─────────────────────────────────────                  │   │
│  │                                                         │   │
│  │  Q: Current pattern features                            │   │
│  │  K: History pattern features (magnitude-based)         │   │
│  │  V: History full content (including absolute phase)    │   │
│  │                                                         │   │
│  │  Output: Attended patterns + attended position info    │   │
│  │                                                         │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                                                         │   │
│  │  DECODER                                                │   │
│  │  ───────                                                │   │
│  │                                                         │   │
│  │  Input: Attended features                               │   │
│  │  Output: Predicted |F| and φ for next frame            │   │
│  │                                                         │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 8.3 Component Summary

```
COMPONENT SUMMARY:

┌────────────────────────────────────────────────────────────────┐
│                                                                │
│  Component           Stores          Learning    Purpose       │
│  ─────────           ──────          ────────    ───────       │
│                                                                │
│  Magnitude Encoder   |F| patterns    Very slow   WHAT exists  │
│                                                                │
│  Coherence Encoder   φ_structure     Moderate    Structural   │
│                       patterns                    WHERE        │
│                                                                │
│  Attention Layer     Matching        N/A         Bind WHAT    │
│                       weights                     to WHERE     │
│                                                                │
│  Phase Instanti-     Position        Very fast   Instance     │
│  ation Layer         adjustment      /computed   position     │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

---

## 9. Learning Dynamics

### 9.1 Learning Rate Hierarchy

```
COMPLETE LEARNING RATE SCHEDULE:

┌────────────────────────────────────────────────────────────────┐
│                                                                │
│  Component                  Base LR    Band Scaling            │
│  ─────────                  ───────    ─────────────           │
│                                                                │
│  Magnitude Encoder:                                            │
│    Coarse band             0.0001     ×1.0                    │
│    Mid band                0.0001     ×1.5                    │
│    Fine band               0.0001     ×2.0                    │
│                                                                │
│  Coherence Encoder:                                            │
│    Coarse band             0.001      ×1.0                    │
│    Mid band                0.001      ×1.5                    │
│    Fine band               0.001      ×2.0                    │
│                                                                │
│  Phase Instantiation:                                          │
│    All bands               0.01       (or computed, lr=0)     │
│                                                                │
│  Cross-Manifold Attention:                                     │
│    All                     0.0005     ×1.0                    │
│                                                                │
│  Routing (if learned):                                         │
│    All                     0.0003     ×1.0                    │
│                                                                │
└────────────────────────────────────────────────────────────────┘

RATIONALE:
• Magnitude slowest (patterns are stable)
• Coherence moderate (structural but somewhat position-dependent)
• Phase instantiation fastest (must track motion)
• Fine bands slightly faster (more dynamic content)
```

### 9.2 Regularization Strategy

```
REGULARIZATION PER COMPONENT:

┌────────────────────────────────────────────────────────────────┐
│                                                                │
│  Component              Weight Decay   Spectral Reg   Dropout  │
│  ─────────              ────────────   ────────────   ───────  │
│                                                                │
│  Magnitude Encoder      0.01           High           0.0      │
│  Coherence Encoder      0.003          Medium         0.1      │
│  Phase Instantiation    0.0001         Low            0.0      │
│  Cross-Attention        0.001          Medium         0.1      │
│                                                                │
└────────────────────────────────────────────────────────────────┘

SPECTRAL REGULARIZATION:
Penalize if component encodes "wrong" information.

For Magnitude Encoder:
L_spectral_mag = ||gradient w.r.t. position||²
(Should be position-invariant)

For Coherence Encoder:
L_spectral_coh = ||correlation with absolute position||²
(Should encode structure, not absolute position)
```

### 9.3 Training Phases

```
PHASED TRAINING SCHEDULE:

PHASE 1: MAGNITUDE PRE-TRAINING (steps 0 - 5000)
──────────────────────────────────────────────────
• Train only magnitude encoders
• Freeze coherence, attention, phase layers
• Input: |F| from various patterns at various positions
• Loss: Magnitude reconstruction + pattern classification
• Goal: Learn what patterns exist (position-invariant)

PHASE 2: COHERENCE TRAINING (steps 5000 - 15000)
─────────────────────────────────────────────────
• Train coherence encoders
• Magnitude encoders: slow fine-tuning
• Freeze attention, phase layers
• Loss: Coherence prediction + edge detection auxiliary
• Goal: Learn structural phase patterns

PHASE 3: ATTENTION TRAINING (steps 15000 - 25000)
──────────────────────────────────────────────────
• Train attention layers
• Magnitude and coherence: slow fine-tuning
• Freeze phase instantiation
• Loss: Attention-based reconstruction
• Goal: Learn to match patterns across time/space

PHASE 4: FULL TRAINING (steps 25000+)
─────────────────────────────────────
• Train all components jointly
• Phase instantiation activated
• Full reconstruction loss
• Goal: Fine-tune complete system
```

### 9.4 Loss Function

```
COMPLETE LOSS FUNCTION:

L_total = L_recon + L_magnitude + L_coherence + L_phase + L_reg

WHERE:

L_recon = ||x_pred - x_target||²
         Main reconstruction loss

L_magnitude = Σ_band λ_mag × ||pred_mag_band - target_mag_band||²
              Per-band magnitude loss
              λ_mag = [1.0, 0.7, 0.5] for [coarse, mid, fine]

L_coherence = Σ_band λ_coh × ||pred_coh_band - target_coh_band||²
              Per-band coherence loss
              λ_coh = [0.5, 0.3, 0.2]

L_phase = ||angle_diff(pred_phase, target_phase)||²
          Phase loss (use angular difference to handle wrapping)

L_reg = L_spectral_reg + L_weight_decay + L_entropy
        Regularization terms
```

---

## 10. Implementation Details

### 10.1 Magnitude Encoder

```python
class MagnitudeEncoder(nn.Module):
    """Encodes magnitude patterns - the WHAT pathway."""
    
    def __init__(self, band_idx: int, dim: int):
        super().__init__()
        self.band_idx = band_idx
        
        # Slower learning for coarse, slightly faster for fine
        self.lr_scale = 0.1 * (1 + band_idx * 0.5)
        
        # Kernel size: larger for coarse (7), smaller for fine (3)
        kernel_size = 7 - 2 * band_idx
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, dim // 2, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm2d(dim // 2),
            nn.ReLU(),
            nn.Conv2d(dim // 2, dim, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm2d(dim),
            nn.ReLU(),
        )
        
        # Heavy weight decay for stability
        self.weight_decay = 0.01
        
        # Very stable EMA
        self.ema_decay = 0.9999 - 0.001 * band_idx
        self.register_buffer('ema_weights', None)
    
    def forward(self, magnitude: torch.Tensor) -> torch.Tensor:
        """
        Args:
            magnitude: [B, 1, H, W] magnitude spectrum
        
        Returns:
            features: [B, dim, H, W] magnitude pattern features
        """
        # Normalize magnitude (log scale for better dynamics)
        mag_norm = torch.log1p(magnitude)
        mag_norm = (mag_norm - mag_norm.mean()) / (mag_norm.std() + 1e-8)
        
        features = self.encoder(mag_norm)
        return features
    
    def update_ema(self):
        """Update exponential moving average of weights."""
        if self.ema_weights is None:
            self.ema_weights = {k: v.clone() for k, v in self.state_dict().items()}
        else:
            for k, v in self.state_dict().items():
                self.ema_weights[k] = (
                    self.ema_decay * self.ema_weights[k] + 
                    (1 - self.ema_decay) * v
                )
```

### 10.2 Coherence Encoder

```python
class CoherenceEncoder(nn.Module):
    """Encodes phase coherence patterns - structural WHERE."""
    
    def __init__(self, band_idx: int, dim: int):
        super().__init__()
        self.band_idx = band_idx
        
        # Moderate learning rate
        self.lr_scale = 1.0 * (1 + band_idx * 0.5)
        
        kernel_size = 5 - band_idx  # 5, 4, 3
        
        # Phase must be encoded with sin/cos (handles wrapping)
        self.phase_embedding = nn.Sequential(
            # Input: 2 channels (sin(φ), cos(φ))
            nn.Conv2d(2, dim // 2, kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
        )
        
        self.encoder = nn.Sequential(
            nn.Conv2d(dim // 2, dim, 3, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(),
        )
        
        self.weight_decay = 0.003
        self.ema_decay = 0.999 - 0.005 * band_idx
    
    def forward(self, phase_structure: torch.Tensor) -> torch.Tensor:
        """
        Args:
            phase_structure: [B, 1, H, W] structural phase component
        
        Returns:
            features: [B, dim, H, W] coherence pattern features
        """
        # Encode phase with sin/cos to handle wrapping
        sin_phase = torch.sin(phase_structure)
        cos_phase = torch.cos(phase_structure)
        phase_encoded = torch.cat([sin_phase, cos_phase], dim=1)
        
        embedded = self.phase_embedding(phase_encoded)
        features = self.encoder(embedded)
        
        return features
```

### 10.3 Phase Decomposer

```python
class PhaseDecomposer(nn.Module):
    """Decomposes phase into absolute (position) and structural (coherence)."""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, fft_complex: torch.Tensor) -> tuple:
        """
        Args:
            fft_complex: [B, 1, H, W] complex FFT output
        
        Returns:
            magnitude: [B, 1, H, W]
            phase_absolute: [B, 1, H, W] position-encoding phase
            phase_structure: [B, 1, H, W] structure-encoding phase
        """
        B, C, H, W = fft_complex.shape
        device = fft_complex.device
        
        # Extract magnitude and phase
        magnitude = torch.abs(fft_complex)
        phase = torch.angle(fft_complex)
        
        # Unwrap phase for gradient computation
        phase_unwrapped = self._unwrap_phase_2d(phase)
        
        # Compute phase gradients
        # Pad to handle boundaries
        phase_padded = F.pad(phase_unwrapped, (1, 1, 1, 1), mode='replicate')
        
        grad_u = (phase_padded[:, :, 1:-1, 2:] - phase_padded[:, :, 1:-1, :-2]) / 2
        grad_v = (phase_padded[:, :, 2:, 1:-1] - phase_padded[:, :, :-2, 1:-1]) / 2
        
        # Weighted mean gradient (weight by magnitude)
        weights = magnitude + 1e-8
        
        mean_grad_u = (grad_u * weights).sum(dim=(-2, -1), keepdim=True) / weights.sum(dim=(-2, -1), keepdim=True)
        mean_grad_v = (grad_v * weights).sum(dim=(-2, -1), keepdim=True) / weights.sum(dim=(-2, -1), keepdim=True)
        
        # Construct frequency grid
        u = torch.fft.fftfreq(W, device=device).view(1, 1, 1, W)
        v = torch.fft.fftfreq(H, device=device).view(1, 1, H, 1)
        
        # Estimated position from mean gradient
        # grad = -2π × position, so position = -grad / (2π)
        x_est = -mean_grad_u / (2 * torch.pi + 1e-8)
        y_est = -mean_grad_v / (2 * torch.pi + 1e-8)
        
        # Absolute phase (linear ramp from position)
        phase_absolute = -2 * torch.pi * (u * x_est + v * y_est)
        
        # Structural phase (residual)
        phase_structure = phase - phase_absolute
        
        # Wrap to [-π, π]
        phase_structure = torch.atan2(
            torch.sin(phase_structure),
            torch.cos(phase_structure)
        )
        
        return magnitude, phase_absolute, phase_structure
    
    def _unwrap_phase_2d(self, phase: torch.Tensor) -> torch.Tensor:
        """Simple 2D phase unwrapping (approximate)."""
        # This is a simplified version
        # For production, use a proper 2D unwrapping algorithm
        
        # Unwrap along each dimension
        phase_uw = phase.clone()
        
        # Horizontal unwrap
        diff = phase_uw[:, :, :, 1:] - phase_uw[:, :, :, :-1]
        diff = torch.where(diff > torch.pi, diff - 2*torch.pi, diff)
        diff = torch.where(diff < -torch.pi, diff + 2*torch.pi, diff)
        phase_uw[:, :, :, 1:] = phase_uw[:, :, :, :1] + diff.cumsum(dim=-1)
        
        # Vertical unwrap
        diff = phase_uw[:, :, 1:, :] - phase_uw[:, :, :-1, :]
        diff = torch.where(diff > torch.pi, diff - 2*torch.pi, diff)
        diff = torch.where(diff < -torch.pi, diff + 2*torch.pi, diff)
        phase_uw[:, :, 1:, :] = phase_uw[:, :, :1, :] + diff.cumsum(dim=-2)
        
        return phase_uw
```

### 10.4 Phase Instantiation Layer

```python
class PhaseInstantiation(nn.Module):
    """Adjusts phase for predicted position - instance WHERE."""
    
    def __init__(self, dim: int):
        super().__init__()
        
        # Very fast learning (or can be purely computed)
        self.lr_scale = 10.0  # Or set to 0 for no learning
        
        # Position predictor
        self.position_predictor = nn.Sequential(
            nn.Conv2d(dim, dim // 2, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(dim // 2, 2, 1),  # Predict (Δx, Δy)
        )
        
        self.weight_decay = 0.0001
    
    def forward(
        self,
        features: torch.Tensor,
        attended_phase: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            features: [B, dim, H, W] pattern features
            attended_phase: [B, 1, H, W] phase from attention
        
        Returns:
            adjusted_phase: [B, 1, H, W] phase for predicted position
        """
        B, _, H, W = features.shape
        device = features.device
        
        # Predict position adjustment
        delta_pos = self.position_predictor(features)  # [B, 2, H, W]
        delta_x = delta_pos[:, 0:1, :, :]
        delta_y = delta_pos[:, 1:2, :, :]
        
        # Create frequency grid
        u = torch.fft.fftfreq(W, device=device).view(1, 1, 1, W)
        v = torch.fft.fftfreq(H, device=device).view(1, 1, H, 1)
        
        # Compute phase adjustment from position change
        # Δφ = -2π(u × Δx + v × Δy)
        delta_phase = -2 * torch.pi * (u * delta_x + v * delta_y)
        
        # Apply adjustment
        adjusted_phase = attended_phase + delta_phase
        
        # Wrap to [-π, π]
        adjusted_phase = torch.atan2(
            torch.sin(adjusted_phase),
            torch.cos(adjusted_phase)
        )
        
        return adjusted_phase
```

### 10.5 Complete Band Manifold

```python
class MagnitudePhaseManifold(nn.Module):
    """Complete manifold for one frequency band."""
    
    def __init__(self, band_idx: int, dim: int):
        super().__init__()
        self.band_idx = band_idx
        self.dim = dim
        
        # Phase decomposition
        self.phase_decomposer = PhaseDecomposer()
        
        # Magnitude encoder (WHAT)
        self.magnitude_encoder = MagnitudeEncoder(band_idx, dim)
        
        # Coherence encoder (structural WHERE)
        self.coherence_encoder = CoherenceEncoder(band_idx, dim // 2)
        
        # Pattern combiner
        self.pattern_combiner = nn.Sequential(
            nn.Conv2d(dim + dim // 2, dim, 1),
            nn.ReLU(),
        )
        
        # Temporal attention
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=4,
            batch_first=True,
        )
        
        # Phase instantiation
        self.phase_instantiation = PhaseInstantiation(dim)
        
        # Decoders
        self.magnitude_decoder = nn.Sequential(
            nn.Conv2d(dim, dim // 2, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(dim // 2, 1, 3, padding=1),
            nn.Softplus(),  # Magnitude is positive
        )
        
        self.phase_decoder = nn.Sequential(
            nn.Conv2d(dim, dim // 2, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(dim // 2, 2, 3, padding=1),  # sin, cos
        )
    
    def forward(
        self,
        current_fft: torch.Tensor,
        history_fft: torch.Tensor = None,
    ) -> dict:
        """
        Args:
            current_fft: [B, 1, H, W] complex FFT of current frame (this band)
            history_fft: [B, T, 1, H, W] complex FFT of history (this band)
        
        Returns:
            dict with predictions and features
        """
        B, C, H, W = current_fft.shape
        
        # Decompose current frame
        mag, phase_abs, phase_struct = self.phase_decomposer(current_fft)
        
        # Encode magnitude (WHAT)
        mag_features = self.magnitude_encoder(mag)  # [B, dim, H, W]
        
        # Encode coherence (structural WHERE)
        coh_features = self.coherence_encoder(phase_struct)  # [B, dim//2, H, W]
        
        # Combine into pattern representation
        pattern = self.pattern_combiner(
            torch.cat([mag_features, coh_features], dim=1)
        )  # [B, dim, H, W]
        
        # Temporal attention if history provided
        attended_pattern = pattern
        attended_phase = phase_abs
        
        if history_fft is not None:
            T = history_fft.shape[1]
            
            # Process history
            history_patterns = []
            history_phases = []
            for t in range(T):
                h_mag, h_phase_abs, h_phase_struct = self.phase_decomposer(
                    history_fft[:, t]
                )
                h_mag_feat = self.magnitude_encoder(h_mag)
                h_coh_feat = self.coherence_encoder(h_phase_struct)
                h_pattern = self.pattern_combiner(
                    torch.cat([h_mag_feat, h_coh_feat], dim=1)
                )
                history_patterns.append(h_pattern)
                history_phases.append(h_phase_abs)
            
            history_patterns = torch.stack(history_patterns, dim=1)  # [B, T, dim, H, W]
            history_phases = torch.stack(history_phases, dim=1)  # [B, T, 1, H, W]
            
            # Reshape for attention
            pattern_flat = pattern.view(B, self.dim, -1).permute(0, 2, 1)  # [B, H*W, dim]
            history_flat = history_patterns.view(B, T, self.dim, -1).permute(0, 3, 1, 2)
            history_flat = history_flat.reshape(B, -1, T, self.dim)  # [B, H*W, T, dim]
            
            # Attention per spatial position
            attended_list = []
            phase_attended_list = []
            for i in range(pattern_flat.shape[1]):  # For each spatial position
                q = pattern_flat[:, i:i+1, :]  # [B, 1, dim]
                k = history_flat[:, i, :, :]   # [B, T, dim]
                v = history_flat[:, i, :, :]   # [B, T, dim]
                
                attn_out, attn_weights = self.temporal_attention(q, k, v)
                attended_list.append(attn_out)
                
                # Also attend over phases
                phase_vals = history_phases[:, :, :, i // W, i % W]  # [B, T, 1]
                phase_attended = (attn_weights.squeeze(1) @ phase_vals.squeeze(-1).unsqueeze(-1)).squeeze(-1)
                phase_attended_list.append(phase_attended)
            
            attended_pattern = torch.cat(attended_list, dim=1).permute(0, 2, 1).view(B, self.dim, H, W)
            attended_phase = torch.stack(phase_attended_list, dim=-1).view(B, 1, H, W)
        
        # Phase instantiation (adjust for predicted position)
        output_phase = self.phase_instantiation(attended_pattern, attended_phase)
        
        # Decode magnitude
        output_magnitude = self.magnitude_decoder(attended_pattern)
        
        # Decode phase (as sin/cos to handle wrapping)
        phase_sincos = self.phase_decoder(attended_pattern)
        pred_phase_sin = phase_sincos[:, 0:1]
        pred_phase_cos = phase_sincos[:, 1:2]
        pred_phase = torch.atan2(pred_phase_sin, pred_phase_cos)
        
        # Combine with instantiated phase
        final_phase = output_phase + pred_phase
        final_phase = torch.atan2(torch.sin(final_phase), torch.cos(final_phase))
        
        return {
            'magnitude': output_magnitude,
            'phase': final_phase,
            'pattern_features': pattern,
            'attended_features': attended_pattern,
            'magnitude_features': mag_features,
            'coherence_features': coh_features,
        }
    
    def get_param_groups(self, base_lr: float) -> list:
        """Get parameter groups with appropriate learning rates."""
        return [
            {
                'params': self.magnitude_encoder.parameters(),
                'lr': base_lr * self.magnitude_encoder.lr_scale,
                'weight_decay': self.magnitude_encoder.weight_decay,
                'name': f'band{self.band_idx}_magnitude',
            },
            {
                'params': self.coherence_encoder.parameters(),
                'lr': base_lr * self.coherence_encoder.lr_scale,
                'weight_decay': self.coherence_encoder.weight_decay,
                'name': f'band{self.band_idx}_coherence',
            },
            {
                'params': self.phase_instantiation.parameters(),
                'lr': base_lr * self.phase_instantiation.lr_scale,
                'weight_decay': self.phase_instantiation.weight_decay,
                'name': f'band{self.band_idx}_phase_inst',
            },
            {
                'params': list(self.pattern_combiner.parameters()) + 
                         list(self.temporal_attention.parameters()) +
                         list(self.magnitude_decoder.parameters()) +
                         list(self.phase_decoder.parameters()),
                'lr': base_lr,
                'name': f'band{self.band_idx}_other',
            },
        ]
```

---

## 11. Reconstruction and Synthesis

### 11.1 Per-Band Reconstruction

```python
def reconstruct_band(magnitude: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
    """
    Reconstruct complex FFT from magnitude and phase.
    
    Args:
        magnitude: [B, 1, H, W] predicted magnitude
        phase: [B, 1, H, W] predicted phase
    
    Returns:
        fft_complex: [B, 1, H, W] complex FFT
    """
    # Construct complex number: z = |z| × e^(iφ)
    real = magnitude * torch.cos(phase)
    imag = magnitude * torch.sin(phase)
    
    fft_complex = torch.complex(real, imag)
    
    return fft_complex
```

### 11.2 Multi-Band Synthesis

```python
def synthesize_output(
    band_ffts: list,
    original_size: tuple,
) -> torch.Tensor:
    """
    Combine frequency bands into final output.
    
    Args:
        band_ffts: List of [B, 1, H, W] complex FFTs per band
        original_size: (H, W) of output
    
    Returns:
        output: [B, 1, H, W] reconstructed signal
    """
    # Sum all bands in frequency domain
    combined_fft = sum(band_ffts)
    
    # Inverse FFT
    output = torch.fft.ifft2(combined_fft).real
    
    # Ensure correct size
    if output.shape[-2:] != original_size:
        output = F.interpolate(
            output, 
            size=original_size, 
            mode='bilinear', 
            align_corners=False
        )
    
    return output
```

### 11.3 Complete Forward Pass

```python
class HierarchicalMagnitudePhaseModel(nn.Module):
    """Complete model with magnitude-phase manifolds."""
    
    def __init__(
        self,
        n_bands: int = 3,
        base_dim: int = 32,
        grid_size: int = 64,
    ):
        super().__init__()
        self.n_bands = n_bands
        self.grid_size = grid_size
        
        # Frequency band masks
        self.register_buffer('band_masks', self._create_band_masks())
        
        # Per-band manifolds
        self.manifolds = nn.ModuleList([
            MagnitudePhaseManifold(
                band_idx=i,
                dim=base_dim * (2 ** i),  # 32, 64, 128
            )
            for i in range(n_bands)
        ])
        
        # Cross-band attention (optional)
        dims = [base_dim * (2 ** i) for i in range(n_bands)]
        self.cross_band_attention = CrossBandAttention(dims)
    
    def _create_band_masks(self):
        """Create frequency masks for each band."""
        H = W = self.grid_size
        cy, cx = H // 2, W // 2
        
        y = torch.arange(H).float() - cy
        x = torch.arange(W).float() - cx
        Y, X = torch.meshgrid(y, x, indexing='ij')
        
        freq_dist = torch.sqrt(X ** 2 + Y ** 2) / min(cx, cy)
        freq_dist = torch.fft.fftshift(freq_dist)
        
        # Band boundaries (logarithmically spaced)
        bounds = [0.0, 0.1, 0.3, 1.0]  # For 3 bands
        
        masks = []
        for i in range(self.n_bands):
            low, high = bounds[i], bounds[i + 1]
            mask = torch.sigmoid(20 * (freq_dist - low)) * \
                   torch.sigmoid(20 * (high - freq_dist))
            masks.append(mask)
        
        return torch.stack(masks)  # [n_bands, H, W]
    
    def decompose_to_bands(self, x: torch.Tensor) -> list:
        """Decompose signal into frequency bands."""
        x_fft = torch.fft.fft2(x)
        
        bands = []
        for i in range(self.n_bands):
            mask = self.band_masks[i].unsqueeze(0).unsqueeze(0)
            band_fft = x_fft * mask
            bands.append(band_fft)
        
        return bands
    
    def forward(
        self,
        x: torch.Tensor,
        history: torch.Tensor = None,
    ) -> dict:
        """
        Args:
            x: [B, 1, H, W] current frame
            history: [B, T, 1, H, W] past frames
        
        Returns:
            dict with predictions and info
        """
        # Decompose into frequency bands
        current_bands = self.decompose_to_bands(x)
        
        history_bands = None
        if history is not None:
            T = history.shape[1]
            history_bands = [
                [self.decompose_to_bands(history[:, t])[i] for t in range(T)]
                for i in range(self.n_bands)
            ]
            # Reorganize: [n_bands, T, B, 1, H, W] → [n_bands, B, T, 1, H, W]
            history_bands = [
                torch.stack(band_history, dim=1)
                for band_history in history_bands
            ]
        
        # Process each band through its manifold
        band_outputs = []
        all_features = []
        
        for i, manifold in enumerate(self.manifolds):
            hist_band = history_bands[i] if history_bands else None
            
            result = manifold(current_bands[i], hist_band)
            band_outputs.append(result)
            all_features.append(result['pattern_features'])
        
        # Cross-band attention (optional refinement)
        refined_features = self.cross_band_attention(all_features)
        
        # Reconstruct each band
        reconstructed_bands = []
        for i, (result, refined) in enumerate(zip(band_outputs, refined_features)):
            band_fft = reconstruct_band(result['magnitude'], result['phase'])
            reconstructed_bands.append(band_fft)
        
        # Synthesize final output
        output = synthesize_output(reconstructed_bands, (self.grid_size, self.grid_size))
        
        # Compile info for monitoring
        info = {
            'output': output,
            'band_magnitudes': [r['magnitude'] for r in band_outputs],
            'band_phases': [r['phase'] for r in band_outputs],
            'band_features': all_features,
        }
        
        return info
```

---

## 12. Observability and Debugging

### 12.1 Monitoring Dashboard

```
MAGNITUDE-PHASE MONITORING:

┌─────────────────────────────────────────────────────────────────┐
│                    MANIFOLD MONITOR                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  STEP: 5000  |  LOSS: 0.0234  |  TIME: 45ms/step               │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ COARSE BAND                                              │   │
│  │ ───────────                                              │   │
│  │                                                          │   │
│  │ Magnitude Encoder:                                       │   │
│  │   Loss: 0.0012 ▂▂▂▂▁▁▁  (stable ✓)                      │   │
│  │   Grad: 0.0001           (small, as expected)            │   │
│  │   EMA div: 0.0001        (frozen)                        │   │
│  │                                                          │   │
│  │ Coherence Encoder:                                       │   │
│  │   Loss: 0.0034 ▃▃▂▂▂▁▁  (learning)                      │   │
│  │   Grad: 0.0012           (moderate)                      │   │
│  │   EMA div: 0.0008        (stable)                        │   │
│  │                                                          │   │
│  │ Phase Instantiation:                                     │   │
│  │   Loss: 0.0089 ▆▅▅▄▄▃▃  (tracking motion)               │   │
│  │   Grad: 0.0045           (active)                        │   │
│  │                                                          │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ FINE BAND                                                │   │
│  │ ─────────                                                │   │
│  │                                                          │   │
│  │ Magnitude Encoder:                                       │   │
│  │   Loss: 0.0045 ▃▃▃▂▂▂▂  (learning patterns)             │   │
│  │   Grad: 0.0008           (moderate)                      │   │
│  │                                                          │   │
│  │ Coherence Encoder:                                       │   │
│  │   Loss: 0.0067 ▄▄▃▃▃▂▂  (learning edges)                │   │
│  │   Grad: 0.0023           (active)                        │   │
│  │                                                          │   │
│  │ Phase Instantiation:                                     │   │
│  │   Loss: 0.0123 ▇▆▆▅▅▄▄  (tracking precisely)            │   │
│  │   Grad: 0.0089           (very active)                   │   │
│  │                                                          │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  COMPONENT HEALTH:                                              │
│  ─────────────────                                              │
│  Magnitude/Coherence ratio: 3.2 (healthy: 2-5)                 │
│  Phase prediction accuracy: 89%                                 │
│  Cross-band correlation: 0.45 (bands cooperating)              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 12.2 Diagnostic Visualizations

```
WHAT TO VISUALIZE:

1. MAGNITUDE ENCODER OUTPUTS
   • Show what patterns each band's encoder activates on
   • Coarse should light up on structure (rings, blobs)
   • Fine should light up on edges and textures

2. COHERENCE ENCODER OUTPUTS
   • Show what phase patterns each encoder detects
   • High activation at edges (in-phase components)
   • Low activation in smooth regions

3. PHASE DECOMPOSITION
   • Visualize absolute vs structural phase
   • Absolute phase: should look like linear gradients
   • Structural phase: should highlight edges

4. ATTENTION PATTERNS
   • Where does current frame attend to in history?
   • Should show temporal correspondence

5. RECONSTRUCTION QUALITY PER BAND
   • Each band's contribution to final output
   • Identify which band has issues
```

### 12.3 Alert Conditions

```
AUTOMATIC ALERTS:

┌────────────────────────────────────────────────────────────────┐
│                                                                │
│  CONDITION                          ACTION                     │
│  ─────────                          ──────                     │
│                                                                │
│  Magnitude encoder gradient         Check learning rate,       │
│  too high (> 0.01)                  clip gradients            │
│                                                                │
│  Coherence encoder not learning     Increase lr or check       │
│  (loss plateau)                     phase decomposition       │
│                                                                │
│  Phase instantiation wildly off     Check position prediction │
│  (> 30% error)                      network                    │
│                                                                │
│  Magnitude encoder learning too     Check if magnitude is     │
│  fast relative to coherence         position-invariant        │
│                                                                │
│  Cross-band correlation too high    Bands may be redundant,   │
│  (> 0.8)                            check band boundaries     │
│                                                                │
│  Phase wrapping errors              Check sin/cos encoding    │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

---

## 13. Summary and Guidelines

### 13.1 Key Takeaways

```
┌────────────────────────────────────────────────────────────────┐
│                                                                │
│  MAGNITUDE-PHASE STORAGE: KEY INSIGHTS                         │
│                                                                │
│  1. MAGNITUDE and PHASE are mathematically inseparable        │
│     but should be TREATED DIFFERENTLY for storage.            │
│                                                                │
│  2. MAGNITUDE represents WHAT (patterns, identity)            │
│     → Store in slow-learning manifolds                        │
│     → Heavy regularization                                    │
│     → Very stable EMA                                         │
│                                                                │
│  3. PHASE must be decomposed into:                            │
│     • ABSOLUTE (position) → Transient, compute dynamically   │
│     • STRUCTURAL (coherence) → Learn in manifold (moderate)  │
│                                                                │
│  4. ATTENTION bridges WHAT and WHERE                          │
│     → Matches using magnitude (position-invariant)           │
│     → Transfers full content (position-specific)             │
│                                                                │
│  5. PHASE INSTANTIATION handles positioning                   │
│     → Very fast learning or purely computed                  │
│     → Instantiates patterns at specific locations            │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### 13.2 Quick Reference

```
COMPONENT CHEAT SHEET:

┌──────────────────┬───────────────┬──────────────┬──────────────┐
│ Component        │ Encodes       │ Learning     │ Stability    │
├──────────────────┼───────────────┼──────────────┼──────────────┤
│ Magnitude Enc.   │ WHAT          │ lr: 0.0001   │ Very high    │
│                  │ (patterns)    │ decay: 0.01  │ EMA: 0.9999  │
├──────────────────┼───────────────┼──────────────┼──────────────┤
│ Coherence Enc.   │ Structural    │ lr: 0.001    │ High         │
│                  │ WHERE         │ decay: 0.003 │ EMA: 0.999   │
├──────────────────┼───────────────┼──────────────┼──────────────┤
│ Phase Instant.   │ Instance      │ lr: 0.01     │ Low          │
│                  │ WHERE         │ decay: 0.0001│ (transient)  │
├──────────────────┼───────────────┼──────────────┼──────────────┤
│ Attention        │ Binding       │ lr: 0.0005   │ Medium       │
│                  │ WHAT↔WHERE    │              │              │
└──────────────────┴───────────────┴──────────────┴──────────────┘
```

### 13.3 Implementation Checklist

```
BEFORE IMPLEMENTING:

□ Define frequency band boundaries
□ Choose number of bands (recommend 3)
□ Set base dimension (recommend 32)
□ Define learning rate scales per component
□ Implement phase decomposition

DURING IMPLEMENTATION:

□ Use sin/cos encoding for phase (handles wrapping)
□ Use log-scale for magnitude (better dynamics)
□ Implement EMA for magnitude encoder stability
□ Add spectral regularization
□ Add monitoring for each component

AFTER IMPLEMENTING:

□ Verify magnitude encoder is position-invariant
□ Verify coherence encoder captures edge patterns
□ Verify phase instantiation tracks motion
□ Check learning rate hierarchy is correct
□ Monitor for component imbalances
```

### 13.4 Final Architecture Summary

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│              MAGNITUDE-PHASE MANIFOLD ARCHITECTURE              │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                                                         │   │
│  │  INPUT → FFT → DECOMPOSE → [|F|, φ_struct, φ_abs]      │   │
│  │                                                         │   │
│  │  |F| ─────────→ MAGNITUDE ENCODER ───┐                 │   │
│  │                 (very slow lr)        │                 │   │
│  │                                       ├──→ PATTERN     │   │
│  │  φ_struct ───→ COHERENCE ENCODER ────┘                 │   │
│  │                 (moderate lr)                           │   │
│  │                                                         │   │
│  │  PATTERN + HISTORY ───→ TEMPORAL ATTENTION             │   │
│  │                          (magnitude-based matching)     │   │
│  │                                                         │   │
│  │  ATTENDED + φ_abs ───→ PHASE INSTANTIATION             │   │
│  │                          (fast lr / computed)           │   │
│  │                                                         │   │
│  │  DECODED |F| + φ ───→ RECONSTRUCT ───→ OUTPUT          │   │
│  │                                                         │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  BENEFITS:                                                      │
│  ✓ Magnitude patterns protected (slow learning)                │
│  ✓ Structural phase learned (moderate learning)                │
│  ✓ Absolute phase tracked dynamically (fast/computed)          │
│  ✓ Natural WHAT/WHERE separation                               │
│  ✓ Memory-based architecture (patterns vs instances)           │
│  ✓ High observability per component                            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

*This document provides a complete treatment of magnitude-phase storage in spectral manifolds. The key insight is that while magnitude and phase are mathematically inseparable, they encode fundamentally different information (WHAT vs WHERE) and should be stored with different learning dynamics. Magnitude represents stable patterns (store in slow-learning manifolds), structural phase represents coherence patterns (store with moderate learning), and absolute phase represents transient position (compute dynamically or learn very fast).*

