# FFT Interference and Superposition

Understanding how multiple patterns combine in the frequency domain, the emergence of interference patterns, and implications for spectral attention.

---

## Table of Contents

1. [The Superposition Principle](#1-the-superposition-principle)
2. [Interference: Constructive and Destructive](#2-interference-constructive-and-destructive)
3. [FFT of Combined Signals](#3-fft-of-combined-signals)
4. [Multiple Objects in Frequency Domain](#4-multiple-objects-in-frequency-domain)
5. [Wave Interference Patterns](#5-wave-interference-patterns)
6. [The Double-Slit Analogy](#6-the-double-slit-analogy)
7. [Beating and Modulation](#7-beating-and-modulation)
8. [Interference in Prediction Error](#8-interference-in-prediction-error)
9. [Implications for Spectral Attention](#9-implications-for-spectral-attention)
10. [Mathematical Framework](#10-mathematical-framework)

---

## 1. The Superposition Principle

### 1.1 The Fundamental Principle

The Fourier Transform is **linear**:

```
FFT(A + B) = FFT(A) + FFT(B)

This is the SUPERPOSITION PRINCIPLE:
The transform of a sum equals the sum of transforms.
```

### 1.2 What This Means Visually

```
IMAGE A:                  IMAGE B:                  A + B:
┌───────────────┐         ┌───────────────┐         ┌───────────────┐
│               │         │               │         │               │
│    ○          │    +    │          ○    │    =    │    ○     ○    │
│               │         │               │         │               │
└───────────────┘         └───────────────┘         └───────────────┘
 Blob on left              Blob on right             Two blobs

        │                         │                         │
        │ FFT                     │ FFT                     │ FFT
        ▼                         ▼                         ▼

SPECTRUM A:               SPECTRUM B:               A + B SPECTRUM:
┌───────────────┐         ┌───────────────┐         ┌───────────────┐
│               │         │               │         │               │
│      ●        │    +    │      ●        │    =    │   ● ═══ ●     │
│               │         │               │         │   (with fringes)
└───────────────┘         └───────────────┘         └───────────────┘
Same spectrum!            Same spectrum!             INTERFERENCE pattern
(shifted phase)           (shifted phase)
```

### 1.3 Key Insight: Spectra Add as Complex Numbers

```
FFT(A) + FFT(B) means COMPLEX ADDITION at each frequency:

At frequency (u, v):

F_A(u,v) = |A| · exp(i·φ_A)    (amplitude A, phase φ_A)
F_B(u,v) = |B| · exp(i·φ_B)    (amplitude B, phase φ_B)

Sum = F_A + F_B = complex addition

The result depends on the PHASE DIFFERENCE (φ_A - φ_B)!
```

---

## 2. Interference: Constructive and Destructive

### 2.1 Phase Alignment Determines Interference Type

```
CONSTRUCTIVE INTERFERENCE (phases aligned):
──────────────────────────────────────────

φ_A ≈ φ_B (phases similar)

Wave A:    ╭───╮   ╭───╮   ╭───╮
           │   │   │   │   │   │
       ────╯   ╰───╯   ╰───╯   ╰────

Wave B:    ╭───╮   ╭───╮   ╭───╮
           │   │   │   │   │   │
       ────╯   ╰───╯   ╰───╯   ╰────

Sum:       ╭═══╮   ╭═══╮   ╭═══╮
           ║   ║   ║   ║   ║   ║        AMPLITUDE DOUBLES!
       ════╝   ╚═══╝   ╚═══╝   ╚════


DESTRUCTIVE INTERFERENCE (phases opposite):
───────────────────────────────────────────

φ_A ≈ φ_B + π (phases opposite)

Wave A:    ╭───╮   ╭───╮   ╭───╮
           │   │   │   │   │   │
       ────╯   ╰───╯   ╰───╯   ╰────

Wave B:        ╭───╮   ╭───╮   ╭───╮
               │   │   │   │   │   │
       ────────╯   ╰───╯   ╰───╯   ╰

Sum:       ─────────────────────────        CANCELLATION!
                   (flat line)
```

### 2.2 The Interference Formula

```
For two waves with same frequency:

A₁·cos(ωt + φ₁) + A₂·cos(ωt + φ₂)

= A_total · cos(ωt + φ_total)

Where:
A_total² = A₁² + A₂² + 2·A₁·A₂·cos(φ₁ - φ₂)
                        ↑
              INTERFERENCE TERM
              
When φ₁ = φ₂:         A_total = A₁ + A₂    (constructive)
When φ₁ = φ₂ + π:     A_total = |A₁ - A₂|  (destructive)
When φ₁ = φ₂ + π/2:   A_total = √(A₁² + A₂²) (quadrature)
```

### 2.3 Interference Visualization in 2D

```
TWO BLOBS AT DIFFERENT POSITIONS:

Spatial Domain:                  Frequency Domain (Amplitude):
┌─────────────────────────┐      ┌─────────────────────────┐
│                         │      │                         │
│    ○           ○        │      │    ═══════════════      │
│                         │      │    ║ ║ ║ ║ ║ ║ ║       │
│                         │      │    ═══════════════      │
│                         │      │    ║ ║ ║ ║ ║ ║ ║       │
│                         │      │    ═══════════════      │
│                         │      │                         │
└─────────────────────────┘      └─────────────────────────┘

The FRINGES in the frequency domain are interference patterns!
Their spacing encodes the DISTANCE between the two blobs.
```

---

## 3. FFT of Combined Signals

### 3.1 Single Object vs Multiple Objects

```
SINGLE BLOB:                    TWO BLOBS:

Spatial:                        Spatial:
┌───────────────┐               ┌───────────────┐
│       ▓       │               │   ▓       ▓   │
│      ▓▓▓      │               │  ▓▓▓     ▓▓▓  │
│       ▓       │               │   ▓       ▓   │
└───────────────┘               └───────────────┘

Amplitude Spectrum:             Amplitude Spectrum:
┌───────────────┐               ┌───────────────┐
│               │               │   │ │ │ │ │   │
│       ●       │               │   ● ● ● ● ●   │  ← Fringes!
│               │               │   │ │ │ │ │   │
└───────────────┘               └───────────────┘
 
Smooth, single peak             Modulated by interference
```

### 3.2 How Distance Affects Fringes

```
FRINGE SPACING ∝ 1/OBJECT SEPARATION

Close objects (small d):        Far objects (large d):
┌───────────────┐               ┌───────────────┐
│    ▓   ▓      │               │  ▓         ▓  │
└───────────────┘               └───────────────┘
        │                               │
        ▼                               ▼
┌───────────────┐               ┌───────────────┐
│ │   │   │   │ │               │││││││││││││││││
│ ●   ●   ●   ● │               │●●●●●●●●●●●●●●●│
│ │   │   │   │ │               │││││││││││││││││
└───────────────┘               └───────────────┘
 Wide fringes                    Narrow fringes
 (low fringe freq)               (high fringe freq)


FORMULA:
Fringe spacing in frequency domain = Image size / Object separation
```

### 3.3 Fringe Orientation

```
FRINGE DIRECTION ⊥ OBJECT ALIGNMENT

Objects aligned horizontally:   Objects aligned vertically:
┌───────────────┐               ┌───────────────┐
│   ○       ○   │               │       ○       │
│               │               │               │
│               │               │       ○       │
└───────────────┘               └───────────────┘
        │                               │
        ▼                               ▼
┌───────────────┐               ┌───────────────┐
│   │ │ │ │ │   │               │   ─────────   │
│   │ │ │ │ │   │               │   ─────────   │
│   │ │ │ │ │   │               │   ─────────   │
└───────────────┘               └───────────────┘
 Vertical fringes                Horizontal fringes


Objects aligned diagonally:
┌───────────────┐               ┌───────────────┐
│  ○            │               │   ╲ ╲ ╲ ╲ ╲   │
│            ○  │       →       │    ╲ ╲ ╲ ╲    │
└───────────────┘               │     ╲ ╲ ╲     │
                                └───────────────┘
                                 Diagonal fringes
```

---

## 4. Multiple Objects in Frequency Domain

### 4.1 Decomposition of Multi-Object Scenes

```
SCENE WITH THREE OBJECTS:

Spatial Domain:
┌─────────────────────────────┐
│                             │
│     ○         □         △   │
│                             │
└─────────────────────────────┘

= Blob A + Square B + Triangle C

FFT:
┌─────────────────────────────┐
│                             │
│   FFT(A) + FFT(B) + FFT(C)  │
│                             │
└─────────────────────────────┘

Each object contributes:
• Its own amplitude pattern (shape signature)
• Its own phase (position encoding)
• Interference terms with every other object
```

### 4.2 Interference Between Different Shapes

```
BLOB + RING:

Spatial:
┌─────────────────────────────┐
│                             │
│     ▓▓▓       ░░░░░         │
│    ▓▓▓▓▓     ░░   ░░        │
│     ▓▓▓       ░░░░░         │
│                             │
└─────────────────────────────┘

Amplitude Spectrum:
┌─────────────────────────────┐
│                             │
│     Blob       Ring         │
│   spectrum + spectrum       │
│       ↓           ↓         │
│       ●─────────○○○         │
│       │         ↑           │
│       └─ Interference ─┘    │
│          fringes            │
└─────────────────────────────┘

The spectrum shows BOTH shape signatures
PLUS interference fringes between them.
```

### 4.3 Object Count from Fringe Complexity

```
NUMBER OF OBJECTS → FRINGE COMPLEXITY:

1 object:    No fringes (clean spectrum)
2 objects:   1 set of fringes
3 objects:   3 sets of fringes (1-2, 1-3, 2-3)
4 objects:   6 sets of fringes
N objects:   N(N-1)/2 sets of fringes

Complexity grows quadratically!

Example with 3 blobs:
┌─────────────────────────────┐
│     ○       ○       ○       │
│     A       B       C       │
└─────────────────────────────┘

Fringes from:
• A-B interference (horizontal, spacing d_AB)
• A-C interference (horizontal, spacing d_AC)
• B-C interference (horizontal, spacing d_BC)

Result: Complex overlapping fringe patterns
```

---

## 5. Wave Interference Patterns

### 5.1 Ripple Analogy

```
TWO POINT SOURCES (like stones in water):

Source A        Source B
    ●               ●
    │               │
    ▼               ▼

┌─────────────────────────────────────────────────┐
│  ╭─╮   ╭─╮   ╭─╮   ╭─╮   ╭─╮   ╭─╮   ╭─╮   ╭─╮ │
│ ╭╯ ╰╮ ╭╯ ╰╮ ╭╯ ╰╮ ╭╯ ╰╮ ╭╯ ╰╮ ╭╯ ╰╮ ╭╯ ╰╮ ╭╯ ╰╮│
│╭╯   ╰─╯   ╰─╯   ╰─╯   ╰─╯   ╰─╯   ╰─╯   ╰─╯   ╰│
│                                                 │
│  Interference pattern: alternating              │
│  constructive (bright) and destructive (dark)   │
│                                                 │
└─────────────────────────────────────────────────┘

Where peaks align: CONSTRUCTIVE (amplitude adds)
Where peak meets trough: DESTRUCTIVE (cancellation)
```

### 5.2 Standing Wave Patterns

```
When two identical waves travel in opposite directions:

→ Wave 1:    ╭───╮   ╭───╮   ╭───╮   →
             │   │   │   │   │   │
         ────╯   ╰───╯   ╰───╯   ╰────

← Wave 2:    ╭───╮   ╭───╮   ╭───╮   ←
             │   │   │   │   │   │
         ────╯   ╰───╯   ╰───╯   ╰────

= Standing Wave:
                    
         N   A   N   A   N   A   N
         │   │   │   │   │   │   │
         ▼   ▼   ▼   ▼   ▼   ▼   ▼
             ╱╲      ╱╲      ╱╲
            ╱  ╲    ╱  ╲    ╱  ╲
         ──╱────╲──╱────╲──╱────╲──
            ╲  ╱    ╲  ╱    ╲  ╱
             ╲╱      ╲╱      ╲╱

N = Node (always zero)
A = Antinode (maximum oscillation)

Standing waves appear in FFT when periodic patterns
interact with frame boundaries.
```

### 5.3 Moiré Patterns

```
MOIRÉ: Interference between similar patterns

Pattern A (vertical lines):    Pattern B (slightly rotated):
┌───────────────────┐          ┌───────────────────┐
│ │ │ │ │ │ │ │ │ │ │          │ / / / / / / / / / │
│ │ │ │ │ │ │ │ │ │ │          │ / / / / / / / / / │
│ │ │ │ │ │ │ │ │ │ │          │ / / / / / / / / / │
└───────────────────┘          └───────────────────┘

A + B = Moiré pattern:
┌───────────────────────────────────────┐
│     ║           ║           ║         │
│    ║ ║         ║ ║         ║ ║        │
│   ║   ║       ║   ║       ║   ║       │
│  ║     ║     ║     ║     ║     ║      │
│   ║   ║       ║   ║       ║   ║       │
│    ║ ║         ║ ║         ║ ║        │
│     ║           ║           ║         │
└───────────────────────────────────────┘

Large-scale "beat" pattern emerges from
interference of similar high-frequency patterns.

In FFT: Moiré appears as low-frequency components
        created by high-frequency interference.
```

---

## 6. The Double-Slit Analogy

### 6.1 Classic Double-Slit Setup

```
DOUBLE-SLIT EXPERIMENT:

Light source → [  ═  ] → Screen
                │   │
               slit slit

Screen pattern:
┌─────────────────────────────────────────┐
│                    │                    │
│  █  █  █  ███████████████████  █  █  █  │
│                    │                    │
│ dim              bright              dim│
└─────────────────────────────────────────┘

Central bright fringe + alternating dark/bright bands
```

### 6.2 Two Objects as "Double Slit"

```
TWO OBJECTS = TWO SOURCES OF FREQUENCY CONTENT

Object A          Object B
   ○                 ○
   │                 │
   ▼                 ▼
 
 FFT(A)           FFT(B)
   │                 │
   └────────┬────────┘
            │
            ▼
      Interference
        Pattern
            │
            ▼
┌─────────────────────────────────────────┐
│   │   │   │   ███████   │   │   │   │   │
│   │   │   │   ███████   │   │   │   │   │
│   │   │   │   ███████   │   │   │   │   │
└─────────────────────────────────────────┘
                Fringes

Just like light through two slits!
```

### 6.3 Fringe Formula (Double-Slit Analogy)

```
FRINGE SPACING:

In optics:    Δy = λL/d
              
              λ = wavelength
              L = distance to screen
              d = slit separation

In FFT:       Δf = N/d
              
              N = image size (pixels)
              d = object separation (pixels)
              Δf = fringe spacing in frequency domain


Example:
64×64 image, objects 16 pixels apart:
Δf = 64/16 = 4 frequency units between fringes
```

---

## 7. Beating and Modulation

### 7.1 Beating: Two Close Frequencies

```
BEATING occurs when two similar frequencies combine:

Frequency f₁:     ╭╮╭╮╭╮╭╮╭╮╭╮╭╮╭╮╭╮╭╮╭╮╭╮╭╮╭╮╭╮╭╮╭╮╭╮╭╮╭╮
                  ╰╯╰╯╰╯╰╯╰╯╰╯╰╯╰╯╰╯╰╯╰╯╰╯╰╯╰╯╰╯╰╯╰╯╰╯╰╯╰╯

Frequency f₂:     ╭╮╭╮╭╮╭╮╭╮╭╮╭╮╭╮╭╮╭╮╭╮╭╮╭╮╭╮╭╮╭╮╭╮╭╮╭╮╭╮╭╮
(slightly higher) ╰╯╰╯╰╯╰╯╰╯╰╯╰╯╰╯╰╯╰╯╰╯╰╯╰╯╰╯╰╯╰╯╰╯╰╯╰╯╰╯╰╯

Sum (beating):
                  ╭                                        ╭
                 ╱ ╲                                      ╱
                ╱   ╲                                    ╱
               ╱     ╲                                  ╱
         ─────╯       ╲────────────────────────────────╯

Beat frequency = |f₁ - f₂|

The ENVELOPE oscillates at the difference frequency!
```

### 7.2 Beating in Spatial Domain

```
TWO SIMILAR GRATINGS:

Grating 1 (period 10px):     Grating 2 (period 11px):
│ │ │ │ │ │ │ │ │ │          │  │  │  │  │  │  │  │

Sum = Beat pattern:
┌─────────────────────────────────────────────────────┐
│█   █   █   █   ██████████████████   █   █   █   █  │
│█   █   █   █   ██████████████████   █   █   █   █  │
└─────────────────────────────────────────────────────┘
 dark            bright                dark

Beat period = (10 × 11) / |11 - 10| = 110 pixels

A LARGE-SCALE pattern emerges from two fine patterns!
```

### 7.3 Amplitude Modulation in FFT

```
When object A modulates object B:

Carrier (object B):           Modulator (object A):
High frequency                Low frequency
┌───────────────┐             ┌───────────────┐
│││││││││││││││││             │     ████      │
│││││││││││││││││             │    ██████     │
│││││││││││││││││             │     ████      │
└───────────────┘             └───────────────┘

Product A × B:
┌───────────────┐
│    ││││││     │
│   ││││||||    │
│    ││││││     │
└───────────────┘
Carrier visible only where modulator is present

FFT of A × B:
┌───────────────┐
│               │
│  ●    ●   ●   │  Three peaks:
│               │  - Carrier frequency
└───────────────┘  - Carrier ± Modulator (sidebands)

MULTIPLICATION in space → CONVOLUTION in frequency
```

---

## 8. Interference in Prediction Error

### 8.1 Error as Interference Pattern

```
PREDICTION ERROR = Target - Prediction

If target and prediction are similar but shifted:

Target (actual):              Prediction (model):
┌───────────────┐             ┌───────────────┐
│               │             │               │
│      ○        │             │        ○      │
│               │             │               │
└───────────────┘             └───────────────┘

Error = Target - Prediction:
┌───────────────┐
│               │
│    + ○  - ○   │  ← Dipole pattern!
│               │
└───────────────┘

This is INTERFERENCE between two versions
of the same object at different positions.
```

### 8.2 Wave Packet Error Structure

```
When prediction slightly misses position:

Target:         ○
Prediction:       ○ (shifted slightly)

Error Pattern:
┌─────────────────────────────────────────┐
│                                         │
│           +       -                     │
│          +++     ---                    │
│         +++++   -----                   │
│          +++     ---                    │
│           +       -                     │
│                                         │
└─────────────────────────────────────────┘
        Positive   Negative
         lobe       lobe

This creates a WAVE PACKET shape:
- Positive where target exists but prediction doesn't
- Negative where prediction exists but target doesn't
- Zero where they overlap
```

### 8.3 FFT of Error Pattern

```
ERROR PATTERN FFT:

The dipole/wave-packet error has specific frequency content:

Spatial error:               FFT of error:
┌───────────────┐            ┌───────────────┐
│    + -        │            │               │
│   ++ --       │    FFT     │   ○ ○ ○ ○ ○   │
│   ++ --       │   ────►    │               │
│    + -        │            │               │
└───────────────┘            └───────────────┘

Dipole pattern              Fringes in frequency domain

The error FFT shows:
1. Same overall spectrum (similar objects)
2. Fringe modulation (position difference)
3. Fringe orientation (direction of error)
```

### 8.4 Error Direction from Interference Fringes

```
ERROR DIRECTION ENCODING:

Error to the RIGHT:          Error to the LEFT:
   + -                          - +
  ++ --                        -- ++
   + -                          - +

FFT fringes:                 FFT fringes:
│ │ │ │ │                    │ │ │ │ │
(same spacing,               (same spacing,
 phase shift 0)               phase shift π)

The PHASE of the fringes encodes error direction!
```

---

## 9. Implications for Spectral Attention

### 9.1 Multiple Objects in Scene

```
SPECTRAL ATTENTION WITH MULTIPLE OBJECTS:

Scene: Two blobs A and B
┌─────────────────────────────────────┐
│                                     │
│      ○ (A)              ○ (B)       │
│                                     │
└─────────────────────────────────────┘

Low-frequency amplitude:
┌─────────────────────────────────────┐
│                                     │
│     ▒▒▒              ▒▒▒            │
│    ▒▒▒▒▒            ▒▒▒▒▒           │
│     ▒▒▒              ▒▒▒            │
│                                     │
└─────────────────────────────────────┘
+ Interference fringes (encodes their relative position)

WHAT: Two blob-like objects
WHERE (relative): Encoded in fringe pattern
```

### 9.2 Attention Between Objects

```
WORMHOLE ATTENTION considers:

Query: Object A
Keys: History frames

If history has object similar to A at different position:
→ Low-freq amplitude matches (same WHAT)
→ Phase differs (different WHERE)
→ Creates interference when comparing

The interference pattern ENCODES the spatial relationship!
```

### 9.3 Disentangling Multiple Objects

```
CHALLENGE: Separate objects from interference

Scene FFT contains:
1. Object A spectrum (shape info)
2. Object B spectrum (shape info)
3. A-B interference (relationship info)

Strategy for spectral attention:
┌─────────────────────────────────────────────────────┐
│                                                     │
│  Focus on CENTRAL low frequencies:                  │
│  • Less affected by interference fringes            │
│  • Captures overall scene structure                 │
│  • More position-invariant                          │
│                                                     │
│  Use PHASE for position:                            │
│  • Separates position from interference             │
│  • Each object contributes phase gradient           │
│  • Weighted by amplitude contribution               │
│                                                     │
└─────────────────────────────────────────────────────┘
```

### 9.4 Interference as Information

```
INTERFERENCE IS NOT NOISE - IT'S SIGNAL!

The fringe pattern encodes:

Information              How It's Encoded
───────────              ────────────────
Object separation        Fringe spacing (closer = wider fringes)
Object alignment         Fringe orientation (perpendicular to alignment)
Relative phase           Fringe position (encodes relative position)
Number of objects        Fringe complexity (more objects = more fringes)

This can be EXPLOITED:
• Count objects from fringe complexity
• Measure distances from fringe spacing
• Determine arrangements from fringe orientations
```

---

## 10. Mathematical Framework

### 10.1 Superposition in Complex Form

```
For N objects at positions (x_n, y_n):

f(x, y) = Σ g_n(x - x_n, y - y_n)
          n=1

Where g_n is the shape of object n.

FFT:
F(u, v) = Σ G_n(u, v) · exp(-i·2π(u·x_n + v·y_n))
          n=1

Each object contributes:
• Shape spectrum G_n(u, v)
• Phase factor encoding position
```

### 10.2 Interference Term Derivation

```
For two identical objects at positions (x₁, y₁) and (x₂, y₂):

F(u, v) = G(u, v) · [exp(-i·2π(u·x₁ + v·y₁)) + exp(-i·2π(u·x₂ + v·y₂))]

        = G(u, v) · exp(-i·π(u(x₁+x₂) + v(y₁+y₂))) ·
                   [exp(-i·π(u·Δx + v·Δy)) + exp(+i·π(u·Δx + v·Δy))]

        = G(u, v) · exp(-i·φ_center) · 2·cos(π(u·Δx + v·Δy))
                                       ↑
                              INTERFERENCE TERM

Where:
Δx = x₁ - x₂, Δy = y₁ - y₂ (object separation)
φ_center = phase encoding center position

AMPLITUDE:
|F(u, v)| = 2·|G(u, v)| · |cos(π(u·Δx + v·Δy))|

The cosine creates the FRINGES!
Zeros when: u·Δx + v·Δy = n + 1/2 (n integer)
```

### 10.3 Fringe Properties

```
FRINGE PROPERTIES:

Spacing:
─────────
Fringe period in u-direction: 1/|Δx|
Fringe period in v-direction: 1/|Δy|

Orientation:
────────────
Fringes perpendicular to (Δx, Δy) vector
Angle = atan2(Δy, Δx) + 90°

Visibility:
───────────
Maximum when objects identical: V = 1
Decreases when objects differ: V = 2·√(I₁·I₂)/(I₁ + I₂)


Example calculation:
64×64 image, objects 10 pixels apart horizontally:

Δx = 10, Δy = 0
Fringe period = 64/10 = 6.4 frequency units
Fringe orientation = vertical (90° from horizontal separation)
```

### 10.4 N-Object Interference

```
For N identical objects:

|F(u, v)|² = |G(u, v)|² · |Σ exp(-i·2π(u·x_n + v·y_n))|²
                          n

            = |G(u, v)|² · [N + 2·Σ cos(2π(u·Δx_mn + v·Δy_mn))]
                               m<n

Number of interference terms = N(N-1)/2

Each pair contributes one set of fringes.

For 3 objects:
|F|² ∝ |G|² · [3 + 2cos(φ₁₂) + 2cos(φ₁₃) + 2cos(φ₂₃)]

where φ_mn = 2π(u·Δx_mn + v·Δy_mn)
```

### 10.5 Extracting Object Positions from Interference

```python
def extract_positions_from_fringes(spectrum, num_objects=2):
    """
    Extract object positions from interference pattern.
    
    Method: Autocorrelation of spectrum reveals fringe spacing.
    """
    # Compute autocorrelation of amplitude spectrum
    amplitude = np.abs(spectrum)
    autocorr = np.fft.ifft2(amplitude ** 2)
    autocorr = np.fft.fftshift(np.abs(autocorr))
    
    # Find peaks (correspond to object separations)
    # Central peak = DC (ignore)
    # Other peaks = separation vectors
    
    H, W = autocorr.shape
    center = (H // 2, W // 2)
    
    # Find secondary peaks
    # Each peak at (dy, dx) means objects separated by (dy, dx)
    
    peaks = find_peaks_2d(autocorr, exclude_center=True)
    
    # Positions are determined up to global translation
    # First object at origin, others relative to it
    positions = [(0, 0)]
    for peak in peaks[:num_objects - 1]:
        dy, dx = peak[0] - center[0], peak[1] - center[1]
        positions.append((dy, dx))
    
    return positions
```

---

## Summary

### Key Concepts

```
SUPERPOSITION:
─────────────
FFT(A + B) = FFT(A) + FFT(B)
Signals add linearly, spectra add as complex numbers.

INTERFERENCE:
─────────────
Phases aligned → Constructive (amplify)
Phases opposite → Destructive (cancel)
Creates fringes in frequency domain.

FRINGES ENCODE:
───────────────
• Object separation (fringe spacing)
• Object arrangement (fringe orientation)
• Object count (fringe complexity)

IMPLICATIONS:
─────────────
• Multiple objects create complex spectra
• Interference patterns carry spatial information
• Error patterns show interference structure
• Can extract relationships from fringes
```

### The Big Picture

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  SINGLE OBJECT:                                                 │
│  FFT = Shape spectrum × Position phase                          │
│  Clean, simple, easy to interpret                               │
│                                                                 │
│  MULTIPLE OBJECTS:                                              │
│  FFT = Σ(Shape × Phase) + Interference terms                    │
│  Complex, but information-rich                                  │
│                                                                 │
│  INTERFERENCE IS INFORMATION:                                   │
│  • Not noise to be removed                                      │
│  • Encodes spatial relationships                                │
│  • Can be decoded to understand scene structure                 │
│                                                                 │
│  FOR SPECTRAL ATTENTION:                                        │
│  • Low-freq focuses on individual structures                    │
│  • Interference patterns encode relationships                   │
│  • Phase provides position for each component                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

*This document explains how multiple signals combine in the frequency domain and how interference patterns emerge and encode spatial information. Understanding interference is crucial for interpreting FFT results in scenes with multiple objects.*

