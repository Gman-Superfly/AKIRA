# FFT Components: Amplitude, Frequency, and Phase

A comprehensive guide to understanding how the three components of the Fourier Transform encode different types of information, and their roles in the WHAT/WHERE separation.

---

## Table of Contents

1. [The Three Components of FFT](#1-the-three-components-of-fft)
2. [Amplitude: The WHAT Signal](#2-amplitude-the-what-signal)
3. [Phase: The WHERE Signal](#3-phase-the-where-signal)
4. [Frequency: The Resolution Control](#4-frequency-the-resolution-control)
5. [Frequency Splitting: Low vs High](#5-frequency-splitting-low-vs-high)
6. [How Frequency Affects Experiments](#6-how-frequency-affects-experiments)
7. [The Complete WHAT/WHERE Framework](#7-the-complete-whatwhere-framework)
8. [Practical Guidelines](#8-practical-guidelines)

---

## 1. The Three Components of FFT

The 2D Fourier Transform decomposes any image into three fundamental components:

```
FFT(image) → Complex number at each frequency (u, v)

F(u, v) = A(u, v) · exp(i · φ(u, v))

Where:
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  A(u, v) = AMPLITUDE (magnitude)                                │
│            How much of frequency (u, v) is present              │
│            |F(u, v)| = √(Re² + Im²)                             │
│                                                                 │
│  φ(u, v) = PHASE (angle)                                        │
│            Where that frequency component is positioned         │
│            arg(F(u, v)) = atan2(Im, Re)                         │
│                                                                 │
│  (u, v)  = FREQUENCY (coordinates in frequency domain)          │
│            How fast the pattern oscillates                      │
│            u = horizontal frequency, v = vertical frequency     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Visual Representation

```
SPATIAL DOMAIN                    FREQUENCY DOMAIN
(what we see)                     (FFT output)

┌─────────────────┐               ┌─────────────────┐
│                 │               │        v        │
│    ░░░░░░░      │               │        ↑        │
│   ░░    ░░      │    FFT        │        │        │
│  ░░      ░░     │  ═══════►     │   ─────●───── u │
│   ░░    ░░      │               │        │        │
│    ░░░░░░░      │               │        │        │
│                 │               │                 │
└─────────────────┘               └─────────────────┘
     Ring image                   Amplitude + Phase
                                  at each (u, v)
```

---

## 2. Amplitude: The WHAT Signal

### 2.1 What Amplitude Represents

Amplitude tells us **how much** of each frequency is present in the image:

```
HIGH AMPLITUDE at frequency (u, v):
─────────────────────────────────────
The image contains strong oscillations at that frequency.

LOW AMPLITUDE at frequency (u, v):
─────────────────────────────────────
The image has weak or no oscillations at that frequency.
```

### 2.2 Amplitude Encodes Structure (WHAT)

Different structures have different amplitude spectra:

```
BLOB (Gaussian):                 RING:                      CROSS:

Spatial:                         Spatial:                   Spatial:
    ▓▓▓▓                            ░░░░░                      │
   ▓████▓                         ░░   ░░                   ───┼───
  ▓██████▓                       ░░     ░░                     │
   ▓████▓                         ░░   ░░
    ▓▓▓▓                            ░░░░░

Amplitude:                       Amplitude:                 Amplitude:
    ●                               ○○○                        │
   ●●●        Single              ○   ○      Concentric       ─●─
    ●         central              ○○○        rings            │
              peak                                            Cross
                                                              pattern

Each structure has a UNIQUE amplitude signature = WHAT
```

### 2.3 Why Amplitude is Position-Invariant (Theoretically)

The Shift Theorem:

```
If f(x, y) has FFT F(u, v), then:

f(x - Δx, y - Δy) has FFT F(u, v) · exp(-i·2π(u·Δx + v·Δy))

Taking magnitude:
|F(u, v) · exp(-i·2π(u·Δx + v·Δy))| = |F(u, v)| · |exp(...)| = |F(u, v)| · 1

THE AMPLITUDE IS UNCHANGED BY TRANSLATION!
```

Visualized:

```
Pattern at Position A:           Pattern at Position B:
┌─────────────────┐              ┌─────────────────┐
│  ○              │              │              ○  │
│                 │              │                 │
└─────────────────┘              └─────────────────┘

       │                                │
       │ FFT                            │ FFT
       ▼                                ▼

Amplitude A:                     Amplitude B:
┌─────────────────┐              ┌─────────────────┐
│                 │              │                 │
│       ●●●       │      =       │       ●●●       │
│       ●●●       │              │       ●●●       │
│                 │              │                 │
└─────────────────┘              └─────────────────┘

        SAME AMPLITUDE = SAME STRUCTURE = SAME "WHAT"
```

### 2.4 Amplitude Spectrum Examples

```
WHAT DOES THE AMPLITUDE TELL US?

Image                           Amplitude Spectrum          Interpretation
─────                           ──────────────────          ──────────────

Single blob                          ●                      Single frequency
  ▓                                 ●●●                     component
 ▓▓▓                                 ●                      (smooth object)
  ▓


Two blobs                           ●●●●                    Multiple frequencies
 ▓   ▓                             ●●●●●●                   (repetition pattern)
▓▓▓ ▓▓▓                            ●●●●●●
 ▓   ▓                              ●●●●


Fine texture                       ●●●●●●●●●                Strong high frequencies
░░░░░░░░                          ●●●●●●●●●●●               (rapid variation)
░░░░░░░░                          ●●●●●●●●●●●
░░░░░░░░                           ●●●●●●●●●


Sharp edges                          │                      Oriented frequencies
────────                          ───●───                   (directional)
                                     │


Noise                             ●●●●●●●●●●●               Flat spectrum
▓░▓░▓░▓░                         ●●●●●●●●●●●●●             (all frequencies)
░▓░▓░▓░▓                         ●●●●●●●●●●●●●
▓░▓░▓░▓░                          ●●●●●●●●●●●
```

### 2.5 The Central Peak (DC Component)

```
THE DC COMPONENT: F(0, 0)

Location: Center of amplitude spectrum
Meaning:  Average brightness of entire image
Value:    Sum of all pixel values

┌─────────────────────────────────────────┐
│                                         │
│           Low frequencies               │
│              (smooth)                   │
│                                         │
│              ┌─────┐                    │
│              │  ●  │ ← DC (average)     │
│              └─────┘                    │
│                                         │
│          High frequencies               │
│            (detail)                     │
│                                         │
└─────────────────────────────────────────┘

Bright DC = High average brightness
Dark DC = Low average brightness
```

---

## 3. Phase: The WHERE Signal

### 3.1 What Phase Represents

Phase tells us **where** each frequency component is positioned:

```
PHASE = 0°:
──────────
Cosine wave starts at maximum (peak at origin)

PHASE = 90°:
───────────
Cosine wave starts at zero (shifted by quarter wavelength)

PHASE = 180°:
────────────
Cosine wave starts at minimum (inverted)
```

### 3.2 Phase Encodes Position (WHERE)

```
SHIFT IN SPACE → LINEAR CHANGE IN PHASE

Pattern shift: (Δx, Δy)

Phase change at frequency (u, v):
    Δφ(u, v) = -2π(u·Δx + v·Δy)

This is a LINEAR RAMP in the frequency domain!
```

Visualized:

```
Pattern at CENTER:               Pattern SHIFTED RIGHT:
┌─────────────────┐              ┌─────────────────┐
│        ○        │              │              ○  │
│                 │              │                 │
└─────────────────┘              └─────────────────┘

Phase (centered):                Phase (shifted):
┌─────────────────┐              ┌─────────────────┐
│                 │              │ R → O → Y → G   │
│   All same      │              │ ↓               │
│   (uniform)     │              │ Phase ramp      │
│                 │              │ pointing right  │
└─────────────────┘              └─────────────────┘

Phase gradient direction = Pattern position direction
```

### 3.3 Reading Position from Phase

```
DECODING POSITION FROM PHASE:

Phase Gradient Direction    Pattern Position
────────────────────────    ────────────────
        ↑                   Bottom of frame
        ↓                   Top of frame
        ←                   Right of frame
        →                   Left of frame
        ↗                   Bottom-left
        ↙                   Top-right
        
The gradient points AWAY from the pattern!
(Or equivalently, toward the center from the pattern)
```

### 3.4 Phase Contains Critical Information

A famous demonstration:

```
SWAP EXPERIMENT:

Image A (face):        Image B (house):
┌───────────┐          ┌───────────┐
│   ◠ ◠     │          │    ___    │
│    ─      │          │   |   |   │
│   ───     │          │   |___|   │
└───────────┘          └───────────┘

Amplitude A + Phase B:  Amplitude B + Phase A:
┌───────────┐          ┌───────────┐
│    ___    │          │   ◠ ◠     │
│   |   |   │          │    ─      │
│   |___|   │          │   ───     │
└───────────┘          └───────────┘
  HOUSE!                  FACE!
  
PHASE DOMINATES PERCEPTION!
The structure we see comes primarily from phase.
```

### 3.5 Phase vs Amplitude: Information Content

```
INFORMATION DISTRIBUTION:

              Amplitude              Phase
              ─────────              ─────
Contains:     Power spectrum         Spatial arrangement
              (how much)             (where)

Determines:   Contrast               Structure
              Frequency content      Edges
              Overall texture        Alignment

Perception:   Less critical          More critical
              for recognition        for recognition

Position:     Invariant              Encodes position
              (WHAT)                 (WHERE)
```

---

## 4. Frequency: The Resolution Control

### 4.1 What Frequency Represents

Frequency (u, v) tells us **how fast** the pattern oscillates:

```
FREQUENCY INTERPRETATION:

(u, v) = (0, 0):   DC component (no oscillation, just average)
(u, v) = (1, 0):   One complete cycle across image width
(u, v) = (0, 1):   One complete cycle across image height
(u, v) = (5, 3):   5 cycles horizontal, 3 cycles vertical

Higher ||(u,v)|| = Faster oscillation = Finer detail
```

### 4.2 Frequency Domain Layout

```
2D FREQUENCY DOMAIN:

              ─v (vertical frequency)
              │
              │    High freq
              │    (fine vertical detail)
              │
    High freq │         High freq
    (fine     │         (fine diagonal)
    diagonal) │
              │
──────────────●────────────────► u (horizontal frequency)
              │(0,0)
              │DC
    Low freq  │         Low freq
    (coarse)  │         (coarse)
              │
              │
              │    High freq
              │    (fine vertical)
              │

Distance from center = Total frequency = Detail level
```

### 4.3 Frequency Rings

```
FREQUENCY MAGNITUDE: ||(u, v)|| = √(u² + v²)

Equal frequency forms RINGS around the origin:

┌─────────────────────────────────────┐
│                                     │
│        ┌─────────────────┐          │
│        │   ┌─────────┐   │          │
│        │   │  ┌───┐  │   │          │
│        │   │  │ ● │  │   │          │  ● = DC
│        │   │  └───┘  │   │          │  Inner = Low freq
│        │   └─────────┘   │          │  Middle = Mid freq
│        └─────────────────┘          │  Outer = High freq
│                                     │
└─────────────────────────────────────┘

All points on same ring = Same frequency magnitude
                        = Same detail level
```

### 4.4 Frequency and Spatial Scale

```
FREQUENCY ↔ SPATIAL SCALE RELATIONSHIP:

Frequency (f)         Wavelength (λ)        Spatial Feature
─────────────         ──────────────        ───────────────

f = 1 cycle/image     λ = image width       Largest features
f = 2 cycles/image    λ = image/2           Half-image features
f = 4 cycles/image    λ = image/4           Quarter-image features
f = 8 cycles/image    λ = image/8           Eighth-image features
...                   ...                   ...
f = N/2 cycles        λ = 2 pixels          Finest possible detail
                                            (Nyquist limit)


EXAMPLE (64×64 image):

Frequency    Wavelength    Feature Size    What It Captures
─────────    ──────────    ────────────    ────────────────
1            64 px         Large           Overall shape
4            16 px         Medium          Major features
8            8 px          Small           Details
16           4 px          Fine            Fine texture
32           2 px          Finest          Pixel-level
```

---

## 5. Frequency Splitting: Low vs High

### 5.1 The Concept of Frequency Splitting

```
FREQUENCY SPLITTING:

Original image → Low frequencies + High frequencies

┌──────────────┐      ┌──────────────┐   ┌──────────────┐
│              │      │              │   │              │
│   Original   │  =   │  Low-freq    │ + │  High-freq   │
│    Image     │      │  (blurry)    │   │  (edges)     │
│              │      │              │   │              │
└──────────────┘      └──────────────┘   └──────────────┘
```

### 5.2 How Splitting Works

```
FREQUENCY DOMAIN MASKING:

Step 1: Compute FFT
┌──────────────┐         ┌──────────────┐
│   Image      │  FFT    │   Spectrum   │
│              │ ──────► │      ●       │
└──────────────┘         └──────────────┘

Step 2: Apply masks
┌──────────────┐         ┌──────────────┐
│  Low-pass    │         │  High-pass   │
│   mask:      │         │   mask:      │
│    ┌───┐     │         │  ████████    │
│    │ 1 │     │         │  ███┌─┐███   │
│    └───┘     │         │  ███│0│███   │
│  (0 outside) │         │  ███└─┘███   │
│              │         │  ████████    │
└──────────────┘         └──────────────┘

Step 3: Inverse FFT each
┌──────────────┐         ┌──────────────┐
│  Low-freq    │         │  High-freq   │
│  image       │         │  image       │
│  (blurry)    │         │  (edges)     │
└──────────────┘         └──────────────┘
```

### 5.3 What Each Band Contains

```
LOW FREQUENCY BAND:                    HIGH FREQUENCY BAND:
═══════════════════                    ════════════════════

Contains:                              Contains:
• Smooth variations                    • Rapid variations
• Overall shape                        • Edges and boundaries
• Average color regions                • Fine texture
• Large-scale structure                • Small-scale detail

Visual appearance:                     Visual appearance:
• Blurry                               • Edge-only
• Spread out                           • Localized
• Soft gradients                       • Sharp transitions

Information type:                      Information type:
• WHAT (identity)                      • WHERE (boundaries)
• Position-tolerant                    • Position-specific
```

### 5.4 Ring Pattern Decomposition Example

```
ORIGINAL RING:
┌─────────────────────────────────────┐
│                                     │
│           ░░░░░░░░░░░               │
│         ░░░        ░░░              │
│        ░░            ░░             │
│       ░░              ░░            │
│       ░░              ░░            │
│        ░░            ░░             │
│         ░░░        ░░░              │
│           ░░░░░░░░░░░               │
│                                     │
└─────────────────────────────────────┘

LOW FREQUENCY (cutoff 0.15):           HIGH FREQUENCY (cutoff 0.15):
┌─────────────────────────────────────┐ ┌─────────────────────────────────────┐
│                                     │ │                                     │
│          ▒▒▒▒▒▒▒▒▒▒▒▒               │ │           ░░░░░░░░░░░               │
│        ▒▒▒▒▒▒▒▒▒▒▒▒▒▒               │ │         ░░         ░░               │
│       ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒              │ │        ░             ░              │
│      ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒             │ │       ░               ░             │
│      ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒             │ │       ░               ░             │
│       ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒              │ │        ░             ░              │
│        ▒▒▒▒▒▒▒▒▒▒▒▒▒▒               │ │         ░░         ░░               │
│          ▒▒▒▒▒▒▒▒▒▒▒▒               │ │           ░░░░░░░░░░░               │
│                                     │ │                                     │
└─────────────────────────────────────┘ └─────────────────────────────────────┘

"There's a round blob"                  "The boundary is here"
(filled in, no hole)                    (only the ring edge)
= WHAT                                  = WHERE
```

### 5.5 Cutoff Frequency Effects

```
EFFECT OF DIFFERENT CUTOFF VALUES:

Cutoff = 0.05 (very low):
├── Low band: Extremely blurry, just a smear
├── High band: Almost everything, very detailed
└── Split: Minimal WHAT, maximum WHERE

Cutoff = 0.15 (low):
├── Low band: Blurry blob shape
├── High band: Clear edges
└── Split: Good WHAT/WHERE separation

Cutoff = 0.25 (medium):
├── Low band: Some structure visible
├── High band: Fine edges only
└── Split: Balanced

Cutoff = 0.40 (high):
├── Low band: Most detail retained
├── High band: Only finest edges
└── Split: Maximum WHAT, minimal WHERE


VISUAL:

Cutoff:     0.05          0.15          0.25          0.40
            ────          ────          ────          ────
            
Low:         ▒            ▒▒▒          ▒▒▒▒▒         ▒▒▒▒▒▒▒
            ▒▒▒          ▒▒▒▒▒        ▒▒▒▒▒▒▒       ▒▒▒ ▒▒▒
             ▒            ▒▒▒          ▒▒▒▒▒         ▒▒▒▒▒▒▒
           (smear)      (blob)       (shape)      (structure)

High:      ░░░░░░        ░░░░░          ░░░            ░
          ░░░░░░░░      ░░   ░░       ░   ░           ░ ░
          ░░░░░░░░       ░░░░░          ░░░            ░
         (detailed)    (edges)       (faint)       (finest)
```

### 5.6 Frequency Splitting Formula

```python
def split_frequencies(image, cutoff=0.2):
    """
    Split image into low and high frequency components.
    
    Args:
        image: 2D array [H, W]
        cutoff: Frequency cutoff (0-1, fraction of max frequency)
    
    Returns:
        low_freq, high_freq: The two frequency bands
    """
    H, W = image.shape
    
    # Compute FFT
    fft = np.fft.fft2(image)
    fft_shifted = np.fft.fftshift(fft)
    
    # Create frequency distance grid
    cy, cx = H // 2, W // 2
    y = np.arange(H) - cy
    x = np.arange(W) - cx
    X, Y = np.meshgrid(x, y)
    freq_dist = np.sqrt(X**2 + Y**2) / min(cx, cy)
    
    # Create masks (smooth transition)
    low_mask = 1 / (1 + np.exp(20 * (freq_dist - cutoff)))
    high_mask = 1 - low_mask
    
    # Apply masks and inverse FFT
    low_fft = fft_shifted * low_mask
    high_fft = fft_shifted * high_mask
    
    low_freq = np.fft.ifft2(np.fft.ifftshift(low_fft)).real
    high_freq = np.fft.ifft2(np.fft.ifftshift(high_fft)).real
    
    return low_freq, high_freq
```

---

## 6. How Frequency Affects Experiments

### 6.1 Experiment Setup

Consider tracking a pattern (ring) moving through a frame:

```
EXPERIMENT: Ring moving in circle

Frame 0:        Frame 10:       Frame 20:       Frame 30:
    ○               ○                               ○
                        ○           ○
    
Position:       Position:       Position:       Position:
Top             Top-right       Right           Bottom-right
```

### 6.2 Effect on Amplitude (WHAT)

```
AMPLITUDE BEHAVIOR BY FREQUENCY:

                    LOW FREQUENCY           HIGH FREQUENCY
                    AMPLITUDE               AMPLITUDE
                    ─────────────           ──────────────

What it shows:      Blob presence           Ring edges
                    (filled shape)          (boundary only)

Position change:    Minimal change          Cross rotates
                    (robust)                (boundary effect)

Information:        "There's a ring"        "Ring has edges"
                    (WHAT)                  (partial WHERE)

Reliability:        HIGH                    MEDIUM
                    (stable)                (boundary artifacts)
```

### 6.3 Effect on Phase (WHERE)

```
PHASE BEHAVIOR BY FREQUENCY:

                    LOW FREQUENCY           HIGH FREQUENCY
                    PHASE                   PHASE
                    ─────────               ───────────

What it shows:      Smooth gradient         Rapid gradient
                    (position ramp)         (wrapped)

Visual:             Clear color ramp        Checkerboard pattern
                    
Position change:    Gradient rotates        Gradient rotates
                    (smooth)                (with wrapping)

Information:        Coarse position         Fine position
                    (WHERE rough)           (WHERE precise)

Reliability:        HIGH                    HIGH
                    (easy to read)          (wrapping aside)
```

### 6.4 Step-by-Step: What Happens as Ring Moves

```
STEP 1: Ring at TOP (0.5, 0.2)
─────────────────────────────────

Spatial:          Low-Freq Amp:     High-Freq Amp:    Phase (all):
    ○                 ▒▒▒               ───              ↓
                     ▒▒▒▒▒           ──────────        Vertical
                      ▒▒▒                              gradient


STEP 2: Ring at TOP-RIGHT (0.7, 0.3)
────────────────────────────────────

Spatial:          Low-Freq Amp:     High-Freq Amp:    Phase (all):
        ○             ▒▒▒               ╲ ───           ↙
                     ▒▒▒▒▒           ──────────        Diagonal
                      ▒▒▒               ╱              gradient


STEP 3: Ring at RIGHT (0.8, 0.5)
────────────────────────────────

Spatial:          Low-Freq Amp:     High-Freq Amp:    Phase (all):
            ○         ▒▒▒               │              ←
                     ▒▒▒▒▒           ───●───          Horizontal
                      ▒▒▒               │              gradient


STEP 4: Ring at BOTTOM-RIGHT (0.7, 0.7)
───────────────────────────────────────

Spatial:          Low-Freq Amp:     High-Freq Amp:    Phase (all):
                      ▒▒▒               ╱ ───           ↖
        ○            ▒▒▒▒▒           ──────────        Diagonal
                      ▒▒▒               ╲              gradient


OBSERVATIONS:
─────────────
• Low-freq amplitude: STABLE (always blob shape)
• High-freq amplitude: ROTATES (cross pattern changes)
• Phase gradient: ROTATES (tracks position)
```

### 6.5 Frequency Cutoff Impact on Experiments

```
EXPERIMENT: Measure WHAT/WHERE separation quality

Metric: Cross-position similarity
        (same pattern at different positions)


CUTOFF = 0.10 (very low):
─────────────────────────
Low-freq similarity:  0.95  (very high - same blob)
High-freq similarity: 0.85  (high - most is high-freq)
Separation ratio:     1.1x  (poor separation)

Problem: Almost everything is "high frequency"


CUTOFF = 0.20 (balanced):
─────────────────────────
Low-freq similarity:  0.92  (high - same structure)
High-freq similarity: 0.45  (medium - edges differ)
Separation ratio:     2.0x  (good separation)

Result: Clear WHAT/WHERE distinction


CUTOFF = 0.35 (high):
─────────────────────
Low-freq similarity:  0.88  (high - includes some edges)
High-freq similarity: 0.15  (low - only finest detail)
Separation ratio:     5.9x  (very high separation)

Problem: Low-freq now includes edge information


OPTIMAL: Cutoff around 0.15-0.25 for most patterns
```

### 6.6 Pattern Size vs Frequency Cutoff

```
PATTERN SIZE AFFECTS OPTIMAL CUTOFF:

Small pattern (size 0.08):
├── Main energy at higher frequencies
├── Optimal cutoff: 0.25-0.35
└── Lower cutoff misses the pattern

Medium pattern (size 0.12):
├── Energy spread across frequencies
├── Optimal cutoff: 0.15-0.25
└── Standard default works well

Large pattern (size 0.20):
├── Main energy at lower frequencies
├── Optimal cutoff: 0.10-0.20
└── Higher cutoff splits the pattern


RULE OF THUMB:
Optimal cutoff ≈ 1 / (pattern_diameter_in_pixels × 2)

For 64×64 grid with pattern size 0.12:
Pattern diameter ≈ 0.12 × 64 ≈ 8 pixels
Optimal cutoff ≈ 1 / (8 × 2) ≈ 0.06 to 0.15
```

### 6.7 Grid Size Effects

```
GRID SIZE AFFECTS FREQUENCY RESOLUTION:

32×32 Grid:
├── Frequency resolution: 32 discrete frequencies
├── Low-freq band: ~5-6 frequencies
├── High-freq band: ~26 frequencies
├── Boundary effects: SEVERE
└── Recommendation: Use cutoff 0.20-0.30

64×64 Grid:
├── Frequency resolution: 64 discrete frequencies
├── Low-freq band: ~10-12 frequencies
├── High-freq band: ~52 frequencies
├── Boundary effects: MODERATE
└── Recommendation: Use cutoff 0.15-0.25

128×128 Grid:
├── Frequency resolution: 128 discrete frequencies
├── Low-freq band: ~20-25 frequencies
├── High-freq band: ~103 frequencies
├── Boundary effects: MILD
└── Recommendation: Use cutoff 0.10-0.20
```

---

## 7. The Complete WHAT/WHERE Framework

### 7.1 Summary Table

```
┌────────────────┬─────────────────────┬─────────────────────┐
│   COMPONENT    │       WHAT          │       WHERE         │
├────────────────┼─────────────────────┼─────────────────────┤
│                │                     │                     │
│   AMPLITUDE    │   Low-freq amp      │   High-freq amp     │
│                │   = Structure       │   = Edges           │
│                │   (position-        │   (position-        │
│                │    invariant)       │    specific)        │
│                │                     │                     │
├────────────────┼─────────────────────┼─────────────────────┤
│                │                     │                     │
│   PHASE        │   (less relevant)   │   All frequencies   │
│                │                     │   = Position        │
│                │                     │   (gradient encodes │
│                │                     │    location)        │
│                │                     │                     │
├────────────────┼─────────────────────┼─────────────────────┤
│                │                     │                     │
│   FREQUENCY    │   Low frequencies   │   High frequencies  │
│                │   = Spread out      │   = Localized       │
│                │   = Coarse          │   = Fine            │
│                │                     │                     │
└────────────────┴─────────────────────┴─────────────────────┘
```

### 7.2 Information Flow

```
            INPUT IMAGE
                 │
                 ▼
            ┌────────┐
            │  FFT   │
            └────────┘
                 │
       ┌─────────┴─────────┐
       │                   │
       ▼                   ▼
  ┌─────────┐         ┌─────────┐
  │AMPLITUDE│         │  PHASE  │
  └─────────┘         └─────────┘
       │                   │
   ┌───┴───┐               │
   │       │               │
   ▼       ▼               ▼
┌─────┐ ┌─────┐       ┌─────────┐
│ Low │ │High │       │Gradient │
│freq │ │freq │       │direction│
└─────┘ └─────┘       └─────────┘
   │       │               │
   ▼       ▼               ▼
┌─────┐ ┌─────┐       ┌─────────┐
│WHAT │ │WHERE│       │  WHERE  │
│(id) │ │(edge│       │(position│
└─────┘ └─────┘       └─────────┘
```

### 7.3 Practical WHAT/WHERE Extraction

```python
def extract_what_where(image, freq_cutoff=0.2):
    """
    Extract WHAT and WHERE information from image.
    
    Returns:
        what: Structure information (position-invariant)
        where_edges: Edge locations (from high-freq amplitude)
        where_position: Pattern position (from phase gradient)
    """
    H, W = image.shape
    
    # FFT
    fft = np.fft.fft2(image)
    fft_shifted = np.fft.fftshift(fft)
    
    # Amplitude and phase
    amplitude = np.abs(fft_shifted)
    phase = np.angle(fft_shifted)
    
    # Frequency mask
    cy, cx = H // 2, W // 2
    y, x = np.ogrid[-cy:H-cy, -cx:W-cx]
    freq_dist = np.sqrt(x**2 + y**2) / min(cx, cy)
    low_mask = freq_dist < freq_cutoff
    high_mask = ~low_mask
    
    # WHAT: Low-frequency amplitude
    what = amplitude * low_mask
    
    # WHERE (edges): High-frequency amplitude  
    where_edges = amplitude * high_mask
    
    # WHERE (position): Phase gradient
    # Compute gradient of phase (unwrapped in center region)
    phase_center = phase[cy-5:cy+5, cx-5:cx+5]
    grad_y, grad_x = np.gradient(phase_center)
    where_position = {
        'x': -np.mean(grad_x) / (2 * np.pi) * W,
        'y': -np.mean(grad_y) / (2 * np.pi) * H
    }
    
    return what, where_edges, where_position
```

---

## 8. Practical Guidelines

### 8.1 Choosing Frequency Cutoff

```
DECISION TREE:

What is your goal?
│
├─► Robust object recognition (WHAT focus)
│   └─► Use lower cutoff (0.10-0.15)
│       More spread, more position-tolerant
│
├─► Precise localization (WHERE focus)
│   └─► Use higher cutoff (0.25-0.35)
│       More edges, more position-specific
│
├─► Balanced WHAT/WHERE
│   └─► Use medium cutoff (0.15-0.25)
│       Good separation of both
│
└─► Pattern-dependent
    └─► Match cutoff to pattern scale
        Small patterns: higher cutoff
        Large patterns: lower cutoff
```

### 8.2 Common Pitfalls

```
PITFALL 1: Cutoff too low
───────────────────────────
Symptom: Everything is "high frequency"
Result: Poor WHAT extraction
Fix: Increase cutoff

PITFALL 2: Cutoff too high
──────────────────────────
Symptom: Low-freq includes edges
Result: WHAT contains WHERE information
Fix: Decrease cutoff

PITFALL 3: Ignoring grid size
─────────────────────────────
Symptom: Inconsistent results across scales
Result: WHAT/WHERE separation varies
Fix: Scale cutoff with grid size

PITFALL 4: Ignoring pattern size
────────────────────────────────
Symptom: Pattern split incorrectly
Result: Ring appears as blob in both bands
Fix: Match cutoff to pattern scale

PITFALL 5: Hard cutoff (step function)
──────────────────────────────────────
Symptom: Ringing artifacts in spatial domain
Result: Artificial patterns in reconstructions
Fix: Use smooth (sigmoid) transition
```

### 8.3 Experimental Checklist

```
BEFORE RUNNING EXPERIMENTS:

□ Choose appropriate grid size (64×64 minimum recommended)
□ Set pattern size relative to grid (0.1-0.2 typical)
□ Calculate appropriate frequency cutoff
□ Verify pattern is not near boundaries
□ Apply windowing if boundary effects matter

DURING EXPERIMENTS:

□ Monitor low-freq amplitude for WHAT stability
□ Watch for high-freq cross rotation (boundary effect)
□ Check phase gradient tracks position correctly
□ Compare cross-position similarities

INTERPRETING RESULTS:

□ Low-freq similarity high → Same structure detected
□ High-freq similarity varies → Position-specific edges
□ Phase gradient direction → Position location
□ Ratio (low/high similarity) → WHAT/WHERE separation quality
```

### 8.4 Quick Reference

```
FREQUENCY CUTOFF QUICK REFERENCE:

Cutoff    Low-Freq Character    High-Freq Character    Best For
──────    ──────────────────    ───────────────────    ────────
0.05      Extreme blur          Almost everything      -
0.10      Very blurry           Most detail            Large patterns
0.15      Blurry                Clear edges            Default start
0.20      Soft                  Good edges             Balanced
0.25      Some detail           Fine edges             Medium patterns
0.30      More detail           Finest edges           Small patterns
0.40      Mostly intact         Only finest            Edge focus


AMPLITUDE vs PHASE QUICK REFERENCE:

Need                          Use
────                          ───
Object identity               Low-freq amplitude
Edge locations                High-freq amplitude
Pattern position              Phase gradient
Position-invariant matching   Low-freq amplitude comparison
Position-specific matching    High-freq amplitude comparison
```

---

## Summary

```
THE THREE COMPONENTS:

1. AMPLITUDE
   └─► Encodes HOW MUCH of each frequency
   └─► Low-freq amplitude = WHAT (structure)
   └─► High-freq amplitude = WHERE (edges)
   └─► Theoretically position-invariant

2. PHASE  
   └─► Encodes WHERE each frequency is positioned
   └─► Gradient direction = pattern position
   └─► Critical for perception
   └─► True position signal (WHERE)

3. FREQUENCY
   └─► Controls RESOLUTION of analysis
   └─► Low frequency = spread out, coarse
   └─► High frequency = localized, fine
   └─► Cutoff determines WHAT/WHERE split


THE WHAT/WHERE SEPARATION:

WHAT = "What object is this?"
     = Low-frequency amplitude
     = Position-invariant structure
     = Spread out, blurry representation

WHERE = "Where exactly is it?"
      = High-frequency amplitude (edges)
      = Phase gradient (position)
      = Localized, precise representation


FREQUENCY SPLITTING enables this separation by isolating
the coarse structure (WHAT) from the fine details (WHERE).
```

---

*This document provides the theoretical foundation for understanding how spectral decomposition enables the separation of object identity from spatial location in the spectral attention framework.*

