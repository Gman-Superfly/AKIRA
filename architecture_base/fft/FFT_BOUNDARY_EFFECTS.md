# FFT Boundary Effects: Magnitude Oscillation and Phase Behavior

This document explains why FFT magnitude appears to oscillate when a pattern moves in a finite domain, and how phase faithfully encodes position despite these boundary effects.

---

## Table of Contents

1. [The Theoretical Promise](#1-the-theoretical-promise)
2. [What We Actually Observe](#2-what-we-actually-observe)
3. [The Root Cause: Finite Domains](#3-the-root-cause-finite-domains)
4. [Step-by-Step Breakdown](#4-step-by-step-breakdown)
5. [Visual Explanation](#5-visual-explanation)
6. [The Three Regimes](#6-the-three-regimes)
7. [**Phase Behavior: The WHERE Signal**](#7-phase-behavior-the-where-signal)
8. [**Low vs High Frequency: Spread vs Detail**](#8-low-vs-high-frequency-spread-vs-detail)
9. [Solutions: Windowing](#9-solutions-windowing)
10. [Implications for Spectral Attention](#10-implications-for-spectral-attention)
11. [Mathematical Details](#11-mathematical-details)

---

## 1. The Theoretical Promise

The Fourier Shift Theorem states:

```
If f(x) has Fourier transform F(ω), then:

f(x - Δ) → F(ω) · exp(-i·ω·Δ)
```

This means:
- **Magnitude** `|F(ω)|` should be **unchanged** by translation
- **Phase** `arg(F(ω))` should change linearly with shift

This is the mathematical foundation of the what/where separation:
- Magnitude = WHAT (position-invariant structure)
- Phase = WHERE (position-specific location)

**But in practice, we see the magnitude oscillate. Why?**

---

## 2. What We Actually Observe

When a ring pattern moves in a circular path around the center of a 64×64 grid:

```
Position A: Ring at (0.25, 0.5)        Position B: Ring at (0.5, 0.25)
┌────────────────────────────┐        ┌────────────────────────────┐
│                            │        │              ○○            │
│      ○○○                   │        │             ○  ○           │
│     ○   ○                  │        │              ○○            │
│      ○○○                   │        │                            │
│                            │        │                            │
└────────────────────────────┘        └────────────────────────────┘

FFT Magnitude A:                       FFT Magnitude B:
┌────────────────────────────┐        ┌────────────────────────────┐
│            │               │        │            ─               │
│            │               │        │            ─               │
│     ───────●───────        │        │     ───────●───────        │
│            │               │        │            ─               │
│            │               │        │            ─               │
└────────────────────────────┘        └────────────────────────────┘
   Horizontal cross                      Vertical cross
```

The FFT magnitude **rotates** as the pattern moves! This contradicts the shift theorem.

---

## 3. The Root Cause: Finite Domains

The shift theorem assumes an **infinite, continuous domain**. Our actual situation:

```
THEORY (Infinite Domain):
────────────────────────────────────────────────────────────────────
... empty ... │ ○○○ │ ... empty ... │ ○○○ │ ... empty ...
────────────────────────────────────────────────────────────────────
              ↑ pattern              ↑ same pattern shifted
              
              Both have IDENTICAL Fourier transforms (in magnitude)


PRACTICE (Finite Domain with Periodic Boundaries):
┌──────────────────┐     ┌──────────────────┐
│ ○○○              │     │              ○○○ │
│○   ○             │     │             ○   ○│
│ ○○○              │     │              ○○○ │
│                  │     │                  │
│                  │     │                  │
└──────────────────┘     └──────────────────┘
 Ring on LEFT side        Ring on RIGHT side
 
 These are DIFFERENT structures to the FFT!
```

The FFT sees these as different because:

1. **The finite frame acts as an implicit window**
2. **Different parts of the frame are "filled" vs "empty"**
3. **The periodic boundary assumption creates different tilings**

---

## 4. Step-by-Step Breakdown

### Step 1: What the FFT Actually "Sees"

The 2D FFT assumes the input tiles infinitely in all directions:

```
Ring at (0.25, 0.5) - What FFT sees:

┌────┬────┬────┐
│ ○  │ ○  │ ○  │
│    │    │    │
├────┼────┼────┤
│ ○  │ ○  │ ○  │  ← Infinite tiling
│    │    │    │
├────┼────┼────┤
│ ○  │ ○  │ ○  │
│    │    │    │
└────┴────┴────┘

The ring is always on the LEFT of each tile.
Creates a LEFT-biased periodic structure.
```

```
Ring at (0.75, 0.5) - What FFT sees:

┌────┬────┬────┐
│  ○ │  ○ │  ○ │
│    │    │    │
├────┼────┼────┤
│  ○ │  ○ │  ○ │  ← Different tiling!
│    │    │    │
├────┼────┼────┤
│  ○ │  ○ │  ○ │
│    │    │    │
└────┴────┴────┘

The ring is always on the RIGHT of each tile.
Creates a RIGHT-biased periodic structure.
```

### Step 2: The Asymmetry Creates Directional Frequencies

When the ring is off-center, the frame has an **asymmetric mass distribution**:

```
Ring at LEFT (cx = 0.25):
                                    
Intensity profile (horizontal):     
                                    
  ████                              
  █  █                              
  ████        ← Mass concentrated left
  │                                 
  └──────────────────────────────►  
  0                               1  
                                    
  This creates a HORIZONTAL GRADIENT
  FFT of horizontal gradient → HORIZONTAL frequencies
```

```
Ring at TOP (cy = 0.25):

Intensity profile (vertical):

  ────────── 1
       │
       ████ ← Mass concentrated top
       │
       │
  ────────── 0

  This creates a VERTICAL GRADIENT
  FFT of vertical gradient → VERTICAL frequencies
```

### Step 3: The Cross Pattern Emerges

The cross/plus pattern in the FFT magnitude comes from **edge interactions**:

```
A ring in a finite frame creates edges:

Frame:  ┌─────────────────┐
        │                 │
        │    ○○○○○        │
        │   ○     ○       │
        │    ○○○○○        │
        │                 │← Empty space = implicit edge
        │                 │
        └─────────────────┘
              ↑
        Ring edge
        
Two types of edges:
1. Ring's own circular edge (always present)
2. Implicit rectangular edges from frame boundaries
```

The FFT of these edges:

```
Vertical edge (ring on left)     Horizontal edge (ring on top)
        │                                ────
        │                                ────
        │                                ────
        ↓                                 ↓
Horizontal frequency line        Vertical frequency line
in FFT magnitude                 in FFT magnitude
```

### Step 4: The Rotation Effect

As the ring moves in a circle, the dominant edge orientation rotates:

```
Position vs FFT Cross Orientation:

Ring Position          FFT Magnitude Pattern
─────────────          ─────────────────────

    ○                        │
   (top)                     │
                       ──────●──────
                             │
                             │
                      (vertical dominant)

○        (left)        ──────●──────
                             
                      (horizontal dominant)

    ○                        │
   (bottom)                  │
                       ──────●──────
                             │
                             │
                      (vertical dominant)

        ○    (right)   ──────●──────
                             
                      (horizontal dominant)
```

The cross pattern **rotates 90° every quarter turn** of the ring's circular path.

---

## 5. Visual Explanation

### The Complete Picture

```
                    RING MOVING IN CIRCLE
                    
                         (0.5, 0.2)
                            ○
                           /│\
                          / │ \
                         /  │  \
         (0.2, 0.5) ○───┼───●───┼───○ (0.8, 0.5)
                         \  │  /
                          \ │ /
                           \│/
                            ○
                         (0.5, 0.8)
                         
                         
                    FFT MAGNITUDE RESPONSE
                    
         Position          FFT Cross         Dominant Direction
         ────────          ─────────         ──────────────────
         Top               Vertical          Up-Down asymmetry
         Right             Horizontal        Left-Right asymmetry
         Bottom            Vertical          Up-Down asymmetry
         Left              Horizontal        Left-Right asymmetry
```

### Why the Central Peak Stays

Despite the cross rotating, the **central bright spot** remains stable:

```
FFT Magnitude Structure:

         Rotating cross (boundary effect)
                   │
                   ▼
         ─────────────────
               │
               │
         ──────●──────  ← Central peak (true structure)
               │           Always present, encodes "ring-ness"
               │
         ─────────────────
         
The central peak = DC component + low frequencies
This part IS position-invariant!
```

---

## 6. The Three Regimes

Position invariance depends on where the pattern is:

### Regime 1: Interior (Far from Boundaries)

```
┌────────────────────────────────┐
│                                │
│                                │
│          ○○○○                  │
│         ○    ○                 │
│          ○○○○    ← Ring fully inside
│                                │
│                                │
└────────────────────────────────┘

Magnitude: Nearly identical for small shifts
Boundary effects: Minimal
Shift theorem: Approximately valid
```

### Regime 2: Near Boundary

```
┌────────────────────────────────┐
│                                │
│ ○○○○                           │
│○    ○   ← Ring near edge       │
│ ○○○○                           │
│                                │
│                                │
│                                │
└────────────────────────────────┘

Magnitude: Oscillates with position
Boundary effects: Dominant
Shift theorem: Breaks down
```

### Regime 3: Wrapping (Crosses Boundary)

```
┌────────────────────────────────┐
│○                             ○○│
│                               ○│ ← Ring wraps around!
│○                             ○○│
│                                │
│                                │
└────────────────────────────────┘

Magnitude: Very different structure
Boundary effects: Severe discontinuity
Shift theorem: Completely invalid
```

---

## 7. Phase Behavior: The WHERE Signal

While magnitude shows boundary-induced oscillations, the **phase** behaves exactly as the shift theorem predicts. Phase is the true WHERE signal.

### 7.1 What Phase Looks Like

When visualizing FFT phase as the ring moves, we observe two distinct appearances:

```
SMOOTH GRADIENT PHASE                    CHECKERBOARD PHASE
(Ring at cardinal positions)             (Ring at diagonal positions)

┌─────────────────────────┐              ┌─────────────────────────┐
│ Red    → Orange → Yellow│              │▓░▓░▓░▓░▓░▓░▓░▓░▓░▓░▓░▓░│
│   ↘                     │              │░▓░▓░▓░▓░▓░▓░▓░▓░▓░▓░▓░▓│
│     Green → Cyan        │              │▓░▓░▓░▓░▓░▓░▓░▓░▓░▓░▓░▓░│
│       ↘                 │              │░▓░▓░▓░▓░▓░▓░▓░▓░▓░▓░▓░▓│
│         Blue → Purple   │              │▓░▓░▓░▓░▓░▓░▓░▓░▓░▓░▓░▓░│
└─────────────────────────┘              └─────────────────────────┘

Smooth diagonal gradient                  Rapid color oscillations
Position encoded in gradient angle        Phase wrapping creates pattern
```

Both appearances encode position information - they just look different due to phase wrapping.

### 7.2 The Phase Ramp: Position Encoding

The shift theorem tells us exactly how phase encodes position:

```
For a pattern shifted by (Δx, Δy):

Phase change at frequency (u, v) = -2π(u·Δx + v·Δy)

This creates a LINEAR PHASE RAMP across the frequency domain.
```

Visualized:

```
Ring at CENTER (0.5, 0.5):          Ring at TOP-LEFT (0.25, 0.25):
No shift → No phase ramp            Shifted → Diagonal phase ramp

Phase:                              Phase:
┌─────────────────────┐             ┌─────────────────────┐
│                     │             │ 0°                  │
│         0°          │             │    45°              │
│    (flat phase)     │             │       90°           │
│                     │             │          135°       │
│                     │             │             180°    │
└─────────────────────┘             └─────────────────────┘
                                    
Uniform color                       Diagonal gradient
```

### 7.3 Why Phase Alternates Between Smooth and Checkerboard

The phase spans from -π to +π (or 0 to 2π). When mapped to colors:

```
PHASE VALUE TO COLOR MAPPING (HSV):

-π ─────────────────────────────────────────────────────── +π
 │                                                          │
 Red   Orange  Yellow  Green   Cyan   Blue   Purple   Red   │
 │                                                          │
 └──────────────────────────────────────────────────────────┘
                    (wraps around)
```

At **high frequencies**, the phase changes rapidly across pixels:

```
HIGH FREQUENCY REGION (edges of FFT):

Pixel 1:  Phase = 0°      → Red
Pixel 2:  Phase = 180°    → Cyan
Pixel 3:  Phase = 360°=0° → Red
Pixel 4:  Phase = 180°    → Cyan
...

Result: Checkerboard pattern (rapid alternation)
```

At **low frequencies** (center of FFT), phase changes slowly:

```
LOW FREQUENCY REGION (center of FFT):

Pixel 1:  Phase = 0°   → Red
Pixel 2:  Phase = 10°  → Red-Orange
Pixel 3:  Phase = 20°  → Orange
Pixel 4:  Phase = 30°  → Orange-Yellow
...

Result: Smooth gradient
```

### 7.4 How Position Affects Phase Gradient Direction

The direction of the phase gradient directly encodes the ring's position:

```
                    RING POSITION vs PHASE GRADIENT
                    
Ring at TOP (0.5, 0.25):                Ring at BOTTOM (0.5, 0.75):
                                        
  Spatial:    ○                         Spatial:
             ───                                   ───
                                                    ○
                                        
  Phase gradient: VERTICAL              Phase gradient: VERTICAL
                                        (opposite direction)
  ┌───────────────┐                     ┌───────────────┐
  │ Blue          │                     │ Red           │
  │ ↓             │                     │ ↓             │
  │ Cyan          │                     │ Orange        │
  │ ↓             │                     │ ↓             │
  │ Green         │                     │ Yellow        │
  │ ↓             │                     │ ↓             │
  │ Yellow        │                     │ Green         │
  │ ↓             │                     │ ↓             │
  │ Red           │                     │ Blue          │
  └───────────────┘                     └───────────────┘


Ring at LEFT (0.25, 0.5):               Ring at RIGHT (0.75, 0.5):

  Spatial:  ○ │                         Spatial:    │ ○
              │                                     │
                                        
  Phase gradient: HORIZONTAL            Phase gradient: HORIZONTAL
                                        (opposite direction)
  ┌───────────────┐                     ┌───────────────┐
  │Blue→Cyan→Green│                     │Red→Org→Yel→Grn│
  │→Yellow→Red    │                     │→Cyan→Blue     │
  └───────────────┘                     └───────────────┘
```

### 7.5 The Complete Phase Rotation

As the ring moves in a circle, the phase gradient rotates:

```
                        PHASE GRADIENT ROTATION
                        
                              Ring at Top
                                  ○
                              Gradient: ↓
                              (vertical)
                                  
                                  │
                                  │
     Ring at Left                 │                Ring at Right
         ○ ←─── Gradient: →  ────┼───── Gradient: → ───→ ○
                (horizontal)      │                (horizontal)
                                  │
                                  │
                                  
                              Gradient: ↓
                              (vertical)
                                  ○
                              Ring at Bottom


Phase gradient angle = Position angle + 90°

Position    Gradient Direction    Phase Appearance
────────    ──────────────────    ─────────────────
0° (right)      0° (→)           Horizontal gradient
45° (diag)      45° (↘)          Diagonal + wrapping = checkerboard  
90° (top)       90° (↓)          Vertical gradient
135° (diag)     135° (↙)         Diagonal + wrapping = checkerboard
180° (left)     180° (←)         Horizontal gradient (reversed)
225° (diag)     225° (↖)         Diagonal + wrapping = checkerboard
270° (bottom)   270° (↑)         Vertical gradient (reversed)
315° (diag)     315° (↗)         Diagonal + wrapping = checkerboard
```

### 7.6 Why Checkerboard at Diagonal Positions

The checkerboard appears specifically at diagonal positions (45°, 135°, etc.):

```
DIAGONAL PHASE GRADIENT (45°):

The gradient runs diagonally across the frequency domain:

┌─────────────────────────────────┐
│ 0°    45°    90°    135°   180° │
│   45°    90°    135°   180°     │
│      90°    135°   180°   225°  │  ← Phase increases diagonally
│         135°   180°   225°      │
│            180°   225°   270°   │
└─────────────────────────────────┘

At HIGH frequencies (edges), adjacent pixels differ by ~180°:

Pixel[i,j]   = θ
Pixel[i+1,j] = θ + 90°
Pixel[i,j+1] = θ + 90°
Pixel[i+1,j+1] = θ + 180° (opposite!)

This creates the checkerboard pattern.
```

For cardinal directions (0°, 90°, 180°, 270°), the gradient aligns with pixel axes:

```
HORIZONTAL PHASE GRADIENT (0°):

┌─────────────────────────────────┐
│ 0°   30°   60°   90°  120° 150° │
│ 0°   30°   60°   90°  120° 150° │  ← Same phase in each column
│ 0°   30°   60°   90°  120° 150° │
│ 0°   30°   60°   90°  120° 150° │
└─────────────────────────────────┘

Adjacent pixels in Y have SAME phase → smooth columns
Adjacent pixels in X have DIFFERENT phase → smooth gradient

Result: Smooth horizontal color bands
```

### 7.7 Phase Wrapping Visualization

Phase wraps at ±π, creating discontinuities in the visualization:

```
UNWRAPPED PHASE:                    WRAPPED PHASE (what we see):

Phase value                         Color
    │                                   │
 3π ┤                                Red ┤ ──────
    │         /                         │      │
 2π ┤        /                     Violet┤      │
    │       /                           │      │
  π ┤      /                       Blue ┤──────│──────
    │     /                             │      │
  0 ┤    /                         Green┤      │
    │   /                               │      │
 -π ┤  /                          Yellow┤──────│──────
    │ /                                 │      │
-2π ┤/                              Red ┤      │
    └────────────────► x                └──────┴──────► x
    
    Continuous ramp                     Repeating pattern
                                        (sawtooth wrapped to color)
```

### 7.8 Mathematical Description of Phase Behavior

The phase at frequency (u, v) for a pattern at position (x₀, y₀):

```
φ(u, v) = φ₀(u, v) - 2π(u·x₀ + v·y₀)

Where:
  φ₀(u, v) = phase of centered pattern
  x₀, y₀   = position offset from center
  u, v     = frequency coordinates
```

The phase gradient vector:

```
∇φ = (-2π·x₀, -2π·y₀)

Gradient magnitude: |∇φ| = 2π·√(x₀² + y₀²) = 2π·r
Gradient direction: θ = atan2(y₀, x₀)

The gradient points TOWARD the pattern position (from center)
```

### 7.9 Phase vs Magnitude: The Complete Picture

```
SUMMARY: MAGNITUDE vs PHASE BEHAVIOR

                    MAGNITUDE                    PHASE
                    ─────────                    ─────
What it encodes     Structure (WHAT)             Position (WHERE)
                    
Theoretical         Invariant to shift           Changes linearly with shift
behavior            
                    
Practical           Oscillates due to            Behaves as predicted!
observation         boundary effects             
                    
Visual              Cross rotates as             Gradient rotates as
appearance          pattern moves                pattern moves
                    
At boundaries       Severely affected            Minimally affected
                    
Reliability         Approximate                  Exact (wrapping aside)
                    
```

### 7.10 Why Phase is the True Position Signal

Despite magnitude suffering from boundary effects, phase reliably encodes position:

```
PHASE RELIABILITY TEST:

Ring at (0.3, 0.5):
  Phase gradient direction: 0° (horizontal)
  Phase gradient magnitude: 2π × 0.2 = 0.4π per unit frequency

Ring at (0.5, 0.3):
  Phase gradient direction: 90° (vertical)  
  Phase gradient magnitude: 2π × 0.2 = 0.4π per unit frequency

Ring at (0.35, 0.35):
  Phase gradient direction: 45° (diagonal)
  Phase gradient magnitude: 2π × 0.21 = 0.42π per unit frequency


In ALL cases:
- Direction encodes position angle ✓
- Magnitude encodes position distance ✓
- Works regardless of boundary effects ✓
```

### 7.11 Extracting Position from Phase

To decode position from phase:

```python
def extract_position_from_phase(phase_2d):
    """
    Extract pattern position from FFT phase.
    
    The phase gradient encodes the position shift.
    """
    H, W = phase_2d.shape
    
    # Compute phase gradient
    # (unwrap phase first to handle discontinuities)
    phase_unwrapped = np.unwrap(np.unwrap(phase_2d, axis=0), axis=1)
    
    # Gradient in x and y
    grad_y, grad_x = np.gradient(phase_unwrapped)
    
    # Average gradient (weighted by magnitude if available)
    mean_grad_x = np.mean(grad_x)
    mean_grad_y = np.mean(grad_y)
    
    # Convert to position
    # grad = -2π × position_offset × (frequency_step)
    freq_step_x = 2 * np.pi / W
    freq_step_y = 2 * np.pi / H
    
    position_x = -mean_grad_x / freq_step_x / (2 * np.pi)
    position_y = -mean_grad_y / freq_step_y / (2 * np.pi)
    
    return position_x + 0.5, position_y + 0.5  # Offset from center
```

### 7.12 The Checkerboard-Smooth Alternation Cycle

As the ring completes one full circle, the phase visualization cycles:

```
COMPLETE ROTATION CYCLE:

Position Angle    Phase Appearance         Why
──────────────    ─────────────────        ───────────────────────────
0° (right)        Smooth horizontal        Gradient aligned with X-axis
22.5°             Mild checkerboard        Gradient slightly diagonal
45° (diag)        Strong checkerboard      Gradient at 45° maximizes wrap
67.5°             Mild checkerboard        Gradient approaching Y-axis
90° (top)         Smooth vertical          Gradient aligned with Y-axis
112.5°            Mild checkerboard        Gradient slightly diagonal
135° (diag)       Strong checkerboard      Gradient at 45° maximizes wrap
157.5°            Mild checkerboard        Gradient approaching X-axis
180° (left)       Smooth horizontal        Gradient aligned with X-axis
... (continues symmetrically)

Pattern: Smooth → Checker → Smooth → Checker (4 cycles per revolution)
```

### 7.13 Frequency-Dependent Phase Behavior

Different frequency regions show different phase characteristics:

```
PHASE BEHAVIOR BY FREQUENCY REGION:

┌─────────────────────────────────────────────────────────┐
│                    HIGH FREQ                             │
│                 (rapid wrapping)                         │
│         ┌───────────────────────────┐                   │
│         │                           │                   │
│  HIGH   │      LOW FREQUENCY        │   HIGH            │
│  FREQ   │     (smooth gradient)     │   FREQ            │
│         │                           │                   │
│         │           ●               │                   │
│         │        (DC=0)             │                   │
│         │                           │                   │
│         └───────────────────────────┘                   │
│                    HIGH FREQ                             │
│                 (rapid wrapping)                         │
└─────────────────────────────────────────────────────────┘

LOW FREQUENCIES (center):
- Phase changes slowly
- Clear gradient visible
- Best for position estimation

HIGH FREQUENCIES (edges):
- Phase changes rapidly  
- Wrapping creates checkerboard
- Still encodes position (just harder to see)
```

---

## 8. Low vs High Frequency: Spread vs Detail

Beyond boundary effects, there's a fundamental difference in what low and high frequencies represent. This is the core of the WHAT/WHERE separation.

### 8.1 The Fundamental Nature of Frequency Bands

```
LOW FREQUENCY                           HIGH FREQUENCY
════════════                            ══════════════

Spatial Extent: SPREAD OUT              Spatial Extent: LOCALIZED
               (covers large area)                     (fine details)

Information:   STRUCTURE                Information:   EDGES
               (overall shape)                         (boundaries, texture)

Visual:        BLURRY                   Visual:        SHARP
               (smooth gradients)                      (crisp transitions)

Changes:       SLOW                     Changes:       RAPID
               (gradual variation)                     (sudden jumps)
```

### 8.2 Visual Comparison: Ring Pattern Decomposition

When we decompose a ring pattern into frequency bands:

```
ORIGINAL RING:                    
┌─────────────────────────────┐   
│                             │   
│         ░░░░░░░             │   
│       ░░      ░░            │   
│      ░░        ░░           │   
│      ░░        ░░           │   
│       ░░      ░░            │   
│         ░░░░░░░             │   
│                             │   
└─────────────────────────────┘   
Sharp edges, clear boundary


LOW FREQUENCY BAND:               HIGH FREQUENCY BAND:
┌─────────────────────────────┐   ┌─────────────────────────────┐
│                             │   │                             │
│        ▒▒▒▒▒▒▒▒▒            │   │         ░░░░░░░             │
│      ▒▒▒▒▒▒▒▒▒▒▒▒           │   │       ░░      ░░            │
│     ▒▒▒▒▒▒▒▒▒▒▒▒▒▒          │   │      ░          ░           │
│     ▒▒▒▒▒▒▒▒▒▒▒▒▒▒          │   │      ░          ░           │
│      ▒▒▒▒▒▒▒▒▒▒▒▒           │   │       ░░      ░░            │
│        ▒▒▒▒▒▒▒▒▒            │   │         ░░░░░░░             │
│                             │   │                             │
└─────────────────────────────┘   └─────────────────────────────┘
Spread out, blurry blob           Only the edges remain
"There's something round here"    "The boundary is exactly here"
        = WHAT                            = WHERE
```

### 8.3 Why Low Frequency Spreads

Low frequencies correspond to slow spatial variations:

```
FREQUENCY vs WAVELENGTH:

Low Frequency (ω small):
───────────────────────────────────────────────────────
    ╭───────────────────╮       ╭───────────────────╮
    │                   │       │                   │
────╯                   ╰───────╯                   ╰────
    
    Long wavelength = Slow variation = Spread out


High Frequency (ω large):
───────────────────────────────────────────────────────
╭─╮ ╭─╮ ╭─╮ ╭─╮ ╭─╮ ╭─╮ ╭─╮ ╭─╮ ╭─╮ ╭─╮ ╭─╮ ╭─╮ ╭─╮ ╭─╮
│ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │
╯ ╰─╯ ╰─╯ ╰─╯ ╰─╯ ╰─╯ ╰─╯ ╰─╯ ╰─╯ ╰─╯ ╰─╯ ╰─╯ ╰─╯ ╰─╯ ╰

    Short wavelength = Rapid variation = Fine detail
```

### 8.4 Mathematical Basis: Uncertainty Principle

There's a fundamental trade-off between spatial and frequency localization:

```
UNCERTAINTY PRINCIPLE (Gabor limit):

    Δx · Δω ≥ 1/2

Where:
    Δx = spatial spread
    Δω = frequency spread

IMPLICATION:
─────────────
Low frequencies (small ω) → Must have large Δx (spread out)
High frequencies (large ω) → Can have small Δx (localized)
```

This is why:

```
Low-freq filter:                    High-freq filter:
┌───────────────┐                   ┌───────────────┐
│               │                   │               │
│    ●●●●●●●    │  Large spatial    │      ●        │  Small spatial
│   ●●●●●●●●●   │  footprint        │     ●●●       │  footprint
│    ●●●●●●●    │                   │      ●        │
│               │                   │               │
└───────────────┘                   └───────────────┘

Averages over                       Responds to
large region                        local changes
```

### 8.5 Edge Detection: Why High Frequency Shows Edges

An edge is a rapid spatial change - exactly what high frequencies capture:

```
EDGE IN SPATIAL DOMAIN:

Intensity
    │
1.0 ┤        ████████████
    │        █
    │        █
    │        █  ← Sharp edge (rapid change)
0.0 ┤████████
    └────────────────────► x
             edge


FREQUENCY CONTENT OF EDGE:

Amplitude
    │
    │████
    │████████
    │████████████
    │████████████████
    │████████████████████
    └────────────────────────► frequency
    low                high
    
    Edges have STRONG high-frequency content
```

### 8.6 Blob vs Edge: The Complete Picture

```
BLOB (Gaussian):                     RING (has edges):
                                     
Spatial:                             Spatial:
    ▓▓▓▓                                 ░░░░░
   ▓████▓                              ░░   ░░
  ▓██████▓                            ░░     ░░
   ▓████▓                              ░░   ░░
    ▓▓▓▓                                 ░░░░░
                                     
Low-freq:                            Low-freq:
    ▒▒▒▒                                 ▒▒▒▒▒
   ▒▒▒▒▒▒                              ▒▒▒▒▒▒▒
  ▒▒▒▒▒▒▒▒  ← Almost                  ▒▒▒▒▒▒▒▒▒  ← Filled blob
   ▒▒▒▒▒▒      unchanged              ▒▒▒▒▒▒▒       (no hole)
    ▒▒▒▒                                 ▒▒▒▒▒
                                     
High-freq:                           High-freq:
                                          ░░░░░
     ·                                  ░░   ░░  ← Ring edges
    · ·     ← Minimal                  ░░     ░░    visible!
     ·         edges                    ░░   ░░
                                          ░░░░░


Blob is mostly low-freq              Ring has both:
(smooth structure)                   - Low: overall presence
                                     - High: the ring edges
```

### 8.7 Information Content by Frequency

Different information lives at different frequencies:

```
FREQUENCY SPECTRUM BREAKDOWN:

         Low          Mid           High
         ────         ───           ────
    ┌─────────┬─────────────┬─────────────┐
    │         │             │             │
    │  Shape  │  Features   │   Edges     │
    │  Color  │  Texture    │   Noise     │
    │  Mass   │  Patterns   │   Detail    │
    │         │             │             │
    └─────────┴─────────────┴─────────────┘
         │           │             │
         ▼           ▼             ▼
      WHAT        WHAT+WHERE     WHERE
    (identity)   (recognition)  (location)
```

### 8.8 Reconstruction from Bands

What happens when we reconstruct from only one band:

```
LOW-FREQ ONLY:                       HIGH-FREQ ONLY:
─────────────                        ──────────────

Original + Low-pass filter           Original + High-pass filter

┌─────────────────────┐              ┌─────────────────────┐
│                     │              │                     │
│      ▒▒▒▒▒▒▒        │              │       ░   ░         │
│    ▒▒▒▒▒▒▒▒▒▒       │              │      ░     ░        │
│   ▒▒▒▒▒▒▒▒▒▒▒▒      │              │     ░       ░       │
│    ▒▒▒▒▒▒▒▒▒▒       │              │      ░     ░        │
│      ▒▒▒▒▒▒▒        │              │       ░   ░         │
│                     │              │                     │
└─────────────────────┘              └─────────────────────┘

Can tell: "There's a round                Can tell: "There's a ring
thing in the upper-left"                  boundary at this exact location"

CANNOT tell: Exact boundary              CANNOT tell: What's inside,
                                         or overall shape


LOW + HIGH = COMPLETE INFORMATION
```

### 8.9 Spatial Spread Quantified

The spatial extent of frequency bands can be measured:

```
EFFECTIVE SPATIAL SUPPORT:

For a frequency cutoff at f_c:

Low-freq spatial spread ≈ 1/f_c pixels
High-freq spatial spread ≈ 1/f_max pixels (typically 1-2 pixels)


Example with 64×64 grid, cutoff at 0.2:
────────────────────────────────────────

f_c = 0.2 × 32 = 6.4 cycles across image

Low-freq spread ≈ 64/6.4 ≈ 10 pixels (blurry)
High-freq spread ≈ 1-2 pixels (sharp)


        Low-freq "smear"              High-freq "precision"
        ┌──────────┐                  ┌─┐
        │ ≈10 px   │                  │1│
        │  wide    │                  └─┘
        └──────────┘                  
```

### 8.10 Why This Matters for WHAT/WHERE

The spread vs detail property directly enables what/where separation:

```
WHAT PATHWAY (Ventral):              WHERE PATHWAY (Dorsal):
═══════════════════════              ═══════════════════════

Uses: Low frequencies                Uses: High frequencies

Properties:                          Properties:
• Spread out → Position-tolerant     • Localized → Position-specific
• Blurry → Ignores fine details      • Sharp → Captures exact edges
• Slow-varying → Stable features     • Fast-varying → Precise boundaries

Result:                              Result:
"I recognize this as a ring"         "The ring edge is at (23, 45)"


        The WHAT pathway can recognize objects even when
        they're shifted, because low-freq spreads across
        the shift distance.
        
        The WHERE pathway knows exact positions because
        high-freq is localized to the boundary.
```

### 8.11 Boundary Effects + Spread/Detail Combined

Both effects are happening simultaneously:

```
COMPLETE PICTURE:

                    MAGNITUDE                 PHASE
                    ─────────                 ─────
                    
LOW FREQUENCY:      
  - Spread out (inherent)            - Smooth gradient
  - Boundary effect: Minimal         - Encodes position
  - Shows: Overall structure         - Low wrap frequency
  
  Visual: Blurry blob                Visual: Clear color ramp
  
  
HIGH FREQUENCY:
  - Localized/sharp (inherent)       - Rapid gradient  
  - Boundary effect: Cross pattern   - Encodes position
  - Shows: Edges only                - High wrap frequency
  
  Visual: Ring edges + cross         Visual: Checkerboard


COMBINED INTERPRETATION:
────────────────────────

Low-freq magnitude:  WHAT (structure type, position-tolerant)
High-freq magnitude: WHERE (edge locations) + boundary artifacts
Low-freq phase:      WHERE (coarse position)
High-freq phase:     WHERE (fine position, wrapped)
```

### 8.12 Practical Frequency Cutoff Selection

Choosing the right cutoff balances spread vs detail:

```
CUTOFF SELECTION GUIDE:

Cutoff     Low-Freq Spread    High-Freq Detail    Use Case
──────     ───────────────    ────────────────    ────────
0.05       Very spread        Almost everything   Coarse WHAT only
0.10       Spread             Most detail         General WHAT
0.15       Moderate           Good detail         Balanced
0.20       Less spread        Fine detail         Balanced (default)
0.30       Compact            Very fine           More WHERE
0.50       Very compact       Finest              Precise WHERE


Visualization of cutoff effect:

Original ring:   ░░░░░
                ░░   ░░
               ░░     ░░
                ░░   ░░
                 ░░░░░

Cutoff 0.10:    ▒▒▒▒▒▒▒▒▒        │        ·  
               ▒▒▒▒▒▒▒▒▒▒▒       │       · ·
              ▒▒▒▒▒▒▒▒▒▒▒▒▒      │      ·   ·
               ▒▒▒▒▒▒▒▒▒▒▒       │       · ·
                ▒▒▒▒▒▒▒▒▒        │        ·
                Very blurry              Very faint edges

Cutoff 0.20:     ▒▒▒▒▒▒▒          │        ░░░░░
                ▒▒▒▒▒▒▒▒▒        │       ░   ░░
               ▒▒▒▒▒▒▒▒▒▒▒       │      ░     ░
                ▒▒▒▒▒▒▒▒▒        │       ░   ░░
                 ▒▒▒▒▒▒▒         │        ░░░░░
                Moderately blurry         Clear edges

Cutoff 0.35:      ▒▒▒▒▒           │        ░░░░░
                 ▒▒▒▒▒▒▒         │       ░░   ░░
                ▒▒▒▒▒▒▒▒▒        │      ░░     ░░
                 ▒▒▒▒▒▒▒         │       ░░   ░░
                  ▒▒▒▒▒          │        ░░░░░
                Slightly blurry           Strong edges
```

### 8.13 Summary: Two Distinct Effects

```
┌─────────────────────────────────────────────────────────────────┐
│                    TWO EFFECTS IN PLAY                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. INHERENT FREQUENCY PROPERTIES (always present):             │
│     ─────────────────────────────────────────────               │
│     • Low-freq = spread out, blurry, captures structure         │
│     • High-freq = localized, sharp, captures edges              │
│     • This is fundamental signal processing                     │
│                                                                 │
│  2. BOUNDARY EFFECTS (finite domain artifact):                  │
│     ────────────────────────────────────────────                │
│     • Magnitude cross pattern rotates with position             │
│     • Caused by finite frame acting as implicit window          │
│     • Can be mitigated with windowing                           │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  LOW FREQUENCY:                    HIGH FREQUENCY:              │
│  ├─ Spread (inherent)              ├─ Sharp (inherent)          │
│  ├─ Structure (WHAT)               ├─ Edges (WHERE)             │
│  └─ Minimal boundary effect        └─ Rotating cross (boundary) │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 9. Solutions: Windowing

To restore magnitude shift invariance, we can apply **windowing** to smooth the boundaries:

### Hanning Window

```python
def apply_hanning_window(frame):
    H, W = frame.shape
    hanning_h = np.hanning(H)
    hanning_w = np.hanning(W)
    window = np.outer(hanning_h, hanning_w)
    return frame * window
```

Effect:

```
Original frame:              Windowed frame:
┌────────────────────┐      ┌────────────────────┐
│████████████████████│      │                    │
│████████████████████│      │  ████████████████  │
│████  RING  ████████│  →   │  ██  RING  ████  │
│████████████████████│      │  ████████████████  │
│████████████████████│      │                    │
└────────────────────┘      └────────────────────┘
 Sharp edges                  Smooth edges (tapered to zero)
```

### Before vs After Windowing

```
WITHOUT windowing:                 WITH windowing:
─────────────────                  ─────────────────

Ring moves → FFT cross rotates    Ring moves → FFT stays stable

Position A:    Position B:        Position A:    Position B:
    │              ─                   ●              ●
────●────      ────●────          ────●────      ────●────
    │              ─                   ●              ●
    
DIFFERENT!                         SAME! (nearly)
```

### Gaussian Window (Alternative)

```python
def apply_gaussian_window(frame, sigma=0.3):
    H, W = frame.shape
    y = np.linspace(-1, 1, H)
    x = np.linspace(-1, 1, W)
    X, Y = np.meshgrid(x, y)
    window = np.exp(-(X**2 + Y**2) / (2 * sigma**2))
    return frame * window
```

---

## 10. Implications for Spectral Attention

### What This Means for the What/Where Separation

The practical what/where separation works differently depending on context:

```
IDEAL (with windowing or interior patterns):
─────────────────────────────────────────────

Magnitude = WHAT (truly position-invariant)
Phase = WHERE (position-specific)

Clean separation, shift theorem holds.


PRACTICAL (near boundaries, no windowing):
──────────────────────────────────────────

Magnitude ≈ WHAT (approximately invariant)
           + Boundary signature (position-dependent)
           
Phase = WHERE (position-specific)

Partial separation, boundary effects add noise.
```

### Design Recommendations

1. **Use windowing** when computing FFT for matching
2. **Focus on central frequencies** (less affected by boundaries)
3. **Use relative similarity** rather than exact matching
4. **Larger grids** reduce boundary effects proportionally

```
Boundary Effect vs Grid Size:

Grid Size    Boundary Zone    Interior Zone    Effect Severity
─────────    ─────────────    ─────────────    ───────────────
32×32        ~8 pixels        ~16 pixels       High
64×64        ~8 pixels        ~48 pixels       Medium
128×128      ~8 pixels        ~112 pixels      Low
256×256      ~8 pixels        ~240 pixels      Minimal
```

### The Robust Core

Despite boundary effects, the **core insight remains valid**:

```
What magnitude encodes (reliably):
─────────────────────────────────
• Presence of ring structure (central peak pattern)
• Ring radius (concentric rings in FFT)
• Ring thickness (spread of FFT pattern)
• General shape type (ring vs blob vs cross)

What phase encodes (reliably):
─────────────────────────────────
• Exact position (linear phase shift)
• Fine spatial alignment
• Edge locations
```

---

## 11. Mathematical Details

### The Discrete Fourier Transform

For a 2D signal f[m,n] of size M×N:

```
F[k,l] = Σ Σ f[m,n] · exp(-i2π(km/M + ln/N))
         m n
```

### The Shift Theorem (Continuous)

For continuous signals:

```
f(x - x₀, y - y₀) ↔ F(u,v) · exp(-i2π(ux₀ + vy₀))

|F(u,v)| unchanged
arg(F(u,v)) → arg(F(u,v)) - 2π(ux₀ + vy₀)
```

### Why It Breaks in Discrete Finite Domains

The DFT implicitly assumes periodic boundaries:

```
f[m,n] = f[m + M, n] = f[m, n + N] = f[m + M, n + N]
```

When a pattern shifts, it **wraps around**, creating discontinuities:

```
Original:        Shifted right by Δ:
┌──────────┐     ┌──────────┐
│  ○       │     │       ○  │  ← Looks like shift
│          │     │          │
└──────────┘     └──────────┘

But with periodic boundary (what FFT sees):
───────────────────────────────────────────
│  ○       │  ○       │  ○       │  ← Tiled
───────────────────────────────────────────

vs.

───────────────────────────────────────────
│       ○  │       ○  │       ○  │  ← Different tiling!
───────────────────────────────────────────
```

### The Convolution Perspective

The finite frame acts as a rectangular window W[m,n]:

```
What we compute:  FFT(f[m,n] · W[m,n])
                = FFT(f) ∗ FFT(W)   (convolution)
                
FFT(W) = sinc function with sidelobes

When f shifts, f·W changes (different parts of f are windowed)
So FFT(f·W) changes!
```

### Quantifying the Effect

The boundary effect magnitude scales as:

```
Boundary Effect ∝ (Pattern Size / Frame Size) × (Distance to Edge)⁻¹

For a ring of radius r in a frame of size L:
- Interior: Effect ≈ 0 when distance to edge > 2r
- Boundary: Effect ≈ O(r/L) when touching edge
```

---

## Summary

### Magnitude (WHAT) Behavior

| Aspect | Theory | Practice |
|--------|--------|----------|
| Shift invariance | Perfect | Approximate (boundary effects) |
| Cross pattern | Should not exist | Rotates with position |
| Central peak | Encodes structure | Reliable, stable |
| Solution | N/A | Windowing, larger grids |

### Phase (WHERE) Behavior

| Aspect | Theory | Practice |
|--------|--------|----------|
| Position encoding | Linear phase ramp | Works exactly as predicted |
| Gradient direction | Points to pattern | Reliable position signal |
| Visual appearance | Smooth gradient | Alternates smooth/checkerboard |
| Checkerboard cause | N/A | Phase wrapping at high frequencies |

### Combined Understanding

```
WHAT (Magnitude):
├── Core structure (central peak) → Reliable
├── Boundary effects (cross pattern) → Position-dependent artifact
└── Solution: Windowing or focus on low frequencies

WHERE (Phase):
├── Gradient direction → Encodes position angle
├── Gradient magnitude → Encodes position distance
├── Low frequencies → Smooth, easy to read
└── High frequencies → Checkerboard (wrapping), still valid
```

### The Key Insights

1. **Magnitude oscillation is a boundary artifact** - In finite domains, the FFT sees different effective structures as patterns move, causing the cross pattern to rotate.

2. **Phase faithfully encodes position** - Despite magnitude issues, phase gradient reliably points to pattern location.

3. **Checkerboard is phase wrapping, not noise** - The alternating smooth/checkerboard appearance is due to phase wrapping at high frequencies when the gradient is diagonal.

4. **Both signals remain useful** - Magnitude still encodes structure type (ring vs blob), phase still encodes exact position. The what/where separation holds, with understood limitations.

The fundamental truth: **The shift theorem is mathematically exact in infinite continuous domains. In finite discrete domains, magnitude shows boundary artifacts while phase remains reliable. Understanding both behaviors is essential for robust spectral attention systems.**

---

## References

- Oppenheim, A. V., & Schafer, R. W. (2009). Discrete-Time Signal Processing
- Bracewell, R. N. (2000). The Fourier Transform and Its Applications
- Harris, F. J. (1978). On the Use of Windows for Harmonic Analysis with the DFT

---

*Document created for the Spectral Attention project. The boundary effects observed in the What/Where Lab demonstrate these principles in action.*

