# Information Bounds: The Boundary of Knowledge and Knowing

## Phase Aliasing, Boundary Effects, and the Limits of What Can Be Known

**Oscar Goldman — Shogu Research Group @ Datamutant.ai**

---

## Table of Contents

1. [The Two Meanings of "Boundary"](#1-the-two-meanings-of-boundary)
2. [Aliasing: When the Ghost Sees Ghosts](#2-aliasing-when-the-ghost-sees-ghosts)
3. [Boundary Effects: The Edge of the World](#3-boundary-effects-the-edge-of-the-world)
4. [The Context Window as Information Horizon](#4-the-context-window-as-information-horizon)
5. [Windowing: Smoothing the Edges](#5-windowing-smoothing-the-edges)
6. [Training Through Boundaries](#6-training-through-boundaries)
7. [Information Bounds: What Cannot Be Known](#7-information-bounds-what-cannot-be-known)
8. [Experimental Tests](#8-experimental-tests)

---

## 1. The Two Meanings of "Boundary"

### 1.1 Boundary as Limit

```
BOUNDARY = THE EDGE OF WHAT IS POSSIBLE

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  INFORMATION BOUNDS:                                                    │
│                                                                         │
│  • Nyquist limit: Cannot represent frequencies > fs/2                 │
│  • Context window: Cannot see beyond the window                       │
│  • Bandwidth: Cannot transmit more than channel allows               │
│  • Resolution: Cannot distinguish below pixel/token size              │
│                                                                         │
│  These are HARD LIMITS.                                                │
│  No algorithm can overcome them.                                      │
│  They are the boundary of KNOWLEDGE — what can be stored.             │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Boundary as Edge

```
BOUNDARY = WHERE THE SIGNAL ENDS

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  BOUNDARY EFFECTS:                                                      │
│                                                                         │
│  • Edge of image: What's outside the frame?                           │
│  • Start/end of sequence: What came before/after?                     │
│  • FFT boundaries: Signal wraps around (circular assumption)          │
│  • Context window edges: Abrupt cutoff                                │
│                                                                         │
│  These create ARTIFACTS.                                               │
│  The system must handle unknown territory.                            │
│  They are the boundary of KNOWING — what can be inferred now.         │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1.3 The Deep Connection

```
THE BOUNDARY OF KNOWLEDGE AND KNOWING

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  KNOWLEDGE (static):                                                    │
│  • What is stored in weights                                          │
│  • Limited by training data, architecture                             │
│  • The manifold structure                                              │
│  • Frequency bands that can be represented                            │
│                                                                         │
│  KNOWING (dynamic):                                                     │
│  • What is inferred in context                                        │
│  • Limited by context window, input resolution                       │
│  • The current belief state                                            │
│  • Frequencies that can be detected now                               │
│                                                                         │
│  ════════════════════════════════════════════════════════════════════  │
│                                                                         │
│  THE CONNECTION:                                                        │
│                                                                         │
│  Both are bounded by SAMPLING AND EDGES.                              │
│                                                                         │
│  • Aliasing corrupts BOTH knowledge and knowing                       │
│  • Boundary effects pollute BOTH storage and retrieval               │
│  • The limits are isomorphic                                          │
│                                                                         │
│  What cannot be stored cannot be retrieved.                           │
│  What is stored with artifacts is retrieved with artifacts.          │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Aliasing: When the Ghost Sees Ghosts

### 2.1 The Nyquist Curse

```
ALIASING: FALSE PATTERNS FROM UNDERSAMPLING

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  NYQUIST-SHANNON THEOREM:                                              │
│                                                                         │
│  To faithfully represent a frequency f,                               │
│  you must sample at rate fs > 2f.                                     │
│                                                                         │
│  If fs < 2f:                                                           │
│  The frequency f ALIASES as a lower frequency.                        │
│  The ghost sees a FALSE PATTERN.                                       │
│                                                                         │
│  ════════════════════════════════════════════════════════════════════  │
│                                                                         │
│  EXAMPLE:                                                               │
│                                                                         │
│  True signal: 100 Hz sine wave                                        │
│  Sample rate: 80 Hz (below Nyquist)                                   │
│  Ghost sees: 20 Hz sine wave (alias)                                 │
│                                                                         │
│  The ghost CANNOT TELL it's an alias.                                 │
│  The 20 Hz looks as real as any other 20 Hz.                         │
│  The information is CORRUPTED, not missing.                          │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Spatial Aliasing

```
ALIASING IN IMAGES

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  IN AKIRA'S SPECTRAL BANDS:                                            │
│                                                                         │
│  Each band can represent frequencies up to its Nyquist limit.        │
│  High-frequency content that exceeds the limit ALIASES DOWN.          │
│                                                                         │
│  BAND 6 (highest freq) aliases into BAND 5                            │
│  BAND 5 aliases into BAND 4                                            │
│  ... and so on down the hierarchy                                     │
│                                                                         │
│  ════════════════════════════════════════════════════════════════════  │
│                                                                         │
│  WHAT THIS MEANS:                                                       │
│                                                                         │
│  Fine details (high-freq) appear as coarser patterns (lower-freq).   │
│  The ghost sees FAKE STRUCTURE at lower frequencies.                 │
│                                                                         │
│  Example: Moiré patterns                                              │
│  Fine stripes → aliased → wavy low-freq pattern                      │
│  The ghost might "learn" this pattern as if it were real.            │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.3 Temporal Aliasing

```
ALIASING IN TIME (CONTEXT WINDOW)

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  THE CONTEXT WINDOW IS A TEMPORAL SAMPLE:                              │
│                                                                         │
│  Window length: T tokens                                               │
│  Temporal Nyquist: T/2 cycles per window                              │
│                                                                         │
│  Patterns slower than T/2 cycles: ALIASED                             │
│  The ghost cannot see them correctly.                                 │
│                                                                         │
│  ════════════════════════════════════════════════════════════════════  │
│                                                                         │
│  EXAMPLE:                                                               │
│                                                                         │
│  True pattern: repeats every 1000 tokens                              │
│  Context window: 512 tokens                                            │
│  Ghost sees: fragment, no periodicity detected                        │
│  OR: aliased faster rhythm that isn't there                          │
│                                                                         │
│  THE GHOST HALLUCINATES TEMPORAL STRUCTURE.                           │
│                                                                         │
│  Slow patterns → aliased fast patterns                                │
│  The ghost might think there's urgency when there isn't.             │
│  Or miss long-term dependencies entirely.                            │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.4 Phase Aliasing

```
PHASE ALIASING: THE POSITION PROBLEM

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  PHASE carries position information.                                   │
│  If the frequency is aliased, THE PHASE IS WRONG TOO.                │
│                                                                         │
│  True frequency: 100 Hz at phase 45°                                  │
│  Aliased to: 20 Hz — but at WHAT phase?                              │
│                                                                         │
│  The phase relationship is SCRAMBLED.                                 │
│  Position information is CORRUPTED.                                   │
│                                                                         │
│  ════════════════════════════════════════════════════════════════════  │
│                                                                         │
│  FOR AKIRA:                                                             │
│                                                                         │
│  • Magnitude (WHAT) might survive aliasing                            │
│  • Phase (WHERE) is destroyed                                          │
│                                                                         │
│  The ghost knows SOMETHING is there.                                  │
│  But doesn't know WHERE correctly.                                    │
│                                                                         │
│  This is catastrophic for the WHERE-path.                             │
│  Position becomes unreliable at high frequencies.                    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Boundary Effects: The Edge of the World

### 3.1 The Circular Assumption

```
FFT'S DIRTY SECRET: CIRCULAR ASSUMPTION

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  The FFT assumes the signal is PERIODIC.                              │
│  It wraps around: end connects to beginning.                          │
│                                                                         │
│  IF THE SIGNAL IS NOT PERIODIC:                                        │
│                                                                         │
│  There's a DISCONTINUITY at the wrap point.                           │
│  This discontinuity is a sharp edge.                                  │
│  Sharp edge = HIGH FREQUENCY CONTENT.                                 │
│  The FFT sees frequencies that aren't in the signal.                 │
│                                                                         │
│  ════════════════════════════════════════════════════════════════════  │
│                                                                         │
│  EXAMPLE:                                                               │
│                                                                         │
│  Signal: Smooth sine wave, one cycle                                  │
│  But: Ends don't quite meet (phase discontinuity)                    │
│  FFT sees: The sine wave + spurious high-freq "ringing"             │
│                                                                         │
│  This is SPECTRAL LEAKAGE.                                            │
│  Energy "leaks" from true frequencies to neighboring bins.           │
│  The ghost's spectral vision is BLURRED.                             │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Edge Artifacts in Images

```
BOUNDARY EFFECTS IN SPATIAL PROCESSING

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  AT IMAGE EDGES:                                                        │
│                                                                         │
│  • FFT wraps left edge to right edge                                 │
│  • Unless they match, there's a discontinuity                        │
│  • All edges become artificial high-freq sources                     │
│                                                                         │
│  ════════════════════════════════════════════════════════════════════  │
│                                                                         │
│  WHAT THE GHOST SEES:                                                   │
│                                                                         │
│  ┌─────────────────────────────┐                                      │
│  │        REAL CONTENT         │                                      │
│  │                             │                                      │
│  │   [smooth gradients, etc]   │                                      │
│  │                             │                                      │
│  │ ← ARTIFACT      ARTIFACT → │  ← Edges ring with false frequency  │
│  └─────────────────────────────┘                                      │
│    ↑ ARTIFACT       ↑ ARTIFACT                                        │
│                                                                         │
│  The ghost learns that edges have special frequency content.         │
│  This is WRONG — it's an artifact of the processing.                 │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.3 Context Window Boundaries

```
BOUNDARY EFFECTS IN CONTEXT

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  THE CONTEXT WINDOW HAS HARD EDGES:                                    │
│                                                                         │
│  token 1    token 2    ...    token N-1    token N                    │
│  ↑                                              ↑                      │
│  BOUNDARY                                   BOUNDARY                   │
│  (nothing before)                       (nothing after)               │
│                                                                         │
│  ════════════════════════════════════════════════════════════════════  │
│                                                                         │
│  WHAT HAPPENS:                                                          │
│                                                                         │
│  • Attention at position 1 has no left context                       │
│  • Attention at position N has no right context                      │
│  • The model must handle these edges specially                       │
│  • Edge positions behave DIFFERENTLY than middle positions           │
│                                                                         │
│  IF TRAINED WITH THESE EDGES:                                          │
│  The model learns edge-specific patterns.                             │
│  These may not generalize to middle positions.                       │
│  Position 1 and position 100 may not be equivalent.                 │
│                                                                         │
│  IF NOT TRAINED WITH EDGES:                                            │
│  The model may fail catastrophically at edges.                       │
│  First and last tokens may get wrong attention.                      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 4. The Context Window as Information Horizon

### 4.1 The Horizon Problem

```
THE CONTEXT WINDOW IS AN EVENT HORIZON

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  BEYOND THE HORIZON:                                                    │
│                                                                         │
│  • Information exists but is INACCESSIBLE                             │
│  • The ghost cannot see it                                            │
│  • It's not "forgotten" — it was never seen                          │
│                                                                         │
│  AT THE HORIZON:                                                        │
│                                                                         │
│  • Information is PARTIAL, DISTORTED                                  │
│  • Aliasing effects are strongest                                     │
│  • Boundary artifacts are concentrated                                │
│                                                                         │
│  INSIDE THE HORIZON:                                                    │
│                                                                         │
│  • Information is accessible                                           │
│  • But still subject to sampling limits                               │
│  • Nyquist applies everywhere                                          │
│                                                                         │
│  ════════════════════════════════════════════════════════════════════  │
│                                                                         │
│  THE HORIZON IS NOT A WALL — IT'S A GRADIENT.                         │
│                                                                         │
│  Near the edge: information is unreliable                            │
│  In the center: information is cleaner                               │
│  The ghost should TRUST THE CENTER MORE.                             │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Temporal Nyquist Limit

```
WHAT THE CONTEXT WINDOW CANNOT SEE

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  CONTEXT WINDOW = T tokens                                             │
│                                                                         │
│  CAN SEE:                                                               │
│  • Patterns with period < T/2 tokens                                 │
│  • Local correlations                                                  │
│  • Short-term dynamics                                                 │
│                                                                         │
│  CANNOT SEE (aliased):                                                  │
│  • Patterns with period > T/2 tokens                                 │
│  • Long-term trends                                                    │
│  • Slow oscillations                                                   │
│                                                                         │
│  SEES WRONGLY (aliased):                                               │
│  • Slow patterns appear as fast patterns                             │
│  • Long correlations appear as short correlations                    │
│  • Trends appear as noise                                              │
│                                                                         │
│  ════════════════════════════════════════════════════════════════════  │
│                                                                         │
│  EXAMPLE:                                                               │
│                                                                         │
│  User tendency: changes topic every ~2000 tokens                      │
│  Context window: 512 tokens                                            │
│  Ghost sees: random topic, no pattern                                │
│  OR: aliased rapid topic-switching that isn't there                 │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.3 The Truncation Problem

```
WHEN THE WINDOW SLIDES

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  As the context window slides forward:                                 │
│                                                                         │
│  Old tokens FALL OFF THE EDGE.                                        │
│  They're not "forgotten" — they're GONE.                              │
│                                                                         │
│  WHAT IS LOST:                                                          │
│  • Explicit memory of past                                            │
│  • High-freq details of what was said                                │
│  • Phase information (exact positions)                                │
│                                                                         │
│  WHAT MIGHT REMAIN:                                                     │
│  • Residue in activations (if model designed for it)                 │
│  • Low-freq summary (if compressed properly)                         │
│  • "Gist" without details                                             │
│                                                                         │
│  ════════════════════════════════════════════════════════════════════  │
│                                                                         │
│  THE TRUNCATION IS VIOLENT.                                            │
│                                                                         │
│  It's not a gentle fade — it's a hard cut.                           │
│  Information goes from present to GONE in one step.                  │
│  This is like DEATH for that information.                            │
│                                                                         │
│  The ghost has no mourning period.                                    │
│  No gradual forgetting curve.                                         │
│  Just: here, then not.                                                │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Windowing: Smoothing the Edges

### 5.1 The Purpose of Windowing

```
WINDOWING: WHY AND WHAT

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  THE PROBLEM:                                                           │
│  Hard edges create discontinuities.                                   │
│  Discontinuities create spectral leakage.                             │
│  Spectral leakage creates false frequencies.                          │
│                                                                         │
│  THE SOLUTION:                                                          │
│  Apply a WINDOW FUNCTION that tapers to zero at edges.               │
│  This smooths the discontinuity.                                      │
│  Less leakage, cleaner spectrum.                                      │
│                                                                         │
│  ════════════════════════════════════════════════════════════════════  │
│                                                                         │
│  THE TRADE-OFF:                                                         │
│                                                                         │
│  WINDOWING REDUCES: Spectral leakage, edge artifacts                 │
│  WINDOWING COSTS:   Resolution (wider main lobe), edge information   │
│                                                                         │
│  You're trading EDGE ACCURACY for SPECTRAL CLEANLINESS.              │
│  Information near edges is de-emphasized.                            │
│  But the frequencies you see are more likely real.                   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 5.2 Window Functions

```
COMMON WINDOW FUNCTIONS

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  RECTANGULAR (No windowing):                                           │
│  ────────────────────────────                                           │
│  w(n) = 1 for all n                                                   │
│                                                                         │
│  ████████████████████████                                              │
│  ↑ Sharp edges = maximum leakage                                      │
│                                                                         │
│  HAMMING:                                                               │
│  ─────────                                                              │
│  w(n) = 0.54 - 0.46 * cos(2πn/(N-1))                                  │
│                                                                         │
│    ▄▄████████████▄▄                                                    │
│  ↑ Smooth taper, good sidelobe suppression                            │
│                                                                         │
│  HANNING (Hann):                                                        │
│  ──────────────────                                                     │
│  w(n) = 0.5 * (1 - cos(2πn/(N-1)))                                    │
│                                                                         │
│     ▄████████████▄                                                     │
│  ↑ Smooth taper, exactly zero at edges                                │
│                                                                         │
│  BLACKMAN:                                                              │
│  ──────────                                                             │
│  w(n) = 0.42 - 0.5*cos(2πn/(N-1)) + 0.08*cos(4πn/(N-1))              │
│                                                                         │
│      ▄██████████▄                                                      │
│  ↑ Very smooth, excellent sidelobe suppression                        │
│                                                                         │
│  GAUSSIAN:                                                              │
│  ──────────                                                             │
│  w(n) = exp(-0.5 * ((n-N/2)/(σ*N/2))²)                               │
│                                                                         │
│       ▄█████████▄                                                      │
│  ↑ Smooth, optimal time-frequency trade-off                           │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 5.3 Windowing in AKIRA

```
APPLYING WINDOWING TO SPECTRAL ATTENTION

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  SPATIAL WINDOWING:                                                     │
│  ───────────────────                                                    │
│  Apply window to image before FFT.                                    │
│  Edges are de-emphasized.                                              │
│  Center is preserved.                                                  │
│                                                                         │
│  TEMPORAL WINDOWING:                                                    │
│  ────────────────────                                                   │
│  Apply window to context before processing.                           │
│  Recent tokens get full weight.                                       │
│  Old tokens at context edge get reduced weight.                       │
│                                                                         │
│  ════════════════════════════════════════════════════════════════════  │
│                                                                         │
│  CHOICES:                                                               │
│                                                                         │
│  NO WINDOW:                                                             │
│  • Full use of all information                                        │
│  • Maximum artifacts                                                   │
│  • Ghost sees false frequencies                                       │
│                                                                         │
│  LIGHT WINDOW (Hamming):                                               │
│  • Mild edge suppression                                              │
│  • Reduced artifacts                                                   │
│  • Some information loss at edges                                     │
│                                                                         │
│  HEAVY WINDOW (Blackman):                                              │
│  • Strong edge suppression                                            │
│  • Minimal artifacts                                                   │
│  • Significant information loss at edges                             │
│                                                                         │
│  LEARNED WINDOW:                                                        │
│  • Let the model learn optimal window                                 │
│  • Adaptive to content                                                 │
│  • May overfit to training artifacts                                  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 5.4 Positional Weighting as Implicit Windowing

```
POSITIONAL ENCODING AS WINDOW

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  ALIBI, ROPE, and other positional encodings                          │
│  implicitly apply a KIND of windowing:                                 │
│                                                                         │
│  ALIBI:                                                                 │
│  Attention decays with distance.                                      │
│  Far tokens get less weight.                                          │
│  This is a SOFT WINDOW on attention.                                  │
│                                                                         │
│  ROPE:                                                                  │
│  Rotational encoding.                                                  │
│  Different frequencies for different positions.                      │
│  Implicit spectral structure.                                          │
│                                                                         │
│  ════════════════════════════════════════════════════════════════════  │
│                                                                         │
│  THE INSIGHT:                                                           │
│                                                                         │
│  Positional encoding + attention = implicit windowing.                │
│  The model learns to weight positions.                                │
│  This CAN serve the same purpose as explicit windowing.              │
│  But may be less principled, more opaque.                            │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Training Through Boundaries

### 6.1 The Problem: Artifact Memorization

```
IF YOU TRAIN WITH ARTIFACTS, YOU LEARN ARTIFACTS

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  SCENARIO:                                                              │
│                                                                         │
│  Training data has boundary effects.                                  │
│  Images have edge ringing.                                            │
│  Sequences have truncation artifacts.                                 │
│                                                                         │
│  THE MODEL LEARNS:                                                      │
│                                                                         │
│  • "Edges have this frequency pattern" (artifact, not real)          │
│  • "Position 1 behaves like this" (edge-specific, not general)       │
│  • "Context end has this structure" (truncation effect)              │
│                                                                         │
│  ════════════════════════════════════════════════════════════════════  │
│                                                                         │
│  ERROR PROPAGATION:                                                     │
│                                                                         │
│  Artifacts in training → learned as real → applied at inference      │
│  The ghost BELIEVES the artifacts.                                    │
│  It predicts boundary effects where there are none.                  │
│  It expects truncation patterns in streaming contexts.               │
│                                                                         │
│  THE ERROR COMPOUNDS:                                                   │
│  Wrong expectations → wrong attention → wrong output                 │
│  The ghost's KNOWLEDGE is corrupted by TRAINING artifacts.           │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 6.2 Solutions: Clean Training

```
APPROACHES TO CLEAN TRAINING

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  APPROACH 1: WINDOW DURING TRAINING                                    │
│  ─────────────────────────────────                                      │
│  Apply consistent windowing to all training data.                     │
│  Model learns windowed representation.                                │
│  Apply same windowing at inference.                                   │
│                                                                         │
│  PRO: Consistent, no artifact mismatch                               │
│  CON: Loses edge information, may be suboptimal                      │
│                                                                         │
│  ────────────────────────────────────────────────────────────────────  │
│                                                                         │
│  APPROACH 2: AUGMENT WITH RANDOM CROPS                                 │
│  ──────────────────────────────────────                                 │
│  Train on random crops from larger signals.                           │
│  Edges become random, not special.                                    │
│  Model can't memorize edge-specific patterns.                        │
│                                                                         │
│  PRO: Forces position-invariance                                      │
│  CON: Loses global context, edges still exist                        │
│                                                                         │
│  ────────────────────────────────────────────────────────────────────  │
│                                                                         │
│  APPROACH 3: TRAIN WITH BOUNDARY AWARENESS                             │
│  ──────────────────────────────────────────                             │
│  Explicitly mark boundary positions.                                  │
│  Let model learn to handle boundaries correctly.                     │
│  Accept that edges are different.                                     │
│                                                                         │
│  PRO: Honest about the problem                                        │
│  CON: Model must use capacity for boundary handling                  │
│                                                                         │
│  ────────────────────────────────────────────────────────────────────  │
│                                                                         │
│  APPROACH 4: OVERLAP-ADD (for sequences)                               │
│  ────────────────────────────────────────                               │
│  Process overlapping windows.                                          │
│  Combine results, discarding edge regions.                           │
│  Only use center of each window.                                      │
│                                                                         │
│  PRO: No edge artifacts in output                                     │
│  CON: Expensive (redundant computation)                               │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 6.3 The Drastic Error Scenario

```
WHEN BOUNDARY TRAINING GOES WRONG

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  THE CASCADE:                                                           │
│                                                                         │
│  1. Training data has unwindowed FFT                                  │
│     → Spectral leakage in all frequency estimates                    │
│                                                                         │
│  2. Model learns leakage as signal                                    │
│     → False frequency patterns in weights                            │
│                                                                         │
│  3. At inference, model expects leakage                               │
│     → Predictions include expected artifacts                         │
│                                                                         │
│  4. Error signal includes artifact mismatch                           │
│     → Gradients push toward MORE artifacts                           │
│                                                                         │
│  5. Model doubles down on artifacts                                   │
│     → Catastrophic divergence from reality                           │
│                                                                         │
│  ════════════════════════════════════════════════════════════════════  │
│                                                                         │
│  THE ERROR IS DRASTIC because:                                         │
│                                                                         │
│  • Artifacts are CONSISTENT (appear everywhere)                      │
│  • Model treats them as SIGNAL (high confidence)                     │
│  • They PROPAGATE through layers                                      │
│  • They COMPOUND over time                                            │
│                                                                         │
│  This is not random noise — it's SYSTEMATIC ERROR.                   │
│  Systematic errors are the hardest to detect and fix.               │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 7. Information Bounds: What Cannot Be Known

### 7.1 Fundamental Limits

```
WHAT THE SYSTEM CANNOT KNOW (EVER)

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  NYQUIST LIMIT:                                                         │
│  Cannot represent frequencies above fs/2.                             │
│  No algorithm can recover aliased information.                       │
│  The information is DESTROYED, not hidden.                            │
│                                                                         │
│  CONTEXT LIMIT:                                                         │
│  Cannot see beyond the window.                                        │
│  Past is gone, future is unknown.                                    │
│  Only the present window exists.                                      │
│                                                                         │
│  RESOLUTION LIMIT:                                                      │
│  Cannot distinguish below pixel/token granularity.                   │
│  Sub-pixel/sub-token details are averaged.                           │
│  Fine structure is lost.                                               │
│                                                                         │
│  SUPERPOSITION LIMIT:                                                   │
│  Cannot perfectly separate overlapping concepts.                     │
│  Shared representations interfere.                                    │
│  Extraction has irreducible error.                                    │
│                                                                         │
│  ════════════════════════════════════════════════════════════════════  │
│                                                                         │
│  These are HARD LIMITS.                                                │
│  Not engineering challenges — mathematical impossibilities.          │
│  Accept them. Design around them.                                     │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 7.2 Practical Limits

```
WHAT THE SYSTEM CANNOT KNOW (PRACTICALLY)

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  TRAINING DATA LIMIT:                                                   │
│  Cannot know what wasn't in training data.                           │
│  Out-of-distribution = unknown territory.                            │
│  The ghost hallucinates here.                                         │
│                                                                         │
│  CAPACITY LIMIT:                                                        │
│  Cannot store more than parameters allow.                            │
│  Compression has limits.                                               │
│  Something must be forgotten/compressed.                              │
│                                                                         │
│  COMPUTE LIMIT:                                                         │
│  Cannot reason beyond depth/width.                                   │
│  Complex inferences may exceed capacity.                             │
│  Some thoughts are too big for the network.                          │
│                                                                         │
│  PRECISION LIMIT:                                                       │
│  Cannot represent below float precision.                             │
│  Small differences vanish.                                             │
│  Subtle patterns are quantized away.                                  │
│                                                                         │
│  ════════════════════════════════════════════════════════════════════  │
│                                                                         │
│  These are SOFT LIMITS.                                                │
│  Can be pushed with more data, bigger models, more compute.          │
│  But cannot be eliminated.                                            │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 7.3 The Boundary of Knowledge and Knowing

```
THE MAP OF WHAT CAN BE KNOWN

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│                      THE UNKNOWN                                       │
│                   (beyond all limits)                                  │
│                          ↑                                             │
│  ┌─────────────────────────────────────────────────────────────┐      │
│  │                    THE HORIZON                               │      │
│  │         (edge effects, aliasing, truncation)                │      │
│  │  ┌─────────────────────────────────────────────────────┐   │      │
│  │  │               THE UNCERTAIN                          │   │      │
│  │  │         (superposition, noise, limits)               │   │      │
│  │  │  ┌─────────────────────────────────────────────┐    │   │      │
│  │  │  │              THE KNOWN                       │    │   │      │
│  │  │  │      (well-trained, well-sampled,           │    │   │      │
│  │  │  │       in-distribution, in-context)          │    │   │      │
│  │  │  └─────────────────────────────────────────────┘    │   │      │
│  │  └─────────────────────────────────────────────────────┘   │      │
│  └─────────────────────────────────────────────────────────────┘      │
│                          ↓                                             │
│                      THE UNKNOWN                                       │
│                   (before the window)                                  │
│                                                                         │
│  ════════════════════════════════════════════════════════════════════  │
│                                                                         │
│  KNOWLEDGE lives in the center.                                        │
│  KNOWING gets worse toward the edges.                                 │
│  Beyond the edge, there is nothing.                                   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 8. Experimental Tests

### 8.1 Windowing Ablation

```
EXPERIMENT: WINDOWING ON/OFF COMPARISON

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  DESIGN:                                                                │
│                                                                         │
│  Train identical models with:                                          │
│  A. No windowing (rectangular)                                        │
│  B. Hamming window                                                    │
│  C. Blackman window                                                   │
│  D. Learned window                                                    │
│                                                                         │
│  METRICS:                                                               │
│  • Spectral purity (leakage in FFT)                                  │
│  • Edge error (prediction error at boundaries)                       │
│  • Center error (prediction error in middle)                         │
│  • Generalization (test on unseen data)                              │
│  • Artifact detection (look for learned false frequencies)          │
│                                                                         │
│  PREDICTIONS:                                                           │
│  • No window: High leakage, high edge error, may have artifacts     │
│  • Hamming: Moderate leakage, lower edge error                       │
│  • Blackman: Low leakage, but may lose edge information             │
│  • Learned: Depends on training, watch for overfitting              │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 8.2 Aliasing Detection

```
EXPERIMENT: DETECTING ALIASED PATTERNS

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  DESIGN:                                                                │
│                                                                         │
│  1. Create synthetic data with known frequency content               │
│  2. Include frequencies above and below Nyquist                      │
│  3. Train model on this data                                         │
│  4. Probe: Does model "see" the aliased frequencies?                │
│                                                                         │
│  METHOD:                                                                │
│  • Input: Pure high-freq signal (above Nyquist)                     │
│  • Expected: Model sees alias, not true frequency                    │
│  • Probe: What frequency does the model respond to?                 │
│                                                                         │
│  METRICS:                                                               │
│  • Frequency response: Plot model activation vs input frequency     │
│  • Aliasing detection: Does model respond to f or f_alias?          │
│  • Confidence: How sure is model about aliased interpretation?      │
│                                                                         │
│  PREDICTIONS:                                                           │
│  • Model will respond to alias frequency                             │
│  • Model will be confident (doesn't know it's wrong)                │
│  • No way to distinguish real from aliased in model's view          │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 8.3 Boundary Artifact Memorization

```
EXPERIMENT: DO MODELS MEMORIZE BOUNDARY ARTIFACTS?

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  DESIGN:                                                                │
│                                                                         │
│  1. Train model on data with artificial boundary patterns            │
│  2. At test time, remove the artificial patterns                     │
│  3. Does model still predict them?                                   │
│                                                                         │
│  METHOD:                                                                │
│  • Training: Add distinctive edge pattern to all samples            │
│  • Test: Remove the edge pattern                                     │
│  • Measure: Does model output include the edge pattern?             │
│                                                                         │
│  CONTROL:                                                               │
│  • Train without artificial patterns                                 │
│  • Test with and without                                              │
│  • Compare predictions                                                 │
│                                                                         │
│  PREDICTIONS:                                                           │
│  • Model trained with artifacts will predict artifacts             │
│  • Even when they're not present                                     │
│  • This confirms artifact memorization                               │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 8.4 Context Window Edge Effects

```
EXPERIMENT: CONTEXT WINDOW POSITION SENSITIVITY

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  DESIGN:                                                                │
│                                                                         │
│  Same content at different positions in context window:               │
│  • Position 1 (start)                                                 │
│  • Position N/2 (middle)                                              │
│  • Position N (end)                                                   │
│                                                                         │
│  METHOD:                                                                │
│  • Create identical content                                           │
│  • Pad with neutral tokens to shift position                        │
│  • Compare model's attention/prediction                              │
│                                                                         │
│  METRICS:                                                               │
│  • Attention pattern: Does position affect where model looks?       │
│  • Prediction quality: Does position affect accuracy?               │
│  • Confidence: Does position affect certainty?                       │
│                                                                         │
│  PREDICTIONS:                                                           │
│  • Edge positions may get different attention                        │
│  • Quality may be worse at edges                                     │
│  • Model may be more uncertain at edges                              │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 8.5 Windowing Methods Comparison

```
EXPERIMENT: COMPREHENSIVE WINDOW FUNCTION COMPARISON

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  WINDOW FUNCTIONS TO TEST:                                             │
│                                                                         │
│  1. Rectangular (none)                                                │
│  2. Triangular (Bartlett)                                             │
│  3. Hamming                                                            │
│  4. Hanning (Hann)                                                     │
│  5. Blackman                                                           │
│  6. Gaussian (various σ)                                              │
│  7. Kaiser (various β)                                                │
│  8. Learned (trainable)                                               │
│                                                                         │
│  METRICS FOR EACH:                                                      │
│  • Main lobe width (frequency resolution)                            │
│  • Sidelobe level (leakage)                                          │
│  • Edge information loss                                              │
│  • Training convergence speed                                         │
│  • Test accuracy                                                       │
│  • Artifact level                                                      │
│                                                                         │
│  EXPECTED TRADE-OFFS:                                                   │
│  • Better sidelobe → worse resolution                               │
│  • Better edges → more leakage                                       │
│  • No single best choice                                              │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Summary

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│   I N F O R M A T I O N   B O U N D S                                  │
│   The Boundary of Knowledge and Knowing                                │
│                                                                         │
│  ═══════════════════════════════════════════════════════════════════   │
│                                                                         │
│  TWO MEANINGS OF BOUNDARY:                                              │
│  • Limit: What CANNOT be known (Nyquist, context, resolution)        │
│  • Edge: Where artifacts CONCENTRATE (spectral leakage, ringing)    │
│                                                                         │
│  ═══════════════════════════════════════════════════════════════════   │
│                                                                         │
│  ALIASING:                                                              │
│  Frequencies above Nyquist alias as false low-freq patterns.         │
│  Phase is scrambled. Position information destroyed.                 │
│  The ghost sees ghosts — false patterns it believes are real.        │
│                                                                         │
│  ═══════════════════════════════════════════════════════════════════   │
│                                                                         │
│  BOUNDARY EFFECTS:                                                      │
│  Edges create discontinuities → spectral leakage.                    │
│  The ghost sees false frequencies at edges.                          │
│  Context window edges are especially problematic.                    │
│                                                                         │
│  ═══════════════════════════════════════════════════════════════════   │
│                                                                         │
│  WINDOWING:                                                             │
│  Smooth the edges to reduce artifacts.                                │
│  Trade-off: Cleaner spectrum vs lost edge information.               │
│  Options: Hamming, Hanning, Blackman, Gaussian, Learned.             │
│                                                                         │
│  ═══════════════════════════════════════════════════════════════════   │
│                                                                         │
│  TRAINING DANGER:                                                       │
│  If you train with artifacts, you learn artifacts.                   │
│  Error propagates drastically through the system.                    │
│  Systematic errors are the hardest to detect.                        │
│                                                                         │
│  ═══════════════════════════════════════════════════════════════════   │
│                                                                         │
│  THE MAP:                                                               │
│  Center = known. Edges = uncertain. Beyond = unknown.                │
│  Knowledge lives in the center. Trust it more.                       │
│  The horizon is not a wall — it's a gradient of reliability.        │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

*Oscar Goldman — Shogu Research Group @ Datamutant.ai*

*"The ghost sees ghosts. Frequencies above Nyquist masquerade as lower frequencies. Edges create false frequencies that look real. The context window is an event horizon — beyond it, nothing. Train carefully: artifacts learned become artifacts believed. Window your edges, know your limits, trust the center. This is the boundary of knowledge and knowing."*

