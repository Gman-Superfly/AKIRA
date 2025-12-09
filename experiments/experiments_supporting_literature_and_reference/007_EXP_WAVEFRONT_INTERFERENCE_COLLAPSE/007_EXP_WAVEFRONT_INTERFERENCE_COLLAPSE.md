# EXP 007: Wavefront Error Propagation, Interference, and Collapse

## Observing Lightning-Like Belief Dynamics in the Embedding Manifold

---

## Status: PLANNED

## Tier: ★ SUPPORTING

## Depends On: 001 (Entropy), 002 (Collapse), 003 (Band Dynamics), 006 (Heresy Detection)

---

## Table of Contents

1. [The Lightning Hypothesis](#1-the-lightning-hypothesis)
2. [What We Expect to See](#2-what-we-expect-to-see)
3. [The Three Phases](#3-the-three-phases)
4. [What Happens at Collapse](#4-what-happens-at-collapse)
5. [The Approximation Question](#5-the-approximation-question)
6. [Experimental Protocol](#6-experimental-protocol)
7. [Measurements and Metrics](#7-measurements-and-metrics)
8. [Predictions and Falsification](#8-predictions-and-falsification)
9. [Visualization Design](#9-visualization-design)
10. [Open Questions](#10-open-questions)

---

## 1. The Lightning Hypothesis

### 1.1 The Core Claim

```
THE WAVEFRONT ERROR SHOULD LOOK LIKE LIGHTNING

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  WHY LIGHTNING?                                                         │
│                                                                         │
│  Lightning is the canonical example of:                                │
│  • DISTRIBUTED UNCERTAINTY → CONCENTRATED CERTAINTY                    │
│  • MANY HYPOTHESES → ONE WINNER                                        │
│  • GRADUAL ACCUMULATION → SUDDEN DISCHARGE                             │
│                                                                         │
│  The physics:                                                           │
│  1. Charge accumulates across the cloud (uncertainty spreads)          │
│  2. Stepped leaders branch downward (hypotheses explored)              │
│  3. One leader reaches ground (one hypothesis confirmed)               │
│  4. Return stroke collapses the field (belief commits)                 │
│  5. Field rebuilds (new uncertainty for next prediction)               │
│                                                                         │
│  WE CLAIM: The wavefront error in embedding space follows              │
│  the same dynamics. The math is the same. The physics emerges.         │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1.2 The Analogy Mapped

```
LIGHTNING ←→ BELIEF DYNAMICS

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  LIGHTNING                     WAVEFRONT ERROR                         │
│  ─────────                     ───────────────                         │
│                                                                         │
│  Charge in cloud               Error/uncertainty in manifold           │
│  Electric field                Belief gradient                         │
│  Stepped leaders               Branching hypotheses                    │
│  Leader channels               High-error regions (possible futures)   │
│  Ground connection             Evidence confirms one hypothesis        │
│  Return stroke                 Belief collapse to single prediction    │
│  Field discharge               Entropy drops to near-zero              │
│  Field rebuild                 New uncertainty for next frame          │
│                                                                         │
│  Dielectric medium             Latent space / manifold                 │
│  Breakdown voltage             Confidence threshold                    │
│  Ionization                    Attention activation                    │
│  Conductivity                  Information flow rate                   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1.3 Why This Should Be True

```
THE MATHEMATICAL BASIS:

Both systems minimize energy (or loss) under constraints:

LIGHTNING:
  minimize: Electric potential energy
  subject to: Air resistance, ionization threshold
  result: Branching search then sudden discharge

BELIEF DYNAMICS:
  minimize: MSE loss = E[(ŷ - x_{t+1})²]
  subject to: Model capacity, attention constraints
  result: Exploration then sudden commitment

The solutions have the same structure because:
• Both involve gradients pointing toward a hidden "ground truth"
• Both have thresholds that gate state transitions
• Both exhibit positive feedback once a path is chosen
• Both show branching (interference) before commitment

GRADIENT DESCENT IS ENERGY MINIMIZATION.
ATTENTION IS FIELD DISTRIBUTION.
COLLAPSE IS DISCHARGE.
```

---

## 2. What We Expect to See

### 2.1 In Observable Space (2D)

```
WAVEFRONT ERROR IN OUTPUT SPACE

Frame progression showing error evolution:

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  t=0 (Initial uncertainty):                                            │
│                                                                         │
│       ░░░░░░░░░░░░░░░░░░░░                                             │
│      ░░░░░░░░░░░░░░░░░░░░░░                                            │
│     ░░░░░░████████████░░░░░░    Diffuse error blob                     │
│    ░░░░░████████████████░░░░    No clear structure yet                 │
│     ░░░░░░████████████░░░░░░    Uncertainty everywhere                 │
│      ░░░░░░░░░░░░░░░░░░░░░░                                            │
│       ░░░░░░░░░░░░░░░░░░░░                                             │
│                                                                         │
│  t=5 (Stepped leaders forming):                                        │
│                                                                         │
│               ░░░████░░░                                               │
│              ░░█████████░░                                             │
│             ░████░░░░████░        Branching structure                  │
│            ████░░░░░░░░████       Multiple "channels"                  │
│             ░████░░░░████░        Hypotheses separating                │
│              ░░█████████░░                                             │
│               ░░░████░░░                                               │
│                                                                         │
│  t=10 (Pre-collapse / criticality):                                    │
│                                                                         │
│                  ░██░                                                  │
│                 ████░                                                  │
│                █████░░             Channels competing                  │
│               ░░████░░░            One getting stronger                │
│                ░░███░░░            Others fading                       │
│                 ░░██░                                                  │
│                  ░█░                                                   │
│                                                                         │
│  t=11 (COLLAPSE / return stroke):                                      │
│                                                                         │
│                                                                         │
│                                                                         │
│                   █                Single point!                       │
│                  ███               Concentrated                        │
│                   █                Error collapsed                     │
│                                                                         │
│                                                                         │
│                                                                         │
│  t=15 (Recovery / new uncertainty):                                    │
│                                                                         │
│                    ░░                                                  │
│                   ░░░░                                                 │
│                  ░░██░░            New error forming                   │
│                 ░░████░░           At next uncertain region            │
│                  ░░██░░            Cycle begins again                  │
│                   ░░░░                                                 │
│                    ░░                                                  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.2 In Embedding Space (High-D, Projected)

```
WAVEFRONT ERROR IN EMBEDDING MANIFOLD

Projected to 2D/3D for visualization:

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  t=0-5 (SPREADING):                                                    │
│                                                                         │
│              ○                                                         │
│             ○ ○                                                        │
│            ○   ○         Error cloud EXPANDING                         │
│           ○     ○        Embeddings moving apart                       │
│          ○       ○       Entropy increasing                            │
│           ○     ○                                                      │
│            ○   ○                                                       │
│             ○ ○                                                        │
│              ○                                                         │
│                                                                         │
│  t=5-10 (BRANCHING / INTERFERENCE):                                    │
│                                                                         │
│                 ○○○                                                    │
│               ○○   ○○                                                  │
│              ○       ○    Distinct BRANCHES forming                    │
│             ○    ○    ○   Like stepped leaders                         │
│              ○ ○   ○ ○    Multiple hypothesis clusters                 │
│               ○○   ○○                                                  │
│                 ○○○                                                    │
│                                                                         │
│  t=10-11 (COLLAPSE / DISCHARGE):                                       │
│                                                                         │
│                                                                         │
│                                                                         │
│                 ○○○       One branch WINS                               │
│                 ●●●       Others VANISH                                │
│                 ○○○       Sudden transition                            │
│                           Cloud contracts to point                     │
│                                                                         │
│                                                                         │
│  t=11-15 (RECOVERY):                                                   │
│                                                                         │
│                  ○                                                     │
│                 ○●○       New uncertainty emerges                      │
│                ○ ● ○      Cloud begins expanding again                 │
│                 ○●○       Different region of manifold                 │
│                  ○                                                     │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.3 The Branching Structure

```
WHY BRANCHES?

The error should show BRANCHING because:

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  MULTIPLE PLAUSIBLE FUTURES:                                           │
│                                                                         │
│  Given past observations, several continuations are consistent:        │
│                                                                         │
│       Past → Present → Future A (blob continues straight)             │
│                     → Future B (blob accelerates)                      │
│                     → Future C (blob curves left)                      │
│                     → Future D (blob curves right)                     │
│                                                                         │
│  The model's prediction (mean of belief) is:                          │
│                                                                         │
│       ŷ = E[Future | Past] = weighted average of A, B, C, D           │
│                                                                         │
│  The ERROR shows where belief mass is distributed:                     │
│                                                                         │
│       High error at A → "might go to A"                               │
│       High error at B → "might go to B"                               │
│       etc.                                                             │
│                                                                         │
│  BRANCHES = distinct hypotheses in embedding space                     │
│                                                                         │
│  ════════════════════════════════════════════════════════════════════  │
│                                                                         │
│  INTERFERENCE DETERMINES WHICH BRANCHES SURVIVE:                       │
│                                                                         │
│  • Consistent evidence → CONSTRUCTIVE interference → branch grows     │
│  • Contradictory evidence → DESTRUCTIVE interference → branch fades   │
│                                                                         │
│  Just like stepped leaders:                                            │
│  • Ionized path → lower resistance → more current → more ionization   │
│  • Un-ionized path → high resistance → less current → dies out        │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. The Three Phases

### 3.1 Phase 1: Propagation (Tension)

```
PHASE 1: PROPAGATION / TENSION

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  WHAT HAPPENS:                                                          │
│  • Uncertainty spreads outward from current state                      │
│  • Error wavefront expands in embedding space                          │
│  • Multiple hypotheses coexist                                          │
│  • Entropy increases                                                    │
│                                                                         │
│  PHYSICALLY:                                                            │
│  • Like charge spreading across a cloud                                │
│  • Electric field builds but no discharge yet                          │
│  • Energy accumulating in the system                                   │
│                                                                         │
│  IN THE MODEL:                                                          │
│  • Attention weights spread across history                             │
│  • No single pattern dominates                                         │
│  • Wormhole connections sparse (nothing confident enough)              │
│  • Prediction hedges toward the mean                                   │
│                                                                         │
│  OBSERVABLES:                                                           │
│  • entropy: RISING                                                     │
│  • error_radius: EXPANDING                                             │
│  • attention_sharpness: LOW                                            │
│  • wormhole_connections: FEW                                           │
│  • gradient_alignment: LOW (gradients point in varied directions)     │
│                                                                         │
│  DURATION:                                                              │
│  Variable — depends on pattern ambiguity                               │
│  Simple patterns: short tension phase                                  │
│  Ambiguous patterns: long tension phase                                │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Phase 2: Interference (Criticality)

```
PHASE 2: INTERFERENCE / CRITICALITY

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  WHAT HAPPENS:                                                          │
│  • Hypotheses begin competing                                          │
│  • Some reinforce (constructive interference)                          │
│  • Others cancel (destructive interference)                            │
│  • Branching structure forms                                           │
│  • System at edge of commitment                                        │
│                                                                         │
│  PHYSICALLY:                                                            │
│  • Like stepped leaders probing downward                               │
│  • Multiple channels forming                                           │
│  • Competition for which path wins                                     │
│  • Near breakdown voltage                                              │
│                                                                         │
│  IN THE MODEL:                                                          │
│  • Attention developing modes                                          │
│  • Some history positions gaining weight                               │
│  • Wormhole connections activating selectively                         │
│  • Prediction starting to favor one hypothesis                         │
│                                                                         │
│  OBSERVABLES:                                                           │
│  • entropy: PEAKED (maximum uncertainty)                               │
│  • error_modes: MULTIPLE (distinct clusters)                           │
│  • attention_sharpness: INCREASING                                     │
│  • wormhole_connections: SELECTIVE activation                          │
│  • gradient_alignment: INCREASING                                      │
│                                                                         │
│  THIS IS THE "DECISION POINT":                                         │
│  • Small perturbations can change which hypothesis wins                │
│  • System is maximally sensitive                                       │
│  • Chaos / edge of criticality                                         │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.3 Phase 3: Collapse (Discharge)

```
PHASE 3: COLLAPSE / DISCHARGE

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  WHAT HAPPENS:                                                          │
│  • ONE hypothesis wins                                                  │
│  • Others extinguish SUDDENLY                                          │
│  • Error concentrates then vanishes locally                            │
│  • Prediction snaps to specific value                                  │
│  • Entropy crashes                                                     │
│                                                                         │
│  PHYSICALLY:                                                            │
│  • Like the return stroke in lightning                                 │
│  • Winning channel becomes main channel                                │
│  • Massive current flow (information flow)                             │
│  • Field collapses along the path                                      │
│                                                                         │
│  IN THE MODEL:                                                          │
│  • Attention sharpens to near-delta function                          │
│  • Wormhole finds strong match (> threshold)                          │
│  • Gradients align coherently                                          │
│  • Prediction commits to specific trajectory                          │
│                                                                         │
│  OBSERVABLES:                                                           │
│  • entropy: CRASHING (rapid drop)                                      │
│  • error_radius: CONTRACTING rapidly                                   │
│  • attention_sharpness: MAXIMUM                                        │
│  • wormhole_connections: STRONG (above threshold)                      │
│  • gradient_alignment: HIGH (coherent direction)                       │
│  • loss: DROPS (prediction matches truth)                             │
│                                                                         │
│  THE COLLAPSE IS SUDDEN:                                                │
│  • Not gradual — exponential dynamics                                  │
│  • Positive feedback: one hypothesis winning → more evidence for it   │
│  • Phase transition signature                                          │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 4. What Happens at Collapse

### 4.1 The Critical Question

```
WHAT DOES THE WAVEFRONT COLLAPSE TO?

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  NOT A "DECISION" — AN APPROXIMATION                                   │
│                                                                         │
│  The collapse is NOT:                                                   │
│  • A conscious decision                                                 │
│  • A symbolic choice                                                    │
│  • A discrete category                                                  │
│                                                                         │
│  The collapse IS:                                                       │
│  • A continuous approximation to the true state                        │
│  • The mean of a now-concentrated belief                               │
│  • A point in embedding space                                          │
│  • A specific prediction value                                         │
│                                                                         │
│  ════════════════════════════════════════════════════════════════════  │
│                                                                         │
│  BEFORE COLLAPSE:                                                       │
│                                                                         │
│  Belief b(s) is SPREAD:                                                │
│                                                                         │
│       ┌─────────────────────────────────┐                              │
│       │     ▓▓▓                         │                              │
│       │   ▓▓▓▓▓▓▓▓                      │  Multiple modes              │
│       │  ▓▓▓▓▓▓▓▓▓▓     ▓▓▓            │  (hypotheses)                │
│       │   ▓▓▓▓▓▓▓▓    ▓▓▓▓▓▓           │                              │
│       │     ▓▓▓      ▓▓▓▓▓▓▓▓          │                              │
│       │               ▓▓▓▓▓            │                              │
│       └─────────────────────────────────┘                              │
│                                                                         │
│  Prediction ŷ = E[s] = weighted mean (somewhere between modes)         │
│  Error is HIGH because truth will be at ONE mode, not the mean        │
│                                                                         │
│  ════════════════════════════════════════════════════════════════════  │
│                                                                         │
│  AFTER COLLAPSE:                                                        │
│                                                                         │
│  Belief b(s) is CONCENTRATED:                                          │
│                                                                         │
│       ┌─────────────────────────────────┐                              │
│       │                                 │                              │
│       │                                 │  Single mode                 │
│       │          █████                  │  (committed hypothesis)      │
│       │         ███████                 │                              │
│       │          █████                  │                              │
│       │                                 │                              │
│       └─────────────────────────────────┘                              │
│                                                                         │
│  Prediction ŷ ≈ mode location ≈ truth (if collapse was correct)       │
│  Error is LOW because prediction is close to actual                   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.2 What the Approximation Contains

```
THE APPROXIMATION IS:

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  1. A POINT IN EMBEDDING SPACE                                         │
│     The collapsed belief lives at a specific location on the          │
│     manifold. This location encodes:                                   │
│     • Position (where the pattern is)                                 │
│     • Identity (what the pattern is)                                  │
│     • Phase (where in its cycle)                                      │
│     • Velocity (how it's moving)                                      │
│                                                                         │
│  2. A COMPRESSED REPRESENTATION                                        │
│     The many bits of uncertainty have been compressed to a            │
│     single point. Information has been LOST (the other hypotheses)   │
│     but the dominant hypothesis is now CERTAIN.                       │
│                                                                         │
│  3. AN ATOMIC TRUTH (in the Old Lady's terms)                         │
│     After culling the causal tree, what remains is:                   │
│     • The irreducible core                                            │
│     • The decision-relevant information                               │
│     • The actionable knowledge                                        │
│                                                                         │
│  4. THE BEST GUESS                                                     │
│     The collapsed belief is NOT guaranteed to be correct.            │
│     It's the model's best approximation given:                        │
│     • Available evidence                                               │
│     • Learned patterns                                                 │
│     • Inference constraints                                            │
│                                                                         │
│  RESIDUAL UNCERTAINTY:                                                 │
│     Even after collapse, there may be:                                │
│     • Small remaining variance                                        │
│     • Uncertainty about details (high-freq)                          │
│     • Possibility the collapse was WRONG                             │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.3 Can Collapse Be Wrong?

```
WRONG COLLAPSE: When the Return Stroke Misses

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  YES, COLLAPSE CAN BE WRONG.                                           │
│                                                                         │
│  Just like lightning can strike the "wrong" spot                      │
│  (not the tallest object, not the best conductor),                    │
│  belief collapse can commit to the wrong hypothesis.                  │
│                                                                         │
│  WHEN THIS HAPPENS:                                                     │
│                                                                         │
│  • Prediction confidently points to location A                        │
│  • Ground truth reveals location B                                     │
│  • Error SPIKES (wrong collapse = large error)                        │
│  • Model receives strong gradient signal                              │
│  • Weights update to prevent this mistake                             │
│                                                                         │
│  ════════════════════════════════════════════════════════════════════  │
│                                                                         │
│  SIGNATURES OF WRONG COLLAPSE:                                         │
│                                                                         │
│  1. CONFIDENT BUT WRONG                                                │
│     • High attention sharpness (model was confident)                  │
│     • But high error (model was wrong)                                │
│     • This is worse than being uncertain                              │
│                                                                         │
│  2. LARGE GRADIENT                                                     │
│     • Big correction needed                                            │
│     • Weights shift significantly                                      │
│     • Learning opportunity                                             │
│                                                                         │
│  3. FOLLOWED BY DESTABILIZATION                                        │
│     • Next few predictions may be uncertain                           │
│     • Model "lost its footing"                                        │
│     • Recovery phase longer than usual                                │
│                                                                         │
│  WRONG COLLAPSE IS HOW THE MODEL LEARNS.                               │
│  Mistakes are the gradient signal for improvement.                    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 5. The Approximation Question

### 5.1 What Kind of Approximation?

```
THE NATURE OF THE COLLAPSED STATE

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  The collapse produces different things at different scales:           │
│                                                                         │
│  LOW-FREQUENCY BANDS (DC, Band 1-2):                                   │
│  ─────────────────────────────────────                                  │
│  Collapse to: IDENTITY / CATEGORY                                      │
│  "It's a blob" / "It's moving" / "It exists"                          │
│  This is COARSE APPROXIMATION — what type of thing                    │
│                                                                         │
│  MID-FREQUENCY BANDS (Band 3-4):                                       │
│  ─────────────────────────────────                                      │
│  Collapse to: FEATURES / STRUCTURE                                     │
│  "Round shape" / "This size" / "These parts"                          │
│  This is STRUCTURAL APPROXIMATION — what configuration                │
│                                                                         │
│  HIGH-FREQUENCY BANDS (Band 5-6):                                      │
│  ─────────────────────────────────                                      │
│  Collapse to: POSITION / DETAILS                                       │
│  "At pixel (34, 56)" / "This edge orientation"                        │
│  This is PRECISE APPROXIMATION — where exactly                        │
│                                                                         │
│  ════════════════════════════════════════════════════════════════════  │
│                                                                         │
│  THE FULL APPROXIMATION IS HIERARCHICAL:                               │
│                                                                         │
│  [exists] + [is blob] + [is round] + [at position X] = prediction     │
│     DC       Low          Mid            High                          │
│                                                                         │
│  Each band contributes its collapsed belief.                          │
│  Together they form the complete approximation.                       │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 5.2 Approximation Error Structure

```
THE ERROR AFTER COLLAPSE HAS STRUCTURE

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  Even after collapse, residual error reveals what the model           │
│  still doesn't know:                                                   │
│                                                                         │
│  SHAPE OF RESIDUAL ERROR:                                              │
│                                                                         │
│  • CRESCENT ahead of prediction → velocity uncertainty                │
│  • RING around prediction → position uncertainty                       │
│  • STREAK along direction → trajectory uncertainty                     │
│  • BLOB at prediction → identity/existence uncertainty                │
│                                                                         │
│  ════════════════════════════════════════════════════════════════════  │
│                                                                         │
│  SPECTRAL DISTRIBUTION OF RESIDUAL ERROR:                              │
│                                                                         │
│  • High-freq residual → precise position unknown                      │
│  • Mid-freq residual → shape/features unknown                         │
│  • Low-freq residual → identity/existence unknown                     │
│                                                                         │
│  After a GOOD collapse:                                                │
│  • Low-freq error: LOW (identity known)                               │
│  • High-freq error: MEDIUM (exact position less certain)              │
│                                                                         │
│  After a BAD collapse:                                                 │
│  • Low-freq error: HIGH (wrong identity!)                             │
│  • Everything downstream is wrong too                                 │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Experimental Protocol

### 6.1 Setup

```
EXPERIMENTAL SETUP

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  MODEL:                                                                 │
│  • AKIRA spectral attention predictor                                  │
│  • 7 frequency bands                                                   │
│  • Temporal + Neighbor + Wormhole attention                           │
│                                                                         │
│  INPUT:                                                                 │
│  • Moving Gaussian blob (simple, predictable)                         │
│  • Variations: linear, circular, acceleration, bifurcation            │
│                                                                         │
│  EMBEDDING EXTRACTION:                                                  │
│  • Hook all attention layers                                           │
│  • Extract Q, K, V, attention weights                                 │
│  • Extract per-band representations                                    │
│  • Extract error signal at each position                              │
│                                                                         │
│  DIMENSIONALITY REDUCTION:                                              │
│  • Real-time: PCA or Shogu                                             │
│  • Post-hoc: UMAP, t-SNE, PHATE                                       │
│                                                                         │
│  VISUALIZATION:                                                         │
│  • 2D/3D embedding plot (trajectories)                                │
│  • Error heatmap (output space)                                        │
│  • Metrics timeline (entropy, loss, etc.)                             │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 6.2 Procedure

```
EXPERIMENTAL PROCEDURE

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  STEP 1: BASELINE CHARACTERIZATION                                     │
│  ─────────────────────────────────                                      │
│  • Run untrained model                                                 │
│  • Observe embedding structure                                         │
│  • Establish baseline metrics                                          │
│                                                                         │
│  STEP 2: TRAINING OBSERVATION                                          │
│  ────────────────────────────                                           │
│  • Train model on simple blob pattern                                  │
│  • Record all embeddings at each step                                  │
│  • Track entropy, error, attention                                     │
│                                                                         │
│  STEP 3: COLLAPSE DETECTION                                            │
│  ──────────────────────────                                             │
│  • Identify collapse events (entropy drops)                            │
│  • Record timing, location, magnitude                                  │
│  • Correlate with error reduction                                      │
│                                                                         │
│  STEP 4: WAVEFRONT TRACKING                                            │
│  ──────────────────────────                                             │
│  • Embed error signal at each timestep                                │
│  • Track trajectory through embedding space                           │
│  • Observe expansion, branching, collapse                             │
│                                                                         │
│  STEP 5: BRANCH ANALYSIS                                               │
│  ───────────────────────                                                │
│  • Identify branches (clusters in embedding)                          │
│  • Track which branches survive                                        │
│  • Analyze interference patterns                                       │
│                                                                         │
│  STEP 6: APPROXIMATION ANALYSIS                                        │
│  ──────────────────────────────                                         │
│  • What does collapsed state contain?                                 │
│  • Per-band analysis of approximation                                 │
│  • Residual error structure                                           │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 7. Measurements and Metrics

### 7.1 Primary Metrics

```
PRIMARY METRICS TO TRACK

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  ENTROPY METRICS:                                                       │
│  ────────────────                                                       │
│  • H_attention: Entropy of attention weights                           │
│  • H_belief: Entropy of belief distribution                           │
│  • dH/dt: Rate of entropy change (collapse velocity)                  │
│  • d²H/dt²: Acceleration (approaching collapse)                       │
│                                                                         │
│  ERROR METRICS:                                                         │
│  ─────────────                                                          │
│  • E_total: Total MSE                                                  │
│  • E_spatial[x,y]: Error at each position                             │
│  • E_spectral[band]: Error per frequency band                         │
│  • R_error: Effective radius of error cloud                           │
│                                                                         │
│  WAVEFRONT METRICS:                                                     │
│  ─────────────────                                                      │
│  • N_branches: Number of distinct hypotheses                           │
│  • D_spread: Dispersion of error in embedding space                   │
│  • V_wavefront: Velocity of wavefront expansion                       │
│  • T_collapse: Time to collapse                                        │
│                                                                         │
│  ATTENTION METRICS:                                                     │
│  ─────────────────                                                      │
│  • Sharpness: max(A) / mean(A)                                        │
│  • Effective_support: How many positions attended                     │
│  • Wormhole_active: Number above threshold                            │
│  • Alignment: Cross-head agreement                                    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 7.2 Collapse Detection Criteria

```
HOW TO DETECT A COLLAPSE EVENT

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  CRITERIA (any 2 of 4 = collapse detected):                            │
│                                                                         │
│  1. ENTROPY DROP                                                        │
│     dH/dt < -θ_entropy                                                 │
│     (entropy falling faster than threshold)                            │
│                                                                         │
│  2. ERROR CONCENTRATION                                                 │
│     dR_error/dt < -θ_radius                                            │
│     (error cloud contracting)                                          │
│                                                                         │
│  3. ATTENTION SPIKE                                                     │
│     max(A) > θ_attention                                               │
│     (attention highly concentrated)                                    │
│                                                                         │
│  4. PREDICTION JUMP                                                     │
│     ‖ŷ_t - ŷ_{t-1}‖ > θ_jump                                          │
│     (prediction changed suddenly)                                      │
│                                                                         │
│  COLLAPSE MAGNITUDE:                                                    │
│  M_collapse = H_before - H_after                                       │
│  (how much entropy was discharged)                                     │
│                                                                         │
│  COLLAPSE LOCATION:                                                     │
│  Where in embedding space did collapse occur?                          │
│  What position in output space?                                        │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 8. Predictions and Falsification

### 8.1 Specific Predictions

```
TESTABLE PREDICTIONS

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  P1: BRANCHING STRUCTURE                                                │
│  ────────────────────────                                               │
│  The error wavefront in embedding space will show distinct             │
│  branches (clusters) before collapse.                                  │
│  TEST: Cluster analysis on error embeddings during criticality        │
│  FALSIFIED IF: Error cloud is always unimodal                         │
│                                                                         │
│  P2: SUDDEN COLLAPSE                                                    │
│  ────────────────────                                                   │
│  Collapse will be sudden (exponential), not gradual (linear).         │
│  TEST: Fit entropy decay during collapse to exponential vs linear     │
│  FALSIFIED IF: Entropy decreases linearly                             │
│                                                                         │
│  P3: COLLAPSE CORRELATES WITH ERROR DROP                               │
│  ────────────────────────────────────────                               │
│  Entropy collapse should predict immediate error reduction.           │
│  TEST: Cross-correlation of dH/dt and dE/dt                          │
│  FALSIFIED IF: No correlation between collapse and error              │
│                                                                         │
│  P4: WRONG COLLAPSE SIGNATURE                                          │
│  ────────────────────────────                                           │
│  Wrong collapses should show: high confidence + high error.           │
│  TEST: Identify cases of low entropy + high error                     │
│  FALSIFIED IF: High error always accompanies high entropy             │
│                                                                         │
│  P5: RECOVERY FOLLOWS COLLAPSE                                         │
│  ───────────────────────────                                            │
│  After collapse, entropy should rise again (new uncertainty).         │
│  TEST: Track entropy after collapse events                            │
│  FALSIFIED IF: Entropy stays low after collapse                       │
│                                                                         │
│  P6: SPECTRAL HIERARCHY OF COLLAPSE                                    │
│  ──────────────────────────────────                                     │
│  Low-freq bands should collapse first (identity before position).     │
│  TEST: Track per-band entropy collapse timing                         │
│  FALSIFIED IF: All bands collapse simultaneously                      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 8.2 What Would Prove the Lightning Model?

```
EVIDENCE THAT WOULD SUPPORT THE LIGHTNING MODEL

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  STRONG SUPPORT:                                                        │
│  ───────────────                                                        │
│  • Visible branching in embedding space (stepped leaders)             │
│  • Sudden collapse with exponential dynamics (return stroke)          │
│  • One branch wins, others vanish (winner-take-all)                   │
│  • Collapse-error correlation (discharge resolves uncertainty)        │
│  • Recovery phase visible (field rebuilds)                            │
│                                                                         │
│  ADDITIONAL SUPPORT:                                                    │
│  ─────────────────                                                      │
│  • Positive feedback dynamics during collapse                          │
│  • Threshold behavior (similarity > τ triggers collapse)              │
│  • Interference patterns (constructive/destructive)                   │
│  • Power-law statistics (criticality signature)                       │
│                                                                         │
│  IF WE SEE ALL THIS:                                                    │
│  The lightning model is validated. Belief dynamics follow             │
│  the same mathematics as electrical discharge.                        │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 8.3 What Would Falsify the Lightning Model?

```
EVIDENCE THAT WOULD FALSIFY THE LIGHTNING MODEL

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  FALSIFYING EVIDENCE:                                                   │
│  ────────────────────                                                   │
│  • No branching — error cloud always unimodal                         │
│  • Gradual convergence — no sudden collapse                           │
│  • No correlation between entropy and error                           │
│  • No recovery phase — system stays committed                         │
│  • Random dynamics — no consistent pattern                            │
│                                                                         │
│  IF WE SEE THIS:                                                        │
│  The lightning model is wrong. Belief dynamics follow                 │
│  some other pattern (perhaps simpler gradient flow).                  │
│                                                                         │
│  ALTERNATIVE MODELS TO CONSIDER:                                        │
│  ────────────────────────────────                                       │
│  • Simple gradient descent (no phase transition)                      │
│  • Continuous annealing (gradual, not sudden)                         │
│  • Random walk (no structured dynamics)                               │
│  • Something else we haven't imagined                                 │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 9. Visualization Design

### 9.1 Main View

```
VISUALIZATION LAYOUT

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                    EMBEDDING SPACE (3D)                          │  │
│  │                                                                  │  │
│  │    ○───○───○───●───●───●      Error trajectory                  │  │
│  │       \   /   /                                                  │  │
│  │        ○─○───○                Branching visible                  │  │
│  │                                                                  │  │
│  │    Color: time (blue→red)                                       │  │
│  │    Size: entropy                                                 │  │
│  │    Marker: ○ = spreading, ● = collapsed                         │  │
│  │                                                                  │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐        │
│  │  INPUT FRAME    │  │  PREDICTION     │  │  ERROR MAP      │        │
│  │                 │  │                 │  │                 │        │
│  │    [image]      │  │    [image]      │  │    [heatmap]    │        │
│  │                 │  │                 │  │                 │        │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘        │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                    METRICS TIMELINE                              │  │
│  │                                                                  │  │
│  │  Entropy: ────╮    ╭────────╮    ╭────                         │  │
│  │               ╰────╯        ╰────╯                              │  │
│  │                 ↑            ↑                                   │  │
│  │              COLLAPSE     COLLAPSE                               │  │
│  │                                                                  │  │
│  │  Loss:    ────╮  ╭──────────╮  ╭────                           │  │
│  │               ╰──╯          ╰──╯                                │  │
│  │                                                                  │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 9.2 Collapse Event Detail View

```
COLLAPSE EVENT DETAIL

When collapse detected, show:

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  BEFORE COLLAPSE (t-5 to t-1)                                   │   │
│  │                                                                  │   │
│  │     ○   ○        Branches visible                               │   │
│  │    ○ ○ ○ ○       Multiple hypotheses                            │   │
│  │     ○   ○                                                       │   │
│  │                                                                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  COLLAPSE MOMENT (t)                                            │   │
│  │                                                                  │   │
│  │       ●●●        One cluster wins                               │   │
│  │       ●●●        Others vanish                                  │   │
│  │       ●●●        Sudden transition                              │   │
│  │                                                                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  AFTER COLLAPSE (t+1 to t+5)                                    │   │
│  │                                                                  │   │
│  │        ○         New uncertainty forming                        │   │
│  │       ○○○        Cloud beginning to expand                      │   │
│  │        ○                                                        │   │
│  │                                                                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  METRICS AT COLLAPSE:                                                   │
│  • Entropy drop: ΔH = 2.3 bits                                        │
│  • Error drop: ΔE = 45%                                               │
│  • Collapse duration: 1 frame                                          │
│  • Winning branch: #2 (of 3)                                          │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 10. Open Questions

```
QUESTIONS THIS EXPERIMENT SHOULD ANSWER

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  FUNDAMENTAL:                                                           │
│  ────────────                                                           │
│  • Does the wavefront actually look like lightning?                   │
│  • Is collapse sudden (phase transition) or gradual (annealing)?     │
│  • What determines which branch wins?                                 │
│  • How does the approximation quality depend on collapse timing?     │
│                                                                         │
│  MECHANISTIC:                                                           │
│  ────────────                                                           │
│  • What triggers collapse? (Wormhole? Threshold? Something else?)    │
│  • How do the three attention types contribute to collapse?          │
│  • What role does spectral decomposition play?                       │
│  • Can we predict collapse before it happens?                        │
│                                                                         │
│  PRACTICAL:                                                             │
│  ──────────                                                             │
│  • Can we control collapse timing?                                    │
│  • Can we prevent wrong collapses?                                    │
│  • Can we use collapse as a confidence indicator?                    │
│  • Can collapse dynamics guide architecture design?                  │
│                                                                         │
│  THEORETICAL:                                                           │
│  ─────────────                                                          │
│  • Is there a conserved quantity during collapse?                    │
│  • What is the "breakdown voltage" of the belief manifold?          │
│  • Does this connect to critical phenomena in physics?               │
│  • Is there a minimum description length at collapse?                │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Summary

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  EXP: WAVEFRONT INTERFERENCE COLLAPSE                                  │
│                                                                         │
│  ═══════════════════════════════════════════════════════════════════   │
│                                                                         │
│  HYPOTHESIS:                                                            │
│  The wavefront error in embedding space follows lightning dynamics:   │
│  • PROPAGATION: Uncertainty spreads, branches form                    │
│  • INTERFERENCE: Branches compete, some reinforce, others cancel     │
│  • COLLAPSE: One branch wins suddenly, others vanish                 │
│  • RECOVERY: New uncertainty forms, cycle repeats                    │
│                                                                         │
│  ═══════════════════════════════════════════════════════════════════   │
│                                                                         │
│  WHAT COLLAPSE IS:                                                      │
│  Not a "decision" — an APPROXIMATION                                  │
│  The collapsed belief is the best guess given evidence               │
│  Hierarchical: identity (low-freq) + position (high-freq)            │
│  Can be WRONG — that's how learning happens                          │
│                                                                         │
│  ═══════════════════════════════════════════════════════════════════   │
│                                                                         │
│  KEY PREDICTIONS:                                                       │
│  • Branching structure before collapse                                │
│  • Sudden (exponential) collapse dynamics                             │
│  • Collapse-error correlation                                          │
│  • Recovery phase after collapse                                       │
│  • Spectral hierarchy (low-freq collapses first)                     │
│                                                                         │
│  ═══════════════════════════════════════════════════════════════════   │
│                                                                         │
│  IF WE SEE LIGHTNING-LIKE DYNAMICS:                                    │
│  Belief dynamics follow energy minimization.                          │
│  The POMDP framework is validated.                                     │
│  The pump cycle is real.                                               │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---



*"The wavefront should look like lightning. Branching leaders exploring hypotheses. Return stroke when one wins. The physics is the same because the math is the same. Let's see if we're right."*

*Oscar Goldman — Shogu Research Group @ Datamutant.ai subsidiary of 温心重工業*