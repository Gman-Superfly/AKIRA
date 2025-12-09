# Pretraining AKIRA

## Building the Ghost, Teaching the Dream Language

---

## Table of Contents

1. [What Pretraining Is](#1-what-pretraining-is)
2. [What We're Building](#2-what-were-building)
3. [Spectral Curriculum](#3-spectral-curriculum)
4. [Boundary and Aliasing Considerations](#4-boundary-and-aliasing-considerations)
5. [Learning Rate Hierarchy](#5-learning-rate-hierarchy)
6. [Wormhole Emergence](#6-wormhole-emergence)
7. [Monitoring for Phase Transitions](#7-monitoring-for-phase-transitions)
8. [The Pump Cycle: Does It Emerge?](#8-the-pump-cycle-does-it-emerge)
9. [Practical Training Protocol](#9-practical-training-protocol)
10. [What Success Looks Like](#10-what-success-looks-like)

---

## 1. What Pretraining Is

### 1.1 Building the Ghost

```
PRETRAINING = SUMMONING THE GHOST INTO THE WEIGHTS

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  BEFORE PRETRAINING:                                                    │
│                                                                         │
│  • Random weights                                                      │
│  • No structure in the manifold                                       │
│  • No learned frequencies                                              │
│  • No collapse dynamics                                                │
│  • No ghost — just noise                                              │
│                                                                         │
│  AFTER PRETRAINING:                                                     │
│                                                                         │
│  • Structured manifold                                                 │
│  • Frequency-tuned bands                                               │
│  • Learned collapse patterns                                           │
│  • Wormhole connectivity                                               │
│  • The ghost lives in the weights                                     │
│                                                                         │
│  ════════════════════════════════════════════════════════════════════  │
│                                                                         │
│  Pretraining is not just "fitting to data."                           │
│  It's building the INFRASTRUCTURE for belief.                         │
│  The ghost emerges from the training dynamics.                        │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Teaching the Dream Language

```
THE GHOST LEARNS ITS VOCABULARY

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  During pretraining, the model learns:                                 │
│                                                                         │
│  ACTION QUANTA:                                                     │
│  • What magnitudes matter                                              │
│  • What phase relationships mean                                       │
│  • What frequencies carry what information                            │
│  • What coherence patterns indicate                                   │
│                                                                         │
│  SPECTRAL GRAMMAR:                                                      │
│  • Low-freq = stable, identity, "what"                                │
│  • High-freq = transient, position, "where"                           │
│  • Cross-band relationships                                            │
│  • Temporal dynamics across bands                                      │
│                                                                         │
│  COLLAPSE DYNAMICS:                                                     │
│  • When to commit                                                      │
│  • How to resolve competition                                         │
│  • What triggers collapse                                              │
│  • How to recover after collapse                                      │
│                                                                         │
│  ════════════════════════════════════════════════════════════════════  │
│                                                                         │
│  The training data TEACHES the vocabulary.                            │
│  The architecture ENABLES the grammar.                                │
│  The ghost EMERGES from their interaction.                            │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1.3 The Explicit → Implicit Transition

```
FROM MEMORIZATION TO UNDERSTANDING

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  EARLY TRAINING (Explicit phase):                                      │
│                                                                         │
│  • Model memorizes training examples                                  │
│  • Train loss drops, test loss flat                                   │
│  • High-freq details stored                                            │
│  • Weights chaotic, no structure                                      │
│  • Information in DATA, not weights                                   │
│                                                                         │
│  TRANSITION (Grokking point):                                          │
│                                                                         │
│  • Sudden generalization                                               │
│  • Test loss drops to match train                                     │
│  • Compression happens                                                 │
│  • Weights organize                                                    │
│  • Information moves to WEIGHTS                                       │
│                                                                         │
│  LATE TRAINING (Implicit phase):                                       │
│                                                                         │
│  • Model generalizes                                                   │
│  • Low-freq structure stable                                          │
│  • High-freq adapts quickly                                           │
│  • Manifold structured                                                 │
│  • Ghost is present                                                    │
│                                                                         │
│  ════════════════════════════════════════════════════════════════════  │
│                                                                         │
│  PRETRAINING GOAL: Reach the implicit phase.                          │
│  The ghost lives in the WEIGHTS, not the DATA.                        │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. What We're Building

### 2.1 The Manifold

```
BUILDING THE BELIEF MANIFOLD

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  THE MANIFOLD IS:                                                       │
│                                                                         │
│  • The geometric structure of learned representations                 │
│  • Where concepts "live" in weight space                              │
│  • The hypersphere for similarity computation                         │
│  • The attractor landscape for collapse                               │
│                                                                         │
│  DURING PRETRAINING:                                                    │
│                                                                         │
│  Early: Manifold is flat, unstructured                               │
│  Middle: Clusters form, distances become meaningful                  │
│  Late: Clear topology, stable attractors                             │
│                                                                         │
│  ════════════════════════════════════════════════════════════════════  │
│                                                                         │
│  WHAT WE WANT:                                                          │
│                                                                         │
│  • Similar concepts near each other                                   │
│  • Different concepts far apart                                       │
│  • Smooth interpolation between related concepts                     │
│  • Clear boundaries between unrelated concepts                       │
│  • Stable attractors for collapse destinations                       │
│                                                                         │
│  The manifold IS the ghost's "mind."                                  │
│  Pretraining sculpts it.                                              │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.2 The Spectral Bands

```
TUNING THE FREQUENCY BANDS

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  THE BANDS ARE:                                                         │
│                                                                         │
│  Band 0 (DC):     Existence, presence                                 │
│  Band 1:          Identity, category                                  │
│  Band 2:          Properties, features                                │
│  Band 3:          Configuration, structure                            │
│  Band 4:          Relationships, connections                          │
│  Band 5:          Position, location                                  │
│  Band 6:          Details, texture                                    │
│                                                                         │
│  DURING PRETRAINING:                                                    │
│                                                                         │
│  Each band must learn:                                                 │
│  • What frequencies to respond to                                     │
│  • What information to extract                                        │
│  • How to interact with other bands                                   │
│  • When to activate wormhole connections                             │
│                                                                         │
│  ════════════════════════════════════════════════════════════════════  │
│                                                                         │
│  BAND-SPECIFIC ROLES EMERGE:                                           │
│                                                                         │
│  Low bands: Stable, slow-learning, identity-focused                  │
│  High bands: Adaptive, fast-learning, position-focused               │
│                                                                         │
│  This hierarchy is not imposed — it EMERGES.                          │
│  But we can ENCOURAGE it with learning rate schedules.               │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.3 The Wormhole Network

```
GROWING THE WORMHOLE CONNECTIONS

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  WORMHOLES CONNECT:                                                     │
│                                                                         │
│  • Distant but similar regions                                        │
│  • Across time (temporal memory)                                      │
│  • Across space (non-local matching)                                  │
│  • Across bands (spectral communication)                              │
│                                                                         │
│  DURING PRETRAINING:                                                    │
│                                                                         │
│  Early: Few wormholes (threshold too high, no matches)               │
│  Middle: Wormholes begin forming (similarity structure emerges)      │
│  Late: Rich wormhole network (manifold supports matching)            │
│                                                                         │
│  ════════════════════════════════════════════════════════════════════  │
│                                                                         │
│  WHAT ENABLES WORMHOLES:                                                │
│                                                                         │
│  • Structured manifold (so similarities are meaningful)              │
│  • Normalized representations (hypersphere for cosine sim)           │
│  • Learned threshold (when to connect)                               │
│  • History buffer (what to connect to)                               │
│                                                                         │
│  Wormholes are useless without manifold structure.                   │
│  The manifold must form FIRST.                                        │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Spectral Curriculum

### 3.1 The Order of Learning

```
SHOULD LEARNING BE CURRICULUM-ORDERED?

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  OPTION 1: ALL AT ONCE (Standard)                                      │
│  ─────────────────────────────────                                      │
│                                                                         │
│  All frequencies, all bands, from the start.                          │
│  Let the model sort it out.                                            │
│                                                                         │
│  PRO: Simple, no curriculum design needed                             │
│  CON: May struggle to establish low-freq first                       │
│                                                                         │
│  ────────────────────────────────────────────────────────────────────  │
│                                                                         │
│  OPTION 2: FREQUENCY CURRICULUM (Low → High)                          │
│  ───────────────────────────────────────────                            │
│                                                                         │
│  Start with low-frequency content only.                               │
│  Gradually add higher frequencies.                                    │
│                                                                         │
│  PRO: Establishes stable base first                                   │
│  CON: May underfit high-freq, complex to implement                   │
│                                                                         │
│  ────────────────────────────────────────────────────────────────────  │
│                                                                         │
│  OPTION 3: COARSE-TO-FINE (Resolution curriculum)                     │
│  ─────────────────────────────────────────────────                      │
│                                                                         │
│  Start with low-resolution data.                                      │
│  Gradually increase resolution.                                       │
│                                                                         │
│  PRO: Natural frequency progression                                   │
│  CON: Needs multi-resolution data                                     │
│                                                                         │
│  ────────────────────────────────────────────────────────────────────  │
│                                                                         │
│  OPTION 4: IMPLICIT CURRICULUM (via learning rates)                   │
│  ─────────────────────────────────────────────────                      │
│                                                                         │
│  All data at once, but low-freq bands learn slower.                  │
│  High-freq adapts fast, low-freq stabilizes.                         │
│                                                                         │
│  PRO: Simple, respects hierarchy                                      │
│  CON: May not be strong enough ordering                              │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Recommended Approach

```
RECOMMENDED: IMPLICIT CURRICULUM + MONITORING

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  STRATEGY:                                                              │
│                                                                         │
│  1. Train on full data from the start                                 │
│  2. Use per-band learning rates (low bands slower)                   │
│  3. Monitor per-band dynamics                                         │
│  4. Adjust if hierarchy doesn't emerge                               │
│                                                                         │
│  ════════════════════════════════════════════════════════════════════  │
│                                                                         │
│  WHY THIS APPROACH:                                                     │
│                                                                         │
│  • The spectral hierarchy SHOULD emerge naturally                     │
│  • If it doesn't, we learn something                                  │
│  • Forcing it might be fighting the data                             │
│  • Monitoring tells us what's happening                              │
│                                                                         │
│  FALLBACK:                                                              │
│  If hierarchy doesn't emerge, consider:                               │
│  • Stronger learning rate differential                               │
│  • Explicit frequency curriculum                                      │
│  • Architecture changes                                                │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Boundary and Aliasing Considerations

### 4.1 The Critical Decision: Windowing

```
WINDOWING DURING PRETRAINING

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  THE QUESTION:                                                          │
│  Should we apply windowing to training data?                          │
│                                                                         │
│  ════════════════════════════════════════════════════════════════════  │
│                                                                         │
│  IF NO WINDOWING:                                                       │
│                                                                         │
│  • Model sees raw edges                                               │
│  • Learns boundary artifacts as real                                  │
│  • May generalize artifacts to inference                             │
│  • But: Uses all available information                               │
│                                                                         │
│  IF WINDOWING:                                                          │
│                                                                         │
│  • Edges are smoothed                                                  │
│  • Less artifact learning                                              │
│  • Cleaner spectral representation                                    │
│  • But: Loses edge information                                        │
│                                                                         │
│  ════════════════════════════════════════════════════════════════════  │
│                                                                         │
│  RECOMMENDATION:                                                        │
│                                                                         │
│  Train WITH windowing (Hamming or Hanning).                           │
│  Apply SAME windowing at inference.                                   │
│  Accept the edge information loss.                                    │
│  The alternative (artifact learning) is worse.                       │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Avoiding Aliasing

```
ALIASING PREVENTION

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  THE DANGER:                                                            │
│  Training data with frequencies above Nyquist                         │
│  → Model learns aliased patterns                                      │
│  → Believes false low-freq structure                                  │
│                                                                         │
│  ════════════════════════════════════════════════════════════════════  │
│                                                                         │
│  PREVENTION STRATEGIES:                                                 │
│                                                                         │
│  1. LOWPASS FILTER TRAINING DATA                                       │
│     Remove frequencies above what bands can represent.                │
│     Ensures no aliasing in input.                                     │
│     Cost: Loses high-freq detail.                                     │
│                                                                         │
│  2. OVERSAMPLE THEN DOWNSAMPLE                                         │
│     Process at higher resolution.                                     │
│     Apply anti-aliasing filter.                                       │
│     Downsample to target resolution.                                  │
│     Cost: More computation.                                            │
│                                                                         │
│  3. RANDOM SHIFTS/CROPS                                                 │
│     Vary the phase of aliased patterns.                               │
│     Prevents consistent alias learning.                               │
│     Cost: Less consistent training.                                   │
│                                                                         │
│  4. MONITOR FOR ALIASING                                                │
│     Track per-band frequency content.                                 │
│     Detect if low bands contain high-freq patterns.                  │
│     Cost: Additional monitoring overhead.                             │
│                                                                         │
│  ════════════════════════════════════════════════════════════════════  │
│                                                                         │
│  RECOMMENDATION:                                                        │
│  Use anti-aliasing filtering + monitoring.                            │
│  Better to lose true high-freq than learn false low-freq.            │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.3 Context Window Boundary Training

```
TRAINING WITH CONTEXT BOUNDARIES

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  THE PROBLEM:                                                           │
│  Context windows have hard edges.                                     │
│  Position 1 and position N are special.                               │
│  Model might learn edge-specific patterns.                            │
│                                                                         │
│  ════════════════════════════════════════════════════════════════════  │
│                                                                         │
│  STRATEGIES:                                                            │
│                                                                         │
│  1. RANDOM CROPPING                                                     │
│     Take random segments from longer sequences.                       │
│     Edges are random, not special.                                    │
│     Forces position-invariance.                                       │
│                                                                         │
│  2. OVERLAP TRAINING                                                    │
│     Train on overlapping windows.                                     │
│     Same content appears at different positions.                     │
│     Reduces position-specific learning.                               │
│                                                                         │
│  3. EXPLICIT BOUNDARY TOKENS                                           │
│     Add [START] and [END] tokens.                                     │
│     Model knows where edges are.                                      │
│     Can learn to handle them appropriately.                          │
│                                                                         │
│  4. TEMPORAL WINDOWING                                                  │
│     Apply window function to positional weights.                     │
│     Edges get less attention.                                          │
│     Similar to spatial windowing.                                     │
│                                                                         │
│  ════════════════════════════════════════════════════════════════════  │
│                                                                         │
│  RECOMMENDATION:                                                        │
│  Random cropping + boundary tokens.                                   │
│  Model knows where edges are, but doesn't overfit them.             │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Learning Rate Hierarchy

### 5.1 Per-Band Learning Rates

```
DIFFERENTIAL LEARNING RATES BY BAND

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  THE PRINCIPLE:                                                         │
│                                                                         │
│  Low-freq bands should be STABLE:                                      │
│  • Slow learning rate                                                  │
│  • Represent identity, category                                       │
│  • Should not change rapidly                                          │
│                                                                         │
│  High-freq bands should be ADAPTIVE:                                   │
│  • Fast learning rate                                                  │
│  • Represent position, detail                                         │
│  • Should adapt quickly                                                │
│                                                                         │
│  ════════════════════════════════════════════════════════════════════  │
│                                                                         │
│  ⚠️  SEE CANONICAL_PARAMETERS.md FOR AUTHORITATIVE SPECIFICATION      │
│                                                                         │
│  CANONICAL (Specification A) — Target for experiments:                │
│                                                                         │
│  Band 0:  0.00001    (1×)                                             │
│  Band 1:  0.0001     (10×)                                            │
│  Band 2:  0.0003     (30×)                                            │
│  Band 3:  0.001      (100×)                                           │
│  Band 4:  0.003      (300×)                                           │
│  Band 5:  0.01       (1000×)                                          │
│  Band 6:  0.03       (3000×)                                          │
│  Band 7:  0.001      (100×, same as Band 3 — both integrative)       │
│                                                                         │
│  WARM-START (Specification B) — Gentler first 50k-100k steps:        │
│                                                                         │
│  Band 0:  lr_base × 0.1    (10% of base)                             │
│  Band 1:  lr_base × 0.2                                               │
│  Band 2:  lr_base × 0.4                                               │
│  Band 3:  lr_base × 0.6                                               │
│  Band 4:  lr_base × 0.8                                               │
│  Band 5:  lr_base × 1.0    (100% of base)                            │
│  Band 6:  lr_base × 1.2    (120% of base)                            │
│  Band 7:  lr_base × 0.6    (same as Band 3)                          │
│                                                                         │
│  USAGE:                                                                 │
│  1. Start training with Specification B (lr_base = 0.0001)           │
│  2. Gradually transition to Specification A over 50k-100k steps      │
│  3. Continue training with Specification A (canonical)               │
│                                                                         │
│  This gives a 12× ratio initially, transitioning to 3000× ratio.    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 5.2 Wormhole Learning Rate

```
WORMHOLE ATTENTION LEARNING

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  WORMHOLE COMPONENTS:                                                   │
│                                                                         │
│  Query/Key projections:                                                │
│  • Learn what to match                                                 │
│  • Should be MODERATE learning rate                                   │
│  • Need stable similarity computation                                 │
│                                                                         │
│  Value projections:                                                     │
│  • Learn what to retrieve                                             │
│  • Can be FASTER learning rate                                        │
│  • Adapts to content                                                   │
│                                                                         │
│  Threshold (if learned):                                               │
│  • Learn when to connect                                               │
│  • Should be SLOW learning rate                                       │
│  • Critical decision, shouldn't oscillate                            │
│                                                                         │
│  ════════════════════════════════════════════════════════════════════  │
│                                                                         │
│  SUGGESTED:                                                             │
│                                                                         │
│  Query/Key: lr_base × 0.5                                             │
│  Value:     lr_base × 1.0                                             │
│  Threshold: lr_base × 0.1                                             │
│                                                                         │
│  Or: Keep threshold fixed initially, learn later.                    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 5.3 Warmup and Schedule

```
LEARNING RATE SCHEDULE

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  WARMUP PHASE:                                                          │
│                                                                         │
│  Steps 0-1000: Linear warmup                                          │
│  • Start from near-zero                                               │
│  • Prevents early instability                                         │
│  • Lets random initialization settle                                  │
│                                                                         │
│  PEAK PHASE:                                                            │
│                                                                         │
│  Steps 1000-100000: Full learning rate                               │
│  • Per-band differential rates active                                │
│  • Main learning happens here                                         │
│  • Watch for phase transitions                                        │
│                                                                         │
│  DECAY PHASE:                                                           │
│                                                                         │
│  Steps 100000+: Cosine or linear decay                               │
│  • Gradually reduce all rates                                         │
│  • Fine-tune the manifold                                             │
│  • Stabilize representations                                          │
│                                                                         │
│  ════════════════════════════════════════════════════════════════════  │
│                                                                         │
│  THE HIERARCHY SHOULD BE MAINTAINED THROUGHOUT.                        │
│  Even during decay, low bands decay slower than high bands.          │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Wormhole Emergence

### 6.1 When to Enable Wormholes

```
WORMHOLE ACTIVATION STRATEGY

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  THE QUESTION:                                                          │
│  Should wormholes be active from the start?                           │
│                                                                         │
│  ════════════════════════════════════════════════════════════════════  │
│                                                                         │
│  OPTION A: ALWAYS ON                                                    │
│                                                                         │
│  • Wormholes active from step 0                                       │
│  • Learn together with everything else                               │
│  • May be unstable early (random similarities)                       │
│  • But: Fully integrated from start                                  │
│                                                                         │
│  OPTION B: DELAYED ACTIVATION                                          │
│                                                                         │
│  • Wormholes disabled until step N                                   │
│  • Let manifold form first                                            │
│  • Then enable wormholes on structured manifold                      │
│  • Cleaner learning, potentially faster                              │
│                                                                         │
│  OPTION C: GRADUAL ACTIVATION                                          │
│                                                                         │
│  • Start with high threshold (few wormholes)                         │
│  • Gradually lower threshold                                          │
│  • More wormholes as manifold structures                             │
│  • Natural curriculum                                                  │
│                                                                         │
│  ════════════════════════════════════════════════════════════════════  │
│                                                                         │
│  RECOMMENDATION: Option C (Gradual Activation)                        │
│                                                                         │
│  FOR COHERENCE-GATED WORMHOLES (theory-aligned):                      │
│  Initial coherence_threshold: 0.3 (very strict, only very coherent)  │
│  Final coherence_threshold: 0.5 (normal operation)                   │
│  Schedule: Linear increase over first 50k steps                       │
│                                                                         │
│  FOR ENERGY-GATED WORMHOLES (simplified hybrid):                      │
│  Initial similarity_threshold: 0.99 (almost nothing connects)        │
│  Final similarity_threshold: 0.92 (normal operation)                 │
│  Schedule: Linear decrease over first 50k steps                       │
│                                                                         │
│  NOTE: coherence_threshold is normalized entropy (lower = stricter). │
│        similarity_threshold is cosine similarity (higher = stricter). │
│        They work in opposite directions!                              │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 6.2 Monitoring Wormhole Health

```
WORMHOLE DIAGNOSTICS

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  HEALTHY WORMHOLE SIGNS:                                                │
│                                                                         │
│  • Connection count increases over training                           │
│  • Connections are SELECTIVE (not everything connects)               │
│  • Similar inputs get similar connections                            │
│  • Connections contribute to prediction accuracy                     │
│                                                                         │
│  UNHEALTHY WORMHOLE SIGNS:                                              │
│                                                                         │
│  • All connections or no connections (wrong threshold)               │
│  • Random connections (manifold not structured)                      │
│  • Connections don't help prediction                                 │
│  • Instability (connections oscillate)                               │
│                                                                         │
│  ════════════════════════════════════════════════════════════════════  │
│                                                                         │
│  METRICS TO TRACK:                                                      │
│                                                                         │
│  FOR COHERENCE-GATED WORMHOLES:                                        │
│  • Coherence gate activation rate (fraction with gate > 0.5)         │
│  • Mean normalized entropy of attention distributions               │
│  • Gate value distribution (should have bimodal: open/closed)       │
│  • Contribution to final prediction                                  │
│                                                                         │
│  FOR ENERGY-GATED WORMHOLES:                                           │
│  • Wormhole activation rate (fraction above similarity threshold)    │
│  • Mean/std of similarity values                                     │
│  • Entropy of connection distribution                                │
│  • Contribution to final prediction                                  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 7. Monitoring for Phase Transitions

### 7.1 Grokking Detection

```
DETECTING THE GROKKING TRANSITION

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  GROKKING = Sudden generalization after apparent convergence          │
│                                                                         │
│  SIGNATURES:                                                            │
│                                                                         │
│  1. LOSS DYNAMICS                                                       │
│     • Train loss plateaus                                             │
│     • Test loss high and flat                                         │
│     • Then: Test loss suddenly drops                                  │
│                                                                         │
│  2. WEIGHT DYNAMICS                                                     │
│     • Weight norm may spike then settle                               │
│     • Representation structure suddenly changes                      │
│     • Entropy of representations drops                               │
│                                                                         │
│  3. GENERALIZATION METRICS                                             │
│     • OOD performance suddenly improves                               │
│     • Perturbation stability increases                               │
│     • Representations become more clustered                          │
│                                                                         │
│  ════════════════════════════════════════════════════════════════════  │
│                                                                         │
│  WHEN YOU SEE GROKKING:                                                 │
│                                                                         │
│  • DON'T STOP EARLY (would miss the transition)                      │
│  • The model is moving from memorization to understanding            │
│  • This is GOOD — keep going                                          │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 7.2 Collapse Event Detection

```
DETECTING COLLAPSE DYNAMICS DURING TRAINING

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  AS THE MODEL LEARNS, IT SHOULD DEVELOP COLLAPSE BEHAVIOR:            │
│                                                                         │
│  EARLY TRAINING:                                                        │
│  • Attention is diffuse                                               │
│  • Entropy stays high                                                  │
│  • No clear collapse events                                           │
│                                                                         │
│  MIDDLE TRAINING:                                                       │
│  • Attention starts sharpening                                        │
│  • Some collapse events appear                                        │
│  • Entropy shows variation                                            │
│                                                                         │
│  LATE TRAINING:                                                         │
│  • Clear collapse dynamics                                            │
│  • Sharp attention patterns                                            │
│  • Entropy shows pump cycle                                           │
│                                                                         │
│  ════════════════════════════════════════════════════════════════════  │
│                                                                         │
│  METRICS TO TRACK:                                                      │
│                                                                         │
│  • Mean attention entropy over time                                   │
│  • Entropy variance (should increase)                                │
│  • Collapse event frequency                                           │
│  • Collapse sharpness (how sudden)                                   │
│                                                                         │
│  IF COLLAPSE DOESN'T EMERGE:                                           │
│  • May need sharper attention (temperature)                          │
│  • May need more training                                             │
│  • May need architecture adjustment                                   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 7.3 Spectral Hierarchy Emergence

```
DETECTING SPECTRAL HIERARCHY

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  THE HIERARCHY SHOULD EMERGE:                                          │
│                                                                         │
│  Low bands: More stable, lower entropy, slower adaptation            │
│  High bands: More variable, higher entropy, faster adaptation        │
│                                                                         │
│  ════════════════════════════════════════════════════════════════════  │
│                                                                         │
│  METRICS TO TRACK PER BAND:                                            │
│                                                                         │
│  • Mean entropy                                                        │
│  • Entropy variance                                                    │
│  • Weight change rate                                                  │
│  • Contribution to prediction                                         │
│  • Collapse timing (does low-freq collapse first?)                   │
│                                                                         │
│  WHAT WE EXPECT:                                                        │
│                                                                         │
│  entropy[band_0] < entropy[band_1] < ... < entropy[band_6]           │
│  collapse_time[band_0] < collapse_time[band_1] < ...                 │
│  weight_change[band_0] < weight_change[band_1] < ...                 │
│                                                                         │
│  IF NOT EMERGING:                                                       │
│  • Increase learning rate differential                               │
│  • Consider explicit curriculum                                       │
│  • Check for architecture issues                                      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 8. The Pump Cycle: Does It Emerge?

### 8.1 What Is the Pump Cycle?

```
THE PUMP CYCLE: TENSION → DISCHARGE → RECOVERY

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  TENSION (High entropy):                                                │
│  • Many hypotheses active                                              │
│  • Uncertainty is high                                                 │
│  • Belief cloud is spread                                             │
│                                                                         │
│  DISCHARGE (Collapse):                                                  │
│  • Hypotheses compete                                                  │
│  • Winner emerges                                                      │
│  • Entropy drops sharply                                               │
│                                                                         │
│  RECOVERY (Post-collapse):                                              │
│  • New input arrives                                                   │
│  • Uncertainty builds again                                            │
│  • Entropy rises                                                       │
│                                                                         │
│  ════════════════════════════════════════════════════════════════════  │
│                                                                         │
│  THE CYCLE SHOULD EMERGE during inference.                            │
│  But does it need to be LEARNED during training?                     │
│  Or does it emerge from the architecture?                             │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 8.2 Does Training Create the Pump?

```
IS THE PUMP LEARNED OR ARCHITECTURAL?

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  ARCHITECTURAL FACTORS (Given):                                        │
│                                                                         │
│  • Softmax creates competition (winner-take-all tendency)            │
│  • Temporal input creates recovery (new info raises entropy)         │
│  • Threshold creates discrete activation                             │
│                                                                         │
│  LEARNED FACTORS (Training):                                           │
│                                                                         │
│  • WHEN to collapse (threshold tuning)                                │
│  • HOW SHARP to collapse (attention temperature)                     │
│  • WHAT triggers recovery (input patterns)                           │
│                                                                         │
│  ════════════════════════════════════════════════════════════════════  │
│                                                                         │
│  HYPOTHESIS:                                                            │
│                                                                         │
│  The pump is ARCHITECTURALLY ENABLED but TRAINING-TUNED.             │
│                                                                         │
│  The architecture provides the mechanism.                             │
│  Training adjusts the parameters (threshold, sharpness).             │
│  The cycle period is partly learned.                                  │
│                                                                         │
│  WHAT TO MONITOR:                                                       │
│                                                                         │
│  • Does cycle behavior appear during training?                       │
│  • Does cycle period stabilize?                                       │
│  • Does cycle amplitude change?                                       │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 9. Practical Training Protocol

### 9.1 Data Preparation

```
PREPARING TRAINING DATA

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  1. ANTI-ALIASING                                                       │
│     • Apply lowpass filter before any downsampling                    │
│     • Ensure no frequency above Nyquist in final data                │
│     • Better to lose true high-freq than learn false low-freq        │
│                                                                         │
│  2. WINDOWING                                                           │
│     • Apply Hamming or Hanning window to spatial data                │
│     • Reduces spectral leakage                                        │
│     • Apply consistently to all data                                  │
│                                                                         │
│  3. TEMPORAL PREPARATION                                                │
│     • Random crops from longer sequences                              │
│     • Add [START] and [END] tokens                                    │
│     • Consider overlapping windows                                    │
│                                                                         │
│  4. NORMALIZATION                                                       │
│     • Normalize to consistent range                                   │
│     • Per-sample or global, be consistent                            │
│     • Track normalization stats                                       │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 9.2 Training Loop

```python
"""
AKIRA PRETRAINING LOOP (Pseudocode)
"""

def pretrain_akira(model, data_loader, config):
    
    # Initialize per-band learning rates
    band_lr_multipliers = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2]
    
    optimizer = setup_optimizer_with_per_band_lr(
        model, 
        base_lr=config.base_lr,
        band_multipliers=band_lr_multipliers
    )
    
    scheduler = setup_scheduler(
        optimizer,
        warmup_steps=config.warmup_steps,
        total_steps=config.total_steps
    )
    
    # Initialize monitors
    entropy_monitor = EntropyMonitor()
    collapse_detector = CollapseDetector()
    grokking_detector = GrokkingDetector()
    wormhole_monitor = WormholeMonitor()
    
    # Wormhole threshold schedule (theory-aligned: coherence gate)
    # NOTE: For energy-gated wormholes, use similarity_threshold instead
    initial_coherence_threshold = 0.3  # Strict: only very coherent attention
    final_coherence_threshold = 0.5    # Normal operation
    threshold_steps = 50000
    
    for step, batch in enumerate(data_loader):
        
        # Update wormhole coherence threshold (linear increase)
        if step < threshold_steps:
            progress = step / threshold_steps
            model.wormhole_coherence_threshold = (
                initial_coherence_threshold + 
                (final_coherence_threshold - initial_coherence_threshold) * progress
            )
        
        # Forward pass
        output, stats = model(batch, return_stats=True)
        
        # Compute loss
        loss = compute_loss(output, batch.target)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
        
        # Optimizer step
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
        # Monitoring
        entropy_monitor.update(stats['entropy'], step)
        collapse_detector.update(stats['entropy'], step)
        wormhole_monitor.update(stats['wormhole_stats'], step)
        
        if step % config.eval_interval == 0:
            test_loss = evaluate(model, test_loader)
            grokking_detector.update(loss.item(), test_loss, step)
            
            log_metrics({
                'train_loss': loss.item(),
                'test_loss': test_loss,
                'entropy': stats['entropy'].mean().item(),
                'wormhole_connections': stats['wormhole_stats']['num_connections'],
                'collapse_events': collapse_detector.recent_count(),
                'grokking_status': grokking_detector.status()
            })
        
        # Early stopping checks
        if grokking_detector.detected() and config.stop_after_grokking:
            print(f"Grokking detected at step {step}")
            continue_training = config.continue_after_grokking
        
    return model
```

### 9.3 Checkpointing and Recovery

```
CHECKPOINTING STRATEGY

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  WHAT TO SAVE:                                                          │
│                                                                         │
│  • Model weights                                                       │
│  • Optimizer state                                                     │
│  • Scheduler state                                                     │
│  • Current wormhole threshold                                         │
│  • Monitor histories                                                   │
│  • Random states                                                       │
│                                                                         │
│  WHEN TO SAVE:                                                          │
│                                                                         │
│  • Every N steps (regular checkpoints)                                │
│  • When test loss improves (best model)                               │
│  • When grokking detected (transition point)                         │
│  • Before any risky operation                                         │
│                                                                         │
│  RECOVERY:                                                              │
│                                                                         │
│  • Resume from checkpoint with all states                             │
│  • Verify training continues correctly                                │
│  • Check that dynamics match pre-interruption                        │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 10. What Success Looks Like

### 10.1 Manifold Quality

```
SIGNS OF A GOOD MANIFOLD

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  GEOMETRIC STRUCTURE:                                                   │
│  • Similar inputs have similar representations                        │
│  • Different inputs are well-separated                               │
│  • Smooth interpolation between related concepts                     │
│  • Clear clusters visible in embedding space                         │
│                                                                         │
│  SPECTRAL STRUCTURE:                                                    │
│  • Low bands capture identity/category                               │
│  • High bands capture position/detail                                │
│  • Hierarchy is evident                                               │
│  • Bands have distinct dynamics                                       │
│                                                                         │
│  ATTRACTOR STRUCTURE:                                                   │
│  • Clear collapse destinations                                        │
│  • Stable attractors for common patterns                             │
│  • Recovery from collapse is smooth                                   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 10.2 Collapse Dynamics

```
SIGNS OF HEALTHY COLLAPSE

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  COLLAPSE SHAPE:                                                        │
│  • Sudden, not gradual (exponential decay)                           │
│  • Winner-take-all (clear dominant winner)                           │
│  • Appropriate timing (not too early, not too late)                  │
│                                                                         │
│  PUMP CYCLE:                                                            │
│  • Visible oscillation in entropy                                     │
│  • Recovery after collapse                                            │
│  • Consistent cycle period                                            │
│                                                                         │
│  SPECTRAL ORDERING:                                                     │
│  • Low-freq collapses before high-freq                               │
│  • Hierarchy in collapse timing                                       │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 10.3 Wormhole Health

```
SIGNS OF HEALTHY WORMHOLES

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  CONNECTION PATTERNS:                                                   │
│  • Selective (not everything connects)                               │
│  • Meaningful (similar things connect)                               │
│  • Stable (connections don't oscillate wildly)                       │
│                                                                         │
│  CONTRIBUTION:                                                          │
│  • Wormholes improve prediction                                       │
│  • Removing wormholes hurts performance                              │
│  • Wormhole attention is informative                                 │
│                                                                         │
│  DISTRIBUTION:                                                          │
│  • Connections span different regions                                │
│  • Not all connections to same place                                 │
│  • Healthy near-threshold population                                 │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 10.4 The Ghost Is Present

```
SIGNS THE GHOST HAS EMERGED

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  GENERALIZATION:                                                        │
│  • Performs well on unseen data                                       │
│  • Interpolates reasonably                                            │
│  • Doesn't just memorize                                              │
│                                                                         │
│  ROBUSTNESS:                                                            │
│  • Stable under perturbation                                          │
│  • Graceful degradation                                               │
│  • Consistent responses                                                │
│                                                                         │
│  STRUCTURE:                                                             │
│  • Representations are interpretable                                  │
│  • Spectral decomposition is meaningful                              │
│  • Collapse dynamics are predictable                                  │
│                                                                         │
│  RESPONSIVENESS:                                                        │
│  • Responds to probes consistently                                    │
│  • Entropy reflects uncertainty                                       │
│  • Tickling reveals structure                                         │
│                                                                         │
│  ════════════════════════════════════════════════════════════════════  │
│                                                                         │
│  WHEN THESE ARE ALL PRESENT:                                           │
│  The ghost lives in the weights.                                      │
│  Pretraining is complete.                                             │
│  The séance can begin.                                                │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Summary

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│                    P R E T R A I N I N G   A K I R A                   │
│                                                                         │
│  ═══════════════════════════════════════════════════════════════════   │
│                                                                         │
│  WHAT WE'RE BUILDING:                                                   │
│  • The belief manifold (geometric structure)                         │
│  • The spectral bands (frequency-tuned processing)                   │
│  • The wormhole network (non-local connections)                      │
│  • The collapse dynamics (belief resolution)                         │
│  • The ghost (emergent understanding)                                │
│                                                                         │
│  ═══════════════════════════════════════════════════════════════════   │
│                                                                         │
│  KEY DECISIONS:                                                         │
│  • Windowing: Yes (Hamming), applied consistently                    │
│  • Aliasing: Anti-aliasing filter + monitoring                       │
│  • Learning rates: Per-band hierarchy (slow → fast)                 │
│  • Wormholes: Gradual activation (threshold schedule)               │
│  • Curriculum: Implicit via learning rates                           │
│                                                                         │
│  ═══════════════════════════════════════════════════════════════════   │
│                                                                         │
│  WHAT TO MONITOR:                                                       │
│  • Per-band entropy and dynamics                                     │
│  • Collapse events and timing                                        │
│  • Wormhole activation and health                                    │
│  • Grokking transition                                               │
│  • Spectral hierarchy emergence                                      │
│                                                                         │
│  ═══════════════════════════════════════════════════════════════════   │
│                                                                         │
│  SUCCESS CRITERIA:                                                      │
│  • Structured manifold                                               │
│  • Healthy collapse dynamics                                         │
│  • Working wormhole network                                          │
│  • Spectral hierarchy present                                        │
│  • Ghost responsive to probes                                        │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

*Oscar Goldman — Shogu Research Group @ Datamutant.ai*

*"Pretraining is not fitting curves to data. It's building a mind. The manifold is the geometry of understanding. The spectral bands are the vocabulary. The wormholes are the connections between ideas. The collapse is decision. Train carefully: what you build here is what the ghost will become. Window your edges. Respect the Nyquist limit. Let the hierarchy emerge. When it's done, the ghost will speak."*

