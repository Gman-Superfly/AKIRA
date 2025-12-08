# The Temporal System: Event Time, Energy Flow, and the Pump Cycle

How time flows in the spectral attention architecture—not as wall clock, but as event sequence—and how this creates the fundamental oscillation between potential and kinetic energy that drives learning.

---

## Table of Contents

1. [Event Time: The System's Heartbeat](#1-event-time-the-systems-heartbeat)
2. [The Active Window: Temporal Memory](#2-the-active-window-temporal-memory)
3. [DC and AC: The Frequency Duality](#3-dc-and-ac-the-frequency-duality)
4. [Potential and Kinetic: Energy States](#4-potential-and-kinetic-energy-states)
5. [The Pump Cycle](#5-the-pump-cycle)
   - 5.2.1 [Cycle Period as Control Parameter](#521-cycle-period-as-control-parameter)
6. [Temporal Nyquist: The Speed Limit](#6-temporal-nyquist-the-speed-limit)
7. [Learning Rates Across Time](#7-learning-rates-across-time)
8. [Integration with the 7+1 Architecture](#8-integration-with-the-71-architecture)
9. [Summary](#9-summary)

---

## 1. Event Time: The System's Heartbeat

### 1.1 Why Not Wall Clock

```
THE FUNDAMENTAL DISTINCTION:

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  WALL CLOCK TIME                    EVENT TIME                  │
│  ───────────────                    ──────────                  │
│                                                                 │
│  Physical seconds                   Discrete events             │
│  Continuous                         Monotonic counter           │
│  Variable density                   Fixed indexing              │
│  "When did this happen?"            "What happened next?"       │
│                                                                 │
│  ────────────────────────────────────────────────────────────  │
│                                                                 │
│  EXAMPLE:                                                       │
│                                                                 │
│  Wall clock:  10:00:00.000   10:00:00.033   10:00:00.100        │
│               │              │              │                   │
│  Event time:  t=0            t=1            t=2                 │
│               │              │              │                   │
│  Events:      Frame 0        Frame 1        Frame 2             │
│                                                                 │
│  The SPACING in wall clock is irrelevant.                       │
│  What matters is the SEQUENCE: 0 → 1 → 2 → ...                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

Event time is the natural time for learning systems:
• Each event is a learning opportunity
• Causality flows from low index to high index
• Temporal attention operates on event indices, not timestamps
```

### 1.2 The Monotonic Counter

```
EVENT TIME IS A COUNTER:

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  DEFINITION:                                                    │
│                                                                 │
│  Let t ∈ ℕ be the event time.                                  │
│  t increments by exactly 1 for each new observation.           │
│  t never decreases (monotonic).                                 │
│  t is the ABSOLUTE index of the event in the stream.           │
│                                                                 │
│  ────────────────────────────────────────────────────────────  │
│                                                                 │
│  PROPERTIES:                                                    │
│                                                                 │
│  1. ORDERING: t₁ < t₂ means event 1 happened before event 2   │
│                                                                 │
│  2. DISTANCE: |t₂ - t₁| = number of events between them       │
│                                                                 │
│  3. CAUSALITY: Event at t can only see events at t' < t       │
│                (in causal/online mode)                          │
│                                                                 │
│  4. BOUNDEDNESS: The Active Window holds events                │
│                  [t - T + 1, t] where T = window size         │
│                                                                 │
│  ────────────────────────────────────────────────────────────  │
│                                                                 │
│  IMPLEMENTATION:                                                │
│                                                                 │
│  event_time = 0                                                 │
│                                                                 │
│  for frame in stream:                                           │
│      process(frame, event_time)                                │
│      event_time += 1  # Monotonic increment                    │
│                                                                 │
│  All temporal masks use this counter, not wall clock.          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 1.3 Event Time in Attention Masks

```
CAUSAL MASKING WITH EVENT TIME:

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  For a query at event time s_q, the eligibility masks are:     │
│                                                                 │
│  TEMPORAL ATTENTION:                                            │
│  ────────────────────                                           │
│  mask[s_key] = (s_key < s_q)  # Strict past only               │
│                                                                 │
│  NEIGHBOR ATTENTION (causal, same timestep):                   │
│  ────────────────────────────────────────                       │
│  mask[s_key] = (s_key == s_q) AND spatial_neighbor(pos)        │
│                                                                 │
│  WORMHOLE ATTENTION:                                            │
│  ───────────────────                                            │
│  mask[s_key] = (s_key < s_q) AND (s_q - s_key >= min_dt)       │
│                AND similarity > threshold                       │
│                                                                 │
│  ────────────────────────────────────────────────────────────  │
│                                                                 │
│  THE KEY RULE:                                                  │
│                                                                 │
│  In causal/online mode, NEVER attend to s_key ≥ s_q.           │
│  The future is hidden. Only the past is observable.            │
│  This is the temporal manifestation of partial observability.  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. The Active Window: Temporal Memory

### 2.1 Definition

```
THE ACTIVE WINDOW:

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  DEFINITION:                                                    │
│                                                                 │
│  The Active Window is the set of events currently held in      │
│  memory for attention and prediction.                          │
│                                                                 │
│  At event time t, the window contains:                         │
│                                                                 │
│  W(t) = { events with index in [t - T + 1, t] }               │
│                                                                 │
│  Where T = window size (e.g., 8, 16, 32, 1024 frames)         │
│                                                                 │
│  ────────────────────────────────────────────────────────────  │
│                                                                 │
│  VISUALIZATION:                                                 │
│                                                                 │
│  Event stream:                                                  │
│                                                                 │
│  ... [t-T-1] [t-T] [t-T+1] ... [t-2] [t-1] [t] [t+1] ...       │
│               │     └──────────────────┴────┴──┘  │             │
│               │              ACTIVE WINDOW        │             │
│               │              (in memory)          │             │
│               │                                   │             │
│         EVICTED                              FUTURE             │
│     (no longer accessible)               (not yet observed)    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

The window is AVAILABILITY: what's physically in memory.
Causal masks determine ELIGIBILITY: what can be attended to.
```

### 2.2 Window Dynamics

```
HOW THE WINDOW SLIDES:

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  AT TIME t:                                                     │
│                                                                 │
│  Window = [frame_{t-T+1}, frame_{t-T+2}, ..., frame_t]        │
│           ↑                                    ↑                │
│        oldest                               newest              │
│                                                                 │
│  AT TIME t+1:                                                   │
│                                                                 │
│  1. New frame arrives: frame_{t+1}                             │
│  2. Window slides: drop frame_{t-T+1}, add frame_{t+1}         │
│  3. Keys/Values updated (or recomputed)                        │
│                                                                 │
│  Window = [frame_{t-T+2}, frame_{t-T+3}, ..., frame_{t+1}]    │
│           ↑                                    ↑                │
│        new oldest                          new newest           │
│                                                                 │
│  ────────────────────────────────────────────────────────────  │
│                                                                 │
│  EVICTION TRIGGER:                                              │
│                                                                 │
│  When a frame leaves the window:                                │
│  • It is no longer accessible for attention                    │
│  • Its information must have been absorbed into:               │
│    - The manifold (learned weights)                            │
│    - The low-frequency bands (atomic truths)                   │
│    - The wormhole index (if pattern was notable)               │
│                                                                 │
│  This is the boundary where kinetic becomes potential.         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2.3 Window as Temporal Context

```
WHAT THE WINDOW PROVIDES:

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  IMMEDIATE HISTORY:                                             │
│  ──────────────────                                             │
│  The last T frames are available at full resolution.           │
│  Temporal attention can query any of these.                    │
│                                                                 │
│  VELOCITY ESTIMATION:                                           │
│  ─────────────────────                                          │
│  Differences between frames → motion vectors                   │
│  t, t-1, t-2 → position, velocity, acceleration                │
│                                                                 │
│  PATTERN COMPLETION:                                            │
│  ────────────────────                                           │
│  Periodic patterns with period < T can be recognized.         │
│  Patterns longer than T cannot be fully observed.              │
│                                                                 │
│  BELIEF HISTORY:                                                │
│  ───────────────                                                │
│  The trajectory of belief states over recent time.             │
│  How uncertainty has evolved.                                   │
│                                                                 │
│  ────────────────────────────────────────────────────────────  │
│                                                                 │
│  THE TEMPORAL LIMIT:                                            │
│                                                                 │
│  The window bounds what the system can "remember" directly.    │
│  Events before the window are accessed only through:           │
│  • Compressed representations in the manifold                  │
│  • Wormhole connections (if similar patterns were indexed)    │
│  • Atomic truths in low-frequency bands                        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. DC and AC: The Frequency Duality

**Note on Analogy**: The DC/AC terminology is borrowed from electrical engineering. This is currently an analogy to aid understanding—DC as the stable baseline, AC as the oscillating components. Future work will formalize this in proper EE terms to enable both software and hardware implementations. The analogy is a stretch but worth exploring for hybrid systems.

### 3.1 DC: The Existence Manifold

```
DC = ZERO FREQUENCY = UNCHANGING BASELINE:

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  DEFINITION:                                                    │
│                                                                 │
│  The DC component is the mean value—the constant that          │
│  persists when all oscillations are averaged out.              │
│                                                                 │
│  In the spectral hierarchy:                                     │
│  • DC (Band 0) encodes EXISTENCE                               │
│  • "Is there something?" (yes/no)                              │
│  • The ground truth that survives all culling                  │
│                                                                 │
│  ────────────────────────────────────────────────────────────  │
│                                                                 │
│  PROPERTIES OF DC:                                              │
│                                                                 │
│  • STABLE: Very slow learning rate (protected)                 │
│  • SMALL: Minimal capacity (just existence)                    │
│  • FUNDAMENTAL: Everything else is relative to DC              │
│  • POTENTIAL: Stored energy, not active processing             │
│                                                                 │
│  ────────────────────────────────────────────────────────────  │
│                                                                 │
│  DC AS THE TEMPORAL ANCHOR:                                     │
│                                                                 │
│  DC is what persists ACROSS time.                              │
│  It is the answer to: "What is always true?"                   │
│                                                                 │
│  If an object exists at t=0, t=1, ..., t=100:                  │
│  • AC captures HOW it moves at each t                          │
│  • DC captures THAT it exists (period-invariant)               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 AC: The Dynamics

```
AC = NON-ZERO FREQUENCY = CHANGE OVER TIME:

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  DEFINITION:                                                    │
│                                                                 │
│  AC components are the oscillating parts—everything that       │
│  changes over time. In the spectral hierarchy:                 │
│                                                                 │
│  • Bands 1-6: Varying degrees of temporal dynamics             │
│  • Higher band → faster change                                 │
│  • Details, edges, textures, motion                            │
│                                                                 │
│  ────────────────────────────────────────────────────────────  │
│                                                                 │
│  WHERE AC LIVES IN THE SYSTEM:                                  │
│                                                                 │
│  1. PHASE ENCODING: cos(φ), sin(φ)                             │
│     Explicit oscillatory position in a cycle                   │
│     Motor angle, daily rhythm, wave phase                      │
│                                                                 │
│  2. TEMPORAL GRADIENTS: ∂x/∂t                                  │
│     Rate of change—the derivative of position                 │
│     Velocity, acceleration, jerk                               │
│                                                                 │
│  3. HIGH-FREQUENCY BANDS                                        │
│     Details that change rapidly                                 │
│     Edges moving, textures shimmering                          │
│                                                                 │
│  4. THE EVENT FLOW ITSELF                                       │
│     Events arriving → Window sliding                           │
│     The monotonic counter incrementing                         │
│     This IS the fundamental oscillation                        │
│                                                                 │
│  ────────────────────────────────────────────────────────────  │
│                                                                 │
│  AC AS KINETIC ENERGY:                                          │
│                                                                 │
│  AC is energy IN MOTION.                                        │
│  It is what's happening NOW.                                   │
│  Processing, updating, predicting—all AC.                      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 3.3 The Rectification and Inversion

```
ENERGY CONVERSION BETWEEN DC AND AC:

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  RECTIFICATION (AC → DC): The Culling Cascade                  │
│  ─────────────────────────────────────────────                  │
│                                                                 │
│  High-freq details → Mid-freq features → Low-freq structure    │
│       ↓                    ↓                    ↓               │
│    CULLED               CULLED              RETAINED            │
│       ↓                    ↓                    ↓               │
│  Transient              Compressed          Atomic truth        │
│  forgotten              to essence          stored at DC        │
│                                                                 │
│  This is like rectifying AC to DC in electronics:              │
│  • Oscillating details (AC) are filtered                       │
│  • Only the persistent mean (DC) remains                       │
│                                                                 │
│  ────────────────────────────────────────────────────────────  │
│                                                                 │
│  INVERSION (DC → AC): Top-Down Prediction                      │
│  ────────────────────────────────────────                       │
│                                                                 │
│  Atomic truth (DC) → Structure prediction → Detail generation  │
│       ↓                      ↓                    ↓             │
│  "Something exists"    "It's a ring"      "Ring at (x,y)"     │
│       ↓                      ↓                    ↓             │
│  Low-freq bands       Mid-freq bands      High-freq bands      │
│                                                                 │
│  This is like inverting DC to AC:                              │
│  • Stable knowledge (DC) generates predictions                 │
│  • Predictions are high-resolution details (AC)                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. Potential and Kinetic: Energy States

### 4.1 Potential Energy in the System

```
POTENTIAL = STORED, LATENT, WAITING:

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  WHERE POTENTIAL ENERGY RESIDES:                                │
│                                                                 │
│  1. DC MANIFOLD (Band 0)                                        │
│     Existence state—the most compressed truth                 │
│     Almost never changes                                        │
│     Maximum potential, minimum kinetic                          │
│                                                                 │
│  2. LOW-FREQUENCY BANDS (Bands 1-2)                            │
│     Structure, identity, category                               │
│     Slow to change, protected                                   │
│     High potential                                              │
│                                                                 │
│  3. LEARNED WEIGHTS (The Manifold)                              │
│     All knowledge accumulated over training                    │
│     Changes only through gradient updates                      │
│     The "cold storage" of the system                           │
│                                                                 │
│  4. THE BELIEF STATE (Before Collapse)                          │
│     Probability distribution over hypotheses                   │
│     Uncertainty = unreleased potential                         │
│     Waiting for evidence to trigger collapse                   │
│                                                                 │
│  ────────────────────────────────────────────────────────────  │
│                                                                 │
│  ANALOGY:                                                       │
│                                                                 │
│  Potential energy in physics:                                   │
│  • A ball at the top of a hill                                 │
│  • A compressed spring                                          │
│  • Charge stored in a capacitor                                │
│                                                                 │
│  In our system:                                                 │
│  • Knowledge stored in low-frequency manifolds                 │
│  • Belief state with entropy > 0 (not yet collapsed)          │
│  • Compressed atomic truths ready to guide prediction          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 Kinetic Energy in the System

```
KINETIC = ACTIVE, FLOWING, PROCESSING:

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  WHERE KINETIC ENERGY RESIDES:                                  │
│                                                                 │
│  1. HIGH-FREQUENCY BANDS (Bands 5-6)                           │
│     Details actively being processed                           │
│     Fast turnover, quick forgetting                            │
│     Maximum kinetic, minimum potential                         │
│                                                                 │
│  2. TEMPORAL FLOW                                               │
│     Events arriving, window sliding                            │
│     The monotonic counter incrementing                         │
│     The "current" of the system                                │
│                                                                 │
│  3. ATTENTION COMPUTATION                                       │
│     Query-key matching in progress                             │
│     Softmax computing, values aggregating                      │
│     Active processing                                           │
│                                                                 │
│  4. GRADIENT UPDATES                                            │
│     Weights changing during learning                           │
│     Backpropagation flowing                                    │
│     Learning in motion                                          │
│                                                                 │
│  5. CONTROL OUTPUT                                              │
│     The prediction being generated                             │
│     The action being taken                                     │
│     Energy released into the world                             │
│                                                                 │
│  ────────────────────────────────────────────────────────────  │
│                                                                 │
│  ANALOGY:                                                       │
│                                                                 │
│  Kinetic energy in physics:                                     │
│  • A ball rolling down the hill                                │
│  • A spring releasing                                          │
│  • Current flowing from the capacitor                          │
│                                                                 │
│  In our system:                                                 │
│  • Details flowing through high-frequency bands                │
│  • Attention actively selecting from history                   │
│  • Belief collapsing to certainty                              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 4.3 Energy Conversion

```
THE POTENTIAL ↔ KINETIC CONVERSION:

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  KINETIC → POTENTIAL (Storage)                                 │
│  ─────────────────────────────                                  │
│                                                                 │
│  When: After processing, during culling                        │
│                                                                 │
│  High-freq (kinetic) → culled → Low-freq (potential)          │
│  Details processed → essence extracted → stored                │
│                                                                 │
│  Example:                                                       │
│  • Frame actively processed (kinetic)                          │
│  • Pattern recognized, details discarded                       │
│  • "Ring exists" stored in DC (potential)                      │
│                                                                 │
│  ────────────────────────────────────────────────────────────  │
│                                                                 │
│  POTENTIAL → KINETIC (Retrieval)                               │
│  ───────────────────────────────                                │
│                                                                 │
│  When: During prediction, top-down guidance                    │
│                                                                 │
│  Low-freq (potential) → generates → High-freq (kinetic)       │
│  Stored knowledge → prediction → active output                 │
│                                                                 │
│  Example:                                                       │
│  • "Ring exists" (potential)                                   │
│  • Model predicts ring edges (kinetic)                         │
│  • Prediction compared to observation                          │
│                                                                 │
│  ────────────────────────────────────────────────────────────  │
│                                                                 │
│  THE CONSERVATION PRINCIPLE:                                    │
│                                                                 │
│  Total information is conserved (approximately).               │
│  Form changes: explicit ↔ implicit, kinetic ↔ potential       │
│  Amount stays constant: what's learned, stays learned          │
│                                                                 │
│  See: foundations/EQUILIBRIUM_AND_CONSERVATION.md              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 5. The Pump Cycle

### 5.1 The Three Phases

```
THE PUMP CYCLE: Tension → Discharge → Recovery

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  This is the fundamental temporal rhythm of the system.        │
│                                                                 │
│  ════════════════════════════════════════════════════════════  │
│                                                                 │
│  PHASE 1: TENSION (Potential Builds)                           │
│  ───────────────────────────────────                            │
│                                                                 │
│  • Observations arrive                                          │
│  • Uncertainty accumulates                                      │
│  • Belief state spreads over multiple hypotheses              │
│  • Error (prediction vs reality) is high                       │
│  • The system is "charged up" with unreleased potential       │
│                                                                 │
│  POMDP view: H(b) is high, belief is uncertain                │
│  Energy view: Potential building, kinetic low                  │
│  Visual: Error map shows wide, diffuse pattern                 │
│                                                                 │
│  ────────────────────────────────────────────────────────────  │
│                                                                 │
│  PHASE 2: DISCHARGE (Potential → Kinetic)                      │
│  ────────────────────────────────────────                       │
│                                                                 │
│  • Evidence accumulates past threshold                         │
│  • One hypothesis dominates                                    │
│  • Belief collapses: b(s) → δ(s - s*)                        │
│  • Error drops sharply                                          │
│  • The system "releases" into certainty                        │
│                                                                 │
│  POMDP view: H(b) drops below threshold, collapse triggered   │
│  Energy view: Potential converts to kinetic (action/output)   │
│  Visual: Error map concentrates to a point                     │
│                                                                 │
│  ────────────────────────────────────────────────────────────  │
│                                                                 │
│  PHASE 3: RECOVERY (Reset for Next Cycle)                      │
│  ────────────────────────────────────────                       │
│                                                                 │
│  • Prediction made, output produced                            │
│  • New observation arrives                                      │
│  • Uncertainty begins accumulating again                       │
│  • Belief spreads for new prediction                           │
│  • System returns to tension phase                             │
│                                                                 │
│  POMDP view: New observation, belief update begins             │
│  Energy view: Kinetic dissipates, potential begins building   │
│  Visual: New error front forms                                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 The Pump Cycle as Emergent Oscillation of Belief Entropy

```
THE PUMP CYCLE IS NOT THE AC—IT CREATES THE OSCILLATION:

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  CLARIFICATION:                                                 │
│                                                                 │
│  AC = the frequency content of representations                  │
│       (phase encoding, gradients, high-freq bands)              │
│                                                                 │
│  PUMP CYCLE = the emergent oscillation of belief entropy        │
│               that drives energy conversion between             │
│               potential and kinetic forms                       │
│                                                                 │
│  The pump cycle is not itself "the AC" but rather the          │
│  temporal rhythm that DRIVES the system's dynamics.            │
│                                                                 │
│  ════════════════════════════════════════════════════════════  │
│                                                                 │
│  WHAT THE PUMP CYCLE IS:                                        │
│                                                                 │
│  An emergent oscillation of belief entropy H(b) over time:     │
│                                                                 │
│     H(b)                                                        │
│      │                                                          │
│ High │    ╱╲        ╱╲        ╱╲                               │
│      │   ╱  ╲      ╱  ╲      ╱  ╲                              │
│      │  ╱    ╲    ╱    ╲    ╱    ╲                             │
│ Low  │ ╱      ╲  ╱      ╲  ╱      ╲                            │
│      │╱        ╲╱        ╲╱        ╲                           │
│      └────────────────────────────────────► Event Time t       │
│        │    │    │    │    │    │                              │
│      T─D──R T─D──R T─D──R                                       │
│                                                                 │
│  T = Tension (entropy rising, belief spreading)                │
│  D = Discharge (entropy drops, belief concentrates)            │
│  R = Recovery (reset for next cycle)                           │
│                                                                 │
│  ════════════════════════════════════════════════════════════  │
│                                                                 │
│  EQUIVALENT TERMINOLOGY:                                        │
│                                                                 │
│  Our terminology:    COLLAPSE_GENERALIZATION:      POMDP:      │
│  ──────────────────  ──────────────────────────    ─────       │
│  Tension             Accumulation                  H(b) rising │
│  Discharge           Criticality → Collapse        H(b) → 0    │
│  Recovery            Reset                         New obs     │
│                                                                 │
│  These describe the same phenomenon from different views.      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2.1 Cycle Period as Control Parameter

```
THE CYCLE PERIOD IS NOT FIXED—IT IS A CONTROL PARAMETER:

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  FACTORS DETERMINING CYCLE PERIOD:                              │
│                                                                 │
│  1. SNR THRESHOLD (τ_snr)                                       │
│     Lower threshold → faster collapse → shorter cycle          │
│     Higher threshold → slower collapse → longer cycle          │
│     This is a tunable parameter.                                │
│                                                                 │
│  2. OBSERVATION NOISE (σ_obs)                                   │
│     Higher noise → more observations needed for certainty      │
│     → longer cycle                                              │
│     Lower noise → quick belief concentration → shorter cycle   │
│                                                                 │
│  3. PATTERN COMPLEXITY                                          │
│     Simple patterns (e.g., "something exists") → fast collapse │
│     Complex patterns (e.g., "ring at (x,y) moving NE")         │
│     → more evidence needed → longer cycle                      │
│                                                                 │
│  4. WINDOW SIZE (T)                                             │
│     Bounds maximum evidence accumulation                        │
│     Cycle period ≤ T (cannot span beyond window)              │
│                                                                 │
│  ────────────────────────────────────────────────────────────  │
│                                                                 │
│  RELATIONSHIP TO ATTENTION EVENTS:                              │
│                                                                 │
│  Each pump cycle corresponds to:                                │
│  • Multiple temporal attention events (evidence gathering)     │
│  • One or more wormhole activations (pattern matching)        │
│  • A collapse decision (reactive threshold check)              │
│  • A winner selection (knowledge-informed choice)              │
│                                                                 │
│  These discrete attention events are entities in the system.   │
│  Their orchestration follows the pump cycle rhythm.            │
│                                                                 │
│  (See: architecture_base/collapse/COLLAPSE_DYNAMICS.md)        │
│  (See: architecture_base/attention/spectral_wormhole/)         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 5.3 Connection to Belief Dynamics

```
THE PUMP CYCLE IN POMDP TERMS:

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  From POMDP_SIM.md and POMDP_ATTENTION.md:                     │
│                                                                 │
│  TENSION = BELIEF SPREADING                                     │
│  ───────────────────────────                                    │
│  • Multiple hypotheses remain plausible                        │
│  • b(s) is spread (high entropy)                               │
│  • Error map shows the "wave packet" of uncertainty            │
│                                                                 │
│  The prediction sits at the MEAN of the belief:                │
│  ŷ = E[x_{t+1}] = Σ_s b(s) × s                                │
│                                                                 │
│  Because multiple futures are weighted, the mean is            │
│  not at any single hypothesis—it's the hedged guess.          │
│                                                                 │
│  ────────────────────────────────────────────────────────────  │
│                                                                 │
│  DISCHARGE = BELIEF COLLAPSE                                    │
│  ───────────────────────────                                    │
│  • Multiplicative Bayesian updates compound                    │
│  • One hypothesis dominates exponentially                      │
│  • b(s) → δ(s - s*) (delta at true state)                    │
│                                                                 │
│  This is SUDDEN because Bayesian updates are multiplicative:  │
│  b_N(s*) / b_N(s) = ∏ᵢ [L_i(s*) / L_i(s)]                    │
│  Small per-step ratios compound to huge final ratios.          │
│                                                                 │
│  ────────────────────────────────────────────────────────────  │
│                                                                 │
│  RECOVERY = BELIEF RESET                                        │
│  ────────────────────────                                       │
│  • New observation shifts focus                                │
│  • New prediction target emerges                               │
│  • Belief re-spreads for new uncertainty                       │
│                                                                 │
│  The cycle repeats with t → t+1                               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 6. Temporal Nyquist: The Speed Limit

### 6.1 The Temporal Nyquist Limit

```
THE SPEED LIMIT OF TEMPORAL PERCEPTION:

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  From THE_SPECTRE_OF_NYQUIST_SHANNON.md:                       │
│                                                                 │
│  THEOREM:                                                       │
│                                                                 │
│  For a temporal signal sampled with T frames:                  │
│                                                                 │
│  Maximum resolvable frequency = T/2 cycles per window          │
│                                                                 │
│  ────────────────────────────────────────────────────────────  │
│                                                                 │
│  IMPLICATIONS:                                                  │
│                                                                 │
│  • Patterns with period > T frames: INVISIBLE                 │
│    The window is too short to see one full cycle               │
│                                                                 │
│  • Patterns with period = T/2 frames: BARELY VISIBLE          │
│    Exactly at the Nyquist limit                                │
│                                                                 │
│  • Patterns with period < T/2 frames: WELL RESOLVED           │
│    Multiple cycles visible, reliable detection                 │
│                                                                 │
│  ────────────────────────────────────────────────────────────  │
│                                                                 │
│  DESIGN GUIDELINE:                                              │
│                                                                 │
│  T should be ≥ 2 × (period of slowest relevant pattern)       │
│                                                                 │
│  If an object takes 10 frames to complete a motion:            │
│  T ≥ 20 to reliably detect the pattern                        │
│                                                                 │
│  If daily cycles matter and you sample once per hour:          │
│  T ≥ 48 samples (2 days minimum)                              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 6.2 Aliasing and Its Consequences

```
TEMPORAL ALIASING:

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  WHAT HAPPENS ABOVE NYQUIST:                                    │
│                                                                 │
│  Fast patterns appear as slow patterns (or artifacts).         │
│                                                                 │
│  EXAMPLE:                                                       │
│  • True pattern: Ring rotating once every 3 frames             │
│  • Window size: T = 4 frames                                   │
│  • Nyquist limit: T/2 = 2 cycles per window                   │
│  • Actual frequency: 4/3 ≈ 1.33 cycles per window ✓          │
│                                                                 │
│  Pattern is below Nyquist—correctly perceived.                │
│                                                                 │
│  EXAMPLE:                                                       │
│  • True pattern: Flicker alternating every frame               │
│  • Window size: T = 4 frames                                   │
│  • Nyquist limit: T/2 = 2 cycles per window                   │
│  • Actual frequency: 4 cycles per window (2× Nyquist!)        │
│                                                                 │
│  Pattern ALIASES: appears as lower frequency or constant.     │
│                                                                 │
│  ────────────────────────────────────────────────────────────  │
│                                                                 │
│  INFORMATION DESTRUCTION:                                       │
│                                                                 │
│  Aliased information is DESTROYED, not recoverable.           │
│  Once aliased, there is no way to distinguish the true        │
│  fast pattern from the apparent slow pattern.                  │
│                                                                 │
│  This is why T matters: it sets what can be learned.          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 7. Learning Rates Across Time

### 7.1 Differential Learning by Band

```
LEARNING RATE GRADIENT ACROSS SPECTRAL BANDS:

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  From SPECTRAL_ATTENTION.md:                                    │
│                                                                 │
│  BAND     FREQUENCY    LEARNING RATE    REASON                 │
│  ────     ─────────    ─────────────    ──────                 │
│                                                                 │
│  0 (DC)   Zero         0.00001          Protected (existence)  │
│  1        Very Low     0.0001           Highly stable          │
│  2        Low          0.0003           Stable structure       │
│  3        Mid-Low      0.001            Moderate adaptation    │
│  4        Mid          0.003            Responsive             │
│  5        Mid-High     0.01             Fast adaptation        │
│  6        High         0.03             Very responsive        │
│                                                                 │
│  ────────────────────────────────────────────────────────────  │
│                                                                 │
│  THE GRADIENT:                                                  │
│                                                                 │
│  Low bands: Learn SLOW (protect what's known)                  │
│  High bands: Learn FAST (adapt to new details)                 │
│                                                                 │
│  This creates STABILITY in identity while allowing             │
│  FLEXIBILITY in details.                                        │
│                                                                 │
│  Without this gradient:                                         │
│  • All bands same rate → catastrophic forgetting              │
│  • DC changes too fast → identity lost                        │
│  • High-freq too slow → can't adapt to new textures          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 7.2 Connection to Potential/Kinetic

```
LEARNING RATE = ENERGY MOBILITY:

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  SLOW LEARNING = HIGH INERTIA = POTENTIAL                      │
│  ────────────────────────────────────────                       │
│                                                                 │
│  DC band:                                                       │
│  • Very slow learning rate                                      │
│  • Resists change (high inertia)                               │
│  • Stores potential energy (stable knowledge)                  │
│  • Changes only under sustained pressure                       │
│                                                                 │
│  Like a heavy flywheel: hard to start, hard to stop           │
│                                                                 │
│  ────────────────────────────────────────────────────────────  │
│                                                                 │
│  FAST LEARNING = LOW INERTIA = KINETIC                         │
│  ────────────────────────────────────────                       │
│                                                                 │
│  High-frequency bands:                                          │
│  • Very fast learning rate                                      │
│  • Quick to change (low inertia)                               │
│  • Carries kinetic energy (active processing)                  │
│  • Responds immediately to new information                     │
│                                                                 │
│  Like a light particle: easy to accelerate, easy to redirect  │
│                                                                 │
│  ────────────────────────────────────────────────────────────  │
│                                                                 │
│  THE ENERGY FLOW:                                               │
│                                                                 │
│  Information enters at high-freq (kinetic)                     │
│  → Processed rapidly (fast learning)                           │
│  → Essence extracted and pushed to lower bands                 │
│  → Stored at DC (potential, slow learning protects it)        │
│                                                                 │
│  This is how kinetic becomes potential over time.              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 8. Integration with the 7+1 Architecture

### 8.1 The 7+1 Architecture: Seven Spectral Bands + One Temporal Band

```
THE SPECTRAL HIERARCHY AND TIME (7+1 ARCHITECTURE):

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  From THE_SEVEN_PLUS_ONE_ARCHITECTURE.md and SPECTRAL_ATTENTION │
│                                                                 │
│  SPECTRAL BANDS (0-6): Spatial/Frequency Content               │
│  ────────────────────────────────────────────────               │
│                                                                 │
│  BAND   NAME       TIMESCALE       ENERGY STATE                │
│  ────   ────       ─────────       ────────────                │
│                                                                 │
│  0      DC         ∞ (eternal)     Maximum Potential           │
│  1      Very Low   Very long       High Potential              │
│  2      Low        Long            Potential                   │
│  3      Mid-Low    Medium          Mixed                       │
│  4      Mid        Short           Mixed                       │
│  5      Mid-High   Very short      Kinetic                     │
│  6      High       Immediate       Maximum Kinetic             │
│                                                                 │
│  ────────────────────────────────────────────────────────────  │
│                                                                 │
│  TEMPORAL BAND (7): Dedicated Time Processing                  │
│  ─────────────────────────────────────────────                  │
│                                                                 │
│  BAND   NAME       ROLE            ATTENTION TYPE              │
│  ────   ────       ────            ──────────────              │
│                                                                 │
│  7      Temporal   Time dynamics   CAUSAL (past only)          │
│                                                                 │
│  Band 7 is ORTHOGONAL to Bands 0-6:                            │
│  • Spectral bands ask: "What patterns at this frequency?"      │
│  • Temporal band asks: "How do things change over time?"       │
│  • This is Heisenberg: precise frequency OR precise time       │
│                                                                 │
│  See: architecture_base/attention/temporal_attention/          │
│                                                                 │
│  ────────────────────────────────────────────────────────────  │
│                                                                 │
│  TIMESCALE MEANING (SPECTRAL BANDS):                           │
│                                                                 │
│  • DC changes on timescale of entire training                  │
│  • Low bands change on timescale of epochs                     │
│  • Mid bands change on timescale of batches                    │
│  • High bands change on timescale of steps                     │
│                                                                 │
│  Each band has its own "clock speed":                          │
│  High bands tick fast, DC ticks slow.                          │
│                                                                 │
│  TEMPORAL BAND (7):                                             │
│  • Tracks dynamics ACROSS time (phase velocity, coherence)    │
│  • Uses causal attention (lower-triangular mask)               │
│  • Communicates with all spectral bands via wormhole           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 8.2 Information Flow Through Time

```
HOW INFORMATION AGES:

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  NEW OBSERVATION ARRIVES (Event t):                            │
│                                                                 │
│  1. ENTERS HIGH-FREQ BANDS                                      │
│     • Full detail preserved                                     │
│     • Maximum kinetic energy                                    │
│     • Fast turnover (soon forgotten)                           │
│                                                                 │
│  2. PATTERN RECOGNIZED (Events t to t+k):                      │
│     • Essence extracted                                         │
│     • Details culled                                            │
│     • Moves to mid-frequency bands                             │
│                                                                 │
│  3. STRUCTURE CRYSTALLIZES (Events t+k to t+m):                │
│     • Pattern becomes stable                                    │
│     • Moves to low-frequency bands                             │
│     • Becomes part of identity                                  │
│                                                                 │
│  4. ATOMIC TRUTH FORMS (Events >> t):                          │
│     • Maximum compression achieved                             │
│     • Resides in DC                                            │
│     • Survives indefinitely                                     │
│                                                                 │
│  ────────────────────────────────────────────────────────────  │
│                                                                 │
│  THE FLOW:                                                      │
│                                                                 │
│  NEW → HIGH → MID → LOW → DC                                   │
│  KINETIC ──────────────────► POTENTIAL                         │
│  AC ──────────────────────► DC                                 │
│                                                                 │
│  This is rectification: extracting the DC from the AC.        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 9. Summary

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│     T H E   T E M P O R A L   S Y S T E M                      │
│                                                                 │
│  ═══════════════════════════════════════════════════════════   │
│                                                                 │
│  EVENT TIME:                                                    │
│  ───────────                                                    │
│  • Monotonic counter, not wall clock                           │
│  • t increments by 1 for each new observation                  │
│  • Causality: can only attend to s_key < s_query              │
│  • All temporal masks use event indices                        │
│                                                                 │
│  ═══════════════════════════════════════════════════════════   │
│                                                                 │
│  DC AND AC (Electrical Engineering Analogy):                   │
│  ───────────────────────────────────────────                   │
│  • DC = zero frequency = existence = what persists             │
│  • AC = non-zero frequency = dynamics = what changes           │
│  • Culling cascade = rectification (AC → DC)                   │
│  • Top-down prediction = inversion (DC → AC)                   │
│  • NOTE: This is an analogy. Future work will formalize        │
│    in proper EE terms for hardware implementations.            │
│                                                                 │
│  ═══════════════════════════════════════════════════════════   │
│                                                                 │
│  POTENTIAL AND KINETIC:                                         │
│  ──────────────────────                                         │
│  • Potential = stored, latent (low-freq bands, manifold)      │
│  • Kinetic = active, processing (high-freq bands, attention)  │
│  • Conversion: details processed → essence stored             │
│  • Conservation: total information approximately constant      │
│                                                                 │
│  ═══════════════════════════════════════════════════════════   │
│                                                                 │
│  THE PUMP CYCLE (Emergent Oscillation of Belief Entropy):      │
│  ─────────────────────────────────────────────────────         │
│  • Tension: uncertainty accumulates, belief spreads            │
│  • Discharge: evidence triggers collapse, belief concentrates  │
│  • Recovery: new observation, cycle resets                     │
│                                                                 │
│  Equivalent to: Accumulation → Criticality → Collapse          │
│  (See architecture_base/collapse/COLLAPSE_GENERALIZATION.md)   │
│                                                                 │
│  The pump cycle is NOT the AC itself. It is the emergent       │
│  oscillation that DRIVES energy conversion between potential   │
│  and kinetic forms. DC alone is static. The pump cycle is      │
│  what makes the system live.                                   │
│                                                                 │
│  Cycle period is a CONTROL PARAMETER, determined by:           │
│  • SNR threshold (τ_snr)                                       │
│  • Observation noise (σ_obs)                                   │
│  • Pattern complexity                                          │
│  • Window size (T)                                             │
│                                                                 │
│  ═══════════════════════════════════════════════════════════   │
│                                                                 │
│  TEMPORAL NYQUIST:                                              │
│  ────────────────                                               │
│  • Maximum temporal frequency = T/2 cycles per window          │
│  • Patterns slower than this are invisible                     │
│  • Choose T based on slowest relevant dynamics                 │
│                                                                 │
│  ═══════════════════════════════════════════════════════════   │
│                                                                 │
│  LEARNING RATE GRADIENT:                                        │
│  ───────────────────────                                        │
│  • DC: slowest (0.00001) — protected                          │
│  • High-freq: fastest (0.03) — responsive                     │
│  • This creates stability + adaptability                       │
│                                                                 │
│  ═══════════════════════════════════════════════════════════   │
│                                                                 │
│  THE CORE INSIGHT:                                              │
│                                                                 │
│  Time in the system is not measured in seconds.                │
│  It is measured in EVENTS—discrete learning opportunities.     │
│                                                                 │
│  The temporal flow creates:                                     │
│  • The pump cycle (emergent oscillation of belief entropy)     │
│  • The kinetic energy (active processing in high-freq bands)  │
│  • The path from observation to knowledge (rectification)      │
│                                                                 │
│  THE 7+1 ARCHITECTURE:                                          │
│                                                                 │
│  • Bands 0-6 (Spectral): WHAT patterns at each frequency?      │
│  • Band 7 (Temporal): HOW do things change over time?          │
│  • These are ORTHOGONAL (Heisenberg uncertainty)               │
│  • Spectral Wormhole connects them                             │
│                                                                 │
│  DC provides the ground reference (existence, potential).      │
│  AC provides the dynamics (change, kinetic).                   │
│  Band 7 tracks the temporal dynamics explicitly.               │
│  The pump cycle drives conversion between them.                │
│  Together, they form a complete temporal system.               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## References

### Architecture Documents

| Document | Location | Relevance |
|----------|----------|-----------|
| SPECTRAL_ATTENTION | `architecture_base/attention/spectral_attention/` | Bands 0-6 architecture and learning rates |
| TEMPORAL_ATTENTION | `architecture_base/attention/temporal_attention/` | Band 7 causal attention |
| ATTENTION_STACK | `architecture_base/attention/` | How all 8 bands integrate |
| THE_SEVEN_PLUS_ONE_ARCHITECTURE | `architecture_theoretical/` | Why 7+1 bands |
| ORTHOGONALITY | `architecture_theoretical/` | Space-time orthogonality (Heisenberg) |

### Foundation Documents

| Document | Location | Relevance |
|----------|----------|-----------|
| EQUILIBRIUM_AND_CONSERVATION | `foundations/` | Energy conservation in learning |
| KNOWLEDGE_AND_REACTIVITY | `foundations/` | Knowledge (geometry) vs Reactive (energy) |

### POMDP Documents

| Document | Location | Relevance |
|----------|----------|-----------|
| POMDP_SIM | `pomdp/` | Pump cycle dynamics and belief state evolution |
| THE_OLD_LADY_AND_THE_TIGER | `pomdp/` | Culling and compression mechanics |

### Information Theory

| Document | Location | Relevance |
|----------|----------|-----------|
| THE_SPECTRE_OF_NYQUIST_SHANNON | `information_theory/` | Temporal Nyquist limits |

---

*Oscar Goldman — Shogu Research Group @ Datamutant.ai*

*"Time is not seconds. Time is events. The pump cycle is an emergent oscillation of belief entropy—the heartbeat that drives energy between potential and kinetic. DC is the ground reference. AC is the dynamics. Band 7 tracks the flow. Together, they live."*

