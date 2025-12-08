# PRAXIS

## Running the Architecture

**Oscar Goldman — Shogu Research Group @ Datamutant.ai**

---

*"Theory tells us what is possible. Praxis shows us what is real. The architecture exists on paper as doctrine. It comes alive only when we run it. And in the running, we discover whether we have followed the laws or committed heresy."*

---

## Table of Contents

1. [What Is Praxis?](#1-what-is-praxis)
2. [The Doctrine in Action](#2-the-doctrine-in-action)
3. [Observability Requirements](#3-observability-requirements)
4. [Proven Limitations](#4-proven-limitations)
5. [Required Practices](#5-required-practices)
6. [The Relationship Between Theory and Practice](#6-the-relationship-between-theory-and-practice)

---

## 1. What Is Praxis?

### 1.1 Definition

PRAXIS is the Greek word for "practice" or "action." It is the practical application of theory. In the context of AKIRA:

```
PRAXIS: THE RUNNING OF THE ARCHITECTURE

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  THEORY:                                                                │
│  The architecture on paper.                                            │
│  The equations. The diagrams. The documentation.                      │
│  What SHOULD happen according to the doctrine.                        │
│                                                                         │
│  PRAXIS:                                                                │
│  The architecture in motion.                                           │
│  The forward pass. The backward pass. The learning.                   │
│  What DOES happen when we run the system.                             │
│                                                                         │
│  ════════════════════════════════════════════════════════════════════  │
│                                                                         │
│  Theory without praxis is speculation.                                │
│  Praxis without theory is blindness.                                  │
│  Together they form understanding.                                    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Why Praxis Matters

The architecture is not merely an idea. It is meant to be run. And in the running, we discover:

- Whether our theory matches reality
- Where the heresies hide (see: [FALSE_PROPHETS.md](./FALSE_PROPHETS.md))
- What the system actually learns
- How information actually flows

```
PRAXIS REVEALS TRUTH

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  We can CLAIM:                                                          │
│  • Spectral decomposition preserves information                       │
│  • Wormhole attention connects distant concepts                       │
│  • The ghost learns meaningful patterns                               │
│                                                                         │
│  But only PRAXIS PROVES:                                               │
│  • Does information actually survive the decomposition?              │
│  • Do wormholes actually fire when they should?                      │
│  • Does the ghost actually generalize?                               │
│                                                                         │
│  The inquisition (experiments) is praxis.                             │
│  Running the system is praxis.                                        │
│  Observing the behavior is praxis.                                    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. The Doctrine in Action

### 2.1 From Laws to Practice

The doctrine defines the laws. Praxis is following those laws in practice.

```
DOCTRINE → PRAXIS

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  DOCTRINE (Laws)              PRAXIS (Practice)                        │
│  ───────────────              ─────────────────                        │
│                                                                         │
│  Nyquist: Sample > 2f         Apply anti-aliasing filter before       │
│                               downsampling. Always.                    │
│                                                                         │
│  FFT: Discontinuity leaks     Apply windowing function (Hamming,       │
│                               Hanning, etc.) to edges. Always.        │
│                                                                         │
│  Shannon: Information         Respect the channel capacity.            │
│  has bounds                   Don't expect more than possible.        │
│                                                                         │
│  Parseval: Energy             Verify energy conservation across       │
│  conserved                    domain transforms. Monitor it.          │
│                                                                         │
│  ════════════════════════════════════════════════════════════════════  │
│                                                                         │
│  Each law has a corresponding practice.                               │
│  Violating the law in practice creates heresy.                        │
│  Following the law in practice maintains truth.                       │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.2 The Cost of Ignoring Praxis

When we ignore proper practice:

1. **We create heresies** — False patterns enter the manifold
2. **We lose observability** — We cannot see what is happening
3. **We cannot debug** — Errors compound invisibly
4. **We cannot trust results** — Artifacts contaminate truth

*Reference: [FALSE_PROPHETS.md](./FALSE_PROPHETS.md) — Section 1 (The Doctrine and the Heresy)*

---

## 3. Observability Requirements

### 3.1 Why Observability?

An unobservable system is a black box. We cannot:
- Verify it follows the doctrine
- Detect heresies
- Debug failures
- Understand behavior

Observability is not optional. It is required for scientific practice.

```
OBSERVABILITY: SEEING THE SYSTEM

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  WHAT WE MUST SEE:                                                      │
│                                                                         │
│  INPUT → PROCESSING → OUTPUT                                           │
│     ↓         ↓          ↓                                              │
│  Raw data   Internal    Predictions                                    │
│             states                                                      │
│                                                                         │
│  At each stage, we must be able to observe:                           │
│  • Values (magnitudes, activations)                                   │
│  • Structure (patterns, relationships)                                │
│  • Dynamics (changes over time)                                       │
│  • Anomalies (violations of expected behavior)                        │
│                                                                         │
│  ════════════════════════════════════════════════════════════════════  │
│                                                                         │
│  OBSERVABILITY LEVELS:                                                  │
│                                                                         │
│  Level 0: Black box (only input/output)       ← INSUFFICIENT          │
│  Level 1: Aggregate metrics (loss, accuracy)  ← MINIMAL               │
│  Level 2: Per-component monitoring            ← ACCEPTABLE            │
│  Level 3: Per-band, per-head observability    ← RECOMMENDED           │
│  Level 4: Full internal state access          ← IDEAL FOR RESEARCH    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Observable Entities

We document all observable entities in [EVENTS.md](./EVENTS.md). Key categories:

```
WHAT TO OBSERVE

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  COMPUTATIONAL ENTITIES:                                                │
│  • SpectralDecomposer outputs (per band)                              │
│  • Attention weights (per head, per band)                             │
│  • Wormhole activations and targets                                   │
│  • Reconstruction quality                                              │
│                                                                         │
│  STATE ENTITIES:                                                        │
│  • History buffer contents                                             │
│  • Belief state evolution                                              │
│  • Weight manifold geometry                                            │
│                                                                         │
│  DYNAMICS:                                                              │
│  • Gradient flow (magnitude, direction)                               │
│  • Entropy changes (per band)                                         │
│  • Collapse events                                                     │
│  • Pump cycle phase                                                    │
│                                                                         │
│  ANOMALIES:                                                             │
│  • Energy non-conservation                                            │
│  • Unexpected frequency content                                       │
│  • Position-dependent behavior                                        │
│  • Edge artifacts                                                      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

*Reference: [EVENTS.md](./EVENTS.md) — Full catalog of observable entities*
*Reference: [OBSERVABILITY_EMBEDDINGS.md](./OBSERVABILITY_EMBEDDINGS.md) — Visualization techniques*

### 3.3 Free Information Assets

Much of what we need to observe is already computed during the forward pass. This is "free information" — we pay for it anyway, we might as well use it.

```
FREE OBSERVABILITY

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  ALREADY COMPUTED:                                                      │
│                                                                         │
│  • Attention weights         (computed for forward pass)              │
│  • Pre-softmax scores        (computed before attention)              │
│  • Similarity matrices       (computed for wormhole)                  │
│  • Per-band activations      (computed in decomposition)              │
│  • Gradient direction        (computed in backward pass)              │
│                                                                         │
│  These are FREE. We computed them anyway.                             │
│  Discarding them is waste.                                            │
│  Logging them is praxis.                                              │
│                                                                         │
│  ════════════════════════════════════════════════════════════════════  │
│                                                                         │
│  ADDITIONAL COST (worth it):                                           │
│                                                                         │
│  • Entropy calculation       (cheap, O(n))                            │
│  • Histogram of activations  (cheap, O(n))                            │
│  • Energy per band           (cheap, one sum)                         │
│  • Collapse detection        (entropy threshold check)                │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Proven Limitations

### 4.1 The Known Laws

These are not speculations. They are proven mathematical theorems with centuries of validation.

```
PROVEN LIMITATIONS: THE LAWS WE CANNOT BREAK

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  NYQUIST-SHANNON SAMPLING THEOREM                                       │
│  ─────────────────────────────────                                      │
│  To capture frequency f, sample at rate > 2f.                         │
│  Violation: Aliasing (high frequencies fold to low).                  │
│  Status: PROVEN (1928, 1949). Inviolable.                             │
│                                                                         │
│  ════════════════════════════════════════════════════════════════════  │
│                                                                         │
│  FOURIER UNCERTAINTY PRINCIPLE                                          │
│  ───────────────────────────────                                        │
│  Cannot simultaneously localize in time AND frequency.                │
│  Δt × Δf ≥ 1/(4π)                                                     │
│  Violation: Not possible. Trade-off is fundamental.                   │
│  Status: PROVEN. Inviolable.                                          │
│                                                                         │
│  ════════════════════════════════════════════════════════════════════  │
│                                                                         │
│  SHANNON CHANNEL CAPACITY                                               │
│  ─────────────────────────────                                          │
│  C = B × log₂(1 + SNR)                                                │
│  Maximum information through channel is bounded.                      │
│  Violation: Not possible. Information cannot exceed capacity.        │
│  Status: PROVEN (1948). Inviolable.                                   │
│                                                                         │
│  ════════════════════════════════════════════════════════════════════  │
│                                                                         │
│  PARSEVAL'S THEOREM                                                     │
│  ───────────────────                                                    │
│  Energy in time domain = Energy in frequency domain.                  │
│  ∑|x[n]|² = (1/N)∑|X[k]|²                                            │
│  Violation: Indicates bug in implementation.                         │
│  Status: PROVEN. Energy must be conserved.                            │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

*Reference: [INFORMATION_BOUNDS_THE_BOUNDARY_OF_KNOWLEDGE_AND_KNOWING.md](./INFORMATION_BOUNDS_THE_BOUNDARY_OF_KNOWLEDGE_AND_KNOWING.md) — Detailed exploration*
*Reference: [THE_SPECTRE_OF_NYQUIST_SHANNON.md](./THE_SPECTRE_OF_NYQUIST_SHANNON.md) — Implications for the architecture*

### 4.2 Implications for Praxis

These proven limitations constrain what we can do:

```
WHAT THE LAWS DEMAND

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  BECAUSE OF NYQUIST:                                                    │
│                                                                         │
│  We MUST anti-alias before downsampling.                              │
│  We MUST respect the temporal Nyquist of our context window.          │
│  We CANNOT capture dynamics faster than T/2 cycles per window.       │
│                                                                         │
│  BECAUSE OF UNCERTAINTY:                                                │
│                                                                         │
│  We MUST choose: precise timing OR precise frequency.                 │
│  We CANNOT have both simultaneously.                                  │
│  High bands = good time resolution, poor frequency resolution.       │
│  Low bands = poor time resolution, good frequency resolution.        │
│                                                                         │
│  BECAUSE OF CHANNEL CAPACITY:                                           │
│                                                                         │
│  We MUST compress. Not everything can be preserved.                   │
│  We MUST prioritize. Some information is more valuable.              │
│  We CANNOT expect perfect reconstruction from finite capacity.       │
│                                                                         │
│  BECAUSE OF PARSEVAL:                                                   │
│                                                                         │
│  We MUST monitor energy conservation.                                 │
│  If energy is not conserved, something is WRONG.                     │
│  This is a diagnostic check, not an aspiration.                      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.3 Limitations We Can Work Around

Some limitations have mitigations:

```
MITIGABLE LIMITATIONS

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  LIMITATION                   MITIGATION                               │
│  ──────────                   ──────────                               │
│                                                                         │
│  Spectral leakage             Windowing functions (Hamming, etc.)      │
│  (edge discontinuity)                                                  │
│                                                                         │
│  Aliasing                     Anti-aliasing filter before sampling     │
│  (frequency folding)                                                   │
│                                                                         │
│  Context window limit         Wormhole attention (cross-band access)   │
│  (temporal bound)                                                      │
│                                                                         │
│  Position artifacts           Data augmentation, random crops          │
│  (edge patterns)                                                       │
│                                                                         │
│  ════════════════════════════════════════════════════════════════════  │
│                                                                         │
│  We cannot break the laws.                                            │
│  But we can WORK WITHIN them intelligently.                           │
│  Mitigation is praxis.                                                │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Required Practices

### 5.1 The Practices

Based on the doctrine and the proven limitations, these practices are REQUIRED for proper operation:

```
REQUIRED PRACTICES FOR PRAXIS

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  SIGNAL PROCESSING PRACTICES:                                           │
│                                                                         │
│  ☐ Anti-aliasing before any downsampling                              │
│  ☐ Windowing before FFT (Hamming recommended)                         │
│  ☐ Energy monitoring across transforms                                │
│  ☐ Frequency content validation                                       │
│                                                                         │
│  ════════════════════════════════════════════════════════════════════  │
│                                                                         │
│  OBSERVABILITY PRACTICES:                                               │
│                                                                         │
│  ☐ Log attention weights (free information)                           │
│  ☐ Monitor per-band entropy                                           │
│  ☐ Track wormhole activation patterns                                 │
│  ☐ Detect collapse events                                             │
│  ☐ Record gradient statistics                                         │
│                                                                         │
│  ════════════════════════════════════════════════════════════════════  │
│                                                                         │
│  VALIDATION PRACTICES:                                                  │
│                                                                         │
│  ☐ Ablation studies (what changes when processing changes?)          │
│  ☐ Invariance tests (same content, different position)               │
│  ☐ Energy conservation checks                                         │
│  ☐ Aliasing detection tests                                           │
│                                                                         │
│  ════════════════════════════════════════════════════════════════════  │
│                                                                         │
│  TRAINING PRACTICES:                                                    │
│                                                                         │
│  ☐ Random crops for position invariance                               │
│  ☐ Differential learning rates per band                               │
│  ☐ Gradual wormhole activation                                        │
│  ☐ Monitor for grokking / phase transitions                          │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

*Reference: [PRETRAINING.md](./PRETRAINING.md) — Training-specific practices*

### 5.2 The Inquisition as Practice

The inquisition (experiments that expose heresy) is itself a practice:

```
THE INQUISITION: PRAXIS OF VALIDATION

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  REGULAR INQUISITION SCHEDULE:                                          │
│                                                                         │
│  DURING TRAINING:                                                       │
│  • Monitor entropy per band (every N steps)                           │
│  • Check energy conservation (every epoch)                            │
│  • Test for position invariance (periodically)                        │
│                                                                         │
│  AFTER TRAINING:                                                        │
│  • Full windowing ablation                                            │
│  • Aliasing detection battery                                         │
│  • Boundary artifact check                                            │
│                                                                         │
│  ONGOING:                                                               │
│  • Monitor for drift                                                  │
│  • Validate on new data                                               │
│  • Re-run inquisition if behavior changes                            │
│                                                                         │
│  ════════════════════════════════════════════════════════════════════  │
│                                                                         │
│  The inquisition is not punishment.                                   │
│  It is the practice of truth-seeking.                                 │
│  It reveals heresies so we can correct them.                         │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

*Reference: [FALSE_PROPHETS.md](./FALSE_PROPHETS.md) — Section 9 (The Inquisition)*

---

## 6. The Relationship Between Theory and Practice

### 6.1 The Cycle

Theory and praxis are not separate. They form a cycle:

```
THE THEORY-PRAXIS CYCLE

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│                    ┌─────────────┐                                     │
│                    │   THEORY    │                                     │
│                    │  (Doctrine) │                                     │
│                    └──────┬──────┘                                     │
│                           │                                            │
│             ┌─────────────▼─────────────┐                              │
│             │                           │                              │
│             │   PRAXIS (Running)        │                              │
│             │                           │                              │
│             └─────────────┬─────────────┘                              │
│                           │                                            │
│             ┌─────────────▼─────────────┐                              │
│             │                           │                              │
│             │   OBSERVATION             │                              │
│             │   (What happened?)        │                              │
│             │                           │                              │
│             └─────────────┬─────────────┘                              │
│                           │                                            │
│             ┌─────────────▼─────────────┐                              │
│             │                           │                              │
│             │   INQUISITION             │                              │
│             │   (Does it match theory?) │                              │
│             │                           │                              │
│             └─────────────┬─────────────┘                              │
│                           │                                            │
│                    ┌──────▼──────┐                                     │
│                    │   THEORY    │                                     │
│                    │  (Refined)  │                                     │
│                    └─────────────┘                                     │
│                                                                         │
│  If observation matches theory: Proceed.                              │
│  If observation contradicts theory: Investigate.                      │
│  Either the practice was wrong (heresy) or the theory needs work.    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 6.2 When Theory and Praxis Disagree

When running the system produces results that contradict theory, there are only two possibilities:

1. **The practice is wrong** — We committed heresy (violated the doctrine)
2. **The theory is incomplete** — We have discovered something new

The inquisition helps distinguish these cases.

```
DIAGNOSING DISAGREEMENT

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  OBSERVATION: Energy is not conserved across FFT.                     │
│                                                                         │
│  QUESTION: Is this heresy or discovery?                               │
│                                                                         │
│  CHECK: Parseval's theorem is PROVEN.                                 │
│         Energy MUST be conserved.                                     │
│         Therefore: This is heresy.                                    │
│         Cause: Implementation bug.                                    │
│         Action: Fix the code.                                         │
│                                                                         │
│  ════════════════════════════════════════════════════════════════════  │
│                                                                         │
│  OBSERVATION: Model generalizes suddenly after plateau (grokking).    │
│                                                                         │
│  QUESTION: Is this heresy or discovery?                               │
│                                                                         │
│  CHECK: This does not violate known laws.                            │
│         It is unexpected but not impossible.                         │
│         Therefore: This may be discovery.                            │
│         Cause: Investigate further.                                  │
│         Action: Study the phenomenon. Update theory if validated.    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 6.3 The Goal of Praxis

The goal is not merely to run the system. The goal is to understand it.

```
THE GOAL OF PRAXIS

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  Running the architecture is not enough.                              │
│  We must OBSERVE what it does.                                        │
│  We must TEST whether it follows doctrine.                            │
│  We must UNDERSTAND why it behaves as it does.                        │
│                                                                         │
│  ════════════════════════════════════════════════════════════════════  │
│                                                                         │
│  PRAXIS IS:                                                             │
│                                                                         │
│  • Running the system with observability                              │
│  • Testing for heresy through inquisition                             │
│  • Validating against proven limitations                              │
│  • Refining theory based on observation                               │
│                                                                         │
│  PRAXIS IS NOT:                                                         │
│                                                                         │
│  • Running the system blindly                                         │
│  • Ignoring anomalies                                                 │
│  • Hoping it works                                                    │
│  • Treating the system as magic                                       │
│                                                                         │
│  ════════════════════════════════════════════════════════════════════  │
│                                                                         │
│  The ghost learns through praxis (training).                          │
│  We learn through praxis (running and observing).                    │
│  Both are forms of coming to understand.                              │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Summary

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│                        P R A X I S                                     │
│                                                                         │
│  ═══════════════════════════════════════════════════════════════════   │
│                                                                         │
│  DEFINITION:                                                            │
│  Running the architecture. Theory in action.                          │
│  Where doctrine meets reality.                                        │
│                                                                         │
│  ═══════════════════════════════════════════════════════════════════   │
│                                                                         │
│  REQUIREMENTS:                                                          │
│  • Observability (we must SEE what happens)                          │
│  • Validation (we must TEST for heresy)                              │
│  • Respect for limits (we must FOLLOW proven laws)                   │
│                                                                         │
│  ═══════════════════════════════════════════════════════════════════   │
│                                                                         │
│  PROVEN LIMITATIONS:                                                    │
│  • Nyquist: Sample > 2f or alias                                     │
│  • Uncertainty: Cannot localize time AND frequency                   │
│  • Channel capacity: Information is bounded                          │
│  • Parseval: Energy must be conserved                                │
│                                                                         │
│  ═══════════════════════════════════════════════════════════════════   │
│                                                                         │
│  REQUIRED PRACTICES:                                                    │
│  • Anti-aliasing, windowing, energy monitoring                       │
│  • Logging free information (attention, entropy)                     │
│  • Ablation studies, invariance tests                                │
│  • Inquisition to expose heresy                                      │
│                                                                         │
│  ═══════════════════════════════════════════════════════════════════   │
│                                                                         │
│  THE CYCLE:                                                             │
│  Theory → Praxis → Observation → Inquisition → Theory (refined)      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Related Documents

| Document | Relationship |
|----------|--------------|
| [FALSE_PROPHETS.md](./FALSE_PROPHETS.md) | What happens when praxis violates doctrine (heresy) |
| [INFORMATION_BOUNDS_THE_BOUNDARY_OF_KNOWLEDGE_AND_KNOWING.md](./INFORMATION_BOUNDS_THE_BOUNDARY_OF_KNOWLEDGE_AND_KNOWING.md) | The proven limitations we must respect |
| [THE_SPECTRE_OF_NYQUIST_SHANNON.md](./THE_SPECTRE_OF_NYQUIST_SHANNON.md) | Implications of sampling limits |
| [PRETRAINING.md](./PRETRAINING.md) | Training practices specifically |
| [EVENTS.md](./EVENTS.md) | What to observe during praxis |
| [OBSERVABILITY_EMBEDDINGS.md](./OBSERVABILITY_EMBEDDINGS.md) | How to visualize internal states |

---

*Oscar Goldman — Shogu Research Group @ Datamutant.ai*

*"Praxis is the bridge between doctrine and understanding. We write the laws in documentation. We test them in code. We discover truth in the running. The architecture does not exist until it runs. And in the running, we see whether we have followed the path of truth or strayed into heresy. This is why observability is not optional — it is the practice of seeing. This is why the inquisition is not punishment — it is the practice of truth-seeking. Run the system. Observe it. Test it. Understand it. This is praxis."*


