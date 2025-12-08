# AKIRA Design Decisions

**Oscar Goldman — Shogu Research Group @ Datamutant.ai**

---

## Epistemological Note

This document captures **key design choices** where the reasoning chain is:

```
ESTABLISHED FOUNDATION → THEORETICAL ARGUMENT → DESIGN CHOICE → TESTABLE PREDICTION
```

Each decision should:
1. Cite established foundations (mathematics, physics, information theory)
2. Present the theoretical argument (logical derivation)
3. Document the design choice and alternatives considered
4. Point to experiments that validate or refute the choice

Some foundations are well-established (Fourier, Shannon, Parseval). The design choices apply these foundations to neural architectures. The experiments test whether the applications are valid.

---

## The Core Dynamic: Tension and Collapse

All design decisions serve the fundamental cycle of belief evolution:

```
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
│  Design choices should enable this cycle, not hinder it.              │
│  See foundations/TERMINOLOGY.md for formal definitions.               │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

This document captures **key design choices** in AKIRA where:
1. The choice is non-obvious
2. Alternative approaches were considered
3. The rationale should be documented for future reference

---

## Table of Contents

1. [Band 6 Temperature: Why τ = 1.0](#1-band-6-temperature-why-τ--10)
2. [Learning Rate Hierarchy: 3000× Ratio](#2-learning-rate-hierarchy-3000-ratio)
3. [Coherence-Gated Wormholes: Entropy vs Similarity](#3-coherence-gated-wormholes-entropy-vs-similarity)
4. [7+1 Architecture: Why Not 6+1 or 8+1](#4-71-architecture-why-not-61-or-81)
5. [Partial Information Decomposition for Cross-Band Analysis](#5-partial-information-decomposition-for-cross-band-analysis)
6. [Duality Methods for Observability](#6-duality-methods-for-observability)

---

## 1. Band 6 Temperature: Why τ = 1.0

### The Question

Why does Band 6 (highest frequency) use τ = 1.0, the same as Band 3 (bridge)?

**Naive expectation:** High-frequency bands should have **higher** temperature (more diffuse attention) because they handle rapidly changing, uncertain details.

**Actual value:** τ = 1.0 (moderate, like Band 3)

### The Knowledge vs Reactive Framework

AKIRA bands operate in **two distinct processing modes**:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  KNOWLEDGE MODE (Bands 1-2, some 3-4)                                  │
│  ────────────────────────────────────────                               │
│  • Geometric, manifold-based processing                               │
│  • "What is this?" queries                                            │
│  • Compares positions on learned manifold                             │
│  • Semantic, relational, structural                                   │
│                                                                         │
│  Benefits from: DIFFUSE attention (explore possibilities)             │
│  Temperature: MODERATE to HIGH (0.7-1.0)                              │
│                                                                         │
│  ═══════════════════════════════════════════════════════════════════   │
│                                                                         │
│  REACTIVE MODE (Bands 0, 5-6)                                          │
│  ───────────────────────────────                                        │
│  • Energy/threshold-based processing                                  │
│  • "Is there signal HERE?" detection                                  │
│  • Responds to magnitude, not geometry                                │
│  • Fast, automatic, local                                             │
│                                                                         │
│  Benefits from: SHARP attention (decisive response)                   │
│  Temperature: LOW to MODERATE (0.5-1.0)                               │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Band-Specific Assignments

```
Band 0 (DC):      τ = 0.5   REACTIVE    (existence = threshold)
Band 1 (VeryLow): τ = 0.6   KNOWLEDGE   (category = manifold query)
Band 2 (Low):     τ = 0.7   KNOWLEDGE   (identity = comparison)
Band 3 (MidLow):  τ = 1.0   HYBRID      (features = manifold + threshold)
Band 4 (Mid):     τ = 0.8   HYBRID      (details = detection + classification)
Band 5 (MidHigh): τ = 0.9   REACTIVE    (fine structure = energy-based)
Band 6 (High):    τ = 1.0   REACTIVE    (edges = pure magnitude)
Band 7 (Time):    τ = 0.8   HYBRID      (temporal = integration)
```

### Band 6 Specifically

**Role:** Edge detection, transients, pixel-level changes, high-frequency noise filtering

**Processing mode:** REACTIVE
- Detects presence/absence of signal
- Energy-based: responds to magnitude
- Threshold-driven: binary-like decisions

**Why τ = 1.0 (not higher):**

1. **Sharp responses needed:** Edge detection requires decisive attention to strong gradients
2. **Not geometric:** Band 6 doesn't query "what is this edge?" — it just detects "there IS an edge"
3. **Fast gating:** Acts as signal gate for lower bands ("is there anything here to process?")
4. **Balance:** τ = 1.0 is sharp enough for detection but not brittle

**Why τ = 1.0 (not lower):**

1. **Avoid over-sharpening:** τ < 1.0 might create brittle, noise-sensitive edges
2. **Allow some exploration:** Small amount of diffusion helps with edge continuity
3. **Learnable parameter:** Will adapt if sharper is actually better

**Contrast with Band 5:**

- Band 5 (τ = 0.9): Still somewhat geometric, fine structure has meaning
- Band 6 (τ = 1.0): Pure energy detection, no semantic content

### Alternative Considered

**Option A:** Make Band 6 τ = 1.2+ (high temperature, diffuse)
- **Rationale:** High frequency = high uncertainty
- **Rejected:** Misunderstands Band 6's role. It's not uncertain about edges — edges ARE the information.

**Option B:** Make Band 6 τ = 0.5 (very sharp, like Band 0)
- **Rationale:** Maximum decisiveness for edge detection
- **Rejected:** Too brittle, might create artifacts

**Option C (CHOSEN):** Band 6 τ = 1.0 (moderate, balanced)
- **Rationale:** Sharp enough for reactive processing, not so sharp it's brittle
- **Benefits:** Learnable, balanced, matches integrative role similar to Band 3

### Validation

**Expected behaviors if this is correct:**

1. ✅ Band 6 activations should be sparse (only at edges/transients)
2. ✅ Band 6 attention should be more peaked than Band 5
3. ✅ Learned τ for Band 6 should stay near 1.0 (not drift to 1.5+)
4. ✅ Band 6 should respond quickly (reactive) not deliberatively

**Experiments:** See `experiments/003_EXP_SPECTRAL_BAND_DYNAMICS.md`

### Status

- ✅ Documented in `CANONICAL_PARAMETERS.md`
- ✅ Explained in `AKIRA_OVERVIEW.md` (Knowledge vs Reactivity section)
- ✅ Implementation matches specification
- ⏭️ TODO: Add explicit test for reactive vs knowledge mode behaviors

---

## 2. Learning Rate Hierarchy: The Two Specifications

### The Question

Why were there **two incompatible learning rate specifications**?

**The Problem Found:**

**Specification A** (absolute values):
- Band 0: 0.00001
- Band 6: 0.03
- **Ratio: 3000×**

**Specification B** (relative multipliers):
- Band 0: lr_base × 0.1
- Band 6: lr_base × 1.2
- **Ratio: 12×** (if lr_base = 0.0001)

These were **incompatible** — experiments couldn't know which to use.

### The Resolution

**RESOLVED:** Established clear hierarchy and purpose for each:

```
SPECIFICATION A (CANONICAL)
─────────────────────────────
Band 0: 0.00001  →  3000× ratio to Band 6
Band 6: 0.03

PURPOSE: Target values for full training
WHEN: All experiments, production systems
WHY: Reflects true timescales (identity vs position)
```

```
SPECIFICATION B (WARM-START)
─────────────────────────────
Band 0: lr_base × 0.1  →  12× ratio to Band 6
Band 6: lr_base × 1.2

PURPOSE: Gentler hierarchy for initial training
WHEN: First 50k-100k steps, then transition to Spec A
WHY: Prevents instability during random initialization
```

### The Rationale for 3000× (Spec A)

**Different timescales for different concepts:**

```
LOW FREQUENCY = SLOW CHANGE (identity, what something IS)
HIGH FREQUENCY = FAST CHANGE (position, where something IS)

Band 0 (DC):   0.00001  — Identity is eternal (~100k steps to change)
Band 6 (High): 0.03     — Position changes every frame (~30 steps to adapt)
```

**Physical intuition:**
- The identity of an object persists across frames
- The position of an object can change every frame
- Standard transformers treat both the same → suboptimal

**3000× ratio ensures:**
1. Low bands learn slowly → stable concepts, no catastrophic forgetting
2. High bands learn quickly → responsive to transients
3. Natural separation of concerns

### The Rationale for 12× (Spec B)

**For early training stability:**

When starting from random initialization:
- Extreme 3000× ratio can cause instability
- High-freq bands might learn too fast before structure emerges
- Gentler 12× ratio allows coordinated development

**Transition strategy:**
1. Start: Spec B (12× ratio, conservative)
2. After 50k-100k steps: Gradually increase to Spec A
3. Final: Spec A (3000× ratio, full differentiation)

### What We Fixed

**Before:** Two specifications in conflict, unclear which to use

**After:** 
- ✅ Spec A is CANONICAL for all experiments
- ✅ Spec B is optional warm-start schedule
- ✅ Clear documentation in `CANONICAL_PARAMETERS.md`
- ✅ All experiment docs reference canonical parameters

### Status

- ✅ Resolved in `CANONICAL_PARAMETERS.md`
- ✅ Both specifications documented with clear purposes
- ✅ Experiments updated to reference canonical spec
- ⏭️ Experiment 013 validates the 3000× ratio choice

---

## 3. Coherence-Gated Wormholes: Entropy vs Similarity

### The Question

Should wormhole connections activate based on:
1. **Similarity threshold** (original design): `sim > 0.92`
2. **Entropy-based coherence gate** (current design): `g = sigmoid((h - threshold) × sharpness)`

### The Evolution

**Original design (simplified):**
```python
# WORMHOLE_HYBRID.md
if similarity > 0.92:
    activate_wormhole()
```

**Current design (canonical):**
```python
# SPECTRAL_BELIEF_MACHINE.md
h = normalized_entropy(attention_distribution)
g = sigmoid((h - coherence_threshold) * gate_sharpness)
response = g * wormhole_response
```

### The Rationale for Entropy Gate

**Problem with similarity threshold:**
- Similarity can be high even when model is uncertain
- Static threshold doesn't adapt to context
- Binary decision (on/off) loses information

**Benefits of entropy gate:**
1. **Uncertainty-aware:** High entropy → disable wormholes (model is confused)
2. **Adaptive:** Gate strength varies smoothly with confidence
3. **Interpretable:** Entropy has clear meaning (information theory)
4. **Trainable:** `coherence_threshold` and `gate_sharpness` are learnable

**Physical intuition:**
- When belief is uncertain (high entropy), don't make bold cross-band connections
- When belief is coherent (low entropy), enable structured communication
- This prevents "hallucinating" connections when model doesn't know

### Status

- ✅ Implemented in `SPECTRAL_BELIEF_MACHINE.md`
- ✅ Documented in `CANONICAL_PARAMETERS.md`
- ✅ Simplified version preserved in `WORMHOLE_HYBRID.md` (marked non-canonical)
- ⏭️ Experiment 024 tests this design

---

## 4. 7+1 Architecture: Why Not 6+1 or 8+1

### The Question

Why exactly **7 spectral bands + 1 temporal band**?

### The Theoretical Justification

**See `architecture_theoretical/THE_SEVEN_PLUS_ONE_ARCHITECTURE.md` for full derivation.**

**Summary:**

1. **Heisenberg uncertainty:** Δx·Δω ≥ 1 constrains joint localization
2. **Perceptual evidence:** Human vision has ~7 spatial frequency channels
3. **Cognitive evidence:** Working memory capacity ≈ 7±2 items
4. **Network diameter:** log₂(1024) ≈ 10 → 7 bands gives good coverage
5. **Computational:** Powers of 2 align well (FFT, GPU warps)

**The +1 (temporal band):**
- Time is fundamentally different from space (causal, not reversible)
- Requires separate processing
- Connects to all spatial bands (temporal integration)

### Status

- ✅ Full derivation in `THE_SEVEN_PLUS_ONE_ARCHITECTURE.md`
- ✅ Multiple independent justifications converge on 7±1
- ⏭️ Could test 6+1 or 8+1 empirically to validate

---

## 5. Partial Information Decomposition for Cross-Band Analysis

### The Question

How should we measure and interpret information flow between spectral bands?

**Problem:** Traditional mutual information I(X;Y) tells you *how much* information is shared but not *what kind*.

**Solution:** Use Partial Information Decomposition (PID) from Williams & Beer (2010).

### The PID Framework

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  Given two source bands (S1, S2) providing information about target T: │
│                                                                         │
│  I(S1,S2 ; T) = I_red + I_uni(S1) + I_uni(S2) + I_syn                 │
│                                                                         │
│  REDUNDANCY (I_red): Both bands share this about T                    │
│  UNIQUE (I_uni):     Only one band knows this about T                 │
│  SYNERGY (I_syn):    Neither alone, but together they know about T    │
│                                                                         │
│  ════════════════════════════════════════════════════════════════════ │
│                                                                         │
│  COLLAPSE INTERPRETATION:                                              │
│                                                                         │
│  Before collapse: High synergy (bands must cooperate)                 │
│  After collapse:  High redundancy (bands agree)                       │
│  Unique remains:  Band-specific expertise preserved                   │
│                                                                         │
│  WORMHOLE INTERPRETATION:                                              │
│                                                                         │
│  High I_syn between bands → Wormhole enables binding                  │
│  High I_red between bands → Wormhole confirms consensus               │
│  High I_uni in source → Wormhole transfers novel info                 │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Why PID Over Plain Mutual Information

**Mutual information alone:**
- Tells you bands share information
- Doesn't distinguish redundancy from synergy
- Can give negative "interaction information" (confusing)

**PID provides:**
- Nonnegative decomposition (always interpretable)
- Distinguishes shared vs. emergent information
- Aligns with AKIRA's view of bands as complementary specialists

### Status

- Added to `pomdp/REFERENCES.md` (Williams & Beer 2010)
- Used in `experiments/005_EXP_CONSERVATION_LAWS`
- Used in `experiments/020_EXP_CROSS_BAND_FLOW`
- TODO: Implement practical PID estimators for attention weights

---

## 6. Duality Methods for Observability

### The Question

How should we observe internal model states when direct inspection is difficult?

### The Design Choice

Use **duality methods** — transform to a domain where the quantity is easy to observe.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  THE PATTERN: TRANSFORM → CONSERVED → INVERSION                        │
│                                                                         │
│  1. TRANSFORM: Cheap operation to switch representations              │
│  2. CONSERVED: Something preserved (use for verification)             │
│  3. INVERSION: What was hard to compute is now easy (and vice versa)  │
│                                                                         │
│  EXAMPLE: Spatial ↔ Frequency (FFT)                                   │
│  Transform: FFT, O(N log N)                                           │
│  Conserved: Energy (Parseval's theorem)                               │
│  Inversion: Convolution (hard) ↔ Multiplication (easy)                │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### The Seven Dualities

| Duality | When to Use | What It Reveals |
|---------|-------------|-----------------|
| Spatial ↔ Frequency | Pattern analysis | Scale of patterns |
| Magnitude ↔ Phase | WHAT vs WHERE | Identity vs position |
| Forward ↔ Backward | Attribution | Why model predicted this |
| Sharp ↔ Soft | Uncertainty | How confident the model is |
| Local ↔ Global | Dependencies | What needs what |
| Explicit ↔ Implicit | Learning | Memorization vs generalization |
| Energy ↔ Geometry | Processing mode | Reactive vs deliberative |

### Status

- Full documentation in `observability/DUALITY_METHODS.md`
- Theoretical foundation in `foundations/DUALITY_AND_EFFICIENCY.md`
- Each duality includes code examples and intuitive explanations

---

## Design Principles

These decisions reflect underlying principles:

1. **Respect physical structure** (timescales, frequencies, causality)
2. **Make processing modes explicit** (knowledge vs reactive)
3. **Use information theory** (entropy as decision criterion)
4. **Learn what you can, specify what you must** (learnable parameters with good initializations)
5. **Test everything** (all choices should be experimentally validated)

---

## Related Documents

| Document | Relationship |
|----------|--------------|
| `CANONICAL_PARAMETERS.md` | Authoritative parameter values |
| `AKIRA_OVERVIEW.md` | High-level architecture rationale |
| `architecture_theoretical/THE_SEVEN_PLUS_ONE_ARCHITECTURE.md` | Theoretical justification for 7+1 |
| `status_check_and_fixes/MATHEMATICAL_FOUNDATIONS_CHECK.md` | Consistency verification |
| `pomdp/REFERENCES.md` | Williams & Beer (2010) PID citation |
| `observability/DUALITY_METHODS.md` | Duality methods for observation |
| `foundations/DUALITY_AND_EFFICIENCY.md` | Transform-conserved-inversion theory |

---

*"Every design choice should be testable. If we can't experiment to validate or falsify it, we haven't made a design decision — we've made an assumption."*

