# EXPERIMENT 019: Belief Geometry

## Does Uncertainty Have Shape?

**Oscar Goldman — Shogu Research Group @ Datamutant.ai**

---

## Status: PLANNED

## Tier: ★ SUPPORTING

## Depends On: 001 (Entropy), 007 (Wavefront)

---

## 1. Problem Statement

### 1.1 The Question

SHAPE_OF_UNCERTAINTY.md claims that prediction error has geometric structure:

**Does the error/uncertainty have a meaningful shape — a crescent, wave packet, or other structured geometry rather than uniform spread?**

### 1.2 Why This Matters

```
THE BELIEF GEOMETRY HYPOTHESIS

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  From SHAPE_OF_UNCERTAINTY.md:                                         │
│                                                                         │
│  For a moving blob:                                                     │
│  • Error forms a CRESCENT ahead of prediction                         │
│  • Crescent width = uncertainty about SPEED                           │
│  • Crescent orientation = uncertainty about DIRECTION                 │
│                                                                         │
│  The error shape IS the belief projected to observable space.         │
│  This is the wave packet interpretation.                              │
│                                                                         │
│  If true:                                                               │
│  • Uncertainty is not just magnitude but STRUCTURE                    │
│  • The shape reveals what the model knows and doesn't know           │
│  • We can read the belief from the error geometry                    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Hypothesis

### 2.1 Primary Hypothesis

**H1: Error has meaningful geometric structure.**

Not uniform spread, but shaped by the type of uncertainty.

### 2.2 Secondary Hypotheses

**H2: Error shape encodes velocity uncertainty (direction and magnitude).**

**H3: Different motion types produce different error shapes.**

**H4: Error shape is predictable from stimulus properties.**

### 2.3 Null Hypothesis

**H0:** Error is uniform/random (no meaningful geometry).

---

## 3. Scientific Basis

### 3.1 AKIRA Theory Basis

**Relevant Theory Documents:**
- `pomdp/SHAPE_OF_UNCERTAINTY.md` — §2 (Error Geometry), §3 (Crescent Structure)
- `architecture_theoretical/SPECTRAL_BELIEF_MACHINE.md` — §6 (Belief State)
- `pomdp/WAVEFRONT_PROPAGATION.md` — §2 (Wave Packet Interpretation)

**Key Concepts:**
- **Belief as wave packet:** Prediction is mean of distribution, error is its spatial extent
- **Crescent geometry:** For moving objects, error forms structured shape (not uniform blob)
- **Shape encodes uncertainty type:** Width = speed uncertainty, orientation = direction uncertainty
- **Observable projection:** Error map is belief state projected to observable space

**From SHAPE_OF_UNCERTAINTY.md (§3.1):**
> "For moving blob, error forms CRESCENT ahead of prediction. Crescent width encodes speed uncertainty (wider = less certain about speed). Crescent orientation encodes direction uncertainty. The shape IS the belief — not just magnitude but structure."

**From WAVEFRONT_PROPAGATION.md (§2.3):**
> "Wave packet interpretation: Prediction is packet center (⟨x⟩), error is packet width (Δx). For moving packet, spatial distribution reveals velocity uncertainty (Δv). Heisenberg: ΔxΔv ≥ ℏ/2. Tighter position → broader velocity spread."

**This experiment validates:**
1. Whether **error has geometric structure** (crescent, streak, etc.)
2. Whether **shape encodes uncertainty type** (direction, speed)
3. Whether **different stimuli produce different shapes** (predictable patterns)
4. Whether geometry reveals **what model knows/doesn't know**

**Falsification:** If error is uniform blob (no structure) → belief has no geometric representation → simplifies to point estimates with scalar uncertainty.

## 3. Methods

### 3.1 Protocol

```
BELIEF GEOMETRY PROTOCOL

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  STEP 1: Create controlled motion stimuli                              │
│  • Moving blob at constant velocity                                   │
│  • Moving blob changing direction                                     │
│  • Moving blob changing speed                                         │
│  • Ambiguous motion (could go either way)                            │
│                                                                         │
│  STEP 2: Measure error maps                                            │
│  • Prediction error at each spatial position                         │
│  • For each frame                                                     │
│                                                                         │
│  STEP 3: Analyze error geometry                                        │
│  • Fit geometric templates (crescent, blob, streak)                  │
│  • Extract parameters (width, orientation, eccentricity)            │
│                                                                         │
│  STEP 4: Correlate with stimulus properties                            │
│  • Does crescent width correlate with speed uncertainty?            │
│  • Does orientation correlate with direction uncertainty?           │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Results

```
[ TO BE FILLED AFTER EXPERIMENT ]

Error geometry detected: YES / NO

Shape classifications:
• Constant velocity: _____
• Direction change: _____
• Speed change: _____
• Ambiguous: _____

Correlation with uncertainty:
• Width vs speed uncertainty: r = _____
• Orientation vs direction uncertainty: r = _____

[INSERT ERROR SHAPE VISUALIZATIONS]
```

---

## 5. Conclusion

```
[ TO BE FILLED AFTER EXPERIMENT ]

H1 (meaningful geometry): SUPPORTED / NOT SUPPORTED
H2 (encodes velocity uncertainty): SUPPORTED / NOT SUPPORTED
H3 (different shapes for different motions): SUPPORTED / NOT SUPPORTED
H4 (predictable from stimulus): SUPPORTED / NOT SUPPORTED

Belief geometry is INFORMATIVE / NOT INFORMATIVE.
```

---

*Oscar Goldman — Shogu Research Group @ Datamutant.ai*


