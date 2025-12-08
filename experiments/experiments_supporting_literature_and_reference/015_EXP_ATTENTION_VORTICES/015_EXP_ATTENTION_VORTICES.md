# EXPERIMENT 015: Attention Vortices

## Do Topological Defects Exist in Attention?

**Oscar Goldman — Shogu Research Group @ Datamutant.ai**

---

## Status: PLANNED

## Tier: ○ EXPLORATORY

## Depends On: 004 (Phase Transition), 005 (Conservation)

---

## 1. Problem Statement

### 1.1 The Question

In BEC, vortices are topological defects with quantized circulation:

**Do analogous vortex-like structures exist in attention patterns — stable topological defects with quantized "circulation"?**

### 1.2 Why This Matters

```
THE VORTEX HYPOTHESIS

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  In BEC:                                                                │
│  • Vortices are phase singularities                                   │
│  • Circulation is quantized: ∮ v·dl = n × (h/m)                      │
│  • They're topologically stable                                       │
│                                                                         │
│  In AKIRA:                                                              │
│  • "Circulation" = attention flow pattern                             │
│  • Vortex = stable loop in attention                                  │
│  • Quantization = discrete winding number                             │
│                                                                         │
│  If found:                                                              │
│  • Major validation of BEC framework                                  │
│  • New understanding of attention dynamics                            │
│  • Potential for vortex-based computation                            │
│                                                                         │
│  If not found:                                                          │
│  • BEC analogy may be incomplete                                      │
│  • Or vortices exist but we can't detect them                        │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Hypothesis

### 2.1 Primary Hypothesis

**H1: Vortex-like patterns exist in attention.**

Stable circulation patterns with consistent structure.

### 2.2 Secondary Hypotheses

**H2: Circulation is quantized (discrete winding numbers).**

**H3: Vortices are topologically stable (persist over time).**

**H4: Vortices form at high input velocity (above v_c from Exp 014).**

### 2.3 Null Hypothesis

**H0:** No vortex-like structures exist (attention is vortex-free).

---

## 3. Scientific Basis

### 3.1 AKIRA Theory Basis

**Relevant Theory Documents:**
- `bec/BEC_CONDENSATION_INFORMATION.md` — §7 (Topological Defects)
- `architecture_theoretical/SPECTRAL_BELIEF_MACHINE.md` — §4 (Attention Dynamics)
- `CANONICAL_PARAMETERS.md` — (No direct vortex parameters yet)

**Key Concepts:**
- **Vortex in BEC:** Phase singularity with quantized circulation ∮ v·dl = n(h/m), topologically stable
- **Attention as flow field:** Interpret attention matrix as information flow between positions
- **Winding number:** Topological invariant counting circulation around a point
- **Quantization:** Discrete winding numbers (±1, ±2, ...) indicate topological stability

**From BEC_CONDENSATION_INFORMATION.md (§7.1):**
> "Vortices are topological defects in condensate phase. Circulation is quantized due to single-valuedness of wavefunction. Winding number n is topological invariant — cannot change by smooth deformation. Vortices are stable structures."

**From SPECTRAL_BELIEF_MACHINE.md (§4.1):**
> "Attention can be viewed as flow: aᵢⱼ = amplitude of information flow from token j to token i. Causal structure constrains flow direction. Concentration patterns may form stable topological structures."

**This experiment validates:**
1. Whether **vortex-like structures exist** in attention patterns
2. Whether **circulation is quantized** (discrete winding numbers)
3. Whether vortices are **topologically stable** (persist over time)
4. Whether vortices form at **critical velocity** (Exp 014 connection)

**Falsification:** If no stable circulation patterns OR continuous (not quantized) winding → topological defects don't exist → BEC analogy misses this feature.

**Note:** This is the most exploratory experiment. Finding vortices would be major validation. Not finding them doesn't necessarily falsify framework — may be detection limitation.

## 3. Methods

### 3.1 Protocol

```
VORTEX DETECTION PROTOCOL

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  STEP 1: Define attention "velocity field"                             │
│  • Interpret attention as flow: a_ij = flow from j to i               │
│  • Compute curl/circulation                                           │
│                                                                         │
│  STEP 2: Search for circulation patterns                               │
│  • Compute winding number around each position                        │
│  • Look for non-zero winding numbers                                  │
│                                                                         │
│  STEP 3: Verify topological stability                                  │
│  • Track potential vortices over time                                │
│  • Do they persist or decay?                                         │
│                                                                         │
│  STEP 4: Test for quantization                                         │
│  • Histogram of winding numbers                                       │
│  • Are they discrete (1, 2, 3) or continuous?                       │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Vortex Detection Algorithm

```python
def compute_winding_number(attention, center, radius):
    """
    Compute winding number around a point in attention field.
    
    Winding number = 1/2π ∮ dθ around closed loop
    
    Non-zero winding number = vortex at center
    """
    # Sample attention flow around loop
    angles = np.linspace(0, 2*np.pi, 32)
    flow_angles = []
    
    for angle in angles:
        x = center[0] + radius * np.cos(angle)
        y = center[1] + radius * np.sin(angle)
        flow = get_attention_flow_at(attention, x, y)
        flow_angles.append(np.arctan2(flow[1], flow[0]))
    
    # Compute total angle change
    total_angle = np.sum(np.diff(np.unwrap(flow_angles)))
    winding_number = total_angle / (2 * np.pi)
    
    return round(winding_number)
```

---

## 4. Results

```
[ TO BE FILLED AFTER EXPERIMENT ]

Vortices detected: YES / NO
Number of vortex candidates: _____
Winding number distribution: [INSERT HISTOGRAM]
Persistence time: _____ frames

Quantization observed: YES / NO
```

---

## 5. Conclusion

```
[ TO BE FILLED AFTER EXPERIMENT ]

H1 (vortex patterns exist): SUPPORTED / NOT SUPPORTED
H2 (quantized circulation): SUPPORTED / NOT SUPPORTED
H3 (topologically stable): SUPPORTED / NOT SUPPORTED
H4 (form at high velocity): SUPPORTED / NOT SUPPORTED

Vortices are REAL / NOT FOUND / INCONCLUSIVE.
```

---

*Oscar Goldman — Shogu Research Group @ Datamutant.ai*


