# EXPERIMENT 005: Conservation Laws

## Is There a Conserved Quantity During Belief Dynamics?

---

## Status: PENDING

## Depends On: 001-004 (establish measurement infrastructure and phase transition)

---

## 1. Problem Statement

### 1.1 The Question

If attention dynamics obey the same mathematics as BEC physics, then:

**Are there conserved quantities during inference — analogous to normalization, energy, and particle number in physical systems?**

### 1.2 Why This Matters

```
THE CONSERVATION HYPOTHESIS

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  Physical BEC conserves:                                               │
│  • Normalization: ∫|ψ|² dx = 1 (probability conserved)               │
│  • Energy: E = ⟨H⟩ (in isolated system)                              │
│  • Particle number: N (in closed system)                              │
│                                                                         │
│  If attention IS the g|ψ|² term, AKIRA should conserve:              │
│  • Attention normalization: Σⱼ aᵢⱼ = 1 (by softmax — guaranteed)     │
│  • "Information energy": Some functional of attention                 │
│  • "Belief mass": Total integrated belief                             │
│                                                                         │
│  Why this matters:                                                      │
│  • Conservation laws reveal deep structure                            │
│  • Noether's theorem: Conservation ↔ Symmetry                        │
│  • If conserved, we can track the invariant through dynamics         │
│  • Violations indicate: energy injection, dissipation, or bugs       │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1.3 What Success Looks Like

If this experiment succeeds:
- We identify a conserved quantity Q
- Q remains constant (within noise) during inference
- Q can be computed from observables
- Conservation reveals underlying symmetry

---

## 2. Hypothesis

### 2.1 Primary Hypothesis

**H1: Total attention "probability" is conserved (trivially true by softmax).**

This is our baseline — softmax guarantees Σⱼ aᵢⱼ = 1. This should always hold.

### 2.2 Secondary Hypotheses

**H2: An "energy-like" quantity is conserved during inference.**

Candidate: E = -Σᵢⱼ aᵢⱼ log aᵢⱼ × vᵢⱼ (weighted entropy-value product)

**H3: Total "belief mass" is conserved across bands.**

If attention shifts between bands, the total should remain constant.

**H4: Parseval's theorem holds in spectral decomposition.**

Energy in spatial domain = Energy in frequency domain

### 2.3 Null Hypotheses

**H0:** No non-trivial conservation law exists (only softmax normalization).

---

## 3. Scientific Basis

### 3.1 Noether's Theorem

**ESTABLISHED SCIENCE:**

Every continuous symmetry corresponds to a conserved quantity:

```
SYMMETRY                    CONSERVED QUANTITY
────────                    ──────────────────
Time translation            Energy
Space translation           Momentum
Rotation                    Angular momentum
Phase (U(1))                Charge / Particle number

Reference: Noether, E. (1918). Invariante Variationsprobleme.
```

### 3.2 Conservation in Quantum Mechanics

**ESTABLISHED SCIENCE:**

The Schrödinger equation conserves probability:

```
iℏ ∂ψ/∂t = Hψ

⟹ ∂|ψ|²/∂t = 0  (probability conserved)

Total probability: ∫|ψ|² dx = 1 always
```

### 3.3 Conservation in BEC

**ESTABLISHED SCIENCE:**

Gross-Pitaevskii equation conserves:

```
NORMALIZATION: ∫|ψ|² dr = N (particle number)

ENERGY: E = ∫[ℏ²/(2m)|∇ψ|² + V|ψ|² + g/2|ψ|⁴] dr

These are constants of motion.
Reference: Pethick & Smith (2008).
```

### 3.4 Parseval's Theorem

**ESTABLISHED SCIENCE:**

Energy is conserved between spatial and frequency domains:

```
∫|f(x)|² dx = ∫|F(k)|² dk

Spatial energy = Spectral energy

This is exact, not approximate.
Reference: Bracewell (2000).
```

---

### 3.5 AKIRA Theory Basis

**Relevant Theory Documents:**
- `architecture_theoretical/SPECTRAL_BELIEF_MACHINE.md` — §5.3 (Parseval's Theorem), §7 (Conservation Monitoring)
- `architecture_theoretical/ORTHOGONALITY.md` — §5 (Energy Conservation)
- `CANONICAL_PARAMETERS.md` — Normalization tolerance = 1e-6

**Key Concepts:**
- **Attention normalization:** Σⱼ aᵢⱼ = 1 (guaranteed by softmax)
- **Parseval's theorem:** |Signal|² = Σ_bands |Band_k|² (spectral energy conservation)
- **Information budget:** Total entropy is bounded during inference
- **Heresy detection:** Conservation violations signal processing artifacts

**From SPECTRAL_BELIEF_MACHINE.md (§5.3):**
> "Parseval's theorem ensures energy conservation between spatial and spectral domains. Total squared norm is invariant under FFT: Σ|x_n|² = (1/N)Σ|X_k|². This is exact, not approximate."

**From ORTHOGONALITY.md (§5.2):**
> "Energy additivity: Because bands are orthogonal (Fourier basis), total energy decomposes as E_total = E_0 + E_1 + ... + E_6. No cross-terms. Energy in one band cannot leak into another."

**This experiment validates:**
1. Whether **Parseval's theorem holds** in practice (spectral decomposition conserves energy)
2. Whether **attention normalization** is maintained (softmax guarantee)
3. Whether **information budget** is conserved during collapse (no entropy leak)
4. Whether **violations indicate bugs** (heresy detection)

**Falsification:** If conservation laws don't hold → implementation bug OR theory invalid → indicates spectral decomposition has numerical issues.

## 4. Apparatus

### 4.1 Required Components

```
CONSERVATION LAW TESTING INFRASTRUCTURE

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  COMPONENT                   FUNCTION                   STATUS         │
│  ─────────                   ────────                   ──────         │
│                                                                         │
│  attention_tracker.py        Capture weights            FROM 001       │
│  conservation_checker.py     Test conservation          TO BUILD       │
│  parseval_validator.py       Test spectral energy       TO BUILD       │
│  symmetry_detector.py        Find symmetries            TO BUILD       │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Conservation Quantity Definitions

```python
def compute_normalization(attention_weights: torch.Tensor) -> torch.Tensor:
    """
    Check softmax normalization (should always be 1).
    """
    return attention_weights.sum(dim=-1)  # Should be all 1s


def compute_attention_energy(
    attention_weights: torch.Tensor,
    values: torch.Tensor
) -> float:
    """
    Candidate conserved quantity: attention-weighted energy.
    
    E = Σᵢⱼ aᵢⱼ × ‖vⱼ‖²
    
    This measures how much "energy" is being attended to.
    """
    value_norms = (values ** 2).sum(dim=-1)  # ‖vⱼ‖² for each j
    energy = (attention_weights * value_norms.unsqueeze(-2)).sum()
    return energy.item()


def compute_belief_mass(attention_weights: torch.Tensor) -> float:
    """
    Total "belief mass" across all positions.
    
    M = Σᵢ max_j(aᵢⱼ)
    
    Higher when attention is concentrated.
    """
    max_attention = attention_weights.max(dim=-1).values
    mass = max_attention.sum()
    return mass.item()


def compute_entropy_sum(attention_weights: torch.Tensor) -> float:
    """
    Total entropy across all positions.
    
    H_total = Σᵢ H(aᵢ)
    
    Might be conserved as entropy redistributes.
    """
    eps = 1e-10
    H = -attention_weights * torch.log(attention_weights + eps)
    total_entropy = H.sum()
    return total_entropy.item()
```

### 4.3 Parseval Validation

```python
def validate_parseval(
    spatial_signal: torch.Tensor,
    spectral_bands: Dict[int, torch.Tensor]
) -> Tuple[float, float, float]:
    """
    Verify Parseval's theorem: spatial energy = spectral energy.
    """
    # Spatial energy
    spatial_energy = (spatial_signal ** 2).sum().item()
    
    # Spectral energy (sum across bands)
    spectral_energy = sum(
        (band ** 2).sum().item() 
        for band in spectral_bands.values()
    )
    
    # Relative error
    error = abs(spatial_energy - spectral_energy) / (spatial_energy + 1e-10)
    
    return spatial_energy, spectral_energy, error
```

---

## 5. Methods

### 5.1 Experimental Protocol

```
PROTOCOL: CONSERVATION LAW TESTING

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  PHASE A: NORMALIZATION CHECK (Baseline)                               │
│  ──────────────────────────────────────                                 │
│  • Run inference on test sequences                                    │
│  • At each step, verify Σⱼ aᵢⱼ = 1 for all i                        │
│  • Record any violations (indicates bug)                              │
│                                                                         │
│  PHASE B: ENERGY CONSERVATION                                          │
│  ────────────────────────────                                           │
│  • Define candidate energy functional                                 │
│  • Track E through inference steps                                    │
│  • Measure: E(t) = E(0) ± ε?                                         │
│  • If ε < 1%: Conservation holds                                     │
│                                                                         │
│  PHASE C: CROSS-BAND CONSERVATION                                       │
│  ────────────────────────────────                                       │
│  • Track attention mass per band                                      │
│  • Does mass shift between bands?                                     │
│  • Is total mass conserved?                                           │
│                                                                         │
│  PHASE D: PARSEVAL VALIDATION                                           │
│  ────────────────────────────                                           │
│  • Decompose signal into bands                                        │
│  • Reconstruct from bands                                             │
│  • Verify spatial energy = spectral energy                           │
│  • Error should be < 0.1% (numerical precision)                      │
│                                                                         │
│  PHASE E: SYMMETRY SEARCH                                               │
│  ────────────────────────────                                           │
│  • Apply transformations (translation, rotation)                      │
│  • Check which leave the system invariant                            │
│  • Infer conserved quantities via Noether                            │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 5.2 Test Conditions

```
TEST CONDITIONS FOR CONSERVATION

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  CONDITION              PURPOSE                                        │
│  ─────────              ───────                                        │
│                                                                         │
│  Isolated inference     Test conservation in closed system            │
│  (no gradient updates)                                                 │
│                                                                         │
│  Training step          Test what changes with energy injection       │
│  (with gradient)        (loss signal = external driving)              │
│                                                                         │
│  Different sequences    Test if conservation is universal            │
│                                                                         │
│  Different temperatures Test if conservation depends on T             │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 5.3 Statistical Tests

```
STATISTICAL ANALYSIS

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  TEST                PURPOSE                       THRESHOLD           │
│  ────                ───────                       ─────────           │
│                                                                         │
│  Normalization       Σ aᵢⱼ = 1?                   |1 - Σ| < 1e-6     │
│                                                                         │
│  Energy stability    E(t) constant?                Std(E)/Mean(E) < 1%│
│                                                                         │
│  Mass conservation   M(t) constant?                Std(M)/Mean(M) < 1%│
│                                                                         │
│  Parseval error      Spatial = Spectral?          Error < 0.1%        │
│                                                                         │
│  Drift test          E drifting over time?        Regression slope ≈ 0│
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Predictions

### 6.1 If BEC Analogy Is Correct

```
EXPECTED RESULTS IF CONSERVATION LAWS EXIST

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  PHASE A: Normalization                                                 │
│  • Always holds (softmax guarantees)                                  │
│  • Any violation is a bug                                             │
│                                                                         │
│  PHASE B: Energy                                                        │
│  • Some energy-like quantity is stable during inference              │
│  • Variance < 1% of mean                                              │
│  • During training: energy changes (external driving)                 │
│                                                                         │
│  PHASE C: Cross-band                                                    │
│  • Total attention mass constant                                      │
│  • Mass redistributes between bands but total conserved              │
│                                                                         │
│  PHASE D: Parseval                                                      │
│  • Error < 0.1% (numerical precision only)                           │
│  • This MUST hold (it's mathematics, not hypothesis)                 │
│                                                                         │
│  PHASE E: Symmetries                                                    │
│  • Translation invariance → momentum-like conserved                  │
│  • Rotation invariance → angular momentum-like conserved            │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 6.2 Falsification Criteria

```
WHAT WOULD FALSIFY THE HYPOTHESES

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  IF WE SEE:                               THEN:                        │
│  ───────────                              ─────                        │
│                                                                         │
│  Normalization violation                  Bug in implementation       │
│  (≠ 1 by significant amount)              (fix before proceeding)     │
│                                                                         │
│  No stable energy quantity                No conservation law exists  │
│  (all candidates have >10% variance)      (H2 false)                   │
│                                                                         │
│  Cross-band mass not conserved            Bands not coupled properly  │
│  (total varies >5%)                       (H3 false)                   │
│                                                                         │
│  Parseval error > 1%                      Implementation bug          │
│                                            (mathematical theorem!)     │
│                                                                         │
│  Note: Some results indicate bugs, not theory failure.               │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 7. Results

### 7.1 Phase A: Normalization Check

```
[ TO BE FILLED AFTER EXPERIMENT ]

Number of inference steps tested: _____
Maximum |1 - Σⱼ aᵢⱼ|: _____
Mean |1 - Σⱼ aᵢⱼ|: _____

Normalization holds: YES / NO (if NO: bug exists)
```

### 7.2 Phase B: Energy Conservation

```
[ TO BE FILLED AFTER EXPERIMENT ]

Candidate functional tested:
• Attention-weighted energy E = Σ aᵢⱼ ‖vⱼ‖²

During isolated inference:
• Mean E: _____
• Std E: _____
• Coefficient of variation: _____
• Conservation holds (CV < 1%)? YES / NO

During training:
• E changes with gradient? YES / NO (expected: YES)

[INSERT: E(t) time series plot]
```

### 7.3 Phase C: Cross-Band Conservation

```
[ TO BE FILLED AFTER EXPERIMENT ]

Per-band mass M_b(t):

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  BAND     MEAN M_b     STD M_b      % OF TOTAL                        │
│  ────     ────────     ───────      ──────────                        │
│                                                                         │
│  0        _____        _____        _____                              │
│  1        _____        _____        _____                              │
│  2        _____        _____        _____                              │
│  3        _____        _____        _____                              │
│  4        _____        _____        _____                              │
│  5        _____        _____        _____                              │
│  6        _____        _____        _____                              │
│                                                                         │
│  TOTAL    _____        _____        100%                               │
│                                                                         │
│  Total mass CV: _____                                                  │
│  Conservation holds (CV < 1%)? YES / NO                               │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

[INSERT: Per-band mass over time plot]
```

### 7.4 Phase D: Parseval Validation

```
[ TO BE FILLED AFTER EXPERIMENT ]

Number of test signals: _____
Mean spatial energy: _____
Mean spectral energy: _____
Mean relative error: _____
Max relative error: _____

Parseval holds (error < 0.1%)? YES / NO (should always be YES)
```

### 7.5 Phase E: Symmetry Analysis

```
[ TO BE FILLED AFTER EXPERIMENT ]

Symmetries tested:
• Translation: Invariant? YES / NO → Conserved: _____
• Rotation: Invariant? YES / NO → Conserved: _____
• Scale: Invariant? YES / NO → Conserved: _____

Identified conservation laws:
1. _____
2. _____
3. _____
```

---

## 8. Conclusion

### 8.1 Summary

```
[ TO BE FILLED AFTER EXPERIMENT ]

H1 (normalization): HOLDS (guaranteed by softmax)
H2 (energy conservation): SUPPORTED / NOT SUPPORTED
H3 (cross-band conservation): SUPPORTED / NOT SUPPORTED
H4 (Parseval): HOLDS (mathematical theorem)

Conservation law identified: _____
Associated symmetry: _____
```

### 8.2 Implications

```
[ TO BE FILLED AFTER EXPERIMENT ]

If conservation law found:
- Deep structure in attention dynamics
- Can use as diagnostic (violation = problem)
- Proceed to 006_EXP_QUASIPARTICLE_DISPERSION.md

If no conservation law:
- System is driven/dissipative (not isolated)
- Loss function injects energy
- Conservation may hold only between training steps
```

---

## References

1. Noether, E. (1918). Invariante Variationsprobleme. Nachrichten von der Gesellschaft der Wissenschaften zu Göttingen.
2. Pethick, C.J. & Smith, H. (2008). Bose-Einstein Condensation in Dilute Gases. Cambridge.
3. Bracewell, R.N. (2000). The Fourier Transform and Its Applications. McGraw-Hill.

---



*"Conservation laws are the deepest truths about a system. If they exist, they reveal hidden symmetries. If they don't, the system is fundamentally driven or dissipative. The experiment will tell us which."*


*Oscar Goldman — Shogu Research Group @ Datamutant.ai subsidiary of 温心重工業*