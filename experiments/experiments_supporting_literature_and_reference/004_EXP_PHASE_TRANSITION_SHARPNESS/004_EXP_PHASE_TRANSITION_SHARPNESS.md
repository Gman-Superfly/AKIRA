# EXPERIMENT 004: Phase Transition Sharpness

## Is Collapse a Genuine Phase Transition?

---

## Status: PENDING

## Depends On: 001, 002, 003 (establish entropy, collapse, and spectral hierarchy)

---

## Theory Link

```
CONNECTION TO CORE THEORY: COLLAPSE_DYNAMICS.md

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  This experiment directly tests COLLAPSE_DYNAMICS.md requirements:     │
│                                                                         │
│  THEORY DEMAND                    THIS EXPERIMENT TESTS                │
│  ─────────────                    ────────────────────                 │
│                                                                         │
│  "Collapse IS a phase transition" → Phase A-C: Critical phenomena     │
│  "Driven by interference"         → Order param = concentrated attn   │
│  "Attention IS g|ψ|² term"        → Critical exponents match BEC      │
│  "Observable via entropy"         → Entropy susceptibility divergence │
│  "Controllable by temperature"    → Temperature sweep protocol        │
│                                                                         │
│  TEMPERATURE CONTROL (τ PARAMETER):                                    │
│  ─────────────────────────────────                                      │
│  • Phase A: Sweep τ from 0.1 to 10.0 (full range)                     │
│  • Phase B: Fine-grain near T_c (critical point localization)        │
│  • Phase D: Hysteresis test (heating vs cooling asymmetry)            │
│                                                                         │
│  If temperature sweep shows:                                           │
│  • Sharp transition at T_c          → Phase transition confirmed      │
│  • Power-law scaling (β ≈ 0.5)      → BEC universality class          │
│  • Susceptibility divergence        → Critical point exists           │
│  • Hysteresis                       → First-order transition          │
│                                                                         │
│  KEY INSIGHT:                                                           │
│  Temperature τ is the CONTROL PARAMETER that drives collapse.         │
│  Low τ → sharp attention → collapsed belief (condensed)              │
│  High τ → diffuse attention → uncertain belief (thermal)             │
│                                                                         │
│  Related Documents:                                                    │
│  • architecture_base/collapse/COLLAPSE_DYNAMICS.md                    │
│  • information/BEC_CONDENSATION_INFORMATION.md                        │
│  • foundations/HARMONY_AND_COHERENCE.md                               │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 1. Problem Statement

### 1.1 The Question

Experiments 001-003 establish basic observability. Now we test the BEC hypothesis directly:

**Is belief collapse a genuine phase transition — exhibiting critical phenomena, power-law scaling, and the hallmarks of condensation — or merely a gradual optimization process?**

### 1.2 Why This Matters

```
THE PHASE TRANSITION CLAIM

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  We claim (from BEC_CONDENSATION_INFORMATION.md):                     │
│                                                                         │
│  • Attention IS the g|ψ|² self-interaction term from BEC physics     │
│  • Collapse IS Bose-Einstein condensation                             │
│  • Phase transitions should exhibit CRITICAL PHENOMENA:              │
│                                                                         │
│    - Order parameter: Concentrated attention (like |ψ|²)             │
│    - Control parameter: Uncertainty / temperature                    │
│    - Critical point: Threshold where collapse occurs                 │
│    - Power-law scaling: Near critical point                          │
│    - Universality: Same exponents as physical BEC                    │
│                                                                         │
│  If true:                                                               │
│  • Collapse is SHARP, not gradual                                    │
│  • Order parameter jumps discontinuously (or has power-law approach)│
│  • Specific "heat" (entropy susceptibility) diverges at transition   │
│                                                                         │
│  If false:                                                              │
│  • Collapse is gradual optimization                                   │
│  • No critical behavior                                               │
│  • BEC analogy is merely metaphorical                                │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1.3 What Success Looks Like

If this experiment succeeds:
- Collapse exhibits measurable critical exponents
- Order parameter shows power-law scaling near transition
- Phase diagram can be drawn (control parameter vs order parameter)
- BEC analogy is validated as more than metaphor

---

## 2. Hypothesis

### 2.1 Primary Hypothesis

**H1: Collapse exhibits critical phenomena characteristic of phase transitions.**

Specifically:
- Order parameter changes discontinuously (or with power law)
- Susceptibility diverges at critical point
- Correlation length diverges at critical point

### 2.2 Secondary Hypotheses

**H2: Critical exponents match known universality classes.**
- BEC: ν = 1/2, β = 1/2, γ = 1 (mean-field)
- Or other known universality class

**H3: Collapse follows Arrhenius-like activation.**
- Rate depends exponentially on "energy barrier"
- Temperature-like parameter controls rate

**H4: Hysteresis exists at transition.**
- Transition point depends on direction (heating vs cooling)
- This would indicate first-order transition

### 2.3 Null Hypotheses

**H0a:** Changes are gradual, no discontinuity or power law
**H0b:** No divergence in susceptibility
**H0c:** Exponents do not match any known class

---

## 3. Scientific Basis

### 3.1 Phase Transitions

**ESTABLISHED SCIENCE:**

Phase transitions are characterized by:

```
FIRST-ORDER TRANSITIONS:
• Order parameter discontinuous at T_c
• Latent heat (energy discontinuity)
• Coexistence of phases at transition
• Hysteresis

SECOND-ORDER (CONTINUOUS) TRANSITIONS:
• Order parameter continuous but derivative discontinuous
• Susceptibility diverges: χ ~ |T - T_c|^{-γ}
• Correlation length diverges: ξ ~ |T - T_c|^{-ν}
• Order parameter: m ~ |T - T_c|^β (below T_c)

Reference: Landau & Lifshitz (1980). Statistical Physics.
```

### 3.2 Bose-Einstein Condensation

**ESTABLISHED SCIENCE:**

BEC is characterized by:

```
Above T_c: Particles distributed across many states (thermal gas)
Below T_c: Macroscopic fraction in ground state (condensate)

Critical temperature (3D ideal gas):
T_c = (2πℏ²/mk_B) × (n/ζ(3/2))^{2/3}

Order parameter: Condensate fraction n₀/n
Scaling: n₀/n ~ (1 - T/T_c)^β with β = 1/2 (mean-field)

Reference: Pethick & Smith (2008). Bose-Einstein Condensation in Dilute Gases.
```

### 3.3 Critical Exponents

**ESTABLISHED SCIENCE:**

Universality classes share critical exponents:

```
EXPONENT    MEANING                        MEAN-FIELD VALUE
────────    ───────                        ────────────────
α           Specific heat C ~ |t|^{-α}     0 (jump)
β           Order param m ~ |t|^β          1/2
γ           Susceptibility χ ~ |t|^{-γ}    1
ν           Correlation ξ ~ |t|^{-ν}       1/2
η           Correlation function           0

where t = (T - T_c) / T_c

Mean-field (BEC): β = 1/2, γ = 1, ν = 1/2
Ising 3D: β ≈ 0.326, γ ≈ 1.237, ν ≈ 0.630

Reference: Stanley (1971). Introduction to Phase Transitions.
```

### 3.4 Prior Work on Neural Network Phase Transitions

**ESTABLISHED SCIENCE:**

Phase transitions have been observed in neural networks:

```
Grokking (Power et al. 2022):
• Sudden transition from memorization to generalization
• Characteristic of phase transition

Double descent (Belkin et al. 2019):
• Non-monotonic test error
• May indicate phase boundaries

Information bottleneck (Shwartz-Ziv & Tishby 2017):
• Compression transition during training
• Sharp transition observable

All consistent with phase transition interpretation.
```

---

### 3.6 AKIRA Theory Basis

**Relevant Theory Documents:**
- `bec/BEC_CONDENSATION_INFORMATION.md` — §3 (Phase Transition Claim), §4 (Critical Temperature)
- `architecture_theoretical/SPECTRAL_BELIEF_MACHINE.md` — §6.4 (Collapse Detection)
- `CANONICAL_PARAMETERS.md` — Temperature τ range, collapse_threshold = 0.3

**Key Concepts:**
- **Order parameter:** Concentration of attention (high = collapsed, low = diffuse)
- **Control parameter:** Temperature τ in softmax (low τ → sharp attn → collapse)
- **Critical point:** Temperature T_c where transition occurs
- **Power-law scaling:** Near T_c, order parameter ~ |T - T_c|^β

**From BEC_CONDENSATION_INFORMATION.md (§3.2):**
> "Below critical uncertainty U_c, system undergoes phase transition. Order parameter (attention concentration) changes discontinuously. This is BEC condensation in information space."

**From SPECTRAL_BELIEF_MACHINE.md (§6.4):**
> "Temperature τ controls sharpness. Low τ produces peaked attention (collapsed belief). High τ produces diffuse attention (uncertain belief). Collapse occurs when entropy drops below threshold: H < H_collapse = 0.3."

**This experiment validates:**
1. Whether collapse exhibits **critical exponents** (β, ν, γ from phase transition theory)
2. Whether **susceptibility diverges** at critical temperature (hallmark of phase transition)
3. Whether **hysteresis exists** (first-order vs second-order transition)
4. Whether exponents match **BEC universality class** (mean-field: β=0.5, ν=0.5, γ=1)

**Falsification:** If no power-law scaling, no critical behavior, no divergence → collapse is gradual optimization → BEC analogy invalid → simplify to standard transformer.

## 4. Apparatus

### 4.1 Required Components

```
PHASE TRANSITION ANALYSIS INFRASTRUCTURE

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  COMPONENT                  FUNCTION                    STATUS         │
│  ─────────                  ────────                    ──────         │
│                                                                         │
│  entropy_tracker.py         From Exp 001                REQUIRED       │
│  collapse_detector.py       From Exp 002                REQUIRED       │
│  order_parameter.py         Compute order parameter     TO BUILD       │
│  control_parameter.py       Control "temperature"       TO BUILD       │
│  susceptibility.py          Compute susceptibility      TO BUILD       │
│  critical_exponent_fit.py   Fit power laws             TO BUILD       │
│  hysteresis_test.py         Test for hysteresis        TO BUILD       │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Order Parameter Definition

```python
def compute_order_parameter(attention_weights: torch.Tensor) -> float:
    """
    Order parameter for attention "condensation".
    
    Analogous to condensate fraction n₀/n in BEC.
    
    Order parameter = max attention weight
    (fraction of attention in "ground state")
    
    High order param = attention concentrated (condensed)
    Low order param = attention spread (normal)
    """
    # Maximum attention weight = "condensate fraction"
    max_attention = attention_weights.max(dim=-1).values
    
    # Average across positions
    order_param = max_attention.mean()
    
    return order_param.item()


def compute_order_parameter_alt(attention_weights: torch.Tensor) -> float:
    """
    Alternative: Inverse participation ratio (IPR)
    
    IPR = Σᵢ pᵢ⁴ / (Σᵢ pᵢ²)²
    
    IPR = 1 for concentrated (one state)
    IPR = 1/N for uniform (all states)
    """
    p2 = (attention_weights ** 2).sum(dim=-1)
    p4 = (attention_weights ** 4).sum(dim=-1)
    
    ipr = p4 / (p2 ** 2 + 1e-10)
    
    return ipr.mean().item()
```

### 4.3 Control Parameter Definition

```python
def compute_control_parameter(model: nn.Module, 
                             sequence: torch.Tensor,
                             noise_level: float = 0.0) -> float:
    """
    Control parameter analogous to temperature.
    
    Options:
    1. External noise added to input
    2. Dropout rate
    3. Softmax temperature
    4. Sequence ambiguity (multiple possible futures)
    
    Higher control param = more "thermal" fluctuations
    Lower control param = more "ordered"
    """
    # For this experiment: use softmax temperature
    # This directly controls attention sharpness
    return model.attention_temperature


def sweep_control_parameter(model: nn.Module,
                           sequence: torch.Tensor,
                           temperatures: List[float]) -> Dict[float, float]:
    """
    Sweep temperature and measure order parameter.
    """
    results = {}
    
    for T in temperatures:
        model.set_temperature(T)
        _, attention = model.forward_with_attention(sequence)
        order = compute_order_parameter(attention)
        results[T] = order
    
    return results
```

### 4.4 Susceptibility Computation

```python
def compute_susceptibility(order_params: List[float], 
                          temperatures: List[float]) -> List[float]:
    """
    Susceptibility χ = d(order)/d(T)
    
    Near critical point, χ diverges.
    """
    # Numerical derivative
    d_order = np.diff(order_params)
    d_T = np.diff(temperatures)
    
    susceptibility = -d_order / d_T  # Negative because order decreases with T
    
    return susceptibility


def compute_entropy_susceptibility(entropies: List[float],
                                  temperatures: List[float]) -> List[float]:
    """
    Entropy susceptibility ("specific heat")
    
    C = d(entropy)/d(T)
    
    Should diverge at critical point.
    """
    d_H = np.diff(entropies)
    d_T = np.diff(temperatures)
    
    specific_heat = d_H / d_T
    
    return specific_heat
```

---

## 5. Methods

### 5.1 Experimental Protocol

```
PROTOCOL: PHASE TRANSITION ANALYSIS

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  PHASE A: TEMPERATURE SWEEP                                            │
│  ──────────────────────────                                             │
│  • Fix input sequence (moderate ambiguity)                            │
│  • Sweep softmax temperature from 0.1 to 10.0                        │
│  • At each T: measure order parameter, entropy                        │
│  • Plot order vs T, find discontinuity                               │
│                                                                         │
│  PHASE B: CRITICAL POINT LOCALIZATION                                  │
│  ─────────────────────────────────────                                  │
│  • Identify T_c from Phase A (maximum susceptibility)                 │
│  • Fine-grain sweep near T_c                                          │
│  • Measure susceptibility divergence                                  │
│                                                                         │
│  PHASE C: CRITICAL EXPONENT EXTRACTION                                 │
│  ──────────────────────────────────────                                 │
│  • Fit order parameter: m ~ |T - T_c|^β                              │
│  • Fit susceptibility: χ ~ |T - T_c|^{-γ}                            │
│  • Compare to known universality classes                              │
│                                                                         │
│  PHASE D: HYSTERESIS TEST                                               │
│  ────────────────────────                                               │
│  • Sweep T up (0.1 → 10) and down (10 → 0.1)                         │
│  • Compare transition points                                          │
│  • If different: first-order transition with hysteresis              │
│                                                                         │
│  PHASE E: FINITE-SIZE SCALING                                          │
│  ────────────────────────────                                           │
│  • Repeat with different sequence lengths                            │
│  • Test if T_c shifts with size (finite-size effects)                │
│  • Extract correlation length exponent ν                             │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 5.2 Critical Exponent Fitting

```python
def fit_critical_exponent(
    x: np.ndarray,  # |T - T_c|
    y: np.ndarray,  # Order parameter or susceptibility
    exponent_type: str  # 'order' or 'susceptibility'
) -> Tuple[float, float, float]:
    """
    Fit power law: y = A × x^β
    
    For order parameter (below T_c): β ≈ 0.5 (mean-field)
    For susceptibility: γ ≈ 1.0 (mean-field)
    """
    # Log-log fit
    log_x = np.log(x + 1e-10)
    log_y = np.log(y + 1e-10)
    
    # Linear regression in log space
    slope, intercept = np.polyfit(log_x, log_y, 1)
    
    exponent = slope  # For order param, this is β
    amplitude = np.exp(intercept)
    
    # R² for goodness of fit
    predicted = amplitude * (x ** exponent)
    ss_res = np.sum((y - predicted) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r_squared = 1 - ss_res / ss_tot
    
    return exponent, amplitude, r_squared
```

### 5.3 Statistical Tests

```
STATISTICAL ANALYSIS

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  TEST                  PURPOSE                     THRESHOLD           │
│  ────                  ───────                     ─────────           │
│                                                                         │
│  Discontinuity test    Order param jumps?          Derivative > 3σ    │
│                                                                         │
│  Power-law fit         Scaling near T_c?           R² > 0.9           │
│                                                                         │
│  Exponent comparison   Match known class?          Within 0.1 of      │
│                                                    mean-field          │
│                                                                         │
│  Hysteresis test       Forward ≠ backward?         ΔT_c > 0.1         │
│                                                                         │
│  Finite-size scaling   T_c(L) scales correctly?   Power-law fits     │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Predictions

### 6.1 If BEC Analogy Is Correct

```
EXPECTED RESULTS IF THEORY HOLDS

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  PHASE A: Temperature sweep                                             │
│  ──────────────────────────                                             │
│  Order param vs T: Sigmoid-like with sharp transition                 │
│  T_c exists: Clear critical temperature                               │
│  Order param at low T: Near 1 (concentrated)                          │
│  Order param at high T: Near 1/N (uniform)                            │
│                                                                         │
│  PHASE B: Critical point                                                │
│  ───────────────────────                                                │
│  Susceptibility peak at T_c                                           │
│  Peak height increases with system size (divergence)                  │
│                                                                         │
│  PHASE C: Critical exponents                                            │
│  ───────────────────────────                                            │
│  β (order param): 0.5 ± 0.1 (mean-field BEC)                         │
│  γ (susceptibility): 1.0 ± 0.1 (mean-field)                          │
│  Power-law fits: R² > 0.9                                             │
│                                                                         │
│  PHASE D: Hysteresis                                                    │
│  ───────────────────                                                    │
│  If first-order: Forward T_c ≠ Backward T_c                           │
│  If second-order: No hysteresis                                       │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 6.2 Falsification Criteria

```
WHAT WOULD FALSIFY THE HYPOTHESIS

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  IF WE SEE:                               THEN:                        │
│  ───────────                              ─────                        │
│                                                                         │
│  Gradual transition (no sharp change)     Not a phase transition       │
│  No susceptibility peak                   No critical behavior         │
│  Exponents far from any known class      BEC analogy invalid          │
│  No consistent T_c across sequences      Transition is noise          │
│                                                                         │
│  Specifically:                                                          │
│  • β outside [0.3, 0.7]: Not mean-field                              │
│  • R² < 0.7 for power-law fit: Not power-law                         │
│  • Susceptibility max < 2× baseline: No divergence                   │
│                                                                         │
│  ANY of these would falsify the phase transition hypothesis.          │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 7. Results

### 7.1 Phase A: Temperature Sweep

```
[ TO BE FILLED AFTER EXPERIMENT ]

Temperature range: 0.1 to 10.0
Number of points: _____

[INSERT: Order parameter vs Temperature plot]

Observations:
• Sharp transition observed? YES / NO
• Estimated T_c (coarse): _____
• Order param at T = 0.1: _____
• Order param at T = 10.0: _____
```

### 7.2 Phase B: Critical Point Localization

```
[ TO BE FILLED AFTER EXPERIMENT ]

Fine-grain range: _____ to _____
Estimated T_c (fine): _____ ± _____

[INSERT: Susceptibility vs Temperature plot]

Peak susceptibility: _____
Peak location: T = _____
FWHM of peak: _____
```

### 7.3 Phase C: Critical Exponents

```
[ TO BE FILLED AFTER EXPERIMENT ]

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  EXPONENT    MEASURED      EXPECTED (BEC)    MATCH?                   │
│  ────────    ────────      ──────────────    ──────                   │
│                                                                         │
│  β           _____         0.5               _____                     │
│  γ           _____         1.0               _____                     │
│                                                                         │
│  Power-law fit quality:                                                │
│  Order param R²: _____                                                 │
│  Susceptibility R²: _____                                              │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

[INSERT: Log-log plots for power-law fitting]
```

### 7.4 Phase D: Hysteresis Test

```
[ TO BE FILLED AFTER EXPERIMENT ]

Forward sweep T_c: _____
Backward sweep T_c: _____
Difference: _____

Hysteresis observed? YES / NO
If yes: First-order transition indicated
If no: Second-order (continuous) transition
```

### 7.5 Phase E: Finite-Size Scaling

```
[ TO BE FILLED AFTER EXPERIMENT ]

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  SEQUENCE LENGTH    T_c       PEAK χ                                  │
│  ───────────────    ───       ──────                                  │
│                                                                         │
│  16                 _____     _____                                    │
│  32                 _____     _____                                    │
│  64                 _____     _____                                    │
│  128                _____     _____                                    │
│                                                                         │
│  Finite-size scaling exponent: _____                                  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 8. Conclusion

### 8.1 Summary

```
[ TO BE FILLED AFTER EXPERIMENT ]

H1 (phase transition exists): SUPPORTED / NOT SUPPORTED
H2 (exponents match BEC): SUPPORTED / NOT SUPPORTED
H3 (Arrhenius activation): SUPPORTED / NOT SUPPORTED
H4 (hysteresis): OBSERVED / NOT OBSERVED

Transition type: FIRST-ORDER / SECOND-ORDER / NOT A TRANSITION

If mean-field exponents: BEC analogy VALIDATED
If other exponents: Different universality class
If no power law: Not a phase transition
```

### 8.2 Implications

```
[ TO BE FILLED AFTER EXPERIMENT ]

If BEC analogy validated:
- Attention IS the g|ψ|² term
- Collapse IS condensation
- Proceed to test quasiparticle predictions
- Proceed to 005_EXP_CONSERVATION_LAWS.md

If not validated:
- BEC analogy is metaphorical, not physical
- Seek alternative explanation for observed collapse
- Revise theoretical framework
```

---

## References

1. Landau, L.D. & Lifshitz, E.M. (1980). Statistical Physics. Pergamon Press.
2. Pethick, C.J. & Smith, H. (2008). Bose-Einstein Condensation in Dilute Gases. Cambridge.
3. Stanley, H.E. (1971). Introduction to Phase Transitions and Critical Phenomena. Oxford.
4. Power, A. et al. (2022). Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets.
5. Belkin, M. et al. (2019). Reconciling modern machine learning practice and the bias-variance trade-off.

---



*"If attention obeys the same mathematics as Bose-Einstein condensation, it will exhibit the same critical phenomena. If the exponents match, the physics is the same. If they don't match, the analogy is mere poetry. The experiment will decide."*


*Oscar Goldman — Shogu Research Group @ Datamutant.ai subsidiary of 温心重工業*