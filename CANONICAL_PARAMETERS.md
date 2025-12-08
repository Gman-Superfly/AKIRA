# Canonical Learning Rate Specification

## Single Source of Truth for All Hyperparameters

**Date:** December 2025  
**Status:**  CANONICAL — All implementations must use these values  
**Purpose:** Eliminate inconsistencies across documents

**Resolution:** Learning rate inconsistency resolved. Specification A (absolute values, 3000× ratio) is canonical for all experiments. Specification B (relative multipliers, 12× ratio) is for optional warm-start only.

---

## Core Principle

Different frequency bands represent different types of knowledge that change at different rates:
- **Low-frequency bands (0-2):** Stable concepts, identity, structure → learn slowly
- **High-frequency bands (4-6):** Details, textures, position → adapt quickly
- **Mid-frequency band (3):** Bridge, transitional → medium rate
- **Temporal band (7):** Sequential patterns → medium rate

The **3000× ratio** between Band 0 and Band 6 is not arbitrary — it reflects the different timescales at which meaning operates at different scales.

---

## Canonical Learning Rates (Theory-Aligned)

**These are the target values for full training:**

```
CANONICAL LEARNING RATES — SPECIFICATION A
(Use these for experiments and production)

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  Band    Learning Rate    Ratio to Band 0    Processing Mode           │
│  ────    ─────────────    ─────────────────    ───────────────           │
│                                                                         │
│  0       0.00001          1×                   Geometric (belief)       │
│  1       0.0001           10×                  Geometric (belief)       │
│  2       0.0003           30×                  Hybrid                   │
│  3       0.001            100×                 Hybrid (bridge)          │
│  4       0.003            300×                 Hybrid                   │
│  5       0.01             1000×                Reactive (energy)        │
│  6       0.03             3000×                Reactive (energy)        │
│  7       0.001            100×                 Temporal (causal)        │
│                                                                         │
│  TOTAL RANGE: 3000× ratio (Band 0 to Band 6)                          │
│                                                                         │
│  NOTE: Band 7 (temporal) has medium LR like Band 3 (bridge)           │
│        Both serve integrative functions across other dimensions        │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Theoretical Justification

- **Band 0 (DC):** Identity is eternal. Learning rate 0.00001 means it takes ~100k steps to significantly change. This prevents catastrophic forgetting of what things *are*.

- **Band 6 (High):** Position changes every frame. Learning rate 0.03 means it adapts in ~30 steps. This enables rapid response to details.

- **Band 3 & 7 (Bridge/Temporal):** Both integrate information across dimensions. Medium LR (0.001) balances stability and adaptation.

---

## Warm-Start Schedule (Gentler Alternative)

**For early training or when you want more conservative adaptation:**

```
WARM-START LEARNING RATES — SPECIFICATION B
(Use these for first 50k-100k steps, then transition to Spec A)

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  Band    Multiplier    Absolute LR           Target LR (Spec A)        │
│  ────    ──────────    ─────────────────    ────────────────────        │
│                                                                         │
│  0       0.1           lr_base × 0.1        → 0.00001                  │
│  1       0.2           lr_base × 0.2        → 0.0001                   │
│  2       0.4           lr_base × 0.4        → 0.0003                   │
│  3       0.6           lr_base × 0.6        → 0.001                    │
│  4       0.8           lr_base × 0.8        → 0.003                    │
│  5       1.0           lr_base × 1.0        → 0.01                     │
│  6       1.2           lr_base × 1.2        → 0.03                     │
│  7       0.6           lr_base × 0.6        → 0.001                    │
│                                                                         │
│  RATIO: 12× (Band 0 to Band 6)                                        │
│                                                                         │
│  RATIONALE: Gentler hierarchy for initial training.                   │
│  Prevents high-frequency bands from learning too aggressively         │
│  before low-frequency structure is established.                       │
│                                                                         │
│  USAGE:                                                                 │
│  1. Start with lr_base = 0.0001 → Band 5 = 0.0001, Band 6 = 0.00012  │
│  2. After warm-start (50k-100k steps), gradually transition to Spec A │
│  3. Final state matches Spec A exactly                                │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### When to Use Warm-Start Schedule

**Use Spec B warm-start if:**
- ✓ Training from scratch with random initialization
- ✓ You observe instability with full Spec A ratios initially
- ✓ Dataset is very noisy or unstructured
- ✓ You want conservative, stable early training

**Transition to Spec A when:**
- ✓ Loss has plateaued at warm-start rates
- ✓ Spectral structure has emerged (check band activation patterns)
- ✓ Training is stable (no gradient explosions)
- ✓ Typically after 50k-100k steps

---

## Other Key Hyperparameters

### Attention Temperatures (Per-Band)

**Theory-aligned initialization:**

```
Band 0: τ = 0.5   (very peaked, confident)
Band 1: τ = 0.6
Band 2: τ = 0.7
Band 3: τ = 1.0   (bridge: balanced)
Band 4: τ = 0.8
Band 5: τ = 0.9
Band 6: τ = 1.0   (high-freq: reactive/energy-based, doesn't need diffuse)
Band 7: τ = 0.8   (temporal: moderately peaked)
```

**All temperatures are learnable** (set `learnable_temperature=True`)

**Note on Band 6:** Despite being high-frequency, τ = 1.0 (not higher) because Band 6 uses reactive/energy-based processing rather than geometric belief processing. Diffuse attention is not needed for fast reactive responses. The value is learnable and will adapt during training.

---

### Logarithm Convention

**For entropy calculations:**
- **Code/Implementation:** Use natural log `ln(x)` or `torch.log(x)`
- **Theory/Discussion:** May use `log₂(x)` when discussing bits
- **Normalized entropy:** `h = H/H_max` is **base-independent**

**Why both?**
- Natural log (ln) is standard in ML, calculus, continuous math
- Log₂ is standard in discrete information theory (bits, Shannon)
- They differ by constant: `log₂(x) = ln(x) / ln(2) ≈ 1.443 ln(x)`
- The *shape* of entropy function is identical

**Practical impact:** None. Normalized entropy `h ∈ [0,1]` is the same regardless of base.

---

### Wormhole Parameters

**Coherence-gated (theory-aligned):**

```
top_k: 16                        # Sparse connections
coherence_threshold: 0.5         # Normalized entropy threshold
gate_sharpness: 10.0             # Sigmoid steepness
learnable_temperature: True      # Per-band τ
```

**Warm-start annealing (optional):**

```
Initial: coherence_threshold = 0.3  (strict)
Final:   coherence_threshold = 0.5  (normal)
Schedule: Linear increase over 50k steps
```

---

### Collapse Detection

```
collapse_threshold: 0.3          # dH/dt threshold for detecting collapse
temperature_init: 1.0            # Initial belief temperature
```

---

### Optimization

```
base_lr: 0.0001                  # For Spec B warm-start
max_grad_norm: 1.0               # Gradient clipping
warmup_steps: 10000              # LR warmup
weight_decay: 0.01               # L2 regularization
adam_beta1: 0.9
adam_beta2: 0.999
adam_eps: 1e-8
```

---

## Implementation Examples

### Using Canonical Rates (Spec A)

```python
import torch.optim as optim

# Explicit per-band learning rates (Spec A)
spectral_params = [
    {'params': model.band_0.parameters(), 'lr': 0.00001},
    {'params': model.band_1.parameters(), 'lr': 0.0001},
    {'params': model.band_2.parameters(), 'lr': 0.0003},
    {'params': model.band_3.parameters(), 'lr': 0.001},
    {'params': model.band_4.parameters(), 'lr': 0.003},
    {'params': model.band_5.parameters(), 'lr': 0.01},
    {'params': model.band_6.parameters(), 'lr': 0.03},
    {'params': model.band_7.parameters(), 'lr': 0.001},
]

# Other components use base rate
other_params = [
    {'params': model.wormhole.parameters(), 'lr': 0.0005},
    {'params': model.belief_tracker.parameters(), 'lr': 0.0001},
]

optimizer = optim.AdamW(spectral_params + other_params, weight_decay=0.01)
```

---

### Using Warm-Start then Transition (Spec B → Spec A)

```python
def setup_optimizer_with_transition(model, step, warmup_steps=50000):
    """
    Start with Spec B (gentle ratios), transition to Spec A (full ratios).
    """
    base_lr = 0.0001
    
    # Compute transition progress
    if step < warmup_steps:
        progress = step / warmup_steps
        
        # Interpolate between Spec B and Spec A
        # Spec B ratios: [0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2]
        # Spec A ratios: [0.1, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0] (relative to Band 0)
        
        spec_b_lrs = [
            base_lr * 0.1,   # Band 0
            base_lr * 0.2,   # Band 1
            base_lr * 0.4,   # Band 2
            base_lr * 0.6,   # Band 3
            base_lr * 0.8,   # Band 4
            base_lr * 1.0,   # Band 5
            base_lr * 1.2,   # Band 6
            base_lr * 0.6,   # Band 7
        ]
        
        spec_a_lrs = [
            0.00001,  # Band 0
            0.0001,   # Band 1
            0.0003,   # Band 2
            0.001,    # Band 3
            0.003,    # Band 4
            0.01,     # Band 5
            0.03,     # Band 6
            0.001,    # Band 7
        ]
        
        # Linear interpolation
        current_lrs = [
            spec_b_lrs[i] + (spec_a_lrs[i] - spec_b_lrs[i]) * progress
            for i in range(8)
        ]
    else:
        # After warm-start, use Spec A directly
        current_lrs = [0.00001, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.001]
    
    # Create optimizer with interpolated rates
    param_groups = [
        {'params': model.bands[i].parameters(), 'lr': current_lrs[i]}
        for i in range(8)
    ]
    
    return optim.AdamW(param_groups, weight_decay=0.01)
```

---

## Monitoring Learning Rate Effects

**Track these metrics per band:**

```python
class LearningRateMonitor:
    """Monitor per-band learning dynamics."""
    
    def __init__(self):
        self.band_weight_changes = [[] for _ in range(8)]
        self.band_gradient_norms = [[] for _ in range(8)]
    
    def log_step(self, model, step):
        for i in range(8):
            # Weight change magnitude
            weight_change = self._compute_weight_change(model.bands[i])
            self.band_weight_changes[i].append(weight_change)
            
            # Gradient norm
            grad_norm = self._compute_grad_norm(model.bands[i])
            self.band_gradient_norms[i].append(grad_norm)
    
    def check_health(self):
        """
        Healthy signs:
        - Band 0 changes slowly (small weight changes)
        - Band 6 changes quickly (large weight changes)
        - Ratio of changes matches ratio of LRs (~3000×)
        - No gradient explosions
        
        Unhealthy signs:
        - Band 0 changing as fast as Band 6 (ratios not working)
        - Any band has zero gradient (dead)
        - Any band has exploding gradients (instability)
        """
        pass
```

---

## Validation Experiments

To confirm these learning rates are correct, run:

**Experiment 013:** Differential Learning Rate Validation
- Measure per-band convergence rates
- Verify Band 6 adapts ~3000× faster than Band 0
- Check that low-freq bands stabilize identity
- Confirm high-freq bands adapt to details

**Expected results:**
- Band 0-1: Converge slowly, hold stable patterns
- Band 5-6: Converge quickly, adapt rapidly to new inputs
- Band 3,7: Medium convergence, integrate information
- Ratio of convergence rates should match LR ratios

---

## Experiments Using These Parameters

| Parameter | Experiments | Purpose |
|-----------|-------------|---------|
| Learning rates (Spec A) | 003, 013 | Validate differential LR hierarchy |
| coherence_threshold | 012, 020, 024 | Test wormhole activation |
| Temperature values (per-band) | 002, 004, 022 | Collapse dynamics, phase transitions |
| top_k (wormholes) | 012, 020, 024 | Sparse cross-band attention |
| collapse_threshold | 002, 004 | Detect entropy drop events |

**See `experiments/000_EXPERIMENT_INDEX.md` for complete experimental program.**

---

## Information-Theoretic Parameters (PID Analysis)

For experiments using Partial Information Decomposition (Williams & Beer 2010):

```
PID ANALYSIS PARAMETERS

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  ESTIMATION METHOD:                                                     │
│  • Use BROJA measure for redundancy (Bertschinger et al., 2014)        │
│  • Use I_min for simple two-source cases                               │
│                                                                         │
│  SAMPLING:                                                              │
│  • Discretize attention weights into 8-16 bins for MI estimation       │
│  • Use k-nearest neighbor estimator for continuous variables          │
│  • Bootstrap for confidence intervals (100 resamples minimum)          │
│                                                                         │
│  THRESHOLDS:                                                            │
│  • Synergy-dominant: I_syn / I_total > 0.3                             │
│  • Redundancy-dominant: I_red / I_total > 0.5                          │
│  • Unique-dominant: I_uni / I_total > 0.4                              │
│                                                                         │
│  EXPERIMENTS: 005 (Conservation), 020 (Cross-Band Flow)                │
│  REFERENCE: pomdp/REFERENCES.md (Williams & Beer 2010)                 │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Duality Method Parameters

For observability using duality methods (see `observability/DUALITY_METHODS.md`):

```
DUALITY PARAMETERS

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  TEMPERATURE SWEEP (Sharp ↔ Soft duality):                            │
│  temperatures: [0.1, 0.5, 1.0, 2.0, 10.0]                             │
│  uncertainty_measure: divergence(output@τ=0.1, output@τ=10.0)         │
│  fragile_threshold: divergence > 0.1 → belief is fragile              │
│                                                                         │
│  ENERGY/GEOMETRY RATIO:                                                 │
│  energy: activation_magnitudes.pow(2).mean()                          │
│  geometry: attention_entropy.mean()                                    │
│  E_G_ratio: energy / (geometry + 1e-10)                               │
│  reactive_threshold: E_G_ratio > 10.0 → reactive mode                 │
│  deliberative_threshold: E_G_ratio < 1.0 → deliberative mode          │
│                                                                         │
│  EXPERIMENTS: 006 (Heresy), 010 (Tickling)                            │
│  REFERENCE: observability/DUALITY_METHODS.md                          │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## References

**Documents using Specification A (canonical):**
- `architecture_theoretical/SPECTRAL_BELIEF_MACHINE.md` (lines 232-244)
- `architecture_theoretical/ORTHOGONALITY.md` (lines 797-804)
- `AKIRA_OVERVIEW.md` (lines 269-275)
- `architecture_base/attention/spectral_attention/SPECTRAL_ATTENTION.md` (lines 728-735)
- `experiments/013_EXP_DIFFERENTIAL_LR_VALIDATION.md` (lines 22-23)

**Documents previously using Specification B:**
- `praxis/PRETRAINING.md` — NOW UPDATED to use Spec B as warm-start only

---

## Summary

| Aspect | Specification A (Canonical) | Specification B (Warm-Start) |
|--------|----------------------------|------------------------------|
| **Status** | ✅ Canonical for all work | Optional gentle start |
| **Ratio** | 3000× (Band 0 to 6) | 12× (Band 0 to 6) |
| **Band 0** | 0.00001 | lr_base × 0.1 |
| **Band 6** | 0.03 | lr_base × 1.2 |
| **When** | All experiments, production | First 50k-100k steps only |
| **Justification** | Theory-aligned timescales | Conservative warm-start |
| **Transition** | Use directly | Gradually shift to Spec A |

---

*"One canonical specification. Many experiments. No ambiguity."*

