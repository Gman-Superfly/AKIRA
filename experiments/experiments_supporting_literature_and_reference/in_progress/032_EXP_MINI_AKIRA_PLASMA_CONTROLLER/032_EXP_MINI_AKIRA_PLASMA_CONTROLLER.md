# Experiment 032: Mini AKIRA Plasma Controller

## A Theory-Aligned Real-Time Control Experiment

**Tier:** EXPLORATORY (Architecture Validation)

**Status:** PENDING

**Dependencies:** Architecture documents (SPECTRAL_BELIEF_MACHINE.md, THE_SEVEN_PLUS_ONE_ARCHITECTURE.md)

---

## The Problem

```
THE CONTROL PROBLEM AS BELIEF DYNAMICS

+-----------------------------------------------------------------------+
|                                                                       |
|  AKIRA claims that prediction IS belief update under partial          |
|  observability. Control is choosing actions that collapse belief      |
|  toward desired states.                                               |
|                                                                       |
|  Most control experiments test whether "bands help" or "attention     |
|  helps" without testing the AKIRA-specific claims:                    |
|                                                                       |
|  1. Does the 7+1 spectral-temporal decomposition outperform           |
|     simpler alternatives at EQUAL parameter count?                    |
|                                                                       |
|  2. Does explicit belief tracking (entropy) provide actionable        |
|     signal for control?                                               |
|                                                                       |
|  3. Does collapse dynamics (entropy rate) correlate with control      |
|     effectiveness?                                                    |
|                                                                       |
|  4. Does a homeostat (PSON-aligned) improve stability over            |
|     standard optimizers?                                              |
|                                                                       |
|  This experiment tests THESE claims, not generic "does spectral       |
|  stuff help" claims.                                                  |
|                                                                       |
+-----------------------------------------------------------------------+
```

---

## The Opportunity

```
A TOY PLASMA ENVIRONMENT FOR BELIEF-CONTROL TESTING

+-----------------------------------------------------------------------+
|                                                                       |
|  WHY PLASMA-LIKE DYNAMICS:                                            |
|                                                                       |
|  1. CONTINUOUS FIELD: Unlike discrete token prediction, plasma        |
|     is a continuous 2D field - tests spectral decomposition           |
|     naturally (FFT is native to the domain).                          |
|                                                                       |
|  2. PARTIAL OBSERVABILITY: The controller sees the field but          |
|     not the underlying dynamics parameters - must infer them.         |
|                                                                       |
|  3. MULTI-SCALE STRUCTURE: Plasma has coherent structures at          |
|     different scales - tests whether bands capture this.              |
|                                                                       |
|  4. CONTROL CHALLENGE: Keeping a blob at target requires              |
|     predicting drift and counteracting - tests belief-action          |
|     coupling.                                                         |
|                                                                       |
|  5. FAST ITERATION: Small grid (64x64), simple physics -              |
|     can run many experiments quickly on CPU.                          |
|                                                                       |
|  This is NOT meant to solve real plasma control.                      |
|  It is a TESTBED for AKIRA's architectural claims.                    |
|                                                                       |
+-----------------------------------------------------------------------+
```

---

## Hypotheses

### H1: 7+1 Architecture Outperforms Baselines
A proper 7-band spectral + 1 temporal (causal) architecture achieves lower prediction error than:
- Flat baseline (same parameters, no spectral structure)
- 4-band baseline (fewer bands)
- Spectral-only baseline (no temporal band)

at matched parameter count.

### H2: Belief Entropy Predicts Control Difficulty
Explicit entropy tracking in the belief state correlates with control error:
- High entropy periods precede large control errors
- Low entropy (post-collapse) periods have lower control errors
- Entropy rate (dH/dt) predicts imminent error spikes

### H3: Collapse Events Correlate with Control Transitions
Sharp entropy drops (collapse events) correspond to:
- Successful target acquisition
- State transitions in the controlled system
- Changes in optimal control strategy

### H4: Homeostat Improves Stability Over Adam
A PSON-aligned homeostat (orthogonal noise injection, constrained relaxation) achieves:
- Lower variance in control error
- Faster recovery from perturbations
- More stable long-term tracking

than Adam with equivalent learning rate.

---

## Method

### Phase 1: Environment Design

```
MINI PLASMA ENVIRONMENT

+-----------------------------------------------------------------------+
|                                                                       |
|  GRID: 64 x 64 scalar field (density/temperature analogue)            |
|                                                                       |
|  DYNAMICS:                                                            |
|  - Diffusion: Laplacian smoothing (5-point stencil)                   |
|  - Advection: Mild drift in one direction                             |
|  - Actuators: 6 localized Gaussian "magnets" that push/pull           |
|  - Noise: Small additive Gaussian perturbation                        |
|                                                                       |
|  CONTROL TASK:                                                        |
|  - Target: Keep a blob centered at grid center                        |
|  - Observation: Current field (partially observable - no velocity)    |
|  - Action: 6-dim vector (actuator strengths)                          |
|                                                                       |
|  METRICS:                                                             |
|  - Field error: MSE(field, target)                                    |
|  - Blob tracking: Distance from blob centroid to target center        |
|                                                                       |
+-----------------------------------------------------------------------+
```

### Phase 2: Predictor Architecture (7+1)

```
THEORY-ALIGNED SPECTRAL BELIEF MACHINE

+-----------------------------------------------------------------------+
|                                                                       |
|  INPUT: Current field (B, 1, 64, 64)                                  |
|                                                                       |
|  SPECTRAL DECOMPOSITION:                                              |
|  1. Apply Hamming window (heresy resistance)                          |
|  2. FFT2 -> complex spectrum                                          |
|  3. Radial log-spaced masks -> 7 bands (DC to Nyquist)                |
|  4. Extract magnitude + phase per band                                |
|                                                                       |
|  PER-BAND PROCESSING (Bands 0-6):                                     |
|  - Each band: Conv2d blocks on (real, imag) channels                  |
|  - Differential learning rates: Band 0 = 0.00001, Band 6 = 0.03       |
|  - Output: Processed (real, imag) per band                            |
|                                                                       |
|  TEMPORAL BAND (Band 7):                                              |
|  - Input: Concatenated band features over history                     |
|  - Causal self-attention (lower-triangular mask)                      |
|  - Output: Temporal context vector                                    |
|                                                                       |
|  WORMHOLE CROSS-BAND:                                                 |
|  - Pairs: (0<->6), (1<->5), (2<->4)                                   |
|  - Band 3 (bridge) -> all                                             |
|  - Band 7 (temporal) -> all spectral                                  |
|  - Cosine similarity on hypersphere, top-k sparse                     |
|                                                                       |
|  BELIEF TRACKING:                                                     |
|  - Compute attention entropy per band                                 |
|  - Track entropy rate dH/dt                                           |
|  - Detect collapse: |dH/dt| > threshold                               |
|                                                                       |
|  RECONSTRUCTION:                                                      |
|  - Combine processed bands -> complex spectrum                        |
|  - iFFT2 -> predicted next field                                      |
|                                                                       |
|  OUTPUT: Predicted field + belief state (entropy per band)            |
|                                                                       |
+-----------------------------------------------------------------------+
```

### Phase 3: Baselines

```
BASELINE MODELS (MATCHED PARAMETERS)

+-----------------------------------------------------------------------+
|                                                                       |
|  BASELINE 1: FLAT (No spectral structure)                             |
|  - Standard ConvNet: Conv2d -> GELU -> Conv2d -> ... -> Conv2d        |
|  - Same total parameters as 7+1                                       |
|  - No FFT, no bands, no temporal attention                            |
|                                                                       |
|  BASELINE 2: 4-BAND (Fewer bands)                                     |
|  - 4 radial bands instead of 7                                        |
|  - No temporal band (all spectral)                                    |
|  - Same total parameters                                              |
|                                                                       |
|  BASELINE 3: SPECTRAL-ONLY (No temporal)                              |
|  - 7 bands but no temporal band                                       |
|  - History via concatenation, not attention                           |
|  - Same total parameters                                              |
|                                                                       |
|  CRITICAL: All baselines have MATCHED parameter count.                |
|  We are testing architecture, not capacity.                           |
|                                                                       |
+-----------------------------------------------------------------------+
```

### Phase 4: Control Head

```
CONTROL HEAD DESIGN

+-----------------------------------------------------------------------+
|                                                                       |
|  INPUT: Predicted next field + belief state                           |
|                                                                       |
|  ARCHITECTURE:                                                        |
|  - Flatten belief entropy vector (8 values)                           |
|  - Concatenate with pooled field features                             |
|  - MLP: Linear -> GELU -> Linear -> tanh (bounded output)             |
|                                                                       |
|  OUTPUT: 6-dim control vector (actuator strengths in [-1, 1])         |
|                                                                       |
|  KEY INSIGHT:                                                         |
|  Control head receives BELIEF STATE explicitly.                       |
|  It can modulate control based on uncertainty.                        |
|                                                                       |
+-----------------------------------------------------------------------+
```

### Phase 5: Homeostat (PSON-Aligned)

```
HOMEOSTATIC CONTROLLER

+-----------------------------------------------------------------------+
|                                                                       |
|  THEORY: The homeostat maintains stability via constrained            |
|  relaxation with orthogonal noise injection.                          |
|                                                                       |
|  IMPLEMENTATION:                                                      |
|                                                                       |
|  1. INNER LOOP (per control step):                                    |
|     - Compute gradient of control loss w.r.t. control output          |
|     - Inject ORTHOGONAL noise (PSON):                                 |
|       noise = randn_like(grad)                                        |
|       noise_orth = noise - proj(noise, grad)                          |
|       grad_pson = grad + alpha * noise_orth                           |
|     - Clip step magnitude (stability)                                 |
|     - Update control parameters                                       |
|                                                                       |
|  2. SETPOINT MAINTENANCE:                                             |
|     - Track running mean of control error                             |
|     - If error exceeds setpoint: increase gain                        |
|     - If error below setpoint: decrease gain (relax)                  |
|                                                                       |
|  3. STABILITY CONSTRAINT:                                             |
|     - Monitor control output variance                                 |
|     - If variance too high: dampen updates                            |
|     - Prevent oscillation                                             |
|                                                                       |
|  COMPARISON: Adam (baseline optimizer)                                |
|  - Same learning rate                                                 |
|  - No orthogonal noise                                                |
|  - No setpoint maintenance                                            |
|                                                                       |
+-----------------------------------------------------------------------+
```

### Phase 6: Experiment Protocol

```
EXPERIMENTAL PROTOCOL

+-----------------------------------------------------------------------+
|                                                                       |
|  STAGE 1: PREDICTOR TRAINING                                          |
|  - Generate 10000 random trajectories (100 steps each)                |
|  - Train each predictor (7+1, baselines) on next-frame prediction     |
|  - Match epochs so all see same data volume                           |
|  - Record: final MSE, training curve, per-band entropy dynamics       |
|                                                                       |
|  STAGE 2: CONTROL TRAINING                                            |
|  - Freeze predictor weights                                           |
|  - Train control head to minimize field error over horizon            |
|  - Compare: Adam vs Homeostat                                         |
|  - Record: control error, variance, recovery time                     |
|                                                                       |
|  STAGE 3: BELIEF-CONTROL CORRELATION                                  |
|  - Run trained controller on test trajectories                        |
|  - Log: entropy per band, control error, collapse events              |
|  - Compute: correlation(entropy, future_error)                        |
|  - Identify: collapse -> control transition correspondence            |
|                                                                       |
|  STAGE 4: ABLATION                                                    |
|  - Remove temporal band: measure degradation                          |
|  - Remove wormholes: measure degradation                              |
|  - Remove belief input to control: measure degradation                |
|                                                                       |
+-----------------------------------------------------------------------+
```

---

## Apparatus

### Required Code Modules

```
CODE STRUCTURE

032_EXP_MINI_AKIRA_PLASMA_CONTROLLER/
  032_EXP_MINI_AKIRA_PLASMA_CONTROLLER.md  (this document)
  code/
    README.md                  (usage instructions)
    mini_plasma_env.py         (environment)
    spectral_belief_machine.py (7+1 predictor)
    baselines.py               (flat, 4-band, spectral-only)
    belief_tracker.py          (entropy, collapse detection)
    control_head.py            (control network)
    homeostat.py               (PSON-aligned optimizer)
    run_experiment.py          (main script)
    analysis.py                (metrics, plots)
  results/
    (to be populated)
```

### Key Metrics

```
METRICS TO COMPUTE

+-----------------------------------------------------------------------+
|                                                                       |
|  PREDICTION METRICS:                                                  |
|  - MSE: Mean squared error on next-frame prediction                   |
|  - Per-band MSE: Error contribution per spectral band                 |
|  - Temporal gain: Improvement from temporal band                      |
|                                                                       |
|  BELIEF METRICS:                                                      |
|  - H_band[k]: Entropy of attention in band k                          |
|  - H_global: Sum of band entropies                                    |
|  - dH/dt: Rate of entropy change                                      |
|  - Collapse count: Number of |dH/dt| > threshold events               |
|                                                                       |
|  CONTROL METRICS:                                                     |
|  - Field error: MSE(field, target) over episode                       |
|  - Centroid error: Distance from blob center to target                |
|  - Variance: Std of control error over episode                        |
|  - Recovery time: Steps to return to low error after perturbation     |
|                                                                       |
|  CORRELATION METRICS:                                                 |
|  - r(H, future_error): Does entropy predict difficulty?               |
|  - r(collapse, transition): Do collapses mark transitions?            |
|                                                                       |
+-----------------------------------------------------------------------+
```

---

## Predictions

### If H1 is True (7+1 Outperforms)

```
EXPECTED RESULTS IF 7+1 ARCHITECTURE IS CORRECT

+-----------------------------------------------------------------------+
|                                                                       |
|  PREDICTION:                                                          |
|  - 7+1 achieves 15-30% lower MSE than flat baseline                   |
|  - 7+1 achieves 5-15% lower MSE than 4-band baseline                  |
|  - 7+1 achieves 10-20% lower MSE than spectral-only baseline          |
|                                                                       |
|  WHY:                                                                 |
|  - 7 bands capture multi-scale structure optimally                    |
|  - Temporal band provides causal context                              |
|  - Wormholes enable what<->where integration                          |
|                                                                       |
|  SIGNATURE:                                                           |
|  - Low bands (0-2) converge first and remain stable                   |
|  - High bands (5-6) adapt faster to local changes                     |
|  - Temporal band shows clear sequential structure                     |
|                                                                       |
+-----------------------------------------------------------------------+
```

### If H2 is True (Entropy Predicts Difficulty)

```
EXPECTED RESULTS IF BELIEF TRACKING MATTERS

+-----------------------------------------------------------------------+
|                                                                       |
|  PREDICTION:                                                          |
|  - Correlation r(H_global, future_error) > 0.3                        |
|  - High entropy precedes error spikes by 2-5 steps                    |
|  - Post-collapse low entropy corresponds to successful tracking       |
|                                                                       |
|  WHY:                                                                 |
|  - Entropy measures uncertainty about future state                    |
|  - High uncertainty -> controller can't plan -> error                 |
|  - Collapse -> certainty -> effective control                         |
|                                                                       |
|  SIGNATURE:                                                           |
|  - Control head that receives belief state outperforms one without    |
|  - Ablating belief input increases control variance                   |
|                                                                       |
+-----------------------------------------------------------------------+
```

### If H3 is True (Collapse = Transition)

```
EXPECTED RESULTS IF COLLAPSE DYNAMICS MATTER

+-----------------------------------------------------------------------+
|                                                                       |
|  PREDICTION:                                                          |
|  - Collapse events (|dH/dt| > threshold) occur near control changes   |
|  - Successful target acquisition preceded by collapse                 |
|  - Failed tracking shows diffuse entropy (no collapse)                |
|                                                                       |
|  WHY:                                                                 |
|  - Collapse = belief crystallization = decision point                 |
|  - Control transitions require commitment (collapse)                  |
|  - Indecision (high entropy) leads to poor control                    |
|                                                                       |
|  SIGNATURE:                                                           |
|  - Phase plot: entropy vs control_error shows distinct regions        |
|  - Collapse timing correlates with state transitions                  |
|                                                                       |
+-----------------------------------------------------------------------+
```

### If H4 is True (Homeostat Beats Adam)

```
EXPECTED RESULTS IF HOMEOSTAT IS BENEFICIAL

+-----------------------------------------------------------------------+
|                                                                       |
|  PREDICTION:                                                          |
|  - Homeostat achieves 10-30% lower control variance                   |
|  - Homeostat recovers from perturbations 20-50% faster                |
|  - Homeostat maintains tracking longer without drift                  |
|                                                                       |
|  WHY:                                                                 |
|  - Orthogonal noise explores without fighting gradient                |
|  - Setpoint maintenance prevents drift                                |
|  - Stability constraint prevents oscillation                          |
|                                                                       |
|  SIGNATURE:                                                           |
|  - Adam shows occasional large spikes; Homeostat is smoother          |
|  - Under perturbation, Homeostat re-stabilizes faster                 |
|                                                                       |
+-----------------------------------------------------------------------+
```

---

## Falsification Criteria

```
WHAT WOULD PROVE US WRONG

+-----------------------------------------------------------------------+
|                                                                       |
|  H1 FALSIFIED IF:                                                     |
|  - 7+1 performs equal or worse than flat baseline at matched params   |
|  - 4-band performs equal to 7-band (7 bands unnecessary)              |
|  - Temporal band provides no benefit over history concatenation       |
|                                                                       |
|  H2 FALSIFIED IF:                                                     |
|  - No correlation between entropy and future error (r < 0.1)          |
|  - Control head performs same with or without belief input            |
|                                                                       |
|  H3 FALSIFIED IF:                                                     |
|  - Collapse events are random w.r.t. control transitions              |
|  - No phase structure in entropy-error space                          |
|                                                                       |
|  H4 FALSIFIED IF:                                                     |
|  - Homeostat performs equal or worse than Adam                        |
|  - Orthogonal noise hurts rather than helps                           |
|                                                                       |
|  IMPLICATIONS IF FALSIFIED:                                           |
|  - H1 false: 7+1 may not be optimal for continuous fields             |
|  - H2 false: Belief tracking is overhead, not useful signal           |
|  - H3 false: Collapse is epiphenomenal, not causal                    |
|  - H4 false: Homeostat is not better than standard optimizers         |
|                                                                       |
+-----------------------------------------------------------------------+
```

---

## Results

**[ TO BE FILLED AFTER EXPERIMENT ]**

### Stage 1: Predictor Training
- 7+1 final MSE: [ ]
- Flat baseline MSE: [ ]
- 4-band baseline MSE: [ ]
- Spectral-only baseline MSE: [ ]

### Stage 2: Control Training
- Adam control error (mean +/- std): [ ]
- Homeostat control error (mean +/- std): [ ]
- Recovery time (Adam vs Homeostat): [ ]

### Stage 3: Belief-Control Correlation
- r(H_global, future_error): [ ]
- Collapse-transition correspondence: [ ]%

### Stage 4: Ablation
- Remove temporal band: +[ ]% error
- Remove wormholes: +[ ]% error
- Remove belief input: +[ ]% error

---

## Conclusions

**[ TO BE FILLED AFTER EXPERIMENT ]**

### Verdict on H1 (7+1 Architecture):
[ CONFIRMED / PARTIALLY CONFIRMED / FALSIFIED ]

### Verdict on H2 (Belief Entropy):
[ CONFIRMED / PARTIALLY CONFIRMED / FALSIFIED ]

### Verdict on H3 (Collapse Dynamics):
[ CONFIRMED / PARTIALLY CONFIRMED / FALSIFIED ]

### Verdict on H4 (Homeostat):
[ CONFIRMED / PARTIALLY CONFIRMED / FALSIFIED ]

### Implications for AKIRA:
[ ]

---

## References

### Internal Documents

| Document | Location | Relevance |
|----------|----------|-----------|
| SPECTRAL_BELIEF_MACHINE | `architecture_theoretical/` | Core architecture specification |
| THE_SEVEN_PLUS_ONE_ARCHITECTURE | `architecture_theoretical/` | 7+1 justification |
| ORTHOGONALITY | `architecture_theoretical/` | Five orthogonalities |
| TERMINOLOGY | `foundations/` | Collapse, entropy definitions |
| EQUILIBRIUM_AND_CONSERVATION | `foundations/` | Conservation laws |

### External References

1. Mao, J., Lozano-Perez, T., Tenenbaum, J.B., & Kaelbling, L.P. (2023). "What Planning Problems Can A Relational Neural Network Solve?" ICLR 2024.

2. Miller, G. A. (1956). "The magical number seven, plus or minus two." Psychological Review.

3. Ashby, W.R. (1952). "Design for a Brain." Chapman & Hall. (Homeostat concept)

---

*Oscar Goldman - Shogu Research Group @ Datamutant.ai*

*"Control is belief collapse toward desired futures. The architecture that best represents belief should best enable control. This experiment tests that claim directly."*
