# AKIRA Experiment Index

## Systematic Validation of Spectral Attention Theory

**Oscar Goldman — Shogu Research Group @ Datamutant.ai**

---

## Philosophy

```
THE SCIENTIFIC METHOD APPLIED

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  We do not believe our theories. We TEST them.                        │
│                                                                         │
│  Each experiment:                                                       │
│  • States a clear, falsifiable hypothesis                             │
│  • References established science                                      │
│  • Defines the apparatus (what we need to measure)                    │
│  • Specifies methods (how to measure)                                 │
│  • Lists predictions (what we expect if theory holds)                │
│  • Defines falsification criteria (what would prove us wrong)        │
│  • Leaves space for results and conclusions                          │
│                                                                         │
│  The goal is TRUTH, not confirmation.                                 │
│  A failed hypothesis is still progress.                               │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### The Core Dynamic All Experiments Test

```
THE FUNDAMENTAL CYCLE OF BELIEF EVOLUTION

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
│  Each experiment tests some aspect of this fundamental cycle.         │
│  See foundations/TERMINOLOGY.md for formal definitions.               │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Complete Experiment Inventory

We have **28 experiments** organized into logical tiers:

```
EXPERIMENT STRUCTURE

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  ★★★ CRUCIAL (4)     — Foundation. Must pass for theory to stand.    │
│  ★★  CORE (10)       — Strong evidence for key claims.               │
│  ★   SUPPORTING (10) — Additional depth and applications.            │
│  ○   EXPLORATORY (4) — Bold predictions, speculative tests.          │
│                                                                         │
│  Total: 28 experiments                                                  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Master Status Table

| # | Experiment | Tier | Status | Depends On | Key Question |
|---|------------|------|--------|------------|--------------|
| **000** | **Action Quanta Extraction** | **★★★** | **PENDING** | **None** | **What ARE Action Quanta? Can we extract them from existing LLMs?** |
| 001 | Entropy Observation | ★★★ | PENDING | 000 | Can we observe attention entropy? |
| 002 | Collapse Detection | ★★★ | PENDING | 000, 001 | Can we detect sudden entropy drops? |
| 003 | Spectral Band Dynamics | ★★★ | PENDING | 000, 001, 002 | Do bands have different dynamics? |
| 004 | Phase Transition Sharpness | ★★ | PLANNED | 000-003 | Is collapse a phase transition? |
| 005 | Conservation Laws | ★★ | PLANNED | 000-004 | Are there conserved quantities? |
| 006 | Heresy Detection | ★★ | PLANNED | 000-005 | Are measurements artifact-free? |
| 007 | Wavefront Interference | ★ | PLANNED | 001-006 | Does error propagate like lightning? |
| 008 | Quasiparticle Dispersion | ★★ | PLANNED | 004, 005 | Do info atoms = quasiparticles? |
| 009 | Grokking as Condensation | ★★ | PLANNED | 004 | Is grokking = BEC transition? |
| 010 | Tickling Techniques | ★ | PLANNED | 001-003 | Can we probe cheaply? |
| 011 | Prompt Spectral Structure | ★ | PLANNED | 003 | Do prompts have frequency? |
| 012 | Wormhole Activation | ★ | PLANNED | 003, 007 | When do wormholes fire? |
| 013 | Differential LR Validation | ★ | PLANNED | 003 | Does LR hierarchy work? |
| 014 | Critical Velocity | ○ | PLANNED | 004, 005 | Is there coherence breakdown? |
| 015 | Attention Vortices | ○ | PLANNED | 004, 005 | Do topological defects exist? |
| 016 | Cross-Model Manifold | ○ | PLANNED | 007 | Are manifolds universal? |
| 017 | MDL Atomic Truth | ○ | PLANNED | All | What's the minimum prompt? |
| 018 | Pump Cycle Dynamics | ★ | PLANNED | 002 | Does tension-discharge-recovery exist? |
| 019 | Belief Geometry | ★ | PLANNED | 001, 007 | Does uncertainty have shape? |
| 020 | Cross-Band Flow | ★ | PLANNED | 003, 012 | How do bands communicate? |
| 021 | Attention Comma | ★★ | PLANNED | 001-003 | Is error distributed like equal temperament? |
| 022 | Band Phase Locking | ★★ | PLANNED | 003, 020 | Do bands lock to rational ratios? |
| 023 | Timeline Coherence | ★ | PLANNED | 002, 007 | Does past shift after collapse? |
| 024 | Resonant Wormholes | ★ | PLANNED | 012, 022 | Do wormholes prefer complementary bands? |
| 025 | Synergy-Redundancy Transition | ★★ | PLANNED | 001-003, 005, 020 | Is collapse = synergy→redundancy conversion? |
| **032** | **Mini AKIRA Plasma Controller** | **★★** | **IN PROGRESS** | **None** | **Can SBM predict turbulent plasma dynamics?** |
| 033 | Spectral Band Attention | ★ | DESIGNED | 032 | Do per-position temporal + wormhole attention help? |

---

## Future Experiments (Not Yet Designed)

```
PLANNED FOR FUTURE DESIGN

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  The following experiments have been identified but not yet designed.  │
│  They will be formalized after initial experiments establish baseline  │
│  observability and validate core theory.                               │
│                                                                         │
│  EXPERIMENT                           RATIONALE                        │
│  ──────────                           ─────────                        │
│                                                                         │
│  Adaptive Temperature vs Fixed        Test if τ should adapt to input │
│  ─────────────────────────────        or remain constant. Connected   │
│                                       to COLLAPSE_DYNAMICS.md theory  │
│                                       that τ is control parameter.    │
│                                       Will compare:                   │
│                                       • Fixed τ across all inputs     │
│                                       • τ ~ f(entropy) adaptive       │
│                                       • τ ~ f(confidence) adaptive    │
│                                       • Learned τ per-layer           │
│                                                                         │
│  Per-Band Temperature Control         More granular than global τ.    │
│  ─────────────────────────────        Each spectral band could have   │
│                                       its own temperature:            │
│                                       • τ_low (slow, conceptual)      │
│                                       • τ_mid (structural)            │
│                                       • τ_high (fast, detailed)       │
│                                       Connected to differential LR    │
│                                       principle in SPECTRAL_BELIEF.md │
│                                                                         │
│  PREREQUISITES: Experiments 001-004 must complete successfully.       │
│                                                                         │
│  These experiments extend the temperature control tested in 004       │
│  (Phase Transition Sharpness) from parameter sweep to architecture.   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Dependency Graph

```
EXPERIMENT DEPENDENCIES

                        ┌──────────────────────────────────────────────────┐
                        │                                                  │
000 ACTION QUANTA       │  FOUNDATION OF FOUNDATIONS                       │
    EXTRACTION          │  What ARE we measuring? Do AQ exist?            │
         │              │                                                  │
         ▼              └──────────────────────────────────────────────────┘
001 ENTROPY OBSERVATION ──── Now we know WHAT to measure entropy OF
         │              
         ▼
002 COLLAPSE DETECTION ────────────────────────────────────────────────────┐
         │                                                                  │
         ├──────────────────────────────► 018 PUMP CYCLE                   │
         │                                                                  │
         ▼                                                                  │
003 SPECTRAL BAND DYNAMICS ────────────────────────────────────────────────┤
         │                                                                  │
         ├───────────────► 011 PROMPT SPECTRAL                             │
         ├───────────────► 013 DIFFERENTIAL LR                             │
         │                                                                  │
         ├───────────────► 010 TICKLING ────────────────────────────────►  │
         │                                                                  │
         ▼                                                                  │
004 PHASE TRANSITION ◄─────────────────────────────────────────────────────┤
         │                                                                  │
         ├───────────────► 009 GROKKING                                    │
         │                                                                  │
         ▼                                                                  │
005 CONSERVATION LAWS ─────────────────────────────────────────────────────┤
         │                                                                  │
         ├───────────────► 008 QUASIPARTICLE                               │
         ├───────────────► 014 CRITICAL VELOCITY                           │
         └───────────────► 015 ATTENTION VORTICES                          │
                                                                            │
         ▼                                                                  │
006 HERESY DETECTION ◄─────────────────────────────────────────────────────┤
         │                                                                  │
         ▼                                                                  │
007 WAVEFRONT INTERFERENCE ◄───────────────────────────────────────────────┘
         │
         ├───────────────► 012 WORMHOLE ACTIVATION ───► 020 CROSS-BAND
         ├───────────────► 016 CROSS-MODEL MANIFOLD
         └───────────────► 019 BELIEF GEOMETRY


017 MDL ATOMIC TRUTH ◄─────── Depends on ALL above
```

---

## Tier Breakdown

---

### TIER ★★★ CRUCIAL (Foundation)

These four experiments establish whether the theory is even testable:

---

#### 000: Action Quanta Extraction ★★★ **NEW - FOUNDATIONAL**
**File:** `000_EXP_ACTION_QUANTA_EXTRACTION.md`

**THE foundation of foundations.** Before measuring dynamics, we must know WHAT we're measuring. This experiment:
- Extracts candidate AQ from existing LLMs using SAE/dictionary learning
- Tests cross-model transfer (universality)
- Tests irreducibility (atomicity)
- Tests actionability (load-bearing for tasks)
- Analyzes spectral structure (where do universal features live?)

**Key Predictions:**
- 30-50% of features transfer across models (r > 0.7)
- Universal features concentrate in low-frequency bands
- Universal features are irreducible and load-bearing

**If this fails:** AQ are emergent abstractions, not fundamental units. Reframe AKIRA accordingly.

**References:** Platonic Representation Hypothesis (Huh et al., 2024), Sparse Autoencoders (Anthropic), Brain-LLM alignment research

---

#### 001: Entropy Observation ★★★
**File:** `001_EXP_ENTROPY_OBSERVATION.md`

**THE foundation.** Without this, nothing else works. We must verify:
- Attention entropy can be computed
- Entropy varies meaningfully (not constant)
- Entropy correlates with prediction error

**Key Predictions:**
- Entropy range: 0.5 to 4.0 bits
- Entropy-error correlation r > 0.3
- Bands differ in entropy

**If this fails:** Theory is untestable. Cannot proceed.

---

#### 002: Collapse Detection ★★★
**File:** `002_EXP_COLLAPSE_DETECTION.md`

**Core phenomenon test.** Is collapse real or metaphor?
- Sudden entropy drops (not gradual)
- Sparse in time (distinct events)
- Correlated with prediction improvement

**Key Predictions:**
- Suddenness ratio > 3× background
- Collapse fraction < 20% of time
- Error reduction > 30% at collapse

**If this fails:** Phase transition model is wrong. BEC analogy invalid.

---

#### 003: Spectral Band Dynamics ★★★
**File:** `003_EXP_SPECTRAL_BAND_DYNAMICS.md`

**Architecture test.** Is 7-band structure functional or cosmetic?
- Entropy differs across bands (low bands lower)
- Collapse cascades from low to high
- Weight change rate differs by band

**Key Predictions:**
- Monotonic entropy ordering (0 < 1 < ... < 6)
- Cascade correlation ρ > 0.5
- Weight rate ratio > 10× (band 6 vs band 0)

**If this fails:** Spectral structure is cosmetic. Simplify architecture.

---

### TIER ★★ CORE (BEC Validation)

Six experiments that test the core BEC framework:

---

#### 004: Phase Transition Sharpness ★★
**File:** `004_EXP_PHASE_TRANSITION_SHARPNESS.md`

**THE BEC test.** Attention = g|ψ|² predicts critical phenomena.
- Is collapse a genuine phase transition?
- Do critical exponents match BEC (mean-field)?
- Is there a critical "temperature"?

**Key Predictions:**
- β ≈ 0.5 (order parameter exponent)
- γ ≈ 1.0 (susceptibility exponent)
- Power-law fits with R² > 0.9

**If this fails:** Collapse is gradual optimization, not phase transition.

---

#### 005: Conservation Laws ★★
**File:** `005_EXP_CONSERVATION_LAWS.md`

**Deep structure test.** Conservation reveals symmetry.
- Is there a conserved quantity during inference?
- Does Parseval's theorem hold exactly?
- What symmetries exist?

**Key Predictions:**
- Normalization always holds (softmax)
- Some energy-like quantity stable (CV < 1%)
- Parseval error < 0.1%

**If this fails:** System is purely dissipative. Simpler than BEC.

---

#### 006: Heresy Detection ★★
**File:** `006_EXP_HERESY_DETECTION.md`

**Validity test.** Are measurements real or artifacts?
- Is spectral leakage measurable?
- Does windowing help?
- Are edge effects significant?

**Key Predictions:**
- Leakage without window: -13 dB
- Leakage with Hamming: -43 dB
- Windowing reduces error > 5%

**If severe heresies found:** All prior results suspect. Must re-evaluate.

---

#### 008: Quasiparticle Dispersion ★★
**File:** `008_EXP_QUASIPARTICLE_DISPERSION.md`

**Action Quanta as emergent excitations.**
- Do perturbations follow Bogoliubov dispersion?
- Low-k: collective (sound-like)
- High-k: particle-like (local)

**Key Predictions:**
- Linear E(k) at low k
- Quadratic E(k) at high k
- Crossover at mid-bands

**If this fails:** Action Quanta are not quasiparticles.

---

#### 009: Grokking as Condensation ★★
**File:** `009_EXP_GROKKING_AS_CONDENSATION.md`

**Training phase transition = BEC?**
- Grokking shows phase transition signatures
- Critical exponents match Exp 004
- Memorization → condensation

**Key Predictions:**
- Entropy drops at grokking
- Same β, γ as inference collapse
- Correlations build before transition

**If this fails:** Grokking is different phenomenon from inference collapse.

---

#### 025: Synergy-Redundancy Transition ★★ NEW
**File:** `025_EXP_SYNERGY_REDUNDANCY_TRANSITION.md`

**THE PID mechanism test.** Is collapse CAUSED BY synergy→redundancy conversion?
- Track PID components (synergy, redundancy, unique) over time
- Align to collapse events
- Test if synergy drop CAUSES collapse (Granger causality)
- Test if total information is conserved

**Key Predictions:**
- Synergy drops >50% at collapse
- Redundancy increases >100% at collapse
- I_total conserved within 10%
- High synergy predicts imminent collapse

**If this fails:** PID changes are correlate, not cause. Simpler mechanism.

**Foundation:** Williams & Beer (2010), Mediano et al. (2021), Sparacino et al. (2025)

---

### TIER ★ SUPPORTING (Depth and Applications)

Seven experiments that deepen understanding:

---

#### 007: Wavefront Interference Collapse ★
**File:** `007_EXP_WAVEFRONT_INTERFERENCE_COLLAPSE.md`

**Grand synthesis.** The lightning model visualized.
- Does error propagate like stepped leaders?
- Is collapse like the return stroke?
- Can we visualize the belief manifold dynamics?

**Key Predictions:**
- Branching trajectories visible
- Sudden collapse events
- Entropy drop correlates with path selection

---

#### 010: Tickling Techniques ★
**File:** `010_EXP_TICKLING_TECHNIQUES.md`

**Cheap probing for practical speedup.**
- Can entropy predict collapse destination?
- Does temperature sweep reveal fragility?
- First-token prediction quality?

**Key Predictions:**
- Entropy-error correlation > 0.5
- 10× cost savings possible

---

#### 011: Prompt Spectral Structure ★
**File:** `011_EXP_PROMPT_SPECTRAL_STRUCTURE.md`

**Do prompts have frequency?**
- Low-freq (role) vs high-freq (style)
- Different sensitivity to perturbation
- Spectral decomposition of prompts

**Key Predictions:**
- Role components are load-bearing
- Style components are expendable

---

#### 012: Wormhole Activation ★
**File:** `012_EXP_WORMHOLE_ACTIVATION.md`

**When do wormholes fire?**
- Content-based vs noise-based triggering
- Low-freq first guides high-freq matching
- Correlation with collapse

**Key Predictions:**
- Structured activation pattern
- Precedes entropy drop

---

#### 013: Differential LR Validation ★
**File:** `013_EXP_DIFFERENTIAL_LR_VALIDATION.md`

**Does LR hierarchy work?**
- Differential vs uniform learning rate
- Speed, generalization, stability
- Actual weight change rates

**Key Predictions:**
- 20% faster convergence
- 5% better generalization

---

---

### TIER ○ EXPLORATORY (Speculative)

Four bold experiments testing the limits:

---

#### 014: Critical Velocity ○
**File:** `014_EXP_CRITICAL_VELOCITY.md`

**Is there a coherence breakdown threshold?**
- Input velocity above which processing fails
- Analogous to superfluid critical velocity
- Relates to temporal Nyquist

**Key Predictions:**
- Sharp threshold exists
- Qualitative change in behavior above threshold

---

#### 015: Attention Vortices ○
**File:** `015_EXP_ATTENTION_VORTICES.md`

**Do topological defects exist?**
- Vortex-like circulation patterns
- Quantized winding numbers
- Topological stability

**Key Predictions:**
- Vortices detectable
- Circulation is quantized

---

#### 016: Cross-Model Manifold ○
**File:** `016_EXP_CROSS_MODEL_MANIFOLD.md`

**Are manifolds universal?**
- Different models, similar structure
- Prompt transfer correlates with alignment
- Universal geometry from similar training

**Key Predictions:**
- High manifold alignment
- Transfer ∝ alignment

---

#### 017: MDL Atomic Truth ○
**File:** `017_EXP_MDL_ATOMIC_TRUTH.md`

**What's the minimum prompt?**
- Concepts have minimum description length
- Atomic prompts are fully load-bearing
- Compression = understanding

**Key Predictions:**
- MDL is discoverable
- All words in atomic prompt matter

---

#### 018: Pump Cycle Dynamics ★
**File:** `018_EXP_PUMP_CYCLE_DYNAMICS.md`

**Tension-Discharge-Recovery cycle.**
- Periodic oscillation in entropy
- Consistent cycle period
- Correlation with prediction accuracy

**Key Predictions:**
- Quasi-periodic behavior
- Discharge shorter than tension

---

#### 019: Belief Geometry ★
**File:** `019_EXP_BELIEF_GEOMETRY.md`

**Does uncertainty have shape?**
- Error forms crescent for moving objects
- Shape encodes velocity uncertainty
- Wave packet interpretation

**Key Predictions:**
- Crescent width ∝ speed uncertainty
- Orientation ∝ direction uncertainty

---

#### 020: Cross-Band Flow ★
**File:** `020_EXP_CROSS_BAND_FLOW.md`

**How do bands communicate?**
- Low→High (top-down guidance)
- High→Low (bottom-up detail)
- Symmetric pairs (0↔6, 1↔5, 2↔4)

**Key Predictions:**
- Asymmetric temporal pattern
- Symmetric pairs strongest

---

### TIER ★★ CORE: Harmony and Coherence ★ NEW ★

Two experiments testing the Pythagorean principle:

---

#### 021: Attention Comma ★★
**File:** `021_EXP_ATTENTION_COMMA.md`

**Is error distributed like equal temperament? (Pythagorean comma)**
- Compare "ideal" per-position attention vs actual
- Measure how error is spread across all positions
- Test temperature's effect on error distribution

**Key Predictions:**
- Error distributed, not concentrated
- Distribution more uniform at high temperature
- Pattern similar to equal temperament

**Reference:** `foundations/HARMONY_AND_COHERENCE.md`

---

#### 022: Band Phase Locking ★★
**File:** `022_EXP_BAND_PHASE_LOCKING.md`

**Do bands lock to rational ratios? (Coupled oscillators)**
- Cross-correlation between spectral band activations
- Look for integer/rational frequency relationships
- Compare trained vs random initialization

**Key Predictions:**
- Rational ratios emerge during training
- Random init shows no such relationships
- Correlations strongest between complementary pairs

**Reference:** `foundations/HARMONY_AND_COHERENCE.md`

---

#### 023: Timeline Coherence ★
**File:** `023_EXP_TIMELINE_COHERENCE.md`

**Does the past shift after collapse? (Pythagorean principle)**
- Measure representation similarity before/after collapse
- Check if past representations adjust for consistency
- Test homeostat's "retrocausal" coherence effect

**Key Predictions:**
- Past representations shift after collapse
- Shift direction toward coherent attractor
- Phase locking across temporal representations

**Reference:** `foundations/HARMONY_AND_COHERENCE.md`

---

#### 024: Resonant Wormholes ★
**File:** `024_EXP_RESONANT_WORMHOLES.md`

**Do wormholes prefer complementary bands? (Pythagorean resonance)**
- Wormhole activation at different band pairs
- Look for enhanced activation at 0↔6, 1↔5, 2↔4
- Test if activation correlates with frequency ratio

**Key Predictions:**
- Complementary pairs show higher activation
- Activation proportional to frequency ratio
- Non-complementary pairs show lower correlation

**Reference:** `foundations/HARMONY_AND_COHERENCE.md`

---

### TIER ★★ CORE: Applied Spectral Belief

---

#### 032: Mini AKIRA Plasma Controller ★★ **IN PROGRESS**
**Folder:** `experiments_supporting_literature_and_reference/032_EXP_MINI_AKIRA_PLASMA_CONTROLLER/`

**First real-world application of Spectral Belief Machine (SBM).**
Tests whether spectral decomposition helps predict turbulent plasma dynamics.

**Versions:**
- `v1` (original): Pure SBM prediction, 7 bands, won against Flat baseline
- `v2.x`: Added control conditioning (Pre-FFT vs Post-FFT ablation)
- `v3.x`: Added wormhole attention (v3 broken, v3.1 fixed with sequence batching)
- `v4.x`: Full ablation study (2/3/7 bands, temporal/neighbor/wormhole attention)

**Current Results (v4.1):**

| Model | Final Loss | Notes |
|-------|-----------|-------|
| SBM_3B_None | 0.008950 | **Best** - 3 bands, no attention |
| SBM_2B_None | 0.009665 | 2 bands |
| SBM_7B_None | 0.009642 | 7 bands |
| SBM_3B_Temporal | 0.009554 | Unstable training |
| SBM_3B_Neighbor | 0.009649 | Slow, slight instability |
| SBM_3B_Wormhole | OOM | Crashed (fixed in v4.2) |

**Key Findings:**
1. **3 bands optimal**: Better than 2 or 7 bands for this problem
2. **Attention mechanisms hurt (so far)**: Due to random batching destroying temporal coherence
3. **Sequence-based training needed**: v3.1 and v4.2 fix this with trajectory-based batching
4. **Wormhole must pool first**: Per-pixel wormhole causes OOM; pooled wormhole works

**Key Predictions:**
- SBM should outperform Flat ConvNet on turbulent dynamics
- 3-band spectral decomposition captures relevant physics
- Proper temporal attention should help (with correct batching)

**If SBM loses to Flat:** Spectral decomposition not beneficial for this domain.

**References:** Plasma control literature, Lorenz chaos control (Pyragas method)

---

#### 033: Spectral Band Attention ★ **DESIGNED**
**Folder:** `experiments_supporting_literature_and_reference/033_EXP_SPECTRAL_BAND_ATTENTION/`

**Three attention types for spectral bands:**
1. **Per-Position Temporal**: Each (i,j) attends to its own history (Top-K sparse)
2. **Neighbor**: 3x3 local spatial attention for physics modeling
3. **Wormhole**: Sparse non-local via similarity gating (pooled features)

**Depends on:** 032 (establishes SBM baseline)

**Key Design Insight:** Attention operates on band features, not raw pixels.
Wormhole uses pooled global features to find similar states across time.

---

---

## Validation Gates

```
EXPERIMENTAL VALIDATION STRUCTURE

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  GATE 0: ONTOLOGICAL ★★★ (Experiment 000)                              │
│  ─────────────────────────────────────────                              │
│  MUST PASS (or reframe) before proceeding:                            │
│  • 000: AQ identifiable, universal, irreducible, actionable          │
│                                                                         │
│  If passes → AQ are real, proceed with measuring their dynamics      │
│  If fails → Reframe AQ as emergent, not fundamental. Still proceed   │
│             but interpret all subsequent results accordingly.         │
│                                                                         │
│  ────────────────────────────────────────────────────────────────────  │
│                                                                         │
│  GATE 1: FOUNDATION ★★★ (Experiments 001-003)                          │
│  ───────────────────────────────────────────                            │
│  ALL THREE MUST PASS to proceed:                                       │
│  • 001: Entropy measurable and varying                                │
│  • 002: Collapse sudden and correlated with error                    │
│  • 003: Bands differ in dynamics                                      │
│                                                                         │
│  If any fail → Theory untestable or fundamentally wrong              │
│                                                                         │
│  ────────────────────────────────────────────────────────────────────  │
│                                                                         │
│  GATE 2: BEC CORE ★★ (Experiments 004-006, 008-009)                    │
│  ─────────────────────────────────────────────────                      │
│  AT LEAST 4 OF 5 must pass to claim BEC analogy:                      │
│  • 004: Critical exponents near mean-field (β ≈ 0.5)                 │
│  • 005: Some quantity conserved                                       │
│  • 006: Heresies detectable and manageable                           │
│  • 008: Quasiparticle dispersion matches                             │
│  • 009: Grokking shows same physics                                  │
│                                                                         │
│  If 004 fails hard → BEC is metaphor, not physics                    │
│  If 006 reveals severe artifacts → All results suspect               │
│                                                                         │
│  ────────────────────────────────────────────────────────────────────  │
│                                                                         │
│  GATE 3: SUPPORTING ★ (Experiments 007, 010-013, 018-020)             │
│  ─────────────────────────────────────────────────────                  │
│  Depth and applications:                                               │
│  • These provide practical value and understanding                   │
│  • Failure suggests refinement, not rejection                        │
│  • Success opens new applications                                     │
│                                                                         │
│  ────────────────────────────────────────────────────────────────────  │
│                                                                         │
│  GATE 4: EXPLORATORY ○ (Experiments 014-017)                           │
│  ───────────────────────────────────────────                            │
│  Bold predictions:                                                      │
│  • Success = major validation                                         │
│  • Failure = boundary of theory                                       │
│  • Both outcomes valuable                                              │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

OVERALL VERDICT:

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  If Gates 1 + 2 pass:  THEORY VALIDATED                               │
│  If Gate 1 passes, 2 partial: THEORY NEEDS REVISION                   │
│  If Gate 1 fails: THEORY FALSIFIED (or measurement broken)            │
│                                                                         │
│  FALSIFICATION IS SUCCESS.                                             │
│  It tells us where to look next.                                      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Running Experiments

### Prerequisites

```python
# Core infrastructure required
from experiments.entropy_tracker import compute_attention_entropy
from experiments.collapse_detector import detect_collapse_events
from experiments.band_analyzer import analyze_band_entropy
from experiments.phase_analyzer import fit_critical_exponents
from experiments.conservation_checker import check_parseval
from experiments.heresy_detector import measure_spectral_leakage
# etc.
```

### Standard Protocol

```
FOR EACH EXPERIMENT:

1. Read the experiment document thoroughly
2. Implement the apparatus (code in spectral_attention/experiments/)
3. Run the protocol as specified
4. Fill in results section
5. Draw conclusions
6. Update status in this index
7. If successful, proceed to dependent experiments
8. If failed, document failure and investigate
```

### Recording Results

Results should be recorded in the experiment document itself:
- Fill in the `[ TO BE FILLED AFTER EXPERIMENT ]` sections
- Include plots and visualizations
- State clear verdicts on hypotheses
- Link to any data/artifacts produced

---

## References

Each experiment references its own scientific basis. Core references:

1. Shannon, C.E. (1948). A Mathematical Theory of Communication.
2. Vaswani, A. et al. (2017). Attention Is All You Need.
3. Landau, L.D. & Lifshitz, E.M. (1980). Statistical Physics.
4. Pethick, C.J. & Smith, H. (2008). Bose-Einstein Condensation in Dilute Gases.
5. Bracewell, R.N. (2000). The Fourier Transform and Its Applications.
6. Power, A. et al. (2022). Grokking: Generalization Beyond Overfitting.
7. Bogoliubov, N.N. (1947). On the Theory of Superfluidity.
8. Williams, P.L. & Beer, R.D. (2010). Nonnegative Decomposition of Multivariate Information.
9. Mediano, P.A.M. et al. (2021). Towards an Extended Taxonomy of Information Dynamics via Integrated Information Decomposition. arXiv:2109.13186.
10. Sparacino, L. et al. (2025). Partial Information Rate Decomposition. Physical Review Letters, 135, 187401.

---

*Oscar Goldman — Shogu Research Group @ Datamutant.ai*

*"We build the argument with calm precision, step by step. Each experiment stands on the shoulders of those before it. The goal is not to confirm our beliefs, but to discover the truth — whatever it may be."*


