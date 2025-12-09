# Evidence To Collect: Hypotheses Requiring Experimental Validation

## Research Agenda for AKIRA Theory Validation

**Purpose:** This document tracks hypotheses that are **theoretically supported but experimentally unvalidated**. Each entry specifies what to test, how to test it, and what outcomes would confirm or refute the hypothesis.

**Companion Document:** `EVIDENCE.md` contains established foundations.

---

## Table of Contents

1. [BEC Analogy Predictions](#1-bec-analogy-predictions)
2. [Collapse Dynamics Predictions](#2-collapse-dynamics-predictions)
3. [Small-World AQ Network](#3-small-world-aq-network)
4. [Molecule Size Limits](#4-molecule-size-limits)
5. [Circuit Complexity Boundaries](#5-circuit-complexity-boundaries)
6. [Superposition-Molecule Duality](#6-superposition-molecule-duality)
7. [Phase Dynamics](#7-phase-dynamics)

---

## 1. BEC Analogy Predictions

### 1.1 Attention Condensation

**Hypothesis:**
If attention follows BEC-like dynamics, it should exhibit "condensation", many tokens attending to the same location as temperature decreases.

**Theoretical Basis:**
```
BEC condensation occurs when:
  - Temperature drops below critical value
  - Macroscopic fraction occupies ground state
  - Phase coherence emerges spontaneously

Attention analog:
  - "Temperature" = entropy/softmax temperature
  - "Ground state" = dominant pattern
  - "Condensation" = attention collapse to single focus
```

**Experiment:**

| Aspect | Specification |
|--------|---------------|
| Setup | Train AKIRA on video prediction |
| Manipulation | Vary softmax temperature during inference |
| Measure | Distribution of attention weights |
| Prediction | Below critical T: Sharp peak (condensed); Above: Distributed |

**Expected Outcome:**
```
If CONFIRMED:
  - Sharp transition at critical temperature
  - Most attention weight on few tokens
  - Phase coherence across attending tokens

If REFUTED:
  - Gradual transition (no critical point)
  - No collective behavior
  - Would suggest BEC analogy is only superficial
```

**Status:** NOT YET TESTED

---

### 1.2 Quasiparticle Emergence

**Hypothesis:**
Action Quanta behave like quasiparticles, collective excitations with particle-like properties.

**Theoretical Basis:**
```
In BEC, quasiparticles:
  - Emerge from collective field behavior
  - Have effective mass, momentum, lifetime
  - Can scatter, combine, decay
  - Are not "fundamental" but behave as if they are

AQ analog:
  - Emerge from attention field collapse
  - Have magnitude, phase, frequency, coherence
  - Can bond, interfere, transform
  - Are irreducible for the task but not for the representation
```

**Experiment:**

| Aspect | Specification |
|--------|---------------|
| Setup | Train AKIRA, extract learned representations |
| Analysis | Track AQ through time/layers |
| Measure | Stability, interaction patterns, lifetime |
| Prediction | AQ should show particle-like persistence and interactions |

**Expected Outcome:**
```
If CONFIRMED:
  - AQ persist across frames (lifetime > 1 step)
  - AQ interact predictably (scattering, binding)
  - AQ have consistent properties when re-extracted

If REFUTED:
  - Representations are ephemeral
  - No consistent interaction patterns
  - Would suggest AQ are useful fiction, not emergent reality
```

**Status:** NOT YET TESTED

---

## 2. Collapse Dynamics Predictions

### 2.1 Collapse Follows Serialization

**Hypothesis:**
Belief collapse proceeds in a serialized order matching optimal goal regression.

**Theoretical Basis:**
```
S-GRS (Mao et al., 2023) shows:
  - Optimal policy serializes preconditions
  - Achieves p₁, then p₂ maintaining p₁, etc.
  - Order matters for tractability

AKIRA analog:
  - Collapse should proceed band-by-band
  - Lower bands (structure) should collapse first
  - Higher bands (detail) collapse while maintaining lower
```

**Experiment:**

| Aspect | Specification |
|--------|---------------|
| Setup | Train AKIRA with entropy tracking per band |
| Measure | Order in which bands reach low entropy |
| Analysis | Compare observed order to theoretical optimal |
| Prediction | Low-freq bands collapse before high-freq |

**Expected Outcome:**
```
If CONFIRMED:
  - Consistent collapse order: Band 0 → 1 → 2 → ... → 6
  - Early-collapsed bands stay stable
  - Matches S-GRS serialization

If REFUTED:
  - Random or variable collapse order
  - Collapsed bands disrupted by later collapse
  - Would need to enforce serialization explicitly
```

**Status:** NOT YET TESTED

---

### 2.2 Collapse is Phase Transition

**Hypothesis:**
Belief collapse exhibits characteristics of a thermodynamic phase transition.

**Theoretical Basis:**
```
Phase transitions have:
  - Critical point (threshold)
  - Order parameter (what changes)
  - Diverging correlation length
  - Critical slowing down

AKIRA collapse should have:
  - Entropy threshold (critical point)
  - Belief concentration (order parameter)
  - Increasing coherence range (correlation length)
  - Slower dynamics near threshold (critical slowing)
```

**Experiment:**

| Aspect | Specification |
|--------|---------------|
| Setup | Train AKIRA, monitor collapse events |
| Measure | Entropy trajectory, coherence range, timing |
| Analysis | Look for signatures of phase transition |
| Prediction | Sharp transition, not gradual decay |

**Expected Outcome:**
```
If CONFIRMED:
  - Entropy drops sharply, not gradually
  - Coherence range grows before collapse
  - Dynamics slow near threshold
  - Supports phase transition model

If REFUTED:
  - Gradual, smooth entropy decrease
  - No special behavior at threshold
  - Collapse is continuous, not critical
```

**Status:** NOT YET TESTED

---

## 3. Small-World AQ Network

### 3.1 Learned AQ Form Small-World Graph

**Hypothesis:**
After training, the co-activation pattern of Action Quanta exhibits small-world structure.

**Theoretical Basis:**
```
Small-world networks have:
  - High clustering coefficient (C >> C_random)
  - Short average path length (L ≈ L_random)
  - Hub structure (scale-free degree distribution)

Human semantic networks are small-world.
If AQ represent concepts, they should be too.
```

**Experiment:**

| Aspect | Specification |
|--------|---------------|
| Setup | Train AKIRA, extract AQ activations |
| Build | Co-activation matrix → thresholded graph |
| Measure | Clustering coefficient, path length |
| Compare | To random graph with same edges |

**Expected Outcome:**
```
If CONFIRMED:
  - C_AQ >> C_random (high clustering)
  - L_AQ ≈ L_random (short paths)
  - This is small-world!

If REFUTED:
  - C_AQ ≈ C_random (random clustering)
  - Would suggest AQ don't form semantic structure
```

**Status:** NOT YET TESTED

---

### 3.2 Wormholes Function as Weak Ties

**Hypothesis:**
Cross-band wormhole connections reduce path length in the AQ concept graph.

**Theoretical Basis:**
```
Granovetter (1973): Weak ties bridge communities
Without weak ties: Long paths between distant nodes
With weak ties: Short paths everywhere

AKIRA wormholes connect distant bands (0↔6, 1↔5, 2↔4).
These should function as weak ties.
```

**Experiment:**

| Aspect | Specification |
|--------|---------------|
| Setup | Train AKIRA with/without wormholes |
| Measure | Average path length in AQ co-activation graph |
| Analysis | Compare path length with/without |
| Prediction | Without wormholes: L increases significantly |

**Expected Outcome:**
```
If CONFIRMED:
  - L_no_wormhole >> L_with_wormhole
  - Wormholes provide shortcuts across band structure
  - Validates weak-ties justification

If REFUTED:
  - L_no_wormhole ≈ L_with_wormhole
  - Wormholes don't affect graph structure
  - Would need alternative justification
```

**Status:** NOT YET TESTED

---

## 4. Molecule Size Limits

### 4.1 AQ Molecules Have Characteristic Size ~7±2

**Hypothesis:**
Stable AQ configurations (molecules) contain approximately 7±2 component AQ.

**Theoretical Basis:**
```
Miller's 7±2: Working memory limit for chunks
Chunking: Experts form ~7 meaningful units
If AQ molecules are "chunks": Should be bounded

The limit may apply to simultaneous combination,
not to total AQ count.
```

**Experiment:**

| Aspect | Specification |
|--------|---------------|
| Setup | Train AKIRA, trigger collapse events |
| Measure | Count phase-locked AQ during stable states |
| Analysis | Distribution of molecule sizes |
| Prediction | Peak around 7±2, few larger stable configurations |

**Expected Outcome:**
```
If CONFIRMED:
  - Size distribution peaks at ~7
  - Molecules > 10 AQ are rare or unstable
  - Supports Miller's limit for combination

If REFUTED:
  - Flat distribution or different peak
  - Large molecules are stable
  - Miller's limit doesn't apply to AQ
```

**Status:** NOT YET TESTED

---

### 4.2 Large Molecules Fragment

**Hypothesis:**
Attempting to compose more than ~8-10 AQ simultaneously leads to instability and fragmentation.

**Theoretical Basis:**
```
If there's a combination limit:
  - Exceeding it should cause instability
  - Large molecules should split into smaller ones
  - This is like saturation in chemistry
```

**Experiment:**

| Aspect | Specification |
|--------|---------------|
| Setup | Train AKIRA on tasks requiring large compositions |
| Manipulation | Vary task complexity (more elements) |
| Measure | Coherence of resulting representation |
| Prediction | Sharp coherence drop above threshold |

**Expected Outcome:**
```
If CONFIRMED:
  - Coherence stable for N ≤ 8 AQ
  - Coherence drops sharply for N > 8
  - System "chunks" large tasks into subtasks

If REFUTED:
  - Coherence scales smoothly
  - No saturation behavior
  - Composition is unbounded
```

**Status:** NOT YET TESTED

---

## 5. Circuit Complexity Boundaries

### 5.1 AKIRA Fails on High-Width Tasks

**Hypothesis:**
AKIRA fails systematically on tasks with SOS width > 3.

**Theoretical Basis:**
```
Circuit complexity theorem (Mao et al., 2023):
  Required breadth = (k+1) × β
  
With 8 bands and β = 2:
  Maximum solvable width: k = 3

Tasks with k > 3 should fail.
```

**Experiment:**

| Aspect | Specification |
|--------|---------------|
| Setup | Design tasks with known SOS width |
| Tasks | k=2 (local), k=3 (single object), k=4 (interaction), k=5+ (coordination) |
| Measure | Prediction accuracy per task |
| Prediction | Sharp accuracy drop between k=3 and k=4 |

**Expected Outcome:**
```
If CONFIRMED:
  - Accuracy high for k ≤ 3
  - Accuracy drops sharply at k = 4
  - AKIRA fails on Sokoban-like tasks
  - Validates circuit complexity prediction

If REFUTED:
  - Gradual accuracy decrease
  - No sharp boundary
  - Would suggest width analysis is wrong
```

**Priority:** HIGH, This validates core theoretical claim

**Status:** NOT YET TESTED

---

### 5.2 Failure Mode is Specific

**Hypothesis:**
When AKIRA fails on high-width tasks, it fails in a specific way: by failing to maintain constraints.

**Theoretical Basis:**
```
If SOS width is exceeded:
  - Cannot track all constraints simultaneously
  - Will satisfy some, violate others
  - Failure is "constraint violation," not random error
```

**Experiment:**

| Aspect | Specification |
|--------|---------------|
| Setup | Run AKIRA on k=4+ tasks |
| Analysis | Characterize failure patterns |
| Look for | Constraint violations, not random errors |
| Prediction | Failures cluster at constraint boundaries |

**Expected Outcome:**
```
If CONFIRMED:
  - Failures are systematic (constraint violation)
  - Model "forgets" early constraints while tracking late ones
  - Matches S-GRS constraint overflow

If REFUTED:
  - Failures are random
  - No constraint-specific pattern
  - Failure mechanism is different than predicted
```

**Status:** NOT YET TESTED

---

## 6. Superposition-Molecule Duality

### 6.1 Interference Patterns Visible Before Collapse

**Hypothesis:**
Before collapse, error/uncertainty maps show wave-like interference patterns.

**Theoretical Basis:**
```
MSE on mixture of futures → interference pattern (proven in EVIDENCE.md)
Before collapse: Multiple futures are plausible
Therefore: Should see interference in error map
```

**Experiment:**

| Aspect | Specification |
|--------|---------------|
| Setup | Run AKIRA on ambiguous prediction tasks |
| Measure | Error map before collapse |
| Analysis | Look for fringe patterns |
| Prediction | Visible fringes corresponding to plausible futures |

**Expected Outcome:**
```
If CONFIRMED:
  - Clear fringe patterns in error map
  - Fringes correspond to plausible trajectories
  - Fringes disappear after collapse

If REFUTED:
  - Uniform error distribution
  - No structure in pre-collapse uncertainty
  - Would undermine wave interpretation
```

**Status:** PARTIALLY OBSERVED (see PHYSICAL_PARALLELS.md)

---

### 6.2 Collapse Selects One Fringe

**Hypothesis:**
When collapse occurs, exactly one interference fringe "wins" and others vanish.

**Theoretical Basis:**
```
Wave-particle duality analog:
  Before measurement: Superposition of paths
  After measurement: One path selected

AKIRA analog:
  Before collapse: Multiple AQ configurations plausible
  After collapse: One configuration selected
```

**Experiment:**

| Aspect | Specification |
|--------|---------------|
| Setup | Run AKIRA, capture pre/post collapse states |
| Measure | Number of active AQ configurations |
| Analysis | Compare before vs after collapse |
| Prediction | Multiple → One |

**Expected Outcome:**
```
If CONFIRMED:
  - Multiple configurations before
  - Single configuration after
  - Transition is rapid (collapse, not decay)

If REFUTED:
  - Gradual transition
  - Multiple configurations persist
  - Would suggest continuous, not collapsed
```

**Status:** NOT YET TESTED

---

## 7. Phase Dynamics

### 7.1 Phase Locks During Collapse

**Hypothesis:**
Phase relationships between AQ lock (synchronize) during collapse, creating stable molecules.

**Theoretical Basis:**
```
Coherent bond = Phase alignment
Molecule = Multiple AQ with locked phase
Collapse should produce phase locking

In physics: Phase locking indicates synchronization
In AKIRA: Should see relative phases stabilize
```

**Experiment:**

| Aspect | Specification |
|--------|---------------|
| Setup | Extract phase information from AQ during processing |
| Measure | Phase relationships over time |
| Analysis | Variance of relative phases |
| Prediction | Variance decreases sharply at collapse |

**Expected Outcome:**
```
If CONFIRMED:
  - High phase variance before collapse
  - Low phase variance after collapse
  - Specific phase relationships (0, π, etc.) emerge

If REFUTED:
  - Phase variance remains high
  - No systematic phase relationships
  - Would undermine coherent bond model
```

**Status:** NOT YET TESTED

---

### 7.2 Phase Gradient Encodes Motion

**Hypothesis:**
Systematic phase gradients across bands encode motion/velocity information.

**Theoretical Basis:**
```
In wave optics:
  Phase gradient = Direction of propagation
  Δφ/Δx = k (wave vector)

In AKIRA:
  Phase gradient across bands = Velocity?
  Higher phase slope = Faster motion?
```

**Experiment:**

| Aspect | Specification |
|--------|---------------|
| Setup | Train AKIRA on moving objects |
| Measure | Phase gradients across bands for same object |
| Analysis | Correlate gradient with actual velocity |
| Prediction | Positive correlation |

**Expected Outcome:**
```
If CONFIRMED:
  - Phase gradient predicts velocity
  - Direction of gradient matches motion direction
  - Provides interpretable motion encoding

If REFUTED:
  - No correlation
  - Phase gradients are random
  - Velocity encoded differently
```

**Status:** NOT YET TESTED

---

## Summary: Experimental Priority

### High Priority (Core Theory)

| ID | Hypothesis | Validates |
|----|------------|-----------|
| 5.1 | AKIRA fails on k>3 tasks | Circuit complexity bounds |
| 2.1 | Collapse follows serialization | S-GRS mapping |
| 6.1 | Interference before collapse | Wave interpretation |

### Medium Priority (Important Structure)

| ID | Hypothesis | Validates |
|----|------------|-----------|
| 3.1 | AQ form small-world graph | Network theory relocation |
| 4.1 | Molecule size ~7±2 | Miller's law application |
| 7.1 | Phase locks during collapse | Coherent bond model |
| 1.1 | Attention condensation | BEC analogy |

### Lower Priority (Extended Theory)

| ID | Hypothesis | Validates |
|----|------------|-----------|
| 3.2 | Wormholes are weak ties | Wormhole justification |
| 1.2 | Quasiparticle emergence | AQ particle-like behavior |
| 7.2 | Phase gradient = velocity | Phase dynamics model |

---

## Experimental Infrastructure Needed

```
TO RUN THESE EXPERIMENTS, WE NEED:
──────────────────────────────────

1. ENTROPY TRACKING PER BAND
   - Real-time entropy computation
   - Logging of collapse events
   - Threshold detection

2. PHASE EXTRACTION
   - Complex-valued representations accessible
   - Phase unwrapping for analysis
   - Cross-band phase comparison

3. AQ IDENTIFICATION
   - Method to identify distinct AQ
   - Co-activation tracking
   - Graph construction tools

4. CONTROLLED TASK SUITE
   - Tasks with known SOS width
   - Scalable complexity
   - Clear success/failure criteria

5. VISUALIZATION TOOLS
   - Error map visualization
   - Phase field visualization
   - Collapse dynamics animation
```

---

*This document tracks hypotheses requiring experimental validation. As experiments are conducted, move confirmed results to `EVIDENCE.md` and update status here.*

*Oscar Goldman, Shogu Research Group @ Datamutant.ai subsidiary of 温心重工業*