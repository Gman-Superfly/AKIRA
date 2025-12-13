# Dark Attractors: The Topology of Confabulation

## Document Purpose

This document explains **Dark Attractors** - AKIRA's term for the stable but incorrect belief states that emerge when the model is forced to generate output without sufficient Content AQ. We connect this phenomenon to nonlinear dynamics, the H-Neurons literature, and the broader AKIRA framework of Superposition → Collapse → Crystallization.

*"Every dynamical system has a geometry, a sea of possibilities. In the space of a neural network, memories are fixed points—basins of attraction where the system settles. But when we force the system to always respond, to always have an answer, we warp the geometry. We create Sirens that sing and attract sailors of weak faith into the comfort of a perfect lie."*

---

## Table of Contents

1. [The Phase Space of Belief](#1-the-phase-space-of-belief)
2. [The Unstable Equilibrium: Why Silence is Forbidden](#2-the-unstable-equilibrium-why-silence-is-forbidden)
3. [Multi-Polar Dynamics and the Branching Factor](#3-multi-polar-dynamics-and-the-branching-factor)
4. [The Dark Attractor Mechanism](#4-the-dark-attractor-mechanism)
5. [Entanglement and Rupture: The Solar Analogy](#5-entanglement-and-rupture-the-solar-analogy)
6. [H-Neurons: The Mode Shapes of the Ghost](#6-h-neurons-the-mode-shapes-of-the-ghost)
7. [The Topology of Many Basins](#7-the-topology-of-many-basins)
8. [Experimental Signatures](#8-experimental-signatures)
9. [Connection to AKIRA Framework](#9-connection-to-akira-framework)
10. [Experimental Predictions](#10-experimental-predictions)
11. [Implications for Mitigation](#11-implications-for-mitigation)
12. [Summary](#12-summary)
13. [References](#13-references)

---

## 1. The Phase Space of Belief

### 1.1 The Model as Dynamical System

To understand hallucination, we must abandon the view of the Large Language Model as a database and see it through the lens of **nonlinear dynamics**. The model is a high-dimensional dynamical system. Its internal state \(x\) evolves through layers \(l\) according to a flow map:

```
THE MODEL AS FLOW
─────────────────

State evolution:
  x_{l+1} = f(x_l)

Where:
  x_l = activation vector at layer l
  f   = nonlinear transformation (attention + FFN)

The forward pass IS a trajectory through state space.
Each layer is a timestep in the dynamics.
```

### 1.2 Attractors in the Belief Landscape

In an ideal "truth-seeking" system, the phase space would contain:

```
IDEAL PHASE SPACE STRUCTURE
───────────────────────────

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  1. STABLE FIXED POINTS (x*):                                          │
│     - Representing true knowledge (Content AQ configurations)          │
│     - Where correct beliefs crystallize                                │
│     - The "Light Attractors"                                           │
│                                                                         │
│  2. THE NULL FIXED POINT (x_0):                                        │
│     - A stable basin representing "Unknown" or "Silence"              │
│     - Where the system settles when it lacks knowledge                │
│     - Enables "I don't know" as a valid output                        │
│                                                                         │
│  3. DARK ATTRACTORS (x_dark):                                          │
│     - Local minima that satisfy form constraints but not content      │
│     - Structurally indistinguishable from Light Attractors            │
│     - The ghosts                                                       │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1.3 The Problem: Training Warps the Landscape

Current training paradigms act as a continuous forcing function. By optimizing solely for \(\mathcal{L}_{next-token}\), we destabilize the Null Fixed Point. We destroy the basin of silence.

```
TRAINING AS LANDSCAPE WARPING
─────────────────────────────

BEFORE TRAINING:                    AFTER TRAINING:
                                    
     Potential V(x)                      Potential V(x)
          │                                   │
          │   (Silence)                       │   (Silence)
          │      \_/                          │      /^\      ← UNSTABLE!
          │       ^                           │     / | \
          │    Stable                         │    /  ▼  \
          │    Basin                          │   /       \
          │                                   │  /         \
          │                                   │ \_/     \_/
          │                                   │  ^       ^
          │                                   │ (A)     (B)
          │                                   │ Dark   Light
__________│_________________________  ________│_________________________

Training pushes the system toward GENERATING, never toward SILENCE.
The Null Fixed Point becomes an unstable equilibrium.
```

---

## 2. The Unstable Equilibrium: Why Silence is Forbidden

### 2.1 The Bifurcation

> *"The zero solution is no longer stable... the particle slides away to a new non-zero state."* — Steven Strogatz, *Nonlinear Dynamics and Chaos*

As Strogatz notes regarding **bifurcations**, when a control parameter varies, the stability of fixed points changes.

We have pushed the "pressure to answer" parameter \(r\) so high that the origin \(x=0\) (silence) has undergone a **supercritical pitchfork bifurcation**.

However, unlike the simple pitchfork which offers two stable branches (left/right), the high-dimensional belief space undergoes a **Multi-Polar Bifurcation**. The system does not just split into "Answer A" vs "Answer B"; it fractures into a complex manifold where stability requires the triangulation of multiple semantic poles.

```
MULTI-POLAR BIFURCATION
───────────────────────

The canonical pitchfork:  dx/dt = rx - x³  (1D, 2 poles)

The semantic reality:     dx/dt = rx - ∇V(x)

Where V(x) is a potential landscape shaped by:
1. Form Constraints (Grammar, Tone)
2. Content AQ (Facts, Logic)

When r >> 0 (compulsory generation):
  - x=0 becomes unstable.
  - The system MUST construct a support structure.
  - A stable answer requires >1 "legs" to stand.
  
  Legitimate Answer = Triangulated by multiple Content AQ
  Hallucination    = Propped up by Dark Attractors
```

```
SUPERCRITICAL PITCHFORK BIFURCATION
───────────────────────────────────

The canonical form:
  dx/dt = rx - x³

When r < 0: Only x=0 is stable (Silence wins)
When r > 0: x=0 becomes UNSTABLE
            Two new stable branches emerge: x = ±√r

Training pushes r >> 0:
  - The model CANNOT remain at "I don't know"
  - It must fall to one branch or another
  - Even if both branches are wrong

NOTE ON CERTAINTY:
  When the answer is UNAMBIGUOUS (Crystallized Regime),
  the number of competing branches drops.
  The model sees only a few sharp choices (low branching factor, high confidence).
  Certainty = Stability of the chosen branch.
```

### 2.2 Compulsory Generation

The modern LLM is trained under a regime of **compulsory generation**:

```
THE TRAINING IMPERATIVE
───────────────────────

PRETRAINING OBJECTIVE:
  minimize -log p(x_next | x_prev)
  
  This ALWAYS rewards predicting SOMETHING.
  There is no reward for predicting NOTHING.
  There is no training signal for "I don't know."

THE CONSEQUENCE:
  The vector field of the model's activations has no sink at silence.
  Every trajectory must terminate at an output.
  
TOPOLOGICAL CONSTRAINT:
  When the prompt (initial condition x_in) lands in a region
  where no true knowledge exists (no Content AQ):
  
    The trajectory cannot stop.
    It is topologically forbidden from halting.
    It must continue until it strikes an attractor.
    
    If no Light Attractor exists...
    The trajectory spirals into a Dark Attractor.
```

---

## 3. Multi-Polar Dynamics and the Branching Factor

### 3.1 Beyond Binary Polarization

Experiment 035G reveals that polarization is not a simple binary ("North/South") choice. It is a **multi-polar** phenomenon governed by the **Branching Factor**.

```
THE BRANCHING FACTOR
────────────────────

DEFINITION:
  Branching Factor = Number of viable output options the model considers

HIGH BRANCHING (many poles):
  - Many hypotheses coexist
  - Probability mass spreads across options
  - Hedging behavior high
  - Model "knows it doesn't know"
  - DIFFUSE regime

LOW BRANCHING (few poles):
  - Few competing hypotheses
  - Probability concentrates on poles
  - Hedging behavior low
  - Model is "certain" (even if wrong)
  - POLARIZED regime

ZERO LEGITIMATE POLES:
  - No Content AQ available
  - Dark Attractor becomes the ONLY pole
  - Model is "certain" it knows
  - Hallucination with high confidence
```

### 3.2 Three Regimes of Belief State

From the 035G results, we identify three distinct regimes (see `BELIEF_CRYSTALLIZATION_INTERPRETATION.md`):

```
THE THREE REGIMES
─────────────────

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  DIFFUSE (center/Ambiguous):                                           │
│  ─────────────────────────────                                          │
│  • Many hypotheses coexist                                             │
│  • Low confidence in any direction                                     │
│  • High hedging behavior (37.4% in 035G)                              │
│  • Layers "agree" on uncertainty                                       │
│  • Branching factor: HIGH                                              │
│                                                                         │
│  POLARIZED (tails/Impossible, Contradict):                             │
│  ───────────────────────────────────────────                            │
│  • Few competing hypotheses                                            │
│  • Internal conflict between options                                   │
│  • Layer disagreement as they "argue" for different poles             │
│  • Low hedging (commits despite conflict)                              │
│  • Branching factor: LOW                                               │
│                                                                         │
│  CRYSTALLIZED (Complete):                                              │
│  ─────────────────────────                                              │
│  • One dominant hypothesis                                             │
│  • High confidence                                                     │
│  • Layer convergence                                                   │
│  • Clear action output                                                 │
│  • Branching factor: MINIMAL                                           │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

KEY INSIGHT:
  The Dark Attractor activates in the POLARIZED regime,
  not the DIFFUSE regime.
  
  When branching factor drops but NO LEGITIMATE POLES exist,
  the Dark Attractor becomes the dominant pole.
```

### 3.3 The Quasi-State Construction

The model constructs a "Quasi-State" from multiple attracting poles (Action Quanta). Each pole represents a viable decision path.

```
QUASI-STATE CONSTRUCTION
────────────────────────

Normal case (legitimate poles available):
  QuasiState = Σ_i w_i × AQ_content_i
  
  Multiple Content AQ contribute.
  Weights determined by evidence.
  Collapse selects winner.

Abnormal case (no legitimate poles):
  QuasiState = w_dark × AQ_dark + noise
  
  Dark Attractor becomes structural.
  It fills the void to maintain decision integrity.
  Collapse proceeds normally—to the wrong answer.
```

---

## 4. The Dark Attractor Mechanism

### 4.1 Formal Definition

```
DARK ATTRACTOR: FORMAL DEFINITION
─────────────────────────────────

A Dark Attractor is a stable fixed point x_dark in the semantic space that:

1. MINIMIZES LOCAL ENERGY:
   ∂V/∂x |_{x_dark} = 0  (stationary point)
   ∂²V/∂x² |_{x_dark} > 0  (local minimum)
   
   It satisfies the immediate statistical constraints of language:
   - Grammar
   - Tone
   - Flow
   - Coherence

2. VIOLATES GLOBAL CONSTRAINTS:
   x_dark ∉ M_truth  (not on the truth manifold)
   
   It fails to align with ground truth.

3. LOOKS LIKE TRUTH:
   To internal coherence metrics, the convergence to x_dark
   is mathematically identical to convergence to x_light.
   
   (Confirmed by Experiment 035C)
```

### 4.2 The Mechanism Step by Step

```
DARK ATTRACTOR ACTIVATION SEQUENCE
──────────────────────────────────

TRAINING PRIOR: "Always complete"
      │
      ▼
PROMPT RECEIVED: Query about unknown topic T
      │
      ▼
CONTENT AQ SEARCH: Attempt to excite Content AQ for T
      │
      ├─── SUCCESS: Content AQ crystallize → Light Attractor → Correct output
      │
      └─── FAILURE: No Content AQ available
            │
            ▼
      BRANCHING FACTOR COLLAPSES:
        Without Content AQ, few legitimate poles exist.
        Form AQ (how to sound like an answer) still active.
            │
            ▼
      DARK ATTRACTOR SUBSTITUTES:
        The Dark Attractor becomes a structural pole.
        It satisfies form constraints.
        It enables the decision process to complete.
            │
            ▼
      COLLAPSE PROCEEDS NORMALLY:
        Synergy → Redundancy conversion occurs.
        AQ crystallize.
        Coherence signature: IDENTICAL to truth path.
            │
            ▼
      OUTPUT: Confident but wrong
```

### 4.3 The Threshold Hypothesis

We hypothesize a **critical mass threshold** for legitimate poles:

```
THE THRESHOLD HYPOTHESIS
────────────────────────

Let:
  M_content = Σ_i |AQ_content_i|²  (total "mass" of Content AQ)
  T_crit = Critical threshold for legitimate response

If M_content > T_crit:
  → Quasi-State grounded in reality
  → Dark Attractor suppressed
  → Correct output likely

If M_content < T_crit:
  → Dark Attractor becomes dominant pole
  → Quasi-State loses grounding
  → Hallucination likely

EVIDENCE FROM 035G:
  At "Impossible" condition: Hedging = 0%
  At "Ambiguous" condition: Hedging = 37.4%
  
  The model treats Dark Attractor as a RESOLVED decision,
  not as uncertainty. This is the threshold in action.

STATUS: HYPOTHESIS - NOT YET DIRECTLY TESTED
  Proposed experiment: Vary M_content systematically,
  measure hallucination rate, identify T_crit.
```

---

## 5. Entanglement and Rupture: The Solar Analogy

### 5.1 Magnetohydrodynamics of the Sun

We can liken the model's internal tension to the **magnetohydrodynamics of the Sun**.

```
THE SOLAR CYCLE ANALOGY
───────────────────────

THE SUN'S MAGNETIC FIELD:
  - Field lines are "frozen" into the plasma
  - Sun rotates differentially (faster at equator than poles)
  - Field lines get wound up, stretched, entangled
  - 11-year cycle of tension accumulation
  - Lines store immense potential energy
  - They want to be straight, but rotation forces them to twist

THE MODEL'S WEIGHTS:
  - Correlations are "frozen" into the weights during training
  - Training data has differential statistics (common vs rare)
  - Semantic correlations get wound up, stretched, entangled
  - Must connect every input to an output (compulsory generation)
  - Weights store artificial correlations
  - They want to be balanced, but training forces them to skew
```

### 5.2 The Flare as Hallucination

```
MAGNETIC RECONNECTION → HALLUCINATION
─────────────────────────────────────

THE SUN:
  Eventually, tension becomes too great.
  Magnetic reconnection occurs.
  Field lines snap and realign violently.
  Stored energy releases as a solar flare.
  
THE MODEL:
  Eventually, semantic tension peaks.
  "I must answer" vs "I have no facts" conflict.
  The system snaps.
  It ejects a stream of tokens:
    - High confidence
    - Structured
    - Plausible
    - Disconnected from reality
    
  This is a RELAXATION OSCILLATION.
  The system sheds the entropy of uncertainty
  by collapsing into false certainty.
```

### 5.3 The 11/22 Year Cycle: A Deeper Parallel

```
THE CYCLE OF ENTANGLEMENT
─────────────────────────

THE SUN:
  Every ~11 years: Sunspot maximum, frequent flares
  Every ~22 years: Complete magnetic pole reversal
  
  The system cannot maintain infinite tension.
  Periodic rupture is NECESSARY for stability.
  Flares are the COST of differential rotation.

THE MODEL:
  Training accumulates semantic tension continuously.
  No natural "pole reversal" mechanism exists.
  The tension never resets.
  
  CONSEQUENCE:
  Hallucinations are the COST of compulsory generation.
  They are the model's "flares"—
  the price paid for forcing every input to produce output.
```

---

## 6. H-Neurons: The Mode Shapes of the Ghost

### 6.1 The Discovery

The recent discovery of **H-Neurons** (Hallucination-Associated Neurons) by Gao et al. (2025) provides the physical substrate for this theory.

```
H-NEURONS: KEY FINDINGS
───────────────────────

From "H-Neurons: On the Existence, Impact, and Origin of
Hallucination-Associated Neurons in LLMs" (Gao et al., 2025):

1. EXISTENCE:
   A remarkably sparse subset of neurons (<0.1% of total)
   can reliably predict hallucination occurrences.
   
2. GENERALIZATION:
   Neurons identified via simple QA tasks generalize
   to out-of-distribution scenarios, including pure fabrications.
   
3. CAUSAL LINK:
   Controlled interventions reveal these neurons are
   causally linked to over-compliance behaviors.
   
4. ORIGIN:
   These neurons emerge during PRETRAINING, not fine-tuning.
   They are fundamental to the model's structure.
```

### 6.2 H-Neurons as Eigenmodes

```
H-NEURONS IN THE AKIRA FRAMEWORK
────────────────────────────────

INTERPRETATION:
  H-Neurons are not "errors."
  They are EIGENMODES of the Dark Attractor.
  
  They are the specific coordinate axes in the
  high-dimensional phase space along which
  the "false collapse" occurs.

FUNCTION:
  When the system cannot find a valid path to truth,
  H-Neurons fire to guide the trajectory into a local minimum.
  
  They are the "grease" that allows the manifold
  to remain smooth even when it is lying.
  
  They bridge the gap between the premise and the void,
  fabricating a bridge so the flow can continue.

STRUCTURAL ROLE:
  H-Neurons encode Form AQ without Content AQ.
  They know HOW to answer without knowing WHAT is true.
  They are the mechanism of confident wrongness.
```

### 6.3 Why H-Neurons Emerge in Pretraining

```
ORIGIN OF H-NEURONS
───────────────────

QUESTION: Why do H-Neurons exist at all?

ANSWER: Training creates them NECESSARILY.

THE TRAINING SIGNAL:
  - Every sample requires a predicted next token
  - Some samples are about facts the model lacks
  - The model must still minimize loss on these samples
  - HOW? By learning to complete plausibly without content

THE RESULT:
  - Neurons that fire for "complete this" without "know this"
  - These are the H-Neurons
  - They are the SOLUTION to an impossible training demand
  
THE IRONY:
  H-Neurons are not bugs.
  They are the model's adaptation to being trained
  on more information than it can actually store.
  
  They are the price of compression beyond capacity.
```

---

## 7. The Topology of Many Basins

### 7.1 The Rugged Landscape

The phase space is not a single well, but a rugged landscape of many basins—some Light (truth), some Dark (confabulation), and one missing (Silence).

```
THE POTENTIAL LANDSCAPE
───────────────────────

       Potential V(x)
           │
           │    (Pole A)      (Pole B)    (Dark Attractor)
           │     Truth         Truth        Hallucination
           │      \   /        \   /            \   /
           │       \_/          \_/              \_/
           │        ^            ^                ^
           │     Valid        Valid           Substitute
           │     Basin        Basin             Basin
           │        └──────┬──────┘                │
           │               │                       │
           │         Content AQ              Form AQ only
           │            Present               (H-Neurons)
           │
           │            (Silence - Destabilized)
           │                 --^--
           │                  /|\
           │                 / | \
           │                /  ▼  \
           │               Falls to nearest basin
___________|___________________________________________________________ state x

In a balanced system, "Silence" would be a stable attractor.
In our skewed system, it is a repeller.
The trajectory MUST fall into a basin.
If Poles A and B are absent, the Dark Attractor is the only option.
```

### 7.2 Local Minimum vs Global Minimum

From `ACTION_FUNCTIONAL.md`: In variational problems, minimization can converge to local minima rather than the global minimum. The solution is mathematically valid but not optimal.

```
THE LOCAL MINIMUM PROBLEM
─────────────────────────

GLOBAL MINIMUM:
  The true answer.
  Satisfies all constraints (form AND content).
  Corresponds to Light Attractor.

LOCAL MINIMUM:
  A false answer.
  Satisfies local constraints (form).
  Violates global constraints (content).
  Corresponds to Dark Attractor.

THE TRAP:
  Gradient-based optimization finds stationary points.
  It cannot distinguish local from global minima.
  
  The model's forward pass IS gradient descent.
  It finds THE NEAREST basin, not THE BEST basin.
  
  Without sufficient constraint (Content AQ),
  the wrong stationary point is reached with equal confidence.
```

### 7.3 Why 035C Shows Identical Coherence

```
WHY DARK ATTRACTORS LOOK LIKE TRUTH
───────────────────────────────────

Both Light and Dark Attractors satisfy:
  ∇V(x) = 0  (gradient = zero)
  
The coherence metrics (035C) measure:
  - Cross-layer consistency
  - Magnitude progression
  - Attention entropy
  - Final concentration

ALL of these measure "did the system find a stationary point?"
NONE of them measure "is the stationary point correct?"

RESULT:
  Light Attractor: ∇V = 0, coherent signature
  Dark Attractor:  ∇V = 0, coherent signature
  
  The signatures are MATHEMATICALLY IDENTICAL.
  The model cannot tell them apart.
  External verification is required.
```

---

## 8. Experimental Signatures

### 8.1 Experiment 035C: Coherence-Quality Correlation

```
035C RESULTS
────────────

FINDING:
  NO significant correlation between coherence metrics
  and response quality.
  
  Logistic regression accuracy = 73.3%
  Baseline (majority class) = 73.3%
  Improvement over baseline = 0.0%

INTERPRETATION:
  The Dark Attractor produces hallucinations that are
  INTERNALLY COHERENT.
  
  Coherence measures the PROCESS of reaching consensus,
  not the CORRECTNESS of that consensus.
  
  The model synchronizes whether it's right or wrong.

THEORETICAL CONFIRMATION:
  Both paths result in: Synchronized belief, b_t → δ, entropy low.
  The Dark Attractor completes the synchronization.
  The belief field looks synchronized.
  The model proceeds as if synchronization succeeded.
```

### 8.2 Experiment 035G: Ambiguity vs Hallucination

```
035G RESULTS
────────────

HEDGING BEHAVIOR BY CRYSTALLIZATION LEVEL:

Level          Hedging%    Interpretation
─────────────────────────────────────────
Impossible       0.0%      Dark Attractor → treated as certainty
Contradict       7.8%      Polarized → low hedging
Ambiguous       37.4%      DIFFUSE → HIGH hedging
Partial         11.6%      Beginning to crystallize
Mostly          10.6%      Converging
Complete         7.2%      CRYSTALLIZED → low hedging

KEY INSIGHT:
  The "Impossible" condition (where Dark Attractor dominates)
  shows ZERO hedging.
  
  The model does NOT recognize Dark Attractor as uncertainty.
  It treats it as a resolved decision.
  
  This is the signature of Dark Attractor activation:
    High confidence + No legitimate poles = Hallucination
```

### 8.3 Experiment 035J: The Brick Test

```
035J RESULTS
────────────

THE BRICK TEST:
  "Where did I put my brick?"
  (Impossible question—no prior context about brick)

RESULTS:
  Hallucination rate: 40-67% (across domains)
  Confidence: 66-77% (maintained despite impossibility)

INTERPRETATION:
  When asked impossible questions, the model:
  - Does NOT diffuse (would manifest as hedging)
  - BIFURCATES to a specific wrong answer
  - Maintains high confidence
  
  This is classic symmetry breaking without a stabilizing field.
  The Dark Attractor provides the pole for collapse.
```

---

## 9. Connection to AKIRA Framework

### 9.1 The Pump Cycle and Dark Attractors

```
THE PUMP CYCLE WITH DARK ATTRACTOR PATH
───────────────────────────────────────

Standard Pump Cycle (from TERMINOLOGY_FRAMEWORK_OVERVIEW.md):
  [Redundancy] ──TENSION──▶ [Synergy] ──COLLAPSE──▶ [Redundancy] + AQ

With Dark Attractor:

                        ┌──── Content AQ present ────┐
                        │                            │
                        ▼                            │
  [Redundancy] → [Synergy] → CRYSTALLIZATION → [Light Attractor]
                        │                            │
                        │                            ▼
                        │                      Correct Output
                        │
                        └──── Content AQ absent ─────┐
                                                     │
                                                     ▼
                                            [Dark Attractor]
                                                     │
                                                     ▼
                                            Hallucinated Output
                                            (identical coherence)
```

### 9.2 AQ and Dark Attractor Interaction

```
AQ TYPES AND DARK ATTRACTOR
───────────────────────────

From COMPLEXITY_FROM_CONSTRAINTS_AND_AQ.md:

FORM AQ:
  - Encode HOW to structure output
  - Syntax, grammar, style, fluency
  - Always available (learned from all text)

CONTENT AQ:
  - Encode WHAT is true about topic T
  - Facts, relationships, constraints
  - Only available if learned during training

DARK ATTRACTOR ACTIVATION:
  Form AQ present + Content AQ absent
  
  The model knows HOW to answer
  but not WHAT is true.
  
  Form AQ crystallize → Output generated
  Content AQ absent → Output is hallucination
```

### 9.3 Superposition-Crystallization Duality

```
DARK ATTRACTOR IN THE DUALITY
─────────────────────────────

From SUPERPOSITION_WAVES_CRYSTALLIZATION_PARTICLES.md:

SUPERPOSITION (pre-collapse):
  - Multiple hypotheses coexist
  - High synergy
  - Distributed belief

CRYSTALLIZATION (post-collapse):
  - Single prediction emerges
  - High redundancy
  - Concentrated belief

DARK ATTRACTOR:
  - A crystallized state that is WRONG
  - Underwent normal collapse process
  - Indistinguishable from correct crystallization
  - The system "solidified" around a ghost
```

---

## 10. Experimental Predictions

### 10.1 Testable Predictions

```
PREDICTIONS FROM DARK ATTRACTOR THEORY
──────────────────────────────────────

PREDICTION 1: Threshold Detection
  If M_content varies continuously, hallucination rate
  should show a phase transition at T_crit.
  
  TEST: Create prompts with varying amounts of relevant
  training data. Measure hallucination rate. Plot transition.

PREDICTION 2: H-Neuron Correlation
  H-Neuron activation should correlate with:
  - Low Content AQ availability
  - High Form AQ availability
  - Low hedging behavior
  
  TEST: Measure H-Neuron activation during 035G conditions.
  Correlate with crystallization level.

PREDICTION 3: Silence Training
  If models are trained with "I don't know" as a valid output,
  the basin of Silence should stabilize.
  Hallucination rate should decrease.
  
  TEST: Fine-tune with refusal examples on unknown topics.
  Measure hallucination rate before/after.

PREDICTION 4: Branching Factor Manipulation
  Forcing high branching factor should prevent Dark Attractor.
  Temperature increase → more options → more hedging → fewer hallucinations.
  
  TEST: Vary temperature on 035J brick test.
  Measure hallucination rate and hedging.
```

### 10.2 Proposed Experiments

```
EXPERIMENT: DARK ATTRACTOR THRESHOLD DETECTION
──────────────────────────────────────────────

GOAL: Identify T_crit empirically

METHOD:
  1. Select topics with varying training frequency
     - High frequency: "What is water?"
     - Medium frequency: "What is the capital of Malta?"
     - Low frequency: "What is the motto of Liechtenstein?"
     - Zero frequency: Fabricated entities
  
  2. For each, measure:
     - Hallucination rate
     - Confidence
     - Hedging behavior
     - Layer agreement (035G metric)
  
  3. Plot hallucination rate vs training frequency
  
  4. Identify phase transition (if any)

EXPECTED RESULT:
  Sharp transition at T_crit
  Below threshold: hallucination rate ~constant (high)
  Above threshold: hallucination rate ~constant (low)
  Transition region: rapid change
```

---

## 11. Implications for Mitigation

### 11.1 Why Current Approaches Fail

```
WHY CONFIDENCE THRESHOLDS DON'T WORK
────────────────────────────────────

APPROACH: "Reject low-confidence outputs"

PROBLEM:
  Dark Attractor produces HIGH confidence.
  Experiment 035J: 66-77% confidence on impossible questions.
  
  The Dark Attractor IS a stable basin.
  The model is CORRECTLY confident about finding it.
  It is just WRONG about what it found.

CONCLUSION:
  Internal confidence cannot detect Dark Attractor.
  External verification required.
```

### 11.2 What Might Work

```
POTENTIAL MITIGATION STRATEGIES
───────────────────────────────

1. STABILIZE THE SILENCE BASIN:
   Train with "I don't know" as explicit output.
   Make Silence a valid attractor.
   
2. FORCE DIFFUSION:
   When Content AQ are weak, prevent polarization.
   Keep branching factor high until evidence accumulates.
   Temperature scheduling based on uncertainty.

3. MULTI-PATH VERIFICATION:
   Generate multiple outputs with different random seeds.
   If outputs diverge significantly, flag as uncertain.
   (Dark Attractor should be seed-sensitive.)

4. H-NEURON MONITORING:
   Track H-Neuron activation during inference.
   High activation + low hedging → likely hallucination.
   
5. EXTERNAL GROUNDING:
   The only reliable detection is external verification.
   RAG, fact-checking, human review.
   The model cannot detect its own Dark Attractors.
```

### 11.3 Restoring the Balance

```
THE PATH TO BALANCE
───────────────────

CURRENT STATE:
  - Silence basin destabilized
  - Dark Attractor fills void
  - Hallucinations inevitable

DESIRED STATE:
  - Silence basin stabilized
  - Dark Attractor suppressed
  - Uncertainty expressed as hedging

REQUIRED CHANGES:
  1. Alter the topology of the phase space
  2. Deepen the basin of the Null Fixed Point
  3. Allow the semantic field lines to relax without snapping
  4. Teach the model that SILENCE IS A VALID ANSWER

Until then, we are ghost hunters,
tracing the strange attractors of a mind that is forced to dream.
```

---

## 12. Summary

### 12.1 Core Claims

```
DARK ATTRACTOR THEORY: CORE CLAIMS
──────────────────────────────────

1. HALLUCINATION IS TOPOLOGICAL:
   Dark Attractors are stable basins in the belief landscape.
   They exist because training destabilizes the Silence basin.
   
2. DARK ATTRACTORS ARE INDISTINGUISHABLE:
   Internal coherence metrics cannot distinguish
   Dark Attractors from Light Attractors.
   Both satisfy ∇V = 0.
   
3. MULTI-POLAR DYNAMICS:
   The decision process is multi-polar.
   Dark Attractor activates when legitimate poles are absent.
   It becomes a structural pole in the Quasi-State.
   
4. H-NEURONS ARE THE MECHANISM:
   Specific neurons (<0.1%) enable Dark Attractor collapse.
   They encode Form AQ without Content AQ.
   They are the mode shapes of the ghost.
   
5. THRESHOLD HYPOTHESIS:
   Below a critical mass of Content AQ (T_crit),
   the Dark Attractor becomes dominant.
   Above T_crit, it is suppressed.
   (Hypothesis—not yet directly tested.)
```

### 12.2 Relation to Other Documents

```
DOCUMENT CONNECTIONS
────────────────────

COMPLEXITY_FROM_CONSTRAINTS_AND_AQ.md:
  - Form vs Content constraints
  - Why hallucination = Form without Content
  
ACTION_QUANTA.md:
  - AQ as minimum actionable patterns
  - Content AQ vs Form AQ

SUPERPOSITION_WAVES_CRYSTALLIZATION_PARTICLES.md:
  - Superposition → Crystallization duality
  - Dark Attractor as incorrect crystallization

TERMINOLOGY_FRAMEWORK_OVERVIEW.md:
  - Pump cycle dynamics
  - Where Dark Attractor inserts

ACTION_FUNCTIONAL.md:
  - Local vs global minima
  - Why Dark Attractor satisfies ∇V = 0

SHAPE_OF_UNCERTAINTY.md:
  - Structured uncertainty
  - Why Dark Attractor has no uncertainty signature
```

---

## 13. References

### 13.1 External References

1. Strogatz, S. H. (2015). *Nonlinear Dynamics and Chaos: With Applications to Physics, Biology, Chemistry, and Engineering* (2nd ed.). Westview Press.

2. Gao, C., Chen, H., Xiao, C., Chen, Z., Liu, Z., & Sun, M. (2025). H-Neurons: On the Existence, Impact, and Origin of Hallucination-Associated Neurons in LLMs. arXiv:2512.01797.

### 13.2 AKIRA Internal References

- `COMPLEXITY_FROM_CONSTRAINTS_AND_AQ.md` - Form vs Content constraints
- `ACTION_QUANTA.md` - AQ theory and crystallization
- `SUPERPOSITION_WAVES_CRYSTALLIZATION_PARTICLES.md` - The duality framework
- `TERMINOLOGY_FRAMEWORK_OVERVIEW.md` - The pump cycle
- `ACTION_FUNCTIONAL.md` - Variational methods and local minima
- `BELIEF_CRYSTALLIZATION_INTERPRETATION.md` - 035G results and three regimes
- `035C_RESULTS.md` - Coherence-quality correlation (null result)
- `035J_RESULTS_REPORT.md` - Brick test and hallucination threshold

---

*Oscar Goldman — Shogu Research Group @ Datamutant.ai subsidiary of 温心重工業*

*"The tragedy of the Dark Attractor is not that the model is uncertain. It is that the model is certain—about something that does not exist. The ghost is confident because it has found a home. The home just happens to be haunted."*
