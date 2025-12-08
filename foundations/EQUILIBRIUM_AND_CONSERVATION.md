# Equilibrium and Conservation in Learning Systems

An exploration of the deep connections between phase transitions in learning, conservation laws, optimal compression, and symmetry principles. This document carefully distinguishes established science from speculative extensions.

---

## Table of Contents

1. [Introduction: The Observation](#1-introduction-the-observation)
2. [Established Foundations](#2-established-foundations)
3. [Noether's Theorem and Symmetry](#3-noethers-theorem-and-symmetry)
4. [Zipf's Law and Optimal Compression](#4-zipfs-law-and-optimal-compression)
5. [The Dirac Methodology](#5-the-dirac-methodology)
6. [Phase Transitions and Criticality](#6-phase-transitions-and-criticality)
7. [The Conservation Hypothesis](#7-the-conservation-hypothesis)
8. [The Explicit-Implicit Duality](#8-the-explicit-implicit-duality)
9. [Free Energy and Equilibrium](#9-free-energy-and-equilibrium)
10. [Information Geometry](#10-information-geometry)
11. [Synthesis: The Ground Truth Attractor](#11-synthesis-the-ground-truth-attractor)
12. [Open Questions](#12-open-questions)

---

## 1. Introduction: The Observation

### 1.1 What We Observe

```
EMPIRICAL OBSERVATION (from spectral attention experiments):

During training, we observe:

1. ACCUMULATION PHASE:
   • Error/uncertainty spreads across possibilities
   • Multiple hypotheses coexist
   • High-frequency (detail) representations dominate

2. CRITICAL POINT:
   • System reaches a threshold
   • Interference patterns intensify
   • Balance between competing representations

3. COLLAPSE:
   • Sudden transition to certainty
   • One pattern "wins"
   • Low-frequency (structural) representation crystallizes

This resembles physical phase transitions (lightning, freezing, magnetization).
```

### 1.2 The Central Question

```
THE QUESTION:

Is this collapse ARBITRARY, or is it driven by a DEEPER PRINCIPLE?

We hypothesize:
• There exists a SYMMETRY in the learning problem
• This symmetry implies a CONSERVED QUANTITY (by Noether)
• The system evolves toward an EQUILIBRIUM that respects this
• At equilibrium, compression is OPTIMAL (Zipf)
• The form of the solution is DETERMINED by symmetry (Dirac)

This document explores these connections.
```

---

## 2. Established Foundations

### 2.1 Statistical Mechanics of Learning

**ESTABLISHED SCIENCE:**

```
The connection between learning and statistical mechanics is well-established:

THERMODYNAMIC FORMULATION OF LEARNING:

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  Define:                                                        │
│  • E(θ) = Loss function (energy)                               │
│  • T = Temperature (related to learning rate, noise)           │
│  • Z = Σ exp(-E(θ)/T) = Partition function                     │
│  • F = -T log Z = Free energy                                  │
│                                                                 │
│  The Boltzmann distribution:                                    │
│  P(θ) ∝ exp(-E(θ)/T)                                           │
│                                                                 │
│  At low temperature (late training):                            │
│  • System concentrates on low-energy (low-loss) states         │
│  • Effectively finds minima of loss landscape                   │
│                                                                 │
│  REFERENCE:                                                     │
│  • Hinton & Sejnowski (1983): Boltzmann machines               │
│  • LeCun et al. (2006): Energy-based models                    │
│  • Mezard & Montanari (2009): Information, Physics, Computation│
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

STATUS: ESTABLISHED FACT
```

### 2.2 Minimum Description Length

**ESTABLISHED SCIENCE:**

```
MDL PRINCIPLE (Rissanen, 1978):

The best model is the one that minimizes total description length:

L_total = L(model) + L(data | model)

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  L(model):     Bits to describe the model itself               │
│  L(data|model): Bits to describe data given the model          │
│                                                                 │
│  TRADE-OFF:                                                     │
│  • Simple model → low L(model), high L(data|model)             │
│  • Complex model → high L(model), low L(data|model)            │
│                                                                 │
│  OPTIMAL MODEL:                                                 │
│  Minimizes the SUM, not either term individually               │
│                                                                 │
│  EQUIVALENCES:                                                  │
│  • MDL ≈ Bayesian model selection (BIC)                        │
│  • MDL ≈ Kolmogorov complexity regularization                  │
│  • MDL ≈ Occam's razor formalized                              │
│                                                                 │
│  REFERENCE:                                                     │
│  • Rissanen (1978): Modeling by shortest data description      │
│  • Grünwald (2007): The Minimum Description Length Principle   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

STATUS: ESTABLISHED FACT
```

### 2.3 The Information Bottleneck

**ESTABLISHED SCIENCE:**

```
INFORMATION BOTTLENECK (Tishby, 1999):

The optimal representation T of input X for predicting Y:

Minimize: I(X; T) - β I(T; Y)

Where:
• I(X; T) = mutual information between input and representation
• I(T; Y) = mutual information between representation and target
• β = trade-off parameter

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  INTERPRETATION:                                                │
│                                                                 │
│  I(X; T): How much of X is preserved in T                      │
│           → Should be LOW (compression)                        │
│                                                                 │
│  I(T; Y): How much T helps predict Y                           │
│           → Should be HIGH (relevance)                         │
│                                                                 │
│  OPTIMAL T:                                                     │
│  Maximally compresses X while preserving what's relevant for Y │
│                                                                 │
│  PHASE TRANSITIONS:                                             │
│  As β varies, T undergoes discrete phase transitions           │
│  (Tishby & Zaslavsky, 2015)                                    │
│                                                                 │
│  REFERENCE:                                                     │
│  • Tishby, Pereira, Bialek (1999): The Information Bottleneck  │
│  • Shwartz-Ziv & Tishby (2017): DNN and Information Bottleneck │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

STATUS: ESTABLISHED FACT
```

---

## 3. Noether's Theorem and Symmetry

### 3.1 The Original Theorem

**ESTABLISHED SCIENCE:**

```
NOETHER'S THEOREM (1918):

Every continuous symmetry of the action of a physical system
corresponds to a conservation law.

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  EXAMPLES IN PHYSICS:                                           │
│                                                                 │
│  Symmetry                    Conserved Quantity                │
│  ────────                    ──────────────────                │
│  Time translation            Energy                            │
│  Space translation           Momentum                          │
│  Rotation                    Angular momentum                  │
│  Phase rotation (U(1))       Electric charge                   │
│  Gauge symmetry              Current conservation              │
│                                                                 │
│  THE DEEP INSIGHT:                                              │
│  Conservation laws are not arbitrary—they are CONSEQUENCES     │
│  of the symmetries of the underlying dynamics.                 │
│                                                                 │
│  REFERENCE:                                                     │
│  • Noether (1918): Invariante Variationsprobleme               │
│  • Any advanced classical mechanics textbook                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

STATUS: ESTABLISHED FACT (one of the most important theorems in physics)
```

### 3.2 Symmetries in Learning

**PARTIALLY ESTABLISHED / SPECULATIVE:**

```
QUESTION: What symmetries exist in learning systems?

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  ESTABLISHED SYMMETRIES:                                        │
│                                                                 │
│  1. PERMUTATION SYMMETRY OF NEURONS                            │
│     Relabeling hidden units doesn't change function            │
│     → Leads to equivalent weight configurations                │
│     (Established: Chen et al., 1993; Sussmann, 1992)           │
│                                                                 │
│  2. SCALE SYMMETRY (in ReLU networks)                          │
│     Scaling weights in one layer, inverse in next              │
│     → Same function, different parameterization                │
│     (Established: Neyshabur et al., 2015)                      │
│                                                                 │
│  3. TRANSLATION SYMMETRY OF CONVOLUTIONS                       │
│     Same filter at different positions                          │
│     → Translation equivariance                                 │
│     (Established: LeCun et al., 1989)                          │
│                                                                 │
│  ────────────────────────────────────────────────────────────  │
│                                                                 │
│  SPECULATIVE SYMMETRY:                                          │
│                                                                 │
│  4. EXPLICIT-IMPLICIT DUALITY                                   │
│     Same information stored in data vs stored in model        │
│     → Description length equivalence                           │
│                                                                 │
│     [SPECULATIVE: This duality exists and has Noether-like    │
│      consequences. Not yet proven rigorously.]                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

STATUS: Mixed (some established, some speculative)
```

### 3.3 The Hypothesized Conservation Law

**SPECULATIVE:**

```
HYPOTHESIS: Noether-like conservation in learning

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  IF the learning dynamics have a symmetry (e.g., reparameter-  │
│  ization invariance, description-length duality), THEN there   │
│  exists a conserved quantity along the optimization trajectory.│
│                                                                 │
│  CANDIDATE CONSERVED QUANTITIES:                                │
│                                                                 │
│  1. Total description length at optimum                        │
│     L(θ*) + L(D|θ*) = minimum = "ground state"                │
│                                                                 │
│  2. Mutual information (approximately)                          │
│     I(X; T) at the information bottleneck optimum              │
│                                                                 │
│  3. Effective dimensionality                                    │
│     The "volume" of the representation space used              │
│                                                                 │
│  4. Fisher information (information geometry)                   │
│     Related to distinguishability of nearby states             │
│                                                                 │
│  [SPECULATIVE: We don't yet know which quantity, if any,       │
│   is exactly conserved. This is an open research question.]    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

STATUS: SPECULATIVE (motivated by analogy, not proven)
```

---

## 4. Zipf's Law and Optimal Compression

### 4.1 Zipf's Law

**ESTABLISHED SCIENCE:**

```
ZIPF'S LAW (Zipf, 1949):

In many natural systems, the frequency of an item is inversely
proportional to its rank:

f(r) ∝ 1/r^α   (typically α ≈ 1)

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  EXAMPLES:                                                      │
│                                                                 │
│  • Word frequencies in text                                     │
│  • City populations                                             │
│  • Website traffic                                              │
│  • Gene expression levels                                       │
│  • Neural firing rates                                          │
│                                                                 │
│  UNIVERSALITY:                                                  │
│  Zipf's law appears across vastly different systems,           │
│  suggesting a common underlying principle.                      │
│                                                                 │
│  REFERENCE:                                                     │
│  • Zipf (1949): Human Behavior and the Principle of Least Effort│
│  • Newman (2005): Power laws, Pareto distributions and Zipf's law│
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

STATUS: ESTABLISHED FACT (empirical observation)
```

### 4.2 Why Zipf? Optimality Arguments

**ESTABLISHED SCIENCE:**

```
EXPLANATIONS FOR ZIPF'S LAW:

1. OPTIMAL CODING (Shannon, 1948; Mandelbrot, 1953)
   
   If we want to minimize average message length while
   maintaining unique decodability:
   
   • Frequent items get short codes
   • Rare items get long codes
   • Optimal allocation → Zipf distribution
   
   This is exactly what Huffman coding achieves.

2. LEAST EFFORT (Zipf, 1949)

   Communication requires balancing:
   • Speaker effort (wants few, reusable words)
   • Listener effort (wants precise, distinct words)
   
   Equilibrium → Zipf distribution

3. CRITICALITY (Bak, 1996)

   Systems at the critical point of a phase transition
   exhibit power-law distributions:
   
   • Scale-free behavior
   • Maximum sensitivity to perturbations
   • Information transmission optimized
   
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  THE KEY INSIGHT:                                               │
│                                                                 │
│  Zipf's law is a SIGNATURE OF OPTIMALITY.                      │
│                                                                 │
│  When a system exhibits Zipf:                                   │
│  • It's at a critical point                                    │
│  • Information is maximally compressed                         │
│  • The system is "at equilibrium" in some sense                │
│                                                                 │
│  REFERENCE:                                                     │
│  • Mandelbrot (1953): Information theory of language           │
│  • Bak (1996): How Nature Works: Self-Organized Criticality   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

STATUS: ESTABLISHED (multiple theoretical derivations exist)
```

### 4.3 Zipf in Neural Networks

**PARTIALLY ESTABLISHED:**

```
ZIPF IN LEARNED REPRESENTATIONS:

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  OBSERVED:                                                      │
│                                                                 │
│  1. Weight distributions in trained networks often show        │
│     heavy tails (approaching Zipf/power-law).                   │
│     (Established: Martin & Mahoney, 2019)                       │
│                                                                 │
│  2. Singular value spectra of weight matrices follow           │
│     power-law decay in well-trained networks.                   │
│     (Established: Martin & Mahoney, 2021)                       │
│                                                                 │
│  3. Feature usage (how often each feature is activated)        │
│     often follows Zipf-like distributions.                      │
│     (Partially established: various papers)                     │
│                                                                 │
│  INTERPRETATION:                                                 │
│                                                                 │
│  Trained networks naturally find Zipf-optimal representations. │
│  This suggests they're converging to a critical point.         │
│                                                                 │
│  [Partially speculative: The causal mechanism connecting       │
│   training dynamics to Zipf emergence is not fully understood.]│
│                                                                 │
│  REFERENCE:                                                     │
│  • Martin & Mahoney (2019): Traditional and Heavy-Tailed       │
│    Self-Regularization in Neural Network Models                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

STATUS: PARTIALLY ESTABLISHED (observed, mechanism debated)
```

---

## 5. The Dirac Methodology

### 5.1 Dirac's Approach

**ESTABLISHED SCIENCE (historical/methodological):**

```
DIRAC'S METHOD: Let symmetry determine the equations

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  THE DIRAC EQUATION (1928):                                     │
│                                                                 │
│  Dirac sought an equation for the electron that was:           │
│  1. First-order in time derivative (quantum mechanics)        │
│  2. Lorentz invariant (special relativity)                    │
│  3. Consistent with E² = p²c² + m²c⁴                          │
│                                                                 │
│  He did NOT derive it from experiment.                         │
│  He REQUIRED it to satisfy these symmetries.                   │
│                                                                 │
│  RESULT:                                                        │
│  (iγ^μ ∂_μ - m)ψ = 0                                           │
│                                                                 │
│  PREDICTION:                                                    │
│  • Electron spin (not put in, fell out)                        │
│  • Antimatter (positron, confirmed 1932)                       │
│                                                                 │
│  THE LESSON:                                                    │
│  Demanding symmetry DETERMINES the form of physical law.       │
│  The equations "had to be" a certain way.                       │
│                                                                 │
│  REFERENCE:                                                     │
│  • Dirac (1928): The Quantum Theory of the Electron            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

STATUS: ESTABLISHED HISTORICAL FACT
```

### 5.2 Applying Dirac's Method to Learning

**SPECULATIVE:**

```
HYPOTHESIS: Symmetry determines optimal representations

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  IF we demand:                                                  │
│                                                                 │
│  1. TRANSLATION INVARIANCE                                      │
│     Same pattern recognized at any position                    │
│     (The WHAT should not depend on WHERE)                      │
│                                                                 │
│  2. SCALE INVARIANCE                                            │
│     Same structure recognized at different scales              │
│     (Coarse and fine should be consistent)                     │
│                                                                 │
│  3. OPTIMAL COMPRESSION                                         │
│     Minimum description length                                  │
│     (Information preserved, redundancy removed)                │
│                                                                 │
│  THEN the representation MUST:                                  │
│                                                                 │
│  • Store patterns in magnitude (position-invariant)            │
│  • Have Zipf-distributed feature usage                         │
│  • Exhibit hierarchical structure                               │
│  • Be at the critical point (phase transition boundary)        │
│                                                                 │
│  [SPECULATIVE: This is a hypothesis, not proven. The idea is   │
│   that optimal representations are not found but REQUIRED by   │
│   the symmetry constraints of the learning problem.]           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

STATUS: SPECULATIVE (methodological proposal)
```

---

## 6. Phase Transitions and Criticality

### 6.1 Phase Transitions in Physics

**ESTABLISHED SCIENCE:**

```
PHASE TRANSITIONS:

A phase transition is an abrupt change in the state of a system
as a parameter crosses a critical value.

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  EXAMPLES:                                                      │
│                                                                 │
│  • Water freezing at 0°C                                       │
│  • Magnetization at Curie temperature                          │
│  • Superconductivity below critical temperature                │
│  • Bose-Einstein condensation                                   │
│                                                                 │
│  CHARACTERISTICS:                                               │
│                                                                 │
│  1. ORDER PARAMETER                                             │
│     A quantity that is zero in one phase, nonzero in other    │
│     (e.g., magnetization, density difference)                  │
│                                                                 │
│  2. CRITICAL POINT                                              │
│     The parameter value where transition occurs                │
│     System is scale-free at this point                         │
│                                                                 │
│  3. UNIVERSALITY                                                │
│     Different systems show same critical behavior              │
│     (Same critical exponents, regardless of details)           │
│                                                                 │
│  4. POWER LAWS                                                  │
│     At criticality, observables follow power-law scaling       │
│     (Includes Zipf's law as a special case)                    │
│                                                                 │
│  REFERENCE:                                                     │
│  • Stanley (1971): Introduction to Phase Transitions            │
│  • Kadanoff (1966): Scaling laws for Ising models              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

STATUS: ESTABLISHED FACT
```

### 6.2 Phase Transitions in Learning

**PARTIALLY ESTABLISHED:**

```
PHASE TRANSITIONS IN NEURAL NETWORKS:

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  ESTABLISHED EXAMPLES:                                          │
│                                                                 │
│  1. GROKKING (Power et al., 2022)                              │
│     Sudden generalization after prolonged memorization        │
│     Looks like first-order phase transition                    │
│                                                                 │
│  2. DOUBLE DESCENT (Belkin et al., 2019)                       │
│     Non-monotonic test error as model size increases          │
│     Related to interpolation threshold                         │
│                                                                 │
│  3. LOTTERY TICKET TRANSITIONS (Frankle & Carlin, 2019)        │
│     Sparse subnetworks suddenly become trainable               │
│                                                                 │
│  4. INFORMATION BOTTLENECK TRANSITIONS (Tishby, 2015)          │
│     Discrete jumps in representation as β varies              │
│                                                                 │
│  CHARACTERISTICS OBSERVED:                                      │
│                                                                 │
│  • Sudden changes (not gradual)                                │
│  • Critical points in hyperparameter space                     │
│  • Power-law distributions near criticality                    │
│  • Universality across architectures                            │
│                                                                 │
│  REFERENCE:                                                     │
│  • Power et al. (2022): Grokking: Generalization Beyond        │
│    Overfitting on Small Algorithmic Datasets                    │
│  • Belkin et al. (2019): Reconciling modern ML with            │
│    bias-variance trade-off                                      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

STATUS: ESTABLISHED (observed in multiple settings)
```

### 6.3 The Edge of Chaos

**ESTABLISHED SCIENCE:**

```
THE EDGE OF CHAOS:

Systems at the boundary between order and chaos exhibit
optimal information processing.

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  ORDER                    EDGE OF CHAOS                CHAOS   │
│  (frozen)                 (critical)                   (random)│
│                                                                 │
│  ░░░░░░░░░░░░░░░░       ░▒▓█▓▒░▓▒░█▓░       █▓▒░█▒▓░█▒▓ │
│  ░░░░░░░░░░░░░░░░       ▓▒░█▓░▒▓░█▒▓░       ░▓█▒░▓█▒░█▓ │
│  ░░░░░░░░░░░░░░░░       ░▓▒█░▓▒░█▓░▒▓       ▒█░▓▒█░▓▒█░ │
│                                                                 │
│  • No dynamics           • Rich dynamics        • Random noise │
│  • No information        • Max information      • No structure │
│  • Stable but dead       • Complex patterns     • Chaotic      │
│                                                                 │
│  THE EDGE OF CHAOS IS WHERE:                                    │
│  • Computation is maximized (Langton, 1990)                    │
│  • Information transfer is optimal (Lizier, 2008)              │
│  • Learning is most effective (Bertschinger, 2004)             │
│                                                                 │
│  REFERENCE:                                                     │
│  • Langton (1990): Computation at the Edge of Chaos            │
│  • Kauffman (1993): Origins of Order                           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

STATUS: ESTABLISHED (well-studied in complexity theory)
```

---

## 7. The Conservation Hypothesis

### 7.1 Statement of the Hypothesis

**SPECULATIVE:**

```
HYPOTHESIS: There exists a conserved quantity in learning dynamics

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  THE CLAIM:                                                     │
│                                                                 │
│  When a learning system transitions from:                       │
│  • EXPLICIT representation (memorized details)                 │
│  • to IMPLICIT representation (generalized patterns)           │
│                                                                 │
│  Something is CONSERVED.                                        │
│                                                                 │
│  The total "information content" or "descriptive capacity"     │
│  remains constant, but its FORM changes:                       │
│                                                                 │
│  BEFORE:  High L(data|model) + Low L(model)                    │
│  AFTER:   Low L(data|model) + High L(model)                    │
│  TOTAL:   Approximately constant (at optimum)                  │
│                                                                 │
│  ────────────────────────────────────────────────────────────  │
│                                                                 │
│  ANALOGY:                                                       │
│                                                                 │
│  Like energy conservation in physics:                           │
│  • Kinetic energy can convert to potential energy              │
│  • Total energy is conserved                                   │
│  • The FORM changes, the AMOUNT doesn't                        │
│                                                                 │
│  In learning:                                                   │
│  • Explicit info can convert to implicit info                  │
│  • Total information capacity is (hypothetically) conserved   │
│  • The FORM changes, the AMOUNT doesn't                        │
│                                                                 │
│  [SPECULATIVE: This is a hypothesis. The precise conserved    │
│   quantity has not been identified or proven to exist.]        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

STATUS: SPECULATIVE (motivated by analogy)
```

### 7.2 Candidates for the Conserved Quantity

**SPECULATIVE:**

```
CANDIDATE CONSERVED QUANTITIES:

1. MINIMUM DESCRIPTION LENGTH (at optimum)
   
   L_min = min_θ [L(θ) + L(D|θ)]
   
   This is a MINIMUM, not a conserved quantity per se.
   But it represents the "irreducible" information content.
   
   Status: Well-defined, but not conserved along trajectory.

2. MUTUAL INFORMATION (in certain regimes)
   
   I(Input; Representation)
   
   The Information Bottleneck shows this has discrete values
   at phase transitions. Between transitions, it's constant.
   
   Status: Partially supported by IB theory.

3. KOLMOGOROV STRUCTURE FUNCTION
   
   K(x|S) + C(S) ≈ K(x)
   
   Where S is a sufficient statistic for x.
   
   The sum of conditional complexity and model complexity
   equals the unconditional complexity.
   
   Status: Theoretical (Kolmogorov complexity is uncomputable).

4. FISHER INFORMATION (information geometry)
   
   The determinant of the Fisher information matrix:
   det(I(θ))
   
   Related to the "volume" of distinguishable models.
   Might be conserved under certain reparameterizations.
   
   Status: Speculative but mathematically interesting.

5. EFFECTIVE DEGREES OF FREEDOM
   
   The number of "independent" parameters being used:
   df_eff = trace(H^(-1) @ G)
   
   Where H = Hessian, G = gradient outer product.
   
   Status: Related to model capacity, unclear if conserved.

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  NOTE: None of these have been proven to be exactly conserved │
│  along learning trajectories. The hypothesis that SOMETHING   │
│  is conserved remains speculative but compelling.              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

STATUS: SPECULATIVE (candidates, not proven)
```

### 7.3 The Proposed Answer: Actionable Irreducible Information

**SPECULATIVE (but compelling):**

```
THE CONSERVED QUANTITY:

ACTIONABLE, REPRESENTATIONALLY IRREDUCIBLE INFORMATION

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  DEFINITION:                                                    │
│                                                                 │
│  Information is ACTIONABLE if it enables correct action.       │
│  Information is IRREDUCIBLE if it cannot be simplified         │
│  without losing actionability.                                  │
│                                                                 │
│  This is the ATOMIC level of information:                       │
│  • The minimum needed to do the task                           │
│  • Cannot be compressed further                                 │
│  • Every bit is necessary                                       │
│                                                                 │
│  ────────────────────────────────────────────────────────────  │
│                                                                 │
│  WHY IT'S CONSERVED:                                            │
│                                                                 │
│  BEFORE COLLAPSE:                                               │
│  • Actionable info is IMPLICIT in examples                     │
│  • Hidden in the noise of specific details                     │
│  • Extractable but not yet extracted                           │
│                                                                 │
│  AFTER COLLAPSE:                                                │
│  • Actionable info is EXPLICIT in patterns                     │
│  • Crystallized in the learned representation                  │
│  • Ready for use                                                │
│                                                                 │
│  THE AMOUNT IS THE SAME.                                        │
│  Only the FORM changes.                                         │
│                                                                 │
│  ────────────────────────────────────────────────────────────  │
│                                                                 │
│  WHY IT'S IRREDUCIBLE:                                          │
│                                                                 │
│  If it could be reduced further:                               │
│  • The learning process WOULD reduce it                        │
│  • Gradient descent + regularization finds minimal forms      │
│  • What survives IS the irreducible core                       │
│                                                                 │
│  The collapsed state is the ATOMIC level:                       │
│  • Remove any bit → action capacity decreases                 │
│  • Add any bit → no improvement (already complete)            │
│  • This is the floor of compression                            │
│                                                                 │
│  ────────────────────────────────────────────────────────────  │
│                                                                 │
│  CONNECTION TO OTHER PRINCIPLES:                                │
│                                                                 │
│  ZIPF: Irreducible atoms have Zipf-optimal distribution       │
│  NOETHER: Explicit↔Implicit symmetry conserves this quantity │
│  DIRAC: Symmetry requirements determine the atomic structure  │
│                                                                 │
│  See: THE_ATOMIC_STRUCTURE_OF_INFORMATION.md for full details │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

STATUS: SPECULATIVE (proposed answer to the central mystery)
```

---

## 8. The Explicit-Implicit Duality

### 8.1 The Duality Principle

**PARTIALLY ESTABLISHED:**

```
THE EXPLICIT-IMPLICIT DUALITY:

Information can be stored in two complementary forms:

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  EXPLICIT                        IMPLICIT                       │
│  ────────                        ────────                       │
│                                                                 │
│  Stored in data                  Stored in model               │
│  High-frequency                  Low-frequency                  │
│  Phase information               Magnitude information          │
│  Instance-specific               Pattern-general                │
│  Memorized                       Generalized                    │
│  L(data|model) high              L(data|model) low             │
│  L(model) low                    L(model) high                 │
│                                                                 │
│  These are COMPLEMENTARY, like:                                 │
│  • Position and momentum (Fourier duality)                     │
│  • Time and frequency (Heisenberg)                             │
│  • Data and model (MDL)                                        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

ESTABLISHED ASPECTS:
• Fourier duality is mathematical fact
• MDL trade-off is well-established
• Position-momentum uncertainty is fundamental physics

SPECULATIVE ASPECTS:
• That this duality has Noether-like consequences
• That there's a conserved quantity associated with it
• That the learning collapse is a "symmetry restoration"

STATUS: PARTIALLY ESTABLISHED (duality yes, consequences speculative)
```

### 8.2 Fourier Duality as Foundation

**ESTABLISHED SCIENCE:**

```
FOURIER DUALITY (Pontryagin duality, Heisenberg):

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  THE MATHEMATICAL FACT:                                         │
│                                                                 │
│  Any function f(x) can be represented equivalently as:         │
│                                                                 │
│  f(x) ↔ F(ω) = ∫ f(x) e^(-iωx) dx                             │
│                                                                 │
│  These are THE SAME INFORMATION in different forms.            │
│                                                                 │
│  ────────────────────────────────────────────────────────────  │
│                                                                 │
│  THE UNCERTAINTY PRINCIPLE:                                     │
│                                                                 │
│  Δx · Δω ≥ 1/2                                                 │
│                                                                 │
│  You cannot be localized in both domains simultaneously.       │
│                                                                 │
│  ────────────────────────────────────────────────────────────  │
│                                                                 │
│  IN LEARNING TERMS:                                             │
│                                                                 │
│  • x = position/instance (explicit, high-freq)                 │
│  • ω = pattern/frequency (implicit, low-freq)                  │
│                                                                 │
│  A representation can be:                                       │
│  • Localized in instance space (memorized)                     │
│  • Localized in pattern space (generalized)                    │
│  • But not both maximally                                       │
│                                                                 │
│  REFERENCE:                                                     │
│  • Any Fourier analysis textbook                                │
│  • Folland (1999): Real Analysis, Chapter 8                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

STATUS: ESTABLISHED MATHEMATICAL FACT
```

---

## 9. Free Energy and Equilibrium

### 9.1 The Free Energy Principle

**ESTABLISHED SCIENCE (with speculative extensions):**

```
FREE ENERGY IN STATISTICAL MECHANICS:

F = E - TS

Where:
• E = Energy (want to minimize)
• T = Temperature
• S = Entropy (disorder)
• F = Free energy (what's actually minimized)

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  INTERPRETATION:                                                │
│                                                                 │
│  At low temperature: Minimize E (energy dominates)             │
│  At high temperature: Maximize S (entropy dominates)           │
│  At finite T: Balance E and S via F                            │
│                                                                 │
│  THE EQUILIBRIUM IS WHERE F IS MINIMIZED.                      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

STATUS: ESTABLISHED (fundamental thermodynamics)
```

### 9.2 Free Energy in Learning

**PARTIALLY ESTABLISHED:**

```
FREE ENERGY INTERPRETATION OF LEARNING:

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  MAPPING:                                                       │
│                                                                 │
│  Physics              Learning                                  │
│  ───────              ────────                                  │
│  Energy E             Loss L(θ)                                │
│  Entropy S            Model complexity / log volume             │
│  Temperature T        Learning rate / regularization           │
│  Free energy F        Regularized loss = L + λR                │
│                                                                 │
│  ESTABLISHED:                                                   │
│  • PAC-Bayes bounds give F = L + complexity term               │
│  • Variational inference minimizes variational free energy    │
│  • MDL = log(1/P) which is related to F                        │
│                                                                 │
│  THE EQUILIBRIUM:                                               │
│                                                                 │
│  Training converges to minimum of regularized loss,            │
│  which is analogous to free energy minimization.               │
│                                                                 │
│  At equilibrium:                                                │
│  • Fit to data (low E) and simplicity (low complexity) balanced│
│  • This is the "ground truth" representation                   │
│                                                                 │
│  REFERENCE:                                                     │
│  • McAllester (1999): PAC-Bayesian bounds                      │
│  • Friston (2010): Free energy principle in neuroscience       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

STATUS: PARTIALLY ESTABLISHED (analogy is well-studied)
```

### 9.3 The Ground State

**PARTIALLY ESTABLISHED / SPECULATIVE:**

```
THE GROUND STATE OF LEARNING:

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  In physics:                                                    │
│  • Ground state = minimum energy configuration                 │
│  • System "falls" into ground state as temperature → 0        │
│  • Ground state has special symmetry properties                │
│                                                                 │
│  In learning:                                                   │
│  • Ground state = optimal representation (min free energy)    │
│  • System "collapses" into it as training progresses          │
│  • Ground state has: Zipf distribution, hierarchical structure│
│                                                                 │
│  PROPERTIES OF THE GROUND STATE:                                │
│  ────────────────────────────────                               │
│                                                                 │
│  ESTABLISHED:                                                   │
│  • Minimum regularized loss                                    │
│  • Flat minima (good generalization)                           │
│  • Low effective dimensionality                                 │
│                                                                 │
│  SPECULATIVE:                                                   │
│  • Conserved quantity is extremized                            │
│  • Maximum symmetry expression                                  │
│  • Zipf-optimal compression achieved                           │
│                                                                 │
│  THE COLLAPSE IS THE TRANSITION TO THIS GROUND STATE.          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

STATUS: PARTIALLY ESTABLISHED (ground state concept yes, 
        specific properties speculative)
```

---

## 10. Information Geometry

### 10.1 Fisher Information

**ESTABLISHED SCIENCE:**

```
FISHER INFORMATION:

The Fisher information matrix measures how much information
data provides about parameters:

I(θ)_ij = E[ (∂ log p(x|θ) / ∂θ_i) (∂ log p(x|θ) / ∂θ_j) ]

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  INTERPRETATION:                                                │
│                                                                 │
│  • I(θ) defines a Riemannian metric on parameter space        │
│  • Distance between θ and θ+dθ is: ds² = dθᵀ I(θ) dθ          │
│  • This measures distinguishability of nearby models           │
│                                                                 │
│  PROPERTIES:                                                    │
│                                                                 │
│  • High Fisher info = parameters well-determined by data      │
│  • Low Fisher info = parameters poorly determined             │
│  • Cramér-Rao bound: Var(θ̂) ≥ I(θ)^(-1)                       │
│                                                                 │
│  REFERENCE:                                                     │
│  • Amari (1985): Differential-Geometrical Methods              │
│  • Rao (1945): Information and accuracy in estimation          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

STATUS: ESTABLISHED FACT
```

### 10.2 Natural Gradient

**ESTABLISHED SCIENCE:**

```
NATURAL GRADIENT (Amari, 1998):

Gradient descent in the Riemannian metric defined by Fisher info:

θ_new = θ - η I(θ)^(-1) ∇L(θ)

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  WHY NATURAL GRADIENT:                                          │
│                                                                 │
│  • Invariant to reparameterization                             │
│  • Moves in direction of steepest descent in probability space │
│  • Converges faster than vanilla gradient descent              │
│                                                                 │
│  THE KEY INSIGHT:                                               │
│                                                                 │
│  There's a NATURAL geometry on the space of models.            │
│  Optimization should respect this geometry.                    │
│                                                                 │
│  REFERENCE:                                                     │
│  • Amari (1998): Natural gradient works efficiently            │
│  • Martens (2020): New insights on natural gradient            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

STATUS: ESTABLISHED FACT
```

### 10.3 Information Geometry and Conservation

**SPECULATIVE:**

```
HYPOTHESIS: Geometric invariants as conserved quantities

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  THE IDEA:                                                      │
│                                                                 │
│  In Riemannian geometry, there are invariants:                 │
│  • Scalar curvature                                            │
│  • Geodesic length                                              │
│  • Volume elements                                              │
│                                                                 │
│  If learning follows geodesics in information geometry,        │
│  certain quantities might be conserved along the path.         │
│                                                                 │
│  CANDIDATES:                                                    │
│                                                                 │
│  1. det(I(θ)) - volume in parameter space                      │
│  2. trace(I(θ)) - total information                            │
│  3. Geodesic action ∫ √(dθᵀ I(θ) dθ)                          │
│                                                                 │
│  [SPECULATIVE: This is an open research direction.             │
│   No definitive results yet on what's conserved.]              │
│                                                                 │
│  REFERENCE:                                                     │
│  • Amari & Nagaoka (2000): Methods of Information Geometry     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

STATUS: SPECULATIVE (promising direction, not proven)
```

---

## 11. Synthesis: The Ground Truth Attractor

### 11.1 Pulling It Together

```
THE SYNTHESIZED PICTURE:

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  ESTABLISHED FACTS:                                             │
│  ────────────────                                               │
│  1. Learning systems undergo phase transitions (grokking)      │
│  2. Optimal representations have Zipf distributions            │
│  3. MDL principle governs model selection                      │
│  4. Information Bottleneck shows discrete transitions          │
│  5. Noether's theorem links symmetry to conservation           │
│  6. Fourier duality connects explicit/implicit                 │
│  7. Free energy is minimized at equilibrium                    │
│                                                                 │
│  SPECULATIVE SYNTHESIS:                                         │
│  ────────────────────                                           │
│  These facts suggest a unified picture:                        │
│                                                                 │
│  • There is a SYMMETRY in learning (explicit ↔ implicit)      │
│  • This symmetry implies a CONSERVED QUANTITY                  │
│  • The system evolves toward EQUILIBRIUM (min free energy)    │
│  • At equilibrium, compression is OPTIMAL (Zipf)              │
│  • The transition is a PHASE TRANSITION (collapse)            │
│  • The final state is DETERMINED by symmetry (Dirac)          │
│                                                                 │
│  STATUS: The individual facts are established.                  │
│          The synthesis is speculative but compelling.           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 11.2 The Ground Truth as Attractor

**SPECULATIVE:**

```
THE GROUND TRUTH ATTRACTOR:

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  CONJECTURE:                                                    │
│                                                                 │
│  There exists a unique "ground truth" representation that:     │
│                                                                 │
│  1. MINIMIZES free energy (optimal fit + simplicity)          │
│  2. SATISFIES symmetry constraints (translation invariance)   │
│  3. ACHIEVES Zipf-optimal compression                          │
│  4. CONSERVES the (unknown) conserved quantity                 │
│  5. EXISTS at the critical point (edge of chaos)              │
│                                                                 │
│  This ground truth is an ATTRACTOR:                            │
│  • All successful training trajectories converge to it        │
│  • It's unique (up to symmetry transformations)                │
│  • It's stable (perturbations return to it)                    │
│                                                                 │
│  THE COLLAPSE is the transition from:                           │
│  • Wandering in high-dimensional explicit space               │
│  • To falling into the low-dimensional implicit attractor     │
│                                                                 │
│  [SPECULATIVE: This is a hypothesis for future research.       │
│   Proving it would require identifying the conserved quantity  │
│   and showing it determines the attractor uniquely.]           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

STATUS: SPECULATIVE (compelling hypothesis, not proven)
```

### 11.3 The "Constructively Optimal" Point

**SPECULATIVE:**

```
WHEN COMPLEXITY BECOMES "CONSTRUCTIVELY OPTIMAL":

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  THE ORIGINAL INSIGHT (from discussion):                        │
│                                                                 │
│  "When complexity becomes constructively optimal, the          │
│   information phase transitions to its ground truth,           │
│   to conserve a symmetrical quantity."                         │
│                                                                 │
│  INTERPRETATION:                                                │
│                                                                 │
│  "Constructively optimal" = the point where:                   │
│  • Constructive interference (structure) maximized             │
│  • Destructive interference (noise) maximized                  │
│  • The ratio (signal/noise) is optimal                         │
│                                                                 │
│  At this point:                                                 │
│  • Zipf distribution emerges (optimal compression)             │
│  • Phase transition occurs (collapse to ground truth)          │
│  • Conserved quantity reaches extremum                          │
│                                                                 │
│  THE "SYMMETRICAL QUANTITY":                                    │
│                                                                 │
│  Might be related to:                                           │
│  • Self-duality (explicit ↔ implicit have equal "weight")     │
│  • Scale invariance (same structure at all levels)            │
│  • Reparameterization invariance (description is canonical)   │
│                                                                 │
│  This is the deep unknown we're trying to identify.            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

STATUS: SPECULATIVE (the core open question)
```

---

## 12. Open Questions

### 12.1 Theoretical Questions

```
OPEN THEORETICAL QUESTIONS:

1. WHAT IS THE CONSERVED QUANTITY?
   - Is there an exact conservation law?
   - What is its mathematical form?
   - How does it relate to Noether's theorem?

2. WHAT IS THE SYMMETRY?
   - Beyond translation invariance, what symmetries matter?
   - Is there a group structure?
   - What's the corresponding Lie algebra?

3. IS THE GROUND TRUTH UNIQUE?
   - Given constraints, is the optimal representation unique?
   - Up to what equivalences?
   - What determines the symmetry breaking?

4. CAN WE PREDICT COLLAPSE?
   - What are the order parameters (if any)?
   - Does collapse exhibit any measurable scaling behavior?
   - Note: "universality class" language is aspirational, not established

5. IS THERE AN ACTION PRINCIPLE?
   - Can learning be derived from a variational principle?
   - What is the Lagrangian?
   - Are there "path integrals" over learning trajectories?
```

### 12.2 Empirical Questions

```
OPEN EMPIRICAL QUESTIONS:

1. MEASURING THE CONSERVED QUANTITY
   - Can we identify and measure it experimentally?
   - Does it stay constant during training?
   - Does it predict collapse timing?

2. VERIFYING ZIPF OPTIMALITY
   - Do all well-trained networks show Zipf distributions?
   - In what observables?
   - What deviations indicate problems?

3. DETECTING THE CRITICAL POINT
   - What are measurable signatures of criticality?
   - Can we engineer systems to stay at the critical point?
   - What's the effect of hyperparameters?

4. SIMILAR BEHAVIOR ACROSS ARCHITECTURES?
   - Do CNNs, transformers, RNNs show similar transitions?
   - Are there common patterns in how collapse occurs?
   - What aspects are architecture-dependent?
```

### 12.3 The Central Mystery

```
THE CENTRAL MYSTERY:

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  We observe that learning systems:                              │
│                                                                 │
│  1. Start with explicit, detailed representations              │
│  2. Undergo sudden transitions (collapse)                       │
│  3. End with implicit, abstract representations                 │
│  4. Exhibit Zipf-optimal distributions                          │
│  5. Achieve generalization                                       │
│                                                                 │
│  This looks like a physical phase transition.                   │
│  Phase transitions conserve certain quantities.                 │
│  What is being conserved here?                                  │
│                                                                 │
│  THE ANSWER TO THIS QUESTION WOULD:                             │
│                                                                 │
│  • Unify learning theory with physics                          │
│  • Explain why generalization is possible                      │
│  • Predict optimal architectures                               │
│  • Guide the design of learning systems                        │
│                                                                 │
│  This is the deep question we are circling.                     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Summary

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  EQUILIBRIUM AND CONSERVATION IN LEARNING                       │
│                                                                 │
│  ESTABLISHED:                                                   │
│  ───────────                                                    │
│  • Learning systems undergo phase transitions                   │
│  • Optimal compression leads to Zipf distributions             │
│  • MDL and Information Bottleneck govern representations       │
│  • Noether's theorem links symmetry to conservation            │
│  • Free energy minimization drives equilibration               │
│                                                                 │
│  SPECULATIVE:                                                   │
│  ───────────                                                    │
│  • There exists a conserved quantity in learning               │
│  • This quantity is related to a symmetry (explicit↔implicit) │
│  • The collapse is a transition to a "ground truth" state     │
│  • Zipf distribution is a signature of this equilibrium       │
│  • Symmetry requirements determine the form of solutions       │
│                                                                 │
│  THE CENTRAL HYPOTHESIS:                                        │
│  ──────────────────────                                         │
│  When complexity becomes constructively optimal,               │
│  information phase-transitions to its ground truth,            │
│  to conserve a symmetrical quantity related to:                │
│  • Zipf (optimal compression)                                  │
│  • Noether (symmetry → conservation)                           │
│  • Dirac (symmetry determines form)                            │
│                                                                 │
│  The identity of this quantity remains the open question.       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Related Documents

| Document | Connection |
|----------|------------|
| `CONSERVATION_OF_ACTION.md` | The fire analogy - what is actually conserved |
| `pandora/PANDORA.md` | Action as universal operator between dual forms |
| `pandora/PANDORA_AFTERMATH.md` | Hope as the conserved generative capacity (the generator NOT consumed by generating) |

---

## References

### Established Science

1. **Noether, E.** (1918). Invariante Variationsprobleme. *Nachr. D. König. Gesellsch. D. Wiss. Zu Göttingen*.

2. **Dirac, P.A.M.** (1928). The Quantum Theory of the Electron. *Proceedings of the Royal Society A*.

3. **Zipf, G.K.** (1949). *Human Behavior and the Principle of Least Effort*. Addison-Wesley.

4. **Shannon, C.E.** (1948). A Mathematical Theory of Communication. *Bell System Technical Journal*.

5. **Rissanen, J.** (1978). Modeling by Shortest Data Description. *Automatica*.

6. **Tishby, N., Pereira, F., & Bialek, W.** (1999). The Information Bottleneck Method. *37th Allerton Conference*.

7. **Amari, S.** (1998). Natural Gradient Works Efficiently in Learning. *Neural Computation*.

8. **Bak, P.** (1996). *How Nature Works: The Science of Self-Organized Criticality*. Copernicus.

9. **Martin, C.H. & Mahoney, M.W.** (2021). Implicit Self-Regularization in Deep Neural Networks. *JMLR*.

10. **Power, A., et al.** (2022). Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets. *arXiv:2201.02177*.

### Speculative Extensions

The connections proposed in this document between these established results—particularly the hypothesis of a conserved quantity and the role of symmetry in determining optimal representations—are speculative and represent directions for future research.

---

*This document explores the deep connections between physics and learning theory. While grounded in established science, it ventures into speculative territory in hypothesizing a conserved quantity and symmetry-determined equilibrium. The central question—what is conserved during the collapse from explicit to implicit representation—remains open.*

