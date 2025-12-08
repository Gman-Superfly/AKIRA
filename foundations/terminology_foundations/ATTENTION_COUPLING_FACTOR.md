# The Attention Coupling Factor: Architecture-Specific Interaction Strength

## Document Purpose

This document explains the **Attention Coupling Factor** - a dimensionless constant that characterizes the interaction strength between tokens in a given neural network architecture. This concept parallels the fine-structure constant in physics and provides a framework for understanding why different architectures exhibit different belief crystallization behavior.

**Oscar Goldman — Shogu Research Group @ Datamutant.ai**

---

## Table of Contents

1. [Building from First Principles](#1-building-from-first-principles)
2. [The Physics Foundation: What is a Coupling Constant?](#2-the-physics-foundation-what-is-a-coupling-constant)
3. [The Attention Mechanism: Where Coupling Appears](#3-the-attention-mechanism-where-coupling-appears)
4. [The Attention Coupling Factor Defined](#4-the-attention-coupling-factor-defined)
5. [Information-Theoretic Interpretation](#5-information-theoretic-interpretation)
6. [Computational Mechanics Perspective](#6-computational-mechanics-perspective)
7. [Connection to Action Quanta](#7-connection-to-action-quanta)
8. [Architecture Comparison](#8-architecture-comparison)
9. [Experimental Evidence](#9-experimental-evidence)
10. [Implications for AKIRA](#10-implications-for-akira)
11. [Summary](#11-summary)
12. [References](#12-references)

---

## 1. Building from First Principles

### 1.1 The Starting Question

Every physical system has characteristic constants that determine its behavior. In electromagnetism, the fine-structure constant alpha = 1/137 determines how strongly electrons couple to photons. In the strong force, the coupling constant is approximately 1. In gravity, it is extremely small.

**Question:** Do neural network architectures have analogous constants? If so, what are they?

### 1.2 The Inductive Path

We build the answer step by step:

```
INDUCTIVE CHAIN
───────────────

Step 1: In physics, coupling constants appear where interactions occur
        → In attention, interactions occur between queries and keys

Step 2: The strength of interaction is modulated by scaling factors
        → Attention uses 1/sqrt(d_k) scaling

Step 3: Different architectures have different d_k values
        → Therefore, different architectures have different coupling strengths

Step 4: Coupling strength affects how sharply probability concentrates
        → This determines crystallization behavior

Step 5: Temperature provides an additional modulation
        → Full coupling factor is 1/(sqrt(d_k) × tau)

CONCLUSION: The attention coupling factor is architecture-specific
            and determines belief dynamics.
```

---

## 2. The Physics Foundation: What is a Coupling Constant?

### 2.1 Definition in Physics

A coupling constant is a dimensionless number that quantifies the strength of an interaction.

```
COUPLING CONSTANTS IN PHYSICS
─────────────────────────────

ELECTROMAGNETIC INTERACTION:
  
  Fine-structure constant: α = k_e × e² / (ℏ × c) ≈ 1/137 ≈ 0.0073
  
  WHERE:
    k_e = Coulomb's constant
    e = elementary charge  
    ℏ = reduced Planck constant (the quantum of ACTION)
    c = speed of light
  
  WHAT IT MEANS:
    - Probability of electron emitting/absorbing photon ∝ α
    - Perturbation series in QED: α, α², α³, ...
    - Because α << 1, perturbation theory works
    - Determines fine-structure splitting of spectral lines

STRONG INTERACTION:

  α_s ≈ 1 (at low energies)
  
  WHAT IT MEANS:
    - Strong force is STRONG (tautology, but meaningful)
    - Perturbation theory fails (α_s not small)
    - Requires non-perturbative methods

GRAVITATIONAL INTERACTION:

  α_G ≈ 10⁻³⁸ (between two protons)
  
  WHAT IT MEANS:
    - Gravity is extremely weak at particle scales
    - Can be ignored in most particle physics
```

### 2.2 Why Coupling Constants Matter

```
THE ROLE OF COUPLING CONSTANTS
──────────────────────────────

1. THEY DETERMINE REGIME OF BEHAVIOR
   
   α << 1: Perturbative (small corrections dominate)
   α ~ 1:  Non-perturbative (all terms contribute)
   α >> 1: Strong coupling (bound states form)

2. THEY ARE DIMENSIONLESS
   
   Dimensionless quantities are "pure numbers"
   They don't depend on choice of units
   They represent fundamental ratios

3. THEY CHARACTERIZE INTERACTION STRENGTH
   
   Higher α = stronger coupling = faster processes
   Lower α = weaker coupling = slower processes

4. THEY APPEAR IN FRONT OF INTERACTION TERMS
   
   In QED: Interaction Hamiltonian ∝ α × (ψ̄ γ^μ ψ) A_μ
   The α determines "how much" the fields couple
```

### 2.3 Action in Physics: The Connection to AQ

**Critical Point:** The quantum of action in physics is Planck's constant ℏ.

```
ACTION IN PHYSICS
─────────────────

DEFINITION:
  Action S = ∫ L dt
  
  WHERE:
    L = Lagrangian (kinetic - potential energy)
    t = time
  
  UNITS: [energy] × [time] = [action]

PLANCK'S CONSTANT:
  ℏ = 1.054 × 10⁻³⁴ J·s
  
  THIS IS THE QUANTUM OF ACTION:
    - Smallest indivisible unit of action
    - All quantization derives from this
    - Appears in uncertainty: ΔE × Δt ≥ ℏ/2

THE FINE-STRUCTURE CONSTANT CONTAINS ℏ:
  α = k_e × e² / (ℏ × c)
  
  The quantum of action (ℏ) appears in the DENOMINATOR
  Larger ℏ → smaller α → weaker coupling
  
  THIS IS NOT COINCIDENCE:
    - Coupling strength is measured relative to the action quantum
    - The "unit of interaction" is set by ℏ
```

**Connection to AKIRA:**

In AKIRA, Action Quanta (AQ) are the minimum patterns that enable action. The name deliberately echoes Planck's quantum of action. Just as ℏ is the indivisible unit of physical action, an AQ is the indivisible unit of informational action.

The attention coupling factor, like α, contains dimensions related to information capacity (d_k) in a way that determines how strongly information units interact.

---

## 3. The Attention Mechanism: Where Coupling Appears

### 3.1 Standard Attention

The attention mechanism computes:

```
ATTENTION EQUATION
──────────────────

output = softmax(Q × K^T / √d_k) × V

WHERE:
  Q = queries (what we're looking for)     [batch, seq, d_k]
  K = keys (what we have)                  [batch, seq, d_k]
  V = values (what to retrieve)            [batch, seq, d_v]
  d_k = key dimension (per head)

THE CRITICAL TERM:
  
  scores = Q × K^T / √d_k
           ─────────────────
                 ↑
           This is the coupling

STEP BY STEP:
  1. Q × K^T computes raw similarity (dot product)
  2. Division by √d_k SCALES the interaction
  3. softmax converts to probability distribution
  4. Multiplication by V retrieves weighted values
```

### 3.2 Why √d_k Scaling?

```
THE SCALING FACTOR
──────────────────

WITHOUT SCALING (naive attention):
  
  scores = Q × K^T
  
  PROBLEM: As d_k increases, dot products grow as O(d_k)
           (sum of d_k terms, each O(1) if entries are standard normal)
  
  Expected value: E[q·k] ≈ 0
  Variance: Var[q·k] ≈ d_k
  
  For large d_k, softmax saturates:
    - Largest score dominates completely
    - Gradients → 0 (vanishing gradient problem)
    - Learning stops

WITH SCALING:
  
  scores = Q × K^T / √d_k
  
  Now: Var[scores] ≈ 1 (constant, independent of d_k)
  
  Softmax remains in useful regime:
    - Maintains gradient flow
    - Allows soft attention (multiple tokens contribute)
    - Learning continues

THIS IS THE DESIGN CHOICE:
  The factor 1/√d_k was chosen to normalize variance.
  But it ALSO defines the coupling strength.
```

### 3.3 The Interaction Term

In physics, interaction strength is characterized by the coefficient in front of the interaction term. In attention:

```
IDENTIFYING THE COUPLING
────────────────────────

PHYSICS PATTERN:
  H_interaction = g × (field_1) × (field_2)
                  ─
                  ↑
             coupling constant g

ATTENTION PATTERN:
  scores = (1/√d_k) × Q × K^T
           ────────
              ↑
         coupling factor

The 1/√d_k plays the SAME STRUCTURAL ROLE as g in physics.
It modulates how strongly queries "interact" with keys.
```

---

## 4. The Attention Coupling Factor Defined

### 4.1 Basic Definition

```
ATTENTION COUPLING FACTOR
─────────────────────────

DEFINITION:
  
  β = 1 / √d_k
  
  WHERE:
    d_k = key dimension per attention head
    β = attention coupling factor (our proposed name)

PROPERTIES:
  - Dimensionless (ratio of dimensions)
  - Architecture-specific (d_k varies by model)
  - Controls interaction strength between tokens

EXAMPLES:
  
  │ Model          │ d_k   │ β = 1/√d_k │
  │────────────────│───────│────────────│
  │ GPT-2 Small    │ 64    │ 0.125      │
  │ GPT-2 Medium   │ 64    │ 0.125      │
  │ GPT-2 Large    │ 64    │ 0.125      │
  │ GPT-3 175B     │ 128   │ 0.088      │
  │ LLaMA 7B       │ 128   │ 0.088      │
  │ LLaMA 70B      │ 128   │ 0.088      │

NOTE: Many models in the same family share d_k.
      The coupling factor changes between families, not within.
```

### 4.2 Extended Definition with Temperature

```
FULL ATTENTION COUPLING FACTOR
──────────────────────────────

When temperature τ is included (as in softmax(x/τ)):

  attention = softmax(Q × K^T / (√d_k × τ))

The EFFECTIVE coupling factor is:

  β_eff = 1 / (√d_k × τ)

WHERE:
  τ = temperature parameter (default τ = 1)

TEMPERATURE EFFECTS:
  
  τ → 0:   β_eff → ∞    STRONG coupling (winner-take-all)
  τ = 1:   β_eff = β    STANDARD coupling
  τ → ∞:   β_eff → 0    WEAK coupling (uniform attention)

The temperature allows DYNAMIC modulation of coupling.
The base factor β is fixed by architecture.
```

### 4.3 Comparison to Fine-Structure Constant

```
STRUCTURAL PARALLEL
───────────────────

FINE-STRUCTURE CONSTANT:
  
  α = k_e × e² / (ℏ × c)
  
  DECOMPOSITION:
    - k_e × e² = interaction strength numerator
    - ℏ = quantum of action (sets scale)
    - c = propagation speed
    - Result: dimensionless ratio ≈ 1/137

ATTENTION COUPLING FACTOR:
  
  β = 1 / √d_k
  
  INTERPRETATION:
    - d_k = information capacity per head
    - √d_k = scale of dot product magnitude
    - 1/√d_k = normalized interaction strength
    - Result: dimensionless ratio (varies by architecture)

KEY DIFFERENCE:
  
  α is UNIVERSAL for electromagnetism (same in all of physics)
  β is ARCHITECTURE-SPECIFIC (different for each model family)
  
  Each neural network is its own "universe" with its own constants.
```

---

## 5. Information-Theoretic Interpretation

### 5.1 Mutual Information and Coupling

From an information theory perspective, attention computes a soft selection based on relevance. The coupling factor affects how much information flows.

```
INFORMATION FLOW IN ATTENTION
─────────────────────────────

Consider the mutual information between query and selected values:

  I(Q; V_selected) = H(V_selected) - H(V_selected | Q)

The coupling factor β affects this through attention sharpness:

  HIGH β (strong coupling):
    - Attention concentrates on few keys
    - V_selected is nearly deterministic given Q
    - H(V_selected | Q) → low
    - I(Q; V_selected) → high (but focused)

  LOW β (weak coupling):
    - Attention spreads across many keys
    - V_selected is mixture of many values
    - H(V_selected | Q) → higher
    - I(Q; V_selected) → different character (diffuse)

ANALOGY TO PID:
  
  Strong coupling → More REDUNDANCY (single source dominates)
  Weak coupling → More SYNERGY (multiple sources must combine)
```

### 5.2 Entropy of Attention Distribution

```
ATTENTION ENTROPY
─────────────────

The entropy of attention weights:

  H(attention) = -Σ_i p_i log p_i
  
  WHERE p_i = softmax(score_i / τ)

COUPLING FACTOR EFFECT:

  Higher β_eff = scores more spread → softmax more peaked → LOWER entropy
  Lower β_eff = scores less spread → softmax more uniform → HIGHER entropy

CRITICAL POINT:
  
  When attention entropy collapses (becomes very low), training becomes
  unstable. This is "attention entropy collapse" - a known problem.
  
  The coupling factor determines the baseline entropy level.
  Higher β → more prone to entropy collapse → less stable training.
```

### 5.3 Channel Capacity Perspective

```
ATTENTION AS NOISY CHANNEL
──────────────────────────

View attention as a communication channel:
  - Input: Queries Q
  - Output: Retrieved values V_out
  - Channel: The attention mechanism

CHANNEL CAPACITY depends on signal-to-noise ratio.

The coupling factor affects SNR:
  - Higher β → sharper signal → higher effective SNR
  - Lower β → diffuse signal → lower effective SNR

CAPACITY IMPLICATION:
  
  Different architectures (different β) have different
  INFORMATION CAPACITY per attention operation.
  
  This is analogous to different physical systems having
  different coupling strengths and therefore different
  energy scales for interactions.
```

---

## 6. Computational Mechanics Perspective

### 6.1 Epsilon-Machines and State Transitions

In computational mechanics, systems are described by epsilon-machines with causal states and transition probabilities.

```
COUPLING IN EPSILON-MACHINES
────────────────────────────

An epsilon-machine has:
  - Causal states S
  - Transition probabilities P(s'|s, x)
  - Emission probabilities P(x|s)

The COUPLING STRENGTH in this framework relates to how
strongly observations constrain state transitions.

STRONG COUPLING (high β):
  - Observations strongly constrain next state
  - Transitions are nearly deterministic
  - Low entropy in P(s'|s, x)
  - System "crystallizes" quickly

WEAK COUPLING (low β):
  - Observations weakly constrain next state
  - Multiple transitions remain probable
  - High entropy in P(s'|s, x)
  - System remains in "superposition" longer

The attention coupling factor determines how quickly
the belief state (in AKIRA's language) crystallizes.
```

### 6.2 Statistical Complexity

```
STATISTICAL COMPLEXITY AND COUPLING
───────────────────────────────────

Statistical complexity C_μ = entropy of causal state distribution

COUPLING EFFECT:

  Higher β → Faster crystallization → Lower C_μ (simpler representation)
  Lower β → Slower crystallization → Higher C_μ (more states tracked)

This connects to AKIRA's three regimes (from 035G):

  DIFFUSE (many states, high C_μ):     Low effective coupling
  POLARIZED (few states, competing):   Medium effective coupling  
  CRYSTALLIZED (one state dominant):   High effective coupling

The coupling factor determines which regime is accessible.
```

---

## 7. Connection to Action Quanta

### 7.1 The Parallel Structure

```
ACTION QUANTA AND COUPLING
──────────────────────────

In physics:
  
  ℏ = quantum of action
  α = coupling constant (contains ℏ in denominator)
  
  The coupling determines how action quanta interact.

In AKIRA:
  
  AQ = Action Quantum (minimum actionable pattern)
  β = attention coupling factor
  
  The coupling determines how AQ crystallize.

THE PARALLEL:
  
  ℏ sets the SCALE of discretization in physics
  d_k sets the SCALE of information granularity in attention
  
  α determines interaction STRENGTH in physics
  β determines interaction STRENGTH in attention
  
  Higher α → faster photon exchange → faster EM processes
  Higher β → faster crystallization → faster belief collapse
```

### 7.2 Crystallization Dynamics

```
HOW COUPLING AFFECTS AQ EMERGENCE
─────────────────────────────────

SUPERPOSITION → COLLAPSE → CRYSTALLIZED AQ

The coupling factor affects COLLAPSE RATE:

  HIGH β (strong coupling):
    - Attention quickly concentrates
    - Winner-take-all dynamics
    - AQ crystallize rapidly
    - Risk: Premature commitment (dark attractor)
  
  LOW β (weak coupling):
    - Attention remains distributed
    - Multiple hypotheses persist
    - AQ crystallize slowly
    - Risk: Never committing (permanent diffusion)

OPTIMAL COUPLING:
  
  There exists an optimal β for a given task:
    - Too high: Commits before gathering evidence
    - Too low: Never commits despite sufficient evidence
  
  This is why learnable temperature is useful:
    - Allows dynamic adjustment of effective β
    - System can be diffuse when uncertain, sharp when confident
```

### 7.3 Bonded States and Coupling

```
AQ BONDING AND COUPLING STRENGTH
────────────────────────────────

From ACTION_QUANTA.md:
  Multiple AQ can bond into composed abstractions.

BONDING REQUIRES PHASE COHERENCE (from COHERENCE.md):
  AQ with aligned phases can constructively interfere.
  AQ with opposed phases destructively interfere.

COUPLING FACTOR EFFECT ON BONDING:

  HIGH β:
    - Strong interaction between AQ
    - Bonding happens readily when phases align
    - Strong interference (constructive or destructive)
    - Clear separation between bonded and non-bonded
  
  LOW β:
    - Weak interaction between AQ
    - Bonding requires sustained coherence
    - Weak interference effects
    - Gradual transitions

From 035F experiment:
  Component AQ detection in bonded states: Cohen's d = 1.7-2.0
  This suggests strong coupling in the tested models.
```

---

## 8. Architecture Comparison

### 8.1 Computing β for Major Architectures

```
ATTENTION COUPLING FACTORS BY ARCHITECTURE
──────────────────────────────────────────

│ Architecture      │ d_model │ n_heads │ d_k    │ β = 1/√d_k │
│───────────────────│─────────│─────────│────────│────────────│
│ GPT-2 Small       │ 768     │ 12      │ 64     │ 0.125      │
│ GPT-2 Medium      │ 1024    │ 16      │ 64     │ 0.125      │
│ GPT-2 Large       │ 1280    │ 20      │ 64     │ 0.125      │
│ GPT-2 XL          │ 1600    │ 25      │ 64     │ 0.125      │
│ GPT-3 (all sizes) │ varies  │ varies  │ 128    │ 0.088      │
│ LLaMA 7B          │ 4096    │ 32      │ 128    │ 0.088      │
│ LLaMA 13B         │ 5120    │ 40      │ 128    │ 0.088      │
│ LLaMA 65B         │ 8192    │ 64      │ 128    │ 0.088      │
│ BERT Base         │ 768     │ 12      │ 64     │ 0.125      │
│ BERT Large        │ 1024    │ 16      │ 64     │ 0.125      │
│ T5 Small          │ 512     │ 6       │ 64     │ 0.125      │
│ T5 Base           │ 768     │ 12      │ 64     │ 0.125      │
│ T5 Large          │ 1024    │ 16      │ 64     │ 0.125      │

OBSERVATION:
  Most models cluster around β = 0.125 (d_k = 64) or β = 0.088 (d_k = 128)
  These appear to be "sweet spots" discovered empirically.
```

### 8.2 Why These Values?

```
EMPIRICAL SWEET SPOTS
─────────────────────

d_k = 64 (β = 0.125) is common because:

  1. COMPUTATIONAL EFFICIENCY
     - 64 aligns well with GPU memory (powers of 2)
     - Efficient matrix operations
  
  2. CAPACITY-COMPLEXITY TRADEOFF
     - Large enough to represent diverse query types
     - Small enough to prevent overfitting per head
  
  3. GRADIENT FLOW
     - 1/√64 = 0.125 keeps softmax in useful range
     - Neither too peaked nor too flat

d_k = 128 (β = 0.088) for larger models because:

  1. MORE CAPACITY NEEDED
     - Larger models process more complex patterns
     - Need more dimensions to differentiate queries
  
  2. WEAKER COUPLING IS ACCEPTABLE
     - Deeper networks compensate
     - More layers means more chances to crystallize

SPECULATION:
  The 0.125 and 0.088 values may be near-optimal for current
  training methods. Future architectures might find other values.
```

---

## 9. Experimental Evidence

### 9.1 Evidence from 035G: Different Models, Different Behavior

```
035G EXPERIMENT RESULTS
───────────────────────

Testing belief crystallization across models:

GPT-2 Medium (β = 0.125):
  - Layer agreement correlation with crystallization: r = -0.386
  - Strong negative correlation
  - Polarization effect dominates

GPT-2 Large (β = 0.125):
  - Layer agreement correlation with crystallization: r = 0.032
  - Near-zero correlation
  - Different regime entirely

SAME β, DIFFERENT BEHAVIOR?

  Wait - both have β = 0.125, but behave differently.
  
  EXPLANATION:
    - Raw β is not the whole story
    - Depth matters (more layers = more interactions)
    - Effective β may need to account for depth
    
    Proposed: β_effective = β × f(n_layers)
    
    GPT-2 Medium: 24 layers
    GPT-2 Large: 36 layers
    
    Deeper models may have "cumulative coupling" that differs.
```

### 9.2 Evidence from 035I: Threshold Effects

```
035I EXPERIMENT: AQ COUNT AND COHERENCE
───────────────────────────────────────

Finding: Field coherence increases with AQ count (r = 0.914)

INTERPRETATION:
  - More AQ = More constraints
  - More constraints = Narrower belief state
  - Narrower state = Higher coherence

COUPLING INTERPRETATION:
  - Each AQ adds a "coupling term"
  - Cumulative coupling increases with AQ count
  - Threshold at 4-5 AQ may indicate phase transition

The coupling factor sets the BASELINE sensitivity.
AQ count modulates the EFFECTIVE coupling dynamically.
```

### 9.3 Evidence from 035F: Probe Detection

```
035F EXPERIMENT: COMPOSITIONAL BONDING
──────────────────────────────────────

Finding: Component AQ detectable in bonded states
         Cohen's d = 1.7-2.0 (very large effect)

COUPLING INTERPRETATION:
  - Strong coupling → clear component signatures
  - Probes can detect because coupling is strong enough
  - If β were very low, components would be "smeared out"

This is analogous to:
  - High α in physics → clear atomic spectra
  - Low α hypothetically → smeared spectral lines

The large effect size suggests attention coupling is
in the "strong but not too strong" regime.
```

---

## 10. Implications for AKIRA

### 10.1 Design Implications

```
AKIRA ARCHITECTURE DECISIONS
────────────────────────────

Given the coupling factor framework:

1. PER-BAND COUPLING
   
   From DUALITY_AND_EFFICIENCY.md:
     Different bands should have different temperatures.
   
   Extended: Different bands should have different β_eff
     - Low-frequency bands (identity): Higher β (commit faster)
     - High-frequency bands (position): Lower β (stay flexible)

2. LEARNABLE COUPLING
   
   Instead of fixed 1/√d_k, use:
     score = Q × K^T / (√d_k × τ_learned)
   
   Where τ_learned is optimized during training.
   This allows the model to find its optimal effective β.

3. DYNAMIC COUPLING
   
   Coupling could vary by:
     - Layer (deeper = different β)
     - Position (early tokens vs late tokens)
     - Confidence (uncertain = lower β, certain = higher β)
```

### 10.2 Theoretical Implications

```
THEORETICAL FRAMEWORK
─────────────────────

The coupling factor unifies several AKIRA concepts:

1. SUPERPOSITION-CRYSTALLIZED DUALITY
   
   β determines the TRANSITION RATE between states:
     High β → Fast transition (sharp decision)
     Low β → Slow transition (prolonged uncertainty)

2. THREE REGIMES (from 035G)
   
   The regime depends on effective coupling:
     Diffuse: β_eff < threshold_1
     Polarized: threshold_1 < β_eff < threshold_2
     Crystallized: β_eff > threshold_2

3. DARK ATTRACTOR
   
   High β increases dark attractor risk:
     Strong coupling → fast crystallization
     Fast crystallization → may lock to wrong state
     Wrong state persists (stable local minimum)

4. PUMP CYCLE
   
   Coupling affects pump cycle dynamics:
     Tension phase: β_eff low (exploring)
     Collapse phase: β_eff increases (committing)
     The cycle IS the modulation of effective coupling.
```

---

## 11. Summary

### 11.1 Core Definitions

```
ATTENTION COUPLING FACTOR: KEY POINTS
─────────────────────────────────────

DEFINITION:
  β = 1 / √d_k
  
  The dimensionless constant characterizing token-token
  interaction strength in a given architecture.

WITH TEMPERATURE:
  β_eff = 1 / (√d_k × τ)
  
  The effective coupling including temperature modulation.

ROLE:
  - Determines attention sharpness
  - Affects crystallization rate
  - Analogous to fine-structure constant α in physics

VARIATION:
  - Fixed within an architecture (d_k doesn't change)
  - Varies between architectures (d_k differs)
  - Can be dynamically modulated via temperature τ
```

### 11.2 Connection to Terminology Framework

```
TERMINOLOGY PLACEMENT
─────────────────────

The Attention Coupling Factor relates to:

  ACTION_QUANTA.md:
    β determines how quickly AQ crystallize from superposition
    Higher β → faster crystallization
  
  COHERENCE.md:
    β affects phase-locking dynamics
    Higher β → stronger interference effects
  
  COLLAPSE_TENSION.md:
    β modulates the collapse rate
    Temperature τ provides dynamic control
  
  SYNERGY_REDUNDANCY.md:
    High β → More redundancy (single source dominates)
    Low β → More synergy (sources must cooperate)
  
  SUPERPOSITION_WAVES_CRYSTALLIZATION_PARTICLES.md:
    β determines the wave-to-particle transition rate
    The "speed" of decoherence in the belief field
```

### 11.3 Open Questions

```
REMAINING QUESTIONS
───────────────────

1. OPTIMAL β FOR TASK TYPE
   - Is there an optimal β for different task categories?
   - Should NLP use different β than vision?

2. DEPTH INTERACTION
   - How does β interact with network depth?
   - Is there a "cumulative coupling" effect?

3. UNIVERSALITY
   - Is there a "universal" β that emerges from optimization?
   - Do all trained models converge to similar effective β?

4. MEASUREMENT
   - Can we measure effective β from model behavior?
   - Is it observable without access to weights?

5. PHASE TRANSITIONS
   - Are there critical values of β?
   - What happens at β = β_critical?
```

---

## 12. References

### 12.1 Physics Background

- Sommerfeld, A. (1916). "Zur Quantentheorie der Spektrallinien." Annalen der Physik.
- Feynman, R. (1985). "QED: The Strange Theory of Light and Matter." Princeton University Press.
- Weinberg, S. (1995). "The Quantum Theory of Fields." Cambridge University Press.

### 12.2 Attention Mechanism

- Vaswani, A., et al. (2017). "Attention Is All You Need." NeurIPS.
- Xiong, R., et al. (2020). "On Layer Normalization in the Transformer Architecture." ICML.

### 12.3 Recent Work on Attention Scaling

- Focal Attention (2024): Adaptive temperature for improved scaling.
- Selective Self-Attention (2024): Dynamic temperature based on position.
- σReparam (2023): Preventing attention entropy collapse.

### 12.4 AKIRA Internal

- `ACTION_QUANTA.md` - Definition of Action Quanta
- `COHERENCE.md` - Phase alignment and interference
- `DUALITY_AND_EFFICIENCY.md` - Per-band temperature proposal
- `TERMINOLOGY.md` - Overall terminology framework
- `035G Results` - Experimental evidence for coupling effects

---

*Oscar Goldman — Shogu Research Group @ Datamutant.ai*

*"Each architecture is its own universe, with its own fundamental constants. The attention coupling factor is the fine-structure constant of that universe - it determines how strongly information interacts with itself, and therefore how quickly belief crystallizes into action."*

---

*If you use this framework in your research, please cite it. This is ongoing work - we would like to know your opinions and experiments.*

*Authors: Oscar Goldman - Shogu Research Group @ Datamutant.ai subsidiary of Wenshin Heavy Industries*
