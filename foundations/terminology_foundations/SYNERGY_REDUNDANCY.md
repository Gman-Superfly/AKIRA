# Synergy and Redundancy: Foundational Concepts

## Document Purpose

This document explains **synergy** and **redundancy** from first principles. These concepts come from Partial Information Decomposition (PID) and are fundamental to understanding how AKIRA represents and processes belief states.

---

## Table of Contents

1. [The Core Question](#1-the-core-question)
2. [Redundancy Explained](#2-redundancy-explained)
3. [Synergy Explained](#3-synergy-explained)
4. [The Complete PID Decomposition](#4-the-complete-pid-decomposition)
5. [Theory and Mathematical Foundation](#5-theory-and-mathematical-foundation)
6. [General Applications](#6-general-applications)
7. [Synergy and Redundancy in AKIRA](#7-synergy-and-redundancy-in-akira)
8. [How This Informs AKIRA's Theoretical Foundations](#8-how-this-informs-akiras-theoretical-foundations)
9. [References](#9-references)

---

## 1. The Core Question

Imagine you want to know something (the **target**). You have two sources of information. The core question is:

> **How is the information about the target distributed across the sources?**

There are fundamentally different ways information can be distributed:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  QUESTION: What can each source tell you about the target?             │
│                                                                         │
│  CASE 1: Either source alone can tell you                              │
│          → They SHARE the information                                  │
│          → This is REDUNDANCY                                          │
│                                                                         │
│  CASE 2: Only one specific source can tell you                         │
│          → That source has EXCLUSIVE information                       │
│          → This is UNIQUE information                                  │
│                                                                         │
│  CASE 3: Neither source alone can tell you, but together they can      │
│          → The information EMERGES from combination                    │
│          → This is SYNERGY                                             │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Redundancy Explained

### 2.1 Definition

**Redundancy** is information that **both sources provide** about the target. You can ask either source and get the answer.

### 2.2 Intuitive Example: Temperature

```
EXAMPLE: Redundant Temperature Information
──────────────────────────────────────────

Target (T):     Is it hot outside? (yes/no)
Source 1 (S₁):  Temperature in Celsius
Source 2 (S₂):  Temperature in Fahrenheit

Ask S₁: "It's 35°C"  →  Yes, it's hot.
Ask S₂: "It's 95°F"  →  Yes, it's hot.

Both sources can answer the question independently.
They AGREE. They provide the SAME information about T.
This shared information is REDUNDANCY.
```

### 2.3 Key Properties of Redundancy

```
REDUNDANCY PROPERTIES:
──────────────────────

1. EITHER source suffices
   - You don't need both
   - One source is "backup" for the other

2. Sources AGREE
   - They point to the same answer
   - No contradiction possible

3. Information is SHARED
   - The same bits are in both sources
   - Overlap, not combination

4. Robust to loss
   - Lose one source? Still have the answer
   - This is why redundancy exists in nature (backup systems)
```

### 2.4 Why Redundancy Matters

Redundancy indicates **agreement** and **certainty**. When multiple independent sources all point to the same answer, you can be confident.

In signal processing: Redundancy means the signal is **overdetermined** - you have more information than strictly necessary.

---

## 3. Synergy Explained

### 3.1 Definition

**Synergy** is information that **neither source alone** can provide, but **together they can**. The information only exists in the combination.

### 3.2 The XOR Example (Canonical)

This is the standard example used in the PID literature to demonstrate pure synergy:

```
EXAMPLE: XOR - Pure Synergy
───────────────────────────

Source 1 (S₁):  A random bit (0 or 1)
Source 2 (S₂):  Another random bit (0 or 1)
Target (T):     S₁ XOR S₂ (exclusive or)

XOR truth table:
  S₁=0, S₂=0  →  T=0
  S₁=0, S₂=1  →  T=1
  S₁=1, S₂=0  →  T=1
  S₁=1, S₂=1  →  T=0

Now try to predict T:

Ask S₁ alone: "I'm 0"
  → T could be 0 (if S₂=0) or 1 (if S₂=1)
  → S₁ tells you NOTHING about T
  → I(S₁; T) = 0 bits

Ask S₂ alone: "I'm 1"
  → T could be 0 (if S₁=1) or 1 (if S₁=0)
  → S₂ tells you NOTHING about T
  → I(S₂; T) = 0 bits

Ask BOTH: "S₁=0, S₂=1"
  → T = 0 XOR 1 = 1
  → Now you know EXACTLY what T is
  → I(S₁, S₂; T) = 1 bit

WHERE DID THAT 1 BIT COME FROM?

Neither source contributed any information alone.
But together they contribute 1 bit.
That bit is SYNERGY - it exists only in the combination.
```

### 3.3 Why XOR is the Canonical Example

The XOR function is special because:
- Each input alone is **completely uninformative** about the output
- Both inputs together are **completely informative** about the output
- There is **zero redundancy** (inputs don't share information)
- There is **zero unique information** (neither input alone helps)
- ALL information is **synergistic**

This makes XOR a "pure" example of synergy.

### 3.4 More Intuitive Examples

```
EXAMPLE: Recipe Synergy
───────────────────────

Target:     What dish am I making?
Source 1:   Flour
Source 2:   Eggs

Flour alone: Could be bread, cake, pasta, cookies...
Eggs alone:  Could be omelet, cake, custard, pasta...

Flour + Eggs: Now we're narrowing down (pasta? cake?)

Add more sources (sugar, butter, vanilla):
Now it's clearly: CAKE

The answer emerges from COMBINATION.
No single ingredient tells you the dish.
```

```
EXAMPLE: Context Synergy (Language)
───────────────────────────────────

Target:     What does "bank" mean?
Source 1:   The word "bank"
Source 2:   The word "river" or "money"

"Bank" alone: Ambiguous (financial? riverside?)

"River bank": Now you know - it's the land beside water
"Money bank": Now you know - it's a financial institution

The meaning EMERGES from combination.
This is why context matters in language.
```

```
EXAMPLE: Triangulation Synergy
──────────────────────────────

Target:     Where is the hidden object?
Source 1:   "It's north of the tree"
Source 2:   "It's east of the rock"

Source 1 alone: Infinite possibilities (anywhere north of tree)
Source 2 alone: Infinite possibilities (anywhere east of rock)

BOTH together: The intersection! One specific location.

Position information emerged from combination.
```

### 3.5 Key Properties of Synergy

```
SYNERGY PROPERTIES:
───────────────────

1. BOTH sources required
   - Neither alone helps
   - Must have the combination

2. Information EMERGES
   - Not "in" either source
   - Created by relationship between sources

3. Cannot be decomposed
   - Can't say "this part came from S₁, that from S₂"
   - It's genuinely holistic

4. Vulnerable to loss
   - Lose one source? Lose ALL synergistic information
   - No graceful degradation

5. Often indicates STRUCTURE
   - XOR is a specific relationship
   - Synergy reveals that relationship exists
```

### 3.6 Why Synergy Matters

Synergy indicates **structure** and **relationship**. When information only exists in combination, it means there's a non-trivial relationship between the sources.

In computation: Synergy corresponds to **actual computation** - combining inputs to produce outputs that neither input alone determines.

---

## 4. The Complete PID Decomposition

### 4.1 The Four Atoms

Williams and Beer (2010) showed that for two sources, information decomposes into exactly four parts:

```
PID DECOMPOSITION (Two Sources)
───────────────────────────────

Total information: I(S₁, S₂ ; T)

Decomposes into:

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  I(S₁, S₂ ; T) = I_red + I_uni(S₁) + I_uni(S₂) + I_syn                │
│                                                                         │
│  Where:                                                                 │
│                                                                         │
│  I_red      = REDUNDANCY                                               │
│               Information both sources share about T                   │
│               Either source alone provides this                        │
│                                                                         │
│  I_uni(S₁)  = UNIQUE information from S₁                              │
│               Information only S₁ provides                             │
│               S₂ cannot provide this                                   │
│                                                                         │
│  I_uni(S₂)  = UNIQUE information from S₂                              │
│               Information only S₂ provides                             │
│               S₁ cannot provide this                                   │
│                                                                         │
│  I_syn      = SYNERGY                                                  │
│               Information requiring both sources                       │
│               Neither alone provides this                              │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Visual Representation

```
VENN DIAGRAM INTUITION (imperfect but helpful)
──────────────────────────────────────────────

        ┌───────────────────────────────────┐
        │                                   │
        │    ┌─────────┐   ┌─────────┐     │
        │    │         │   │         │     │
        │    │  Uni(1) ├───┤  Uni(2) │     │
        │    │         │RED│         │     │
        │    └────┬────┘   └────┬────┘     │
        │         │             │          │
        │         └──────┬──────┘          │
        │                │                 │
        │           ┌────┴────┐            │
        │           │ SYNERGY │            │
        │           │ (below) │            │
        │           └─────────┘            │
        │                                   │
        └───────────────────────────────────┘

Note: Synergy doesn't fit in a Venn diagram because it's
NOT overlap - it's information that exists ONLY in combination.
The diagram is a rough intuition, not exact.
```

### 4.3 Examples with All Four Atoms

```
EXAMPLE: Weather Prediction
───────────────────────────

Target: Will it rain tomorrow?

S₁ = Barometric pressure
S₂ = Humidity

REDUNDANCY:
  Both low pressure and high humidity indicate rain
  Either alone provides SOME rain prediction
  → The overlap in what they tell you

UNIQUE(S₁):
  Pressure patterns indicate storm systems
  Humidity doesn't capture this
  → Storm-specific information

UNIQUE(S₂):
  Local humidity indicates local conditions
  Pressure might miss local variations
  → Location-specific information

SYNERGY:
  Pressure TREND + humidity CHANGE together
  indicate something neither alone shows
  → Complex weather pattern information
```

---

## 5. Theory and Mathematical Foundation

### 5.1 Information-Theoretic Basis

PID is built on Shannon's mutual information:

```
MUTUAL INFORMATION
──────────────────

I(X; Y) = H(X) - H(X|Y)
        = H(Y) - H(Y|X)
        = H(X) + H(Y) - H(X,Y)

Where H is Shannon entropy:
  H(X) = -Σ p(x) log p(x)

Mutual information measures: How much does knowing X
reduce uncertainty about Y (and vice versa)?
```

### 5.2 The PID Problem

Traditional information theory gives you:
- I(S₁; T) - Information from source 1 about target
- I(S₂; T) - Information from source 2 about target
- I(S₁, S₂; T) - Information from both sources about target

But it doesn't tell you HOW that information is distributed. PID solves this.

### 5.3 Williams and Beer's Approach

Williams and Beer (2010) defined redundancy using the concept of **specificity**:

```
REDUNDANCY DEFINITION (Williams & Beer)
───────────────────────────────────────

I_red(S₁, S₂ → T) = min{ I(S₁; T), I(S₂; T) }

Intuition: Redundancy is bounded by the LESS informative source.
If S₁ provides 3 bits and S₂ provides 5 bits,
redundancy can be at most 3 bits (the minimum).

This is called I_min (minimum mutual information).
```

Once redundancy is defined, the other atoms follow:

```
DERIVING OTHER ATOMS
────────────────────

Given I_red, we can compute:

I_uni(S₁) = I(S₁; T) - I_red
           (what S₁ provides beyond redundancy)

I_uni(S₂) = I(S₂; T) - I_red
           (what S₂ provides beyond redundancy)

I_syn = I(S₁, S₂; T) - I(S₁; T) - I(S₂; T) + I_red
       (the "extra" information from combination)

These must sum correctly:
I(S₁, S₂; T) = I_red + I_uni(S₁) + I_uni(S₂) + I_syn  ✓
```

### 5.4 Alternative Measures

Several alternative redundancy measures have been proposed:

| Measure | Authors | Key Idea |
|---------|---------|----------|
| I_min | Williams & Beer (2010) | Minimum mutual information |
| I_broja | Bertschinger et al. (2014) | Maximum entropy approach |
| I_ccs | Ince (2017) | Common change in surprisal |
| I_sx | Griffith & Koch (2014) | Shared exclusions |

These give different values in some cases but agree on key examples like XOR.

### 5.5 Mathematical Properties

```
PID PROPERTIES
──────────────

1. NON-NEGATIVITY
   All atoms are ≥ 0
   (information cannot be negative)

2. CONSISTENCY
   Atoms sum to total mutual information
   I_red + I_uni(S₁) + I_uni(S₂) + I_syn = I(S₁, S₂; T)

3. MONOTONICITY
   Adding sources cannot decrease information
   I(S₁, S₂, S₃; T) ≥ I(S₁, S₂; T)

4. CHAIN RULE COMPATIBILITY
   PID is consistent with information chain rules
```

---

## 6. General Applications

### 6.1 Neuroscience

PID is widely used to analyze neural coding:

```
NEURAL CODING APPLICATION
─────────────────────────

Question: How do neurons encode stimuli?

Setup:
  S₁ = Firing rate of neuron 1
  S₂ = Firing rate of neuron 2
  T  = Stimulus (e.g., orientation of a bar)

Analysis:
  High redundancy → Neurons encode same feature (backup)
  High synergy → Neurons encode feature TOGETHER (binding)
  High unique → Neurons specialize (division of labor)

Finding:
  Visual cortex shows high synergy at binding sites
  Motor cortex shows high redundancy (robust control)
```

### 6.2 Genetics

```
GENE REGULATORY NETWORKS
────────────────────────

Question: How do genes interact to control traits?

Setup:
  S₁ = Expression of gene A
  S₂ = Expression of gene B
  T  = Phenotype (observable trait)

Analysis:
  High synergy → Genes interact (epistasis)
  High redundancy → Genes are backups (robustness)
  High unique → Genes have distinct roles
```

### 6.3 Machine Learning

```
FEATURE ANALYSIS
────────────────

Question: How do features contribute to prediction?

Setup:
  S₁ = Feature 1
  S₂ = Feature 2
  T  = Label to predict

Analysis:
  High redundancy → Features are correlated (may prune one)
  High synergy → Features interact (need both)
  High unique → Features capture different aspects

Used for:
  - Feature selection
  - Model interpretability
  - Understanding learned representations
```

### 6.4 Communication Systems

```
CHANNEL CODING
──────────────

Question: How to encode messages reliably?

Redundancy in coding:
  - Error correction codes ADD redundancy
  - If one bit is corrupted, others recover it
  - Robustness vs efficiency tradeoff

Synergy in coding:
  - Encryption creates synergy
  - Need ALL of the key to decrypt
  - Security through combination
```

---

## 7. Synergy and Redundancy in AKIRA

### 7.1 The Band Structure

AKIRA's 7+1 spectral bands can be analyzed through PID:

```
AKIRA BAND DECOMPOSITION
────────────────────────

Sources: 8 bands (Band 0, 1, 2, 3, 4, 5, 6, Temporal)
Target:  Correct prediction

Question: How is prediction information distributed?

LOW-FREQUENCY BANDS (0, 1, 2):
  - Carry coarse structure
  - High REDUNDANCY with each other
  - Any can provide general shape

HIGH-FREQUENCY BANDS (4, 5, 6):
  - Carry fine details
  - More UNIQUE information
  - Each captures different details

CROSS-BAND:
  - Structure + detail = full picture
  - This combination is SYNERGISTIC
  - Need both coarse and fine for accurate prediction

TEMPORAL BAND:
  - Time dimension is ORTHOGONAL to spatial
  - Creates synergy with ALL spatial bands
  - Temporal + spatial = motion prediction
```

### 7.2 States of the Belief System

AKIRA's belief state oscillates between high-synergy and high-redundancy:

```
THE SYNERGY-REDUNDANCY SPECTRUM
───────────────────────────────

HIGH SYNERGY STATE (Before Collapse):
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  • Band 0 alone: Cannot predict target                             │
│  • Band 1 alone: Cannot predict target                             │
│  • ...                                                              │
│  • ALL bands together: CAN predict target                          │
│                                                                     │
│  Information is DISTRIBUTED                                         │
│  Bands must COOPERATE                                               │
│  Uncertainty is HIGH (many possibilities)                           │
│  Entropy is HIGH                                                    │
│                                                                     │
│  This is the SUPERPOSITION phase                                    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

HIGH REDUNDANCY STATE (After Collapse):
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  • Band 0 alone: CAN predict target                                │
│  • Band 1 alone: CAN predict target                                │
│  • ...                                                              │
│  • ANY band: CAN predict target                                    │
│                                                                     │
│  Information is SHARED                                              │
│  Bands AGREE                                                        │
│  Uncertainty is LOW (one answer)                                    │
│  Entropy is LOW                                                     │
│                                                                     │
│  This is the MOLECULE phase                                         │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 7.3 Collapse as Synergy-to-Redundancy Transition

```
THE COLLAPSE PROCESS
────────────────────

BEFORE:
  Multiple hypotheses compete
  Each band "votes" for different possibilities
  Information distributed (high synergy)
  
  Analogy: A jury deliberating - each juror has partial view

DURING COLLAPSE:
  Evidence accumulates
  Hypotheses eliminated
  Bands begin to AGREE
  Synergy converts to redundancy
  
  Analogy: Jury reaching consensus

AFTER:
  Single prediction selected
  All bands point to same answer
  Information concentrated (high redundancy)
  
  Analogy: Unanimous verdict
```

### 7.4 Why This Matters for AKIRA

```
FUNCTIONAL SIGNIFICANCE
───────────────────────

1. HIGH SYNERGY = UNCERTAINTY
   When bands must cooperate to predict,
   the system is "uncertain" - multiple possibilities
   
   This is appropriate for:
   - Ambiguous inputs
   - Multiple plausible futures
   - Exploration phase

2. HIGH REDUNDANCY = CERTAINTY
   When any band can predict,
   the system is "certain" - single answer
   
   This is appropriate for:
   - Clear inputs
   - Single future
   - Exploitation/action phase

3. THE TRANSITION IS INFORMATION PROCESSING
   Going from synergy to redundancy IS the computation
   The system "figures out" the answer
   This is what collapse accomplishes
```

---

## 8. How This Informs AKIRA's Theoretical Foundations

### 8.1 Inductive Foundation: Why Bands Should Exist

```
THEORETICAL JUSTIFICATION FOR BANDS
───────────────────────────────────

Question: Why have multiple bands at all?

PID Answer:
  - Different frequency bands carry different information
  - Low bands: Redundant (robust structure)
  - High bands: Unique (specific details)
  - Cross-band: Synergistic (combined understanding)

If we used ONE band:
  - No redundancy (fragile to noise)
  - No synergy (can't combine scales)
  - Limited representation

Multiple bands enable:
  - Robust core structure (redundancy)
  - Rich detail (unique)
  - Scale combination (synergy)
```

### 8.2 Inductive Foundation: Why Collapse Should Occur

```
THEORETICAL JUSTIFICATION FOR COLLAPSE
──────────────────────────────────────

Question: Why must belief "collapse" at all?

PID Answer:
  - Synergistic information cannot drive action alone
  - To act, you need a definite prediction
  - Definite prediction = high redundancy (bands agree)

High synergy problem:
  - "I need both X and Y to decide"
  - But which X? Which Y?
  - Infinite regression of combinations

High redundancy solution:
  - "Any band says the same thing"
  - Clear answer, no combination needed
  - Actionable prediction

Collapse is necessary:
  Synergy is useful for REPRESENTING uncertainty
  Redundancy is useful for ACTING on certainty
  Collapse converts representation → action
```

### 8.3 Inductive Foundation: Measuring Belief State

```
SYNERGY/REDUNDANCY AS OBSERVABLE
────────────────────────────────

We can MEASURE the belief state:

Compute for each band pair:
  I_red(Band_i, Band_j → Target)
  I_syn(Band_i, Band_j → Target)

Aggregate:
  Total_Synergy = Σ I_syn(i,j)
  Total_Redundancy = Σ I_red(i,j)

Ratio tells us belief state:
  Syn/Red high → Uncertain (pre-collapse)
  Syn/Red low → Certain (post-collapse)

This is COMPUTABLE and OBSERVABLE
Not metaphor - actual measurement
```

### 8.4 Connection to Other AKIRA Concepts

```
SYNERGY/REDUNDANCY CONNECTS TO:
───────────────────────────────

ENTROPY:
  High synergy ↔ High entropy (many possibilities)
  High redundancy ↔ Low entropy (few possibilities)
  See: TERMINOLOGY.md Section 5

SUPERPOSITION/CRYSTALLIZED:
  Superposition ↔ High synergy (distributed)
  Crystallized ↔ High redundancy (concentrated, IRREDUCIBLE)
  See: TERMINOLOGY.md Section 6

ACTION QUANTA:
  AQ crystallize when redundancy dominates
  Before that, they're "potential" in synergistic state
  See: TERMINOLOGY.md Section 7

WORMHOLE ATTENTION:
  Wormholes enable synergy across distant bands
  Without them, only adjacent bands could combine
  See: architecture_theoretical/SPECTRAL_BELIEF_MACHINE.md
```

---

## 9. References

### 9.1 Original PID Literature

1. Williams, P.L., & Beer, R.D. (2010). *Nonnegative Decomposition of Multivariate Information.* arXiv:1004.2515.
   - The foundational paper defining PID
   - Introduces redundancy, unique, synergy decomposition
   - Defines I_min measure

2. Griffith, V., & Koch, C. (2014). *Quantifying Synergistic Mutual Information.* arXiv:1205.4265.
   - Alternative synergy measure
   - Applications to neuroscience

3. Bertschinger, N., Rauh, J., Olbrich, E., Jost, J., & Ay, N. (2014). *Quantifying Unique Information.* Entropy, 16(4), 2161-2183.
   - BROJA measure of redundancy
   - Maximum entropy approach

4. Ince, R.A.A. (2017). *Measuring Multivariate Redundant Information with Pointwise Common Change in Surprisal.* Entropy, 19(7), 318.
   - CCS measure
   - Pointwise information approach

### 9.2 Applications

5. Timme, N., Alford, W., Flecker, B., & Beggs, J.M. (2014). *Synergy, redundancy, and multivariate information measures: an experimentalist's perspective.* Journal of Computational Neuroscience, 36(2), 119-140.
   - Practical guide to computing PID
   - Neuroscience applications

6. Mediano, P.A.M., et al. (2025). *Toward a unified taxonomy of information dynamics via Integrated Information Decomposition.* PNAS 122(39).
   - Extends PID to temporal dynamics
   - ΦID framework

### 9.3 AKIRA Internal Documents

7. `AKIRA/foundations/TERMINOLOGY.md`
   - Complete terminology reference
   - Sections 2, 4, 5 relate to synergy/redundancy

8. `AKIRA/architecture_theoretical/EVIDENCE.md`
   - Evidence for AKIRA's theoretical foundations
   - Section 2.4 (MSE produces interference)

9. `AKIRA/architecture_theoretical/EVIDENCE_TO_COLLECT.md`
   - Experimental predictions
   - Section 6 (Superposition-Crystallized Duality)

10. `AKIRA/experiments/025_EXP_SYNERGY_REDUNDANCY_TRANSITION/`
    - Experiment design for measuring synergy-redundancy transitions
    - Empirical validation of theoretical framework

---

## Summary

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  SYNERGY AND REDUNDANCY: KEY TAKEAWAYS                                 │
│                                                                         │
│  ═══════════════════════════════════════════════════════════════════   │
│                                                                         │
│  REDUNDANCY = Information BOTH sources share                           │
│    • Either source alone can provide it                                │
│    • Sources AGREE                                                     │
│    • Robust to loss                                                    │
│    • Indicates CERTAINTY                                               │
│                                                                         │
│  SYNERGY = Information NEITHER source alone provides                   │
│    • Requires BOTH sources                                             │
│    • EMERGES from combination                                          │
│    • Vulnerable to loss                                                │
│    • Indicates STRUCTURE/RELATIONSHIP                                  │
│                                                                         │
│  ═══════════════════════════════════════════════════════════════════   │
│                                                                         │
│  IN AKIRA:                                                              │
│                                                                         │
│  High Synergy = Uncertainty (pre-collapse, superposition)              │
│  High Redundancy = Certainty (post-collapse, crystallized)             │
│  Collapse = Synergy → Redundancy transition                            │
│                                                                         │
│  This is not metaphor - it's measurable information theory.            │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

*This document is part of AKIRA's terminology foundations. For the complete terminology framework, see `AKIRA/foundations/TERMINOLOGY.md`.*

