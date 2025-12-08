# Collapse Mechanisms: Decision Trees for Uncertainty Resolution

A technical framework for implementing collapse—the transition from uncertainty to certainty—using decision tree structures that combine reactive triggers with knowledge-informed selection.

---

## Table of Contents

1. [Introduction: Collapse as Decision](#1-introduction-collapse-as-decision)
   - 1.3 [Collapse = Belief State Dynamics (POMDP)](#13-collapse--belief-state-dynamics-pomdp)
2. [The Decision Tree Structure](#2-the-decision-tree-structure)
3. [Reactive Triggers vs Knowledge Selection](#3-reactive-triggers-vs-knowledge-selection)
4. [Types of Collapse](#4-types-of-collapse)
5. [The Collapse Decision Tree](#5-the-collapse-decision-tree)
6. [Branch Generation (Accumulation)](#6-branch-generation-accumulation)
7. [Branch Competition (Criticality)](#7-branch-competition-criticality)
8. [Branch Selection (Collapse)](#8-branch-selection-collapse)
9. [Hierarchical Collapse](#9-hierarchical-collapse)
10. [Implementation Patterns](#10-implementation-patterns)
11. [Integration with Spectral Attention](#11-integration-with-spectral-attention)
12. [Design Principles](#12-design-principles)

---

## 1. Introduction: Collapse as Decision

### 1.1 Recap: What Is Collapse?

```
FROM COLLAPSE_GENERALIZATION.md:

Collapse is the sudden transition from uncertainty to certainty:

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  PHASE 1: ACCUMULATION                                          │
│  Multiple hypotheses coexist                                    │
│  Uncertainty spreads across possibilities                       │
│                                                                 │
│       ░░░░░░░░░░░░░░░░░░░░                                     │
│      ░░░░░░████████████░░░░░░                                  │
│     ░░░░░████████████████░░░░░                                 │
│      ░░░░░░████████████░░░░░░                                  │
│       ░░░░░░░░░░░░░░░░░░░░                                     │
│                                                                 │
│  PHASE 2: CRITICALITY                                           │
│  Hypotheses compete via interference                            │
│  Some reinforce, some cancel                                    │
│                                                                 │
│  PHASE 3: COLLAPSE                                              │
│  One hypothesis wins, others extinguish                        │
│                                                                 │
│           ████                                                 │
│         ████████                                               │
│           ████                                                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 Collapse as a Decision Problem

```
REFRAMING COLLAPSE:

Collapse is fundamentally a DECISION:

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  DECISION 1: Should we collapse NOW?                           │
│  ─────────────────────────────────────                          │
│  • Is uncertainty high enough to warrant resolution?           │
│  • Is evidence accumulated enough to decide?                   │
│  • Are we at the critical threshold?                           │
│                                                                 │
│  TYPE: REACTIVE (energy/threshold based)                       │
│                                                                 │
│  ────────────────────────────────────────────────────────────  │
│                                                                 │
│  DECISION 2: What should we collapse TO?                       │
│  ─────────────────────────────────────────                      │
│  • Which hypothesis has the most support?                      │
│  • Which pattern best matches the evidence?                    │
│  • Which branch should survive?                                │
│                                                                 │
│  TYPE: KNOWLEDGE-INFORMED (evidence/geometry based)            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

This naturally maps to a DECISION TREE structure.
```

### 1.3 Collapse = Belief State Dynamics (POMDP)

```
FORMAL CONNECTION TO POMDP:

Collapse is BELIEF STATE DYNAMICS in a Partially Observable system.

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  POMDP COMPONENTS → OUR SYSTEM:                                │
│                                                                 │
│  State space S         → All possible next frames              │
│  Observation space O   → Current frame + history               │
│  Belief state b(s)     → Decision tree branches (hypotheses)   │
│  Belief update         → Evidence accumulation                  │
│  Belief collapse       → Branch pruning (our "collapse")       │
│                                                                 │
│  ════════════════════════════════════════════════════════════  │
│                                                                 │
│  KEY INSIGHT:                                                   │
│                                                                 │
│  The decision tree IS a discrete representation of b(s).       │
│  Each branch is a hypothesis with associated probability.      │
│  Collapse is b(s) → δ(s - s*) (concentrating on winner).      │
│                                                                 │
│  ════════════════════════════════════════════════════════════  │
│                                                                 │
│  BAYESIAN UPDATE = EVIDENCE ACCUMULATION:                      │
│                                                                 │
│  b'(s') ∝ O(o|s') × Σ_s T(s'|s,a) × b(s)                      │
│                                                                 │
│  Where:                                                         │
│  • O(o|s') = observation likelihood (manifold similarity)      │
│  • T(s'|s,a) = transition model (temporal dynamics)            │
│  • b(s) = prior (from history/manifold)                        │
│                                                                 │
│  This is exactly what our attention mechanisms compute.        │
│                                                                 │
│  See: POMDP_SIM.md for full formalization.                     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 1.4 Why Decision Trees?

```
DECISION TREES FOR COLLAPSE:

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  ADVANTAGES:                                                    │
│                                                                 │
│  1. EXPLICIT STRUCTURE                                          │
│     • Each branch is a hypothesis                              │
│     • Tree structure shows relationships                        │
│     • Clear lineage from root to leaf                          │
│                                                                 │
│  2. NATURAL PRUNING                                             │
│     • Collapse = pruning all but one branch                    │
│     • Partial collapse = soft pruning                          │
│     • Hierarchical collapse = level-by-level pruning           │
│                                                                 │
│  3. INTERPRETABLE                                               │
│     • Can trace why a decision was made                        │
│     • Can identify which branch won and why                    │
│     • Debuggable                                                │
│                                                                 │
│  4. ALIGNS WITH LIGHTNING ANALOGY                               │
│     • Stepped leaders = growing branches                       │
│     • Return stroke = collapse to single path                  │
│     • Natural metaphor                                          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. The Decision Tree Structure

### 2.1 Anatomy of a Collapse Decision Tree

```
COLLAPSE DECISION TREE:

                              ROOT
                          (current state)
                               │
              ┌────────────────┼────────────────┐
              │                │                │
              ▼                ▼                ▼
         ┌────────┐       ┌────────┐       ┌────────┐
         │ Hyp A  │       │ Hyp B  │       │ Hyp C  │
         │ p=0.3  │       │ p=0.5  │       │ p=0.2  │
         └────┬───┘       └────┬───┘       └────┬───┘
              │                │                │
         ┌────┴────┐      ┌────┴────┐      ┌────┴────┐
         │         │      │         │      │         │
         ▼         ▼      ▼         ▼      ▼         ▼
       A.1       A.2    B.1       B.2    C.1       C.2
      p=0.1     p=0.2  p=0.3     p=0.2  p=0.1     p=0.1


COMPONENTS:

ROOT:       The current uncertain state
            Contains all possibilities

BRANCHES:   Hypotheses about what the answer might be
            Each branch has an associated probability/weight

LEAVES:     Specific predictions/outcomes
            Most refined level of hypothesis

COLLAPSE:   Selecting one branch and pruning others
```

### 2.2 Branch Attributes

```
EACH BRANCH HAS ATTRIBUTES:

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  BRANCH = {                                                     │
│                                                                 │
│    hypothesis:     The specific prediction this branch makes   │
│                                                                 │
│    evidence:       Accumulated support for this hypothesis     │
│                   (from manifold similarity, past matches)     │
│                                                                 │
│    probability:    Estimated likelihood (softmax over evidence)│
│                                                                 │
│    energy:         Current "activation" level                  │
│                   (scalar, for reactive decisions)             │
│                                                                 │
│    parent:         Link to parent branch                       │
│                                                                 │
│    children:       Links to sub-hypotheses                     │
│                                                                 │
│    depth:          Level in the tree                           │
│                   (coarse = low, fine = high)                  │
│                                                                 │
│  }                                                              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2.3 Tree States

```
TREE STATE TRANSITIONS:

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  STATE 1: EXPANDING                                             │
│  ──────────────────                                             │
│  • New branches being created                                   │
│  • More hypotheses explored                                     │
│  • Tree is growing                                              │
│                                                                 │
│        ●                      ●                                │
│       ╱ ╲         →         ╱ │ ╲                              │
│      ●   ●                 ● ● ● ●                             │
│                                                                 │
│  ────────────────────────────────────────────────────────────  │
│                                                                 │
│  STATE 2: COMPETING                                             │
│  ──────────────────                                             │
│  • Branches accumulating evidence                              │
│  • Probabilities shifting                                       │
│  • Some branches strengthening, others weakening               │
│                                                                 │
│        ●                      ●                                │
│       ╱│╲          →        ╱ │ ╲                              │
│      ● ● ●                 █  ░  ▒     (varying weights)       │
│                                                                 │
│  ────────────────────────────────────────────────────────────  │
│                                                                 │
│  STATE 3: COLLAPSING                                            │
│  ───────────────────                                            │
│  • One branch selected                                          │
│  • Others pruned                                                 │
│  • Tree reduces to single path                                 │
│                                                                 │
│        ●                      ●                                │
│       ╱│╲          →          │                                │
│      █ ░ ▒                    █                                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Reactive Triggers vs Knowledge Selection

### 3.1 The Dual Nature of Collapse

```
COLLAPSE REQUIRES BOTH MODES:

FROM KNOWLEDGE_AND_REACTIVITY.md:

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  REACTIVE COMPONENT:                                            │
│  ───────────────────                                            │
│  Answers: WHEN to collapse                                      │
│                                                                 │
│  • Monitors energy levels (SNR, gradient norms)                │
│  • Compares to threshold                                       │
│  • Triggers collapse when conditions met                       │
│  • Fast, automatic, no deliberation                            │
│                                                                 │
│  Implementation:                                                │
│  if SNR > threshold:                                            │
│      trigger_collapse()                                         │
│                                                                 │
│  ────────────────────────────────────────────────────────────  │
│                                                                 │
│  KNOWLEDGE-INFORMED COMPONENT:                                  │
│  ─────────────────────────────                                  │
│  Answers: WHAT to collapse to                                   │
│                                                                 │
│  • Evaluates evidence for each branch                          │
│  • Computes geometric similarity to manifold                   │
│  • Selects branch with most support                            │
│  • Slower, deliberate, evidence-based                          │
│                                                                 │
│  Implementation:                                                │
│  winner = argmax([manifold_similarity(branch) for branch in tree])
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 The Interaction

```
HOW REACTIVE AND KNOWLEDGE INTERACT:

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│                    CONTINUOUS LOOP                              │
│                                                                 │
│     ┌──────────────────────────────────────────────────────┐   │
│     │                                                      │   │
│     ▼                                                      │   │
│  ┌──────────────┐                                          │   │
│  │   OBSERVE    │  ← Measure current state                 │   │
│  │   (signal)   │                                          │   │
│  └──────┬───────┘                                          │   │
│         │                                                  │   │
│         ▼                                                  │   │
│  ┌──────────────┐     No                                   │   │
│  │   REACTIVE   │ ──────────────────────────────────────────   │
│  │  (threshold) │  Collapse not triggered                      │
│  └──────┬───────┘                                              │
│         │ Yes                                                  │
│         ▼                                                      │
│  ┌──────────────┐                                              │
│  │  KNOWLEDGE   │  ← Consult manifold                         │
│  │ (selection)  │                                              │
│  └──────┬───────┘                                              │
│         │                                                      │
│         ▼                                                      │
│  ┌──────────────┐                                              │
│  │   COLLAPSE   │  ← Prune tree to winner                     │
│  │   (execute)  │                                              │
│  └──────────────┘                                              │
│                                                                 │
│  REACTIVE GATES, KNOWLEDGE FILLS                               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 3.3 Thresholds and Evidence

```
WHAT EACH MODE PROVIDES:

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  REACTIVE THRESHOLDS:                                           │
│  ────────────────────                                           │
│                                                                 │
│  1. SNR_threshold                                              │
│     Signal-to-noise ratio for collapse trigger                 │
│     Higher → collapse happens later, more certainty needed    │
│                                                                 │
│  2. energy_min                                                 │
│     Minimum energy for branch survival                         │
│     Below this → branch is pruned automatically               │
│                                                                 │
│  3. divergence_max                                             │
│     Maximum allowed divergence between branches                │
│     Exceeded → force collapse to resolve ambiguity            │
│                                                                 │
│  4. time_limit                                                 │
│     Maximum time before forced collapse                        │
│     Prevents indefinite accumulation                           │
│                                                                 │
│  ────────────────────────────────────────────────────────────  │
│                                                                 │
│  KNOWLEDGE EVIDENCE:                                            │
│  ───────────────────                                            │
│                                                                 │
│  1. manifold_similarity                                        │
│     How well branch matches stored patterns                   │
│     Higher → branch is more likely correct                    │
│                                                                 │
│  2. historical_match                                           │
│     How often similar states led to this branch               │
│     Higher → branch has track record                          │
│                                                                 │
│  3. coherence                                                  │
│     Internal consistency of branch hypothesis                  │
│     Higher → branch is well-formed                            │
│                                                                 │
│  4. interference_score                                         │
│     Accumulated constructive interference                      │
│     Higher → branch reinforced by multiple examples           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. Types of Collapse

### 4.1 Hard Collapse

```
HARD COLLAPSE: Winner-Take-All

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  DEFINITION:                                                    │
│  One branch selected with 100% weight, all others eliminated   │
│                                                                 │
│  BEFORE:                          AFTER:                       │
│                                                                 │
│      ●                               ●                         │
│    ╱ │ ╲                             │                         │
│   █  ▒  ░                            █                         │
│  p=.5 .3 .2                        p=1.0                       │
│                                                                 │
│  ────────────────────────────────────────────────────────────  │
│                                                                 │
│  MECHANISM:                                                     │
│                                                                 │
│  winner = argmax(evidence)                                      │
│  for branch in tree:                                            │
│      if branch != winner:                                       │
│          prune(branch)                                          │
│                                                                 │
│  ────────────────────────────────────────────────────────────  │
│                                                                 │
│  USE CASES:                                                     │
│  • Final prediction (must commit to one answer)               │
│  • Resource-constrained (can only pursue one path)            │
│  • Binary decisions (yes/no, left/right)                      │
│                                                                 │
│  ADVANTAGES:                                                    │
│  • Decisive                                                     │
│  • Efficient (single path)                                     │
│  • Clear commitment                                             │
│                                                                 │
│  RISKS:                                                         │
│  • May lose information if wrong branch selected              │
│  • Irreversible                                                 │
│  • All-or-nothing                                               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 Soft Collapse

```
SOFT COLLAPSE: Winner-Take-Most

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  DEFINITION:                                                    │
│  Winner gets boosted weight, others reduced but not eliminated │
│                                                                 │
│  BEFORE:                          AFTER:                       │
│                                                                 │
│      ●                               ●                         │
│    ╱ │ ╲                           ╱ │ ╲                       │
│   █  ▒  ░                         ██ ░  ·                      │
│  p=.5 .3 .2                      p=.8 .15 .05                  │
│                                                                 │
│  ────────────────────────────────────────────────────────────  │
│                                                                 │
│  MECHANISM:                                                     │
│                                                                 │
│  # Sharpen the distribution                                    │
│  evidence_sharpened = evidence ** temperature  # temp > 1      │
│  probabilities = softmax(evidence_sharpened)                   │
│                                                                 │
│  # Or explicit winner boosting                                 │
│  winner = argmax(evidence)                                      │
│  for branch in tree:                                            │
│      if branch == winner:                                       │
│          branch.weight *= boost_factor                          │
│      else:                                                      │
│          branch.weight *= decay_factor                          │
│                                                                 │
│  ────────────────────────────────────────────────────────────  │
│                                                                 │
│  USE CASES:                                                     │
│  • Uncertain situations (keep backup hypotheses)               │
│  • Intermediate states (not yet ready for hard collapse)      │
│  • Attention mechanisms (soft weighting over options)         │
│                                                                 │
│  ADVANTAGES:                                                    │
│  • Reversible (can recover suppressed branches)               │
│  • Encodes uncertainty                                          │
│  • Gradual commitment                                           │
│                                                                 │
│  RISKS:                                                         │
│  • Never fully commits (indecision)                            │
│  • Computationally expensive (maintains all branches)         │
│  • Can oscillate                                                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 4.3 Staged Collapse

```
STAGED COLLAPSE: Hierarchical Pruning

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  DEFINITION:                                                    │
│  Collapse happens level-by-level, coarse before fine           │
│                                                                 │
│  STAGE 1: Collapse coarse level                                │
│                                                                 │
│      ●                               ●                         │
│    ╱ │ ╲          →                  │                         │
│   A  B  C                            B                         │
│  ╱╲ ╱╲ ╱╲                           ╱ ╲                        │
│                                    B1  B2                      │
│                                                                 │
│  STAGE 2: Collapse fine level                                  │
│                                                                 │
│      ●                               ●                         │
│      │              →                │                         │
│      B                               B                         │
│     ╱ ╲                              │                         │
│    B1  B2                            B1                        │
│                                                                 │
│  ────────────────────────────────────────────────────────────  │
│                                                                 │
│  MECHANISM:                                                     │
│                                                                 │
│  for level in range(max_depth):                                 │
│      if should_collapse_level(level):                          │
│          winner = select_winner_at_level(level)                │
│          prune_siblings(winner)                                 │
│          # Children of winner remain for next level            │
│                                                                 │
│  ────────────────────────────────────────────────────────────  │
│                                                                 │
│  USE CASES:                                                     │
│  • Multi-scale decisions (coarse structure, then fine details)│
│  • Hierarchical classification (kingdom → species)            │
│  • Spectral bands (low-freq first, then high-freq)            │
│                                                                 │
│  ADVANTAGES:                                                    │
│  • Matches natural hierarchy                                    │
│  • Early commitment to structure                               │
│  • Details resolved within committed context                   │
│                                                                 │
│  RISKS:                                                         │
│  • Early wrong choice propagates                               │
│  • Can't recover from coarse-level mistakes                    │
│  • Requires well-defined hierarchy                              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 4.4 Comparison Table

```
COLLAPSE TYPE COMPARISON:

┌──────────────┬──────────────┬──────────────┬──────────────────┐
│ Property     │ Hard         │ Soft         │ Staged           │
├──────────────┼──────────────┼──────────────┼──────────────────┤
│ Speed        │ Fast         │ Gradual      │ Multi-step       │
│ Commitment   │ Total        │ Partial      │ Level-by-level   │
│ Reversibility│ None         │ Possible     │ Per-level        │
│ Uncertainty  │ Eliminated   │ Preserved    │ Hierarchical     │
│ Branches     │ 1 survives   │ All weighted │ Subtree survives │
│ Memory       │ Minimal      │ High         │ Medium           │
│ Use case     │ Final output │ Attention    │ Multi-scale      │
└──────────────┴──────────────┴──────────────┴──────────────────┘
```

---

## 5. The Collapse Decision Tree

### 5.1 The Meta-Decision

```
DECIDING HOW TO COLLAPSE:

Before collapsing branches, we must decide:
• WHEN to collapse (reactive)
• WHAT TYPE of collapse (could be knowledge-informed)
• WHAT to collapse to (knowledge-informed)

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│                    META-DECISION TREE                           │
│                                                                 │
│                       [Current State]                           │
│                             │                                   │
│              ┌──────────────┴──────────────┐                   │
│              │                             │                   │
│              ▼                             │                   │
│       [SNR > threshold?]                   │                   │
│         (REACTIVE)                         │                   │
│        ╱           ╲                       │                   │
│       No            Yes                    │                   │
│       │              │                     │                   │
│       ▼              ▼                     │                   │
│  [Continue       [What type?]              │                   │
│   accumulating]   (KNOWLEDGE)              │                   │
│       │          ╱    │    ╲               │                   │
│       │       Hard   Soft   Staged         │                   │
│       │         │     │       │            │                   │
│       │         ▼     ▼       ▼            │                   │
│       │      [Select winner]               │                   │
│       │       (KNOWLEDGE)                  │                   │
│       │            │                       │                   │
│       └────────────┴───────────────────────┘                   │
│                    │                                            │
│                    ▼                                            │
│               [New State]                                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 Collapse Decision Factors

```
FACTORS DETERMINING COLLAPSE TYPE:

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  FACTOR                  HARD          SOFT          STAGED    │
│  ──────                  ────          ────          ──────    │
│                                                                 │
│  Confidence level        High          Low           Medium    │
│  Time pressure           High          Low           Medium    │
│  Resource constraint     High          Low           Medium    │
│  Reversibility need      Low           High          Medium    │
│  Hierarchy present       No            No            Yes       │
│  Output type             Discrete      Continuous    Multi-scale│
│                                                                 │
│  ────────────────────────────────────────────────────────────  │
│                                                                 │
│  DECISION LOGIC:                                                │
│                                                                 │
│  if confidence > high_threshold:                               │
│      return HARD                                               │
│  elif hierarchy_present:                                       │
│      return STAGED                                             │
│  else:                                                          │
│      return SOFT                                                │
│                                                                 │
│  The type decision can itself be REACTIVE (threshold-based)    │
│  or KNOWLEDGE-INFORMED (context-dependent).                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 6. Branch Generation (Accumulation)

### 6.1 How Branches Are Created

```
BRANCH GENERATION:

During accumulation, new branches (hypotheses) are created.

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  KNOWLEDGE-INFORMED BRANCH GENERATION:                          │
│  ────────────────────────────────────                           │
│                                                                 │
│  1. MANIFOLD QUERY                                              │
│     Query manifold for patterns similar to current state       │
│     Each match becomes a potential branch                      │
│                                                                 │
│     matches = manifold.query(current_state, top_k=N)           │
│     for match in matches:                                       │
│         create_branch(hypothesis=match)                         │
│                                                                 │
│  2. HISTORY LOOKUP                                              │
│     Check what happened in similar past situations             │
│     Past outcomes suggest possible branches                    │
│                                                                 │
│     past_outcomes = history.lookup(current_state)              │
│     for outcome in past_outcomes:                               │
│         create_branch(hypothesis=outcome)                       │
│                                                                 │
│  3. GENERATIVE SAMPLING                                         │
│     Sample from the generative model (hope)                    │
│     Samples become hypotheses                                   │
│                                                                 │
│     samples = model.sample(current_state, num_samples=N)       │
│     for sample in samples:                                      │
│         create_branch(hypothesis=sample)                        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 6.2 Branch Expansion Control

```
CONTROLLING BRANCH EXPANSION (REACTIVE):

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  EXPANSION IS GATED BY REACTIVE THRESHOLDS:                    │
│                                                                 │
│  1. MAX_BRANCHES                                                │
│     if num_branches >= MAX_BRANCHES:                           │
│         stop_expanding()                                        │
│                                                                 │
│  2. MIN_PROBABILITY                                             │
│     if new_branch.probability < MIN_PROBABILITY:               │
│         don't_create(new_branch)  # Too unlikely              │
│                                                                 │
│  3. DEPTH_LIMIT                                                 │
│     if new_branch.depth > MAX_DEPTH:                           │
│         don't_create(new_branch)  # Too detailed              │
│                                                                 │
│  4. COMPUTATIONAL_BUDGET                                        │
│     if compute_used >= BUDGET:                                  │
│         stop_expanding()  # Resource limit                     │
│                                                                 │
│  ────────────────────────────────────────────────────────────  │
│                                                                 │
│  KNOWLEDGE DETERMINES WHAT; REACTIVE DETERMINES HOW MANY       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 6.3 The Expansion Decision Tree

```
EXPANSION DECISION FLOW:

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│                    [Should expand?]                             │
│                          │                                      │
│         ┌────────────────┼────────────────┐                    │
│         │                │                │                    │
│         ▼                ▼                ▼                    │
│  [branches < max?]  [budget ok?]   [depth ok?]                │
│   (REACTIVE)        (REACTIVE)     (REACTIVE)                  │
│         │                │                │                    │
│         └────────────────┴────────────────┘                    │
│                          │                                      │
│                    All yes?                                     │
│                    ╱     ╲                                     │
│                  No       Yes                                   │
│                  │         │                                    │
│                  ▼         ▼                                    │
│             [Stop]    [What to expand?]                        │
│                        (KNOWLEDGE)                              │
│                             │                                   │
│                 ┌───────────┼───────────┐                      │
│                 │           │           │                      │
│                 ▼           ▼           ▼                      │
│            [Manifold]  [History]  [Generative]                 │
│             query       lookup      sample                     │
│                 │           │           │                      │
│                 └───────────┴───────────┘                      │
│                             │                                   │
│                    [Create branches]                            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 7. Branch Competition (Criticality)

### 7.1 Evidence Accumulation

```
HOW BRANCHES ACCUMULATE EVIDENCE:

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  EACH BRANCH ACCUMULATES EVIDENCE OVER TIME:                   │
│                                                                 │
│  branch.evidence += compute_evidence(branch, new_observation) │
│                                                                 │
│  EVIDENCE SOURCES (Knowledge-Informed):                        │
│                                                                 │
│  1. MANIFOLD SIMILARITY                                        │
│     How well does branch match stored patterns?                │
│     similarity = cosine(branch.embedding, manifold.patterns)  │
│                                                                 │
│  2. PREDICTION ACCURACY                                         │
│     How well did branch predict what happened?                 │
│     accuracy = 1 - error(branch.prediction, actual)           │
│                                                                 │
│  3. CONSTRUCTIVE INTERFERENCE                                   │
│     Is branch reinforced by similar hypotheses?                │
│     interference = sum(phase_aligned_contributions)            │
│                                                                 │
│  4. PRIOR PROBABILITY                                           │
│     How likely is this branch based on past experience?       │
│     prior = history.frequency(branch.hypothesis)               │
│                                                                 │
│  TOTAL EVIDENCE:                                                │
│  evidence = w₁×similarity + w₂×accuracy + w₃×interference + w₄×prior
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 7.2 Branch Interaction

```
HOW BRANCHES INTERACT (Interference):

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  CONSTRUCTIVE INTERFERENCE:                                     │
│  Similar branches reinforce each other                         │
│                                                                 │
│  Branch A: "Ring moving right"                                 │
│  Branch B: "Circular object moving right"                      │
│                                                                 │
│  These have overlapping evidence → both strengthen             │
│                                                                 │
│  if similarity(A, B) > threshold:                              │
│      A.evidence += transfer × B.evidence                       │
│      B.evidence += transfer × A.evidence                       │
│                                                                 │
│  ────────────────────────────────────────────────────────────  │
│                                                                 │
│  DESTRUCTIVE INTERFERENCE:                                      │
│  Contradictory branches weaken each other                      │
│                                                                 │
│  Branch A: "Ring moving RIGHT"                                 │
│  Branch C: "Ring moving LEFT"                                  │
│                                                                 │
│  These contradict → one must lose                              │
│                                                                 │
│  if contradiction(A, C):                                        │
│      # Evidence flows to stronger branch                       │
│      if A.evidence > C.evidence:                               │
│          A.evidence += steal × C.evidence                      │
│          C.evidence -= steal × C.evidence                      │
│                                                                 │
│  This creates winner-take-all dynamics at criticality          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 7.3 Monitoring Criticality

```
DETECTING CRITICALITY (Reactive Signals):

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  CRITICALITY INDICATORS:                                        │
│                                                                 │
│  1. LEADER GAP                                                  │
│     gap = evidence[1st] - evidence[2nd]                        │
│     Large gap → approaching collapse                           │
│                                                                 │
│     SNR = gap / std(all_evidence)                              │
│                                                                 │
│  2. EVIDENCE VARIANCE                                           │
│     variance = var([branch.evidence for branch in tree])      │
│     High variance → differentiation happening                  │
│                                                                 │
│  3. CONVERGENCE RATE                                            │
│     d_evidence/dt for top branch                               │
│     Accelerating → positive feedback starting                  │
│                                                                 │
│  4. BRANCH PRUNING RATE                                         │
│     How fast are weak branches dying?                          │
│     Increasing → system approaching collapse                   │
│                                                                 │
│  ────────────────────────────────────────────────────────────  │
│                                                                 │
│  CRITICALITY REACHED WHEN:                                      │
│                                                                 │
│  SNR > threshold  AND  d_evidence/dt > acceleration_threshold │
│                                                                 │
│  This is the REACTIVE trigger for collapse.                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 8. Branch Selection (Collapse)

### 8.1 Winner Selection Algorithm

```
SELECTING THE WINNER (Knowledge-Informed):

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  ALGORITHM: SELECT_WINNER                                       │
│                                                                 │
│  def select_winner(branches, manifold, history):               │
│      """                                                        │
│      Knowledge-informed winner selection.                      │
│      Uses geometric evidence and accumulated support.          │
│      """                                                        │
│      scores = []                                                │
│                                                                 │
│      for branch in branches:                                   │
│          # Accumulated evidence                                 │
│          evidence = branch.evidence                            │
│                                                                 │
│          # Manifold similarity (geometric)                     │
│          manifold_score = manifold.similarity(branch.hyp)     │
│                                                                 │
│          # Historical support                                   │
│          history_score = history.support(branch.hyp)          │
│                                                                 │
│          # Coherence (internal consistency)                    │
│          coherence = compute_coherence(branch)                 │
│                                                                 │
│          # Weighted combination                                 │
│          total = (w_e * evidence +                             │
│                   w_m * manifold_score +                       │
│                   w_h * history_score +                        │
│                   w_c * coherence)                              │
│                                                                 │
│          scores.append((branch, total))                        │
│                                                                 │
│      # Winner is highest scoring                                │
│      winner = max(scores, key=lambda x: x[1])[0]              │
│      return winner                                              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 8.2 Pruning Procedure

```
PRUNING NON-WINNERS:

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  HARD COLLAPSE PRUNING:                                         │
│                                                                 │
│  def prune_hard(tree, winner):                                  │
│      """Remove all branches except winner."""                  │
│      for branch in tree.branches:                              │
│          if branch != winner:                                   │
│              tree.remove(branch)  # Complete removal           │
│                                                                 │
│  ────────────────────────────────────────────────────────────  │
│                                                                 │
│  SOFT COLLAPSE PRUNING:                                         │
│                                                                 │
│  def prune_soft(tree, winner, boost=2.0, decay=0.5):          │
│      """Boost winner, decay others."""                         │
│      for branch in tree.branches:                              │
│          if branch == winner:                                   │
│              branch.weight *= boost                             │
│          else:                                                  │
│              branch.weight *= decay                             │
│              if branch.weight < MIN_WEIGHT:                    │
│                  tree.remove(branch)  # Too weak              │
│                                                                 │
│  ────────────────────────────────────────────────────────────  │
│                                                                 │
│  STAGED COLLAPSE PRUNING:                                       │
│                                                                 │
│  def prune_staged(tree, level):                                │
│      """Prune at specified level only."""                      │
│      winner = select_winner_at_level(tree, level)             │
│      for branch in tree.branches_at_level(level):             │
│          if branch != winner:                                   │
│              tree.remove_subtree(branch)                        │
│      # Winner's children remain for next level                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 8.3 Post-Collapse State

```
AFTER COLLAPSE:

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  THE TREE AFTER COLLAPSE:                                       │
│                                                                 │
│  HARD:                                                          │
│       ●               Single path: ROOT → Winner → Prediction │
│       │                                                         │
│       █                                                         │
│                                                                 │
│  SOFT:                                                          │
│       ●               Winner dominant, others suppressed       │
│     ╱ │ ╲             Can recover if new evidence arrives     │
│    ░  █  ·                                                     │
│                                                                 │
│  STAGED:                                                        │
│       ●               Subtree remains for finer decisions     │
│       │                                                         │
│       █                                                         │
│      ╱ ╲                                                       │
│     ▒   ▒                                                      │
│                                                                 │
│  ────────────────────────────────────────────────────────────  │
│                                                                 │
│  WHAT TO DO WITH ELIMINATED BRANCHES:                          │
│                                                                 │
│  Option 1: FORGET                                               │
│            Delete completely (save memory)                     │
│                                                                 │
│  Option 2: ARCHIVE                                              │
│            Store in history (may be useful later)             │
│                                                                 │
│  Option 3: MERGE                                                │
│            Combine information into winner                     │
│            Winner inherits aspects of losers                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 9. Hierarchical Collapse

### 9.1 Multi-Scale Decision Trees

```
HIERARCHICAL COLLAPSE FOR MULTI-SCALE PROBLEMS:

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  CONNECTION TO SPECTRAL BANDS:                                  │
│                                                                 │
│  Level 0 (DC):     WHAT exists at all                          │
│                    Coarsest level, collapse first              │
│                                                                 │
│  Level 1 (Low):    WHAT category                               │
│                    Major structure, collapse second            │
│                                                                 │
│  Level 2 (Mid):    WHAT features                               │
│                    Parts and relationships                      │
│                                                                 │
│  Level 3 (High):   WHERE exactly                               │
│                    Fine details, collapse last                  │
│                                                                 │
│  ────────────────────────────────────────────────────────────  │
│                                                                 │
│  TREE STRUCTURE:                                                │
│                                                                 │
│  Level 0:          ●─────────────────────────                  │
│                    │                                            │
│  Level 1:    ●─────┼─────●─────●                               │
│              │     │     │     │                               │
│  Level 2:  ●─┼─● ●─┼─● ●─┼─● ●─┼─●                            │
│            │ │ │ │ │ │ │ │ │ │ │ │                            │
│  Level 3:  · · · · · · · · · · · ·                             │
│                                                                 │
│  Collapse proceeds TOP-DOWN: Level 0 → 1 → 2 → 3              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 9.2 Cascading Collapse

```
CASCADING COLLAPSE ALGORITHM:

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  def cascading_collapse(tree):                                  │
│      """                                                        │
│      Collapse level-by-level, coarse to fine.                  │
│      Each level collapses when its SNR exceeds threshold.      │
│      """                                                        │
│      for level in range(tree.num_levels):                      │
│                                                                 │
│          # REACTIVE: Check if this level ready                 │
│          if not level_ready_to_collapse(tree, level):          │
│              continue  # Not yet                               │
│                                                                 │
│          # KNOWLEDGE: Select winner at this level              │
│          winner = select_winner_at_level(tree, level)          │
│                                                                 │
│          # Prune siblings of winner                            │
│          prune_siblings_at_level(tree, winner, level)          │
│                                                                 │
│          # Winner's children become active for next iteration │
│          # (they inherit the collapsed context)                │
│                                                                 │
│      return tree                                                │
│                                                                 │
│  ────────────────────────────────────────────────────────────  │
│                                                                 │
│  def level_ready_to_collapse(tree, level):                     │
│      """REACTIVE: Check energy-based threshold."""             │
│      branches = tree.branches_at_level(level)                  │
│      evidence = [b.evidence for b in branches]                 │
│      snr = compute_snr(evidence)                               │
│      return snr > THRESHOLD_FOR_LEVEL[level]                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 9.3 Context Inheritance

```
HOW CHILDREN INHERIT FROM COLLAPSED PARENTS:

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  BEFORE LEVEL-1 COLLAPSE:                                       │
│                                                                 │
│       ●  (Root: "Something is moving")                         │
│      ╱│╲                                                       │
│     A B C  (Level 1: "Ring", "Square", "Blob")                │
│    ╱│  │╲                                                      │
│   ···· ···  (Level 2: Position variants)                       │
│                                                                 │
│  AFTER LEVEL-1 COLLAPSE (B wins):                              │
│                                                                 │
│       ●  (Root: "Something is moving")                         │
│       │                                                         │
│       B  (Level 1: "Ring" CONFIRMED)                           │
│      ╱╲                                                        │
│    B1  B2  (Level 2: "Ring at position X" vs "Ring at Y")     │
│                                                                 │
│  ────────────────────────────────────────────────────────────  │
│                                                                 │
│  WHAT B1 AND B2 INHERIT:                                        │
│                                                                 │
│  • CONTEXT: The fact that it's a "Ring" is now certain        │
│  • PRIOR: Their prior probability is renormalized             │
│  • EVIDENCE: They can now accumulate position evidence        │
│  • CONSTRAINT: They must be consistent with "Ring"            │
│                                                                 │
│  The collapsed parent CONSTRAINS the children's hypotheses.   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 10. Implementation Patterns

### 10.1 Data Structures

```python
IMPLEMENTATION: DATA STRUCTURES

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  class Branch:                                                  │
│      """A hypothesis branch in the collapse tree."""           │
│                                                                 │
│      def __init__(self, hypothesis, parent=None):              │
│          self.hypothesis = hypothesis  # The prediction        │
│          self.parent = parent                                   │
│          self.children = []                                     │
│          self.depth = parent.depth + 1 if parent else 0       │
│                                                                 │
│          # Evidence (knowledge-informed)                       │
│          self.evidence = 0.0                                   │
│          self.manifold_score = 0.0                             │
│          self.history_score = 0.0                              │
│                                                                 │
│          # Energy (reactive)                                   │
│          self.energy = 1.0                                      │
│          self.weight = 1.0                                      │
│                                                                 │
│  ────────────────────────────────────────────────────────────  │
│                                                                 │
│  class CollapseTree:                                            │
│      """Decision tree for collapse."""                         │
│                                                                 │
│      def __init__(self, root_state):                           │
│          self.root = Branch(root_state)                        │
│          self.all_branches = [self.root]                       │
│          self.collapsed_levels = set()                         │
│                                                                 │
│      def add_branch(self, hypothesis, parent):                 │
│          branch = Branch(hypothesis, parent)                   │
│          parent.children.append(branch)                        │
│          self.all_branches.append(branch)                      │
│          return branch                                          │
│                                                                 │
│      def branches_at_level(self, level):                       │
│          return [b for b in self.all_branches                  │
│                  if b.depth == level and b.weight > 0]        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 10.2 Reactive Components

```python
IMPLEMENTATION: REACTIVE TRIGGERS

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  class CollapseTrigger:                                         │
│      """Reactive collapse triggering."""                       │
│                                                                 │
│      def __init__(self, config):                               │
│          # Thresholds (reactive parameters)                    │
│          self.snr_threshold = config.snr_threshold             │
│          self.min_energy = config.min_energy                   │
│          self.max_branches = config.max_branches               │
│          self.time_limit = config.time_limit                   │
│                                                                 │
│      def should_collapse(self, tree, level=None):              │
│          """Check if collapse should trigger (REACTIVE)."""   │
│                                                                 │
│          # Get branches at level (or all active)               │
│          branches = (tree.branches_at_level(level)             │
│                     if level is not None                       │
│                     else tree.active_branches())               │
│                                                                 │
│          if len(branches) <= 1:                                │
│              return False  # Nothing to collapse               │
│                                                                 │
│          # Compute SNR                                          │
│          evidence = [b.evidence for b in branches]             │
│          snr = self._compute_snr(evidence)                     │
│                                                                 │
│          # Check threshold (REACTIVE DECISION)                 │
│          return snr > self.snr_threshold                       │
│                                                                 │
│      def _compute_snr(self, evidence):                         │
│          """Signal-to-noise ratio of evidence."""              │
│          sorted_ev = sorted(evidence, reverse=True)            │
│          if len(sorted_ev) < 2:                                │
│              return float('inf')                               │
│          gap = sorted_ev[0] - sorted_ev[1]                     │
│          noise = np.std(evidence) + 1e-8                       │
│          return gap / noise                                     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 10.3 Knowledge Components

```python
IMPLEMENTATION: KNOWLEDGE-INFORMED SELECTION

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  class CollapseSelector:                                        │
│      """Knowledge-informed branch selection."""                │
│                                                                 │
│      def __init__(self, manifold, history, weights):           │
│          self.manifold = manifold      # Stored patterns       │
│          self.history = history        # Past outcomes         │
│          self.w_evidence = weights['evidence']                 │
│          self.w_manifold = weights['manifold']                 │
│          self.w_history = weights['history']                   │
│          self.w_coherence = weights['coherence']               │
│                                                                 │
│      def select_winner(self, branches):                        │
│          """Select winning branch (KNOWLEDGE-INFORMED)."""    │
│          best_score = -float('inf')                            │
│          winner = None                                          │
│                                                                 │
│          for branch in branches:                               │
│              score = self._compute_score(branch)               │
│              if score > best_score:                            │
│                  best_score = score                             │
│                  winner = branch                                │
│                                                                 │
│          return winner                                          │
│                                                                 │
│      def _compute_score(self, branch):                         │
│          """Geometric evidence computation."""                 │
│          # Accumulated evidence                                 │
│          ev = branch.evidence                                   │
│                                                                 │
│          # Manifold similarity (GEOMETRIC)                     │
│          ms = self.manifold.similarity(branch.hypothesis)     │
│                                                                 │
│          # Historical support                                   │
│          hs = self.history.support(branch.hypothesis)         │
│                                                                 │
│          # Internal coherence                                   │
│          coh = self._compute_coherence(branch)                 │
│                                                                 │
│          return (self.w_evidence * ev +                        │
│                  self.w_manifold * ms +                        │
│                  self.w_history * hs +                         │
│                  self.w_coherence * coh)                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 10.4 Complete Collapse System

```python
IMPLEMENTATION: COMPLETE COLLAPSE SYSTEM

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  class CollapseSystem:                                          │
│      """                                                        │
│      Complete collapse mechanism combining:                    │
│      - Reactive triggers (WHEN)                                │
│      - Knowledge selection (WHAT)                              │
│      """                                                        │
│                                                                 │
│      def __init__(self, config, manifold, history):            │
│          self.trigger = CollapseTrigger(config)                │
│          self.selector = CollapseSelector(                     │
│              manifold, history, config.weights                 │
│          )                                                      │
│          self.collapse_type = config.collapse_type             │
│                                                                 │
│      def step(self, tree, observation):                        │
│          """One step of collapse processing."""                │
│                                                                 │
│          # 1. Update evidence for all branches                 │
│          self._update_evidence(tree, observation)              │
│                                                                 │
│          # 2. Check if collapse should trigger (REACTIVE)     │
│          if self.collapse_type == 'staged':                    │
│              self._staged_collapse(tree)                       │
│          elif self.trigger.should_collapse(tree):              │
│              self._execute_collapse(tree)                       │
│                                                                 │
│          return tree                                            │
│                                                                 │
│      def _execute_collapse(self, tree):                        │
│          """Execute collapse based on type."""                 │
│                                                                 │
│          # Select winner (KNOWLEDGE-INFORMED)                  │
│          winner = self.selector.select_winner(                 │
│              tree.active_branches()                            │
│          )                                                      │
│                                                                 │
│          # Prune based on type                                  │
│          if self.collapse_type == 'hard':                      │
│              self._hard_prune(tree, winner)                    │
│          elif self.collapse_type == 'soft':                    │
│              self._soft_prune(tree, winner)                    │
│                                                                 │
│      def _staged_collapse(self, tree):                         │
│          """Level-by-level collapse."""                        │
│          for level in range(tree.max_depth):                   │
│              if level in tree.collapsed_levels:                │
│                  continue  # Already collapsed                 │
│                                                                 │
│              if self.trigger.should_collapse(tree, level):    │
│                  winner = self.selector.select_winner(         │
│                      tree.branches_at_level(level)             │
│                  )                                              │
│                  self._prune_level(tree, winner, level)        │
│                  tree.collapsed_levels.add(level)              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 11. Integration with Spectral Attention

### 11.1 Collapse in Spectral Attention

```
HOW COLLAPSE INTEGRATES WITH SPECTRAL ATTENTION:

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  SPECTRAL ATTENTION SYSTEM:                                     │
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │                    INPUT FRAME                            │ │
│  └─────────────────────────┬─────────────────────────────────┘ │
│                            │                                   │
│                            ▼                                   │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │              SPECTRAL BAND DECOMPOSITION                  │ │
│  │         DC │ Low │ Mid │ High                             │ │
│  └─────────────────────────┬─────────────────────────────────┘ │
│                            │                                   │
│                            ▼                                   │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │                 ATTENTION MECHANISMS                      │ │
│  │                                                           │ │
│  │  Temporal ──► COLLAPSE TREE (time hypotheses)            │ │
│  │  Neighbor ──► COLLAPSE TREE (spatial hypotheses)         │ │
│  │  Wormhole ──► COLLAPSE TREE (non-local hypotheses)       │ │
│  │                                                           │ │
│  └─────────────────────────┬─────────────────────────────────┘ │
│                            │                                   │
│                            ▼                                   │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │                  PREDICTION OUTPUT                        │ │
│  │            (after collapse to single answer)             │ │
│  └───────────────────────────────────────────────────────────┘ │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 11.2 Collapse Points in the Pipeline

```
WHERE COLLAPSE HAPPENS:

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  COLLAPSE POINT 1: BAND SELECTION                              │
│  ────────────────────────────────                               │
│  • Which spectral bands to use for this prediction?           │
│  • Decision tree: One branch per band combination             │
│  • Collapse type: SOFT (weighted attention over bands)        │
│  • Trigger: Always (continuous weighting)                      │
│                                                                 │
│  COLLAPSE POINT 2: TEMPORAL ATTENTION                          │
│  ─────────────────────────────────────                          │
│  • Which past frames are relevant?                             │
│  • Decision tree: One branch per history frame                │
│  • Collapse type: SOFT (top-k selection)                      │
│  • Trigger: Similarity threshold                               │
│                                                                 │
│  COLLAPSE POINT 3: WORMHOLE CONNECTION                         │
│  ──────────────────────────────────────                         │
│  • Which non-local region to attend to?                        │
│  • Decision tree: One branch per candidate connection         │
│  • Collapse type: SOFT or HARD depending on threshold         │
│  • Trigger: Similarity > threshold                             │
│                                                                 │
│  COLLAPSE POINT 4: FINAL PREDICTION                            │
│  ───────────────────────────────────                            │
│  • What is the next frame?                                     │
│  • Decision tree: Multiple possible predictions               │
│  • Collapse type: HARD (must commit to one answer)            │
│  • Trigger: End of inference step                              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 11.3 Spectral Collapse Hierarchy

```
HIERARCHICAL COLLAPSE MATCHING SPECTRAL BANDS:

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  LEVEL 0: DC BAND COLLAPSE                                     │
│  ─────────────────────────                                      │
│  Question: Is there SOMETHING moving?                          │
│  Branches: {Present, Absent}                                   │
│  Collapse: HARD (binary decision)                              │
│  Threshold: Very low (easy to decide)                          │
│                                                                 │
│  LEVEL 1: LOW-FREQ COLLAPSE                                    │
│  ──────────────────────────                                     │
│  Question: WHAT category of thing?                             │
│  Branches: {Ring, Square, Blob, Line, ...}                    │
│  Collapse: STAGED (commit to category)                         │
│  Threshold: Medium                                              │
│                                                                 │
│  LEVEL 2: MID-FREQ COLLAPSE                                    │
│  ───────────────────────────                                    │
│  Question: WHAT specific features?                             │
│  Branches: {Large/Small, Filled/Hollow, Fast/Slow}            │
│  Collapse: STAGED (refine within category)                     │
│  Threshold: Medium-high                                         │
│                                                                 │
│  LEVEL 3: HIGH-FREQ COLLAPSE                                   │
│  ────────────────────────────                                   │
│  Question: WHERE exactly?                                      │
│  Branches: {Position variants}                                 │
│  Collapse: SOFT → HARD (continuous then commit)               │
│  Threshold: High (need certainty for position)                 │
│                                                                 │
│  ────────────────────────────────────────────────────────────  │
│                                                                 │
│  This hierarchy matches the WHAT/WHERE separation:             │
│  • Low-freq decides WHAT (collapses first)                    │
│  • High-freq decides WHERE (collapses last)                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 12. Design Principles

### 12.1 Core Principles

```
DESIGN PRINCIPLES FOR COLLAPSE MECHANISMS:

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  PRINCIPLE 1: SEPARATE TRIGGER FROM SELECTION                  │
│  ─────────────────────────────────────────────                  │
│  • Trigger = REACTIVE (energy threshold)                       │
│  • Selection = KNOWLEDGE (geometric evidence)                  │
│  • Never mix these concerns in one module                      │
│                                                                 │
│  PRINCIPLE 2: MATCH COLLAPSE TYPE TO USE CASE                  │
│  ─────────────────────────────────────────────                  │
│  • Hard: Final outputs, binary decisions                       │
│  • Soft: Intermediate states, uncertain situations            │
│  • Staged: Multi-scale problems, hierarchies                   │
│                                                                 │
│  PRINCIPLE 3: COARSE BEFORE FINE                               │
│  ───────────────────────────────                                │
│  • Collapse structural decisions first                         │
│  • Then refine details within committed structure             │
│  • Prevents detail-level noise from affecting structure       │
│                                                                 │
│  PRINCIPLE 4: EVIDENCE IS GEOMETRIC                            │
│  ──────────────────────────────────                             │
│  • Evidence comes from manifold similarity                     │
│  • Evidence accumulates via constructive interference         │
│  • Winner selection is always knowledge-informed              │
│                                                                 │
│  PRINCIPLE 5: THRESHOLDS ARE TUNABLE                           │
│  ───────────────────────────────────                            │
│  • Different thresholds for different levels                   │
│  • Thresholds can be learned (slowly)                         │
│  • Start conservative, tune with experience                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 12.2 Implementation Checklist

```
IMPLEMENTATION CHECKLIST:

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  FOR EACH COLLAPSE POINT:                                       │
│                                                                 │
│  □ Define the hypothesis space (what are the branches?)        │
│  □ Choose collapse type (hard/soft/staged)                     │
│  □ Set reactive thresholds (SNR, energy, time)                │
│  □ Define evidence sources (manifold, history, coherence)     │
│  □ Implement branch generation (knowledge-informed)           │
│  □ Implement trigger check (reactive)                          │
│  □ Implement winner selection (knowledge-informed)             │
│  □ Implement pruning procedure (based on type)                │
│  □ Handle post-collapse state (forget/archive/merge)          │
│  □ Test collapse behavior                                      │
│  □ Monitor collapse metrics (timing, accuracy)                 │
│                                                                 │
│  ────────────────────────────────────────────────────────────  │
│                                                                 │
│  FOR HIERARCHICAL COLLAPSE:                                     │
│                                                                 │
│  □ Define level structure (what decisions at each level?)     │
│  □ Set per-level thresholds                                    │
│  □ Implement cascading logic                                   │
│  □ Implement context inheritance                               │
│  □ Test level-by-level collapse                                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 12.3 Summary Framework

```
THE COMPLETE COLLAPSE FRAMEWORK:

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│         C O L L A P S E   M E C H A N I S M S                  │
│                                                                 │
│  ═══════════════════════════════════════════════════════════   │
│                                                                 │
│  STRUCTURE:                                                     │
│  Decision tree where branches = hypotheses                     │
│  Collapse = selecting winner and pruning others                │
│                                                                 │
│  ═══════════════════════════════════════════════════════════   │
│                                                                 │
│  TWO COMPONENTS:                                                │
│                                                                 │
│  REACTIVE (WHEN):                                               │
│  • SNR threshold check                                         │
│  • Energy monitoring                                           │
│  • Time limits                                                  │
│  → Fast, automatic trigger                                     │
│                                                                 │
│  KNOWLEDGE-INFORMED (WHAT):                                     │
│  • Manifold similarity                                         │
│  • Historical evidence                                         │
│  • Accumulated support                                          │
│  → Deliberate selection                                        │
│                                                                 │
│  ═══════════════════════════════════════════════════════════   │
│                                                                 │
│  THREE TYPES:                                                   │
│                                                                 │
│  HARD: Winner-take-all, final decisions                        │
│  SOFT: Winner-take-most, uncertain states                      │
│  STAGED: Level-by-level, hierarchies                           │
│                                                                 │
│  ═══════════════════════════════════════════════════════════   │
│                                                                 │
│  KEY INSIGHT:                                                   │
│                                                                 │
│  "Reactive gates, knowledge fills"                              │
│                                                                 │
│  Reactive mechanisms decide IF and WHEN to collapse.           │
│  Knowledge mechanisms decide WHAT to collapse to.              │
│  Both are necessary. Neither is sufficient alone.              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Summary

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  C O L L A P S E   M E C H A N I S M S                         │
│                                                                 │
│  ═══════════════════════════════════════════════════════════   │
│                                                                 │
│  COLLAPSE IS A DECISION PROBLEM:                                │
│  • Decision tree structure with hypothesis branches            │
│  • Collapse = pruning to select winner                         │
│                                                                 │
│  TWO MODES WORK TOGETHER:                                       │
│  • REACTIVE: When to collapse (threshold triggers)            │
│  • KNOWLEDGE: What to collapse to (evidence selection)        │
│                                                                 │
│  THREE COLLAPSE TYPES:                                          │
│  • HARD: Winner-take-all (final outputs)                       │
│  • SOFT: Winner-take-most (intermediate states)               │
│  • STAGED: Level-by-level (hierarchies)                        │
│                                                                 │
│  PHASES:                                                        │
│  • Branch Generation: Knowledge creates hypotheses             │
│  • Branch Competition: Evidence accumulates via interference  │
│  • Branch Selection: Knowledge picks winner when reactive fires│
│                                                                 │
│  HIERARCHY MATCHES SPECTRAL BANDS:                              │
│  • DC/Low collapse first (WHAT)                                │
│  • Mid/High collapse later (WHERE)                             │
│                                                                 │
│  CORE PRINCIPLE:                                                │
│  "Reactive gates, knowledge fills"                              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

**References:**

- COLLAPSE_GENERALIZATION.md - The collapse phenomenon
- KNOWLEDGE_AND_REACTIVITY.md - Reactive vs knowledge-informed
- EQUILIBRIUM_AND_CONSERVATION.md - Phase transitions
- MANIFOLD_BELIEF_INTERFERENCE.md - Belief interference
- FFT_AMPLITUDE_FREQ_PHASE.md - Spectral decomposition

*This document provides the technical framework for implementing collapse mechanisms using decision tree structures. The key insight is that collapse requires both reactive triggers (energy-based, automatic) and knowledge-informed selection (geometric, evidence-based). These correspond to WHEN and WHAT respectively, following the principle "reactive gates, knowledge fills."*

---

*Oscar Goldman - Shogu Research Group @ Datamutant.ai*

