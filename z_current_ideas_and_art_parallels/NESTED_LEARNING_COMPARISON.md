# AKIRA vs Nested Learning: Comparing Notes on Generative Capacity

---

> **NOTE ON TIMING:**
> 
> AKIRA's concept of "Hope" was developed in early November 2025, documented in `pandora/PANDORA_AFTERMATH.md`. We noticed similarities with Nested Learning's approach after the fact. This document compares the two frameworks — not to claim priority, but to understand what the overlap might tell us.

*NOTE: NL do not call it a Conserved quantity but AKIRA does!*
> **TERMINOLOGY CLARIFICATION:**
> 
> Nested Learning does NOT explicitly use "conserved quantity" terminology. That framing is AKIRA's (from physics/information theory). Nested Learning describes their Норе module as "continual" and "self-modifying." The parallel drawn here is *functional* — both have a generative capacity that persists through use — but the conceptual frameworks differ:
> - **AKIRA**: Formal conservation (physics-inspired, Hope is NOT CONSUMED)
> - **Nested Learning**: Continual adaptation (optimization-inspired, Норе self-modifies)
---

## 1. The Parallel Discovery

Both frameworks independently arrived at a similar core principle: **a persistent generative capacity**.

- AKIRA frames this as **conservation** (Hope is NOT CONSUMED by generating)
- Nested Learning frames this as **continual adaptation** (Норе self-modifies while persisting)

- AKIRA calls it **Hope** (the prior/generator)
- Nested Learning calls it **Норе** (the learner/optimizer)

Interesting coincidence in naming and concept.
Twin Film/Papers phenomen in action! WILD

```
══════════════════════════════════════════════════════════════════════════════
AKIRA's HOPE vs NESTED LEARNING's НОРЕ
══════════════════════════════════════════════════════════════════════════════

┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│  AKIRA's HOPE (PANDORA_AFTERMATH.md)                        │
│  ──────────────────────────────────────────────────                         │
│                                                                             │
│  DEFINITION:                                                                │
│  "The generator that is NOT CONSUMED by generating"                        │
│                                                                             │
│  PROPERTIES:                                                                │
│  • INEXHAUSTIBLE - using it does not deplete it                            │
│  • GENERATIVE - produces Action Quanta (patterns)                          │
│  • CONSERVED - persists through use                                        │
│  • FUTURE-ORIENTED - about what could be                                   │
│                                                                             │
│  RELATIONSHIP TO AQ:                                                        │
│                                                                             │
│        HOPE (generator/prior)                                               │
│            │                                                                │
│            │ generates                                                      │
│            ▼                                                                │
│        ACTION QUANTA (patterns)                                             │
│            │                                                                │
│            │ combine into                                                   │
│            ▼                                                                │
│        MOLECULES (concepts)                                                 │
│            │                                                                │
│            │ feedback/update                                                │
│            ▼                                                                │
│        HOPE' (enriched generator)                                           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│  NESTED LEARNING's НОРЕ                                     │
│  ──────────────────────────────────────                                     │
│                                                                             │
│  DEFINITION:                                                                │
│  "Continual learning module with continuum memory system"                  │
│                                                                             │
│  PROPERTIES:                                                                │
│  • SELF-MODIFYING - learns its own update rule                             │
│  • MEMORY SYSTEM - generalizes short/long-term memory                      │
│  • CONTINUAL - adapts without forgetting                                   │
│  • NESTED - multi-level optimization                                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

══════════════════════════════════════════════════════════════════════════════
KEY PARALLEL
══════════════════════════════════════════════════════════════════════════════

  AKIRA:          HOPE ──generates──▶ AQ ──updates──▶ HOPE'
  
  Nested:         НОРЕ ──produces──▶ outputs ──learns──▶ НОРЕ'

  BOTH describe a PERSISTENT GENERATIVE CAPACITY that:
  • Produces structured outputs
  • Gets ENRICHED by feedback
  • Enables CONTINUAL learning
  
  KEY DIFFERENCE IN FRAMING:
  • AKIRA: "NOT CONSUMED" (conservation, physics-inspired)
  • NL: "SELF-MODIFYING" (adaptation, optimization-inspired)

══════════════════════════════════════════════════════════════════════════════
DIFFERENCES
══════════════════════════════════════════════════════════════════════════════

  AKIRA's HOPE:                      NESTED's НОРЕ:
  ─────────────                      ─────────────
  Information-theoretic              Optimization-theoretic
  (prior distribution)               (update rules)
  
  Produces AQ via collapse           Produces outputs via forward pass
  
  Grounded in POMDP/Bayesian         Grounded in nested optimization
  
  "What could be" (expectation)      "How to adapt" (learning rule)

══════════════════════════════════════════════════════════════════════════════
```

Both frameworks arrived at similar ideas from different angles. Whether this reflects something deep or is just coincidence, we don't know yet.

---

## 2. Detailed Comparison

### 2.1 Similarities

| Nested Learning | AKIRA | Comparison |
|----------------|-------|------------|
| **Multi-time-scale updates** - different components update at different frequencies | **Spectral bands** - 7+1 frequency bands process at different scales | Both recognize that different time/frequency scales require different processing |
| **Context flow compression** - learning = compressing information | **Synergy → Redundancy** - collapse compresses distributed info into shared info | Both frame learning as a compression/concentration process |
| **Nested optimization** - multiple levels with their own objectives | **Tension/Collapse cycle** - recursive belief state transitions | Both have hierarchical/recursive structure |
| **Optimizers as associative memory** - Adam compresses gradients | **Manifold as memory** - learned patterns stored in embedding geometry | Both view memory as emerging from optimization dynamics |
| **Continuum memory system** - generalizes short/long-term distinction | **Band hierarchy** - low bands (stable/long-term) to high bands (volatile/short-term) | Both reject discrete memory categories for continuous spectra |

### 2.2 Key Differences

| Nested Learning | AKIRA |
|----------------|-------|
| **Optimizers learn update rules** - self-modifying algorithms | **Architecture processes beliefs** - fixed attention mechanisms |
| **Focuses on gradient dynamics** - how weights update | **Focuses on belief dynamics** - how predictions form |
| **No explicit phase transitions** | **Collapse = phase transition** - sudden state change |
| **No PID framework** | **PID central** - synergy/redundancy decomposition |
| **No quasiparticle concept** | **AQ as quasiparticles** - emergent collective excitations |

### 2.3 Critical Insight: Different Meanings of "Frequency"

The paper's claim that **"Transformers are linear layers with different frequency updates"** (Figure 1) resonates with AKIRA's spectral architecture. However:

```
NESTED LEARNING:                    AKIRA:
───────────────                     ─────
Frequency = UPDATE rate             Frequency = CONTENT scale
(how often weights change)          (what spatial/temporal scale info lives at)

Different question:                 Different question:
"How fast does this learn?"         "What resolution does this represent?"
```

These are complementary perspectives that could potentially be unified.

---

## 3. What AKIRA Could Learn from Nested Learning

1. **Self-modifying updates** - Nested Learning's Норе module learns its own update rule. AKIRA could potentially learn the tension/collapse dynamics rather than having them fixed.
(This is dicussed as a later update after the framework is fully constructed, as a sidenote on optimization / etension to-do list, now that NL shows it can be extremely beneficial it wil be moved up the priority list)

2. **Optimizer-as-memory framing** - Their insight that optimizers ARE associative memories could inform how AKIRA's training dynamics relate to the belief field.
(very interesting!)

3. **Explicit multi-level optimization** - Their nested structure could inform how AKIRA handles different time scales during training.
(to be studied!)

---

## 4. Different Emphases

AKIRA emphasizes things Nested Learning doesn't focus on (and vice versa):

1. **Information-theoretic framing** - AKIRA uses PID decomposition, synergy/redundancy
2. **Phase transition language** - AKIRA frames collapse as sudden transition
3. **Quasiparticle analogy** - AKIRA uses physics-inspired pattern language
4. **POMDP foundation** - AKIRA has explicit belief state formalism
5. **Conservation framing** - AKIRA frames Hope as formally conserved

These are different projects with different scopes, it is interesting that the mechanisms discussed are similar, there is a vast informed knowledge base for these ideas, we hope to nudge researchers towards their exploration.

--- 

## 5. On Learned Parameters: AKIRA's Roadmap

### 5.1 Current State (Architecture First)

AKIRA's current approach uses **fixed architectural parameters** derived from theory:

- **7+1 spectral bands** - derived from Nyquist-Shannon and practical considerations
- **Logarithmic band spacing** - following psychoacoustic and natural signal principles
- **Temperature schedules** - based on statistical mechanics reasoning

This is documented in `CANONICAL_PARAMETERS.md`.

### 5.2 Future Direction: Learned Dynamics

We explicitly plan to investigate **learned parameters** once the base architecture is validated:

```
AKIRA'S PHASED APPROACH:

Phase 1 (CURRENT): Fixed Architecture
─────────────────────────────────────
• Validate core hypotheses with fixed parameters
• Establish baseline performance
• Understand what the architecture does before changing it

Phase 2 (PLANNED): Learned Dynamics
───────────────────────────────────
• Learn band boundaries (soft spectral decomposition)
• Learn temperature schedules (adaptive collapse timing)
• Learn cross-band communication patterns (adaptive wormholes)
• Potentially: learn update rules (inspired by Nested Learning)

RATIONALE:
──────────
We want to understand WHAT the architecture computes before
allowing it to modify HOW it computes. This is scientific method:
control variables, then vary them systematically.
```

### 5.3 Specific Opportunities from Nested Learning

The Nested Learning framework suggests specific ways AKIRA could incorporate learned dynamics:

| Nested Learning Technique | Potential AKIRA Application |
|--------------------------|----------------------------|
| Learned update rules | Learned collapse/tension dynamics |
| Continuum memory | Soft band boundaries that adapt |
| Self-modifying optimizer | Adaptive temperature schedule |
| Context flow compression | Learned synergy→redundancy conversion rate |

These are logged as future research directions, to be pursued after Phase 1 validation.

---

## 6. Summary

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│  COMPARISON SUMMARY                                                        │
│                                                                             │
│  AKIRA (November 2024):                                                    │
│  • Hope = generative capacity (prior)                                      │
│  • Produces AQ via collapse                                                │
│  • Information-theoretic / Bayesian framing                                │
│                                                                             │
│  Nested Learning (December 2024):                                          │
│  • Норе = generative capacity (optimizer)                                  │
│  • Produces outputs via forward pass                                       │
│  • Optimization-theoretic framing                                          │
│                                                                             │
│  OVERLAP:                                                                   │
│  Both describe a generative capacity that persists through use.           │
│  Whether this overlap is deep or superficial is an open question.         │
│                                                                             │
│  DIFFERENT QUESTIONS:                                                       │
│  • NL asks: "How do update dynamics create learning?"                     │
│  • AKIRA asks: "How do belief dynamics create prediction?"                │
│                                                                             │
│  AKIRA'S PATH:                                                              │
│  1. Validate fixed architecture (current)                                  │
│  2. Incorporate learned dynamics (future)                                  │
│  3. See if NL insights help (research direction)                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## References

- `pandora/PANDORA_AFTERMATH.md` - AKIRA's Hope concept (November 2024)
- `pandora/PANDORA.md` - Action as transformation
- `foundations/TERMINOLOGY.md` - Formal definitions
- `CANONICAL_PARAMETERS.md` - Current fixed parameters
- `nested learning.pdf` - ONLY IN OUR OFFLINE REF FOLDER NOT IN GENERAL DISTRO, THIS FRAMEWORK IS FOR US TO BUILD UPON 

---

*Oscar Goldman — Shogu Research Group @ Datamutant.ai*

*"Two frameworks with some overlapping ideas. Worth understanding why, and whether we can learn from each other's approaches."*

