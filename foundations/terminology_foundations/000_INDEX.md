# Terminology Foundations Index

## Purpose of This Folder

This folder contains **foundational explanations** for AKIRA's core theoretical concepts. Each document explains a concept from first principles, shows how it works in theory and practice, and demonstrates its role in AKIRA.

These documents serve as:
1. **Learning resources** for understanding AKIRA
2. **Reference materials** for the theoretical framework
3. **Bridges** between established science and AKIRA-specific applications

---

## Document Overview

```
DOCUMENT DEPENDENCY GRAPH
─────────────────────────

                ┌─────────────────────────┐
                │  SYNERGY_REDUNDANCY.md  │ ← Foundation
                │  (PID decomposition)    │
                └───────────┬─────────────┘
                            │
            ┌───────────────┼───────────────┐
            │               │               │
            ▼               ▼               ▼
┌───────────────────┐ ┌─────────────┐ ┌─────────────────────┐
│INFORMATION_DYNAMICS│ │COLLAPSE_    │ │CIRCUIT_COMPLEXITY.md│
│.md (temporal)     │ │TENSION.md   │ │(tractability)       │
└─────────┬─────────┘ └──────┬──────┘ └──────────┬──────────┘
          │                  │                    │
          │         ┌────────┴────────┐          │
          │         │                 │          │
          ▼         ▼                 ▼          │
    ┌─────────────────────┐  ┌────────────────┐ │
    │  ACTION_QUANTA.md   │  │SUPERPOSITION_  │ │
    │  (emergent units)   │  │MOLECULE.md     │ │
    └──────────┬──────────┘  │(duality)       │ │
               │             └───────┬────────┘ │
               │                     │          │
               └──────────┬──────────┘          │
                          │                     │
                          ▼                     │
                 ┌────────────────┐             │
                 │  COHERENCE.md  │ ◄───────────┘
                 │  (bonding,     │
                 │   selection)   │
                 └───────┬────────┘
                                          │
                                          ▼
                              ┌─────────────────────┐
                              │    AKIRA SYSTEM     │
                              │  (full integration) │
                              └─────────────────────┘
```

---

## Documents in This Folder

### 1. SYNERGY_REDUNDANCY.md

**What it explains:**
- PID (Partial Information Decomposition)
- Redundancy: Information sources share
- Synergy: Information only from combination
- Unique information

**Key insight:** Synergy indicates structure/relationship; redundancy indicates agreement/certainty.

**In AKIRA:** High synergy = pre-collapse uncertainty; high redundancy = post-collapse certainty.

**Read if:** You need to understand how information is distributed across bands.

---

### 2. INFORMATION_DYNAMICS.md

**What it explains:**
- Storage: Information persisting within a variable
- Transfer: Information flowing between variables
- Modification: Information emerging from combination

**Key insight:** Modification = Synergy observed temporally. This is where computation happens.

**In AKIRA:** Collapse IS modification IS where AKIRA actually computes.

**Read if:** You need to understand how information flows over time in AKIRA.

---

### 3. CIRCUIT_COMPLEXITY.md

**What it explains:**
- SOS Width: Simultaneous constraints to track
- Circuit Breadth: Parallel processing capacity
- The theorem: Required Breadth = (k+1) × β
- Problem classes: What architectures can/cannot solve

**Key insight:** The 7+1 = 8 bands are formally justified for problems with SOS width ≤ 3.

**In AKIRA:** Provides mathematical bounds on what AKIRA can solve.

**Read if:** You need to understand why AKIRA has 8 bands and what its limits are.

---

### 4. COLLAPSE_TENSION.md

**What it explains:**
- Tension: Certainty → Uncertainty (belief expansion)
- Collapse: Uncertainty → Certainty (posterior contraction)
- The pump cycle: Redundancy → Synergy → Redundancy + AQ

**Key insight:** Collapse is where computation happens; tension is where exploration happens.

**In AKIRA:** The fundamental rhythm of processing: explore (tension) then commit (collapse).

**Read if:** You need to understand how AKIRA processes information over time.

---

### 5. ACTION_QUANTA.md

**What it explains:**
- AQ: Minimum patterns that enable correct decision
- Properties: Magnitude, Phase, Frequency, Coherence
- AQ vs PID atoms: Different concepts
- Molecular bonds: How AQ combine

**Key insight:** AQ are task-relative; they're what AKIRA actually computes.

**In AKIRA:** AQ emerge during collapse; they are the output of processing.

**Read if:** You need to understand what AKIRA produces and how patterns crystallize.

---

### 6. SUPERPOSITION_MOLECULE.md

**What it explains:**
- Superposition: Pre-collapse distributed state (wave-like, reducible)
- Crystallized: Post-collapse concentrated state (particle-like, IRREDUCIBLE)
- Why "crystallized" is the correct term (phase transition, irreducibility)
- Why both phases are needed
- The wave-particle parallel

**Key insight:** Crystallized = IRREDUCIBLE. Like a crystal lattice, you cannot remove one part without breaking the structure. This is the defining property of post-collapse AQ.

**In AKIRA:** Architecture supports both phases; different processing in each.

**Read if:** You need to understand the two states belief can be in.

**Note:** "Bonded state" is reserved for when multiple crystallized AQ combine (terminology under consideration).

---

### 7. COHERENCE.md

**What it explains:**
- Coherence: Phase alignment between components
- Interference: How coherent/incoherent combinations combine
- Role in bonding: Phase coherence determines valid AQ combinations
- Role in attention: Dot product measures phase alignment
- Role in collapse: Coherent states survive, incoherent cancel

**Key insight:** Coherence is not imposed — it emerges from survival. What survives interference is what harmonizes. Coherent states dominate by a factor of N (quadratic advantage).

**In AKIRA:** Coherence determines which AQ combinations bond, which hypotheses survive collapse, and what attention selects.

**Read if:** You need to understand why some patterns survive and others cancel, how bonding works, or why attention acts as a selection mechanism.

---

## Reading Order

### For complete understanding (recommended):

```
1. SYNERGY_REDUNDANCY.md     (foundation)
2. INFORMATION_DYNAMICS.md   (temporal extension)
3. CIRCUIT_COMPLEXITY.md     (architecture bounds)
4. COLLAPSE_TENSION.md       (the processes)
5. ACTION_QUANTA.md          (what emerges)
6. SUPERPOSITION_MOLECULE.md (the duality)
7. COHERENCE.md              (bonding and selection)
```

### For specific topics:

**"Why 8 bands?"**
→ CIRCUIT_COMPLEXITY.md

**"What does AKIRA compute?"**
→ ACTION_QUANTA.md

**"How does processing work?"**
→ COLLAPSE_TENSION.md → INFORMATION_DYNAMICS.md

**"What are the two states?"**
→ SUPERPOSITION_MOLECULE.md ← SYNERGY_REDUNDANCY.md

**"How do AQ combine? Why do some patterns survive?"**
→ COHERENCE.md ← ACTION_QUANTA.md

**"How does attention select?"**
→ COHERENCE.md

---

## Relationship to Other AKIRA Documents

```
THESE DOCUMENTS EXPLAIN:          REFERENCED BY:
────────────────────────          ──────────────

SYNERGY_REDUNDANCY.md       →     TERMINOLOGY.md Section 2
                                  EVIDENCE.md throughout
                                  Many architecture docs

INFORMATION_DYNAMICS.md     →     TERMINOLOGY.md Section 3
                                  SPECTRAL_BELIEF_MACHINE.md

CIRCUIT_COMPLEXITY.md       →     TERMINOLOGY.md Section 4
                                  THEORY_NOTES_FOR_DEEP_CONSIDERATION.md
                                  THE_SEVEN_PLUS_ONE_ARCHITECTURE.md

COLLAPSE_TENSION.md         →     TERMINOLOGY.md Section 5
                                  COLLAPSE_DYNAMICS.md
                                  PHYSICAL_PARALLELS.md

ACTION_QUANTA.md            →     TERMINOLOGY.md Section 7
                                  THE_ATOMIC_STRUCTURE_OF_INFORMATION.md
                                  OBSERVABILITY_EMBEDDINGS.md

SUPERPOSITION_MOLECULE.md   →     TERMINOLOGY.md Section 6
                                  EVIDENCE.md Section 5

COHERENCE.md                →     TERMINOLOGY.md Section 5 (AQ properties)
                                  HARMONY_AND_COHERENCE.md (philosophical)
                                  RADAR_ARRAY.md (physical parallel)
                                  COMPUTATIONAL_MECHANICS_EQUIVALENCE.md
```

---

## Key Equations Reference

| Document | Key Equation | Meaning |
|----------|--------------|---------|
| SYNERGY_REDUNDANCY | I(S₁,S₂;T) = I_red + I_uni(S₁) + I_uni(S₂) + I_syn | Total information decomposes into four atoms |
| INFORMATION_DYNAMICS | Modification = Synergy | The key bridge between static and dynamic |
| CIRCUIT_COMPLEXITY | B = (k+1) × β | Required breadth for given problem width |
| COLLAPSE_TENSION | b(s) → δ(s-s*) | Posterior contraction (collapse) |
| ACTION_QUANTA | AQ = min pattern for action | Definition of Action Quantum (IRREDUCIBLE) |
| SUPERPOSITION_MOLECULE | \|b⟩ = Σᵢ cᵢ\|ψᵢ⟩ → \|ψⱼ⟩ | Superposition collapses to crystallized (irreducible) |
| COHERENCE | Coherent: \|A\|² = N², Incoherent: \|A\|² ~ N | Coherent states dominate by factor of N |

---

## References Across All Documents

### Primary Sources (External)

1. Williams & Beer (2010) - PID foundations
2. Lizier et al. (2013) - Information dynamics, Modification = Synergy
3. Mao et al. (2023) - Circuit complexity, SOS width theorem
4. van der Vaart & van Zanten (2008) - Posterior contraction
5. Kalman (1960) - Belief expansion/prediction step
6. Born & Wolf (1999) - Coherence in optics, wave interference
7. Pikovsky et al. (2001) - Phase locking, synchronization

### AKIRA Internal

- `AKIRA/foundations/TERMINOLOGY.md` - Complete terminology
- `AKIRA/architecture_theoretical/EVIDENCE.md` - Solid evidence
- `AKIRA/architecture_theoretical/EVIDENCE_TO_COLLECT.md` - Hypotheses to test
- `AKIRA/foundations/THE_ATOMIC_STRUCTURE_OF_INFORMATION.md` - Full AQ theory

---

*This folder provides the conceptual foundations for understanding AKIRA. For the complete terminology reference, see `AKIRA/foundations/TERMINOLOGY.md`. For evidence and experimental predictions, see `AKIRA/architecture_theoretical/EVIDENCE.md` and `EVIDENCE_TO_COLLECT.md`.*

