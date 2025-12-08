# Supporting Literature for EXP 025: Synergy-Redundancy Transition

## Evidence for architecture experiment choices

This document provides the scientific foundation for Experiment 025, which tests whether collapse IS the synergy→redundancy conversion mechanism (not merely correlated with it).

---

## Primary References

### Williams, P. L., & Beer, R. D. (2010). *Nonnegative Decomposition of Multivariate Information.*

**Why this paper is foundational:**

The original PID paper solves a fundamental problem in information theory. Traditional mutual information I(X;Y) tells you how much information X has about Y, but when you have multiple sources S1, S2, ... about a target T, the interaction information I(S1;S2;T) can be NEGATIVE. This is confusing — how can information be negative?

Williams & Beer showed that multivariate information decomposes into:
- **Redundancy (I_red):** Information ALL sources share about target
- **Unique (I_uni):** Information only ONE source has
- **Synergy (I_syn):** Information that requires ALL sources together — no subset suffices

The key insight: negative interaction information arises when synergy exceeds redundancy. PID separates these, giving nonnegative terms.

**Relevance to EXP 025:**

This framework lets us ask: Before collapse, is information about the answer SYNERGISTIC (need all bands)? After collapse, is it REDUNDANT (any band suffices)? This is the core testable prediction.

---

### Sparacino, L., Mijatovic, G., Antonacci, Y., Ricci, L., Marinazzo, D., Stramaglia, S., & Faes, L. (2025). *Partial Information Rate Decomposition.* Physical Review Letters, 135, 187401.

**Why this paper is critical:**

This 2025 paper extends PID to information RATES — not just static information decomposition, but how synergy and redundancy flow over time. Published in Physical Review Letters, it provides a rigorous framework for decomposing the rate at which information flows through a system into redundant, unique, and synergistic components.

Key contributions:
- Defines partial information rate (not just mutual information)
- Provides formulas for temporal synergy/redundancy dynamics
- Applicable to continuous-time processes

**Relevance to EXP 025:**

This is THE paper for our experiment. It provides:
1. Mathematical framework for measuring information RATE decomposition over time
2. Methods to track synergy→redundancy conversion as it happens
3. Tools for testing whether synergy rate changes PRECEDE collapse events

The rate formulation is more appropriate than static PID for testing causality — we need to know how information structure changes moment-to-moment, not just snapshots.

**Link:** [arXiv:2502.04550](https://arxiv.org/pdf/2502.04550)

---

### Bertschinger, N., Rauh, J., Olbrich, E., Jost, J., & Ay, N. (2014). *Quantifying Unique Information.*

**Why this paper matters:**

Provides the BROJA measure — a specific algorithm for computing PID that has desirable properties (uniqueness, consistency). Different PID measures exist; BROJA is well-studied and recommended.

**Relevance to EXP 025:**

Tells us HOW to actually compute PID. The experiment protocol uses BROJA by default.

---

## Secondary References

### Griffith, V., & Koch, C. (2014). *Quantifying synergistic mutual information.*

Discusses synergy in neural systems and provides context for why synergy matters for cognition. High synergy = binding, integration, "more than sum of parts."

**Relevance:** Supports the interpretation that high synergy before collapse represents binding (combining WHAT and WHERE into unified percept), and collapse resolves this into redundant representation.

---

### Tononi, G. (2004). *An information integration theory of consciousness.*

Introduces Phi (Φ), integrated information. While not identical to PID synergy, the concepts are related. High Phi = high integration = synergy-like.

**Relevance:** Provides broader context. If AKIRA's collapse involves synergy→redundancy, this connects to theories of consciousness and integration.

---

### Timme, N. M., & Lapish, C. (2018). *A Tutorial for Information Theory in Neuroscience.*

Practical tutorial on applying information theory to neural data, including mutual information estimation and pitfalls.

**Relevance:** Helps avoid estimation artifacts in PID computation. Important for interpreting EXP 025 results correctly.

---

## Key Concepts Explained

### Why Synergy Matters for Collapse

```
BEFORE COLLAPSE:
─────────────────
Multiple hypotheses coexist. The model is uncertain.
To predict the answer, you need ALL bands working together.
No single band "knows" the answer — the knowledge is distributed.
This is HIGH SYNERGY: I(Bands ; Answer) comes mostly from I_syn.

┌────────────────────────────────────────────────────────────┐
│  Band 0: "Something exists, maybe A or B"                 │
│  Band 3: "Has feature X, consistent with A or B"          │
│  Band 6: "Edge at position P, could be A or B"            │
│                                                            │
│  Together: "Given existence + feature + edge → It's A"    │
│  Separately: None can conclude "A"                        │
└────────────────────────────────────────────────────────────┘

AFTER COLLAPSE:
───────────────
One hypothesis wins. The model commits.
ANY band alone can predict the answer (they all "know").
This is HIGH REDUNDANCY: I(Bands ; Answer) comes mostly from I_red.

┌────────────────────────────────────────────────────────────┐
│  Band 0: "It's A" (existence crystallized)                │
│  Band 3: "It's A" (features crystallized)                 │
│  Band 6: "It's A" (position crystallized)                 │
│                                                            │
│  Each band independently encodes the answer.              │
└────────────────────────────────────────────────────────────┘
```

### The Causality Question

The key question EXP 025 addresses: Is synergy→redundancy merely CORRELATED with collapse, or does it CAUSE collapse?

**If causal:** Synergy drop → entropy drop (Granger causality test passes)
**If correlated:** Both happen together, but neither causes the other (Granger fails)

This matters because:
- If CAUSAL: We can CONTROL collapse by manipulating information structure
- If CORRELATED: We need a different mechanism for collapse

### Conservation Prediction

If collapse is truly synergy→redundancy conversion, then:

```
I_total = I_syn + I_red + I_uni

Before collapse: High I_syn, low I_red
After collapse: Low I_syn, high I_red
But I_total should stay constant (or increase monotonically)
```

This is testable via Experiment 005 (Conservation Laws).

---

## Experimental Design Choices

### Mediano, P. A. M. et al. (2021). *Towards an Extended Taxonomy of Information Dynamics via Integrated Information Decomposition.* arXiv:2109.13186.

**Why this paper matters:**

Introduces **ΦID (Integrated Information Decomposition)** — combining PID with Integrated Information Theory. The key insight: complex systems exhibit behavior where the "whole is more than the sum of parts." ΦID provides tools to quantify this.

Key contributions:
- Reveals previously unreported modes of collective information flow
- Expresses well-known measures (transfer entropy, etc.) as aggregates of these modes
- Extends explanatory power beyond traditional causal discovery

**Relevance to EXP 025:**

ΦID provides the *conceptual framework* — why synergy matters for understanding "emergence." When bands work together to produce information neither has alone, that's the "whole > sum of parts." Collapse might be the transition from high-ΦID state to low-ΦID state.

**Link:** [arXiv:2109.13186](https://arxiv.org/abs/2109.13186)

---

### Why BROJA Measure?

Multiple PID measures exist. We use BROJA because:
1. Well-studied theoretically
2. Software implementations available (dit library)
3. Consistent with intuitions about redundancy
4. Compatible with both ΦID framework (Mediano et al. 2021) and rate decomposition (Sparacino et al. 2025)

### Why Discretization?

PID computation typically requires discrete variables. We discretize continuous attention weights into 8-16 bins. This introduces some estimation error but is standard practice.

Alternative: Use k-nearest neighbor estimators for continuous PID (more complex, less well-established).

### Why Granger Causality?

Granger causality tests whether past values of X help predict Y beyond what past values of Y predict. If synergy Granger-causes entropy drop, then synergy changes PRECEDE collapse — consistent with causal relationship.

Limitations: Granger causality is not true causality (correlation + temporal precedence), but it's a reasonable first test.

---

## Predictions From Literature

Based on the PID literature applied to AKIRA's architecture:

| Prediction | Basis | Experiment Section |
|------------|-------|-------------------|
| Synergy drops >50% at collapse | Binding resolves to agreement | Phase 2 |
| Redundancy increases >100% | Single-band sufficiency | Phase 2 |
| I_total conserved within 10% | Information conservation | Phase 2 |
| Granger causality p < 0.05 | Synergy causes collapse | Phase 3 |
| Complementary pairs show most change | Spectral structure | Phase 4 |

---

## Connection to Other AKIRA Experiments

| Experiment | Relationship to EXP 025 |
|------------|------------------------|
| 005 (Conservation) | Tests I_total conservation |
| 020 (Cross-Band Flow) | Measures flow direction; 025 measures information type |
| 002 (Collapse Detection) | Provides collapse events to align |
| 022 (Phase Locking) | Phase locking may enable conversion |
| 024 (Resonant Wormholes) | Complementary pairs may show highest conversion |

---

*"The mechanism matters. If collapse IS synergy→redundancy conversion, we have a lever to pull. If it merely correlates, we must look elsewhere."*

