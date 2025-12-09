# Experiment 020: Cross-Band Information Flow - Supporting Literature

## Evidence for architecture experiment choices

This experiment tests whether information flows asymmetrically between spectral bands (Low→High early, High→Low late) and whether symmetric band pairs (0↔6, 1↔5, 2↔4) have special status. Standard mutual information I(Band_i; Band_j) tells you *how much* information is shared, but not *how* — whether bands carry the same information redundantly, unique information, or synergistic information that emerges only when combined. **Partial Information Decomposition (PID)** provides the correct mathematical framework to answer these questions.

---

## Primary References

**Williams, P.L., & Beer, R.D. (2010).** *Nonnegative Decomposition of Multivariate Information.* arXiv:1004.2515.  
📄 [arXiv PDF](https://arxiv.org/pdf/1004.2515)

### Why This Paper Supports EXP_020

**Core Finding:** Mutual information between multiple sources and a target decomposes into always-nonnegative atoms: **redundancy**, **unique information**, and **synergy**.

**Key Concepts:**

| Atom | Definition | Cross-Band Meaning |
|------|------------|-------------------|
| **Redundancy** | Information any single source could provide | Both bands encode the same thing (e.g., both know "it's a ring") |
| **Unique** | Information only one source provides | Band 0 knows identity, Band 6 knows position — non-overlapping |
| **Synergy** | Information that emerges only when sources combine | Prediction accuracy requires *both* coarse structure AND fine detail |

**Why PID matters for EXP_020:**

1. **Standard MI conflates redundancy and synergy**  
   High MI between Band 0 and Band 6 could mean:
   - (a) They encode the *same* information (high redundancy)
   - (b) They encode *complementary* information that together enables prediction (high synergy)
   
   PID separates these. The experiment's H2 (symmetric pairs have strongest connections) is ambiguous without PID — "strongest" could mean redundant or synergistic.

2. **Collapse converts synergy to redundancy**  
   Before collapse: bands hold different hypotheses (high synergy — need all bands to predict).  
   After collapse: all bands agree on winner (high redundancy — any band suffices).  
   PID can track this conversion during the collapse event (H3: cross-band flow precedes within-band collapse).

3. **Asymmetric flow (H1) has a PID interpretation**  
   - **Low→High (top-down):** Coarse identity (Band 0) *constrains* fine localization (Band 6). This is redundancy injection — Band 6's uncertainty should decrease because Band 0's information propagates.
   - **High→Low (bottom-up):** Fine details (Band 6) provide evidence that updates coarse hypothesis (Band 0). This is synergy resolution — combining local evidence across positions enables identity inference.

4. **Interaction information can be negative — PID explains why**  
   Williams & Beer show that negative interaction information (McGill's measure) occurs when synergy > redundancy. If you observe negative interaction between bands, that's not "negative information" — it's a signature that combining bands provides *more* than the sum of parts. EXP_020 may observe this during the interference phase (EXP 007 terminology).

### Implementation Implications

**For EXP_020 Protocol (§3.1):**

```
STEP 1: Measure band-to-band information transfer
• Mutual information between bands           ← Replace/augment with:
• PID: Redundancy(Band_i; Band_j; Target)
• PID: Synergy(Band_i; Band_j; Target)
• PID: Unique(Band_i; Target) and Unique(Band_j; Target)
```

**Concrete PID metrics to add:**

```python
# Target = next-frame prediction or ground truth
# For each band pair (i, j):

R_ij = Redundancy(Target; Band_i, Band_j)   # Shared information
S_ij = Synergy(Target; Band_i, Band_j)      # Emergent information  
U_i  = Unique(Target; Band_i \ Band_j)      # Info only Band_i has
U_j  = Unique(Target; Band_j \ Band_i)      # Info only Band_j has

# Track R_ij / S_ij ratio over time:
# - High S/R ratio: bands are complementary (early processing?)
# - High R/S ratio: bands are redundant (post-collapse?)
```

**Predictions refined by PID:**

| Hypothesis | PID-based Prediction |
|------------|---------------------|
| H1 (asymmetric flow) | Early: Synergy(0,6) > Redundancy(0,6). Late: Redundancy(0,6) > Synergy(0,6). |
| H2 (symmetric pairs strongest) | Symmetric pairs should have high *synergy* (complementary) not just high MI. |
| H3 (cross-band precedes collapse) | Synergy→Redundancy conversion should precede entropy drop in individual bands. |
| H4 (flow correlates with task stage) | Top-down = Redundancy injection. Bottom-up = Synergy resolution. |

### Theoretical Connections

**To AKIRA Theory:**

1. **WHAT↔WHERE Wormholes:** PID formalizes the intuition. Band 0 (WHAT) and Band 6 (WHERE) should have high *synergy* — identity alone doesn't predict pixel, position alone doesn't identify object, but together they do.

2. **Collapse as Redundancy Emergence:** Pre-collapse belief is distributed (synergistic — need all bands). Post-collapse belief is concentrated (redundant — any band suffices). PID tracks this transition.

3. **Conservation Laws (EXP 005):** Total information I(Target; All Bands) is conserved. PID decomposes *how* it's distributed: does collapse destroy synergy or convert it to redundancy?

4. **Orthogonality Principle:** If bands are truly orthogonal (no leakage), then cross-band MI must flow through explicit wormholes. PID can verify: Redundancy should be near-zero without wormholes, positive only when wormholes active.

---

## Additional References

**Griffith, V., & Koch, C. (2014).** *Quantifying Synergistic Mutual Information.* arXiv:1205.4265.  
Extends PID with alternative synergy measures. Useful if Williams-Beer I_min proves too conservative for neural data.

**Timme, N. et al. (2014).** *Synergy, Redundancy, and Multivariate Information Measures.* Journal of Computational Neuroscience.  
Application of PID to neural population coding. Directly relevant methodology for applying PID to AKIRA band activations.

**Mediano, P. et al. (2021).** *Towards an Extended Taxonomy of Information Dynamics.* arXiv:2109.13186.  
Modern extensions including temporal PID (information dynamics). Could inform time-resolved cross-band flow analysis.

---

## Citation

Williams, P.L., & Beer, R.D. (2010). Nonnegative Decomposition of Multivariate Information. *arXiv preprint arXiv:1004.2515.*

---

*Oscar Goldman — Shogu Research Group @ Datamutant.ai subsidiary of 温心重工業*
