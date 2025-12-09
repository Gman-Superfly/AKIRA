# POMDP Framework: Supporting Literature

## Verification as Belief Update

**Wu, S., & Yao, Q. (2025).** *Asking LLMs to Verify First is Almost Free Lunch.* arXiv:2511.21734v1.  
Local: `supporting_literature_and_reference/010_EXP_TICKLING_TECHNIQUES/Wu_Yao_2024_Verification_First.pdf`

**Key Findings:**
- **Verification is cognitively easier than generation** (Baker et al., 1975)
- Verification triggers "reverse reasoning" complementary to forward CoT
- Minimal computational overhead (often fewer tokens than standard reasoning)
- Overcomes egocentric bias by critiquing external answers
- **Error correction through verification avoids pitfalls of self-reflection**
- Iter-VF (iterative verification-generation) uses Markovian refinement: previous answer only, not full history

**Relevance to AKIRA:**
- Supports **The Old Lady parable**: Verification (listening) is easier than decision (acting)
- Validates **optimal stopping framework**: When to gather information vs commit
- Confirms **POMDP belief update**: Verification refines belief state efficiently
- Demonstrates **cheap probing**: High information gain with low computational cost

**Connection to Theory:**
- Old Lady learns by observing (verification of others' actions)
- Optimal stopping: Balance cost of listening vs cost of wrong decision
- Belief state updates through verification reduce uncertainty efficiently
- Reverse reasoning provides complementary information to forward generation

---

## Optimal Stopping and Sequential Decision Making

*(Space for additional references)*

---

## Belief State Representation

**Shai, A. (2024).** *Transformers Represent Belief State Geometry in their Residual Stream.* LessWrong.  
🔗 Online: [link](https://www.lesswrong.com/posts/gTZ2SxesbHckJ3CkF/transformers-represent-belief-state-geometry-in-their)

**Key Findings:**
- **Transformers trained on HMM-generated sequences linearly encode Bayesian belief states** over hidden Markov states in their residual stream.
- **Belief-state geometry forms a structured (often fractal) subset of the probability simplex**, matching the mixed-state presentation (MSP) from Computational Mechanics.
- **Residual stream activations encode both a world model and synchronization dynamics**, how an observer updates beliefs over latent states.

**Relevance to AKIRA:**
- Grounds our use of **POMDP belief states** as the correct abstraction for internal transformer state.
- Supports the **collapse-as-belief-concentration** view developed in `architecture_base/collapse/COLLAPSE_GENERALIZATION.md` and `architecture_base/collapse/COLLAPSE_DYNAMICS.md`.
- Suggests that monitoring **residual/attention geometry** is a principled way to study belief evolution and collapse in AKIRA.

**Connection to Theory:**
- Connects **Computational Mechanics mixed-state geometry** to AKIRA’s **spectral collapse and POMDP belief updates**.
- Clarifies that an LLM’s “world model” inherently includes both **environment structure** and the **agent’s synchronization process**, aligning with our collapse framework.

**Wentworth, J., & Lorell, D. (2024).** *Why Would Belief-States Have A Fractal Structure, And Why Would That Matter For Interpretability? An Explainer.* LessWrong.  
🔗 Online: [link](https://www.lesswrong.com/posts/mBw7nc4ipdyeeEpWs/why-would-belief-states-have-a-fractal-structure-and-why)

**Key Findings:**
- Shows that **Bayesian belief states for HMMs implement a chaos game whose attractor is a fractal set** in the belief simplex.
- Emphasizes that **fractal belief geometry is a generic property of Bayesian tracking of latent Markov states**, not specific to transformers.
- Proposes using **self-similar activation sets in trained nets to infer their latent variables and update symmetries**.

**Relevance to AKIRA:**
- Provides the **conceptual bridge** from POMDP belief updates to **fractal belief manifolds**, supporting the geometry assumed in our collapse analysis.
- Motivates **searching for self-similar subsets of AKIRA's internal activations** as a way to identify latent factors and collapse structure.

---

## Multivariate Information Theory

### Terminology Clarification

**IMPORTANT:** Williams & Beer (2010) use "information atoms" to refer to the PID decomposition terms (Redundancy, Unique, Synergy). AKIRA uses **Action Quanta (AQ)** for irreducible actionable patterns to avoid collision with this established terminology. See `foundations/TERMINOLOGY.md` for full framework.

---

**Williams, P.L., & Beer, R.D. (2010).** *Nonnegative Decomposition of Multivariate Information.* arXiv:1004.2515.  
📄 Online: [PDF](https://arxiv.org/pdf/1004.2515)

**Key Findings:**
- **Mutual information between multiple sources and a target decomposes into nonnegative atoms:** redundancy, unique information per source, and synergy.
- **Synergy** = information that emerges only when sources combine; neither source alone can provide it.
- **Redundancy** = information any single source could provide; duplication across sources.
- Resolves the problem that **classical interaction information can be negative**, which is uninterpretable as "negative information."

**Relevance to AKIRA:**
- **Justifies wormhole architecture:** Complementary band pairs (0↔6, 1↔5, 2↔4) have HIGH SYNERGY, combining them enables predictions neither can make alone. Wormholes realize this synergy. See `architecture_base/attention/spectral_wormhole/SPECTRAL_WORMHOLE_ATTENTION.md` §3.3.
- **Explains collapse as information phase transition:** Pre-collapse states have HIGH SYNERGY (need all bands to predict). Post-collapse states have HIGH REDUNDANCY (any band suffices). Collapse = synergy→redundancy conversion. See `architecture_theoretical/ORTHOGONALITY.md` §11.3.
- **Refines conservation laws:** Total information I(Target; All Bands) is conserved, but its *decomposition* into R/U/S atoms changes during dynamics. This is the information-theoretic signature of a phase transition.
- **Action Quanta (AQ) crystallize during collapse:** AQ are the PRODUCT of synergy→redundancy transition. They emerge when the belief field condenses.

**Connection to AKIRA Theory:**
- Resolves the apparent paradox between orthogonality (bands should be independent) and wormholes (bands should communicate): orthogonal bands have low REDUNDANCY (good) but can have high SYNERGY (exploited by wormholes).
- Provides measurable quantities for experiments: track Redundancy(t), Synergy(t), Unique(t) during inference to observe collapse dynamics.
- See: `EXP_005_CONSERVATION_LAWS/README.md`, `EXP_020_CROSS_BAND_FLOW/README.md` for experimental applications.

---

## Additional Supporting References

**Griffith, V., & Koch, C. (2014).** *Quantifying Synergistic Mutual Information.* arXiv:1205.4265.  
Alternative synergy measure. Useful if Williams-Beer I_min underestimates synergy in neural data.

**Mediano, P. A. M. et al. (2025).** *Toward a unified taxonomy of information dynamics via Integrated Information Decomposition.* PNAS 122(39).  
📄 Online: [DOI](https://doi.org/10.1073/pnas.2423297122)  
Introduces **ΦID (Integrated Information Decomposition)**, extending PID to many-to-many dynamics. Provides framework for understanding "whole > sum of parts" dynamics and reveals modes of collective information flow. WHY synergy matters: it's the emergence of information that no single component has alone. Defines TDMI (Time-Delayed Mutual Information), information transfer, and information storage.

**Lizier, J.T., Flecker, B., & Williams, P.L. (2013).** *Towards a Synergy-based Approach to Measuring Information Modification.* arXiv:1303.3440.  
📄 Online: [PDF](https://arxiv.org/pdf/1303.3440)  
**KEY PAPER for terminology:** Defines **information modification = synergy** in distributed computation. Shows that storage, transfer, and modification form the triad of information dynamics operations. Particle collisions in cellular automata = information modification = synergy. **AKIRA collapse events are information modification events.**

**Sparacino, L. et al. (2025).** *Partial Information Rate Decomposition.* Physical Review Letters, 135, 187401. [arXiv:2502.04550](https://arxiv.org/pdf/2502.04550)  
Extends PID to information **RATES**, the temporal dynamics of synergy/redundancy. HOW to measure synergy/redundancy changes over time. Critical for EXP_025 causality testing: does synergy rate drop precede collapse?

**Complementary roles:**
- Mediano (2021): Conceptual framework, WHY synergy/integration matters
- Sparacino (2025): Measurement framework, HOW to track temporal dynamics


*Oscar Goldman, Shogu Research Group @ Datamutant.ai subsidiary of 温心重工業*
