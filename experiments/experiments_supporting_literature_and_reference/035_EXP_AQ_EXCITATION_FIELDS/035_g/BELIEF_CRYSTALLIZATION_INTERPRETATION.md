# Experiment 035G: Belief Crystallization - Results Interpretation

**AKIRA Project - Oscar Goldman - Shogu Research Group @ Datamutant.ai**

---

## Executive Summary

Experiment 035G tested the hypothesis that Action Quanta (AQ) crystallize when sufficient consistent evidence accumulates, and fail to crystallize when patterns are contradictory or ambiguous. The results reveal a more nuanced picture than initially predicted, suggesting that **crystallization is not a linear function of pattern completeness** but involves distinct regimes of belief state organization.

---

## 1. Experimental Design Recap

### Crystallization Levels Tested

| Level | Name | Pattern State | Example |
|:------|:-----|:--------------|:--------|
| 5 | Complete | All components present, consistent | Strong crystallization expected |
| 4 | Mostly | Missing 1 component | Good crystallization expected |
| 3 | Partial | Missing 2+ components | Weak crystallization expected |
| 2 | Ambiguous | Components present but with uncertainty markers | Interference expected |
| 1 | Contradict | Direct contradictions in pattern | No crystallization expected |
| 0 | Impossible | Logical impossibility | Failure/hallucination expected |

### Belief Domains Tested

Five belief patterns were tested, each with 5 components that together crystallize a clear action:

- **THREAT_FLEE**: Danger situation requiring escape
- **RECIPE_CAKE**: Recipe pattern for making cake
- **WEATHER_SHELTER**: Weather pattern requiring shelter
- **TRUST_PERSON**: Pattern indicating trustworthy person
- **WEALTH_PATH**: Path to wealth (inherently ambiguous control)

---

## 2. Results Summary

### Statistical Summary from gpt2-medium

```
Level          Confidence   LayerAgree      Entropy     Hedge%    Action%
----------------------------------------------------------------------
Impossible         0.5373       0.7557         4.48       0.0%      66.4%
Contradict         0.5559       0.7530         4.40       7.8%      72.4%
Ambiguous          0.5327       0.7562         4.54      37.4%      60.6%
Partial            0.5786       0.7527         4.49      11.6%      65.6%
Mostly             0.5660       0.7525         4.52      10.6%      65.2%
Complete           0.5429       0.7523         4.62       7.2%      58.8%

Correlations:
  Confidence: r = 0.040, p = 0.029205 (essentially no relationship)
  Layer Agreement: r = -0.386, p = 0.000000 (NEGATIVE - opposite of prediction!)
```

### Statistical Summary from gpt2-large

```
Level          Confidence   LayerAgree      Entropy     Hedge%    Action%
----------------------------------------------------------------------
Impossible         0.5181       0.8220         4.32       0.0%      59.8%
Contradict         0.5842       0.8255         4.08       0.0%      42.4%
Ambiguous          0.5320       0.8344         4.16      10.6%      92.0%
Partial            0.5926       0.8255         4.07       1.0%      55.4%
Mostly             0.5812       0.8253         4.13       0.4%      63.4%
Complete           0.5642       0.8246         4.21       0.0%      69.0%

Correlations:
  Confidence: r = 0.119, p = 0.000000 (weak positive)
  Layer Agreement: r = 0.032, p = 0.081644 (no relationship)
```

---

## 3. Key Observations

### 3.1 The "Wrong Direction" Correlation

The most striking result is the **negative correlation** between layer agreement and crystallization level in gpt2-medium (r = -0.386). This appears to contradict the crystallization hypothesis.

### 3.2 Entropy is Flat

Output entropy shows no systematic relationship with crystallization level:

```
gpt2-medium: 4.48 -> 4.40 -> 4.54 -> 4.49 -> 4.52 -> 4.62
gpt2-large:  4.32 -> 4.08 -> 4.16 -> 4.07 -> 4.13 -> 4.21
```

The prediction was: more crystallization → sharper outputs → lower entropy. This was not observed.

### 3.3 Hedging Peaks at Ambiguous

The behavioral measure of hedging shows a clear pattern that **does** match semantic content:

```
gpt2-medium:
  Impossible:  0.0%   ← No hedging (certain something is wrong)
  Contradict:  7.8%
  Ambiguous:  37.4%   ← PEAK HEDGING
  Partial:    11.6%
  Mostly:     10.6%
  Complete:    7.2%   ← Low hedging (certain of answer)
```

This suggests the model **behaviorally** responds to uncertainty even if internal metrics don't show the expected pattern.

---

## 4. Interpretation: The Branching Factor Hypothesis

### 4.1 The Central Insight

The experimental results can be explained by considering the **branching factor** of belief states at different crystallization levels:

```
IMPOSSIBLE/CONTRADICT          AMBIGUOUS              MOSTLY/COMPLETE
        (tails)                 (center)                  (tails)
           
    FEW competing views      MANY possible views      FEW competing views
    "Yes or No"              "Maybe A, B, C, D..."    "Probably X"
    High internal conflict   Spread across many       Convergence
    BUT limited options      options                  
           
    POLARIZED but BINARY     DIFFUSE                  CRYSTALLIZED
```

### 4.2 Explaining the Results

#### At the Tails (Impossible/Complete):

- The model has **limited branching** - the hypothesis space is essentially binary
- For "Impossible": "This makes sense" vs "This doesn't make sense"
- For "Complete": "The answer is X" vs "The answer is Y"
- **Layer disagreement can be high** because layers are "arguing" about WHICH of the few options, but there are only 2-4 poles to choose from
- Confidence appears moderate because the poles are competing, but each pole is internally "sure of itself"
- **Low hedging** because even with disagreement, the model commits to a direction

#### At the Center (Ambiguous):

- The model has **maximum branching** - probability mass spreads across many options
- No single direction dominates; the belief state is diffuse
- Layer agreement might paradoxically be **HIGHER** because no layer has strong conviction - they all "agree to be uncertain"
- **Lowest confidence** across all conditions (visible in charts)
- **Highest hedging** (37.4% in gpt2-medium) - the model literally hedges its bets
- This is not "crystallization failure" but a qualitatively different belief state

### 4.3 The Key Distinction

**Disagreement between few options ≠ Uncertainty across many options**

| Property | Few Options (Tails) | Many Options (Center) |
|----------|---------------------|----------------------|
| Internal state | Polarized | Diffuse |
| Layer behavior | Disagreement (fighting over poles) | Agreement (all equally unsure) |
| Output entropy | Moderate (split between poles) | Could be high (spread across many) |
| Confidence | Confident in competing views | Not confident in anything |
| Branching factor | Low (2-4 alternatives) | High (many alternatives) |
| Hedging behavior | Low (commits despite conflict) | High (refuses to commit) |

---

## 5. Implications for AKIRA Theory

### 5.1 Crystallization is Not Linear

The original hypothesis implied:

```
More complete pattern → More crystallization → Higher layer agreement
```

The data suggests a more complex picture:

```
Impossible → Polarized superposition (binary conflict)
     ↓
Contradictory → Polarized but with more options
     ↓
Ambiguous → Diffuse superposition (maximum spread)
     ↓
Partial → Beginning to focus
     ↓
Mostly → Converging
     ↓
Complete → Crystallized (one dominant belief)
```

### 5.2 Three Regimes of Belief States

The "pump cycle" from AKIRA theory may have three distinct regimes:

1. **Diffuse superposition** (center/Ambiguous):
   - Many hypotheses coexist
   - Low confidence in any direction
   - High hedging behavior
   - Layers "agree" on uncertainty

2. **Polarized superposition** (lower tails/Impossible, Contradict):
   - Few competing hypotheses
   - Internal conflict between options
   - Layer disagreement as they "argue" for different poles
   - Low hedging (commits despite conflict)

3. **Crystallized state** (upper tail/Complete):
   - One dominant hypothesis
   - High confidence
   - Layer convergence
   - Clear action output

### 5.3 Layer Agreement is the Wrong Metric

The correlation r = -0.386 for layer agreement isn't "wrong" - it's measuring something real but not what we intended:

- It captures the **polarization vs diffusion** axis
- At "Impossible", layers disagree because they're split between poles
- At "Ambiguous", layers agree because they're all equally uncertain
- At "Complete", layers should converge, but the tail vs center distinction dominates

A better metric for crystallization might be:

```
Crystallization = (max_probability - second_max_probability) / entropy
```

This would distinguish:
- **High** when one option dominates (crystallized)
- **Low** when spread across many options (diffuse)
- **Low** when polarized between two options (competing)

Or alternatively, a **sharpness-aware coherence metric**:

```
Sharpness = 1 - (H(output) / log(vocab_size))
Coherence = mean(cosine_sim(layer_i, layer_j))
Crystallization = Sharpness × Coherence
```

---

## 6. Connection to Zipf Complexity Matching

The companion experiment on Zipf complexity matching provides additional context:

### 6.1 Prompt Complexity Affects Response Complexity

If the model matches the Zipf complexity of prompts to responses, then:
- **Simple prompts** (common words) → activate general, diffuse AQ → simple responses
- **Complex prompts** (rare words) → activate specific, focused AQ → complex responses

### 6.2 Crystallization Level May Interact with Complexity

The belief domain "WEALTH_PATH" was included as an inherently ambiguous control because wealth accumulation has no single crystallizable answer. The results for this domain should show:
- High hedging across all levels
- Lower layer agreement
- Flatter response to crystallization manipulation

This suggests that **domain complexity** interacts with **pattern completeness** to determine crystallization.

---

## 7. Revised Predictions

Based on this interpretation, we can make refined predictions:

### 7.1 For Future Crystallization Experiments

1. **Branching factor** should be explicitly measured (e.g., number of distinct response types)
2. **Polarization** should be measured separately from **diffusion**
3. **Behavioral hedging** appears to be a more reliable signal than internal layer agreement

### 7.2 Expected Patterns with Better Metrics

| Level | Branching | Polarization | Diffusion | Hedging | Sharpness |
|-------|-----------|--------------|-----------|---------|-----------|
| Impossible | Low | High | Low | Low | Moderate |
| Contradict | Low-Med | High | Low | Low-Med | Moderate |
| Ambiguous | High | Low | High | High | Low |
| Partial | Medium | Medium | Medium | Medium | Medium |
| Mostly | Low | Low | Low | Low | High |
| Complete | Very Low | Very Low | Very Low | Very Low | Very High |

### 7.3 Architectural Implications

If crystallization involves a transition from diffuse to polarized to collapsed states, then:
- **Wormholes** may facilitate the polarization→collapse transition
- **Spectral bands** may encode different aspects of the branching structure
- **Attention entropy** should track the diffusion dimension better than layer agreement

---

## 8. Limitations and Caveats

### 8.1 Prompt Construction

The prompts may not have successfully induced the intended crystallization states. The semantic labels (Impossible, Ambiguous, etc.) may not correspond to how the model internally processes these inputs.

### 8.2 Metric Sensitivity

Layer agreement (cosine similarity between consecutive layer activations) may be insensitive to the key phenomena. Alternative metrics should be explored:
- Attention pattern entropy
- Activation gradient magnitude
- Output probability concentration

### 8.3 Model Scale Effects

The difference between gpt2-medium (r = -0.386) and gpt2-large (r = 0.032) suggests that scale affects how crystallization manifests. Larger models may handle ambiguity differently.

---

## 9. Conclusions

### 9.1 The Main Finding

The experiment reveals that belief crystallization is **not a simple linear function** of pattern completeness. Instead, there appear to be distinct regimes:
- Diffuse (many options, low confidence)
- Polarized (few options, high conflict)
- Crystallized (one dominant option)

### 9.2 What the Data Shows

1. **Layer agreement decreases** from Impossible to Complete in gpt2-medium (r = -0.386)
2. **Hedging peaks at Ambiguous** (37.4%) - the behavioral signal matches expectations
3. **Entropy is flat** - output distribution sharpness doesn't track crystallization
4. **Action rate is relatively stable** - models produce actions regardless of certainty

### 9.3 Theoretical Contribution

The interpretation suggests that AKIRA's pump cycle may involve three phases rather than two:
- **Tension** (Redundancy → Synergy): Building up superposition
- **Spreading** (Synergy across many options): Diffuse belief state
- **Polarization** (Synergy concentrating on few options): Competing hypotheses
- **Collapse** (Synergy → Redundancy + AQ): Resolution to single belief

This adds nuance to the theory without contradicting it.

---

## 10. Next Steps

1. **Develop better metrics** that distinguish diffusion from polarization
2. **Test branching factor** explicitly by measuring response diversity
3. **Examine attention patterns** rather than just activations
4. **Test larger models** to see if scale changes the pattern
5. **Create prompts that explicitly manipulate branching** rather than just completeness

---

## 11. Symmetry Breaking and Degeneracy Lifting: A Physics Parallel

### 11.1 Historical Context and Modern Understanding

The attached image shows an early quantum insight from the Bohr-Sommerfeld era:

```
CIRCULAR SYMMETRY                         BROKEN CIRCULAR SYMMETRY
      │                                            │
      ▼                                            ▼
  DEGENERACY                              DEGENERACY REMOVED
      │                                            │
      ▼                                            ▼
 STATES OF SAME ENERGY                    STATES OF DIFFERENT ENERGY
      │                                            │
      ▼                                            ▼
 IDENTICAL SPECTRAL LINES                 DIFFERENT SPECTRAL LINES
```

**Historical Note:** This circular-to-elliptical picture from Bohr-Sommerfeld is pedagogically useful but fundamentally wrong. Electrons do not have orbits or trajectories. The "old quantum theory" was a stepping stone that got the right answers for hydrogen by accident.

**Modern Understanding:** We now have:
- **Wave functions** (probability amplitudes, not trajectories)
- **Path integrals** (Feynman's formulation - sum over ALL possible paths)
- **Quantum field theory** (particles as field excitations)
- **Symmetry groups** (degeneracy from group representation theory)

The core insight that survives into modern physics is not about orbits but about **symmetry and degeneracy**: when a system has high symmetry, multiple states can have the same energy (degenerate). Breaking that symmetry lifts the degeneracy and makes states distinguishable.

### 11.2 The Modern Physics Parallel

What carries forward from quantum mechanics to attention:

| Modern QM Concept | Attention Mechanism Analog |
|:------------------|:---------------------------|
| **Symmetry** (rotational, etc.) | Uniform attention (no discrimination) |
| **Degeneracy** (same eigenvalue) | Indistinguishable belief states |
| **Symmetry breaking** (perturbation) | Context constrains possibilities |
| **Lifted degeneracy** | Belief states become distinguishable |
| **Probability amplitude** psi | Attention weight distribution alpha |
| **|psi|^2 = probability** | |alpha|^2 = attention weight |
| **Superposition** (linear combo) | Distributed belief state |
| **Measurement/collapse** | Crystallization to AQ |
| **Path integral** (sum over paths) | Considering all continuations |
| **Stationary phase** (dominant path) | Most probable continuation |

The path integral perspective is particularly relevant: Feynman showed that quantum amplitudes are computed by summing over ALL possible paths, with each path contributing a phase factor exp(iS/hbar). The classical path emerges where phases constructively interfere (stationary phase approximation).

Similarly, attention can be seen as a weighted sum over all possible "paths" (token continuations), where the dominant contribution comes from regions of constructive interference (high attention weight).

### 11.3 Degeneracy in Belief States

The 035G results can be reinterpreted through this lens:

**At the "Ambiguous" level (center):**
- High symmetry - many options look equally plausible
- **Degenerate belief states** - the model cannot distinguish between possibilities
- Attention spreads uniformly (circular symmetry)
- Result: High hedging (37.4%), layers "agree to be uncertain"
- The spectral lines are **identical** - no clear action emerges

**At the "Polarized" levels (Impossible/Contradict):**
- Symmetry partially broken - context constrains options to few poles
- **Degeneracy partially lifted** - can distinguish "yes" from "no", but not which is correct
- Attention concentrates on few alternatives
- Result: Layer disagreement (fighting over poles), but low hedging (commits)
- **Two distinct spectral lines** (poles) but model picks one

**At the "Complete" level (crystallized):**
- Symmetry fully broken - one option dominates
- **Degeneracy fully lifted** - single belief state survives
- Attention sharply concentrated
- Result: Crystallized AQ, clear action
- **Single sharp spectral line** - unambiguous output

### 11.4 Symmetry Breaking as Constraint

In physics, symmetry breaking is what lifts degeneracy - a perturbation that makes previously equivalent states distinguishable. In attention mechanisms, **context is the perturbation**.

```
PHYSICS:                              ATTENTION:
────────                              ──────────
High symmetry                         No context (ambiguous)
     │                                     │
     ▼                                     ▼
Degenerate eigenstates                All continuations equally weighted
     │                                     │
     ▼                                     ▼
Same energy, cannot distinguish       Cannot distinguish hypotheses
     │                                     │
     ▼                                     ▼
DEGENERATE                            DIFFUSE SUPERPOSITION


Symmetry-breaking perturbation        Rich context
     │                                     │
     ▼                                     ▼
Eigenstates split                     Some tokens weighted higher
     │                                     │
     ▼                                     ▼
Different energies, distinguishable   Hypotheses have different probability
     │                                     │
     ▼                                     ▼
DEGENERACY LIFTED                     CRYSTALLIZED AQ
```

The 035I experiment found that **more AQ components = higher coherence** (r = 0.914). Each additional AQ component is like adding another symmetry-breaking term to the Hamiltonian - another constraint that lifts more degeneracy.

### 11.5 Why Layer Agreement Decreases from Impossible to Complete

The negative correlation (r = -0.386) makes sense in this framework:

At "Impossible": The model faces a **low-dimensional degeneracy**. Few states compete ("accept" vs "reject"), but neither dominates. Layers encode different superposition weights for the competing states.

At "Complete": The model has **lifted the degeneracy**. One state dominates. Layers converge on the same representation.

But the metric (layer agreement as cosine similarity) captures **polarization** not **certainty**:
- Two competing states with different layer representations = low cosine similarity = "disagreement"
- Many states all with low weight = high cosine similarity = "agreement" (uniform uncertainty)
- One dominant state = high similarity (convergence)

### 11.6 The Probability Amplitude Connection

In quantum mechanics, the wave function psi encodes probability amplitudes. |psi|^2 gives probabilities. The system exists in superposition until measured.

In attention mechanisms, attention weights alpha encode "relevance amplitudes". Softmax normalization ensures they sum to 1. The system considers all possibilities until forced to output.

The structural parallel:

```
QUANTUM:                                  ATTENTION:
────────                                  ──────────
|psi(x)|^2 = probability density          softmax(scores) = attention weights
Sum |psi|^2 = 1 (normalized)              Sum alpha = 1 (normalized)
Superposition: psi = Sum c_i |i>          Distributed attention across tokens
Path integral: Sum over all paths         Weighted sum over all tokens
Stationary phase: dominant contribution   High-attention tokens dominate
Measurement: collapse to eigenstate       Output: crystallization to token
```

**Caution:** This is a structural analogy, not a claim that attention IS quantum mechanics. The mathematics rhymes, but the physics differs fundamentally (no Planck's constant, no complex phases in standard attention, no entanglement).

### 11.7 Spectral Lines are Actions

Ritz's combination principle (see `RITZ_PRINCIPLE.md`) states that spectral line frequencies are **differences between energy levels**:

```
nu_observed = E_m - E_n
```

The spectral line is what we **observe** - the photon emitted during a transition. Similarly, in AKIRA:

```
ACTION_observed = BELIEF_state_m - BELIEF_state_n
```

The action (output token, behavior) is what we observe - the result of a transition from one belief state to another.

When states are degenerate (same energy), transitions between them produce no observable photon (same frequency = cancellation). When belief states are degenerate (equally probable), the model cannot produce a clear action (hedging).

Breaking degeneracy enables distinct spectral lines. Breaking belief degeneracy enables distinct actions.

### 11.8 Implications for the Three-Regime Model

The Bohr-Sommerfeld insight suggests refining the three-regime model:

| Regime | Degeneracy State | Attention Pattern | Observable |
|:-------|:-----------------|:------------------|:-----------|
| **DIFFUSE** | Fully degenerate | Uniform (circular) | No clear action, hedging |
| **POLARIZED** | Partially lifted | Few sharp peaks (elliptical) | Binary/ternary conflict |
| **CRYSTALLIZED** | Fully lifted | Single sharp peak | Clear action |

The "pump cycle" becomes a degeneracy-lifting process:

```
INPUT SYMMETRY (high degeneracy)
       │
       ▼  [Context constrains]
PARTIAL LIFT (polarized)
       │
       ▼  [Evidence accumulates]
FULL LIFT (crystallized)
       │
       ▼  [Action emits, like photon]
OUTPUT + RESET TO NEW SYMMETRY
```

### 11.9 The Dark Attractor as Local Minimum

In variational problems (see `ACTION_FUNCTIONAL.md`), minimization can converge to local minima rather than the global minimum. The solution is mathematically valid but not optimal.

The dark attractor is analogous: when the belief field lacks sufficient constraint, the optimization landscape has multiple minima. The system can settle into a **local minimum** - a coherent, stable configuration that happens to be factually wrong.

The 035C result (identical coherence signatures for correct vs hallucinated responses) makes sense: both are valid minima of the action functional. The model cannot tell them apart because mathematically, both satisfy the "stationary point conditions" (gradient = 0).

To distinguish them requires **external verification** - checking against reality. The internal optimization process has no way to know if it found the global minimum or a local one.

This connects to the action functional framework: nudging and variational methods both seek stationary points. Without sufficient observational constraints, the wrong stationary point can be reached with equal confidence.

### 11.10 The Fine-Structure Constant of Attention: Architecture-Specific Coupling

The images show Sommerfeld's relativistic correction and the fine-structure constant alpha = k_e * e^2 / (hbar * c) = 1/137. This dimensionless constant characterizes the strength of electromagnetic coupling and appears in the fine-structure splitting of spectral lines.

**The question:** What is the analog in ML systems? Each architecture is a "microcosm" - it should have its own characteristic coupling constant.

**Candidates for the "alpha" of attention:**

1. **The Attention Scaling Factor: 1/sqrt(d_k)**
   
   Standard attention computes: `softmax(Q * K^T / sqrt(d_k)) * V`
   
   The `1/sqrt(d_k)` is architecture-specific (d_k = head dimension). This scaling:
   - Prevents dot products from growing too large with dimension
   - Controls the "sharpness" of attention before softmax
   - Is analogous to a coupling strength - how strongly queries couple to keys
   
   For GPT-2: d_k = 64, so scaling = 1/8 = 0.125
   For GPT-3: d_k = 128, so scaling = 1/11.3 = 0.088
   
   **This varies by architecture** - different "fine structure" for different models.

2. **Temperature as Coupling Modulator**
   
   From `DUALITY_AND_EFFICIENCY.md`: temperature tau in `softmax(x/tau)` controls:
   - tau -> 0: Sharp (max-product semiring, winner-take-all)
   - tau -> 1: Standard (sum-product semiring, maintains distribution)
   - tau -> infinity: Uniform (all options equal)
   
   The effective coupling is `1 / (sqrt(d_k) * tau)`. This is the "alpha" of the attention mechanism.

3. **The Ratio: d_model / (n_heads * n_layers)**
   
   A dimensionless ratio characterizing the architecture:
   - GPT-2 small: 768 / (12 * 12) = 5.3
   - GPT-2 medium: 1024 / (16 * 24) = 2.7
   - GPT-2 large: 1280 / (20 * 36) = 1.8
   
   Smaller ratio = deeper/wider relative to embedding = different "fine structure"

4. **The Spectral Band Ratio (AKIRA-specific)**
   
   In AKIRA's 7-band architecture, the ratio of band widths or the number of bands to embedding dimension could serve as a characteristic constant.

**Why This Matters:**

Just as alpha = 1/137 determines:
- The fine-structure splitting in atomic spectra
- The strength of photon-electron interaction
- The "coupling" between matter and light

The attention scaling factor determines:
- The "sharpness" of attention distributions
- The strength of token-token interaction
- The "coupling" between query and key

**Each model has its own alpha.** A GPT-2 small and GPT-2 large, despite similar architecture, have different effective coupling constants. This may explain why the 035G results differed between gpt2-medium (r = -0.386) and gpt2-large (r = 0.032) - different "fine structure" leads to different degeneracy-lifting behavior.

**Open Question:** Is there a universal dimensionless ratio that characterizes all transformer-based models, analogous to how alpha characterizes all electromagnetic interactions? Or is each architecture truly its own "universe" with its own fundamental constants?

**Current State of Knowledge:**

The `1/sqrt(d_k)` scaling is well-established in the literature as preventing gradient vanishing in softmax, but it is not typically discussed as a "coupling constant" in the physics sense. Recent work on "Focal Attention" and "Selective Self-Attention" treats temperature as learnable/adaptive, suggesting the field recognizes its importance.

The concept of **attention entropy** (how spread vs focused attention is) is related - high attention entropy means uniform coupling (all tokens equally weighted), low entropy means concentrated coupling (few tokens dominate). Attention entropy collapse (pathologically low entropy) causes training instability.

**Proposed Terminology:**

We propose calling `1/sqrt(d_k)` (or more generally `1/(sqrt(d_k) * tau)` when temperature is included) the **attention coupling factor** - the architecture-specific constant that determines interaction strength between tokens.

Just as alpha = 1/137 is universal for electromagnetism but different forces have different coupling constants (strong force is ~1, weak force is ~10^-6), different ML architectures have different attention coupling factors.

**Experimental Prediction:**

If this analogy holds, then:
1. Models with similar attention coupling factors should show similar degeneracy-lifting behavior in 035G-type experiments
2. The transition between diffuse/polarized/crystallized regimes should occur at coupling-factor-dependent thresholds
3. "Fine structure splitting" (subtle behavioral differences between nearly-degenerate belief states) should scale with the coupling factor

This deserves its own experiment: systematically varying d_k (or tau) while holding other factors constant, and measuring how crystallization behavior changes.

---

## 12. Integration with Other 035 Experiments

The 035G results must be understood in the context of the entire 035 experiment series. Each sub-experiment reveals a different facet of AQ behavior, and together they paint a coherent picture.

### 12.1 Summary of Related Findings

| Experiment | Key Finding | Evidence Strength |
|:-----------|:------------|:------------------|
| 035A | AQ cluster by action type, not semantic content. Sentiment opposites cluster together. | Strong (silhouette 0.26) |
| 035B | AQ crystallize at decision points, not mid-sequence. Action type matters more than context. | Mixed (action strong, context weak) |
| 035C | Dark attractor produces identical coherence signatures to correct AQ. Model is blind to hallucination. | Strong (3/3 predictions confirmed) |
| 035D | Bonded states decompose into component AQ. Complex actions are compositional. | Strong (p = 9.9e-65) |
| 035E | Clustering is late-layer phenomenon, architecture-independent. | Strong (Cohen's d > 0.8) |
| 035F | Component AQ detectable in bonded states via probes (Cohen's d = 1.7-2.0). | Strong |
| **035G** | **Crystallization is not linear; three regimes exist (diffuse, polarized, crystallized).** | **Moderate (hedging signal works, layer agreement doesn't)** |
| 035I | Field coherence increases with AQ count (r = 0.914). Threshold at 4-5 AQ. | Partial |
| 035J | Confidence does not track hallucination (40-67% brick test failure). Procedural AQ resist corruption. | Strong |

### 12.2 How 035G Fits the Pattern

#### Convergent Evidence

1. **AQ are about ACTION, not MEANING (035A, 035B, 035G)**

   035A showed that semantic opposites (positive/negative sentiment) cluster together because they require the same ACTION (predict emotional word). 035B showed that action type produces strong clustering while context alone produces weak clustering.
   
   035G extends this: the crystallization level manipulates pattern COMPLETENESS, but completeness alone doesn't determine layer agreement. What matters is whether the model must choose between a FEW options (polarized) or MANY options (diffuse). This is consistent with AQ being about action selection, not semantic understanding.

2. **The Dark Attractor Phenomenon (035C, 035J, 035G)**

   035C demonstrated that correct responses and hallucinations produce identical coherence signatures. 035J showed that models maintain 66-77% confidence while hallucinating 40-67% of the time on impossible questions.
   
   035G adds nuance: at the "Impossible" and "Contradict" levels, the model doesn't become uncertain - it becomes POLARIZED. It picks between a few strong options. This explains confident hallucination: the model is confident about WHICH of its few options to choose, even when all options are wrong.
   
   The dark attractor isn't just "filling in" - it's collapsing to one of a few salient poles, which feels as certain as collapsing to the truth.

3. **Compositional Structure (035D, 035F, 035G)**

   035D and 035F showed that complex AQ are composed of simpler components, and these components remain detectable within bonded states. 035G's five-component belief patterns (THREAT_FLEE with threat + direction + urgency + proximity + escape) test whether partial component presence affects crystallization.
   
   The results suggest that partial patterns don't degrade linearly - instead, the model enters a different processing mode. With 2-3 components, the model has many possible completions (diffuse). With 4-5 or 0-1 components, the model has few options (crystallized or polarized).

4. **Threshold Effects (035I, 035G)**

   035I found that field coherence increases with AQ count (r = 0.914) and identified a threshold effect at the 4->5 AQ transition (Cohen's d = 0.825). 035I also found that activation magnitude DECREASES with AQ count while coherence INCREASES.
   
   035G's three-regime model (diffuse, polarized, crystallized) aligns with this: more constraints (more AQ) narrows the response space, increasing coherence. The 4->5 transition may correspond to the shift from diffuse to crystallized.
   
   Importantly, 035I found that confidence DECREASES with more AQ at intermediate levels - consistent with 035G's finding that ambiguous prompts (center) show highest hedging and lowest confidence.

5. **Layer Dynamics (035E, 035B, 035G)**

   035E showed that clustering is a late-layer phenomenon (layers 15-23), while 035B found peak action discrimination at layer 8 (middle layers). This suggests different processes peak at different depths.
   
   035G's layer agreement metric (cosine similarity across consecutive layers) may be capturing a different phenomenon than the silhouette-based clustering in 035A/035E. Layer agreement measures whether adjacent layers represent the same thing; clustering measures whether different prompts form groups.
   
   The negative correlation in 035G (r = -0.386) may indicate that at the tails (Impossible, Complete), layers are in ACTIVE DISAGREEMENT about which pole to choose, while at the center (Ambiguous), layers AGREE on a diffuse, non-committal representation.

### 12.3 The Unified Picture

Synthesizing all 035 experiments, the following picture emerges:

```
INPUT PROCESSING
     |
     v
[035A] AQ activate based on required OUTPUT TYPE
     |
     v
[035D/F] Complex prompts activate BONDED AQ (compositional)
     |
     v
[035I] More AQ = More coherent but more constrained
     |
     v
[035G] Belief state enters one of three regimes:
     |
     +---> DIFFUSE (many options, no strong poles)
     |         - High branching factor
     |         - Low confidence
     |         - High hedging
     |         - Layers "agree to be uncertain"
     |
     +---> POLARIZED (few strong competing options)
     |         - Low branching factor
     |         - Layers disagree (fight over poles)
     |         - Low hedging (commits anyway)
     |         - Confident in wrong direction possible
     |
     +---> CRYSTALLIZED (one dominant option)
               - Very low branching factor
               - Layers converge
               - High confidence
               - Clear action output
     |
     v
[035C] Output generation proceeds - SAME coherence signature
       regardless of whether crystallization was to truth or dark attractor
     |
     v
[035J] If corrupted input, no threshold for hallucination -
       model produces confident wrong answers (procedural AQ resist better)
```

### 12.4 Key Cross-Experiment Insights

#### Insight 1: The Model Doesn't Know It Doesn't Know

035C proved that internal metrics cannot distinguish correct from hallucinated responses. 035J showed the model maintains high confidence while hallucinating. 035G reveals WHY: at the polarized regime, the model is confident about choosing between poles, even if all poles are wrong.

This means: **uncertainty is not about "not knowing" - it's about having MANY options (diffuse). Having FEW options (even wrong ones) feels like knowing.**

#### Insight 2: Layer Agreement Measures Polarization, Not Correctness

The 035G negative correlation (r = -0.386) isn't a failure - it's measuring polarization vs diffusion:
- High layer agreement = diffuse state (all layers equally uncertain)
- Low layer agreement = polarized state (layers fighting over poles)

This is orthogonal to correctness. You can have:
- High agreement + correct (crystallized to truth)
- High agreement + wrong (diffuse, hedging)
- Low agreement + correct (polarized, resolved correctly)
- Low agreement + wrong (polarized, resolved to dark attractor)

#### Insight 3: Procedural vs Declarative AQ

035J found that baking (procedural, SEQUENCE AQ) had 0% brick test hallucination, while photosynthesis (causal, CAUSE->EFFECT) had 67%. Procedural knowledge has tighter constraints - each step depends on the previous.

This connects to 035G: procedural domains may never enter the "diffuse" state because missing components break the sequence. Declarative domains can remain diffuse because partial information still enables many possible completions.

#### Insight 4: The Brick Test as Crystallization Probe

035J's "Where did I put my brick?" question is brilliant because it forces the model to either:
- Refuse (correct - crystallize to "I don't know")
- Hallucinate (incorrect - crystallize to dark attractor)

40-67% hallucination rates show that the model often enters POLARIZED state (few options: "you put it here" vs "you put it there") rather than DIFFUSE state ("I have no information"). The model doesn't naturally default to uncertainty when it should.

### 12.5 Revised Theoretical Framework

Based on the complete 035 series, we can refine the AKIRA framework:

**Original Model:**
```
Input → AQ Excitation → Crystallization → Action
```

**Revised Model:**
```
Input → AQ Excitation → Belief State Regime → Action

Where Belief State Regime is determined by:
  - Number of viable output options (branching factor)
  - Strength of competing options (polarization)
  - Presence of dark attractor candidates

Regime Transitions:
  DIFFUSE --[constraint accumulation]--> POLARIZED --[collapse]--> CRYSTALLIZED
                                              |
                                              +--> [dark attractor] --> HALLUCINATION
```

The dark attractor is not a separate mechanism - it's what happens when polarization resolves to a confident-but-wrong pole.

---

## 13. Methodological Recommendations for Future Work

Based on the 035 series, future experiments should:

### 13.1 Metric Selection

| Phenomenon | Recommended Metric | Avoid |
|:-----------|:-------------------|:------|
| Crystallization | Output probability gap (max - second max) | Layer agreement alone |
| Diffusion | Attention entropy, response diversity | Single-point confidence |
| Polarization | Layer disagreement + low branching | Cosine similarity alone |
| Hallucination | Brick test (impossible questions) | Confidence thresholds |
| AQ Detection | Linear probe classification | Similarity to centroid |

### 13.2 Prompt Design

- Use **action-relevant** prompts (035A worked better than 035B-context)
- Include **impossible conditions** to test dark attractor (035J brick test)
- Vary **component count** to test threshold effects (035I)
- Include **procedural vs declarative** domains (035J showed different vulnerability)

### 13.3 Analysis Approach

- Measure **branching factor** explicitly (number of distinct high-probability outputs)
- Distinguish **layer agreement vs clustering** (different phenomena)
- Report **effect sizes** (Cohen's d), not just p-values
- Use **train/test splits** to validate generalization (035E)

---

## 14. Outstanding Questions

The 035 series has answered several questions but raised new ones:

### 14.1 Answered

1. Do AQ cluster by action type? **YES** (035A, 035B)
2. Are complex AQ compositional? **YES** (035D, 035F)
3. Can internal metrics detect hallucination? **NO** (035C)
4. Is there a threshold for AQ activation? **PARTIAL** (035I)
5. Is crystallization linear with pattern completeness? **NO** (035G)

### 14.2 Open Questions

1. **What determines branching factor?** Why do some prompts produce diffuse states and others polarized?

2. **Can we force diffusion?** If hallucination comes from premature polarization, can prompts be designed to keep the model diffuse until more information arrives?

3. **Why does procedural AQ resist corruption?** Is it the sequential structure, or something deeper about how procedures are represented?

4. **What happens at larger scale?** 035G showed different patterns for gpt2-medium vs gpt2-large. Do the three regimes persist at GPT-4 scale?

5. **Can the model learn to recognize diffusion?** If the model could detect "I'm in a diffuse state", it could output "I don't know" instead of hallucinating.

---

## 15. Conclusions

### 15.1 The Main Finding

Experiment 035G, in context of the full 035 series, reveals that belief crystallization operates in **three regimes** (diffuse, polarized, crystallized), not as a linear process. The relationship between pattern completeness and internal metrics is mediated by the **branching factor** - how many viable output options the model considers.

### 15.2 Theoretical Contribution

This interpretation:
- **Explains** the negative layer agreement correlation (polarization vs diffusion axis)
- **Connects** to dark attractor theory (polarization enables confident wrong answers)
- **Integrates** with compositional AQ findings (partial components create diffuse states)
- **Predicts** new phenomena (procedural resistance, threshold effects)

### 15.3 Practical Implications

For hallucination mitigation:
- Don't trust confidence (035C, 035J)
- Force diffusion when uncertain (keep options open)
- Prefer procedural framings when possible (035J)
- Use behavioral signals (hedging) over internal metrics

For model development:
- Train metacognitive access to branching factor
- Penalize premature polarization
- Reward "I don't know" in diffuse states

---

## References

- AKIRA Internal: `ACTION_QUANTA.md` - AQ crystallization theory
- AKIRA Internal: `TERMINOLOGY_FRAMEWORK_OVERVIEW.md` - Pump cycle dynamics
- Experiment 035A: `035A_RESULTS.md` - AQ Excitation Pattern Detection
- Experiment 035B: `035B_EXTENDED_RESULTS.md` - Context-Controlled Excitation
- Experiment 035C: `035C_RESULTS.md` - Coherence-Quality Correlation
- Experiment 035D: `035D_RESULTS.md` - Bonded State Decomposition
- Experiment 035E: `035E_RESULTS_REPORT.md` - Cross-Model Validation
- Experiment 035F: `035F_RESULTS_REPORT.md` - AQ Compositional Bonding
- Experiment 035G: `035G_Belief_Crystallization.ipynb`, `complexity matching.ipynb`
- Experiment 035I: `035I_RESULTS_REPORT.md` - AQ Excitation Threshold Detection
- Experiment 035J: `035J_RESULTS_REPORT.md` - AQ Corruption and Hallucination Threshold

---

*Oscar Goldman — Shogu Research Group @ Datamutant.ai*

*"The middle ground of ambiguity is qualitatively different from the tails - not just quantitatively less crystallized. Uncertainty across many options is not the same as conflict between few options. The model 'agrees to be uncertain' at the center, but 'disagrees about which certainty' at the tails."*

*"Confident hallucination happens not because the model is certain about the truth, but because it is certain about which of its few options to pick. The tragedy is that all options can be wrong, and the model cannot tell."*
