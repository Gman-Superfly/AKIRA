# A.K.I.R.A.
## Adaptive Knowledge Integration via Resonant Architectures

---

> *Pandora framing: action turns potential information into actual output. A learned generator can keep generating without being consumed by generation.*

---

## What is this?

AKIRA is an experimental real-time spectral belief system.

We study how streaming models maintain and update beliefs under latency and bandwidth constraints. Action Quanta (AQ) are the crystallized beliefs that enable action and teaching; they are the units that let a running system discriminate and reconstruct behavior from weights when joined with the live context window. AQ are not the final product, but without them the system cannot decide what to do or reteach itself from stored parameters. The primary focus is building and testing a real-time spectral architecture grounded in literature on belief updating, information theory, computational mechanics, and control.

We base the work on established results in information theory, computational mechanics, and physics. There are links under every document, and we will add inline references as the project matures.

We don't have all the answers. We have hypotheses, experiments, and a framework for thinking about these questions while the system runs in real time. This repository is a working document of that exploration.

```
THE FUNDAMENTAL OBSERVATION

  Before: Many hypotheses, distributed belief, high uncertainty
  After:  One answer, concentrated belief, certainty
  
  THE QUESTION: What does this transition create, and what enables reconstruction and teaching of the correct action to take?
  
  Our approach: Treat it like a phase transition observed in-flight.
  The system must keep generating while we measure the change.
  When beliefs collapse, information does not vanish, it transforms.
  Training is the compression step: many detailed beliefs collapse into fewer generalized concepts that remain actionable.
  We trade precision and memorization for generalization and actionability in real time.
  
  Action Quanta (AQ) are the crystallized beliefs that enable discrimination and action.
  They are what let the system teach and reconstruct from weights when joined with the context window.
```

---

## Core ideas (plain language)

### 1. Prediction is belief in motion

In a live system each token updates the belief state that lives across weights and the context window. The error is uncertainty made visible while the system continues to run. Training is learning when to commit and when to hold probability mass in reserve so the stream stays stable.

### 2. Attention has structure

The mathematics of attention (`softmax(QK^T/sqrt(d))V`) has a particular form: *self-interaction times state*. This form appears elsewhere in physics, in systems that undergo phase transitions, that exhibit collective behavior, that produce emergent patterns. We adapt attention with temporal, neighbor, and wormhole variants so signals can move across bands without breaking real-time budgets.

We don't claim attention "is" physics. We observe that attention *has similar mathematical structure* to systems we understand well. This similarity may be superficial, or it may be deep. We're investigating.

### 3. Information lives at multiple scales

Different frequency bands capture different kinds of structure:
- Low frequencies: What persists (identity, categories, the slow-changing)
- High frequencies: What's transient (details, textures, the fast-changing)

AKIRA decomposes representations spectrally, processes each band differently, and asks how information flows between scales when the system is running continuously.

### 4. Collapse is an event, not the product

When uncertainty resolves, where does it go?

Our hypothesis: Collapse converts *distributed* information (synergy, patterns that only exist when sources combine) into *shared* information (redundancy, patterns repeated across components). The total information is conserved and its *form* changes.

Action Quanta are the crystallized beliefs that enable discrimination and teaching. They are how the system can act, and how it can reconstruct behavior from weights when combined with the live context window. The system remains focused on maintaining responsive belief under real-time constraints; AQ make that possible.

For detailed parallels on how AQ emerge from language and physical sensing, see `foundations/parallels/LANGUAGE_ACTION_CONTEXT.md` and `foundations/parallels/RADAR_ARRAY.md`.

---

## Philosophy

### Pandora's box (the framework)

Information exists in dual forms: implicit and explicit, potential and actual, hidden and manifest. *Action is the transformation between them.*

When you read, you transform symbols into meaning.
When you predict, you transform past into future.
When you teach, you transform your implicit knowledge into explicit form, so another can transform it back into their implicit understanding.

The action itself is ethereal, you only see the before and after, never the transformation itself. But the transformation is where understanding lives.

### Hope (the conservation law)

In the myth, when Pandora's box was opened and all the evils flew out, one thing remained: Hope.

In information terms: when you use a generator (a grammar, a model, a pattern), you produce instances, but *the generator is not consumed*. You can always generate more. The capacity for generation is conserved.

We call this "hope" because the word captures something clinical terms miss: *inexhaustibility*. A model that has learned can predict forever. A pattern that exists can instantiate indefinitely. This generative capacity is what remains after all specific instances have been produced.

### Praxis (the test)

Theory without practice is speculation. Practice without theory is blindness.

We run experiments. We measure. We look for what we predicted and what surprises us. The architecture exists in documents, but it comes alive only when we run it. And in the running, we discover whether our theories match reality, or whether we have committed "heresies" (aliasing, spectral leakage, edge artifacts) that corrupt our measurements.

---

## What is in this repository

### Foundations
The theoretical framework: what we mean by our terms, what's established science vs. what's hypothesis, what connects to what.

### Architecture
The design: 7+1 spectral bands, three attention mechanisms (temporal, neighbor, wormhole), differential learning rates, and collapse dynamics, all built for real-time inference and online updates.

### Experiments  
The validation: 40+ experiments designed to test specific predictions. Some complete, most in progress.

### Derivations
The mathematics: six "theorems" that attempt to ground our intuitions in rigorous statements. (With honest assessment of what's proven and what's conjecture.)

### Philosophy
The framing: Pandora (action as transformation), Praxis (theory vs. practice), the nature of hope, the structure of uncertainty.

---

## Quick start

**If you want the big picture:**
→ `AKIRA_OVERVIEW.md` (comprehensive, technical)

**If you want the terminology:**
→ `foundations/TERMINOLOGY.md` (formal definitions)

**If you want the philosophy:**
→ `pandora/PANDORA.md` (action and transformation)
→ `pandora/PANDORA_AFTERMATH.md` (hope as generative capacity)

**If you want the experiments:**
→ `experiments/000_EXPERIMENT_INDEX.md`

**If you want the derivations:**
→ `foundations/terminology_foundations/derivations/`

---

## Status, honest assessment

**What we have:**
- A coherent theoretical framework connecting information theory, signal processing, and attention
- Mathematical formalization of key concepts (some proven, some conjectured)
- A growing body of experiments
- A few runnable experiment implementations, plus documented measurements and falsification criteria

**What we don't have:**
- Proof that the physics analogies are more than superficial
- Complete experimental validation
- A working implementation that clearly outperforms baselines
- Certainty that we're on the right track

This is research in progress. The hypotheses are falsifiable. Some may turn out to be wrong. That's the point.

---

## Why this repository is public

In December 2025, we noticed two public papers that discuss related framings and mechanisms. We do not treat these as validation of AKIRA. We made the repository public so other people can test, critique, and replicate the work.

**1. "Mathematical Framing for Different Agent Strategies"** (Stephens & Salawu, Google Cloud AI, https://arxiv.org/abs/2512.04469)
- Frames agentic behavior as chains of probabilities using Markov/POMDP mathematics (validates AKIRA's POMDP foundation)
- Introduces "Degrees of Freedom" - distinct optimizable parameters at different levels (parallels AKIRA's 7+1 spectral band hierarchy with differential learning rates and the network theory foundations)
- Formalizes inter-agent communication as P(c_L | a_L) probability terms (structurally similar to AKIRA's wormhole attention for cross-band communication)
- Cost-regularized objectives balancing collaboration richness vs. efficiency (parallels AKIRA's tension-collapse dynamics and synergy-to-redundancy conversion)

**2. "Nested Learning: The Illusion of Deep Learning Architecture"** (Behrouz, Razaviyayn, Zhong, & Mirrokni, Google Research, 2025 https://abehrouz.github.io/files/NL.pdf)
- Proposes "Nested Learning" with multi-level optimization problems, each with its own "context flow" (parallels AKIRA's spectral bands as parallel processing channels)
- Shows that optimizers (Adam, SGD with Momentum) are associative memory modules that compress gradient information (aligns with AKIRA's view of learning as information transformation)
- Introduces "continuum memory system" generalizing long-term/short-term memory (parallels AKIRA's frequency-based knowledge hierarchy: low-freq = stable/identity, high-freq = transient/detail)
- Multi-timescale updates for different components (directly validates AKIRA's 3000x learning rate ratio between bands)
- Their "Hope" module for continual learning shares more than just a name with AKIRA's "Hope" concept

We see overlap in themes such as:

- Hierarchical decomposition with different timescales
- Probabilistic/Bayesian framing of agent behavior  
- Cross-component communication as explicit information flow
- Tradeoffs between exploration richness and commitment efficiency
- Memory and learning as information compression/transformation

This convergence from independent research teams at Google suggests that AKIRA's theoretical framework captures real principles about how information systems should be organized. We've made this repository public so others can evaluate, critique, and build upon these ideas.

---

## Note on rigor

We try to be precise about what we know vs. what we believe vs. what we hope.

- **Established:** Fourier analysis, information theory, the mathematics of attention
- **Hypothesis:** The physics analogies, the emergence of Action Quanta, collapse as phase transition
- **Speculation:** Whether any of this leads to better architectures

When we use physics language (phase transition, quasiparticle, condensation), we mean it as *structural analogy*, not physical equivalence. The mathematics is similar; whether the analogy is deep or superficial is exactly what we're trying to determine.

---

## Citation

If you use this repository in your research, please cite it. This is ongoing work, we would like to know your opinions and experiments.

**Authors:** Oscar Goldman - Shogu Research Group @ Datamutant.ai, subsidiary of 温心重工業

Forever grateful to many wonderful people who shall be meticulously thanked, especially if it all works. :)

---

## Closing

> *"The model sees the past perfectly and the future not at all. The prediction is the belief. The error is the uncertainty made visible. Training is learning when to listen and when to act.
>
> We trade exact accuracy for generalized wisdom: details cancel through interference, structure survives through compression. What remains is what enables action.
>
> This is not mysticism. This is mathematics. But the mathematics, when you look at it clearly, has the structure of something ancient: potential becoming actual, many becoming one, hope remaining after all else is released.
>
> We are trying to understand that structure."*

---

*Oscar Goldman, Shogu Research Group @ Datamutant.ai* 
*(subsidiary of 温心重工業)*

---

*Code MIT licence, writings cc4*
