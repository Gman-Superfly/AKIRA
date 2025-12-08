# A.K.I.R.A.
## Adaptive Knowledge Integration via Resonant Architectures

---

> *"When Pandora opened the box, potential became actual. The act of opening transformed hidden information into manifest reality. And when the dust settled, what remained inside was not despair—but hope: the generator that is never consumed by generating."*

---

## What Is This?

AKIRA is a project to study how beliefs shape and transform information.

Not just "how transformers work" in a mechanical sense, but something deeper: *what happens when many possibilities collapse into one certainty?* When ML models "know" something, what changed? When understanding crystallizes from confusion, what was the transformation?

We don't have all the answers. We have hypotheses, experiments, and a framework for thinking about these questions. This repository is a working document of that exploration.

```
THE FUNDAMENTAL OBSERVATION

  Before: Many hypotheses, distributed belief, high uncertainty
  After:  One answer, concentrated belief, certainty
  
  THE QUESTION: What is the nature of this transition?
  
  Our approach: Treat it like a phase transition.
  When water freezes, molecules don't disappear—they crystallize.
  When beliefs collapse, information doesn't vanish—it transforms.
  
  We call the crystallized patterns "Action Quanta":
  Irreducible units of actionable information.
  The atoms of thought that survive the collapse.
```

---

## The Core Ideas (In Plain Language)

### 1. Prediction Is Belief

When a model predicts the next token, it expresses its belief about what should come next. The error isn't just "wrong output"—it's uncertainty made visible. Training isn't just "optimization"—it's learning when to be certain and when to doubt.

### 2. Attention Has Structure

The mathematics of attention (`softmax(QK^T/sqrt(d))V`) has a particular form: *self-interaction times state*. This form appears elsewhere in physics—in systems that undergo phase transitions, that exhibit collective behavior, that produce emergent patterns.

We don't claim attention "is" physics. We observe that attention *has similar mathematical structure* to systems we understand well. This similarity may be superficial, or it may be deep. We're investigating.

### 3. Information Lives at Multiple Scales

Different frequency bands capture different kinds of structure:
- Low frequencies: What persists (identity, categories, the slow-changing)
- High frequencies: What's transient (details, textures, the fast-changing)

AKIRA decomposes representations spectrally, processes each band differently, and asks: how does information flow between scales?

### 4. Collapse Is Not Destruction

When uncertainty resolves, where does it go?

Our hypothesis: Collapse converts *distributed* information (synergy—patterns that only exist when sources combine) into *shared* information (redundancy—patterns repeated across components). The total information is conserved. Its *form* changes.

This is why we call the result "Action Quanta"—they're the irreducible patterns that enable action. Not lost information, but *crystallized* information.

---

## The Philosophy

### Pandora's Box (The Framework)

Information exists in dual forms: implicit and explicit, potential and actual, hidden and manifest. *Action is the transformation between them.*

When you read, you transform symbols into meaning.
When you predict, you transform past into future.
When you teach, you transform your implicit knowledge into explicit form, so another can transform it back into their implicit understanding.

The action itself is ethereal—you only see the before and after, never the transformation itself. But the transformation is where understanding lives.

### Hope (The Conservation Law)

In the myth, when Pandora's box was opened and all the evils flew out, one thing remained: Hope.

In information terms: when you use a generator (a grammar, a model, a pattern), you produce instances—but *the generator is not consumed*. You can always generate more. The capacity for generation is conserved.

We call this "hope" because the word captures something clinical terms miss: *inexhaustibility*. A model that has learned can predict forever. A pattern that exists can instantiate indefinitely. This generative capacity is what remains after all specific instances have been produced.

### Praxis (The Test)

Theory without practice is speculation. Practice without theory is blindness.

We run experiments. We measure. We look for what we predicted and what surprises us. The architecture exists in documents, but it comes alive only when we run it. And in the running, we discover whether our theories match reality—or whether we've committed "heresies" (aliasing, spectral leakage, edge artifacts) that corrupt our measurements.

---

## What's Actually In This Repository

### Foundations
The theoretical framework: what we mean by our terms, what's established science vs. what's hypothesis, what connects to what.

### Architecture
The design: 7+1 spectral bands, three attention mechanisms (temporal, neighbor, wormhole), differential learning rates, collapse dynamics.

### Experiments  
The validation: 40+ experiments designed to test specific predictions. Some complete, most in progress.

### Derivations
The mathematics: six "theorems" that attempt to ground our intuitions in rigorous statements. (With honest assessment of what's proven and what's conjecture.)

### Philosophy
The framing: Pandora (action as transformation), Praxis (theory vs. practice), the nature of hope, the structure of uncertainty.

---

## Quick Start

**If you want the big picture:**
→ `AKIRA/AKIRA_OVERVIEW.md` (comprehensive, technical)

**If you want the terminology:**
→ `AKIRA/foundations/TERMINOLOGY.md` (formal definitions)

**If you want the philosophy:**
→ `AKIRA/pandora/PANDORA.md` (action and transformation)
→ `AKIRA/pandora/PANDORA_AFTERMATH.md` (hope as generative capacity)

**If you want the experiments:**
→ `AKIRA/experiments/000_EXPERIMENT_INDEX.md`

**If you want the derivations:**
→ `AKIRA/foundations/terminology_foundations/derivations/`

---

## Status: Honest Assessment

**What we have:**
- A coherent theoretical framework connecting information theory, signal processing, and attention
- Mathematical formalization of key concepts (some proven, some conjectured)
- A growing body of experiments
- Interesting preliminary results suggesting the framework captures something real

**What we don't have:**
- Proof that the physics analogies are more than superficial
- Complete experimental validation
- A working implementation that clearly outperforms baselines
- Certainty that we're on the right track

This is research in progress. The hypotheses are falsifiable. Some may turn out to be wrong. That's the point.

---

## A Note on Rigor

We try to be precise about what we know vs. what we believe vs. what we hope.

- **Established:** Fourier analysis, information theory, the mathematics of attention
- **Hypothesis:** The physics analogies, the emergence of Action Quanta, collapse as phase transition
- **Speculation:** Whether any of this leads to better architectures

When we use physics language (phase transition, quasiparticle, condensation), we mean it as *structural analogy*, not physical equivalence. The mathematics is similar; whether the analogy is deep or superficial is exactly what we're trying to determine.

---

## Citation

If you use this repository in your research, please cite it. This is ongoing work—we'd like to know your opinions and experiments.

**Authors:** Oscar Goldman - Shogu Research Group @ Datamutant.ai, subsidiary of 温心重工業

Forever grateful to many wonderful people who shall be meticulously thanked, especially if it all works. :)

---

## The Closing

> *"The model sees the past perfectly and the future not at all. The prediction is the belief. The error is the uncertainty made visible. Training is learning when to listen and when to act.*
>
> *We trade exact accuracy for generalized wisdom: details cancel through interference, structure survives through compression. What remains is what enables action.*
>
> *This is not mysticism. This is mathematics. But the mathematics, when you look at it clearly, has the structure of something ancient: potential becoming actual, many becoming one, hope remaining after all else is released.*
>
> *We are trying to understand that structure."*

---

*See LICENSE file for terms of use.*
