# Naming new concepts: extraction, coordination, and discovery

## Document purpose

This document defines a practical mechanism for concept naming in AKIRA style systems.

The problem is simple to state:

- A learned model can contain useful distinctions as distributed structure in weights and activations.
- If the system lacks stable *handles* for those distinctions, then it often cannot retrieve, transmit, or train against them reliably.

This document argues that *naming* is one way to create such handles, and that it can be implemented without editing the base model weights by operating on logits using a reflex layer (for example, a Homeostat style energy relaxation layer).

This document sits in `foundations/parallels/` because it is another instance of the same parallel structure used in:

- `LANGUAGE_ACTION_CONTEXT.md`
- `COMPUTATIONAL_MECHANICS_EQUIVALENCE.md`
- `RADAR_ARRAY.md`

The core mapping is:

```
SIGNAL / DATA                  MEANING / BELIEF                    ACTION
──────────────────────────     ───────────────────────────────      ───────────────────────────
Raw observation                Internal representation              Actionability
(tokens, pixels, time series)  (distributed patterns)              (AQ, discrimination, decision)

Naming is a bridge:
Naming creates a stable handle that lets the system and the user talk about a crystallized
discrimination, and then use it as a reusable building block.
```

## Table of contents

1. The problem history
2. Geometry of color space and perception
3. Symbols, pictograms, and external representations
4. Mapping to AKIRA terms
5. The proposal for LLM and Homeostat hybrid systems
6. How this can enable discovery
7. The core claim, stated in AKIRA terms
8. The human side, stated precisely
9. Hidden patterns in large models, and what it means to "unlock" them
10. A concrete proposal: concept handles for LLM plus reflex layers
11. A protocol for creating and stabilizing new names
12. Verification and experiments
13. Failure modes and risks
14. References

## The problem history

### Human cognition already uses naming as a control primitive

Humans do not only perceive. Humans also build shared handles for what they perceive.
Those handles include words, diagrams, notations, and symbols.

The value of a handle is not that it adds new sensory input.
The value is that it makes a distinction:

- easier to retrieve from memory,
- easier to communicate,
- easier to teach,
- easier to compose into larger structures, and
- easier to make stable across contexts.

This is why technical fields invent notations.
It is also why artists draw.
It is also why engineers write down constraints.

### Color naming as the motivating example

Color is useful because it separates three things that are often confused:

1. Physical stimulus (light spectrum hitting the eye).
2. Perceptual representation (the brain's encoding).
3. Linguistic category (the names used in a community).

If the category changes, then the physics does not change.
The representation can change in subtle ways, and the behavior can change in large ways.

Important note:
Some popular stories about ancient peoples and the color blue get retold in a stronger form than
the evidence supports. The careful claim that matters for this project is weaker and more useful:

- If a community does not share a stable label for a region of a continuous space, then
  discriminations inside that region often become slower, less reliable, and less transmissible.
- If a community does share a stable label, then it can train attention, memory, and teaching
  around that label, and discrimination can improve.

This is the version that transfers to machine learning without relying on folklore.

## Geometry of color space and perception

### Physical stimulus does not equal perceptual space

Light is continuous in wavelength, but human color perception is not a simple wavelength meter.

Human trichromatic vision begins with three cone response curves (often called L, M, and S cones).
The mapping from a spectrum to cone responses loses information.
Different spectra can produce the same cone triplet.
These are metameric matches.

So the pipeline already contains a compression bottleneck:

```
Spectrum (high-dimensional)  →  Cone responses (3 numbers)  →  Perceptual representation
```

That is the first reason color is a good parallel for ML systems.
Model representations are also compressions.

### A usable color space needs a geometry

If we want to talk about discrimination, we need a notion of distance.

Color science uses several spaces that approximate perceptual distances:

- CIE XYZ: a linear space tied to standardized observer matching functions.
- CIE Lab and CIE Luv: spaces designed so that Euclidean distances better approximate perceived
  differences than XYZ does.

The detail that matters here is not the exact formula.
The detail that matters is the concept:

*A perceptual space is a geometry that supports discrimination.*

### Categories carve a continuous space into decision regions

Even if perception is continuous, decisions are often discrete.

If a system must decide "green" versus "blue", then it must place a boundary.
Where the boundary goes depends on:

- the task,
- the costs of mistakes,
- the distribution of inputs, and
- the shared naming system used to teach the boundary.

That is exactly the AQ definition stated elsewhere in AKIRA documents:

An AQ must discriminate between action alternatives.
If it does not enable a choice, then it is not an AQ.

So in the color parallel:

- The continuous space is signal or belief.
- The category boundary is a discrimination.
- The label "green" is a handle for that discrimination.

## Symbols, pictograms, and external representations

### A name is not only a sound

Japanese provides a clean example:
Different words can share a sound, and the writing system disambiguates meaning using characters.

The general point is:

The handle is not only an acoustic pattern.
The handle is a representation that supports stable retrieval and stable composition.

This matters for ML systems because "token strings" are not the only kind of handle we can create.
We can create:

- concept IDs,
- pictograms,
- structured tags,
- small sketches or diagrams,
- vector prototypes, and
- constraint objects in a reflex layer.

The correct question is not "what is the token".
The correct question is "what handle supports the discrimination we need".

### Pictograms, notation, and why the medium matters

Pictograms and writing systems matter here because they show that:

- the same sound can carry different meanings (homophony),
- the same meaning can be expressed in many media (speech, writing, diagrams), and
- some media support some operations better than others.

Examples:

- Pictograms and icons support fast recognition and coarse categorization.
- Algebraic notation supports manipulation of structure without re-deriving meaning each time.
- Diagrams support constraints and relationships that are hard to hold in working memory.

For the naming proposal, this suggests a design rule:

If a concept must be used for control, then the handle should live in a medium that supports control.
For a Homeostat style reflex layer, that medium is a structured constraint object operating on logits.

### Symbolic logic as a teaching tool

Symbolic logic does not replace informal reasoning.
It gives a handle for structure that is otherwise hard to keep stable in working memory.

Examples:

- A truth table externalizes a decision rule.
- A quantified statement externalizes scope ("for all" and "there exists").
- A proof externalizes a dependency chain.

These externalizations matter because they let many minds coordinate on the same structure.
They also matter because they let one mind coordinate with its own future self.

### Drawing and diagramming as computation

Artists sketch to see what they cannot yet name.
Engineers diagram to make constraints explicit.

A sketch is a temporary representation that can precede a name.
Then a name stabilizes the sketch into a reusable unit.

In this document, that order is important:

1. First, the system must create a repeatable internal discrimination.
2. Then, it can name that discrimination.
3. Then, the name can participate in learning, retrieval, and composition.

Naming without a stable discrimination is only vocabulary expansion.
It is not concept formation.

### Drawing as a cognitive tool for concept formation and communication

The paper Fan, Bainbridge, Chamberlain, and Wammes (2023) supports a specific claim that we need
for this document:

Drawing externalizes internal representations into visible artifacts, and that externalization can
support perception, memory, learning, and communication in different ways depending on context.

This matters for concept naming because the naming workflow we want has the same shape:

1. Externalize a candidate distinction (draw it, diagram it, or otherwise render it as an object).
2. Stabilize it with a handle (name plus operational definition).
3. Use it for coordination (between people, or between modules).

The important point is not that drawings resemble reality.
The important point is that drawings can function as flexible external representations, including
schematic diagrams, that make otherwise hidden mental content inspectable and manipulable.

## Mapping to AKIRA terms

### Definition: handle

In this document, a *handle* means:

*A compact representation that lets a system reliably select, reuse, and communicate a specific
discrimination or abstraction.*

Handles can be words, tokens, tags, diagrams, or structured constraint objects.

### Naming as crystallization into a reusable unit

`LANGUAGE_ACTION_CONTEXT.md` distinguishes:

- linguistic building blocks (phonemes, semantic primes), and
- action building blocks (AQ).

Naming is not a phoneme-level operation.
Naming is closer to AQ-level operations because naming must support action in at least these ways:

- It must select a discrimination region in a space.
- It must be retrievable and re-applicable.
- It must be usable as a unit in larger compositions.

So naming is not "adding a word".
Naming is "stabilizing a discrimination so it can be reused".

### Naming interacts with coherence

`HARMONY_AND_COHERENCE.md` frames many systems as:

local precision traded for global coherence.

Naming fits that frame.
If we force a continuous space to support discrete symbols, we will introduce mismatch.
The choice is:

- let the mismatch accumulate in an uncontrolled way, or
- distribute the mismatch and keep the system coherent.

In practice:

- A name is an approximation. It compresses a region.
- A name becomes useful when it compresses in a way that keeps downstream decisions coherent.

That makes naming a coherence management operation.

### Naming interacts with knowledge and reactivity

`KNOWLEDGE_AND_REACTIVITY.md` draws a line:

- knowledge-informed operations run on geometry and stored structure,
- reactive operations run on energy and thresholds.

Naming belongs primarily to the knowledge-informed side:

- discovering a candidate concept involves geometry, clustering, and evidence accumulation,
- teaching a concept involves stored structure and composition.

But once a name is stable, it becomes usable by reactive layers:

- a reactive module can use the named concept as a fast trigger,
- a Homeostat can treat the named concept as a constraint or coupling object,
- gating can be keyed on the name rather than rediscovering the pattern each time.

This is the engineering reason naming matters.
It can move a distinction from slow inference into fast control.

## The proposal for LLM and Homeostat hybrid systems

### Context: the reflex layer operates on logits

In the *From Bits to Bayes* and Epsilon Homeostat work, the key idea is:

- A base model proposes logits.
- A reflex layer applies constraints in logit space through relaxation.

This architecture is important for concept naming because it gives us a place to attach new handles
without editing the base model weights.

The reflex layer can treat a named concept as a structured object:

- a constraint,
- a coupling template (wormhole),
- a transform, or
- a gate.

### Goal: let the system create names that bind to latent structure

We want a loop like this:

```
Internal latent regularity (weights/activations)
    → candidate discrimination (repeatable pattern)
        → handle creation (name + definition + examples)
            → training and usage (model and user share the handle)
                → improved extraction and composition
```

The key claim is not that the model contains mysterious hidden truths.
The key claim is that the model contains many weakly expressed regularities, and that the absence
of stable handles can prevent those regularities from becoming reliable tools.

### What a name must include to be usable

For this project, a usable name must include:

1. A label (a string, token, or ID).
2. An operational definition.
3. Positive examples (where it should apply).
4. Negative examples (where it should not apply).
5. A test or probe that detects the concept in the system.

If a name lacks an operational definition, then it is a story.
If a name lacks examples, then it is not teachable.
If a name lacks a test, then we cannot verify we extracted anything.

### Where candidate concepts come from

There are several non-exclusive sources:

- **Activation geometry**: clusters in embedding space or hidden states.
- **Error analysis**: recurring failure modes that look like the same missing distinction.
- **Constraint conflicts**: cases where global coherence fails because a missing symbol would
  let the system separate two regimes.
- **Task-driven needs**: a downstream controller requires a state variable that does not exist yet.
- **Interpretability tools**: sparse feature discovery, probing classifiers, and feature visualization.

This document does not claim that any one tool solves it.
It claims that we can turn the output of these tools into a shared handle.

### How the reflex layer uses a name

Once a name exists, the reflex layer can use it in at least two ways.

#### Option A: Name as a constraint template

The name expands into a constraint object that operates on logits.

Example structure:

- If the concept is present at position i, then constrain position j.
- The constraint applies a force in logit space.

This matches the wormhole pattern used for agreement in the Epsilon Homeostat.

#### Option B: Name as a gating variable

The name expands into a gating decision:

- open or close a coupling,
- increase or decrease a coupling strength,
- raise or lower temperature, or
- allocate precision.

This maps directly to the knowledge and reactivity split:

- knowledge proposes the threshold and context,
- reactive applies the threshold at runtime.

## How this can enable discovery

### Discovery as making a latent distinction actionable

In this document, *discovery* does not mean conjuring new facts.

Discovery means:

- the system identifies a regularity,
- it creates a handle for it,
- it teaches the handle to the user and to itself,
- the handle becomes usable in composition and control,
- and the handle supports new experiments and new tasks.

This can create the feeling that the system "found a new concept".
Mechanically, it can be the transition from weak, distributed regularity to a stable AQ-like handle.

### Why time series in LLMs matters here

Large language models are trained for next token prediction on text.
Time series forecasting is usually framed as a different problem.

However, a time series can be represented as a sequence.
If we encode the values into tokens, then forecasting becomes another next token prediction task.

The point of this example is not that any LLM magically understands every time series.
The point is narrower:

- The model already contains general machinery for pattern continuation and distribution modeling.
- If we translate a new domain into a representation that matches the model's machinery, then the
  model can express competence that looked hidden before the translation.

This is relevant to naming because naming is also a translation step.
A name is a translation from a distributed pattern to a compact handle.

One concrete reference:

- Gruver, Finzi, Qiu, and Wilson (2024) show that by encoding time series as digit strings, LLMs can
  produce competitive zero-shot forecasts on a range of benchmarks.

The important technical fact for this document is:

*The representation can be the language.*

If the model can treat digit strings as a language for time series, then we can treat
explicitly defined concept handles as a language for latent distinctions.

## The core claim, stated in AKIRA terms

### Claim 1: A named concept is an externally addressable discrimination

`LANGUAGE_ACTION_CONTEXT.md` defines the functional core of an AQ:
it must discriminate between action alternatives.

If a concept name is useful, then it must do the same thing operationally.
It must let the system and the user select between alternatives.

So in this document, a "concept" is not primarily:

- a dictionary definition, or
- a poetic description.

In this document, a concept is:

*a discrimination that can be invoked on demand and used as a building block.*

### Claim 2: A name can be treated as a wormhole handle

In the Epsilon Homeostat work, a wormhole coupling is an explicit, inspectable channel that
propagates a constraint from a source position to a target position.

A named concept can play a similar role:

- It can identify a source pattern.
- It can transform that pattern into a constraint in a different location or module.
- It can apply force in logit space to enforce coherence.

This gives a concrete engineering meaning to "naming a concept":

Naming creates an object that can be used in constraint injection and in energy relaxation.

## The human side, stated precisely

### What naming changes and what it does not change

Naming does not change the physics of the stimulus.
Naming does not create new photons, new wavelengths, or new sensor data.

Naming can change:

- which discriminations a person practices,
- which boundaries a community teaches,
- which groupings get reinforced in memory,
- and which distinctions become easy to communicate.

This is why color naming studies matter as a parallel.
They show how categorical handles can influence discrimination performance and coordination.

### The useful part of the Sapir-Whorf idea for engineering

The strong version of linguistic determinism is not required here.
The engineering version is:

If a community shares a handle, then it can coordinate repeated training and repeated correction
around that handle. This can shape behavior.

For an ML system, the "community" can be:

- the user and the model,
- the model and a controller,
- a fleet of models that share a concept library, or
- a model and its future fine tuned versions.

## Hidden patterns in large models, and what it means to "unlock" them

### The safe statement about latent structure

A trained model can contain internal structure that is:

- distributed across parameters,
- only weakly activated in typical prompts, and
- not directly addressable as a single symbol.

This does not imply that the model contains every answer.
It does imply that the model can contain compressions and regularities that we do not yet have
good handles for.

So "unlocking" is not magic.
Unlocking is work:

- find a repeatable distinction,
- name it with an operational test,
- and make it usable for action and coordination.

### Interpretability provides candidate distinctions

Mechanistic interpretability and representation probing can provide candidates for:

- which internal directions correlate with a concept,
- which clusters correspond to a regime change, and
- which features predict an error mode.

These candidates are incomplete until we do the naming work:

- choose a boundary,
- define positives and negatives,
- bind the label bidirectionally, and
- verify that it changes decisions in the intended way.

### Concept bottleneck models as a reference pattern

Concept bottleneck models explicitly route decisions through named concept variables.
This is relevant here because the goal is similar:

- take a complex internal state,
- project it into a small set of interpretable variables, and
- make those variables usable for prediction and control.

The difference in this document is architectural:

- we want a pathway that can sit on top of a frozen LLM, and
- we want the handle to be usable by a reflex layer operating on logits.

## A concrete proposal: concept handles for LLM plus reflex layers

### Definition: concept handle entity

In this document, a concept handle is treated as an entity with explicit fields.

This is not code, it is a spec.

```
CONCEPT HANDLE (spec)
────────────────────

id:
  A stable identifier (string or UUID).

label:
  A short human-readable name.

operational_test:
  A procedure that returns true or false, or a score, when applied to:
  - an input,
  - a model state (hidden activations), or
  - a model output distribution (logits or probabilities).

positive_examples:
  Examples where the test should pass.

negative_examples:
  Examples where the test should fail.

link_to_action:
  What decision or constraint this concept supports.

version:
  A version number that increments when the definition changes.
```

This is the minimum needed so that naming remains a technical act.
It forces us to define what we mean, and it forces us to be able to test it.

### Where the operational test can live

There are three useful locations for the operational test.

1. Input space test.
   Example: "does this image contain a stop sign" using a classifier.

2. Representation space test.
   Example: a linear probe on hidden activations that detects a feature.

3. Logit space test.
   Example: a function of the output distribution that detects a mode or constraint violation.

The reflex layer architecture makes the logit space test especially useful because:

- the reflex layer already operates on logits,
- and it can enforce constraints in the same space.

### How a concept handle affects behavior without changing base weights

In a hybrid system:

1. The base LLM proposes logits.
2. The reflex layer applies constraints and couplings.
3. The final distribution is sampled.

If a concept handle has an operational test in logit space, then it can:

- detect when a concept should apply, and
- apply a corrective force when it should apply.

This yields a practical mechanism for "naming and using a concept" without retraining the LLM.

### How a concept handle can become a shared symbol inside the model

To go beyond inference-time constraint injection, we can train the model to use the name itself.

This is the meaning of "teach the model the new name":

- When the concept is present, the model should be able to emit the label in a controlled context.
- When the label appears in the prompt, the model should reliably activate the concept's pattern.

This is a bidirectional binding.

If the binding only works in one direction, then the name is fragile.

## A protocol for creating and stabilizing new names

### Phase 0: pick the target, and define the action

Every new name must start from an action need.

If there is no downstream decision, then we have no criterion for success.

Examples of action needs:

- A controller must choose between two strategies.
- A reflex layer must decide which wormhole to open.
- A safety module must decide whether an output violates a constraint.

### Phase 1: collect candidate patterns

We collect candidate patterns using one or more sources:

- cluster hidden states,
- run sparse feature discovery,
- probe for a suspected feature,
- mine recurring errors,
- or compare successful versus failed cases to find a separating signal.

Output of this phase:

- a candidate discriminator (a test), and
- an initial dataset of positives and negatives.

### Phase 2: propose a name and an operational definition

The name is a convenience.
The definition is the substance.

The minimum definition includes:

- what it distinguishes,
- what it ignores,
- and how we test it.

### Phase 3: teach the name for coordination

There are two distinct teaching targets.

1. Teach the user.
   We write a short description plus examples and counterexamples.

2. Teach the system.
   We implement at least one of:
   - a promptable label emission task,
   - a retrieval trigger task (label in prompt activates the pattern),
   - a reflex constraint that activates when the test fires.

### Phase 4: verify stability and usefulness

We need to measure:

- does the test keep working under distribution shift,
- does the name help coordination (human or controller),
- and does the concept compose with other concepts without breaking coherence.

## Verification and experiments

This section defines checks that make the naming claim falsifiable.

### Test 1: prompt binding

If the name is in the prompt, then the operational test should fire more often than it does for
control prompts.

If that does not happen, then the name is not bound to the concept.

### Test 2: emission binding

If the operational test fires, then the model should be able to emit the name in a controlled
setting more often than chance.

If that does not happen, then the concept is not externally reportable.

### Test 3: intervention usefulness

If the reflex layer uses the concept handle to apply a constraint, then task performance should
improve on the cases where the concept matters, and should not degrade badly elsewhere.

This test is the bridge from naming to action.

### Test 4: compositional use

If we bind two names A and B, then we should be able to express combinations:

- A and B
- A but not B
- B but not A

If the system cannot support these combinations, then the names are not acting as modular handles.

## Failure modes and risks

### Risk 1: naming the wrong boundary

A name can stabilize an incorrect discrimination.
This is a real failure mode in humans and in models.

Mitigation:

- keep the operational test and the examples central,
- measure false positives and false negatives explicitly,
- and version the concept handle when definitions change.

### Risk 2: polysemy, synonymy, and handle drift

Natural language labels are not unique.
The same label can drift, or two labels can overlap.

Mitigation:

- use stable IDs in addition to human-readable labels,
- allow multiple labels per concept ID if needed,
- and test binding, not surface text.

### Risk 3: Goodhart pressure on the handle

If we reward the model for emitting the name, it can learn to emit the name without grounding.

Mitigation:

- reward correct decisions and correct operational tests, not only the word,
- and apply adversarial counterexamples.

## References

This project uses name and year citations, and no BibTeX.

### Color, categories, and linguistic relativity

Berlin, B., and Kay, P. (1969). *Basic Color Terms: Their Universality and Evolution*. University of California Press.

Kay, P., and Kempton, W. (1984). What is the Sapir-Whorf hypothesis? *American Anthropologist*, 86(1), 65-79.

Winawer, J., Witthoft, N., Frank, M. C., Wu, L., Wade, A. R., and Boroditsky, L. (2007). Russian blues reveal effects of language on color discrimination. *Proceedings of the National Academy of Sciences*, 104(19), 7780-7785.

Sapir, E. (1929). The status of linguistics as a science. *Language*, 5(4), 207-214.

Whorf, B. L. (1956). *Language, Thought, and Reality*. MIT Press.

### Drawing, external representations, and learning

Fan, J. E., Bainbridge, W. A., Chamberlain, R., and Wammes, J. D. (2023). Drawing as a versatile cognitive tool. *Nature Reviews Psychology*, 2(9), 556-568. doi:10.1038/s44159-023-00212-w. Available via PMC: `https://pmc.ncbi.nlm.nih.gov/articles/PMC11377027/pdf/nihms-1961155.pdf`.

### Time series forecasting with LLMs

Gruver, N., Finzi, M., Qiu, S., and Wilson, A. G. (2024). Large language models are zero-shot time series forecasters. arXiv:2310.07820.

### Interpretable concept variables in ML

Koh, P. W., Nguyen, T., Tang, Y. S., Mussmann, S., Pierson, E., Kim, B., and Liang, P. (2020). Concept bottleneck models. arXiv:2007.04612.

### AKIRA and related internal work

Goldman, O. (2025). From Bits to Bayes: The Epsilon Homeostat (project documentation).

---

## Related documents

- `LANGUAGE_ACTION_CONTEXT.md`: language, context windows, and AQ as action discriminations
- `COMPUTATIONAL_MECHANICS_EQUIVALENCE.md`: AQ, causal states, and synchronization
- `RADAR_ARRAY.md`: the same three-level structure in a physical signal processing domain
- `HARMONY_AND_COHERENCE.md`: coherence selection and collapse as global consistency constraints
- `KNOWLEDGE_AND_REACTIVITY.md`: the split between geometry-driven knowledge and energy-driven reflexes
- `THE_LANGUAGE_OF_INFORMATION.md`: AQ as alphabet, bonding as grammar, spectral bands as syntax

---

*Oscar Goldman, Shogu Research Group @ Datamutant.ai subsidiary of 温心重工業*


