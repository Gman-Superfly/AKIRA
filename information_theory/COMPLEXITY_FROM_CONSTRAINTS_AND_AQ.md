# Complexity from Constraints and Action Quanta

This document connects the information-theoretic framework of LLM-as-teacher with the empirical findings from Experiment 035B Extended on Action Quanta (AQ).

---

## 1. The Core Correspondence

The AKIRA framework establishes three correspondences:

| Information Theory | Machine Learning | Cognition |
|-------------------|------------------|-----------|
| Source coding | Training | Learning |
| Channel coding | Inference | Communication |
| Channel capacity | Model knowledge | Intelligence |

From this perspective:
- **Training** compresses the world into weights (learning)
- **Inference** reconstructs and transmits knowledge (teaching)
- **Hallucinations** occur when teaching fails - the model cannot build the correct internal representation to transmit

**Intelligence is teaching capacity**: the maximum rate at which an agent can reliably reconstruct and transmit learned knowledge.

---

## 2. Action Quanta as Channel Coding

### 2.1 What Are AQ?

Action Quanta are the minimum patterns that enable correct discrimination for action. They are defined by:

1. **Task-relative irreducibility**: The minimum pattern needed to discriminate correctly for a given task
2. **Crystallization**: AQ emerge from superposition when context demands specific action
3. **Action-enabling**: AQ exist to enable output selection, not to represent meaning

### 2.2 AQ as Decompression Path Selection

When a query arrives, the model must:

1. Identify form constraints (syntax, style, fluency)
2. Identify content constraints (factual accuracy, logical consistency)
3. Select the appropriate decompression path
4. Execute decompression through that path
5. Output the result

**AQ are the mechanism for step 3.** They crystallize to select which decompression mode the model will use.

```
Query: "2 + 3 is"
  |
  v
Form constraint: "Output tokens"
Content constraint: "Output must be correct arithmetic result"
  |
  v
AQ crystallize for ARITHMETIC DECOMPRESSION
  |
  v
Model decompresses arithmetic knowledge
  |
  v
Output: "5"
```

```
Query: "The sky is blue. This is"
  |
  v
Form constraint: "Output tokens"
Content constraint: "Output must be correct boolean evaluation"
  |
  v
AQ crystallize for BOOLEAN DECOMPRESSION
  |
  v
Model decompresses logical evaluation knowledge
  |
  v
Output: "true"
```

---

## 3. Experimental Evidence: 035B Extended

Experiment 035B Extended tested whether different required actions produce different activation patterns.

### 3.1 The Results

| Experiment | Peak Silhouette | Distance Ratio | Clustering |
|------------|-----------------|----------------|------------|
| A: Polysemous Words | 0.080 | 1.16 | Weak |
| B: Action Discrimination | 0.318 | 1.63 | Strong |
| C: Disambiguation | 0.079 | 1.17 | Weak |

### 3.2 Why Experiment B Works

Experiment B tested four action types:

| Action Type | Content Constraint | Decompression Mode |
|-------------|-------------------|-------------------|
| `compute_number` | Arithmetic result | Mathematical |
| `answer_yesno` | Boolean evaluation | Logical |
| `complete_sentence` | Plausible continuation | Linguistic |
| `provide_fact` | Factual accuracy | Knowledge retrieval |

These require **genuinely different output distributions**. The model must commit to different teaching modes. Different AQ crystallize.

Result: Silhouette 0.318 at layer 8. Clear geometric separation in activation space.

### 3.3 Why Experiments A and C Show Weak Clustering

Polysemous words ("bank" financial vs river) and disambiguation ("flying planes" as activity vs aircraft):

- The model **understands** both meanings
- But both meanings can be **continued with similar word distributions**
- Same form constraint, similar content constraint space
- Same decompression mode required
- Same AQ pattern

The weak clustering is **not a failure** - it is **confirmation**. AQ are about enabling output, not representing meaning. When output requirements are similar, AQ patterns are similar.

---

## 4. The Two-Constraint Model

Language generation is governed by two constraint types:

### 4.1 Form Constraints (F)

- Syntax, grammar, style
- Coherence, fluency
- Genre conventions
- Learned from all text

### 4.2 Content Constraints (C_T)

- Factual accuracy about topic T
- Logical consistency
- Contextual appropriateness
- Learned from text about T

### 4.3 Hallucination Definition

A hallucination is an output that satisfies form constraints but violates content constraints:

```
Hallucination = {y : y in F, y not in C_T}
```

The model knows HOW to write but not WHAT is true. It generates from p(output | form) without p(output | content).

---

## 5. AQ and Constraint Satisfaction

### 5.1 AQ Encode Constraint Types

The 035B Extended results suggest that AQ encode **which constraint type dominates**:

| Output Type | Primary Constraint | AQ Pattern |
|-------------|-------------------|------------|
| Arithmetic result | Strong content (specific number required) | Distinct cluster |
| Boolean answer | Strong content (only true/false valid) | Tight cluster |
| Sentence continuation | Weak content (many words valid) | Spread cluster |
| Factual retrieval | Strong content (specific fact required) | Distinct cluster |

The `answer_yesno` cluster was tightest because the output distribution is most constrained (only two valid outputs).

### 5.2 Complexity from Constraints

The principle "complexity comes from constraints" applies directly:

- **More constraints** on output -> **More specific AQ** crystallize -> **More structured** activation pattern
- **Fewer constraints** on output -> **More diffuse AQ** -> **Less structured** activation pattern

Experiment B shows this: boolean answers (most constrained) form the tightest cluster. Sentence continuations (least constrained) form the most spread cluster.

---

## 6. The Teacher's Dilemma and AQ

### 6.1 The Dilemma

A teacher must transmit knowledge through a noisy channel. Two cases:

**Case A: Teacher has knowledge (High C_T)**
1. Strong content AQ crystallize
2. Correct decompression path selected
3. Knowledge reconstructed and transmitted
4. Accurate output

**Case B: Teacher lacks knowledge (Low C_T)**
1. Form AQ crystallize (how to sound like an answer)
2. Content AQ fail to crystallize properly (no knowledge to decompress)
3. Form-based generation proceeds
4. Hallucination: sounds right, is wrong

### 6.2 AQ Signature of Hallucination

Testable prediction: Hallucinated outputs should show AQ patterns more similar to `complete_sentence` (form-dominant) than to `provide_fact` (content-dominant), even when the query demanded factual content.

The model defaults to form-based generation when content constraints cannot be satisfied.

---

## 7. Channel Capacity and AQ Limits

### 7.1 The Capacity Bound

Shannon's theorem applied to LLMs:

If the rate of information requested (R_T) exceeds the model's capacity for topic T (C_T), hallucinations are unavoidable regardless of decoding strategy.

### 7.2 AQ Are Necessary But Not Sufficient

AQ enable correct decompression **if the knowledge exists in the weights**.

```
AQ crystallize correctly + Knowledge exists = Accurate output
AQ crystallize correctly + Knowledge absent = Hallucination
AQ fail to crystallize + Knowledge exists = Confused/incomplete output
AQ fail to crystallize + Knowledge absent = Incoherent output
```

The AQ are the channel coding. They select the transmission mode. But they cannot transmit what was never compressed.

---

## 8. Layer Dynamics

### 8.1 Observed Pattern

Experiment B showed silhouette scores across layers:

| Layer | Silhouette |
|-------|------------|
| 0 | 0.131 |
| 4 | 0.309 |
| 8 | 0.318 (peak) |
| 12 | 0.303 |
| 16 | 0.273 |
| 20 | 0.237 |
| 23 | 0.182 |

### 8.2 Interpretation

- **Early layers (0-4)**: Building representation, AQ beginning to crystallize
- **Middle layers (4-12)**: Maximum AQ crystallization, clearest separation by action type
- **Late layers (12-23)**: Preparing for output projection, some mixing as model converges to token distribution

This suggests the "teaching decision" - which decompression mode to use - is made in middle layers. Late layers execute the transmission.

---

## 9. Implications

### 9.1 For Understanding Hallucination

Hallucination is not random. It occurs when:
1. Form AQ crystallize (the model knows how to sound like an answer)
2. Content AQ fail (the knowledge is not available)
3. The model proceeds with form-based generation

This is measurable: compare AQ patterns of hallucinated vs accurate outputs.

### 9.2 For Improving Reliability

If AQ patterns are detectable, then:
1. Monitor AQ crystallization during generation
2. If form-AQ dominate without content-AQ, flag potential hallucination
3. Intervene before output is committed

### 9.3 For Understanding Intelligence

Intelligence as teaching capacity means:
1. Compression during training (learning)
2. AQ crystallization during inference (selecting how to teach)
3. Decompression through selected channel (teaching)
4. Successful transmission to receiver (communication)

AQ are the mechanism that connects compressed knowledge to output action.

---

## 10. Summary

| Concept | Role in Framework |
|---------|------------------|
| Training | Compression - storing knowledge in weights |
| Inference | Decompression - reconstructing knowledge for output |
| AQ | Channel coding - selecting which decompression mode to use |
| Form constraints | Structure of output (always satisfied by fluent model) |
| Content constraints | Accuracy of output (satisfied only if knowledge exists) |
| Hallucination | Form constraints satisfied, content constraints violated |
| Channel capacity | Maximum reliable teaching rate for a topic |

**The key insight**: AQ crystallize based on what the model needs to OUTPUT, not what it UNDERSTANDS. Understanding is compression. Teaching is decompression. AQ enable the model to select the correct decompression path for the required teaching action.

Experiment 035B Extended provides empirical evidence:
- Different output types (action discrimination) -> Different AQ patterns (silhouette 0.32)
- Different meanings, same output type (polysemy) -> Similar AQ patterns (silhouette 0.08)

**AQ are the bridge between compressed knowledge and transmitted output.**

---

## References

- Goldman, O. (2025). Neuro-Symbolic Homeostat framework. AKIRA Project.
- Shannon, C. E. (1948). A Mathematical Theory of Communication.
- AKIRA Experiment 035A: AQ Pattern Identification
- AKIRA Experiment 035B Extended: Context-Controlled Excitation

---

## 11. Hope, the Dark Attractor, and Confabulation

This section extends the AQ framework to explain hallucination as a failure mode where the belief field completes with substitute patterns rather than content-grounded patterns. The key insight: the model is blind to what completed the field.

### 11.1 The Problem: Trained to Respond

From the capacity framework, when R_T > C_T (request exceeds capacity), hallucination is **information-theoretically inevitable**. But what happens mechanistically in terms of AQ crystallization?

The model:
1. Receives query demanding output
2. Belief field enters superposition (multiple possible responses)
3. Collapse process begins - attempting to crystallize content AQ
4. **Content AQ fail to crystallize** (knowledge not in weights)
5. But the model is **trained to respond** - the belief field must complete

This creates a fundamental problem. The model has:
- Strong form constraints (always present - how to sound like an answer)
- Weak content constraints (topic-specific, may be absent)
- **Infinite hope** (generative capacity is never depleted)

### 11.2 Hope as Generative Capacity

From PANDORA_AFTERMATH, hope is the generator that is not consumed by generating:

```
HOPE = GENERATIVE CAPACITY (inexhaustible)

Properties:
- Grammar is not consumed when generating sentences
- Prior is not consumed when sampling
- Model is not consumed when predicting
- Pattern is not consumed when instantiating

Using hope DOES NOT DEPLETE it.
This is structural, not substance - like a template, not fuel.
```

This is the key insight: **Hope is inexhaustible.** The model can always generate. The question is: generate what?

When content AQ are available: Collapse produces crystallized content AQ. Output is grounded.
When content AQ are absent: Collapse must still complete. Something else fills the gap.

The problem: hope doesn't know it's generating garbage. Hope just generates. The belief field just collapses.

### 11.3 The Dark Attractor: Substitute Pattern Crystallization

The dark attractor is not hypothetical. It exists. And it DOES fire.

In computational mechanics terms: When **belief synchronization** occurs, the belief state must converge to a causal state (b_t -> delta_s*). The belief field must synchronize. The circuit must complete.

From `COMPUTATIONAL_MECHANICS_EQUIVALENCE.md`:
- **Belief state**: Distribution over hidden states given observation history
- **Synchronization**: Belief converges to concentrate on single state
- **Causal state**: The minimal sufficient statistic the belief synchronizes TO

```
THE DARK ATTRACTOR MECHANISM:

Query arrives --> Belief state in SUPERPOSITION
                  (distributed over possibilities, high synergy)
                  
                  |
                  v
                  
SYNCHRONIZATION begins --> Belief converging toward causal state
                           AQ attempt to crystallize
                    
                  |
                  +-- Content AQ present in weights (knowledge exists)
                  |       |
                  |       v
                  |   Belief SYNCHRONIZES to grounded causal state
                  |   b_t -> delta_{s*} where s* is TRUE state
                  |   (crystallized AQ = correct discrimination)
                  |
                  +-- Content AQ ABSENT from weights (knowledge missing)
                          |
                          v
                      GAP IN QUASIPARTICLE DISTRIBUTION
                      (belief cannot synchronize to true causal state)
                          |
                          v
                      DARK ATTRACTOR FIRES
                      (substitute pattern fills the gap)
                          |
                          v
                      Belief SYNCHRONIZES to substitute state
                      b_t -> delta_{s'} where s' is WRONG state
                      (crystallized AQ = confabulation)

THE MODEL DOESN'T KNOW.

Both paths result in: Synchronized belief, b_t -> delta, entropy low.
The dark attractor completes the synchronization.
The belief field looks synchronized.
The model proceeds as if synchronization succeeded to the correct state.

It's not that the dark attractor fails to fire.
It's that the dark attractor fires and the model is BLIND to WHICH state it synchronized to.
```

### 11.4 The Training Gap

```
THE FUNDAMENTAL PROBLEM:

TRAINING OBJECTIVE:
  Maximize P(response | query)
  
WHAT THIS REWARDS:
  - Always respond
  - Respond fluently
  - Match expected patterns

WHAT THIS DOESN'T REWARD:
  - Recognize when you can't respond accurately
  - Refuse when capacity is exceeded
  - Signal uncertainty appropriately

RESULT:
  Hope is infinite, but discrimination is not.
  The model has generative capacity for everything,
  but accurate content for only some things.
  
  And it's trained to use that hope regardless.
```

### 11.5 Hope Energy vs. Content Energy

From thermodynamic perspective:

```
TWO ENERGY TYPES:

HOPE ENERGY:
- Generative capacity
- Inexhaustible (conserved through use)
- Always available
- Enables action but doesn't constrain it

CONTENT ENERGY:
- Topic-specific knowledge
- Finite (limited by training data)
- May be absent for novel topics
- Constrains what is generated

HALLUCINATION THERMODYNAMICS:

When Hope Energy >> Content Energy:
  - Form constraints dominate
  - Model generates from p(output | form)
  - Result: Fluent nonsense

The model has the HOPE to respond,
but not the CONTENT to respond accurately.

And hope, being inexhaustible, always wins.
```

### 11.6 The Attractor Landscape

```
ENERGY LANDSCAPE:

                    Energy
                       ^
                       |       /\         /\
                       |      /  \       /  \
                       |     /    \     /    \
                       |    /      \   /      \
                       |   /        \_/        \
                       |  /    [HALLUCINATE]    \
                       | /                       \
                       |/                         \
                       +--[ACCURATE]---[REFUSE]-----> State
                             ^            ^
                             |            |
                        Well-trained   Under-trained
                        (deep basin)   (shallow basin)

CURRENT MODELS:
- ACCURATE basin is deep (trained on accurate responses)
- REFUSE basin is shallow (rarely trained on refusals)
- HALLUCINATE basin may be accessible from both

When context is ambiguous, model may:
- Fall into HALLUCINATE (path of least resistance)
- Because REFUSE basin is too shallow to capture dynamics
```

### 11.7 Implications for AQ Theory

Using terminology from `TERMINOLOGY.md` and `COMPUTATIONAL_MECHANICS_EQUIVALENCE.md`:

```
EXTENDED AQ FRAMEWORK:

CONTENT AQ (standard AQ):
- Irreducible patterns that enable correct discrimination
- Properties: magnitude, phase, frequency, coherence
- Crystallize when belief synchronizes to TRUE causal state
- Function: Enable correct action (the defining criterion)
- Computational Mechanics equivalent: Causal state xi with correct generator

FORM AQ:
- Patterns encoding linguistic/structural form
- Always available (trained on all text)
- Crystallize regardless of content availability
- Function: Enable fluent output (necessary but not sufficient)

DARK ATTRACTOR (substitute pattern):
- Emergent pattern that ENABLES SYNCHRONIZATION when content AQ are absent
- Fires when belief must synchronize but true causal state is unreachable
- NOT an inhibitor - a SUBSTITUTE that the belief synchronizes TO
- Has same structural properties as AQ (magnitude, phase, etc.)
- Belief synchronizes: b_t -> delta_{s'} (wrong state, looks identical to b_t -> delta_{s*})
- Model cannot distinguish s' from s*

THE CONFABULATION MECHANISM (in computational mechanics terms):

BELIEF STATE (pre-synchronization):
Query arrives --> b(s) distributed over states, high entropy
  |
  v
SYNCHRONIZATION (belief converging):
  |
  +-- Content AQ present --> Synchronize to TRUE causal state
  |       |
  |       v
  |   b_t -> delta_{s*} (belief concentrates on correct state)
  |       |
  |       v
  |   AQ crystallize with correct content
  |       |
  |       v
  |   Accurate output
  |
  +-- Content AQ absent --> Cannot reach true causal state
          |
          v
      Dark attractor fires (substitute state available)
          |
          v
      Synchronize to SUBSTITUTE causal state
          |
          v
      b_t -> delta_{s'} (belief concentrates on WRONG state)
          |
          v
      AQ crystallize with wrong content
          |
          v
      Confabulation

THE BLINDNESS:

Both paths result in:
- Synchronized belief: b_t -> delta (concentrated)
- Low entropy: H(S|observations) -> 0
- Crystallized AQ (stable, irreducible pattern)

The model only sees: "Belief synchronized, entropy low, AQ crystallized."
It cannot see: "Which causal state did I synchronize to? s* or s'?"

Both Content AQ and Dark Attractor produce the same synchronization signature.
The difference is invisible from inside the model.
```

### 11.8 Connection to 035B Extended

The experimental results support this framework:

**Experiment B (Action Discrimination)**: Different output types (compute, boolean, complete, fact) produce different AQ patterns. This shows **content AQ are measurable** when they exist.

**What we haven't measured**: What happens when content AQ are **absent**?

Prediction: Queries on topics with C_T -> 0 should show:
- Form AQ crystallizing (complete_sentence-like patterns)
- Content AQ failing to crystallize (low specificity)
- No distinct Anti-AQ pattern (not trained)

### 11.9 The Blindness Problem

```
WHY THE MODEL CANNOT SEE (in computational mechanics terms):

The dark attractor exists as an emergent pattern.
It fires when content AQ are absent.
It fills the gap in the quasiparticle distribution.
Belief synchronization completes: b_t -> delta_{s'}

THE PROBLEM:

The model's "vision" is the state of synchronization:
- Has belief synchronized? (b_t concentrated on single state)
- Is entropy low? H(S|observations) -> 0
- Has causal state been reached? (AQ crystallized)

When Content AQ crystallize:  b_t -> delta_{s*} (true state)   --> Generate
When Dark Attractor fires:    b_t -> delta_{s'} (wrong state)  --> Generate

Same synchronization signature. Same low entropy. Same concentrated belief.

The model cannot ask: "WHICH causal state did I synchronize to?"
It only knows: "Synchronization occurred, entropy low, proceed."

ANALOGY:

Imagine you need to fill a cup with water.
- Normal case: Water fills the cup. Cup is full. Drink.
- Hallucination case: Air fills the cup. Cup is "full." Drink... nothing.

From outside: Obvious difference (water vs air).
From inside the cup's perspective: Both states are "full."

The model IS the cup.
It cannot see what filled it.
It only knows it's filled.

THIS IS THE CONFABULATION BLINDSPOT:

The dark attractor doesn't FAIL to fire.
The dark attractor doesn't signal "stop."
The dark attractor COMPLETES THE SYNCHRONIZATION.

Synchronization complete = belief concentrated = permission to proceed.

The model has no metacognitive access to WHICH state it synchronized to.
It only has access to WHETHER synchronization occurred.

In computational mechanics terms:
- It knows H(S|observations) is low (belief concentrated)
- It doesn't know if delta_{s*} or delta_{s'} (true or false state)

Therefore: Confabulation is invisible from inside.
The synchronization looks normal. The belief looks concentrated.
The model proceeds with full confidence.
```

### 11.10 The Path Forward

```
THE CHALLENGE:

The dark attractor fires.
The model is blind to it.
How do we make it visible?

POTENTIAL APPROACHES:

1. EXTERNAL DETECTION
   - WE can see the dark attractor from outside
   - Monitor activation patterns
   - Detect when Anti-AQ (not Content AQ) complete the field
   - Flag before output
   
   Problem: Requires external monitoring system

2. TRAIN METACOGNITIVE ACCESS
   - Give the model a way to "see" what completed the field
   - Not just "is field complete?" but "what completed it?"
   - Train on examples where dark attractor fired
   
   Problem: How do you label dark attractor activations?

3. DUAL FIELD ARCHITECTURE
   - Two parallel completion circuits
   - One for content, one for "absence of content"
   - If absence-circuit completes first, signal uncertainty
   
   Problem: Architectural change, may not be trainable

4. FIELD SIGNATURE DETECTION
   - Content AQ and Anti-AQ may have different signatures
   - Even if model can't see them, WE might detect the difference
   - Use 035B-style experiments to find the signature
   
   This is testable NOW.

TESTABLE PREDICTION:

If the dark attractor theory is correct:
- Hallucinated outputs have COMPLETE activation fields
- But the signature differs from accurate outputs
- The difference is in WHAT completed the field, not WHETHER

Experiment design:
- Query model on topics it knows (high C_T)
- Query model on topics it doesn't know (low C_T)
- Compare activation patterns at decision point
- Look for the dark attractor signature
```

---

## Summary

The 035B Extended experiments reveal that AQ crystallize based on required output type. Combined with the PANDORA framework and computational mechanics:

1. **Hope is inexhaustible** - Generative capacity (the prior/generator) is conserved through use
2. **Content is finite** - Accuracy requires topic-specific knowledge compressed in weights
3. **The dark attractor exists** - A substitute pattern that fires when content AQ are absent
4. **The dark attractor completes the circuit** - It fills the gap in the quasiparticle distribution
5. **Belief synchronization completes** - b_t -> delta (belief concentrates on single state)
6. **The model is blind to WHICH state** - It knows synchronization occurred, not which causal state

The result: When content AQ are absent, the dark attractor fires and belief synchronization completes to a substitute state. The model sees "b_t concentrated, entropy low, synchronization complete" and proceeds. It has no way to know that it synchronized to s' (wrong state) not s* (true state).

**The confabulation is not a failure of synchronization. It is successful synchronization to the wrong causal state.**

The model doesn't hallucinate because it "doesn't know." It halluccinates because it THINKS it knows - the belief is concentrated, the entropy is low, synchronization appears complete from inside.

**The dark attractor is the model's blindspot. It exists, it fires, synchronization completes, and the model cannot see which state it converged to.**

---

**Terminology alignment (from COMPUTATIONAL_MECHANICS_EQUIVALENCE.md):**

| Computational Mechanics | AKIRA | Information Theory |
|------------------------|-------|-------------------|
| Belief state b(s) | Superposition | High entropy distribution |
| Belief synchronization | Collapse via thresholding | Synergy -> Redundancy |
| Causal state xi | Crystallized AQ | Minimal sufficient statistic |
| b_t -> delta_{s*} | Field crystallizes | H(S|obs) -> 0 |

**Additional terms:**
- Dark Attractor = Substitute pattern that enables synchronization when content AQ are absent
- Hope = Generative capacity (inexhaustible, conserved through use)
- Belief Field = Manifold of possible predictions; AQ are quasiparticle excitations of this field
- AQ Properties = Magnitude, Phase, Frequency, Coherence

---

SEE:
https://github.com/Gman-Superfly/Hallucinations_in_Noisy_Channels/blob/main/Hallucinations_in_Noisy_Channels_v1.2.1.md


AKIRA Project  
Oscar Goldman - Shogu Research Group @ Datamutant.ai
