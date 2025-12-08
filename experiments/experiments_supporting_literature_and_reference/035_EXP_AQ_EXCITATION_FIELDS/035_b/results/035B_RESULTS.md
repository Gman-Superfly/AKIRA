# Experiment 035B Results: Context-Controlled Excitation

**Date**: December 2024  
**Model**: GPT-2 (124M parameters)  
**Platform**: Google Colab (T4 GPU)

---

## Summary

**Evidence Score: 1/3 - WEAK EVIDENCE for context-controlled excitation**

The experiment shows some separation between context types but clustering is not clean. The identical tokens "The answer is" do produce somewhat different activations based on preceding context, but the effect is weaker than expected.

---

## Key Question

Do IDENTICAL tokens ("The answer is") produce DIFFERENT activations based on context?

---

## Results

### Metrics by Layer

| Layer | Silhouette | Distance Ratio | Within Dist | Between Dist |
|-------|------------|----------------|-------------|--------------|
| 0     | 0.158      | 1.360          | 4.10        | 5.58         |
| 3     | 0.176      | 1.461          | 7.41        | 10.83        |
| 6     | 0.160      | 1.378          | 15.51       | 21.37        |
| 9     | 0.159      | 1.317          | 41.51       | 54.67        |
| 11    | 0.090      | 1.226          | 85.48       | 104.84       |

### Assessment

1. **Do same-context probes cluster together?**
   - UNCLEAR - Silhouette score 0.090 is low (threshold: > 0.1)
   - Clusters exist but overlap significantly

2. **Do different contexts produce different patterns?**
   - YES - Distance ratio 1.226 > 1.2
   - Between-context distance is 23% larger than within-context
   - Context does influence activation patterns

3. **Does context control strengthen with depth?**
   - NO - Silhouette decreases from 0.176 (layer 3) to 0.090 (layer 11)
   - Distance ratio also decreases with depth

---

## Interpretation

### What We Observed

The results show a **weak but present** context effect:

1. **Context does matter**: Distance ratio > 1.0 at all layers means between-context activations are more distant than within-context activations. The tokens "The answer is" do produce somewhat different patterns depending on whether they follow a math question vs a geography question.

2. **But clustering is poor**: Low silhouette scores indicate significant overlap between context categories. The separation exists but is not clean.

3. **Effect weakens with depth**: Unlike 035A where clustering improved with depth, here the context signal degrades. By layer 11, the activations are more homogeneous.

### Why Weaker Than 035A?

Several factors may explain why 035B shows weaker results than 035A:

**1. Token Position Effects**

In 035A, we measured the LAST token (where prediction happens). In 035B, we measured the middle of the sequence ("The answer is" tokens). The model may not have fully "committed" to a context-specific representation at these positions.

**2. Query Phrase is Generic**

"The answer is" is a very common phrase that appears in many contexts during training. The model may have learned a relatively context-independent representation for this specific phrase.

**3. Averaging Across Positions**

We averaged activations across the 3-4 token positions of "The answer is". This may have washed out position-specific context effects.

**4. Context Types May Be Too Similar**

Math, geography, science, sentiment, and yes/no all share the pattern of "question followed by answer". The contextual difference may be subtler than in 035A where we had sentiment vs math vs factual.

### What This Tells Us About AQ Theory

The weak result is actually informative:

1. **Context effect exists but is partial**: The weights don't produce identical activations for identical tokens - context matters - but the effect is not as strong as the discrimination-type effect in 035A.

2. **AQ excitation may be more about "what to predict" than "what came before"**: In 035A, we measured at the prediction point. In 035B, we measured mid-sequence. The strong AQ patterns may only crystallize at the decision point.

3. **Some tokens may be "AQ-agnostic"**: Common phrases like "The answer is" may function more like structural scaffolding than discrimination-bearing elements. The AQ excitation may happen more strongly on content words.

---

## Comparison with 035A

| Metric | 035A (Final Layer) | 035B (Final Layer) |
|--------|--------------------|--------------------|
| Silhouette | 0.263 | 0.090 |
| Distance Ratio | 1.648 | 1.226 |
| Evidence Score | 3/4 (Strong) | 1/3 (Weak) |

The difference is striking:
- 035A showed clear clustering by discrimination type
- 035B shows weak clustering by context type

This suggests: **What the model needs to DO (discrimination type) is more strongly encoded than what came before (context)**. The AQ are about action, not history.

---

## Revised Hypothesis

Based on 035A + 035B together:

1. **AQ patterns are strongest at decision points**: The last token position (where next-token prediction happens) shows clear AQ excitation patterns.

2. **Mid-sequence representations are more mixed**: Tokens in the middle of a sequence carry context information but haven't fully collapsed to a specific AQ pattern.

3. **Common structural phrases may be AQ-neutral**: "The answer is" might not carry strong AQ signal because it's a structural element, not a discrimination-bearing element.

---

## Suggested Follow-ups

1. **Measure at last token**: Modify 035B to capture activations at the LAST token position after "The answer is", where prediction actually happens.

2. **Use more distinctive contexts**: Try contexts that require genuinely different actions (e.g., "translate to French" vs "solve the equation" vs "continue the story").

3. **Measure specific content words**: Instead of averaging "The answer is", track a single token that varies in meaning based on context.

---

## Conclusion

Experiment 035B provides **weak evidence** (1/3 criteria met) for context-controlled excitation. The identical tokens "The answer is" do produce somewhat different activations based on context (distance ratio > 1.2), but the clustering is poor (silhouette < 0.1) and the effect degrades with depth.

This result, combined with 035A's strong results, suggests that AQ patterns are most clearly observable at **decision points** (last token) where the model must commit to an action, rather than at mid-sequence positions where context is still being integrated.

The weak result is not a failure of AQ theory but rather a refinement: **AQ excitation is about what-to-do-next, and that signal crystallizes most clearly at the prediction point.**

---

AKIRA Project - Experiment 035B  
Oscar Goldman - Shogu Research Group @ Datamutant.ai
