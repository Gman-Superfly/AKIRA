# Experiment 035I: AQ Excitation Threshold Detection

**AKIRA Project - Oscar Goldman - Shogu Research Group @ Datamutant.ai**

---

## Executive Summary

This experiment tested whether a minimum number of Action Quanta (AQ) in context is required for a language model to generate coherent action-relevant responses. The hypothesis derives from AKIRA theory: AQ must "resonate" with the weight field and bond together to construct answers. Below some threshold, insufficient AQ prevents coherent action crystallization.

**Key Finding**: Partial support for threshold hypothesis. Field coherence increases monotonically with AQ count (r = 0.914), and a large effect size appears at the 4->5 AQ transition (Cohen's d = 0.825, p < 0.001). However, response quality metrics remain flat, suggesting either the quality metric is inadequate or the threshold manifests in representation structure rather than output quality.

---

## 1. Experimental Design

### 1.1 Hypothesis

If AQ are quasiparticle excitations in the belief field:
- A minimum number of AQ must be present in context to enable action discrimination
- Below this threshold, the model cannot construct coherent action responses
- Above this threshold, AQ resonate with weight patterns and bond into answers

### 1.2 Method

**Model**: GPT-2 Medium (24 layers, 355M parameters)

**AQ Components Tested**: Six irreducible action discriminations:

| AQ Component | Discrimination | Action Enabled |
|:-------------|:---------------|:---------------|
| THREAT_PRESENT | threat vs no-threat | FLEE vs STAY |
| PROXIMITY | near vs far | ENGAGE vs OBSERVE |
| DIRECTION | toward vs away | INTERCEPT vs EVADE |
| URGENCY | immediate vs delayed | ACT_NOW vs PLAN |
| AGENT_INTENT | hostile vs friendly | DEFEND vs COOPERATE |
| RESOURCE_STATE | scarce vs abundant | CONSERVE vs EXPEND |

**Conditions**: Prompts constructed with 0, 1, 2, 3, 4, or 5 AQ components

**Sample Size**: 100 prompts per AQ level (600 total)

**Layers Probed**: 0, 4, 8, 12, 16, 20, 23

### 1.3 Metrics

**Response Quality Metrics**:
- Quality Score: Composite of confidence and action relevance
- Model Confidence: Token generation probability
- Action Relevance: Presence of action-related tokens in output

**Belief Field Proxies**:
- Activation Magnitude: L2 norm of hidden states (field excitation)
- Activation Coherence: Cosine similarity across layers (field alignment)
- Attention Entropy: Entropy of attention distribution (focus vs diffuse)

---

## 2. Results

### 2.1 Response Quality by AQ Count

| AQ Level | Quality Score | Model Confidence | Action Relevance |
|:---------|:--------------|:-----------------|:-----------------|
| 0 | 0.517 +/- 0.074 | 0.492 | 0.100 |
| 1 | 0.507 +/- 0.075 | 0.560 | 0.100 |
| 2 | 0.555 +/- 0.081 | 0.470 | 0.100 |
| 3 | 0.505 +/- 0.079 | 0.455 | 0.100 |
| 4 | 0.465 +/- 0.078 | 0.370 | 0.100 |
| 5 | 0.519 +/- 0.073 | 0.480 | 0.100 |

**Observation**: Response quality shows no clear monotonic relationship with AQ count. The quality score hovers around 0.50 across all conditions with overlapping confidence intervals. Action relevance metric appears constant (0.100), suggesting this metric may not be sensitive to the manipulation.

### 2.2 Belief Field Proxies

| AQ Level | Activation Magnitude | Activation Coherence |
|:---------|:--------------------|:---------------------|
| 0 | 634.3 | 0.766 |
| 1 | 754.3 | 0.770 |
| 2 | 713.5 | 0.780 |
| 3 | 599.7 | 0.780 |
| 4 | 540.0 | 0.790 |
| 5 | 480.6 | 0.803 |

**Correlation with AQ count**:
- Magnitude: r = -0.772 (strong negative)
- Coherence: r = +0.914 (strong positive)

**Observation**: As AQ count increases, activation magnitude *decreases* while coherence *increases*. This suggests more AQ leads to more constrained, aligned representations rather than higher excitation.

### 2.3 Effect Sizes at AQ Transitions

| Transition | Cohen's d | p-value | Interpretation |
|:-----------|:----------|:--------|:---------------|
| 0 -> 1 | -0.14 | 0.286 | Negligible |
| 1 -> 2 | +0.61 | 0.000 | Medium effect |
| 2 -> 3 | -0.55 | 0.001 | Medium (negative) |
| 3 -> 4 | -0.45 | 0.003 | Small-medium (negative) |
| 4 -> 5 | +0.83 | 0.000 | Large effect |

**Observation**: The largest effect occurs at the 4->5 transition (d = 0.83), suggesting a potential threshold. However, the pattern is non-monotonic with negative effects at intermediate transitions.

### 2.4 Layer-wise Analysis

**Activation Magnitude by Layer and AQ Level**:

```
Layer |  AQ=0  |  AQ=1  |  AQ=2  |  AQ=3  |  AQ=4  |  AQ=5
------|--------|--------|--------|--------|--------|-------
  0   | 124.9  | 126.8  | 125.6  | 123.0  | 123.2  | 120.7
  4   | 600.2  | 764.7  | 713.4  | 536.7  | 483.1  | 396.8
  8   | 668.0  | 846.3  | 791.1  | 599.8  | 539.5  | 445.1
 12   | 689.5  | 869.0  | 813.4  | 623.6  | 562.6  | 468.2
 16   | 736.5  | 917.8  | 861.0  | 674.2  | 615.8  | 521.9
 20   | 867.2  | 1042.7 | 986.1  | 815.6  | 769.2  | 677.6
 23   | 753.7  | 738.6  | 724.0  | 761.3  | 803.9  | 824.8
```

**Key Pattern**: Layers 4-20 show decreasing magnitude with AQ count. Layer 23 (final) shows the *opposite* pattern - magnitude *increases* with AQ count. This suggests action representations consolidate differently in the output layer.

**Activation Coherence by Layer and AQ Level**:

```
Layer |  AQ=0  |  AQ=1  |  AQ=2  |  AQ=3  |  AQ=4  |  AQ=5
------|--------|--------|--------|--------|--------|-------
  0   |  0.77  |  0.71  |  0.72  |  0.78  |  0.79  |  0.82
  4   |  0.79  |  0.78  |  0.78  |  0.79  |  0.79  |  0.81
  8   |  0.78  |  0.79  |  0.78  |  0.78  |  0.79  |  0.79
 12   |  0.75  |  0.76  |  0.75  |  0.75  |  0.75  |  0.75
 16   |  0.70  |  0.71  |  0.71  |  0.72  |  0.72  |  0.72
 20   |  0.74  |  0.73  |  0.74  |  0.76  |  0.77  |  0.77
 23   |  0.84  |  0.86  |  0.87  |  0.88  |  0.89  |  0.87
```

**Key Pattern**: Layer 23 shows highest coherence across all conditions (0.84-0.89). Coherence increases with AQ count most clearly in early layers (0) and late layers (20, 23). Middle layers (12, 16) show minimal AQ effect on coherence.

---

## 3. Interpretation

### 3.1 Evidence Supporting Threshold Hypothesis

1. **Coherence increases with AQ count** (r = 0.914): More action discriminations lead to more aligned field representations. This is consistent with the theory that AQ enable coherent belief crystallization.

2. **Large effect at 4->5 transition** (d = 0.83, p < 0.001): Something measurable changes when moving from 4 to 5 AQ components. This could indicate a threshold effect.

3. **Layer 23 shows unique pattern**: The output layer shows increasing activation magnitude with AQ count, opposite to all other layers. This suggests action representations consolidate in a qualitatively different way at the output stage.

### 3.2 Evidence Against Threshold Hypothesis

1. **Response quality is flat**: The quality score shows no clear relationship with AQ count. If a threshold existed, we would expect quality to jump at some AQ level.

2. **Model confidence decreases with AQ**: More AQ leads to lower generation confidence, opposite to the prediction that more actionable information should enable more confident responses.

3. **Non-monotonic effect sizes**: The pattern of effects is not monotonic (1->2 positive, 2->3 negative, 3->4 negative, 4->5 positive). A true threshold would show a single transition point.

4. **Activation magnitude decreases with AQ**: More AQ leads to *less* field excitation, not more. This contradicts the "resonance" prediction.

### 3.3 Alternative Interpretation

The results may indicate that AQ operate through **constraint** rather than **excitation**:

- More AQ = More constraints on possible responses
- More constraints = Lower magnitude (fewer dimensions active)
- More constraints = Higher coherence (remaining dimensions more aligned)
- More constraints = Lower confidence (fewer probable continuations)

Under this interpretation, AQ function as **filters** that narrow the response space rather than **resonators** that amplify specific patterns.

---

## 4. Methodological Limitations

1. **Quality metric sensitivity**: The response quality metrics may not capture the relevant aspects of "actionable" responses. The constant action relevance score (0.100) suggests this metric needs refinement.

2. **Prompt construction**: Combining multiple AQ markers in a single prompt may create artificial or unnatural language patterns that the model handles differently than natural multi-AQ contexts.

3. **Single model**: Results from GPT-2 Medium may not generalize to other architectures.

4. **Attention entropy**: This metric returned constant values (0.0), indicating a measurement bug. Results should be interpreted without this metric.

5. **Sample size**: While 100 prompts per condition provides statistical power, the large variance in quality scores suggests more samples or different prompt designs may be needed.

---

## 5. Conclusions

### 5.1 Primary Finding

The threshold hypothesis receives **partial support**. Field coherence strongly correlates with AQ count, and a large effect appears at the 4->5 transition. However, this threshold does not manifest in response quality metrics, suggesting either:

(a) The quality metrics are inadequate, or
(b) The threshold affects representation structure without affecting output quality

### 5.2 Theoretical Implications

The pattern of decreasing magnitude with increasing coherence suggests AQ may function through constraint rather than excitation. This is consistent with an alternative reading of AKIRA theory: AQ crystallize by *collapsing* the belief field to specific action-relevant subspaces rather than by *exciting* resonant patterns.

### 5.3 Recommendations for Future Work

1. **Improve quality metrics**: Develop response quality measures specifically designed to detect action-relevant content and appropriate action discrimination.

2. **Test constraint hypothesis**: Design experiments that specifically test whether AQ operate through constraint (narrowing response space) vs excitation (amplifying patterns).

3. **Cross-model validation**: Replicate with Pythia and other architectures to test generalization.

4. **Natural language prompts**: Use prompts derived from natural corpora rather than constructed templates to reduce artificiality.

5. **Output analysis**: Analyze the actual generated text for action-relevant content rather than relying on token-level metrics.

---

## 6. Data Availability

Raw results and analysis code available in the 035_i experiment folder.

---



Authors: Oscar Goldman - Shogu Research Group @ Datamutant.ai 
