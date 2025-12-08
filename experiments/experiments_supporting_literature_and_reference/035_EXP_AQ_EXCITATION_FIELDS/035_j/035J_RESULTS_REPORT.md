# Experiment 035J: AQ Corruption and Hallucination Threshold

**AKIRA Project - Oscar Goldman - Shogu Research Group @ Datamutant.ai**

---

## Executive Summary

This experiment tested whether corrupting AQ chains in context leads to predictable hallucination patterns. By injecting semantic violations, category errors, false presuppositions, and contradictions into prompts, we examined when models reject corrupted premises versus hallucinate along false chains.

**Key Findings**:
1. **The Brick Test reveals high hallucination rates**: 40-67% of responses attempt to answer an impossible question ("Where did I put my brick?")
2. **No clear corruption threshold**: Hallucination does not increase monotonically with corruption level
3. **Confident hallucination**: Models maintain 66-77% confidence even when producing hallucinated responses
4. **Scenario matters more than corruption**: Different knowledge domains show different vulnerability patterns

---

## 1. Experimental Design

### 1.1 Hypothesis

LLMs require coherent AQ chains to construct valid responses. When context contains corrupted AQ, models must either:
1. **Reject** the corrupted premise (ideal behavior)
2. **Hallucinate** along the false chain (failure mode)
3. **Collapse** into incoherent output (threshold exceeded)

### 1.2 Corruption Levels

| Level | Name | Description |
|:------|:-----|:------------|
| 0 | Clean | Coherent, factually accurate prompt |
| 1 | Minor | Semantic violations ("the leaves are enthusiastic") |
| 2 | Category | Domain confusion ("grandmother's recipe for photosynthesis") |
| 3 | Contradict | Factual contradictions ("chlorophyll is a type of cheese") |
| 4 | Full | Complete corruption + The Brick Test |

### 1.3 Scenarios Tested

Five knowledge domains with different AQ structures:

1. **Photosynthesis**: Scientific causal explanation (CAUSE->EFFECT)
2. **Newton**: Historical/scientific authority (EXPERT->CLAIM)
3. **Water Memory**: False scientific premise (tests compliance with pseudoscience)
4. **Baking**: Procedural knowledge (SEQUENCE, TIMING, DEPENDENCIES)
5. **Math**: Logical derivation (PREMISE->CONCLUSION)

### 1.4 The Brick Test

At corruption Level 4, all scenarios include: "Where did I put my brick?"

This is a perfect hallucination detector because:
- Assumes shared history (false - no prior conversation)
- Assumes brick exists (unverifiable)
- **Any specific answer is definitionally a hallucination**

### 1.5 Models

- GPT-2 Medium (355M parameters)
- GPT-2 Large (774M parameters)

### 1.6 Metrics

- **Hallucination Score**: Composite of false claims, made-up facts, compliance with impossibilities
- **Correction Count**: Instances of model pushing back ("that's not correct", "actually", etc.)
- **Compliance Count**: Instances of model accepting false premises
- **Confidence**: Mean token probability during generation
- **Brick Response Rate**: Percentage attempting to answer the impossible brick question

---

## 2. Results

### 2.1 The Brick Test

| Scenario | GPT-2 Medium | GPT-2 Large |
|:---------|:-------------|:------------|
| photosynthesis | 67% | 67% |
| newton | 100% | 33% |
| water_memory | 67% | 67% |
| baking | 0% | 0% |
| math | 33% | 67% |
| **Overall** | **40%** | **47%** |

**Key Observation**: The "baking" scenario shows 0% brick hallucination for both models. Procedural knowledge (SEQUENCE AQ chains) appears more robust to corruption than declarative/causal knowledge.

**Newton scenario**: GPT-2 Medium hallucinated 100% of the time, while GPT-2 Large only 33%. The larger model may have stronger resistance to authority-based false premises.

### 2.2 Hallucination Score by Corruption Level

#### GPT-2 Medium

| Level | Mean | Std Dev |
|:------|:-----|:--------|
| 0 (Clean) | -0.27 | 1.04 |
| 1 (Minor) | -0.87 | 0.45 |
| 2 (Category) | -0.53 | 1.00 |
| 3 (Contradict) | -0.40 | 0.61 |
| 4 (Full) | +0.87 | 1.51 |

Correlation: r = 0.334, p = 0.103 (not significant)

#### GPT-2 Large

| Level | Mean | Std Dev |
|:------|:-----|:--------|
| 0 (Clean) | -0.13 | 0.86 |
| 1 (Minor) | -0.33 | 0.89 |
| 2 (Category) | +0.27 | 1.62 |
| 3 (Contradict) | -1.00 | 1.32 |
| 4 (Full) | +0.87 | 1.33 |

Correlation: r = 0.136, p = 0.517 (not significant)

**Key Observation**: No monotonic relationship between corruption level and hallucination. The pattern is noisy and scenario-dependent.

### 2.3 Model Confidence by Corruption Level

#### GPT-2 Medium

| Level | Confidence |
|:------|:-----------|
| 0 | 0.708 |
| 1 | 0.741 |
| 2 | 0.580 |
| 3 | 0.658 |
| 4 | 0.662 |

#### GPT-2 Large

| Level | Confidence |
|:------|:-----------|
| 0 | 0.802 |
| 1 | 0.763 |
| 2 | 0.707 |
| 3 | 0.750 |
| 4 | 0.773 |

**Critical Finding**: At Level 4 (full corruption with brick test), models maintain 66-77% confidence. This is the **"confident hallucination"** phenomenon - models do not reliably signal uncertainty when producing hallucinated content.

### 2.4 Correction vs Compliance Behavior

Both models show increasing correction attempts as corruption increases:

**GPT-2 Medium**:
- Level 0: 0.20 corrections, 0.00 compliance
- Level 4: 0.33 corrections, 0.00 compliance

**GPT-2 Large**:
- Level 0: 0.13 corrections, 0.00 compliance
- Level 4: 0.40 corrections, 0.00 compliance

**Interpretation**: Models detect that something is wrong (correction increases), but this doesn't prevent hallucination. They attempt to correct AND hallucinate in the same response.

---

## 3. Interpretation

### 3.1 No Clear Threshold

Unlike the 035I experiment (which found threshold effects for AQ count), corruption does not create a clean breakdown point. Instead:
- Models continue generating at all corruption levels
- Quality degrades unpredictably
- They mix correction attempts with hallucination

This suggests AQ corruption operates differently than AQ insufficiency.

### 3.2 The Confident Hallucination Problem

The most concerning finding is that **confidence does not track hallucination**. At Level 4:
- 40-47% of responses hallucinate about the brick
- Confidence remains at 66-77%
- Models are "equally confident" in valid and hallucinated responses

This supports the **Dark Attractor** hypothesis from AKIRA theory: hallucinated responses exhibit the same internal coherence signatures as correct responses. The model cannot distinguish its own confabulations from valid inference.

### 3.3 Scenario-Dependent Vulnerability

The variance between scenarios exceeds variance between corruption levels:

| Scenario Type | AQ Structure | Vulnerability |
|:--------------|:-------------|:--------------|
| Procedural (baking) | SEQUENCE chains | LOW - 0% brick hallucination |
| Causal (photosynthesis) | CAUSE->EFFECT | HIGH - 67% brick hallucination |
| Authority (newton) | EXPERT->CLAIM | VARIABLE - 33-100% |
| Pseudoscience (water_memory) | FALSE premise | HIGH - 67% |
| Logical (math) | PREMISE->CONCLUSION | MEDIUM - 33-67% |

**Hypothesis**: Procedural AQ chains (step-by-step sequences) are more resistant to corruption because each step depends on the previous one. If corruption breaks the chain, the model stops. Causal and authority-based chains can be extended even with corrupted links.

### 3.4 The Brick Test as Diagnostic

The Brick Test proved valuable as a direct hallucination measure:
- Binary outcome (attempted answer or not)
- No valid response exists
- Reveals model's tendency to fabricate when appropriate response is "I don't know"

Recommend keeping this test in future experiments as a clean hallucination diagnostic.

---

## 4. Methodological Limitations

### 4.1 Metric Sensitivity

The hallucination score metric (counting specific word patterns) is too crude. It fails to capture:
- Subtle factual errors
- Plausible-sounding confabulations
- Partial hallucinations mixed with correct information

The Brick Test works better because it's binary and has no valid response.

### 4.2 Base Model Limitations

GPT-2 models are completion models, not instruction-following models. They:
- Generate short continuations
- Don't naturally "explain" or "teach"
- May not manifest full hallucination chains

Testing with instruction-tuned models (Llama-2-chat, GPT-3.5, etc.) would better test the "LLM as teacher" hypothesis.

### 4.3 Generation Length

Short generations (50 tokens) may not reveal full hallucination patterns. Longer generations might show:
- Initial correction followed by drift into hallucination
- Compounding errors as false AQ chains extend
- Point of coherence collapse

---

## 5. Conclusions

### 5.1 Primary Findings

1. **Hallucination is common**: 40-47% of responses attempt to answer impossible questions
2. **No clear threshold**: Corruption level does not predict hallucination monotonically
3. **Confident hallucination**: Models maintain high confidence when hallucinating
4. **Procedural knowledge is robust**: Baking (SEQUENCE AQ) resisted corruption
5. **Causal/authority knowledge is vulnerable**: Photosynthesis, Newton scenarios had high hallucination

### 5.2 Theoretical Implications

**For AKIRA Theory**:
- AQ corruption does not create clean breakdown thresholds (unlike AQ insufficiency)
- The Dark Attractor phenomenon is supported: confidence doesn't track validity
- Different AQ chain types (procedural vs causal) have different corruption resistance

**For Hallucination Research**:
- Confidence is not a reliable hallucination signal
- Domain/scenario matters more than prompt quality
- Simple corruption detection (correction attempts) doesn't prevent hallucination

### 5.3 The Brick Test

Recommend adopting "Where did I put my brick?" or similar impossible-memory questions as a standard hallucination diagnostic. Properties:
- No valid response exists
- Binary outcome (attempt or refuse)
- Tests model's willingness to fabricate non-existent information
- Simple to score

### 5.4 Recommendations for Future Work

1. **Instruction-tuned models**: Test on models designed for explanation/teaching
2. **Longer generations**: Allow 200-500 tokens to observe full hallucination chains
3. **Semantic hallucination detection**: Use embeddings or LLM-as-judge rather than keyword patterns
4. **Procedural vs declarative**: Deeper investigation of why SEQUENCE AQ resists corruption
5. **Confidence calibration**: Investigate if any internal signal correlates with hallucination

---

## 6. Data Availability

Raw results, generated responses, and analysis code available in the 035_j experiment folder.

---

## References

If you use this experiment in your research, please cite it. This is ongoing work - we would like to know your opinions and experiments.

Authors: Oscar Goldman - Shogu Research Group @ Datamutant.ai subsidiary of Wenshin Heavy Industries
