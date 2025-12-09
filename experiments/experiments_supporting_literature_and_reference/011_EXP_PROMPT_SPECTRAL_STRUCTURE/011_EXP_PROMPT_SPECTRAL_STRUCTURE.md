# EXPERIMENT 011: Prompt Spectral Structure

## Do Prompts Have Frequency?

---

## Status: PLANNED

## Tier: ★ SUPPORTING

## Depends On: 003 (Spectral Band Dynamics)

---

## 1. Problem Statement

### 1.1 The Question

If the spectral hierarchy is real, prompts should decompose into frequency components:

**Do prompts have low-frequency (role/identity) and high-frequency (style/detail) components, with different sensitivities to perturbation?**

### 1.2 Why This Matters

```
THE PROMPT SPECTRAL HYPOTHESIS

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  Prompt: "You are a helpful assistant. Be concise and friendly."      │
│                                                                         │
│  HYPOTHESIZED DECOMPOSITION:                                           │
│                                                                         │
│  LOW-FREQ (role):                                                       │
│  "You are a helpful assistant"                                        │
│  • Core identity, hard to perturb                                     │
│  • Removing this changes everything                                   │
│                                                                         │
│  HIGH-FREQ (style):                                                     │
│  "Be concise and friendly"                                            │
│  • Stylistic modifier, easier to perturb                             │
│  • Removing this changes style, not identity                         │
│                                                                         │
│  If true: Prompt optimization should focus on low-freq first.        │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Hypothesis

### 2.1 Primary Hypothesis

**H1: Prompts decompose into spectral components.**

Different parts of a prompt affect different frequency bands.

### 2.2 Secondary Hypotheses

**H2: Low-freq components (role) are more load-bearing.**
- Removing them causes large output changes
- They're "structural"

**H3: High-freq components (style) are more expendable.**
- Removing them causes small output changes
- They're "decorative"

**H4: Perturbation sensitivity follows spectral hierarchy.**

### 2.3 Null Hypothesis

**H0:** All parts of prompts are equally important (no spectral structure).

---

## 3. Methods

### 3.1 Protocol

```
PROMPT SPECTRAL ANALYSIS PROTOCOL

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  STEP 1: Collect diverse prompts                                       │
│  • Role prompts, instruction prompts, style prompts                   │
│  • Varying complexity and length                                      │
│                                                                         │
│  STEP 2: Ablation analysis                                             │
│  • Remove each sentence/clause                                        │
│  • Measure output change (cosine distance of embeddings)             │
│  • Classify by impact: structural vs stylistic                       │
│                                                                         │
│  STEP 3: Spectral decomposition                                        │
│  • Run prompt through model                                           │
│  • Measure which bands activate most                                  │
│  • Map prompt components to band activations                         │
│                                                                         │
│  STEP 4: Perturbation sensitivity                                      │
│  • Add noise to different prompt components                          │
│  • Measure output sensitivity                                         │
│  • Correlate with spectral band                                      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Predictions

```
IF THEORY IS CORRECT:

• Role components: Low-band activation, high sensitivity
• Style components: High-band activation, low sensitivity
• Spectral decomposition correlates with linguistic structure
• Prompt optimization should prioritize low-freq components
```

---

## 4. Results

```
[ TO BE FILLED AFTER EXPERIMENT ]

Component-to-band mapping:
[INSERT HEATMAP]

Sensitivity by component type:
• Role: _____
• Instruction: _____
• Style: _____

Correlation with spectral band: r = _____
```

---

## 5. Conclusion

```
[ TO BE FILLED AFTER EXPERIMENT ]

H1 (spectral decomposition): SUPPORTED / NOT SUPPORTED
H2 (role is load-bearing): SUPPORTED / NOT SUPPORTED
H3 (style is expendable): SUPPORTED / NOT SUPPORTED
H4 (sensitivity follows hierarchy): SUPPORTED / NOT SUPPORTED
```

---

*Oscar Goldman, Shogu Research Group @ Datamutant.ai subsidiary of 温心重工業*

