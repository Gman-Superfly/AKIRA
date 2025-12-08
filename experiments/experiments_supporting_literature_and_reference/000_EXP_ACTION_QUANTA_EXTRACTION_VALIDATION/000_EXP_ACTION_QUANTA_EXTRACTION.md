# Experiment 000: Action Quanta Extraction and Validation

## The Foundational Experiment

**Tier:** CRUCIAL (Foundation of foundations)

**Status:** PENDING

**Dependencies:** None - this experiment comes FIRST

---

## The Problem

```
THE FUNDAMENTAL GAP

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  AKIRA defines Action Quanta (AQ) as:                                  │
│  "Irreducible units of actionable information that emerge from         │
│   collapse - quasiparticles of the belief field"                       │
│                                                                         │
│  But we have NO:                                                        │
│  • Operational method for identifying an AQ                            │
│  • Way to extract them from representations                            │
│  • Validation that they exist as discrete units                        │
│                                                                         │
│  All other experiments measure DYNAMICS of something undefined.        │
│                                                                         │
│  THIS EXPERIMENT MUST COME FIRST.                                       │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## The Opportunity

```
IF ACTION QUANTA ARE REAL AND UNIVERSAL, THEY ALREADY EXIST IN TRAINED LLMs

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  SUPPORTING RESEARCH:                                                   │
│                                                                         │
│  1. PLATONIC REPRESENTATION HYPOTHESIS (Huh et al., 2024)              │
│     Different models trained on different data converge to             │
│     similar internal representations.                                   │
│     → Universal structure exists                                       │
│                                                                         │
│  2. BRAIN-LLM ALIGNMENT (Antonello, Goldstein, Schrimpf, etc.)        │
│     LLM internal representations predict neural activity.              │
│     → Representations may be biologically grounded                    │
│                                                                         │
│  3. SPARSE AUTOENCODERS (Anthropic, OpenAI interpretability)          │
│     Extracting interpretable, sparse features from LLMs.              │
│     → Method exists to find discrete "atoms"                          │
│                                                                         │
│  4. UNIVERSAL NEURONS (Olah et al., 2020)                             │
│     Same features appear across different vision models.              │
│     → Universality is empirically observed                            │
│                                                                         │
│  5. EXCESS CORRELATION METHOD (Shuyang, 2025)                         │
│     Validated method to identify universal neurons in LLMs.           │
│     Compare actual vs random-rotation baseline correlation.           │
│     → Concrete, reproducible methodology exists                       │
│                                                                         │
│  IMPLICATION:                                                           │
│  We can EXTRACT AQ candidates from existing LLMs and VALIDATE them.   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Hypotheses

### H0 (Null): No Universal Discrete Features
Features extracted from LLMs are model-specific, continuous, and do not transfer.

### H1: Action Quanta Exist and Are Universal
There exist discrete, irreducible, universal features that:
- Transfer across different LLM architectures
- Are atomic (cannot be decomposed further without losing actionability)
- Enable downstream task performance
- Correlate with biological neural representations

### H2: AQ Live in Low-Frequency Bands
Universal features (AQ candidates) concentrate in low-frequency spectral bands.
Model-specific implementation details live in high-frequency bands.

---

## Method

### Phase 1: Feature Extraction

```
EXTRACT CANDIDATE AQ FROM MULTIPLE LLMs

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  MODELS TO ANALYZE:                                                     │
│  • GPT-2 / GPT-J (open weights, different scales)                     │
│  • Llama 2/3 (different architecture family)                          │
│  • Mistral (different training approach)                              │
│  • Pythia (same architecture, different scales - scaling analysis)    │
│                                                                         │
│  EXTRACTION METHODS:                                                    │
│                                                                         │
│  1. SPARSE AUTOENCODERS (SAE)                                          │
│     - Train SAE on residual stream activations                        │
│     - Extract sparse, interpretable features                          │
│     - Each feature = candidate AQ                                     │
│     Reference: Anthropic's interpretability work                      │
│                                                                         │
│  2. DICTIONARY LEARNING                                                 │
│     - Find overcomplete basis via sparse coding                       │
│     - Atoms of the dictionary = candidate AQ                          │
│     Reference: Olshausen & Field (1996)                               │
│                                                                         │
│  3. NON-NEGATIVE MATRIX FACTORIZATION (NMF)                           │
│     - Parts-based decomposition                                        │
│     - Components = candidate AQ                                        │
│                                                                         │
│  4. INDEPENDENT COMPONENT ANALYSIS (ICA)                               │
│     - Find statistically independent components                       │
│     - Independent sources = candidate AQ                              │
│                                                                         │
│  OUTPUT: Set of candidate AQ for each model                           │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Phase 2: Cross-Model Transfer Test

```
TEST: DO FEATURES TRANSFER ACROSS MODELS?

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  PROCEDURE:                                                             │
│                                                                         │
│  1. Extract features F_A from Model A                                  │
│  2. Extract features F_B from Model B                                  │
│                                                                         │
│  3. ALIGNMENT TEST:                                                     │
│     - Compute similarity matrix S(F_A, F_B)                           │
│     - Use CKA, linear regression, or optimal transport                │
│     - Identify matching feature pairs                                  │
│                                                                         │
│  4. TRANSFER TEST:                                                      │
│     - Train linear probe on F_A to predict task Y                     │
│     - Apply same probe to aligned F_B features                        │
│     - Measure transfer accuracy                                        │
│                                                                         │
│  5. UNIVERSALITY SCORE:                                                 │
│     - Features that transfer across ALL model pairs = AQ candidates   │
│     - Features that don't transfer = model-specific                   │
│                                                                         │
│  PREDICTION (H1):                                                       │
│  - At least 30% of features should transfer with r > 0.7             │
│  - Transfer features form coherent semantic clusters                  │
│                                                                         │
│  PREDICTION (H0 - null):                                               │
│  - Less than 10% transfer, or transfer is random                     │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

#### Phase 2a: Excess Correlation Method (Validated Approach)

Reference: Shuyang (2025), "What is Universality in LLMs? How to Find Universal Neurons"
https://towardsdatascience.com/what-is-universality-in-llm-and-how-to-find-universal-neurons/

```
EXCESS CORRELATION METHOD - CONCRETE IMPLEMENTATION

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  This method has been validated on toy transformers.                   │
│  It provides a simple, reproducible baseline for Phase 2.             │
│                                                                         │
│  PROCEDURE:                                                             │
│                                                                         │
│  1. Extract MLP activations from both models on test data:            │
│     Shape: [num_samples, sequence_length, mlp_dim]                    │
│                                                                         │
│  2. For each neuron i, compute ACTUAL correlation:                    │
│                                                                         │
│            Σ (a_t,i - a_mean_i)(b_t,i - b_mean_i)                      │
│     r_i = ─────────────────────────────────────────                    │
│                  σ(a_i) × σ(b_i)                                       │
│                                                                         │
│     where a_t,i, b_t,i are activations of neuron i at time t          │
│                                                                         │
│  3. Compute BASELINE correlation:                                       │
│     - Apply random rotation matrix R to model_b neurons               │
│     - This destroys alignment but preserves distribution              │
│     - Compute correlation with rotated neurons                        │
│                                                                         │
│  4. EXCESS CORRELATION:                                                 │
│     excess_i = actual_correlation_i - baseline_correlation_i          │
│                                                                         │
│  5. FLAG UNIVERSAL NEURONS:                                            │
│     IF excess_i > 0.5 → neuron i is UNIVERSAL (AQ candidate)         │
│                                                                         │
│  KEY INSIGHT:                                                           │
│  Random rotation baseline eliminates chance correlations.             │
│  Only genuine cross-model alignment survives.                         │
│                                                                         │
│  EXPECTED RESULT (from literature):                                    │
│  - Most neurons: low excess correlation (near zero)                   │
│  - Subset of neurons: high excess correlation (> 0.5)                │
│  - These are universal neurons = AQ candidates                        │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

```python
# CONCRETE IMPLEMENTATION (from Shuyang, 2025)

def compute_excess_correlation(activations_a, activations_b):
    """
    Compute excess correlation between two models' MLP activations.
    
    Args:
        activations_a: [num_samples, seq_len, mlp_dim] from model A
        activations_b: [num_samples, seq_len, mlp_dim] from model B
    
    Returns:
        excess_correlations: [mlp_dim] excess correlation per neuron
        universal_mask: [mlp_dim] boolean mask for universal neurons
    """
    from scipy.stats import pearsonr
    from scipy.stats import ortho_group
    import numpy as np
    
    # Flatten to [num_samples * seq_len, mlp_dim]
    flat_a = activations_a.reshape(-1, activations_a.shape[-1])
    flat_b = activations_b.reshape(-1, activations_b.shape[-1])
    
    # Compute actual correlation per neuron
    actual_corr = np.array([
        pearsonr(flat_a[:, i], flat_b[:, i])[0] 
        for i in range(flat_a.shape[1])
    ])
    
    # Generate random rotation baseline
    random_rotation = ortho_group.rvs(flat_b.shape[1])
    rotated_b = flat_b @ random_rotation
    
    baseline_corr = np.array([
        pearsonr(flat_a[:, i], rotated_b[:, i])[0] 
        for i in range(flat_a.shape[1])
    ])
    
    # Excess correlation
    excess_corr = actual_corr - baseline_corr
    
    # Universal neurons have excess > 0.5
    universal_mask = excess_corr > 0.5
    
    return excess_corr, universal_mask
```

### Phase 3: Irreducibility Test

```
TEST: ARE TRANSFERRING FEATURES ATOMIC?

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  DEFINITION OF IRREDUCIBILITY:                                          │
│  A feature is IRREDUCIBLE if decomposing it loses actionability.      │
│                                                                         │
│  PROCEDURE:                                                             │
│                                                                         │
│  1. Take candidate AQ (transferring features)                          │
│                                                                         │
│  2. DECOMPOSITION ATTEMPT:                                              │
│     - Apply further sparse coding to each AQ                          │
│     - Try to split into sub-components                                │
│                                                                         │
│  3. ACTIONABILITY TEST:                                                 │
│     - Use original AQ for downstream task                             │
│     - Use sub-components for same task                                │
│     - Compare performance                                              │
│                                                                         │
│  4. IRREDUCIBILITY CRITERION:                                          │
│     IF sub-components perform WORSE than original:                    │
│       → Original is IRREDUCIBLE (true AQ)                            │
│     IF sub-components perform EQUAL or BETTER:                        │
│       → Original is REDUCIBLE (not atomic)                           │
│                                                                         │
│  PREDICTION (H1):                                                       │
│  - Universal features are more irreducible than model-specific       │
│  - Irreducibility correlates with transfer score                     │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Phase 4: Actionability Test

```
TEST: DO AQ ENABLE ACTION?

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  DEFINITION:                                                            │
│  AQ must enable correct action - that's the defining criterion.       │
│                                                                         │
│  PROCEDURE:                                                             │
│                                                                         │
│  1. SELECT DOWNSTREAM TASKS:                                            │
│     - Sentiment classification                                         │
│     - Named entity recognition                                         │
│     - Reasoning (simple arithmetic, logic)                            │
│     - Factual recall                                                   │
│                                                                         │
│  2. FEATURE ABLATION:                                                   │
│     - Remove AQ candidate features                                     │
│     - Measure task performance drop                                    │
│                                                                         │
│  3. LOAD-BEARING TEST:                                                  │
│     - If removing feature X drops performance on task Y:              │
│       → X is load-bearing for Y                                       │
│     - If no drop: X is not actionable for Y                          │
│                                                                         │
│  4. CROSS-TASK ACTIONABILITY:                                          │
│     - True AQ should be load-bearing for MULTIPLE tasks              │
│     - Task-specific features are NOT AQ (too narrow)                 │
│                                                                         │
│  PREDICTION (H1):                                                       │
│  - Universal features are load-bearing for more tasks                │
│  - Removing them causes larger performance drops                     │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Phase 5: Spectral Analysis

```
TEST: WHERE DO UNIVERSAL FEATURES LIVE IN FREQUENCY SPACE?

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  MOTIVATION (H2):                                                       │
│  AKIRA predicts that low-frequency = structure, high-frequency = detail│
│  Universal AQ should concentrate in LOW frequency bands.              │
│                                                                         │
│  PROCEDURE:                                                             │
│                                                                         │
│  1. Apply spectral decomposition to activations:                       │
│     - FFT on activation vectors                                        │
│     - Or: Apply AKIRA's 7-band structure                              │
│                                                                         │
│  2. For each extracted feature:                                        │
│     - Compute its spectral profile                                     │
│     - Determine dominant frequency band                               │
│                                                                         │
│  3. CORRELATION ANALYSIS:                                               │
│     - Plot: Transfer score vs. dominant frequency                     │
│     - Plot: Irreducibility vs. dominant frequency                     │
│                                                                         │
│  PREDICTION (H2):                                                       │
│  - Negative correlation: Low freq → high transfer                    │
│  - Universal AQ peak in bands 0-2 (DC, structure)                    │
│  - Model-specific features peak in bands 4-6 (detail)                │
│                                                                         │
│  NULL PREDICTION:                                                       │
│  - No correlation with frequency                                       │
│  - Universal features distributed uniformly across spectrum          │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Phase 6: Biological Validation (Optional but Powerful)

```
TEST: DO AQ CORRELATE WITH NEURAL REPRESENTATIONS?

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  MOTIVATION:                                                            │
│  If AQ are truly universal, they should appear in biological brains.  │
│  Existing brain-LLM alignment research provides the data.             │
│                                                                         │
│  PROCEDURE:                                                             │
│                                                                         │
│  1. Use existing brain imaging datasets:                               │
│     - Pereira et al. (2018) - fMRI during language processing        │
│     - Goldstein et al. (2022) - neural tracking of LLM features      │
│     - Caucheteux & King (2022) - brain-GPT alignment                 │
│                                                                         │
│  2. For each extracted AQ:                                             │
│     - Compute correlation with neural activity patterns               │
│     - Compare: Universal AQ vs. model-specific features              │
│                                                                         │
│  3. PREDICTION:                                                         │
│     - Universal AQ show higher brain correlation                      │
│     - Model-specific features show lower/no correlation              │
│                                                                         │
│  SIGNIFICANCE:                                                          │
│  If confirmed, this suggests AQ are not just computational artifacts  │
│  but reflect fundamental structure of information processing.         │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Apparatus

### Required Tools

```python
# Feature extraction
from transformer_lens import HookedTransformer  # Activation access
from sae_lens import SAE  # Sparse autoencoder training
from sklearn.decomposition import NMF, FastICA, DictionaryLearning

# Cross-model alignment
from torch_cka import CKA  # Centered Kernel Alignment
import ot  # Optimal transport for feature matching

# Analysis
import numpy as np
from scipy.fft import fft, fftfreq
from scipy.stats import spearmanr, pearsonr

# Models
MODELS = [
    "gpt2",
    "gpt2-medium",
    "gpt2-large",
    "EleutherAI/gpt-j-6B",
    "meta-llama/Llama-2-7b",
    "mistralai/Mistral-7B-v0.1",
    "EleutherAI/pythia-70m",
    "EleutherAI/pythia-410m",
    "EleutherAI/pythia-1.4b",
]
```

### Key Metrics

```
METRICS TO COMPUTE

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  TRANSFER SCORE (per feature):                                          │
│  T(f) = mean correlation of f across all model pairs                  │
│  Range: [-1, 1], higher = more universal                              │
│                                                                         │
│  IRREDUCIBILITY INDEX (per feature):                                   │
│  I(f) = performance(f) / performance(decompose(f))                    │
│  Range: [0, ∞], >1 = irreducible                                     │
│                                                                         │
│  ACTIONABILITY SCORE (per feature):                                    │
│  A(f) = mean performance drop when f ablated, across tasks           │
│  Range: [0, 1], higher = more load-bearing                           │
│                                                                         │
│  SPECTRAL CENTROID (per feature):                                      │
│  S(f) = Σ k × |F(k)|² / Σ |F(k)|²                                   │
│  Range: [0, N/2], lower = more low-frequency                         │
│                                                                         │
│  AQ SCORE (composite):                                                  │
│  AQ(f) = T(f) × I(f) × A(f) × (1 - S(f)/S_max)                       │
│  Higher = more likely to be true Action Quantum                       │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Predictions

### If H1 is True (AQ Exist)

```
EXPECTED RESULTS IF ACTION QUANTA ARE REAL

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  1. TRANSFER:                                                           │
│     - 30-50% of features transfer with r > 0.7                        │
│     - Transfer features cluster semantically                          │
│     - Examples: "negation", "entity", "causation", "quantity"        │
│                                                                         │
│  2. IRREDUCIBILITY:                                                     │
│     - Universal features resist decomposition                         │
│     - Sub-components lose actionability                               │
│     - Correlation: transfer ↔ irreducibility r > 0.5                 │
│                                                                         │
│  3. ACTIONABILITY:                                                      │
│     - Universal features are load-bearing                             │
│     - Ablation causes 20-50% performance drop                        │
│     - Load-bearing for multiple tasks                                 │
│                                                                         │
│  4. SPECTRAL:                                                           │
│     - Universal features concentrate in bands 0-2                     │
│     - Model-specific in bands 4-6                                    │
│     - Correlation: transfer ↔ low-frequency r < -0.4                 │
│                                                                         │
│  5. NEURAL:                                                             │
│     - Universal features predict brain activity                       │
│     - r > 0.3 with neural data                                        │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### If H0 is True (No Universal AQ)

```
EXPECTED RESULTS IF ACTION QUANTA DON'T EXIST

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  1. TRANSFER:                                                           │
│     - Less than 10% of features transfer                              │
│     - Transfer is random, no semantic structure                       │
│                                                                         │
│  2. IRREDUCIBILITY:                                                     │
│     - All features are decomposable                                   │
│     - No correlation with transfer                                     │
│                                                                         │
│  3. ACTIONABILITY:                                                      │
│     - Features are task-specific, not general                         │
│     - No multi-task load-bearing features                             │
│                                                                         │
│  4. SPECTRAL:                                                           │
│     - No frequency structure                                           │
│     - Uniform distribution across bands                               │
│                                                                         │
│  IMPLICATION IF H0 CONFIRMED:                                          │
│  "Action Quanta" is a useful abstraction, not a physical reality.    │
│  AKIRA would need to reframe AQ as emergent, not fundamental.        │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Falsification Criteria

```
WHAT WOULD PROVE US WRONG

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  H1 FALSIFIED IF:                                                       │
│  • Transfer rate < 10% across model pairs                             │
│  • No correlation between transfer and irreducibility                 │
│  • No correlation between transfer and actionability                  │
│  • Features are uniformly distributed across frequency bands          │
│                                                                         │
│  H2 FALSIFIED IF:                                                       │
│  • Universal features are NOT concentrated in low-frequency bands    │
│  • Frequency has no predictive power for universality                │
│                                                                         │
│  FULL FALSIFICATION:                                                   │
│  If transfer < 10% AND no frequency structure AND no irreducibility  │
│  correlation, then AQ as discrete universal units do not exist.      │
│                                                                         │
│  This would NOT invalidate AKIRA entirely - but would require        │
│  reframing AQ as emergent abstractions rather than fundamental       │
│  units. The dynamics (collapse, pump cycle) could still be valid.    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Results

**[ TO BE FILLED AFTER EXPERIMENT ]**

### Phase 1: Feature Extraction
- Number of features extracted per model: [ ]
- Sparsity achieved: [ ]

### Phase 2: Transfer
- Transfer rate (r > 0.7): [ ]%
- Semantic clusters identified: [ ]

### Phase 3: Irreducibility
- Irreducibility index distribution: [ ]
- Correlation with transfer: r = [ ]

### Phase 4: Actionability
- Mean ablation impact: [ ]%
- Multi-task features: [ ]%

### Phase 5: Spectral
- Frequency-transfer correlation: r = [ ]
- Band distribution of universal features: [ ]

### Phase 6: Neural
- Brain correlation (universal): r = [ ]
- Brain correlation (model-specific): r = [ ]

---

## Conclusions

**[ TO BE FILLED AFTER EXPERIMENT ]**

### Verdict on H1 (AQ Exist):
[ CONFIRMED / PARTIALLY CONFIRMED / FALSIFIED ]

### Verdict on H2 (AQ are Low-Frequency):
[ CONFIRMED / PARTIALLY CONFIRMED / FALSIFIED ]

### Implications for AKIRA:
[ ]

### Identified Action Quanta (if found):
[ List of confirmed AQ with properties ]

---

## Connection to Other Experiments

```
WHY THIS MUST COME FIRST

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  Experiment 000 → ALL OTHER EXPERIMENTS                                │
│                                                                         │
│  Without knowing WHAT AQ are:                                          │
│  • 001-003: Measuring dynamics of what?                               │
│  • 004-009: BEC analogy for what particles?                          │
│  • 008: Quasiparticle dispersion of what?                            │
│  • 025: Synergy/redundancy of what units?                            │
│                                                                         │
│  IF Experiment 000 identifies concrete AQ:                             │
│  • All subsequent experiments have defined observables                │
│  • We know what to track during collapse                             │
│  • We can measure AQ crystallization directly                        │
│                                                                         │
│  IF Experiment 000 falsifies discrete AQ:                              │
│  • Reframe AKIRA: AQ as emergent, not fundamental                    │
│  • Focus on field dynamics, not particle counting                    │
│  • Still valid: collapse, pump cycle, spectral structure             │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## References

1. Huh, M., Cheung, B., Wang, T., & Isola, P. (2024). The Platonic Representation Hypothesis. arXiv:2405.07987.

2. Bricken, T., et al. (2023). Towards Monosemanticity: Decomposing Language Models With Dictionary Learning. Anthropic.

3. Cunningham, H., et al. (2023). Sparse Autoencoders Find Highly Interpretable Features in Language Models. arXiv:2309.08600.

4. Olah, C., et al. (2020). Zoom In: An Introduction to Circuits. Distill.

5. Caucheteux, C., & King, J.-R. (2022). Brains and algorithms partially converge in natural language processing. Communications Biology.

6. Goldstein, A., et al. (2022). Shared computational principles for language processing in humans and deep language models. Nature Neuroscience.

7. Antonello, R., et al. (2023). Scaling laws for language encoding models in fMRI. NeurIPS.

8. Olshausen, B. A., & Field, D. J. (1996). Emergence of simple-cell receptive field properties by learning a sparse code for natural images. Nature.

9. Williams, P.L., & Beer, R.D. (2010). Nonnegative Decomposition of Multivariate Information. arXiv:1004.2515.

10. Shuyang (2025). What is Universality in LLMs? How to Find Universal Neurons. Towards Data Science. https://towardsdatascience.com/what-is-universality-in-llm-and-how-to-find-universal-neurons/

---

*Oscar Goldman — Shogu Research Group @ Datamutant.ai*

*"Before measuring dynamics, we must know what we're measuring. Before tracking collapse, we must identify what collapses. Before counting quasiparticles, we must find the particles. This experiment comes first because it answers the fundamental question: Do Action Quanta exist, and if so, what are they?"*
