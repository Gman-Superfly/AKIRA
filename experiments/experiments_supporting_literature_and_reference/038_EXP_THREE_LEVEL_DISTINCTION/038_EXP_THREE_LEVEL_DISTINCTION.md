# Experiment 038: Three-Level Distinction

## Measurement vs Inference vs Action Quanta

**Tier:** ★★ CORE  
**Status:** PLANNED  
**Depends On:** 000 (AQ Extraction), 001 (Entropy Observation), 002 (Collapse Detection)  
**References:** ACTION_QUANTA.md Section 5.4, RADAR_ARRAY.md Section 5.2.1-5.2.2

---

## Motivation

### The Core Claim Being Tested

```
THREE DISTINCT LEVELS OF INFORMATION
────────────────────────────────────

From ACTION_QUANTA.md Section 5.4 (Radar Example):

  MEASUREMENT: +2000 Hz Doppler shift
    - Physical quantity from sensor (DATA)
    - Result of sensor and processing
    
  INFERENCE: "Approaching very fast"
    - Interpretation of measurement (MEANING)
    - Context-dependent
    
  AQ: "CLOSING RAPIDLY"
    - Minimum PATTERN enabling discrimination (FOR ACTION)
    - The crystallized belief

┌───────────────┬────────────────────┬─────────────────────────────────┐
│ LEVEL         │ EXAMPLE            │ WHAT IT IS                      │
├───────────────┼────────────────────┼─────────────────────────────────┤
│ MEASUREMENT   │ +2000 Hz           │ Physical quantity from sensor   │
│               │                    │ (DATA)                          │
├───────────────┼────────────────────┼─────────────────────────────────┤
│ INFERENCE     │ "Approaching fast" │ Interpretation in context       │
│               │                    │ (MEANING)                       │
├───────────────┼────────────────────┼─────────────────────────────────┤
│ AQ            │ "CLOSING RAPIDLY"  │ Minimum PATTERN enabling        │
│               │                    │ discrimination (FOR ACTION)     │
└───────────────┴────────────────────┴─────────────────────────────────┘

THE HYPOTHESIS:
  These three levels are empirically distinguishable.
  L1 (Measurement): Raw activations
  L2 (Inference): Interpretable features via probing
  L3 (AQ): Features that predict downstream action
  
  L3 should be a STRICT SUBSET of L2.
  L3 should be MORE STABLE than L2.
  Ablating L3 should be CATASTROPHIC; ablating L2-only should be TOLERABLE.
```

### Why This Matters

1. **Theoretical Clarity:** Distinguishing these levels clarifies what "understanding" means. A system can MEASURE without INFERRING, and INFER without having actionable AQ.

2. **Interpretability:** If we can identify the AQ level separately from inference, we know exactly what the model uses to make decisions.

3. **Efficiency:** If AQ are a small subset of inferences, we can compress representations dramatically while preserving action capacity.

4. **Failure Diagnosis:** Failures might occur at different levels:
   - Bad measurement → wrong data
   - Bad inference → wrong interpretation
   - Bad AQ → wrong action despite correct interpretation

---

## Foundation

**Established Science:**

1. **Linear Probing** - Linear classifiers on hidden representations can extract semantic features. These represent "inferences" the model has made.

2. **Causal Tracing** - Ablating specific features reveals what is "load-bearing" for output.

3. **Sparse Coding** - Representations can be decomposed into sparse, interpretable components.

**Bridge to AKIRA:**

- Probed features = Inferences (L2)
- Load-bearing features = AQ candidates (L3)
- Raw activations = Measurements (L1)

If L3 is a strict subset of L2, and L3 is more stable and ablation-sensitive, then the three-level distinction is empirically validated.

**Hypothesis:** 
1. L3 (AQ) is a strict subset of L2 (Inference): |L3| < |L2|
2. L3 is more stable across similar inputs than L2
3. Ablating L3 is catastrophic for output; ablating L2-only is tolerable

---

## Apparatus

### Required Infrastructure

```python
from typing import Dict, List, Set, Tuple
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

class ThreeLevelExtractor:
    """
    Extract and distinguish the three levels of information:
    L1: Measurements (raw activations)
    L2: Inferences (probed semantic features)
    L3: AQ (action-predictive features)
    """
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.probes = {}  # Trained probes for semantic features
        
    def extract_L1_measurements(
        self, 
        input_text: str, 
        layer_idx: int
    ) -> torch.Tensor:
        """
        Extract raw activations (L1 - Measurements).
        
        These are uninterpreted: just the numbers.
        """
        inputs = self.tokenizer(input_text, return_tensors="pt")
        
        activations = {}
        def hook(module, inp, out):
            activations['output'] = out[0].detach()
        
        handle = self.model.transformer.h[layer_idx].register_forward_hook(hook)
        
        with torch.no_grad():
            self.model(**inputs)
        
        handle.remove()
        return activations['output']
    
    def train_semantic_probes(
        self,
        training_data: List[Tuple[str, Dict[str, int]]],
        layer_idx: int,
        semantic_features: List[str]
    ):
        """
        Train linear probes for semantic features (L2 - Inferences).
        
        Args:
            training_data: List of (text, feature_labels) pairs
            layer_idx: Which layer to probe
            semantic_features: List of feature names to probe for
                e.g., ['is_positive', 'mentions_person', 'is_question']
        """
        # Extract activations for all training examples
        X = []
        Y = {feat: [] for feat in semantic_features}
        
        for text, labels in training_data:
            act = self.extract_L1_measurements(text, layer_idx)
            # Pool over sequence (mean)
            X.append(act.mean(dim=1).squeeze().numpy())
            
            for feat in semantic_features:
                Y[feat].append(labels.get(feat, 0))
        
        X = np.array(X)
        
        # Train probe for each semantic feature
        for feat in semantic_features:
            probe = LogisticRegression(max_iter=1000)
            probe.fit(X, Y[feat])
            self.probes[feat] = probe
            
            # Evaluate probe accuracy
            score = cross_val_score(probe, X, Y[feat], cv=5).mean()
            print(f"Probe for '{feat}': accuracy = {score:.3f}")
    
    def extract_L2_inferences(
        self,
        input_text: str,
        layer_idx: int
    ) -> Dict[str, float]:
        """
        Extract inferred features using trained probes (L2 - Inferences).
        
        Returns dict of feature_name -> predicted_probability
        """
        act = self.extract_L1_measurements(input_text, layer_idx)
        X = act.mean(dim=1).squeeze().numpy().reshape(1, -1)
        
        inferences = {}
        for feat, probe in self.probes.items():
            prob = probe.predict_proba(X)[0, 1]  # Probability of class 1
            inferences[feat] = prob
        
        return inferences
    
    def identify_L3_AQ(
        self,
        input_text: str,
        target_output: str,
        layer_idx: int,
        threshold: float = 0.1
    ) -> Set[str]:
        """
        Identify which features are AQ (L3 - Action Quanta).
        
        AQ = features that, when ablated, significantly degrade 
        the model's ability to produce the target output.
        
        Returns set of feature names that qualify as AQ.
        """
        # Get baseline output probability
        baseline_prob = self._get_output_probability(input_text, target_output)
        
        aq_candidates = set()
        
        for feat in self.probes.keys():
            # Ablate this feature
            degraded_prob = self._get_output_probability_with_ablation(
                input_text, target_output, layer_idx, feat
            )
            
            # Compute degradation
            degradation = (baseline_prob - degraded_prob) / baseline_prob
            
            if degradation > threshold:
                aq_candidates.add(feat)
        
        return aq_candidates
    
    def _get_output_probability(
        self, 
        input_text: str, 
        target_output: str
    ) -> float:
        """Get probability of target output token."""
        inputs = self.tokenizer(input_text, return_tensors="pt")
        target_id = self.tokenizer.encode(target_output, add_special_tokens=False)[0]
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits[0, -1, :], dim=-1)
        
        return probs[target_id].item()
    
    def _get_output_probability_with_ablation(
        self,
        input_text: str,
        target_output: str,
        layer_idx: int,
        feature_to_ablate: str
    ) -> float:
        """
        Get output probability after ablating a specific feature.
        
        Uses activation patching: project out the direction
        corresponding to the feature probe.
        """
        # This is a simplified version - full implementation would
        # use proper activation patching
        
        inputs = self.tokenizer(input_text, return_tensors="pt")
        target_id = self.tokenizer.encode(target_output, add_special_tokens=False)[0]
        
        # Get probe direction
        probe = self.probes[feature_to_ablate]
        direction = torch.tensor(probe.coef_[0], dtype=torch.float32)
        direction = direction / direction.norm()
        
        ablated_activations = {}
        
        def ablation_hook(module, inp, out):
            # Project out the feature direction
            act = out[0]
            # For each position, remove component along direction
            for pos in range(act.shape[1]):
                proj = (act[0, pos, :] @ direction) * direction
                act[0, pos, :] = act[0, pos, :] - proj
            ablated_activations['output'] = act
            return (act,) + out[1:]
        
        handle = self.model.transformer.h[layer_idx].register_forward_hook(ablation_hook)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits[0, -1, :], dim=-1)
        
        handle.remove()
        
        return probs[target_id].item()


class StabilityAnalyzer:
    """
    Analyze stability of features across similar inputs.
    
    AQ should be MORE stable than mere inferences.
    """
    
    def __init__(self, extractor: ThreeLevelExtractor):
        self.extractor = extractor
    
    def measure_stability(
        self,
        base_input: str,
        perturbations: List[str],
        layer_idx: int
    ) -> Dict[str, float]:
        """
        Measure how stable each feature is across perturbations.
        
        Args:
            base_input: Original input text
            perturbations: List of perturbed versions (same meaning)
            layer_idx: Layer to analyze
        
        Returns:
            Dict of feature_name -> stability_score (variance across perturbations)
        """
        # Get base inferences
        base_inferences = self.extractor.extract_L2_inferences(base_input, layer_idx)
        
        # Get inferences for each perturbation
        all_inferences = [base_inferences]
        for perturbed in perturbations:
            inferences = self.extractor.extract_L2_inferences(perturbed, layer_idx)
            all_inferences.append(inferences)
        
        # Compute stability (inverse of variance) for each feature
        stability = {}
        for feat in base_inferences.keys():
            values = [inf[feat] for inf in all_inferences]
            variance = np.var(values)
            stability[feat] = 1.0 / (1.0 + variance)  # Higher = more stable
        
        return stability
```

### Test Scenarios

```python
# Define test scenarios for distinguishing the three levels
TEST_SCENARIOS = {
    'sentiment_classification': {
        'task': 'Classify sentiment as positive or negative',
        'examples': [
            {
                'input': "This movie was absolutely wonderful, I loved every minute.",
                'target': "positive",
                'semantic_features': [
                    'mentions_movie', 'has_positive_word', 'has_negative_word',
                    'is_first_person', 'has_intensifier', 'is_exclamatory'
                ],
                'expected_L3': {'has_positive_word', 'has_negative_word'}  # Only these are load-bearing
            },
            {
                'input': "The food was terrible and the service was even worse.",
                'target': "negative",
                'perturbations': [
                    "The food tasted terrible and service was even worse.",
                    "Food was terrible, service was even worse than that.",
                    "Terrible food and service that was even worse."
                ]
            }
        ]
    },
    'entity_extraction': {
        'task': 'Identify the main entity',
        'examples': [
            {
                'input': "John went to the store to buy milk.",
                'target': "John",
                'semantic_features': [
                    'has_person_name', 'has_location', 'has_action',
                    'has_object', 'is_past_tense', 'has_purpose_clause'
                ],
                'expected_L3': {'has_person_name'}  # Only entity-related features matter
            }
        ]
    },
    'question_answering': {
        'task': 'Answer a factual question',
        'examples': [
            {
                'input': "The capital of France is Paris. Q: What is the capital of France?",
                'target': "Paris",
                'semantic_features': [
                    'mentions_country', 'mentions_city', 'is_question',
                    'has_capital_word', 'mentions_france', 'mentions_paris'
                ],
                'expected_L3': {'mentions_paris', 'mentions_france', 'has_capital_word'}
            }
        ]
    }
}
```

---

## Protocol

### Phase 1: Train Semantic Probes (Establish L2)

```
PROBE TRAINING PROTOCOL:

1. For each test scenario:
   a. Collect 500+ labeled examples for each semantic feature
   b. Extract activations at layers [8, 12, 16, 20]
   c. Train linear probes for each semantic feature
   d. Record probe accuracy (must be > 70% to count as "inference")

2. Establish L2 set:
   - L2 = {features where probe accuracy > 70%}
   - These are "inferences" the model makes

3. Quality check:
   - Ensure probes are not trivial (baseline accuracy < 60%)
   - Ensure probes capture meaningful distinctions
```

### Phase 2: Identify Load-Bearing Features (Establish L3)

```
ABLATION PROTOCOL:

1. For each test scenario and example:
   a. Get baseline output probability for target
   b. For each feature in L2:
      - Ablate the feature (project out probe direction)
      - Measure output probability degradation
      - If degradation > 10%: feature is load-bearing (L3 candidate)

2. Establish L3 set:
   - L3 = {features in L2 where ablation causes > 10% degradation}
   - These are "AQ candidates"

3. Key measurement:
   - Is L3 a strict subset of L2? (|L3| < |L2|)
   - What fraction? (|L3| / |L2|)
   - Expected: |L3| / |L2| < 0.5
```

### Phase 3: Stability Analysis

```
STABILITY PROTOCOL:

1. For each test example:
   a. Generate 10 paraphrased versions (same meaning, different words)
   b. Extract L2 inferences for all versions
   c. Compute stability (inverse variance) for each feature

2. Compare stability:
   - Mean stability of L3 features
   - Mean stability of L2-only features (in L2 but not L3)

3. Key prediction:
   - Stability(L3) > Stability(L2-only)
   - L3 features should be MORE stable (they capture essence, not surface)
```

### Phase 4: Cross-Level Ablation

```
CROSS-LEVEL ABLATION PROTOCOL:

1. For each test example:
   a. Ablate ALL L3 features simultaneously
   b. Measure output degradation (should be CATASTROPHIC: >50%)
   
   c. Ablate ALL L2-only features simultaneously  
   d. Measure output degradation (should be TOLERABLE: <20%)

2. Key prediction:
   - L3 ablation: degradation > 50%
   - L2-only ablation: degradation < 20%
   - Ratio > 2.5x

3. This demonstrates:
   - L3 is NECESSARY for action
   - L2-only is INSUFFICIENT for action (nice to have, not load-bearing)
```

---

## Predictions

### If Hypothesis is Correct

```
EXPECTED RESULTS:

┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│  SET RELATIONSHIP:                                                          │
│                                                                             │
│  ┌─────────────────────────────────────────────┐                            │
│  │              L2 (Inferences)                │                            │
│  │                                             │                            │
│  │    ┌─────────────────────┐                  │                            │
│  │    │    L3 (AQ)          │                  │                            │
│  │    │                     │                  │                            │
│  │    │  Load-bearing       │   Non-load-      │                            │
│  │    │  for action         │   bearing        │                            │
│  │    │                     │   inferences     │                            │
│  │    └─────────────────────┘                  │                            │
│  │                                             │                            │
│  └─────────────────────────────────────────────┘                            │
│                                                                             │
│  L3 ⊂ L2 (strict subset)                                                   │
│  |L3| / |L2| < 0.5                                                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

QUANTITATIVE PREDICTIONS:

1. Subset relationship:
   - L3 ⊆ L2 (all AQ are inferences, but not all inferences are AQ)
   - |L3| / |L2| < 0.5 (AQ are less than half of inferences)

2. Stability difference:
   - Mean stability of L3 > 1.5 × Mean stability of L2-only
   - L3 features vary less across paraphrases

3. Ablation asymmetry:
   - Ablating L3: >50% output degradation
   - Ablating L2-only: <20% output degradation
   - Ratio: >2.5x

4. Specificity:
   - L3 features match expected_L3 in test scenarios (>80% overlap)
   - Non-expected features in L2-only (not load-bearing as predicted)
```

### Falsification Criteria

```
FALSIFICATION:

The hypothesis is FALSIFIED if:

1. STRONG: L3 = L2 (all inferences are load-bearing)
   - |L3| / |L2| > 0.9
   → No distinction between inference and AQ
   → All interpretable features are equally necessary

2. STRONG: L3 ⊄ L2 (some AQ are not inferences)
   - Features load-bearing but not probeable
   → AQ are not a subset of explicit inferences
   → Black-box features matter

3. STRONG: Stability(L3) ≤ Stability(L2-only)
   - L3 features are not more stable
   → AQ don't capture "essence" better than other inferences

4. STRONG: Ablating L2-only causes >40% degradation
   - Non-load-bearing features actually matter
   → The distinction is not meaningful

5. MODERATE: L3 doesn't match expected_L3
   - Wrong features identified as load-bearing
   → Our intuition about what should be AQ is wrong
   → Or the extraction method is flawed
```

---

## Analysis

### Primary Metrics

```python
def analyze_three_levels(
    L2_features: Set[str],
    L3_features: Set[str],
    stability_L2: Dict[str, float],
    stability_L3: Dict[str, float],
    ablation_L3_degradation: float,
    ablation_L2only_degradation: float
) -> Dict:
    """
    Analyze the three-level distinction.
    """
    # 1. Subset relationship
    is_subset = L3_features.issubset(L2_features)
    subset_ratio = len(L3_features) / len(L2_features) if L2_features else 0
    
    # 2. Stability comparison
    L2_only = L2_features - L3_features
    mean_stability_L3 = np.mean([stability_L3[f] for f in L3_features])
    mean_stability_L2only = np.mean([stability_L2[f] for f in L2_only]) if L2_only else 0
    stability_ratio = mean_stability_L3 / mean_stability_L2only if mean_stability_L2only > 0 else float('inf')
    
    # 3. Ablation asymmetry
    ablation_ratio = ablation_L3_degradation / max(ablation_L2only_degradation, 0.01)
    
    return {
        'is_strict_subset': is_subset and subset_ratio < 1.0,
        'subset_ratio': subset_ratio,
        'stability_ratio': stability_ratio,
        'ablation_ratio': ablation_ratio,
        
        # Verdict
        'hypothesis_supported': (
            is_subset and 
            subset_ratio < 0.5 and 
            stability_ratio > 1.5 and 
            ablation_ratio > 2.5
        )
    }


def visualize_three_levels(L1_activations, L2_inferences, L3_aq):
    """
    Create visualization of the three levels.
    """
    # PCA reduction for visualization
    from sklearn.decomposition import PCA
    
    # L1: Raw activations
    pca_L1 = PCA(n_components=2)
    L1_2d = pca_L1.fit_transform(L1_activations)
    
    # L2: Inference space (probe predictions)
    L2_array = np.array([[inf[f] for f in sorted(inf.keys())] for inf in L2_inferences])
    pca_L2 = PCA(n_components=2)
    L2_2d = pca_L2.fit_transform(L2_array)
    
    # L3: AQ subspace
    L3_features = sorted(L3_aq)
    L3_array = np.array([[inf[f] for f in L3_features] for inf in L2_inferences])
    if L3_array.shape[1] >= 2:
        pca_L3 = PCA(n_components=2)
        L3_2d = pca_L3.fit_transform(L3_array)
    else:
        L3_2d = L3_array
    
    return {
        'L1_2d': L1_2d,
        'L2_2d': L2_2d,
        'L3_2d': L3_2d,
        'L1_explained_var': pca_L1.explained_variance_ratio_,
        'L2_explained_var': pca_L2.explained_variance_ratio_
    }
```

---

## Expected Outcomes

### If Hypothesis Validated

```
IMPLICATIONS IF TRUE:

1. THEORETICAL CLARITY:
   Three levels ARE empirically distinct
   Measurement ≠ Inference ≠ AQ
   Validates the conceptual framework

2. INTERPRETABILITY:
   We can identify EXACTLY what the model uses for decisions
   L3 = the causal bottleneck
   Everything else (L1, L2-only) is "noise" for this task

3. COMPRESSION:
   Can compress to L3 without losing action capacity
   |L3| / |L1| = compression ratio achievable
   Dramatic reduction possible (potentially 90%+)

4. DEBUGGING:
   Failures can be diagnosed by level:
   - L1 wrong → sensor/input problem
   - L2 wrong → inference problem
   - L3 wrong → wrong action selection
   - L3 correct but output wrong → decoder problem

5. ARCHITECTURE:
   Design systems that explicitly separate levels
   L1 → L2 (inference heads)
   L2 → L3 (action selection)
   L3 → Output (action execution)
```

### If Hypothesis Falsified

```
IMPLICATIONS IF FALSE:

1. IF L3 = L2:
   All inferences are equally load-bearing
   No privileged "action quanta"
   Simpler than thought

2. IF L3 ⊄ L2:
   Some AQ are not explicit inferences
   Black-box features matter
   Interpretability is incomplete

3. IF stability is equal:
   AQ don't capture "essence" better
   Surface features equally important
   Compression may not work
```

---

## Connection to Other Experiments

| Experiment | Relationship |
|------------|--------------|
| 000 (AQ Extraction) | Provides candidate features; this tests if they form distinct level |
| 037 (Task-Relative) | Different tasks may have different L3 sets |
| 025 (Synergy-Redundancy) | L3 may correspond to redundant (crystallized) information |
| 017 (MDL) | MDL may correspond to |L3| |
| 001 (Entropy) | Lower entropy may indicate L3 dominance |

---

## Results

*[ TO BE FILLED AFTER EXPERIMENT ]*

### L2 and L3 Set Sizes

```
[ PLACEHOLDER FOR RESULTS ]

Scenario: Sentiment Classification
  |L2| = ____ features
  |L3| = ____ features
  Ratio: ____

Scenario: Entity Extraction
  |L2| = ____ features
  |L3| = ____ features
  Ratio: ____
```

### Stability Analysis

```
[ PLACEHOLDER FOR RESULTS ]

Mean stability (L3 features): ____
Mean stability (L2-only features): ____
Ratio: ____
```

### Ablation Results

```
[ PLACEHOLDER FOR RESULTS ]

Ablating L3 features: ____% degradation
Ablating L2-only features: ____% degradation
Ratio: ____
```

---

## Conclusion

*[ TO BE FILLED AFTER EXPERIMENT ]*

---

## References

1. **Alain, G., & Bengio, Y. (2016).** *Understanding intermediate layers using linear classifier probes.*, Foundation for probing methodology.

2. **Meng, K., et al. (2022).** *Locating and Editing Factual Associations in GPT.*, Causal tracing for load-bearing features.

3. **AKIRA Internal:** `ACTION_QUANTA.md` Section 5.4, Three-level distinction (Radar example).

4. **AKIRA Internal:** `RADAR_ARRAY.md` Section 5.2.1-5.2.2, Measurement vs Inference vs AQ detailed breakdown.

5. **AKIRA Internal:** `TERMINOLOGY_FRAMEWORK_OVERVIEW.md`, Structural vs Functional terminology.

---



*"The model measures many things, infers some things, but acts on few things. Action Quanta are the bottleneck between understanding and doing. Everything else is scaffolding."*


*Oscar Goldman, Shogu Research Group @ Datamutant.ai subsidiary of 温心重工業*

