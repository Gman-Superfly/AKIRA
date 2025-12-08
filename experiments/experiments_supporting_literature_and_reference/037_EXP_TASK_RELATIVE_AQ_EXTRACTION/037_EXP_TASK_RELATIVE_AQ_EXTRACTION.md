# Experiment 037: Task-Relative Action Quanta Extraction

## Same Signal, Different AQ for Different Tasks

**Tier:** ★★ CORE  
**Status:** PLANNED  
**Depends On:** 000 (AQ Extraction), 001 (Entropy Observation)  
**References:** ACTION_QUANTA.md, RADAR_ARRAY.md Section 8, LANGUAGE_ACTION_CONTEXT.md

---

## Motivation

### The Core Claim Being Tested

```
TASK-RELATIVITY OF ACTION QUANTA
────────────────────────────────

From RADAR_ARRAY.md:

  "The SAME radar return produces DIFFERENT AQ depending on task:
  
  WEATHER RADAR:
    Task = Predict precipitation
    AQ = Reflectivity patterns, velocity fields
    
  AIR DEFENSE RADAR:
    Task = Detect threats
    AQ = Target signatures, trajectory indicators
    
  SAME SIGNAL → DIFFERENT AQ → DIFFERENT ACTION"

From ACTION_QUANTA.md Section 1.3:

  "AQ are defined relative to a TASK:
    - Different tasks → Different AQ
    - Same data → Different decompositions"

THE HYPOTHESIS:
  Action Quanta are NOT fixed properties of the input.
  AQ EMERGE from the input-task relationship.
  The task defines what is IRREDUCIBLE (load-bearing for correct action).
```

### Why This Matters

1. **Theoretical Foundation:** If AQ are task-relative, then "meaning" is not intrinsic to signals but emerges from signal-task interaction. This validates the pragmatist view (meaning = action-enablement).

2. **Architecture Implications:** Systems should not extract "universal features" but task-conditioned features. The same input should be processed differently depending on the task.

3. **Efficiency:** Task-relative extraction means we can discard task-irrelevant information early, reducing computational cost.

4. **Failure Analysis:** If a system fails on a task, we can check: Did it extract the WRONG AQ (task-inappropriate features)?

---

## Foundation

**Established Science:**

1. **Attention as Feature Selection** - Attention mechanisms select which features to emphasize. Different queries produce different attention patterns over the same keys/values.

2. **Task-Specific Representations** - Multi-task learning shows that different tasks learn different feature detectors even from shared representations.

3. **Gibson's Affordances** - The meaning of an object IS what actions it affords. Different tasks reveal different affordances of the same object.

**Bridge to AKIRA:**

If AQ are "the minimum pattern that enables correct action," then changing the task changes what counts as "correct action," which should change which patterns qualify as AQ.

**Hypothesis:** Given the same input, different task framings will produce demonstrably different AQ, measurable as:
- Different attention patterns
- Different load-bearing features (via ablation)
- Different minimal sufficient representations

---

## Apparatus

### Required Infrastructure

```python
from typing import Dict, List, Tuple
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

class TaskRelativeAQExtractor:
    """
    Extract AQ candidates under different task framings.
    
    An AQ candidate is load-bearing for a task if:
    1. Ablating it degrades task performance significantly
    2. It is in the minimal sufficient set for the task
    """
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.hooks = {}
        self.activations = {}
        
    def register_hooks(self, layer_indices: List[int]):
        """Register hooks to capture activations at specified layers."""
        for idx in layer_indices:
            layer = self.model.transformer.h[idx]  # Adjust for model architecture
            self.hooks[idx] = layer.register_forward_hook(
                lambda m, inp, out, idx=idx: self._save_activation(idx, out)
            )
    
    def _save_activation(self, layer_idx, output):
        """Save activation for analysis."""
        self.activations[layer_idx] = output[0].detach()
    
    def extract_under_task(
        self, 
        input_text: str, 
        task_prompt: str
    ) -> Dict:
        """
        Extract activations when processing input under a specific task framing.
        
        Args:
            input_text: The signal to process (e.g., "A red ball on a blue table")
            task_prompt: The task framing (e.g., "What color is the ball?")
        
        Returns:
            Dict with activations, attention patterns, and output distribution
        """
        # Combine input with task prompt
        full_prompt = f"{task_prompt}\n\nContext: {input_text}\n\nAnswer:"
        
        inputs = self.tokenizer(full_prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=True)
        
        return {
            'activations': dict(self.activations),
            'attention_patterns': [a.detach() for a in outputs.attentions],
            'logits': outputs.logits.detach(),
            'output_distribution': torch.softmax(outputs.logits[0, -1, :], dim=-1)
        }
    
    def compute_feature_importance(
        self,
        input_text: str,
        task_prompt: str,
        layer_idx: int,
        n_features: int = 100
    ) -> torch.Tensor:
        """
        Compute feature importance via gradient-based attribution.
        
        Returns importance scores for each feature dimension.
        """
        full_prompt = f"{task_prompt}\n\nContext: {input_text}\n\nAnswer:"
        inputs = self.tokenizer(full_prompt, return_tensors="pt")
        
        # Enable gradients
        self.model.eval()
        inputs['input_ids'].requires_grad = False
        
        # Get activations with gradients
        embeddings = self.model.get_input_embeddings()(inputs['input_ids'])
        embeddings.requires_grad = True
        
        outputs = self.model(inputs_embeds=embeddings, output_attentions=True)
        
        # Get predicted token and compute gradient
        pred_logit = outputs.logits[0, -1, :].max()
        pred_logit.backward()
        
        # Feature importance = gradient magnitude
        importance = embeddings.grad.abs().mean(dim=(0, 1))  # Average over batch and sequence
        
        return importance


class AblationTester:
    """
    Test which features are load-bearing for specific tasks via ablation.
    """
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
    
    def ablate_and_measure(
        self,
        input_text: str,
        task_prompt: str,
        features_to_ablate: List[int],
        layer_idx: int
    ) -> Dict:
        """
        Ablate specific features and measure task performance degradation.
        
        Returns:
            Dict with original and ablated performance metrics
        """
        # This would require implementing activation patching
        # For now, return placeholder structure
        return {
            'original_performance': None,
            'ablated_performance': None,
            'degradation': None,
            'features_ablated': features_to_ablate
        }
    
    def find_minimal_sufficient_set(
        self,
        input_text: str,
        task_prompt: str,
        layer_idx: int,
        threshold: float = 0.9
    ) -> List[int]:
        """
        Find minimal set of features sufficient for task performance.
        
        Uses iterative ablation to find smallest set that maintains
        at least `threshold` of original performance.
        
        Returns:
            List of feature indices in the minimal sufficient set
        """
        # Greedy search for minimal sufficient set
        # Start with all features, iteratively remove least important
        pass
```

### Task Definitions

```python
# Define contrasting tasks for the same input
TASK_DEFINITIONS = {
    'visual_scene': {
        'input': "A red ball on a blue table, next to a green cup.",
        'tasks': {
            'color_query': "What color is the ball?",
            'location_query': "Where is the ball?",
            'existence_query': "Is there a ball?",
            'relation_query': "What is the ball next to?",
            'count_query': "How many objects are on the table?"
        },
        'expected_aq': {
            'color_query': ['color_red', 'object_ball'],
            'location_query': ['position_on', 'object_table', 'object_ball'],
            'existence_query': ['object_ball', 'existence'],
            'relation_query': ['spatial_next_to', 'object_ball', 'object_cup'],
            'count_query': ['quantity', 'object_ball', 'object_cup']
        }
    },
    'temporal_event': {
        'input': "John picked up the key and walked to the door.",
        'tasks': {
            'agent_query': "Who performed the action?",
            'object_query': "What did John pick up?",
            'sequence_query': "What did John do after picking up the key?",
            'location_query': "Where did John go?"
        },
        'expected_aq': {
            'agent_query': ['agent_john'],
            'object_query': ['object_key', 'action_pick_up'],
            'sequence_query': ['sequence', 'action_pick_up', 'action_walk'],
            'location_query': ['destination_door', 'action_walk']
        }
    },
    'ambiguous_signal': {
        'input': "Fire!",
        'tasks': {
            'emergency_context': "You are in a crowded building. Fire!",
            'military_context': "You are a soldier with weapon ready. Fire!",
            'campfire_context': "You are camping and someone says: Fire!",
            'pottery_context': "You are in a pottery studio. Fire!"
        },
        'expected_aq': {
            'emergency_context': ['threat', 'escape', 'urgency'],
            'military_context': ['command', 'weapon', 'action_shoot'],
            'campfire_context': ['warmth', 'camp', 'gathering'],
            'pottery_context': ['kiln', 'ceramics', 'process']
        }
    }
}
```

---

## Protocol

### Phase 1: Baseline Activation Comparison

```
BASELINE PROTOCOL:

1. For each input scenario in TASK_DEFINITIONS:
   a. Present the SAME input under DIFFERENT task framings
   b. Extract activations at layers [0, 4, 8, 12, 16, 20, 24] (adjust for model)
   c. Compute cosine similarity between activation patterns

2. Measurements:
   - Activation divergence: How different are activations under different tasks?
   - Divergence progression: Does divergence increase with layer depth?
   - Per-token divergence: Which tokens show highest divergence?

3. Predictions:
   - Activations should DIVERGE across tasks (cosine sim < 0.9)
   - Divergence should INCREASE in later layers (later layers more task-specific)
   - Task-relevant tokens should show HIGHEST divergence
```

### Phase 2: Attention Pattern Analysis

```
ATTENTION PROTOCOL:

1. For each input-task pair:
   a. Extract full attention matrices at all layers
   b. Identify which tokens receive highest attention
   c. Compare attention patterns across tasks

2. Measurements:
   - Attention overlap: What fraction of top-attended tokens is shared?
   - Task-specific attention: Which tokens are attended ONLY for specific tasks?
   - Attention entropy: Does task framing reduce attention entropy?

3. Predictions:
   - Different tasks should attend to DIFFERENT tokens
   - Task-relevant tokens should receive MORE attention
   - Specific task → Lower attention entropy (more focused)
```

### Phase 3: Ablation Study (Load-Bearing Features)

```
ABLATION PROTOCOL:

1. For each input-task pair:
   a. Identify top-K important features via gradient attribution
   b. Ablate these features (zero out or mean-replace)
   c. Measure task performance degradation

2. Key test: CROSS-TASK ABLATION
   a. Find features important for Task A
   b. Ablate these features
   c. Measure degradation on Task A (should be HIGH)
   d. Measure degradation on Task B (should be LOW if AQ are task-relative)

3. Predictions:
   - Ablating Task-A features degrades Task-A performance >50%
   - Ablating Task-A features degrades Task-B performance <10%
   - The load-bearing features are DISJOINT across tasks
```

### Phase 4: Minimal Sufficient Set

```
MINIMAL SET PROTOCOL:

1. For each input-task pair:
   a. Start with full representation (all features)
   b. Iteratively remove least important features
   c. Stop when performance drops below 90% of original
   d. The remaining features = Minimal Sufficient Set = AQ candidates

2. Measurements:
   - Size of minimal set for each task
   - Overlap between minimal sets across tasks
   - Intersection size / Union size (Jaccard index)

3. Predictions:
   - Minimal sets should be SMALL (<<50% of total features)
   - Minimal sets should be DISJOINT across tasks (Jaccard < 0.3)
   - Minimal set IS the AQ for that task
```

---

## Predictions

### If Hypothesis is Correct

```
EXPECTED RESULTS:

┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│  ACTIVATION DIVERGENCE BY LAYER:                                           │
│                                                                             │
│  Cosine   │                                                                 │
│  Similarity│                                                                │
│           │                                                                 │
│    1.0    │ ●───●                                                           │
│           │      ╲                                                          │
│    0.9    │       ●───●                                                     │
│           │            ╲                                                    │
│    0.8    │             ●───●                                               │
│           │                  ╲                                              │
│    0.7    │                   ●───●                                         │
│           │                        ╲                                        │
│    0.6    │                         ●───●                                   │
│           │                                                                 │
│           └────────────────────────────────────────► Layer                  │
│              0    4    8   12   16   20   24                                │
│                                                                             │
│  Prediction: Similarity DECREASES with depth                               │
│  (Later layers are more task-specialized)                                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

QUANTITATIVE PREDICTIONS:

1. Activation divergence:
   - Early layers (0-4): cosine similarity > 0.95
   - Middle layers (8-16): cosine similarity 0.7-0.9
   - Late layers (20+): cosine similarity < 0.7

2. Attention patterns:
   - Top-10 attended tokens overlap < 50% across tasks
   - Task-relevant tokens in top-5 for relevant task, not in top-10 for others

3. Cross-task ablation:
   - Same-task degradation: >50% performance drop
   - Cross-task degradation: <10% performance drop
   - Degradation ratio: >5x between same-task and cross-task

4. Minimal sufficient sets:
   - Set size: <30% of total features
   - Jaccard similarity across tasks: <0.3
   - Union coverage: >80% (tasks together cover most features)
```

### Falsification Criteria

```
FALSIFICATION:

The hypothesis is FALSIFIED if:

1. STRONG: Activations do NOT diverge (cosine sim > 0.95 at all layers)
   → Task framing does not change processing
   → AQ are input-intrinsic, not task-relative

2. STRONG: Cross-task ablation shows SYMMETRIC degradation
   → Ablating Task-A features hurts Task-B equally
   → Features are universal, not task-specific

3. STRONG: Minimal sufficient sets are IDENTICAL across tasks
   → Jaccard > 0.8
   → Same features are load-bearing regardless of task

4. MODERATE: Divergence does not increase with depth
   → Task-specificity is not computed, it's immediate
   → Contradicts hierarchical processing view

5. MODERATE: Attention patterns are identical across tasks
   → Task framing does not guide attention
   → Selection mechanism is task-blind
```

---

## Analysis

### Primary Metrics

```python
def analyze_task_relativity(results: Dict) -> Dict:
    """
    Analyze task-relativity of extracted AQ.
    
    Args:
        results: Dict containing extraction results for all input-task pairs
    
    Returns:
        Analysis summary with divergence, overlap, and ablation metrics
    """
    analysis = {}
    
    # 1. Activation divergence across tasks
    for input_key, tasks in results.items():
        task_pairs = list(combinations(tasks.keys(), 2))
        divergences = []
        
        for task_a, task_b in task_pairs:
            for layer in tasks[task_a]['activations'].keys():
                act_a = tasks[task_a]['activations'][layer]
                act_b = tasks[task_b]['activations'][layer]
                
                # Cosine similarity
                sim = torch.cosine_similarity(
                    act_a.flatten(), 
                    act_b.flatten(), 
                    dim=0
                )
                divergences.append({
                    'input': input_key,
                    'task_pair': (task_a, task_b),
                    'layer': layer,
                    'similarity': sim.item()
                })
        
        analysis[input_key] = {'divergences': divergences}
    
    # 2. Attention overlap
    # ... compute attention overlap metrics
    
    # 3. Ablation asymmetry
    # ... compute cross-task ablation results
    
    return analysis


def compute_jaccard_index(set_a: set, set_b: set) -> float:
    """Compute Jaccard similarity index."""
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def test_ablation_asymmetry(
    same_task_degradation: float,
    cross_task_degradation: float
) -> Dict:
    """
    Test if ablation effects are asymmetric (task-specific).
    """
    ratio = same_task_degradation / max(cross_task_degradation, 0.01)
    
    return {
        'same_task': same_task_degradation,
        'cross_task': cross_task_degradation,
        'ratio': ratio,
        'task_specific': ratio > 5.0  # Strong evidence threshold
    }
```

---

## Expected Outcomes

### If Hypothesis Validated

```
IMPLICATIONS IF TRUE:

1. THEORETICAL:
   AQ are EMERGENT from input-task relationship
   Meaning is not intrinsic to signals
   Validates pragmatist view: meaning = action-enablement

2. ARCHITECTURAL:
   Design systems for task-conditioned feature extraction
   Early layers can be shared; late layers should be task-specific
   Attention should be task-guided, not purely content-based

3. EFFICIENCY:
   Task-irrelevant features can be discarded early
   Compression depends on task (different MDL for different tasks)
   Same input, different compression under different tasks

4. DEBUGGING:
   Check if model extracted WRONG AQ (task-inappropriate features)
   Failure = extracted features for wrong task
   Can diagnose by examining which features are load-bearing
```

### If Hypothesis Falsified

```
IMPLICATIONS IF FALSE:

1. THEORETICAL:
   AQ are INPUT-INTRINSIC, not task-relative
   Universal features exist independent of task
   Contradicts strong pragmatism

2. ARCHITECTURAL:
   Universal feature extractors are valid
   Task-specificity is only in decoder, not encoder
   Simplifies architecture (one encoder, multiple decoders)

3. IMPLICATIONS FOR AKIRA:
   Spectral decomposition extracts universal features
   Task-relativity is in how we READ features, not in features themselves
```

---

## Connection to Other Experiments

| Experiment | Relationship |
|------------|--------------|
| 000 (AQ Extraction) | Provides baseline AQ candidates; this tests if they're task-relative |
| 001 (Entropy) | Task framing should affect entropy (more focused = lower entropy) |
| 003 (Spectral Bands) | Different tasks may emphasize different bands |
| 011 (Prompt Spectral) | Task prompts may have spectral structure that biases extraction |
| 017 (MDL) | Different tasks should have different MDL for same input |

---

## Results

*[ TO BE FILLED AFTER EXPERIMENT ]*

### Activation Divergence by Layer

```
[ PLACEHOLDER FOR RESULTS ]

Layer 0:  Similarity = ____
Layer 4:  Similarity = ____
Layer 8:  Similarity = ____
Layer 12: Similarity = ____
Layer 16: Similarity = ____
Layer 20: Similarity = ____
Layer 24: Similarity = ____
```

### Cross-Task Ablation

```
[ PLACEHOLDER FOR RESULTS ]

Task A features ablated:
  Task A degradation: ____%
  Task B degradation: ____%
  Ratio: ____

Task B features ablated:
  Task A degradation: ____%
  Task B degradation: ____%
  Ratio: ____
```

### Minimal Sufficient Sets

```
[ PLACEHOLDER FOR RESULTS ]

Task A minimal set size: ____ features (____ % of total)
Task B minimal set size: ____ features (____ % of total)
Jaccard similarity: ____
```

---

## Conclusion

*[ TO BE FILLED AFTER EXPERIMENT ]*

---

## References

1. **Gibson, J.J. (1979).** *The Ecological Approach to Visual Perception.* — Affordances: meaning is action-possibility, task-relative.

2. **Peirce, C.S.** — Pragmatic maxim: meaning is practical consequence.

3. **Vaswani, A. et al. (2017).** *Attention Is All You Need.* — Attention as task-guided feature selection.

4. **AKIRA Internal:** `ACTION_QUANTA.md` Section 1.3 — Task-relativity of AQ.

5. **AKIRA Internal:** `RADAR_ARRAY.md` Section 8 — Same signal, different AQ for different tasks (radar example).

6. **AKIRA Internal:** `LANGUAGE_ACTION_CONTEXT.md` — Context determines AQ crystallization.

---

*Oscar Goldman — Shogu Research Group @ Datamutant.ai*

*"The same signal carries different meaning for different tasks. The ball is RED when you ask about color, but ON THE TABLE when you ask about location. Action Quanta are not discovered; they are crystallized from the signal-task interaction."*
