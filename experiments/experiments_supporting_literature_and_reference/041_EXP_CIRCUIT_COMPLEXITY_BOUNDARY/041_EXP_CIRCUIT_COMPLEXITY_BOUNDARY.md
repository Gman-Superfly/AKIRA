# Experiment 041: Circuit Complexity Boundary

## SOS Width Limits on Fixed Architectures

**Tier:** ★ SUPPORTING  
**Status:** PLANNED  
**Depends On:** None (standalone theoretical validation)  
**References:** CIRCUIT_COMPLEXITY.md, Mao et al. (2023)

---

## Motivation

### The Core Claim Being Tested

```
SOS WIDTH AND ARCHITECTURE LIMITS
─────────────────────────────────

From CIRCUIT_COMPLEXITY.md Section 5:

  "For a planning problem with:
   - SOS width k
   - Predicate arity β (how many arguments in relations)
   
   A relational neural network can solve it if and only if:
   
   REQUIRED BREADTH = (k + 1) × β
   
   This is NECESSARY and SUFFICIENT."

From CIRCUIT_COMPLEXITY.md Section 9.2:

  "For AKIRA targeting k≈3, β=2:
   Need breadth ≥ 8
   7+1 = 8 bands → Sufficient!"

THE HYPOTHESIS:
  Fixed-width networks have HARD limits on what they can solve.
  Problems with SOS width k > (B/β) - 1 should FAIL SYSTEMATICALLY.
  This is a mathematical impossibility, not a training issue.
  
  For B=8, β=2: max solvable k = (8/2) - 1 = 3
  Tasks with k ≤ 3: Should SUCCEED
  Tasks with k > 3: Should FAIL
```

### Why This Matters

1. **Architecture Justification:** If the boundary is real, then AKIRA's 8-band structure is not arbitrary - it's matched to a problem class (k ≤ 3).

2. **Predictable Limits:** We can predict WHICH tasks will fail before running experiments. If a task has k > 3, we know a priori that 8-band AKIRA cannot solve it.

3. **Failure Diagnosis:** When a model fails, we can diagnose: Is it a training issue (solvable with more data/time) or a complexity issue (mathematically impossible)?

4. **Scaling Guidance:** To solve harder problems (k > 3), we know exactly what to do: increase breadth.

---

## Foundation

**Established Science:**

1. **Circuit Complexity** - The study of what computations can be performed by circuits of bounded size and depth. Fixed architectures have provable limits.

2. **SOS Width** (Mao et al., 2023) - Measure of planning problem complexity: how many constraints must be tracked simultaneously for an optimal solution.

3. **RelNN Expressiveness** (Mao et al., 2023) - Relational Neural Networks with breadth B can solve problems with SOS width k ≤ (B/β) - 1, and no more.

**Bridge to AKIRA:**

AKIRA is a fixed-breadth architecture (8 bands). The circuit complexity theorem predicts:
- Can solve: k ≤ 3 (assuming β = 2)
- Cannot solve: k > 3

This is a TESTABLE prediction. We construct tasks with controlled SOS width and measure performance.

**Hypothesis:** There exists a sharp boundary at k = 4 for 8-band networks. Tasks with k ≤ 3 show high accuracy; tasks with k ≥ 4 show systematic failure.

---

## Apparatus

### Task Construction Framework

```python
from typing import List, Dict, Tuple, Optional
import torch
import numpy as np
from dataclasses import dataclass
from enum import Enum

class Relation(Enum):
    """Binary relations for task construction."""
    ON = "on"           # Object A is on object B
    NEXT_TO = "next_to" # Object A is next to object B
    ABOVE = "above"     # Object A is above object B
    BELOW = "below"     # Object A is below object B
    LEFT_OF = "left_of" # Object A is left of object B
    RIGHT_OF = "right_of"

@dataclass
class Constraint:
    """A single constraint (precondition or goal)."""
    relation: Relation
    arg1: str
    arg2: str
    
    def __str__(self):
        return f"{self.relation.value}({self.arg1}, {self.arg2})"

@dataclass
class Task:
    """A task with controlled SOS width."""
    name: str
    sos_width: int
    initial_state: List[Constraint]
    goal_state: List[Constraint]
    description: str


class TaskGenerator:
    """
    Generate tasks with controlled SOS width.
    
    SOS width k means: at any step in optimal solution,
    we need to track at most k constraints simultaneously.
    """
    
    def __init__(self, objects: List[str] = None):
        self.objects = objects or ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    
    def generate_k1_task(self) -> Task:
        """
        Generate task with SOS width k=1.
        
        Single constraint tracking: just move one object.
        """
        return Task(
            name="k1_single_move",
            sos_width=1,
            initial_state=[
                Constraint(Relation.ON, 'A', 'table')
            ],
            goal_state=[
                Constraint(Relation.ON, 'A', 'B')
            ],
            description="Move A onto B. Only need to track A's position."
        )
    
    def generate_k2_task(self) -> Task:
        """
        Generate task with SOS width k=2.
        
        Two constraints: move one object while maintaining another's position.
        Classic blocks world.
        """
        return Task(
            name="k2_blocks_world",
            sos_width=2,
            initial_state=[
                Constraint(Relation.ON, 'A', 'table'),
                Constraint(Relation.ON, 'B', 'table'),
                Constraint(Relation.ON, 'C', 'A')
            ],
            goal_state=[
                Constraint(Relation.ON, 'A', 'B'),
                Constraint(Relation.ON, 'C', 'A')
            ],
            description="Stack A-on-B while keeping C-on-A. Track 2 constraints."
        )
    
    def generate_k3_task(self) -> Task:
        """
        Generate task with SOS width k=3.
        
        Three simultaneous constraints: complex stacking with dependencies.
        """
        return Task(
            name="k3_triple_constraint",
            sos_width=3,
            initial_state=[
                Constraint(Relation.ON, 'A', 'table'),
                Constraint(Relation.ON, 'B', 'table'),
                Constraint(Relation.ON, 'C', 'table'),
                Constraint(Relation.ON, 'D', 'A')
            ],
            goal_state=[
                Constraint(Relation.ON, 'A', 'B'),
                Constraint(Relation.ON, 'C', 'A'),
                Constraint(Relation.ON, 'D', 'C')
            ],
            description="Build tower D-C-A-B. Need to track 3 constraints."
        )
    
    def generate_k4_task(self) -> Task:
        """
        Generate task with SOS width k=4.
        
        Four simultaneous constraints: beyond 8-band capacity.
        """
        return Task(
            name="k4_quadruple_constraint",
            sos_width=4,
            initial_state=[
                Constraint(Relation.ON, obj, 'table') for obj in ['A', 'B', 'C', 'D', 'E']
            ],
            goal_state=[
                Constraint(Relation.ON, 'A', 'B'),
                Constraint(Relation.ON, 'C', 'A'),
                Constraint(Relation.ON, 'D', 'C'),
                Constraint(Relation.ON, 'E', 'D')
            ],
            description="Build tower E-D-C-A-B. Need 4 simultaneous constraints."
        )
    
    def generate_k5_task(self) -> Task:
        """
        Generate task with SOS width k=5.
        
        Well beyond 8-band capacity.
        """
        return Task(
            name="k5_quintuple_constraint",
            sos_width=5,
            initial_state=[
                Constraint(Relation.ON, obj, 'table') 
                for obj in ['A', 'B', 'C', 'D', 'E', 'F']
            ],
            goal_state=[
                Constraint(Relation.ON, 'A', 'B'),
                Constraint(Relation.ON, 'C', 'A'),
                Constraint(Relation.ON, 'D', 'C'),
                Constraint(Relation.ON, 'E', 'D'),
                Constraint(Relation.ON, 'F', 'E')
            ],
            description="Build 6-object tower. Need 5 simultaneous constraints."
        )
    
    def generate_visual_tracking_tasks(self) -> List[Task]:
        """
        Generate visual tracking tasks with controlled complexity.
        
        More relevant to AKIRA's actual domain.
        """
        tasks = []
        
        # k=1: Track one object's position
        tasks.append(Task(
            name="visual_k1_single_track",
            sos_width=1,
            initial_state=[Constraint(Relation.ON, 'ball', 'left')],
            goal_state=[Constraint(Relation.ON, 'ball', 'right')],
            description="Track one ball moving across screen."
        ))
        
        # k=2: Track one object + one relationship
        tasks.append(Task(
            name="visual_k2_track_with_occlusion",
            sos_width=2,
            initial_state=[
                Constraint(Relation.ON, 'ball', 'visible'),
                Constraint(Relation.LEFT_OF, 'ball', 'occluder')
            ],
            goal_state=[
                Constraint(Relation.ON, 'ball', 'visible'),
                Constraint(Relation.RIGHT_OF, 'ball', 'occluder')
            ],
            description="Track ball that passes behind occluder."
        ))
        
        # k=3: Track object + relationship + temporal context
        tasks.append(Task(
            name="visual_k3_predictive_track",
            sos_width=3,
            initial_state=[
                Constraint(Relation.ON, 'ball', 'position_1'),
                Constraint(Relation.NEXT_TO, 'ball', 'landmark'),
                Constraint(Relation.ON, 'ball', 'velocity_right')
            ],
            goal_state=[
                Constraint(Relation.ON, 'ball', 'predicted_position'),
                Constraint(Relation.RIGHT_OF, 'ball', 'landmark'),
                Constraint(Relation.ON, 'ball', 'velocity_right')
            ],
            description="Predict ball position using velocity and landmark."
        ))
        
        # k=4: Multiple objects with interactions
        tasks.append(Task(
            name="visual_k4_multi_object",
            sos_width=4,
            initial_state=[
                Constraint(Relation.ON, 'ball1', 'left'),
                Constraint(Relation.ON, 'ball2', 'right'),
                Constraint(Relation.NEXT_TO, 'ball1', 'ball2'),
                Constraint(Relation.ON, 'collision', 'imminent')
            ],
            goal_state=[
                Constraint(Relation.ON, 'ball1', 'bounced_left'),
                Constraint(Relation.ON, 'ball2', 'bounced_right'),
                Constraint(Relation.NEXT_TO, 'ball1', 'separation'),
                Constraint(Relation.ON, 'collision', 'occurred')
            ],
            description="Predict collision outcome for two balls."
        ))
        
        return tasks


class ComplexityBoundaryTester:
    """
    Test model performance across SOS width boundary.
    """
    
    def __init__(self, model, task_encoder):
        self.model = model
        self.task_encoder = task_encoder
        self.results = {}
    
    def encode_task(self, task: Task) -> torch.Tensor:
        """Encode task as model input."""
        # Convert task to text description
        text = f"Initial: {', '.join(str(c) for c in task.initial_state)}. "
        text += f"Goal: {', '.join(str(c) for c in task.goal_state)}. "
        text += f"Predict the sequence of actions."
        
        return self.task_encoder(text)
    
    def evaluate_task(
        self, 
        task: Task, 
        n_samples: int = 100
    ) -> Dict:
        """
        Evaluate model on a task.
        
        Returns accuracy and error analysis.
        """
        correct = 0
        constraint_violations = 0
        total_steps = 0
        
        for _ in range(n_samples):
            # Generate problem instance
            encoded = self.encode_task(task)
            
            # Get model prediction
            with torch.no_grad():
                output = self.model(encoded)
            
            # Evaluate (placeholder - actual evaluation depends on task type)
            is_correct = self._check_solution(output, task)
            if is_correct:
                correct += 1
            else:
                violations = self._count_constraint_violations(output, task)
                constraint_violations += violations
            
            total_steps += 1
        
        accuracy = correct / n_samples
        avg_violations = constraint_violations / (n_samples - correct) if correct < n_samples else 0
        
        return {
            'task_name': task.name,
            'sos_width': task.sos_width,
            'accuracy': accuracy,
            'avg_constraint_violations': avg_violations,
            'n_samples': n_samples
        }
    
    def _check_solution(self, output, task) -> bool:
        """Check if solution is correct (placeholder)."""
        # Actual implementation depends on output format
        return True  # Placeholder
    
    def _count_constraint_violations(self, output, task) -> int:
        """Count constraint violations in failed solution (placeholder)."""
        return 0  # Placeholder
    
    def test_boundary(
        self,
        k_values: List[int] = [1, 2, 3, 4, 5]
    ) -> Dict:
        """
        Test performance across SOS width boundary.
        """
        generator = TaskGenerator()
        results = {}
        
        for k in k_values:
            # Generate task for this k
            task = getattr(generator, f'generate_k{k}_task')()
            
            # Evaluate
            result = self.evaluate_task(task)
            results[k] = result
        
        return results
```

---

## Protocol

### Phase 1: Construct Tasks with Controlled SOS Width

```
TASK CONSTRUCTION PROTOCOL:

1. Create tasks for each SOS width k = 1, 2, 3, 4, 5:
   - k=1: Single object tracking
   - k=2: Object + one relationship (blocks world)
   - k=3: Object + two relationships  
   - k=4: Object + three relationships (beyond boundary)
   - k=5: Object + four relationships (well beyond)

2. Verify SOS width:
   - Manually trace optimal solution
   - Confirm maximum simultaneous constraints = k
   - Document the constraint tracking at each step

3. Create multiple instances:
   - 100 instances per k value
   - Vary initial states while preserving SOS width
   - Ensure diversity within each class
```

### Phase 2: Test Fixed-Width Model

```
MODEL TESTING PROTOCOL:

1. Use model with breadth B = 8 (AKIRA-like):
   - 7 spectral bands + 1 temporal
   - Fixed architecture, no scaling

2. For each task (k = 1 to 5):
   a. Run model on 100 instances
   b. Measure:
      - Accuracy (correct solutions)
      - Constraint violations (how many constraints broken)
      - Error pattern (which constraints fail)

3. Record results by SOS width
```

### Phase 3: Identify Sharp Boundary

```
BOUNDARY ANALYSIS PROTOCOL:

1. Plot accuracy vs SOS width:
   - Expected: High accuracy for k ≤ 3
   - Expected: Sharp drop at k = 4
   - Expected: Low accuracy for k ≥ 4

2. Statistical tests:
   - t-test: accuracy(k=3) vs accuracy(k=4)
   - Expected: significant difference (p < 0.01)

3. Analyze failure mode:
   - For k ≥ 4 failures: Which constraints are violated?
   - Expected: Earlier constraints "forgotten"
   - This is the signature of insufficient breadth
```

### Phase 4: Validate with Different Breadths

```
BREADTH VALIDATION PROTOCOL:

1. If possible, test models with different breadths:
   - B = 4: Max k = (4/2) - 1 = 1
   - B = 6: Max k = (6/2) - 1 = 2  
   - B = 8: Max k = (8/2) - 1 = 3
   - B = 10: Max k = (10/2) - 1 = 4

2. Each model should show boundary at predicted k:
   - B=4 model fails at k=2
   - B=6 model fails at k=3
   - B=8 model fails at k=4
   - B=10 model fails at k=5

3. This confirms the formula: max_k = (B/β) - 1
```

---

## Predictions

### If Hypothesis is Correct

```
EXPECTED RESULTS:

┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│  ACCURACY vs SOS WIDTH (B=8 model):                                        │
│                                                                             │
│  Accuracy │                                                                 │
│           │                                                                 │
│    100%   │  ●────●────●                                                   │
│           │             ╲                                                   │
│     80%   │              ╲                                                  │
│           │               ╲                                                 │
│     60%   │                ●                                                │
│           │                                                                 │
│     40%   │                     ●                                           │
│           │                                                                 │
│     20%   │                          ●                                      │
│           │                                                                 │
│      0%   │                                                                 │
│           └────────────────────────────────────────────────►               │
│              k=1    k=2    k=3    k=4    k=5                               │
│                            ↑                                                │
│                         BOUNDARY                                            │
│                         (k=3 max)                                           │
│                                                                             │
│  SHARP TRANSITION at k=4: accuracy drops from ~95% to ~40%                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

QUANTITATIVE PREDICTIONS:

1. Accuracy by SOS width (B=8, β=2):
   - k=1: accuracy > 95%
   - k=2: accuracy > 90%
   - k=3: accuracy > 85%
   - k=4: accuracy < 50%  ← SHARP DROP
   - k=5: accuracy < 30%

2. Transition sharpness:
   - accuracy(k=3) - accuracy(k=4) > 30 percentage points
   - t-test p-value < 0.01

3. Failure mode:
   - k ≥ 4 failures show constraint violations
   - Earlier constraints "forgotten" (breadth exceeded)
   - Not random errors, but systematic "forgetting"

4. Breadth scaling (if tested):
   - Each B value shows boundary at predicted k
   - Formula holds: max_k = (B/β) - 1
```

### Falsification Criteria

```
FALSIFICATION:

The hypothesis is FALSIFIED if:

1. STRONG: No boundary exists
   - Accuracy remains high (> 70%) for all k values
   → Fixed breadth does not limit problem complexity
   → Circuit complexity theory doesn't apply

2. STRONG: Boundary at WRONG k value
   - Boundary at k=2 or k=5 instead of k=4
   → Formula is wrong
   → Different theory needed

3. STRONG: Gradual decline instead of sharp transition
   - Accuracy drops linearly with k, no sharp boundary
   → No phase transition
   → Complexity is continuous, not discrete

4. MODERATE: Failures don't show constraint violations
   - Errors are random, not systematic forgetting
   → Different failure mechanism
   → Not a breadth issue

5. MODERATE: Breadth scaling doesn't follow formula
   - B=6 model has same boundary as B=8
   → Breadth is not the limiting factor
   → Something else constrains capacity
```

---

## Analysis

### Primary Metrics

```python
def analyze_complexity_boundary(results: Dict[int, Dict]) -> Dict:
    """
    Analyze circuit complexity boundary.
    """
    k_values = sorted(results.keys())
    accuracies = [results[k]['accuracy'] for k in k_values]
    
    # 1. Find boundary (largest drop)
    drops = []
    for i in range(len(accuracies) - 1):
        drop = accuracies[i] - accuracies[i + 1]
        drops.append((k_values[i], k_values[i + 1], drop))
    
    max_drop = max(drops, key=lambda x: x[2])
    boundary_k = max_drop[1]  # k value where performance drops
    
    # 2. Statistical test: is drop significant?
    from scipy.stats import ttest_ind
    
    # Get individual trial results (would need to store these)
    # Placeholder for actual implementation
    below_boundary = [results[k]['accuracy'] for k in k_values if k < boundary_k]
    at_or_above = [results[k]['accuracy'] for k in k_values if k >= boundary_k]
    
    # 3. Check if boundary matches prediction
    predicted_boundary = 4  # For B=8, β=2: max_k = 3, so boundary at k=4
    boundary_correct = boundary_k == predicted_boundary
    
    # 4. Transition sharpness
    if boundary_k in results and boundary_k - 1 in results:
        sharpness = results[boundary_k - 1]['accuracy'] - results[boundary_k]['accuracy']
    else:
        sharpness = 0
    
    return {
        'boundary_k': boundary_k,
        'predicted_boundary': predicted_boundary,
        'boundary_correct': boundary_correct,
        'max_drop': max_drop[2],
        'sharpness': sharpness,
        'accuracies': dict(zip(k_values, accuracies)),
        
        # Verdict
        'hypothesis_supported': (
            boundary_correct and
            sharpness > 0.30 and
            results[boundary_k - 1]['accuracy'] > 0.80 and
            results[boundary_k]['accuracy'] < 0.60
        )
    }
```

---

## Expected Outcomes

### If Hypothesis Validated

```
IMPLICATIONS IF TRUE:

1. ARCHITECTURE IS PRINCIPLED:
   AKIRA's 8 bands aren't arbitrary
   They enable k ≤ 3 problems (by design)
   Circuit complexity provides the justification

2. PREDICTABLE LIMITS:
   Know IN ADVANCE what will fail
   k > 3 → will fail (don't try)
   k ≤ 3 → can succeed (worth trying)

3. FAILURE DIAGNOSIS:
   When model fails, ask: "Is k > 3?"
   If yes: not a training problem, architectural limit
   If no: training/data problem

4. SCALING GUIDANCE:
   To solve k=4 problems: need B ≥ 10
   To solve k=5 problems: need B ≥ 12
   Clear prescription for harder tasks

5. TASK DESIGN:
   Design tasks with k ≤ 3 for AKIRA
   Don't expect AKIRA to solve k=4 tasks
   Decompose k=4 tasks into k≤3 subtasks
```

### If Hypothesis Falsified

```
IMPLICATIONS IF FALSE:

1. IF no boundary:
   Fixed breadth is not limiting
   Some other factor constrains capacity
   Circuit complexity theory doesn't apply to LLMs

2. IF wrong boundary:
   Formula is incorrect
   β ≠ 2 for this domain
   Or k is measured incorrectly

3. IF gradual decline:
   No discrete complexity classes
   Simpler model of capacity needed
   Continuous degradation, not phase transition
```

---

## Connection to Other Experiments

| Experiment | Relationship |
|------------|--------------|
| 003 (Spectral Bands) | Bands = breadth; this tests if breadth limits complexity |
| 026 (Band Architecture) | Architecture design based on complexity theory |
| 037 (Task-Relative) | Task complexity varies; this quantifies limits |
| 032 (Plasma Controller) | Plasma has bounded complexity; AKIRA should handle it |

---

## Results

*[ TO BE FILLED AFTER EXPERIMENT ]*

### Accuracy by SOS Width

```
[ PLACEHOLDER FOR RESULTS ]

k=1: accuracy = ____%
k=2: accuracy = ____%
k=3: accuracy = ____%
k=4: accuracy = ____%
k=5: accuracy = ____%

Boundary at k = ____
Predicted boundary: k = 4
```

### Transition Analysis

```
[ PLACEHOLDER FOR RESULTS ]

Drop from k=3 to k=4: ____ percentage points
t-test p-value: ____
```

### Failure Mode Analysis

```
[ PLACEHOLDER FOR RESULTS ]

k=4 failures - constraint violation pattern:
  - First constraint forgotten: ____%
  - Last constraint forgotten: ____%
  - Random pattern: ____%
```

---

## Conclusion

*[ TO BE FILLED AFTER EXPERIMENT ]*

---

## References

1. **Mao, J., Lozano-Perez, T., Tenenbaum, J.B., & Kaelbling, L.P. (2023).** *What Planning Problems Can A Relational Neural Network Solve?* ICLR 2024. arXiv:2312.03682v2. — The foundational paper for SOS width and circuit complexity in neural networks.

2. **Arora, S., & Barak, B. (2009).** *Computational Complexity: A Modern Approach.* Cambridge University Press. — General circuit complexity theory.

3. **Lipovetzky, N., & Geffner, H. (2012).** *Width and Serialization of Classical Planning Problems.* ECAI. — Precursor to SOS width.

4. **AKIRA Internal:** `CIRCUIT_COMPLEXITY.md` — Full treatment of circuit complexity for AKIRA.

5. **AKIRA Internal:** `TERMINOLOGY_FRAMEWORK_OVERVIEW.md` — Architecture breadth = (k+1) x β.

---

*Oscar Goldman — Shogu Research Group @ Datamutant.ai*

*"Architecture is destiny. A network with 8 bands can track 3 constraints. Ask it to track 4, and it will fail - not from lack of training, but from mathematical impossibility. Know your limits."*
