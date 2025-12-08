# 041: Circuit Complexity Boundary

## Quick Summary

**Question:** Do fixed-width networks have a hard performance boundary determined by SOS width?

**Hypothesis:** For breadth B=8 and predicate arity β=2, problems with SOS width k > 3 should systematically fail.

**Method:** Construct tasks with controlled SOS width (k=1 to 5), test model performance, identify sharp boundary.

**Key Prediction:** Sharp accuracy drop at k=4 (from >85% to <50%).

## Status

- **Tier:** SUPPORTING
- **Status:** PLANNED
- **Dependencies:** None (standalone)

## Files

- `041_EXP_CIRCUIT_COMPLEXITY_BOUNDARY.md` - Full experiment specification
- `code/` - Implementation (to be added)
- `results/` - Results (to be filled after experiment)

## The Circuit Complexity Theorem

```
REQUIRED BREADTH = (k + 1) × β

For AKIRA (B=8, β=2):
  Max solvable k = (8/2) - 1 = 3

k ≤ 3: CAN solve (mathematically possible)
k > 3: CANNOT solve (mathematically impossible)
```

## Task Examples

```
k=1: Track one object
k=2: Track object + one relationship (blocks world)
k=3: Track object + two relationships
k=4: Track object + three relationships (BEYOND CAPACITY)
k=5: Track object + four relationships (WELL BEYOND)
```

## Quick Start

```python
# Pseudocode for experiment
generator = TaskGenerator()
tester = ComplexityBoundaryTester(model)

# Test each k value
for k in [1, 2, 3, 4, 5]:
    task = generator.generate_task(sos_width=k)
    result = tester.evaluate_task(task)
    # Expect: accuracy drops sharply at k=4
```

## Connection to Theory

From `CIRCUIT_COMPLEXITY.md`:

> "A relational neural network can solve it if and only if:
> REQUIRED BREADTH = (k + 1) × β
> This is NECESSARY and SUFFICIENT.
> Less breadth → Cannot solve the problem (mathematical impossibility)"
