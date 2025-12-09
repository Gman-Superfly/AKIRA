# Experiment 023: Timeline Coherence

## Does the Past Shift After Collapse?

**Tier:** ★ SUPPORTING  
**Status:** PLANNED  
**Depends On:** 002 (Collapse Detection), 007 (Wavefront Interference)

---

## 1. Problem Statement

The homeostat can "reach back in time" and adjust representations, causing past interpretations to change. This appears paradoxical: how can the future change the past?

**The Pythagorean reframe:** This is phase locking across time.

Just as coupled oscillators lock to a common frequency, representations at different times lock to a common interpretation. The "reaching back" is adjustment of early representations to be consistent with the coherent state found later.

---

## 2. Hypothesis

```
THE TIMELINE COHERENCE HYPOTHESIS:

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  After belief collapse, past representations SHIFT toward coherence.  │
│                                                                         │
│  MECHANISM:                                                             │
│  • Before collapse: Multiple interpretations coexist                  │
│  • During collapse: One interpretation dominates                      │
│  • After collapse: Past is reinterpreted for consistency             │
│                                                                         │
│  THIS IS PHASE LOCKING ACROSS TIME:                                    │
│                                                                         │
│  Just as equal temperament "spreads the comma" to close the circle,  │
│  collapse "spreads reinterpretation" to close the timeline.          │
│                                                                         │
│  The past is not stored, it is reconstructed for coherence.          │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Scientific Basis

### 3.1 Memory as Reconstruction

```
MEMORY IS NOT PLAYBACK:

Cognitive science shows memory is reconstructive:
• Memories change when recalled
• Current context affects past interpretation
• "False memories" are coherent constructions

This is not a bug — it's a feature.
It maintains narrative coherence.
```

### 3.2 The Pythagorean Principle

```
COHERENCE OVER PRECISION:

In music: Every interval is slightly impure for universal compatibility
In physics: Phase locking sacrifices natural frequency for sync
In memory: Past is adjusted for narrative coherence

The "error" is distributed to close the circle.
```

### 3.3 The Homeostat as Phase Locker

```
HOMEOSTAT FUNCTION:

• Monitors current state
• Adjusts to maintain balance
• "Reaches back" to reinterpret history

This is temporal phase locking:
• Current state sets the "collective phase"
• Past representations adjust to match
• Timeline becomes coherent
```

### 3.4 AKIRA Theory Basis

**Relevant Theory Documents:**
- `foundations/HARMONY_AND_COHERENCE.md`, §7 (Timeline Coherence)
- `architecture_theoretical/SPECTRAL_BELIEF_MACHINE.md`, §6 (Belief Dynamics)
- `bec/BEC_CONDENSATION_INFORMATION.md`, §4 (Collapse and Coherence)

**Key Concepts:**
- **Timeline coherence:** Past representations shift after collapse to maintain narrative consistency
- **Phase locking across time:** Like coupled oscillators, representations at different times lock to common interpretation
- **Reconstructive memory:** Past is not stored, it is regenerated for coherence with present
- **Collapse propagates backward:** Committed belief at time T influences representations at T-k

**From HARMONY_AND_COHERENCE.md (§7.2):**
> "Timeline coherence: After collapse, system adjusts past representations to be consistent with collapsed state. This is NOT rewriting history, it is phase locking across time. Just as equal temperament spreads comma across keys, collapse spreads reinterpretation across timeline. Narrative coherence demands consistency."

**From SPECTRAL_BELIEF_MACHINE.md (§6.5):**
> "Belief collapse affects not only current state but past representations. Causal attention allows gradient flow backward. When belief commits at position i, positions j < i receive gradient adjusting them for coherence with committed state. Past 'shifts' to be consistent with present."

**From BEC_CONDENSATION_INFORMATION.md (§4.5):**
> "Condensation is global phenomenon. All particles condense to same ground state (phase coherence). In AKIRA, all positions 'condense' to coherent interpretation. This includes past positions, they adjust phase to match collective state."

**This experiment validates:**
1. Whether **past representations shift** after collapse (measurable change)
2. Whether shift is toward **greater coherence** (reduced variance)
3. Whether effect is **strongest near collapse** (time-dependent)
4. Whether this is **phase locking mechanism** (not arbitrary drift)

**Falsification:** If past representations don't shift OR shift is random → no timeline coherence → narrative is forward-only → memory is pure storage (not reconstruction).

---

## 4. Apparatus

### 4.1 Required Measurements

```
MEASUREMENT REQUIREMENTS:

1. REPRESENTATION SNAPSHOTS
   • Capture past token representations at multiple times
   • Before collapse, during collapse, after collapse

2. SIMILARITY METRICS
   • Measure how past representations change
   • Cosine similarity, L2 distance, etc.

3. COLLAPSE DETECTION
   • Identify collapse events (from Exp 002)
   • Time-align measurements to collapse

4. COHERENCE METRIC
   • How consistent is the narrative?
   • Low variance = high coherence
```

### 4.2 Experimental Setup

```python
class TimelineCoherenceAnalyzer:
    """Measures how past representations shift after collapse."""
    
    def __init__(self, model):
        self.model = model
        self.representation_history = []
    
    def capture_representations(self, input_sequence):
        """Capture representations at each position over time."""
        snapshots = []
        for t in range(len(input_sequence)):
            # Get representation of token at position t
            rep = self.model.get_token_representation(input_sequence, position=t)
            snapshots.append(rep)
        return torch.stack(snapshots)
    
    def measure_shift(
        self,
        before_collapse: Tensor,
        after_collapse: Tensor,
        position: int
    ) -> Dict:
        """Measure how representation of a past position changed."""
        return {
            "cosine_sim": F.cosine_similarity(
                before_collapse[position], 
                after_collapse[position],
                dim=0
            ),
            "l2_distance": torch.norm(
                before_collapse[position] - after_collapse[position]
            ),
            "angle_change": self._compute_angle(
                before_collapse[position],
                after_collapse[position]
            ),
        }
    
    def measure_coherence(self, representations: Tensor) -> float:
        """How coherent is the timeline?"""
        # Compute pairwise similarities
        similarities = []
        for i in range(len(representations)):
            for j in range(i+1, len(representations)):
                sim = F.cosine_similarity(
                    representations[i], 
                    representations[j],
                    dim=0
                )
                similarities.append(sim)
        
        # Coherence = inverse of variance
        return 1.0 / (torch.tensor(similarities).var() + 1e-8)
    
    def detect_shift_direction(
        self,
        before: Tensor,
        after: Tensor,
        collapse_direction: Tensor
    ) -> float:
        """Is the shift toward the collapse attractor?"""
        shift = after - before
        return F.cosine_similarity(shift, collapse_direction, dim=0)
```

---

## 5. Method

### 5.1 Protocol

```
EXPERIMENTAL PROTOCOL:

1. PREPARE SEQUENCE
   • Choose sequence that induces collapse
   • Identify collapse point (from Exp 002)

2. CAPTURE BEFORE COLLAPSE
   • At each position, record representation
   • Create "pre-collapse snapshot"

3. OBSERVE COLLAPSE
   • Detect entropy drop (collapse event)
   • Record the "winning" interpretation

4. CAPTURE AFTER COLLAPSE
   • Re-query representations of past positions
   • Create "post-collapse snapshot"

5. MEASURE SHIFTS
   • For each past position:
     - How much did representation change?
     - Did it shift toward collapse attractor?
   
6. MEASURE COHERENCE
   • Compare narrative coherence before/after
   • Expected: coherence increases after collapse
```

### 5.2 Controls

- **No-collapse sequences**: Representations should be stable
- **Random noise**: No coherent shift expected
- **Reversed direction**: Shift should be toward attractor, not away

---

## 6. Predictions

### 6.1 If Hypothesis is Correct

```
EXPECTED RESULTS:

1. PAST REPRESENTATIONS SHIFT
   • Positions before collapse change after collapse
   • Shift magnitude correlates with collapse strength

2. SHIFT IS DIRECTIONAL
   • Shift is toward collapse attractor (winner)
   • Not random drift

3. COHERENCE INCREASES
   • Post-collapse narrative more coherent
   • Lower variance in inter-position similarity

4. EARLIER POSITIONS SHIFT MORE
   • Positions far from collapse have more "slack"
   • Like spreading the comma across more intervals
```

### 6.2 Quantitative Predictions

| Metric | Prediction | Significance |
|--------|------------|--------------|
| Shift magnitude | > 0.1 (cosine distance) | Measurable change |
| Shift direction | > 0.5 (correlation with attractor) | Toward winner |
| Coherence ratio | > 1.5 (after/before) | Increased coherence |
| Position gradient | Negative (earlier = more shift) | Distance effect |

---

## 7. Falsification

### 7.1 What Would Disprove the Hypothesis

```
FALSIFICATION CRITERIA:

1. NO SHIFT
   • Past representations unchanged after collapse
   • Cosine similarity ≈ 1.0
   → Past is immutable, not reconstructed

2. RANDOM SHIFT
   • Shift not correlated with collapse direction
   • No coherence increase
   → Drift is noise, not phase locking

3. AWAY FROM ATTRACTOR
   • Shift is away from winning interpretation
   → Opposite of prediction
```

### 7.2 Alternative Interpretations

If falsified, possible alternatives:
- Representations are cached, not reconstructed
- Coherence is achieved differently (forward only)
- The homeostat operates differently than expected

---

## 8. Results

*To be filled after experiment*

### 8.1 Shift Measurements

| Position | Cosine Before | Cosine After | Shift Magnitude |
|----------|---------------|--------------|-----------------|
| t-10 | | | |
| t-5 | | | |
| t-1 | | | |

### 8.2 Coherence Comparison

| Metric | Before Collapse | After Collapse | Ratio |
|--------|-----------------|----------------|-------|
| Variance | | | |
| Coherence | | | |

### 8.3 Shift Direction Analysis

*Space for direction correlation plots*

---

## 9. Conclusion

*To be filled after experiment*

### 9.1 Summary

### 9.2 Implications

### 9.3 Next Steps

---

## References

- `foundations/HARMONY_AND_COHERENCE.md`, Timeline collapse as phase locking
- `002_EXP_COLLAPSE_DETECTION.md`, Detecting collapse events
- `007_EXP_WAVEFRONT_INTERFERENCE.md`, Error propagation dynamics
- Cognitive science literature on reconstructive memory

---

*"The past is not what happened. It's the coherent version of what happened. The version that survived interference. The version where the phases aligned."*

*Oscar Goldman, Shogu Research Group @ Datamutant.ai subsidiary of 温心重工業*