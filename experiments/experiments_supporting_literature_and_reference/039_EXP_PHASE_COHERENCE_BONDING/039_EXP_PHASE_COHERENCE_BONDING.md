# Experiment 039: Phase Coherence Bonding

## AQ Combination via Phase Alignment

**Tier:** ★★ CORE  
**Status:** PLANNED  
**Depends On:** 000 (AQ Extraction), 003 (Spectral Band Dynamics), 022 (Band Phase Locking)  
**References:** THE_LANGUAGE_OF_INFORMATION.md Section 3, ACTION_QUANTA.md Section 6

---

## Motivation

### The Core Claim Being Tested

```
AQ BONDING VIA PHASE COHERENCE
──────────────────────────────

From THE_LANGUAGE_OF_INFORMATION.md Section 3:

  "COHERENT BONDS (Constructive Interference):
   Atoms with aligned phase combine to reinforce.
   'Cat' + 'sits' + 'mat' = coherent meaning
   The whole is MORE than the sum of parts.
   Energy increases."

  "INHIBITORY BONDS (Destructive Interference):
   Atoms with opposing phase combine to cancel.
   'Cat' + 'not cat' = confusion or null
   The whole is LESS than the sum of parts.
   Energy decreases."

From ACTION_QUANTA.md Section 6.2:

  "COHERENT BONDS (Phase Alignment):
   AQ at SAME frequency, ALIGNED phase
   Result: Constructive interference
   
   Example: Multiple edges forming a contour
   Edge₁ (phase 0°) + Edge₂ (phase 0°) + Edge₃ (phase 0°)
   = Coherent contour"

THE HYPOTHESIS:
  Semantically COHERENT combinations show ALIGNED phases
    → E_combined > E_1 + E_2 (constructive)
  Semantically CONFLICTING combinations show OPPOSING phases
    → E_combined < E_1 + E_2 (destructive)
  Phase difference predicts combination type
```

### Why This Matters

1. **Mechanism of Composition:** If phase alignment determines bonding, we understand HOW meaning composes. This is the "grammar" of information combination.

2. **Prediction:** Given two AQ, we can predict whether their combination will be coherent or conflicting by measuring phase.

3. **Architecture Design:** Systems should be designed to align phases for coherent combination and misalign for inhibition.

4. **Error Detection:** Incoherent outputs might be diagnosed as phase misalignment failures.

---

## Foundation

**Established Science:**

1. **Superposition Principle** - Waves combine according to their relative phases. In-phase = constructive, out-of-phase = destructive.

2. **Coherence in Neural Networks** - Oscillatory neural activity shows phase locking during binding (Fries, 2005). Phase synchrony correlates with feature binding in perception.

3. **Transformer Representations** - Embeddings can be decomposed into magnitude and phase components. Relative phase affects combination.

**Connection to AKIRA:**

If AQ behave like wave-like patterns (as claimed), then their combination should follow superposition rules:
- Aligned phase → constructive interference → amplified combined representation
- Opposing phase → destructive interference → cancelled combined representation

**Hypothesis:** Semantically coherent concept pairs ("hot" + "fire") show phase alignment (|Δφ| < π/2) and energy amplification. Semantically conflicting pairs ("big" + "small") show phase opposition (|Δφ| > π/2) and energy reduction.

---

## Apparatus

### Required Infrastructure

```python
from typing import Dict, List, Tuple
import torch
import numpy as np
from scipy.fft import fft, ifft

class PhaseCoherenceAnalyzer:
    """
    Analyze phase relationships in embeddings and their effect on combination.
    """
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
    
    def get_embedding(self, text: str) -> torch.Tensor:
        """Get embedding for a word or phrase."""
        inputs = self.tokenizer(text, return_tensors="pt")
        
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
        
        # Get last hidden state, average over tokens
        hidden = outputs.hidden_states[-1]  # [1, seq_len, hidden_dim]
        return hidden.mean(dim=1).squeeze()  # [hidden_dim]
    
    def decompose_to_magnitude_phase(
        self, 
        embedding: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decompose embedding into magnitude and phase components.
        
        Uses FFT to extract spectral representation.
        """
        emb_np = embedding.numpy()
        
        # FFT decomposition
        fft_result = fft(emb_np)
        
        magnitude = np.abs(fft_result)
        phase = np.angle(fft_result)
        
        return torch.tensor(magnitude), torch.tensor(phase)
    
    def compute_phase_difference(
        self,
        embedding_1: torch.Tensor,
        embedding_2: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute phase difference between two embeddings.
        
        Returns phase difference for each frequency component.
        """
        _, phase_1 = self.decompose_to_magnitude_phase(embedding_1)
        _, phase_2 = self.decompose_to_magnitude_phase(embedding_2)
        
        # Phase difference (wrapped to [-π, π])
        delta_phase = phase_1 - phase_2
        delta_phase = torch.atan2(torch.sin(delta_phase), torch.cos(delta_phase))
        
        return delta_phase
    
    def compute_coherence_score(
        self,
        embedding_1: torch.Tensor,
        embedding_2: torch.Tensor
    ) -> float:
        """
        Compute overall phase coherence between two embeddings.
        
        High coherence (near 1) = aligned phases
        Low coherence (near 0) = random phases
        Negative coherence = opposing phases
        """
        delta_phase = self.compute_phase_difference(embedding_1, embedding_2)
        
        # Coherence = mean of cos(delta_phase)
        # cos(0) = 1 (aligned), cos(π) = -1 (opposing)
        coherence = torch.cos(delta_phase).mean().item()
        
        return coherence
    
    def compute_combination_energy(
        self,
        embedding_1: torch.Tensor,
        embedding_2: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compute energy of individual embeddings vs their combination.
        
        Returns:
            Dict with E_1, E_2, E_combined, and interference_type
        """
        # Individual energies
        E_1 = (embedding_1 ** 2).sum().item()
        E_2 = (embedding_2 ** 2).sum().item()
        
        # Combined embedding (element-wise addition, as per superposition)
        combined = embedding_1 + embedding_2
        E_combined = (combined ** 2).sum().item()
        
        # Expected energy if no interference
        E_expected = E_1 + E_2
        
        # Interference ratio
        interference_ratio = E_combined / E_expected
        
        # Classify interference type
        if interference_ratio > 1.1:
            interference_type = "constructive"
        elif interference_ratio < 0.9:
            interference_type = "destructive"
        else:
            interference_type = "neutral"
        
        return {
            'E_1': E_1,
            'E_2': E_2,
            'E_combined': E_combined,
            'E_expected': E_expected,
            'interference_ratio': interference_ratio,
            'interference_type': interference_type
        }
    
    def analyze_pair(
        self,
        word_1: str,
        word_2: str
    ) -> Dict:
        """
        Full analysis of a word pair.
        """
        emb_1 = self.get_embedding(word_1)
        emb_2 = self.get_embedding(word_2)
        
        coherence = self.compute_coherence_score(emb_1, emb_2)
        energy = self.compute_combination_energy(emb_1, emb_2)
        phase_diff = self.compute_phase_difference(emb_1, emb_2)
        
        return {
            'word_1': word_1,
            'word_2': word_2,
            'coherence': coherence,
            'energy': energy,
            'mean_phase_diff': phase_diff.abs().mean().item(),
            'phase_diff_std': phase_diff.std().item()
        }


class BondingExperiment:
    """
    Run experiments on different types of word pairs.
    """
    
    def __init__(self, analyzer: PhaseCoherenceAnalyzer):
        self.analyzer = analyzer
        
    def run_coherent_pairs(self) -> List[Dict]:
        """Test semantically coherent pairs."""
        coherent_pairs = [
            ("hot", "fire"),
            ("cold", "ice"),
            ("happy", "joy"),
            ("sad", "tears"),
            ("fast", "speed"),
            ("slow", "turtle"),
            ("big", "giant"),
            ("small", "tiny"),
            ("cat", "meow"),
            ("dog", "bark"),
            ("sun", "bright"),
            ("moon", "night"),
            ("water", "wet"),
            ("desert", "dry"),
            ("doctor", "hospital"),
            ("teacher", "school"),
            ("king", "throne"),
            ("ocean", "waves"),
            ("mountain", "peak"),
            ("forest", "trees")
        ]
        
        results = []
        for w1, w2 in coherent_pairs:
            result = self.analyzer.analyze_pair(w1, w2)
            result['expected_type'] = 'coherent'
            results.append(result)
        
        return results
    
    def run_conflicting_pairs(self) -> List[Dict]:
        """Test semantically conflicting pairs."""
        conflicting_pairs = [
            ("hot", "cold"),
            ("big", "small"),
            ("fast", "slow"),
            ("happy", "sad"),
            ("true", "false"),
            ("up", "down"),
            ("left", "right"),
            ("good", "bad"),
            ("light", "dark"),
            ("alive", "dead"),
            ("open", "closed"),
            ("full", "empty"),
            ("wet", "dry"),
            ("young", "old"),
            ("love", "hate"),
            ("peace", "war"),
            ("success", "failure"),
            ("rich", "poor"),
            ("strong", "weak"),
            ("early", "late")
        ]
        
        results = []
        for w1, w2 in conflicting_pairs:
            result = self.analyzer.analyze_pair(w1, w2)
            result['expected_type'] = 'conflicting'
            results.append(result)
        
        return results
    
    def run_neutral_pairs(self) -> List[Dict]:
        """Test semantically unrelated pairs (control)."""
        neutral_pairs = [
            ("cat", "telephone"),
            ("mountain", "keyboard"),
            ("happy", "purple"),
            ("doctor", "banana"),
            ("ocean", "pencil"),
            ("king", "sandwich"),
            ("tree", "algorithm"),
            ("sun", "carpet"),
            ("music", "brick"),
            ("paper", "gravity"),
            ("book", "volcano"),
            ("chair", "dream"),
            ("window", "philosophy"),
            ("garden", "mathematics"),
            ("bridge", "poetry"),
            ("clock", "elephant"),
            ("mirror", "thunder"),
            ("ladder", "symphony"),
            ("candle", "democracy"),
            ("blanket", "equation")
        ]
        
        results = []
        for w1, w2 in neutral_pairs:
            result = self.analyzer.analyze_pair(w1, w2)
            result['expected_type'] = 'neutral'
            results.append(result)
        
        return results
```

---

## Protocol

### Phase 1: Pair Analysis

```
PAIR ANALYSIS PROTOCOL:

1. Run analysis on all three pair types:
   - 20 coherent pairs (semantically related, reinforcing)
   - 20 conflicting pairs (semantic opposites)
   - 20 neutral pairs (unrelated, control)

2. For each pair, measure:
   - Phase coherence score (mean cos(Δφ))
   - Interference ratio (E_combined / E_expected)
   - Mean absolute phase difference
   - Phase difference variance

3. Compare distributions:
   - Coherent pairs should have HIGH coherence, HIGH interference ratio
   - Conflicting pairs should have LOW/NEGATIVE coherence, LOW interference ratio
   - Neutral pairs should be in between (baseline)
```

### Phase 2: Sentence-Level Bonding

```
SENTENCE BONDING PROTOCOL:

1. Construct sentences with coherent vs conflicting elements:
   
   COHERENT: "The hot fire burned brightly."
   CONFLICTING: "The hot cold burned brightly."
   
   COHERENT: "The big giant towered above."
   CONFLICTING: "The big small towered above."

2. Measure:
   - Full sentence embedding energy
   - Component embedding energies
   - Per-word phase alignment within sentence

3. Predictions:
   - Coherent sentences: Higher combined energy, more phase alignment
   - Conflicting sentences: Lower combined energy, phase opposition
```

### Phase 3: Triplet Bonding

```
TRIPLET BONDING PROTOCOL:

1. Test three-way combinations:
   
   FULLY COHERENT: "hot" + "fire" + "burn"
   PARTIALLY COHERENT: "hot" + "fire" + "ice"
   FULLY CONFLICTING: "hot" + "cold" + "warm"

2. Measure:
   - Pairwise phase coherences
   - Three-way interference
   - Coherence transitivity (if A~B and B~C, does A~C?)

3. Predictions:
   - Fully coherent: All pairs aligned, maximum combined energy
   - Partially coherent: Mixed alignment, moderate energy
   - Fully conflicting: Maximum cancellation, minimum energy
```

### Phase 4: Layer-Wise Analysis

```
LAYER ANALYSIS PROTOCOL:

1. Extract embeddings at multiple layers (not just final):
   - Early layers (0-4)
   - Middle layers (8-16)
   - Late layers (20-24)

2. Measure phase coherence at each layer for same pairs

3. Predictions:
   - Phase alignment should INCREASE in later layers
   - Early layers: Raw representations, less structured
   - Late layers: Semantic alignment emerges
```

---

## Predictions

### If Hypothesis is Correct

```
EXPECTED RESULTS:

┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│  COHERENCE DISTRIBUTION BY PAIR TYPE:                                       │
│                                                                             │
│  Frequency │                                                                │
│           │                    ╭───╮                                        │
│           │                   ╱     ╲                                       │
│           │   ╭───╮          ╱       ╲         ╭───╮                        │
│           │  ╱     ╲        ╱         ╲       ╱     ╲                       │
│           │ ╱       ╲      ╱           ╲     ╱       ╲                      │
│           │╱         ╲____╱             ╲___╱         ╲                     │
│           └───────────────────────────────────────────────► Coherence      │
│             -1          0                              +1                   │
│                                                                             │
│             CONFLICTING   NEUTRAL          COHERENT                        │
│                                                                             │
│  Conflicting pairs: Mean coherence < 0 (opposing phases)                   │
│  Neutral pairs: Mean coherence ≈ 0 (random phases)                         │
│  Coherent pairs: Mean coherence > 0.3 (aligned phases)                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│  INTERFERENCE RATIO BY PAIR TYPE:                                           │
│                                                                             │
│  E_combined / E_expected                                                    │
│           │                                                                 │
│     2.0   │                              ●                                  │
│           │                           ●     ●                               │
│     1.5   │                        ●           ●                            │
│           │                                       Coherent                  │
│     1.0   │  ─────────────●────●────────────────────────── (baseline)      │
│           │            ●                          Neutral                   │
│     0.5   │      ●  ●                                                       │
│           │   ●        Conflicting                                          │
│     0.0   │                                                                 │
│           └─────────────────────────────────────────────►                   │
│                                                                             │
│  Conflicting: ratio < 0.9 (destructive interference)                       │
│  Neutral: ratio ≈ 1.0 (no systematic interference)                         │
│  Coherent: ratio > 1.1 (constructive interference)                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

QUANTITATIVE PREDICTIONS:

1. Phase coherence:
   - Coherent pairs: mean coherence > 0.3
   - Conflicting pairs: mean coherence < 0 (or < 0.1)
   - Separation: coherent_mean - conflicting_mean > 0.3

2. Interference ratio:
   - Coherent pairs: ratio > 1.1
   - Conflicting pairs: ratio < 0.9
   - Neutral pairs: 0.95 < ratio < 1.05

3. Correlation:
   - Coherence should correlate with interference ratio (r > 0.5)
   - Higher coherence → higher interference ratio

4. Layer progression:
   - Coherence difference (coherent - conflicting) increases with depth
   - Early layers: small difference
   - Late layers: large difference (semantics emerge)
```

### Falsification Criteria

```
FALSIFICATION:

The hypothesis is FALSIFIED if:

1. STRONG: No coherence difference between pair types
   - |mean_coherent - mean_conflicting| < 0.1
   → Phase is not related to semantic coherence
   → Bonding mechanism is not phase-based

2. STRONG: Interference ratio uncorrelated with coherence
   - Correlation |r| < 0.2
   → Energy combination is not phase-dependent
   → Superposition model is wrong

3. STRONG: Conflicting pairs show CONSTRUCTIVE interference
   - Conflicting ratio > 1.0
   → Opposite prediction to hypothesis
   → Fundamental misunderstanding of mechanism

4. MODERATE: No layer progression
   - Coherence patterns same at all layers
   → Semantic alignment is not computed
   → It's a property of embeddings, not processing

5. MODERATE: Neutral pairs show strong interference
   - Neutral ratio significantly different from 1.0
   → Interference is not semantically driven
   → Random relationships show systematic patterns
```

---

## Analysis

### Primary Metrics

```python
def analyze_phase_bonding(
    coherent_results: List[Dict],
    conflicting_results: List[Dict],
    neutral_results: List[Dict]
) -> Dict:
    """
    Analyze phase bonding experiment results.
    """
    def extract_metrics(results: List[Dict]) -> Dict:
        coherences = [r['coherence'] for r in results]
        ratios = [r['energy']['interference_ratio'] for r in results]
        
        return {
            'mean_coherence': np.mean(coherences),
            'std_coherence': np.std(coherences),
            'mean_ratio': np.mean(ratios),
            'std_ratio': np.std(ratios),
            'coherences': coherences,
            'ratios': ratios
        }
    
    coherent_metrics = extract_metrics(coherent_results)
    conflicting_metrics = extract_metrics(conflicting_results)
    neutral_metrics = extract_metrics(neutral_results)
    
    # Compute separations
    coherence_separation = (
        coherent_metrics['mean_coherence'] - 
        conflicting_metrics['mean_coherence']
    )
    ratio_separation = (
        coherent_metrics['mean_ratio'] - 
        conflicting_metrics['mean_ratio']
    )
    
    # Compute correlation (coherence vs ratio)
    all_coherences = (
        coherent_metrics['coherences'] + 
        conflicting_metrics['coherences'] + 
        neutral_metrics['coherences']
    )
    all_ratios = (
        coherent_metrics['ratios'] + 
        conflicting_metrics['ratios'] + 
        neutral_metrics['ratios']
    )
    correlation = np.corrcoef(all_coherences, all_ratios)[0, 1]
    
    # Statistical tests
    from scipy.stats import ttest_ind
    
    t_coherence, p_coherence = ttest_ind(
        coherent_metrics['coherences'],
        conflicting_metrics['coherences']
    )
    
    t_ratio, p_ratio = ttest_ind(
        coherent_metrics['ratios'],
        conflicting_metrics['ratios']
    )
    
    return {
        'coherent': coherent_metrics,
        'conflicting': conflicting_metrics,
        'neutral': neutral_metrics,
        'coherence_separation': coherence_separation,
        'ratio_separation': ratio_separation,
        'coherence_ratio_correlation': correlation,
        'coherence_ttest': {'t': t_coherence, 'p': p_coherence},
        'ratio_ttest': {'t': t_ratio, 'p': p_ratio},
        
        # Verdict
        'hypothesis_supported': (
            coherence_separation > 0.2 and
            p_coherence < 0.05 and
            correlation > 0.3
        )
    }
```

---

## Expected Outcomes

### If Hypothesis Validated

```
IMPLICATIONS IF TRUE:

1. MECHANISM REVEALED:
   Phase alignment IS the mechanism of semantic composition
   We understand HOW meaning combines
   Not just correlation, but causal mechanism

2. PREDICTION CAPABILITY:
   Given two embeddings, predict if they will combine constructively
   Phase measurement → combination outcome
   Enable "semantic compatibility" estimation

3. ARCHITECTURE DESIGN:
   Design attention to align phases for combination
   Inhibition = phase inversion
   Binding = phase synchronization

4. COMPOSITIONAL SEMANTICS:
   Compositional meaning follows wave mechanics
   "Hot" + "fire" = constructive superposition
   "Hot" + "cold" = destructive interference
   Formal semantics via physics

5. ERROR DETECTION:
   Incoherent outputs = phase misalignment
   Can diagnose WHERE combination failed
   Phase analysis as debugging tool
```

### If Hypothesis Falsified

```
IMPLICATIONS IF FALSE:

1. IF no coherence difference:
   Phase is not semantically meaningful
   Bonding mechanism is different
   Look elsewhere for composition rules

2. IF no energy interference:
   Superposition model is wrong
   Combination is not additive
   Different mathematical framework needed

3. IF layer-invariant:
   Semantic structure is in embeddings, not processing
   Simpler than thought
   Static rather than dynamic
```

---

## Connection to Other Experiments

| Experiment | Relationship |
|------------|--------------|
| 000 (AQ Extraction) | Tests whether extracted AQ show phase structure |
| 003 (Spectral Bands) | Phase may differ by band |
| 022 (Band Phase Locking) | Tests phase relationships between bands |
| 024 (Resonant Wormholes) | Phase alignment may trigger wormholes |
| 025 (Synergy-Redundancy) | Phase alignment may correspond to redundancy |

---

## Results

*[ TO BE FILLED AFTER EXPERIMENT ]*

### Coherence by Pair Type

```
[ PLACEHOLDER FOR RESULTS ]

Coherent pairs:
  Mean coherence: ____
  Std: ____

Conflicting pairs:
  Mean coherence: ____
  Std: ____

Neutral pairs:
  Mean coherence: ____
  Std: ____

Separation (coherent - conflicting): ____
t-test p-value: ____
```

### Interference Ratio by Pair Type

```
[ PLACEHOLDER FOR RESULTS ]

Coherent pairs:
  Mean ratio: ____
  
Conflicting pairs:
  Mean ratio: ____
  
Correlation (coherence vs ratio): ____
```

### Layer Progression

```
[ PLACEHOLDER FOR RESULTS ]

Layer 4:  coherence separation = ____
Layer 12: coherence separation = ____
Layer 20: coherence separation = ____
```

---

## Conclusion

*[ TO BE FILLED AFTER EXPERIMENT ]*

---

## References

1. **Fries, P. (2005).** *A mechanism for cognitive dynamics: neuronal communication through neuronal coherence.* Trends in Cognitive Sciences., Phase synchrony in neural binding.

2. **Singer, W. (1999).** *Neuronal synchrony: A versatile code for the definition of relations?* Neuron., Phase as binding mechanism.

3. **AKIRA Internal:** `THE_LANGUAGE_OF_INFORMATION.md` Section 3, Grammatical rules of AQ bonding.

4. **AKIRA Internal:** `ACTION_QUANTA.md` Section 6, Types of AQ bonds.

5. **AKIRA Internal:** `HARMONY_AND_COHERENCE.md`, Phase alignment as harmony.

---

*

*"Meaning composes like waves. In-phase concepts reinforce; out-of-phase concepts cancel. The grammar of thought is the physics of interference."*


*Oscar Goldman, Shogu Research Group @ Datamutant.ai subsidiary of 温心重工業*

