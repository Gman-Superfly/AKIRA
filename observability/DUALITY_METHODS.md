# DUALITY METHODS FOR OBSERVABILITY

## Hard-to-Observe ↔ Easy-to-Observe Swaps

**Oscar Goldman — Shogu Research Group @ Datamutant.ai**

---

## The Observability Principle

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  DUALITY FOR OBSERVABILITY:                                            │
│                                                                         │
│  Every duality provides a HARD↔EASY swap.                             │
│  What's HARD to observe in Domain A is EASY in Domain B.              │
│  The transform cost is usually cheap (O(N log N) or O(N)).            │
│                                                                         │
│  THE OBSERVABILITY PATTERN:                                            │
│                                                                         │
│  1. Identify what's HARD to observe directly                          │
│  2. Find a duality where it becomes EASY                              │
│  3. Transform → Observe → Transform back (if needed)                  │
│  4. Verify via the conserved quantity                                 │
│                                                                         │
│  ════════════════════════════════════════════════════════════════════  │
│                                                                         │
│  This document catalogs HARD↔EASY swaps for observing AKIRA.          │
│  Each duality section lists:                                           │
│  • What's hard to see in the original domain                          │
│  • What becomes easy in the dual domain                               │
│  • How to use this for observability                                  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 1. Spatial ↔ Frequency Duality

### The Intuition

```
THE SPATIAL-FREQUENCY RELATIONSHIP:

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  SPATIAL DOMAIN:                                                        │
│  Tells you WHERE things are.                                           │
│  Like looking at a photo — you see positions, edges, objects.          │
│                                                                         │
│  FREQUENCY DOMAIN:                                                      │
│  Tells you WHAT PATTERNS exist.                                        │
│  Like listening to music — you hear bass (low freq), treble (high).   │
│                                                                         │
│  THE RELATIONSHIP:                                                      │
│  • A sharp spike in space → spread across all frequencies             │
│  • A pure tone (one frequency) → spread across all space              │
│  • This is Heisenberg uncertainty: Δx·Δk ≥ constant                   │
│                                                                         │
│  SWITCHING IS CHEAP: FFT costs only O(N log N).                        │
│  USE BOTH. They show complementary information.                        │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### The Observability Swap

```
SPATIAL ↔ FREQUENCY: HARD↔EASY FOR OBSERVABILITY

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  TRANSFORM: FFT — O(N log N)                                           │
│  CONSERVED: Energy (Parseval) — use for validation                     │
│                                                                         │
│  ════════════════════════════════════════════════════════════════════  │
│                                                                         │
│  HARD TO OBSERVE IN SPATIAL:    EASY TO OBSERVE IN FREQUENCY:         │
│  ────────────────────────────   ─────────────────────────────         │
│  Global patterns                One coefficient per pattern            │
│  Multi-scale structure          Band boundaries                        │
│  Dominant wavelengths           Peak locations                         │
│  Texture periodicity            Spectral peaks                         │
│  Energy per scale               Band energy directly                   │
│                                                                         │
│  HARD TO OBSERVE IN FREQUENCY:  EASY TO OBSERVE IN SPATIAL:           │
│  ────────────────────────────   ─────────────────────────────         │
│  Position of features           Direct coordinates                     │
│  Local anomalies                Local deviation                        │
│  Boundary locations             Edge detection                         │
│  Spatial relationships          Distance metrics                       │
│                                                                         │
│  ════════════════════════════════════════════════════════════════════  │
│                                                                         │
│  OBSERVABILITY STRATEGY:                                               │
│                                                                         │
│  To observe WHAT patterns exist:    Use frequency domain              │
│  To observe WHERE patterns are:     Use spatial domain                │
│  To observe BOTH:                   Compare both representations      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### How to Use

```python
"""
SPATIAL ↔ FREQUENCY DUALITY
"""

def spatial_to_frequency(x):
    """Transform to frequency domain."""
    return torch.fft.fft2(x)


def frequency_to_spatial(X):
    """Transform back to spatial domain."""
    return torch.fft.ifft2(X).real


def analyze_both_domains(x):
    """Analyze in both domains for complementary info."""
    X = spatial_to_frequency(x)
    magnitude = X.abs()
    phase = X.angle()
    
    spatial_info = {
        'mean': x.mean().item(),
        'std': x.std().item(),
        'max_pos': x.argmax().item(),
        'local_variance': compute_local_variance(x)
    }
    
    frequency_info = {
        'dominant_freq': magnitude.argmax().item(),
        'low_freq_energy': magnitude[:, :magnitude.size(1)//4].pow(2).sum().item(),
        'high_freq_energy': magnitude[:, magnitude.size(1)//4:].pow(2).sum().item(),
        'phase_coherence': compute_phase_coherence(phase)
    }
    
    return {
        'spatial': spatial_info,
        'frequency': frequency_info,
        'energy_ratio': frequency_info['low_freq_energy'] / 
                       (frequency_info['high_freq_energy'] + 1e-10)
    }
```

### What It Reveals

```
SWITCHING DOMAINS REVEALS:

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  PATTERN                SPATIAL SIGNATURE      FREQUENCY SIGNATURE    │
│  ───────                ─────────────────      ───────────────────    │
│                                                                         │
│  Object                 Localized blob          Spread spectrum       │
│  Edge                   Sharp transition        High frequency        │
│  Texture                Repeated pattern        Peaks at period       │
│  Noise                  Random variation        Flat spectrum         │
│  Smooth gradient        Slow change             Low frequency only    │
│                                                                         │
│  USE CASE:                                                              │
│  • Filtering: easier in frequency                                     │
│  • Localization: easier in spatial                                    │
│  • Scale analysis: frequency                                          │
│  • Boundary detection: spatial                                        │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Magnitude ↔ Phase Duality

### The Intuition

```
THE MAGNITUDE-PHASE RELATIONSHIP:

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  MAGNITUDE:                                                             │
│  Tells you HOW MUCH of each frequency is present.                      │
│  Like a music equalizer showing bass/mid/treble levels.                │
│                                                                         │
│  PHASE:                                                                 │
│  Tells you WHERE the waves line up.                                    │
│  Like knowing when the beats synchronize.                              │
│                                                                         │
│  THE INSIGHT:                                                           │
│  • Phase carries STRUCTURE (edges, positions, layout)                  │
│  • Magnitude carries IDENTITY (what kind of thing)                     │
│                                                                         │
│  SURPRISING FACT:                                                       │
│  If you swap the phase of two images but keep their magnitudes,        │
│  the result looks like the image whose PHASE you used.                 │
│  → Phase matters more for structure than magnitude does.               │
│                                                                         │
│  AKIRA APPLICATION:                                                     │
│  Phase alignment across bands = coherent understanding                 │
│  Magnitude distribution = what patterns are present                    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### The Observability Swap

```
MAGNITUDE ↔ PHASE: HARD↔EASY FOR OBSERVABILITY

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  TRANSFORM: Polar decomposition — O(N)                                 │
│  CONSERVED: Complex structure (mag × e^(iφ) = original)               │
│                                                                         │
│  ════════════════════════════════════════════════════════════════════  │
│                                                                         │
│  HARD TO OBSERVE IN MAGNITUDE:  EASY TO OBSERVE IN PHASE:             │
│  ─────────────────────────────  ───────────────────────────           │
│  Object positions               Phase gradients → positions           │
│  Edge locations                 Phase discontinuities                  │
│  Structural alignment           Phase coherence                        │
│  Spatial layout                 Phase relationships                    │
│                                                                         │
│  HARD TO OBSERVE IN PHASE:      EASY TO OBSERVE IN MAGNITUDE:         │
│  ─────────────────────────────  ───────────────────────────           │
│  Object identity (WHAT)         Power at each frequency               │
│  Texture type                   Spectral profile                       │
│  Energy distribution            Magnitude histogram                    │
│  Dominant frequencies           Peak magnitudes                        │
│                                                                         │
│  ════════════════════════════════════════════════════════════════════  │
│                                                                         │
│  OBSERVABILITY INSIGHT:                                                │
│                                                                         │
│  Phase carries WHERE information → observe phase for position         │
│  Magnitude carries WHAT information → observe magnitude for identity  │
│                                                                         │
│  If you see phase scrambled but magnitude intact:                     │
│  → Model knows WHAT but lost WHERE                                    │
│                                                                         │
│  If you see magnitude scrambled but phase intact:                     │
│  → Model knows WHERE but lost WHAT                                    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### How to Use

```python
"""
MAGNITUDE ↔ PHASE DUALITY
"""

def decompose_magnitude_phase(x):
    """Decompose into magnitude and phase."""
    X = torch.fft.fft2(x)
    magnitude = X.abs()
    phase = X.angle()
    return magnitude, phase


def recombine(magnitude, phase):
    """Recombine magnitude and phase into signal."""
    X = magnitude * torch.exp(1j * phase)
    return torch.fft.ifft2(X).real


def swap_phases(x1, x2):
    """Swap phases between two signals (demonstrates phase importance)."""
    mag1, phase1 = decompose_magnitude_phase(x1)
    mag2, phase2 = decompose_magnitude_phase(x2)
    
    # Magnitude from x2, phase from x1
    hybrid = recombine(mag2, phase1)
    
    return hybrid  # Will look like x1!


def analyze_magnitude_phase(x):
    """Extract information from both representations."""
    magnitude, phase = decompose_magnitude_phase(x)
    
    magnitude_info = {
        'total_energy': magnitude.pow(2).sum().item(),
        'spectral_centroid': compute_spectral_centroid(magnitude),
        'energy_concentration': (magnitude.max() ** 2) / magnitude.pow(2).sum().item()
    }
    
    phase_info = {
        'phase_gradient': compute_phase_gradient(phase),
        'phase_coherence': compute_local_phase_coherence(phase),
        'dominant_orientation': compute_dominant_orientation(phase)
    }
    
    return {
        'magnitude': magnitude_info,
        'phase': phase_info
    }
```

### What It Reveals

```
MAGNITUDE vs PHASE: Different Information

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  MAGNITUDE reveals:              PHASE reveals:                        │
│  ─────────────────              ─────────────                          │
│                                                                         │
│  • Frequency content             • Edge locations                     │
│  • Dominant scales               • Object positions                   │
│  • Texture type                  • Structural alignment               │
│  • Energy distribution           • Geometric relationships            │
│                                                                         │
│  FOR AKIRA SPECIFICALLY:                                                │
│                                                                         │
│  • WHAT-path uses magnitude (object identity)                         │
│  • WHERE-path uses phase (object position)                            │
│  • Band 0-2 care more about magnitude (what exists)                   │
│  • Band 4-6 care more about phase (where exactly)                     │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Forward ↔ Backward Duality

### The Intuition

```
THE FORWARD-BACKWARD RELATIONSHIP:

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  FORWARD PASS:                                                          │
│  Tells you WHAT the model computes.                                    │
│  You see: predictions, activations, attention weights.                 │
│  "What does the model think?"                                          │
│                                                                         │
│  BACKWARD PASS:                                                         │
│  Tells you WHY the model computes it.                                  │
│  You see: which inputs matter, sensitivity, blame assignment.          │
│  "Why does the model think this?"                                      │
│                                                                         │
│  THE INSIGHT:                                                           │
│  • Forward tells you the model's conclusion                            │
│  • Backward tells you the model's reasoning                            │
│                                                                         │
│  PRACTICAL:                                                             │
│  • Model makes weird prediction → check forward (what did it see?)     │
│  • Want to understand why → check backward (what influenced it?)       │
│                                                                         │
│  BOTH TOGETHER:                                                         │
│  High attention + High gradient = Model is UNCERTAIN here              │
│  High attention + Low gradient = Model is CONFIDENT here               │
│  Low attention + High gradient = Model is IGNORING something important │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### The Observability Swap

```
FORWARD ↔ BACKWARD: HARD↔EASY FOR OBSERVABILITY

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  TRANSFORM: Autodiff backward pass — O(N)                              │
│  CONSERVED: Inner products (gradient accuracy)                         │
│                                                                         │
│  ════════════════════════════════════════════════════════════════════  │
│                                                                         │
│  HARD TO OBSERVE IN FORWARD:   EASY TO OBSERVE IN BACKWARD:           │
│  ───────────────────────────   ─────────────────────────────          │
│  Feature importance            Gradient magnitude                      │
│  Input attribution             Saliency maps                           │
│  Why this prediction           ∂Loss/∂input                           │
│  Sensitivity to inputs         Local gradient norm                    │
│  Which neurons matter          Gradient flow per layer                │
│                                                                         │
│  HARD TO OBSERVE IN BACKWARD:  EASY TO OBSERVE IN FORWARD:            │
│  ───────────────────────────   ─────────────────────────────          │
│  Current prediction            Direct output                          │
│  Activation patterns           Layer activations                       │
│  Attention distribution        Attention weights                      │
│  What model represents         Embedding values                        │
│                                                                         │
│  ════════════════════════════════════════════════════════════════════  │
│                                                                         │
│  COMBINED OBSERVABILITY MATRIX:                                        │
│                                                                         │
│  ┌────────────────────┬────────────────────┬────────────────────────┐  │
│  │                    │ HIGH GRADIENT      │ LOW GRADIENT           │  │
│  ├────────────────────┼────────────────────┼────────────────────────┤  │
│  │ HIGH ATTENTION     │ Important+Malleable│ Important+Settled      │  │
│  │                    │ → Active learning  │ → Committed belief     │  │
│  ├────────────────────┼────────────────────┼────────────────────────┤  │
│  │ LOW ATTENTION      │ Ignored but matters│ Irrelevant             │  │
│  │                    │ → Bug/misalignment │ → Safely ignore        │  │
│  └────────────────────┴────────────────────┴────────────────────────┘  │
│                                                                         │
│  High attention + High gradient = PROBE HERE (model is uncertain)     │
│  High attention + Low gradient = STABLE (model is confident)          │
│  Low attention + High gradient = BUG (model ignores important input)  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### How to Use

```python
"""
FORWARD ↔ BACKWARD DUALITY
"""

def forward_backward_analysis(model, input_data, target):
    """Extract information from both directions."""
    # Forward
    input_data.requires_grad_(True)
    output = model(input_data)
    
    forward_info = {
        'prediction': output.detach(),
        'activations': extract_activations(model),
        'attention': extract_attention_weights(model, input_data)
    }
    
    # Backward
    loss = F.mse_loss(output, target)
    loss.backward()
    
    backward_info = {
        'input_gradient': input_data.grad.detach(),
        'saliency': input_data.grad.abs().detach(),
        'gradient_direction': input_data.grad.sign().detach(),
        'sensitivity': compute_layer_gradients(model)
    }
    
    # Combined insights
    combined = {
        'forward': forward_info,
        'backward': backward_info,
        'saliency_overlap_attention': compute_overlap(
            backward_info['saliency'],
            forward_info['attention']
        ),
        'gradient_entropy': compute_entropy(backward_info['saliency'].flatten())
    }
    
    return combined


def attribution_analysis(model, input_data, target_class):
    """Use backward pass for attribution."""
    input_data.requires_grad_(True)
    output = model(input_data)
    
    # Target specific output
    target_output = output[:, target_class] if output.dim() > 1 else output
    target_output.sum().backward()
    
    attribution = input_data.grad.detach()
    
    return {
        'attribution': attribution,
        'positive_attribution': (attribution > 0).float(),
        'negative_attribution': (attribution < 0).float(),
        'most_important': attribution.abs().argmax()
    }
```

### What It Reveals

```
FORWARD vs BACKWARD: Complementary Views

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  FORWARD reveals:               BACKWARD reveals:                      │
│  ───────────────               ────────────────                        │
│                                                                         │
│  • Current model beliefs        • What drives those beliefs           │
│  • Attention patterns           • Why those patterns                  │
│  • Layer activations            • Layer sensitivities                 │
│  • Prediction confidence        • Prediction fragility                │
│                                                                         │
│  COMBINING VIEWS:                                                       │
│                                                                         │
│  High attention + High gradient = Important AND malleable             │
│  High attention + Low gradient  = Important AND settled               │
│  Low attention + High gradient  = Ignored BUT should matter           │
│  Low attention + Low gradient   = Irrelevant                          │
│                                                                         │
│  USE FOR:                                                               │
│  • Understanding where model is wrong and why                         │
│  • Finding features that matter but are misused                       │
│  • Detecting when model ignores important input                       │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Sharp ↔ Soft Duality

### The Intuition

```
THE TEMPERATURE RELATIONSHIP:

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  SHARP (low temperature):                                               │
│  Model commits to ONE answer strongly.                                 │
│  Like asking "What's your final answer?"                               │
│  You see: the decision, but not the alternatives considered.           │
│                                                                         │
│  SOFT (high temperature):                                               │
│  Model shows its full uncertainty distribution.                        │
│  Like asking "What are all the possibilities you considered?"          │
│  You see: all alternatives, but not what it would actually pick.       │
│                                                                         │
│  THE INSIGHT:                                                           │
│  • Temperature doesn't change WHAT the model knows                     │
│  • Temperature only changes HOW MUCH it shows you                      │
│  • High temp = see all beliefs; Low temp = see only winner             │
│                                                                         │
│  DIAGNOSTIC POWER:                                                      │
│  Same output at τ=0.1 and τ=10? → Model is CONFIDENT (clear winner)   │
│  Different outputs?              → Model is UNCERTAIN (close race)     │
│                                                                         │
│  THE GAP between sharp and soft outputs = UNCERTAINTY MEASURE          │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### The Observability Swap

```
SHARP ↔ SOFT: HARD↔EASY FOR OBSERVABILITY

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  TRANSFORM: Temperature scaling — O(1) (just divide by τ)             │
│  CONSERVED: Score ordering (same ranking at all temperatures)          │
│                                                                         │
│  ════════════════════════════════════════════════════════════════════  │
│                                                                         │
│  HARD TO OBSERVE AT SHARP (τ→0):  EASY TO OBSERVE AT SOFT (τ→∞):     │
│  ────────────────────────────────  ─────────────────────────────────  │
│  Alternative hypotheses            Full belief distribution           │
│  Second-best options               All candidates visible             │
│  Uncertainty                       Entropy directly readable          │
│  Fragility of commitment           How close the competition          │
│                                                                         │
│  HARD TO OBSERVE AT SOFT (τ→∞):   EASY TO OBSERVE AT SHARP (τ→0):    │
│  ────────────────────────────────  ─────────────────────────────────  │
│  What model will commit to         Winner-take-all decision           │
│  Final prediction                  argmax directly                     │
│  Operational behavior              Deployment-time output             │
│                                                                         │
│  ════════════════════════════════════════════════════════════════════  │
│                                                                         │
│  TEMPERATURE SWEEP DIAGNOSTIC:                                          │
│                                                                         │
│  Run same input at τ = [0.1, 0.5, 1.0, 2.0, 10.0]                     │
│                                                                         │
│  PATTERN                         INTERPRETATION                        │
│  ───────                         ──────────────                        │
│  Same output at all τ            Clear winner, confident              │
│  Output changes at τ = 0.1       Fragile commitment, close race       │
│  Output changes at τ = 10        Multiple viable alternatives         │
│  Big divergence sharp vs soft    Rich belief structure (uncertain)    │
│  Small divergence                Sparse belief (confident)            │
│                                                                         │
│  THE GAP (output@τ=0.1 - output@τ=10) = UNCERTAINTY MEASURE           │
│                                                                         │
│  DECISION RULE:                                                         │
│  If sharp ≈ soft:  Commit (belief is stable, trust it)               │
│  If sharp ≠ soft:  Tickle more (belief is fragile, probe further)    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### How to Use

```python
"""
SHARP ↔ SOFT DUALITY
"""

def temperature_comparison(model, input_data, temperatures=[0.1, 1.0, 10.0]):
    """Compare outputs at different temperatures."""
    outputs = {}
    attention_entropies = {}
    
    for temp in temperatures:
        with torch.no_grad():
            output = model.forward(input_data, temperature=temp)
        outputs[temp] = output
        
        attention = extract_attention_weights(model, input_data)
        _, entropy = compute_entropy(list(attention.values())[0])
        attention_entropies[temp] = entropy.mean().item()
    
    # Compute divergence
    base = outputs[1.0]
    divergence = {
        temp: (output - base).pow(2).mean().sqrt().item()
        for temp, output in outputs.items()
    }
    
    return {
        'outputs': outputs,
        'entropies': attention_entropies,
        'divergence': divergence,
        'is_fragile': divergence[0.1] > 0.1,
        'uncertainty': divergence[10.0] - divergence[0.1]
    }


def sharp_soft_analysis(attention_weights, temperatures=[0.1, 1.0, 10.0]):
    """Analyze attention at different sharpness levels."""
    # Get pre-softmax scores
    scores = torch.log(attention_weights + 1e-10)  # Approximate inverse
    
    results = {}
    for temp in temperatures:
        weights = F.softmax(scores / temp, dim=-1)
        
        results[temp] = {
            'entropy': compute_entropy(weights)[1].mean().item(),
            'max_weight': weights.max(dim=-1).values.mean().item(),
            'sparsity': (weights < 0.01).float().mean().item()
        }
    
    return results
```

### What It Reveals

```
TEMPERATURE SWEEP → BELIEF STRUCTURE

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  PATTERN                      INTERPRETATION                           │
│  ───────                      ──────────────                           │
│                                                                         │
│  Same output at all temps     Clear winner, confident                 │
│  Different output at τ=0.1    Fragile commitment                      │
│  Different output at τ=10     Multiple alternatives                   │
│  Big divergence               Rich belief structure                   │
│  Small divergence             Sparse belief structure                 │
│                                                                         │
│  OPERATIONAL USE:                                                       │
│                                                                         │
│  If sharp ≈ soft:             Commit (belief is stable)               │
│  If sharp ≠ soft:             Tickle more (belief is fragile)        │
│                                                                         │
│  The GAP (sharp - soft) = measure of uncertainty                      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Local ↔ Global Duality

### The Intuition

```
THE LOCAL-GLOBAL RELATIONSHIP:

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  LOCAL:                                                                 │
│  Information about specific positions, individual tokens, exact spots. │
│  Like examining one tree in a forest.                                  │
│                                                                         │
│  GLOBAL:                                                                │
│  Information about the whole, overall patterns, aggregate properties.  │
│  Like seeing the forest from a plane.                                  │
│                                                                         │
│  THE TRADE-OFF:                                                         │
│  • Local details are EASY to observe locally, HARD globally           │
│  • Global patterns are EASY to observe globally, HARD locally         │
│                                                                         │
│  EXAMPLE:                                                               │
│  • "Is there a red pixel at position (45, 67)?" → Ask LOCAL            │
│  • "Is the image mostly blue?" → Ask GLOBAL                            │
│  • "Is there ANY anomaly?" → Need GLOBAL (scan everything)             │
│  • "What is the anomaly?" → Need LOCAL (examine it)                    │
│                                                                         │
│  EFFICIENCY INSIGHT:                                                    │
│  Global pooling loses local information but is cheap.                  │
│  Keeping everything local is expensive but complete.                   │
│  AKIRA uses BANDS: each band pools at different scales.                │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### The Observability Swap

```
LOCAL ↔ GLOBAL: HARD↔EASY FOR OBSERVABILITY

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  TRANSFORM: Aggregation / Attention — varies                           │
│  CONSERVED: Total information (local + global = complete picture)     │
│                                                                         │
│  ════════════════════════════════════════════════════════════════════  │
│                                                                         │
│  HARD TO OBSERVE LOCALLY:       EASY TO OBSERVE GLOBALLY:             │
│  ─────────────────────────      ───────────────────────────           │
│  Long-range dependencies        Attention to distant positions        │
│  Object identity                Semantic clustering                    │
│  Context effects                Cross-position correlations           │
│  Compositional structure        Hierarchical patterns                 │
│                                                                         │
│  HARD TO OBSERVE GLOBALLY:      EASY TO OBSERVE LOCALLY:              │
│  ─────────────────────────      ───────────────────────────           │
│  Exact positions                Direct coordinates                    │
│  Local anomalies                Point-wise deviation                  │
│  Edge details                   Gradient at position                  │
│  Texture patterns               Neighborhood statistics               │
│                                                                         │
│  ════════════════════════════════════════════════════════════════════  │
│                                                                         │
│  WORMHOLE OBSERVABILITY:                                               │
│                                                                         │
│  Wormhole attention weights reveal WHICH distant regions connect.     │
│                                                                         │
│  To observe wormhole behavior:                                         │
│  1. Extract wormhole attention matrix A[source_band][target_band]    │
│  2. Threshold: A > 0.1 shows active connections                       │
│  3. Visualize: which Band 0 positions query which Band 6 positions   │
│                                                                         │
│  HIGH wormhole activity: Model needs cross-scale integration          │
│  LOW wormhole activity: Per-band processing suffices                  │
│                                                                         │
│  If wormhole fires between same positions: colocated WHAT/WHERE      │
│  If wormhole fires between distant positions: object tracking        │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### How to Use

```python
"""
LOCAL ↔ GLOBAL DUALITY
"""

def local_global_comparison(model, input_data):
    """Compare local and global processing results."""
    # Local: per-band attention (within spatial neighborhoods)
    local_outputs = {}
    for band_idx in range(7):
        local_outputs[band_idx] = model.per_band_attention[band_idx](
            model.spectral_decomposer(input_data)[band_idx]
        )
    
    # Global: wormhole attention (across space and time)
    global_output = model.wormhole_attention(input_data)
    
    # Compare
    local_mean = torch.stack(list(local_outputs.values())).mean(dim=0)
    
    comparison = {
        'local_outputs': local_outputs,
        'global_output': global_output,
        'local_global_correlation': torch.corrcoef(
            torch.stack([local_mean.flatten(), global_output.flatten()])
        )[0, 1].item(),
        'global_unique': (global_output - local_mean).abs().mean().item()
    }
    
    return comparison


def receptive_field_analysis(model, input_data, position):
    """Analyze effective receptive field at a position."""
    # Perturb each position, measure effect on target position
    receptive_field = torch.zeros_like(input_data[0, 0])
    
    for i in range(input_data.size(-2)):
        for j in range(input_data.size(-1)):
            perturbed = input_data.clone()
            perturbed[:, :, i, j] += 0.1
            
            with torch.no_grad():
                base_output = model(input_data)
                pert_output = model(perturbed)
            
            effect = (pert_output[:, :, position[0], position[1]] - 
                     base_output[:, :, position[0], position[1]]).abs().mean()
            receptive_field[i, j] = effect.item()
    
    return {
        'receptive_field': receptive_field,
        'effective_size': (receptive_field > 0.01).sum().item(),
        'is_local': (receptive_field > 0.01).float().mean().item() < 0.1
    }
```

### What It Reveals

```
LOCAL vs GLOBAL: Different Information

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  LOCAL processing finds:       GLOBAL processing finds:               │
│  ──────────────────────       ────────────────────────                │
│                                                                         │
│  • Edges and textures          • Object identity                      │
│  • Local patterns              • Semantic relationships               │
│  • Spatial gradients           • Long-range dependencies              │
│  • Boundary detection          • Context integration                  │
│                                                                         │
│  THE COMBINATION:                                                       │
│                                                                         │
│  Local gives: "There's an edge here"                                  │
│  Global gives: "This edge is part of a car"                          │
│                                                                         │
│  Local gives: "This pixel matches texture"                            │
│  Global gives: "This texture matches object from 10 frames ago"      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Explicit ↔ Implicit Duality

### The Intuition

```
THE LEARNING TRANSITION:

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  Before training: All in data (explicit)                               │
│  During training: Transfer to weights (explicit → implicit)            │
│  After training:  Structure in weights, details in data                │
│                                                                         │
│  GROKKING is the sudden completion of this transfer.                   │
│  COLLAPSE is the point where implicit takes over.                      │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  EXPLICIT:                                                              │
│  Information is in the DATA you can see.                               │
│  Like having a phone book — you look up each number.                   │
│                                                                         │
│  IMPLICIT:                                                              │
│  Information is in the WEIGHTS you can't easily see.                   │
│  Like knowing the pattern — you can generate the answer.               │
│                                                                         │
│  THE TRADE-OFF:                                                         │
│  • Explicit: easy to observe (it's the data!)                          │
│  • Implicit: efficient (compressed into weights)                       │
│                                                                         │
│  OBSERVABILITY CHALLENGE:                                               │
│  You can SEE training data.                                            │
│  You can PROBE weights (but interpreting them is hard).                │
│  Grokking moment = watch test loss suddenly drop after plateau.        │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### The Observability Swap

```
EXPLICIT ↔ IMPLICIT: HARD↔EASY FOR OBSERVABILITY

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  TRANSFORM: Training (data → weights)                                  │
│  CONSERVED: Predictive information (same input-output mapping)        │
│                                                                         │
│  ════════════════════════════════════════════════════════════════════  │
│                                                                         │
│  HARD TO OBSERVE IMPLICITLY:    EASY TO OBSERVE EXPLICITLY:           │
│  ───────────────────────────    ─────────────────────────────         │
│  What patterns are learned      Input-output examples                  │
│  Generalization rules           Training data directly                │
│  Compressed knowledge           Raw data statistics                   │
│  Category structure             Individual instances                   │
│                                                                         │
│  HARD TO OBSERVE EXPLICITLY:    EASY TO OBSERVE IMPLICITLY:           │
│  ───────────────────────────    ─────────────────────────────         │
│  Why model generalizes          Weight structure / sparsity           │
│  Internal representations       Activation patterns                   │
│  Learned features               Layer outputs                          │
│  Compression achieved           Weight count, norms                   │
│                                                                         │
│  ════════════════════════════════════════════════════════════════════  │
│                                                                         │
│  GROKKING DETECTION:                                                    │
│                                                                         │
│  Track: train_loss, test_loss, weight_norm over epochs                │
│                                                                         │
│  PHASE              SIGNATURE                INTERPRETATION            │
│  ─────              ─────────                ──────────────            │
│  Explicit           train↓, test flat        Memorizing (explicit)    │
│  (early)            weights growing          Storing instances        │
│                                                                         │
│  Transition         both↓, weights shrink    Compressing              │
│  (middle)           sudden test drop         Finding patterns         │
│                                                                         │
│  Implicit           both low, weights stable Generalized (implicit)   │
│  (late)             robust to perturbation   Knowledge in weights     │
│                                                                         │
│  GROKKING = sudden test loss drop after long plateau                  │
│  Observable via: test_loss time series, weight compression ratio      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### How to Use

```python
"""
EXPLICIT ↔ IMPLICIT DUALITY
"""

def measure_explicit_implicit_split(model, data_loader):
    """Measure how much information is in data vs weights."""
    
    # Information in weights: model can generalize
    generalization_score = 0
    
    # Information in data: model memorizes
    memorization_score = 0
    
    for batch in data_loader:
        input_data, target = batch
        
        # Test generalization: perturb input slightly
        perturbed = input_data + torch.randn_like(input_data) * 0.1
        
        with torch.no_grad():
            output_original = model(input_data)
            output_perturbed = model(perturbed)
        
        # Generalization: output stable under perturbation
        generalization_score += (
            1 - (output_original - output_perturbed).abs().mean().item()
        )
        
        # Memorization: exact recall of training data
        memorization_score += (
            1 - (output_original - target).abs().mean().item()
        )
    
    n = len(data_loader)
    
    return {
        'generalization': generalization_score / n,
        'memorization': memorization_score / n,
        'ratio': generalization_score / (memorization_score + 1e-10),
        'phase': 'implicit' if generalization_score > memorization_score else 'explicit'
    }


def track_explicit_implicit_over_training(model, train_loader, test_loader, epochs):
    """Track the explicit→implicit transition during training."""
    history = []
    
    for epoch in range(epochs):
        # Train
        train_model(model, train_loader)
        
        # Measure
        train_metrics = measure_explicit_implicit_split(model, train_loader)
        test_metrics = measure_explicit_implicit_split(model, test_loader)
        
        history.append({
            'epoch': epoch,
            'train': train_metrics,
            'test': test_metrics,
            'generalization_gap': test_metrics['generalization'] - train_metrics['generalization']
        })
    
    return history
```

### What It Reveals

```
EXPLICIT vs IMPLICIT: Learning Dynamics

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  PHASE          CHARACTERISTICS           WHAT MODEL DOES             │
│  ─────          ───────────────           ───────────────             │
│                                                                         │
│  Explicit       Train loss ↓, test flat   Memorizing data            │
│  (early)        High-freq details stored  Looking up training set    │
│                 Weights random-ish        No generalization           │
│                                                                         │
│  Transition     Both losses ↓             Extracting patterns        │
│  (middle)       Compression happening     Building structure          │
│                 Weights organizing        Beginning to generalize     │
│                                                                         │
│  Implicit       Test loss ↓ (catches up)  Generalized!               │
│  (late)         Low-freq structure        Understands category        │
│                 Weights stable            Predicts new examples       │
│                                                                         │
│  GROKKING = Sudden transition from explicit to implicit              │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 7. Energy ↔ Geometry Duality

### The Intuition

```
THE ENERGY-GEOMETRY RELATIONSHIP:

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  ENERGY:                                                                │
│  HOW MUCH is happening — magnitudes, norms, intensities.               │
│  Like volume in music. Loud = something important.                     │
│                                                                         │
│  GEOMETRY:                                                              │
│  WHAT STRUCTURE exists — patterns, relationships, distances.           │
│  Like harmony in music. Structure = meaningful arrangement.            │
│                                                                         │
│  THE ANALOGY:                                                           │
│  • Energy alone → loud noise (urgent but meaningless)                  │
│  • Geometry alone → quiet whisper (meaningful but weak)                │
│  • Both together → clear important signal                              │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  SYSTEM 1 vs SYSTEM 2:                                                  │
│                                                                         │
│  High energy, low geometry = REFLEX (fast, reactive)                   │
│  Low energy, high geometry = REASONING (slow, deliberate)              │
│                                                                         │
│  OBSERVE: Energy/Geometry ratio tells you which mode is active.        │
│                                                                         │
│  Emergency   → High E, Low G → Act now, think later                    │
│  Normal ops  → Low E, High G → Reason carefully                        │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### The Observability Swap

```
ENERGY ↔ GEOMETRY: HARD↔EASY FOR OBSERVABILITY

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  TRANSFORM: Magnitude vs structure decomposition                       │
│  CONSERVED: Total representational capacity                            │
│                                                                         │
│  ════════════════════════════════════════════════════════════════════  │
│                                                                         │
│  HARD TO OBSERVE VIA ENERGY:    EASY TO OBSERVE VIA GEOMETRY:         │
│  ───────────────────────────    ─────────────────────────────         │
│  Semantic relationships         Attention patterns                     │
│  Conceptual similarity          Embedding distances                    │
│  Knowledge structure            Manifold geometry                      │
│  Category boundaries            Clustering structure                   │
│                                                                         │
│  HARD TO OBSERVE VIA GEOMETRY:  EASY TO OBSERVE VIA ENERGY:           │
│  ───────────────────────────    ─────────────────────────────         │
│  Signal intensity               Activation magnitudes                  │
│  Urgency / salience             Gradient norms                        │
│  Anomaly detection              Deviation from mean energy            │
│  Threshold triggers             Energy > threshold?                   │
│                                                                         │
│  ════════════════════════════════════════════════════════════════════  │
│                                                                         │
│  OBSERVABILITY QUADRANT:                                               │
│                                                                         │
│  ┌────────────────────┬─────────────────────┬─────────────────────────┐│
│  │                    │ HIGH GEOMETRY       │ LOW GEOMETRY            ││
│  │                    │ (rich structure)    │ (sparse structure)      ││
│  ├────────────────────┼─────────────────────┼─────────────────────────┤│
│  │ HIGH ENERGY        │ Full engagement     │ Emergency / Anomaly     ││
│  │ (strong signal)    │ Important+Understood│ Reflex trigger          ││
│  │                    │ → Trust it          │ → Investigate           ││
│  ├────────────────────┼─────────────────────┼─────────────────────────┤│
│  │ LOW ENERGY         │ Background process  │ Noise / Irrelevant      ││
│  │ (weak signal)      │ Subtle deliberation │ Nothing happening       ││
│  │                    │ → Monitor           │ → Ignore                ││
│  └────────────────────┴─────────────────────┴─────────────────────────┘│
│                                                                         │
│  REACTIVE vs DELIBERATIVE:                                              │
│                                                                         │
│  Energy-dominant response = REACTIVE (System 1, reflex)               │
│  Geometry-dominant response = DELIBERATIVE (System 2, reasoning)      │
│                                                                         │
│  Observe E/G ratio to diagnose processing mode.                        │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### How to Use

```python
"""
ENERGY ↔ GEOMETRY DUALITY
"""

def energy_geometry_analysis(model, input_data):
    """Separate energy and geometry contributions."""
    
    # Energy: magnitude-based measures
    energy_info = {
        'input_energy': input_data.pow(2).sum().item(),
        'activation_energies': {},
        'gradient_magnitudes': {}
    }
    
    # Geometry: structure-based measures
    geometry_info = {
        'attention_patterns': {},
        'similarity_structure': {},
        'manifold_distances': {}
    }
    
    # Forward pass
    with torch.no_grad():
        output = model(input_data)
    
    # Extract energy
    for name, activation in extract_activations(model, input_data).items():
        energy_info['activation_energies'][name] = activation.pow(2).mean().item()
    
    # Extract geometry
    for name, weights in extract_attention_weights(model, input_data).items():
        geometry_info['attention_patterns'][name] = {
            'entropy': compute_entropy(weights)[1].mean().item(),
            'sparsity': (weights < 0.01).float().mean().item()
        }
    
    # Wormhole geometry
    similarity = extract_full_similarity(model.wormhole_attention, input_data)
    geometry_info['similarity_structure'] = {
        'clustering': compute_similarity_clustering(similarity),
        'dimensionality': estimate_intrinsic_dim(similarity)
    }
    
    return {
        'energy': energy_info,
        'geometry': geometry_info,
        'energy_geometry_ratio': sum(energy_info['activation_energies'].values()) /
                                 (sum(g['entropy'] for g in geometry_info['attention_patterns'].values()) + 1e-10)
    }
```

### What It Reveals

```
ENERGY vs GEOMETRY: Decision Types

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  HIGH ENERGY + LOW GEOMETRY:                                           │
│  → Reactive mode (reflexes)                                           │
│  → Large magnitudes, little structure                                 │
│  → Emergency response, immediate action                               │
│                                                                         │
│  LOW ENERGY + HIGH GEOMETRY:                                           │
│  → Deliberative mode (knowledge)                                      │
│  → Rich structure, moderate magnitudes                                │
│  → Careful reasoning, manifold queries                                │
│                                                                         │
│  HIGH ENERGY + HIGH GEOMETRY:                                          │
│  → Full engagement                                                     │
│  → Strong signals with clear structure                                │
│  → Important and understood                                           │
│                                                                         │
│  LOW ENERGY + LOW GEOMETRY:                                            │
│  → Noise or irrelevance                                               │
│  → Weak signals, no structure                                         │
│  → Ignore                                                              │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Summary: The Observability Toolkit

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│        D U A L I T Y   M E T H O D S   F O R   O B S E R V A B I L I T Y│
│                                                                         │
│  ═══════════════════════════════════════════════════════════════════   │
│                                                                         │
│  DUALITY               COST          HARD→EASY SWAP                    │
│  ───────               ────          ─────────────                      │
│                                                                         │
│  Spatial ↔ Frequency   O(N log N)    Global patterns → single coeffs  │
│  Magnitude ↔ Phase     O(N)          Position → phase gradients       │
│  Forward ↔ Backward    O(N)          Attribution → one backward pass  │
│  Sharp ↔ Soft          O(1)          Uncertainty → temperature sweep  │
│  Local ↔ Global        varies        Dependencies → attention weights │
│  Explicit ↔ Implicit   training      Generalization → train/test gap  │
│  Energy ↔ Geometry     analysis      Processing mode → E/G ratio      │
│                                                                         │
│  ═══════════════════════════════════════════════════════════════════   │
│                                                                         │
│  THE OBSERVABILITY PRINCIPLE:                                           │
│                                                                         │
│  1. What you want to observe is HARD in the current domain            │
│  2. Find the duality where it becomes EASY                            │
│  3. Transform, observe, transform back                                │
│  4. Verify via the conserved quantity (if it fails → bug)            │
│                                                                         │
│  ═══════════════════════════════════════════════════════════════════   │
│                                                                         │
│  QUICK REFERENCE: WHAT TO OBSERVE WHERE                                │
│                                                                         │
│  TO OBSERVE...                  USE THIS DOMAIN...                     │
│  ─────────────                  ─────────────────                       │
│  Scale of patterns              Frequency (FFT)                        │
│  Position of features           Spatial (direct) or Phase             │
│  Feature importance             Backward (gradients)                   │
│  Model uncertainty              Soft (temperature sweep)              │
│  Long-range dependencies        Global (attention weights)            │
│  Generalization vs memorization Explicit/Implicit (train vs test)    │
│  Reactive vs deliberative       Energy/Geometry (E/G ratio)          │
│                                                                         │
│  ═══════════════════════════════════════════════════════════════════   │
│                                                                         │
│  CONSERVATION CHECKS (validation):                                      │
│                                                                         │
│  Duality                Conservation Law                               │
│  ───────                ────────────────                               │
│  Spatial↔Frequency      Σ|x|² = Σ|X|² (Parseval)                      │
│  Forward↔Backward       Gradient accuracy (autodiff correctness)      │
│  Sharp↔Soft             Score ordering preserved                       │
│                                                                         │
│  If conservation fails, the observation is invalid (bug in transform).│
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Related Documents

| Document | Connection |
|----------|------------|
| `foundations/DUALITY_AND_EFFICIENCY.md` | Companion doc: dualities for ARCHITECTURE (FFT, Forward/Backward, Viterbi) |
| `foundations/TERMINOLOGY.md` | Formal definitions of collapse, tension, AQ |
| `pandora/PANDORA.md` | Action as transformation between dual forms |

---

*Oscar Goldman — Shogu Research Group @ Datamutant.ai*

*"What's hard to observe directly is easy to observe dually. The transform is cheap. Use both domains. See everything."*

