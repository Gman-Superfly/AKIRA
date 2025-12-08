# EXPERIMENTAL TECHNIQUES

## A Practical Guide to Observing and Validating AKIRA Psyche

**Oscar Goldman — Shogu Research Group @ Datamutant.ai**

---

## Table of Contents

1. [Overview](#1-overview)
2. [Free Information Extraction](#2-free-information-extraction)
3. [Tickling Techniques](#3-tickling-techniques)
4. [Collapse Detection](#4-collapse-detection)
5. [Spectral Analysis](#5-spectral-analysis)
6. [Embedding and Visualization](#6-embedding-and-visualization)
7. [Statistical Validation](#7-statistical-validation)
8. [Implementation Patterns](#8-implementation-patterns)

---

## 1. Overview

### 1.1 Philosophy

```
EXPERIMENTAL PHILOSOPHY

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  PRINCIPLE 1: EXTRACT BEFORE COMPUTING                                 │
│  ──────────────────────────────────────                                 │
│  Before running new computations, ask:                                 │
│  "Is this information already available somewhere?"                   │
│                                                                         │
│  PRINCIPLE 2: CHEAP BEFORE EXPENSIVE                                   │
│  ────────────────────────────────────                                   │
│  Always try the cheapest technique first:                             │
│  Entropy (free) → Temperature (2-3 FP) → Full search (N FP)          │
│                                                                         │
│  PRINCIPLE 3: FALSIFY BEFORE VALIDATE                                  │
│  ─────────────────────────────────────                                  │
│  Design experiments to DISPROVE the theory first.                     │
│  If they fail to disprove, that's stronger than confirmation.        │
│                                                                         │
│  PRINCIPLE 4: DUAL VIEWS BEFORE SINGLE VIEW                            │
│  ────────────────────────────────────────────                           │
│  Look at data in multiple representations:                            │
│  Spatial AND frequency, magnitude AND phase, etc.                    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Technique Categories

```
TECHNIQUE CATEGORIES BY COST

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  FREE (no additional computation):                                     │
│  • Attention weight reading                                           │
│  • Entropy computation                                                 │
│  • Pre-softmax score extraction                                       │
│  • Similarity matrix reading                                          │
│  • Near-threshold counting                                            │
│                                                                         │
│  CHEAP (1-5 forward passes):                                           │
│  • Temperature sweeping                                                │
│  • Threshold sweeping                                                  │
│  • Sparse probing                                                      │
│  • First-token analysis                                                │
│                                                                         │
│  MODERATE (10-100 forward passes):                                     │
│  • Ensemble uncertainty                                                │
│  • Jacobian estimation                                                 │
│  • Random perturbation analysis                                        │
│                                                                         │
│  EXPENSIVE (100+ forward passes):                                      │
│  • Full prompt search                                                  │
│  • Exhaustive manifold mapping                                        │
│  • Full trajectory analysis                                            │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Free Information Extraction

### 2.1 Attention Weight Reading

```python
"""
TECHNIQUE: ATTENTION WEIGHT READING
COST: FREE (already computed)
REVEALS: Where model attends, uncertainty structure
"""

def extract_attention_weights(model, input_data, layer_idx=None):
    """
    Extract attention weights from forward pass.
    
    Returns:
        weights: Attention weight tensors per layer
        metadata: Layer info, shapes, etc.
    """
    attention_weights = {}
    
    # Register hooks to capture attention
    hooks = []
    
    def capture_attention(name):
        def hook(module, input, output):
            # Most attention modules return (output, weights) or have .attn_weights
            if isinstance(output, tuple) and len(output) >= 2:
                attention_weights[name] = output[1].detach()
            elif hasattr(module, 'attn_weights'):
                attention_weights[name] = module.attn_weights.detach()
        return hook
    
    # Hook relevant layers
    for name, module in model.named_modules():
        if 'attention' in name.lower() or 'attn' in name.lower():
            if layer_idx is None or str(layer_idx) in name:
                hooks.append(module.register_forward_hook(capture_attention(name)))
    
    # Forward pass
    with torch.no_grad():
        output = model(input_data)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    return attention_weights


def analyze_attention_patterns(weights):
    """
    Analyze patterns in attention weights.
    """
    analysis = {}
    
    for name, w in weights.items():
        # w shape: [batch, heads, seq, seq] or similar
        analysis[name] = {
            'entropy': compute_entropy(w),
            'sparsity': (w < 0.01).float().mean().item(),
            'max_attention': w.max(dim=-1).values.mean().item(),
            'concentration': (w ** 2).sum(dim=-1).mean().item(),  # Gini-like
        }
    
    return analysis
```

### 2.2 Entropy Computation

```python
"""
TECHNIQUE: ENTROPY COMPUTATION
COST: FREE (simple computation on existing weights)
REVEALS: Uncertainty, leader count, edge of error
"""

def compute_entropy(attention_weights, dim=-1, eps=1e-10):
    """
    Compute entropy of attention distribution.
    
    Returns:
        entropy: Per-position entropy values
        normalized_entropy: Entropy / max_entropy (0-1 range)
    """
    # Clamp for numerical stability
    p = attention_weights.clamp(min=eps)
    
    # Entropy: -sum(p * log(p))
    entropy = -(p * p.log()).sum(dim=dim)
    
    # Normalize by max entropy (uniform distribution)
    n = attention_weights.size(dim)
    max_entropy = torch.log(torch.tensor(n, dtype=p.dtype, device=p.device))
    normalized_entropy = entropy / max_entropy
    
    return entropy, normalized_entropy


def entropy_landscape(attention_weights):
    """
    Create a spatial map of attention entropy.
    """
    _, normalized = compute_entropy(attention_weights)
    
    # Average over heads if present
    if normalized.dim() == 4:  # [batch, heads, seq, seq]
        normalized = normalized.mean(dim=1)
    
    return normalized


def identify_high_entropy_regions(entropy_map, threshold=0.7):
    """
    Find positions where entropy is above threshold (uncertain regions).
    """
    high_entropy_mask = entropy_map > threshold
    positions = high_entropy_mask.nonzero()
    
    return {
        'mask': high_entropy_mask,
        'positions': positions,
        'count': high_entropy_mask.sum().item(),
        'fraction': high_entropy_mask.float().mean().item()
    }
```

### 2.3 Pre-Softmax Score Extraction

```python
"""
TECHNIQUE: PRE-SOFTMAX SCORE EXTRACTION
COST: FREE (requires minor hook modification)
REVEALS: True competition between hypotheses
"""

def extract_presoftmax_scores(model, input_data):
    """
    Extract attention scores BEFORE softmax is applied.
    
    These reveal how close the competition is between hypotheses.
    """
    presoftmax_scores = {}
    
    def capture_presoftmax(name):
        def hook(module, input, output):
            # Need to intercept the scores before softmax
            # This depends on the attention implementation
            if hasattr(module, 'last_scores'):
                presoftmax_scores[name] = module.last_scores.detach()
        return hook
    
    # Alternative: Modify attention to store pre-softmax
    # This is architecture-specific
    
    return presoftmax_scores


def analyze_competition(presoftmax_scores):
    """
    Analyze how close the competition is between top hypotheses.
    """
    analysis = {}
    
    for name, scores in presoftmax_scores.items():
        # Get top-2 scores
        top2_scores, top2_indices = scores.topk(2, dim=-1)
        
        # Competition ratio: how close is #2 to #1?
        winner = top2_scores[..., 0]
        runner_up = top2_scores[..., 1]
        competition_ratio = runner_up / (winner.abs() + 1e-10)
        
        # Gap: absolute difference between #1 and #2
        gap = winner - runner_up
        
        analysis[name] = {
            'competition_ratio': competition_ratio.mean().item(),
            'gap_mean': gap.mean().item(),
            'gap_std': gap.std().item(),
            'tight_races': (competition_ratio > 0.8).float().mean().item(),
            'clear_winners': (competition_ratio < 0.3).float().mean().item()
        }
    
    return analysis
```

### 2.4 Similarity Matrix Extraction

```python
"""
TECHNIQUE: SIMILARITY MATRIX EXTRACTION
COST: FREE (for wormhole attention, already computed)
REVEALS: Full landscape of potential connections
"""

def extract_similarity_matrix(wormhole_module, query, keys):
    """
    Extract full similarity matrix before top-k selection.
    """
    # Normalize (already done in wormhole)
    query_norm = F.normalize(query.flatten(2), p=2, dim=-1)
    key_norm = F.normalize(keys.flatten(2), p=2, dim=-1)
    
    # Full similarity matrix
    similarity = torch.matmul(query_norm, key_norm.transpose(-1, -2))
    
    return similarity


def analyze_similarity_landscape(similarity, threshold=0.92):
    """
    Analyze the full similarity landscape.
    """
    analysis = {
        # Distribution statistics
        'mean': similarity.mean().item(),
        'std': similarity.std().item(),
        'max': similarity.max().item(),
        'min': similarity.min().item(),
        
        # Threshold analysis
        'above_threshold': (similarity > threshold).float().mean().item(),
        'near_threshold': ((similarity > threshold - 0.05) & 
                          (similarity < threshold)).float().mean().item(),
        
        # Histogram data for visualization
        'histogram': torch.histogram(similarity.flatten(), bins=50),
        
        # Peak analysis
        'num_peaks': count_similarity_peaks(similarity),
    }
    
    return analysis


def count_similarity_peaks(similarity, threshold=0.85):
    """
    Count distinct peaks (clusters) in similarity distribution.
    """
    # Simple approach: count connected components above threshold
    above = (similarity > threshold).float()
    # More sophisticated: use clustering
    return above.sum().item()  # Simplified
```

---

## 3. Tickling Techniques

### 3.1 Temperature Probing

```python
"""
TECHNIQUE: TEMPERATURE PROBING
COST: 2-3 forward passes
REVEALS: Robustness of prediction, number of competing hypotheses
"""

def temperature_probe(model, input_data, temperatures=[0.1, 1.0, 10.0]):
    """
    Run inference at multiple softmax temperatures.
    
    If outputs diverge, prediction is fragile (multiple leaders).
    If outputs stable, prediction is robust (clear winner).
    """
    outputs = {}
    entropies = {}
    
    for temp in temperatures:
        with torch.no_grad():
            # Modify temperature in attention
            output = model.forward(input_data, temperature=temp)
        outputs[temp] = output
        
        # Also track entropy at each temperature
        attention_weights = extract_attention_weights(model, input_data)
        for name, w in attention_weights.items():
            if name not in entropies:
                entropies[name] = {}
            _, norm_ent = compute_entropy(w)
            entropies[name][temp] = norm_ent.mean().item()
    
    # Compute divergence between outputs
    base_output = outputs[1.0]
    divergence = {}
    for temp, out in outputs.items():
        divergence[temp] = {
            'l2': (out - base_output).pow(2).mean().sqrt().item(),
            'linf': (out - base_output).abs().max().item(),
            'cosine': F.cosine_similarity(
                out.flatten(), base_output.flatten(), dim=0
            ).item()
        }
    
    return {
        'outputs': outputs,
        'entropies': entropies,
        'divergence': divergence,
        'is_fragile': divergence[0.1]['l2'] > 0.1 or divergence[10.0]['l2'] > 0.1
    }
```

### 3.2 Threshold Sweeping

```python
"""
TECHNIQUE: THRESHOLD SWEEPING
COST: 5-10 forward passes
REVEALS: Near-threshold connections (almost-leaders)
"""

def threshold_sweep(model, input_data, 
                   thresholds=[0.95, 0.92, 0.90, 0.85, 0.80, 0.75]):
    """
    Sweep wormhole threshold to see what's just below activation.
    """
    results = {}
    
    for thresh in thresholds:
        with torch.no_grad():
            # Run with modified threshold
            output, stats = model.forward(
                input_data, 
                wormhole_threshold=thresh,
                return_stats=True
            )
        
        results[thresh] = {
            'output': output,
            'num_connections': stats.get('num_connections', 0),
            'mean_similarity': stats.get('mean_similarity', 0),
            'max_similarity': stats.get('max_similarity', 0),
        }
    
    # Analyze where connections "turn on"
    connection_curve = {t: r['num_connections'] for t, r in results.items()}
    
    # Find steepest rise (where leaders are hiding)
    thresholds_sorted = sorted(thresholds, reverse=True)
    max_gradient = 0
    leader_zone_threshold = None
    
    for i in range(len(thresholds_sorted) - 1):
        t_high, t_low = thresholds_sorted[i], thresholds_sorted[i+1]
        gradient = (results[t_low]['num_connections'] - 
                   results[t_high]['num_connections']) / (t_high - t_low)
        if gradient > max_gradient:
            max_gradient = gradient
            leader_zone_threshold = (t_low + t_high) / 2
    
    return {
        'results': results,
        'connection_curve': connection_curve,
        'leader_zone_threshold': leader_zone_threshold,
        'max_connection_gradient': max_gradient
    }
```

### 3.3 Sparse Probing

```python
"""
TECHNIQUE: SPARSE PROBING
COST: 2 forward passes (targeted)
REVEALS: Sensitivity at specific locations (leader zones)
"""

def sparse_probe(model, input_data, probe_positions, epsilon=0.01):
    """
    Perturb only at specified positions, measure response.
    
    Use this AFTER identifying high-entropy zones.
    """
    # Create sparse perturbation
    perturbation = torch.zeros_like(input_data)
    noise = torch.randn(len(probe_positions), input_data.shape[-1]) * epsilon
    
    for i, pos in enumerate(probe_positions):
        perturbation[pos] = noise[i]
    
    # Base output
    with torch.no_grad():
        output_base = model(input_data)
        output_probe = model(input_data + perturbation)
    
    # Measure sensitivity
    delta_output = output_probe - output_base
    sensitivity = delta_output.abs()
    
    return {
        'probe_positions': probe_positions,
        'sensitivity_map': sensitivity,
        'mean_sensitivity': sensitivity.mean().item(),
        'max_sensitivity': sensitivity.max().item(),
        'high_sensitivity_positions': (sensitivity > sensitivity.mean() * 2).nonzero()
    }


def entropy_guided_probing(model, input_data, n_probes=10, epsilon=0.01):
    """
    Automatically identify high-entropy zones and probe them.
    """
    # Step 1: Get entropy map (FREE)
    attention_weights = extract_attention_weights(model, input_data)
    entropy_map = entropy_landscape(list(attention_weights.values())[0])
    
    # Step 2: Find high-entropy positions
    high_entropy = identify_high_entropy_regions(entropy_map, threshold=0.7)
    positions = high_entropy['positions']
    
    # Step 3: Sample positions to probe
    if len(positions) > n_probes:
        indices = torch.randperm(len(positions))[:n_probes]
        probe_positions = positions[indices]
    else:
        probe_positions = positions
    
    # Step 4: Probe
    return sparse_probe(model, input_data, probe_positions, epsilon)
```

### 3.4 PSON (Precision-Scaled Orthogonal Noise)

```python
"""
TECHNIQUE: PSON EXPLORATION
COST: 1-5 forward passes
REVEALS: Manifold structure without disrupting descent
"""

def compute_orthogonal_noise(gradient, noise_scale=0.1):
    """
    Generate noise orthogonal to the gradient.
    """
    # Random noise
    noise = torch.randn_like(gradient)
    
    # Project out gradient component
    grad_norm = gradient / (gradient.norm() + 1e-10)
    parallel_component = (noise * grad_norm).sum() * grad_norm
    orthogonal_noise = noise - parallel_component
    
    # Scale
    orthogonal_noise = orthogonal_noise * noise_scale
    
    return orthogonal_noise


def pson_exploration(model, input_data, n_directions=5, noise_scale=0.1):
    """
    Explore manifold structure using PSON.
    """
    # Get gradient direction
    input_data_grad = input_data.clone().requires_grad_(True)
    output = model(input_data_grad)
    loss = output.mean()  # Simple loss for gradient direction
    loss.backward()
    gradient = input_data_grad.grad
    
    # Explore multiple orthogonal directions
    explorations = []
    
    for i in range(n_directions):
        orth_noise = compute_orthogonal_noise(gradient, noise_scale)
        
        with torch.no_grad():
            output_base = model(input_data)
            output_perturbed = model(input_data + orth_noise)
        
        explorations.append({
            'direction': orth_noise,
            'output_change': (output_perturbed - output_base).abs().mean().item(),
            'output_base': output_base,
            'output_perturbed': output_perturbed
        })
    
    return {
        'gradient_direction': gradient,
        'explorations': explorations,
        'mean_sensitivity': np.mean([e['output_change'] for e in explorations]),
        'max_sensitivity': max([e['output_change'] for e in explorations])
    }
```

---

## 4. Collapse Detection

### 4.1 Entropy Drop Detection

```python
"""
TECHNIQUE: ENTROPY DROP DETECTION
COST: FREE (computed from existing entropy tracking)
REVEALS: Collapse events, transition from uncertainty to certainty
"""

class CollapseDetector:
    def __init__(self, window_size=10, drop_threshold=0.3, min_start_entropy=0.6):
        self.window_size = window_size
        self.drop_threshold = drop_threshold
        self.min_start_entropy = min_start_entropy
        self.entropy_history = []
        self.collapses = []
        
    def update(self, entropy_value, step):
        """
        Update with new entropy value, detect collapse.
        """
        self.entropy_history.append({
            'step': step,
            'entropy': entropy_value
        })
        
        if len(self.entropy_history) < self.window_size:
            return None
        
        # Look for sudden drop
        recent = [e['entropy'] for e in self.entropy_history[-self.window_size:]]
        start_entropy = recent[0]
        end_entropy = recent[-1]
        drop = start_entropy - end_entropy
        
        # Detect collapse
        if (start_entropy > self.min_start_entropy and 
            drop > self.drop_threshold):
            
            collapse_event = {
                'step': step,
                'start_entropy': start_entropy,
                'end_entropy': end_entropy,
                'drop': drop,
                'rate': drop / self.window_size
            }
            self.collapses.append(collapse_event)
            return collapse_event
        
        return None
    
    def get_collapse_statistics(self):
        """
        Analyze all detected collapses.
        """
        if not self.collapses:
            return {'count': 0}
        
        drops = [c['drop'] for c in self.collapses]
        rates = [c['rate'] for c in self.collapses]
        
        return {
            'count': len(self.collapses),
            'mean_drop': np.mean(drops),
            'max_drop': max(drops),
            'mean_rate': np.mean(rates),
            'collapses': self.collapses
        }
```

### 4.2 Leader Tracking

```python
"""
TECHNIQUE: LEADER TRACKING
COST: FREE (computed from attention weights)
REVEALS: Which hypotheses compete before collapse
"""

class LeaderTracker:
    def __init__(self, top_k=5, min_weight=0.05):
        self.top_k = top_k
        self.min_weight = min_weight
        self.leader_history = []
        
    def update(self, attention_weights, step):
        """
        Track top-k leaders at each step.
        """
        # Get top-k weights and indices
        top_weights, top_indices = attention_weights.topk(self.top_k, dim=-1)
        
        # Filter by minimum weight
        significant_mask = top_weights > self.min_weight
        
        leaders = {
            'step': step,
            'top_indices': top_indices,
            'top_weights': top_weights,
            'num_significant': significant_mask.sum(dim=-1).float().mean().item(),
            'winner_weight': top_weights[..., 0].mean().item(),
            'competition_ratio': (top_weights[..., 1] / 
                                 (top_weights[..., 0] + 1e-10)).mean().item()
        }
        
        self.leader_history.append(leaders)
        return leaders
    
    def detect_winner_emergence(self, window=5):
        """
        Detect when a clear winner emerges (end of competition).
        """
        if len(self.leader_history) < window:
            return None
        
        recent = self.leader_history[-window:]
        competition_ratios = [l['competition_ratio'] for l in recent]
        
        # Competition decreasing = winner emerging
        if (competition_ratios[0] > 0.5 and 
            competition_ratios[-1] < 0.3 and
            all(competition_ratios[i] >= competition_ratios[i+1] - 0.1 
                for i in range(len(competition_ratios)-1))):
            
            return {
                'step': recent[-1]['step'],
                'initial_competition': competition_ratios[0],
                'final_competition': competition_ratios[-1],
                'winner_index': recent[-1]['top_indices'][..., 0]
            }
        
        return None
```

### 4.3 Pump Cycle Detection

```python
"""
TECHNIQUE: PUMP CYCLE DETECTION
COST: FREE (computed from entropy history)
REVEALS: Oscillation pattern of tension → discharge → recovery
"""

class PumpCycleDetector:
    def __init__(self, min_cycle_length=10, min_amplitude=0.2):
        self.min_cycle_length = min_cycle_length
        self.min_amplitude = min_amplitude
        self.entropy_history = []
        self.cycles = []
        
    def update(self, entropy, step):
        """
        Update and detect pump cycles.
        """
        self.entropy_history.append({'step': step, 'entropy': entropy})
        
        if len(self.entropy_history) < self.min_cycle_length * 3:
            return None
        
        # Look for oscillation pattern
        return self._detect_cycle()
    
    def _detect_cycle(self):
        """
        Detect a complete tension → discharge → recovery cycle.
        """
        entropies = [e['entropy'] for e in self.entropy_history]
        
        # Find local maxima (tension peaks)
        maxima = []
        for i in range(1, len(entropies) - 1):
            if entropies[i] > entropies[i-1] and entropies[i] > entropies[i+1]:
                maxima.append(i)
        
        # Find local minima (discharge troughs)
        minima = []
        for i in range(1, len(entropies) - 1):
            if entropies[i] < entropies[i-1] and entropies[i] < entropies[i+1]:
                minima.append(i)
        
        # Look for pattern: max → min → max (one complete cycle)
        if len(maxima) >= 2 and len(minima) >= 1:
            for peak1 in maxima[:-1]:
                for trough in minima:
                    if trough > peak1:
                        for peak2 in maxima:
                            if peak2 > trough:
                                amplitude = entropies[peak1] - entropies[trough]
                                if amplitude > self.min_amplitude:
                                    cycle = {
                                        'tension_step': self.entropy_history[peak1]['step'],
                                        'discharge_step': self.entropy_history[trough]['step'],
                                        'recovery_step': self.entropy_history[peak2]['step'],
                                        'amplitude': amplitude,
                                        'period': peak2 - peak1
                                    }
                                    self.cycles.append(cycle)
                                    return cycle
        
        return None
```

---

## 5. Spectral Analysis

### 5.1 Per-Band Entropy Tracking

```python
"""
TECHNIQUE: PER-BAND ENTROPY TRACKING
COST: LOW (one FFT, already part of architecture)
REVEALS: Which frequency bands are uncertain
"""

def per_band_entropy(model, input_data, num_bands=7):
    """
    Compute entropy separately for each spectral band.
    """
    # Get spectral decomposition
    bands = model.spectral_decomposer(input_data)
    
    # For each band, get attention and entropy
    band_entropies = {}
    
    for band_idx, band_data in bands.items():
        # Get attention for this band
        attention = model.per_band_attention[band_idx](band_data)
        attention_weights = extract_attention_weights(
            model.per_band_attention[band_idx], 
            band_data
        )
        
        # Compute entropy
        _, norm_entropy = compute_entropy(list(attention_weights.values())[0])
        band_entropies[band_idx] = {
            'entropy': norm_entropy.mean().item(),
            'spatial_entropy': norm_entropy  # Full spatial map
        }
    
    return band_entropies


def spectral_collapse_order(band_entropy_history):
    """
    Determine the order in which bands collapse.
    
    Theory predicts: low-freq bands collapse before high-freq.
    """
    collapse_times = {}
    
    for band_idx in band_entropy_history[0].keys():
        entropies = [h[band_idx]['entropy'] for h in band_entropy_history]
        
        # Find first significant drop
        for i in range(1, len(entropies)):
            if entropies[i-1] > 0.6 and entropies[i] < 0.4:
                collapse_times[band_idx] = i
                break
    
    # Order by collapse time
    order = sorted(collapse_times.keys(), key=lambda b: collapse_times.get(b, float('inf')))
    
    return {
        'collapse_times': collapse_times,
        'collapse_order': order,
        'low_freq_first': order[:3] == list(range(3)) if len(order) >= 3 else None
    }
```

### 5.2 Cross-Band Correlation

```python
"""
TECHNIQUE: CROSS-BAND CORRELATION
COST: LOW (correlation on existing data)
REVEALS: How bands interact, information flow
"""

def cross_band_correlation(band_activations):
    """
    Compute correlation between activations of different bands.
    """
    num_bands = len(band_activations)
    correlation_matrix = torch.zeros(num_bands, num_bands)
    
    for i in range(num_bands):
        for j in range(num_bands):
            # Flatten and compute correlation
            flat_i = band_activations[i].flatten()
            flat_j = band_activations[j].flatten()
            
            corr = torch.corrcoef(torch.stack([flat_i, flat_j]))[0, 1]
            correlation_matrix[i, j] = corr
    
    return correlation_matrix


def band_information_flow(model, input_data):
    """
    Measure information flow between bands via wormhole attention.
    """
    # Track wormhole connections
    wormhole_stats = {}
    
    for source_band in range(7):
        for target_band in range(7):
            if source_band != target_band:
                # Get wormhole connections between these bands
                connections = model.wormhole_attention.get_connections(
                    source_band, target_band
                )
                wormhole_stats[(source_band, target_band)] = {
                    'num_connections': len(connections),
                    'mean_strength': connections.mean().item() if len(connections) > 0 else 0
                }
    
    return wormhole_stats
```

### 5.3 Prompt Spectral Decomposition

```python
"""
TECHNIQUE: PROMPT SPECTRAL DECOMPOSITION
COST: LOW (FFT on embeddings)
REVEALS: Frequency content of prompts
"""

def decompose_prompt_spectrally(model, prompt_embedding):
    """
    Decompose a prompt embedding into spectral components.
    """
    # Apply FFT to embedding
    fft = torch.fft.fft(prompt_embedding, dim=-1)
    
    # Compute magnitude spectrum
    magnitude = torch.abs(fft)
    
    # Compute phase spectrum
    phase = torch.angle(fft)
    
    # Band-wise energy
    n_freq = magnitude.size(-1)
    band_edges = [0, n_freq//32, n_freq//16, n_freq//8, 
                  n_freq//4, n_freq//2, 3*n_freq//4, n_freq]
    
    band_energies = []
    for i in range(len(band_edges) - 1):
        start, end = band_edges[i], band_edges[i+1]
        energy = magnitude[..., start:end].pow(2).sum().item()
        band_energies.append(energy)
    
    return {
        'magnitude': magnitude,
        'phase': phase,
        'band_energies': band_energies,
        'dominant_band': np.argmax(band_energies),
        'low_freq_ratio': sum(band_energies[:2]) / sum(band_energies)
    }


def compare_prompt_spectra(prompt_embedding_1, prompt_embedding_2):
    """
    Compare spectral content of two prompts.
    """
    spec1 = decompose_prompt_spectrally(None, prompt_embedding_1)
    spec2 = decompose_prompt_spectrally(None, prompt_embedding_2)
    
    # Spectral distance
    spectral_distance = np.sqrt(sum(
        (e1 - e2) ** 2 
        for e1, e2 in zip(spec1['band_energies'], spec2['band_energies'])
    ))
    
    # Phase coherence
    phase_diff = spec1['phase'] - spec2['phase']
    phase_coherence = torch.cos(phase_diff).mean().item()
    
    return {
        'spectral_distance': spectral_distance,
        'phase_coherence': phase_coherence,
        'same_dominant_band': spec1['dominant_band'] == spec2['dominant_band']
    }
```

---

## 6. Embedding and Visualization

### 6.1 Embedding Extraction

```python
"""
TECHNIQUE: EMBEDDING EXTRACTION
COST: VARIES (depends on what's extracted)
REVEALS: State of belief manifold at any point
"""

class EmbeddingExtractor:
    def __init__(self, model, layers_to_extract=None):
        self.model = model
        self.layers_to_extract = layers_to_extract or ['all']
        self.hooks = []
        self.embeddings = {}
        
    def setup_hooks(self):
        """
        Set up hooks to capture intermediate representations.
        """
        def capture(name):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    self.embeddings[name] = output[0].detach()
                else:
                    self.embeddings[name] = output.detach()
            return hook
        
        for name, module in self.model.named_modules():
            if (self.layers_to_extract == ['all'] or 
                any(layer in name for layer in self.layers_to_extract)):
                self.hooks.append(
                    module.register_forward_hook(capture(name))
                )
    
    def extract(self, input_data):
        """
        Extract embeddings from a forward pass.
        """
        self.embeddings = {}
        self.setup_hooks()
        
        with torch.no_grad():
            output = self.model(input_data)
        
        # Remove hooks
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        
        return self.embeddings.copy()
    
    def extract_trajectory(self, input_sequence):
        """
        Extract embedding trajectory over a sequence of inputs.
        """
        trajectory = []
        for input_data in input_sequence:
            embeddings = self.extract(input_data)
            trajectory.append(embeddings)
        return trajectory
```

### 6.2 Dimensionality Reduction

```python
"""
TECHNIQUE: DIMENSIONALITY REDUCTION
COST: VARIES (PCA fast, UMAP slow)
REVEALS: Structure of high-dimensional space in visualizable form
"""

from sklearn.decomposition import PCA
import umap

class DimensionalityReducer:
    def __init__(self, method='pca', n_components=3):
        self.method = method
        self.n_components = n_components
        self.reducer = None
        
    def fit(self, embeddings):
        """
        Fit reducer to embeddings.
        """
        # Flatten if needed
        if embeddings.dim() > 2:
            embeddings = embeddings.flatten(start_dim=1)
        
        embeddings_np = embeddings.cpu().numpy()
        
        if self.method == 'pca':
            self.reducer = PCA(n_components=self.n_components)
        elif self.method == 'umap':
            self.reducer = umap.UMAP(n_components=self.n_components)
        elif self.method == 'shogu':
            # Custom Shogu reducer (placeholder)
            self.reducer = ShoguReducer(n_components=self.n_components)
        
        self.reducer.fit(embeddings_np)
        return self
    
    def transform(self, embeddings):
        """
        Project embeddings to low dimensions.
        """
        if embeddings.dim() > 2:
            embeddings = embeddings.flatten(start_dim=1)
        
        embeddings_np = embeddings.cpu().numpy()
        return self.reducer.transform(embeddings_np)
    
    def fit_transform(self, embeddings):
        """
        Fit and transform in one step.
        """
        self.fit(embeddings)
        return self.transform(embeddings)


class TrajectoryVisualizer:
    def __init__(self, reducer_method='pca'):
        self.reducer = DimensionalityReducer(method=reducer_method, n_components=3)
        
    def visualize_trajectory(self, trajectory, labels=None, title="Embedding Trajectory"):
        """
        Visualize a trajectory through embedding space.
        """
        # Stack all embeddings
        all_embeddings = torch.cat([emb.flatten().unsqueeze(0) 
                                    for emb in trajectory], dim=0)
        
        # Reduce
        projected = self.reducer.fit_transform(all_embeddings)
        
        # Plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Color by step
        colors = plt.cm.viridis(np.linspace(0, 1, len(projected)))
        
        # Plot trajectory
        ax.plot(projected[:, 0], projected[:, 1], projected[:, 2], 
                'k-', alpha=0.3, linewidth=0.5)
        ax.scatter(projected[:, 0], projected[:, 1], projected[:, 2], 
                  c=colors, s=20)
        
        # Mark start and end
        ax.scatter(*projected[0], c='green', s=100, marker='o', label='Start')
        ax.scatter(*projected[-1], c='red', s=100, marker='x', label='End')
        
        ax.set_title(title)
        ax.legend()
        
        return fig, projected
```

### 6.3 Real-Time Visualization

```python
"""
TECHNIQUE: REAL-TIME VISUALIZATION
COST: LOW (incremental updates)
REVEALS: Live dynamics of belief evolution
"""

class RealTimeVisualizer:
    def __init__(self, update_interval=10):
        self.update_interval = update_interval
        self.data_buffer = []
        self.fig = None
        self.ax = None
        
    def initialize(self):
        """
        Initialize the live plot.
        """
        plt.ion()
        self.fig, self.ax = plt.subplots(2, 2, figsize=(12, 10))
        
    def update(self, step, embeddings, entropy, error, collapse_events):
        """
        Update visualization with new data.
        """
        self.data_buffer.append({
            'step': step,
            'embeddings': embeddings,
            'entropy': entropy,
            'error': error,
            'collapse': collapse_events
        })
        
        if step % self.update_interval == 0:
            self._redraw()
    
    def _redraw(self):
        """
        Redraw all subplots.
        """
        # Clear axes
        for ax in self.ax.flat:
            ax.clear()
        
        steps = [d['step'] for d in self.data_buffer]
        
        # Plot 1: Entropy over time
        entropies = [d['entropy'] for d in self.data_buffer]
        self.ax[0, 0].plot(steps, entropies, 'b-')
        self.ax[0, 0].set_title('Entropy Over Time')
        self.ax[0, 0].set_ylabel('Entropy')
        
        # Mark collapse events
        for d in self.data_buffer:
            if d['collapse']:
                self.ax[0, 0].axvline(x=d['step'], color='r', linestyle='--', alpha=0.5)
        
        # Plot 2: Error over time
        errors = [d['error'] for d in self.data_buffer]
        self.ax[0, 1].plot(steps, errors, 'r-')
        self.ax[0, 1].set_title('Error Over Time')
        self.ax[0, 1].set_ylabel('Error')
        
        # Plot 3: Embedding trajectory (2D projection)
        if len(self.data_buffer) > 10:
            embeddings = torch.stack([d['embeddings'].flatten() 
                                      for d in self.data_buffer[-100:]])
            pca = PCA(n_components=2)
            projected = pca.fit_transform(embeddings.cpu().numpy())
            
            colors = plt.cm.viridis(np.linspace(0, 1, len(projected)))
            self.ax[1, 0].scatter(projected[:, 0], projected[:, 1], c=colors, s=10)
            self.ax[1, 0].set_title('Embedding Trajectory (PCA)')
        
        # Plot 4: Entropy heatmap (spatial)
        if self.data_buffer[-1]['embeddings'].dim() >= 2:
            entropy_map = self.data_buffer[-1]['entropy']
            if isinstance(entropy_map, torch.Tensor):
                self.ax[1, 1].imshow(entropy_map.cpu().numpy(), cmap='hot')
                self.ax[1, 1].set_title('Entropy Map (Current)')
        
        self.fig.tight_layout()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
```

---

## 7. Statistical Validation

### 7.1 Hypothesis Testing

```python
"""
TECHNIQUE: STATISTICAL HYPOTHESIS TESTING
COST: LOW (post-processing)
REVEALS: Whether observed patterns are significant
"""

from scipy import stats

def test_collapse_suddenness(collapse_events, threshold_exponent=0.5):
    """
    Test whether collapse is sudden (exponential) vs gradual (linear).
    
    H0: Collapse is linear
    H1: Collapse is exponential
    """
    results = []
    
    for event in collapse_events:
        entropy_curve = event['entropy_curve']
        steps = np.arange(len(entropy_curve))
        
        # Fit linear model
        slope, intercept, r_linear, _, _ = stats.linregress(steps, entropy_curve)
        
        # Fit exponential model: entropy = a * exp(-b * t)
        def exp_model(t, a, b):
            return a * np.exp(-b * t)
        
        try:
            from scipy.optimize import curve_fit
            popt, _ = curve_fit(exp_model, steps, entropy_curve, p0=[1, 0.1])
            fitted_exp = exp_model(steps, *popt)
            r_exp = np.corrcoef(entropy_curve, fitted_exp)[0, 1]
        except:
            r_exp = 0
        
        results.append({
            'r_linear': r_linear ** 2,
            'r_exponential': r_exp ** 2,
            'is_sudden': r_exp ** 2 > r_linear ** 2
        })
    
    # Aggregate test
    n_sudden = sum(r['is_sudden'] for r in results)
    n_total = len(results)
    p_value = stats.binom_test(n_sudden, n_total, 0.5, alternative='greater')
    
    return {
        'individual_results': results,
        'n_sudden': n_sudden,
        'n_total': n_total,
        'p_value': p_value,
        'hypothesis_supported': p_value < 0.05
    }


def test_spectral_hierarchy(collapse_orders, expected_order=list(range(7))):
    """
    Test whether low-freq bands collapse before high-freq.
    
    H0: Collapse order is random
    H1: Collapse order follows spectral hierarchy
    """
    # Compute Kendall's tau (rank correlation)
    taus = []
    
    for order in collapse_orders:
        tau, p = stats.kendalltau(order, expected_order)
        taus.append(tau)
    
    # One-sample t-test: is mean tau > 0?
    t_stat, p_value = stats.ttest_1samp(taus, 0)
    
    return {
        'mean_tau': np.mean(taus),
        'std_tau': np.std(taus),
        't_statistic': t_stat,
        'p_value': p_value / 2,  # One-tailed
        'hypothesis_supported': p_value / 2 < 0.05 and np.mean(taus) > 0
    }
```

### 7.2 Effect Size Measurement

```python
"""
TECHNIQUE: EFFECT SIZE MEASUREMENT
COST: LOW (post-processing)
REVEALS: Practical significance of findings
"""

def compute_effect_sizes(experimental_data, control_data):
    """
    Compute various effect sizes.
    """
    exp_mean = np.mean(experimental_data)
    ctrl_mean = np.mean(control_data)
    pooled_std = np.sqrt((np.var(experimental_data) + np.var(control_data)) / 2)
    
    # Cohen's d
    cohens_d = (exp_mean - ctrl_mean) / pooled_std
    
    # Glass's delta (use control std)
    glass_delta = (exp_mean - ctrl_mean) / np.std(control_data)
    
    # Hedges' g (corrected for small samples)
    n = len(experimental_data) + len(control_data)
    hedges_g = cohens_d * (1 - 3 / (4 * n - 9))
    
    return {
        'cohens_d': cohens_d,
        'glass_delta': glass_delta,
        'hedges_g': hedges_g,
        'effect_interpretation': interpret_effect_size(cohens_d)
    }


def interpret_effect_size(d):
    """
    Interpret Cohen's d.
    """
    d = abs(d)
    if d < 0.2:
        return 'negligible'
    elif d < 0.5:
        return 'small'
    elif d < 0.8:
        return 'medium'
    else:
        return 'large'
```

---

## 8. Implementation Patterns

### 8.1 Experiment Runner

```python
"""
PATTERN: EXPERIMENT RUNNER
Standardized experiment execution and logging
"""

import json
from datetime import datetime
from pathlib import Path

class ExperimentRunner:
    def __init__(self, name, model, output_dir='experiments'):
        self.name = name
        self.model = model
        self.output_dir = Path(output_dir) / name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.config = {}
        self.results = {}
        self.start_time = None
        
    def configure(self, **kwargs):
        """
        Set experiment configuration.
        """
        self.config.update(kwargs)
        
    def run(self, data_loader, techniques):
        """
        Run experiment with specified techniques.
        """
        self.start_time = datetime.now()
        
        # Initialize techniques
        for technique_name, technique in techniques.items():
            technique.reset()
        
        # Run through data
        for batch_idx, batch in enumerate(data_loader):
            # Forward pass
            with torch.no_grad():
                output = self.model(batch)
            
            # Apply each technique
            for technique_name, technique in techniques.items():
                result = technique.analyze(self.model, batch, output, batch_idx)
                if technique_name not in self.results:
                    self.results[technique_name] = []
                self.results[technique_name].append(result)
        
        # Finalize
        self._save_results()
        
    def _save_results(self):
        """
        Save experiment results.
        """
        end_time = datetime.now()
        
        output = {
            'name': self.name,
            'config': self.config,
            'start_time': str(self.start_time),
            'end_time': str(end_time),
            'duration': str(end_time - self.start_time),
            'results': {k: [self._serialize(r) for r in v] 
                       for k, v in self.results.items()}
        }
        
        with open(self.output_dir / 'results.json', 'w') as f:
            json.dump(output, f, indent=2)
    
    def _serialize(self, obj):
        """
        Convert results to JSON-serializable format.
        """
        if isinstance(obj, torch.Tensor):
            return obj.cpu().numpy().tolist()
        elif isinstance(obj, dict):
            return {k: self._serialize(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._serialize(v) for v in obj]
        else:
            return obj
```

### 8.2 Technique Interface

```python
"""
PATTERN: TECHNIQUE INTERFACE
Standardized interface for all experimental techniques
"""

from abc import ABC, abstractmethod

class ExperimentalTechnique(ABC):
    """
    Base class for all experimental techniques.
    """
    
    def __init__(self, name, cost_category='free'):
        self.name = name
        self.cost_category = cost_category  # 'free', 'cheap', 'moderate', 'expensive'
        self.history = []
        
    @abstractmethod
    def analyze(self, model, input_data, output, step):
        """
        Analyze model state at this step.
        
        Returns:
            dict: Analysis results
        """
        pass
    
    def reset(self):
        """
        Reset technique state for new experiment.
        """
        self.history = []
    
    def get_summary(self):
        """
        Get summary statistics from history.
        """
        if not self.history:
            return {}
        
        # Default: aggregate numeric values
        summary = {}
        for key in self.history[0].keys():
            values = [h[key] for h in self.history if isinstance(h[key], (int, float))]
            if values:
                summary[key] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': min(values),
                    'max': max(values)
                }
        return summary


class EntropyTechnique(ExperimentalTechnique):
    """
    Example technique: Entropy analysis.
    """
    
    def __init__(self):
        super().__init__('entropy', cost_category='free')
        
    def analyze(self, model, input_data, output, step):
        attention_weights = extract_attention_weights(model, input_data)
        
        results = {}
        for name, weights in attention_weights.items():
            _, normalized = compute_entropy(weights)
            results[name] = {
                'mean_entropy': normalized.mean().item(),
                'max_entropy': normalized.max().item(),
                'min_entropy': normalized.min().item()
            }
        
        self.history.append(results)
        return results
```

---

## Summary

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│          E X P E R I M E N T A L   T E C H N I Q U E S                │
│                                                                         │
│  ═══════════════════════════════════════════════════════════════════   │
│                                                                         │
│  FREE TECHNIQUES:                                                       │
│  • Attention weight reading                                           │
│  • Entropy computation                                                 │
│  • Pre-softmax score extraction                                       │
│  • Similarity matrix extraction                                       │
│                                                                         │
│  TICKLING TECHNIQUES:                                                   │
│  • Temperature probing (2-3 FP)                                       │
│  • Threshold sweeping (5-10 FP)                                       │
│  • Sparse probing (2 FP)                                              │
│  • PSON exploration (1-5 FP)                                          │
│                                                                         │
│  COLLAPSE DETECTION:                                                    │
│  • Entropy drop detection                                             │
│  • Leader tracking                                                    │
│  • Pump cycle detection                                               │
│                                                                         │
│  SPECTRAL ANALYSIS:                                                     │
│  • Per-band entropy tracking                                          │
│  • Cross-band correlation                                             │
│  • Prompt spectral decomposition                                      │
│                                                                         │
│  VISUALIZATION:                                                         │
│  • Embedding extraction                                               │
│  • Dimensionality reduction (PCA, UMAP, Shogu)                       │
│  • Real-time visualization                                            │
│                                                                         │
│  STATISTICAL VALIDATION:                                                │
│  • Hypothesis testing                                                 │
│  • Effect size measurement                                            │
│                                                                         │
│  ═══════════════════════════════════════════════════════════════════   │
│                                                                         │
│  KEY PRINCIPLES:                                                        │
│  1. Extract before computing (use free information)                   │
│  2. Cheap before expensive (always try cheap first)                   │
│  3. Falsify before validate (design to disprove)                     │
│  4. Dual views before single view (look from multiple angles)        │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

*Oscar Goldman — Shogu Research Group @ Datamutant.ai*

*"The best experiment is the one you don't have to run because the information is already there. Extract everything free. Tickle before searching. Validate with statistics. This is the experimental method."*

