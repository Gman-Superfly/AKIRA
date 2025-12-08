# FREE INFORMATION ASSETS

## Everything We Already Compute (And Usually Throw Away)

**Oscar Goldman — Shogu Research Group @ Datamutant.ai**

---

## The Core Insight

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  THE INSIGHT:                                                           │
│                                                                         │
│  Neural network inference ALREADY COMPUTES information about:          │
│  • Where the model is uncertain                                       │
│  • What hypotheses are competing                                      │
│  • Which connections almost activated                                 │
│  • How close the race is between winners                             │
│                                                                         │
│  WE THROW IT ALL AWAY.                                                 │
│                                                                         │
│  We take the argmax, the top-k, the final output.                     │
│  All the structure is lost.                                            │
│                                                                         │
│  This document catalogs what's available FOR FREE.                    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 1. Attention Weights

### What's Computed

```
ATTENTION WEIGHTS: Already Computed in Every Forward Pass

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  Standard attention computes:                                          │
│                                                                         │
│  scores = Q @ K.T / sqrt(d)                                           │
│  weights = softmax(scores)        ← THIS IS THROWN AWAY               │
│  output = weights @ V             ← THIS IS KEPT                      │
│                                                                         │
│  THE WEIGHTS CONTAIN:                                                   │
│  • WHERE the model attends (position information)                     │
│  • HOW MUCH it attends (confidence per connection)                    │
│  • HOW SPREAD the attention is (uncertainty structure)                │
│  • WHICH alternatives were considered (other high weights)            │
│                                                                         │
│  COST TO EXTRACT: 0 (just read it before discarding)                  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### How to Extract

```python
# Hook-based extraction
def extract_attention_weights(model, input_data):
    """Extract attention weights from forward pass."""
    weights = {}
    
    def hook(name):
        def fn(module, input, output):
            if hasattr(module, 'attn_weights'):
                weights[name] = module.attn_weights.detach()
            elif isinstance(output, tuple) and len(output) >= 2:
                weights[name] = output[1].detach()
        return fn
    
    hooks = []
    for name, module in model.named_modules():
        if 'attn' in name.lower():
            hooks.append(module.register_forward_hook(hook(name)))
    
    with torch.no_grad():
        model(input_data)
    
    for h in hooks:
        h.remove()
    
    return weights
```

### What It Reveals

```
ATTENTION WEIGHTS → INFORMATION AVAILABLE

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  WEIGHT PATTERN          INTERPRETATION                                │
│  ──────────────          ──────────────                                │
│                                                                         │
│  One high weight         Clear winner, confident                       │
│  Two similar weights     Competition, uncertain                        │
│  Many small weights      Confused, high entropy                        │
│  Clustered weights       Focused on region                             │
│  Sparse weights          Specific matches                              │
│  Uniform weights         No preference, maximum uncertainty            │
│                                                                         │
│  FROM THIS YOU CAN DERIVE:                                              │
│  • Entropy (uncertainty)                                               │
│  • Sparsity (focus)                                                    │
│  • Leader positions (top-k indices)                                   │
│  • Competition ratio (2nd best / best)                                │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Entropy

### What's Computed

```
ENTROPY: Simple Function of Attention Weights

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  DEFINITION:                                                            │
│  H(p) = -Σ p_i log(p_i)                                               │
│                                                                         │
│  WHERE:                                                                 │
│  • H = 0 means one hypothesis dominates (collapsed)                   │
│  • H = log(N) means all hypotheses equal (maximum uncertainty)        │
│  • H = medium means few competing hypotheses (leaders visible)        │
│                                                                         │
│  COST: One pass over weights (microseconds)                           │
│                                                                         │
│  NORMALIZED:                                                            │
│  H_norm = H / log(N)  →  range [0, 1]                                 │
│                                                                         │
│  0 = collapsed, 1 = maximum entropy                                   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### How to Extract

```python
def compute_entropy(attention_weights, dim=-1, eps=1e-10):
    """Compute entropy of attention distribution."""
    p = attention_weights.clamp(min=eps)
    entropy = -(p * p.log()).sum(dim=dim)
    
    n = attention_weights.size(dim)
    max_entropy = torch.log(torch.tensor(n, dtype=p.dtype))
    normalized = entropy / max_entropy
    
    return entropy, normalized


def entropy_map(attention_weights):
    """Create spatial map of entropy."""
    _, normalized = compute_entropy(attention_weights)
    
    if normalized.dim() == 4:  # [batch, heads, seq, seq]
        normalized = normalized.mean(dim=1)  # Average over heads
    
    return normalized
```

### What It Reveals

```
ENTROPY → OPERATIONAL DECISIONS

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  ENTROPY VALUE        STATUS                 ACTION                    │
│  ─────────────        ──────                 ──────                    │
│                                                                         │
│  H_norm > 0.9         Before edge            Keep probing              │
│                       (leaders not formed)                             │
│                                                                         │
│  0.3 < H_norm < 0.9   AT THE EDGE            Extract leaders          │
│                       (leaders visible)       Maximum information      │
│                                                                         │
│  H_norm < 0.3         Past edge              Commit / Collapse        │
│                       (winner clear)                                   │
│                                                                         │
│  SPATIAL ENTROPY MAP:                                                   │
│  • High-H regions = uncertain zones (leader zones)                    │
│  • Low-H regions = confident zones (already decided)                  │
│  • Edge regions = transition zones (probe here)                       │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Pre-Softmax Scores

### What's Computed

```
PRE-SOFTMAX SCORES: The Raw Competition

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  STANDARD FLOW:                                                         │
│  scores = Q @ K.T / sqrt(d)      ← RAW SCORES (informative!)          │
│  weights = softmax(scores)       ← POST-SOFTMAX (loses info!)         │
│                                                                         │
│  WHAT SOFTMAX HIDES:                                                    │
│                                                                         │
│  Raw scores:    [2.1, 1.9, 0.3, 0.1]                                  │
│  After softmax: [0.52, 0.43, 0.03, 0.02]                              │
│                                                                         │
│  The raw scores show: 2.1 and 1.9 are CLOSE!                          │
│  The post-softmax weights hide the gap.                               │
│                                                                         │
│  COST: Store scores before softmax (no additional computation)        │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### How to Extract

```python
# Modify attention to store pre-softmax scores
class AttentionWithScores(nn.Module):
    def forward(self, q, k, v):
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))
        
        # Store for extraction
        self.last_scores = scores.detach()
        
        weights = F.softmax(scores, dim=-1)
        return torch.matmul(weights, v), weights
```

### What It Reveals

```
PRE-SOFTMAX SCORES → TRUE COMPETITION

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  METRIC                FORMULA                    MEANING              │
│  ──────                ───────                    ───────              │
│                                                                         │
│  Gap                   s[0] - s[1]                How clear is winner │
│  Competition ratio     s[1] / s[0]                How close is race   │
│  Margin to threshold   s[0] - threshold           Safety margin       │
│  Score variance        var(s)                     Competition spread  │
│                                                                         │
│  INTERPRETATION:                                                        │
│                                                                         │
│  Gap large, ratio small → Clear winner, safe to commit               │
│  Gap small, ratio high  → Close race, tickle more                    │
│  Many scores near top   → Multiple leaders, rich structure           │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Similarity Matrix (Wormhole)

### What's Computed

```
SIMILARITY MATRIX: The Full Landscape

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  WORMHOLE ATTENTION COMPUTES:                                          │
│                                                                         │
│  query_norm = normalize(query)                                        │
│  key_norm = normalize(keys)                                           │
│  similarity = query_norm @ key_norm.T    ← FULL MATRIX (valuable!)   │
│  top_k_sim, top_k_idx = topk(similarity) ← ONLY TOP-K KEPT           │
│  mask = top_k_sim > threshold            ← GATING                     │
│  output = aggregate(values[masked])      ← FINAL OUTPUT               │
│                                                                         │
│  THE FULL SIMILARITY MATRIX CONTAINS:                                  │
│  • Every query-to-key similarity (not just winners)                   │
│  • Near-threshold connections (almost-leaders)                        │
│  • The distribution of potential connections                          │
│                                                                         │
│  COST: 0 (already computed, usually discarded after top-k)            │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### How to Extract

```python
def extract_full_similarity(wormhole_module, query, keys):
    """Extract full similarity matrix before top-k selection."""
    q_norm = F.normalize(query.flatten(2), p=2, dim=-1)
    k_norm = F.normalize(keys.flatten(2), p=2, dim=-1)
    
    similarity = torch.matmul(q_norm, k_norm.transpose(-1, -2))
    return similarity


def analyze_similarity_distribution(similarity, threshold=0.92):
    """Analyze full similarity distribution."""
    return {
        'mean': similarity.mean().item(),
        'std': similarity.std().item(),
        'above_threshold': (similarity > threshold).sum().item(),
        'near_threshold': ((similarity > threshold - 0.05) & 
                          (similarity <= threshold)).sum().item(),
        'histogram': torch.histogram(similarity.flatten(), bins=50)
    }
```

### What It Reveals

```
SIMILARITY MATRIX → MANIFOLD STRUCTURE

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  FROM THE FULL MATRIX YOU CAN SEE:                                     │
│                                                                         │
│  PEAKS:              Local maxima = potential wormhole targets        │
│  NEAR-THRESHOLD:     Almost-activated connections = leaders waiting   │
│  CLUSTERS:           Groups of high similarity = manifold regions     │
│  DISTRIBUTION:       How uniform/concentrated matches are             │
│                                                                         │
│  SPECIFIC INSIGHTS:                                                     │
│                                                                         │
│  Many near-threshold → Rich leader structure, probe more              │
│  Few near-threshold  → Sparse manifold, committed structure           │
│  Bimodal distribution → Two distinct regions (regime split)          │
│  Uniform distribution → No structure (high entropy)                   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Gradient Direction

### What's Computed

```
GRADIENT: Direction of Improvement

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  DURING TRAINING:                                                       │
│  loss = criterion(output, target)                                     │
│  loss.backward()                  ← GRADIENT COMPUTED                 │
│  optimizer.step()                 ← GRADIENT APPLIED                  │
│                                                                         │
│  THE GRADIENT TELLS YOU:                                                │
│  • Which direction reduces loss                                       │
│  • How sensitive each parameter is                                    │
│  • Where the model is "wrong"                                         │
│                                                                         │
│  COST: Already computed if training                                    │
│  COST: One backward pass if inference-only                            │
│                                                                         │
│  CAN READ WITHOUT STEPPING:                                            │
│  Just compute gradient, don't apply update.                           │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### How to Extract

```python
def extract_gradient_info(model, input_data, target):
    """Extract gradient without updating."""
    model.zero_grad()
    
    output = model(input_data)
    loss = F.mse_loss(output, target)
    loss.backward()
    
    gradients = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            gradients[name] = {
                'grad': param.grad.detach().clone(),
                'norm': param.grad.norm().item(),
                'mean': param.grad.mean().item(),
                'std': param.grad.std().item()
            }
    
    # Don't actually step
    model.zero_grad()
    
    return gradients
```

### What It Reveals

```
GRADIENT → SENSITIVITY MAP

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  GRADIENT PATTERN         MEANING                                      │
│  ────────────────         ───────                                      │
│                                                                         │
│  Large gradient at X      X is uncertain/wrong                        │
│  Small gradient at Y      Y is confident/correct                      │
│  Gradient direction       Which hypothesis would win if we stepped    │
│  Gradient magnitude       How much we'd change                        │
│                                                                         │
│  INPUT GRADIENTS:                                                       │
│  Large → input pixels that matter for prediction                      │
│  Small → input pixels that don't affect prediction                    │
│                                                                         │
│  PARAMETER GRADIENTS:                                                   │
│  Large → parameters that need updating                                │
│  Small → parameters that are well-tuned                               │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Hidden States / Intermediate Activations

### What's Computed

```
INTERMEDIATE ACTIVATIONS: The Belief Evolution

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  EVERY LAYER COMPUTES:                                                  │
│  h_1 = layer_1(input)                                                 │
│  h_2 = layer_2(h_1)                                                   │
│  h_3 = layer_3(h_2)                                                   │
│  ...                                                                   │
│  output = final_layer(h_n)                                            │
│                                                                         │
│  THE INTERMEDIATE h_i ARE THE BELIEF EVOLUTION:                        │
│  • Early layers: low-level features                                   │
│  • Middle layers: composition, abstraction                            │
│  • Late layers: decision formation                                    │
│                                                                         │
│  COST: 0 (just read them during forward pass)                         │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### How to Extract

```python
class ActivationExtractor:
    def __init__(self, model, layers=None):
        self.model = model
        self.layers = layers
        self.activations = {}
        self.hooks = []
    
    def extract(self, input_data):
        def hook(name):
            def fn(module, input, output):
                if isinstance(output, tuple):
                    self.activations[name] = output[0].detach()
                else:
                    self.activations[name] = output.detach()
            return fn
        
        for name, module in self.model.named_modules():
            if self.layers is None or name in self.layers:
                self.hooks.append(module.register_forward_hook(hook(name)))
        
        with torch.no_grad():
            output = self.model(input_data)
        
        for h in self.hooks:
            h.remove()
        self.hooks = []
        
        return self.activations
```

### What It Reveals

```
LAYER ACTIVATIONS → BELIEF TRAJECTORY

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  LAYER-WISE ANALYSIS:                                                   │
│                                                                         │
│  Early layers:   Features extracted, many hypotheses alive            │
│  Middle layers:  Competition happens, leaders emerge                  │
│  Late layers:    Winner dominates, collapse complete                  │
│                                                                         │
│  THE "LOGIT LENS" TECHNIQUE:                                            │
│  Decode hidden state h_i through output projection at each layer.    │
│  See which outputs are "winning" at each depth.                       │
│                                                                         │
│  Layer 5:  ["cat", "dog", "animal", "pet", "mammal"]  ← Many options  │
│  Layer 10: ["cat", "dog", "pet"]                       ← Narrowing    │
│  Layer 15: ["cat", "dog"]                              ← Competition  │
│  Output:   ["cat"]                                     ← Collapsed    │
│                                                         │                │
│  YOU CAN SEE THE LEADERS BEFORE FINAL OUTPUT!                         │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 7. Spectral Decomposition

### What's Computed

```
SPECTRAL BANDS: Already Part of AKIRA Architecture

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  AKIRA COMPUTES:                                                        │
│  bands = FFT(input)  →  [band_0, band_1, ..., band_6]                 │
│                                                                         │
│  EACH BAND HAS:                                                         │
│  • Magnitude (energy at that frequency)                               │
│  • Phase (position information)                                       │
│  • Spatial structure (where the frequency content is)                 │
│                                                                         │
│  PER-BAND ATTENTION COMPUTES:                                          │
│  For each band: attention weights, entropy, output                    │
│                                                                         │
│  COST: 0 (already computed as part of forward pass)                   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### How to Extract

```python
def extract_per_band_info(model, input_data):
    """Extract information from each spectral band."""
    bands = model.spectral_decomposer(input_data)
    
    band_info = {}
    for band_idx, band_data in bands.items():
        # Energy
        energy = band_data.pow(2).sum().item()
        
        # Per-band attention
        attn_weights = model.per_band_attention[band_idx].get_weights()
        _, entropy = compute_entropy(attn_weights)
        
        band_info[band_idx] = {
            'energy': energy,
            'entropy': entropy.mean().item(),
            'magnitude': band_data.abs().mean().item()
        }
    
    return band_info
```

### What It Reveals

```
PER-BAND INFO → SPECTRAL DYNAMICS

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  BAND       FREQUENCY    CONTENT            DYNAMICS                   │
│  ────       ─────────    ───────            ────────                   │
│                                                                         │
│  Band 0     DC           Existence          Stable, slow collapse      │
│  Band 1     Very low     Identity           Slow adaptation            │
│  Band 2     Low          Category           Moderate adaptation        │
│  Band 3     Mid-low      Features           Adaptive                   │
│  Band 4     Mid-high     Configuration      Fast adaptation            │
│  Band 5     High         Position           Very fast                  │
│  Band 6     Very high    Details            Immediate                  │
│                                                                         │
│  SPECTRAL HIERARCHY PREDICTIONS:                                        │
│  • Low bands collapse first (what) → High bands last (where)         │
│  • Low bands have lower entropy (more stable)                        │
│  • High bands have higher entropy (more uncertain)                   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 8. Near-Threshold Connections

### What's Computed

```
NEAR-THRESHOLD: The Almost-Leaders

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  WORMHOLE GATING:                                                       │
│  active = similarity > threshold (e.g., 0.92)                         │
│                                                                         │
│  WHAT'S NEAR THRESHOLD:                                                │
│  near = (similarity > threshold - ε) & (similarity <= threshold)     │
│                                                                         │
│  NEAR-THRESHOLD CONNECTIONS ARE:                                       │
│  • Connections that ALMOST activated                                  │
│  • Leaders that are WAITING                                           │
│  • Hypotheses that are CLOSE but didn't win                          │
│                                                                         │
│  COST: 0 (just filter the similarity matrix you already have)        │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### How to Extract

```python
def count_near_threshold(similarity, threshold=0.92, epsilon=0.05):
    """Count connections just below threshold."""
    near_mask = (similarity > threshold - epsilon) & (similarity <= threshold)
    
    return {
        'count': near_mask.sum().item(),
        'fraction': near_mask.float().mean().item(),
        'positions': near_mask.nonzero(),
        'values': similarity[near_mask]
    }


def threshold_sensitivity(similarity, thresholds=[0.90, 0.91, 0.92, 0.93, 0.94]):
    """How does connection count change with threshold?"""
    counts = []
    for t in thresholds:
        count = (similarity > t).sum().item()
        counts.append(count)
    
    # Gradient: where does count change fastest?
    gradients = [counts[i] - counts[i+1] for i in range(len(counts)-1)]
    
    return {
        'thresholds': thresholds,
        'counts': counts,
        'gradients': gradients,
        'steepest_threshold': thresholds[np.argmax(gradients)]
    }
```

### What It Reveals

```
NEAR-THRESHOLD → LEADER STRUCTURE

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  NEAR-THRESHOLD COUNT        MEANING                                   │
│  ────────────────────        ───────                                   │
│                                                                         │
│  Many near-threshold         Rich leader structure                    │
│                              Many hypotheses waiting                   │
│                              Lower threshold → many activations        │
│                                                                         │
│  Few near-threshold          Sparse manifold                          │
│                              Clear winners                             │
│                              Threshold doesn't matter much            │
│                                                                         │
│  THRESHOLD SENSITIVITY:                                                 │
│  High gradient at τ → many leaders just below τ                       │
│  Low gradient at τ  → gap in similarity distribution                  │
│                                                                         │
│  USE FOR:                                                               │
│  • Adaptive threshold selection                                       │
│  • Detecting when to probe more                                       │
│  • Identifying rich vs sparse regions                                 │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 9. Summary: The Free Information Catalog

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│           F R E E   I N F O R M A T I O N   C A T A L O G              │
│                                                                         │
│  ═══════════════════════════════════════════════════════════════════   │
│                                                                         │
│  ASSET                  COST    REVEALS                                │
│  ─────                  ────    ───────                                │
│                                                                         │
│  Attention weights      0       Where model attends                    │
│  Entropy                0       Uncertainty per position               │
│  Pre-softmax scores     0       True competition                       │
│  Similarity matrix      0       Full connection landscape              │
│  Gradient (training)    0       Direction of improvement               │
│  Hidden states          0       Belief evolution                       │
│  Spectral bands         0       Frequency decomposition                │
│  Near-threshold count   0       Almost-leaders                         │
│                                                                         │
│  ═══════════════════════════════════════════════════════════════════   │
│                                                                         │
│  DERIVED METRICS (microseconds):                                       │
│                                                                         │
│  Competition ratio      s[1]/s[0]      How close the race is         │
│  Leader count           count(w > τ)   Number of hypotheses          │
│  Edge-of-error          H in [0.3,0.9] At optimal operating point    │
│  Spectral energy        |band|²        Activity per frequency        │
│  Collapse detection     dH/dt          Sudden entropy drop            │
│                                                                         │
│  ═══════════════════════════════════════════════════════════════════   │
│                                                                         │
│  THE PRINCIPLE:                                                         │
│  READ BEFORE COMPUTING.                                                │
│  Everything in this catalog is ALREADY COMPUTED.                      │
│  We just throw it away. Stop throwing it away.                        │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

*Oscar Goldman — Shogu Research Group @ Datamutant.ai*

*"The electric field is already there. The leaders are in the attention weights. The competition is in the pre-softmax scores. We compute all of this and throw it away. Stop. Read it first. The information is free."*

