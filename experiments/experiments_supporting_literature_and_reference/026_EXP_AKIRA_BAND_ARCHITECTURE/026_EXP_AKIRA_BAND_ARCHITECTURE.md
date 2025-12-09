# Experiment 026: AKIRA Band Architecture vs Standard Transformer

**Status:** PLANNED

**Date:** December 5, 2025

**Author:** Oscar Goldman, Shogu Research Group @ Datamutant.ai

---

## Motivation

Experiment 000 revealed that universal features (Action Quanta) exist in standard transformers like GPT-2, and that middle layers (Layer 5) show low-frequency concentration. However, this spectral structure is **emergent and implicit** - the model discovered it through training, not by design.

AKIRA proposes that explicit spectral band structure should produce:
1. More universal features (better organized)
2. Cleaner spectral separation (features know their band)
3. Stronger theoretical grounding (designed, not emergent)
4. Better interpretability (band = frequency scale)

This experiment tests whether explicit band architecture produces measurably different Action Quanta properties.

---

## Research Questions

1. **Does explicit band structure increase AQ count?**
   - Standard transformer: ~3-4% of neurons are AQ
   - AKIRA prediction: Higher percentage due to organized structure

2. **Does explicit band structure improve spectral separation?**
   - Standard transformer: Layer 5 shows low-freq concentration (emergent)
   - AKIRA prediction: Low-freq bands should show strong AQ concentration by design

3. **Does explicit band structure improve universality?**
   - Standard transformer: ~25% overlap between structural and semantic AQ
   - AKIRA prediction: Higher overlap due to more stable feature formation

4. **Does explicit band structure affect learning dynamics?**
   - Track per-band loss curves
   - Observe grokking patterns per band
   - Test if low-freq bands stabilize earlier

---

## Experimental Design

### Architecture Comparison

**Model A: Standard Transformer (Baseline)**
```
- 6 layers
- 512 hidden dim
- 8 attention heads
- Single MLP per layer (512 -> 2048 -> 512)
- ~25M parameters
- Standard learning rate: 3e-4
```

**Model B: AKIRA Band Transformer**
```
- 6 layers
- 512 hidden dim (distributed across 7 bands)
- 8 attention heads (with wormhole capability)
- 7 parallel MLPs per layer (one per band)
- Band dimensions: [128, 96, 80, 64, 64, 48, 32] = 512 total
- ~25M parameters (same as baseline)
- Per-band learning rates:
  - Band 0 (DC):        1e-5  (slowest, most stable)
  - Band 1 (low):       3e-5
  - Band 2 (low-mid):   1e-4
  - Band 3 (mid):       3e-4
  - Band 4 (mid-high):  1e-3
  - Band 5 (high):      3e-3
  - Band 6 (highest):   1e-2  (fastest, most adaptive)
```

### AKIRA Band Architecture Details

```python
class AKIRABandTransformer:
    """
    Transformer with explicit spectral band structure.
    
    Key differences from standard transformer:
    1. Parallel MLPs instead of single MLP
    2. Each band has own learning rate
    3. Wormhole attention allows cross-band communication
    4. Band sizes follow AKIRA's decreasing pattern (low-freq = larger)
    """
    
    def __init__(self, config):
        self.num_bands = 7
        self.band_dims = [128, 96, 80, 64, 64, 48, 32]  # Sum = 512
        self.band_lrs = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
        
        # Per-band MLPs
        self.band_mlps = [
            MLP(dim, dim * 4, dim) for dim in self.band_dims
        ]
        
        # Standard attention (operates on concatenated bands)
        self.attention = MultiHeadAttention(512, 8)
        
        # Wormhole attention (cross-band, optional)
        self.wormhole = WormholeAttention(self.band_dims)
    
    def forward(self, x):
        # Split input into bands
        bands = self.split_bands(x)
        
        # Process each band through its MLP
        for i, (band, mlp) in enumerate(zip(bands, self.band_mlps)):
            bands[i] = mlp(band)
        
        # Concatenate for attention
        x = self.concat_bands(bands)
        x = self.attention(x)
        
        # Optional: wormhole cross-band communication
        x = self.wormhole(x, bands)
        
        return x
```

### Training Configuration

```yaml
# Common settings
dataset: WikiText-103
batch_size: 64
sequence_length: 512
total_steps: 100,000
warmup_steps: 1,000
weight_decay: 0.01

# Evaluation
eval_every: 1,000 steps
save_every: 10,000 steps

# Hardware
device: A100 (40GB)
mixed_precision: bf16
```

### Ablations

1. **Band structure only** (no differential LR)
   - Tests if architecture matters without LR schedule

2. **Differential LR only** (no band structure)
   - Tests if LR schedule matters without explicit bands

3. **Full AKIRA** (bands + differential LR + wormhole)
   - Tests complete system

4. **AKIRA without wormhole**
   - Tests if cross-band communication is necessary

---

## Measurements

### Phase 1: Training Dynamics

1. **Per-band loss curves**
   - Do low-freq bands converge first?
   - Do high-freq bands show more variance?

2. **Grokking detection**
   - Monitor generalization gap per band
   - Does grokking happen band-by-band?

3. **Feature emergence timeline**
   - When do AQ-like features appear in each band?

### Phase 2: AQ Analysis (Post-Training)

Using methods from Experiment 000:

1. **AQ Detection**
   - Train 2 identical AKIRA models with different seeds
   - Extract per-band activations
   - Apply Procrustes alignment
   - Count AQ candidates per band

2. **Spectral Analysis**
   - Compute spectral centroid per neuron
   - Compare: Do low-freq bands have lower centroids?
   - This should be TRUE BY DESIGN in AKIRA

3. **Universality Comparison**
   - Random tokens vs real text (as in Exp 000)
   - Measure overlap (Universal AQ)
   - Compare AKIRA vs baseline

### Phase 3: Cross-Architecture Comparison

1. **AKIRA vs GPT-2**
   - Same AQ detection methods
   - Does AKIRA have more AQ?
   - Are AKIRA AQ more universal?

2. **AKIRA Band 0 vs GPT-2 Layer 5**
   - Both should be "semantic abstraction"
   - Compare AQ properties

---

## Predictions

Based on AKIRA theory:

| Metric | Baseline | AKIRA | Rationale |
|--------|----------|-------|-----------|
| Total AQ % | ~3-4% | ~8-12% | Explicit structure encourages universality |
| Universal AQ overlap | ~25% | ~50% | Stable bands = stable features |
| Low-freq AQ concentration | Layer 5 only | Bands 0-2 | By design |
| Grokking | Global | Per-band | Bands mature at different rates |
| Interpretability | Low | High | Band = frequency scale |

### Falsification Criteria

The experiment **fails to support** AKIRA if:
1. AKIRA has fewer or equal AQ compared to baseline
2. Low-freq bands do NOT show AQ concentration
3. Universality overlap is not improved
4. Per-band learning rates cause instability

---

## Implementation Plan

### Step 1: Baseline Training (~2 hours on A100)
```bash
python train_baseline.py \
    --model standard_transformer \
    --hidden_dim 512 \
    --num_layers 6 \
    --dataset wikitext-103 \
    --steps 100000 \
    --output baseline_model/
```

### Step 2: AKIRA Training (~3 hours on A100)
```bash
python train_akira.py \
    --model akira_band_transformer \
    --num_bands 7 \
    --band_dims 128,96,80,64,64,48,32 \
    --band_lrs 1e-5,3e-5,1e-4,3e-4,1e-3,3e-3,1e-2 \
    --dataset wikitext-103 \
    --steps 100000 \
    --output akira_model/
```

### Step 3: Second AKIRA Training (different seed)
```bash
python train_akira.py \
    --seed 43 \
    --output akira_model_seed43/
```

### Step 4: AQ Analysis
```bash
python analyze_aq.py \
    --model1 akira_model/ \
    --model2 akira_model_seed43/ \
    --output aq_results/
```

### Step 5: Comparison
```bash
python compare_architectures.py \
    --baseline baseline_model/ \
    --akira akira_model/ \
    --output comparison_results/
```

---

## Expected Timeline

| Phase | Duration | Output |
|-------|----------|--------|
| Implementation | 2-3 days | Training code |
| Baseline training | 2 hours | baseline_model/ |
| AKIRA training x2 | 6 hours | akira_model/, akira_model_seed43/ |
| Ablation training x3 | 9 hours | ablation_models/ |
| AQ analysis | 1 hour | aq_results/ |
| Documentation | 1 day | Results writeup |

**Total: ~4-5 days**

---

## Code Structure

```
026_EXP_AKIRA_BAND_ARCHITECTURE/
├── 026_EXP_AKIRA_BAND_ARCHITECTURE.md  (this file)
├── code/
│   ├── models/
│   │   ├── baseline_transformer.py
│   │   ├── akira_band_transformer.py
│   │   └── wormhole_attention.py
│   ├── training/
│   │   ├── train_baseline.py
│   │   ├── train_akira.py
│   │   └── config.yaml
│   ├── analysis/
│   │   ├── analyze_aq.py
│   │   ├── spectral_analysis.py
│   │   └── compare_architectures.py
│   └── utils/
│       ├── data_loading.py
│       └── metrics.py
├── results/
│   ├── training_logs/
│   ├── aq_analysis/
│   └── comparisons/
└── README.md
```

---

## Dependencies

```
torch>=2.0
transformers>=4.30
datasets
wandb (optional, for logging)
numpy
scipy
matplotlib
```

---

## Notes

This experiment directly tests AKIRA's core architectural claim: that explicit spectral band structure produces qualitatively different learning dynamics and feature organization compared to standard transformers.

The key insight is that GPT-2's Layer 5 low-frequency concentration is **emergent** - the model discovered it through gradient descent without explicit guidance. AKIRA proposes that **designing** for this structure should produce stronger, more reliable, and more interpretable universal features.

If this experiment succeeds, it provides evidence that:
1. Spectral organization is not just an observation but a design principle
2. Differential learning rates across frequency scales improve feature stability
3. The "pump cycle" (tension/collapse) might emerge naturally from band interactions

If this experiment fails, it suggests:
1. The implicit organization in standard transformers is sufficient
2. Explicit band structure may add overhead without benefit
3. AKIRA's architectural claims need revision

Either outcome advances our understanding.

---

*"The difference between emergence and design is the difference between discovering fire and building a furnace. Both give heat, but only one gives control."*

---
*Oscar Goldman, Shogu Research Group @ Datamutant.ai subsidiary of 温心重工業*

---
## References

- AKIRA Framework Documentation (this repository)
- Experiment 000: Action Quanta Extraction and Validation
- Williams and Beer (2010) - Partial Information Decomposition
- Shuyang (2025) - Universal Neurons in LLMs
