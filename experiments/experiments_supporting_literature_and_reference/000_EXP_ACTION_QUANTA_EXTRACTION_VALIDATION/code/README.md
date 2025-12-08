# Experiment 000 Code: Action Quanta Extraction

AKIRA Project - Foundational Experiment

## Quick Start (Google Colab)

### Option 1: Fastest Setup

1. Go to [Google Colab](https://colab.research.google.com/)
2. Create new notebook
3. **Runtime > Change runtime type > T4 GPU**
4. In first cell, install dependencies:

```python
!pip install transformer-lens torch numpy scipy matplotlib tqdm --quiet
```

5. In second cell, copy entire contents of `000_action_quanta_extraction.py`
6. Run

### Option 2: Upload Script

1. Upload `000_action_quanta_extraction.py` to Colab
2. Run:

```python
!pip install transformer-lens torch numpy scipy matplotlib tqdm --quiet
%run 000_action_quanta_extraction.py
```

---

## Configuration

Edit the `Config` class to adjust:

```python
class Config:
    # Models (choose based on GPU memory)
    MODEL_A = "gpt2"           # 124M, ~500MB VRAM
    MODEL_B = "gpt2-medium"    # 355M, ~1.5GB VRAM
    
    # Or use Pythia (cleaner architecture)
    # MODEL_A = "EleutherAI/pythia-70m"
    # MODEL_B = "EleutherAI/pythia-160m"
    
    # Layers to analyze
    LAYERS_TO_ANALYZE = [3, 5, 7]
    
    # Threshold for Action Quanta detection
    AQ_THRESHOLD = 0.3
```

---

## Model Options by GPU Memory

| GPU | VRAM | Recommended Models |
|-----|------|-------------------|
| Colab T4 | 16GB | gpt2 + gpt2-medium, pythia-70m to pythia-410m |
| Colab A100 | 40GB | gpt2-large + gpt-j, pythia up to 1.4b |
| Local 8GB | 8GB | pythia-70m + pythia-160m only |

---

## Expected Output

```
EXPERIMENT 000: ACTION QUANTA EXTRACTION
AKIRA Project - Foundational Experiment
======================================================================

Model A: gpt2
Model B: gpt2-medium
...

ANALYZING LAYER 5
==================================================
  Computing correlations for 3072 neurons...
  
  RESULTS:
  - Total neurons: 3072
  - Action Quanta candidates: 412 (13.4%)
  - Mean excess correlation: 0.0823
  - Max excess correlation: 0.6721

VERDICT: STRONG evidence for universal Action Quanta
```

---

## Interpreting Results

| AQ % | Interpretation |
|------|----------------|
| > 20% | Very strong evidence - Action Quanta likely exist as discrete units |
| 10-20% | Strong evidence - AQ candidates worth investigating further |
| 5-10% | Moderate - may need lower threshold or different models |
| < 5% | Weak - AQ may be emergent (not discrete), see AKIRA framework notes |

---

## Next Steps After Running

If you find AQ candidates (> 10%):

1. **Export the neuron indices** for further analysis
2. **Run Phase 3** (Irreducibility) - can these AQ be decomposed?
3. **Run Phase 4** (Actionability) - are they load-bearing for tasks?
4. **Run Phase 5** (Spectral) - do they live in low-frequency bands?

---

## Troubleshooting

**Out of Memory (OOM):**
- Reduce `BATCH_SIZE` to 8 or 4
- Use smaller models (pythia-70m)
- Reduce `NUM_SAMPLES`

**Low AQ detection:**
- Try different model pairs (same family often shows more AQ transfer)
- Lower `AQ_THRESHOLD` to 0.2
- Increase `NUM_SAMPLES` for more stable estimates
- If still low: AQ may be emergent rather than discrete (valid AKIRA outcome)

**Import errors:**
- Make sure transformer-lens is installed: `!pip install transformer-lens`

---

## Advanced: Aligned Version

The basic version (`000_action_quanta_extraction.py`) compares neurons by index, which is naive.

The aligned version (`001_action_quanta_extraction_aligned.py`) uses proper methods:

### Three Alignment Methods

| Method | What it measures | Output |
|--------|------------------|--------|
| **CKA** | Global representation similarity | Score 0-1 (higher = more similar) |
| **Optimal Transport** | Best neuron-to-neuron matching | Matched pairs + correlations |
| **Procrustes** | Learned linear transformation | Aligned correlations + AQ |

### Key Improvements

1. **CKA** - Invariant to neuron ordering, measures if representations encode similar information
2. **Optimal Transport** - Finds which neuron in B best corresponds to each neuron in A
3. **Procrustes** - Learns rotation matrix, then computes correlations in aligned space

### Running the Aligned Version

```python
!pip install transformer-lens torch numpy scipy matplotlib tqdm pot --quiet
# Note: 'pot' is Python Optimal Transport library
```

Then copy `001_action_quanta_extraction_aligned.py` and run.

### Statistical Threshold

The aligned version computes thresholds from data:
- `"statistical"`: mean + 2*std (outliers)
- `"percentile"`: top N% of correlations
- `"fixed"`: arbitrary value (not recommended)

---

*Oscar Goldman - Shogu Research Group @ Datamutant.ai*
