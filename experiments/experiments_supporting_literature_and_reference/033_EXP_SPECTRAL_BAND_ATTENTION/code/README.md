# Experiment 033: Spectral Band Attention

A working reference implementation of the three-attention-type architecture for 2D spatiotemporal prediction.

## Quick Start

```bash
cd AKIRA/experiments/experiments_supporting_literature_and_reference/033_EXP_SPECTRAL_BAND_ATTENTION/code

# Basic run (blob pattern, fast config)
python synthetic_2d.py --config fast --steps 300

# With visualization output
python synthetic_2d.py --config fast --steps 500 --viz-interval 5 --viz-dir viz_out

# Different patterns
python synthetic_2d.py --pattern-train interference --steps 500
python synthetic_2d.py --pattern-train switching --steps 500 --switch-period 50

# Log attention statistics
python synthetic_2d.py --steps 300 --log-stats-csv ../results/attention_stats.csv

# Larger grid
python synthetic_2d.py --config large --grid-size 64 --steps 500
```

## Architecture

Three complementary attention mechanisms:

1. **Temporal Attention** (`temporal_attention.py`)
   - Per-position self-history attention
   - Top-K selection with exponential decay
   - Purpose: Object permanence

2. **Neighbor Attention** (`neighbor_attention.py`)
   - 8-connected local spatial attention
   - Temporal window support
   - Purpose: Local physics (diffusion, propagation)

3. **Wormhole Attention** (`wormhole_attention.py`)
   - Sparse similarity-gated non-local connections
   - Separate bands for similarity vs value retrieval
   - Purpose: Global resonance patterns

## Files

| File | Description |
|------|-------------|
| `synthetic_2d.py` | Main experiment script |
| `spectral_attention.py` | Band attention coordinator |
| `temporal_attention.py` | Per-position temporal attention |
| `neighbor_attention.py` | Local 8-connected attention |
| `wormhole_attention.py` | Sparse non-local attention |
| `configs.py` | Configuration presets |

## Configuration Presets

| Config | baseDim | attnDim | timeDepth | Speed |
|--------|---------|---------|-----------|-------|
| `turbo` | 16 | 16 | 4 | Fastest |
| `fast` | 32 | 32 | 8 | Quick |
| `default` | 64 | 64 | 16 | Balanced |
| `large` | 32 | 32 | 8 | For 64x64 |
| `xlarge` | 32 | 32 | 8 | For 128x128 |

## Available Patterns

- `blob` - Moving Gaussian blob (circular motion)
- `interference` - Wave interference pattern
- `switching` - Alternates blob/interference
- `double_slit` - Double-slit diffraction
- `counter_rotate` - Two counter-rotating blobs
- `chirp` - Frequency sweep
- `phase_jump` - Abrupt phase changes
- `noisy_motion` - Blob with trajectory noise
- `bifurcation` - Blob that occasionally splits
- `wave_collision` - Standing wave nodes

## Key Insights

1. **Per-position temporal attention** is crucial - each pixel tracks its own history
2. **Explicit neighbor attention** captures local physics CNNs learn implicitly
3. **Wormhole with similarity gating** finds resonant patterns efficiently
4. **Cross-band routing** allows structural matching (low-freq) while retrieving intensity values

## Dependencies

- PyTorch
- tqdm
- matplotlib (optional, for visualization)

## Output

The experiment outputs:
- Training loss trajectory
- Held-out evaluation metrics
- Optional: per-step attention statistics CSV
- Optional: visualization snapshots

## Example Output

```
Using device: cpu
Config: fast
Grid size: 32
Time depth: 8

Training for 300 steps...
  Step 0: Loss = 0.142857
  Step 50: Loss = 0.023456
  ...

RESULTS:
  Total time: 45.23s
  Steps/sec: 6.6
  Final loss: 0.001234
  Min loss: 0.000987
  Trend: IMPROVING (+87.3%)

HELD-OUT EVALUATION:
  Eval mean loss: 0.001567
```
