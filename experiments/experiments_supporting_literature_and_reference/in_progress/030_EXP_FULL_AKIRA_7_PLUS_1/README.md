# Experiment 030: Full AKIRA 7+1 Architecture (As Documented)

## Purpose

Test the **actual** AKIRA 7+1 architecture as specified in the theoretical documents, without the experimental additions (cross-batch history) that were tested in experiments 028-029.

## What This Tests

The REAL AKIRA architecture from `architecture_theoretical/`:
- **7 Spectral Bands**: Decompose input by frequency (DC through high-freq)
- **1 Temporal Band**: Causal attention over SEQUENCE (not cross-batch)
- **Spectral Wormhole**: Cross-band communication with **coherence/entropy-based gating**
- **Complementary Pairs**: (0-6), (1-5), (2-4), Bridge(3), Temporal(7)->all

## What This Does NOT Include

- Cross-batch history attention (Transformer-XL style) - this was Exp 028-029
- Differential temporal windows in wormhole - docs say coherence-gated
- BandHistoryAttention - not in the AKIRA spec

## Architecture Summary (From Docs)

```
INPUT
  |
  v
SPECTRAL DECOMPOSITION (FFT-inspired, 7 bands)
  |
  +---> Band 0 (DC)     ---> Spectral Attention (non-causal) ---> Geometric Processor
  +---> Band 1          ---> Spectral Attention (non-causal) ---> Geometric Processor  
  +---> Band 2          ---> Spectral Attention (non-causal) ---> Geometric Processor
  +---> Band 3 (Bridge) ---> Spectral Attention (non-causal) ---> Hybrid Processor
  +---> Band 4          ---> Spectral Attention (non-causal) ---> Hybrid Processor
  +---> Band 5          ---> Spectral Attention (non-causal) ---> Reactive Processor
  +---> Band 6          ---> Spectral Attention (non-causal) ---> Reactive Processor
  |
  v
SPECTRAL WORMHOLE (entropy/coherence-gated, sparse top-k)
  - Pair 0 <-> 6 (identity <-> position)
  - Pair 1 <-> 5 (shape <-> texture)
  - Pair 2 <-> 4 (structure <-> detail)
  - Bridge 3 -> all
  |
  v
TEMPORAL BAND (Band 7)
  - Aggregates from spectral bands
  - Causal attention over sequence
  - Provides temporal context back to reconstruction
  |
  v
RECONSTRUCTION + OUTPUT
```

## Key Differences from Exp 029

| Aspect | Exp 029 (v2/v3) | Exp 030 (This) |
|--------|-----------------|----------------|
| Wormhole | Differential temporal windows | Entropy/coherence gated |
| History | Cross-batch (Transformer-XL) | None (within sequence only) |
| Temporal Band | After wormhole | After wormhole, causal over sequence |
| Focus | Does history help? | Does the base AKIRA architecture work? |

## Ablations

1. **Baseline**: Standard GPT-2 (same as v2)
2. **AKIRA 7+1**: Full architecture as documented
3. **AKIRA No-Wormhole**: 7+1 without cross-band wormhole
4. **AKIRA No-Temporal**: 7 spectral only (no band 7)

## Theoretical References

- `architecture_theoretical/THE_SEVEN_PLUS_ONE_ARCHITECTURE.md`
- `architecture_theoretical/SPECTRAL_BELIEF_MACHINE.md`
- `architecture_theoretical/ORTHOGONALITY.md`

## Expected Outcome

If the AKIRA theory is correct:
- AKIRA 7+1 should outperform baseline
- Wormhole should provide additional benefit (cross-band helps)
- Temporal band should help sequence modeling (band 7 is needed)

If AKIRA 7+1 does NOT outperform baseline, the theory needs revision.

---

**Oscar Goldman - Shogu Research Group @ Datamutant.ai**
