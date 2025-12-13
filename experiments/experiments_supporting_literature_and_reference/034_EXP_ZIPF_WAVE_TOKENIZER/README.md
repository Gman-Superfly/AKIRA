# Experiment 034: Zipf-Grounded Wave Tokenizer

## Hypothesis

If token wave frequencies are assigned based on Zipf's Law (usage frequency), then spectral decomposition of token sequences becomes **information-theoretically grounded** rather than arbitrary.

## Motivation

Experiment 031 failed because wave parameters (frequency, phase, amplitude) were learned from scratch with no inductive bias. There was no natural "wave signature" for tokens - the model had to discover structure that may not exist.

**Key insight**: Natural language already HAS frequency structure - Zipf's Law. The token usage distribution IS a power spectrum.

### The Zipf-Information Connection

| Token Type | Usage Frequency | Information Content | Wave Frequency |
|------------|-----------------|---------------------|----------------|
| "the", "a", "is" | Very High | Low (structural) | LOW (DC-like) |
| Common content | High | Medium | LOW-MID |
| Domain words | Medium | Medium-High | MID |
| Technical/rare | Low | High (specific) | HIGH |

This mapping is not arbitrary:
1. **Shannon entropy**: Common words carry fewer bits of information
2. **Zipf's Law**: Word frequency follows power law f^(-alpha)
3. **Spectral analogy**: Power spectra also follow power laws
4. **Band separation**: Information density naturally maps to spectral bands

## Theory

### Zipf's Law

Word frequency in natural language follows:
```
f(r) ~ r^(-alpha)
```
where r is the rank (1 = most common) and alpha ~ 1.0 for most languages.

### Mapping to Wave Frequency

```python
# Token rank from corpus statistics
rank = zipf_rank[token_id]  # 1 = most common, V = rarest

# Map to wave frequency (log scale for power law)
wave_freq = log(rank) / log(vocab_size)  # Normalized to [0, 1]

# Result:
# - "the" (rank ~1) -> wave_freq ~ 0.0 (DC component)
# - rare token (rank ~50000) -> wave_freq ~ 1.0 (high frequency)
```

### Why This Works

1. **Grounded in data**: Frequencies come from actual usage statistics, not learned
2. **Information-theoretic**: Aligns with Shannon entropy (common = low info = low freq)
3. **Natural band structure**: Zipf distribution creates natural spectral bands
4. **Phase encodes context**: Same token at different positions gets different phase

## Architecture

### ZipfWaveTokenizer

```
Input: token_ids [B, T]
    |
    v
+-------------------+
| Zipf Rank Lookup  |  <- Pre-computed from corpus
+-------------------+
    |
    v
+-------------------+
| Rank -> Frequency |  <- f = log(rank) / log(V)
+-------------------+
    |
    v
+-------------------+
| Wave Generation   |  <- sin(2*pi*f*t + phase) * amplitude
+-------------------+
    |
    v
Output: wave_embeddings [B, T, D]
```

### Components

1. **Zipf Rank Table**: Pre-computed token frequency ranks from corpus
2. **Frequency Mapper**: Converts rank to wave frequency (log scale)
3. **Phase Generator**: Learnable per-token phase (or derived from position)
4. **Amplitude**: Can be fixed (1.0) or learnable per-token
5. **Multi-frequency**: Each token can have multiple harmonic components

## Experiment Design

### Phase 1: Tokenizer Validation

Test that the tokenizer produces meaningful spectral structure:

1. **Band Distribution**: Verify tokens distribute across frequency bands according to Zipf
2. **Sentence Spectra**: Analyze spectral content of real sentences
3. **Reconstruction**: Can we recover tokens from wave representation?

### Phase 2: Language Model Integration

Test the tokenizer in actual LM training:

1. **Baseline**: Standard embedding + transformer
2. **ZipfWave + Linear**: Zipf wave embedding + standard transformer + linear decode
3. **ZipfWave + FFT**: Zipf wave embedding + FFT decomposition + band attention
4. **ZipfWave + Symmetric**: Zipf wave embedding + wave-based decode

### Ablations

1. **Frequency source**: Zipf rank vs random vs learned
2. **Phase**: Fixed vs learnable vs position-derived
3. **Harmonics**: Single frequency vs multiple harmonics per token
4. **Band boundaries**: Natural Zipf bands vs uniform split

## Expected Outcomes

**If Zipf-grounded frequencies help:**
- FFT decomposition should cleanly separate information types
- Low-frequency bands capture syntax/structure
- High-frequency bands capture specific content
- Coherence patterns should be meaningful

**If it doesn't help:**
- Discrete tokens may not map naturally to continuous waves
- Information structure may require different representation
- Learned embeddings may already capture this implicitly

## Plan for real-time Zipf wave tokenization

Goals:
- Ground token waves in Zipf ranks, keep streaming-friendly generation, and make collapse/observability measurable in-band.
- Verify high-frequency to low-frequency information contraction at collapse, or record that discrete tokens already carry the needed structure.

Data and mapping:
- Compute Zipf ranks from the deployed tokenizer vocab; refresh when the vocab shifts. Store rank→frequency as log(rank)/log(V), clamp to [0, 1].
- Default amplitude = 1.0; allow optional learned per-token amplitude only if it improves reconstruction without destabilizing streaming.
- Harmonics: start with fundamental plus first harmonic; keep the count small for real-time cost.
- Phase: derive from position for determinism; allow a small learned phase offset per token to capture context-specific shifts.

Streaming wave generation:
- Maintain a per-sequence phase accumulator so incremental tokens reuse phase history instead of recomputing prefixes.
- Generate waves on the fly: sin(2π f t + φ) with t as absolute position or wall-clock step for live streams.
- Cache waveforms for the most frequent ranks to cut latency; fall back to on-the-fly compute for tail tokens.

Banding and collapse checkpoints:
- Use the 7 Zipf-aligned spectral bands plus temporal band; track band energies per step (FFT or fixed filterbank).
- Track entropy per band and cross-band coherence before and after attention; expect synergy (distributed, high + low) pre-collapse and redundancy (low-dominant) post-collapse.
- Observe high→low band energy shift at collapse; if absent, note that the wave map adds little beyond discrete tokens.

Attention-of-attention probes (cheap, online):
- Attention entropy maps: high entropy marks competing leaders; log per head and layer.
- Temperature sweep on attention logits (τ in {0.1, 1, 10}); divergence shows fragility of routing.
- Wormhole threshold sweep: count connections versus threshold; steep rises flag latent links just below gate.
- Gradients on attention logits (if training): large magnitude means routing would change most there.
- Cross-head similarity: cluster heads to see alternative hypotheses versus consensus.

Reconstruction path (waves → tokens):
- Demodulate per position: recover frequency peak → candidate ranks; combine with phase to disambiguate collisions.
- Decode with a small beam over rank candidates; compare argmax vs beam vs hybrid (rank prior × phase match).
- Metrics: exact token match, rank error (Zipf distance), phase error; stratify by band.

Experiment steps (real-time focus):
1) Build Zipf rank table and frequency mapper; add streaming phase accumulator and harmonic support.
2) Instrument band energy, attention entropy, temperature and threshold sweeps; log high→low band energy shifts at collapse.
3) Validate the tokenizer: band distribution, sentence spectra, reconstruction accuracy (full and streaming), sensitivity to phase shifts.
4) Integrate in real-time LM path: ZipfWave + linear decode and ZipfWave + FFT band attention; measure latency and throughput.
5) Compare against the discrete embedding baseline: look for clearer band separation, more stable collapse (entropy drop, coherence rise), and better reconstruction in streaming.
6) If high→low contraction or coherence spike is weak, document that tokens already encode the needed structure and limit the wave path to observability-only use.

## Connection to AKIRA

This experiment tests whether AKIRA's spectral framework can be grounded in information theory:

| AKIRA Concept | Zipf Wave Implementation |
|---------------|--------------------------|
| Band 0 (DC) | Most common tokens (the, a, is) |
| Band 1-2 | Common content words |
| Band 3 (bridge) | Transition vocabulary |
| Band 4-5 | Domain-specific terms |
| Band 6 (HF) | Rare/technical tokens |
| Coherence | Phase alignment of related tokens |

## Files

- `code/zipf_wave_tokenizer.py` - Core tokenizer implementation
- `code/test_tokenizer.py` - Validation and visualization
- `code/colab_wave_lm.py` - Language model experiment (Colab)

## References

- Zipf, G.K. (1949). Human Behavior and the Principle of Least Effort
- Shannon, C.E. (1948). A Mathematical Theory of Communication
- Li et al. (2024). Large Language Models for Limited Noisy Data (arxiv:2512.04031)
  - Key insight: tokenizing time-frequency representations preserves discriminative structure

## Status

- [ ] Zipf rank computation from GPT-2 tokenizer
- [ ] Wave tokenizer implementation
- [ ] Tokenizer validation tests
- [ ] Language model experiments
- [ ] Analysis of spectral patterns

*Oscar Goldman, Shogu Research Group @ Datamutant.ai subsidiary of 温心重工業*

v3 notebook added at code_005/wave_phase_v3_nb.ipynb.

What’s inside
Activation spectra: FFT/Hilbert helpers on head-specific contexts (using attention weights over each layer’s input hidden states).
Metrics: head phase coherence, head interference, spectral concentration of mean context.

Prompts upgraded: quantum, legal indemnity, riddle, coding (GCD), ambiguous “square circle”.
Plots saved under figs_wave_v3/:
v3_activation_metrics.png — layer curves (coherence, spectral concentration, interference, entropy vs final coherence).
v3_quantum_head_spectra.png — per-head FFT magnitude heatmaps at layers 0/5/11 for the quantum prompt, with per-layer coherence/interference bars.
v3_wave_superposition_legal.png — synthetic inverted-Zipf wave superposition for the legal prompt (kept for intuition contrast).
Notes on methodology

Head contexts are approximated by attention-weighted sums over the layer input hidden states (ignores value projections but preserves attention structure).
Spectral concentration now comes from activations, not synthetic waves; synthetic wave plots remain only for visual intuition.