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