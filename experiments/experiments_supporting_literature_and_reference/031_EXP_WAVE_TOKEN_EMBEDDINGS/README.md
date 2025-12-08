# Experiment 031: Wave-Based Token Embeddings

## Hypothesis

If tokens are represented as waves (frequency + phase + amplitude) rather than discrete embeddings, 
AKIRA's spectral decomposition becomes **natural** rather than **learned approximation**.

## Motivation

Current AKIRA implementation:
- Tokens are discrete embeddings (lookup table)
- "Spectral bands" are learned convolutions trying to approximate frequency separation
- The 7 bands are artificial constructs

With wave-based tokens:
- Each token IS a superposition of waves
- Band separation becomes TRUE frequency filtering (FFT/bandpass)
- Coherence gating becomes meaningful (phase alignment between signals)
- Magnitude-phase orthogonality becomes REAL

## The Core Idea

Instead of:
```
token_id → embedding_table[token_id] → [D] vector
```

We do:
```
token_id → (frequencies, phases, amplitudes) → wave at position t
         → sin(2π * f * t + φ) * A for each frequency component
```

This transforms discrete tokens into continuous wave representations that can be 
properly analyzed by spectral methods.

## Why This Matters for AKIRA

| AKIRA Concept | Current Implementation | Wave-Based |
|---------------|----------------------|------------|
| Spectral bands | Learned convolutions | True FFT bands |
| Coherence gating | Abstract similarity | Phase alignment |
| Magnitude-Phase | Implicit in embeddings | Explicit components |
| "What" vs "When" | Learned separation | Frequency vs Time |
| Band 0-6 | Arbitrary splits | True frequency ranges |

## Connection to Theory

From `ORTHOGONALITY.md`:
- **Magnitude-Phase Orthogonality**: Wave representation makes this explicit
- **Space-Time (Heisenberg)**: Frequency-time uncertainty is natural property of waves

From `SPECTRAL_BELIEF_MACHINE.md`:
- Spectral decomposition is a first-class operation
- Coherence-gated wormhole needs meaningful coherence measure

## Experiment Design

### Test 1: Wave Embedding vs Standard Embedding
- Same AKIRA architecture
- Compare: standard embedding lookup vs wave-based embedding
- Metric: Perplexity on WikiText-2

### Test 2: True FFT Decomposition
- Replace learned convolution decomposer with actual FFT
- Band 0 = lowest frequencies, Band 6 = highest frequencies
- See if true spectral decomposition helps

### Test 3: Coherence Analysis
- With wave tokens, measure actual phase coherence between bands
- See if coherence gating activates meaningfully

## Ablations

### Embedding Tests
1. **Baseline**: Standard GPT-2 (discrete embeddings)
2. **Wave Embed + Transformer**: Wave tokens + standard transformer (linear decode)

### Spectral Decomposition Tests
3. **Wave + AKIRA (Learned Bands)**: Wave tokens + learned convolution decomposition
4. **Wave + AKIRA (FFT Bands)**: Wave tokens + true FFT decomposition

### Symmetric Decode Tests (Key Question: Should decode match encode?)
5. **Wave Symmetric Encode/Decode**: Wave embed + match output to token wave signatures
6. **Wave + Frequency Decode**: Wave embed + match by spectrum similarity

The symmetric decode tests ask: if tokens are waves, should we decode by finding 
which token's wave signature best matches the output, rather than using a 
learned linear projection?

## Decode Approaches Explained

### Linear Decode (Asymmetric - Standard)
```
Encode: token_id → wave(freq, phase, amp)
Decode: hidden → Linear(vocab_size) → logits
```
The decode ignores wave structure entirely - just a learned linear projection.

### Symmetric Wave Decode
```
Encode: token_id → wave(freq, phase, amp) 
Decode: hidden → wave_space → compare to ALL token wave signatures → logits
```
Match output wave to find which token's wave it most resembles.
Uses cosine similarity between output wave and each token's reference wave.

### Frequency Spectrum Decode
```
Encode: token_id → wave(freq, phase, amp)
Decode: hidden → spectrum → compare to ALL token spectra → logits
```
Each token has a characteristic frequency spectrum (its amplitude pattern).
Match output spectrum to find the most similar token.

## Expected Outcomes

**If wave representation helps:**
- True FFT decomposition should outperform learned convolutions
- Coherence gating should show meaningful activation patterns
- Phase information should improve predictions

**If wave representation doesn't help:**
- Learned convolutions are sufficient approximation
- Discrete tokens may be fundamentally different from continuous signals
- AKIRA may need different architecture for discrete vs continuous domains

## Implications for Real-Time Systems

If wave tokens work for language:
- Direct application to audio (already waves)
- Direct application to sensor data (already continuous)
- Unified architecture for all modalities

If wave tokens don't work for language but AKIRA works for signals:
- AKIRA is a signal processing architecture
- Language models need different approach
- Domain-specific design is correct

## Files

- `code/colab_wave_tokens.py` - Main experiment notebook
- `results/` - Experiment outputs

## Status

- [ ] Wave embedding implementation
- [ ] FFT-based spectral decomposition
- [ ] Ablation experiments
- [ ] Analysis of coherence patterns
