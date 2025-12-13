
for experiments 1 and 2:

We invented a frequency for each token (common words = high freq), then watched how attention would combine those fake waves. This is a visualization tool, not a measurement of what GPT-2 actually does internally. It's like asking "if tokens were musical notes, how would attention blend them?" Useful for intuition, not proof.

What we learned:

The synthetic wave experiment is useful for intuition about how attention would modulate a hypothetical wave basis. It shows that attention does concentrate on fewer tokens over depth (coherence rises). Valid as visualization, not as measurement.

however it is interesting to note the wave coherence is an analogy that works computationally BUT is not actually what is happening, it's an ANALOGY of what is happenning, which might be even more of a brain twister, because it actually works LMAO
so yet, think about that, it's twisting my melon :D


=========================================================================
=========================================================================


experiment 3

The flaw in v3
When we FFT a hidden state vector, we're treating the 768 embedding dimensions as if they were time samples. They're not. The embedding dimension is an abstract feature space, not a temporal axis. There's no reason the model's representations should have "frequencies" across dimension indices.
So when we find:
Spectral concentration ≈ 0.01 (flat spectrum)
Interference < 1 (cancellation)
This doesn't mean "the model lacks phase coherence." It means FFT isn't the right tool for this data structure. We're measuring noise from a category error.

we should be using spectral realtime attention for signals, we tested the token version for the LOLs

but V3 needs to be applied to time signal spectral attention NOT standard LLMs. different process entirely.

The activation FFT experiment is conceptually flawed. Applying signal processing to embedding dimensions doesn't test the phase theory because embeddings aren't time series.

========================================================================
========================================================================



The code_004 attention entropy results are actually the most valid: they directly measure attention sharpening and output confidence, which is what "belief collapse" means operationally.

The phase/wave metaphor is powerful for intuition but hard to test literally. 

What we can test:
Attention concentration (code_004 does this well)
Output entropy drop through layers (code_004)
Collapse layer correlating with prompt type (code_004 showed this)
These are the solid findings

The project's core insight (belief crystallization through attention) is supported by the simpler code_004 metrics, not by the FFT approach which should be part of spectral attention code.


NOTE part of code4 isn't valid, some is good but this part explained is not

## The phase problem in code_004

Our current "phase" (004) is:
```python
# Maps POSITION to angle (arbitrary!)
angles = 2π × position / seq_len
phase = atan2(Σ attn × sin(angle), Σ attn × cos(angle))
```

This measures **where attention points** (position-weighted circular mean), NOT true oscillatory phase. It's a proxy, not physics.

========================================================

**Better phase approaches:** this exp
1. **FFT phase**: Extract phase from frequency components of activations
2. **Hilbert transform**: Instantaneous phase of activation signals  
3. **Wave embedding phase**: If tokens ARE waves, phase is built-in

===================================================

note on expt 004:


code_004 does NOT use FFT on embeddings
Looking at the code, attention_to_phase does something completely different:
# Map positions to angles on unit circlepositions = torch.arange(seq_len, dtype=torch.float32, device=attn_weights.device)angles = 2 * math.pi * positions / seq_len  # [0, 2pi)# Complex representation: each position is a point on unit circlereal_part = (attn_weights * torch.cos(angles)).sum(dim=-1)imag_part = (attn_weights * torch.sin(angles)).sum(dim=-1)# Coherence is the magnitudecoherence = torch.sqrt(real_part**2 + imag_part**2)
This is valid. It maps token positions (a meaningful ordered axis) to angles, then computes the attention-weighted centroid magnitude. This measures how concentrated the attention is.
What each metric actually measures
Metric	What it actually measures
Phase Coherence R	How concentrated is attention? (spread → 0, focused → 1)
Head Synchronization	Do heads attend to similar positions? (scattered → 0, aligned → 1)
"Phase"	Where in the sequence the attention points (angle of centroid)
Why the plots make sense
xxxx.png (Belief Collapse):
Layer 2 collapse: entropy drops fastest here
Coherence rises: attention becomes more focused after layer 2
Head sync rises: heads start agreeing on where to attend
nnnnnnnnnnn.png (Head phase trajectories):
Left: Individual heads' "pointing direction" through layers
Right: Phase dispersion peaks at layer 2-3 (maximum disagreement), then drops (consensus emerges)
The spike at layer 2-3 then convergence is the "decision being made"
bbbbbbbbbbbb.png:
Same pattern: entropy down, coherence up, sync up
Non-monotonic entropy (rises at final layer) matches what we saw in v2



===============================================================
The notebook is created at `code_005/wave_phase_dynamics.ipynb`. Here's what it addresses:

the measured coherence reflects how the attention pattern would modulate that synthetic wave basis, not what the model truly uses internally.
How this relates to spectral attention

Right now in exp5: attention weights modulate our synthetic waves (visual intuition only).

Spectral attention (what we ultimately want): operate in a real frequency basis of the model’s own signals—e.g., FFT/Hilbert of hidden states or a learned spectral projection—then weight/select bands. That would make frequency part of the computation, not just visualization.

Would we see superposition without the wave conversion?
Yes, attention is already a superposition of value vectors. You can plot the attention-weighted sum of hidden states or embeddings directly.

**However** without a wave basis you lose interpretable phase/frequency. You’d see vector sums, not constructive/destructive interference tied to “low vs high frequency” bands. The wave conversion gives us an interpretable axis; without it, you can still see norm changes and mixing, but not “which bands reinforced or cancelled.”

What the current superposition plots tell us:
They show how the attention pattern would combine token-associated waves: where it concentrates, coherence rises; where it spreads, coherence drops.
The interference metric trending toward ~1 across layers hints attention is balancing constructive/destructive mixing over depth.
Differences by prompt type (technical vs factual) reflect how the attention pattern interacts with the chosen frequency prior (common=high freq, rare=low freq), not the model’s internal spectral content.



What the superposition plots actually mean (deeper cut)
Synthetic lens, not internal signal: We map tokens to synthetic waves (common=high freq, rare=low) and let attention weights mix them. The plots show how the model’s attention pattern would modulate that chosen basis. They do not show true spectral content of GPT-2 activations.

Coherence rises when attention focuses: If a query head concentrates on a few tokens, the weighted wave sum aligns and coherence climbs. When attention spreads across many tokens, phases cancel and coherence drops. This is a direct readout of how sharp vs diffuse the attention is, given our frequency prior.

Interference → ~1 across depth: Interference >1 means constructive reinforcement; <1 means net cancellation. Watching it drift toward ~1 suggests that, layer by layer, attention patterns settle into a balance where gains and cancellations even out. Think of it as the model stabilizing its mixing strategy—less wild constructive/destructive swings by later layers.

Prompt-type differences reflect the prior, not hidden states:
Factual/narrative prompts (many common words → higher assigned freq) start and end with higher coherence.
Technical prompts (rarer words → lower assigned freq) show lower coherence and more dispersion.

This difference is about how the attention pattern interacts with the high/low-frequency tags we assigned, not about true spectral bands inside GPT-2.
Why this matters: The plots are useful for intuition about how focusing vs spreading attention would shape interference if tokens carried these frequencies. They give a way to see “selection” vs “diffusion” of signal energy in a wave metaphor, but they should not be mistaken for actual spectral dynamics of the network.


### The phase problem in code_004

```
code_004 phase = atan2(Σ attn × sin(2π × pos/N), Σ attn × cos(2π × pos/N))
```

This measures **WHERE attention points on a position circle**, not true oscillatory phase. It's a proxy that treats position as angle, which is arbitrary.

### What we need for reliable phase

**True wave representation**:
- Token → Zipf rank → Wave frequency
- Common words ("the", "is") → LOW frequency (0.1 Hz)
- Rare words ("quantum", "crystallization") → HIGH frequency (10 Hz)
- Phase emerges from actual wave interference

## What the new experiment does exp5

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│  EXPERIMENT 034-005: WAVE PHASE DYNAMICS                                   │
│                                                                             │
│  1. Zipf-Wave Encoding                                                     │
│     - Build frequency table from wikitext-2                                │
│     - Map token rank → wave frequency (log scale)                         │
│     - Generate complex wave signals with harmonics                        │
│                                                                             │
│  2. Wave Superposition Through Attention                                   │
│     - Attention weights = mixing coefficients                             │
│     - Superposed wave = Σ attn × wave                                     │
│     - Coherence R = |mean(normalized phasors)|                            │
│                                                                             │
│  3. Multiple Phase Measures                                                │
│     - FFT phase: dominant frequency components                            │
│     - Hilbert phase: instantaneous phase envelope                         │
│     - Wave coherence: direct from complex wave superposition             │
│                                                                             │
│  4. 16+ Prompts Across Categories                                         │
│     - Factual (constrained)                                               │
│     - Narrative (open-ended)                                              │
│     - Technical (rare vocabulary)                                         │
│     - Philosophical (abstract)                                            │
│                                                                             │
│  5. Many Visualizations                                                    │
│     - Individual token waves                                              │
│     - Attention patterns per layer                                        │
│     - Superposed waves per layer                                          │
│     - FFT spectra per layer                                               │
│     - Coherence profiles by prompt type                                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Interesting hypothesis to test

**Zipf-frequency × attention = spectral filtering**

```
Common words (low freq) = Carrier signal / background rhythm
Rare words (high freq) = Information-bearing modulation

Attention SELECTS which frequencies dominate the superposition.

Prediction:
- Factual prompts → narrow spectrum (few frequencies dominate)
- Philosophical prompts → broad spectrum (many frequencies compete)
```

This connects to the radar array parallel: attention is like a phased array antenna that steers the "beam" in frequency space.

Run the notebook to generate all visualizations in `code_005/figs_wave/`.