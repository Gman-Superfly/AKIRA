# Experiment 002: Collapse Event Detection – Supporting Literature

## Evidence for architecture experiment choices

Experiment 002 assumes that **“collapse events” are real belief transitions**, not just artifacts of noisy optimization, and that they should appear as **sudden drops in the entropy of an internal belief distribution**.  
Results on transformers trained on Hidden Markov Models support this by showing that such models maintain **Bayesian belief states over hidden variables**, and that the evolution of those beliefs has a **structured, sometimes fractal, geometry** driven by observation-conditioned updates.

In that setting, a “collapse” is a moment when the belief distribution moves from being spread across multiple hypotheses to being sharply concentrated near one latent state. That is exactly the situation where **entropy should drop abruptly**, and where we expect a **correlated improvement in prediction quality**. The references below therefore justify:

- using **attention/residual-based belief distributions** as the object whose entropy we monitor, and  
- interpreting **large negative spikes in \(dH/dt\)** as evidence of genuine belief commitment events, not just smooth gradient descent.

## Primary References

### Shai, A. (2024). *Transformers Represent Belief State Geometry in their Residual Stream.* LessWrong.  
🔗 Online: [link](https://www.lesswrong.com/posts/gTZ2SxesbHckJ3CkF/transformers-represent-belief-state-geometry-in-their)

**Why this paper supports EXP_002**

- Demonstrates that a transformer trained on HMM‑generated sequences linearly encodes the **Bayesian posterior over hidden states** in its residual stream. This means there is a well‑defined underlying **belief distribution \(p(H_t \mid O_{\le t})\)** whose entropy we can, in principle, track.  
- Because the residual stream is (approximately) an affine image of the belief simplex, **abrupt belief updates** (e.g. when evidence rules out many states at once) correspond to **sharp movements in residual space** and to **sudden entropy drops** in the underlying distribution.  
- This supports EXP_002’s design choice to define collapse in terms of **entropy rate \(dH/dt\)** and to treat detected events as **real changes in inferred hidden state**, rather than arbitrary thresholds on raw activations.

**How it informs architecture choices**

- Justifies building a **collapse detector around entropy dynamics** (as in `collapse_detector.py`) instead of ad‑hoc heuristics on logits or hidden norms.  
- Encourages aligning AKIRA’s notion of collapse with **Bayesian commitment** over latent variables: a move from a multi‑modal belief to a concentrated one, which should manifest both in **entropy** and in **prediction error reduction**.

---

### Wentworth, J., & Lorell, D. (2024). *Why Would Belief-States Have A Fractal Structure, And Why Would That Matter For Interpretability? An Explainer.* LessWrong.  
🔗 Online: [link](https://www.lesswrong.com/posts/mBw7nc4ipdyeeEpWs/why-would-belief-states-have-a-fractal-structure-and-why)

**Why this paper supports EXP_002**

- Shows that Bayesian belief updates for an HMM can be written as a **chaos game on the belief simplex**, where each observation selects a particular update map. The set of belief states visited over time forms a **self‑similar (often fractal) subset** of the simplex.  
- Within that chaos‑game picture, “collapse” corresponds to the belief trajectory entering a **small, low‑entropy region** of this attractor (near one hidden state) after wandering through more diffuse areas; such entries are naturally **event‑like** and **sparse** in time.  
- This gives a concrete model in which **entropy spikes, sparsity of events, and correlation with prediction improvement**—all central metrics in EXP_002—are the expected signatures of commitment to a particular latent hypothesis.

**How it informs architecture choices**

- Motivates treating collapse events as **change‑points in an underlying belief process** with structured dynamics, which is exactly what our `detect_collapse_events` and suddenness metrics are designed to detect.  
- Suggests that, beyond scalar statistics, later work could compare **where in belief/activation space** collapses occur (e.g. which regions of the attractor) to better understand **what kind of latent ambiguity is being resolved**, but EXP_002 is correctly focused first on **event detection via entropy**.

### Citation

- Shai, A. (2024). *Transformers Represent Belief State Geometry in their Residual Stream.* LessWrong.  
  Shows that transformers trained on HMMs linearly encode Bayesian belief states, supporting entropy‑based collapse detection as tracking real belief commitments.  
- Wentworth, J., & Lorell, D. (2024). *Why Would Belief-States Have A Fractal Structure, And Why Would That Matter For Interpretability? An Explainer.* LessWrong.  
  Derives chaos‑game, fractal belief dynamics in HMMs and motivates viewing collapse as sparse, low‑entropy entry events in a structured belief attractor.


*Oscar Goldman — Shogu Research Group @ Datamutant.ai subsidiary of 温心重工業*