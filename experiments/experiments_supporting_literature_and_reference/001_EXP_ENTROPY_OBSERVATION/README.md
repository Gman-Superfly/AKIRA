# Experiment 001: Basic Entropy Observation – Supporting Literature

## Evidence for architecture experiment choices

Experiment 001 assumes that **attention distributions are meaningful belief states** and that **Shannon entropy over those distributions is the right scalar measure of uncertainty**.  
Work on belief-state geometry in transformers supports this assumption directly: in a controlled HMM setting, a small transformer’s residual stream linearly encodes the **Bayesian posterior over hidden states**, and the set of reachable belief states has a **structured geometry** rather than being arbitrary noise.

Under that view, EXP_001 is not just “debugging telemetry” but a necessary first step: if the model’s internal activations (attention or residual) really behave like belief distributions, then **being able to measure their entropy across space, time, and bands** is the minimal capability required before we can legitimately talk about **collapse events (EXP_002)** or **belief geometry (EXP_019)**. The papers below justify:

- treating attention-like vectors as *beliefs over latent states* rather than opaque scores, and  
- using standard information-theoretic entropy as the canonical observable of “how certain” those beliefs are.

## Primary References

### Shai, A. (2024). *Transformers Represent Belief State Geometry in their Residual Stream.* LessWrong.  
🔗 Online: [link](https://www.lesswrong.com/posts/gTZ2SxesbHckJ3CkF/transformers-represent-belief-state-geometry-in-their)

**Why this paper supports EXP_001**

- **Core result**: A transformer trained on sequences from a 3‑state Hidden Markov Model implicitly learns the **Bayesian belief state** \(p(H_t \mid O_{\le t})\); a linear probe on the residual stream can reconstruct the full distribution over hidden states.  
- This means that, at least in the toy setting, some internal vectors are *already normalized probability distributions over latent states*. Computing **Shannon entropy over those vectors** therefore has a clear probabilistic interpretation as belief uncertainty.  
- EXP_001’s design (hook attention, compute \(H = -\sum p_i \log p_i\), study variation over time/space/bands) is exactly the measurement one would choose if one expected transformers to be **approximate Bayesian trackers** of hidden state, as in this work.

**How it informs architecture choices**

- Justifies **treating attention weights (and related internal distributions) as belief states**, not arbitrary scores: they are plausible carriers of \(p(\text{latent} \mid \text{history})\).  
- Supports the choice of **Shannon entropy as the primary observable** for uncertainty, matching the way belief uncertainty is quantified in the HMM setting.  
- Motivates building a **first-class entropy-tracking subsystem** in AKIRA (as specified in EXP_001) so that later experiments can reason about collapse and geometry in terms of *belief* rather than ad‑hoc heuristics.

---

### Wentworth, J., & Lorell, D. (2024). *Why Would Belief-States Have A Fractal Structure, And Why Would That Matter For Interpretability? An Explainer.* LessWrong.  
🔗 Online: [link](https://www.lesswrong.com/posts/mBw7nc4ipdyeeEpWs/why-would-belief-states-have-a-fractal-structure-and-why)

**Why this paper supports EXP_001**

- Shows that a Bayesian agent tracking an HMM’s hidden state implements a **chaos game** on the belief simplex: each observation chooses an update map, and iterating this process produces a **fractal attractor of belief states**.  
- In that picture, belief states are not arbitrary points but elements of a **structured subset of the simplex**; nevertheless, **entropy remains the canonical scalar measure of uncertainty** over hidden state.  
- This underlines that before probing geometry (fractal structure, attractors) it is essential to be able to **measure and monitor entropy over belief states reliably**, which is exactly the goal of EXP_001.

**How it informs architecture choices**

- Encourages us to view AKIRA’s internal belief evolution as **trajectories on a structured manifold**; EXP_001 provides the scalar uncertainty signal (entropy) that later geometric analyses (EXP_019) will build on.  
- Supports the decision to **instrument entropy first**, even if we expect richer geometric structure later: the chaos‑game view says entropy and geometry are two aspects of the same underlying belief dynamics.

### Citation

- Shai, A. (2024). *Transformers Represent Belief State Geometry in their Residual Stream.* LessWrong.  
  Explains how transformers trained on HMM data linearly encode Bayesian belief states in their residual stream, motivating entropy as a meaningful belief-uncertainty observable.  
- Wentworth, J., & Lorell, D. (2024). *Why Would Belief-States Have A Fractal Structure, And Why Would That Matter For Interpretability? An Explainer.* LessWrong.  
  Derives fractal belief-state geometry from Bayesian updates in HMMs and motivates analyzing belief trajectories and uncertainty in terms of both entropy and geometry.


