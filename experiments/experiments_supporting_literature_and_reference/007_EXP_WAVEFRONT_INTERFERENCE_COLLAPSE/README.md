# Experiment 007: Wavefront Error Propagation, Interference, and Collapse – Supporting Literature

## Evidence for architecture experiment choices

Experiment 007 models AKIRA’s **error wavefront in embedding space** as behaving like **lightning**: uncertainty spreads, branches into multiple hypotheses, and then one branch “wins” in a sudden collapse.  
The belief‑geometry results on HMMs give a clean Bayesian underpinning for this picture: they show that when a system maintains a **distribution over hidden states** and updates that distribution with each observation, the trajectory of beliefs behaves like a **chaos game** which:

- explores a **structured (often fractal) subset** of the belief simplex, and  
- exhibits **branching and consolidation** as different observation‑conditioned maps are applied.

Mapping that story into AKIRA’s setting suggests that:

- the **wavefront of error** we see in embedding/output space should be interpreted as the **projection of a branching belief process**, and  
- the **collapse moments** studied in EXP_007 should correspond to belief trajectories snapping into **low‑entropy, single‑hypothesis regions** of that process—just as in the HMM case.

## Primary References

### Shai, A. (2024). *Transformers Represent Belief State Geometry in their Residual Stream.* LessWrong.  
🔗 Online: [link](https://www.lesswrong.com/posts/gTZ2SxesbHckJ3CkF/transformers-represent-belief-state-geometry-in-their)

**Why this paper supports EXP_007**

- Shows that a transformer trained on an HMM learns an internal representation in which a **linear probe can recover the full Bayesian belief state** over hidden Markov states. The residual stream is, in effect, a linear image of the **belief simplex and its geometry**.  
- When such a model is run on real sequences, the **path of residual activations** is therefore a path of **belief states**; in the HMM experiments, that path traces out the same fractal geometry predicted by Computational Mechanics.  
- EXP_007’s decision to treat the **error wavefront and embedding trajectories** as a physical “medium” through which hypotheses propagate and interfere is directly aligned with this view: the wavefront is the observable footprint of a **belief trajectory** inside a structured manifold.

**How it informs architecture choices**

- Supports designing visualizations which track **branching structures in embedding space** (clusters/branches of error or belief) rather than only scalar metrics.  
- Justifies interpreting sudden consolidation of a branch (and disappearance of alternatives) as a **genuine belief collapse**, not an arbitrary artifact, because similar behavior is seen when transformers are forced to track explicit HMM belief states.

---

### Wentworth, J., & Lorell, D. (2024). *Why Would Belief-States Have A Fractal Structure, And Why Would That Matter For Interpretability? An Explainer.* LessWrong.  
🔗 Online: [link](https://www.lesswrong.com/posts/mBw7nc4ipdyeeEpWs/why-would-belief-states-have-a-fractal-structure-and-why)

**Why this paper supports EXP_007**

- Recasts the Bayesian belief update for an HMM as a **chaos game**: at each timestep, the observation picks one of several affine maps on the belief simplex, and iterating this procedure generates a **fractal belief attractor**.  
- In that picture, the belief trajectory naturally exhibits **branching exploration of possibilities** (as different update maps are sampled) and **occasional entries into narrow regions** corresponding to strong commitment to a particular hidden state—exactly the “lightning” pattern EXP_007 aims to observe.  
- The authors explicitly point out that, in more realistic networks, we should expect to find **self‑similar sets of activations** corresponding to distributions over latent variables, and that understanding these sets and their symmetries is central for interpretability. EXP_007 is an instance of this agenda focused on **temporal prediction and collapse**.

**How it informs architecture choices**

- Provides a theoretical reason to expect **branching wavefronts** (multiple hypotheses) and **sudden consolidation** (collapse) when visualizing AKIRA’s error/belief trajectories over time.  
- Motivates analyzing not just whether collapse happens but **where in the manifold** it occurs and how branches relate, as this encodes the **latent variables and update rules** the model is implicitly using.

### Citation

- Shai, A. (2024). *Transformers Represent Belief State Geometry in their Residual Stream.* LessWrong.  
  Demonstrates that transformers trained on HMMs linearly encode belief states and their geometry, backing the interpretation of wavefront dynamics as belief trajectories.  
- Wentworth, J., & Lorell, D. (2024). *Why Would Belief-States Have A Fractal Structure, And Why Would That Matter For Interpretability? An Explainer.* LessWrong.  
  Explains how Bayesian belief updates implement a chaos game with fractal belief attractors, providing a concrete model for branching and collapse in belief dynamics.


