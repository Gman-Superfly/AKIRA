# Experiment 019: Belief Geometry – Supporting Literature

## Evidence for architecture experiment choices

Experiment 019 takes seriously the claim that **uncertainty “has shape”**: error fields and prediction uncertainty are expected to form **structured geometries (crescents, wave packets, branches)** rather than uniform blobs. It interprets these shapes as **projections of an internal belief state** into observable space.  
Recent work on transformers trained on HMMs provides direct support for this framing: it shows that such models maintain **Bayesian belief states over hidden Markov states**, and that the set of reachable beliefs has a **non‑trivial geometric structure** (often fractal) in the simplex.

Combined, these results motivate EXP_019’s architecture choices:

- Treat the model’s internal state as a **belief vector** whose evolution traces out a **geometric object** (manifold/attractor), not a formless cloud.  
- Interpret **spatial error patterns** (e.g. crescents in the moving‑blob experiments) as **projections of that belief geometry** into pixel space.  
- Consider augmenting simple geometric fits (crescent vs blob vs streak) with **geometry‑aware analyses** (e.g. self‑similarity, clustering on the manifold), since theory predicts rich structure even in small HMMs.

## Primary References

### Shai, A. (2024). *Transformers Represent Belief State Geometry in their Residual Stream.* LessWrong.  
🔗 Online: [link](https://www.lesswrong.com/posts/gTZ2SxesbHckJ3CkF/transformers-represent-belief-state-geometry-in-their)

**Why this paper supports EXP_019**

- Shows that a transformer trained on a simple 3‑state HMM learns residual activations from which a **linear probe can reconstruct the full Bayesian belief state** over hidden states; the probe recovers the **mixed‑state presentation (MSP)** from Computational Mechanics.  
- Visualizations demonstrate that these belief states form a **structured subset of the simplex**, with **fractal‑like geometry** predicted by MSP theory, not an amorphous point cloud.  
- This is a direct analogue of the claim in SHAPE_OF_UNCERTAINTY.md: there, the **error field** for a moving blob is treated as the **belief distribution over futures projected to space**, and its **shape** (crescent width/orientation) is supposed to encode uncertainty about speed and direction.

**How it informs architecture choices**

- Justifies modeling AKIRA’s **error/uncertainty fields as geometric objects** tied to underlying belief, rather than only tracking scalar uncertainty.  
- Supports the decision to design EXP_019 around **systematic measurement of error shapes** (crescent vs streak vs blob) and their dependence on motion type, as this closely parallels how belief geometry is analyzed in the HMM setting.  
- Suggests a natural extension: once basic shapes are confirmed, apply **dimensionality reduction to internal states** and look for **linear structure corresponding to belief coordinates**, in the same spirit as the linear probes used here.

---

### Wentworth, J., & Lorell, D. (2024). *Why Would Belief-States Have A Fractal Structure, And Why Would That Matter For Interpretability? An Explainer.* LessWrong.  
🔗 Online: [link](https://www.lesswrong.com/posts/mBw7nc4ipdyeeEpWs/why-would-belief-states-have-a-fractal-structure-and-why)

**Why this paper supports EXP_019**

- Derives the **fractal geometry of belief states** for HMMs by viewing Bayesian updating as a **chaos game** on the belief simplex: each observation picks an affine map, and iterating these maps yields a **self‑similar attractor of beliefs**.  
- Emphasizes that this structure is **generic** for Bayesian trackers of hidden Markov processes, not an artifact of transformers per se, and argues that in larger nets we should look for **self‑similar activation sets** to infer which latent variables the model is tracking.  
- This directly underpins EXP_019’s view that **“belief geometry” is a real object**: if AKIRA’s dynamics are approximately Bayesian in a POMDP sense, then its internal belief should similarly live on a **structured manifold**, with observable error shapes as its shadows.

**How it informs architecture choices**

- Motivates the experiment’s focus on **systematic mapping from stimulus properties → error shapes → inferred uncertainty type**, since in the chaos‑game view the geometry encodes how beliefs over latent states respond to observations.  
- Suggests follow‑up analyses: beyond fitting crescents, we can **sample belief/error states across time** and look for **self‑similarity or low‑dimensional structure**, connecting EXP_019 to broader interpretability agendas around discovering latent variables via geometry.

### Citation

- Shai, A. (2024). *Transformers Represent Belief State Geometry in their Residual Stream.* LessWrong.  
  Provides direct evidence that transformers can linearly embed Bayesian belief geometry (including fractal structure), supporting the treatment of AKIRA’s uncertainty fields as projections of an internal belief manifold.  
- Wentworth, J., & Lorell, D. (2024). *Why Would Belief-States Have A Fractal Structure, And Why Would That Matter For Interpretability? An Explainer.* LessWrong.  
  Explains why Bayesian belief dynamics for HMMs generically yield fractal belief geometry and argues for using self‑similar activation sets to infer a model’s latent variables, aligning with EXP_019’s belief‑geometry agenda.


*Oscar Goldman, Shogu Research Group @ Datamutant.ai subsidiary of 温心重工業*