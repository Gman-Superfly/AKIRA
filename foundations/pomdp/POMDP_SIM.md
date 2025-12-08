# POMDP-Style Dynamics in Spectral Attention

This document describes how the spectral attention predictor behaves like a **Partially Observable Markov Decision Process (POMDP)** combined with a relaxation-oscillator "pump" cycle. We walk through each phase step by step and explain precisely where partial observability arises.

---

## 1. Background: What Is a POMDP?

A POMDP models an agent that:

1. Cannot observe the true state of the environment directly.
2. Maintains a **belief state**—a probability distribution over possible true states.
3. Takes actions that update the belief based on noisy observations.
4. Receives rewards (or incurs costs) that depend on the hidden state.

Formally, a POMDP is defined by the tuple ⟨S, A, T, R, Ω, O, γ⟩:

| Symbol | Meaning |
|--------|---------|
| S | Set of hidden states |
| A | Set of actions |
| T(s'|s,a) | Transition probability |
| R(s,a) | Reward function |
| Ω | Set of observations |
| O(o|s',a) | Observation probability |
| γ | Discount factor |

The agent maintains a belief b(s) over states and updates it via Bayes' rule after each observation.

---

## 2. Mapping the Predictor to POMDP Components

In our spectral attention predictor:

| POMDP Component | Predictor Equivalent | Explanation |
|-----------------|---------------------|-------------|
| **Hidden state s** | Next frame `x_{t+1}` | The ground truth we're trying to predict—never observed until after we commit |
| **Observation o** | Current frame `x_t` + history buffer `[x_{t-T}, ..., x_{t-1}]` | What the model actually sees |
| **Belief b(s)** | The prediction `ŷ_t` | A point estimate representing the model's implicit distribution over futures |
| **Action a** | Weight update `θ ← θ - η∇L` | The model "acts" by adjusting parameters |
| **Reward R** | Negative MSE loss `-‖ŷ_t - x_{t+1}‖²` | Lower error = higher reward |
| **Transition T** | Physics of the animation | How the blob/pattern moves from frame to frame |
| **Observation function O** | Identity on past, zero on future | We see past frames perfectly but future frames not at all |

---

## 3. Where Is the Partial Observability?

This is the key question. The partial observability in our system is **temporal**:

### 3.1 The Temporal Horizon Creates Hidden State

```
Time axis:
  [t-T] [t-T+1] ... [t-1] [t]  |  [t+1] [t+2] ...
  ←——— OBSERVED ———→          |  ←——— HIDDEN ———→
       (history buffer)       |  (future frames)
                              ↑
                         prediction boundary
```

The model observes frames `[t-T, ..., t]` but must predict `t+1`, which is **hidden** until after the prediction is made and evaluated. This is fundamentally different from a fully observable MDP where the agent would see the true state before acting.

### 3.2 Multiple Futures Are Consistent with Observations

Given only past frames, **multiple future trajectories are plausible**:

- The blob could continue on its current path
- The blob could accelerate or decelerate
- The blob could split (bifurcation pattern)
- The blob could reverse (phase jump pattern)

The model cannot distinguish between these possibilities from the observation alone—it must **infer** which future is most likely. This inference is the belief update.

### 3.3 The Prediction Is a Belief State

Under MSE loss, the optimal prediction is the **expected value** (mean) of the belief distribution:

```
ŷ_t = E[x_{t+1} | x_{t-T:t}] = ∫ x_{t+1} · p(x_{t+1} | x_{t-T:t}) dx_{t+1}
```

When the belief is uncertain (multiple futures plausible), this mean sits in the **center of the uncertainty cloud**—not on any single trajectory. This is why:

1. The prediction appears to "lag" behind the true position
2. The error map shows where the belief is spread
3. High error = high uncertainty = the model is hedging

### 3.4 Observation Noise vs. State Uncertainty

In a classic POMDP, partial observability often comes from **noisy sensors**. In our system:

| Source of Uncertainty | Present? | Explanation |
|----------------------|----------|-------------|
| Sensor noise | No | We observe past frames perfectly (no noise added) |
| State uncertainty | **Yes** | The future is fundamentally unobserved |
| Process noise | Pattern-dependent | Some patterns (noisy_motion, bifurcation) have stochastic dynamics |

Our partial observability is "clean"—we see the past perfectly but the future not at all. This is actually a **harder** problem than noisy observations of the current state, because no amount of sensor improvement can reveal the future.

---

## 4. The Belief Update Cycle

### 4.1 Bayesian Interpretation

Each training step performs an implicit belief update:

```
Prior:      p(x_{t+1} | x_{t-T:t-1})     [before seeing frame t]
Likelihood: p(x_t | x_{t+1})             [how likely is this observation given each future?]
Posterior:  p(x_{t+1} | x_{t-T:t})       [after incorporating frame t]
```

The neural network approximates this posterior through gradient descent on the MSE loss. The weights encode the learned transition dynamics.

### 4.2 Why Gradient Descent ≈ Belief Update

Consider what happens when the model makes a prediction and observes the true frame:

1. **Prediction**: Model outputs ŷ_t (its belief about x_{t+1})
2. **Revelation**: True frame x_{t+1} is revealed
3. **Error signal**: L = ‖ŷ_t - x_{t+1}‖² measures how wrong the belief was
4. **Update**: ∇L points toward the true state; weights shift to make future predictions closer

This is analogous to Bayesian updating: the "likelihood" of the true frame under the model's belief was low (high error), so the belief shifts toward the truth.

---

## 5. The Pump Cycle: Tension → Discharge → Recovery

We observe a repeating pattern analogous to a mechanical pump or a relaxation oscillator:

| Phase        | Physical Analogy              | Predictor Behavior                                                                 |
|--------------|-------------------------------|------------------------------------------------------------------------------------|
| **Tension**  | Pressure builds in a chamber  | Uncertainty (error) accumulates in the high-error zone; belief spreads over many possible futures. |
| **Discharge**| Valve opens, fluid moves      | Pattern becomes unambiguous; prediction snaps to the true trajectory; error collapses locally.      |
| **Recovery** | Chamber refills               | Uncertainty front advances to the next ambiguous region; belief re-spreads.                         |

Each cycle corresponds to one "step" of learning: the model commits to a prediction, observes the true frame, updates weights, and the cycle restarts.

### 5.1 Information-Theoretic View

The pump cycle can be understood as information flow:

```
Tension:    H(x_{t+1} | x_{t-T:t}) is high    [many bits of uncertainty]
Discharge:  x_{t+1} revealed → H drops to 0   [uncertainty resolved]
Recovery:   H(x_{t+2} | x_{t-T:t+1}) rises    [new uncertainty about next frame]
```

The model is constantly "pumping" information from the future (via the loss signal) into its weights.

---

## 6. Step-by-Step Walkthrough

### Step 0: Initialization

- The model has no history; the belief is maximally uncertain.
- The history buffer is seeded with small noise so temporal attention has something to differentiate.
- Error is high everywhere because the prediction is essentially random.
- **POMDP state**: b(s) ≈ uniform over all possible frames

### Step 1: First Observation

- The generator produces frame `t=0`.
- The model projects the frame into intensity, low-freq, and high-freq bands.
- Temporal, neighbor, and wormhole attention compute weighted sums over the (nearly empty) history.
- The output projection produces a prediction for frame `t=1`.
- MSE loss is computed; gradients flow back; weights update.
- **POMDP state**: b(s) slightly sharpened around "continuation of frame 0"

### Step 2: Belief Update (Tension Phase)

- Frame `t=1` arrives; the model now has one real history entry.
- The prediction for `t=2` is slightly better, but uncertainty is still high.
- The error map shows a broad blob: the model hedges across many plausible continuations.
- This is the "pressure building" phase—loss gradient is large but diffuse.
- **POMDP state**: b(s) is a wide Gaussian centered on the mean trajectory

### Step 3: Pattern Recognition (Tension Continues)

- After several frames the temporal attention learns the phase and velocity of the pattern.
- The belief sharpens: the error blob shrinks and moves ahead of the prediction.
- Wormhole attention may start finding matches in the history, reinforcing the emerging hypothesis.
- Neighbor attention refines spatial edges.
- **POMDP state**: b(s) narrowing; variance decreasing

### Step 4: Commitment (Discharge)

- The pattern becomes unambiguous; one trajectory dominates the belief.
- The prediction snaps to the true position; error collapses to near zero locally.
- This is the "valve opening"—probability mass transfers from the uncertain manifold to a committed point.
- Temporal entropy drops; wormhole sparsity increases (fewer connections needed).
- **POMDP state**: b(s) ≈ δ(s - s_true), a delta function at the true state

### Step 5: Advance (Recovery)

- The next frame introduces new uncertainty (the pattern has moved).
- The error front advances to the region the model hasn't explained yet.
- Belief re-spreads; tension begins accumulating again.
- The cycle repeats from Step 2.
- **POMDP state**: b(s) spreads again as new future becomes relevant

---

## 2. The Pump Cycle: Tension → Discharge → Recovery

We observe a repeating pattern analogous to a mechanical pump or a relaxation oscillator:

| Phase        | Physical Analogy              | Predictor Behavior                                                                 |
|--------------|-------------------------------|------------------------------------------------------------------------------------|
| **Tension**  | Pressure builds in a chamber  | Uncertainty (error) accumulates in the high-error zone; belief spreads over many possible futures. |
| **Discharge**| Valve opens, fluid moves      | Pattern becomes unambiguous; prediction snaps to the true trajectory; error collapses locally.      |
| **Recovery** | Chamber refills               | Uncertainty front advances to the next ambiguous region; belief re-spreads.                         |

Each cycle corresponds to one "step" of learning: the model commits to a prediction, observes the true frame, updates weights, and the cycle restarts.

---

## 3. Step-by-Step Walkthrough

### Step 0: Initialization

- The model has no history; the belief is maximally uncertain.
- The history buffer is seeded with small noise so temporal attention has something to differentiate.
- Error is high everywhere because the prediction is essentially random.

### Step 1: First Observation

- The generator produces frame `t=0`.
- The model projects the frame into intensity, low-freq, and high-freq bands.
- Temporal, neighbor, and wormhole attention compute weighted sums over the (nearly empty) history.
- The output projection produces a prediction for frame `t=1`.
- MSE loss is computed; gradients flow back; weights update.

### Step 2: Belief Update (Tension Phase)

- Frame `t=1` arrives; the model now has one real history entry.
- The prediction for `t=2` is slightly better, but uncertainty is still high.
- The error map shows a broad blob: the model hedges across many plausible continuations.
- This is the "pressure building" phase—loss gradient is large but diffuse.

### Step 3: Pattern Recognition (Tension Continues)

- After several frames the temporal attention learns the phase and velocity of the pattern.
- The belief sharpens: the error blob shrinks and moves ahead of the prediction.
- Wormhole attention may start finding matches in the history, reinforcing the emerging hypothesis.
- Neighbor attention refines spatial edges.

### Step 4: Commitment (Discharge)

- The pattern becomes unambiguous; one trajectory dominates the belief.
- The prediction snaps to the true position; error collapses to near zero locally.
- This is the "valve opening"—probability mass transfers from the uncertain manifold to a committed point.
- Temporal entropy drops; wormhole sparsity increases (fewer connections needed).

### Step 5: Advance (Recovery)

- The next frame introduces new uncertainty (the pattern has moved).
- The error front advances to the region the model hasn't explained yet.
- Belief re-spreads; tension begins accumulating again.
- The cycle repeats from Step 2.

---

## 7. Why "Most Probable = Most Error"?

Under MSE loss the optimal prediction is the **mean** of the belief distribution. When multiple futures are plausible:

- The mean sits in the middle of the uncertainty cloud.
- That middle region has the **largest expected squared distance** to any single true outcome.
- Therefore the error map peaks exactly where the model places its probability mass.

This is why the high-error zone "moves faster" than the prediction: it marks the belief's center of mass, which leads the committed trajectory until discharge occurs.

### 7.1 Mathematical Derivation

Let the belief be p(x_{t+1} | history). The MSE-optimal prediction is:

```
ŷ* = argmin_ŷ E[(ŷ - x_{t+1})²] = E[x_{t+1}] = μ
```

The expected squared error at this optimal point is:

```
E[(μ - x_{t+1})²] = Var(x_{t+1}) = σ²
```

So **error equals variance**—the more uncertain the belief, the higher the error at the prediction point.

### 7.2 Why Error "Leads" the Prediction

Consider a blob moving right. The model's belief might be:

```
p(x_{t+1}) = 0.6 · δ(position = 10) + 0.4 · δ(position = 12)
```

The optimal prediction is the mean: position = 10.8

But the error at position 10.8 is high because:
- If truth is 10: error = 0.8²
- If truth is 12: error = 1.2²

The error map shows this uncertainty—it's brightest where the model is hedging between hypotheses.

---

## 8. The Role of Each Attention Mechanism

Each attention branch contributes differently to belief formation:

### 8.1 Temporal Attention: Memory of Dynamics

```
Temporal attention asks: "What happened at this position in recent frames?"
```

- Provides velocity/acceleration estimates
- Learns periodic patterns (phase tracking)
- Reduces uncertainty by exploiting temporal continuity
- **POMDP role**: Transition model T(s'|s) estimation

### 8.2 Neighbor Attention: Spatial Coherence

```
Neighbor attention asks: "What's happening at nearby positions?"
```

- Enforces spatial smoothness
- Propagates information from observed regions to uncertain regions
- Prevents fragmented predictions
- **POMDP role**: Observation model O(o|s) regularization

### 8.3 Wormhole Attention: Non-Local Pattern Matching

```
Wormhole attention asks: "Have I seen this situation before, anywhere in history?"
```

- Finds distant frames with similar features
- Enables "teleportation" of information across time
- Reduces partial observability by leveraging pattern recurrence
- **POMDP role**: Belief shortcut—if I've seen this before, I know what comes next

### 8.4 How Wormhole Reduces Partial Observability

This is crucial. Wormhole attention partially **defeats** the partial observability:

```
Without wormhole:
  Observation: [t-T, ..., t]
  Hidden: [t+1, ...]
  
With wormhole:
  Observation: [t-T, ..., t] + similar frames from [0, ..., t-T-1]
  Hidden: [t+1, ...]
  
  But if frame t looks like frame t-100, and we know what happened at t-99,
  we can infer what will happen at t+1!
```

Wormhole attention is like having a **memory oracle** that says "this situation is similar to one you've seen before." It doesn't reveal the future directly, but it provides strong evidence about what futures are likely.

---

## 9. Analogy to Lightning Breakdown

| Lightning Phase       | Predictor Phase                                                                 |
|-----------------------|---------------------------------------------------------------------------------|
| Charge accumulates    | Uncertainty accumulates in high-error zone                                      |
| Field smears across cloud | Belief spreads over manifold of plausible futures                            |
| Stepped leader forms  | Prediction drifts toward steepest loss gradient                                 |
| Return stroke         | Prediction snaps to true trajectory; error collapses                            |
| Field rebuilds        | Uncertainty front advances; belief re-spreads                                   |

The de Broglie-like fringes in the error map correspond to the branching leader channels visible in high-speed lightning footage: they trace the manifold of plausible continuations before one path wins.

### 9.1 The Dielectric Medium Analogy

The latent space acts like a **dielectric medium**:

- **Permittivity** ≈ model capacity (how much uncertainty can be stored)
- **Field strength** ≈ loss gradient magnitude  
- **Breakdown voltage** ≈ threshold where belief commits

Just as a dielectric stores charge until breakdown, the latent space stores uncertainty until the evidence forces commitment.

---

## 10. Controlling the Cycle

| Knob                        | Effect                                                                 |
|-----------------------------|------------------------------------------------------------------------|
| **Sharper loss (GAN, CE)**  | Forces earlier discharge; fringes disappear; prediction commits sooner |
| **Higher temporal top-k**   | More history considered; belief sharpens faster                        |
| **Lower wormhole threshold**| More non-local matches; belief can jump to distant attractors          |
| **Noise injection**         | Delays discharge; fringes persist longer                               |

Tuning these parameters lets us trade off between interpretability (visible fringes) and commitment speed (sharp predictions).

---

## 11. Implications for Latent Geometry

The pump cycle reveals that the latent space is not static—it oscillates between:

1. **Exploration**: belief spreads, fringes form, multiple hypotheses coexist.
2. **Exploitation**: belief collapses, prediction commits, error drops.

This is the same explore/exploit tradeoff seen in reinforcement learning and Bayesian optimization. The spectral attention architecture makes the cycle visible because each attention branch contributes a different "force" on the belief:

- Temporal attention pulls toward recent history.
- Neighbor attention enforces local smoothness.
- Wormhole attention allows non-local jumps.

The interplay of these forces shapes the equipotential surfaces we see in the error map.

### 11.1 Connection to the Platonic Representation Hypothesis

The vec2vec paper (arXiv:2505.12540) argues that diverse models converge to similar geometric structures in their latent spaces. Our observations support this:

1. **The error fringes are geometric invariants**—they appear regardless of initialization
2. **The pump cycle is universal**—it emerges from the MSE loss structure, not model specifics
3. **The equipotential surfaces follow physical laws**—they're shaped by gradient flow, not arbitrary

This suggests the latent space has intrinsic geometry determined by the task, not the architecture.

### 11.2 Why Physics Emerges

The key insight: **gradient descent on MSE loss is equivalent to energy minimization**. The model is solving a variational problem:

```
minimize E[‖ŷ - x_{t+1}‖²]  subject to  ŷ = f_θ(x_{t-T:t})
```

This is a **Lagrangian mechanics** problem. The error map shows the potential energy surface. The prediction follows gradient flow on this surface. The pump cycle is a **limit cycle** in the energy landscape.

Physical analogies emerge because the math is the same:
- MSE ↔ potential energy
- Gradient descent ↔ force = -∇V
- Learning rate ↔ mass/damping
- Belief collapse ↔ phase transition

---

## 12. Summary

1. **Partial observability is temporal**: the model sees past frames but not the future.
2. **The prediction is a belief state**: it represents the model's distribution over possible futures.
3. **Error equals uncertainty**: high error marks where the model is hedging between hypotheses.
4. **The pump cycle is universal**: tension (uncertainty builds) → discharge (belief commits) → recovery (new uncertainty).
5. **Wormhole attention reduces partial observability**: by finding similar past situations, it provides evidence about likely futures.
6. **Physical analogies are mathematical**: gradient descent on MSE is energy minimization, so physical laws emerge naturally.
7. **The error fringes are equipotential surfaces**: they trace the manifold of plausible futures, like leader channels in lightning.

Understanding this POMDP framing explains:
- Why "most probable = most error" (the mean of an uncertain belief has high variance)
- Why error "leads" the prediction (it marks the uncertainty front)
- Why the pump cycle exists (information flows from future to weights via the loss)
- Why physical analogies work (same underlying math)

---

## 13. Experimental Predictions

If this POMDP interpretation is correct, we should observe:

| Prediction | Test |
|------------|------|
| Higher time_depth → faster belief sharpening | Vary `timeDepth`, measure steps to low error |
| More wormhole connections → faster pattern recognition | Vary `wormhole_max_connections`, measure convergence |
| Stochastic patterns → persistent fringes | Use `bifurcation` or `noisy_motion` patterns |
| Deterministic patterns → fast collapse | Use `blob` pattern, observe rapid error drop |
| Sharp loss (GAN) → earlier commitment | Replace MSE with adversarial loss |

These predictions can be tested in the web UI by varying parameters and observing the collapse dynamics panel.

---

## 14. Connection to Adaptive Resonance Theory (ART)

The pump cycle has parallels to Grossberg's Adaptive Resonance Theory:

| ART Concept | Predictor Equivalent |
|-------------|---------------------|
| Bottom-up input | Current frame observation |
| Top-down expectation | Prediction from history |
| Resonance | Belief collapse (discharge) |
| Mismatch reset | Error spike, belief re-spreads |
| Vigilance parameter | Wormhole threshold (how similar must matches be?) |

Both systems exhibit:
1. **Stability-plasticity tradeoff**: learning new patterns vs. preserving old ones
2. **Hypothesis testing**: generate expectation, compare to input, update
3. **Resonant states**: stable configurations where expectation matches input

The spectral attention architecture can be viewed as a continuous relaxation of ART's discrete category dynamics.

---

## References

- POMDP formalism: Kaelbling, Littman, Cassandra, "Planning and Acting in Partially Observable Stochastic Domains," *Artificial Intelligence* 101 (1998).
- Relaxation oscillators: van der Pol, "On Relaxation Oscillations," *Philosophical Magazine* 2 (1926).
- Lightning physics: Uman, *Lightning* (Dover, 1984).
- Platonic Representation Hypothesis: Jha et al., "Harnessing the Universal Geometry of Embeddings," arXiv:2505.12540v3 (2025). [https://arxiv.org/abs/2505.12540](https://arxiv.org/abs/2505.12540)
- Adaptive Resonance Theory: Grossberg, "Adaptive Resonance Theory: How a brain learns to consciously attend, learn, and recognize a changing world," *Neural Networks* 37 (2013).
- Bayesian filtering: Thrun, Burgard, Fox, *Probabilistic Robotics* (MIT Press, 2005), Chapter 2.
- Variational inference: Blei, Kucukelbir, McAuliffe, "Variational Inference: A Review for Statisticians," *JASA* 112 (2017).

