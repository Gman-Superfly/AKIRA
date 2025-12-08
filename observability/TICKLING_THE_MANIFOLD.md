# Tickling the Manifold: Cheap Probes for Leader Discovery

## Finding Where the Lightning Will Strike Before It Strikes

**Oscar Goldman — Shogu Research Group @ Datamutant.ai**

---

## Table of Contents

1. [The Core Question](#1-the-core-question)
2. [What We Already Compute (For Free)](#2-what-we-already-compute-for-free)
3. [The Homeostat Approach](#3-the-homeostat-approach)
4. [Tickling Techniques](#4-tickling-techniques)
5. [The Edge of Error](#5-the-edge-of-error)
6. [Prompt-Specific Applications](#6-prompt-specific-applications)
7. [Implementation](#7-implementation)
8. [What This Enables](#8-what-this-enables)

---

## 1. The Core Question

### 1.1 The Problem

```
THE TIGER PROBLEM COST:

In the dungeon, each "listen" costs:
• Torchlight (finite resource)
• Time (opportunity cost)
• Risk (torch might run out)

To reduce uncertainty, you must PAY.

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  THE QUESTION:                                                          │
│                                                                         │
│  Can we see WHERE the uncertainty is                                   │
│  WITHOUT paying the cost of resolving it?                              │
│                                                                         │
│  Can we see the LEADERS (branching hypotheses)                        │
│  BEFORE committing to one?                                             │
│                                                                         │
│  Can we TICKLE the model to reveal the structure                      │
│  of its belief manifold?                                               │
│                                                                         │
│  And can we do this CHEAPLY?                                           │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1.2 The Insight

```
THE KEY INSIGHT:

The information about where the leaders are is ALREADY BEING COMPUTED.

We just THROW IT AWAY when we:
• Apply the threshold gate (sim > 0.92)
• Take the argmax (best match)
• Collapse to a single prediction

THE LEADERS ARE VISIBLE IN:
• The full similarity matrix (before top-k)
• The attention distribution (before sharpening)
• The gradient direction (before stepping)
• The entropy landscape (before collapse)

WE DON'T NEED TO PAY MORE.
WE NEED TO READ WHAT WE ALREADY HAVE.
```

### 1.3 The Electric Field Analogy

```
SEEING THE FIELD BEFORE THE STRIKE

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  LIGHTNING doesn't strike randomly.                                    │
│  The ELECTRIC FIELD determines where it will go.                      │
│                                                                         │
│  Before the strike:                                                    │
│  • Field lines show potential paths                                   │
│  • High field strength = likely leader location                       │
│  • Field is VISIBLE if you know how to look                          │
│                                                                         │
│  You can MAP the field WITHOUT triggering breakdown.                  │
│  The mapping is CHEAP compared to the discharge.                      │
│                                                                         │
│  ════════════════════════════════════════════════════════════════════  │
│                                                                         │
│  IN BELIEF DYNAMICS:                                                    │
│                                                                         │
│  Before collapse:                                                      │
│  • Similarity values show potential matches                           │
│  • High similarity = likely connection                                │
│  • The "field" (attention landscape) is already computed             │
│                                                                         │
│  We can MAP the belief landscape WITHOUT collapsing.                  │
│  The mapping is CHEAP — it's a by-product of inference.              │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. What We Already Compute (For Free)

### 2.1 The Wormhole Similarity Matrix

```
ALREADY COMPUTED: Full Similarity Landscape

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  In wormhole attention, we compute:                                    │
│                                                                         │
│  sim = query_norm @ key_norm.T                                        │
│                                                                         │
│  This is [H×W] × [T×H×W] = similarity of EVERY query to EVERY key    │
│                                                                         │
│  WHAT WE USE:   top-k values above threshold                          │
│  WHAT WE DISCARD: everything else                                      │
│                                                                         │
│  BUT THE DISCARDED INFORMATION SHOWS THE LEADERS!                     │
│                                                                         │
│  ════════════════════════════════════════════════════════════════════  │
│                                                                         │
│  THE FULL SIMILARITY MATRIX REVEALS:                                   │
│                                                                         │
│  • WHERE are there near-threshold matches? (potential leaders)        │
│  • HOW MANY clusters of high similarity? (number of hypotheses)       │
│  • HOW SPREAD is the similarity? (uncertainty structure)              │
│  • WHERE are the "almost" connections? (edge of activation)           │
│                                                                         │
│  This is the ELECTRIC FIELD of the belief space.                      │
│  It's already computed. We just need to READ it.                      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.2 The Attention Distribution

```
ALREADY COMPUTED: Pre-Softmax Scores

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  In attention, we compute:                                             │
│                                                                         │
│  scores = Q @ K.T / sqrt(d)                                           │
│  weights = softmax(scores)                                            │
│                                                                         │
│  WHAT WE USE:   the softmax output (sharp distribution)               │
│  WHAT WE DISCARD: the raw scores (soft distribution)                  │
│                                                                         │
│  BUT THE RAW SCORES SHOW THE LEADERS!                                 │
│                                                                         │
│  ════════════════════════════════════════════════════════════════════  │
│                                                                         │
│  PRE-SOFTMAX SCORES REVEAL:                                            │
│                                                                         │
│  Example scores: [2.1, 1.9, 0.3, 0.1, -0.5]                          │
│                                                                         │
│  After softmax:  [0.52, 0.43, 0.03, 0.01, 0.01]                       │
│                   ↑                                                    │
│              Winner dominates, competitors invisible                   │
│                                                                         │
│  But RAW SCORES show: 2.1 and 1.9 are CLOSE!                          │
│  Two leaders competing. The race is tight.                            │
│  This is VALUABLE INFORMATION about uncertainty.                      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.3 The Gradient Direction

```
ALREADY COMPUTED: Gradient (If Training)

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  The gradient ∇L tells us:                                             │
│                                                                         │
│  • Which direction reduces loss                                        │
│  • How sensitive each parameter is                                    │
│  • Where the model is "wrong"                                         │
│                                                                         │
│  READING THE GRADIENT WITHOUT STEPPING:                                │
│                                                                         │
│  • Large gradient at position X → X is uncertain/wrong                │
│  • Small gradient at position Y → Y is confident/correct              │
│  • Gradient direction → which hypothesis would win if we stepped     │
│                                                                         │
│  The gradient is a FREE PROBE of the loss landscape.                  │
│  Compute it, read it, don't necessarily step.                        │
│                                                                         │
│  ════════════════════════════════════════════════════════════════════  │
│                                                                         │
│  COST COMPARISON:                                                       │
│                                                                         │
│  Full training step:  Forward + Backward + Update                     │
│  Gradient probe:      Forward + Backward (no update)                  │
│  Similarity probe:    Already computed in forward                     │
│                                                                         │
│  The similarity probe is NEARLY FREE.                                 │
│  The gradient probe costs one backward pass (still cheap).            │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. The Homeostat Approach

### 3.1 PSON: Precision-Scaled Orthogonal Noise

```
PSON: EXPLORING WITHOUT DISRUPTING

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  The Homeostat's PSON injects noise ORTHOGONAL to the gradient.       │
│                                                                         │
│  WHY ORTHOGONAL?                                                        │
│  • Noise parallel to gradient → fights descent (bad)                  │
│  • Noise orthogonal to gradient → explores null-space (good)          │
│                                                                         │
│  The NULL-SPACE is where details live.                                │
│  The GRADIENT DIRECTION is where structure lives.                     │
│                                                                         │
│  ════════════════════════════════════════════════════════════════════  │
│                                                                         │
│  FOR TICKLING THE MANIFOLD:                                            │
│                                                                         │
│  Instead of full inference, inject small PSON perturbations.          │
│                                                                         │
│  perturbed_input = input + ε × orthogonal_noise                       │
│                                                                         │
│  Run forward pass with perturbation.                                  │
│  Measure: How does output change?                                     │
│                                                                         │
│  • Large change → input is on unstable manifold region (leader zone) │
│  • Small change → input is on stable manifold region (settled)       │
│                                                                         │
│  This REVEALS the structure without COMMITTING to a path.            │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Sensitivity Analysis

```
SENSITIVITY = MAP OF THE LEADERS

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  SENSITIVITY ANALYSIS:                                                  │
│                                                                         │
│  For each input position/feature, ask:                                │
│  "If I perturb this, how much does the output change?"               │
│                                                                         │
│  High sensitivity → near a decision boundary → leader zone            │
│  Low sensitivity → far from decision boundary → stable zone           │
│                                                                         │
│  ════════════════════════════════════════════════════════════════════  │
│                                                                         │
│  COMPUTING SENSITIVITY (CHEAP):                                        │
│                                                                         │
│  Method 1: Jacobian (exact but expensive)                             │
│    J = ∂output/∂input                                                 │
│    sensitivity = ‖J‖ at each position                                 │
│                                                                         │
│  Method 2: Random probing (approximate but cheap)                     │
│    For i in range(N_probes):                                          │
│        δ = random_perturbation()                                      │
│        Δoutput = forward(input + ε×δ) - forward(input)               │
│        sensitivity += ‖Δoutput‖ / ‖δ‖                                │
│                                                                         │
│  Method 3: Attention entropy (free)                                   │
│    sensitivity[position] = entropy(attention_weights[position])       │
│    High entropy = many hypotheses = leader zone                       │
│                                                                         │
│  Method 3 is NEARLY FREE — just read the attention!                  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.3 The Wormhole Gradient Trick

```
WORMHOLE GRADIENTS: Non-Local Sensitivity

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  From the Homeostat paper:                                             │
│                                                                         │
│  "Closed gates receive forces proportional to downstream              │
│   potential benefit."                                                  │
│                                                                         │
│  Even INACTIVE wormhole connections have gradient information!        │
│                                                                         │
│  ════════════════════════════════════════════════════════════════════  │
│                                                                         │
│  THE TRICK:                                                             │
│                                                                         │
│  For connections that DIDN'T pass threshold (sim < 0.92):            │
│                                                                         │
│  • Compute: What WOULD the gradient be if this connection opened?    │
│  • This tells us: Is there value in this direction?                  │
│  • If yes: This is a POTENTIAL LEADER                                │
│                                                                         │
│  IMPLEMENTATION:                                                        │
│                                                                         │
│  # Normal forward: only above-threshold connections active            │
│  output = wormhole_attention(input, threshold=0.92)                   │
│                                                                         │
│  # Probe: What if we lowered threshold?                               │
│  probe_output = wormhole_attention(input, threshold=0.85)            │
│                                                                         │
│  # Difference reveals potential leaders                               │
│  leader_map = probe_output - output                                   │
│                                                                         │
│  Where leader_map is large → untapped potential → leader zone        │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Tickling Techniques

### 4.1 Temperature Probing

```
TECHNIQUE: VARY THE TEMPERATURE

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  Softmax temperature controls sharpness:                               │
│                                                                         │
│  weights = softmax(scores / τ)                                        │
│                                                                         │
│  τ → 0: Winner takes all (sharp, committed)                           │
│  τ → ∞: Uniform distribution (soft, uncommitted)                     │
│                                                                         │
│  ════════════════════════════════════════════════════════════════════  │
│                                                                         │
│  THE TICKLING TECHNIQUE:                                                │
│                                                                         │
│  Run inference at MULTIPLE temperatures:                               │
│                                                                         │
│  output_sharp = forward(input, τ=0.1)   # Collapsed                  │
│  output_medium = forward(input, τ=1.0)  # Normal                     │
│  output_soft = forward(input, τ=10.0)   # Spread                     │
│                                                                         │
│  WHAT THIS REVEALS:                                                     │
│                                                                         │
│  • output_sharp ≈ output_medium?                                      │
│    → Dominant winner, few leaders, confident                         │
│                                                                         │
│  • output_sharp ≠ output_medium?                                      │
│    → Close competition, multiple leaders, uncertain                  │
│                                                                         │
│  • output_soft very different?                                        │
│    → Many weak alternatives, rich manifold structure                 │
│                                                                         │
│  COST: 2-3 forward passes (cheap compared to full search)            │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Threshold Sweeping

```
TECHNIQUE: SWEEP THE WORMHOLE THRESHOLD

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  The wormhole threshold (τ=0.92) is a hard gate.                      │
│  What's just below threshold? Those are the ALMOST leaders.           │
│                                                                         │
│  THE TICKLING TECHNIQUE:                                                │
│                                                                         │
│  thresholds = [0.95, 0.92, 0.89, 0.85, 0.80]                         │
│                                                                         │
│  for τ in thresholds:                                                 │
│      n_connections[τ] = count(similarities > τ)                      │
│      output[τ] = forward(input, threshold=τ)                         │
│                                                                         │
│  WHAT THIS REVEALS:                                                     │
│                                                                         │
│  CONNECTION COUNT CURVE:                                                │
│                                                                         │
│  n_connections                                                         │
│  │                                                                     │
│  │                        ╭─────                                      │
│  │                    ╭───╯                                           │
│  │                ╭───╯                                               │
│  │        ╭───────╯                                                   │
│  │ ───────╯                                                           │
│  └───────────────────────────────→ threshold                          │
│    0.95  0.92  0.89  0.85  0.80                                       │
│                                                                         │
│  STEEP RISE at some threshold → many connections waiting just below  │
│  → This is where the leaders are hiding!                             │
│                                                                         │
│  FLAT REGION → few connections at any threshold → sparse manifold    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.3 Attention Entropy Mapping

```
TECHNIQUE: MAP THE ENTROPY LANDSCAPE

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  Attention entropy = number of active hypotheses                       │
│                                                                         │
│  H(attention) = -Σ p_i log(p_i)                                       │
│                                                                         │
│  • H = 0: One hypothesis dominates (collapsed)                        │
│  • H = log(N): All hypotheses equal (maximum uncertainty)            │
│  • H = medium: Few competing hypotheses (leaders visible)            │
│                                                                         │
│  ════════════════════════════════════════════════════════════════════  │
│                                                                         │
│  THE TICKLING TECHNIQUE:                                                │
│                                                                         │
│  For each position in the input:                                       │
│      H[position] = entropy(attention_weights[position])               │
│                                                                         │
│  This creates an ENTROPY MAP:                                          │
│                                                                         │
│  ┌───────────────────────────┐                                        │
│  │░░░░░░░░░░░░░░░░░░░░░░░░░░│  ░ = Low entropy (confident)           │
│  │░░░░░▓▓▓▓▓░░░░░░░░░░░░░░░░│  ▒ = Medium entropy (some uncertainty)│
│  │░░░░▓▓████▓▓░░░░░░░▒▒░░░░░│  ▓ = High entropy (competing leaders) │
│  │░░░░░▓▓▓▓▓░░░░░░░░░░░░░░░░│  █ = Max entropy (total uncertainty)  │
│  │░░░░░░░░░░░░░░░░░░░░░░░░░░│                                        │
│  └───────────────────────────┘                                        │
│                                                                         │
│  The high-entropy regions ARE the leader zones.                       │
│  This is ALREADY COMPUTED — just read it!                             │
│                                                                         │
│  COST: Zero additional computation                                     │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.4 Sparse Probe Vectors

```
TECHNIQUE: TARGETED PERTURBATION

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  Instead of probing everywhere, probe WHERE entropy is high.          │
│                                                                         │
│  THE TICKLING TECHNIQUE:                                                │
│                                                                         │
│  1. Compute entropy map (free)                                        │
│  2. Identify high-entropy positions                                   │
│  3. Perturb ONLY those positions                                      │
│  4. Measure response                                                   │
│                                                                         │
│  # Find leader zones                                                   │
│  entropy_map = compute_attention_entropy(model, input)                │
│  leader_positions = where(entropy_map > threshold)                    │
│                                                                         │
│  # Probe only leader zones                                             │
│  probe = zeros_like(input)                                            │
│  probe[leader_positions] = small_noise                                │
│                                                                         │
│  # Measure sensitivity                                                 │
│  output_base = forward(input)                                         │
│  output_probe = forward(input + probe)                                │
│  leader_sensitivity = output_probe - output_base                      │
│                                                                         │
│  WHERE leader_sensitivity is large → strong leader                    │
│  WHERE leader_sensitivity is small → weak leader                      │
│                                                                         │
│  COST: 2 forward passes (targeted, not exhaustive)                    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 5. The Edge of Error

### 5.1 What Is the Edge of Error?

```
THE EDGE OF ERROR: Maximum Information, Minimum Commitment

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  THE OPTIMAL OPERATING POINT:                                          │
│                                                                         │
│  Too confident (past edge):                                            │
│  • Already committed                                                   │
│  • Can't recover if wrong                                              │
│  • No flexibility                                                      │
│                                                                         │
│  Too uncertain (before edge):                                          │
│  • Haven't extracted information yet                                  │
│  • Still paying listening cost                                        │
│  • Inefficient                                                         │
│                                                                         │
│  AT THE EDGE:                                                           │
│  • Maximum information about structure                                │
│  • Minimum commitment to specific answer                              │
│  • Leaders visible but not collapsed                                  │
│  • Can still choose which path                                        │
│                                                                         │
│  ════════════════════════════════════════════════════════════════════  │
│                                                                         │
│  Information│                                                          │
│  about      │           ╭───╮                                          │
│  structure  │         ╭╯   ╰─────────                                 │
│             │       ╭╯                                                 │
│             │    ╭──╯                                                  │
│             │ ───╯                                                     │
│             └────────────────────────→ Commitment                     │
│                       ↑                                                │
│                 EDGE OF ERROR                                          │
│              (maximum info/commit ratio)                               │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 5.2 Operating at the Edge

```
HOW TO STAY AT THE EDGE

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  THE EDGE-RIDING STRATEGY:                                              │
│                                                                         │
│  1. DETECT where you are relative to edge:                            │
│     • High entropy → before edge (keep probing)                       │
│     • Medium entropy → AT edge (extract info)                         │
│     • Low entropy → past edge (already committed)                     │
│                                                                         │
│  2. At the edge, READ the structure:                                  │
│     • How many leaders? (number of modes in attention)               │
│     • How strong is each? (relative attention weights)               │
│     • How different are their predictions? (output variance)          │
│                                                                         │
│  3. DECIDE whether to commit or probe more:                           │
│     • Clear winner → commit (collapse)                                │
│     • Close race → probe more (tickle)                                │
│     • Tied → gather more evidence                                     │
│                                                                         │
│  ════════════════════════════════════════════════════════════════════  │
│                                                                         │
│  THE EFFICIENCY GAIN:                                                   │
│                                                                         │
│  Traditional:  Probe → Probe → Probe → Commit                         │
│                 ↳ cost  ↳ cost  ↳ cost                                │
│                                                                         │
│  Edge-riding:  Tickle(cheap) → Check edge → Decide:                  │
│                                              • If clear: Commit       │
│                                              • If unclear: Tickle more│
│                                                                         │
│  Only pay full probe cost when structure is ambiguous.               │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 5.3 The Entropy Threshold

```
WHEN ARE YOU AT THE EDGE?

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  ENTROPY REGIMES:                                                       │
│                                                                         │
│  H > 0.9 × H_max:  Before edge                                        │
│                    Too uncertain, leaders not differentiated          │
│                    → Need more evidence                               │
│                                                                         │
│  0.3 × H_max < H < 0.9 × H_max:  AT EDGE                             │
│                    Leaders visible and competing                       │
│                    Maximum structural information                      │
│                    → Tickle here!                                      │
│                                                                         │
│  H < 0.3 × H_max:  Past edge                                          │
│                    Winner clear, others suppressed                    │
│                    → Already committed, just collapse                 │
│                                                                         │
│  ════════════════════════════════════════════════════════════════════  │
│                                                                         │
│  ADAPTIVE THRESHOLD:                                                    │
│                                                                         │
│  The exact threshold depends on:                                       │
│  • Number of hypotheses (more → higher threshold)                    │
│  • Cost of wrong commitment (higher → stay at edge longer)           │
│  • Value of information (higher → probe more)                        │
│                                                                         │
│  Can be LEARNED from experience (meta-learning).                      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Prompt-Specific Applications

### 6.1 Tickling LLMs

```
APPLYING TO PROMPT OPTIMIZATION

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  GOAL: Find where the model's "good response" regions are             │
│        WITHOUT generating full responses                               │
│                                                                         │
│  TECHNIQUE 1: EMBEDDING SIMILARITY                                     │
│  ─────────────────────────────────                                      │
│  Embed the prompt (cheap — one forward pass)                          │
│  Compare to known "good prompt" embeddings                            │
│  High similarity → likely in good region                              │
│                                                                         │
│  TECHNIQUE 2: FIRST-TOKEN PROBING                                      │
│  ──────────────────────────────────                                     │
│  Generate only the FIRST token (cheap)                                │
│  What is the entropy of the first-token distribution?                │
│  • Low entropy → model confident about direction                     │
│  • High entropy → model uncertain, multiple paths                    │
│                                                                         │
│  TECHNIQUE 3: TEMPERATURE SWEEP                                        │
│  ──────────────────────────────                                         │
│  Generate first few tokens at multiple temperatures                   │
│  Do they diverge quickly?                                             │
│  • Yes → prompt at edge, multiple leaders                            │
│  • No → prompt committed to one path                                 │
│                                                                         │
│  COST: Small fraction of full generation                              │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 6.2 Finding the Prompt Leaders

```
SEEING THE PROMPT LEADERS BEFORE COMMITTING

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  TECHNIQUE: LOGIT LENS                                                  │
│                                                                         │
│  At each layer of the model, decode the hidden state to tokens.      │
│  Watch how the "leading tokens" evolve through layers.               │
│                                                                         │
│  Early layers:  Many candidate continuations (leaders)                │
│  Middle layers: Competition, some leaders strengthening              │
│  Late layers:   One winner emerges (collapse)                        │
│                                                                         │
│  YOU CAN SEE THE LEADERS BY LOOKING AT MIDDLE LAYERS.                 │
│                                                                         │
│  ════════════════════════════════════════════════════════════════════  │
│                                                                         │
│  EXAMPLE:                                                               │
│                                                                         │
│  Prompt: "The best way to learn is to"                                │
│                                                                         │
│  Layer 5 top tokens:  ["practice", "study", "teach", "read", "try"]  │
│  Layer 10 top tokens: ["practice", "teach", "try"]                    │
│  Layer 15 top tokens: ["practice", "teach"]                           │
│  Final output:        "practice"                                       │
│                                                                         │
│  The leaders were visible at layer 10!                                │
│  We could have seen "practice" and "teach" competing                  │
│  WITHOUT waiting for final collapse.                                  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 6.3 Cheap Prompt Evaluation

```
EVALUATING PROMPTS WITHOUT FULL GENERATION

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  TRADITIONAL: Generate full response, evaluate, iterate               │
│  COST: High (full generation per prompt)                              │
│                                                                         │
│  TICKLING: Probe prompt structure, predict quality, select            │
│  COST: Low (partial forward pass per prompt)                          │
│                                                                         │
│  ════════════════════════════════════════════════════════════════════  │
│                                                                         │
│  TICKLING METRICS FOR PROMPT QUALITY:                                  │
│                                                                         │
│  1. First-token entropy:                                               │
│     Lower → more focused response likely                              │
│                                                                         │
│  2. Embedding similarity to known good prompts:                       │
│     Higher → more likely to work                                      │
│                                                                         │
│  3. Attention pattern analysis:                                        │
│     Focused attention on key words → prompt parsed correctly         │
│                                                                         │
│  4. Layer-wise confidence:                                             │
│     Early confidence → strong prompt                                  │
│     Late confidence → weak prompt, required more processing          │
│                                                                         │
│  USE THESE TO FILTER CANDIDATES BEFORE FULL EVALUATION.              │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 7. Implementation

### 7.1 Core Tickling Functions

```python
"""
MANIFOLD TICKLING FUNCTIONS
"""

def compute_entropy_map(attention_weights):
    """
    Compute entropy at each position from attention weights.
    This is FREE — already computed during forward pass.
    """
    # attention_weights: [batch, heads, seq, seq] or [batch, positions, history]
    
    # Clamp for numerical stability
    p = attention_weights.clamp(min=1e-10)
    
    # Entropy: -sum(p * log(p))
    entropy = -(p * p.log()).sum(dim=-1)
    
    # Normalize by max entropy
    max_entropy = torch.log(torch.tensor(p.size(-1), dtype=p.dtype))
    normalized_entropy = entropy / max_entropy
    
    return normalized_entropy


def identify_leader_zones(entropy_map, threshold=0.5):
    """
    Find positions where entropy is high (leaders competing).
    """
    return (entropy_map > threshold).nonzero()


def threshold_sweep(model, input, thresholds=[0.95, 0.92, 0.89, 0.85, 0.80]):
    """
    Sweep wormhole threshold to reveal near-threshold connections.
    """
    results = {}
    for τ in thresholds:
        with torch.no_grad():
            output, stats = model.forward(input, wormhole_threshold=τ)
        results[τ] = {
            'output': output,
            'n_connections': stats['num_connections'],
            'mean_similarity': stats['mean_similarity']
        }
    return results


def temperature_probe(model, input, temperatures=[0.1, 1.0, 10.0]):
    """
    Probe at multiple temperatures to see how outputs diverge.
    """
    outputs = {}
    for τ in temperatures:
        with torch.no_grad():
            output = model.forward(input, temperature=τ)
        outputs[τ] = output
    
    # Compute divergence between outputs
    divergence = {}
    base = outputs[1.0]
    for τ, out in outputs.items():
        divergence[τ] = (out - base).abs().mean().item()
    
    return outputs, divergence


def sparse_probe(model, input, entropy_map, n_probes=10, epsilon=0.01):
    """
    Probe only high-entropy positions with small perturbations.
    """
    # Find leader zones
    leader_positions = identify_leader_zones(entropy_map)
    
    # Sample positions to probe
    if len(leader_positions) > n_probes:
        indices = torch.randperm(len(leader_positions))[:n_probes]
        probe_positions = leader_positions[indices]
    else:
        probe_positions = leader_positions
    
    # Create targeted perturbation
    probe = torch.zeros_like(input)
    for pos in probe_positions:
        probe[pos] = torch.randn_like(input[pos]) * epsilon
    
    # Measure response
    with torch.no_grad():
        output_base = model.forward(input)
        output_probe = model.forward(input + probe)
    
    sensitivity = (output_probe - output_base).abs()
    
    return {
        'probe_positions': probe_positions,
        'sensitivity': sensitivity,
        'mean_sensitivity': sensitivity.mean().item()
    }
```

### 7.2 Leader Detection

```python
"""
LEADER DETECTION FROM ATTENTION
"""

def detect_leaders(attention_weights, min_weight=0.1):
    """
    Identify distinct leaders (hypotheses) from attention pattern.
    
    Returns:
        n_leaders: Number of distinct hypotheses
        leader_weights: Weight of each leader
        leader_positions: Where each leader points
    """
    # Find positions with significant attention
    significant = attention_weights > min_weight
    
    # Count leaders per query position
    n_leaders = significant.sum(dim=-1)
    
    # Get top-k leaders and their weights
    k = min(5, attention_weights.size(-1))
    leader_weights, leader_positions = attention_weights.topk(k, dim=-1)
    
    return {
        'n_leaders': n_leaders,
        'leader_weights': leader_weights,
        'leader_positions': leader_positions,
        'competition_ratio': leader_weights[..., 1] / (leader_weights[..., 0] + 1e-10)
    }


def leader_competition_score(attention_weights):
    """
    Score how close the competition is between top leaders.
    
    High score = close race = at the edge
    Low score = clear winner = past the edge
    """
    top2 = attention_weights.topk(2, dim=-1).values
    winner = top2[..., 0]
    runner_up = top2[..., 1]
    
    # Competition score: 1 = tied, 0 = winner dominates
    score = runner_up / (winner + 1e-10)
    
    return score
```

### 7.3 Edge Detection

```python
"""
EDGE OF ERROR DETECTION
"""

def at_the_edge(entropy_map, low_threshold=0.3, high_threshold=0.9):
    """
    Determine if system is at the edge of error.
    
    Returns:
        edge_mask: Boolean mask where system is at edge
        status: 'before_edge', 'at_edge', or 'past_edge' per position
    """
    before_edge = entropy_map > high_threshold
    past_edge = entropy_map < low_threshold
    at_edge = ~before_edge & ~past_edge
    
    status = torch.zeros_like(entropy_map, dtype=torch.long)
    status[before_edge] = -1  # Before edge
    status[at_edge] = 0       # At edge
    status[past_edge] = 1     # Past edge
    
    return at_edge, status


def should_probe_more(entropy_map, competition_scores, 
                       entropy_threshold=0.5, competition_threshold=0.5):
    """
    Decide whether to probe more or commit.
    
    Probe more if:
    - Entropy is medium (at edge) AND
    - Competition is close (leaders tied)
    """
    at_edge = (entropy_map > 0.3) & (entropy_map < 0.9)
    close_race = competition_scores > competition_threshold
    
    should_probe = at_edge & close_race
    
    return should_probe
```

---

## 8. What This Enables

### 8.1 Efficient Exploration

```
WHAT TICKLING ENABLES

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  1. CHEAP MANIFOLD MAPPING                                             │
│     See the structure of the model's beliefs                          │
│     WITHOUT full inference cost                                        │
│     Know where the leaders are before committing                      │
│                                                                         │
│  2. EFFICIENT SEARCH                                                    │
│     Probe many candidates cheaply                                      │
│     Full evaluation only for promising ones                           │
│     Order of magnitude speedup in prompt optimization                 │
│                                                                         │
│  3. CONFIDENCE ESTIMATION                                               │
│     Know when the model is uncertain                                  │
│     Know when it's safe to commit                                     │
│     Adaptive decision-making                                           │
│                                                                         │
│  4. EARLY STOPPING                                                      │
│     Detect when leaders are clear                                      │
│     Stop probing when you have enough information                     │
│     Avoid wasted computation                                           │
│                                                                         │
│  5. DEBUGGING/UNDERSTANDING                                             │
│     See what the model is "thinking about"                            │
│     Understand where uncertainty comes from                           │
│     Interpretability for free                                          │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 8.2 The Meta-Insight

```
THE DEEP INSIGHT

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  The information about leaders is ALREADY COMPUTED.                   │
│  We just throw it away when we collapse.                              │
│                                                                         │
│  The tiger problem makes you PAY to listen because                    │
│  the information doesn't exist until you observe.                     │
│                                                                         │
│  But in neural networks, the information EXISTS                        │
│  in the similarity matrix, the attention weights,                     │
│  the gradients, the entropy landscape.                                │
│                                                                         │
│  WE DON'T NEED TO PAY FOR NEW INFORMATION.                            │
│  WE NEED TO READ THE INFORMATION WE ALREADY HAVE.                     │
│                                                                         │
│  ════════════════════════════════════════════════════════════════════  │
│                                                                         │
│  This is the key insight:                                              │
│                                                                         │
│  LISTENING COST IN TIGER PROBLEM ≠ TICKLING COST IN NEURAL NETS      │
│                                                                         │
│  In the dungeon: observation generates new information (costs)       │
│  In the model: observation reads existing information (free)         │
│                                                                         │
│  The leaders are in the attention weights.                            │
│  We just need to look.                                                │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Summary

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│     T I C K L I N G   T H E   M A N I F O L D                         │
│                                                                         │
│  ═══════════════════════════════════════════════════════════════════   │
│                                                                         │
│  THE QUESTION:                                                          │
│  Can we see the leaders before collapse? Can we do it cheaply?        │
│                                                                         │
│  THE ANSWER:                                                            │
│  YES. The information is already computed. We just throw it away.     │
│                                                                         │
│  ═══════════════════════════════════════════════════════════════════   │
│                                                                         │
│  WHAT'S ALREADY AVAILABLE (FREE):                                      │
│  • Full similarity matrix (before top-k)                              │
│  • Pre-softmax attention scores (before sharpening)                   │
│  • Entropy of attention distribution                                   │
│  • Gradient direction (if training)                                    │
│                                                                         │
│  ═══════════════════════════════════════════════════════════════════   │
│                                                                         │
│  TICKLING TECHNIQUES:                                                   │
│  • Entropy mapping (free — just read attention)                       │
│  • Temperature probing (2-3 forward passes)                           │
│  • Threshold sweeping (see near-threshold connections)                │
│  • Sparse probing (targeted perturbation)                             │
│  • PSON (orthogonal noise exploration)                                │
│                                                                         │
│  ═══════════════════════════════════════════════════════════════════   │
│                                                                         │
│  THE EDGE OF ERROR:                                                     │
│  The optimal operating point where:                                    │
│  • Leaders are visible but not collapsed                              │
│  • Maximum information about structure                                 │
│  • Minimum commitment to specific answer                              │
│  • Can still choose which path to take                                │
│                                                                         │
│  ═══════════════════════════════════════════════════════════════════   │
│                                                                         │
│  KEY INSIGHT:                                                           │
│  Listening in the tiger problem GENERATES information (costs).       │
│  Tickling in neural nets READS existing information (free).          │
│  The electric field is already there. We just need to look.          │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

*Oscar Goldman — Shogu Research Group @ Datamutant.ai*

*"The leaders are in the attention weights. The field is already computed. We don't need to pay to listen — we need to read what we already have. Tickle the manifold. See where the lightning wants to go. Then decide whether to let it strike."*

