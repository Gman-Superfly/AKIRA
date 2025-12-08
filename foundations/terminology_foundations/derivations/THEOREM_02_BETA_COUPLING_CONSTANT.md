# Theorem 02 – β as an Attention Coupling Parameter

## 0. Document Purpose and Scope

This document gives a **precise statement and proof sketch** for the mathematically clean part of:

> **Theorem 2 (Coupling Constant)** – "β = 1/√d_k is the attention coupling factor, analogous to a coupling constant: it controls interaction strength and sharpness of attention."

We aim to prove, under minimal and explicit assumptions, that:

1. The attention mechanism can be written as a **Gibbs/Boltzmann distribution** with inverse temperature β.
2. β is **dimensionless** and directly controls the **sharpness** of this distribution.
3. As β increases, the **entropy** of the attention distribution **strictly decreases**, except in the degenerate equal-logit case.
4. For small β, attention admits a **perturbative expansion** in β, so β plays the role of a coupling/perturbation parameter.

We **do not** claim or prove a true thermodynamic phase transition (with diverging susceptibility) here; that would require an infinite-system limit and much stronger assumptions. Instead, we prove the core facts that justify calling β an "interaction strength" or **coupling parameter** for attention.

---

## 1. Setup and Notation

We again work with a **single attention head** in one layer.

- Number of tokens: \( n \).
- Query/key dimension: \( d_k \).

Let \( Q, K \in \mathbb{R}^{n \times d_k} \) be the query and key matrices. For a fixed query token index \( i \):

- Query vector: \( q \equiv q_i \in \mathbb{R}^{d_k} \).
- Key vectors: \( k_j \in \mathbb{R}^{d_k} \), \( j = 1, \dots, n \).

The standard scaled dot-product attention logits for row \( i \) are:
\[
S_{ij} = \frac{q \cdot k_j}{\sqrt{d_k}}.
\]

Define the **scale parameter**:
\[
\beta := \frac{1}{\sqrt{d_k}}.
\]

Then we can rewrite the logits as:
\[
S_{ij} = \beta \, u_j, \quad \text{where } u_j := q \cdot k_j.
\]

The corresponding attention distribution over keys for query \( i \) is:
\[
 p_j(\beta) := A_{ij} = \frac{\exp(\beta u_j)}{Z(\beta)}, \quad Z(\beta) := \sum_{m=1}^n \exp(\beta u_m).
\]

This is a standard **log-linear / Gibbs distribution** with inverse temperature \( \beta \) and "energy" \( -u_j \).

We will treat \( u = (u_1, \dots, u_n) \) as fixed, and study the family of distributions \( p(\beta) \) as a function of \( \beta \ge 0 \).

---

## 2. Basic Properties of β

### 2.1 Dimensionlessness

The logits \( u_j = q \cdot k_j \) are inner products in \( \mathbb{R}^{d_k} \). In standard implementations, \( q \) and \( k \) are dimensionless feature vectors (pure numbers), so \( u_j \) is also dimensionless.

The scale factor \( \beta = 1/\sqrt{d_k} \) is a **pure number**: it is the reciprocal of the square root of the key dimension, which is itself a pure count. Thus:

- \( \beta \) has **no physical units**; it is dimensionless.

This matches the usual requirement for a coupling constant or inverse temperature: it scales a dimensionless energy/logit.

### 2.2 Gibbs/Boltzmann Form

By construction:
\[
 p_j(\beta) = \frac{e^{\beta u_j}}{\sum_m e^{\beta u_m}}.
\]

This is exactly the form of a **Gibbs measure**:
\[
 p_j(\beta) = \frac{e^{-\beta E_j}}{\sum_m e^{-\beta E_m}}, \quad \text{with } E_j := -u_j.
\]

So attention is a Gibbs distribution with energy levels \( E_j = -q \cdot k_j \) and inverse temperature \( \beta = 1/\sqrt{d_k} \). This is a straightforward reparameterization, but it is the key link to statistical mechanics.

No additional assumptions are needed for this step.

---

## 3. Entropy Monotonicity: β Controls Sharpness

We now prove that as \( \beta \) increases, the **entropy** of the attention distribution **strictly decreases**, except in the degenerate case where all logits are equal.

### 3.1 Entropy of the Attention Distribution

Define the Shannon entropy of \( p(\beta) \):
\[
H(\beta) := - \sum_{j=1}^n p_j(\beta) \log p_j(\beta).
\]

We assume \( u \) is not constant; i.e., there exists \( j, k \) with \( u_j \ne u_k \).

We will show:

> **Theorem 02.1 (Entropy Decreases with β)**:  
> For any non-constant logit vector \( u \), the entropy \( H(\beta) \) is a **strictly decreasing** function of \( \beta \ge 0 \). That is,
> \[
> \frac{dH}{d\beta} < 0 \quad \text{for all } \beta > 0.
> \]
>
> Equality \( dH/d\beta = 0 \) for all \( \beta \) occurs **only** when all \( u_j \) are equal (the trivial uniform case).

Intuitively: as \( \beta \) increases, the Gibbs distribution puts more mass on higher-\( u_j \) states and less on others, becoming more peaked; entropy drops.

### 3.2 Derivative of p with Respect to β

First, compute the derivative of \( p_j(\beta) \):
\[
 p_j(\beta) = \frac{e^{\beta u_j}}{Z(\beta)}, \quad Z(\beta) = \sum_m e^{\beta u_m}.
\]

Differentiate w.r.t. \( \beta \):
\[
 \frac{dp_j}{d\beta} = \frac{u_j e^{\beta u_j} Z(\beta) - e^{\beta u_j} Z'(\beta)}{Z(\beta)^2}.
\]

But:
\[
 Z'(\beta) = \sum_m u_m e^{\beta u_m}.
\]

So:
\[
 \frac{dp_j}{d\beta} = p_j(\beta) \big(u_j - \mathbb{E}_{p(\beta)}[u]\big),
\]
where:
\[
 \mathbb{E}_{p(\beta)}[u] := \sum_m p_m(\beta) u_m.
\]

This is a standard result for exponential families: the derivative of the mean parameter w.r.t. the natural parameter is the centered statistic.

### 3.3 Derivative of Entropy

Recall:
\[
H(\beta) = -\sum_j p_j(\beta) \log p_j(\beta).
\]

Differentiate:
\[
 \frac{dH}{d\beta} = -\sum_j \frac{dp_j}{d\beta} \log p_j - \sum_j p_j \frac{d}{d\beta}(\log p_j).
\]

But:
\[
 \frac{d}{d\beta}(\log p_j) = \frac{1}{p_j} \frac{dp_j}{d\beta}.
\]

So the second sum becomes:
\[
 -\sum_j p_j \cdot \frac{1}{p_j} \frac{dp_j}{d\beta} = -\sum_j \frac{dp_j}{d\beta}.
\]

Since \( \sum_j p_j = 1 \) for all \( \beta \), we have \( \sum_j \frac{dp_j}{d\beta} = 0 \). Therefore that term vanishes, and we obtain the simpler expression:
\[
 \frac{dH}{d\beta} = -\sum_j \frac{dp_j}{d\beta} \log p_j.
\]

Substitute \( \frac{dp_j}{d\beta} = p_j (u_j - \mathbb{E}[u]) \):
\[
 \frac{dH}{d\beta} = -\sum_j p_j (u_j - \mathbb{E}[u]) \log p_j.
\]

This form is slightly opaque; we can use a more elegant route by relating entropy to the **log partition function**.

### 3.4 Alternative Expression via Log Partition Function

Entropy for an exponential family with natural parameter \( \beta \) and sufficient statistic \( u \) can be written as (see standard references on exponential families):
\[
H(\beta) = \log Z(\beta) - \beta \mathbb{E}_{p(\beta)}[u].
\]

**Derivation**:
\[
H(\beta) = -\sum_j p_j \log p_j = -\sum_j p_j (\beta u_j - \log Z) = -\beta \mathbb{E}[u] + \log Z.
\]

Differentiate:
\[
 \frac{dH}{d\beta} = \frac{d}{d\beta} \log Z(\beta) - \mathbb{E}[u] - \beta \frac{d}{d\beta} \mathbb{E}[u].
\]

But:
\[
 \frac{d}{d\beta} \log Z(\beta) = \frac{1}{Z} Z'(\beta) = \frac{1}{Z} \sum_j u_j e^{\beta u_j} = \mathbb{E}_{p(\beta)}[u].
\]

So the first two terms cancel:
\[
 \frac{dH}{d\beta} = -\beta \frac{d}{d\beta} \mathbb{E}[u].
\]

Now, for exponential families we also know:
\[
 \frac{d}{d\beta} \mathbb{E}[u] = \mathrm{Var}_{p(\beta)}(u) = \mathbb{E}[u^2] - (\mathbb{E}[u])^2 \ge 0.
\]

Thus:
\[
 \frac{dH}{d\beta} = -\beta \, \mathrm{Var}_{p(\beta)}(u) \le 0.
\]

Moreover, since we assume \( u \) is not constant, \( \mathrm{Var}_{p(\beta)}(u) > 0 \) for all \( \beta > 0 \). Hence, for \( \beta > 0 \):
\[
 \frac{dH}{d\beta} < 0.
\]

This proves Theorem 02.1.

**Conclusion**:
- β **monotonically decreases** the entropy H(β) for any non-degenerate logit vector u.
- Larger β → sharper, lower-entropy attention distribution.

This is precisely the behavior expected of an **interaction strength / inverse temperature parameter**.

---

## 4. Perturbative Expansion Around β = 0

To further justify calling β a **coupling/perturbation parameter**, we show that for small β, the attention distribution admits a Taylor expansion in β around the **uniform distribution**.

### 4.1 Expansion of Softmax for Small β

At β = 0, we have:
\[
p_j(0) = \frac{1}{n} \quad \text{for all } j.
\]

We expand \( p_j(\beta) \) around 0.

Recall:
\[
p_j(\beta) = \frac{e^{\beta u_j}}{\sum_m e^{\beta u_m}}.
\]

Use the Taylor expansion \( e^{\beta u_j} = 1 + \beta u_j + \frac{\beta^2 u_j^2}{2} + O(\beta^3) \). Then:
\[
Z(\beta) = \sum_m e^{\beta u_m} = n + \beta \sum_m u_m + \frac{\beta^2}{2} \sum_m u_m^2 + O(\beta^3).
\]

Define the mean logit:
\[
\bar{u} := \frac{1}{n} \sum_m u_m.
\]

Then:
\[
Z(\beta) = n \Big(1 + \beta \bar{u} + \frac{\beta^2}{2n} \sum_m u_m^2 + O(\beta^3) \Big).
\]

Similarly, numerator:
\[
e^{\beta u_j} = 1 + \beta u_j + \frac{\beta^2 u_j^2}{2} + O(\beta^3).
\]

Thus:
\[
p_j(\beta) = \frac{1 + \beta u_j + \frac{\beta^2 u_j^2}{2} + O(\beta^3)}{n \Big(1 + \beta \bar{u} + O(\beta^2) \Big)}.
\]

Use \( 1/(1 + a) = 1 - a + a^2 + O(a^3) \) with \( a = \beta \bar{u} + O(\beta^2) \):
\[
\frac{1}{1 + \beta \bar{u} + O(\beta^2)} = 1 - \beta \bar{u} + O(\beta^2).
\]

So:
\[
p_j(\beta) = \frac{1}{n} \Big(1 + \beta u_j + O(\beta^2) \Big) \Big(1 - \beta \bar{u} + O(\beta^2) \Big).
\]

Multiply out, keeping up to first order:
\[
p_j(\beta) = \frac{1}{n} \Big(1 + \beta (u_j - \bar{u}) + O(\beta^2) \Big).
\]

Hence:
\[
 p_j(\beta) = \frac{1}{n} + \frac{\beta}{n} (u_j - \bar{u}) + O(\beta^2).
\]

This shows clearly:

- At β = 0: uniform distribution (no coupling between q and k).
- First-order deviation from uniform is **linear in β** and proportional to the centered logits \( u_j - \bar{u} \).

So β plays exactly the role of a **perturbation parameter**: small β = weak coupling, distribution close to uniform; large β = strong coupling, highly non-uniform distribution.

### 4.2 Interpretation as Coupling Strength

In physics, a coupling constant g often appears in an expansion:
\[
\text{observable} = \text{free part} + g \cdot \text{first-order correction} + g^2 \cdot \text{second-order} + \dots
\]

Here:

- "Free part" = uniform attention \( 1/n \) (no interaction between q and k).
- "Coupling" = β.
- "First-order correction" = \( (u_j - \bar{u})/n \) (how much a particular key deviates from the mean score).

Thus, β truly acts as a **coupling parameter** in the sense of perturbation theory.

---

## 5. Summary of Proven Properties

Collecting results:

> **Theorem 02 (β as Coupling Parameter for Attention)**  
> For scaled dot-product attention with logits \( S_{ij} = \beta \, q_i \cdot k_j \), where \( \beta = 1/\sqrt{d_k} \), we have:
>
> 1. (Gibbs form) Each attention row \( p(\beta) \) is a Gibbs distribution with inverse temperature β and energies \( E_j = -q \cdot k_j \).
> 2. (Dimensionless) β is a pure number (ratio of feature-scale quantities); it has no physical units.
> 3. (Entropy monotonicity) For any non-constant logit vector u, the entropy \( H(\beta) = -\sum_j p_j(\beta) \log p_j(\beta) \) is strictly decreasing in β for all β > 0:
>    \[
>    \frac{dH}{d\beta} = -\beta \, \mathrm{Var}_{p(\beta)}(u) < 0.
>    \]
> 4. (Perturbative expansion) For small β,
>    \[
>    p_j(\beta) = \frac{1}{n} + \frac{\beta}{n} (u_j - \bar{u}) + O(\beta^2),
>    \]
>    so β governs the first-order deviation from uniform attention.
>
> These properties justify interpreting β as an **interaction-strength parameter** (a coupling) for attention: increasing β strengthens the effect of q·k similarity on the attention distribution and monotonically sharpens that distribution.

We **have not** proven a true thermodynamic phase transition or universality-class statement here. Those require:

- An infinite-token or infinite-layer limit;
- Definition of an order parameter (e.g., condensate fraction of maximum attention weight);
- Analysis of critical exponents and susceptibilities.

Those questions are left for future work and are better treated explicitly as **conjectures** or **hypotheses** in the main theory documents.

---

## 6. Relation to `ATTENTION_COUPLING_FACTOR.md` and Experiments

Given this theorem, we can more carefully phrase the core claims in `ATTENTION_COUPLING_FACTOR.md`:

- It is **rigorously correct** to say:
  - β is dimensionless;
  - β is the inverse temperature of the attention Gibbs distribution;
  - β controls the sharpness of attention in a strictly monotone way;
  - β is the perturbation parameter around uniform attention.

- It is **plausible but not yet proven** (requires experiments) to say:
  - There exists an effective \( \beta_c \) at which attention becomes "effectively collapsed" for practical purposes;
  - Different architectures with different \( d_k \) occupy different regions of the (β, task) plane;
  - Layer agreement and AQ counts exhibit sharp changes around some architecture- and task-dependent \( \beta_c \).

Empirically, you can:

1. Measure attention entropy H(β) across models with different d_k, confirming the monotone relationship predicted here.
2. Fit an "operational \( \beta_c \)" where H(β) drops below some fraction of \( \log n \) (e.g., half-max entropy) and correlate that with AQ crystallization behavior.
3. Compare architectures (GPT-2 vs GPT-3 vs LLaMA) to see whether their default β places them in systematically different attention regimes.

---

**Oscar Goldman — Shogu Research Group @ Datamutant.ai**

*This theorem makes precise the sense in which β is a coupling/temperature parameter for attention. It does not overclaim a full phase-transition theory, but it provides a solid mathematical backbone for the "attention coupling factor" concept and for treating β as the fine-structure-like parameter of attention strength.*
