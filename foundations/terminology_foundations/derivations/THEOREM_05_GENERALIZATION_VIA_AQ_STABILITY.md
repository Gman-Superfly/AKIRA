# Theorem 05 – Generalization Bounds via AQ Stability

## 0. Purpose and Scope

This document gives a **stylized, but standard-learning-theory-consistent** derivation of a generalization bound where the key quantity is the **number of Action-Quantum-like factors** \(k\), rather than the total number of parameters \(p\).

It formalizes the slogan from `DERIVED_FORMULATIONS.md`:

> "Generalization = consistent AQ extraction. Similar inputs → same AQ → same output."

in the following way:

- We model an **AQ extractor** as a feature map \(\Phi: \mathcal{X} \to \mathbb{R}^k\) with stability (Lipschitz) properties.
- We consider a **linear classifier/regressor** on top of \(\Phi\).
- We prove a **Rademacher-complexity-based generalization bound** whose dominant term scales like \(\sqrt{k/n}\), not \(\sqrt{p/n}\).

This shows, in a precise setting, how **the number of effective AQ factors** \(k\) can control generalization.

---

## 1. Setup and Definitions

### 1.1 Data and Hypothesis Class

Let:

- Input space: \(\mathcal{X} \subseteq \mathbb{R}^d\).
- Output space: \(\mathcal{Y} \subseteq \mathbb{R}\) (binary or real-valued; extension to multi-class is standard).
- Data distribution: \(D\) over \(\mathcal{X} \times \mathcal{Y}\).

We assume a **fixed feature extractor** (AQ extractor):
\[
\Phi: \mathcal{X} \to \mathbb{R}^k, \quad x \mapsto z = \Phi(x).
\]

Think of \(z \in \mathbb{R}^k\) as the **vector of AQ activations** for input \(x\).

We consider a **linear predictor** on top of \(\Phi\):
\[
 f_w(x) := \langle w, \Phi(x) \rangle, \quad w \in \mathbb{R}^k.
\]

Define the hypothesis class:
\[
 \mathcal{F} := \{ f_w : x \mapsto \langle w, \Phi(x) \rangle \mid \|w\|_2 \le B_w \}.
\]

This is standard in margin-based generalization theory: we restrict the norm of \(w\) to control capacity.

### 1.2 AQ Stability

We capture "consistent AQ extraction" via **Lipschitz stability** of \(\Phi\).

**Definition 1 (AQ Stability)**:  
The feature map \(\Phi\) is \(L\)-Lipschitz if for all \(x_1, x_2 \in \mathcal{X}\):
\[
 \|\Phi(x_1) - \Phi(x_2)\|_2 \le L \cdot \|x_1 - x_2\|_2.
\]

Intuitively:
- Similar inputs (in input norm) have similar AQ vectors.
- AQ extraction is **stable**, not chaotic.

We also assume the AQ vectors are **bounded**:

**Assumption A1 (Bounded AQ Features)**:  
There exists \(R > 0\) such that for all \(x\):
\[
 \|\Phi(x)\|_2 \le R.
\]

This is standard and can be enforced by normalization or architecture design.

### 1.3 Loss Function

Let \(\ell: \mathbb{R} \times \mathcal{Y} \to [0, 1]\) be a **Lipschitz loss** in its first argument (prediction), e.g.:

- Squared loss (clipped),
- Logistic loss,
- Hinge loss (with appropriate normalization).

Assume \(\ell\) is \(L_\ell\)-Lipschitz in its first argument:
\[
 |\ell(f, y) - \ell(g, y)| \le L_\ell |f - g| \quad \forall f, g \in \mathbb{R}, \; y \in \mathcal{Y}.
\]

### 1.4 Risk and Empirical Risk

For \(f \in \mathcal{F}\), define:

- True (expected) risk:
  \[
  \mathcal{R}(f) := \mathbb{E}_{(x,y) \sim D}[\ell(f(x), y)].
  \]
- Empirical risk on sample \(S = \{(x_i, y_i)\}_{i=1}^n\):
  \[
  \hat{\mathcal{R}}_S(f) := \frac{1}{n} \sum_{i=1}^n \ell(f(x_i), y_i).
  \]

We are interested in bounding \(\mathcal{R}(f) - \hat{\mathcal{R}}_S(f)\) uniformly over \(f \in \mathcal{F}\), in terms of \(k\), \(R\), \(B_w\), and \(n\).

---

## 2. Rademacher Complexity of AQ-Based Linear Predictors

We recall a standard result: the **empirical Rademacher complexity** of a class of linear functions with bounded weights and bounded features.

### 2.1 Empirical Rademacher Complexity

Given a fixed sample \(S = \{x_i\}_{i=1}^n\), the empirical Rademacher complexity of \(\mathcal{F}\) is:
\[
 \hat{\mathfrak{R}}_S(\mathcal{F}) := \mathbb{E}_\sigma \left[ \sup_{f \in \mathcal{F}} \frac{1}{n} \sum_{i=1}^n \sigma_i f(x_i) \right],
\]
where \(\sigma_i\) are i.i.d. Rademacher random variables (\(\mathbb{P}(\sigma_i = +1) = \mathbb{P}(\sigma_i = -1) = 1/2\)).

Substitute \(f_w(x) = \langle w, \Phi(x) \rangle\):
\[
 \hat{\mathfrak{R}}_S(\mathcal{F}) = \mathbb{E}_\sigma \left[ \sup_{\|w\|_2 \le B_w} \frac{1}{n} \sum_{i=1}^n \sigma_i \langle w, \Phi(x_i) \rangle \right].
\]

By linearity:
\[
 \frac{1}{n} \sum_{i=1}^n \sigma_i \langle w, \Phi(x_i) \rangle = \left\langle w, \frac{1}{n} \sum_{i=1}^n \sigma_i \Phi(x_i) \right\rangle.
\]

The supremum over \(w\) with \(\|w\|_2 \le B_w\) is given by aligning \(w\) with the direction of the vector inside the inner product:
\[
 \sup_{\|w\|_2 \le B_w} \left\langle w, v \right\rangle = B_w \|v\|_2.
\]

So:
\[
 \hat{\mathfrak{R}}_S(\mathcal{F}) = \mathbb{E}_\sigma \left[ B_w \left\| \frac{1}{n} \sum_{i=1}^n \sigma_i \Phi(x_i) \right\|_2 \right] = B_w \, \mathbb{E}_\sigma \left[ \left\| \frac{1}{n} \sum_{i=1}^n \sigma_i \Phi(x_i) \right\|_2 \right].
\]

Using Jensen and the bound \(\|\Phi(x_i)\|_2 \le R\), we obtain the standard inequality:
\[
 \hat{\mathfrak{R}}_S(\mathcal{F}) \le \frac{B_w R}{\sqrt{n}}.
\]

This bound is **independent of input dimension d** and depends only on the **AQ feature radius R**, the **weight norm bound B_w**, and \(n\).

If we further assume that each coordinate of \(\Phi(x)\) is individually bounded by \(r\) (i.e., \(|\Phi_j(x)| \le r\)), one can also derive a bound with explicit \(\sqrt{k}\) dependence. A simple variant:

- \(\|\Phi(x)\|_2 \le r \sqrt{k}\) implies \(R \le r \sqrt{k}\).
- Then:
  \[
  \hat{\mathfrak{R}}_S(\mathcal{F}) \le \frac{B_w r \sqrt{k}}{\sqrt{n}}.
  \]

This is the form that makes the \(\sqrt{k/n}\) dependence explicit.

---

## 3. Generalization Bound in Terms of k

We now state a standard Rademacher-complexity generalization bound for Lipschitz loss.

### Theorem 05.1 (Generalization via AQ Stability)

Let \(\mathcal{F}\) and \(\ell\) be as in Section 1. Suppose:

1. **Bounded AQ features**: \(\|\Phi(x)\|_2 \le R \le r \sqrt{k}\) for all \(x\).
2. **Weight norm bound**: \(\|w\|_2 \le B_w\) for all \(f_w \in \mathcal{F}\).
3. **Lipschitz loss**: \(\ell\) is \(L_\ell\)-Lipschitz in its first argument and bounded in \([0,1]\).

Then for any \(\delta > 0\), with probability at least \(1 - \delta\) over the draw of an i.i.d. sample \(S\) of size \(n\), the following holds for all \(f \in \mathcal{F}\):
\[
 \mathcal{R}(f) \le \hat{\mathcal{R}}_S(f) + 2 L_\ell \hat{\mathfrak{R}}_S(\mathcal{F}) + 3 \sqrt{\frac{\log(2/\delta)}{2n}}.
\]

Using the bound \(\hat{\mathfrak{R}}_S(\mathcal{F}) \le B_w r \sqrt{k / n}\), we get:
\[
 \mathcal{R}(f) \le \hat{\mathcal{R}}_S(f) + 2 L_\ell B_w r \sqrt{\frac{k}{n}} + 3 \sqrt{\frac{\log(2/\delta)}{2n}}.
\]

So the **generalization gap** scales as:
\[
 \mathcal{R}(f) - \hat{\mathcal{R}}_S(f) = O\left( \sqrt{\frac{k}{n}} \right).
\]

This is the precise version of "sample complexity depends on k, not p" in this setting.

### Proof Sketch

The proof is a straightforward application of the standard Rademacher-complexity generalization bound (e.g., Bartlett & Mendelson, 2002):

1. Use Lipschitz composition: the Rademacher complexity of the loss-composed class \(\ell \circ \mathcal{F}\) is bounded by \(L_\ell \hat{\mathfrak{R}}_S(\mathcal{F})\).
2. Use the standard inequality (for bounded loss):
   \[
   \mathcal{R}(f) \le \hat{\mathcal{R}}_S(f) + 2 \mathfrak{R}_n(\ell \circ \mathcal{F}) + 3 \sqrt{\frac{\log(2/\delta)}{2n}},
   \]
   where \(\mathfrak{R}_n\) is the expectation of \(\hat{\mathfrak{R}}_S\) over samples.
3. Replace \(\mathfrak{R}_n(\ell \circ \mathcal{F})\) by \(L_\ell \mathfrak{R}_n(\mathcal{F})\) and use the empirical bound \(\hat{\mathfrak{R}}_S(\mathcal{F}) \le B_w r \sqrt{k/n}\) plus symmetrization arguments.

All of these steps are standard; the only AQ-specific part is that **we interpret \(k\) as the number of AQ dimensions** in \(\Phi(x)\).

---

## 4. Interpretation: Generalization = Consistent AQ Extraction

Within this framework, we can interpret the generalization bound as follows:

- The **AQ extractor** \(\Phi\) maps high-dimensional inputs \(x\) to a **k-dimensional AQ space**.
- The **classifier** only depends on these \(k\) coordinates, with bounded weight norm.
- The **capacity** of the classifier class is controlled by \(k\), \(R\), and \(B_w\), not by the total number of parameters in the underlying deep network that implements \(\Phi\).

If \(\Phi\) is **stable** (Lipschitz) and **compressive** (small \(k\)), then:

- Similar inputs map to similar AQ vectors (consistent extraction).
- The linear classifier on top has low Rademacher complexity.
- Thus, **generalization is good** even if the underlying network has many parameters.

This makes precise the idea:

> "What matters for generalization is the structure and stability of the AQ space (k, R, L), not the raw parameter count p."

---

## 5. Limitations and Relation to Full AQ Theory

### 5.1 Strong Simplifications

This theorem is deliberately narrow:

- It assumes a **fixed** AQ extractor \(\Phi\); we do not analyze the training of \(\Phi\) itself.
- It uses a **linear** classifier on top of \(\Phi\); real models have deep heads and non-linearities.
- It uses standard Lipschitz and norm-boundedness assumptions, rather than the full AQ structure (magnitude, phase, frequency, coherence).

Nevertheless, it shows that **once you have a stable k-dimensional AQ representation**, standard learning theory gives **k-based generalization bounds**.

### 5.2 Toward Stronger AQ-Specific Results

To connect this more tightly to the full AQ framework, future work would:

1. Incorporate **AQ properties** explicitly: treat magnitude, phase, frequency, coherence as structured coordinates with possibly different norms and Lipschitz constants.
2. Analyze the training of \(\Phi\) itself: show that SGD plus architectural biases (attention, bottlenecks, sparsity) tend to produce low-k, stable AQ representations.
3. Connect \(k\) to **task complexity** \(K(T)\) as in Theorem 04: show that under appropriate regularization, \(k \approx K(T)\).
4. Consider **multi-head and multi-layer** AQ structures, where effective \(k\) may vary by layer and head.

These steps would move from "if you have AQ, generalization depends on k" to "training naturally produces a small k, so generalization follows."

---

## 6. Summary

> **Theorem 05 (Generalization via AQ Stability)**  
> For a hypothesis class consisting of linear predictors on top of a fixed, Lipschitz, k-dimensional AQ feature map \(\Phi\), with bounded feature norm and bounded weight norm, the uniform generalization gap is bounded by
> \[
> \mathcal{R}(f) - \hat{\mathcal{R}}_S(f) = O\left( L_\ell B_w r \sqrt{\frac{k}{n}} + \sqrt{\frac{\log(1/\delta)}{n}} \right).
> \]
> Thus, in this stylized setting, **sample complexity depends on the number of AQ dimensions k**, not on the raw parameter count p of the underlying network that implements \(\Phi\).

This makes the "generalization = consistent AQ extraction" idea precise in a standard statistical-learning-theory framework.

---

**Oscar Goldman — Shogu Research Group @ Datamutant.ai**

*This theorem shows how, once an input is compressed into a stable AQ space of dimension k, classical generalization bounds become functions of k. The broader AQ theory then asks: why and how does training produce such a space in the first place?*
