# Analysis 03 – Mean-Field Toy Model for Attention Concentration

## 0. Purpose and Scope

**Note on Status**: This document analyzes a stylized toy model. It does NOT prove that real attention exhibits phase transitions or belongs to any universality class. The connection to BEC/Gross-Pitaevskii is an **analogy** that motivates the analysis, not a claim of physical equivalence.

This document gives a **rigorous analysis of a toy model** where attention is viewed as a Gibbs distribution. It does **not** prove anything about full, high-dimensional LLM attention. Instead, it shows:

- A simple order parameter for attention collapse,
- A Landau mean-field free-energy description for that order parameter,
- That **if** one postulates a mean-field coupling, the resulting toy model has the same exponents as standard \(\phi^4\) theory.

This provides a mathematically clean **toy model** that illustrates how mean-field analysis might apply to attention-like systems. It does NOT prove that real attention is mean-field, and the connection to BEC should be understood as an analogy that may or may not hold.

---

## 1. Stylized Attention Model

We consider a **single attention row** with one privileged key and \(n-1\) symmetric background keys.

### 1.1 Logits

Let the logits be:
\[
 u_1 = \Delta, \quad u_j = 0 \quad (j = 2, \dots, n),
\]
where \( \Delta \in \mathbb{R} \) is the logit advantage of the privileged key.

Scaled by \(\beta = 1/\sqrt{d_k}\), the attention probabilities are:
\[
 p_1(\beta) = \frac{e^{\beta \Delta}}{e^{\beta \Delta} + (n-1)}, \quad p_j(\beta) = \frac{1}{e^{\beta \Delta} + (n-1)} \quad (j \ge 2).
\]

This is the minimal non-trivial case where one pattern competes against a symmetric background.

### 1.2 Order Parameter

Define the **order parameter** \(\phi\) as the excess mass on the privileged key relative to uniform:
\[
 \phi(\beta) := p_1(\beta) - \frac{1}{n}.
\]

- \( \phi = 0 \) corresponds to uniform attention (no collapse).
- \( \phi > 0 \) corresponds to a collapsed state where the privileged key dominates.

We will analyze \(\phi\) near a critical point.

---

## 2. Exact Expression and Expansion

From the definition:
\[
 p_1(\beta) = \frac{e^{\beta \Delta}}{e^{\beta \Delta} + (n-1)}.
\]

Let:
\[
 x := e^{\beta \Delta}.
\]

Then:
\[
 p_1 = \frac{x}{x + (n-1)}.
\]

Uniform distribution would assign \(1/n\) to each key. We can write:
\[
 p_1 = \frac{1}{n} + \phi.
\]

So:
\[
 \phi = \frac{x}{x + (n-1)} - \frac{1}{n}.
\]

Compute explicitly:
\[
 \phi = \frac{nx - (x + n - 1)}{n(x + n - 1)} = \frac{(n-1)(x - 1)}{n(x + n - 1)}.
\]

Thus:
\[
 \phi(\beta) = \frac{(n-1)(e^{\beta \Delta} - 1)}{n(e^{\beta \Delta} + n - 1)}.
\]

We analyze this near a point where the system passes from effectively uniform (\(\phi \approx 0\)) to strongly peaked (\(\phi \approx 1 - 1/n\)).

---

## 3. Small-φ Expansion and Landau Form

For **small** \(\beta \Delta\), expand \(e^{\beta \Delta}\):
\[
 e^{\beta \Delta} = 1 + \beta \Delta + \frac{(\beta \Delta)^2}{2} + O((\beta \Delta)^3).
\]

Substitute into \(\phi(\beta)\):

Numerator:
\[
 (n-1)(e^{\beta \Delta} - 1) = (n-1)\left(\beta \Delta + \frac{(\beta \Delta)^2}{2} + O((\beta \Delta)^3)\right).
\]

Denominator:
\[
 n(e^{\beta \Delta} + n - 1) = n\left(1 + \beta \Delta + \frac{(\beta \Delta)^2}{2} + (n-1) + O((\beta \Delta)^3)\right) = n\left(n + \beta \Delta + \frac{(\beta \Delta)^2}{2} + O((\beta \Delta)^3)\right).
\]

Factor out \(n^2\) in the denominator:
\[
 n(e^{\beta \Delta} + n - 1) = n^2 \left(1 + \frac{\beta \Delta}{n} + O((\beta \Delta)^2)\right).
\]

Thus, to leading orders:
\[
 \phi(\beta) = \frac{(n-1)\left(\beta \Delta + \frac{(\beta \Delta)^2}{2} + O((\beta \Delta)^3)\right)}{n^2 \left(1 + \frac{\beta \Delta}{n} + O((\beta \Delta)^2)\right)}.
\]

Using \(1/(1+a) = 1 - a + O(a^2)\):
\[
 \phi(\beta) = \frac{n-1}{n^2} \left(\beta \Delta + \frac{(\beta \Delta)^2}{2} + O((\beta \Delta)^3)\right) \left(1 - \frac{\beta \Delta}{n} + O((\beta \Delta)^2)\right).
\]

Multiply and keep up to quadratic order in \(\beta \Delta\):
\[
 \phi(\beta) = \frac{n-1}{n^2} \left[\beta \Delta + \frac{(\beta \Delta)^2}{2} - \frac{(\beta \Delta)^2}{n} + O((\beta \Delta)^3)\right].
\]

So:
\[
 \phi(\beta) = a_1 (\beta \Delta) + a_2 (\beta \Delta)^2 + O((\beta \Delta)^3),
\]
with:
\[
 a_1 = \frac{n-1}{n^2}, \quad a_2 = \frac{n-1}{n^2}\left(\frac{1}{2} - \frac{1}{n}\right).
\]

This is a **regular Taylor expansion** in \(\beta \Delta\); there is no singularity at finite \(\beta\) in this finite-n toy model. To obtain non-analytic behavior (true criticality), one must take a limit where \(n \to \infty\) and/or introduce coupling between many such rows.

Nevertheless, we can extract a **Landau-type free energy** in terms of \(\phi\) for small \(\phi\).

---

## 4. Effective Free Energy and Mean-Field Exponents

### 4.1 Effective Free Energy

For a single Gibbs distribution \(p(\beta)\), the (dimensionless) free energy is:
\[
 F(\beta) = -\log Z(\beta).
\]

In terms of \(\phi\), for small \(\phi\) we can write an effective **Landau expansion**:
\[
 F_{\text{eff}}(\phi; t) = F_0 + a \, t \, \phi^2 + b \, \phi^4 + O(\phi^6),
\]
where:

- \( t \) is a reduced control parameter (analogous to \(T - T_c\) or \(\beta - \beta_c\)),
- \( a > 0 \), \( b > 0 \) for stability.

In standard mean-field theory, minimizing \(F_{\text{eff}}\) gives:

- For \( t > 0 \): minimum at \( \phi = 0 \) (symmetric/uniform phase).
- For \( t < 0 \): minima at \( \phi = \pm \sqrt{-a t / (2b)} \) (symmetry-broken phase), implying **order parameter exponent** \(\beta_{\text{MF}} = 1/2\).

### 4.2 Mapping to Attention Parameters

In our toy model:

- The control parameter is effectively \(\beta \Delta\): how strongly the privileged key is favored.
- For small \(\beta \Delta\), \( \phi(\beta) \approx a_1 (\beta \Delta) \), linear.
- To see mean-field behavior, we consider a **population** of such attention rows coupled through a regularization term that penalizes \(\phi^2\) and allows spontaneous symmetry breaking.

A simple construction:

1. Consider many independent copies of the single-row system, indexed by \(r\).
2. Introduce an effective interaction term favoring alignment of \(\phi_r\):
   \[
   F_{\text{int}} = \frac{J}{2} \sum_{r,r'} (\phi_r - \phi_{r'})^2.
   \]
3. In mean-field approximation, all \(\phi_r\) ≈ \(\phi\), leading to an effective quadratic term in \(\phi\) whose sign can change as a function of \(\beta \Delta\).

Under this construction, the standard Landau analysis applies, and the resulting critical exponents are the **mean-field exponents**:

- Order parameter exponent: \(\beta_{\text{MF}} = 1/2\).
- Susceptibility exponent: \(\gamma_{\text{MF}} = 1\).
- Correlation-length exponent: \(\nu_{\text{MF}} = 1/2\).

These are the same exponents as for mean-field BEC and \(\phi^4\) theory.

### 4.3 What Is Actually Proven Here

Within this stylized construction, we have:

- An explicit order parameter \(\phi\) (excess attention on a privileged key).
- A small-\(\phi\) expansion resembling Landau theory.
- A standard mean-field argument showing how coupling many such rows through a quadratic interaction leads to the **same critical exponents** as classical mean-field models (Ising, BEC in GP regime).

The key point for this toy model: **if** one assumes the stated mean-field coupling, **then** the form of the effective free energy matches that of standard mean-field theory, and the toy model would exhibit the same exponents. This does NOT prove real attention belongs to any universality class.

---

## 5. Limitations and Conjecture for Real Attention

### 5.1 What We Have Not Proven

We have **not** shown that:

- Full multi-head, multi-layer attention in large LLMs can be rigorously reduced to this toy model;
- Real networks, with residuals, MLPs, and complex data distributions, exhibit true thermodynamic phase transitions;
- The full system has the same universality class as BEC beyond this stylized mean-field construction.

Those would require:

- A precise mapping from high-dimensional attention dynamics to an effective \(\phi^4\) field theory;
- Taking an appropriate infinite-size limit (tokens, layers, width);
- Computing critical exponents directly from that theory.

### 5.2 Speculative Direction (Not a Conjecture We Endorse)

The toy model suggests one might **hypothesize** that real attention exhibits similar behavior:

> **Speculative Hypothesis**: In some limit, attention pattern concentration might be describable by an effective Landau-type free energy.

However, we **do not endorse this as a conjecture** because:

- The mean-field coupling between attention rows is postulated, not derived
- Real attention operates on finite sequences, not infinite systems
- The analogy to BEC/GP is structural, not physical
- No critical exponents have been measured in real attention

This section describes a **possible direction for future investigation**, not a claim we expect to be validated. The toy model is a mathematical exercise that shows what mean-field behavior would look like if it existed. Whether it does exist in real attention is unknown and may be unknowable.

---

## 6. Summary

> **Analysis 03 (Toy Model for Attention Concentration)**  
> In a simplified attention model with one privileged key and \(n-1\) symmetric keys, the excess attention \(\phi\) on the privileged key can be treated as an order-parameter-like quantity. For small \(\phi\), this quantity admits a Taylor expansion. **If** one postulates a mean-field coupling between many such rows, **then** the resulting system would have the same exponents as standard \(\phi^4\) theory.

**What this analysis shows**: A toy model where mean-field assumptions lead to familiar physics-like behavior.

**What this analysis does NOT show**:
- That real attention exhibits phase transitions
- That attention belongs to any universality class
- That the BEC analogy is more than suggestive

The connection to BEC/GP is an **analogy** that may provide intuition but should not be interpreted as a physical claim. This document is a mathematical exercise, not a theorem about real neural networks.

---

**Oscar Goldman — Shogu Research Group @ Datamutant.ai**

*This analysis explores a toy model where mean-field assumptions would lead to physics-like behavior. It is a mathematical exercise that may or may not have relevance to real attention mechanisms. The BEC analogy is suggestive but unproven, and should be treated as a source of hypotheses rather than established theory.*
