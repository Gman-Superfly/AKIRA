# Theorem 06 – AQ as Task-Optimal Compression (Stylized Rate–Distortion)

## 0. Purpose and Scope

This document gives a **stylized information-theoretic theorem** that makes precise a limited version of the claim:

> "Action Quanta are task-relevant compression; they achieve an optimal rate–distortion tradeoff for actionable information."

We do **not** prove a full, general rate–distortion theorem for arbitrary tasks and architectures. Instead, we:

- Consider a **simple supervised task** \(Y\) from inputs \(X\),
- Define a representation \(R = \Phi(X)\),
- Use standard information theory (data processing, Fano's inequality) to show:
  - Any representation that allows **perfect decisions** (zero task distortion) must have **mutual information** \(I(X;R) \ge H(Y)\),
  - This lower bound is **attained** by a minimal sufficient representation (e.g., \(R = Y\)), so it is **rate–optimal** for \(D = 0\),
  - With non-zero error \(D = P(\hat Y \ne Y)\), there is a corresponding **information lower bound** \(I(X;R) \ge R(D)\) of standard form.

In this sense, a representation that corresponds to a **minimal sufficient statistic** for the task is a natural mathematical stand-in for an **AQ representation**: it is the **shortest code** that still supports correct (or near-correct) action.

---

## 1. Setup

### 1.1 Variables and Channels

Let:

- \(X\) be the input random variable (e.g., sensor data, tokens),
- \(Y\) be the task label (e.g., class, decision), taking values in a finite set \(\mathcal{Y}\) with \(|\mathcal{Y}| = M\),
- \(R = \Phi(X)\) be a **representation** (code) extracted from \(X\),
- \(\hat Y = g(R)\) be the **decision** made from \(R\).

We assume a Markov chain:
\[
 Y \to X \to R \to \hat Y.
\]

That is, \(R\) is a (possibly stochastic) function of \(X\), and \(\hat Y\) is a function of \(R\).

We are interested in the tradeoff between:

- The **rate** (information content) of the representation \(R\), quantified by \(I(X;R)\),
- The **distortion** in task performance, here measured via classification error.

### 1.2 Task Distortion

Define the **task distortion** as the misclassification probability:
\[
 D := \mathbb{P}(\hat Y \ne Y).
\]

- **Zero distortion**: \(D = 0\) (perfect decisions).
- **Non-zero distortion**: \(D > 0\) (some errors allowed).

We ask: **How small can \(I(X;R)\) be, given that we must keep \(D\) below some target?**

This is exactly a **rate–distortion** question, but with distortion defined in the **task space** (error on \(Y\)), not reconstruction error in \(X\)-space.

---

## 2. Zero-Distortion Case: Minimal Bits = H(Y)

We first treat the simplest case: **perfect decisions**, \(D = 0\).

### 2.1 Perfect Decisions and Sufficiency

If \(D = 0\), then \(\hat Y = Y\) almost surely. Since \(\hat Y\) is a function of \(R\), this means that **Y is a function of R**:
\[
 H(Y \mid R) = 0.
\]

We then have:
\[
 I(Y;R) = H(Y) - H(Y \mid R) = H(Y).
\]

By the **data processing inequality** (DPI) applied to \(Y \to X \to R\), we know:
\[
 I(X;R) \ge I(Y;R) = H(Y).
\]

Thus:

> **Proposition 6.1 (Zero-Distortion Lower Bound)**  
> For any representation \(R = \Phi(X)\) that permits perfect decisions (\(H(Y \mid R) = 0\)), the mutual information between \(X\) and \(R\) satisfies
> \[
> I(X;R) \ge H(Y).
> \]

This is a **hard lower bound**: no representation that supports perfect performance can carry less than \(H(Y)\) bits of information about \(X\). In other words, you cannot compress below \(H(Y)\) bits without losing the ability to predict \(Y\) perfectly.

### 2.2 Achievability

The lower bound is **tight**: there exists at least one representation that achieves it.

- Take \(R = Y\) itself (i.e., encode only the label, and nothing else).
- Then \(H(Y \mid R) = 0\) trivially.
- Also, since \(R\) is a deterministic function of \(Y\) and vice versa,
  \[
  I(X;R) = I(X;Y).
  \]

If we further assume \(Y\) is a deterministic function of \(X\) (no label noise), we have \(H(Y \mid X) = 0\), so:
\[
 I(X;Y) = H(Y) - H(Y \mid X) = H(Y).
\]

Thus:

> **Proposition 6.2 (Zero-Distortion Achievability)**  
> If the task label \(Y\) is a deterministic function of \(X\), then the representation \(R = Y\) achieves both:
> - Perfect decisions (\(D = 0\)), and
> - Minimal mutual information \(I(X;R) = H(Y)\),
> thereby **achieving the lower bound** in Proposition 6.1.

This is a simple but important fact: the minimal number of bits needed to support perfect decisions is exactly **the entropy of the labels**, \(H(Y)\). No representation with fewer bits can possibly encode all task-relevant distinctions.

In the AQ language, a representation that is **equivalent to Y** (or to a minimal sufficient statistic for Y) is an **optimal action code** at zero distortion.

---

## 3. Non-Zero Distortion: Fano-Type Lower Bound

In practice, we often tolerate **non-zero error** \(D > 0\). We can then ask: how does this relax the information requirement?

### 3.1 Fano's Inequality

For a classifier \(\hat Y\) based on \(R\), Fano's inequality states:
\[
 H(Y \mid R) \le H_b(D) + D \log(M-1),
\]
where:

- \(M = |\mathcal{Y}|\) is the number of classes,
- \(H_b(D) = -D \log D - (1-D) \log(1-D)\) is the **binary entropy** function.

Using \(I(Y;R) = H(Y) - H(Y \mid R)\), we get:
\[
 I(Y;R) \ge H(Y) - H_b(D) - D \log(M-1).
\]

Again using DPI, \(Y \to X \to R\):
\[
 I(X;R) \ge I(Y;R).
\]

Therefore:

> **Proposition 6.3 (Fano-Type Lower Bound on I(X;R))**  
> For any representation \(R\) and classifier \(\hat Y\) with misclassification probability \(D = \mathbb{P}(\hat Y \ne Y)\),
> \[
> I(X;R) \ge H(Y) - H_b(D) - D \log(M-1).
> \]

This is a **task-specific rate lower bound**: to achieve error \(D\), any representation must carry at least this many bits of information about \(X\).

### 3.2 Interpretation as Rate–Distortion Lower Bound

In classical rate–distortion theory, the **rate–distortion function** \(R(D)\) is defined as:
\[
 R(D) = \inf_{P(R \mid X): \mathbb{E}[d(Y, \hat Y)] \le D} I(X;R),
\]
where the infimum runs over all channels \(P(R \mid X)\) and decoders \(P(\hat Y \mid R)\) that achieve expected distortion \(D\).

Proposition 6.3 shows that, for the 0–1 loss on \(Y\), we have a **universal lower bound**:
\[
 R(D) \ge H(Y) - H_b(D) - D \log(M-1).
\]

In many symmetric cases (e.g., uniform \(Y\), symmetric channel), this lower bound is **tight** and equals the true rate–distortion function. We do not prove tightness here, but we note:

- The functional form \(H(Y) - H_b(D) - D \log(M-1)\) is the standard information-theoretic lower bound for classification with error \(D\).
- It reduces to \(H(Y)\) when \(D = 0\) (since \(H_b(0) = 0\)).

Thus, in the non-zero distortion case, we can say:

> Any representation \(R\) that supports decisions with error \(D\) must have rate \(I(X;R)\) at least on the order of \(R(D)\), where \(R(D)\) is bounded below by \(H(Y) - H_b(D) - D \log(M-1)\).

---

## 4. Connecting to AQ: Minimal Actionable Representations

In the AQ framework, we are not interested in reconstructing \(X\); we are interested in **supporting correct action** on \(Y\). The above results can be interpreted as follows:

1. **Zero-distortion AQ representation**:
   - Any representation that allows perfect decisions must have \(I(X;R) \ge H(Y)\).
   - A minimal sufficient representation (e.g. \(R = Y\) or a minimal sufficient statistic for Y) **achieves** this lower bound.
   - Therefore, such a representation is **compression-optimal** for the task: no smaller code (in bits) can support perfect action.

2. **Non-zero distortion AQ representation**:
   - If we tolerate error \(D\), we can in principle compress further, but only down to \(R(D)\), the task-relevant rate–distortion bound.
   - An ideal AQ representation at distortion \(D\) would be one that **achieves** this bound.

3. **Action Quanta as units of task-relevant rate**:
   - If each AQ is associated (on average) with a fixed amount of mutual information about \(Y\), then the **number of AQ** required corresponds to the task rate \(R(D)\).
   - In the simplest case (zero distortion), the number of AQ needed is at least sufficient to encode \(H(Y)\) bits.

Thus, in a stylized sense:

> A representation that consists of minimal, irreducible AQ (in the sense of Theorem 04) and just suffices to determine \(Y\) (zero or low \(D\)) is **information-theoretically optimal** among all representations achieving the same task distortion.

---

## 5. Limitations and Directions for Stronger Results

### 5.1 What Is Proven vs. Assumed

What is **rigorously proven** here (using standard information theory):

- Any representation used for a \(M\)-class decision problem with error \(D\) must satisfy \(I(X;R) \ge H(Y) - H_b(D) - D \log(M-1)\).
- In the special case \(D = 0\), this reduces to \(I(X;R) \ge H(Y)\), and there exist representations (e.g. \(R = Y\)) that achieve this exactly.

What is **assumed or left as conjecture** for AQ specifically:

- That the **AQ representation** found by a trained model is (approximately) a minimal sufficient statistic for \(Y\) or for downstream actions.
- That each AQ carries a roughly constant "quantum" of task-relevant information, so that AQ count \(k\) tracks \(R(D)\).
- That, in more complex tasks (beyond classification), an analogous rate–distortion picture holds with appropriate distortion measures.

### 5.2 Toward a Full AQ Rate–Distortion Theory

A full **AQ rate–distortion theory** would:

1. Define a **task distortion** \(d_T(x, R)\) capturing how much worse actions based on \(R\) are than those based on \(X\).
2. Define the **task rate–distortion function** \(R_T(D)\) as the minimal \(I(X;R)\) for expected distortion \(\le D\).
3. Show that, under learning dynamics and architectural biases, the model converges to representations \(R\) whose AQ structure approximates the optimum of this rate–distortion problem.
4. Connect **AQ properties** (magnitude, phase, frequency, coherence) to how information is packed into the representation and how close it is to \(R_T(D)\).

These are non-trivial and largely open questions; the current theorem should be viewed as a **baseline information-theoretic constraint** that any AQ theory must respect.

---

## 6. Summary

> **Theorem 06 (Stylized AQ Optimal Compression)**  
> For a supervised task \(Y\) from inputs \(X\), any representation \(R = \Phi(X)\) that supports decisions with error \(D\) must satisfy
> \[
> I(X;R) \ge H(Y) - H_b(D) - D \log(|\mathcal{Y}|-1).
> \]
> In particular, for **zero error** (\(D = 0\)), we have \(I(X;R) \ge H(Y)\), and this bound is attainable (e.g. by \(R = Y\)). Therefore, any representation that is sufficient for perfect action and has \(I(X;R) = H(Y)\) is **information-theoretically minimal** for the task: no representation with fewer bits of mutual information can achieve the same performance.
>
> In the AQ framework, such minimal sufficient representations are natural mathematical proxies for **Action Quantum codes**: they are the shortest descriptions of the world that still support correct action.

---

**Oscar Goldman — Shogu Research Group @ Datamutant.ai**

*This theorem shows that the intuitive "AQ as optimal compression" idea has a precise core: for a simple classification task, there is a hard lower bound on how much information any actionable code must carry, and minimal sufficient representations achieve that bound.*
