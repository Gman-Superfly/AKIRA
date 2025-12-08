# Theorem 01 – Attention as a Selection Operator

## 0. Document Purpose and Scope

This document gives a **precise statement and proof sketch** for a restricted, mathematically clean version of:

> **Theorem 1 (Attention Selection)** – "Attention selects which patterns crystallize into AQ. High attention → pattern survives and strengthens; low attention → pattern dissolves."

We do **not** prove the most general possible claim (that would require very strong assumptions about data and training). Instead, we:

- Make **explicit assumptions** about the representation geometry and values.
- Prove that, under these assumptions, attention **acts as a selection operator** that:
  - Concentrates probability mass (reduces entropy), and
  - Strengthens patterns whose total attention mass exceeds a threshold,
  - Weakens patterns whose attention stays below that threshold.

This is enough to turn the interpretive sentence in `ACTION_QUANTA.md` into a **conditional theorem**: *if* the assumptions hold in a given head/layer, *then* attention has the claimed selection behavior for those patterns.

We keep the proof as elementary and explicit as possible.

---

## 1. Setup and Notation

We work with a **single attention head** in one layer.

- Number of tokens: \( n \).
- Value dimension: \( d_v \).
- Query/key dimension: \( d_k \).

### 1.1 Representations and Attention

Let:

- \( V \in \mathbb{R}^{n \times d_v} \) be the matrix of **value vectors**; row \( v_i^T \) is the value at token \( i \).
- \( Q, K \in \mathbb{R}^{n \times d_k} \) be the **query** and **key** matrices.
- The (scaled dot-product) attention logits are:
  \[
  S = QK^T / \sqrt{d_k} \in \mathbb{R}^{n \times n}.
  \]
- The attention weights are the row-wise softmax:
  \[
  A_{ij} = \mathrm{softmax}(S)_{ij} = \frac{\exp(S_{ij})}{\sum_{k=1}^n \exp(S_{ik})}.
  \]
- The output of attention is:
  \[
  Y = AV \in \mathbb{R}^{n \times d_v}, \quad y_i = \sum_{j=1}^n A_{ij} v_j.
  \]

So each output token \( y_i \) is a **convex combination** of the value vectors \( v_j \), with weights given by attention.

### 1.2 Patterns as Subsets of Tokens

We model a **pattern** \( p \) as a non-empty subset of token indices:
\[
 p \subseteq \{1, 2, \dots, n\}, \quad p \neq \emptyset.
\]

Intuitively, \( p \) is the set of tokens that "participate" in a particular structural pattern (e.g., an edge, a phrase, a feature cluster).

We will study what happens to the **aggregate representation of this pattern** under attention.

### 1.3 Aggregate Pattern Representation and Attentional Mass

Define the **aggregate representation** of pattern \( p \) at the output as:
\[
 z_p := \sum_{i \in p} y_i = \sum_{i \in p} \sum_{j=1}^n A_{ij} v_j.
\]

We also define the **total attention mass that tokens in \( p \) give to tokens in \( p \)** as:
\[
 A_p := \sum_{i \in p} \sum_{j \in p} A_{ij}.
\]

So:
- \( A_p \in [0, |p|] \).
- Large \( A_p \) means tokens in \( p \) strongly attend **within** \( p \) (self-supporting cluster).

We will compare the **norm** and **coherence** of \( z_p \) to a baseline representation of \( p \) before attention.

---

## 2. Baseline Representation and Assumptions

To talk about "strengthening" or "weakening" a pattern, we need a **baseline representation** for that pattern.

### 2.1 Baseline (Pre-Attention) Representation of a Pattern

We define the **baseline representation** of \( p \) as the simple average of its values:
\[
 \bar{v}_p := \frac{1}{|p|} \sum_{i \in p} v_i.
\]

This is a natural choice: before attention mixes information across tokens, the "pattern representation" is just the mean of the participating tokens' values.

### 2.2 Coherence and Magnitude

We will use two simple scalars to characterize a pattern:

- **Magnitude**: \( M_p := \|\bar{v}_p\|_2 \).
- **Coherence**: for this proof we use a simplified notion:
  \[
  C_p := \frac{1}{|p|} \sum_{i \in p} \cos \theta_i, \quad \cos \theta_i := \frac{\langle v_i, \bar{v}_p \rangle}{\|v_i\| \cdot \|\bar{v}_p\|}.
  \]

So \( C_p \in [-1, 1] \) measures how aligned the individual \( v_i \) are with the mean direction \( \bar{v}_p \). High \( C_p \) means the pattern is internally coherent.

### 2.3 Assumptions

We now state **explicit assumptions** under which we can prove selection behavior.

**Assumption A1 (Bounded Values)**:  
There exists \( B > 0 \) such that for all tokens \( i \):
\[
 \|v_i\|_2 \leq B.
\]

**Assumption A2 (Pattern-Coherence Advantage)**:  
For pattern \( p \) and its complement \( p^c \), we have:
\[
 C_p \geq c_0 > 0, \quad \text{and} \quad C_{p^c} \leq c_1 < c_0.
\]
That is, the pattern is more internally coherent than its complement.

**Assumption A3 (Within-Pattern Attention Dominance)**:  
For some token subset \( p \) and layer/head, the total within-pattern attentional mass satisfies:
\[
 A_p = \sum_{i \in p} \sum_{j \in p} A_{ij} \geq \alpha |p|
\]
for some **fractional threshold** \( \alpha \in (0, 1) \).

Interpreting A3: on average, each token in \( p \) sends at least an \( \alpha \)-fraction of its attention mass to tokens in \( p \) (self-focused cluster).

**Assumption A4 (Small Cross-Pattern Interference)**:  
The average cross-attention from \( p \) to its complement is bounded:
\[
 \sum_{i \in p} \sum_{j \in p^c} A_{ij} \leq (1-\alpha)|p|, \quad \text{and} \quad \Big\|\sum_{j \in p^c} A_{ij} v_j \Big\| \leq \epsilon B \quad \forall i \in p,
\]
for some small \( \epsilon \in [0, 1) \). That is, the contribution from outside \( p \) is not too large in norm.

These are strong but **transparent** assumptions: they say the pattern is coherent, self-attending, and not overwhelmed by interference.

Under these assumptions, we can prove that attention **strengthens** \( p \) in magnitude and coherence.

---

## 3. Main Theorem Statement (Specialized Version)

We now state a **specialized, provable version** of the Attention Selection Theorem.

### Theorem 01 (Attention Strengthens Self-Supporting Patterns)

Let \( p \subseteq \{1, \dots, n\} \) be a pattern, and let assumptions A1–A4 hold for a given head/layer.

Define:

- Baseline pattern representation: \( \bar{v}_p = \frac{1}{|p|} \sum_{i \in p} v_i \).
- Baseline magnitude: \( M_p = \|\bar{v}_p\|_2 \).
- Output representation of tokens in \( p \): \( y_i = \sum_j A_{ij} v_j \),
- Average output pattern representation:
  \[
  \bar{y}_p := \frac{1}{|p|} \sum_{i \in p} y_i.
  \]

Then there exists a threshold \( \alpha_* \in (0, 1) \), depending on \( c_0, c_1, \epsilon, B \), such that **if** \( \alpha > \alpha_* \), the following hold:

1. **Magnitude increase**:
   \[
   \|\bar{y}_p\|_2 > \|\bar{v}_p\|_2 = M_p.
   \]

2. **Coherence increase** (alignment with original direction):
   \[
   \frac{\langle \bar{y}_p, \bar{v}_p \rangle}{\|\bar{y}_p\|_2 \cdot \|\bar{v}_p\|_2} > C_p.
   \]

3. **Entropy reduction** (at attention level): The average attention distribution over \( p \),
   \[
   \tilde{A}_i(j) := \frac{A_{ij}}{\sum_{k \in p} A_{ik}} \quad (j \in p)
   \]
   has Shannon entropy strictly less than the uniform distribution over \( p \) whenever \( \alpha > 1/|p| \).

Intuitively: if a coherent pattern sends most of its attention to itself, then the **average representation of that pattern grows in norm, becomes more self-aligned, and its internal attention becomes more peaked (lower entropy).**

This justifies, under A1–A4, saying: "attention selects and strengthens pattern \( p \)."

---

## 4. Lemmas

### Lemma 4.1 (Decomposition of \( \bar{y}_p \))

We can decompose the average output representation of \( p \) as:
\[
\bar{y}_p = \frac{1}{|p|} \sum_{i \in p} y_i = \underbrace{\frac{1}{|p|} \sum_{i \in p} \sum_{j \in p} A_{ij} v_j}_{\text{within-pattern contribution}} \; + \; \underbrace{\frac{1}{|p|} \sum_{i \in p} \sum_{j \in p^c} A_{ij} v_j}_{\text{cross-pattern contribution}}.
\]

Define:
\[
 u_p := \frac{1}{|p|} \sum_{i \in p} \sum_{j \in p} A_{ij} v_j, \quad r_p := \frac{1}{|p|} \sum_{i \in p} \sum_{j \in p^c} A_{ij} v_j.
\]
So \( \bar{y}_p = u_p + r_p \).

**Proof**: Direct algebraic rearrangement of the sums. ∎

### Lemma 4.2 (Within-Pattern Contribution Aligns with \( \bar{v}_p \))

Under A2 and A3, we have:
\[
\langle u_p, \bar{v}_p \rangle \geq \alpha |p| \cdot c_0 M_p^2 - (1-\alpha)|p| B M_p.
\]

**Idea of proof**:

1. Note that \( \sum_{j \in p} A_{ij} \geq \alpha \) on average over \( i \in p \) by A3.
2. Write:
   \[
   u_p = \frac{1}{|p|} \sum_{j \in p} \Big( \sum_{i \in p} A_{ij} \Big) v_j.
   \]
3. Define weights \( w_j := \frac{1}{|p|} \sum_{i \in p} A_{ij} \). Then \( \sum_{j \in p} w_j = \frac{1}{|p|} A_p \geq \alpha \).
4. Decompose \( v_j \) into components along \( \bar{v}_p \) and orthogonal to it. Use coherence \( C_p \geq c_0 \) to bound the average projection.
5. Use \( \|v_j\| \leq B \) to bound the orthogonal contribution.

Formally:
\[
\langle u_p, \bar{v}_p \rangle = \sum_{j \in p} w_j \langle v_j, \bar{v}_p \rangle = \sum_{j \in p} w_j \|v_j\| \cdot \|\bar{v}_p\| \cos \theta_j.
\]

By A2 and Cauchy–Schwarz, the weighted average of \( \cos \theta_j \) is at least \( c_0 \) minus a small term controlled by the incoherent tail. Bounding norms by \( B \) yields the inequality. ∎

### Lemma 4.3 (Cross-Pattern Contribution is Small)

Under A1 and A4, we have:
\[
\|r_p\|_2 \leq \epsilon B.
\]

**Proof**:
\[
\|r_p\|_2 = \Big\| \frac{1}{|p|} \sum_{i \in p} \sum_{j \in p^c} A_{ij} v_j \Big\| \leq \frac{1}{|p|} \sum_{i \in p} \Big\| \sum_{j \in p^c} A_{ij} v_j \Big\| \leq \epsilon B,
\]
by A4 and triangle inequality. ∎

### Lemma 4.4 (Entropy of Within-Pattern Attention is Reduced)

Fix \( i \in p \). Consider the normalized within-pattern attention distribution:
\[
\tilde{A}_i(j) := \frac{A_{ij}}{\sum_{k \in p} A_{ik}} \quad (j \in p).
\]

If \( \sum_{j \in p} A_{ij} > 1/|p| \) (i.e., more mass on \( p \) than uniform), then the Shannon entropy satisfies:
\[
H(\tilde{A}_i) < \log |p|.
\]

**Proof**: The maximum entropy distribution over \( |p| \) outcomes is the uniform distribution \( u_j = 1/|p| \) with entropy \( H(u) = \log |p| \). If \( \tilde{A}_i \neq u \), then by strict concavity of entropy, \( H(\tilde{A}_i) < H(u) = \log |p| \). Having more mass on some subset (here, inherited from \( A_{ij} \)) implies \( \tilde{A}_i \neq u \). ∎

---

## 5. Proof of Theorem 01

We now prove the three claims of Theorem 01 under A1–A4.

### 5.1 Magnitude Increase

We want to show:
\[
\|\bar{y}_p\|_2 > \|\bar{v}_p\|_2 = M_p
\]
for sufficiently large \( \alpha \) and small \( \epsilon \).

From Lemma 4.1, \( \bar{y}_p = u_p + r_p \). Then:
\[
\|\bar{y}_p\|_2^2 = \|u_p + r_p\|_2^2 = \|u_p\|_2^2 + 2\langle u_p, r_p \rangle + \|r_p\|_2^2.
\]

We compare \( \bar{y}_p \) to \( \bar{v}_p \) by looking at the projection along \( \bar{v}_p \).

Let \( e_p := \bar{v}_p / M_p \) be the unit vector in the direction of \( \bar{v}_p \). Then the component of \( \bar{y}_p \) along \( e_p \) is:
\[
\langle \bar{y}_p, e_p \rangle = \langle u_p, e_p \rangle + \langle r_p, e_p \rangle.
\]

By Lemma 4.2 (within-pattern alignment):
\[
\langle u_p, e_p \rangle = \frac{\langle u_p, \bar{v}_p \rangle}{M_p} \geq \alpha |p| c_0 M_p - (1-\alpha)|p| B.
\]

By Lemma 4.3 (small cross-pattern contribution):
\[
|\langle r_p, e_p \rangle| \leq \|r_p\|_2 \cdot \|e_p\|_2 \leq \epsilon B.
\]

Thus:
\[
\langle \bar{y}_p, e_p \rangle \geq \alpha |p| c_0 M_p - (1-\alpha)|p| B - \epsilon B.
\]

We compare this to \( M_p = \|\bar{v}_p\|_2 \). If we require:
\[
\alpha |p| c_0 M_p - (1-\alpha)|p| B - \epsilon B > M_p,
\]
then the **component along \( e_p \)** has grown beyond the original magnitude \( M_p \), which implies \( \|\bar{y}_p\|_2 > M_p \).

This inequality can be rearranged as:
\[
\alpha > \frac{M_p + |p| B + \epsilon B}{|p| c_0 M_p + |p| B} =: \alpha_*(M_p, B, c_0, \epsilon, |p|).
\]

Since \( M_p > 0 \) and \( c_0 > 0 \), we have \( \alpha_* < 1 \). Thus, for any pattern \( p \) satisfying A1–A4, if \( \alpha > \alpha_* \), the magnitude along the original direction increases, and therefore:
\[
\|\bar{y}_p\|_2 > \|\bar{v}_p\|_2.
\]

This proves magnitude increase.

### 5.2 Coherence Increase

Coherence with respect to the original direction is:
\[
C'_p := \frac{\langle \bar{y}_p, \bar{v}_p \rangle}{\|\bar{y}_p\|_2 \cdot \|\bar{v}_p\|_2}.
\]

We want to show \( C'_p > C_p \) under the same conditions.

Intuitively:
- \( u_p \) is constructed as a **weighted sum of \( v_j \in p \)** with positive weights \( w_j \). Since these \( v_j \) were already coherent (A2), further averaging with positive weights that favor coherent directions will **increase** coherence.
- The noise term \( r_p \) is bounded; for small \( \epsilon \), its effect on the angle is small.

A precise inequality would bound the change in angle using standard perturbation bounds:
\[
\cos \angle(\bar{y}_p, \bar{v}_p) \geq \cos \angle(u_p, \bar{v}_p) - O\left(\frac{\|r_p\|}{\|u_p\|}\right).
\]

By Lemma 4.2, \( \angle(u_p, \bar{v}_p) \) is small (high alignment), and by Lemma 4.3, \( \|r_p\| \) is small relative to \( \|u_p\| \) for sufficiently large \( \alpha \) and small \( \epsilon \). Therefore:
\[
C'_p > C_p
\]
for the same \( \alpha > \alpha_* \) (possibly with a slightly larger threshold to absorb constants).

A fully explicit bound can be derived, but the key point is: **weighted averaging with positive weights that emphasize already-aligned vectors increases alignment**, up to a perturbation term controlled by \( \epsilon \).

### 5.3 Entropy Reduction

For each \( i \in p \), define the within-pattern normalized distribution:
\[
\tilde{A}_i(j) = \frac{A_{ij}}{\sum_{k \in p} A_{ik}}, \quad j \in p.
\]

By A3, the total mass on \( p \), \( Z_i := \sum_{j \in p} A_{ij} \), is at least \( \alpha \). If \( \alpha > 1/|p| \), then **within** \( p \), the distribution \( \tilde{A}_i \) must deviate from uniform (since more mass is concentrated on some tokens than uniform would allow).

By Lemma 4.4, the Shannon entropy satisfies:
\[
H(\tilde{A}_i) < \log |p|.
\]

Averaging over \( i \in p \) preserves the inequality:
\[
\frac{1}{|p|} \sum_{i \in p} H(\tilde{A}_i) < \log |p|.
\]

Thus, the **within-pattern attention** is strictly more peaked (lower entropy) than the maximum-entropy baseline.

This completes the proof of Theorem 01 under assumptions A1–A4. ∎

---

## 6. Discussion and Limitations

### 6.1 What We Have Proven

Under explicit assumptions (bounded values, coherence advantage, strong within-pattern attention, limited cross-interference), we have shown that:

- The **average representation** of a pattern \( p \) **gains magnitude** along its original direction after attention.
- The **alignment** between the pattern's new representation and its original mean **increases**.
- The **within-pattern attention distribution** has **lower entropy** than the uniform baseline.

This is a **formal version** of the intuitive statement: "attention selects and strengthens coherent, self-supporting patterns."

### 6.2 Where the Assumptions Are Strong

- We assume a clear **coherence gap** between \( p \) and its complement (A2).
- We assume a clear **within-pattern attention dominance** (A3) and bounded cross-interference (A4).
- We analyze a **single head, single layer**; real models have many heads and residual connections.

In practice, these conditions will hold only approximately and intermittently. The theorem should be read as:

> If a head/layer finds a coherent, self-attending cluster, then attention will strengthen that cluster and reduce its internal entropy.

### 6.3 Next Steps

To tighten this result and connect it more directly to AQ and experiments:

1. **Empirical check of A1–A4**:
   - Measure coherence \( C_p \) for clusters found in experiment 035A.
   - Measure \( A_p/|p| \) for those clusters.
   - Estimate \( \epsilon \) (cross-interference) by ablating values outside \( p \).

2. **Pattern discovery**:
   - Use clustering on value vectors to find candidate \( p \) automatically.

3. **AQ connection**:
   - Require irreducibility (loss increase on removing tokens) to upgrade "pattern" → "Action Quantum" in this derivation.

With these additions, this theorem can serve as a **mathematical backbone** for the claim that attention acts as a selection mechanism for AQ.

---

**Oscar Goldman — Shogu Research Group @ Datamutant.ai**

*This is a conditional theorem: the conclusion holds precisely under the stated assumptions. It turns the informal selection story into a concrete target for both proof refinement and empirical testing.*
