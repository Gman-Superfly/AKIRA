# EXPERIMENT 012: Wormhole Activation Patterns

## When Do Wormholes Fire?

**Oscar Goldman — Shogu Research Group @ Datamutant.ai**

---

## Status: PLANNED

## Tier: ★ SUPPORTING

## Depends On: 003 (Spectral Band Dynamics), 007 (Wavefront)

---

## 1. Problem Statement

### 1.1 The Question

Wormhole attention enables non-local pattern matching across history. But when does it activate?

**What triggers wormhole connections, and do they follow the spectral hierarchy (low-freq "what" guides high-freq "where")?**

### 1.2 Why This Matters

```
THE WORMHOLE HYPOTHESIS

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  Current implementation:                                               │
│  • Geometric belief (cosine similarity on hypersphere)                │
│  • Energy trigger (fixed threshold τ = 0.92)                         │
│                                                                         │
│  Questions:                                                             │
│  • When does similarity exceed threshold?                             │
│  • What content triggers wormholes?                                   │
│  • Do wormholes follow spectral structure?                           │
│                                                                         │
│  Predicted pattern:                                                     │
│  • Low-freq similarity (category) activates first                    │
│  • Guides high-freq matching (position)                              │
│  • "I know WHAT, wormhole tells me WHERE"                            │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Hypothesis

### 2.1 Primary Hypothesis

**H1: Wormholes activate on content similarity, not just any high similarity.**

Specific patterns trigger wormholes consistently.

### 2.2 Secondary Hypotheses

**H2: Low-freq bands trigger wormholes before high-freq bands.**

**H3: Wormhole activation correlates with belief collapse.**

**H4: Wormholes form "shortcuts" in the belief manifold.**

### 2.3 Null Hypothesis

**H0:** Wormhole activation is random (just noise exceeding threshold).

---

## 3. Scientific Basis

### 3.1 AKIRA Theory Basis

**Relevant Theory Documents:**
- `architecture_expanded/wormhole/WORMHOLE_ARCHITECTURE.md` — §2 (Activation Mechanism)
- `architecture_theoretical/SPECTRAL_BELIEF_MACHINE.md` — §4 (Wormhole Attention), Coherence Gate
- `CANONICAL_PARAMETERS.md` — coherence_threshold = 0.5, gate_sharpness = 10.0

**Key Concepts:**
- **Wormhole pairs:** Symmetric connections (0↔6, 1↔5, 2↔4) + Band 3 hub + Temporal bridge
- **Activation mechanism:** Entropy-based coherence gate, not fixed similarity threshold
- **Coherence gate:** `g = sigmoid((h - coherence_threshold) × gate_sharpness)` where h = normalized entropy
- **Non-local matching:** Structured shortcuts across complementary frequency bands

**From SPECTRAL_BELIEF_MACHINE.md (§4.3 - updated):**
> "Wormholes activate when attention distribution has low entropy (high coherence). Gate value g = sigmoid((h - 0.5) × 10). When g > 0.5, wormhole is open. Low entropy → high coherence → wormhole fires."

**From WORMHOLE_ARCHITECTURE.md:**
> "Complementary pairs enable WHAT↔WHERE communication. Low-freq identifies object (what), high-freq localizes features (where). Band 3 serves as bridge connecting all bands."

**This experiment validates:**
1. Whether wormholes activate on **specific content patterns** (not random noise)
2. Whether **low-freq bands trigger first** (top-down guidance)
3. Whether activation correlates with **belief collapse** (entropy drop)
4. Whether **symmetric pairs** show structured communication

**Falsification:** If activation is random OR all bands trigger uniformly → wormhole structure is cosmetic → simplify to standard attention.

## 3. Methods

### 3.1 Protocol

```
WORMHOLE ACTIVATION PROTOCOL

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  STEP 1: Instrument wormhole layer                                     │
│  • Log all activations (time, position, similarity)                   │
│  • Record what content triggered each                                 │
│                                                                         │
│  STEP 2: Analyze activation patterns                                   │
│  • What stimuli trigger wormholes?                                    │
│  • Which bands activate first?                                        │
│  • Is there temporal structure?                                       │
│                                                                         │
│  STEP 3: Correlate with collapse                                       │
│  • Do wormholes precede collapse?                                     │
│  • Do they accelerate collapse?                                       │
│                                                                         │
│  STEP 4: Map wormhole topology                                         │
│  • Which positions connect to which?                                  │
│  • Is there structure in the connection graph?                       │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Predictions

```
IF THEORY IS CORRECT:

• Wormholes activate on recurring patterns (not noise)
• Low-freq similarity predicts high-freq matching
• Wormhole activation precedes entropy drop
• Connection graph has structure (not random)
```

---

## 4. Results

```
[ TO BE FILLED AFTER EXPERIMENT ]

Wormhole activation rate: _____% of positions
Most common triggers: _____

Band-ordered activation:
[INSERT TIMING ANALYSIS]

Correlation with collapse: r = _____
```

---

## 5. Conclusion

```
[ TO BE FILLED AFTER EXPERIMENT ]

H1 (content-based activation): SUPPORTED / NOT SUPPORTED
H2 (low-freq first): SUPPORTED / NOT SUPPORTED
H3 (correlates with collapse): SUPPORTED / NOT SUPPORTED
H4 (manifold shortcuts): SUPPORTED / NOT SUPPORTED
```

---

*Oscar Goldman — Shogu Research Group @ Datamutant.ai*


