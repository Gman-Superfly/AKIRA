# EXPERIMENT 020: Cross-Band Information Flow

## How Do Bands Communicate?

**Oscar Goldman — Shogu Research Group @ Datamutant.ai**

---

## Status: PLANNED

## Tier: ★ SUPPORTING

## Depends On: 003 (Spectral Band Dynamics), 012 (Wormhole Activation)

---

## 1. Problem Statement

### 1.1 The Question

SPECTRAL_ATTENTION.md describes wormhole shortcuts between bands:

**How does information flow between spectral bands — does low-freq (WHAT) guide high-freq (WHERE) as predicted?**

### 1.2 Why This Matters

```
THE CROSS-BAND FLOW HYPOTHESIS

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  From SPECTRAL_ATTENTION.md:                                           │
│                                                                         │
│  WHAT → WHERE wormhole (top-down):                                     │
│  • Low-freq structure guides high-freq localization                   │
│  • "I know it's a ring, where exactly is the edge?"                  │
│                                                                         │
│  WHERE → WHAT wormhole (bottom-up):                                    │
│  • High-freq details inform identity at low-freq                      │
│  • "I see this edge pattern, what object is this?"                   │
│                                                                         │
│  Symmetric pairs:                                                       │
│  • Band 0 ↔ Band 6                                                     │
│  • Band 1 ↔ Band 5                                                     │
│  • Band 2 ↔ Band 4                                                     │
│                                                                         │
│  If true: There's a structured communication pattern.                 │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Hypothesis

### 2.1 Primary Hypothesis

**H1: Information flows asymmetrically between bands.**

Low→High dominates early, High→Low dominates later.

### 2.2 Secondary Hypotheses

**H2: Symmetric band pairs have strongest connections.**

**H3: Cross-band flow precedes within-band collapse.**

**H4: Flow direction correlates with task stage.**

### 2.3 Null Hypothesis

**H0:** Cross-band flow is random (no structured pattern).

---

## 3. Scientific Basis

### 3.1 AKIRA Theory Basis

**Relevant Theory Documents:**
- `architecture_expanded/wormhole/WORMHOLE_ARCHITECTURE.md` — §3 (WHAT↔WHERE Flow)
- `architecture_theoretical/SPECTRAL_BELIEF_MACHINE.md` — §4 (Wormhole Attention)
- `architecture_theoretical/ORTHOGONALITY.md` — §4 (Band Independence)

**Key Concepts:**
- **Symmetric pairs:** Band 0↔6, 1↔5, 2↔4 (frequency complements)
- **Top-down flow:** Low-freq (WHAT) guides high-freq (WHERE) - "I know it's a ring, where exactly?"
- **Bottom-up flow:** High-freq (WHERE) informs low-freq (WHAT) - "I see edge pattern, what object?"
- **Band 3 hub:** Connects all bands as central bridge
- **Temporal bridge:** Temporal band connects to all spectral bands

**From WORMHOLE_ARCHITECTURE.md (§3):**
> "WHAT→WHERE: Low-freq encodes object identity. High-freq needs to know 'where to look' for details. Wormhole provides direct path. WHERE→WHAT: High-freq detects local features. Low-freq needs to update global interpretation. Bidirectional information flow."

**From ORTHOGONALITY.md (§4.2):**
> "Bands are orthogonal (Fourier basis). Information cannot leak through spectral domain. Cross-band communication REQUIRES explicit mechanism (wormholes). Without wormholes, bands operate independently."

**This experiment validates:**
1. Whether **information transfer** follows predicted asymmetric pattern
2. Whether **symmetric pairs** have strongest connections (0↔6, 1↔5, 2↔4)
3. Whether **flow direction** changes with processing stage
4. Whether wormholes provide **functional benefit** vs standard attention

**Falsification:** If flow is uniform across all band pairs → symmetric pairing is arbitrary → simplify wormhole structure.

## 3. Methods

### 3.1 Protocol

```
CROSS-BAND FLOW PROTOCOL

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  STEP 1: Measure band-to-band information transfer                     │
│  • Mutual information between bands                                   │
│  • Granger causality (which band predicts which)                     │
│  • Attention flow between bands                                       │
│                                                                         │
│  STEP 2: Analyze temporal structure                                    │
│  • When does Low→High flow peak?                                     │
│  • When does High→Low flow peak?                                     │
│  • Is there a sequence?                                               │
│                                                                         │
│  STEP 3: Test symmetric pair hypothesis                                │
│  • Compare Band 0↔6, 1↔5, 2↔4 connections                            │
│  • Are these stronger than other pairs?                              │
│                                                                         │
│  STEP 4: Correlate with processing stages                              │
│  • Early processing: which direction dominates?                      │
│  • Late processing: which direction dominates?                       │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Results

```
[ TO BE FILLED AFTER EXPERIMENT ]

Cross-band information transfer matrix:
[INSERT 7x7 HEATMAP]

Temporal pattern:
• Low→High peaks at: _____% of processing
• High→Low peaks at: _____% of processing

Symmetric pair strength:
• Band 0↔6: _____
• Band 1↔5: _____
• Band 2↔4: _____
• Average non-symmetric: _____
```

---

## 5. Conclusion

```
[ TO BE FILLED AFTER EXPERIMENT ]

H1 (asymmetric flow): SUPPORTED / NOT SUPPORTED
H2 (symmetric pairs strongest): SUPPORTED / NOT SUPPORTED
H3 (cross-band precedes collapse): SUPPORTED / NOT SUPPORTED
H4 (correlates with task stage): SUPPORTED / NOT SUPPORTED

Cross-band communication is STRUCTURED / RANDOM.
```

---

*Oscar Goldman — Shogu Research Group @ Datamutant.ai*


