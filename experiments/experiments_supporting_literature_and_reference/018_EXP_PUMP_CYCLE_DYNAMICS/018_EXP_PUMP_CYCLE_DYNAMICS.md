# EXPERIMENT 018: Pump Cycle Dynamics

## Does the Tension-Discharge-Recovery Cycle Exist?


---

## Status: PLANNED

## Tier: ★ SUPPORTING

## Depends On: 002 (Collapse Detection)

---

## 1. Problem Statement

### 1.1 The Question

POMDP_SIM.md describes a "pump cycle" analogous to a relaxation oscillator:

**Does AKIRA exhibit a recurring Tension → Discharge → Recovery cycle in its belief dynamics?**

### 1.2 Why This Matters

```
THE PUMP CYCLE HYPOTHESIS

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  PHASE 1: TENSION (Accumulation)                                       │
│  • Uncertainty spreads outward                                         │
│  • Multiple hypotheses coexist                                         │
│  • Entropy increasing                                                   │
│  • Like charge accumulating in a cloud                                 │
│                                                                         │
│  PHASE 2: DISCHARGE (Collapse)                                          │
│  • One hypothesis wins                                                  │
│  • Error drops suddenly                                                 │
│  • Entropy collapses                                                    │
│  • Like lightning strike                                                │
│                                                                         │
│  PHASE 3: RECOVERY                                                       │
│  • Belief re-spreads                                                    │
│  • New uncertainty for next prediction                                 │
│  • Cycle repeats                                                        │
│                                                                         │
│  If true: This is a fundamental rhythm of belief dynamics.            │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Hypothesis

### 2.1 Primary Hypothesis

**H1: Entropy exhibits a periodic oscillation.**

Tension (entropy up) → Discharge (entropy down) → Recovery (entropy up).

### 2.2 Secondary Hypotheses

**H2: Cycle period is consistent (quasi-periodic).**

**H3: Discharge phase is shorter than tension phase.**

**H4: Cycle correlates with prediction accuracy.**

### 2.3 Null Hypothesis

**H0:** Entropy is random/chaotic (no cycle structure).

---

## 3. Scientific Basis

### 3.1 AKIRA Theory Basis

**Relevant Theory Documents:**
- `pomdp/POMDP_SIM.md`, §5 (Pump Cycle Dynamics)
- `architecture_theoretical/SPECTRAL_BELIEF_MACHINE.md`, §6 (Belief Dynamics)
- `bec/BEC_CONDENSATION_INFORMATION.md`, §4 (Collapse Dynamics)

**Key Concepts:**
- **Tension phase:** Uncertainty accumulates as input arrives, entropy rises, multiple hypotheses coexist
- **Discharge phase:** Collapse occurs, entropy drops suddenly, one hypothesis wins
- **Recovery phase:** Belief re-spreads to handle next prediction, entropy rises again
- **Relaxation oscillator:** System that naturally oscillates between charging and discharging

**From POMDP_SIM.md (§5.2):**
> "Pump cycle is fundamental rhythm of belief dynamics: TENSION (accumulate uncertainty) → DISCHARGE (collapse to certainty) → RECOVERY (re-spread for next cycle). Like lightning: charge builds in cloud (tension), sudden discharge (strike), return to equilibrium (recovery)."

**From SPECTRAL_BELIEF_MACHINE.md (§6.3):**
> "Belief collapse is not continuous process. System alternates between exploration (high entropy, multiple hypotheses) and commitment (low entropy, collapsed belief). Cycle repeats at natural frequency determined by input dynamics and model capacity."

**From BEC_CONDENSATION_INFORMATION.md (§4.3):**
> "Condensation and evaporation can cycle if system is driven. Below T_c: condensed (collapsed). Drive system above T_c: evaporated (diffuse). Allow cooling: recondense. This is pump cycle at mesoscopic scale."

**This experiment validates:**
1. Whether **entropy oscillates periodically** (tension-discharge-recovery)
2. Whether **cycle has characteristic period** (quasi-periodic vs chaotic)
3. Whether **discharge is shorter than tension** (asymmetric phases)
4. Whether **cycle correlates with prediction quality** (functional significance)

**Falsification:** If entropy dynamics are random/chaotic → no pump cycle → belief dynamics are stochastic, not structured.

## 3. Methods

### 3.1 Protocol

```
PUMP CYCLE DETECTION PROTOCOL

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  STEP 1: Record entropy time series                                    │
│  • Long sequence processing                                           │
│  • High temporal resolution                                           │
│                                                                         │
│  STEP 2: Detect cycle phases                                           │
│  • Rising entropy = Tension                                           │
│  • Sudden drop = Discharge                                            │
│  • Return to baseline = Recovery                                      │
│                                                                         │
│  STEP 3: Analyze periodicity                                           │
│  • Fourier analysis of entropy time series                           │
│  • Autocorrelation analysis                                           │
│  • Is there a characteristic period?                                  │
│                                                                         │
│  STEP 4: Correlate with prediction                                     │
│  • Phase of cycle vs prediction error                                │
│  • Where in cycle are predictions best?                              │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Results

```
[ TO BE FILLED AFTER EXPERIMENT ]

Cycle detected: YES / NO

Cycle statistics:
• Mean period: _____ timesteps
• Tension phase duration: _____
• Discharge phase duration: _____
• Recovery phase duration: _____

Periodicity (autocorrelation peak): _____
Prediction accuracy by phase:
• Tension: _____
• Discharge: _____
• Recovery: _____
```

---

## 5. Conclusion

```
[ TO BE FILLED AFTER EXPERIMENT ]

H1 (oscillation exists): SUPPORTED / NOT SUPPORTED
H2 (consistent period): SUPPORTED / NOT SUPPORTED
H3 (discharge < tension): SUPPORTED / NOT SUPPORTED
H4 (correlates with accuracy): SUPPORTED / NOT SUPPORTED

Pump cycle is REAL / PARTIAL / NOT FOUND.
```

---

*Oscar Goldman, Shogu Research Group @ Datamutant.ai subsidiary of 温心重工業*

