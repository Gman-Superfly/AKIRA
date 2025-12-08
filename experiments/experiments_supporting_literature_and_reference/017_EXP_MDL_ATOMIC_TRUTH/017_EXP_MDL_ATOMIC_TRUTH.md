# EXPERIMENT 017: MDL Atomic Truth

## What's the Minimum Prompt?

**Oscar Goldman — Shogu Research Group @ Datamutant.ai**

---

## Status: PLANNED

## Tier: ○ EXPLORATORY

## Depends On: All previous experiments

---

## 1. Problem Statement

### 1.1 The Question

The "Old Lady" model predicts compression to atomic truths:

**Can we find the minimum description length (MDL) for concepts — the shortest prompt that still produces the desired behavior?**

### 1.2 Why This Matters

```
THE ATOMIC TRUTH HYPOTHESIS

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  From THE_OLD_LADY_AND_THE_TIGER.md:                                   │
│                                                                         │
│  "DON'T INVERT. DISTILL."                                              │
│                                                                         │
│  The Old Lady compresses trajectories to atomic truths:               │
│  • Many details → few load-bearing elements                           │
│  • What survives is sufficient for action                            │
│  • Everything else is noise                                           │
│                                                                         │
│  For prompts:                                                           │
│  • Long prompt → shortest prompt with same effect                     │
│  • The MDL is the "atomic prompt"                                     │
│  • Beyond MDL, additions are redundant                                │
│                                                                         │
│  If we can find MDL:                                                    │
│  • Optimal prompt engineering                                          │
│  • Understanding of what models actually need                         │
│  • Compression = understanding                                         │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Hypothesis

### 2.1 Primary Hypothesis

**H1: Concepts have a minimum description length.**

Prompts can be compressed to a point where further removal changes behavior.

### 2.2 Secondary Hypotheses

**H2: MDL correlates with concept "primitivity."**

Simpler concepts have shorter MDL.

**H3: Atomic prompts are load-bearing.**

Every word in an atomic prompt matters.

**H4: MDL is relatively universal across models.**

### 2.3 Null Hypothesis

**H0:** No clear MDL exists (behavior changes continuously with prompt length).

---

## 3. Methods

### 3.1 Protocol

```
MDL EXTRACTION PROTOCOL

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  STEP 1: Start with working prompt                                     │
│  • Known effective prompt for target behavior                         │
│  • Record baseline performance                                        │
│                                                                         │
│  STEP 2: Progressive compression                                       │
│  • Remove words one at a time                                         │
│  • Measure behavior change                                            │
│  • Use binary search for efficiency                                   │
│                                                                         │
│  STEP 3: Find MDL boundary                                             │
│  • Point where further removal causes failure                        │
│  • This is the atomic prompt                                          │
│                                                                         │
│  STEP 4: Verify load-bearing property                                  │
│  • Remove each word from atomic prompt                               │
│  • All should cause significant change                               │
│                                                                         │
│  STEP 5: Compare MDL across concepts                                   │
│  • Simple concepts (e.g., "be helpful")                              │
│  • Complex concepts (e.g., "write like Shakespeare")                 │
│  • Is there a pattern?                                                │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Results

```
[ TO BE FILLED AFTER EXPERIMENT ]

MDL by concept:
┌─────────────────────────────────────────────────────────────────────────┐
│  CONCEPT                    ORIGINAL LENGTH    MDL     COMPRESSION     │
│  ───────                    ───────────────    ───     ───────────     │
│  Be helpful                 _____              _____   _____×          │
│  Write code                 _____              _____   _____×          │
│  Explain simply             _____              _____   _____×          │
│  Act as expert              _____              _____   _____×          │
│  Creative writing           _____              _____   _____×          │
└─────────────────────────────────────────────────────────────────────────┘

Load-bearing verification:
• All words load-bearing: _____%

Cross-model MDL correlation: r = _____
```

---

## 5. Conclusion

```
[ TO BE FILLED AFTER EXPERIMENT ]

H1 (MDL exists): SUPPORTED / NOT SUPPORTED
H2 (correlates with primitivity): SUPPORTED / NOT SUPPORTED
H3 (atomic prompts are load-bearing): SUPPORTED / NOT SUPPORTED
H4 (universal across models): SUPPORTED / NOT SUPPORTED

Atomic prompts are DISCOVERABLE / NOT DISCOVERABLE.
```

---

*Oscar Goldman — Shogu Research Group @ Datamutant.ai*


