# EXPERIMENT 008: Quasiparticle Dispersion

## Do Action Quanta Behave Like Quasiparticles?

**Oscar Goldman — Shogu Research Group @ Datamutant.ai**

---

## Status: PLANNED

## Tier: ★★ CORE

## Depends On: 004 (Phase Transition), 005 (Conservation Laws)

---

## 1. Problem Statement

### 1.1 The Question

If attention dynamics follow BEC physics, then Action Quanta should be **quasiparticles** — collective excitations with a characteristic dispersion relation:

**Do perturbations at different spatial frequencies propagate differently, following a Bogoliubov-like dispersion relation?**

### 1.2 Why This Matters

```
THE QUASIPARTICLE HYPOTHESIS

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  In BEC, excitations follow Bogoliubov dispersion:                     │
│                                                                         │
│  E(k) = √[ε(k)(ε(k) + 2gn)]                                          │
│                                                                         │
│  LOW k (long wavelength):  E ≈ ℏck    (phonon-like, collective)       │
│  HIGH k (short wavelength): E ≈ ℏ²k²/2m (particle-like, local)       │
│                                                                         │
│  If true for AKIRA:                                                     │
│  • Low-freq perturbations propagate globally (collective)             │
│  • High-freq perturbations stay local (individual)                    │
│  • Crossover at "healing length" = mid-bands                          │
│                                                                         │
│  This would prove Action Quanta are EMERGENT, not fundamental.    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Hypothesis

### 2.1 Primary Hypothesis

**H1: Perturbation propagation depends on spatial frequency.**

- Low-k perturbations: Global, collective response
- High-k perturbations: Local, individual response

### 2.2 Secondary Hypotheses

**H2: Dispersion relation matches Bogoliubov form.**
- Linear at low k (sound-like)
- Quadratic at high k (particle-like)

**H3: Crossover scale corresponds to mid-bands (3-4).**

### 2.3 Null Hypothesis

**H0:** All frequencies propagate the same (no quasiparticle structure).

---

## 3. Scientific Basis

### 3.1 Bogoliubov Theory

**ESTABLISHED SCIENCE:**

In weakly interacting BEC:
```
E(k) = √[ε₀(k)(ε₀(k) + 2μ)]

where:
ε₀(k) = ℏ²k²/2m (free particle)
μ = gn (chemical potential)

Low k: E ≈ ck (sound waves, c = √(gn/m))
High k: E ≈ ε₀(k) (free particles)

Reference: Bogoliubov (1947), Pethick & Smith (2008)
```

### 3.2 AKIRA Theory Basis

**Relevant Theory Documents:**
- `bec/BEC_CONDENSATION_INFORMATION.md` — §8 (Action Quanta as Quasiparticles)
- `architecture_theoretical/SPECTRAL_BELIEF_MACHINE.md` — §5 (Spectral Propagation)
- `foundations/HARMONY_AND_COHERENCE.md` — §8 (Collective Excitations)

**Key Concepts:**
- **Action Quanta:** Emergent collective excitations of belief field (not fundamental units)
- **Dispersion relation:** Energy E(k) vs wavenumber k characterizes propagation
- **Bogoliubov form:** Low-k phonon-like (collective, E ∝ k), high-k particle-like (local, E ∝ k²)
- **Healing length:** Crossover scale where behavior transitions from collective to local

**From BEC_CONDENSATION_INFORMATION.md (§8.1):**
> "Action Quanta are quasiparticles — collective excitations analogous to phonons in solid or Bogoliubov excitations in BEC. NOT fundamental particles but emergent structures. Dispersion E(k) = √[ε(k)(ε(k) + 2gn)]: low-k sound waves (collective), high-k particles (local)."

**From SPECTRAL_BELIEF_MACHINE.md (§5.4):**
> "Perturbation propagation depends on frequency. Low-freq (Bands 0-2): global influence, slow propagation, collective response. High-freq (Bands 5-6): local influence, fast decay, individual response. Mid-bands (3-4): crossover regime (healing length)."

**From HARMONY_AND_COHERENCE.md (§8.2):**
> "Collective excitations exhibit Bogoliubov dispersion. Low-k: linear E ∝ k (sound speed c = √(interaction/mass)). High-k: quadratic E ∝ k² (free particle). This crossover is signature of emergent quasiparticle — proves atoms are collective, not fundamental."

**This experiment validates:**
1. Whether **perturbations propagate differently** by frequency (dispersion exists)
2. Whether **dispersion matches Bogoliubov form** (linear→quadratic crossover)
3. Whether **crossover occurs at mid-bands** (healing length = Band 3-4 boundary)
4. Whether Action Quanta are **emergent collective excitations** (not fundamental)

**Falsification:** If all frequencies propagate identically OR no Bogoliubov-like dispersion → no quasiparticle structure → Action Quanta are fundamental (not emergent) → BEC analogy incomplete.

---

## 4. Methods

### 4.1 Protocol

```
QUASIPARTICLE DISPERSION PROTOCOL

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  STEP 1: Create perturbation at specific k                            │
│  • Add sinusoidal perturbation at frequency k to input                │
│  • Vary k from low (DC-like) to high (edge-like)                     │
│                                                                         │
│  STEP 2: Measure response propagation                                  │
│  • Track how perturbation affects distant positions                   │
│  • Measure propagation velocity and decay                             │
│                                                                         │
│  STEP 3: Extract dispersion relation                                   │
│  • Plot energy (response magnitude) vs k                              │
│  • Fit to Bogoliubov form                                             │
│                                                                         │
│  STEP 4: Identify crossover scale                                      │
│  • Find k where behavior transitions from collective to local        │
│  • Compare to spectral band boundaries                                │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Predictions

```
IF THEORY IS CORRECT:

Low-k (bands 0-2):
• Perturbations spread globally
• Response at distant positions is strong
• Propagation is coherent (sound-like)

High-k (bands 5-6):
• Perturbations stay local
• Response decays rapidly with distance
• Propagation is diffusive

Crossover (bands 3-4):
• Intermediate behavior
• This is the "healing length" of the system
```

---

## 5. Results

```
[ TO BE FILLED AFTER EXPERIMENT ]

Dispersion relation E(k):
[INSERT PLOT]

Fit to Bogoliubov: R² = _____
Crossover scale: k = _____ (corresponds to band _____)
```

---

## 6. Conclusion

```
[ TO BE FILLED AFTER EXPERIMENT ]

H1 (frequency-dependent propagation): SUPPORTED / NOT SUPPORTED
H2 (Bogoliubov form): SUPPORTED / NOT SUPPORTED
H3 (crossover at mid-bands): SUPPORTED / NOT SUPPORTED
```

---

*Oscar Goldman — Shogu Research Group @ Datamutant.ai*


