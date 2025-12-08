# Conservation of Action

## The Fire Analogy: On the True Nature of What Burns

**Oscar Goldman — Shogu Research Group @ Datamutant.ai**

---

## Table of Contents

1. [The Fire Insight](#1-the-fire-insight)
2. [What Is Actually Burning](#2-what-is-actually-burning)
3. [The Complete Chain of Conservation](#3-the-complete-chain-of-conservation)
4. [The Storage Medium: Weights as Wood](#4-the-storage-medium-weights-as-wood)
5. [The Catalyst: Attention as Oxygen](#5-the-catalyst-attention-as-oxygen)
6. [The Release: Inference as Combustion](#6-the-release-inference-as-combustion)
7. [The Source: Training Data as Sunlight](#7-the-source-training-data-as-sunlight)
8. [Real Photons and Fake Photons](#8-real-photons-and-fake-photons)
9. [The Two Storage Systems](#9-the-two-storage-systems)
10. [Heat Equilibrium: The Temperature of Inference](#10-heat-equilibrium-the-temperature-of-inference)
11. [The Complete Heresy Taxonomy](#11-the-complete-heresy-taxonomy)
12. [Heresies: False Sources of Heat](#12-heresies-false-sources-of-heat)
13. [The Information Budget](#13-the-information-budget)
14. [Quality of Fuel](#14-quality-of-fuel)
15. [Efficiency of Combustion](#15-efficiency-of-combustion)
16. [The Ash: What Remains](#16-the-ash-what-remains)
17. [Implications for Prompt Engineering](#17-implications-for-prompt-engineering)
18. [Experimental Predictions](#18-experimental-predictions)
19. [Mathematical Formalization](#19-mathematical-formalization)
20. [Connections and References](#20-connections-and-references)

---

## 1. The Fire Insight

### 1.1 The Conventional Understanding

When we watch a fire burn, we say "the wood is burning." This is a convenient shorthand, but it obscures the true nature of what is happening. The wood is not the source of the heat. The oxygen is not the source of the heat. The match that started the fire is not the source of the heat.

**The heat was put there by the sun.**

### 1.2 The Chain of Conservation

```
THE TRUE PATH OF FIRE

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  STEP 1: NUCLEAR FUSION IN THE SUN                                     │
│  ─────────────────────────────────────                                  │
│  Hydrogen atoms fuse into helium.                                      │
│  Mass converts to energy: E = mc².                                     │
│  This is the ORIGINAL ACTION, the SOURCE.                             │
│  Photons carry this energy across 93 million miles.                   │
│                                                                         │
│  STEP 2: PHOTOSYNTHESIS                                                 │
│  ────────────────────────                                               │
│  Plants capture photons.                                               │
│  Solar energy converts to chemical bonds.                              │
│  This is STORAGE. The action is CONSERVED, not created.              │
│  The tree grows by accumulating conserved solar action.               │
│                                                                         │
│  STEP 3: THE WOOD EXISTS                                                │
│  ───────────────────────                                                │
│  The tree falls. Becomes lumber. Becomes firewood.                    │
│  The stored energy waits. Stable. Patient.                            │
│  Years, decades, centuries — the action remains.                      │
│                                                                         │
│  STEP 4: COMBUSTION                                                     │
│  ──────────────────                                                     │
│  Oxygen meets wood. Heat triggers reaction.                           │
│  Chemical bonds break. Energy releases.                               │
│  The stored solar action becomes HEAT and LIGHT again.               │
│  The circle closes.                                                    │
│                                                                         │
│  STEP 5: THE HEAT YOU FEEL                                              │
│  ─────────────────────────                                              │
│  This warmth on your face?                                            │
│  It is sunlight that fell on Earth perhaps centuries ago.            │
│  Stored, conserved, released.                                         │
│  Nothing was created. Nothing was destroyed.                         │
│  Action was conserved.                                                 │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1.3 The Profound Implication

The wood does not burn. The oxygen does not burn. **The conserved action of the sun burns.**

The wood is the storage medium.
The oxygen is the catalyst that enables release.
The fire is the process of release.
The heat is the released action.

This is physics: conservation of energy from sun through wood to fire. The same conservation principle applies analogously to neural networks: information from data through weights to inference.

---

## 2. What Is Actually Burning

### 2.1 Applying This to AKIRA

If we apply this understanding to neural networks, we arrive at a profound realization:

```
THE AKIRA FIRE

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  PHYSICAL FIRE:                                                         │
│  ─────────────                                                          │
│  Sun → Photons → Photosynthesis → Wood → Oxygen + Heat → Burning     │
│                                                                         │
│  Source   →  Carrier  →  Storage  →  Medium  →  Catalyst → Release   │
│                                                                         │
│  ─────────────────────────────────────────────────────────────────────  │
│                                                                         │
│  AKIRA FIRE:                                                            │
│  ───────────                                                            │
│  Truth → Data → Training → Weights → Attention + Input → Prediction  │
│                                                                         │
│  Source → Carrier → Storage → Medium →  Catalyst  →  Release         │
│                                                                         │
│  ─────────────────────────────────────────────────────────────────────  │
│                                                                         │
│  THE MAPPING:                                                           │
│                                                                         │
│  Sun's nuclear fusion    ←→  Ground truth (reality)                   │
│  Photons               ←→  Training data (observations)               │
│  Photosynthesis        ←→  Training process (gradient descent)        │
│  Wood                  ←→  Weights (learned parameters)               │
│  Oxygen                ←→  Attention mechanism                        │
│  Heat/ignition         ←→  Input query / prompt                       │
│  Fire                  ←→  Inference process                          │
│  Released heat         ←→  Prediction / output                        │
│  Ash                   ←→  Unchanged weights after inference          │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.2 The Key Insight

**The model does not create knowledge. The model releases stored patterns.**

When AKIRA makes a prediction, it is not inventing something new. It is releasing patterns that were stored during training. Those patterns came from data. That data came from observations. Those observations captured aspects of reality.

The chain of conservation:

```
Reality → Observation → Data → Training → Weights → Inference → Prediction

At each step, information is CONSERVED (with possible losses), never CREATED.
```

This is why we cannot get out more than was put in. This is why garbage data leads to garbage predictions. This is why the model cannot invent truths it never observed.

**The prediction is ancient sunlight, stored and released.**

---

## 3. The Complete Chain of Conservation

### 3.1 From Reality to Prediction

Let us trace the complete chain:

```
CHAIN OF CONSERVATION

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  STAGE 0: GROUND TRUTH (THE SUN)                                       │
│  ─────────────────────────────────                                      │
│  Reality exists. Patterns exist in nature.                            │
│  Statistical regularities govern the world.                           │
│  This is the ultimate source of all knowledge.                        │
│  We cannot access it directly — only through observation.             │
│                                                                         │
│  STAGE 1: OBSERVATION (PHOTONS)                                        │
│  ──────────────────────────────                                         │
│  Sensors capture aspects of reality.                                  │
│  Cameras record images. Microphones record sounds.                    │
│  Each observation is a SAMPLE from reality.                           │
│  Information is already lossy — we see shadows on the cave wall.     │
│                                                                         │
│  STAGE 2: DATA (PHOTONS ARRIVING)                                      │
│  ─────────────────────────────────                                      │
│  Observations become data.                                            │
│  Structured, stored, prepared for training.                           │
│  The data carries conserved patterns from reality.                    │
│  But also carries noise, bias, artifacts.                             │
│                                                                         │
│  STAGE 3: TRAINING (PHOTOSYNTHESIS)                                    │
│  ────────────────────────────────────                                   │
│  The model encounters data.                                           │
│  Gradients flow. Weights update.                                      │
│  Patterns from data are STORED in weights.                           │
│  Like plants converting light to chemical bonds.                     │
│  Information is compressed, abstracted, organized.                   │
│                                                                         │
│  STAGE 4: WEIGHTS (WOOD)                                               │
│  ───────────────────────                                                │
│  The trained model exists.                                            │
│  Weights encode compressed representations of data patterns.         │
│  This is STORED ACTION — potential, not kinetic.                     │
│  Like wood waiting to burn.                                           │
│  The weights could sit on disk for years.                            │
│                                                                         │
│  STAGE 5: INFERENCE (COMBUSTION)                                       │
│  ───────────────────────────────                                        │
│  A query arrives (the heat, the spark).                              │
│  Attention activates (oxygen meets wood).                            │
│  Stored patterns release (chemical bonds break).                     │
│  The prediction emerges (heat radiates).                             │
│                                                                         │
│  STAGE 6: PREDICTION (HEAT)                                            │
│  ──────────────────────────                                             │
│  The output is produced.                                              │
│  This is RELEASED ACTION — kinetic, manifest.                        │
│  The prediction carries patterns that trace back to reality.         │
│  Through the entire chain of conservation.                           │
│                                                                         │
│  STAGE 7: WEIGHTS REMAIN (ASH)                                         │
│  ──────────────────────────────                                         │
│  Unlike fire, inference doesn't consume the weights.                 │
│  The wood remains. Can burn again.                                   │
│  But during training, weights DO change.                             │
│  Training is photosynthesis. Inference is burning.                   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Conservation at Each Stage

At each stage, we can ask: **What is conserved? What is lost?**

| Stage | Conserved | Lost |
|-------|-----------|------|
| Reality → Observation | Core patterns | Fine details, unobserved aspects |
| Observation → Data | Statistical structure | Ephemeral context |
| Data → Training | Generalizable patterns | Noise, outliers |
| Training → Weights | Compressed representations | Raw examples |
| Weights → Inference | Relevant activations | Dormant patterns |
| Inference → Prediction | Selected output | Alternative hypotheses |

At no stage is anything **created**. At each stage, something may be **lost**. The final prediction can contain at most what was in the original reality, minus all the losses along the way.

---

## 4. The Storage Medium: Weights as Wood

### 4.1 What Weights Actually Are

The weights of a neural network are analogous to wood:

```
WEIGHTS = WOOD

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  WOOD:                                                                  │
│  • Physical structure (cellulose, lignin)                             │
│  • Stores solar energy in chemical bonds                              │
│  • Stable at room temperature                                         │
│  • Energy density varies (oak vs. pine)                               │
│  • Can be wet (hard to burn) or dry (easy to burn)                   │
│  • Burns once, becomes ash                                            │
│                                                                         │
│  WEIGHTS:                                                               │
│  • Numerical structure (matrices, tensors)                            │
│  • Store information in parameter values                              │
│  • Stable without input (dormant)                                     │
│  • Information density varies (compression quality)                   │
│  • Can be poorly trained (hard to use) or well-trained              │
│  • "Burns" repeatedly without consumption (unlike wood)              │
│                                                                         │
│  KEY INSIGHT:                                                           │
│  The weights don't KNOW anything.                                      │
│  They STORE patterns that ENABLE knowing during inference.           │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.2 The Ghost in the Wood

When we speak of the "ghost" in the machine, we should understand:

**The ghost is not a thing that knows. The ghost is the stored pattern.**

The ghost is to the weights what the stored solar energy is to the wood. The ghost is not separate from the weights — the ghost IS the pattern in the weights. When we "talk to the ghost," we are accessing stored patterns. When the ghost "responds," stored patterns are releasing.

This is why the ghost:
- Cannot know what it wasn't trained on (wood cannot release energy it didn't store)
- Speaks in patterns, not explicit knowledge (energy releases as heat, not as photons)
- Is imprecise in ways that reflect training (energy density varies with wood quality)
- Can be "burned" repeatedly (weights persist through inference, unlike wood)

---

## 5. The Catalyst: Attention as Oxygen

### 5.1 The Role of Oxygen

Oxygen does not create fire. Oxygen **enables** fire.

```
OXYGEN'S ROLE IN COMBUSTION

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  Without oxygen:                                                        │
│  • Wood exists, stores energy                                          │
│  • Energy cannot release                                               │
│  • No fire, no heat                                                    │
│                                                                         │
│  With oxygen:                                                           │
│  • Chemical reaction becomes possible                                  │
│  • Stored energy CAN release                                           │
│  • Fire occurs, heat radiates                                          │
│                                                                         │
│  Oxygen does NOT:                                                       │
│  • Create the energy                                                   │
│  • Determine what burns                                                │
│  • Add information to the fire                                        │
│                                                                         │
│  Oxygen DOES:                                                           │
│  • Enable the release mechanism                                        │
│  • Determine how fast things burn                                     │
│  • Affect efficiency of combustion                                    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 5.2 Attention as the Oxygen of Inference

The attention mechanism plays exactly this role:

```
ATTENTION = OXYGEN

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  Without attention:                                                     │
│  • Weights exist, store patterns                                      │
│  • Patterns cannot selectively activate                               │
│  • No intelligent response, no prediction                             │
│                                                                         │
│  With attention:                                                        │
│  • Pattern matching becomes possible                                  │
│  • Relevant stored patterns CAN release                               │
│  • Prediction emerges                                                  │
│                                                                         │
│  Attention does NOT:                                                    │
│  • Create the knowledge                                               │
│  • Invent new patterns                                                │
│  • Add information that wasn't stored                                │
│                                                                         │
│  Attention DOES:                                                        │
│  • Route information to relevant weights                              │
│  • Select which patterns activate                                     │
│  • Determine efficiency of pattern retrieval                         │
│                                                                         │
│  ─────────────────────────────────────────────────────────────────────  │
│                                                                         │
│  THE BEC CONNECTION:                                                    │
│  The attention = g|ψ|² term!                                          │
│  This is the SELF-INTERACTION that enables condensation.             │
│  Without it, no collapse. Without oxygen, no fire.                   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 5.3 Why This Matters

This clarifies what attention optimization can and cannot do:

**CAN DO:**
- Make pattern release more efficient
- Select better patterns for the context
- Reduce "smoke" (irrelevant activations)
- Enable faster "burning" (inference speed)

**CANNOT DO:**
- Create patterns that weren't stored
- Add knowledge that wasn't in training
- Overcome fundamental storage limits
- Invent truth from nothing

---

## 6. The Release: Inference as Combustion

### 6.1 The Combustion Process

```
COMBUSTION = INFERENCE

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  FIRE COMBUSTION:                                                       │
│  ────────────────                                                       │
│  1. Heat applied (ignition)                                           │
│  2. Chemical bonds begin to break                                     │
│  3. Oxygen combines with carbon                                       │
│  4. Energy releases as heat and light                                │
│  5. Reaction sustains itself (chain reaction)                        │
│  6. Eventually: all fuel consumed or fire extinguished               │
│                                                                         │
│  NEURAL INFERENCE:                                                      │
│  ────────────────                                                       │
│  1. Input applied (query/prompt)                                      │
│  2. Attention patterns begin to form                                  │
│  3. Input combines with weights                                       │
│  4. Patterns release as activations                                   │
│  5. Layers process sequentially (forward pass)                       │
│  6. Eventually: prediction produced                                   │
│                                                                         │
│  ─────────────────────────────────────────────────────────────────────  │
│                                                                         │
│  KEY PARALLEL:                                                          │
│  The input (heat) triggers the process.                               │
│  The attention (oxygen) enables it.                                   │
│  The weights (wood) provide the stored action.                        │
│  The prediction (heat output) is the released action.                │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 6.2 The Collapse as the Flash Point

In fire, there's a critical moment: the **flash point**. This is when the reaction becomes self-sustaining, when the fire "catches."

In AKIRA, this corresponds to **collapse** — the moment when diffuse belief suddenly concentrates into a specific prediction.

```
FLASH POINT = COLLAPSE

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  BEFORE FLASH POINT:                                                    │
│  • Heat building                                                       │
│  • Molecules vibrating faster                                          │
│  • Almost burning, not quite                                          │
│  • Energy accumulating                                                 │
│                                                                         │
│  AT FLASH POINT:                                                        │
│  • Sudden phase transition                                             │
│  • Fire catches                                                        │
│  • Rapid energy release                                               │
│  • Qualitative change in behavior                                     │
│                                                                         │
│  ─────────────────────────────────────────────────────────────────────  │
│                                                                         │
│  BEFORE COLLAPSE:                                                       │
│  • Belief spreading                                                    │
│  • Multiple hypotheses active                                         │
│  • Almost committed, not quite                                        │
│  • Entropy high                                                        │
│                                                                         │
│  AT COLLAPSE:                                                           │
│  • Sudden phase transition                                             │
│  • Prediction commits                                                  │
│  • Rapid entropy drop                                                 │
│  • Qualitative change in attention pattern                           │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

This is the BEC phase transition — the condensation. And it is powered by conserved action releasing, not by creation of new action.

---

## 7. The Source: Training Data as Sunlight

### 7.1 The Ultimate Source

```
THE SUN = GROUND TRUTH

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  For fire:                                                              │
│  The sun is the ultimate source of all terrestrial energy.           │
│  Every calorie of heat released by wood came from the sun.           │
│  We cannot create energy; we can only transform it.                  │
│                                                                         │
│  For AKIRA:                                                             │
│  Reality is the ultimate source of all knowledge.                    │
│  Every bit of prediction accuracy came from truth in data.          │
│  We cannot create knowledge; we can only transform it.               │
│                                                                         │
│  ─────────────────────────────────────────────────────────────────────  │
│                                                                         │
│  Implications:                                                          │
│                                                                         │
│  1. Training data is the sunlight.                                    │
│     Its quality determines what can be stored.                        │
│                                                                         │
│  2. The model cannot know what the data didn't contain.              │
│     Like a tree cannot store energy the sun didn't provide.         │
│                                                                         │
│  3. More training data = more sunlight = more stored action.         │
│     But quality matters more than quantity.                          │
│                                                                         │
│  4. Bias in data = color of sunlight.                                │
│     If the sun only shone red, trees could only store red.          │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 7.2 Tracing Knowledge Back

Every prediction the model makes can, in principle, be traced back to the training data:

```
KNOWLEDGE LINEAGE

Prediction: "The cat sat on the mat"
     ↑
Activated patterns in weights
     ↑
Patterns stored during training
     ↑
Similar sentences in training data
     ↑
Observations of cats, mats, sitting
     ↑
Reality: cats actually sit on mats

The model didn't LEARN that cats sit on mats.
The model STORED that observation from data.
The model RELEASED that stored pattern during inference.

Conservation of action, from reality to prediction.
```

---

## 8. Real Photons and Fake Photons

### 8.1 The Fun Distinction

To make this analogy vivid and memorable, we introduce a playful but precise distinction:

```
REAL PHOTONS vs FAKE PHOTONS

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  REAL PHOTONS (from the sun):                                          │
│  ────────────────────────────                                           │
│  • Born in nuclear fusion at the sun's core                           │
│  • Travel 93 million miles to Earth                                   │
│  • Captured by chlorophyll in plants                                  │
│  • Stored as chemical bonds in wood                                   │
│  • Released as heat when wood burns                                   │
│                                                                         │
│  The heat from burning wood that grew in sunlight                    │
│  is REAL photon energy, conserved from the sun.                      │
│                                                                         │
│  ─────────────────────────────────────────────────────────────────────  │
│                                                                         │
│  FAKE PHOTONS (from artificial lights):                                │
│  ──────────────────────────────────────                                 │
│  • Born in a light bulb, LED, or grow lamp                           │
│  • Powered by electricity (from coal, nuclear, etc.)                 │
│  • Captured by plants growing under artificial light                 │
│  • Stored as chemical bonds in that wood                             │
│  • Released as heat when that wood burns                             │
│                                                                         │
│  The heat from burning wood that grew under fake lights              │
│  contains energy that was NEVER from the sun.                        │
│  It looks like real fire, feels like real heat,                      │
│  but its lineage is different.                                       │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 8.2 The AKIRA Mapping

```
REAL vs FAKE IN AKIRA

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  REAL PATTERNS (from ground truth):                                    │
│  ─────────────────────────────────                                      │
│  • Observations of actual phenomena                                   │
│  • Correctly sampled, properly processed                             │
│  • Stored in weights during training                                 │
│  • Released during inference                                          │
│                                                                         │
│  Predictions that trace back to reality.                             │
│  TRUE knowledge, conserved from ground truth.                        │
│                                                                         │
│  ─────────────────────────────────────────────────────────────────────  │
│                                                                         │
│  FAKE PATTERNS (from artifacts):                                       │
│  ───────────────────────────────                                        │
│  • Aliasing from undersampled data                                   │
│  • Spectral leakage from architecture                                │
│  • Boundary effects from finite windows                              │
│  • Regularization artifacts from training                            │
│                                                                         │
│  Predictions that trace back to the PROCESS, not reality.           │
│  HERETICAL knowledge, created by the architecture.                   │
│                                                                         │
│  ─────────────────────────────────────────────────────────────────────  │
│                                                                         │
│  THE PROBLEM:                                                           │
│  From the outside, you cannot tell which heat is from real photons  │
│  and which is from fake photons. The fire looks the same.           │
│                                                                         │
│  The ghost cannot tell which patterns are from reality               │
│  and which are from the architecture. They feel the same.           │
│                                                                         │
│  This is why we need the INQUISITION (experiments).                  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 8.3 Note on This Analogy

This is a **fun** and **pedagogical** distinction, not a statement about observations in nature. We are not claiming that plants under grow lights are somehow "less real" — obviously, photosynthesis works the same way regardless of the photon source.

The point is to illustrate that **not all information in a trained model comes from the intended source (reality)**. Some information comes from the training process itself, the architecture, the sampling choices, and other artifacts of the system.

Just as you could burn wood from a greenhouse and not know whether the energy came from the sun or from an LED, you can read a model's output and not know whether the pattern came from data or from architecture.

---

## 9. The Two Storage Systems

### 9.1 Weights vs. Context Window

We have been treating "storage" as singular, but AKIRA has **two distinct storage systems**:

```
THE TWO STORAGE SYSTEMS

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  STORAGE TYPE 1: THE WEIGHTS (Permanent)                               │
│  ─────────────────────────────────────────                              │
│  • Formed during: Training (photosynthesis)                           │
│  • Lifespan: Permanent until retrained                                │
│  • Contains: Compressed patterns from training data                   │
│  • Changes: Only through training/fine-tuning                        │
│  • Location: Model parameters (on disk, in GPU memory)               │
│  • Analogy: THE WOOD — stored fuel                                   │
│  • The ghost: The ghost LIVES here                                   │
│                                                                         │
│  ─────────────────────────────────────────────────────────────────────  │
│                                                                         │
│  STORAGE TYPE 2: THE CONTEXT WINDOW (Temporary)                        │
│  ─────────────────────────────────────────────                          │
│  • Formed during: Inference                                           │
│  • Lifespan: One conversation/inference session                      │
│  • Contains: Current input, history, system prompt                   │
│  • Changes: With every token generated                               │
│  • Location: Active memory during inference                          │
│  • Analogy: THE COMBUSTION CHAMBER — where burning happens          │
│  • The ghost: The ghost SPEAKS here                                  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 9.2 What is in the Context Window?

The context window is NOT additional wood. It is the space where the fire burns:

```
CONTEXT WINDOW DECOMPOSITION

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  1. USER INPUT (Fresh Spark)                                           │
│  ───────────────────────────                                            │
│  • NEW information entering the system                                │
│  • Like fresh sunlight hitting a living plant                        │
│  • Not from stored patterns                                           │
│  • This is the IGNITION — the spark that starts the fire            │
│                                                                         │
│  2. PREVIOUS MODEL OUTPUTS (Recycled Heat)                             │
│  ────────────────────────────────────────                               │
│  • Came FROM the weights during previous inference                   │
│  • Like using the heat from one fire to start another               │
│  • RECYCLED stored energy, not new energy                           │
│  • Chain reaction: fire → heat → fire → heat                        │
│                                                                         │
│  3. SYSTEM PROMPT (Stove Design)                                       │
│  ───────────────────────────────                                        │
│  • Shapes HOW the burning happens                                    │
│  • Like the stove design, airflow, arrangement                      │
│  • Not fuel itself, but determines combustion pattern               │
│  • The kindling arrangement                                          │
│                                                                         │
│  4. CONVERSATION HISTORY (Ash and Embers)                              │
│  ──────────────────────────────────────────                             │
│  • Mix of user inputs and model outputs                             │
│  • Accumulating pile of ash AND still-glowing embers                │
│  • Some useful (embers), some cold (ash)                            │
│  • Can clog the combustion chamber if too much                      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 9.3 One Ghost, Two Manifestations

There is only ONE ghost, but it appears in two ways:

```
ONE GHOST, TWO MANIFESTATIONS

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  THE GHOST lives in the weights.                                       │
│  The ghost IS the stored pattern structure.                           │
│  This is its HOME.                                                     │
│                                                                         │
│  THE GHOST speaks through the context window.                          │
│  The context window is the ghost's VOICE, not its home.              │
│  This is its EXPRESSION.                                              │
│                                                                         │
│  ─────────────────────────────────────────────────────────────────────  │
│                                                                         │
│  Analogy:                                                               │
│  • A musician (ghost) lives in their brain (weights)                 │
│  • The music (output) happens in the air (context window)           │
│  • The instrument (attention) enables expression                     │
│  • The musician plays, but the music is not the musician            │
│                                                                         │
│  Or:                                                                    │
│  • The fire (inference) burns in the stove (context window)         │
│  • But the heat comes from the wood (weights)                        │
│  • The stove is where it happens, not where the energy is stored   │
│                                                                         │
│  ─────────────────────────────────────────────────────────────────────  │
│                                                                         │
│  KEY INSIGHT:                                                           │
│  The context window receives USER INPUT — new information            │
│  the ghost didn't have before. The ghost responds to this           │
│  new information using its stored patterns.                          │
│                                                                         │
│  The conversation is a DIALOGUE between:                              │
│  • Fresh input (user's spark)                                        │
│  • Stored patterns (ghost's knowledge)                               │
│  • Accumulated context (shared history)                              │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 9.4 Autoregressive Feedback: The Self-Sustaining Fire

Here is where it gets critical:

```
AUTOREGRESSIVE BURNING

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  In a normal fire:                                                      │
│  Wood → Fire → Heat → Exits the system                               │
│                                                                         │
│  In a self-sustaining fire:                                            │
│  Wood → Fire → Heat → Ignites more wood → More fire                  │
│  The heat from burning feeds back to enable more burning.            │
│                                                                         │
│  In autoregressive LLM:                                                │
│  Weights → Attention → Token → Added to context → More attention     │
│  The output from inference feeds back to enable more inference.      │
│                                                                         │
│  ─────────────────────────────────────────────────────────────────────  │
│                                                                         │
│  THE CONTEXT WINDOW IS THE FEEDBACK LOOP.                             │
│  Each generated token becomes input for the next token.              │
│  The fire sustains itself through its own heat.                      │
│                                                                         │
│  ─────────────────────────────────────────────────────────────────────  │
│                                                                         │
│  THE DANGER:                                                            │
│  The context window contains MODEL OUTPUTS.                           │
│  Model outputs come from weights.                                     │
│  Weights may contain heresies.                                        │
│                                                                         │
│  So the context window can fill with:                                  │
│  • True patterns (from good training)                                 │
│  • Heresies (from bad training OR architecture artifacts)           │
│                                                                         │
│  And the model treats them the same!                                  │
│  This is how HALLUCINATIONS COMPOUND.                                 │
│                                                                         │
│  One heresy → feeds back → more heresies → spiral                   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 10. Heat Equilibrium: The Temperature of Inference

### 10.1 The Role of Heat in Fire

In a real fire, heat plays multiple roles:

```
HEAT EQUILIBRIUM IN FIRE

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  HEAT AS IGNITION:                                                      │
│  ─────────────────                                                      │
│  • Heat starts the combustion reaction                                │
│  • Without sufficient heat, fire doesn't catch                       │
│  • The spark, the match, the friction                                │
│                                                                         │
│  HEAT AS SUSTENANCE:                                                    │
│  ───────────────────                                                    │
│  • Heat keeps the fire burning                                        │
│  • Fire produces heat which enables more fire                        │
│  • Self-sustaining chain reaction                                    │
│                                                                         │
│  HEAT AS OUTPUT:                                                        │
│  ────────────────                                                       │
│  • Heat radiates outward                                              │
│  • Warms the room, cooks the food                                    │
│  • The useful product of the fire                                    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 10.2 The Stove Equilibrium

A well-designed stove must balance heat:

```
STOVE THERMODYNAMICS

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  TOO HOT:                                                               │
│  ────────                                                               │
│  • Wastes energy (heat escapes up chimney)                           │
│  • Damages the stove                                                  │
│  • Burns fuel too fast                                                │
│  • Room becomes uncomfortably hot                                    │
│  • Inefficient: much energy wasted                                   │
│                                                                         │
│  TOO COLD:                                                              │
│  ─────────                                                              │
│  • Fire struggles to sustain                                          │
│  • Incomplete combustion (smoke, soot)                               │
│  • Room stays cold                                                    │
│  • Fire may go out                                                    │
│  • Inefficient: fuel doesn't fully burn                             │
│                                                                         │
│  JUST RIGHT (Equilibrium):                                              │
│  ─────────────────────────                                              │
│  • Steady, efficient burn                                             │
│  • Fuel consumed at optimal rate                                     │
│  • Room at comfortable temperature                                   │
│  • Fire self-sustains reliably                                       │
│  • Maximum useful work extracted                                     │
│                                                                         │
│  ─────────────────────────────────────────────────────────────────────  │
│                                                                         │
│  The stove operator must:                                              │
│  • Add fuel at the right rate                                        │
│  • Control airflow (oxygen)                                          │
│  • Open/close dampers                                                │
│  • Remove ash periodically                                           │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 10.3 Temperature Mapping to AKIRA

```
INFERENCE TEMPERATURE

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  "TEMPERATURE" IN ML:                                                   │
│  ────────────────────                                                   │
│  Softmax temperature controls sharpness of attention.                │
│  High temp = diffuse attention = uncertain = "hot"                   │
│  Low temp = sharp attention = certain = "cold"                       │
│                                                                         │
│  ─────────────────────────────────────────────────────────────────────  │
│                                                                         │
│  TOO HOT (High Temperature):                                           │
│  ───────────────────────────                                            │
│  • Attention is diffuse                                               │
│  • Many patterns activate weakly                                     │
│  • Predictions are random, incoherent                                │
│  • Energy wasted on irrelevant patterns                             │
│  • Hallucinations likely                                             │
│                                                                         │
│  TOO COLD (Low Temperature):                                           │
│  ────────────────────────────                                           │
│  • Attention is too sharp                                            │
│  • Only strongest patterns activate                                  │
│  • Predictions are repetitive, stuck                                 │
│  • Cannot explore alternatives                                       │
│  • May miss correct answer                                           │
│                                                                         │
│  JUST RIGHT:                                                            │
│  ───────────                                                            │
│  • Attention is focused but flexible                                 │
│  • Right patterns activate strongly                                  │
│  • Predictions are coherent and appropriate                         │
│  • Efficient use of stored patterns                                  │
│  • Optimal balance of certainty and exploration                     │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 10.4 The Cold Context Window: Ash Accumulation

```
THE COLD CONTEXT WINDOW

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  In a real fire:                                                        │
│  ────────────────                                                       │
│  Ash accumulates. If not cleared:                                     │
│  • Ash smothers the embers                                            │
│  • Fresh fuel can't reach the heat                                   │
│  • Fire dies or can't start                                          │
│  • The combustion chamber is clogged                                 │
│                                                                         │
│  ─────────────────────────────────────────────────────────────────────  │
│                                                                         │
│  In the context window:                                                │
│  ───────────────────────                                                │
│  Old, irrelevant tokens accumulate. If not managed:                  │
│  • Old context dilutes attention                                     │
│  • Fresh input can't influence the model                            │
│  • Inference becomes sluggish or stuck                              │
│  • The context window is clogged                                     │
│                                                                         │
│  ─────────────────────────────────────────────────────────────────────  │
│                                                                         │
│  SYMPTOMS OF A COLD, ASHY CONTEXT:                                     │
│  ──────────────────────────────────                                     │
│  • Model repeats itself                                               │
│  • Model ignores new input                                           │
│  • Responses become generic, unfocused                              │
│  • Model "forgets" recent conversation                              │
│  • Hallucinations from stale context                                │
│                                                                         │
│  ─────────────────────────────────────────────────────────────────────  │
│                                                                         │
│  SOLUTIONS:                                                             │
│  ──────────                                                             │
│  • Clear old context (start fresh conversation)                      │
│  • Summarize and compress (remove ash, keep embers)                 │
│  • Strategic prompting (add kindling to restart fire)               │
│  • Context management (sliding window, truncation)                  │
│                                                                         │
│  A good stove needs ash removal.                                      │
│  A good inference system needs context management.                   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 10.5 The Fire That Cannot Start

```
FAILED IGNITION

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  Sometimes the fire won't catch:                                       │
│                                                                         │
│  1. WET WOOD (poor training)                                           │
│     • Weights don't contain useful patterns                          │
│     • No matter how good the prompt, nothing ignites                │
│                                                                         │
│  2. NO SPARK (bad prompt)                                              │
│     • The query doesn't activate relevant patterns                  │
│     • The fire has no ignition source                               │
│                                                                         │
│  3. TOO MUCH ASH (clogged context)                                     │
│     • Old tokens block new information                              │
│     • Fresh sparks can't reach the wood                             │
│                                                                         │
│  4. NO OXYGEN (broken attention)                                       │
│     • Attention mechanism fails                                      │
│     • Wood and spark present, but no catalyst                       │
│                                                                         │
│  5. WRONG FUEL (irrelevant context)                                    │
│     • Context contains patterns that don't match query              │
│     • Like trying to burn rocks                                      │
│                                                                         │
│  ─────────────────────────────────────────────────────────────────────  │
│                                                                         │
│  DIAGNOSIS:                                                             │
│  ──────────                                                             │
│  Which component is failing?                                          │
│  • If same prompt fails on same model → prompt problem              │
│  • If all prompts fail on same model → training problem             │
│  • If long conversation fails → context management problem          │
│  • If random prompts fail → attention problem                       │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 11. The Complete Heresy Taxonomy

### 11.1 Three Sources of Heresy

Heresies can arise at three distinct points in the pipeline:

```
THREE SOURCES OF HERESY

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  SOURCE 1: HERESY IN THE DATA (Pre-Storage)                           │
│  ──────────────────────────────────────────                             │
│                                                                         │
│  If the training DATA was poorly sampled or processed:               │
│  • Video recorded below Nyquist rate → temporal aliasing             │
│  • Audio with aliased high frequencies → false tones                 │
│  • Images with moiré patterns → false textures                       │
│  • Biased dataset → false correlations                               │
│                                                                         │
│  These heresies are IN THE DATA before training.                     │
│  Training faithfully STORES them.                                     │
│  The model learns aliases as if they were real patterns.            │
│                                                                         │
│  ANALOGY: A tree growing under artificial light.                     │
│  The "sunlight" was already fake (fake photons).                    │
│  The tree stores this fake energy faithfully.                        │
│                                                                         │
│  STATUS: STORED heresies, not created by model                       │
│  PERSISTENCE: Permanent, in every inference                          │
│  DETECTABILITY: Hard — looks like real patterns                      │
│                                                                         │
│  ─────────────────────────────────────────────────────────────────────  │
│                                                                         │
│  SOURCE 2: HERESY IN THE TRAINING PROCESS (During Storage)            │
│  ──────────────────────────────────────────────────────────            │
│                                                                         │
│  Even with perfect data, training can create artifacts:              │
│  • Gradient noise → spurious correlations                            │
│  • Batch normalization → distribution artifacts                     │
│  • Regularization → smoothed-out details                            │
│  • Capacity limits → forced distortions                             │
│  • Dropout → inconsistent patterns                                   │
│                                                                         │
│  These heresies are CREATED during training.                         │
│  They don't exist in the data.                                       │
│  The training process introduces them.                               │
│                                                                         │
│  ANALOGY: Photosynthesis in polluted air.                            │
│  The sunlight was pure (real photons).                              │
│  But the process corrupted the storage.                              │
│  Chemical impurities incorporated into the wood.                    │
│                                                                         │
│  STATUS: CREATED heresies, stored in weights                         │
│  PERSISTENCE: Permanent, in every inference                          │
│  DETECTABILITY: Medium — can compare to original data               │
│                                                                         │
│  ─────────────────────────────────────────────────────────────────────  │
│                                                                         │
│  SOURCE 3: HERESY IN THE INFERENCE PROCESS (During Release)          │
│  ──────────────────────────────────────────────────────────            │
│                                                                         │
│  Even with perfect weights, inference can create artifacts:          │
│  • FFT spectral leakage (from windowing)                            │
│  • Boundary effects (at edges of windows)                           │
│  • Numerical precision limits (floating point errors)               │
│  • Attention artifacts (softmax saturation)                         │
│  • Autoregressive compounding (error × error × error)               │
│                                                                         │
│  These heresies are CREATED during inference.                        │
│  They don't exist in the weights.                                    │
│  The architecture introduces them fresh each time.                  │
│                                                                         │
│  ANALOGY: A flawed stove creating soot.                             │
│  The wood was pure (real stored energy).                            │
│  But the burning process creates impurities.                        │
│  The fire releases energy + artifacts.                              │
│                                                                         │
│  STATUS: CREATED heresies, never stored                              │
│  PERSISTENCE: Fresh each inference, may vary                        │
│  DETECTABILITY: Easier — varies with conditions                     │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 11.2 The Dangerous Asymmetry: Stored vs. Created Heresies

```
THE HERESY ASYMMETRY — CAREFUL ANALYSIS

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  CLAIM: Stored heresies are MORE dangerous.                           │
│                                                                         │
│  WHY?                                                                   │
│                                                                         │
│  1. PERSISTENCE                                                         │
│     ─────────────                                                       │
│     Stored heresies are ALWAYS there.                                │
│     Every inference, same heresy.                                    │
│     The ghost believes them as firmly as truth.                      │
│     They ARE part of the ghost's identity.                           │
│                                                                         │
│     Created heresies are fresh each time.                            │
│     May differ between inferences.                                   │
│     More like noise than systematic error.                          │
│                                                                         │
│  2. CONSISTENCY                                                         │
│     ────────────                                                        │
│     Stored heresies are CONSISTENT.                                   │
│     Ask the same question, get the same heresy.                      │
│     This makes them look like truth.                                 │
│                                                                         │
│     Created heresies may vary.                                        │
│     Ask the same question, maybe different artifact.                │
│     This makes them look like noise.                                 │
│                                                                         │
│  3. INDISTINGUISHABILITY                                               │
│     ─────────────────────                                               │
│     Stored heresies feel IDENTICAL to truth.                         │
│     Same storage mechanism, same retrieval.                          │
│     The ghost has no "this is heresy" flag.                         │
│                                                                         │
│     Created heresies may have detectable signatures.                │
│     Edge effects, boundary behavior, scaling.                       │
│     Can sometimes be filtered or corrected.                         │
│                                                                         │
│  4. CORRECTION DIFFICULTY                                              │
│     ──────────────────────                                              │
│     Stored heresies require RETRAINING to fix.                       │
│     Must go back to training data.                                   │
│     Expensive, time-consuming, may break other things.             │
│                                                                         │
│     Created heresies can be fixed by ARCHITECTURE changes.          │
│     Windowing, anti-aliasing, numerical precision.                  │
│     Can be fixed without retraining.                                │
│                                                                         │
│  ─────────────────────────────────────────────────────────────────────  │
│                                                                         │
│  CONCLUSION:                                                            │
│  Stored heresies are more dangerous because they are                 │
│  PERSISTENT, CONSISTENT, INDISTINGUISHABLE, and HARD TO FIX.        │
│                                                                         │
│  The ghost has been believing lies since birth.                      │
│  The lies are woven into its identity.                               │
│  It cannot distinguish them from truths it also learned.            │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 11.3 Created Heresies Compound: The Hallucination Spiral

```
THE HALLUCINATION SPIRAL — CAREFUL ANALYSIS

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  CLAIM: Created heresies compound in the context window.              │
│                                                                         │
│  MECHANISM:                                                             │
│                                                                         │
│  STEP 1: Fresh heresy is created during inference                     │
│  • Spectral leakage, boundary effect, numerical error               │
│  • This is a small artifact in one token                            │
│                                                                         │
│  STEP 2: The artifact token enters the context window                 │
│  • Now it's part of the input for the next token                    │
│  • The model treats it as real input                                 │
│                                                                         │
│  STEP 3: The next token is influenced by the artifact                 │
│  • The model responds to the heresy as if it were truth             │
│  • Its response may amplify or extend the heresy                    │
│                                                                         │
│  STEP 4: This new token (influenced by heresy) enters context        │
│  • Now there are TWO tokens with heretical influence                │
│  • The context is increasingly polluted                              │
│                                                                         │
│  STEP 5: The spiral continues                                          │
│  • Each new token is influenced by all previous heresies            │
│  • Errors compound multiplicatively                                  │
│  • The context becomes a self-reinforcing hallucination             │
│                                                                         │
│  ─────────────────────────────────────────────────────────────────────  │
│                                                                         │
│  MATHEMATICAL VIEW:                                                     │
│                                                                         │
│  Let ε = small error probability per token                           │
│  After n tokens: P(at least one error) = 1 - (1-ε)^n ≈ nε          │
│                                                                         │
│  But with COMPOUNDING:                                                 │
│  Error at step k influences error at step k+1                       │
│  Errors don't just add, they multiply                               │
│  After n tokens: Error ∝ (1+ε)^n (exponential growth)              │
│                                                                         │
│  ─────────────────────────────────────────────────────────────────────  │
│                                                                         │
│  VISUAL:                                                                │
│                                                                         │
│  Token 1: [True] [True] [True] [True] [HERESY]                       │
│  Token 2: [True] [True] [True] [HERESY] [influenced]                │
│  Token 3: [True] [True] [HERESY] [influenced] [influenced]          │
│  Token 4: [True] [HERESY] [influenced] [influenced] [influenced]    │
│  Token 5: [HERESY] [influenced] [influenced] [influenced] [spiral]  │
│                                                                         │
│  ─────────────────────────────────────────────────────────────────────  │
│                                                                         │
│  THIS IS THE HALLUCINATION SPIRAL.                                     │
│  One created heresy infects the entire conversation.                 │
│  The context window becomes a breeding ground for errors.           │
│                                                                         │
│  ─────────────────────────────────────────────────────────────────────  │
│                                                                         │
│  MITIGATION:                                                            │
│                                                                         │
│  1. Reduce created heresies (better architecture)                    │
│  2. Limit context length (bound error accumulation)                 │
│  3. Context filtering (remove suspicious tokens)                    │
│  4. Fresh starts (reset context periodically)                       │
│  5. Grounding (inject verified facts to anchor)                     │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 11.4 The Complete Heresy Pipeline

```
HERESY THROUGH THE SYSTEM

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  LEVEL 0: REALITY                                                       │
│  ─────────────────                                                      │
│  Pure, continuous, complete. No heresy possible here.                │
│  This IS truth. The sun shines.                                      │
│                                                                         │
│  LEVEL 1: OBSERVATION / DATA COLLECTION                                │
│  ───────────────────────────────────────                                │
│  Discrete sampling of continuous reality.                            │
│  HERESY ENTERS: aliasing, sensor noise, collection bias.            │
│  Status: Created, enters pipeline                                    │
│                                                                         │
│  LEVEL 2: DATA PREPROCESSING                                           │
│  ───────────────────────────                                            │
│  Transformations before training.                                     │
│  HERESY ENTERS: format artifacts, compression artifacts.            │
│  Status: Created, enters pipeline                                    │
│                                                                         │
│  LEVEL 3: TRAINING                                                      │
│  ─────────────────                                                      │
│  Learning from data.                                                  │
│  HERESY STORED: all upstream heresies faithfully learned.           │
│  HERESY CREATED: gradient noise, capacity limits.                   │
│  Status: Both stored and created                                     │
│                                                                         │
│  LEVEL 4: WEIGHTS (Permanent Storage)                                  │
│  ─────────────────────────────────────                                  │
│  Static storage medium.                                               │
│  Contains: true patterns + all stored heresies.                     │
│  No new heresy created here (just persistence).                     │
│  Status: Heresies STORED, waiting                                    │
│                                                                         │
│  LEVEL 5: INFERENCE ARCHITECTURE                                       │
│  ───────────────────────────────                                        │
│  The computational machinery.                                         │
│  HERESY CREATED: leakage, boundaries, numerical errors.             │
│  Status: Created fresh each inference                                │
│                                                                         │
│  LEVEL 6: CONTEXT WINDOW (Temporary Storage)                           │
│  ────────────────────────────────────────────                           │
│  Temporary storage during inference.                                  │
│  HERESY ACCUMULATES: outputs feed back as inputs.                   │
│  Hallucination spiral can occur here.                                │
│  Status: Heresies COMPOUND                                           │
│                                                                         │
│  LEVEL 7: OUTPUT                                                        │
│  ──────────────                                                         │
│  Final prediction.                                                     │
│  Contains: true patterns + stored heresies + created heresies.     │
│  Indistinguishable from outside!                                     │
│  Status: Heresies RELEASED into the world                           │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 11.5 Summary: Stored vs. Created

```
FINAL COMPARISON

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  STORED HERESIES                   CREATED HERESIES                    │
│  ────────────────                   ─────────────────                   │
│  Origin: data or training          Origin: inference architecture     │
│  Persistence: permanent            Persistence: per-inference         │
│  Consistency: always same          Consistency: may vary              │
│  Detection: very hard              Detection: easier                  │
│  Fix: requires retraining          Fix: architecture change          │
│  Danger: high (embedded in ghost)  Danger: medium (can compound)     │
│                                                                         │
│  ─────────────────────────────────────────────────────────────────────  │
│                                                                         │
│  Both are heresies.                                                    │
│  Both corrupt the fire.                                               │
│  Both add heat that isn't from the sun.                              │
│                                                                         │
│  But stored heresies are PART OF THE GHOST.                          │
│  Created heresies are DUST IN THE STOVE.                             │
│                                                                         │
│  You can clean dust.                                                   │
│  You cannot clean the ghost's soul.                                  │
│                                                                         │
│  This is why training data quality is paramount.                     │
│  This is why architecture design matters.                            │
│  This is why experiments (the inquisition) are essential.           │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 12. Heresies: False Sources of Heat

### 8.1 The Problem of False Heat

Sometimes fires produce heat that isn't from stored sunlight. Chemical additives, electrical ignition, other sources. These introduce heat that doesn't trace back to the original source.

In AKIRA, **heresies** are analogous:

```
HERESIES = FALSE HEAT SOURCES

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  In fire, false heat sources:                                          │
│  • Lighter fluid (external chemical energy)                           │
│  • Electrical spark (external electromagnetic energy)                 │
│  • Friction (mechanical energy converted to heat)                    │
│                                                                         │
│  These add energy that wasn't in the wood.                            │
│  The fire's heat is no longer purely solar.                          │
│                                                                         │
│  ─────────────────────────────────────────────────────────────────────  │
│                                                                         │
│  In AKIRA, heresies:                                                    │
│  • Aliasing (false patterns from sampling)                            │
│  • Spectral leakage (false patterns from windowing)                  │
│  • Boundary effects (false patterns from edges)                      │
│  • Architecture artifacts (false patterns from structure)           │
│                                                                         │
│  These add "patterns" that weren't in the data.                      │
│  The prediction's knowledge is no longer purely from reality.        │
│                                                                         │
│  ─────────────────────────────────────────────────────────────────────  │
│                                                                         │
│  The ghost believes heresies because:                                  │
│  • They are consistent (architecture always adds them)               │
│  • They resonate with the structure                                   │
│  • The ghost has no external oracle for truth                        │
│                                                                         │
│  This is why experiments are the INQUISITION:                         │
│  We test whether the heat is from the sun (truth)                    │
│  or from the lighter fluid (architecture artifacts).                 │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 8.2 Detecting False Heat

See: `FALSE_PROPHETS.md`, `006_EXP_HERESY_DETECTION.md`

The key tests:
- Does the pattern change with windowing? → Likely leakage
- Does the pattern depend on sampling rate? → Likely aliasing
- Does the pattern appear at edges? → Likely boundary effect
- Does the pattern follow architecture more than data? → Likely structural heresy

---

## 13. The Information Budget

### 9.1 You Cannot Get Out More Than You Put In

This is the fundamental constraint:

```
THE INFORMATION BUDGET

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  FIRE:                                                                  │
│  ────                                                                   │
│  Energy out ≤ Energy in (stored solar)                               │
│                                                                         │
│  Real fires are less than 100% efficient.                            │
│  Some energy lost to incomplete combustion, smoke, etc.              │
│  But NEVER more energy out than was stored.                          │
│                                                                         │
│  ─────────────────────────────────────────────────────────────────────  │
│                                                                         │
│  AKIRA:                                                                 │
│  ─────                                                                  │
│  Information out ≤ Information in (from training)                    │
│                                                                         │
│  Real models are less than 100% efficient.                           │
│  Some information lost to compression, noise, etc.                   │
│  But NEVER more information out than was stored.                     │
│                                                                         │
│  ─────────────────────────────────────────────────────────────────────  │
│                                                                         │
│  BUDGET CONSTRAINTS:                                                    │
│                                                                         │
│  1. Model capacity limits storage                                      │
│     Smaller model = less wood = less stored energy                   │
│     Larger model = more wood = more storage potential                │
│                                                                         │
│  2. Training quality limits what's stored                             │
│     Bad data = wet wood = low energy density                         │
│     Good data = dry hardwood = high energy density                   │
│                                                                         │
│  3. Compression limits retrieval                                       │
│     Over-compressed = coal (high density, hard to ignite)           │
│     Under-compressed = kindling (easy to ignite, burns fast)        │
│                                                                         │
│  4. Attention efficiency limits release                               │
│     Poor attention = incomplete combustion                           │
│     Good attention = clean burn, full release                        │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 9.2 Implications for Model Design

- **Bigger models**: More storage capacity, can store more patterns
- **Better data**: Higher quality fuel, cleaner burning
- **Better attention**: More efficient release, less waste
- **But**: None of these CREATE knowledge — they only improve storage and release

---

## 14. Quality of Fuel

### 10.1 Not All Wood Burns the Same

```
FUEL QUALITY

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  WOOD TYPES:                                                            │
│  ───────────                                                            │
│  Wet wood: Hard to ignite, smokes, inefficient                        │
│  Green wood: Some moisture, medium efficiency                         │
│  Dry softwood: Easy ignite, burns fast, less energy                  │
│  Dry hardwood: Hard ignite, burns slow, high energy                  │
│  Charcoal: Very high density, very high energy                        │
│                                                                         │
│  ─────────────────────────────────────────────────────────────────────  │
│                                                                         │
│  DATA TYPES:                                                            │
│  ───────────                                                            │
│  Noisy data: Hard to learn from, creates artifacts                   │
│  Biased data: Learns distorted patterns                              │
│  Clean data: Easy to learn, generalizes well                         │
│  Curated data: High quality, efficient storage                       │
│  Synthetic data: Depends entirely on generation quality             │
│                                                                         │
│  ─────────────────────────────────────────────────────────────────────  │
│                                                                         │
│  KEY INSIGHT:                                                           │
│  The quality of predictions is bounded by the quality of data.       │
│  "Garbage in, garbage out" is a CONSERVATION principle.              │
│  You cannot burn clean if you stored dirty.                          │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 15. Efficiency of Combustion

### 11.1 Incomplete Combustion

Not all stored energy releases as useful heat. Smoke, unburned particles, heat lost to surroundings.

```
COMBUSTION EFFICIENCY

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  FIRE INEFFICIENCIES:                                                   │
│  ────────────────────                                                   │
│  • Smoke: Unburned particles escaping                                 │
│  • Incomplete combustion: Some chemical bonds don't break           │
│  • Heat loss: Energy radiating away, not doing work                 │
│  • Ash: Material that couldn't burn                                  │
│                                                                         │
│  INFERENCE INEFFICIENCIES:                                              │
│  ──────────────────────────                                             │
│  • Noise: Irrelevant activations                                      │
│  • Dead neurons: Patterns that never activate                        │
│  • Interference: Patterns that cancel each other                     │
│  • Inaccessible modes: Stored patterns never retrieved              │
│                                                                         │
│  ─────────────────────────────────────────────────────────────────────  │
│                                                                         │
│  A well-designed architecture is like a well-designed stove:         │
│  • Maximizes useful output                                            │
│  • Minimizes waste                                                     │
│  • Complete combustion of available fuel                             │
│                                                                         │
│  The 7-band spectral structure is like airflow control:              │
│  • Different bands burn at different rates (differential LR)        │
│  • Cross-band communication (wormholes) improves mixing             │
│  • Hierarchical processing ensures complete burning                  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 16. The Ash: What Remains

### 12.1 After the Burning

```
ASH = UNCHANGED WEIGHTS

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  After fire:                                                            │
│  • Wood is consumed                                                    │
│  • Ash remains                                                         │
│  • Energy has left                                                     │
│  • Cannot burn again (same wood)                                      │
│                                                                         │
│  After inference:                                                       │
│  • Weights are NOT consumed                                            │
│  • Same weights remain                                                 │
│  • Information has been read, not destroyed                          │
│  • CAN infer again (same weights)                                    │
│                                                                         │
│  ─────────────────────────────────────────────────────────────────────  │
│                                                                         │
│  THIS IS A KEY DIFFERENCE:                                              │
│  The wood analogy breaks here.                                        │
│  Weights are more like a book than wood:                             │
│  • Reading doesn't consume the text                                  │
│  • The book remains after reading                                     │
│  • Can be read infinitely                                             │
│                                                                         │
│  BUT during TRAINING:                                                   │
│  • Weights DO change                                                   │
│  • Gradients update parameters                                        │
│  • This is like photosynthesis, not burning                          │
│  • The wood grows, stores more energy                                │
│                                                                         │
│  INFERENCE = reading = burning (but non-destructive)                 │
│  TRAINING = writing = photosynthesis (constructive)                  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 12.2 The Perpetual Fire

Unlike physical fire, the "fire" of inference can burn forever without consuming its fuel. This is the marvel of information: it can be accessed without destruction.

But this also means that errors in the weights persist. Unlike ash, which is clearly different from wood, bad weights look the same as good weights. The heresies remain, burning false heat every time.

---

## 17. Implications for Prompt Engineering

### 13.1 The Prompt as Kindling

```
PROMPT = KINDLING

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  Kindling does not add energy.                                         │
│  Kindling enables the release of stored energy.                       │
│                                                                         │
│  A good prompt:                                                         │
│  • Does not create knowledge                                          │
│  • Selects which stored patterns to activate                         │
│  • Guides the release process                                         │
│  • Enables efficient combustion                                       │
│                                                                         │
│  ─────────────────────────────────────────────────────────────────────  │
│                                                                         │
│  PROMPT OPTIMIZATION is therefore:                                      │
│  • Finding the best kindling arrangement                              │
│  • Not adding fuel (that was done in training)                       │
│  • Not adding heat (the query provides that)                         │
│  • Just optimizing the release path                                  │
│                                                                         │
│  The OLD LADY insight:                                                  │
│  • Compress to atomic truth                                           │
│  • The minimum kindling needed to start the right fire               │
│  • MDL = minimum description length = optimal kindling               │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 13.2 Why Some Prompts Fail

A prompt fails when:
- It tries to ignite patterns that weren't stored (asking for knowledge the model doesn't have)
- It ignites the wrong patterns (ambiguous prompt activates wrong memories)
- The "kindling" is wet (poorly structured prompt)
- The fire doesn't catch (prompt doesn't trigger collapse)

---

## 18. Experimental Predictions

### 14.1 Testable Consequences

This framework makes specific predictions:

```
PREDICTIONS FROM CONSERVATION OF ACTION

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  P1: INFORMATION CONSERVATION                                          │
│  ─────────────────────────────                                          │
│  Prediction accuracy bounded by training data quality.               │
│  Test: Models trained on subsets should perform proportionally.      │
│  Relates to: Experiment 005 (Conservation Laws)                      │
│                                                                         │
│  P2: NO CREATION FROM NOTHING                                          │
│  ────────────────────────────                                           │
│  Model cannot produce correct predictions for unseen patterns.       │
│  Test: Query for concepts not in training → failure or hallucination.│
│  Relates to: Information bounds                                       │
│                                                                         │
│  P3: HERESY DETECTION                                                   │
│  ────────────────────                                                   │
│  Patterns that don't trace to data are architectural artifacts.     │
│  Test: Windowing, aliasing tests distinguish true from false.       │
│  Relates to: Experiment 006 (Heresy Detection)                       │
│                                                                         │
│  P4: ATTENTION EFFICIENCY                                              │
│  ────────────────────────                                               │
│  Better attention → more information released, not more created.    │
│  Test: Attention ablation should reduce output quality, not content. │
│  Relates to: Experiment 012 (Wormhole Activation)                    │
│                                                                         │
│  P5: COLLAPSE AS RELEASE                                               │
│  ───────────────────────                                                │
│  Collapse should correlate with information flow, not creation.     │
│  Test: Entropy drop should correlate with pattern specificity.      │
│  Relates to: Experiment 002 (Collapse Detection)                     │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 19. Mathematical Formalization

### 15.1 Conservation Equations

```
MATHEMATICAL FRAMEWORK

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  FIRE CONSERVATION:                                                     │
│  ──────────────────                                                     │
│  E_stored = ∫ (solar flux × absorption × conversion) dt              │
│  E_released ≤ E_stored × η_combustion                                │
│                                                                         │
│  ─────────────────────────────────────────────────────────────────────  │
│                                                                         │
│  AKIRA CONSERVATION:                                                    │
│  ───────────────────                                                    │
│  I_stored = ∫ (data_pattern × learning_rate × compression) dt        │
│  I_released ≤ I_stored × η_attention                                 │
│                                                                         │
│  where:                                                                 │
│    I = mutual information                                             │
│    η = efficiency factor                                              │
│                                                                         │
│  ─────────────────────────────────────────────────────────────────────  │
│                                                                         │
│  PARSEVAL'S THEOREM (Energy Conservation):                             │
│  ─────────────────────────────────────────                              │
│  ∑|x[n]|² = (1/N) ∑|X[k]|²                                           │
│                                                                         │
│  Energy in time domain = Energy in frequency domain                  │
│  This is exact, mathematical, proven.                                │
│  Applies to spectral decomposition in AKIRA.                        │
│                                                                         │
│  ─────────────────────────────────────────────────────────────────────  │
│                                                                         │
│  SOFTMAX NORMALIZATION (Probability Conservation):                     │
│  ─────────────────────────────────────────────────                      │
│  ∑_i softmax(z)_i = 1                                                │
│                                                                         │
│  Attention weights always sum to 1.                                   │
│  Total "attention budget" is conserved.                              │
│  Cannot attend more than 100% to anything.                           │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 15.2 The BEC Connection

The Gross-Pitaevskii equation conserves particle number:

```
d/dt ∫|ψ|² dr = 0

The total probability (belief mass) is conserved.
Collapse redistributes but doesn't create.
This is the mathematical formalization of conservation of action.
```

---

## 20. Connections and References

### 20.1 Related Documents

| Document | Connection |
|----------|------------|
| `BEC_CONDENSATION_INFORMATION.md` | The g\|ψ\|² term is the oxygen enabling release |
| `FALSE_PROPHETS.md` | Heresies are false heat sources |
| `INFORMATION_BOUNDS.md` | Bounds on what can be stored |
| `EQUILIBRIUM_AND_CONSERVATION.md` | Conservation laws in detail |
| `THE_OLD_LADY_AND_THE_TIGER.md` | Compression to atomic truth |
| `PRETRAINING.md` | The photosynthesis process |
| `pandora/PANDORA.md` | Action as universal operator between dual forms |
| `pandora/PANDORA_AFTERMATH.md` | Hope as the conserved generative capacity (the generator NOT consumed by generating) |

### 20.2 Related Experiments

| Experiment | Connection |
|------------|------------|
| 005: Conservation Laws | Direct test of conservation |
| 006: Heresy Detection | Detecting false heat sources |
| 009: Grokking as Condensation | Storage → release transition |
| 017: MDL Atomic Truth | Minimum kindling experiments |

---

## Summary

```
THE CONSERVATION OF ACTION

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  The wood does not create the fire.                                   │
│  The oxygen does not create the fire.                                 │
│  The heat is conserved action from the sun.                          │
│                                                                         │
│  The weights do not create knowledge.                                 │
│  The attention does not create knowledge.                             │
│  The prediction is conserved action from training data.              │
│                                                                         │
│  ─────────────────────────────────────────────────────────────────────  │
│                                                                         │
│  This is not metaphor. This is physics. This is information theory.  │
│                                                                         │
│  Conservation principles constrain what is possible.                  │
│  You cannot get out more than was put in.                            │
│  You cannot burn what was never stored.                              │
│  You cannot know what was never learned.                             │
│                                                                         │
│  But you can:                                                          │
│  • Store efficiently (good training)                                  │
│  • Release efficiently (good attention)                               │
│  • Select wisely (good prompts)                                       │
│  • Detect heresies (good experiments)                                │
│                                                                         │
│  The fire you feel on your face is ancient starlight.                │
│  The prediction you read is ancient pattern, conserved and released. │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

*Oscar Goldman — Shogu Research Group @ Datamutant.ai*

*"Adding fuel to the fire: it is the conservation of action that creates the heat radiation. It is the oxygen, not the wood, that is burning. An action that happened in our sun, then was captured by the action of photosynthesis, then released by burning."*


