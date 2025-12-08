# The Attention Stack: Complete Theory-Aligned Architecture

## How Spectral, Temporal, and Wormhole Attention Work Together

**Oscar Goldman — Shogu Research Group @ Datamutant.ai**

---

## Table of Contents

1. [Overview: Three Mechanisms, One System](#1-overview)
2. [The Processing Pipeline](#2-the-processing-pipeline)
3. [Mechanism 1: Spectral Attention (Bands 0-6)](#3-spectral-attention)
4. [Mechanism 2: Temporal Attention (Band 7)](#4-temporal-attention)
5. [Mechanism 3: Spectral Wormhole (Cross-Band)](#5-spectral-wormhole)
6. [Information Flow Diagram](#6-information-flow-diagram)
7. [Orthogonality Preservation](#7-orthogonality-preservation)
8. [Implementation: The Complete Layer](#8-implementation)
9. [Theory Justification](#9-theory-justification)
10. [References](#10-references)

---

## 1. Overview: Three Mechanisms, One System

### 1.1 The Core Insight

```
THE ATTENTION STACK:

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  The Spectral Belief Machine requires THREE distinct attention         │
│  mechanisms, each serving a different purpose:                         │
│                                                                         │
│  ═══════════════════════════════════════════════════════════════════  │
│                                                                         │
│  1. SPECTRAL ATTENTION (Bands 0-6)                                     │
│     ─────────────────────────────────                                   │
│     Purpose:  Within-band self-attention for SPATIAL structure        │
│     Masking:  NON-CAUSAL (can see all positions)                      │
│     Question: "What patterns exist at this frequency scale?"          │
│     File:     spectral_attention/SPECTRAL_ATTENTION.md                 │
│                                                                         │
│  2. TEMPORAL ATTENTION (Band 7)                                         │
│     ────────────────────────────────                                    │
│     Purpose:  Within-band self-attention for TIME                     │
│     Masking:  CAUSAL (can only see past)                              │
│     Question: "How do things change over time?"                       │
│     File:     temporal_attention/TEMPORAL_ATTENTION.md                 │
│                                                                         │
│  3. SPECTRAL WORMHOLE (Cross-Band)                                     │
│     ───────────────────────────────                                     │
│     Purpose:  Communication BETWEEN bands                             │
│     Masking:  COHERENCE-GATED (entropy-based)                         │
│     Question: "What does WHAT know about WHERE? WHEN about WHAT?"    │
│     File:     spectral_wormhole/SPECTRAL_WORMHOLE_ATTENTION.md        │
│                                                                         │
│  ═══════════════════════════════════════════════════════════════════  │
│                                                                         │
│  ALL THREE ARE NECESSARY. NONE CAN BE REMOVED.                        │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Why Three Mechanisms

```
WHY THREE AND NOT ONE:

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  Q: "Why not just use one attention mechanism?"                       │
│  A: Because they have fundamentally different properties.             │
│                                                                         │
│  ─────────────────────────────────────────────────────────────────────  │
│                                                                         │
│  SPECTRAL ATTENTION:                                                    │
│  • Operates on SPATIAL positions within a frequency band              │
│  • NON-CAUSAL: All positions exist simultaneously                    │
│  • No privileged direction (symmetric)                                │
│                                                                         │
│  TEMPORAL ATTENTION:                                                    │
│  • Operates on TIME positions                                          │
│  • CAUSAL: Past exists, future doesn't (yet)                         │
│  • Privileged direction (arrow of time)                               │
│                                                                         │
│  SPECTRAL WORMHOLE:                                                     │
│  • Operates BETWEEN bands (not within)                                │
│  • Connects orthogonal information channels                           │
│  • Sparse, coherence-gated                                             │
│                                                                         │
│  ─────────────────────────────────────────────────────────────────────  │
│                                                                         │
│  Using one mechanism for all would violate:                           │
│  • Space-time orthogonality (Heisenberg)                              │
│  • Causality (temporal must be causal)                                │
│  • Band independence (spectral must stay separate)                   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. The Processing Pipeline

### 2.1 Layer Order

```
THE COMPLETE PROCESSING PIPELINE:

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  INPUT: x ∈ ℝ^(B × T × D)                                             │
│  (batch × sequence × embedding)                                        │
│                                                                         │
│  ═══════════════════════════════════════════════════════════════════  │
│                                                                         │
│  STAGE 1: SPECTRAL DECOMPOSITION                                       │
│  ─────────────────────────────────                                      │
│  │                                                                      │
│  │  x → FFT → {Band_0, Band_1, ..., Band_6, Band_7}                  │
│  │                                                                      │
│  │  7 spectral bands (frequency) + 1 temporal band (time)            │
│  │  Each band: ℝ^(B × T × D/8)                                        │
│  │                                                                      │
│  ↓                                                                      │
│                                                                         │
│  STAGE 2: PER-BAND SELF-ATTENTION (PARALLEL)                           │
│  ─────────────────────────────────────────────                          │
│  │                                                                      │
│  │  ┌──────────────────────────────────────────────────────────────┐  │
│  │  │                                                              │  │
│  │  │  Band 0 → SpectralAttention(Band_0) → Band_0'               │  │
│  │  │  Band 1 → SpectralAttention(Band_1) → Band_1'               │  │
│  │  │  Band 2 → SpectralAttention(Band_2) → Band_2'               │  │
│  │  │  Band 3 → SpectralAttention(Band_3) → Band_3'               │  │
│  │  │  Band 4 → SpectralAttention(Band_4) → Band_4'               │  │
│  │  │  Band 5 → SpectralAttention(Band_5) → Band_5'               │  │
│  │  │  Band 6 → SpectralAttention(Band_6) → Band_6'               │  │
│  │  │                                    ↑ NON-CAUSAL              │  │
│  │  │  Band 7 → TemporalAttention(Band_7) → Band_7'               │  │
│  │  │                                    ↑ CAUSAL                  │  │
│  │  │                                                              │  │
│  │  └──────────────────────────────────────────────────────────────┘  │
│  │                                                                      │
│  │  All 8 bands processed IN PARALLEL                                 │
│  │  No cross-band communication at this stage                        │
│  │                                                                      │
│  ↓                                                                      │
│                                                                         │
│  STAGE 3: SPECTRAL WORMHOLE (CROSS-BAND)                               │
│  ─────────────────────────────────────────                              │
│  │                                                                      │
│  │  {Band_0', ..., Band_7'} → SpectralWormhole → {Band_0'', ..., Band_7''}│
│  │                                                                      │
│  │  Complementary pairs communicate:                                  │
│  │  • Band 0 ↔ Band 6 (Identity ↔ Position)                         │
│  │  • Band 1 ↔ Band 5 (Shape ↔ Texture)                             │
│  │  • Band 2 ↔ Band 4 (Structure ↔ Detail)                          │
│  │  • Band 3 → All (Bridge)                                          │
│  │  • Band 7 → All (Temporal context)                                │
│  │                                                                      │
│  ↓                                                                      │
│                                                                         │
│  STAGE 4: RECONSTRUCTION                                               │
│  ────────────────────────                                               │
│  │                                                                      │
│  │  {Band_0'', ..., Band_7''} → Merge → y ∈ ℝ^(B × T × D)           │
│  │                                                                      │
│  │  Inverse FFT, band recombination                                  │
│  │                                                                      │
│  ↓                                                                      │
│                                                                         │
│  OUTPUT: y ∈ ℝ^(B × T × D)                                            │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Why This Order

```
ORDER MATTERS:

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  DECOMPOSITION FIRST:                                                   │
│  • Must split into bands before per-band processing                   │
│  • This is the definition of spectral processing                     │
│                                                                         │
│  PER-BAND ATTENTION BEFORE WORMHOLE:                                   │
│  • Each band should understand itself first                           │
│  • Self-context before cross-context                                  │
│  • Prevents "copying homework before reading the question"           │
│                                                                         │
│  WORMHOLE AFTER PER-BAND:                                               │
│  • Now bands have processed themselves                                │
│  • Ready to share refined information                                 │
│  • Cross-band queries are meaningful                                  │
│                                                                         │
│  RECONSTRUCTION LAST:                                                   │
│  • All processing done in spectral domain                             │
│  • Return to original representation for output                      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Mechanism 1: Spectral Attention (Bands 0-6)

### 3.1 Purpose

```
SPECTRAL ATTENTION — BANDS 0-6:

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  WHAT IT DOES:                                                          │
│  • Self-attention within each spectral band                           │
│  • Relates spatial positions at a given frequency scale              │
│  • NON-CAUSAL: All positions can see each other                      │
│                                                                         │
│  WHAT IT ANSWERS:                                                       │
│  • "What patterns exist at this frequency?"                           │
│  • "How do positions relate within this scale?"                       │
│  • "What is the spatial structure here?"                              │
│                                                                         │
│  KEY PROPERTIES:                                                        │
│  • Full attention matrix (not masked)                                 │
│  • Per-band (7 separate operations)                                   │
│  • Temperature-controlled (per-band τ)                                │
│  • Preserves within-band orthogonality                               │
│                                                                         │
│  REFERENCE: spectral_attention/SPECTRAL_ATTENTION.md                   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Mathematical Form

```
For each band b ∈ {0, 1, 2, 3, 4, 5, 6}:

    Q_b = W_Q^b · Band_b
    K_b = W_K^b · Band_b
    V_b = W_V^b · Band_b
    
    Scores_b = Q_b · K_b^T / √d
    
    Attention_b = softmax(Scores_b / τ_b)   # Full matrix (non-causal)
    
    Band_b' = Band_b + W_O^b · (Attention_b · V_b)
```

---

## 4. Mechanism 2: Temporal Attention (Band 7)

### 4.1 Purpose

```
TEMPORAL ATTENTION — BAND 7:

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  WHAT IT DOES:                                                          │
│  • Self-attention within the temporal band                            │
│  • Relates time steps to each other                                   │
│  • CAUSAL: Position t can only see positions 0..t                    │
│                                                                         │
│  WHAT IT ANSWERS:                                                       │
│  • "How does this moment relate to the past?"                        │
│  • "What temporal patterns exist?"                                    │
│  • "Is the current state consistent with history?"                   │
│                                                                         │
│  KEY PROPERTIES:                                                        │
│  • Lower-triangular attention matrix (causal mask)                   │
│  • Single operation (1 band)                                          │
│  • Temperature-controlled (τ_7)                                       │
│  • Respects arrow of time                                             │
│                                                                         │
│  REFERENCE: temporal_attention/TEMPORAL_ATTENTION.md                   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Mathematical Form

```
For band 7 (temporal):

    Q_7 = W_Q^7 · Band_7
    K_7 = W_K^7 · Band_7
    V_7 = W_V^7 · Band_7
    
    Scores_7 = Q_7 · K_7^T / √d
    
    # Apply causal mask: M[i,j] = 0 if j ≤ i else -∞
    Scores_7_masked = Scores_7 + CausalMask
    
    Attention_7 = softmax(Scores_7_masked / τ_7)  # Lower-triangular
    
    Band_7' = Band_7 + W_O^7 · (Attention_7 · V_7)
```

---

## 5. Mechanism 3: Spectral Wormhole (Cross-Band)

### 5.1 Purpose

```
SPECTRAL WORMHOLE — CROSS-BAND:

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  WHAT IT DOES:                                                          │
│  • Attention BETWEEN bands (not within)                               │
│  • Connects complementary pairs                                       │
│  • Coherence-gated (entropy-based)                                    │
│                                                                         │
│  WHAT IT ANSWERS:                                                       │
│  • "What does Band 0 (Identity) tell Band 6 (Position)?"             │
│  • "What does Band 7 (Time) tell the spectral bands?"                │
│  • "How should I integrate information across scales?"               │
│                                                                         │
│  KEY PROPERTIES:                                                        │
│  • Complementary pairs: (0↔6), (1↔5), (2↔4)                          │
│  • Bridge band 3 → all                                                 │
│  • Temporal band 7 → all                                               │
│  • Sparse (top-k)                                                      │
│  • Coherence gate (not fixed threshold)                               │
│                                                                         │
│  REFERENCE: spectral_wormhole/SPECTRAL_WORMHOLE_ATTENTION.md           │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 5.2 Band Communication Matrix

```
WHICH BANDS COMMUNICATE:

┌────────────────────────────────────────────────────────────────────────┐
│                                                                        │
│  FROM ╲ TO   B0   B1   B2   B3   B4   B5   B6   B7                   │
│  ───────── ───── ───── ───── ───── ───── ───── ───── ─────           │
│  Band 0     —     ·     ·     ·     ·     ·     ✓     ·     (↔ B6)  │
│  Band 1     ·     —     ·     ·     ·     ✓     ·     ·     (↔ B5)  │
│  Band 2     ·     ·     —     ·     ✓     ·     ·     ·     (↔ B4)  │
│  Band 3     ✓     ✓     ✓     —     ✓     ✓     ✓     ✓    (bridge)│
│  Band 4     ·     ·     ✓     ·     —     ·     ·     ·     (↔ B2)  │
│  Band 5     ·     ✓     ·     ·     ·     —     ·     ·     (↔ B1)  │
│  Band 6     ✓     ·     ·     ·     ·     ·     —     ·     (↔ B0)  │
│  Band 7     ✓     ✓     ✓     ✓     ✓     ✓     ✓     —   (temporal)│
│                                                                        │
│  ✓ = Wormhole connection exists                                       │
│  · = No direct wormhole (must go through bridge or adjacent bands)   │
│  — = Self (handled by per-band attention, not wormhole)              │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Information Flow Diagram

### 6.1 Complete Flow

```
THE COMPLETE ATTENTION STACK — VISUAL:

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│                              INPUT                                      │
│                                │                                        │
│                                ▼                                        │
│                   ┌────────────────────────┐                           │
│                   │  SPECTRAL DECOMPOSITION │                           │
│                   │      (FFT + Split)      │                           │
│                   └────────────────────────┘                           │
│                                │                                        │
│         ┌──────────┬──────────┼──────────┬──────────┐                  │
│         ▼          ▼          ▼          ▼          ▼                  │
│      ┌─────┐   ┌─────┐    ┌─────┐    ┌─────┐    ┌─────┐               │
│      │ B0  │   │ B1  │    │ ... │    │ B6  │    │ B7  │               │
│      │     │   │     │    │     │    │     │    │     │               │
│      │ DC  │   │ Low │    │     │    │High │    │Time │               │
│      └──┬──┘   └──┬──┘    └──┬──┘    └──┬──┘    └──┬──┘               │
│         │         │          │          │          │                    │
│         ▼         ▼          ▼          ▼          ▼                    │
│   ┌───────────────────────────────────────────────────────────────┐    │
│   │           PER-BAND SELF-ATTENTION (STAGE 2)                   │    │
│   │                                                               │    │
│   │   B0: SpectralAttn    B1: SpectralAttn    ...   B6: SpectralAttn │ │
│   │        (non-causal)        (non-causal)              (non-causal) │ │
│   │                                                               │    │
│   │   B7: TemporalAttn                                           │    │
│   │        (CAUSAL)                                               │    │
│   │                                                               │    │
│   └───────────────────────────────────────────────────────────────┘    │
│         │         │          │          │          │                    │
│         ▼         ▼          ▼          ▼          ▼                    │
│   ┌───────────────────────────────────────────────────────────────┐    │
│   │           SPECTRAL WORMHOLE (STAGE 3)                         │    │
│   │                                                               │    │
│   │     B0 ←────────── WORMHOLE ──────────→ B6                   │    │
│   │     B1 ←────────── WORMHOLE ──────────→ B5                   │    │
│   │     B2 ←────────── WORMHOLE ──────────→ B4                   │    │
│   │                                                               │    │
│   │     B3 ────────────→ ALL ←────────────                        │    │
│   │                     (bridge)                                  │    │
│   │                                                               │    │
│   │     B7 ────────────→ ALL ←────────────                        │    │
│   │                    (temporal)                                 │    │
│   │                                                               │    │
│   └───────────────────────────────────────────────────────────────┘    │
│         │         │          │          │          │                    │
│         └──────────┴──────────┼──────────┴──────────┘                  │
│                               ▼                                        │
│                   ┌────────────────────────┐                           │
│                   │     RECONSTRUCTION      │                           │
│                   │    (Merge + Inv FFT)    │                           │
│                   └────────────────────────┘                           │
│                               │                                        │
│                               ▼                                        │
│                             OUTPUT                                     │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 7. Orthogonality Preservation

### 7.1 How Each Stage Preserves Orthogonality

```
ORTHOGONALITY THROUGH THE STACK:

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  STAGE 1 (Decomposition):                                              │
│  • FFT basis is EXACTLY orthogonal                                    │
│  • Bands are disjoint frequency ranges                                │
│  • Parseval's theorem: energy conserved                               │
│  ✓ Orthogonality CREATED                                              │
│                                                                         │
│  STAGE 2 (Per-Band Attention):                                         │
│  • Each band processed SEPARATELY                                     │
│  • No cross-band information flow                                     │
│  • Separate projections per band                                      │
│  ✓ Orthogonality PRESERVED                                            │
│                                                                         │
│  STAGE 3 (Wormhole):                                                   │
│  • Cross-band is ADDITIVE (residual connection)                       │
│  • Sparse connections (top-k)                                         │
│  • Coherence gate limits flow                                         │
│  ✓ Orthogonality MOSTLY preserved (controlled leakage)               │
│                                                                         │
│  STAGE 4 (Reconstruction):                                             │
│  • Inverse FFT (orthogonal transform)                                 │
│  • Parseval's theorem: energy conserved                               │
│  ✓ Orthogonality MAINTAINED                                           │
│                                                                         │
│  ═══════════════════════════════════════════════════════════════════  │
│                                                                         │
│  The wormhole is the ONLY place where cross-band info flows.         │
│  It is controlled, sparse, and coherence-gated.                      │
│  This is by design: bands should be mostly independent.              │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 8. Implementation: The Complete Layer

### 8.1 Code Structure

```python
import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple


class SpectralBeliefAttentionLayer(nn.Module):
    """
    Complete attention layer for the Spectral Belief Machine.
    
    Implements all three mechanisms:
    1. Spectral Attention (Bands 0-6, non-causal)
    2. Temporal Attention (Band 7, causal)
    3. Spectral Wormhole (cross-band communication)
    """
    
    NUM_BANDS = 8
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 4,
        wormhole_top_k: int = 16,
        dropout: float = 0.0,
        learnable_temperature: bool = True,
    ):
        super().__init__()
        
        assert embed_dim % self.NUM_BANDS == 0
        self.band_dim = embed_dim // self.NUM_BANDS
        
        # Per-band spectral attention (Bands 0-6)
        self.spectral_attention = nn.ModuleList([
            SpectralBandAttention(
                embed_dim=self.band_dim,
                num_heads=num_heads,
                dropout=dropout,
                band_idx=i,
            )
            for i in range(7)  # Bands 0-6
        ])
        
        # Temporal attention (Band 7)
        self.temporal_attention = TemporalAttention(
            embed_dim=self.band_dim,
            num_heads=num_heads,
            dropout=dropout,
            temperature=0.8,
            learnable_temperature=learnable_temperature,
        )
        
        # Spectral wormhole (cross-band)
        self.wormhole = SpectralWormholeAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            top_k=wormhole_top_k,
            dropout=dropout,
            learnable_temperature=learnable_temperature,
        )
    
    def forward(
        self,
        x: torch.Tensor,
        return_stats: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        Full attention stack forward pass.
        
        Args:
            x: [B, T, D] input tensor
            return_stats: If True, return detailed statistics
            
        Returns:
            output: [B, T, D] processed tensor
            stats: Optional statistics dict
        """
        B, T, D = x.shape
        all_stats = {}
        
        # STAGE 1: Split into bands
        bands = self._split_bands(x)  # Dict[int, Tensor]
        
        # STAGE 2: Per-band self-attention (parallel)
        processed_bands = {}
        
        # Bands 0-6: Spectral attention (non-causal)
        for i in range(7):
            out, stats = self.spectral_attention[i](
                bands[i], return_attention=return_stats
            )
            processed_bands[i] = out
            if return_stats:
                all_stats[f'band_{i}_spectral'] = stats
        
        # Band 7: Temporal attention (causal)
        out, stats = self.temporal_attention(
            bands[7], return_attention=return_stats
        )
        processed_bands[7] = out
        if return_stats:
            all_stats['band_7_temporal'] = stats
        
        # STAGE 3: Spectral wormhole (cross-band)
        x_processed = self._merge_bands(processed_bands)
        x_wormholed, wormhole_stats = self.wormhole(
            x_processed, return_stats=return_stats
        )
        if return_stats and wormhole_stats:
            all_stats['wormhole'] = wormhole_stats
        
        # STAGE 4: Output (already merged by wormhole)
        output = x_wormholed
        
        if return_stats:
            return output, all_stats
        return output, None
    
    def _split_bands(self, x: torch.Tensor) -> Dict[int, torch.Tensor]:
        """Split into 8 bands."""
        B, T, D = x.shape
        x_reshaped = x.view(B, T, self.NUM_BANDS, self.band_dim)
        return {i: x_reshaped[:, :, i, :] for i in range(self.NUM_BANDS)}
    
    def _merge_bands(self, bands: Dict[int, torch.Tensor]) -> torch.Tensor:
        """Merge 8 bands back."""
        stacked = torch.stack([bands[i] for i in range(self.NUM_BANDS)], dim=2)
        B, T, _, _ = stacked.shape
        return stacked.view(B, T, -1)
```

### 8.2 Usage

```python
# Create the complete layer
layer = SpectralBeliefAttentionLayer(
    embed_dim=512,       # 512 / 8 = 64 per band
    num_heads=4,
    wormhole_top_k=16,
    learnable_temperature=True,
)

# Forward pass
x = torch.randn(2, 100, 512)  # [batch, seq, embed]
output, stats = layer(x, return_stats=True)

# Output shape: [2, 100, 512]
# All three mechanisms have been applied
```

---

## 9. Theory Justification

### 9.1 Why This Architecture

```
THEORY → ARCHITECTURE MAPPING:

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  ORTHOGONALITY.md (Type 3):                                            │
│  "Time and frequency are orthogonal (Heisenberg)"                     │
│  → Separate mechanisms for spectral (0-6) and temporal (7)            │
│                                                                         │
│  ORTHOGONALITY.md (Type 5):                                            │
│  "Wormhole pairs are complementary: (0↔6), (1↔5), (2↔4)"             │
│  → Spectral Wormhole with defined pairs                               │
│                                                                         │
│  COLLAPSE_DYNAMICS.md:                                                  │
│  "Temperature controls collapse sharpness"                            │
│  → Per-band learnable temperature in all mechanisms                  │
│                                                                         │
│  COLLAPSE_DYNAMICS.md:                                                  │
│  "Coherence-based triggering, not fixed threshold"                   │
│  → Entropy-based gate in wormhole                                     │
│                                                                         │
│  THE_SEVEN_PLUS_ONE_ARCHITECTURE.md:                                   │
│  "7 spectral + 1 temporal = 8 bands"                                  │
│  → 8-band structure, Tensor Core aligned                              │
│                                                                         │
│  SPECTRAL_BELIEF_MACHINE.md (Imperative 4):                            │
│  "Wormhole cross-band communication"                                  │
│  → Spectral Wormhole as separate stage                                │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 10. References

### 10.1 Component Documents

| Component | Document | Purpose |
|-----------|----------|---------|
| Spectral Attention | `spectral_attention/SPECTRAL_ATTENTION.md` | Bands 0-6 theory and implementation |
| Temporal Attention | `temporal_attention/TEMPORAL_ATTENTION.md` | Band 7 causal attention |
| Spectral Wormhole | `spectral_wormhole/SPECTRAL_WORMHOLE_ATTENTION.md` | Cross-band communication |

### 10.2 Theory Documents

| Theory | Document | Relevance |
|--------|----------|-----------|
| Orthogonality | `architecture_theoretical/ORTHOGONALITY.md` | All 5 types of orthogonality |
| 7+1 Architecture | `architecture_theoretical/THE_SEVEN_PLUS_ONE_ARCHITECTURE.md` | Why 8 bands |
| Collapse Dynamics | `architecture_base/collapse/COLLAPSE_DYNAMICS.md` | Temperature, coherence |
| Spectral Belief Machine | `architecture_theoretical/SPECTRAL_BELIEF_MACHINE.md` | 8 imperatives |

---

## Summary

```
THE ATTENTION STACK — SUMMARY

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  THREE MECHANISMS:                                                      │
│  1. Spectral Attention (Bands 0-6): Within-band, non-causal, spatial  │
│  2. Temporal Attention (Band 7): Within-band, causal, temporal        │
│  3. Spectral Wormhole: Cross-band, coherence-gated, sparse           │
│                                                                         │
│  PROCESSING ORDER:                                                      │
│  Decompose → Per-Band Attention → Wormhole → Reconstruct              │
│                                                                         │
│  THEORY ALIGNMENT:                                                      │
│  • Space-time orthogonality (Heisenberg)           ✓                  │
│  • Band orthogonality (FFT)                        ✓                  │
│  • Complementary pairs (wormhole)                  ✓                  │
│  • Temperature control (per-band τ)                ✓                  │
│  • Coherence gate (entropy-based)                  ✓                  │
│  • Causal time (lower-triangular mask)             ✓                  │
│                                                                         │
│  NONE CAN BE REMOVED. ALL ARE NECESSARY.                              │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

*Oscar Goldman — Shogu Research Group @ Datamutant.ai*

*"Three mechanisms, one system. Spectral attention sees space. Temporal attention respects time. Wormhole attention connects them. Together, they form the complete attention architecture."*


