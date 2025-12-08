# Temporal Attention: Band 7 Causal Self-Attention

## The Time Dimension in the Spectral Belief Machine

**Oscar Goldman — Shogu Research Group @ Datamutant.ai**

---

## Table of Contents

1. [What Theory Demands](#1-what-theory-demands)
2. [Why Time is Different](#2-why-time-is-different)
3. [The Causal Constraint](#3-the-causal-constraint)
4. [Band 7 Specification](#4-band-7-specification)
5. [Mathematical Formalization](#5-mathematical-formalization)
6. [Implementation](#6-implementation)
7. [Relationship to Spectral Bands](#7-relationship-to-spectral-bands)
8. [Relationship to Wormhole](#8-relationship-to-wormhole)
9. [Experiments](#9-experiments)
10. [References](#10-references)

---

## 1. What Theory Demands

### 1.1 The Theoretical Requirements

```
FROM ORTHOGONALITY.md (Type 3: Space-Time Orthogonality):

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  Time and frequency are ORTHOGONAL.                                    │
│                                                                         │
│  HEISENBERG UNCERTAINTY:                                                │
│  Δt × Δf ≥ 1/(4π)                                                      │
│                                                                         │
│  You cannot simultaneously have:                                        │
│  • Precise frequency (Δf small)                                        │
│  • Precise time (Δt small)                                             │
│                                                                         │
│  SPECTRAL BANDS: Precise frequency → imprecise time                   │
│  TEMPORAL BAND:  Precise time → imprecise frequency                   │
│                                                                         │
│  These are COMPLEMENTARY views of the same signal.                    │
│  They MUST be processed by DIFFERENT mechanisms.                      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1.2 From THE_SEVEN_PLUS_ONE_ARCHITECTURE.md

```
THE 7+1 PRINCIPLE:

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  7 SPECTRAL BANDS (Bands 0-6):                                         │
│  • Decompose by FREQUENCY                                              │
│  • Each band spans all TIME (within the window)                       │
│  • Captures "what structure exists at this scale"                     │
│  • Precise frequency → imprecise time                                 │
│                                                                         │
│  1 TEMPORAL BAND (Band 7):                                              │
│  • Processes along TIME (sequence)                                     │
│  • Does NOT decompose by frequency                                     │
│  • Captures "how things relate across moments"                        │
│  • Precise time → imprecise frequency                                 │
│                                                                         │
│  ORTHOGONAL BY CONSTRUCTION.                                           │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1.3 What This Rules Out

```
BAND 7 IS NOT:

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  ✗ FFT over time                                                       │
│    → Would give temporal frequencies, not temporal order              │
│    → Requires knowing the future (breaks causality)                   │
│    → Would create 7 × 7 = 49 bands (too many)                        │
│                                                                         │
│  ✗ Just another spatial band                                           │
│    → Time has causality (past → future, not reverse)                 │
│    → Spatial attention can see anywhere                               │
│    → Temporal attention can only see past                            │
│                                                                         │
│  ✗ Same mechanism as Bands 0-6                                         │
│    → Spectral bands are NON-CAUSAL (see all positions)               │
│    → Temporal band MUST be CAUSAL (see only past)                    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Why Time is Different

### 2.1 The Arrow of Time

```
TIME HAS A DIRECTION:

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  SPACE IS SYMMETRIC:                                                    │
│  ───────────────────                                                    │
│  • Left and right are equivalent                                       │
│  • Can look in any direction                                           │
│  • Reversible: go left, then right, you're back                       │
│  • No privileged direction                                             │
│                                                                         │
│  TIME IS ASYMMETRIC:                                                    │
│  ───────────────────                                                    │
│  • Past and future are NOT equivalent                                  │
│  • Can only observe the past (causality)                              │
│  • Irreversible: time moves only forward                              │
│  • Past is known, future is uncertain                                 │
│                                                                         │
│  ═══════════════════════════════════════════════════════════════════  │
│                                                                         │
│  This asymmetry DEMANDS a different attention mechanism.              │
│  Spectral bands can be symmetric (non-causal).                        │
│  Temporal band MUST be asymmetric (causal).                           │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.2 What Temporal Attention Captures

```
SPECTRAL BANDS vs TEMPORAL BAND:

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  SPECTRAL BANDS (0-6) ANSWER:                                          │
│  ─────────────────────────────                                          │
│  • "What patterns exist at this frequency?"                           │
│  • "Where is this pattern in space?"                                  │
│  • "How strong is this frequency component?"                          │
│                                                                         │
│  TEMPORAL BAND (7) ANSWERS:                                             │
│  ───────────────────────────                                            │
│  • "How does this change over time?"                                  │
│  • "What came before this moment?"                                    │
│  • "Is this consistent with the past?"                                │
│  • "What temporal patterns exist?"                                    │
│                                                                         │
│  ═══════════════════════════════════════════════════════════════════  │
│                                                                         │
│  SPECTRAL: Static structure at different scales                       │
│  TEMPORAL: Dynamic relationships across moments                       │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. The Causal Constraint

### 3.1 Causal Masking

```
THE CAUSAL MASK:

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  Position i can only attend to positions j ≤ i.                       │
│                                                                         │
│  ATTENTION MASK (for sequence length 5):                               │
│                                                                         │
│             Keys: t₀  t₁  t₂  t₃  t₄                                  │
│                   ─── ─── ─── ─── ───                                  │
│  Queries:                                                               │
│    t₀           │ ✓   ✗   ✗   ✗   ✗                                  │
│    t₁           │ ✓   ✓   ✗   ✗   ✗                                  │
│    t₂           │ ✓   ✓   ✓   ✗   ✗                                  │
│    t₃           │ ✓   ✓   ✓   ✓   ✗                                  │
│    t₄           │ ✓   ✓   ✓   ✓   ✓                                  │
│                                                                         │
│  ✓ = can attend (past or present)                                     │
│  ✗ = cannot attend (future, masked to -∞)                             │
│                                                                         │
│  This is MANDATORY for temporal attention.                            │
│  Without it, we would "know the future."                              │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Why Spectral Bands Don't Need Causal Masking

```
SPECTRAL BANDS ARE NON-CAUSAL:

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  Within a single frame:                                                 │
│  • All spatial positions exist simultaneously                          │
│  • No "before" or "after" in space                                    │
│  • Position (x₁, y₁) doesn't cause position (x₂, y₂)                 │
│  • Full attention is appropriate                                       │
│                                                                         │
│  Within a spectral band:                                                │
│  • All frequency components exist simultaneously                       │
│  • No causality between frequencies                                    │
│  • Full self-attention is appropriate                                  │
│                                                                         │
│  ═══════════════════════════════════════════════════════════════════  │
│                                                                         │
│  SPECTRAL ATTENTION (Bands 0-6): Full attention matrix                │
│  TEMPORAL ATTENTION (Band 7):    Lower-triangular (causal) matrix    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Band 7 Specification

### 4.1 Input/Output

```
BAND 7 SPECIFICATION:

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  INPUT:                                                                 │
│  • Temporal features: T[t] ∈ ℝ^(d/8) for t = 0, 1, ..., T-1          │
│  • These are the "time slices" of the input                           │
│  • NOT spectrally decomposed (that's Bands 0-6)                       │
│                                                                         │
│  PROCESSING:                                                            │
│  • Causal self-attention across time                                  │
│  • Position t can only see positions 0, 1, ..., t                    │
│  • Temperature-controlled softmax (like spectral bands)               │
│                                                                         │
│  OUTPUT:                                                                │
│  • Enhanced temporal features: T'[t] ∈ ℝ^(d/8)                       │
│  • Each position now "knows" about its past                          │
│                                                                         │
│  ═══════════════════════════════════════════════════════════════════  │
│                                                                         │
│  DIMENSION: d/8 (same as each spectral band)                          │
│  Total: 7 spectral × d/8 + 1 temporal × d/8 = d                      │
│  Perfect Tensor Core alignment with d divisible by 8                 │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.2 What Band 7 Contains

```
TEMPORAL FEATURE CONTENT:

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  Option 1: POOLED SPATIAL INFORMATION                                  │
│  ──────────────────────────────────────                                  │
│  T[t] = pool(frame_t)     # Global average or learned pooling        │
│  Captures: "What happened at time t" (summarized)                     │
│                                                                         │
│  Option 2: POSITION-SPECIFIC TEMPORAL SLICE                           │
│  ────────────────────────────────────────────                           │
│  T[t] = frame_t[position]  # Specific spatial location               │
│  Captures: "What happened at position p over time"                    │
│                                                                         │
│  Option 3: LEARNED TEMPORAL EMBEDDING                                  │
│  ────────────────────────────────────                                    │
│  T[t] = TemporalEncoder(frame_t)                                       │
│  Captures: Learned representation of temporal dynamics                │
│                                                                         │
│  RECOMMENDATION: Option 1 (pooled) for first implementation          │
│  Can be refined based on experiments.                                 │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.3 Temperature for Band 7

```
TEMPERATURE SETTING FOR TEMPORAL BAND:

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  FROM SPECTRAL_WORMHOLE_ATTENTION.md:                                  │
│                                                                         │
│  Band 7 (Time): τ = 0.8 (suggested initial value)                     │
│                                                                         │
│  RATIONALE:                                                             │
│  ─────────                                                              │
│  • Not as decisive as low-freq bands (τ = 0.5-0.6)                   │
│  • Not as exploratory as high-freq bands (τ = 0.9-1.0)               │
│  • Balanced: commit to clear temporal patterns, explore ambiguous    │
│                                                                         │
│  Can be learned alongside spectral band temperatures.                 │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Mathematical Formalization

### 5.1 Causal Self-Attention

```
TEMPORAL SELF-ATTENTION (CAUSAL):

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  INPUT: T ∈ ℝ^(B × L × D)  where B=batch, L=sequence, D=d/8          │
│                                                                         │
│  STEP 1: Project to Q, K, V                                            │
│  ──────────────────────────                                             │
│  Q = W_Q · T     ∈ ℝ^(B × L × D)                                      │
│  K = W_K · T     ∈ ℝ^(B × L × D)                                      │
│  V = W_V · T     ∈ ℝ^(B × L × D)                                      │
│                                                                         │
│  STEP 2: Compute attention scores                                      │
│  ─────────────────────────────────                                      │
│  S = Q · K^T / √D    ∈ ℝ^(B × L × L)                                  │
│                                                                         │
│  STEP 3: Apply causal mask                                             │
│  ─────────────────────────                                              │
│  M[i,j] = 0 if j ≤ i else -∞                                          │
│  S_masked = S + M                                                       │
│                                                                         │
│  STEP 4: Temperature-controlled softmax                                │
│  ────────────────────────────────────────                               │
│  A = softmax(S_masked / τ, dim=-1)    ∈ ℝ^(B × L × L)                │
│                                                                         │
│  STEP 5: Aggregate values                                              │
│  ────────────────────────                                               │
│  O = A · V    ∈ ℝ^(B × L × D)                                         │
│                                                                         │
│  STEP 6: Output projection + residual                                  │
│  ──────────────────────────────────                                     │
│  T' = T + W_O · O                                                       │
│                                                                         │
│  OUTPUT: T' ∈ ℝ^(B × L × D)                                           │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 5.2 Multi-Head Version

```
MULTI-HEAD CAUSAL ATTENTION:

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  For H heads with head dimension D_h = D/H:                           │
│                                                                         │
│  For each head h:                                                       │
│    Q_h = W_Q^h · T    ∈ ℝ^(B × L × D_h)                               │
│    K_h = W_K^h · T    ∈ ℝ^(B × L × D_h)                               │
│    V_h = W_V^h · T    ∈ ℝ^(B × L × D_h)                               │
│                                                                         │
│    S_h = Q_h · K_h^T / √D_h                                            │
│    S_h_masked = S_h + CausalMask                                       │
│    A_h = softmax(S_h_masked / τ, dim=-1)                              │
│    O_h = A_h · V_h                                                      │
│                                                                         │
│  Concatenate: O = [O_1; O_2; ...; O_H]    ∈ ℝ^(B × L × D)            │
│                                                                         │
│  Output: T' = T + W_O · O                                               │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Implementation

### 6.1 Core Module

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict


class TemporalAttention(nn.Module):
    """
    Theory-aligned temporal attention for Band 7.
    
    Implements CAUSAL self-attention across time, allowing each
    position to attend only to past and present positions.
    
    This is fundamentally different from spectral attention (Bands 0-6)
    which is non-causal and operates on spatial structure.
    
    References:
    - ORTHOGONALITY.md (Type 3: Space-Time Orthogonality)
    - THE_SEVEN_PLUS_ONE_ARCHITECTURE.md
    - PHASE_AND_TIME.md
    """
    
    def __init__(
        self,
        embed_dim: int,          # Should be d/8 for Band 7
        num_heads: int = 4,
        dropout: float = 0.0,
        temperature: float = 0.8,
        learnable_temperature: bool = True,
        max_seq_len: int = 1024,
    ):
        """
        Initialize Temporal Attention.
        
        Args:
            embed_dim: Dimension of Band 7 features (typically d/8)
            num_heads: Number of attention heads
            dropout: Dropout probability
            temperature: Initial softmax temperature (τ)
            learnable_temperature: If True, τ is a learned parameter
            max_seq_len: Maximum sequence length (for causal mask caching)
        """
        super().__init__()
        
        assert embed_dim % num_heads == 0, \
            f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = math.sqrt(self.head_dim)
        self.max_seq_len = max_seq_len
        
        # Projections
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # Temperature
        if learnable_temperature:
            self.temperature = nn.Parameter(torch.tensor(temperature))
        else:
            self.register_buffer('temperature', torch.tensor(temperature))
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Pre-compute and cache causal mask
        self.register_buffer(
            'causal_mask',
            self._create_causal_mask(max_seq_len),
            persistent=False
        )
    
    def _create_causal_mask(self, seq_len: int) -> torch.Tensor:
        """
        Create causal (lower-triangular) attention mask.
        
        Position i can only attend to positions j where j <= i.
        Future positions are masked with -inf.
        
        Returns:
            mask: [seq_len, seq_len] tensor with 0 for allowed, -inf for masked
        """
        # Create lower triangular matrix of ones
        mask = torch.triu(
            torch.ones(seq_len, seq_len), 
            diagonal=1
        )
        # Convert to mask: 0 where allowed, -inf where masked
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    
    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        Apply causal temporal self-attention.
        
        Args:
            x: [B, L, D] temporal features (Band 7)
            return_attention: If True, return attention weights for analysis
            
        Returns:
            output: [B, L, D] enhanced temporal features
            stats: Optional dict with attention weights and statistics
        """
        B, L, D = x.shape
        assert D == self.embed_dim
        
        # Project to Q, K, V
        q = self.q_proj(x)  # [B, L, D]
        k = self.k_proj(x)  # [B, L, D]
        v = self.v_proj(x)  # [B, L, D]
        
        # Reshape for multi-head attention
        q = q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        # Shape: [B, H, L, D_h]
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale  # [B, H, L, L]
        
        # Apply causal mask
        causal_mask = self.causal_mask[:L, :L]  # [L, L]
        scores = scores + causal_mask.unsqueeze(0).unsqueeze(0)
        
        # Temperature-controlled softmax
        tau = self.temperature.clamp(min=0.1)
        attention = F.softmax(scores / tau, dim=-1)  # [B, H, L, L]
        
        # Apply dropout
        attention = self.dropout(attention)
        
        # Aggregate values
        out = torch.matmul(attention, v)  # [B, H, L, D_h]
        
        # Reshape back
        out = out.transpose(1, 2).contiguous().view(B, L, D)  # [B, L, D]
        
        # Output projection
        out = self.out_proj(out)
        
        # Residual connection
        output = x + out
        
        # Statistics for observability
        if return_attention:
            # Compute entropy of attention distribution
            attn_clamped = attention.clamp(min=1e-10)
            entropy = -(attn_clamped * attn_clamped.log()).sum(dim=-1)  # [B, H, L]
            
            stats = {
                'attention_weights': attention.detach(),
                'temperature': tau.item(),
                'mean_entropy': entropy.mean().item(),
                'per_position_entropy': entropy.mean(dim=(0, 1)).detach(),  # [L]
            }
            return output, stats
        
        return output, None
    
    def get_attention_pattern(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get the causal attention pattern for visualization.
        
        Returns:
            attention: [B, H, L, L] attention weights (lower triangular)
        """
        _, stats = self.forward(x, return_attention=True)
        return stats['attention_weights']


class TemporalAttentionBlock(nn.Module):
    """
    Complete temporal attention block with LayerNorm and FFN.
    
    This is the full "transformer block" for Band 7.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 4,
        ffn_ratio: float = 4.0,
        dropout: float = 0.0,
        temperature: float = 0.8,
        learnable_temperature: bool = True,
    ):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = TemporalAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            temperature=temperature,
            learnable_temperature=learnable_temperature,
        )
        
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * ffn_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * ffn_ratio), embed_dim),
            nn.Dropout(dropout),
        )
    
    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        Forward pass through temporal block.
        
        Args:
            x: [B, L, D] temporal features
            return_attention: If True, return attention statistics
            
        Returns:
            output: [B, L, D] processed features
            stats: Optional attention statistics
        """
        # Self-attention with pre-norm
        x_norm = self.norm1(x)
        attn_out, stats = self.attn(x_norm, return_attention=return_attention)
        x = x + attn_out
        
        # FFN with pre-norm
        x = x + self.ffn(self.norm2(x))
        
        return x, stats
```

### 6.2 Usage Example

```python
# Create temporal attention for Band 7
# Assuming total embed_dim = 512, Band 7 gets 512/8 = 64

temporal_attn = TemporalAttention(
    embed_dim=64,           # d/8 for Band 7
    num_heads=4,            # 4 heads, each with dim 16
    temperature=0.8,        # Balanced temperature
    learnable_temperature=True,
)

# Input: temporal features [batch, sequence, d/8]
temporal_features = torch.randn(2, 100, 64)  # 100 time steps

# Forward pass
output, stats = temporal_attn(temporal_features, return_attention=True)

# Output shape: [2, 100, 64] — same as input
# Attention is CAUSAL: position t only sees positions 0..t

print(f"Temperature: {stats['temperature']}")
print(f"Mean entropy: {stats['mean_entropy']}")
```

---

## 7. Relationship to Spectral Bands

### 7.1 The Parallel Structure

```
SPECTRAL vs TEMPORAL ATTENTION:

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  SPECTRAL ATTENTION (Bands 0-6):                                       │
│  ─────────────────────────────────                                      │
│  • Within-band self-attention                                          │
│  • NON-CAUSAL (full attention matrix)                                 │
│  • Operates on spatial positions                                       │
│  • 7 separate attention operations (one per band)                     │
│                                                                         │
│  TEMPORAL ATTENTION (Band 7):                                           │
│  ──────────────────────────────                                         │
│  • Within-band self-attention                                          │
│  • CAUSAL (lower-triangular attention matrix)                         │
│  • Operates on time positions                                          │
│  • 1 attention operation                                               │
│                                                                         │
│  ═══════════════════════════════════════════════════════════════════  │
│                                                                         │
│  THEY ARE PARALLEL:                                                     │
│  • All 8 bands do self-attention simultaneously                       │
│  • No cross-band communication at this stage                          │
│  • Orthogonality fully preserved                                       │
│                                                                         │
│  THEY ARE DIFFERENT:                                                    │
│  • Bands 0-6: Can see all spatial positions                          │
│  • Band 7: Can only see past temporal positions                       │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 7.2 Information Flow

```
INFORMATION FLOW:

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  STAGE 1: SPECTRAL DECOMPOSITION                                       │
│  ─────────────────────────────────                                      │
│                                                                         │
│  Input [B, T, D]                                                        │
│       ↓                                                                 │
│  FFT + Band splitting                                                   │
│       ↓                                                                 │
│  8 bands: B0, B1, B2, B3, B4, B5, B6, B7                              │
│  Each: [B, T, D/8]                                                      │
│                                                                         │
│  ─────────────────────────────────────────────────────────────────────  │
│                                                                         │
│  STAGE 2: PER-BAND SELF-ATTENTION (PARALLEL)                           │
│  ─────────────────────────────────────────────                          │
│                                                                         │
│  Bands 0-6: Spectral Self-Attention (non-causal)                      │
│  Band 7:    Temporal Self-Attention (causal) ← THIS FILE              │
│                                                                         │
│  Each band processes itself independently.                             │
│  No cross-talk. Orthogonality preserved.                              │
│                                                                         │
│  ─────────────────────────────────────────────────────────────────────  │
│                                                                         │
│  STAGE 3: SPECTRAL WORMHOLE (CROSS-BAND)                               │
│  ─────────────────────────────────────────                              │
│                                                                         │
│  Band pairs communicate: (0↔6), (1↔5), (2↔4)                         │
│  Band 3 (bridge) → all bands                                           │
│  Band 7 (temporal) → all bands                                         │
│                                                                         │
│  This is where time talks to frequency!                               │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 8. Relationship to Wormhole

### 8.1 Band 7 in Wormhole Communication

```
BAND 7 IN SPECTRAL WORMHOLE:

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  From SPECTRAL_WORMHOLE_ATTENTION.md:                                  │
│                                                                         │
│  Band 7 (Temporal) queries ALL other bands:                           │
│  • Band 7 → Band 0: "How does identity evolve?"                       │
│  • Band 7 → Band 1: "How does shape change?"                          │
│  • Band 7 → Band 2-6: "How do these features evolve?"                │
│                                                                         │
│  All bands CAN query Band 7:                                            │
│  • Band 0 → Band 7: "What's the temporal context for this identity?" │
│  • Band 6 → Band 7: "What's the motion of this edge?"                │
│                                                                         │
│  ═══════════════════════════════════════════════════════════════════  │
│                                                                         │
│  THE KEY INSIGHT:                                                       │
│                                                                         │
│  Within-band: Temporal attention (causal, past-only)                  │
│  Cross-band:  Wormhole attention (sparse, temperature-controlled)    │
│                                                                         │
│  These are SEPARATE mechanisms with DIFFERENT purposes.               │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 8.2 What Each Mechanism Does

```
TWO MECHANISMS, TWO PURPOSES:

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  TEMPORAL ATTENTION (this file):                                       │
│  ─────────────────────────────────                                      │
│  Purpose:  Relate time steps WITHIN Band 7                            │
│  Question: "How does the temporal representation evolve?"             │
│  Mask:     Causal (can only see past)                                 │
│  Scope:    Band 7 only                                                 │
│                                                                         │
│  WORMHOLE ATTENTION (Band 7 component):                                │
│  ─────────────────────────────────────                                  │
│  Purpose:  Connect Band 7 TO other bands                              │
│  Question: "What do spectral bands tell me about time?"              │
│  Mask:     Coherence-gated (entropy-based)                            │
│  Scope:    Cross-band                                                  │
│                                                                         │
│  ═══════════════════════════════════════════════════════════════════  │
│                                                                         │
│  Temporal Attention:  Band 7 talks to ITSELF (across time)           │
│  Wormhole Attention:  Band 7 talks to OTHERS (across frequency)      │
│                                                                         │
│  Both are needed. Neither replaces the other.                         │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 9. Experiments

### 9.1 Validation Experiments

| Experiment | Tests | File |
|------------|-------|------|
| 001_EXP_ENTROPY_OBSERVATION | Temporal attention entropy | `experiments/001_*` |
| 003_EXP_SPECTRAL_BAND_DYNAMICS | Band 7 vs Bands 0-6 dynamics | `experiments/003_*` |
| 018_EXP_PUMP_CYCLE_DYNAMICS | Temporal patterns in belief | `experiments/018_*` |
| 023_EXP_TIMELINE_COHERENCE | Past representations shift | `experiments/023_*` |

### 9.2 Key Predictions

```
TESTABLE PREDICTIONS FOR TEMPORAL ATTENTION:

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  1. CAUSALITY IS RESPECTED                                              │
│     Position t should NOT be influenced by positions t+1, t+2, ...    │
│     Test: Perturb future, measure effect on past predictions.        │
│                                                                         │
│  2. RECENCY BIAS                                                        │
│     Recent positions should receive more attention (on average).      │
│     Test: Measure attention weight distribution over positions.       │
│                                                                         │
│  3. TEMPORAL PATTERNS CAPTURED                                          │
│     Repeating patterns should be recognized.                          │
│     Test: Present periodic input, measure attention to past cycles.  │
│                                                                         │
│  4. DIFFERENT FROM SPECTRAL                                             │
│     Attention patterns should differ from Bands 0-6.                  │
│     Test: Compare attention matrices across bands.                    │
│                                                                         │
│  5. ENTROPY DROPS BEFORE PREDICTION                                     │
│     Temporal attention should concentrate before decision.            │
│     Test: Measure entropy trajectory during inference.                │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 10. References

### 10.1 Internal Documents

| Document | Location | Relevance |
|----------|----------|-----------|
| ORTHOGONALITY | `architecture_theoretical/` | Type 3: Space-Time |
| THE_SEVEN_PLUS_ONE_ARCHITECTURE | `architecture_theoretical/` | 7+1 structure |
| PHASE_AND_TIME | `architecture_theoretical/` | Phase vs temporal dynamics |
| SPECTRAL_WORMHOLE_ATTENTION | `architecture_base/wormhole/` | Cross-band communication |
| SPECTRAL_ATTENTION | `architecture_base/attention/` | Bands 0-6 attention |

### 10.2 Key Distinctions

```
SUMMARY: WHAT EACH FILE COVERS

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  SPECTRAL_ATTENTION.md:                                                 │
│  • Theory of 7 bands                                                   │
│  • Per-band self-attention for Bands 0-6                              │
│  • NON-CAUSAL (spatial)                                                │
│                                                                         │
│  TEMPORAL_ATTENTION.md (this file):                                     │
│  • Band 7 causal self-attention                                       │
│  • CAUSAL (temporal)                                                   │
│  • Why time is different from frequency                               │
│                                                                         │
│  SPECTRAL_WORMHOLE_ATTENTION.md:                                        │
│  • Cross-band communication                                            │
│  • How bands talk to each other                                       │
│  • Including Band 7 ↔ Bands 0-6                                       │
│                                                                         │
│  Together: Complete attention architecture                             │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Summary

```
TEMPORAL ATTENTION (BAND 7) — SUMMARY

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  WHAT IT IS:                                                            │
│  • Causal self-attention for the temporal band                        │
│  • Position t can only attend to positions 0, 1, ..., t              │
│  • Temperature-controlled softmax                                      │
│                                                                         │
│  WHAT IT CAPTURES:                                                      │
│  • Temporal relationships                                              │
│  • How things change over time                                        │
│  • Patterns in sequence                                                │
│                                                                         │
│  WHY IT'S DIFFERENT:                                                    │
│  • Time has a direction (causality)                                   │
│  • Cannot see the future                                               │
│  • Orthogonal to frequency (Heisenberg)                               │
│                                                                         │
│  HOW IT CONNECTS:                                                       │
│  • Runs parallel to spectral attention (Bands 0-6)                   │
│  • Communicates via Spectral Wormhole                                 │
│  • Part of the 8-band architecture                                    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

*Oscar Goldman — Shogu Research Group @ Datamutant.ai*

*"Time is not another frequency. It is the dimension in which frequencies unfold. The temporal band respects this truth: it sees the past, not the future, because causality is not optional."*


