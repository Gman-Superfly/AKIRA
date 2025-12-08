"""
EXPERIMENT 030: Full AKIRA 7+1 Architecture (As Documented)
============================================================

This implements the ACTUAL AKIRA architecture from the theoretical docs:
- 7 spectral bands + 1 temporal band = 8 total
- Spectral Wormhole with COHERENCE/ENTROPY gating (not differential windows)
- NO cross-batch history (that was Exp 028-029)
- Complementary pairs: (0-6), (1-5), (2-4), Bridge(3)

This is a clean test of whether the base AKIRA architecture works.

Run on Google Colab with GPU.

AKIRA Project - Experiment 030
Oscar Goldman - Shogu Research Group @ Datamutant.ai
"""

# ==============================================================================
# CELL 1: INSTALL DEPENDENCIES
# ==============================================================================

# !pip install transformers datasets tqdm

# ==============================================================================
# CELL 2: IMPORTS
# ==============================================================================

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import time

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ==============================================================================
# CELL 3: CONFIGURATION
# ==============================================================================

@dataclass
class AKIRAConfig:
    """Configuration for AKIRA 7+1 architecture (as documented)."""
    vocab_size: int = 50257          # GPT-2 vocabulary
    embed_dim: int = 512             # Must be divisible by 8 (8 bands x 64 dim)
    num_layers: int = 6
    num_heads: int = 8
    max_seq_length: int = 256
    dropout: float = 0.1
    
    # 7 spectral + 1 temporal = 8 bands
    num_spectral_bands: int = 7
    num_total_bands: int = 8
    
    # Wormhole settings (from docs: coherence-gated, sparse)
    wormhole_top_k: int = 16
    coherence_threshold: float = 0.5  # Normalized entropy threshold
    
    # Differential learning rates per band (from docs)
    band_learning_rates: Tuple[float, ...] = (
        3e-5,   # Band 0: DC - slowest
        5e-5,   # Band 1
        1e-4,   # Band 2
        3e-4,   # Band 3: Bridge
        5e-4,   # Band 4
        1e-3,   # Band 5
        3e-3,   # Band 6: fastest
        3e-4,   # Band 7: Temporal - medium
    )
    
    # Training
    batch_size: int = 16
    learning_rate: float = 3e-4
    total_steps: int = 5000
    eval_every: int = 500
    warmup_steps: int = 200
    
    @property
    def dim_per_band(self) -> int:
        return self.embed_dim // self.num_total_bands  # 512/8 = 64
    
    def validate(self):
        assert self.embed_dim % self.num_total_bands == 0
        assert len(self.band_learning_rates) == self.num_total_bands
        return True

print("AKIRAConfig defined")

# ==============================================================================
# CELL 4: CAUSAL CONV1D (For Language Modeling)
# ==============================================================================

class CausalConv1d(nn.Module):
    """Causal 1D convolution - only sees past and current, never future."""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = kernel_size - 1
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]
        x_padded = F.pad(x, (self.padding, 0))  # Left pad only
        return self.conv(x_padded)

print("CausalConv1d defined")

# ==============================================================================
# CELL 5: SPECTRAL DECOMPOSER (7 Bands)
# ==============================================================================

class SpectralDecomposer(nn.Module):
    """
    Decomposes input into 7 spectral bands.
    
    Uses learnable projections + causal convolutions with different kernel sizes
    to capture different frequency components.
    
    Band 0: DC (identity) - kernel 1
    Band 1-6: Increasing frequency - decreasing kernels
    """
    
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.dim_per_band = embed_dim // 8  # 64 for 512
        
        # Kernel sizes: larger = lower freq, smaller = higher freq
        # Band 0 (DC) uses kernel 1 (identity-like)
        kernel_sizes = [1, 15, 11, 9, 7, 5, 3]
        
        # Projections: embed_dim -> dim_per_band
        self.projs = nn.ModuleList([
            nn.Linear(embed_dim, self.dim_per_band) for _ in range(7)
        ])
        
        # Causal convolutions per band (operate on dim_per_band)
        self.convs = nn.ModuleList([
            CausalConv1d(self.dim_per_band, self.dim_per_band, k) if k > 1
            else nn.Identity()
            for k in kernel_sizes
        ])
        
        # Layer norms
        self.norms = nn.ModuleList([
            nn.LayerNorm(self.dim_per_band) for _ in range(7)
        ])
    
    def forward(self, x: torch.Tensor) -> Dict[int, torch.Tensor]:
        """
        Args:
            x: [B, T, embed_dim]
        Returns:
            Dict of 7 bands, each [B, T, dim_per_band]
        """
        B, T, D = x.shape
        bands = {}
        
        for i in range(7):
            # Project to band dimension
            h = self.projs[i](x)  # [B, T, dim_per_band]
            
            # Apply causal conv if not identity
            if not isinstance(self.convs[i], nn.Identity):
                h_t = h.transpose(1, 2)  # [B, dim_per_band, T]
                h_t = self.convs[i](h_t)
                h = h_t.transpose(1, 2)  # [B, T, dim_per_band]
            
            bands[i] = self.norms[i](h)
        
        return bands

print("SpectralDecomposer defined")

# ==============================================================================
# CELL 6: SPECTRAL ATTENTION (Per-Band, Causal for LM)
# ==============================================================================

class SpectralBandAttention(nn.Module):
    """
    Attention within a single spectral band.
    CAUSAL for language modeling (can only see past).
    """
    
    def __init__(self, dim: int, num_heads: int = 2, max_seq_len: int = 256, dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        
        # Causal mask
        self.register_buffer("causal_mask",
            torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1).bool())
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(self.causal_mask[:T, :T].unsqueeze(0).unsqueeze(0), -1e9)
        
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.out_proj(out)

print("SpectralBandAttention defined")

# ==============================================================================
# CELL 7: BAND PROCESSORS (Geometric/Hybrid/Reactive)
# ==============================================================================

class GeometricProcessor(nn.Module):
    """Bands 0-2: Slow, structure-preserving. 2x expansion, gated residual."""
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim * 2)
        self.fc2 = nn.Linear(dim * 2, dim)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Linear(dim, dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        h = self.norm(x)
        h = F.gelu(self.fc1(h))
        h = self.dropout(h)
        h = self.fc2(h)
        gate = torch.sigmoid(self.gate(residual))
        return residual + gate * h


class HybridProcessor(nn.Module):
    """Bands 3-4: Balanced. 3x expansion, standard residual."""
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim * 3)
        self.fc2 = nn.Linear(dim * 3, dim)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        h = self.norm(x)
        h = F.gelu(self.fc1(h))
        h = self.dropout(h)
        return residual + self.fc2(h)


class ReactiveProcessor(nn.Module):
    """Bands 5-6: Fast, adaptive. 4x expansion."""
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim * 4)
        self.fc2 = nn.Linear(dim * 4, dim)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        h = self.norm(x)
        h = F.gelu(self.fc1(h))
        h = self.dropout(h)
        return residual + self.fc2(h)

print("Band processors defined (Geometric/Hybrid/Reactive)")

# ==============================================================================
# CELL 8: SPECTRAL WORMHOLE (Coherence-Gated, from Docs)
# ==============================================================================

class SpectralWormhole(nn.Module):
    """
    Cross-band communication with COHERENCE/ENTROPY gating (as per docs).
    
    NOT differential temporal windows - that was an experimental addition.
    
    Mechanism:
    1. Compute attention between complementary pairs
    2. Gate by coherence (low entropy = gate opens)
    3. Sparse top-k selection
    4. CAUSAL MASK to prevent future information leakage (for LM)
    
    Pairs: (0,6), (1,5), (2,4)
    Bridge: Band 3 connects to all
    """
    
    def __init__(self, dim: int, coherence_threshold: float = 0.5, top_k: int = 16, max_seq_len: int = 256):
        super().__init__()
        self.dim = dim
        self.coherence_threshold = coherence_threshold
        self.top_k = top_k
        self.pairs = [(0, 6), (1, 5), (2, 4)]
        
        # CAUSAL MASK - critical for language modeling
        self.register_buffer("causal_mask",
            torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1).bool())
        
        # Cross-band projections for each pair
        self.cross_q = nn.ModuleDict()
        self.cross_k = nn.ModuleDict()
        self.cross_v = nn.ModuleDict()
        self.cross_out = nn.ModuleDict()
        
        for low, high in self.pairs:
            key = f"{low}_{high}"
            self.cross_q[key] = nn.Linear(dim, dim)
            self.cross_k[key] = nn.Linear(dim, dim)
            self.cross_v[key] = nn.Linear(dim, dim)
            self.cross_out[key] = nn.Linear(dim, dim)
        
        # Bridge band (3) projections
        self.bridge_q = nn.Linear(dim, dim)
        self.bridge_k = nn.Linear(dim, dim)
        self.bridge_v = nn.Linear(dim, dim)
        self.bridge_out = nn.Linear(dim, dim)
        
        # Output gates
        self.gates = nn.ModuleList([nn.Linear(dim * 2, dim) for _ in range(7)])
    
    def _coherence_gate(self, attention: torch.Tensor) -> torch.Tensor:
        """
        Compute coherence gate based on attention entropy.
        Low entropy (focused attention) = high coherence = gate opens.
        """
        # attention: [B, T, T] or [B, H, T, T]
        if attention.dim() == 4:
            attention = attention.mean(dim=1)  # Average over heads
        
        # Compute entropy
        entropy = -(attention * torch.log(attention + 1e-9)).sum(dim=-1)  # [B, T]
        max_entropy = math.log(attention.size(-1))
        normalized_entropy = entropy / max_entropy
        
        # Gate: opens for low entropy (high coherence)
        # Sigmoid centered at threshold
        gate = torch.sigmoid((self.coherence_threshold - normalized_entropy) * 10.0)
        return gate.unsqueeze(-1)  # [B, T, 1]
    
    def forward(self, bands: Dict[int, torch.Tensor]) -> Dict[int, torch.Tensor]:
        """
        Args:
            bands: Dict of 7 bands, each [B, T, dim]
        Returns:
            Dict of 7 bands with wormhole updates
        """
        outputs = {k: v.clone() for k, v in bands.items()}
        
        # Process complementary pairs
        for low, high in self.pairs:
            key = f"{low}_{high}"
            band_low = bands[low]
            band_high = bands[high]
            B, T, D = band_low.shape
            
            # Low queries high
            q_l = self.cross_q[key](band_low)
            k_h = self.cross_k[key](band_high)
            v_h = self.cross_v[key](band_high)
            
            # Attention: low -> high (with CAUSAL MASK)
            scores_l2h = torch.matmul(q_l, k_h.transpose(-2, -1)) / math.sqrt(D)
            scores_l2h = scores_l2h.masked_fill(self.causal_mask[:T, :T].unsqueeze(0), -1e9)
            attn_l2h = F.softmax(scores_l2h, dim=-1)
            
            # Coherence gate
            gate_l2h = self._coherence_gate(attn_l2h)
            
            # Sparse top-k
            if T > self.top_k:
                top_k_vals, top_k_idx = attn_l2h.topk(self.top_k, dim=-1)
                sparse_attn = torch.zeros_like(attn_l2h)
                sparse_attn.scatter_(-1, top_k_idx, top_k_vals)
                sparse_attn = sparse_attn / (sparse_attn.sum(dim=-1, keepdim=True) + 1e-9)
                attn_l2h = sparse_attn
            
            # Apply gated wormhole
            wormhole_to_low = torch.matmul(attn_l2h, v_h)
            wormhole_to_low = self.cross_out[key](wormhole_to_low) * gate_l2h
            
            # High queries low (symmetric, with CAUSAL MASK)
            q_h = self.cross_q[key](band_high)
            k_l = self.cross_k[key](band_low)
            v_l = self.cross_v[key](band_low)
            
            scores_h2l = torch.matmul(q_h, k_l.transpose(-2, -1)) / math.sqrt(D)
            scores_h2l = scores_h2l.masked_fill(self.causal_mask[:T, :T].unsqueeze(0), -1e9)
            attn_h2l = F.softmax(scores_h2l, dim=-1)
            gate_h2l = self._coherence_gate(attn_h2l)
            
            if T > self.top_k:
                top_k_vals, top_k_idx = attn_h2l.topk(self.top_k, dim=-1)
                sparse_attn = torch.zeros_like(attn_h2l)
                sparse_attn.scatter_(-1, top_k_idx, top_k_vals)
                sparse_attn = sparse_attn / (sparse_attn.sum(dim=-1, keepdim=True) + 1e-9)
                attn_h2l = sparse_attn
            
            wormhole_to_high = torch.matmul(attn_h2l, v_l)
            wormhole_to_high = self.cross_out[key](wormhole_to_high) * gate_h2l
            
            # Apply with output gates
            gate_low = torch.sigmoid(self.gates[low](
                torch.cat([outputs[low], wormhole_to_low], dim=-1)))
            gate_high = torch.sigmoid(self.gates[high](
                torch.cat([outputs[high], wormhole_to_high], dim=-1)))
            
            outputs[low] = outputs[low] + gate_low * wormhole_to_low
            outputs[high] = outputs[high] + gate_high * wormhole_to_high
        
        # Bridge band (3) aggregates from all
        bridge = bands[3]
        B, T, D = bridge.shape
        
        q_bridge = self.bridge_q(bridge)
        
        for i in [0, 1, 2, 4, 5, 6]:
            k_i = self.bridge_k(bands[i])
            v_i = self.bridge_v(bands[i])
            
            # Bridge attention with CAUSAL MASK
            scores = torch.matmul(q_bridge, k_i.transpose(-2, -1)) / math.sqrt(D)
            scores = scores.masked_fill(self.causal_mask[:T, :T].unsqueeze(0), -1e9)
            attn = F.softmax(scores, dim=-1)
            gate = self._coherence_gate(attn)
            
            bridge_info = torch.matmul(attn, v_i) * gate * (1.0 / 6.0)  # Average contribution
            outputs[3] = outputs[3] + self.bridge_out(bridge_info)
        
        return outputs

print("SpectralWormhole defined (coherence-gated, CAUSAL for LM)")


class PositionWiseWormhole(nn.Module):
    """
    Position-wise only wormhole (NO temporal attention).
    
    Band 0 position t only attends to Band 6 position t.
    No T x T attention matrix - just element-wise cross-band mixing.
    
    This tests: Is the wormhole value from TEMPORAL attention,
    or just from cross-band communication at the same position?
    """
    
    def __init__(self, dim: int, coherence_threshold: float = 0.5):
        super().__init__()
        self.dim = dim
        self.coherence_threshold = coherence_threshold
        self.pairs = [(0, 6), (1, 5), (2, 4)]
        
        # Cross-band projections for each pair (position-wise only)
        self.cross_proj = nn.ModuleDict()
        self.cross_gate = nn.ModuleDict()
        
        for low, high in self.pairs:
            key = f"{low}_{high}"
            # Project complementary band to current band space
            self.cross_proj[key] = nn.Linear(dim, dim)
            # Gate for mixing
            self.cross_gate[key] = nn.Linear(dim * 2, dim)
        
        # Bridge band (3) aggregates position-wise from all
        self.bridge_proj = nn.ModuleList([nn.Linear(dim, dim) for _ in range(6)])
        self.bridge_gate = nn.Linear(dim * 2, dim)
    
    def forward(self, bands: Dict[int, torch.Tensor]) -> Dict[int, torch.Tensor]:
        """
        Args:
            bands: Dict of 7 bands, each [B, T, dim]
        Returns:
            Dict of 7 bands with position-wise wormhole updates
        """
        outputs = {k: v.clone() for k, v in bands.items()}
        
        # Process complementary pairs (position-wise only)
        for low, high in self.pairs:
            key = f"{low}_{high}"
            band_low = bands[low]   # [B, T, D]
            band_high = bands[high]  # [B, T, D]
            
            # Position t in low band gets info from position t in high band (and vice versa)
            # NO temporal attention - just element-wise transform
            
            # Low <- High (position-wise)
            info_to_low = self.cross_proj[key](band_high)  # [B, T, D]
            gate_low = torch.sigmoid(self.cross_gate[key](
                torch.cat([band_low, info_to_low], dim=-1)))  # [B, T, D]
            
            # High <- Low (position-wise)
            info_to_high = self.cross_proj[key](band_low)  # [B, T, D]
            gate_high = torch.sigmoid(self.cross_gate[key](
                torch.cat([band_high, info_to_high], dim=-1)))  # [B, T, D]
            
            outputs[low] = outputs[low] + gate_low * info_to_low
            outputs[high] = outputs[high] + gate_high * info_to_high
        
        # Bridge band (3) aggregates position-wise from all
        bridge = bands[3]
        other_bands = [bands[i] for i in [0, 1, 2, 4, 5, 6]]
        
        bridge_info = sum(proj(b) for proj, b in zip(self.bridge_proj, other_bands)) / 6.0
        gate_bridge = torch.sigmoid(self.bridge_gate(
            torch.cat([bridge, bridge_info], dim=-1)))
        
        outputs[3] = outputs[3] + gate_bridge * bridge_info
        
        return outputs

print("PositionWiseWormhole defined (no temporal attention)")

# ==============================================================================
# CELL 9: TEMPORAL BAND (Band 7 - Causal over Sequence)
# ==============================================================================

class TemporalBand(nn.Module):
    """
    Band 7: Temporal attention over the sequence.
    
    This is DIFFERENT from cross-batch history.
    It processes temporal relationships WITHIN the current sequence.
    
    Input: Aggregated representation from spectral bands
    Output: Temporally-contextualized representation
    """
    
    def __init__(self, dim: int, max_seq_len: int, num_heads: int = 2, dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        
        # Causal mask (must be causal for LM)
        self.register_buffer("causal_mask",
            torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1).bool())
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, dim] - aggregated spectral representation
        Returns:
            [B, T, dim] - temporally contextualized
        """
        B, T, D = x.shape
        residual = x
        x = self.norm(x)
        
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(self.causal_mask[:T, :T].unsqueeze(0).unsqueeze(0), -1e9)
        
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return residual + self.out_proj(out)

print("TemporalBand defined (Band 7)")

# ==============================================================================
# CELL 10: SPECTRAL RECONSTRUCTOR
# ==============================================================================

class SpectralReconstructor(nn.Module):
    """Reconstructs full embedding from 7 spectral bands."""
    
    def __init__(self, embed_dim: int, dim_per_band: int):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Project each band back to full dimension, then mix
        self.band_projs = nn.ModuleList([
            nn.Linear(dim_per_band, embed_dim) for _ in range(7)
        ])
        self.mix = nn.Linear(embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, bands: Dict[int, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            bands: Dict of 7 bands, each [B, T, dim_per_band]
        Returns:
            [B, T, embed_dim]
        """
        # Sum projections (they should combine constructively)
        combined = sum(self.band_projs[i](bands[i]) for i in range(7))
        return self.norm(self.mix(combined))

print("SpectralReconstructor defined")

# ==============================================================================
# CELL 11: FULL AKIRA 7+1 LAYER
# ==============================================================================

class AKIRALayer(nn.Module):
    """
    Full AKIRA 7+1 layer as documented.
    
    Flow:
    1. Spectral decomposition (7 bands)
    2. Per-band attention + processing
    3. Spectral wormhole (coherence-gated cross-band)
    4. Temporal band (causal over sequence)
    5. Reconstruct + combine
    """
    
    def __init__(self, config: AKIRAConfig, wormhole_type: str = "full", use_temporal: bool = True):
        """
        Args:
            config: AKIRAConfig
            wormhole_type: "full" (T x T causal), "position_wise" (no temporal), or "none"
            use_temporal: Whether to use temporal band (band 7)
        """
        super().__init__()
        self.config = config
        self.wormhole_type = wormhole_type
        self.use_temporal = use_temporal
        dim = config.dim_per_band
        
        # Input norm
        self.input_norm = nn.LayerNorm(config.embed_dim)
        
        # Spectral decomposition
        self.decomposer = SpectralDecomposer(config.embed_dim)
        
        # Per-band attention
        self.band_attns = nn.ModuleList([
            SpectralBandAttention(dim, num_heads=2, max_seq_len=config.max_seq_length, dropout=config.dropout)
            for _ in range(7)
        ])
        
        # Per-band processors
        self.band_processors = nn.ModuleList([
            GeometricProcessor(dim, config.dropout),   # Band 0
            GeometricProcessor(dim, config.dropout),   # Band 1
            GeometricProcessor(dim, config.dropout),   # Band 2
            HybridProcessor(dim, config.dropout),      # Band 3 (bridge)
            HybridProcessor(dim, config.dropout),      # Band 4
            ReactiveProcessor(dim, config.dropout),    # Band 5
            ReactiveProcessor(dim, config.dropout),    # Band 6
        ])
        
        # Wormhole (optional for ablation)
        if wormhole_type == "full":
            self.wormhole = SpectralWormhole(dim, config.coherence_threshold, config.wormhole_top_k, config.max_seq_length)
        elif wormhole_type == "position_wise":
            self.wormhole = PositionWiseWormhole(dim, config.coherence_threshold)
        
        # Temporal band (optional for ablation)
        if use_temporal:
            self.temporal_agg = nn.Linear(7 * dim, dim)  # Aggregate spectral -> temporal
            self.temporal_band = TemporalBand(dim, config.max_seq_length, num_heads=2, dropout=config.dropout)
            self.temporal_out = nn.Linear(dim, config.embed_dim)
        
        # Reconstructor
        self.reconstructor = SpectralReconstructor(config.embed_dim, dim)
        
        # Final projection
        self.output_proj = nn.Linear(config.embed_dim, config.embed_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        residual = x
        
        # Pre-norm
        x = self.input_norm(x)
        
        # 1. Spectral decomposition
        bands = self.decomposer(x)
        
        # 2. Per-band attention + processing
        for i in range(7):
            bands[i] = bands[i] + self.band_attns[i](bands[i])
            bands[i] = self.band_processors[i](bands[i])
        
        # 3. Spectral wormhole (if enabled)
        if self.wormhole_type != "none":
            bands = self.wormhole(bands)
        
        # 4. Temporal band (if enabled)
        temporal_out = None
        if self.use_temporal:
            # Aggregate spectral bands for temporal processing
            spectral_concat = torch.cat([bands[i] for i in range(7)], dim=-1)
            temporal_input = self.temporal_agg(spectral_concat)
            temporal_out = self.temporal_band(temporal_input)
        
        # 5. Reconstruct
        spectral_output = self.reconstructor(bands)
        
        # Combine spectral + temporal
        if temporal_out is not None:
            output = spectral_output + self.temporal_out(temporal_out)
        else:
            output = spectral_output
        
        # Residual
        return residual + self.output_proj(output)

print("AKIRALayer defined")

# ==============================================================================
# CELL 12: AKIRA LANGUAGE MODEL
# ==============================================================================

class AKIRALM(nn.Module):
    """Full AKIRA 7+1 Language Model."""
    
    def __init__(self, config: AKIRAConfig, wormhole_type: str = "full", use_temporal: bool = True):
        """
        Args:
            config: AKIRAConfig
            wormhole_type: "full" (T x T causal), "position_wise" (no temporal), or "none"
            use_temporal: Whether to use temporal band (band 7)
        """
        super().__init__()
        config.validate()
        self.config = config
        
        self.token_embed = nn.Embedding(config.vocab_size, config.embed_dim)
        self.pos_embed = nn.Embedding(config.max_seq_length, config.embed_dim)
        self.dropout = nn.Dropout(config.dropout)
        
        self.layers = nn.ModuleList([
            AKIRALayer(config, wormhole_type, use_temporal) for _ in range(config.num_layers)
        ])
        
        self.final_norm = nn.LayerNorm(config.embed_dim)
        self.lm_head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)
        
        self.apply(self._init_weights)
        self.num_params = sum(p.numel() for p in self.parameters())
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        B, T = input_ids.shape
        
        positions = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, -1)
        x = self.token_embed(input_ids) + self.pos_embed(positions)
        x = self.dropout(x)
        
        for layer in self.layers:
            x = layer(x)
        
        x = self.final_norm(x)
        logits = self.lm_head(x)
        
        output = {'logits': logits}
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100
            )
            output['loss'] = loss
        return output

print("AKIRALM defined")

# ==============================================================================
# CELL 13: BASELINE GPT-2
# ==============================================================================

class BaselineGPT2(nn.Module):
    """Standard GPT-2 for baseline comparison."""
    
    def __init__(self, config: AKIRAConfig):
        super().__init__()
        self.config = config
        
        self.token_embed = nn.Embedding(config.vocab_size, config.embed_dim)
        self.pos_embed = nn.Embedding(config.max_seq_length, config.embed_dim)
        self.dropout = nn.Dropout(config.dropout)
        
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=config.embed_dim,
                nhead=config.num_heads,
                dim_feedforward=config.embed_dim * 4,
                dropout=config.dropout,
                batch_first=True
            )
            for _ in range(config.num_layers)
        ])
        
        self.final_norm = nn.LayerNorm(config.embed_dim)
        self.lm_head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)
        
        self.register_buffer("causal_mask", None)
        self.apply(self._init_weights)
        self.num_params = sum(p.numel() for p in self.parameters())
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def _get_causal_mask(self, T: int, device: torch.device) -> torch.Tensor:
        if self.causal_mask is None or self.causal_mask.shape[0] < T:
            self.causal_mask = torch.triu(torch.ones(T, T, device=device), diagonal=1).bool()
        return self.causal_mask[:T, :T]
    
    def forward(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        B, T = input_ids.shape
        
        positions = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, -1)
        x = self.token_embed(input_ids) + self.pos_embed(positions)
        x = self.dropout(x)
        
        mask = self._get_causal_mask(T, x.device)
        for layer in self.layers:
            x = layer(x, src_mask=mask)
        
        x = self.final_norm(x)
        logits = self.lm_head(x)
        
        output = {'logits': logits}
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100
            )
            output['loss'] = loss
        return output

print("BaselineGPT2 defined")

# ==============================================================================
# CELL 14: DATA LOADING
# ==============================================================================

def load_wikitext2(config: AKIRAConfig):
    """Load WikiText-2 dataset."""
    from datasets import load_dataset
    from transformers import GPT2Tokenizer
    
    print("Loading WikiText-2...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    def tokenize_and_chunk(examples):
        tokens = tokenizer(
            examples["text"],
            truncation=True,
            max_length=config.max_seq_length,
            padding="max_length",
            return_tensors="pt"
        )
        return {"input_ids": tokens["input_ids"]}
    
    train_data = dataset["train"].filter(lambda x: len(x["text"]) > 50)
    val_data = dataset["validation"].filter(lambda x: len(x["text"]) > 50)
    
    train_data = train_data.map(tokenize_and_chunk, batched=True, remove_columns=["text"])
    val_data = val_data.map(tokenize_and_chunk, batched=True, remove_columns=["text"])
    
    train_data.set_format(type="torch", columns=["input_ids"])
    val_data.set_format(type="torch", columns=["input_ids"])
    
    train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=config.batch_size)
    
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    return train_loader, val_loader

print("Data loading defined")

# ==============================================================================
# CELL 15: TRAINING
# ==============================================================================

def train_model(model, train_loader, val_loader, config: AKIRAConfig, name: str):
    """Train a model and return results."""
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.total_steps)
    
    print(f"\nTraining {name}...")
    print(f"Parameters: {model.num_params:,}")
    
    model.train()
    train_iter = iter(train_loader)
    start_time = time.time()
    
    results = {"steps": [], "train_loss": [], "val_loss": [], "val_ppl": []}
    
    pbar = tqdm(range(1, config.total_steps + 1), desc=name)
    for step in pbar:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
        
        input_ids = batch["input_ids"].to(device)
        labels = input_ids.clone()
        
        optimizer.zero_grad()
        output = model(input_ids, labels=labels)
        loss = output["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        if step % config.eval_every == 0:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for i, batch in enumerate(val_loader):
                    if i >= 50:  # Limit eval batches
                        break
                    input_ids = batch["input_ids"].to(device)
                    labels = input_ids.clone()
                    output = model(input_ids, labels=labels)
                    val_losses.append(output["loss"].item())
            
            val_loss = sum(val_losses) / len(val_losses)
            val_ppl = math.exp(val_loss)
            
            results["steps"].append(step)
            results["train_loss"].append(loss.item())
            results["val_loss"].append(val_loss)
            results["val_ppl"].append(val_ppl)
            
            elapsed = time.time() - start_time
            print(f"\n  Step {step}: val_loss={val_loss:.4f}, ppl={val_ppl:.2f}, time={elapsed:.1f}s")
            
            model.train()
    
    elapsed = time.time() - start_time
    print(f"  Training complete in {elapsed:.1f}s")
    
    results["final_loss"] = results["val_loss"][-1]
    results["final_ppl"] = results["val_ppl"][-1]
    results["time"] = elapsed
    
    return results

print("Training function defined")

# ==============================================================================
# CELL 16: RUN ABLATIONS
# ==============================================================================

def run_ablations():
    """Run all ablations."""
    config = AKIRAConfig()
    
    train_loader, val_loader = load_wikitext2(config)
    
    all_results = {}
    
    # 1. Baseline
    print("\n" + "="*60)
    print("1. Baseline (Standard GPT-2)")
    print("="*60)
    model = BaselineGPT2(config)
    all_results["baseline"] = train_model(model, train_loader, val_loader, config, "1. Baseline")
    del model
    torch.cuda.empty_cache() if device == "cuda" else None
    
    # 2. AKIRA 7+1 Full (with causal temporal wormhole + temporal band)
    print("\n" + "="*60)
    print("2. AKIRA 7+1 (Full - causal wormhole + temporal band)")
    print("="*60)
    model = AKIRALM(config, wormhole_type="full", use_temporal=True)
    all_results["akira_full"] = train_model(model, train_loader, val_loader, config, "2. AKIRA 7+1 Full")
    del model
    torch.cuda.empty_cache() if device == "cuda" else None
    
    # 3. AKIRA No Wormhole
    print("\n" + "="*60)
    print("3. AKIRA 7+1 (No Wormhole)")
    print("="*60)
    model = AKIRALM(config, wormhole_type="none", use_temporal=True)
    all_results["akira_no_wormhole"] = train_model(model, train_loader, val_loader, config, "3. AKIRA No Wormhole")
    del model
    torch.cuda.empty_cache() if device == "cuda" else None
    
    # 4. AKIRA No Temporal Band
    print("\n" + "="*60)
    print("4. AKIRA 7 (No Temporal Band)")
    print("="*60)
    model = AKIRALM(config, wormhole_type="full", use_temporal=False)
    all_results["akira_no_temporal"] = train_model(model, train_loader, val_loader, config, "4. AKIRA No Temporal")
    del model
    torch.cuda.empty_cache() if device == "cuda" else None
    
    # 5. AKIRA Position-wise Wormhole (no temporal attention in wormhole)
    print("\n" + "="*60)
    print("5. AKIRA 7+1 (Position-wise Wormhole - no temporal in wormhole)")
    print("="*60)
    model = AKIRALM(config, wormhole_type="position_wise", use_temporal=True)
    all_results["akira_position_wise"] = train_model(model, train_loader, val_loader, config, "5. AKIRA Position-wise")
    del model
    torch.cuda.empty_cache() if device == "cuda" else None
    
    return all_results

print("Ablation runner defined")

# ==============================================================================
# CELL 17: MAIN
# ==============================================================================

if __name__ == "__main__":
    results = run_ablations()
    
    # Print summary
    print("\n" + "="*70)
    print("EXPERIMENT 030 RESULTS: Full AKIRA 7+1 Architecture")
    print("="*70)
    
    baseline_ppl = results["baseline"]["final_ppl"]
    
    print(f"\n{'Model':<40} {'Loss':>10} {'PPL':>10} {'vs Base':>12}")
    print("-"*70)
    
    for name, res in results.items():
        ppl = res["final_ppl"]
        loss = res["final_loss"]
        if name == "baseline":
            delta = "+0.00%"
        else:
            improvement = (baseline_ppl - ppl) / baseline_ppl * 100
            delta = f"+{improvement:.2f}%" if improvement > 0 else f"{improvement:.2f}%"
        
        display_name = {
            "baseline": "1. Baseline (Standard GPT-2)",
            "akira_full": "2. AKIRA 7+1 Full (causal wormhole)",
            "akira_no_wormhole": "3. AKIRA No Wormhole",
            "akira_no_temporal": "4. AKIRA No Temporal",
            "akira_position_wise": "5. AKIRA Position-wise Wormhole"
        }.get(name, name)
        
        print(f"{display_name:<40} {loss:>10.4f} {ppl:>10.2f} {delta:>12}")
    
    print("-"*70)
    
    # Find winner
    best_name = min(results.keys(), key=lambda k: results[k]["final_ppl"])
    best_ppl = results[best_name]["final_ppl"]
    
    print(f"\n>>> WINNER: {best_name}")
    print(f"    Perplexity: {best_ppl:.2f}")
    
    if best_name != "baseline":
        improvement = (baseline_ppl - best_ppl) / baseline_ppl * 100
        print(f"    Improvement over baseline: +{improvement:.2f}%")
    
    print("\n" + "="*70)
    print("EXPERIMENT 030 COMPLETE")
    print("="*70)
