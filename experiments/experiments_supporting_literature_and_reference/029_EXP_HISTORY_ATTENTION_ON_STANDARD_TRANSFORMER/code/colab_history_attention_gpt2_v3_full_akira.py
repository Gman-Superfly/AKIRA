"""
EXPERIMENT 029 v3: Full AKIRA 7+1 Architecture Ablation
========================================================

Complete ablation study with the FULL AKIRA architecture:
- 7 spectral bands + 1 temporal band = 8 total
- Temporal Wormhole with differential windows (Heisenberg)
- Complementary pairs: (0-6), (1-5), (2-4), Bridge(3)
- Processing modes: Geometric/Hybrid/Reactive per band
- Differential learning rates per band
- CAUSAL convolutions (fixed from v2)

Ablations:
1. BASELINE: Standard GPT-2
2. V2_WINNER: 4-band spectral + variable history (our previous best)
3. AKIRA_7_1: Full 7+1 architecture (no history attention)
4. AKIRA_7_1_HIST: Full 7+1 + per-band history attention
5. AKIRA_7_1_VAR_HIST: Full 7+1 + variable history depths per band

embed_dim = 512 (divisible by 8 for 8 bands of 64 each)

AKIRA Project - Experiment 029 v3
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
    """Configuration for full AKIRA 7+1 architecture."""
    vocab_size: int = 50257          # GPT-2 vocabulary
    embed_dim: int = 512             # Must be divisible by 8 (8 bands x 64 dim)
    num_layers: int = 6
    num_heads: int = 8
    max_seq_length: int = 256
    dropout: float = 0.1
    
    # 7 spectral bands + 1 temporal = 8 total
    num_spectral_bands: int = 7
    num_total_bands: int = 8
    
    # Differential learning rates per band (100x range)
    # Band 0 (DC) slowest -> Band 6 fastest, Band 7 (temporal) medium
    band_learning_rates: Tuple[float, ...] = (
        3e-5,   # Band 0: DC - identity, eternal patterns
        5e-5,   # Band 1: Coarse structure
        1e-4,   # Band 2: Medium structure  
        3e-4,   # Band 3: Transitions (bridge band)
        5e-4,   # Band 4: Fine structure
        1e-3,   # Band 5: Textures
        3e-3,   # Band 6: Edges, details
        3e-4,   # Band 7: Temporal - medium rate
    )
    
    # Temporal windows per band (Heisenberg uncertainty)
    # Low freq = good freq resolution, poor time -> see FAR back
    # High freq = poor freq resolution, good time -> see RECENT
    temporal_windows: Tuple[int, ...] = (128, 64, 32, 16, 16, 8, 4)
    
    # History attention settings (for ablation)
    max_history_uniform: int = 64
    band_history_depths: Tuple[int, ...] = (128, 64, 32, 16, 16, 8, 4)  # Match temporal windows
    decay_rate: float = 0.95
    
    # Training
    batch_size: int = 8              # Smaller batch for larger model
    learning_rate: float = 3e-4      # Base LR for non-band params
    total_steps: int = 5000
    eval_every: int = 500
    warmup_steps: int = 200
    
    @property
    def dim_per_band(self) -> int:
        return self.embed_dim // self.num_total_bands  # 512/8 = 64
    
    def validate(self):
        assert self.embed_dim % self.num_total_bands == 0, \
            f"embed_dim ({self.embed_dim}) must be divisible by 8"
        assert len(self.band_learning_rates) == self.num_total_bands
        assert len(self.temporal_windows) == self.num_spectral_bands
        return True

print("AKIRAConfig defined")
print(f"  embed_dim=512 -> dim_per_band={512//8}")
print(f"  7 spectral bands + 1 temporal band = 8 total")

# ==============================================================================
# CELL 4: CAUSAL CONV1D (CRITICAL FOR LM)
# ==============================================================================

class CausalConv1d(nn.Module):
    """Causal 1D convolution - only sees past and current, never future."""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = kernel_size - 1  # All padding on left
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]
        x_padded = F.pad(x, (self.padding, 0))  # Left pad only
        return self.conv(x_padded)

print("CausalConv1d defined")

# ==============================================================================
# CELL 5: HISTORY BUFFER
# ==============================================================================

class HistoryBuffer(nn.Module):
    """History buffer for storing past states (beliefs)."""
    
    def __init__(self, max_history: int, feature_dim: int):
        super().__init__()
        self.max_history = max_history
        self.feature_dim = feature_dim
        self.register_buffer('buffer', None)
        self.register_buffer('current_length', torch.tensor(0))
    
    def reset(self):
        self.buffer = None
        self.current_length = torch.tensor(0)
    
    def update(self, state: torch.Tensor) -> None:
        if state.dim() == 2:
            state = state.unsqueeze(0)
        B, P, D = state.shape
        
        if self.buffer is None:
            self.buffer = torch.zeros(self.max_history, P, D, device=state.device, dtype=state.dtype)
            self.current_length = torch.tensor(0, device=state.device)
        
        state_mean = state.mean(dim=0).detach().clone()
        
        if self.current_length < self.max_history:
            idx = self.current_length.item()
            new_buffer = self.buffer.clone()
            new_buffer[idx] = state_mean
            self.buffer = new_buffer
            self.current_length = self.current_length + 1
        else:
            self.buffer = torch.cat([self.buffer[1:], state_mean.unsqueeze(0)], dim=0)
    
    def get_history(self) -> Optional[torch.Tensor]:
        if self.buffer is None or self.current_length == 0:
            return None
        valid = self.buffer[:self.current_length]
        return valid.transpose(0, 1)  # [P, T_hist, D]

print("HistoryBuffer defined")

# ==============================================================================
# CELL 6: SPECTRAL DECOMPOSER (7 bands, CAUSAL)
# ==============================================================================

class SpectralDecomposer(nn.Module):
    """
    Decomposes input into 7 frequency bands using CAUSAL learnable filters.
    
    Band 0 (DC): kernel=15, captures global/slow patterns
    Band 6 (HF): kernel=1, captures local/fast details
    """
    
    def __init__(self, embed_dim: int, num_bands: int = 7):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_bands = num_bands
        self.dim_per_band = embed_dim // 8  # 64 for 512 embed
        
        # Kernel sizes: larger for low freq, smaller for high freq
        kernel_sizes = [15, 11, 9, 7, 5, 3, 1]
        
        # Input projections per band
        self.input_projs = nn.ModuleList([
            nn.Linear(embed_dim, embed_dim) for _ in range(num_bands)
        ])
        
        # CAUSAL convolutions (critical for LM)
        self.band_convs = nn.ModuleList([
            CausalConv1d(embed_dim, embed_dim, k) if k > 1 
            else nn.Identity()  # kernel=1 needs no conv
            for k in kernel_sizes
        ])
        
        # Output projections to band dimension
        self.output_projs = nn.ModuleList([
            nn.Linear(embed_dim, self.dim_per_band) for _ in range(num_bands)
        ])
        
        # Learnable gates
        self.band_gates = nn.ParameterList([
            nn.Parameter(torch.ones(embed_dim) * (0.5 + 0.1 * i))
            for i in range(num_bands)
        ])
    
    def forward(self, x: torch.Tensor) -> Dict[int, torch.Tensor]:
        B, T, D = x.shape
        bands = {}
        
        for band_idx in range(self.num_bands):
            h = self.input_projs[band_idx](x)
            
            # Apply causal conv (transpose for conv1d)
            if not isinstance(self.band_convs[band_idx], nn.Identity):
                h_t = h.transpose(1, 2)  # [B, D, T]
                h_conv = self.band_convs[band_idx](h_t)
                h = h_conv.transpose(1, 2)  # [B, T, D]
            
            # Gate
            gate = torch.sigmoid(self.band_gates[band_idx])
            h = h * gate.unsqueeze(0).unsqueeze(0)
            
            # Project to band dimension
            bands[band_idx] = self.output_projs[band_idx](h)
        
        return bands

print("SpectralDecomposer defined (7 bands, CAUSAL)")

# ==============================================================================
# CELL 7: SPECTRAL RECONSTRUCTOR
# ==============================================================================

class SpectralReconstructor(nn.Module):
    """Reconstructs full embedding from 7 spectral bands."""
    
    def __init__(self, embed_dim: int, num_bands: int = 7):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_bands = num_bands
        self.dim_per_band = embed_dim // 8
        
        self.band_recon = nn.ModuleList([
            nn.Linear(self.dim_per_band, embed_dim) for _ in range(num_bands)
        ])
        self.mix = nn.Linear(embed_dim * num_bands, embed_dim)
    
    def forward(self, bands: Dict[int, torch.Tensor]) -> torch.Tensor:
        reconstructed = [self.band_recon[i](bands[i]) for i in range(self.num_bands)]
        concat = torch.cat(reconstructed, dim=-1)
        return self.mix(concat)

print("SpectralReconstructor defined")

# ==============================================================================
# CELL 8: BAND PROCESSORS (Geometric/Hybrid/Reactive)
# ==============================================================================

class GeometricProcessor(nn.Module):
    """Low freq bands (0-2): Slow, structure-preserving."""
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim * 2)
        self.fc2 = nn.Linear(dim * 2, dim)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Linear(dim, dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        h = F.gelu(self.fc1(x))
        h = self.dropout(h)
        h = self.fc2(h)
        gate = torch.sigmoid(self.gate(residual))
        return residual + gate * h


class HybridProcessor(nn.Module):
    """Mid freq bands (3-4): Balanced."""
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim * 3)
        self.fc2 = nn.Linear(dim * 3, dim)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = F.gelu(self.fc1(x))
        x = self.dropout(x)
        return residual + self.fc2(x)


class ReactiveProcessor(nn.Module):
    """High freq bands (5-6): Fast, adaptive."""
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim * 4)
        self.fc2 = nn.Linear(dim * 4, dim)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = F.gelu(self.fc1(x))
        x = self.dropout(x)
        return residual + self.fc2(x)

print("Band processors defined (Geometric/Hybrid/Reactive)")

# ==============================================================================
# CELL 9: TEMPORAL WORMHOLE (Differential windows, complementary pairs)
# ==============================================================================

class TemporalWormhole(nn.Module):
    """
    Wormhole attention with DIFFERENTIAL TEMPORAL WINDOWS (Heisenberg).
    
    - Band 0 sees 128 back <-> Band 6 sees 4 back
    - Band 1 sees 64 back  <-> Band 5 sees 8 back
    - Band 2 sees 32 back  <-> Band 4 sees 16 back
    - Band 3 (bridge) aggregates from all
    """
    
    def __init__(self, dim: int, temporal_windows: Tuple[int, ...], max_seq_len: int = 256):
        super().__init__()
        self.dim = dim
        self.temporal_windows = temporal_windows
        self.pairs = [(0, 6), (1, 5), (2, 4)]
        
        # Per-band attention
        self.q_projs = nn.ModuleList([nn.Linear(dim, dim) for _ in range(7)])
        self.k_projs = nn.ModuleList([nn.Linear(dim, dim) for _ in range(7)])
        self.v_projs = nn.ModuleList([nn.Linear(dim, dim) for _ in range(7)])
        
        # Cross-band projections
        self.cross_projs = nn.ModuleDict({
            f"{l}_{h}": nn.Linear(dim, dim) for l, h in self.pairs
        })
        self.cross_projs_rev = nn.ModuleDict({
            f"{l}_{h}": nn.Linear(dim, dim) for l, h in self.pairs
        })
        
        # Output gates
        self.gates = nn.ModuleList([nn.Linear(dim * 2, dim) for _ in range(7)])
        self.bridge_proj = nn.Linear(dim, dim)
        
        # Precompute causal masks with different windows
        for band_idx, window in enumerate(temporal_windows):
            mask = torch.ones(max_seq_len, max_seq_len)
            for i in range(max_seq_len):
                start = max(0, i - window + 1)
                mask[i, start:i+1] = 0
            self.register_buffer(f"mask_{band_idx}", mask.bool())
    
    def forward(self, bands: Dict[int, torch.Tensor]) -> Dict[int, torch.Tensor]:
        outputs = {k: v.clone() for k, v in bands.items()}
        B, T, D = bands[0].shape
        
        # Step 1: Per-band attention within temporal window
        band_contexts = {}
        for band_idx in range(7):
            band = bands[band_idx]
            q = self.q_projs[band_idx](band)
            k = self.k_projs[band_idx](band)
            v = self.v_projs[band_idx](band)
            
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(D)
            mask = getattr(self, f"mask_{band_idx}")[:T, :T]
            scores = scores.masked_fill(mask.unsqueeze(0), -1e4)
            
            attn = F.softmax(scores, dim=-1)
            attn = torch.nan_to_num(attn, nan=0.0)
            band_contexts[band_idx] = torch.matmul(attn, v)
        
        # Step 2: Wormhole transfer between complementary pairs
        for low, high in self.pairs:
            key = f"{low}_{high}"
            
            low_ctx = band_contexts[low]
            high_ctx = band_contexts[high]
            
            wormhole_to_high = self.cross_projs[key](low_ctx)
            wormhole_to_low = self.cross_projs_rev[key](high_ctx)
            
            gate_high = torch.sigmoid(self.gates[high](
                torch.cat([bands[high], wormhole_to_high], dim=-1)))
            gate_low = torch.sigmoid(self.gates[low](
                torch.cat([bands[low], wormhole_to_low], dim=-1)))
            
            outputs[high] = outputs[high] + gate_high * wormhole_to_high
            outputs[low] = outputs[low] + gate_low * wormhole_to_low
        
        # Step 3: Bridge band aggregates
        other_ctx = torch.stack([band_contexts[i] for i in range(7) if i != 3], dim=0).mean(dim=0)
        bridge_info = self.bridge_proj(other_ctx)
        gate_bridge = torch.sigmoid(self.gates[3](
            torch.cat([bands[3], bridge_info], dim=-1)))
        outputs[3] = outputs[3] + gate_bridge * bridge_info
        
        return outputs

print("TemporalWormhole defined (differential windows, complementary pairs)")

# ==============================================================================
# CELL 10: TEMPORAL BAND (Band 7 - Causal attention)
# ==============================================================================

class TemporalBand(nn.Module):
    """Band 7: Causal attention over sequence (orthogonal to spectral)."""
    
    def __init__(self, dim: int, max_seq_len: int, num_heads: int = 4, dropout: float = 0.1):
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
        
        self.register_buffer("causal_mask",
            torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1).bool())
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        residual = x
        x = self.norm(x)
        
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(self.causal_mask[:T, :T].unsqueeze(0).unsqueeze(0), -1e4)
        
        attn = F.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return residual + self.out_proj(out)

print("TemporalBand defined (Band 7)")

# ==============================================================================
# CELL 11: BAND HISTORY ATTENTION (Optional per-band history)
# ==============================================================================

class BandHistoryAttention(nn.Module):
    """Per-band history attention - each band remembers its own beliefs."""
    
    def __init__(self, dim: int, max_history: int, decay_rate: float):
        super().__init__()
        self.dim = dim
        self.max_history = max_history
        self.history_buffer = HistoryBuffer(max_history, dim)
        
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        
        decay_weights = torch.tensor([decay_rate ** i for i in range(max_history)])
        self.register_buffer('decay_weights', decay_weights)
    
    def forward(self, x: torch.Tensor, update_buffer: bool = True) -> torch.Tensor:
        B, T, D = x.shape
        
        history = self.history_buffer.get_history()
        
        if history is None:
            output = self.out_proj(x)
            if update_buffer:
                self.history_buffer.update(output.detach())
            return output
        
        Q = self.q_proj(x)
        T_hist = history.shape[1]
        K = self.k_proj(history)
        V = self.v_proj(history)
        
        Q_exp = Q.unsqueeze(2)
        K_exp = K.unsqueeze(0)
        V_exp = V.unsqueeze(0)
        
        scores = torch.matmul(Q_exp, K_exp.transpose(-2, -1)) / math.sqrt(D)
        
        decay = self.decay_weights[:T_hist].flip(0).view(1, 1, 1, -1)
        scores = scores + torch.log(decay + 1e-10)
        
        attn = F.softmax(scores, dim=-1)
        attended = torch.matmul(attn, V_exp).squeeze(2)
        
        output = self.out_proj(attended + x)
        
        if update_buffer:
            self.history_buffer.update(output.detach())
        
        return output
    
    def reset_history(self):
        self.history_buffer.reset()

print("BandHistoryAttention defined")

# ==============================================================================
# CELL 12: SPECTRAL ATTENTION (Causal, for each band)
# ==============================================================================

class SpectralAttention(nn.Module):
    """Causal attention for spectral bands."""
    
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
        
        self.register_buffer("causal_mask",
            torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1).bool())
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        residual = x
        x = self.norm(x)
        
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(self.causal_mask[:T, :T].unsqueeze(0).unsqueeze(0), -1e4)
        
        attn = F.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return residual + self.out_proj(out)

print("SpectralAttention defined (causal)")

# ==============================================================================
# CELL 13: FULL AKIRA 7+1 LAYER
# ==============================================================================

class AKIRALayer(nn.Module):
    """
    Full AKIRA 7+1 layer with optional per-band history attention.
    
    Flow:
    1. Spectral decomposition into 7 bands
    2. Per-band attention + processing
    3. (Optional) Per-band history attention
    4. Temporal wormhole cross-band communication
    5. Aggregate spectral -> Temporal band processing
    6. Reconstruct + combine
    """
    
    def __init__(self, config: AKIRAConfig, history_mode: str = "none"):
        """
        Args:
            config: AKIRA configuration
            history_mode: "none", "uniform", or "variable"
        """
        super().__init__()
        self.config = config
        self.history_mode = history_mode
        dim = config.dim_per_band
        
        # Spectral decomposition
        self.decomposer = SpectralDecomposer(config.embed_dim, config.num_spectral_bands)
        
        # Per-band attention (causal)
        self.band_attns = nn.ModuleList([
            SpectralAttention(dim, config.max_seq_length, num_heads=2, dropout=config.dropout)
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
        
        # Optional: Per-band history attention
        self.band_history_attns = None
        if history_mode == "uniform":
            self.band_history_attns = nn.ModuleList([
                BandHistoryAttention(dim, config.max_history_uniform, config.decay_rate)
                for _ in range(7)
            ])
            self.band_history_norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(7)])
        elif history_mode == "variable":
            self.band_history_attns = nn.ModuleList([
                BandHistoryAttention(dim, config.band_history_depths[i], config.decay_rate)
                for i in range(7)
            ])
            self.band_history_norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(7)])
        
        # Temporal wormhole
        self.wormhole = TemporalWormhole(dim, config.temporal_windows, config.max_seq_length)
        
        # Temporal band (Band 7)
        self.temporal_band = TemporalBand(dim, config.max_seq_length, num_heads=2, dropout=config.dropout)
        self.temporal_agg = nn.Linear(config.embed_dim - dim, dim)  # 7 bands -> 1
        
        # Reconstructor
        self.reconstructor = SpectralReconstructor(config.embed_dim, config.num_spectral_bands)
        
        # Pre-norm for input (critical for stable training)
        self.input_norm = nn.LayerNorm(config.embed_dim)
        
        # Output (spectral + temporal combined)
        self.output_proj = nn.Linear(config.embed_dim + dim, config.embed_dim)
    
    def forward(self, x: torch.Tensor, update_history: bool = True) -> torch.Tensor:
        B, T, D = x.shape
        residual = x  # CRITICAL: Save for residual connection
        
        # Pre-norm (like Pre-LN transformer for stability)
        x = self.input_norm(x)
        
        # 1. Spectral decomposition
        bands = self.decomposer(x)
        
        # 2. Per-band attention + processing
        for i in range(7):
            bands[i] = self.band_attns[i](bands[i])
            bands[i] = self.band_processors[i](bands[i])
        
        # 3. Optional: Per-band history attention
        if self.band_history_attns is not None:
            for i in range(7):
                bands[i] = bands[i] + self.band_history_attns[i](
                    self.band_history_norms[i](bands[i]),
                    update_buffer=update_history
                )
        
        # 4. Wormhole cross-band
        bands = self.wormhole(bands)
        
        # 5. Temporal band
        spectral_concat = torch.cat([bands[i] for i in range(7)], dim=-1)
        temporal_input = self.temporal_agg(spectral_concat)
        temporal_output = self.temporal_band(temporal_input)
        
        # 6. Reconstruct + combine
        spectral_output = self.reconstructor(bands)
        combined = torch.cat([spectral_output, temporal_output], dim=-1)
        
        # CRITICAL: Add residual connection for gradient flow
        return residual + self.output_proj(combined)
    
    def reset_history(self):
        if self.band_history_attns is not None:
            for ha in self.band_history_attns:
                ha.reset_history()

print("AKIRALayer defined (full 7+1)")

# ==============================================================================
# CELL 14: AKIRA LANGUAGE MODEL
# ==============================================================================

class AKIRALM(nn.Module):
    """Full AKIRA 7+1 Language Model."""
    
    def __init__(self, config: AKIRAConfig, history_mode: str = "none"):
        super().__init__()
        config.validate()
        self.config = config
        self.history_mode = history_mode
        
        self.token_embed = nn.Embedding(config.vocab_size, config.embed_dim)
        self.pos_embed = nn.Embedding(config.max_seq_length, config.embed_dim)
        self.dropout = nn.Dropout(config.dropout)
        
        self.layers = nn.ModuleList([
            AKIRALayer(config, history_mode) for _ in range(config.num_layers)
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
    
    def get_optimizer_param_groups(self) -> List[Dict]:
        """Differential LRs per band."""
        band_params = {i: [] for i in range(8)}
        other_params = []
        
        for name, param in self.named_parameters():
            assigned = False
            for band_idx in range(7):
                if f'band_processors.{band_idx}' in name or f'band_attns.{band_idx}' in name:
                    band_params[band_idx].append(param)
                    assigned = True
                    break
            if 'temporal_band' in name:
                band_params[7].append(param)
                assigned = True
            if not assigned:
                other_params.append(param)
        
        param_groups = []
        for band_idx in range(8):
            if band_params[band_idx]:
                param_groups.append({
                    'params': band_params[band_idx],
                    'lr': self.config.band_learning_rates[band_idx],
                    'name': f'band_{band_idx}'
                })
        if other_params:
            param_groups.append({
                'params': other_params,
                'lr': self.config.learning_rate,
                'name': 'shared'
            })
        return param_groups
    
    def forward(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None,
                update_history: bool = True) -> Dict[str, torch.Tensor]:
        B, T = input_ids.shape
        
        positions = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, -1)
        x = self.token_embed(input_ids) + self.pos_embed(positions)
        x = self.dropout(x)
        
        for layer in self.layers:
            x = layer(x, update_history=update_history)
        
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
    
    def reset_history(self):
        for layer in self.layers:
            layer.reset_history()

print("AKIRALM defined")

# ==============================================================================
# CELL 15: BASELINE GPT-2 (for comparison)
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
    
    def forward(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None,
                update_history: bool = True) -> Dict[str, torch.Tensor]:
        B, T = input_ids.shape
        
        positions = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, -1)
        x = self.token_embed(input_ids) + self.pos_embed(positions)
        x = self.dropout(x)
        
        mask = self._get_causal_mask(T, x.device)
        for layer in self.layers:
            x = layer(x, src_mask=mask, is_causal=True)
        
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
    
    def reset_history(self):
        pass

print("BaselineGPT2 defined")

# ==============================================================================
# CELL 16: DATA LOADING
# ==============================================================================

def load_wikitext2(tokenizer, max_length: int = 256, split: str = "train"):
    from datasets import load_dataset
    print(f"Loading WikiText-2 {split}...")
    dataset = load_dataset("wikitext", "wikitext-2-v1", split=split)
    
    def tokenize_function(examples):
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt"
        )
        tokenized["labels"] = tokenized["input_ids"].clone()
        return tokenized
    
    dataset = dataset.filter(lambda x: len(x["text"].strip()) > 0)
    tokenized = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return tokenized

print("Data loading defined")

# ==============================================================================
# CELL 17: TRAINING FUNCTION
# ==============================================================================

def train_model(model, train_loader, eval_loader, config: AKIRAConfig, model_name: str,
                use_differential_lr: bool = False) -> Dict[str, List[float]]:
    model = model.to(device)
    
    if use_differential_lr and hasattr(model, 'get_optimizer_param_groups'):
        param_groups = model.get_optimizer_param_groups()
        optimizer = AdamW(param_groups, weight_decay=0.01)
    else:
        optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.01)
    
    scheduler = CosineAnnealingLR(optimizer, T_max=config.total_steps, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda') if device == "cuda" else None
    
    history = {'train_loss': [], 'eval_loss': [], 'eval_ppl': [], 'step': []}
    
    model.train()
    train_iter = iter(train_loader)
    
    print(f"\nTraining {model_name}...")
    print(f"Parameters: {model.num_params:,}")
    
    start_time = time.time()
    
    for step in tqdm(range(config.total_steps), desc=model_name):
        try:
            batch = next(train_iter)
        except StopIteration:
            if hasattr(model, 'reset_history'):
                model.reset_history()
            train_iter = iter(train_loader)
            batch = next(train_iter)
        
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        
        optimizer.zero_grad()
        
        if scaler:
            with torch.amp.autocast('cuda'):
                output = model(input_ids, labels=labels)
                loss = output['loss']
            
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            output = model(input_ids, labels=labels)
            loss = output['loss']
            
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        
        scheduler.step()
        
        if (step + 1) % config.eval_every == 0:
            model.eval()
            if hasattr(model, 'reset_history'):
                model.reset_history()
            
            eval_losses = []
            with torch.no_grad():
                for i, batch in enumerate(eval_loader):
                    if i >= 50:
                        break
                    input_ids = batch["input_ids"].to(device)
                    labels = batch["labels"].to(device)
                    
                    if scaler:
                        with torch.amp.autocast('cuda'):
                            output = model(input_ids, labels=labels, update_history=False)
                    else:
                        output = model(input_ids, labels=labels, update_history=False)
                    eval_losses.append(output['loss'].item())
            
            eval_loss = sum(eval_losses) / len(eval_losses)
            eval_ppl = math.exp(min(eval_loss, 10))
            
            history['step'].append(step + 1)
            history['train_loss'].append(loss.item())
            history['eval_loss'].append(eval_loss)
            history['eval_ppl'].append(eval_ppl)
            
            elapsed = time.time() - start_time
            tqdm.write(f"  Step {step+1}: eval_loss={eval_loss:.4f}, ppl={eval_ppl:.2f}, time={elapsed:.1f}s")
            
            model.train()
    
    print(f"  Training complete in {time.time() - start_time:.1f}s")
    return history

print("Training function defined")

# ==============================================================================
# CELL 18: ABLATION CONFIGURATIONS
# ==============================================================================

ABLATIONS = {
    "baseline": True,           # Standard GPT-2
    "akira_7_1": True,          # Full AKIRA 7+1 (no history)
    "akira_7_1_hist": True,     # AKIRA + uniform history
    "akira_7_1_var_hist": True, # AKIRA + variable history (Heisenberg)
}

ABLATION_NAMES = {
    "baseline": "1. Baseline (Standard GPT-2)",
    "akira_7_1": "2. AKIRA 7+1 (No History)",
    "akira_7_1_hist": "3. AKIRA 7+1 + Uniform History",
    "akira_7_1_var_hist": "4. AKIRA 7+1 + Variable History (Full)",
}

print(f"Ablations: {[k for k, v in ABLATIONS.items() if v]}")

# ==============================================================================
# CELL 19: RUN EXPERIMENT
# ==============================================================================

def run_ablation():
    print("\n" + "="*70)
    print("EXPERIMENT 029 v3: Full AKIRA 7+1 Architecture Ablation")
    print("="*70)
    
    from transformers import GPT2Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    config = AKIRAConfig()
    
    print(f"\nConfiguration:")
    print(f"  embed_dim: {config.embed_dim} (8 bands x {config.dim_per_band})")
    print(f"  num_layers: {config.num_layers}")
    print(f"  Bands: 7 spectral + 1 temporal = 8")
    print(f"  Temporal windows: {config.temporal_windows}")
    print(f"  History depths: {config.band_history_depths}")
    print(f"  total_steps: {config.total_steps}")
    
    train_dataset = load_wikitext2(tokenizer, max_length=config.max_seq_length, split="train")
    eval_dataset = load_wikitext2(tokenizer, max_length=config.max_seq_length, split="validation")
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size,
                              shuffle=True, num_workers=2, pin_memory=True)
    eval_loader = DataLoader(eval_dataset, batch_size=config.batch_size,
                             shuffle=False, num_workers=2, pin_memory=True)
    
    print(f"\nDataset: Train={len(train_dataset)}, Eval={len(eval_dataset)}")
    
    results = {}
    
    # 1. Baseline
    if ABLATIONS.get("baseline"):
        print("\n" + "-"*60)
        print(ABLATION_NAMES["baseline"])
        print("-"*60)
        model = BaselineGPT2(config)
        results["baseline"] = train_model(model, train_loader, eval_loader, config,
                                          ABLATION_NAMES["baseline"])
        del model
        if device == "cuda": torch.cuda.empty_cache()
    
    # 2. AKIRA 7+1 (no history)
    if ABLATIONS.get("akira_7_1"):
        print("\n" + "-"*60)
        print(ABLATION_NAMES["akira_7_1"])
        print("-"*60)
        model = AKIRALM(config, history_mode="none")
        results["akira_7_1"] = train_model(model, train_loader, eval_loader, config,
                                           ABLATION_NAMES["akira_7_1"], use_differential_lr=True)
        del model
        if device == "cuda": torch.cuda.empty_cache()
    
    # 3. AKIRA + uniform history
    if ABLATIONS.get("akira_7_1_hist"):
        print("\n" + "-"*60)
        print(ABLATION_NAMES["akira_7_1_hist"])
        print("-"*60)
        model = AKIRALM(config, history_mode="uniform")
        results["akira_7_1_hist"] = train_model(model, train_loader, eval_loader, config,
                                                ABLATION_NAMES["akira_7_1_hist"], use_differential_lr=True)
        del model
        if device == "cuda": torch.cuda.empty_cache()
    
    # 4. AKIRA + variable history
    if ABLATIONS.get("akira_7_1_var_hist"):
        print("\n" + "-"*60)
        print(ABLATION_NAMES["akira_7_1_var_hist"])
        print("-"*60)
        model = AKIRALM(config, history_mode="variable")
        results["akira_7_1_var_hist"] = train_model(model, train_loader, eval_loader, config,
                                                    ABLATION_NAMES["akira_7_1_var_hist"], use_differential_lr=True)
        del model
        if device == "cuda": torch.cuda.empty_cache()
    
    # Results
    print("\n" + "="*70)
    print("ABLATION RESULTS")
    print("="*70)
    
    print(f"\n{'Model':<50} {'Loss':>10} {'PPL':>10} {'vs Base':>12}")
    print("-" * 82)
    
    baseline_ppl = results.get('baseline', {}).get('eval_ppl', [None])[-1]
    
    for mode in ["baseline", "akira_7_1", "akira_7_1_hist", "akira_7_1_var_hist"]:
        if mode in results:
            loss = results[mode]['eval_loss'][-1]
            ppl = results[mode]['eval_ppl'][-1]
            if baseline_ppl:
                impr = (baseline_ppl - ppl) / baseline_ppl * 100
            else:
                impr = 0
            print(f"{ABLATION_NAMES[mode]:<50} {loss:>10.4f} {ppl:>10.2f} {impr:>+11.2f}%")
    
    print("\n" + "="*70)
    print("EXPERIMENT 029 v3 COMPLETE")
    print("="*70)
    
    return results

# ==============================================================================
# CELL 20: RUN
# ==============================================================================

if __name__ == "__main__":
    results = run_ablation()
