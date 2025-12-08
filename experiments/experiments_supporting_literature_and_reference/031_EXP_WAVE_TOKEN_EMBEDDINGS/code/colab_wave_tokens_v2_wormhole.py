# -*- coding: utf-8 -*-
"""
Experiment 031 v2: Wave-Based Token Embeddings WITH WORMHOLE ATTENTION

Adds wormhole cross-band communication to the AKIRA variants.
This tests the full spectral architecture: decompose -> per-band attention -> wormhole -> reconstruct

Pipeline:
1. Spectral decomposition (FFT or learned)
2. Per-band attention (within each band)
3. WORMHOLE attention (cross-band communication) <-- NEW
4. Reconstruction

Wormhole pairs:
- Band 0 <-> Band 6 (identity <-> position)
- Band 1 <-> Band 5 (shape <-> texture)
- Band 2 <-> Band 4 (structure <-> detail)
- Band 3 -> all (bridge band)

Run in Google Colab with GPU runtime.
"""

# ==============================================================================
# CELL 1: SETUP
# ==============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

# Check device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# ==============================================================================
# CELL 2: INSTALL DEPENDENCIES (Colab)
# ==============================================================================

# !pip install datasets transformers -q

# ==============================================================================
# CELL 3: CONFIG
# ==============================================================================

@dataclass
class WaveConfig:
    """Configuration for wave-based token experiments."""
    # Model
    vocab_size: int = 50257  # GPT-2 vocab
    embed_dim: int = 512
    num_layers: int = 6
    num_heads: int = 8
    dropout: float = 0.1
    max_seq_length: int = 256
    
    # Wave embedding specific
    n_frequencies: int = 64  # Number of frequency components per token
    freq_range: Tuple[float, float] = (0.01, 1.0)  # Frequency range
    
    # AKIRA specific
    num_bands: int = 7
    dim_per_band: int = 73  # embed_dim // num_bands = 512 // 7 = 73
    
    # Wormhole specific
    wormhole_top_k: int = 16  # Sparse top-k connections
    wormhole_threshold: float = 0.5  # Coherence threshold
    
    # Training
    batch_size: int = 16
    learning_rate: float = 3e-4
    total_steps: int = 5000
    eval_every: int = 500
    
    def validate(self):
        assert self.embed_dim % self.num_heads == 0
        assert self.n_frequencies * 2 <= self.embed_dim  # sin + cos components

print("Config defined")

# ==============================================================================
# CELL 4: WAVE TOKEN EMBEDDING
# ==============================================================================

class WaveTokenEmbedding(nn.Module):
    """
    Represent each token as a superposition of waves.
    
    Instead of: token_id -> lookup_table[token_id] -> [D] vector
    We do: token_id -> (freq, phase, amp) -> wave(t) -> [D] vector
    """
    
    def __init__(self, vocab_size: int, embed_dim: int, n_frequencies: int, 
                 freq_range: Tuple[float, float] = (0.01, 1.0)):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.n_frequencies = n_frequencies
        
        # Each token has learnable frequency components
        init_freqs = torch.linspace(freq_range[0], freq_range[1], n_frequencies)
        init_freqs = init_freqs.unsqueeze(0).expand(vocab_size, -1)
        init_freqs = init_freqs + torch.randn(vocab_size, n_frequencies) * 0.01
        
        self.frequencies = nn.Parameter(init_freqs)  # [V, n_freq]
        self.phases = nn.Parameter(torch.randn(vocab_size, n_frequencies) * 0.1)  # [V, n_freq]
        self.amplitudes = nn.Parameter(torch.ones(vocab_size, n_frequencies))  # [V, n_freq]
        
        self.proj = nn.Linear(n_frequencies * 2, embed_dim)
        self.token_bias = nn.Parameter(torch.zeros(vocab_size, embed_dim))
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        B, T = token_ids.shape
        
        freq = self.frequencies[token_ids]
        phase = self.phases[token_ids]
        amp = self.amplitudes[token_ids]
        
        t = torch.arange(T, device=token_ids.device, dtype=torch.float)
        theta = 2 * math.pi * freq * t.view(1, T, 1) + phase
        
        sin_component = amp * torch.sin(theta)
        cos_component = amp * torch.cos(theta)
        
        wave_repr = torch.cat([sin_component, cos_component], dim=-1)
        embedded = self.proj(wave_repr)
        bias = self.token_bias[token_ids]
        
        return embedded + bias

print("WaveTokenEmbedding defined")

# ==============================================================================
# CELL 5: STANDARD EMBEDDING (for comparison)
# ==============================================================================

class StandardEmbedding(nn.Module):
    """Standard lookup-table embedding."""
    
    def __init__(self, vocab_size: int, embed_dim: int, max_seq_len: int):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Embedding(max_seq_len, embed_dim)
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        B, T = token_ids.shape
        positions = torch.arange(T, device=token_ids.device).unsqueeze(0).expand(B, -1)
        return self.token_embed(token_ids) + self.pos_embed(positions)

print("StandardEmbedding defined")

# ==============================================================================
# CELL 6: FFT-BASED SPECTRAL DECOMPOSITION
# ==============================================================================

class FFTSpectralDecomposer(nn.Module):
    """TRUE spectral decomposition using FFT."""
    
    def __init__(self, embed_dim: int, num_bands: int = 7):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_bands = num_bands
        self.dim_per_band = embed_dim // num_bands
        
        init_cutoffs = torch.linspace(0, 1, num_bands + 1)[1:-1]
        self.band_cutoffs = nn.Parameter(init_cutoffs)
        
        self.band_projs = nn.ModuleList([
            nn.Linear(embed_dim, self.dim_per_band) for _ in range(num_bands)
        ])
    
    def forward(self, x: torch.Tensor) -> Dict[int, torch.Tensor]:
        B, T, D = x.shape
        
        x_fft = torch.fft.rfft(x, dim=-1)
        magnitude = torch.abs(x_fft)
        phase = torch.angle(x_fft)
        
        n_freqs = magnitude.shape[-1]
        
        cutoffs = torch.sigmoid(self.band_cutoffs)
        cutoffs, _ = torch.sort(cutoffs)
        
        freq_idx = torch.linspace(0, 1, n_freqs, device=x.device)
        
        bands = {}
        prev_cutoff = 0.0
        
        for i in range(self.num_bands):
            if i < self.num_bands - 1:
                curr_cutoff = cutoffs[i]
            else:
                curr_cutoff = 1.0
            
            lower_mask = torch.sigmoid((freq_idx - prev_cutoff) * 20)
            upper_mask = torch.sigmoid((curr_cutoff - freq_idx) * 20)
            band_mask = lower_mask * upper_mask
            
            masked_mag = magnitude * band_mask.view(1, 1, -1)
            band_fft = masked_mag * torch.exp(1j * phase)
            band_signal = torch.fft.irfft(band_fft, n=D, dim=-1)
            
            bands[i] = self.band_projs[i](band_signal)
            prev_cutoff = curr_cutoff
        
        return bands

print("FFTSpectralDecomposer defined")

# ==============================================================================
# CELL 7: LEARNED CONVOLUTION DECOMPOSER
# ==============================================================================

class CausalConv1d(nn.Module):
    """Causal 1D convolution."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        self.padding = kernel_size - 1
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (self.padding, 0))
        return self.conv(x)

class LearnedSpectralDecomposer(nn.Module):
    """Learned convolution-based decomposition."""
    
    def __init__(self, embed_dim: int, num_bands: int = 7):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_bands = num_bands
        self.dim_per_band = embed_dim // num_bands
        
        kernel_sizes = [15, 13, 11, 9, 7, 5, 3]
        
        self.decomposers = nn.ModuleList([
            CausalConv1d(embed_dim, self.dim_per_band, kernel_sizes[i])
            for i in range(num_bands)
        ])
        
        self.norms = nn.ModuleList([
            nn.LayerNorm(self.dim_per_band) for _ in range(num_bands)
        ])
    
    def forward(self, x: torch.Tensor) -> Dict[int, torch.Tensor]:
        B, T, D = x.shape
        x_conv = x.transpose(1, 2)
        
        bands = {}
        for i in range(self.num_bands):
            band = self.decomposers[i](x_conv)
            band = band.transpose(1, 2)
            bands[i] = self.norms[i](band)
        
        return bands

print("LearnedSpectralDecomposer defined")

# ==============================================================================
# CELL 8: CAUSAL SELF-ATTENTION
# ==============================================================================

class CausalSelfAttention(nn.Module):
    """Standard causal self-attention."""
    
    def __init__(self, dim: int, num_heads: int, max_seq_len: int, dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        
        self.register_buffer("causal_mask",
            torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1).bool())
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(self.causal_mask[:T, :T].unsqueeze(0).unsqueeze(0), -1e9)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(B, T, D)
        return self.out_proj(out)

print("CausalSelfAttention defined")

# ==============================================================================
# CELL 9: FFN
# ==============================================================================

class FFN(nn.Module):
    """Standard feed-forward network."""
    
    def __init__(self, dim: int, expansion: int = 4, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim * expansion)
        self.fc2 = nn.Linear(dim * expansion, dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc2(F.gelu(self.fc1(x))))

print("FFN defined")

# ==============================================================================
# CELL 10: TRANSFORMER BLOCK
# ==============================================================================

class TransformerBlock(nn.Module):
    """Standard transformer block with pre-norm."""
    
    def __init__(self, dim: int, num_heads: int, max_seq_len: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = CausalSelfAttention(dim, num_heads, max_seq_len, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = FFN(dim, dropout=dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

print("TransformerBlock defined")

# ==============================================================================
# CELL 11: SPECTRAL BAND ATTENTION
# ==============================================================================

class SpectralBandAttention(nn.Module):
    """Per-band causal attention."""
    
    def __init__(self, dim: int, num_heads: int, max_seq_len: int, dropout: float = 0.1):
        super().__init__()
        self.attn = CausalSelfAttention(dim, num_heads, max_seq_len, dropout)
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.attn(self.norm(x))

print("SpectralBandAttention defined")

# ==============================================================================
# CELL 12: SPECTRAL WORMHOLE ATTENTION (NEW)
# ==============================================================================

class SpectralWormhole(nn.Module):
    """
    Wormhole cross-band attention.
    
    Connects complementary frequency bands:
    - Band 0 <-> Band 6 (identity <-> position)
    - Band 1 <-> Band 5 (shape <-> texture)
    - Band 2 <-> Band 4 (structure <-> detail)
    - Band 3 -> all (bridge band)
    
    Uses cosine similarity on hypersphere with sparse top-k selection.
    """
    
    def __init__(self, dim: int, num_bands: int = 7, top_k: int = 16, dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.num_bands = num_bands
        self.top_k = top_k
        
        # Complementary pairs: (low_band, high_band)
        self.pairs = [(0, 6), (1, 5), (2, 4)]
        self.bridge_band = 3
        
        # Projections for cross-band queries and keys
        # Each pair needs Q/K projections
        self.pair_q_projs = nn.ModuleList([
            nn.Linear(dim, dim) for _ in range(len(self.pairs) * 2)
        ])
        self.pair_k_projs = nn.ModuleList([
            nn.Linear(dim, dim) for _ in range(len(self.pairs) * 2)
        ])
        self.pair_v_projs = nn.ModuleList([
            nn.Linear(dim, dim) for _ in range(len(self.pairs) * 2)
        ])
        self.pair_out_projs = nn.ModuleList([
            nn.Linear(dim, dim) for _ in range(len(self.pairs) * 2)
        ])
        
        # Bridge band projections (connects to 6 other bands)
        self.bridge_q_proj = nn.Linear(dim, dim)
        self.bridge_k_projs = nn.ModuleList([
            nn.Linear(dim, dim) for _ in range(num_bands - 1)
        ])
        self.bridge_v_projs = nn.ModuleList([
            nn.Linear(dim, dim) for _ in range(num_bands - 1)
        ])
        self.bridge_out_proj = nn.Linear(dim, dim)
        
        # Learnable temperature per pair
        self.temperatures = nn.ParameterList([
            nn.Parameter(torch.ones(1)) for _ in range(len(self.pairs) * 2 + 1)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
        # Layer norms for residual
        self.norms = nn.ModuleList([
            nn.LayerNorm(dim) for _ in range(num_bands)
        ])
    
    def _sparse_cross_attention(
        self, 
        q: torch.Tensor,  # [B, T, D] queries from band A
        k: torch.Tensor,  # [B, T, D] keys from band B
        v: torch.Tensor,  # [B, T, D] values from band B
        temperature: torch.Tensor
    ) -> torch.Tensor:
        """
        Sparse cross-band attention using cosine similarity and top-k.
        """
        B, T, D = q.shape
        
        # Normalize onto hypersphere (cosine similarity)
        q_norm = F.normalize(q, dim=-1)
        k_norm = F.normalize(k, dim=-1)
        
        # Compute similarity: [B, T, T]
        sim = torch.matmul(q_norm, k_norm.transpose(-2, -1)) / temperature
        
        # Sparse top-k selection
        if self.top_k < T:
            # Get top-k values and indices
            top_k_vals, top_k_idx = sim.topk(self.top_k, dim=-1)
            
            # Create sparse attention mask
            sparse_attn = torch.zeros_like(sim)
            sparse_attn.scatter_(-1, top_k_idx, F.softmax(top_k_vals, dim=-1))
        else:
            # If T <= top_k, use full attention
            sparse_attn = F.softmax(sim, dim=-1)
        
        sparse_attn = self.dropout(sparse_attn)
        
        # Apply attention to values
        out = torch.matmul(sparse_attn, v)
        
        return out
    
    def forward(self, bands: Dict[int, torch.Tensor]) -> Dict[int, torch.Tensor]:
        """
        Apply wormhole cross-band attention.
        
        Args:
            bands: Dict of 7 band tensors, each [B, T, dim_per_band]
        Returns:
            Dict of 7 band tensors with cross-band information
        """
        # Clone bands for output
        out_bands = {i: bands[i].clone() for i in range(self.num_bands)}
        
        # Process complementary pairs
        for pair_idx, (low_band, high_band) in enumerate(self.pairs):
            low = self.norms[low_band](bands[low_band])
            high = self.norms[high_band](bands[high_band])
            
            # Low -> High direction
            proj_idx_lh = pair_idx * 2
            q_lh = self.pair_q_projs[proj_idx_lh](low)
            k_lh = self.pair_k_projs[proj_idx_lh](high)
            v_lh = self.pair_v_projs[proj_idx_lh](high)
            
            wormhole_lh = self._sparse_cross_attention(
                q_lh, k_lh, v_lh, self.temperatures[proj_idx_lh]
            )
            out_bands[low_band] = out_bands[low_band] + self.pair_out_projs[proj_idx_lh](wormhole_lh)
            
            # High -> Low direction
            proj_idx_hl = pair_idx * 2 + 1
            q_hl = self.pair_q_projs[proj_idx_hl](high)
            k_hl = self.pair_k_projs[proj_idx_hl](low)
            v_hl = self.pair_v_projs[proj_idx_hl](low)
            
            wormhole_hl = self._sparse_cross_attention(
                q_hl, k_hl, v_hl, self.temperatures[proj_idx_hl]
            )
            out_bands[high_band] = out_bands[high_band] + self.pair_out_projs[proj_idx_hl](wormhole_hl)
        
        # Bridge band (3) queries all other bands
        bridge = self.norms[self.bridge_band](bands[self.bridge_band])
        q_bridge = self.bridge_q_proj(bridge)
        
        bridge_accum = torch.zeros_like(bridge)
        other_bands = [i for i in range(self.num_bands) if i != self.bridge_band]
        
        for i, other_idx in enumerate(other_bands):
            other = self.norms[other_idx](bands[other_idx])
            k_other = self.bridge_k_projs[i](other)
            v_other = self.bridge_v_projs[i](other)
            
            bridge_from_other = self._sparse_cross_attention(
                q_bridge, k_other, v_other, self.temperatures[-1]
            )
            bridge_accum = bridge_accum + bridge_from_other
        
        # Average contribution from all bands
        bridge_accum = bridge_accum / len(other_bands)
        out_bands[self.bridge_band] = out_bands[self.bridge_band] + self.bridge_out_proj(bridge_accum)
        
        return out_bands

print("SpectralWormhole defined")

# ==============================================================================
# CELL 13: SPECTRAL RECONSTRUCTOR
# ==============================================================================

class SpectralReconstructor(nn.Module):
    """Reconstruct full embedding from spectral bands."""
    
    def __init__(self, embed_dim: int, dim_per_band: int, num_bands: int = 7):
        super().__init__()
        self.band_projs = nn.ModuleList([
            nn.Linear(dim_per_band, embed_dim) for _ in range(num_bands)
        ])
        self.mix = nn.Linear(embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, bands: Dict[int, torch.Tensor]) -> torch.Tensor:
        combined = sum(self.band_projs[i](bands[i]) for i in range(len(bands)))
        return self.norm(self.mix(combined))

print("SpectralReconstructor defined")

# ==============================================================================
# CELL 14: MODEL VARIANTS
# ==============================================================================

class BaselineGPT2(nn.Module):
    """Baseline: Standard embeddings + standard transformer."""
    
    def __init__(self, config: WaveConfig):
        super().__init__()
        self.config = config
        
        self.embed = StandardEmbedding(config.vocab_size, config.embed_dim, config.max_seq_length)
        self.dropout = nn.Dropout(config.dropout)
        
        self.layers = nn.ModuleList([
            TransformerBlock(config.embed_dim, config.num_heads, config.max_seq_length, config.dropout)
            for _ in range(config.num_layers)
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
        x = self.dropout(self.embed(input_ids))
        
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

print("BaselineGPT2 defined")


class WaveGPT2(nn.Module):
    """Wave embeddings + standard transformer."""
    
    def __init__(self, config: WaveConfig):
        super().__init__()
        self.config = config
        
        self.embed = WaveTokenEmbedding(
            config.vocab_size, config.embed_dim, config.n_frequencies
        )
        self.dropout = nn.Dropout(config.dropout)
        
        self.layers = nn.ModuleList([
            TransformerBlock(config.embed_dim, config.num_heads, config.max_seq_length, config.dropout)
            for _ in range(config.num_layers)
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
    
    def forward(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        x = self.dropout(self.embed(input_ids))
        
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

print("WaveGPT2 defined")


class WaveAKIRA_Learned(nn.Module):
    """Wave embeddings + AKIRA with learned decomposition (NO wormhole)."""
    
    def __init__(self, config: WaveConfig):
        super().__init__()
        self.config = config
        dim = config.dim_per_band
        
        self.embed = WaveTokenEmbedding(
            config.vocab_size, config.embed_dim, config.n_frequencies
        )
        self.dropout = nn.Dropout(config.dropout)
        
        self.decomposers = nn.ModuleList([
            LearnedSpectralDecomposer(config.embed_dim) for _ in range(config.num_layers)
        ])
        
        self.band_attns = nn.ModuleList([
            nn.ModuleList([
                SpectralBandAttention(dim, 2, config.max_seq_length, config.dropout)
                for _ in range(config.num_bands)
            ])
            for _ in range(config.num_layers)
        ])
        
        self.reconstructors = nn.ModuleList([
            SpectralReconstructor(config.embed_dim, dim) for _ in range(config.num_layers)
        ])
        
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(config.embed_dim) for _ in range(config.num_layers)
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
    
    def forward(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        x = self.dropout(self.embed(input_ids))
        
        for i in range(self.config.num_layers):
            residual = x
            x = self.layer_norms[i](x)
            
            bands = self.decomposers[i](x)
            
            for j in range(self.config.num_bands):
                bands[j] = bands[j] + self.band_attns[i][j](bands[j])
            
            x = self.reconstructors[i](bands)
            x = residual + x
        
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

print("WaveAKIRA_Learned defined")


class WaveAKIRA_Learned_Wormhole(nn.Module):
    """
    Wave embeddings + AKIRA with learned decomposition + WORMHOLE attention.
    
    Pipeline per layer:
    1. Decompose into 7 bands
    2. Per-band attention
    3. WORMHOLE cross-band attention
    4. Reconstruct
    """
    
    def __init__(self, config: WaveConfig):
        super().__init__()
        self.config = config
        dim = config.dim_per_band
        
        self.embed = WaveTokenEmbedding(
            config.vocab_size, config.embed_dim, config.n_frequencies
        )
        self.dropout = nn.Dropout(config.dropout)
        
        self.decomposers = nn.ModuleList([
            LearnedSpectralDecomposer(config.embed_dim) for _ in range(config.num_layers)
        ])
        
        self.band_attns = nn.ModuleList([
            nn.ModuleList([
                SpectralBandAttention(dim, 2, config.max_seq_length, config.dropout)
                for _ in range(config.num_bands)
            ])
            for _ in range(config.num_layers)
        ])
        
        # WORMHOLE attention (NEW)
        self.wormholes = nn.ModuleList([
            SpectralWormhole(dim, config.num_bands, config.wormhole_top_k, config.dropout)
            for _ in range(config.num_layers)
        ])
        
        self.reconstructors = nn.ModuleList([
            SpectralReconstructor(config.embed_dim, dim) for _ in range(config.num_layers)
        ])
        
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(config.embed_dim) for _ in range(config.num_layers)
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
    
    def forward(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        x = self.dropout(self.embed(input_ids))
        
        for i in range(self.config.num_layers):
            residual = x
            x = self.layer_norms[i](x)
            
            # 1. Decompose into bands
            bands = self.decomposers[i](x)
            
            # 2. Per-band attention
            for j in range(self.config.num_bands):
                bands[j] = bands[j] + self.band_attns[i][j](bands[j])
            
            # 3. WORMHOLE cross-band attention (NEW)
            bands = self.wormholes[i](bands)
            
            # 4. Reconstruct
            x = self.reconstructors[i](bands)
            x = residual + x
        
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

print("WaveAKIRA_Learned_Wormhole defined")


class WaveAKIRA_FFT(nn.Module):
    """Wave embeddings + AKIRA with FFT decomposition (NO wormhole)."""
    
    def __init__(self, config: WaveConfig):
        super().__init__()
        self.config = config
        dim = config.dim_per_band
        
        self.embed = WaveTokenEmbedding(
            config.vocab_size, config.embed_dim, config.n_frequencies
        )
        self.dropout = nn.Dropout(config.dropout)
        
        self.decomposers = nn.ModuleList([
            FFTSpectralDecomposer(config.embed_dim) for _ in range(config.num_layers)
        ])
        
        self.band_attns = nn.ModuleList([
            nn.ModuleList([
                SpectralBandAttention(dim, 2, config.max_seq_length, config.dropout)
                for _ in range(config.num_bands)
            ])
            for _ in range(config.num_layers)
        ])
        
        self.reconstructors = nn.ModuleList([
            SpectralReconstructor(config.embed_dim, dim) for _ in range(config.num_layers)
        ])
        
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(config.embed_dim) for _ in range(config.num_layers)
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
    
    def forward(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        x = self.dropout(self.embed(input_ids))
        
        for i in range(self.config.num_layers):
            residual = x
            x = self.layer_norms[i](x)
            
            bands = self.decomposers[i](x)
            
            for j in range(self.config.num_bands):
                bands[j] = bands[j] + self.band_attns[i][j](bands[j])
            
            x = self.reconstructors[i](bands)
            x = residual + x
        
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

print("WaveAKIRA_FFT defined")


class WaveAKIRA_FFT_Wormhole(nn.Module):
    """
    Wave embeddings + AKIRA with FFT decomposition + WORMHOLE attention.
    
    Pipeline per layer:
    1. FFT decompose into 7 bands
    2. Per-band attention
    3. WORMHOLE cross-band attention
    4. Reconstruct
    """
    
    def __init__(self, config: WaveConfig):
        super().__init__()
        self.config = config
        dim = config.dim_per_band
        
        self.embed = WaveTokenEmbedding(
            config.vocab_size, config.embed_dim, config.n_frequencies
        )
        self.dropout = nn.Dropout(config.dropout)
        
        self.decomposers = nn.ModuleList([
            FFTSpectralDecomposer(config.embed_dim) for _ in range(config.num_layers)
        ])
        
        self.band_attns = nn.ModuleList([
            nn.ModuleList([
                SpectralBandAttention(dim, 2, config.max_seq_length, config.dropout)
                for _ in range(config.num_bands)
            ])
            for _ in range(config.num_layers)
        ])
        
        # WORMHOLE attention (NEW)
        self.wormholes = nn.ModuleList([
            SpectralWormhole(dim, config.num_bands, config.wormhole_top_k, config.dropout)
            for _ in range(config.num_layers)
        ])
        
        self.reconstructors = nn.ModuleList([
            SpectralReconstructor(config.embed_dim, dim) for _ in range(config.num_layers)
        ])
        
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(config.embed_dim) for _ in range(config.num_layers)
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
    
    def forward(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        x = self.dropout(self.embed(input_ids))
        
        for i in range(self.config.num_layers):
            residual = x
            x = self.layer_norms[i](x)
            
            # 1. FFT decompose into bands
            bands = self.decomposers[i](x)
            
            # 2. Per-band attention
            for j in range(self.config.num_bands):
                bands[j] = bands[j] + self.band_attns[i][j](bands[j])
            
            # 3. WORMHOLE cross-band attention (NEW)
            bands = self.wormholes[i](bands)
            
            # 4. Reconstruct
            x = self.reconstructors[i](bands)
            x = residual + x
        
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

print("WaveAKIRA_FFT_Wormhole defined")

# ==============================================================================
# CELL 15: DATA LOADING
# ==============================================================================

def load_wikitext2(config: WaveConfig):
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
# CELL 16: TRAINING
# ==============================================================================

def train_model(model, train_loader, val_loader, config: WaveConfig, name: str):
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
                    if i >= 50:
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
# CELL 17: RUN EXPERIMENTS
# ==============================================================================

def run_experiments():
    """Run wave token experiments WITH wormhole comparison."""
    config = WaveConfig()
    
    train_loader, val_loader = load_wikitext2(config)
    
    all_results = {}
    
    # 1. Baseline: Standard embeddings + transformer
    print("\n" + "="*60)
    print("1. Baseline (Standard Embeddings)")
    print("="*60)
    model = BaselineGPT2(config)
    all_results["baseline"] = train_model(model, train_loader, val_loader, config, "1. Baseline")
    del model
    torch.cuda.empty_cache() if device == "cuda" else None
    
    # 2. Wave embeddings + standard transformer
    print("\n" + "="*60)
    print("2. Wave Embeddings + Standard Transformer")
    print("="*60)
    model = WaveGPT2(config)
    all_results["wave_gpt2"] = train_model(model, train_loader, val_loader, config, "2. Wave GPT2")
    del model
    torch.cuda.empty_cache() if device == "cuda" else None
    
    # 3. Wave + AKIRA (Learned) WITHOUT wormhole
    print("\n" + "="*60)
    print("3. Wave + AKIRA (Learned) - No Wormhole")
    print("="*60)
    model = WaveAKIRA_Learned(config)
    all_results["wave_akira_learned"] = train_model(model, train_loader, val_loader, config, "3. Wave+AKIRA Learned")
    del model
    torch.cuda.empty_cache() if device == "cuda" else None
    
    # 4. Wave + AKIRA (Learned) WITH wormhole
    print("\n" + "="*60)
    print("4. Wave + AKIRA (Learned) + WORMHOLE")
    print("="*60)
    model = WaveAKIRA_Learned_Wormhole(config)
    all_results["wave_akira_learned_wormhole"] = train_model(model, train_loader, val_loader, config, "4. Wave+AKIRA+Wormhole")
    del model
    torch.cuda.empty_cache() if device == "cuda" else None
    
    # 5. Wave + AKIRA (FFT) WITHOUT wormhole
    print("\n" + "="*60)
    print("5. Wave + AKIRA (FFT) - No Wormhole")
    print("="*60)
    model = WaveAKIRA_FFT(config)
    all_results["wave_akira_fft"] = train_model(model, train_loader, val_loader, config, "5. Wave+AKIRA FFT")
    del model
    torch.cuda.empty_cache() if device == "cuda" else None
    
    # 6. Wave + AKIRA (FFT) WITH wormhole
    print("\n" + "="*60)
    print("6. Wave + AKIRA (FFT) + WORMHOLE")
    print("="*60)
    model = WaveAKIRA_FFT_Wormhole(config)
    all_results["wave_akira_fft_wormhole"] = train_model(model, train_loader, val_loader, config, "6. Wave+FFT+Wormhole")
    del model
    torch.cuda.empty_cache() if device == "cuda" else None
    
    return all_results

print("Experiment runner defined")

# ==============================================================================
# CELL 18: MAIN
# ==============================================================================

if __name__ == "__main__":
    results = run_experiments()
    
    # Print summary
    print("\n" + "="*80)
    print("EXPERIMENT 031 v2 RESULTS: Wave Tokens + Wormhole Attention")
    print("="*80)
    
    baseline_ppl = results["baseline"]["final_ppl"]
    
    print(f"\n{'Model':<50} {'Loss':>10} {'PPL':>10} {'vs Base':>12}")
    print("-"*85)
    
    for name, res in results.items():
        ppl = res["final_ppl"]
        loss = res["final_loss"]
        if name == "baseline":
            delta = "+0.00%"
        else:
            improvement = (baseline_ppl - ppl) / baseline_ppl * 100
            delta = f"+{improvement:.2f}%" if improvement > 0 else f"{improvement:.2f}%"
        
        display_name = {
            "baseline": "1. Baseline (Standard Embed)",
            "wave_gpt2": "2. Wave Embed + Transformer",
            "wave_akira_learned": "3. Wave + AKIRA Learned (no wormhole)",
            "wave_akira_learned_wormhole": "4. Wave + AKIRA Learned + WORMHOLE",
            "wave_akira_fft": "5. Wave + AKIRA FFT (no wormhole)",
            "wave_akira_fft_wormhole": "6. Wave + AKIRA FFT + WORMHOLE"
        }.get(name, name)
        
        print(f"{display_name:<50} {loss:>10.4f} {ppl:>10.2f} {delta:>12}")
    
    print("-"*85)
    
    # Analysis
    print("\n" + "="*80)
    print("WORMHOLE ABLATION ANALYSIS")
    print("="*80)
    
    # Learned: with vs without wormhole
    if "wave_akira_learned" in results and "wave_akira_learned_wormhole" in results:
        no_worm = results["wave_akira_learned"]["final_ppl"]
        with_worm = results["wave_akira_learned_wormhole"]["final_ppl"]
        worm_effect = (no_worm - with_worm) / no_worm * 100
        print(f"\nLearned Decomposition:")
        print(f"  Without wormhole: {no_worm:.2f} PPL")
        print(f"  With wormhole:    {with_worm:.2f} PPL")
        print(f"  Wormhole effect:  {worm_effect:+.2f}%")
    
    # FFT: with vs without wormhole
    if "wave_akira_fft" in results and "wave_akira_fft_wormhole" in results:
        no_worm = results["wave_akira_fft"]["final_ppl"]
        with_worm = results["wave_akira_fft_wormhole"]["final_ppl"]
        worm_effect = (no_worm - with_worm) / no_worm * 100
        print(f"\nFFT Decomposition:")
        print(f"  Without wormhole: {no_worm:.2f} PPL")
        print(f"  With wormhole:    {with_worm:.2f} PPL")
        print(f"  Wormhole effect:  {worm_effect:+.2f}%")
    
    # Best overall
    best_name = min(results.keys(), key=lambda k: results[k]["final_ppl"])
    best_ppl = results[best_name]["final_ppl"]
    
    print(f"\n>>> BEST MODEL: {best_name}")
    print(f"    Perplexity: {best_ppl:.2f}")
    
    if best_name != "baseline":
        improvement = (baseline_ppl - best_ppl) / baseline_ppl * 100
        print(f"    Improvement over baseline: +{improvement:.2f}%")
    
    # Key question: Does wormhole help?
    print("\n" + "="*80)
    print("KEY FINDING: Does wormhole cross-band attention help?")
    print("="*80)
    
    learned_helps = results.get("wave_akira_learned_wormhole", {}).get("final_ppl", float('inf')) < results.get("wave_akira_learned", {}).get("final_ppl", float('inf'))
    fft_helps = results.get("wave_akira_fft_wormhole", {}).get("final_ppl", float('inf')) < results.get("wave_akira_fft", {}).get("final_ppl", float('inf'))
    
    if learned_helps and fft_helps:
        print("YES - Wormhole improves BOTH learned and FFT decomposition")
    elif learned_helps or fft_helps:
        print("MIXED - Wormhole helps one but not the other")
    else:
        print("NO - Wormhole does not improve performance")
    
    print("\n" + "="*80)
    print("EXPERIMENT 031 v2 COMPLETE")
    print("="*80)
