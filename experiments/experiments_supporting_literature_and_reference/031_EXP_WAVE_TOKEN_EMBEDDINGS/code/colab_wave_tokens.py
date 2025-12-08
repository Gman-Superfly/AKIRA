# -*- coding: utf-8 -*-
"""
Experiment 031: Wave-Based Token Embeddings

Tests whether representing tokens as waves (frequency + phase + amplitude)
enables more natural spectral decomposition in AKIRA.

Hypothesis: If tokens ARE waves, spectral decomposition becomes REAL,
not learned approximation.

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
    
    Instead of: token_id → lookup_table[token_id] → [D] vector
    We do: token_id → (freq, phase, amp) → wave(t) → [D] vector
    
    Each token has learnable:
    - Frequencies: what spectral content this token represents
    - Phases: temporal offset/alignment
    - Amplitudes: strength of each frequency component
    
    The embedding at position t incorporates the position into the wave,
    creating a natural time-frequency representation.
    """
    
    def __init__(self, vocab_size: int, embed_dim: int, n_frequencies: int, 
                 freq_range: Tuple[float, float] = (0.01, 1.0)):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.n_frequencies = n_frequencies
        
        # Each token has learnable frequency components
        # Initialize frequencies in a spread across the range
        init_freqs = torch.linspace(freq_range[0], freq_range[1], n_frequencies)
        init_freqs = init_freqs.unsqueeze(0).expand(vocab_size, -1)
        # Add small random variation per token
        init_freqs = init_freqs + torch.randn(vocab_size, n_frequencies) * 0.01
        
        self.frequencies = nn.Parameter(init_freqs)  # [V, n_freq]
        self.phases = nn.Parameter(torch.randn(vocab_size, n_frequencies) * 0.1)  # [V, n_freq]
        self.amplitudes = nn.Parameter(torch.ones(vocab_size, n_frequencies))  # [V, n_freq]
        
        # Project wave components to full embedding dimension
        # Wave gives us n_freq * 2 components (sin + cos)
        self.proj = nn.Linear(n_frequencies * 2, embed_dim)
        
        # Learnable token-specific bias (captures non-wave information)
        self.token_bias = nn.Parameter(torch.zeros(vocab_size, embed_dim))
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_ids: [B, T] - discrete token indices
        Returns:
            [B, T, embed_dim] - wave-based embeddings
        """
        B, T = token_ids.shape
        
        # Get per-token wave parameters: [B, T, n_freq]
        freq = self.frequencies[token_ids]  # [B, T, n_freq]
        phase = self.phases[token_ids]      # [B, T, n_freq]
        amp = self.amplitudes[token_ids]    # [B, T, n_freq]
        
        # Time positions: [T]
        t = torch.arange(T, device=token_ids.device, dtype=torch.float)
        
        # Generate waves: position affects the phase
        # theta = 2π * f * t + φ
        # Shape: [B, T, n_freq] * [1, T, 1] + [B, T, n_freq] → [B, T, n_freq]
        theta = 2 * math.pi * freq * t.view(1, T, 1) + phase
        
        # Wave components (sin and cos for full information)
        sin_component = amp * torch.sin(theta)  # [B, T, n_freq]
        cos_component = amp * torch.cos(theta)  # [B, T, n_freq]
        
        # Concatenate: [B, T, n_freq * 2]
        wave_repr = torch.cat([sin_component, cos_component], dim=-1)
        
        # Project to embedding dimension
        embedded = self.proj(wave_repr)  # [B, T, embed_dim]
        
        # Add token-specific bias (for non-wave semantic content)
        bias = self.token_bias[token_ids]  # [B, T, embed_dim]
        
        return embedded + bias

print("WaveTokenEmbedding defined")

# ==============================================================================
# CELL 4B: WAVE DECODER (Symmetric Decode)
# ==============================================================================

class WaveDecoder(nn.Module):
    """
    Symmetric wave decoding: match output to token wave signatures.
    
    Instead of: hidden → linear → logits (ignores wave structure)
    We do: hidden → wave signature → compare to all token signatures → logits
    
    This tests whether symmetric encode/decode is better.
    """
    
    def __init__(self, embed_dim: int, vocab_size: int, n_frequencies: int,
                 wave_embed: WaveTokenEmbedding):
        super().__init__()
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.n_frequencies = n_frequencies
        
        # Share wave parameters with encoder (tied weights)
        self.wave_embed = wave_embed
        
        # Project hidden state to wave space
        self.hidden_to_wave = nn.Linear(embed_dim, n_frequencies * 2)
        
        # Temperature for similarity matching
        self.temperature = nn.Parameter(torch.ones(1))
    
    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_state: [B, T, embed_dim] - model output
        Returns:
            [B, T, vocab_size] - logits
        """
        B, T, D = hidden_state.shape
        
        # Project to wave space: [B, T, n_freq * 2]
        output_wave = self.hidden_to_wave(hidden_state)
        
        # Generate all token wave signatures at position 0 (reference)
        # We compare wave "shapes" not position-specific waves
        all_token_ids = torch.arange(self.vocab_size, device=hidden_state.device)
        
        # Get wave parameters for all tokens
        freq = self.wave_embed.frequencies  # [V, n_freq]
        phase = self.wave_embed.phases      # [V, n_freq]
        amp = self.wave_embed.amplitudes    # [V, n_freq]
        
        # Generate reference waves at t=0 (just the signature)
        theta = phase  # At t=0, theta = phase
        sin_comp = amp * torch.sin(theta)   # [V, n_freq]
        cos_comp = amp * torch.cos(theta)   # [V, n_freq]
        token_signatures = torch.cat([sin_comp, cos_comp], dim=-1)  # [V, n_freq * 2]
        
        # Normalize for cosine similarity
        output_norm = F.normalize(output_wave, dim=-1)  # [B, T, n_freq * 2]
        token_norm = F.normalize(token_signatures, dim=-1)  # [V, n_freq * 2]
        
        # Compute similarity: [B, T, V]
        logits = torch.matmul(output_norm, token_norm.T) / self.temperature
        
        return logits

print("WaveDecoder defined (symmetric)")


class FrequencyDecoder(nn.Module):
    """
    Frequency-domain decoding: match by spectrum similarity.
    
    Each token has a characteristic frequency spectrum.
    Match output spectrum to find most likely token.
    """
    
    def __init__(self, embed_dim: int, vocab_size: int, n_frequencies: int,
                 wave_embed: WaveTokenEmbedding):
        super().__init__()
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.n_frequencies = n_frequencies
        
        # Share frequency parameters with encoder
        self.wave_embed = wave_embed
        
        # Project hidden to frequency magnitudes
        self.hidden_to_spectrum = nn.Linear(embed_dim, n_frequencies)
        
        # Learnable output scale
        self.scale = nn.Parameter(torch.ones(1) * 10.0)
    
    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_state: [B, T, embed_dim]
        Returns:
            [B, T, vocab_size] - logits
        """
        B, T, D = hidden_state.shape
        
        # Extract output spectrum: [B, T, n_freq]
        output_spectrum = self.hidden_to_spectrum(hidden_state)
        output_spectrum = F.softplus(output_spectrum)  # Positive magnitudes
        
        # Token spectra = their amplitude patterns: [V, n_freq]
        token_spectra = F.softplus(self.wave_embed.amplitudes)
        
        # Normalize
        output_norm = F.normalize(output_spectrum, dim=-1)
        token_norm = F.normalize(token_spectra, dim=-1)
        
        # Similarity: [B, T, V]
        logits = torch.matmul(output_norm, token_norm.T) * self.scale
        
        return logits

print("FrequencyDecoder defined (spectrum matching)")

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
    """
    TRUE spectral decomposition using FFT.
    
    This is what AKIRA's learned convolutions were trying to approximate.
    With wave-based tokens, this becomes the natural decomposition.
    """
    
    def __init__(self, embed_dim: int, num_bands: int = 7):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_bands = num_bands
        self.dim_per_band = embed_dim // num_bands
        
        # Learnable band boundaries (frequency cutoffs)
        # Start with linear spacing
        init_cutoffs = torch.linspace(0, 1, num_bands + 1)[1:-1]  # Exclude 0 and 1
        self.band_cutoffs = nn.Parameter(init_cutoffs)
        
        # Per-band projection to ensure consistent dimension
        self.band_projs = nn.ModuleList([
            nn.Linear(embed_dim, self.dim_per_band) for _ in range(num_bands)
        ])
    
    def forward(self, x: torch.Tensor) -> Dict[int, torch.Tensor]:
        """
        Args:
            x: [B, T, D] - input embeddings
        Returns:
            Dict of 7 bands, each [B, T, dim_per_band]
        """
        B, T, D = x.shape
        
        # Apply FFT along embedding dimension (treating each position as a signal)
        # This decomposes the embedding into frequency components
        x_fft = torch.fft.rfft(x, dim=-1)  # [B, T, D//2+1] complex
        
        # Get magnitude and phase
        magnitude = torch.abs(x_fft)  # [B, T, D//2+1]
        phase = torch.angle(x_fft)    # [B, T, D//2+1]
        
        # Number of frequency bins
        n_freqs = magnitude.shape[-1]
        
        # Sort cutoffs to ensure valid ranges
        cutoffs = torch.sigmoid(self.band_cutoffs)  # Ensure 0-1 range
        cutoffs, _ = torch.sort(cutoffs)
        
        # Create frequency indices
        freq_idx = torch.linspace(0, 1, n_freqs, device=x.device)
        
        bands = {}
        prev_cutoff = 0.0
        
        for i in range(self.num_bands):
            if i < self.num_bands - 1:
                curr_cutoff = cutoffs[i]
            else:
                curr_cutoff = 1.0
            
            # Create soft mask for this frequency band
            # Smooth transitions using sigmoid
            lower_mask = torch.sigmoid((freq_idx - prev_cutoff) * 20)
            upper_mask = torch.sigmoid((curr_cutoff - freq_idx) * 20)
            band_mask = lower_mask * upper_mask  # [n_freqs]
            
            # Apply mask to magnitude (keep phase)
            masked_mag = magnitude * band_mask.view(1, 1, -1)
            
            # Reconstruct with phase
            band_fft = masked_mag * torch.exp(1j * phase)
            
            # Inverse FFT to get band signal
            band_signal = torch.fft.irfft(band_fft, n=D, dim=-1)  # [B, T, D]
            
            # Project to band dimension
            bands[i] = self.band_projs[i](band_signal)  # [B, T, dim_per_band]
            
            prev_cutoff = curr_cutoff
        
        return bands

print("FFTSpectralDecomposer defined")

# ==============================================================================
# CELL 7: LEARNED CONVOLUTION DECOMPOSER (from Exp 030 for comparison)
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
    """Learned convolution-based decomposition (from Exp 030)."""
    
    def __init__(self, embed_dim: int, num_bands: int = 7):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_bands = num_bands
        self.dim_per_band = embed_dim // num_bands
        
        # Different kernel sizes for different frequency bands
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
        x_conv = x.transpose(1, 2)  # [B, D, T]
        
        bands = {}
        for i in range(self.num_bands):
            band = self.decomposers[i](x_conv)  # [B, dim_per_band, T]
            band = band.transpose(1, 2)  # [B, T, dim_per_band]
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
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, T, head_dim]
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
# CELL 11: SPECTRAL BAND ATTENTION (for AKIRA variants)
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
# CELL 12: SPECTRAL RECONSTRUCTOR
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
# CELL 13: MODEL VARIANTS
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
    """Wave embeddings + AKIRA with learned convolution decomposition."""
    
    def __init__(self, config: WaveConfig):
        super().__init__()
        self.config = config
        dim = config.dim_per_band
        
        self.embed = WaveTokenEmbedding(
            config.vocab_size, config.embed_dim, config.n_frequencies
        )
        self.dropout = nn.Dropout(config.dropout)
        
        # Per layer: decompose, attend per-band, reconstruct
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
            
            # Decompose into bands
            bands = self.decomposers[i](x)
            
            # Per-band attention
            for j in range(self.config.num_bands):
                bands[j] = bands[j] + self.band_attns[i][j](bands[j])
            
            # Reconstruct
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


class WaveAKIRA_FFT(nn.Module):
    """Wave embeddings + AKIRA with TRUE FFT decomposition."""
    
    def __init__(self, config: WaveConfig):
        super().__init__()
        self.config = config
        dim = config.dim_per_band
        
        self.embed = WaveTokenEmbedding(
            config.vocab_size, config.embed_dim, config.n_frequencies
        )
        self.dropout = nn.Dropout(config.dropout)
        
        # Per layer: FFT decompose, attend per-band, reconstruct
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
            
            # FFT decompose into bands
            bands = self.decomposers[i](x)
            
            # Per-band attention
            for j in range(self.config.num_bands):
                bands[j] = bands[j] + self.band_attns[i][j](bands[j])
            
            # Reconstruct
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


class WaveGPT2_SymmetricDecode(nn.Module):
    """Wave embeddings + standard transformer + SYMMETRIC wave decoding."""
    
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
        
        # Symmetric wave decoder instead of linear
        self.decoder = WaveDecoder(
            config.embed_dim, config.vocab_size, config.n_frequencies, self.embed
        )
        
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
        logits = self.decoder(x)  # Wave-based decoding
        
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

print("WaveGPT2_SymmetricDecode defined")


class WaveGPT2_FreqDecode(nn.Module):
    """Wave embeddings + standard transformer + FREQUENCY spectrum decoding."""
    
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
        
        # Frequency decoder instead of linear
        self.decoder = FrequencyDecoder(
            config.embed_dim, config.vocab_size, config.n_frequencies, self.embed
        )
        
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
        logits = self.decoder(x)  # Frequency-based decoding
        
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

print("WaveGPT2_FreqDecode defined")

# ==============================================================================
# CELL 14: DATA LOADING
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
# CELL 15: TRAINING
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
# CELL 16: RUN EXPERIMENTS
# ==============================================================================

def run_experiments():
    """Run all wave token experiments."""
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
    
    # 3. Wave embeddings + AKIRA (learned decomposition)
    print("\n" + "="*60)
    print("3. Wave + AKIRA (Learned Bands)")
    print("="*60)
    model = WaveAKIRA_Learned(config)
    all_results["wave_akira_learned"] = train_model(model, train_loader, val_loader, config, "3. Wave+AKIRA Learned")
    del model
    torch.cuda.empty_cache() if device == "cuda" else None
    
    # 4. Wave embeddings + AKIRA (FFT decomposition)
    print("\n" + "="*60)
    print("4. Wave + AKIRA (FFT Bands)")
    print("="*60)
    model = WaveAKIRA_FFT(config)
    all_results["wave_akira_fft"] = train_model(model, train_loader, val_loader, config, "4. Wave+AKIRA FFT")
    del model
    torch.cuda.empty_cache() if device == "cuda" else None
    
    # 5. Wave embeddings + Symmetric Wave Decode (tied encode/decode)
    print("\n" + "="*60)
    print("5. Wave Embed + Symmetric Wave Decode")
    print("="*60)
    model = WaveGPT2_SymmetricDecode(config)
    all_results["wave_symmetric"] = train_model(model, train_loader, val_loader, config, "5. Wave Symmetric")
    del model
    torch.cuda.empty_cache() if device == "cuda" else None
    
    # 6. Wave embeddings + Frequency Spectrum Decode
    print("\n" + "="*60)
    print("6. Wave Embed + Frequency Decode")
    print("="*60)
    model = WaveGPT2_FreqDecode(config)
    all_results["wave_freq_decode"] = train_model(model, train_loader, val_loader, config, "6. Wave Freq Decode")
    del model
    torch.cuda.empty_cache() if device == "cuda" else None
    
    return all_results

print("Experiment runner defined")

# ==============================================================================
# CELL 17: MAIN
# ==============================================================================

if __name__ == "__main__":
    results = run_experiments()
    
    # Print summary
    print("\n" + "="*70)
    print("EXPERIMENT 031 RESULTS: Wave-Based Token Embeddings")
    print("="*70)
    
    baseline_ppl = results["baseline"]["final_ppl"]
    
    print(f"\n{'Model':<45} {'Loss':>10} {'PPL':>10} {'vs Base':>12}")
    print("-"*80)
    
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
            "wave_gpt2": "2. Wave Embed + Transformer", #linear decode
            "wave_akira_learned": "3. Wave + AKIRA (Learned)",
            "wave_akira_fft": "4. Wave + AKIRA (FFT)",
            "wave_symmetric": "5. Wave Symmetric Decode",
            "wave_freq_decode": "6. Wave Frequency Decode"
        }.get(name, name)
        
        print(f"{display_name:<45} {loss:>10.4f} {ppl:>10.2f} {delta:>12}")
    
    print("-"*80)
    
    # Analysis
    print("\n" + "="*70)
    print("ANALYSIS")
    print("="*70)
    
    wave_vs_standard = (baseline_ppl - results["wave_gpt2"]["final_ppl"]) / baseline_ppl * 100
    print(f"\nWave embeddings alone: {wave_vs_standard:+.2f}% vs standard")
    
    if "wave_akira_fft" in results and "wave_akira_learned" in results:
        fft_vs_learned = (results["wave_akira_learned"]["final_ppl"] - results["wave_akira_fft"]["final_ppl"]) / results["wave_akira_learned"]["final_ppl"] * 100
        print(f"FFT vs Learned decomposition: {fft_vs_learned:+.2f}%")
    
    # Decode comparison
    if "wave_gpt2" in results and "wave_symmetric" in results:
        sym_vs_linear = (results["wave_gpt2"]["final_ppl"] - results["wave_symmetric"]["final_ppl"]) / results["wave_gpt2"]["final_ppl"] * 100
        print(f"Symmetric vs Linear decode: {sym_vs_linear:+.2f}%")
    
    if "wave_gpt2" in results and "wave_freq_decode" in results:
        freq_vs_linear = (results["wave_gpt2"]["final_ppl"] - results["wave_freq_decode"]["final_ppl"]) / results["wave_gpt2"]["final_ppl"] * 100
        print(f"Frequency vs Linear decode: {freq_vs_linear:+.2f}%")
    
    # Find winner
    best_name = min(results.keys(), key=lambda k: results[k]["final_ppl"])
    best_ppl = results[best_name]["final_ppl"]
    
    print(f"\n>>> WINNER: {best_name}")
    print(f"    Perplexity: {best_ppl:.2f}")
    
    if best_name != "baseline":
        improvement = (baseline_ppl - best_ppl) / baseline_ppl * 100
        print(f"    Improvement over baseline: +{improvement:.2f}%")
    
    print("\n" + "="*70)
    print("EXPERIMENT 031 COMPLETE")
    print("="*70)
