# -*- coding: utf-8 -*-
"""
Experiment 034: Zipf Wave Language Model

Tests the Zipf-grounded wave tokenizer in actual language modeling.

Ablations:
1. Baseline: Standard embedding + transformer + linear decode
2. ZipfWave + Linear: Zipf wave embedding + transformer + linear decode
3. ZipfWave + Symmetric: Zipf wave embedding + transformer + wave decode
4. ZipfWave + FFT: Zipf wave embedding + FFT decomposition + band attention
5. Random Wave: Random frequencies (ablation control)

Hypothesis:
- Zipf-grounded frequencies should enable meaningful spectral decomposition
- FFT-based band separation should work better with Zipf waves than random waves
- Symmetric wave decode may help preserve spectral structure

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

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# ==============================================================================
# CELL 2: INSTALL DEPENDENCIES (Uncomment for Colab)
# ==============================================================================

# !pip install datasets transformers -q

# ==============================================================================
# CELL 3: CONFIGURATION
# ==============================================================================

@dataclass
class ExperimentConfig:
    """Configuration for wave LM experiments."""
    # Model architecture
    vocab_size: int = 50257  # GPT-2 vocab
    embed_dim: int = 512
    num_layers: int = 6
    num_heads: int = 8
    dropout: float = 0.1
    max_seq_length: int = 256
    
    # Wave embedding
    n_harmonics: int = 8
    freq_min: float = 0.01
    freq_max: float = 1.0
    phase_mode: str = "learnable"  # learnable, position, fixed
    amplitude_mode: str = "zipf_scaled"  # fixed, learnable, zipf_scaled
    
    # AKIRA spectral
    num_bands: int = 7
    dim_per_band: int = 73  # embed_dim // num_bands
    
    # Training
    batch_size: int = 16
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    total_steps: int = 5000
    eval_every: int = 500
    warmup_steps: int = 200
    
    def validate(self):
        assert self.embed_dim % self.num_heads == 0
        return True


print("Config defined")

# ==============================================================================
# CELL 4: ZIPF WAVE EMBEDDING
# ==============================================================================

class ZipfWaveEmbedding(nn.Module):
    """
    Wave embedding with frequencies grounded in Zipf's Law.
    
    Common tokens (low rank) -> low frequency (DC-like)
    Rare tokens (high rank) -> high frequency
    
    This grounds spectral decomposition in information theory.
    """
    
    def __init__(self, config: ExperimentConfig):
        super().__init__()
        self.config = config
        
        # Compute Zipf-based frequencies
        # GPT-2 token IDs roughly correlate with BPE merge order (frequency)
        ranks = torch.arange(1, config.vocab_size + 1, dtype=torch.float32)
        
        # Log mapping: freq = freq_min + (freq_max - freq_min) * log(rank) / log(V)
        log_ranks = torch.log(ranks)
        log_vocab = math.log(config.vocab_size)
        normalized = log_ranks / log_vocab
        
        base_frequencies = config.freq_min + (config.freq_max - config.freq_min) * normalized
        self.register_buffer('base_frequencies', base_frequencies)  # [V]
        
        # Harmonic multipliers
        harmonics = torch.arange(1, config.n_harmonics + 1, dtype=torch.float32)
        self.register_buffer('harmonics', harmonics)  # [H]
        
        # Learnable phases per token per harmonic
        if config.phase_mode == "learnable":
            self.phases = nn.Parameter(torch.randn(config.vocab_size, config.n_harmonics) * 0.1)
        else:
            phases = torch.randn(config.vocab_size, config.n_harmonics) * 2 * math.pi
            self.register_buffer('phases', phases)
        
        # Amplitudes
        if config.amplitude_mode == "learnable":
            self.amplitudes = nn.Parameter(torch.ones(config.vocab_size, config.n_harmonics))
        elif config.amplitude_mode == "zipf_scaled":
            # Inverse Zipf: common tokens get higher amplitude
            inv_normalized = 1.0 - normalized
            amps = inv_normalized.unsqueeze(1) * (1.0 / harmonics.unsqueeze(0))
            self.register_buffer('amplitudes', amps)
        else:
            amps = (1.0 / harmonics).unsqueeze(0).expand(config.vocab_size, -1)
            self.register_buffer('amplitudes', amps)
        
        # Project wave to embedding dim
        wave_dim = config.n_harmonics * 2
        self.proj = nn.Linear(wave_dim, config.embed_dim)
        
        # Residual for non-wave information
        self.residual = nn.Parameter(torch.zeros(config.vocab_size, config.embed_dim))
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_ids: [B, T]
        Returns:
            [B, T, D] wave embeddings
        """
        B, T = token_ids.shape
        
        # Get per-token parameters
        freq = self.base_frequencies[token_ids]  # [B, T]
        phase = self.phases[token_ids]  # [B, T, H]
        amp = self.amplitudes[token_ids] if isinstance(self.amplitudes, nn.Parameter) else self.amplitudes[token_ids]
        
        # Time positions
        t = torch.arange(T, device=token_ids.device, dtype=torch.float32)
        
        # Compute harmonics: [B, T, H]
        frequencies = freq.unsqueeze(-1) * self.harmonics  # [B, T, H]
        theta = 2 * math.pi * frequencies * t.view(1, T, 1) + phase
        
        # Wave components
        sin_comp = amp * torch.sin(theta)
        cos_comp = amp * torch.cos(theta)
        wave = torch.cat([sin_comp, cos_comp], dim=-1)  # [B, T, 2H]
        
        # Project and add residual
        embedded = self.proj(wave) + self.residual[token_ids]
        
        return embedded
    
    def get_band_assignments(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Get spectral band for each token based on frequency."""
        freq = self.base_frequencies[token_ids]
        normalized = (freq - self.config.freq_min) / (self.config.freq_max - self.config.freq_min)
        bands = (normalized * self.config.num_bands).long().clamp(0, self.config.num_bands - 1)
        return bands


class RandomWaveEmbedding(nn.Module):
    """
    Ablation control: Random frequencies instead of Zipf-based.
    
    If Zipf grounding matters, this should perform worse.
    """
    
    def __init__(self, config: ExperimentConfig):
        super().__init__()
        self.config = config
        
        # Random frequencies (not Zipf-based)
        random_frequencies = torch.rand(config.vocab_size) * (config.freq_max - config.freq_min) + config.freq_min
        self.register_buffer('base_frequencies', random_frequencies)
        
        harmonics = torch.arange(1, config.n_harmonics + 1, dtype=torch.float32)
        self.register_buffer('harmonics', harmonics)
        
        self.phases = nn.Parameter(torch.randn(config.vocab_size, config.n_harmonics) * 0.1)
        
        amps = (1.0 / harmonics).unsqueeze(0).expand(config.vocab_size, -1)
        self.register_buffer('amplitudes', amps)
        
        wave_dim = config.n_harmonics * 2
        self.proj = nn.Linear(wave_dim, config.embed_dim)
        self.residual = nn.Parameter(torch.zeros(config.vocab_size, config.embed_dim))
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        B, T = token_ids.shape
        
        freq = self.base_frequencies[token_ids]
        phase = self.phases[token_ids]
        amp = self.amplitudes[token_ids]
        
        t = torch.arange(T, device=token_ids.device, dtype=torch.float32)
        frequencies = freq.unsqueeze(-1) * self.harmonics
        theta = 2 * math.pi * frequencies * t.view(1, T, 1) + phase
        
        sin_comp = amp * torch.sin(theta)
        cos_comp = amp * torch.cos(theta)
        wave = torch.cat([sin_comp, cos_comp], dim=-1)
        
        return self.proj(wave) + self.residual[token_ids]


class StandardEmbedding(nn.Module):
    """Standard lookup embedding (baseline)."""
    
    def __init__(self, config: ExperimentConfig):
        super().__init__()
        self.token_embed = nn.Embedding(config.vocab_size, config.embed_dim)
        self.pos_embed = nn.Embedding(config.max_seq_length, config.embed_dim)
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        B, T = token_ids.shape
        positions = torch.arange(T, device=token_ids.device).unsqueeze(0).expand(B, -1)
        return self.token_embed(token_ids) + self.pos_embed(positions)


print("Embeddings defined")

# ==============================================================================
# CELL 5: WAVE DECODER
# ==============================================================================

class WaveDecoder(nn.Module):
    """
    Symmetric wave decoder: match output to token wave signatures.
    """
    
    def __init__(self, config: ExperimentConfig, wave_embed: ZipfWaveEmbedding):
        super().__init__()
        self.config = config
        self.wave_embed = wave_embed
        
        wave_dim = config.n_harmonics * 2
        self.hidden_to_wave = nn.Linear(config.embed_dim, wave_dim)
        self.temperature = nn.Parameter(torch.ones(1))
    
    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden: [B, T, D]
        Returns:
            [B, T, V] logits
        """
        B, T, D = hidden.shape
        
        # Project to wave space
        output_wave = self.hidden_to_wave(hidden)  # [B, T, 2H]
        
        # Get reference signatures for all tokens
        phases = self.wave_embed.phases  # [V, H]
        if isinstance(self.wave_embed.amplitudes, nn.Parameter):
            amps = self.wave_embed.amplitudes
        else:
            amps = self.wave_embed.amplitudes
        
        # Reference at t=0
        sin_ref = amps * torch.sin(phases)
        cos_ref = amps * torch.cos(phases)
        ref_sigs = torch.cat([sin_ref, cos_ref], dim=-1)  # [V, 2H]
        
        # Cosine similarity
        output_norm = F.normalize(output_wave, dim=-1)
        ref_norm = F.normalize(ref_sigs, dim=-1)
        
        logits = torch.matmul(output_norm, ref_norm.T) / self.temperature.abs().clamp(min=0.01)
        
        return logits


print("Decoder defined")

# ==============================================================================
# CELL 6: TRANSFORMER COMPONENTS
# ==============================================================================

class CausalSelfAttention(nn.Module):
    """Standard causal self-attention."""
    
    def __init__(self, dim: int, num_heads: int, max_seq_len: int, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        
        mask = torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1).bool()
        self.register_buffer("causal_mask", mask)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(self.causal_mask[:T, :T], -1e9)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(B, T, D)
        return self.out_proj(out)


class FFN(nn.Module):
    """Feed-forward network."""
    
    def __init__(self, dim: int, expansion: int = 4, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim * expansion)
        self.fc2 = nn.Linear(dim * expansion, dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc2(F.gelu(self.fc1(x))))


class TransformerBlock(nn.Module):
    """Standard transformer block."""
    
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


print("Transformer components defined")

# ==============================================================================
# CELL 7: FFT SPECTRAL DECOMPOSITION
# ==============================================================================

class FFTSpectralDecomposer(nn.Module):
    """True FFT-based spectral decomposition into bands."""
    
    def __init__(self, embed_dim: int, num_bands: int = 7):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_bands = num_bands
        self.dim_per_band = embed_dim // num_bands
        
        # Learnable band boundaries
        init_cutoffs = torch.linspace(0, 1, num_bands + 1)[1:-1]
        self.band_cutoffs = nn.Parameter(init_cutoffs)
        
        # Per-band projection
        self.band_projs = nn.ModuleList([
            nn.Linear(embed_dim, self.dim_per_band) for _ in range(num_bands)
        ])
    
    def forward(self, x: torch.Tensor) -> Dict[int, torch.Tensor]:
        B, T, D = x.shape
        
        # FFT along embedding dimension
        x_fft = torch.fft.rfft(x, dim=-1)
        magnitude = torch.abs(x_fft)
        phase = torch.angle(x_fft)
        
        n_freqs = magnitude.shape[-1]
        cutoffs = torch.sigmoid(self.band_cutoffs).sort()[0]
        freq_idx = torch.linspace(0, 1, n_freqs, device=x.device)
        
        bands = {}
        prev_cutoff = 0.0
        
        for i in range(self.num_bands):
            curr_cutoff = cutoffs[i].item() if i < self.num_bands - 1 else 1.0
            
            # Soft mask
            lower = torch.sigmoid((freq_idx - prev_cutoff) * 20)
            upper = torch.sigmoid((curr_cutoff - freq_idx) * 20)
            mask = lower * upper
            
            masked_mag = magnitude * mask.view(1, 1, -1)
            band_fft = masked_mag * torch.exp(1j * phase)
            band_signal = torch.fft.irfft(band_fft, n=D, dim=-1)
            
            bands[i] = self.band_projs[i](band_signal)
            prev_cutoff = curr_cutoff
        
        return bands


class SpectralReconstructor(nn.Module):
    """Reconstruct from spectral bands."""
    
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


print("Spectral components defined")

# ==============================================================================
# CELL 8: MODEL VARIANTS
# ==============================================================================

class BaselineModel(nn.Module):
    """Baseline: Standard embedding + transformer + linear decode."""
    
    def __init__(self, config: ExperimentConfig):
        super().__init__()
        self.config = config
        
        self.embed = StandardEmbedding(config)
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
    
    def forward(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None):
        x = self.dropout(self.embed(input_ids))
        for layer in self.layers:
            x = layer(x)
        x = self.final_norm(x)
        logits = self.lm_head(x)
        
        output = {'logits': logits}
        if labels is not None:
            loss = F.cross_entropy(
                logits[..., :-1, :].contiguous().view(-1, self.config.vocab_size),
                labels[..., 1:].contiguous().view(-1),
                ignore_index=-100
            )
            output['loss'] = loss
        return output


class ZipfWaveModel(nn.Module):
    """Zipf wave embedding + transformer + linear decode."""
    
    def __init__(self, config: ExperimentConfig):
        super().__init__()
        self.config = config
        
        self.embed = ZipfWaveEmbedding(config)
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
    
    def forward(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None):
        x = self.dropout(self.embed(input_ids))
        for layer in self.layers:
            x = layer(x)
        x = self.final_norm(x)
        logits = self.lm_head(x)
        
        output = {'logits': logits}
        if labels is not None:
            loss = F.cross_entropy(
                logits[..., :-1, :].contiguous().view(-1, self.config.vocab_size),
                labels[..., 1:].contiguous().view(-1),
                ignore_index=-100
            )
            output['loss'] = loss
        return output


class ZipfWaveSymmetricModel(nn.Module):
    """Zipf wave embedding + transformer + symmetric wave decode."""
    
    def __init__(self, config: ExperimentConfig):
        super().__init__()
        self.config = config
        
        self.embed = ZipfWaveEmbedding(config)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([
            TransformerBlock(config.embed_dim, config.num_heads, config.max_seq_length, config.dropout)
            for _ in range(config.num_layers)
        ])
        self.final_norm = nn.LayerNorm(config.embed_dim)
        self.decoder = WaveDecoder(config, self.embed)
        
        self.apply(self._init_weights)
        self.num_params = sum(p.numel() for p in self.parameters())
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    def forward(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None):
        x = self.dropout(self.embed(input_ids))
        for layer in self.layers:
            x = layer(x)
        x = self.final_norm(x)
        logits = self.decoder(x)
        
        output = {'logits': logits}
        if labels is not None:
            loss = F.cross_entropy(
                logits[..., :-1, :].contiguous().view(-1, self.config.vocab_size),
                labels[..., 1:].contiguous().view(-1),
                ignore_index=-100
            )
            output['loss'] = loss
        return output


class ZipfWaveFFTModel(nn.Module):
    """Zipf wave embedding + FFT decomposition + band attention."""
    
    def __init__(self, config: ExperimentConfig):
        super().__init__()
        self.config = config
        dim = config.dim_per_band
        
        self.embed = ZipfWaveEmbedding(config)
        self.dropout = nn.Dropout(config.dropout)
        
        # Per-layer: FFT decompose -> band attention -> reconstruct
        self.decomposers = nn.ModuleList([
            FFTSpectralDecomposer(config.embed_dim, config.num_bands)
            for _ in range(config.num_layers)
        ])
        
        self.band_attns = nn.ModuleList([
            nn.ModuleList([
                CausalSelfAttention(dim, 2, config.max_seq_length, config.dropout)
                for _ in range(config.num_bands)
            ])
            for _ in range(config.num_layers)
        ])
        
        self.band_norms = nn.ModuleList([
            nn.ModuleList([nn.LayerNorm(dim) for _ in range(config.num_bands)])
            for _ in range(config.num_layers)
        ])
        
        self.reconstructors = nn.ModuleList([
            SpectralReconstructor(config.embed_dim, dim, config.num_bands)
            for _ in range(config.num_layers)
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
    
    def forward(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None):
        x = self.dropout(self.embed(input_ids))
        
        for i in range(self.config.num_layers):
            residual = x
            x = self.layer_norms[i](x)
            
            # FFT decompose
            bands = self.decomposers[i](x)
            
            # Per-band attention
            for j in range(self.config.num_bands):
                bands[j] = bands[j] + self.band_attns[i][j](self.band_norms[i][j](bands[j]))
            
            # Reconstruct
            x = self.reconstructors[i](bands)
            x = residual + x
        
        x = self.final_norm(x)
        logits = self.lm_head(x)
        
        output = {'logits': logits}
        if labels is not None:
            loss = F.cross_entropy(
                logits[..., :-1, :].contiguous().view(-1, self.config.vocab_size),
                labels[..., 1:].contiguous().view(-1),
                ignore_index=-100
            )
            output['loss'] = loss
        return output


class RandomWaveModel(nn.Module):
    """Ablation: Random frequencies instead of Zipf-based."""
    
    def __init__(self, config: ExperimentConfig):
        super().__init__()
        self.config = config
        
        self.embed = RandomWaveEmbedding(config)
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
    
    def forward(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None):
        x = self.dropout(self.embed(input_ids))
        for layer in self.layers:
            x = layer(x)
        x = self.final_norm(x)
        logits = self.lm_head(x)
        
        output = {'logits': logits}
        if labels is not None:
            loss = F.cross_entropy(
                logits[..., :-1, :].contiguous().view(-1, self.config.vocab_size),
                labels[..., 1:].contiguous().view(-1),
                ignore_index=-100
            )
            output['loss'] = loss
        return output


print("Models defined")

# ==============================================================================
# CELL 9: DATA LOADING
# ==============================================================================

def load_wikitext2(config: ExperimentConfig):
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
# CELL 10: TRAINING
# ==============================================================================

def train_model(model, train_loader, val_loader, config: ExperimentConfig, name: str):
    """Train model and return results."""
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
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
        
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
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
    
    results["final_loss"] = results["val_loss"][-1]
    results["final_ppl"] = results["val_ppl"][-1]
    results["time"] = time.time() - start_time
    
    return results


print("Training function defined")

# ==============================================================================
# CELL 11: RUN EXPERIMENTS
# ==============================================================================

def run_experiments():
    """Run all experiments."""
    config = ExperimentConfig()
    config.validate()
    
    train_loader, val_loader = load_wikitext2(config)
    
    all_results = {}
    
    # 1. Baseline
    print("\n" + "="*60)
    print("1. BASELINE (Standard Embedding)")
    print("="*60)
    model = BaselineModel(config)
    all_results["baseline"] = train_model(model, train_loader, val_loader, config, "Baseline")
    del model
    torch.cuda.empty_cache() if device == "cuda" else None
    
    # 2. Zipf Wave + Linear
    print("\n" + "="*60)
    print("2. ZIPF WAVE + LINEAR DECODE")
    print("="*60)
    model = ZipfWaveModel(config)
    all_results["zipf_wave_linear"] = train_model(model, train_loader, val_loader, config, "ZipfWave+Linear")
    del model
    torch.cuda.empty_cache() if device == "cuda" else None
    
    # 3. Zipf Wave + Symmetric
    print("\n" + "="*60)
    print("3. ZIPF WAVE + SYMMETRIC DECODE")
    print("="*60)
    model = ZipfWaveSymmetricModel(config)
    all_results["zipf_wave_symmetric"] = train_model(model, train_loader, val_loader, config, "ZipfWave+Symmetric")
    del model
    torch.cuda.empty_cache() if device == "cuda" else None
    
    # 4. Zipf Wave + FFT
    print("\n" + "="*60)
    print("4. ZIPF WAVE + FFT BANDS")
    print("="*60)
    model = ZipfWaveFFTModel(config)
    all_results["zipf_wave_fft"] = train_model(model, train_loader, val_loader, config, "ZipfWave+FFT")
    del model
    torch.cuda.empty_cache() if device == "cuda" else None
    
    # 5. Random Wave (ablation)
    print("\n" + "="*60)
    print("5. RANDOM WAVE (ABLATION)")
    print("="*60)
    model = RandomWaveModel(config)
    all_results["random_wave"] = train_model(model, train_loader, val_loader, config, "RandomWave")
    del model
    torch.cuda.empty_cache() if device == "cuda" else None
    
    return all_results


print("Experiment runner defined")

# ==============================================================================
# CELL 12: MAIN
# ==============================================================================

if __name__ == "__main__":
    results = run_experiments()
    
    # Print summary
    print("\n" + "="*70)
    print("EXPERIMENT 034 RESULTS: Zipf Wave Language Model")
    print("="*70)
    
    baseline_ppl = results["baseline"]["final_ppl"]
    
    print(f"\n{'Model':<35} {'Loss':>10} {'PPL':>10} {'vs Base':>12}")
    print("-"*70)
    
    display_names = {
        "baseline": "1. Baseline (Standard)",
        "zipf_wave_linear": "2. ZipfWave + Linear",
        "zipf_wave_symmetric": "3. ZipfWave + Symmetric",
        "zipf_wave_fft": "4. ZipfWave + FFT Bands",
        "random_wave": "5. RandomWave (ablation)"
    }
    
    for name, res in results.items():
        ppl = res["final_ppl"]
        loss = res["final_loss"]
        if name == "baseline":
            delta = "---"
        else:
            improvement = (baseline_ppl - ppl) / baseline_ppl * 100
            delta = f"{improvement:+.2f}%"
        
        display = display_names.get(name, name)
        print(f"{display:<35} {loss:>10.4f} {ppl:>10.2f} {delta:>12}")
    
    print("-"*70)
    
    # Key comparisons
    print("\n" + "="*70)
    print("KEY COMPARISONS")
    print("="*70)
    
    # Zipf vs Random (does Zipf grounding help?)
    zipf_ppl = results["zipf_wave_linear"]["final_ppl"]
    random_ppl = results["random_wave"]["final_ppl"]
    zipf_vs_random = (random_ppl - zipf_ppl) / random_ppl * 100
    print(f"\nZipf vs Random frequencies: {zipf_vs_random:+.2f}%")
    print(f"  -> {'Zipf grounding HELPS' if zipf_vs_random > 0 else 'No benefit from Zipf'}")
    
    # Symmetric vs Linear decode
    linear_ppl = results["zipf_wave_linear"]["final_ppl"]
    symmetric_ppl = results["zipf_wave_symmetric"]["final_ppl"]
    sym_vs_linear = (linear_ppl - symmetric_ppl) / linear_ppl * 100
    print(f"\nSymmetric vs Linear decode: {sym_vs_linear:+.2f}%")
    
    # FFT vs Linear (does spectral decomposition help?)
    fft_ppl = results["zipf_wave_fft"]["final_ppl"]
    fft_vs_linear = (linear_ppl - fft_ppl) / linear_ppl * 100
    print(f"\nFFT bands vs Linear: {fft_vs_linear:+.2f}%")
    
    # Find winner
    best_name = min(results.keys(), key=lambda k: results[k]["final_ppl"])
    best_ppl = results[best_name]["final_ppl"]
    
    print(f"\n>>> WINNER: {display_names.get(best_name, best_name)}")
    print(f"    Perplexity: {best_ppl:.2f}")
    
    if best_name != "baseline":
        improvement = (baseline_ppl - best_ppl) / baseline_ppl * 100
        print(f"    vs Baseline: {improvement:+.2f}%")
    
    print("\n" + "="*70)
    print("EXPERIMENT 034 COMPLETE")
    print("="*70)
