# -*- coding: utf-8 -*-
"""
Zipf-Grounded Wave Tokenizer

Maps discrete tokens to wave representations based on Zipf's Law.
Token usage frequency determines wave frequency:
- Common tokens (the, a, is) -> LOW wave frequency (DC-like)
- Rare tokens (technical terms) -> HIGH wave frequency

This grounds spectral decomposition in information theory rather than
arbitrary learned parameters.

Theory:
- Zipf's Law: f(r) ~ r^(-alpha) where r is rank
- Shannon entropy: common words carry fewer bits
- Mapping: wave_freq = log(rank) / log(vocab_size)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import json
from pathlib import Path


@dataclass
class ZipfWaveConfig:
    """Configuration for Zipf-grounded wave tokenizer."""
    
    # Tokenizer
    vocab_size: int = 50257  # GPT-2 vocab size
    
    # Wave parameters
    embed_dim: int = 512  # Output embedding dimension
    n_harmonics: int = 8  # Number of harmonic frequencies per token
    freq_min: float = 0.01  # Minimum frequency (for most common tokens)
    freq_max: float = 1.0  # Maximum frequency (for rarest tokens)
    
    # Phase options
    phase_mode: str = "learnable"  # "learnable", "position", "fixed"
    
    # Amplitude options  
    amplitude_mode: str = "fixed"  # "fixed", "learnable", "zipf_scaled"
    
    # Band structure (for analysis)
    num_bands: int = 7
    
    def validate(self) -> bool:
        assert self.vocab_size > 0, "vocab_size must be positive"
        assert self.embed_dim > 0, "embed_dim must be positive"
        assert self.n_harmonics > 0, "n_harmonics must be positive"
        assert 0 < self.freq_min < self.freq_max, "freq_min must be less than freq_max"
        assert self.phase_mode in ["learnable", "position", "fixed"]
        assert self.amplitude_mode in ["fixed", "learnable", "zipf_scaled"]
        return True


class ZipfRankTable:
    """
    Pre-computed Zipf ranks for a vocabulary.
    
    Can be initialized from:
    1. GPT-2 tokenizer (uses BPE merge order as proxy for frequency)
    2. Corpus statistics (actual frequency counts)
    3. Custom rank mapping
    """
    
    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size
        self._ranks: Optional[torch.Tensor] = None
        self._frequencies: Optional[torch.Tensor] = None
    
    @classmethod
    def from_gpt2(cls, vocab_size: int = 50257) -> "ZipfRankTable":
        """
        Create rank table from GPT-2 tokenizer.
        
        GPT-2 BPE tokens are ordered roughly by frequency:
        - Lower token IDs = more common (earlier in BPE merges)
        - Higher token IDs = less common (later merges or rare)
        
        This is an approximation but captures the Zipf structure.
        """
        table = cls(vocab_size)
        
        # GPT-2 token IDs roughly correspond to frequency rank
        # Token 0-255 are byte tokens (very common)
        # Subsequent tokens are BPE merges in order of frequency
        # This gives us a natural Zipf-like ordering
        
        # Create ranks: lower token_id -> lower rank (more common)
        # We add 1 to avoid log(0) issues
        ranks = torch.arange(1, vocab_size + 1, dtype=torch.float32)
        
        # Adjust for the fact that early tokens (bytes) are all common
        # Smooth the transition from bytes to BPE tokens
        byte_tokens = 256
        ranks[:byte_tokens] = torch.linspace(1, byte_tokens, byte_tokens)
        
        table._ranks = ranks
        return table
    
    @classmethod
    def from_corpus_counts(cls, token_counts: Dict[int, int], vocab_size: int) -> "ZipfRankTable":
        """
        Create rank table from actual corpus frequency counts.
        
        Args:
            token_counts: Dict mapping token_id -> count
            vocab_size: Total vocabulary size
        """
        table = cls(vocab_size)
        
        # Sort tokens by count (descending)
        sorted_tokens = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Assign ranks
        ranks = torch.ones(vocab_size, dtype=torch.float32) * vocab_size  # Default: rarest
        for rank, (token_id, count) in enumerate(sorted_tokens, start=1):
            if token_id < vocab_size:
                ranks[token_id] = rank
        
        # Store frequencies for potential use
        frequencies = torch.zeros(vocab_size, dtype=torch.float32)
        for token_id, count in token_counts.items():
            if token_id < vocab_size:
                frequencies[token_id] = count
        
        table._ranks = ranks
        table._frequencies = frequencies
        return table
    
    @classmethod
    def from_file(cls, path: Union[str, Path]) -> "ZipfRankTable":
        """Load rank table from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        table = cls(data['vocab_size'])
        table._ranks = torch.tensor(data['ranks'], dtype=torch.float32)
        if 'frequencies' in data:
            table._frequencies = torch.tensor(data['frequencies'], dtype=torch.float32)
        return table
    
    def save(self, path: Union[str, Path]) -> None:
        """Save rank table to JSON file."""
        data = {
            'vocab_size': self.vocab_size,
            'ranks': self._ranks.tolist()
        }
        if self._frequencies is not None:
            data['frequencies'] = self._frequencies.tolist()
        
        with open(path, 'w') as f:
            json.dump(data, f)
    
    @property
    def ranks(self) -> torch.Tensor:
        """Get rank tensor [vocab_size]."""
        if self._ranks is None:
            raise ValueError("Ranks not initialized. Use from_gpt2() or from_corpus_counts().")
        return self._ranks
    
    def get_ranks(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Get ranks for token IDs.
        
        Args:
            token_ids: [*] tensor of token IDs
        Returns:
            [*] tensor of ranks (same shape)
        """
        ranks = self.ranks.to(token_ids.device)
        return ranks[token_ids]
    
    def get_normalized_ranks(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Get ranks normalized to [0, 1].
        
        Args:
            token_ids: [*] tensor of token IDs
        Returns:
            [*] tensor of normalized ranks (0 = most common, 1 = rarest)
        """
        ranks = self.get_ranks(token_ids)
        return (ranks - 1) / (self.vocab_size - 1)
    
    def get_band_indices(self, num_bands: int = 7) -> Dict[int, Tuple[int, int]]:
        """
        Get token ID ranges for each spectral band based on Zipf distribution.
        
        Uses logarithmic spacing to account for power law.
        
        Returns:
            Dict mapping band_index -> (start_rank, end_rank)
        """
        # Log-spaced boundaries for Zipf distribution
        boundaries = torch.logspace(0, math.log10(self.vocab_size), num_bands + 1)
        boundaries = boundaries.long()
        boundaries[0] = 1
        boundaries[-1] = self.vocab_size
        
        bands = {}
        for i in range(num_bands):
            bands[i] = (boundaries[i].item(), boundaries[i + 1].item())
        
        return bands


class ZipfWaveEmbedding(nn.Module):
    """
    Zipf-grounded wave embedding layer.
    
    Maps token IDs to wave representations where:
    - Wave frequency is determined by Zipf rank (common = low freq, rare = high freq)
    - Phase can be learned, fixed, or position-dependent
    - Amplitude can be fixed, learned, or Zipf-scaled
    
    Each token generates a superposition of harmonics:
        embedding = sum_k A_k * sin(2*pi*f_k*t + phi_k)
    
    where f_k = base_freq * (k+1) for harmonic k.
    """
    
    def __init__(self, config: ZipfWaveConfig, rank_table: Optional[ZipfRankTable] = None):
        super().__init__()
        self.config = config
        config.validate()
        
        # Initialize rank table
        if rank_table is None:
            rank_table = ZipfRankTable.from_gpt2(config.vocab_size)
        self.register_buffer('zipf_ranks', rank_table.ranks)
        
        # Compute base frequencies from Zipf ranks
        # log scale: freq = freq_min + (freq_max - freq_min) * log(rank) / log(V)
        log_ranks = torch.log(rank_table.ranks)
        log_vocab = math.log(config.vocab_size)
        normalized_log_ranks = log_ranks / log_vocab  # [0, 1]
        
        base_frequencies = config.freq_min + (config.freq_max - config.freq_min) * normalized_log_ranks
        self.register_buffer('base_frequencies', base_frequencies)  # [V]
        
        # Harmonic multipliers: 1, 2, 3, ... n_harmonics
        harmonics = torch.arange(1, config.n_harmonics + 1, dtype=torch.float32)
        self.register_buffer('harmonics', harmonics)  # [H]
        
        # Phase parameters
        if config.phase_mode == "learnable":
            # Learnable phase per token per harmonic
            self.phases = nn.Parameter(
                torch.randn(config.vocab_size, config.n_harmonics) * 0.1
            )  # [V, H]
        elif config.phase_mode == "fixed":
            # Fixed random phases (not trained)
            phases = torch.randn(config.vocab_size, config.n_harmonics) * 2 * math.pi
            self.register_buffer('phases', phases)
        else:  # position
            # Phase derived from position (no per-token phase)
            self.phases = None
        
        # Amplitude parameters
        if config.amplitude_mode == "learnable":
            # Learnable amplitude per token per harmonic
            self.amplitudes = nn.Parameter(
                torch.ones(config.vocab_size, config.n_harmonics)
            )  # [V, H]
        elif config.amplitude_mode == "zipf_scaled":
            # Amplitude inversely proportional to frequency (1/f spectrum)
            # Common tokens get higher amplitude, rare tokens lower
            inv_log_ranks = 1.0 - normalized_log_ranks  # High for common, low for rare
            amplitudes = inv_log_ranks.unsqueeze(1).expand(-1, config.n_harmonics)
            # Also decay with harmonic number (1/k)
            harmonic_decay = 1.0 / harmonics
            amplitudes = amplitudes * harmonic_decay.unsqueeze(0)
            self.register_buffer('amplitudes', amplitudes)
        else:  # fixed
            # Equal amplitude with harmonic decay
            harmonic_decay = 1.0 / harmonics  # [H]
            amplitudes = harmonic_decay.unsqueeze(0).expand(config.vocab_size, -1)
            self.register_buffer('amplitudes', amplitudes)
        
        # Projection from wave components to embedding dimension
        # Wave gives us n_harmonics * 2 components (sin + cos)
        wave_dim = config.n_harmonics * 2
        self.proj = nn.Linear(wave_dim, config.embed_dim)
        
        # Optional: learnable token-specific residual (captures non-wave info)
        self.use_residual = True
        if self.use_residual:
            self.residual = nn.Parameter(
                torch.zeros(config.vocab_size, config.embed_dim)
            )
    
    def get_frequencies(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Get wave frequencies for tokens.
        
        Args:
            token_ids: [B, T] token indices
        Returns:
            [B, T, H] frequencies for each harmonic
        """
        # Base frequency from Zipf rank: [B, T]
        base_freq = self.base_frequencies[token_ids]
        
        # Expand to harmonics: [B, T, H]
        frequencies = base_freq.unsqueeze(-1) * self.harmonics.unsqueeze(0).unsqueeze(0)
        
        return frequencies
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Convert token IDs to wave embeddings.
        
        Args:
            token_ids: [B, T] tensor of token indices
        Returns:
            [B, T, embed_dim] wave-based embeddings
        """
        B, T = token_ids.shape
        device = token_ids.device
        
        # Get frequencies: [B, T, H]
        frequencies = self.get_frequencies(token_ids)
        
        # Get phases: [B, T, H]
        if self.config.phase_mode == "position":
            # Phase from position: phi = 2*pi*position/T
            positions = torch.arange(T, device=device, dtype=torch.float32)
            phases = 2 * math.pi * positions.view(1, T, 1) / T
            phases = phases.expand(B, -1, self.config.n_harmonics)
        else:
            # Per-token phases
            phases = self.phases[token_ids]  # [B, T, H]
        
        # Get amplitudes: [B, T, H]
        if isinstance(self.amplitudes, nn.Parameter):
            amplitudes = self.amplitudes[token_ids]
        else:
            amplitudes = self.amplitudes[token_ids]
        
        # Generate time points: [T]
        t = torch.arange(T, device=device, dtype=torch.float32)
        
        # Compute wave argument: theta = 2*pi*f*t + phi
        # frequencies: [B, T, H], t: [T] -> need [B, T, H]
        t_expanded = t.view(1, T, 1)  # [1, T, 1]
        theta = 2 * math.pi * frequencies * t_expanded + phases  # [B, T, H]
        
        # Generate wave components
        sin_component = amplitudes * torch.sin(theta)  # [B, T, H]
        cos_component = amplitudes * torch.cos(theta)  # [B, T, H]
        
        # Concatenate sin and cos: [B, T, 2H]
        wave_repr = torch.cat([sin_component, cos_component], dim=-1)
        
        # Project to embedding dimension: [B, T, D]
        embedded = self.proj(wave_repr)
        
        # Add residual if enabled
        if self.use_residual:
            residual = self.residual[token_ids]  # [B, T, D]
            embedded = embedded + residual
        
        return embedded
    
    def get_wave_spectrum(self, token_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Analyze the spectral content of token embeddings.
        
        Returns dict with:
        - frequencies: [B, T, H] per-harmonic frequencies
        - amplitudes: [B, T, H] per-harmonic amplitudes  
        - phases: [B, T, H] per-harmonic phases
        - band_assignments: [B, T] which band each token belongs to
        """
        B, T = token_ids.shape
        
        frequencies = self.get_frequencies(token_ids)
        
        if self.config.phase_mode == "position":
            positions = torch.arange(T, device=token_ids.device, dtype=torch.float32)
            phases = 2 * math.pi * positions.view(1, T, 1) / T
            phases = phases.expand(B, -1, self.config.n_harmonics)
        else:
            phases = self.phases[token_ids]
        
        if isinstance(self.amplitudes, nn.Parameter):
            amplitudes = self.amplitudes[token_ids]
        else:
            amplitudes = self.amplitudes[token_ids]
        
        # Compute band assignments based on base frequency
        base_freq = self.base_frequencies[token_ids]  # [B, T]
        freq_normalized = (base_freq - self.config.freq_min) / (self.config.freq_max - self.config.freq_min)
        band_assignments = (freq_normalized * self.config.num_bands).long().clamp(0, self.config.num_bands - 1)
        
        return {
            'frequencies': frequencies,
            'amplitudes': amplitudes,
            'phases': phases,
            'band_assignments': band_assignments,
            'base_frequencies': base_freq
        }


class ZipfWaveDecoder(nn.Module):
    """
    Symmetric wave decoder: match output to token wave signatures.
    
    Instead of linear projection to vocab, we:
    1. Project hidden state to wave space
    2. Compare to all token wave signatures
    3. Return similarity as logits
    
    This enforces symmetry between encoding and decoding.
    """
    
    def __init__(self, config: ZipfWaveConfig, wave_embedding: ZipfWaveEmbedding):
        super().__init__()
        self.config = config
        self.wave_embedding = wave_embedding
        
        # Project hidden state to wave space
        wave_dim = config.n_harmonics * 2
        self.hidden_to_wave = nn.Linear(config.embed_dim, wave_dim)
        
        # Learnable temperature for similarity
        self.temperature = nn.Parameter(torch.ones(1))
        
        # Precompute reference signatures for all tokens
        self._cached_signatures: Optional[torch.Tensor] = None
    
    def _get_reference_signatures(self, device: torch.device) -> torch.Tensor:
        """
        Get wave signatures for all tokens (computed at t=0).
        
        Returns: [V, 2H] tensor of reference signatures
        """
        if self._cached_signatures is not None and self._cached_signatures.device == device:
            return self._cached_signatures
        
        V = self.config.vocab_size
        H = self.config.n_harmonics
        
        # Get phases for all tokens: [V, H]
        if self.config.phase_mode == "position":
            # At t=0, position-based phase is 0
            phases = torch.zeros(V, H, device=device)
        else:
            phases = self.wave_embedding.phases.to(device)
        
        # Get amplitudes: [V, H]
        if isinstance(self.wave_embedding.amplitudes, nn.Parameter):
            amplitudes = self.wave_embedding.amplitudes.to(device)
        else:
            amplitudes = self.wave_embedding.amplitudes.to(device)
        
        # Reference signature at t=0: just phase determines the wave shape
        sin_comp = amplitudes * torch.sin(phases)  # [V, H]
        cos_comp = amplitudes * torch.cos(phases)  # [V, H]
        
        signatures = torch.cat([sin_comp, cos_comp], dim=-1)  # [V, 2H]
        
        self._cached_signatures = signatures
        return signatures
    
    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """
        Decode hidden state to vocabulary logits via wave matching.
        
        Args:
            hidden_state: [B, T, D] model output
        Returns:
            [B, T, V] logits
        """
        B, T, D = hidden_state.shape
        device = hidden_state.device
        
        # Project to wave space: [B, T, 2H]
        output_wave = self.hidden_to_wave(hidden_state)
        
        # Get reference signatures: [V, 2H]
        ref_signatures = self._get_reference_signatures(device)
        
        # Normalize for cosine similarity
        output_norm = F.normalize(output_wave, dim=-1)  # [B, T, 2H]
        ref_norm = F.normalize(ref_signatures, dim=-1)  # [V, 2H]
        
        # Compute similarity: [B, T, V]
        logits = torch.matmul(output_norm, ref_norm.T) / self.temperature.abs().clamp(min=0.01)
        
        return logits
    
    def invalidate_cache(self):
        """Clear cached signatures (call after training step if phases are learnable)."""
        self._cached_signatures = None


# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def compute_zipf_ranks_from_tokenizer(tokenizer_name: str = "gpt2") -> ZipfRankTable:
    """
    Compute Zipf ranks from a HuggingFace tokenizer.
    
    For GPT-2, token IDs roughly correspond to BPE merge order,
    which correlates with frequency.
    """
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        vocab_size = tokenizer.vocab_size
    except ImportError:
        print("transformers not installed, using default GPT-2 vocab size")
        vocab_size = 50257
    
    return ZipfRankTable.from_gpt2(vocab_size)


def compute_zipf_ranks_from_corpus(
    texts: List[str],
    tokenizer_name: str = "gpt2",
    min_count: int = 1
) -> ZipfRankTable:
    """
    Compute actual Zipf ranks from a corpus of texts.
    
    Args:
        texts: List of text strings
        tokenizer_name: HuggingFace tokenizer to use
        min_count: Minimum count to include a token
    
    Returns:
        ZipfRankTable with corpus-derived ranks
    """
    from transformers import AutoTokenizer
    from collections import Counter
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    # Count all tokens
    token_counts: Counter = Counter()
    for text in texts:
        tokens = tokenizer.encode(text)
        token_counts.update(tokens)
    
    # Filter by min count
    filtered_counts = {k: v for k, v in token_counts.items() if v >= min_count}
    
    return ZipfRankTable.from_corpus_counts(filtered_counts, tokenizer.vocab_size)


def analyze_sentence_spectrum(
    sentence: str,
    wave_embedding: ZipfWaveEmbedding,
    tokenizer_name: str = "gpt2"
) -> Dict[str, torch.Tensor]:
    """
    Analyze the spectral content of a sentence.
    
    Returns dict with per-token spectral information and aggregate statistics.
    """
    from transformers import AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    token_ids = torch.tensor([tokenizer.encode(sentence)])  # [1, T]
    
    # Get wave spectrum
    spectrum = wave_embedding.get_wave_spectrum(token_ids)
    
    # Add token strings for reference
    tokens = tokenizer.convert_ids_to_tokens(token_ids[0].tolist())
    
    # Compute band distribution
    band_counts = torch.bincount(
        spectrum['band_assignments'].flatten(),
        minlength=wave_embedding.config.num_bands
    )
    
    return {
        **spectrum,
        'tokens': tokens,
        'band_distribution': band_counts,
        'mean_frequency': spectrum['base_frequencies'].mean(),
        'frequency_std': spectrum['base_frequencies'].std()
    }


# ==============================================================================
# QUICK TEST
# ==============================================================================

if __name__ == "__main__":
    print("Testing ZipfWaveTokenizer...")
    
    # Create config and embedding
    config = ZipfWaveConfig(
        vocab_size=50257,
        embed_dim=512,
        n_harmonics=8,
        phase_mode="learnable",
        amplitude_mode="zipf_scaled"
    )
    
    wave_embed = ZipfWaveEmbedding(config)
    
    # Test forward pass
    batch_size, seq_len = 2, 16
    token_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    embeddings = wave_embed(token_ids)
    print(f"Input shape: {token_ids.shape}")
    print(f"Output shape: {embeddings.shape}")
    
    # Test spectrum analysis
    spectrum = wave_embed.get_wave_spectrum(token_ids)
    print(f"Frequencies shape: {spectrum['frequencies'].shape}")
    print(f"Band assignments shape: {spectrum['band_assignments'].shape}")
    
    # Test decoder
    decoder = ZipfWaveDecoder(config, wave_embed)
    logits = decoder(embeddings)
    print(f"Logits shape: {logits.shape}")
    
    # Show frequency distribution
    print("\nFrequency statistics:")
    print(f"  Min base freq: {spectrum['base_frequencies'].min():.4f}")
    print(f"  Max base freq: {spectrum['base_frequencies'].max():.4f}")
    print(f"  Mean base freq: {spectrum['base_frequencies'].mean():.4f}")
    
    # Show band distribution for sample
    band_counts = torch.bincount(
        spectrum['band_assignments'].flatten(),
        minlength=config.num_bands
    )
    print(f"\nBand distribution: {band_counts.tolist()}")
    
    print("\nZipfWaveTokenizer test complete!")
