"""
EXPERIMENT 029 v2: Spectral Bands + History Attention Ablation Study
=====================================================================

Full ablation study testing:
1. BASELINE: Standard GPT-2
2. + HISTORY: Standard + uniform history attention
3. + SPECTRAL: Standard + spectral band decomposition (no history)
4. + SPECTRAL + UNIFORM HISTORY: Bands with same history depth
5. + SPECTRAL + VARIABLE HISTORY: Bands with Heisenberg-inspired depths
   (low freq = long history, high freq = short history)

This tests whether spectral decomposition + variable temporal windows
provides additional benefit beyond uniform history attention.

Run on Google Colab with GPU for best performance.

AKIRA Project - Experiment 029 v2
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
class AblationConfig:
    """Configuration for ablation study."""
    vocab_size: int = 50257          # GPT-2 vocabulary
    embed_dim: int = 256             # Small for speed
    num_layers: int = 6              # Moderate depth
    num_heads: int = 8               # Standard
    max_seq_length: int = 256        # Context window
    dropout: float = 0.1
    
    # Spectral band settings
    num_bands: int = 4               # Number of spectral bands
    band_dim: int = 64               # Dimension per band (embed_dim / num_bands)
    
    # History attention settings (uniform)
    max_history_uniform: int = 64    # Uniform history depth
    decay_rate: float = 0.95         # Temporal decay
    
    # Variable history depths per band (Heisenberg-inspired)
    # Low frequency bands see further back, high frequency see recent
    band_history_depths: List[int] = field(default_factory=lambda: [128, 64, 32, 16])
    
    # Training
    batch_size: int = 16
    learning_rate: float = 3e-4
    total_steps: int = 5000          # Adjust based on time
    eval_every: int = 500
    warmup_steps: int = 200

    def __post_init__(self):
        assert self.embed_dim == self.num_bands * self.band_dim, \
            f"embed_dim ({self.embed_dim}) must equal num_bands * band_dim ({self.num_bands * self.band_dim})"
        assert len(self.band_history_depths) == self.num_bands, \
            f"band_history_depths length ({len(self.band_history_depths)}) must equal num_bands ({self.num_bands})"

print("AblationConfig defined")

# ==============================================================================
# CELL 4: HISTORY BUFFER
# ==============================================================================

class HistoryBuffer(nn.Module):
    """Generic history buffer for storing past states."""
    
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
        """Add state to buffer. state: [B, P, D] or [P, D]"""
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
    
    def get_history(self) -> torch.Tensor:
        """Returns [P, T, D] for attention."""
        if self.buffer is None:
            return None
        valid = self.buffer[:self.current_length]
        return valid.transpose(0, 1)

print("HistoryBuffer defined")

# ==============================================================================
# CELL 5: CAUSAL SPECTRAL DECOMPOSER
# ==============================================================================

class CausalConv1d(nn.Module):
    """
    Causal 1D convolution - only sees past and current, never future.
    
    For language modeling, this is critical to avoid information leakage.
    """
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size
        # Causal padding: all padding on the left (past) side
        self.padding = kernel_size - 1
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, T] input
        Returns:
            [B, C, T] output (same length, causal)
        """
        # Pad only on the left (past)
        x_padded = F.pad(x, (self.padding, 0))  # [B, C, T + padding]
        out = self.conv(x_padded)  # [B, C, T]
        return out


class SpectralDecomposer(nn.Module):
    """
    Decomposes input into spectral bands using CAUSAL learnable filters.
    
    Each band captures different frequency content:
    - Band 0: Lowest frequencies (slow changes, global patterns)
    - Band N-1: Highest frequencies (fast changes, local details)
    
    IMPORTANT: Uses causal convolutions - position t only sees positions <= t.
    This prevents information leakage in language modeling.
    """
    
    def __init__(self, embed_dim: int, num_bands: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_bands = num_bands
        self.band_dim = embed_dim // num_bands
        
        # Learnable projection to bands
        self.to_bands = nn.Linear(embed_dim, embed_dim)
        
        # CAUSAL frequency filters (1D conv per band)
        # Different kernel sizes capture different frequencies
        self.band_filters = nn.ModuleList()
        self.kernel_sizes = []
        for i in range(num_bands):
            # Lower bands = larger kernels (capture low freq / longer context)
            # Higher bands = smaller kernels (capture high freq / local details)
            kernel_size = max(3, 2 * (num_bands - i) + 1)  # 9, 7, 5, 3 for 4 bands
            self.kernel_sizes.append(kernel_size)
            # Use CAUSAL convolution
            self.band_filters.append(
                CausalConv1d(self.band_dim, self.band_dim, kernel_size)
            )
        
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Decompose input into spectral bands (CAUSALLY).
        
        Args:
            x: [B, T, D] input
            
        Returns:
            List of [B, T, band_dim] tensors, one per band
        """
        B, T, D = x.shape
        
        # Project to band space
        h = self.to_bands(x)  # [B, T, D]
        
        # Split into bands
        bands = h.view(B, T, self.num_bands, self.band_dim)  # [B, T, num_bands, band_dim]
        
        # Apply CAUSAL frequency-specific filters to each band
        filtered_bands = []
        for i in range(self.num_bands):
            band = bands[:, :, i, :]  # [B, T, band_dim]
            band = band.transpose(1, 2)  # [B, band_dim, T] for conv1d
            band = self.band_filters[i](band)  # [B, band_dim, T] - CAUSAL
            band = band.transpose(1, 2)  # [B, T, band_dim]
            filtered_bands.append(band)
        
        return filtered_bands
    
    def reconstruct(self, bands: List[torch.Tensor]) -> torch.Tensor:
        """Reconstruct from bands."""
        # Stack and reshape
        stacked = torch.stack(bands, dim=2)  # [B, T, num_bands, band_dim]
        B, T, _, _ = stacked.shape
        reconstructed = stacked.view(B, T, -1)  # [B, T, D]
        return self.norm(reconstructed)

print("CausalConv1d defined")
print("SpectralDecomposer defined (with CAUSAL convolutions)")

# ==============================================================================
# CELL 6: HISTORY ATTENTION (for single band or full embedding)
# ==============================================================================

class HistoryAttention(nn.Module):
    """
    Per-position history attention.
    Each position attends to its OWN history across time.
    """
    
    def __init__(self, feature_dim: int, max_history: int, decay_rate: float):
        super().__init__()
        self.feature_dim = feature_dim
        self.max_history = max_history
        
        self.history_buffer = HistoryBuffer(max_history, feature_dim)
        
        self.q_proj = nn.Linear(feature_dim, feature_dim)
        self.k_proj = nn.Linear(feature_dim, feature_dim)
        self.v_proj = nn.Linear(feature_dim, feature_dim)
        self.out_proj = nn.Linear(feature_dim, feature_dim)
        
        decay_weights = torch.tensor([decay_rate ** i for i in range(max_history)])
        self.register_buffer('decay_weights', decay_weights)
    
    def forward(self, x: torch.Tensor, update_buffer: bool = True) -> torch.Tensor:
        """
        Args:
            x: [B, T, D] input
            update_buffer: whether to add current output to history
            
        Returns:
            [B, T, D] output
        """
        B, T, D = x.shape
        
        history = self.history_buffer.get_history()
        
        if history is None or history.shape[1] == 0:
            output = self.out_proj(x)
            if update_buffer:
                self.history_buffer.update(output.detach())
            return output
        
        Q = self.q_proj(x)  # [B, T, D]
        T_hist = history.shape[1]
        
        K = self.k_proj(history)  # [T, T_hist, D]
        V = self.v_proj(history)  # [T, T_hist, D]
        
        # Per-position attention
        Q_exp = Q.unsqueeze(2)  # [B, T, 1, D]
        K_exp = K.unsqueeze(0)  # [1, T, T_hist, D]
        V_exp = V.unsqueeze(0)  # [1, T, T_hist, D]
        
        scores = torch.matmul(Q_exp, K_exp.transpose(-2, -1)) / math.sqrt(D)  # [B, T, 1, T_hist]
        
        # Temporal decay (recent = higher weight)
        decay = self.decay_weights[:T_hist].flip(0).view(1, 1, 1, -1)
        scores = scores + torch.log(decay + 1e-10)
        
        attn = F.softmax(scores, dim=-1)
        attended = torch.matmul(attn, V_exp).squeeze(2)  # [B, T, D]
        
        output = self.out_proj(attended + x)
        
        if update_buffer:
            self.history_buffer.update(output.detach())
        
        return output
    
    def reset_history(self):
        self.history_buffer.reset()

print("HistoryAttention defined")

# ==============================================================================
# CELL 7: SPECTRAL BAND PROCESSOR
# ==============================================================================

class SpectralBandProcessor(nn.Module):
    """
    Processes spectral bands with optional per-band history attention.
    
    Can operate in three modes:
    - no_history: Just process bands, no temporal memory
    - uniform_history: All bands use same history depth
    - variable_history: Each band has its own history depth (Heisenberg-inspired)
    """
    
    def __init__(
        self,
        band_dim: int,
        num_bands: int,
        history_mode: str,  # "none", "uniform", "variable"
        uniform_history: int = 64,
        variable_depths: List[int] = None,
        decay_rate: float = 0.95
    ):
        super().__init__()
        self.band_dim = band_dim
        self.num_bands = num_bands
        self.history_mode = history_mode
        
        # Per-band attention (within each band, across positions)
        self.band_attns = nn.ModuleList([
            nn.MultiheadAttention(band_dim, num_heads=4, batch_first=True)
            for _ in range(num_bands)
        ])
        
        # Per-band FFN
        self.band_ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(band_dim, band_dim * 2),
                nn.GELU(),
                nn.Linear(band_dim * 2, band_dim)
            )
            for _ in range(num_bands)
        ])
        
        # Per-band norms
        self.band_norms1 = nn.ModuleList([nn.LayerNorm(band_dim) for _ in range(num_bands)])
        self.band_norms2 = nn.ModuleList([nn.LayerNorm(band_dim) for _ in range(num_bands)])
        
        # History attention per band (if enabled)
        self.band_history_attns = None
        if history_mode == "uniform":
            self.band_history_attns = nn.ModuleList([
                HistoryAttention(band_dim, uniform_history, decay_rate)
                for _ in range(num_bands)
            ])
            self.band_history_norms = nn.ModuleList([nn.LayerNorm(band_dim) for _ in range(num_bands)])
        elif history_mode == "variable":
            assert variable_depths is not None and len(variable_depths) == num_bands
            self.band_history_attns = nn.ModuleList([
                HistoryAttention(band_dim, variable_depths[i], decay_rate)
                for i in range(num_bands)
            ])
            self.band_history_norms = nn.ModuleList([nn.LayerNorm(band_dim) for _ in range(num_bands)])
    
    def forward(
        self,
        bands: List[torch.Tensor],
        update_history: bool = True,
        causal_mask: torch.Tensor = None
    ) -> List[torch.Tensor]:
        """
        Process each band independently.
        
        Args:
            bands: List of [B, T, band_dim] tensors
            update_history: whether to update history buffers
            causal_mask: [T, T] causal attention mask
            
        Returns:
            List of processed [B, T, band_dim] tensors
        """
        processed = []
        
        for i, band in enumerate(bands):
            # 1. History attention (if enabled)
            if self.band_history_attns is not None:
                band = band + self.band_history_attns[i](
                    self.band_history_norms[i](band),
                    update_buffer=update_history
                )
            
            # 2. Self-attention within band
            attn_out, _ = self.band_attns[i](
                self.band_norms1[i](band),
                self.band_norms1[i](band),
                self.band_norms1[i](band),
                attn_mask=causal_mask
            )
            band = band + attn_out
            
            # 3. FFN
            band = band + self.band_ffns[i](self.band_norms2[i](band))
            
            processed.append(band)
        
        return processed
    
    def reset_history(self):
        if self.band_history_attns is not None:
            for ha in self.band_history_attns:
                ha.reset_history()

print("SpectralBandProcessor defined")

# ==============================================================================
# CELL 8: STANDARD TRANSFORMER LAYER
# ==============================================================================

class TransformerLayer(nn.Module):
    """Standard GPT-2 style transformer layer."""
    
    def __init__(self, embed_dim: int, num_heads: int, dropout: float):
        super().__init__()
        
        self.self_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.register_buffer('causal_mask', None)
    
    def _get_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        if self.causal_mask is None or self.causal_mask.shape[0] < seq_len:
            mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
            self.causal_mask = mask
        return self.causal_mask[:seq_len, :seq_len]
    
    def forward(self, x: torch.Tensor, update_history: bool = True) -> torch.Tensor:
        B, T, D = x.shape
        mask = self._get_causal_mask(T, x.device)
        
        attn_out, _ = self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x), attn_mask=mask)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        
        return x
    
    def reset_history(self):
        pass  # No history in baseline

print("TransformerLayer defined")

# ==============================================================================
# CELL 9: TRANSFORMER LAYER WITH HISTORY (NO BANDS)
# ==============================================================================

class TransformerLayerWithHistory(nn.Module):
    """Transformer layer with history attention (no spectral bands)."""
    
    def __init__(self, embed_dim: int, num_heads: int, dropout: float, max_history: int, decay_rate: float):
        super().__init__()
        
        # History attention
        self.history_attn = HistoryAttention(embed_dim, max_history, decay_rate)
        self.norm0 = nn.LayerNorm(embed_dim)
        
        # Standard self-attention
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.register_buffer('causal_mask', None)
    
    def _get_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        if self.causal_mask is None or self.causal_mask.shape[0] < seq_len:
            mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
            self.causal_mask = mask
        return self.causal_mask[:seq_len, :seq_len]
    
    def forward(self, x: torch.Tensor, update_history: bool = True) -> torch.Tensor:
        B, T, D = x.shape
        
        # 1. History attention
        x = x + self.history_attn(self.norm0(x), update_buffer=update_history)
        
        # 2. Self-attention
        mask = self._get_causal_mask(T, x.device)
        attn_out, _ = self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x), attn_mask=mask)
        x = x + attn_out
        
        # 3. FFN
        x = x + self.ffn(self.norm2(x))
        
        return x
    
    def reset_history(self):
        self.history_attn.reset_history()

print("TransformerLayerWithHistory defined")

# ==============================================================================
# CELL 10: SPECTRAL TRANSFORMER LAYER
# ==============================================================================

class SpectralTransformerLayer(nn.Module):
    """
    Transformer layer with spectral band decomposition.
    
    Flow:
    1. Decompose into spectral bands
    2. Process each band (with optional per-band history)
    3. Reconstruct
    4. Standard self-attention on full embedding
    5. FFN
    """
    
    def __init__(
        self,
        config: AblationConfig,
        history_mode: str  # "none", "uniform", "variable"
    ):
        super().__init__()
        self.config = config
        self.history_mode = history_mode
        
        # Spectral decomposition
        self.decomposer = SpectralDecomposer(config.embed_dim, config.num_bands)
        
        # Band processor with optional history
        variable_depths = config.band_history_depths if history_mode == "variable" else None
        self.band_processor = SpectralBandProcessor(
            config.band_dim,
            config.num_bands,
            history_mode,
            uniform_history=config.max_history_uniform,
            variable_depths=variable_depths,
            decay_rate=config.decay_rate
        )
        
        # Post-reconstruction self-attention (full embedding)
        self.self_attn = nn.MultiheadAttention(
            config.embed_dim, config.num_heads,
            dropout=config.dropout, batch_first=True
        )
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(config.embed_dim, config.embed_dim * 4),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.embed_dim * 4, config.embed_dim),
            nn.Dropout(config.dropout)
        )
        
        self.norm1 = nn.LayerNorm(config.embed_dim)
        self.norm2 = nn.LayerNorm(config.embed_dim)
        self.register_buffer('causal_mask', None)
    
    def _get_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        if self.causal_mask is None or self.causal_mask.shape[0] < seq_len:
            mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
            self.causal_mask = mask
        return self.causal_mask[:seq_len, :seq_len]
    
    def forward(self, x: torch.Tensor, update_history: bool = True) -> torch.Tensor:
        B, T, D = x.shape
        mask = self._get_causal_mask(T, x.device)
        
        # 1. Decompose into spectral bands
        bands = self.decomposer(x)
        
        # 2. Process bands (with optional history)
        processed_bands = self.band_processor(bands, update_history=update_history, causal_mask=mask)
        
        # 3. Reconstruct
        reconstructed = self.decomposer.reconstruct(processed_bands)
        x = x + reconstructed  # Residual
        
        # 4. Full-embedding self-attention
        attn_out, _ = self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x), attn_mask=mask)
        x = x + attn_out
        
        # 5. FFN
        x = x + self.ffn(self.norm2(x))
        
        return x
    
    def reset_history(self):
        self.band_processor.reset_history()

print("SpectralTransformerLayer defined")

# ==============================================================================
# CELL 11: ABLATION MODEL
# ==============================================================================

class AblationLM(nn.Module):
    """
    Language model supporting all ablation configurations:
    - baseline: Standard transformer
    - history: Standard + history attention
    - spectral: Spectral bands, no history
    - spectral_uniform: Spectral + uniform history
    - spectral_variable: Spectral + variable history (AKIRA design)
    """
    
    def __init__(self, config: AblationConfig, mode: str):
        """
        Args:
            config: Model configuration
            mode: One of "baseline", "history", "spectral", "spectral_uniform", "spectral_variable"
        """
        super().__init__()
        self.config = config
        self.mode = mode
        
        # Embeddings
        self.token_embed = nn.Embedding(config.vocab_size, config.embed_dim)
        self.pos_embed = nn.Embedding(config.max_seq_length, config.embed_dim)
        self.dropout = nn.Dropout(config.dropout)
        
        # Build layers based on mode
        if mode == "baseline":
            self.layers = nn.ModuleList([
                TransformerLayer(config.embed_dim, config.num_heads, config.dropout)
                for _ in range(config.num_layers)
            ])
        elif mode == "history":
            self.layers = nn.ModuleList([
                TransformerLayerWithHistory(
                    config.embed_dim, config.num_heads, config.dropout,
                    config.max_history_uniform, config.decay_rate
                )
                for _ in range(config.num_layers)
            ])
        elif mode == "spectral":
            self.layers = nn.ModuleList([
                SpectralTransformerLayer(config, history_mode="none")
                for _ in range(config.num_layers)
            ])
        elif mode == "spectral_uniform":
            self.layers = nn.ModuleList([
                SpectralTransformerLayer(config, history_mode="uniform")
                for _ in range(config.num_layers)
            ])
        elif mode == "spectral_variable":
            self.layers = nn.ModuleList([
                SpectralTransformerLayer(config, history_mode="variable")
                for _ in range(config.num_layers)
            ])
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        # Output
        self.final_norm = nn.LayerNorm(config.embed_dim)
        self.lm_head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)
        
        # Initialize
        self.apply(self._init_weights)
        
        # Count parameters
        self.num_params = sum(p.numel() for p in self.parameters())
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        update_history: bool = True
    ) -> Dict[str, torch.Tensor]:
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

print("AblationLM defined")

# ==============================================================================
# CELL 12: DATA LOADING
# ==============================================================================

def load_wikitext2(tokenizer, max_length: int = 256, split: str = "train"):
    """Load WikiText-2 dataset."""
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
    
    tokenized = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    
    return tokenized

print("Data loading function defined")

# ==============================================================================
# CELL 13: TRAINING FUNCTION
# ==============================================================================

def train_model(
    model: AblationLM,
    train_loader: DataLoader,
    eval_loader: DataLoader,
    config: AblationConfig,
    model_name: str
) -> Dict[str, List[float]]:
    """Train a model and return metrics."""
    
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.total_steps, eta_min=1e-6)
    
    if device == "cuda":
        scaler = torch.amp.GradScaler('cuda')
    else:
        scaler = None
    
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
    
    total_time = time.time() - start_time
    print(f"  Training complete in {total_time:.1f}s")
    
    return history

print("Training function defined")

# ==============================================================================
# CELL 14: ABLATION CONFIGURATIONS
# ==============================================================================

# Define which ablations to run (set to True/False)
ABLATIONS = {
    "baseline": True,           # Standard GPT-2
    "history": True,            # + History attention (uniform depth)
    "spectral": True,           # + Spectral bands (no history)
    "spectral_uniform": True,   # + Spectral + uniform history
    "spectral_variable": True,  # + Spectral + variable history (AKIRA)
}

ABLATION_NAMES = {
    "baseline": "1. Baseline (Standard GPT-2)",
    "history": "2. + History (Uniform)",
    "spectral": "3. + Spectral Bands (No History)",
    "spectral_uniform": "4. + Spectral + Uniform History",
    "spectral_variable": "5. + Spectral + Variable History (AKIRA)",
}

print("Ablation configurations defined")
print(f"Running: {[k for k, v in ABLATIONS.items() if v]}")

# ==============================================================================
# CELL 15: MAIN EXPERIMENT
# ==============================================================================

def run_ablation_study():
    """Run full ablation study."""
    
    print("\n" + "="*70)
    print("EXPERIMENT 029 v2: Spectral Bands + History Attention Ablation")
    print("="*70)
    
    # Load tokenizer
    from transformers import GPT2Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Configuration
    config = AblationConfig()
    
    print(f"\nConfiguration:")
    print(f"  embed_dim: {config.embed_dim}")
    print(f"  num_layers: {config.num_layers}")
    print(f"  num_bands: {config.num_bands}")
    print(f"  band_history_depths: {config.band_history_depths}")
    print(f"  total_steps: {config.total_steps}")
    
    # Load data
    train_dataset = load_wikitext2(tokenizer, max_length=config.max_seq_length, split="train")
    eval_dataset = load_wikitext2(tokenizer, max_length=config.max_seq_length, split="validation")
    
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size,
        shuffle=True, num_workers=2, pin_memory=True
    )
    eval_loader = DataLoader(
        eval_dataset, batch_size=config.batch_size,
        shuffle=False, num_workers=2, pin_memory=True
    )
    
    print(f"\nDataset:")
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Eval samples: {len(eval_dataset)}")
    
    results = {}
    
    # Run each ablation
    for mode, enabled in ABLATIONS.items():
        if not enabled:
            continue
        
        print("\n" + "-"*60)
        print(f"{ABLATION_NAMES[mode]}")
        print("-"*60)
        
        model = AblationLM(config, mode=mode)
        history = train_model(model, train_loader, eval_loader, config, ABLATION_NAMES[mode])
        results[mode] = history
        
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
    
    # ==================================
    # RESULTS COMPARISON
    # ==================================
    print("\n" + "="*70)
    print("ABLATION RESULTS COMPARISON")
    print("="*70)
    
    print(f"\n{'Model':<45} {'Loss':>10} {'PPL':>10} {'vs Base':>12}")
    print("-" * 77)
    
    baseline_ppl = results.get('baseline', {}).get('eval_ppl', [None])[-1]
    
    sorted_results = []
    for mode in ["baseline", "history", "spectral", "spectral_uniform", "spectral_variable"]:
        if mode in results:
            loss = results[mode]['eval_loss'][-1]
            ppl = results[mode]['eval_ppl'][-1]
            if baseline_ppl:
                improvement = (baseline_ppl - ppl) / baseline_ppl * 100
            else:
                improvement = 0
            sorted_results.append((mode, loss, ppl, improvement))
            print(f"{ABLATION_NAMES[mode]:<45} {loss:>10.4f} {ppl:>10.2f} {improvement:>+11.2f}%")
    
    # Find winner
    print("\n" + "-"*70)
    if sorted_results:
        winner = min(sorted_results, key=lambda x: x[2])
        print(f">>> WINNER: {ABLATION_NAMES[winner[0]]}")
        print(f"    Perplexity: {winner[2]:.2f}")
        if baseline_ppl:
            print(f"    Improvement over baseline: {winner[3]:+.2f}%")
        
        # Check if spectral+variable (AKIRA) is the best
        if winner[0] == "spectral_variable":
            print("\n>>> AKIRA DESIGN VALIDATED")
            print("    Spectral bands + variable history depths wins!")
        elif winner[0] == "spectral_uniform":
            print("\n>>> Spectral bands help, but variable depths don't add benefit")
        elif winner[0] == "spectral":
            print("\n>>> Spectral bands help, history doesn't add benefit")
        elif winner[0] == "history":
            print("\n>>> History helps, but spectral decomposition doesn't add benefit")
    print("-"*70)
    
    print("\n" + "="*70)
    print("EXPERIMENT 029 v2 COMPLETE")
    print("="*70)
    
    return results

# ==============================================================================
# CELL 16: RUN EXPERIMENT
# ==============================================================================

if __name__ == "__main__":
    results = run_ablation_study()
