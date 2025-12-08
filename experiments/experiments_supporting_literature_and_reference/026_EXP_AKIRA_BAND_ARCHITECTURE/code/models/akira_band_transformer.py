"""
AKIRA Band Transformer Implementation
=====================================

This implements the core AKIRA architectural innovation: explicit spectral band structure
with differential learning rates.

Key differences from standard transformer:
1. Single MLP replaced with 7 parallel band MLPs
2. Each band has its own learning rate (slow for low-freq, fast for high-freq)
3. Wormhole attention enables cross-band communication
4. Band dimensions follow AKIRA's pattern (larger for low-freq)

AKIRA Project - Experiment 026
Oscar Goldman - Shogu Research Group @ Datamutant.ai
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass


@dataclass
class AKIRAConfig:
    """
    Configuration for AKIRA Band Transformer.
    
    Band structure follows AKIRA's 7-band architecture:
    - Band 0: DC (lowest frequency, most stable features)
    - Band 1-2: Low frequency
    - Band 3: Mid frequency
    - Band 4-5: High frequency
    - Band 6: Highest frequency (most adaptive features)
    """
    # Model dimensions
    vocab_size: int = 50257  # GPT-2 vocab
    num_layers: int = 6
    num_heads: int = 8
    max_seq_length: int = 512
    dropout: float = 0.1
    
    # Band configuration (7 bands)
    num_bands: int = 7
    band_dims: Tuple[int, ...] = (128, 96, 80, 64, 64, 48, 32)  # Sum = 512
    
    # Learning rates per band (from slow to fast)
    band_learning_rates: Tuple[float, ...] = (1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2)
    
    # MLP expansion factor
    mlp_expansion: int = 4
    
    # Wormhole attention settings
    use_wormhole: bool = True
    wormhole_threshold: float = 0.5  # Activation threshold for cross-band communication
    
    @property
    def hidden_dim(self) -> int:
        """Total hidden dimension (sum of all bands)."""
        return sum(self.band_dims)
    
    def validate(self) -> bool:
        """Validate configuration consistency."""
        assert len(self.band_dims) == self.num_bands, "band_dims must match num_bands"
        assert len(self.band_learning_rates) == self.num_bands, "band_learning_rates must match num_bands"
        assert self.hidden_dim % self.num_heads == 0, "hidden_dim must be divisible by num_heads"
        return True


class BandMLP(nn.Module):
    """
    MLP for a single frequency band.
    
    Each band has its own MLP with dimension specific to that band.
    Low-frequency bands are larger (more capacity for stable features).
    High-frequency bands are smaller (less capacity, more adaptive).
    """
    
    def __init__(self, band_dim: int, expansion: int = 4, dropout: float = 0.1):
        super().__init__()
        
        self.band_dim = band_dim
        hidden_dim = band_dim * expansion
        
        self.fc1 = nn.Linear(band_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, band_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through band MLP.
        
        Args:
            x: Input tensor of shape [batch, seq, band_dim]
        
        Returns:
            Output tensor of shape [batch, seq, band_dim]
        """
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class WormholeAttention(nn.Module):
    """
    Wormhole attention for cross-band communication.
    
    Allows information to "jump" between frequency bands when needed.
    This implements AKIRA's wormhole attention concept: when high-frequency
    patterns need to inform low-frequency representations (or vice versa),
    wormhole attention enables this communication.
    
    The attention is sparse and gated - it only activates when the
    cross-band relevance exceeds a threshold.
    
    Uses a COMMON projection dimension for all cross-band queries/keys
    to enable attention between bands of different sizes.
    """
    
    def __init__(self, band_dims: Tuple[int, ...], threshold: float = 0.5):
        super().__init__()
        
        self.band_dims = band_dims
        self.num_bands = len(band_dims)
        self.threshold = threshold
        
        # Common dimension for cross-band attention (use smallest band // 2)
        self.cross_dim = min(band_dims) // 2  # = 16 for default config
        
        # Projection matrices for cross-band queries and keys
        # All project to the SAME cross_dim for compatibility
        self.cross_queries = nn.ModuleList([
            nn.Linear(dim, self.cross_dim) for dim in band_dims
        ])
        self.cross_keys = nn.ModuleList([
            nn.Linear(dim, self.cross_dim) for dim in band_dims
        ])
        
        # Values project to target band dimension (for output)
        # We need value projections from each source to each target
        # Simplified: project to common dim, then expand per-target
        self.cross_values = nn.ModuleList([
            nn.Linear(dim, self.cross_dim * 2) for dim in band_dims
        ])
        
        # Output projection: from common value dim to target band dim
        self.out_projs = nn.ModuleList([
            nn.Linear(self.cross_dim * 2, dim) for dim in band_dims
        ])
        
        # Gate to control cross-band information flow
        self.gate = nn.ModuleList([
            nn.Linear(dim + dim, dim) for dim in band_dims  # band + projected output
        ])
    
    def forward(
        self, 
        bands: List[torch.Tensor],
        mask: Optional[torch.Tensor] = None
    ) -> List[torch.Tensor]:
        """
        Apply wormhole attention across bands.
        
        Args:
            bands: List of tensors, one per band, each [batch, seq, band_dim]
            mask: Optional attention mask
        
        Returns:
            List of updated band tensors
        """
        batch_size, seq_len = bands[0].shape[:2]
        updated_bands = []
        
        for i, band in enumerate(bands):
            # Compute cross-band attention for this band
            cross_info = torch.zeros(batch_size, seq_len, self.band_dims[i], device=band.device)
            
            for j, other_band in enumerate(bands):
                if i == j:
                    continue  # Skip self
                
                # Query from current band, key/value from other band
                # All use common cross_dim for compatibility
                q = self.cross_queries[i](band)  # [batch, seq, cross_dim]
                k = self.cross_keys[j](other_band)  # [batch, seq, cross_dim]
                v = self.cross_values[j](other_band)  # [batch, seq, cross_dim*2]
                
                # Attention scores
                scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.cross_dim)
                
                if mask is not None:
                    scores = scores.masked_fill(mask == 0, -1e9)
                
                attn = F.softmax(scores, dim=-1)
                
                # Apply attention to values
                cross_output = torch.matmul(attn, v)  # [batch, seq, cross_dim*2]
                
                # Project to target band dimension
                cross_output = self.out_projs[i](cross_output)  # [batch, seq, band_dim[i]]
                
                # Apply threshold gate
                gate_input = torch.cat([band, cross_output], dim=-1)
                gate_value = torch.sigmoid(self.gate[i](gate_input))
                
                # Only allow cross-band info if gate exceeds threshold
                gate_value = (gate_value > self.threshold).float() * gate_value
                
                cross_info = cross_info + gate_value * cross_output
            
            # Residual connection with cross-band info
            updated_bands.append(band + cross_info / (self.num_bands - 1))
        
        return updated_bands


class AKIRAMultiHeadAttention(nn.Module):
    """
    Multi-head attention that operates on concatenated bands.
    
    Standard attention but aware of band structure for potential
    band-specific attention patterns.
    """
    
    def __init__(self, config: AKIRAConfig):
        super().__init__()
        
        self.num_heads = config.num_heads
        self.hidden_dim = config.hidden_dim
        self.head_dim = config.hidden_dim // config.num_heads
        
        self.q_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.k_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.v_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.out_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(
        self, 
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Standard multi-head attention.
        
        Args:
            x: Input tensor [batch, seq, hidden_dim]
            mask: Optional causal mask
        
        Returns:
            Output tensor [batch, seq, hidden_dim]
        """
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        out = self.out_proj(out)
        
        return out


class AKIRABandLayer(nn.Module):
    """
    Single AKIRA transformer layer with band structure.
    
    Processing order:
    1. Layer norm
    2. Multi-head attention (on concatenated bands)
    3. Split into bands
    4. Per-band MLP processing
    5. Wormhole attention (cross-band communication)
    6. Concatenate bands
    """
    
    def __init__(self, config: AKIRAConfig):
        super().__init__()
        
        self.config = config
        
        # Pre-norm for attention
        self.norm1 = nn.LayerNorm(config.hidden_dim)
        
        # Multi-head attention
        self.attention = AKIRAMultiHeadAttention(config)
        
        # Pre-norm for MLPs
        self.band_norms = nn.ModuleList([
            nn.LayerNorm(dim) for dim in config.band_dims
        ])
        
        # Per-band MLPs
        self.band_mlps = nn.ModuleList([
            BandMLP(dim, config.mlp_expansion, config.dropout) 
            for dim in config.band_dims
        ])
        
        # Wormhole attention (optional)
        if config.use_wormhole:
            self.wormhole = WormholeAttention(config.band_dims, config.wormhole_threshold)
        else:
            self.wormhole = None
        
        self.dropout = nn.Dropout(config.dropout)
    
    def split_bands(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Split concatenated representation into bands."""
        bands = []
        start = 0
        for dim in self.config.band_dims:
            bands.append(x[..., start:start + dim])
            start += dim
        return bands
    
    def concat_bands(self, bands: List[torch.Tensor]) -> torch.Tensor:
        """Concatenate bands into single representation."""
        return torch.cat(bands, dim=-1)
    
    def forward(
        self, 
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through AKIRA layer.
        
        Args:
            x: Input tensor [batch, seq, hidden_dim]
            mask: Optional causal mask
        
        Returns:
            Output tensor [batch, seq, hidden_dim]
        """
        # Attention block
        residual = x
        x = self.norm1(x)
        x = self.attention(x, mask)
        x = self.dropout(x)
        x = residual + x
        
        # Split into bands
        bands = self.split_bands(x)
        
        # Per-band MLP
        processed_bands = []
        for i, (band, norm, mlp) in enumerate(zip(bands, self.band_norms, self.band_mlps)):
            residual = band
            band = norm(band)
            band = mlp(band)
            band = residual + band
            processed_bands.append(band)
        
        # Wormhole attention (cross-band communication)
        if self.wormhole is not None:
            processed_bands = self.wormhole(processed_bands, mask)
        
        # Concatenate bands
        x = self.concat_bands(processed_bands)
        
        return x


class AKIRABandTransformer(nn.Module):
    """
    Full AKIRA Band Transformer model.
    
    This is the main model class that implements AKIRA's spectral band architecture.
    """
    
    def __init__(self, config: AKIRAConfig):
        super().__init__()
        
        config.validate()
        self.config = config
        
        # Token and position embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.position_embedding = nn.Embedding(config.max_seq_length, config.hidden_dim)
        
        self.dropout = nn.Dropout(config.dropout)
        
        # AKIRA layers
        self.layers = nn.ModuleList([
            AKIRABandLayer(config) for _ in range(config.num_layers)
        ])
        
        # Final layer norm
        self.final_norm = nn.LayerNorm(config.hidden_dim)
        
        # Output projection (tied with token embedding by default)
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights following GPT-2 conventions."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def get_band_parameters(self) -> Dict[int, List[nn.Parameter]]:
        """
        Get parameters grouped by band for differential learning rates.
        
        Returns:
            Dictionary mapping band index to list of parameters
        """
        band_params = {i: [] for i in range(self.config.num_bands)}
        
        for layer in self.layers:
            for i, mlp in enumerate(layer.band_mlps):
                band_params[i].extend(mlp.parameters())
            for i, norm in enumerate(layer.band_norms):
                band_params[i].extend(norm.parameters())
        
        return band_params
    
    def get_optimizer_param_groups(self) -> List[Dict]:
        """
        Get parameter groups with per-band learning rates for optimizer.
        
        Returns:
            List of param group dicts for optimizer
        """
        band_params = self.get_band_parameters()
        
        param_groups = []
        
        # Band-specific parameters with differential LRs
        for band_idx, params in band_params.items():
            param_groups.append({
                'params': params,
                'lr': self.config.band_learning_rates[band_idx],
                'name': f'band_{band_idx}'
            })
        
        # Non-band parameters (embeddings, attention, final norm) with base LR
        non_band_params = []
        non_band_params.extend(self.token_embedding.parameters())
        non_band_params.extend(self.position_embedding.parameters())
        non_band_params.extend(self.final_norm.parameters())
        non_band_params.extend(self.lm_head.parameters())
        
        for layer in self.layers:
            non_band_params.extend(layer.norm1.parameters())
            non_band_params.extend(layer.attention.parameters())
            if layer.wormhole is not None:
                non_band_params.extend(layer.wormhole.parameters())
        
        param_groups.append({
            'params': non_band_params,
            'lr': self.config.band_learning_rates[3],  # Use mid-band LR as base
            'name': 'shared'
        })
        
        return param_groups
    
    def forward(
        self, 
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through AKIRA model.
        
        Args:
            input_ids: Token IDs [batch, seq]
            attention_mask: Optional attention mask
            labels: Optional labels for loss computation
        
        Returns:
            Dictionary with 'logits' and optionally 'loss'
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Embeddings
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        x = self.dropout(x)
        
        # Causal mask
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
        causal_mask = ~causal_mask  # Flip for attention
        
        if attention_mask is not None:
            causal_mask = causal_mask & attention_mask.unsqueeze(1).unsqueeze(2)
        
        # Process through layers
        for layer in self.layers:
            x = layer(x, causal_mask)
        
        # Final norm and output
        x = self.final_norm(x)
        logits = self.lm_head(x)
        
        output = {'logits': logits}
        
        # Compute loss if labels provided
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
    
    def extract_band_activations(
        self, 
        input_ids: torch.Tensor,
        layer_idx: int
    ) -> Dict[int, torch.Tensor]:
        """
        Extract per-band activations for AQ analysis.
        
        Args:
            input_ids: Token IDs [batch, seq]
            layer_idx: Which layer to extract from
        
        Returns:
            Dictionary mapping band index to activation tensor
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Forward pass up to target layer
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
        causal_mask = ~causal_mask
        
        for i, layer in enumerate(self.layers):
            x = layer(x, causal_mask)
            if i == layer_idx:
                break
        
        # Split into bands
        bands = self.layers[layer_idx].split_bands(x)
        
        return {i: band.detach() for i, band in enumerate(bands)}


# Utility functions

def create_akira_model(
    num_bands: int = 7,
    hidden_dim: int = 512,
    num_layers: int = 6,
    num_heads: int = 8,
    use_wormhole: bool = True
) -> AKIRABandTransformer:
    """
    Create an AKIRA model with default band structure.
    
    Args:
        num_bands: Number of frequency bands (default 7)
        hidden_dim: Total hidden dimension (will be distributed across bands)
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        use_wormhole: Whether to use wormhole attention
    
    Returns:
        Configured AKIRABandTransformer
    """
    # Distribute dimensions across bands (larger for low-freq)
    if num_bands == 7:
        # Default AKIRA distribution
        band_dims = (128, 96, 80, 64, 64, 48, 32)
        band_lrs = (1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2)
    else:
        # Uniform distribution for other band counts
        base_dim = hidden_dim // num_bands
        band_dims = tuple([base_dim] * num_bands)
        # Logarithmic LR spread
        band_lrs = tuple([1e-5 * (10 ** (i * 3 / (num_bands - 1))) for i in range(num_bands)])
    
    config = AKIRAConfig(
        num_layers=num_layers,
        num_heads=num_heads,
        num_bands=num_bands,
        band_dims=band_dims,
        band_learning_rates=band_lrs,
        use_wormhole=use_wormhole
    )
    
    return AKIRABandTransformer(config)


if __name__ == "__main__":
    # Quick test
    print("Creating AKIRA Band Transformer...")
    
    config = AKIRAConfig()
    model = AKIRABandTransformer(config)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Hidden dim: {config.hidden_dim}")
    print(f"Band dims: {config.band_dims}")
    print(f"Band LRs: {config.band_learning_rates}")
    
    # Test forward pass
    batch_size, seq_len = 2, 64
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    labels = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    output = model(input_ids, labels=labels)
    print(f"Output logits shape: {output['logits'].shape}")
    print(f"Loss: {output['loss'].item():.4f}")
    
    # Test band activation extraction
    band_acts = model.extract_band_activations(input_ids, layer_idx=2)
    for band_idx, acts in band_acts.items():
        print(f"Band {band_idx} activations: {acts.shape}")
    
    # Test optimizer param groups
    param_groups = model.get_optimizer_param_groups()
    for pg in param_groups:
        print(f"Param group '{pg['name']}': {sum(p.numel() for p in pg['params']):,} params, lr={pg['lr']}")
    
    print("\nAKIRA Band Transformer test complete!")
