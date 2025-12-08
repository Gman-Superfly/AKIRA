"""
Spectral Band Attention: Multi-Band Decomposition

Coordinates temporal, neighbor, and wormhole attention across multiple feature bands.

This is the main orchestration layer that:
1. Routes different bands to different attention mechanisms
2. Fuses outputs from all three attention types
3. Enables cross-band pattern matching (e.g., similarity via low_freq, values via intensity)
"""

from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn

from temporal_attention import TemporalAttention
from neighbor_attention import NeighborAttention
from wormhole_attention import WormholeAttention


class SpectralBandAttention(nn.Module):
    """
    Multi-band attention where each mechanism operates on specific feature bands.
    
    Example configuration:
        Temporal:  query=band_0,  key=band_0   (self-consistency)
        Neighbor:  query=band_1,  key=band_2   (cross-band local)
        Wormhole:  query=band_0,  key=band_1   (cross-band global)
    """
    
    def __init__(
        self,
        bands: List[str],
        band_dims: Dict[str, int],
        attn_dim: int,
        config: Dict,
        device: torch.device = torch.device('cpu')
    ):
        """
        Args:
            bands: List of available band names
            band_dims: Dict mapping band name -> feature dimension
            attn_dim: Attention hidden dimension
            config: Configuration dict with band assignments
            device: Torch device
        """
        super().__init__()
        
        assert len(bands) > 0, "Must have at least one band"
        assert attn_dim > 0, "attn_dim must be positive"
        
        self.bands = bands
        self.band_dims = band_dims
        self.attn_dim = attn_dim
        self.config = config
        self.device = device
        
        # Get band assignments from config (with defaults)
        self.temporal_query_band = config.get('temporal_query_band', bands[0])
        self.temporal_key_band = config.get('temporal_key_band', bands[0])
        
        self.neighbor_query_band = config.get('neighbor_query_band', bands[0])
        self.neighbor_key_band = config.get('neighbor_key_band', bands[0])
        
        # Wormhole has three band assignments:
        # - similarity_band: Used for finding similar patterns (compare query to history)
        # - value_band: Where to retrieve K,V from after finding matches
        self.wormhole_query_band = config.get('wormhole_query_band', bands[0])
        self.wormhole_similarity_band = config.get('wormhole_similarity_band', bands[0])
        self.wormhole_value_band = config.get('wormhole_value_band', bands[0])
        
        # Create attention mechanisms
        self.temporal_attn = TemporalAttention(
            feature_dim=band_dims[self.temporal_query_band],
            attn_dim=attn_dim,
            top_k=config.get('attnTopKTemporal', 4),
            decay_rate=config.get('decayRate', 0.95),
            causal=config.get('temporal_causal', True),
            device=device
        )
        
        self.neighbor_attn = NeighborAttention(
            feature_dim=band_dims[self.neighbor_query_band],
            attn_dim=attn_dim,
            layer_range=config.get('layerRange', 5),
            first_layer_only=config.get('neighbor_first_layer_only', False),
            causal=config.get('neighbor_causal', True),
            device=device
        )
        
        self.wormhole_attn = WormholeAttention(
            feature_dim=band_dims[self.wormhole_value_band],  # Use value_band dim (K,V source)
            attn_dim=attn_dim,
            threshold=config.get('wormholeThreshold', 0.92),
            min_temporal_distance=config.get('wormhole_min_temporal_distance', 1),
            max_connections=config.get('wormhole_max_connections', 16),
            causal=config.get('wormhole_causal', True),
            chunk_size=config.get('wormhole_chunk_size', 4096),
            device=device
        )
        
        # Wormhole skip for speed optimization
        self.wormhole_skip_steps = config.get('wormhole_skip_steps', 1)
        self.cached_wormhole_output = None
        self.cached_wormhole_stats = None
        
        # Output fusion layer (each attention outputs its source band's dimension)
        total_dim = (band_dims[self.temporal_query_band] +
                     band_dims[self.neighbor_query_band] +
                     band_dims[self.wormhole_value_band])
        
        self.fusion = nn.Sequential(
            nn.Linear(total_dim, attn_dim),
            nn.ReLU(),
            nn.Linear(attn_dim, attn_dim)
        )
        
        self.to(device)
    
    def forward(
        self,
        current_bands: Dict[str, torch.Tensor],
        history_bands: Dict[str, torch.Tensor],
        current_step: int,
        tau: float = 1.0
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute spectral band attention.
        
        Args:
            current_bands: Current step bands {band_name: [H, W, D_band]}
            history_bands: History bands {band_name: [T, H, W, D_band]}
            current_step: Current absolute step index
            tau: Temperature for softmax
        
        Returns:
            output: Fused attended features [H, W, attn_dim]
            stats: Dictionary with attention statistics
        """
        # 1. Temporal attention
        out_temporal, stats_temporal = self.temporal_attn(
            query_features=current_bands[self.temporal_query_band],
            history_buffer=history_bands[self.temporal_key_band],
            current_step=current_step,
            tau=tau
        )
        
        # 2. Neighbor attention
        out_neighbor, stats_neighbor = self.neighbor_attn(
            query_features=current_bands[self.neighbor_query_band],
            history_buffer=history_bands[self.neighbor_key_band],
            current_step=current_step,
            tau=tau
        )
        
        # 3. Wormhole attention (with optional skip)
        run_wormhole = (current_step % self.wormhole_skip_steps == 0) or (self.cached_wormhole_output is None)
        
        if run_wormhole:
            # Wormhole attention uses:
            # - similarity_band for finding similar patterns (query vs history)
            # - value_band for K,V retrieval after matches are found
            out_wormhole, stats_wormhole = self.wormhole_attn(
                query_features=current_bands[self.wormhole_value_band],      # Q source
                history_buffer=history_bands[self.wormhole_value_band],       # K,V source
                query_band_features=current_bands[self.wormhole_similarity_band],  # similarity query
                key_band_features=history_bands[self.wormhole_similarity_band],    # similarity keys
                current_step=current_step,
                tau=tau
            )
            self.cached_wormhole_output = out_wormhole.detach()
            self.cached_wormhole_stats = stats_wormhole
        else:
            out_wormhole = self.cached_wormhole_output
            stats_wormhole = self.cached_wormhole_stats.copy()
            stats_wormhole['skipped'] = True
        
        # 4. Concatenate and fuse
        combined = torch.cat([
            out_temporal,
            out_neighbor,
            out_wormhole
        ], dim=-1)
        
        output = self.fusion(combined)
        
        stats = {
            'temporal': stats_temporal,
            'neighbor': stats_neighbor,
            'wormhole': stats_wormhole,
        }
        
        return output, stats
