"""
Neighbor Attention: Local Spatial Propagation

Each token attends to its 8-connected spatial neighbors within a temporal window.
This models local physics (diffusion, collision, wavefront propagation).

Key insight: Explicit neighbor attention captures the local dynamics that
convolutional layers learn implicitly, but with learnable attention weights.
"""

from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class NeighborAttention(nn.Module):
    """
    Neighbor attention: Local 8-connected spatial attention.
    
    Mask condition: |i - i'| <= 1 AND |j - j'| <= 1 AND |t - t'| <= layer_range
    
    This is the "local physics" mechanism (like CNNs but with learned weights).
    """
    
    def __init__(
        self,
        feature_dim: int,
        attn_dim: int,
        layer_range: int = 5,
        first_layer_only: bool = False,
        causal: bool = True,
        device: torch.device = torch.device('cpu')
    ):
        """
        Args:
            feature_dim: Input feature dimension
            attn_dim: Hidden attention dimension
            layer_range: Temporal window +/- delta_t for neighbor attention
            first_layer_only: If True, only attend to same timestep
            causal: If True, only attend to past neighbors
            device: Torch device
        """
        super().__init__()
        
        assert feature_dim > 0, "feature_dim must be positive"
        assert attn_dim > 0, "attn_dim must be positive"
        assert layer_range >= 0, "layer_range must be non-negative"
        
        self.feature_dim = feature_dim
        self.attn_dim = attn_dim
        self.layer_range = layer_range
        self.first_layer_only = first_layer_only
        self.causal = causal
        self.device = device
        
        # QKV projections
        self.W_q = nn.Linear(feature_dim, attn_dim, bias=False)
        self.W_k = nn.Linear(feature_dim, attn_dim, bias=False)
        self.W_v = nn.Linear(feature_dim, attn_dim, bias=False)
        
        # Output projection
        self.W_o = nn.Linear(attn_dim, feature_dim, bias=False)
        
        self.to(device)
    
    def forward(
        self,
        query_features: torch.Tensor,
        history_buffer: torch.Tensor,
        current_step: int = 0,
        tau: float = 1.0
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute neighbor attention.
        
        Args:
            query_features: Current step features [H, W, D]
            history_buffer: Recent features [T, H, W, D]
            current_step: Current absolute step index
            tau: Temperature for softmax
        
        Returns:
            output: Attended features [H, W, D]
            stats: Dictionary with attention statistics
        """
        H, W, D = query_features.shape
        T = history_buffer.shape[0]
        
        assert D == self.feature_dim, f"Expected feature_dim={self.feature_dim}, got {D}"
        
        # Determine temporal range
        if self.first_layer_only:
            temporal_indices = [-1]
        else:
            if self.causal:
                temporal_indices = list(range(max(-T, -self.layer_range - 1), 0))
            else:
                temporal_indices = list(range(max(-T, -self.layer_range), 
                                              min(1, T - self.layer_range)))
        
        Q = self.W_q(query_features)
        
        outputs = []
        all_weights = []
        
        for t_offset in temporal_indices:
            key_features = history_buffer[t_offset]
            K = self.W_k(key_features)
            V = self.W_v(key_features)
            
            output_t, weights_t = self._attend_to_neighbors(Q, K, V, tau)
            outputs.append(output_t)
            all_weights.append(weights_t)
        
        if len(outputs) > 1:
            output = torch.stack(outputs).mean(dim=0)
        else:
            output = outputs[0]
        
        output = self.W_o(output)
        
        stats = {
            'num_temporal_steps': len(temporal_indices),
            'mean_weight': torch.stack(all_weights).mean().item(),
            'neighbor_weights': all_weights[-1] if all_weights else None,
        }
        
        return output, stats
    
    def _attend_to_neighbors(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        tau: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Attend to 8-connected spatial neighbors."""
        H, W, attn_dim = Q.shape
        
        # Pad K and V for boundary handling
        K_padded = F.pad(K.permute(2, 0, 1), (1, 1, 1, 1), mode='replicate').permute(1, 2, 0)
        V_padded = F.pad(V.permute(2, 0, 1), (1, 1, 1, 1), mode='replicate').permute(1, 2, 0)
        
        # 9 neighbors (self + 8-connected)
        neighbor_offsets = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),  (0, 0),  (0, 1),
            (1, -1),  (1, 0),  (1, 1)
        ]
        
        K_neighbors = torch.zeros(H, W, 9, attn_dim, device=self.device)
        V_neighbors = torch.zeros(H, W, 9, attn_dim, device=self.device)
        
        for n_idx, (di, dj) in enumerate(neighbor_offsets):
            K_neighbors[:, :, n_idx, :] = K_padded[1+di:H+1+di, 1+dj:W+1+dj, :]
            V_neighbors[:, :, n_idx, :] = V_padded[1+di:H+1+di, 1+dj:W+1+dj, :]
        
        scores = torch.einsum('hwd,hwnd->hwn', Q, K_neighbors)
        scores = scores / math.sqrt(attn_dim)
        weights = F.softmax(scores / tau, dim=-1)
        output = torch.einsum('hwn,hwnd->hwd', weights, V_neighbors)
        
        return output, weights
