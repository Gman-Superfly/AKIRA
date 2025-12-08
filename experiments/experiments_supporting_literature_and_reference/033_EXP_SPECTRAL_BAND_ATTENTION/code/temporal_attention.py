"""
Temporal Attention: Self-History Mechanism

Each spatial position attends to its OWN past states via Top-K selection.
This provides object permanence and inertial memory.

Key insight: Per-position attention is crucial for tracking - each pixel
maintains its own temporal context rather than sharing a pooled summary.
"""

from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class TemporalAttention(nn.Module):
    """
    Temporal attention: Each token attends to its own history.
    
    Mask condition: i' = i AND j' = j AND t' < t
    Gating: Top-K (hard attention on most relevant past moments)
    
    This is the "object permanence" mechanism.
    """
    
    def __init__(
        self,
        feature_dim: int,
        attn_dim: int,
        top_k: int = 4,
        decay_rate: float = 0.95,
        causal: bool = True,
        device: torch.device = torch.device('cpu')
    ):
        """
        Args:
            feature_dim: Input feature dimension (per band)
            attn_dim: Hidden attention dimension (d_k)
            top_k: Number of past states to attend to
            decay_rate: Exponential decay gamma in (0,1) for older states
            causal: If True, only attend to past (t' < t)
            device: Torch device
        """
        super().__init__()
        
        assert feature_dim > 0, "feature_dim must be positive"
        assert attn_dim > 0, "attn_dim must be positive"
        assert 0 < decay_rate < 1, "decay_rate must be in (0, 1)"
        assert top_k > 0, "top_k must be positive"
        
        self.feature_dim = feature_dim
        self.attn_dim = attn_dim
        self.top_k = top_k
        self.decay_rate = decay_rate
        self.causal = causal
        self.device = device
        
        # QKV projections
        self.W_q = nn.Linear(feature_dim, attn_dim, bias=False)
        self.W_k = nn.Linear(feature_dim, attn_dim, bias=False)
        self.W_v = nn.Linear(feature_dim, attn_dim, bias=False)
        
        # Output projection
        self.W_o = nn.Linear(attn_dim, feature_dim, bias=False)
        
        # Move to device
        self.to(device)
    
    def forward(
        self,
        query_features: torch.Tensor,
        history_buffer: torch.Tensor,
        current_step: int = 0,
        tau: float = 1.0
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute temporal attention.
        
        Args:
            query_features: Current step features [H, W, D] or [B, H, W, D]
            history_buffer: Past features [T, H, W, D] or [B, T, H, W, D]
            current_step: Current absolute step index (for logging)
            tau: Temperature for softmax
        
        Returns:
            output: Attended features [H, W, D] or [B, H, W, D]
            stats: Dictionary with attention statistics
        """
        # Handle batched vs unbatched input
        if query_features.dim() == 3:
            return self._forward_unbatched(query_features, history_buffer, current_step, tau)
        else:
            return self._forward_batched(query_features, history_buffer, current_step, tau)
    
    def _forward_unbatched(
        self,
        query_features: torch.Tensor,
        history_buffer: torch.Tensor,
        current_step: int,
        tau: float
    ) -> Tuple[torch.Tensor, Dict]:
        """Unbatched forward pass [H, W, D]."""
        H, W, D = query_features.shape
        T = history_buffer.shape[0]
        
        assert D == self.feature_dim, f"Expected feature_dim={self.feature_dim}, got {D}"
        assert history_buffer.shape[1:] == (H, W, D), \
            f"History shape mismatch: {history_buffer.shape[1:]} vs ({H}, {W}, {D})"
        
        # Project to QKV
        Q = self.W_q(query_features)  # [H, W, attn_dim]
        K = self.W_k(history_buffer)  # [T, H, W, attn_dim]
        V = self.W_v(history_buffer)  # [T, H, W, attn_dim]
        
        # Reshape for per-position attention
        Q_flat = Q.reshape(H * W, self.attn_dim)  # [HW, attn_dim]
        K_flat = K.reshape(T, H * W, self.attn_dim)  # [T, HW, attn_dim]
        V_flat = V.reshape(T, H * W, self.attn_dim)  # [T, HW, attn_dim]
        
        # Transpose K for each position to [HW, T, attn_dim]
        K_per_pos = K_flat.permute(1, 0, 2)  # [HW, T, attn_dim]
        V_per_pos = V_flat.permute(1, 0, 2)  # [HW, T, attn_dim]
        
        # Compute attention scores
        scores = torch.bmm(
            Q_flat.unsqueeze(1),  # [HW, 1, attn_dim]
            K_per_pos.transpose(1, 2)  # [HW, attn_dim, T]
        ).squeeze(1)  # [HW, T]
        
        # Scale by sqrt(d_k)
        scores = scores / math.sqrt(self.attn_dim)
        
        # Top-K selection
        topk_indices = None
        if self.top_k < T:
            topk_scores, topk_indices = torch.topk(scores, self.top_k, dim=1)
            mask = torch.zeros_like(scores)
            mask.scatter_(1, topk_indices, 1.0)
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Apply temporal decay
        time_offsets = torch.arange(T, device=self.device, dtype=torch.float32)
        time_deltas = T - time_offsets
        decay_weights = self.decay_rate ** time_deltas
        decay_weights = decay_weights.unsqueeze(0).expand(H * W, -1)
        scores = scores + torch.log(decay_weights + 1e-10)
        
        # Softmax with temperature
        attn_weights = F.softmax(scores / tau, dim=1)  # [HW, T]
        
        # Apply attention
        output = torch.bmm(
            attn_weights.unsqueeze(1),  # [HW, 1, T]
            V_per_pos  # [HW, T, attn_dim]
        ).squeeze(1)  # [HW, attn_dim]
        
        # Project back
        output = self.W_o(output)
        output = output.reshape(H, W, D)
        
        stats = {
            'attn_weights': attn_weights.reshape(H, W, T).detach(),
            'top_k_indices': topk_indices.reshape(H, W, self.top_k).detach() if topk_indices is not None else None,
            'mean_weight': attn_weights.mean().item(),
            'entropy': self._compute_entropy(attn_weights).mean().item(),
        }
        
        return output, stats
    
    def _forward_batched(
        self,
        query_features: torch.Tensor,
        history_buffer: torch.Tensor,
        current_step: int,
        tau: float
    ) -> Tuple[torch.Tensor, Dict]:
        """Batched forward pass [B, H, W, D]."""
        B, H, W, D = query_features.shape
        T = history_buffer.shape[1]
        
        # Process each batch item
        outputs = []
        all_weights = []
        batch_stats = []
        
        for b in range(B):
            out, stats_b = self._forward_unbatched(
                query_features[b], history_buffer[b], current_step, tau
            )
            outputs.append(out)
            all_weights.append(stats_b['attn_weights'])
            batch_stats.append(stats_b)
        
        output = torch.stack(outputs)
        
        stats = {
            'attn_weights': torch.stack(all_weights),
            'mean_weight': sum(s['mean_weight'] for s in batch_stats) / B,
            'entropy': sum(s['entropy'] for s in batch_stats) / B,
        }
        
        return output, stats
    
    def _compute_entropy(self, weights: torch.Tensor) -> torch.Tensor:
        """Compute entropy of attention distribution."""
        weights_safe = torch.clamp(weights, min=1e-10)
        entropy = -torch.sum(weights_safe * torch.log(weights_safe), dim=1)
        return entropy
