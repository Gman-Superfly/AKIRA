"""
Wormhole Attention: Non-Local Phase-Locking

Connects distant tokens based on content similarity.
This enables instant global synchronization and resonance effects.

Key insight: Uses SEPARATE bands for similarity matching vs value retrieval.
This allows finding structurally similar patterns (via low-freq) while
retrieving appropriate intensity values from matches.
"""

from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class WormholeAttention(nn.Module):
    """
    Wormhole attention: Sparse non-local connections via similarity gating.
    
    Mask condition: Similarity(q, k) > threshold
    
    This is the "teleportation" mechanism for connecting distant similar features.
    """
    
    def __init__(
        self,
        feature_dim: int,
        attn_dim: int,
        threshold: float = 0.92,
        min_temporal_distance: int = 1,
        max_connections: Optional[int] = 16,
        causal: bool = True,
        chunk_size: int = 4096,
        max_spatial_tokens: int = 1_000_000,
        spatial_pool_size: int = 2,
        device: torch.device = torch.device('cpu')
    ):
        """
        Args:
            feature_dim: Input feature dimension
            attn_dim: Hidden attention dimension
            threshold: Similarity threshold tau in [0, 1] for connections
            min_temporal_distance: Minimum time gap for wormholes
            max_connections: Maximum wormhole links per query
            causal: If True, only connect to past
            chunk_size: Chunk size for memory-efficient similarity computation
            max_spatial_tokens: Maximum spatial tokens (H*W) to process
            spatial_pool_size: Pooling size if spatial tokens exceed max
            device: Torch device
        """
        super().__init__()
        
        assert feature_dim > 0, "feature_dim must be positive"
        assert attn_dim > 0, "attn_dim must be positive"
        assert 0 <= threshold <= 1, "threshold must be in [0, 1]"
        assert max_spatial_tokens > 0, "max_spatial_tokens must be positive"
        assert spatial_pool_size > 0, "spatial_pool_size must be positive"
        
        self.feature_dim = feature_dim
        self.attn_dim = attn_dim
        self.threshold = threshold
        self.min_temporal_distance = min_temporal_distance
        self.max_connections = max_connections
        self.causal = causal
        self.chunk_size = chunk_size
        self.max_spatial_tokens = max_spatial_tokens
        self.spatial_pool_size = spatial_pool_size
        self.device = device
        
        # QKV projections
        self.W_q = nn.Linear(feature_dim, attn_dim, bias=False)
        self.W_k = nn.Linear(feature_dim, attn_dim, bias=False)
        self.W_v = nn.Linear(feature_dim, attn_dim, bias=False)
        
        # Output projection
        self.W_o = nn.Linear(attn_dim, feature_dim, bias=False)
        
        self.to(device)
    
    def _adaptive_spatial_pool(
        self,
        features: torch.Tensor,
        pool_size: int
    ) -> Tuple[torch.Tensor, Tuple[int, int], Tuple[int, int]]:
        """
        Adaptively pool spatial features to reduce resolution.
        
        Args:
            features: Input features [H, W, D] or [T, H, W, D]
            pool_size: Pooling kernel size
            
        Returns:
            pooled_features: Pooled features with same format
            original_shape: (H, W) before pooling
            pooled_shape: (H', W') after pooling
        """
        is_temporal = features.ndim == 4
        
        if is_temporal:
            T, H, W, D = features.shape
            original_shape = (H, W)
            # Reshape to [T, D, H, W] for pooling
            features_pooling = features.permute(0, 3, 1, 2)
            # Pool each timestep independently
            pooled_list = []
            for t in range(T):
                pooled_t = F.adaptive_avg_pool2d(
                    features_pooling[t:t+1],
                    (H // pool_size, W // pool_size)
                )
                pooled_list.append(pooled_t)
            pooled_features = torch.cat(pooled_list, dim=0)
            # Back to [T, H', W', D]
            pooled_features = pooled_features.permute(0, 2, 3, 1)
            T, H_new, W_new, D = pooled_features.shape
            pooled_shape = (H_new, W_new)
        else:
            H, W, D = features.shape
            original_shape = (H, W)
            # Reshape to [1, D, H, W] for pooling
            features_pooling = features.permute(2, 0, 1).unsqueeze(0)
            pooled_features = F.adaptive_avg_pool2d(
                features_pooling,
                (H // pool_size, W // pool_size)
            )
            # Back to [H', W', D]
            pooled_features = pooled_features.squeeze(0).permute(1, 2, 0)
            H_new, W_new, D = pooled_features.shape
            pooled_shape = (H_new, W_new)
        
        return pooled_features, original_shape, pooled_shape
    
    def _upsample_output(
        self,
        output: torch.Tensor,
        target_shape: Tuple[int, int]
    ) -> torch.Tensor:
        """
        Upsample output back to original spatial resolution.
        
        Args:
            output: Pooled output [H', W', D]
            target_shape: Target (H, W)
            
        Returns:
            upsampled: Output at original resolution [H, W, D]
        """
        H_target, W_target = target_shape
        H, W, D = output.shape
        
        if (H, W) == (H_target, W_target):
            return output
        
        # Reshape to [1, D, H, W] for interpolation
        output_interp = output.permute(2, 0, 1).unsqueeze(0)
        upsampled = F.interpolate(
            output_interp,
            size=(H_target, W_target),
            mode='bilinear',
            align_corners=False
        )
        # Back to [H, W, D]
        upsampled = upsampled.squeeze(0).permute(1, 2, 0)
        
        return upsampled
    
    def forward(
        self,
        query_features: torch.Tensor,
        history_buffer: torch.Tensor,
        query_band_features: Optional[torch.Tensor] = None,
        key_band_features: Optional[torch.Tensor] = None,
        current_step: int = 0,
        tau: float = 1.0
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute wormhole attention with sparse similarity gating.
        
        Args:
            query_features: Current features [H, W, D]
            history_buffer: Past features [T, H, W, D]
            query_band_features: Optional features for similarity [H, W, D_band]
            key_band_features: Optional features for similarity [T, H, W, D_band]
            current_step: Current step index
            tau: Temperature for softmax
        
        Returns:
            output: Attended features [H, W, D]
            stats: Dictionary with wormhole statistics
        """
        H, W, D = query_features.shape
        T = history_buffer.shape[0]
        
        assert D == self.feature_dim, f"Expected feature_dim={self.feature_dim}, got {D}"
        
        # Store original shapes
        original_H, original_W = H, W
        
        # Use provided band features for similarity, or default to main features
        if query_band_features is None:
            query_band_features = query_features
        if key_band_features is None:
            key_band_features = history_buffer
        
        # Adaptive spatial pooling if spatial tokens exceed max
        HW = H * W
        need_pooling = HW > self.max_spatial_tokens
        
        if need_pooling:
            # Pool query features and band features
            query_features_pooled, _, (H, W) = self._adaptive_spatial_pool(
                query_features, self.spatial_pool_size
            )
            query_band_features_pooled, _, _ = self._adaptive_spatial_pool(
                query_band_features, self.spatial_pool_size
            )
            history_buffer_pooled, _, _ = self._adaptive_spatial_pool(
                history_buffer, self.spatial_pool_size
            )
            key_band_features_pooled, _, _ = self._adaptive_spatial_pool(
                key_band_features, self.spatial_pool_size
            )
            
            # Use pooled versions
            query_features_work = query_features_pooled
            query_band_features_work = query_band_features_pooled
            history_buffer_work = history_buffer_pooled
            key_band_features_work = key_band_features_pooled
            
            HW = H * W
        else:
            # Use original versions
            query_features_work = query_features
            query_band_features_work = query_band_features
            history_buffer_work = history_buffer
            key_band_features_work = key_band_features
        
        Q = self.W_q(query_features_work)
        
        # Flatten for similarity search
        query_flat = query_band_features_work.reshape(H * W, -1)
        key_flat = key_band_features_work.reshape(T * H * W, -1)
        
        query_norm = F.normalize(query_flat, p=2, dim=1)
        key_norm = F.normalize(key_flat, p=2, dim=1)
        
        THW = T * H * W
        
        K_neighbors = self.max_connections if self.max_connections is not None else 16
        K_neighbors = min(K_neighbors, THW)
        
        # Adaptive chunk sizing based on query size
        # Rule: chunk_size should give us a manageable similarity matrix
        # sim_chunk will be [HW, chunk_size], so HW * chunk_size should be reasonable
        # Target: keep sim_chunk under ~100MB (25M float32 elements)
        max_sim_elements = 25_000_000  # ~100MB
        adaptive_chunk_size = max(1, min(self.chunk_size, max_sim_elements // HW))
        
        # Chunked similarity computation
        all_topk_vals = []
        all_topk_inds = []
        
        for i in range(0, THW, adaptive_chunk_size):
            end = min(i + adaptive_chunk_size, THW)
            sim_chunk = torch.matmul(query_norm, key_norm[i:end].T)
            
            k_local = min(K_neighbors, end - i)
            vals, inds = torch.topk(sim_chunk, k_local, dim=1)
            
            all_topk_vals.append(vals)
            all_topk_inds.append(inds + i)
        
        if len(all_topk_vals) > 0:
            merged_vals = torch.cat(all_topk_vals, dim=1)
            merged_inds = torch.cat(all_topk_inds, dim=1)
            
            topk_sim, topk_indices_local = torch.topk(merged_vals, K_neighbors, dim=1)
            topk_indices = torch.gather(merged_inds, 1, topk_indices_local)
        else:
            topk_sim = torch.zeros(HW, K_neighbors, device=self.device)
            topk_indices = torch.zeros(HW, K_neighbors, device=self.device, dtype=torch.long)
        
        # Apply causal/temporal mask
        if self.causal or self.min_temporal_distance > 0:
            key_times = topk_indices // (H * W)
            time_distance = (T - 1) - key_times
            
            invalid_mask = torch.zeros_like(topk_sim, dtype=torch.bool)
            if self.causal:
                invalid_mask |= (time_distance <= 0)
            if self.min_temporal_distance > 0:
                invalid_mask |= (time_distance < self.min_temporal_distance)
            
            topk_sim.masked_fill_(invalid_mask, -1.0)
        
        # Filter by threshold
        mask = topk_sim > self.threshold
        
        # Sparse attention computation
        history_buffer_flat = history_buffer_work.reshape(T * H * W, -1)
        
        gathered_hist = torch.gather(
            history_buffer_flat.unsqueeze(0).expand(H * W, -1, -1),
            1,
            topk_indices.unsqueeze(-1).expand(-1, -1, self.feature_dim)
        )
        
        K_selected = self.W_k(gathered_hist)
        V_selected = self.W_v(gathered_hist)
        
        Q_selected = Q.reshape(H * W, 1, self.attn_dim)
        
        scores = torch.bmm(Q_selected, K_selected.transpose(1, 2)).squeeze(1)
        scores = scores / math.sqrt(self.attn_dim)
        scores = scores.masked_fill(~mask, float('-inf'))
        
        attn_weights = F.softmax(scores / tau, dim=-1)
        
        valid_rows = mask.any(dim=1)
        attn_weights = torch.where(
            valid_rows.unsqueeze(1),
            attn_weights,
            torch.zeros_like(attn_weights)
        )
        
        output_flat = torch.bmm(attn_weights.unsqueeze(1), V_selected).squeeze(1)
        output_flat = self.W_o(output_flat)
        output = output_flat.reshape(H, W, D)
        
        # Upsample output if we pooled
        if need_pooling:
            output = self._upsample_output(output, (original_H, original_W))
        
        num_connections = mask.sum().item()
        total_possible = H * W * T * H * W
        sparsity = num_connections / total_possible if total_possible > 0 else 0.0
        
        # Per-position connection counts: how many connections each (h, w) has
        # Note: if pooled, this is at pooled resolution
        per_position_counts = mask.sum(dim=1).reshape(H, W)  # [H, W]
        
        # If pooled, upsample the counts too
        if need_pooling:
            per_position_counts_upsampled = F.interpolate(
                per_position_counts.unsqueeze(0).unsqueeze(0).float(),
                size=(original_H, original_W),
                mode='nearest'
            ).squeeze().long()
            per_position_counts = per_position_counts_upsampled
        
        stats = {
            'num_connections': num_connections,
            'sparsity': sparsity,
            'mean_similarity': topk_sim[mask].mean().item() if num_connections > 0 else 0.0,
            'positions_with_connections': valid_rows.sum().item(),
            'per_position_counts': per_position_counts,  # [original_H, original_W] tensor
            'spatial_pooling_applied': need_pooling,
            'pooled_resolution': (H, W) if need_pooling else None,
        }
        
        return output, stats
