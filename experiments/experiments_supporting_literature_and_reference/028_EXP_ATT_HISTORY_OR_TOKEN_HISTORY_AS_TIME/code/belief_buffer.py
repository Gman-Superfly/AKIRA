"""
Belief History Buffer - Shared Implementation
==============================================

Maintains rolling buffer of attention outputs (beliefs) rather than raw tokens.
This enables belief propagation over time.

Key insight: Attention outputs contain PROCESSED, CONTEXTUALIZED states -
the system's understanding, not just raw observations.

AKIRA Project - Experiment 028
Oscar Goldman - Shogu Research Group @ Datamutant.ai
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple


class BeliefHistoryBuffer(nn.Module):
    """
    Maintains rolling buffer of attention outputs (beliefs).
    
    Unlike token history (which stores raw input), this stores
    the PROCESSED states - the system's evolving understanding.
    
    Properties:
    - Fixed maximum history length
    - Rolling update (FIFO)
    - Per-position history tracking
    - Supports batched operation
    """
    
    def __init__(
        self,
        max_history: int,
        feature_dim: int,
        num_positions: Optional[int] = None
    ):
        """
        Args:
            max_history: Maximum number of past belief states to store
            feature_dim: Dimension of belief states
            num_positions: Number of positions (optional, can be dynamic)
        """
        super().__init__()
        self.max_history = max_history
        self.feature_dim = feature_dim
        self.num_positions = num_positions
        
        # Buffer will be lazily initialized on first update
        # Shape: [history_length, num_positions, feature_dim]
        self.register_buffer('buffer', None)
        self.register_buffer('current_length', torch.tensor(0))
    
    def reset(self):
        """Clear the buffer."""
        self.buffer = None
        self.current_length = torch.tensor(0)
    
    def update(self, belief_state: torch.Tensor) -> None:
        """
        Add new belief state to buffer.
        
        Args:
            belief_state: [batch, positions, dim] or [positions, dim]
        """
        # Handle batched vs unbatched
        if belief_state.dim() == 2:
            belief_state = belief_state.unsqueeze(0)  # Add batch dim
        
        B, P, D = belief_state.shape
        
        # Initialize buffer if needed
        if self.buffer is None:
            self.buffer = torch.zeros(
                self.max_history, P, D,
                device=belief_state.device,
                dtype=belief_state.dtype
            )
            self.current_length = torch.tensor(0, device=belief_state.device)
        
        # For batched input, we take mean across batch (or could store per-batch)
        # For simplicity, average beliefs across batch
        # Use detach and clone to avoid autograd issues
        belief_mean = belief_state.mean(dim=0).detach().clone()  # [P, D]
        
        # Roll buffer and add new state - avoid inplace operations
        if self.current_length < self.max_history:
            # Buffer not full yet - create new buffer with updated value
            idx = self.current_length.item()
            new_buffer = self.buffer.clone()
            new_buffer[idx] = belief_mean
            self.buffer = new_buffer
            self.current_length = self.current_length + 1
        else:
            # Buffer full - concatenate (drop oldest, add newest)
            self.buffer = torch.cat([self.buffer[1:], belief_mean.unsqueeze(0)], dim=0)
    
    def get_history(self, position_idx: Optional[int] = None) -> torch.Tensor:
        """
        Get belief history.
        
        Args:
            position_idx: If provided, return history for single position
            
        Returns:
            [current_length, positions, dim] or [current_length, dim] if position_idx given
        """
        if self.buffer is None:
            raise RuntimeError("Buffer not initialized. Call update() first.")
        
        valid_history = self.buffer[:self.current_length]
        
        if position_idx is not None:
            return valid_history[:, position_idx, :]  # [T, D]
        return valid_history  # [T, P, D]
    
    def get_all_positions_history(self) -> torch.Tensor:
        """
        Get history for all positions, suitable for per-position attention.
        
        Returns:
            [positions, current_length, dim] - transposed for easier attention
        """
        if self.buffer is None:
            raise RuntimeError("Buffer not initialized. Call update() first.")
        
        valid_history = self.buffer[:self.current_length]  # [T, P, D]
        return valid_history.transpose(0, 1)  # [P, T, D]


class TokenHistoryBuffer(nn.Module):
    """
    Standard token history buffer for comparison.
    
    Stores raw input tokens/values rather than processed beliefs.
    """
    
    def __init__(
        self,
        max_history: int,
        feature_dim: int,
        num_positions: Optional[int] = None
    ):
        super().__init__()
        self.max_history = max_history
        self.feature_dim = feature_dim
        self.num_positions = num_positions
        
        self.register_buffer('buffer', None)
        self.register_buffer('current_length', torch.tensor(0))
    
    def reset(self):
        """Clear the buffer."""
        self.buffer = None
        self.current_length = torch.tensor(0)
    
    def update(self, tokens: torch.Tensor) -> None:
        """
        Add new tokens to buffer.
        
        Args:
            tokens: [batch, positions, dim] or [positions, dim]
        """
        if tokens.dim() == 2:
            tokens = tokens.unsqueeze(0)
        
        B, P, D = tokens.shape
        
        if self.buffer is None:
            self.buffer = torch.zeros(
                self.max_history, P, D,
                device=tokens.device,
                dtype=tokens.dtype
            )
            self.current_length = torch.tensor(0, device=tokens.device)
        
        # Use detach and clone to avoid autograd issues
        token_mean = tokens.mean(dim=0).detach().clone()
        
        # Avoid inplace operations
        if self.current_length < self.max_history:
            idx = self.current_length.item()
            new_buffer = self.buffer.clone()
            new_buffer[idx] = token_mean
            self.buffer = new_buffer
            self.current_length = self.current_length + 1
        else:
            # Concatenate (drop oldest, add newest)
            self.buffer = torch.cat([self.buffer[1:], token_mean.unsqueeze(0)], dim=0)
    
    def get_history(self, position_idx: Optional[int] = None) -> torch.Tensor:
        """Get token history."""
        if self.buffer is None:
            raise RuntimeError("Buffer not initialized. Call update() first.")
        
        valid_history = self.buffer[:self.current_length]
        
        if position_idx is not None:
            return valid_history[:, position_idx, :]
        return valid_history
    
    def get_all_positions_history(self) -> torch.Tensor:
        """Get history for all positions."""
        if self.buffer is None:
            raise RuntimeError("Buffer not initialized. Call update() first.")
        
        valid_history = self.buffer[:self.current_length]
        return valid_history.transpose(0, 1)


class TemporalAttentionWithHistory(nn.Module):
    """
    Temporal attention that can use either token or belief history.
    
    Each position attends to its own history (per-position attention).
    This implements the "each sensor has its own memory" concept.
    """
    
    def __init__(
        self,
        feature_dim: int,
        history_type: str = "belief",  # "belief" or "token"
        max_history: int = 128,
        decay_rate: float = 0.95
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.history_type = history_type
        self.max_history = max_history
        self.decay_rate = decay_rate
        
        # History buffer (belief or token)
        if history_type == "belief":
            self.history_buffer = BeliefHistoryBuffer(max_history, feature_dim)
        else:
            self.history_buffer = TokenHistoryBuffer(max_history, feature_dim)
        
        # Attention projections
        self.q_proj = nn.Linear(feature_dim, feature_dim)
        self.k_proj = nn.Linear(feature_dim, feature_dim)
        self.v_proj = nn.Linear(feature_dim, feature_dim)
        self.out_proj = nn.Linear(feature_dim, feature_dim)
        
        # Precompute decay weights
        decay_weights = torch.tensor([decay_rate ** i for i in range(max_history)])
        self.register_buffer('decay_weights', decay_weights)
    
    def forward(
        self,
        current_input: torch.Tensor,
        update_buffer: bool = True
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass with history-based temporal attention.
        
        Args:
            current_input: [batch, positions, dim] current input
            update_buffer: Whether to update history buffer after attention
            
        Returns:
            output: [batch, positions, dim] attended output
            stats: Dictionary with attention statistics
        """
        B, P, D = current_input.shape
        
        # Query from current input
        Q = self.q_proj(current_input)  # [B, P, D]
        
        # Check if we have history
        if self.history_buffer.current_length == 0:
            # No history yet - just project input and update buffer
            output = self.out_proj(current_input)
            if update_buffer:
                if self.history_type == "belief":
                    self.history_buffer.update(output.detach())
                else:
                    self.history_buffer.update(current_input.detach())
            return output, {'history_length': 0, 'used_history': False}
        
        # Get history [P, T_hist, D]
        history = self.history_buffer.get_all_positions_history()
        T_hist = history.shape[1]
        
        # Keys and values from history
        K = self.k_proj(history)  # [P, T_hist, D]
        V = self.v_proj(history)  # [P, T_hist, D]
        
        # Per-position attention: each position attends to its own history
        # Q: [B, P, D] -> [B, P, 1, D]
        # K: [P, T_hist, D] -> [1, P, T_hist, D]
        Q_exp = Q.unsqueeze(2)  # [B, P, 1, D]
        K_exp = K.unsqueeze(0)  # [1, P, T_hist, D]
        V_exp = V.unsqueeze(0)  # [1, P, T_hist, D]
        
        # Attention scores [B, P, 1, T_hist]
        scores = torch.matmul(Q_exp, K_exp.transpose(-2, -1)) / (D ** 0.5)
        
        # Apply temporal decay (more recent = higher weight)
        decay = self.decay_weights[:T_hist].flip(0)  # Recent first
        decay = decay.view(1, 1, 1, -1)
        scores = scores + torch.log(decay + 1e-10)
        
        # Softmax
        attn_weights = torch.softmax(scores, dim=-1)  # [B, P, 1, T_hist]
        
        # Attend to values
        attended = torch.matmul(attn_weights, V_exp)  # [B, P, 1, D]
        attended = attended.squeeze(2)  # [B, P, D]
        
        # Combine with current input (residual)
        output = self.out_proj(attended + current_input)
        
        # Update buffer
        if update_buffer:
            if self.history_type == "belief":
                self.history_buffer.update(output.detach())
            else:
                self.history_buffer.update(current_input.detach())
        
        stats = {
            'history_length': T_hist,
            'used_history': True,
            'mean_attn_entropy': -(attn_weights * torch.log(attn_weights + 1e-10)).sum(-1).mean().item()
        }
        
        return output, stats
    
    def reset_history(self):
        """Clear history buffer."""
        self.history_buffer.reset()


print("Belief and Token History Buffers defined")
print("TemporalAttentionWithHistory defined")
