"""
Baseline Transformer (Standard Architecture)
=============================================

Standard transformer for comparison with AKIRA Band Transformer.
Same parameter count, no band structure.

AKIRA Project - Experiment 026
Oscar Goldman - Shogu Research Group @ Datamutant.ai
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict
from dataclasses import dataclass


@dataclass
class BaselineConfig:
    """Configuration for baseline transformer."""
    vocab_size: int = 50257
    num_layers: int = 6
    hidden_dim: int = 512
    num_heads: int = 8
    mlp_expansion: int = 4
    max_seq_length: int = 512
    dropout: float = 0.1
    learning_rate: float = 3e-4
    
    def validate(self) -> bool:
        assert self.hidden_dim % self.num_heads == 0
        return True


class MultiHeadAttention(nn.Module):
    """Standard multi-head attention."""
    
    def __init__(self, config: BaselineConfig):
        super().__init__()
        
        self.num_heads = config.num_heads
        self.hidden_dim = config.hidden_dim
        self.head_dim = config.hidden_dim // config.num_heads
        
        self.q_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.k_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.v_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.out_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        out = self.out_proj(out)
        
        return out


class MLP(nn.Module):
    """Standard MLP."""
    
    def __init__(self, config: BaselineConfig):
        super().__init__()
        
        hidden_dim = config.hidden_dim * config.mlp_expansion
        
        self.fc1 = nn.Linear(config.hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, config.hidden_dim)
        self.dropout = nn.Dropout(config.dropout)
        self.activation = nn.GELU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerLayer(nn.Module):
    """Standard transformer layer."""
    
    def __init__(self, config: BaselineConfig):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(config.hidden_dim)
        self.attention = MultiHeadAttention(config)
        self.norm2 = nn.LayerNorm(config.hidden_dim)
        self.mlp = MLP(config)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Attention block
        residual = x
        x = self.norm1(x)
        x = self.attention(x, mask)
        x = self.dropout(x)
        x = residual + x
        
        # MLP block
        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = residual + x
        
        return x


class BaselineTransformer(nn.Module):
    """
    Standard transformer model for baseline comparison.
    
    No band structure - single MLP per layer with uniform learning rate.
    """
    
    def __init__(self, config: BaselineConfig):
        super().__init__()
        
        config.validate()
        self.config = config
        
        # Embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.position_embedding = nn.Embedding(config.max_seq_length, config.hidden_dim)
        
        self.dropout = nn.Dropout(config.dropout)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerLayer(config) for _ in range(config.num_layers)
        ])
        
        # Output
        self.final_norm = nn.LayerNorm(config.hidden_dim)
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)
        
        # Initialize
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Embeddings
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        x = self.dropout(x)
        
        # Causal mask
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
        causal_mask = ~causal_mask
        
        if attention_mask is not None:
            causal_mask = causal_mask & attention_mask.unsqueeze(1).unsqueeze(2)
        
        # Process through layers
        for layer in self.layers:
            x = layer(x, causal_mask)
        
        # Output
        x = self.final_norm(x)
        logits = self.lm_head(x)
        
        output = {"logits": logits}
        
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100
            )
            output["loss"] = loss
        
        return output
    
    def extract_layer_activations(
        self,
        input_ids: torch.Tensor,
        layer_idx: int
    ) -> torch.Tensor:
        """Extract activations from a specific layer for AQ analysis."""
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
        causal_mask = ~causal_mask
        
        for i, layer in enumerate(self.layers):
            x = layer(x, causal_mask)
            if i == layer_idx:
                # Return MLP output (after residual)
                return x.detach()
        
        return x.detach()


if __name__ == "__main__":
    # Test
    print("Creating Baseline Transformer...")
    
    config = BaselineConfig()
    model = BaselineTransformer(config)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    batch_size, seq_len = 2, 64
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    labels = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    output = model(input_ids, labels=labels)
    print(f"Output logits shape: {output['logits'].shape}")
    print(f"Loss: {output['loss'].item():.4f}")
    
    # Test activation extraction
    acts = model.extract_layer_activations(input_ids, layer_idx=2)
    print(f"Layer 2 activations: {acts.shape}")
    
    print("\nBaseline Transformer test complete!")
