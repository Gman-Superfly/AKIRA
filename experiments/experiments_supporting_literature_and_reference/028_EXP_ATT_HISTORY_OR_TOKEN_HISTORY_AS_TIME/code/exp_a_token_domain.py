"""
Experiment A: Token Domain - Attention History vs Token History
===============================================================

Compares two approaches to temporal memory in a language modeling context:
- Model A1: History buffer stores token embeddings (raw input)
- Model A2: History buffer stores attention outputs (beliefs)

Small-scale experiment to validate the belief propagation hypothesis.

AKIRA Project - Experiment 028
Oscar Goldman - Shogu Research Group @ Datamutant.ai
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from belief_buffer import TemporalAttentionWithHistory

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# ==============================================================================
# CONFIGURATION
# ==============================================================================

@dataclass
class TokenExperimentConfig:
    """Configuration for token domain experiment."""
    # Vocabulary
    vocab_size: int = 1000           # Small vocab for speed
    embed_dim: int = 128             # Embedding dimension
    
    # Architecture
    num_layers: int = 4
    num_heads: int = 4
    max_seq_length: int = 64
    dropout: float = 0.1
    
    # History
    max_history: int = 128           # How far back to look
    decay_rate: float = 0.95         # Temporal decay
    
    # Training
    batch_size: int = 32
    learning_rate: float = 3e-4
    num_epochs: int = 20             # 50 og value slow
    num_train_samples: int = 3000    #10000 slow
    num_val_samples: int = 300        #1000 slow remember these because big boy test needs them

print("TokenExperimentConfig defined")

# ==============================================================================
# SYNTHETIC TOKEN DATASET
# ==============================================================================

class SyntheticTokenDataset(Dataset):
    """
    Synthetic token sequences with learnable patterns.
    
    Generates sequences with:
    - Repeating patterns (tests long-range memory)
    - Local correlations (tests short-range)
    - Random noise (baseline difficulty)
    """
    
    def __init__(
        self,
        num_samples: int,
        seq_length: int,
        vocab_size: int,
        pattern_type: str = "mixed"
    ):
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.pattern_type = pattern_type
        
        # Generate sequences
        self.sequences = []
        for _ in range(num_samples):
            seq = self._generate_sequence()
            self.sequences.append(seq)
    
    def _generate_sequence(self) -> torch.Tensor:
        """Generate a single sequence."""
        if self.pattern_type == "repeating":
            # Repeating pattern (tests long-range memory)
            pattern_len = torch.randint(4, 16, (1,)).item()
            pattern = torch.randint(0, self.vocab_size, (pattern_len,))
            repeats = self.seq_length // pattern_len + 1
            seq = pattern.repeat(repeats)[:self.seq_length]
            
        elif self.pattern_type == "local":
            # Local correlations (next token depends on previous few)
            seq = torch.zeros(self.seq_length, dtype=torch.long)
            seq[0] = torch.randint(0, self.vocab_size, (1,))
            for i in range(1, self.seq_length):
                # Next token is (prev + small_offset) % vocab_size
                offset = torch.randint(-3, 4, (1,)).item()
                seq[i] = (seq[i-1] + offset) % self.vocab_size
                
        elif self.pattern_type == "random":
            # Pure random (baseline)
            seq = torch.randint(0, self.vocab_size, (self.seq_length,))
            
        else:  # mixed
            choice = torch.randint(0, 3, (1,)).item()
            self.pattern_type = ["repeating", "local", "random"][choice]
            seq = self._generate_sequence()
            self.pattern_type = "mixed"
        
        return seq
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (input, target) where target is shifted by 1."""
        seq = self.sequences[idx]
        return seq[:-1], seq[1:]

print("SyntheticTokenDataset defined")

# ==============================================================================
# MODEL WITH HISTORY-BASED ATTENTION
# ==============================================================================

class HistoryTransformerLayer(nn.Module):
    """
    Transformer layer that uses history-based temporal attention.
    
    Key difference from standard transformer:
    - Each position attends to its OWN history (past attention outputs or tokens)
    - NOT global attention to all sequence positions
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        history_type: str,
        max_history: int,
        decay_rate: float,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # History-based temporal attention
        self.temporal_attn = TemporalAttentionWithHistory(
            feature_dim=embed_dim,
            history_type=history_type,
            max_history=max_history,
            decay_rate=decay_rate
        )
        
        # Standard causal self-attention (within current sequence)
        self.self_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        
        # Feedforward
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )
        
        # Layer norms
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        
        # Causal mask
        self.register_buffer('causal_mask', None)
    
    def _get_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Get causal attention mask."""
        if self.causal_mask is None or self.causal_mask.shape[0] < seq_len:
            mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
            self.causal_mask = mask.bool()
        return self.causal_mask[:seq_len, :seq_len]
    
    def forward(self, x: torch.Tensor, update_history: bool = True) -> torch.Tensor:
        """Forward pass."""
        B, T, D = x.shape
        
        # 1. History-based temporal attention
        h, _ = self.temporal_attn(self.norm1(x), update_buffer=update_history)
        x = x + h
        
        # 2. Causal self-attention within sequence
        mask = self._get_causal_mask(T, x.device)
        attn_out, _ = self.self_attn(
            self.norm2(x), self.norm2(x), self.norm2(x),
            attn_mask=mask
        )
        x = x + attn_out
        
        # 3. Feedforward
        x = x + self.ffn(self.norm3(x))
        
        return x
    
    def reset_history(self):
        """Reset history buffer."""
        self.temporal_attn.reset_history()


class HistoryTransformerLM(nn.Module):
    """
    Language model with history-based temporal attention.
    
    Compares:
    - history_type="token": Uses raw token embeddings as history
    - history_type="belief": Uses attention outputs as history
    """
    
    def __init__(self, config: TokenExperimentConfig, history_type: str = "belief"):
        super().__init__()
        self.config = config
        self.history_type = history_type
        
        # Embeddings
        self.token_embed = nn.Embedding(config.vocab_size, config.embed_dim)
        self.pos_embed = nn.Embedding(config.max_seq_length, config.embed_dim)
        self.dropout = nn.Dropout(config.dropout)
        
        # Transformer layers with history attention
        self.layers = nn.ModuleList([
            HistoryTransformerLayer(
                config.embed_dim,
                config.num_heads,
                history_type,
                config.max_history,
                config.decay_rate,
                config.dropout
            )
            for _ in range(config.num_layers)
        ])
        
        # Output
        self.final_norm = nn.LayerNorm(config.embed_dim)
        self.lm_head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)
        
        # Initialize
        self.apply(self._init_weights)
    
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
        targets: Optional[torch.Tensor] = None,
        update_history: bool = True
    ) -> Dict[str, torch.Tensor]:
        """Forward pass."""
        B, T = input_ids.shape
        
        # Embeddings
        positions = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, -1)
        x = self.token_embed(input_ids) + self.pos_embed(positions)
        x = self.dropout(x)
        
        # Process through layers
        for layer in self.layers:
            x = layer(x, update_history=update_history)
        
        # Output
        x = self.final_norm(x)
        logits = self.lm_head(x)
        
        output = {'logits': logits}
        
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                targets.view(-1),
                ignore_index=-100
            )
            output['loss'] = loss
        
        return output
    
    def reset_history(self):
        """Reset all history buffers."""
        for layer in self.layers:
            layer.reset_history()

print("HistoryTransformerLM defined")

# ==============================================================================
# TRAINING
# ==============================================================================

def train_model(
    model: HistoryTransformerLM,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: TokenExperimentConfig
) -> Dict[str, List[float]]:
    """Train model and return history."""
    
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.01)
    
    history = {'train_loss': [], 'val_loss': [], 'val_ppl': []}
    
    for epoch in range(config.num_epochs):
        # Training
        model.train()
        model.reset_history()  # Reset at start of epoch
        train_losses = []
        
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            output = model(inputs, targets=targets)
            loss = output['loss']
            
            if torch.isnan(loss):
                continue
                
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_losses.append(loss.item())
        
        # Validation
        model.eval()
        model.reset_history()
        val_losses = []
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                output = model(inputs, targets=targets, update_history=False)
                val_losses.append(output['loss'].item())
        
        # Record
        train_loss = sum(train_losses) / len(train_losses) if train_losses else float('inf')
        val_loss = sum(val_losses) / len(val_losses) if val_losses else float('inf')
        val_ppl = math.exp(min(val_loss, 10))  # Cap for numerical stability
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_ppl'].append(val_ppl)
        
        if epoch % 10 == 0 or epoch == config.num_epochs - 1:
            print(f"Epoch {epoch:3d}: train_loss={train_loss:.4f}, "
                  f"val_loss={val_loss:.4f}, val_ppl={val_ppl:.2f}")
    
    return history

print("Training function defined")

# ==============================================================================
# MAIN EXPERIMENT
# ==============================================================================

def run_experiment():
    """Run comparison experiment."""
    print("\n" + "="*60)
    print("EXPERIMENT 028A: Token Domain - History Type Comparison")
    print("="*60)
    
    config = TokenExperimentConfig()
    
    # Create datasets
    print("\nCreating datasets...")
    train_dataset = SyntheticTokenDataset(
        config.num_train_samples,
        config.max_seq_length,
        config.vocab_size,
        pattern_type="mixed"
    )
    val_dataset = SyntheticTokenDataset(
        config.num_val_samples,
        config.max_seq_length,
        config.vocab_size,
        pattern_type="mixed"
    )
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    results = {}
    
    # ==================================
    # Model A1: Token History
    # ==================================
    print("\n" + "-"*40)
    print("Training Model A1: TOKEN HISTORY")
    print("-"*40)
    
    model_token = HistoryTransformerLM(config, history_type="token")
    num_params = sum(p.numel() for p in model_token.parameters())
    print(f"Parameters: {num_params:,}")
    
    history_token = train_model(model_token, train_loader, val_loader, config)
    results['token'] = history_token
    
    # ==================================
    # Model A2: Belief History
    # ==================================
    print("\n" + "-"*40)
    print("Training Model A2: BELIEF HISTORY (Attention Output)")
    print("-"*40)
    
    model_belief = HistoryTransformerLM(config, history_type="belief")
    print(f"Parameters: {num_params:,}")
    
    history_belief = train_model(model_belief, train_loader, val_loader, config)
    results['belief'] = history_belief
    
    # ==================================
    # Comparison
    # ==================================
    print("\n" + "="*60)
    print("RESULTS COMPARISON")
    print("="*60)
    
    token_final_loss = results['token']['val_loss'][-1]
    token_final_ppl = results['token']['val_ppl'][-1]
    belief_final_loss = results['belief']['val_loss'][-1]
    belief_final_ppl = results['belief']['val_ppl'][-1]
    
    print(f"\nFinal Validation Loss:")
    print(f"  Token History:  {token_final_loss:.4f} (ppl {token_final_ppl:.2f})")
    print(f"  Belief History: {belief_final_loss:.4f} (ppl {belief_final_ppl:.2f})")
    
    improvement = (token_final_loss - belief_final_loss) / token_final_loss * 100
    print(f"\nImprovement with Belief History: {improvement:+.2f}%")
    
    if belief_final_loss < token_final_loss:
        print("\n>>> BELIEF HISTORY WINS - Belief propagation hypothesis supported")
    else:
        print("\n>>> TOKEN HISTORY WINS - More investigation needed")
    
    print("\n" + "="*60)
    print("EXPERIMENT 028A COMPLETE")
    print("="*60)
    
    return results


if __name__ == "__main__":
    results = run_experiment()
