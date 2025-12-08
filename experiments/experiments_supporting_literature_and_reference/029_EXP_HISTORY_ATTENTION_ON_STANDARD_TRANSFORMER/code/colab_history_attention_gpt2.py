"""
EXPERIMENT 029: History Attention on Standard Transformer
=========================================================

Tests whether adding per-position history attention to a standard GPT-2
style transformer improves language modeling on WikiText-2.

Three configurations:
1. BASELINE: Standard GPT-2 (causal self-attention only)
2. + TOKEN HISTORY: Standard + history attention storing raw embeddings
3. + BELIEF HISTORY: Standard + history attention storing attention outputs

Run on Google Colab with GPU for best performance.

AKIRA Project - Experiment 029
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
from dataclasses import dataclass
import time

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ==============================================================================
# CELL 3: CONFIGURATION
# ==============================================================================

@dataclass
class GPT2Config:
    """Configuration for GPT-2 style transformer."""
    vocab_size: int = 50257          # GPT-2 vocabulary
    embed_dim: int = 256             # Small for speed
    num_layers: int = 6              # Moderate depth
    num_heads: int = 8               # Standard
    max_seq_length: int = 256        # Context window
    dropout: float = 0.1
    
    # History attention settings
    max_history: int = 64            # History buffer size
    decay_rate: float = 0.95         # Temporal decay
    
    # Training
    batch_size: int = 16
    learning_rate: float = 3e-4
    total_steps: int = 500          # Adjust based on time
    eval_every: int = 50
    warmup_steps: int = 100

print("GPT2Config defined")
print(f"  embed_dim: {256}, num_layers: {6}, num_heads: {8}")

# ==============================================================================
# CELL 4: HISTORY BUFFER CLASSES
# ==============================================================================

class BeliefHistoryBuffer(nn.Module):
    """Stores attention outputs (beliefs) across time."""
    
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
    
    def get_all_positions_history(self) -> torch.Tensor:
        if self.buffer is None:
            raise RuntimeError("Buffer not initialized")
        valid = self.buffer[:self.current_length]
        return valid.transpose(0, 1)


class TokenHistoryBuffer(nn.Module):
    """Stores raw token embeddings across time."""
    
    def __init__(self, max_history: int, feature_dim: int):
        super().__init__()
        self.max_history = max_history
        self.feature_dim = feature_dim
        self.register_buffer('buffer', None)
        self.register_buffer('current_length', torch.tensor(0))
    
    def reset(self):
        self.buffer = None
        self.current_length = torch.tensor(0)
    
    def update(self, tokens: torch.Tensor) -> None:
        if tokens.dim() == 2:
            tokens = tokens.unsqueeze(0)
        B, P, D = tokens.shape
        
        if self.buffer is None:
            self.buffer = torch.zeros(self.max_history, P, D, device=tokens.device, dtype=tokens.dtype)
            self.current_length = torch.tensor(0, device=tokens.device)
        
        token_mean = tokens.mean(dim=0).detach().clone()
        
        if self.current_length < self.max_history:
            idx = self.current_length.item()
            new_buffer = self.buffer.clone()
            new_buffer[idx] = token_mean
            self.buffer = new_buffer
            self.current_length = self.current_length + 1
        else:
            self.buffer = torch.cat([self.buffer[1:], token_mean.unsqueeze(0)], dim=0)
    
    def get_all_positions_history(self) -> torch.Tensor:
        if self.buffer is None:
            raise RuntimeError("Buffer not initialized")
        valid = self.buffer[:self.current_length]
        return valid.transpose(0, 1)

print("History buffers defined")

# ==============================================================================
# CELL 5: HISTORY ATTENTION LAYER
# ==============================================================================

class HistoryAttention(nn.Module):
    """
    Per-position history attention.
    Each position attends to its OWN history across time.
    """
    
    def __init__(
        self,
        embed_dim: int,
        history_type: str,  # "token" or "belief"
        max_history: int,
        decay_rate: float
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.history_type = history_type
        self.max_history = max_history
        
        if history_type == "belief":
            self.history_buffer = BeliefHistoryBuffer(max_history, embed_dim)
        else:
            self.history_buffer = TokenHistoryBuffer(max_history, embed_dim)
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        decay_weights = torch.tensor([decay_rate ** i for i in range(max_history)])
        self.register_buffer('decay_weights', decay_weights)
    
    def forward(self, x: torch.Tensor, update_buffer: bool = True) -> torch.Tensor:
        B, T, D = x.shape
        
        if self.history_buffer.current_length == 0:
            output = self.out_proj(x)
            if update_buffer:
                if self.history_type == "belief":
                    self.history_buffer.update(output.detach())
                else:
                    self.history_buffer.update(x.detach())
            return output
        
        Q = self.q_proj(x)
        history = self.history_buffer.get_all_positions_history()
        T_hist = history.shape[1]
        
        K = self.k_proj(history)
        V = self.v_proj(history)
        
        Q_exp = Q.unsqueeze(2)
        K_exp = K.unsqueeze(0)
        V_exp = V.unsqueeze(0)
        
        scores = torch.matmul(Q_exp, K_exp.transpose(-2, -1)) / math.sqrt(D)
        
        decay = self.decay_weights[:T_hist].flip(0).view(1, 1, 1, -1)
        scores = scores + torch.log(decay + 1e-10)
        
        attn = F.softmax(scores, dim=-1)
        attended = torch.matmul(attn, V_exp).squeeze(2)
        
        output = self.out_proj(attended + x)
        
        if update_buffer:
            if self.history_type == "belief":
                self.history_buffer.update(output.detach())
            else:
                self.history_buffer.update(x.detach())
        
        return output
    
    def reset_history(self):
        self.history_buffer.reset()

print("HistoryAttention defined")

# ==============================================================================
# CELL 6: STANDARD TRANSFORMER LAYER
# ==============================================================================

class TransformerLayer(nn.Module):
    """Standard GPT-2 style transformer layer."""
    
    def __init__(self, config: GPT2Config):
        super().__init__()
        
        self.self_attn = nn.MultiheadAttention(
            config.embed_dim, config.num_heads,
            dropout=config.dropout, batch_first=True
        )
        
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
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        
        # Self-attention with causal mask
        mask = self._get_causal_mask(T, x.device)
        attn_out, _ = self.self_attn(
            self.norm1(x), self.norm1(x), self.norm1(x),
            attn_mask=mask
        )
        x = x + attn_out
        
        # FFN
        x = x + self.ffn(self.norm2(x))
        
        return x

print("TransformerLayer defined")

# ==============================================================================
# CELL 7: TRANSFORMER LAYER WITH HISTORY ATTENTION
# ==============================================================================

class TransformerLayerWithHistory(nn.Module):
    """Transformer layer with additional history attention."""
    
    def __init__(self, config: GPT2Config, history_type: str):
        super().__init__()
        
        # History attention (vertical - across time)
        self.history_attn = HistoryAttention(
            config.embed_dim,
            history_type,
            config.max_history,
            config.decay_rate
        )
        self.norm0 = nn.LayerNorm(config.embed_dim)
        
        # Standard self-attention (horizontal - within sequence)
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
        
        # 1. History attention (vertical)
        x = x + self.history_attn(self.norm0(x), update_buffer=update_history)
        
        # 2. Self-attention (horizontal)
        mask = self._get_causal_mask(T, x.device)
        attn_out, _ = self.self_attn(
            self.norm1(x), self.norm1(x), self.norm1(x),
            attn_mask=mask
        )
        x = x + attn_out
        
        # 3. FFN
        x = x + self.ffn(self.norm2(x))
        
        return x
    
    def reset_history(self):
        self.history_attn.reset_history()

print("TransformerLayerWithHistory defined")

# ==============================================================================
# CELL 8: GPT-2 MODEL
# ==============================================================================

class GPT2LM(nn.Module):
    """GPT-2 style language model with optional history attention."""
    
    def __init__(self, config: GPT2Config, history_type: Optional[str] = None):
        """
        Args:
            config: Model configuration
            history_type: None (baseline), "token", or "belief"
        """
        super().__init__()
        self.config = config
        self.history_type = history_type
        
        # Embeddings
        self.token_embed = nn.Embedding(config.vocab_size, config.embed_dim)
        self.pos_embed = nn.Embedding(config.max_seq_length, config.embed_dim)
        self.dropout = nn.Dropout(config.dropout)
        
        # Transformer layers
        if history_type is None:
            # Baseline: standard transformer
            self.layers = nn.ModuleList([
                TransformerLayer(config) for _ in range(config.num_layers)
            ])
        else:
            # With history attention
            self.layers = nn.ModuleList([
                TransformerLayerWithHistory(config, history_type)
                for _ in range(config.num_layers)
            ])
        
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
            if self.history_type is not None:
                x = layer(x, update_history=update_history)
            else:
                x = layer(x)
        
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
        if self.history_type is not None:
            for layer in self.layers:
                layer.reset_history()

print("GPT2LM defined")

# ==============================================================================
# CELL 9: DATA LOADING
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
    
    # Filter empty lines
    dataset = dataset.filter(lambda x: len(x["text"].strip()) > 0)
    
    # Tokenize
    tokenized = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    
    return tokenized

print("Data loading function defined")

# ==============================================================================
# CELL 10: TRAINING FUNCTION
# ==============================================================================

def train_model(
    model: GPT2LM,
    train_loader: DataLoader,
    eval_loader: DataLoader,
    config: GPT2Config,
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
            model.reset_history()  # Reset history at epoch boundary
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
        
        # Evaluation
        if (step + 1) % config.eval_every == 0:
            model.eval()
            model.reset_history()
            
            eval_losses = []
            with torch.no_grad():
                for i, batch in enumerate(eval_loader):
                    if i >= 50:  # Limit eval batches for speed
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
# CELL 11: MAIN EXPERIMENT
# ==============================================================================

def run_experiment():
    """Run the full comparison experiment."""
    
    print("\n" + "="*70)
    print("EXPERIMENT 029: History Attention on Standard Transformer")
    print("="*70)
    
    # Load tokenizer
    from transformers import GPT2Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Configuration
    config = GPT2Config()
    
    print(f"\nConfiguration:")
    print(f"  embed_dim: {config.embed_dim}")
    print(f"  num_layers: {config.num_layers}")
    print(f"  num_heads: {config.num_heads}")
    print(f"  max_history: {config.max_history}")
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
    
    # ==================================
    # Model 1: BASELINE (Standard GPT-2)
    # ==================================
    print("\n" + "-"*50)
    print("MODEL 1: BASELINE (Standard GPT-2)")
    print("-"*50)
    
    model_baseline = GPT2LM(config, history_type=None)
    history_baseline = train_model(model_baseline, train_loader, eval_loader, config, "Baseline")
    results['baseline'] = history_baseline
    
    # Clear memory
    del model_baseline
    torch.cuda.empty_cache() if device == "cuda" else None
    
    # ==================================
    # Model 2: + TOKEN HISTORY
    # ==================================
    print("\n" + "-"*50)
    print("MODEL 2: + TOKEN HISTORY")
    print("-"*50)
    
    model_token = GPT2LM(config, history_type="token")
    history_token = train_model(model_token, train_loader, eval_loader, config, "+Token History")
    results['token'] = history_token
    
    del model_token
    torch.cuda.empty_cache() if device == "cuda" else None
    
    # ==================================
    # Model 3: + BELIEF HISTORY
    # ==================================
    print("\n" + "-"*50)
    print("MODEL 3: + BELIEF HISTORY")
    print("-"*50)
    
    model_belief = GPT2LM(config, history_type="belief")
    history_belief = train_model(model_belief, train_loader, eval_loader, config, "+Belief History")
    results['belief'] = history_belief
    
    del model_belief
    torch.cuda.empty_cache() if device == "cuda" else None
    
    # ==================================
    # RESULTS COMPARISON
    # ==================================
    print("\n" + "="*70)
    print("RESULTS COMPARISON")
    print("="*70)
    
    baseline_ppl = results['baseline']['eval_ppl'][-1]
    token_ppl = results['token']['eval_ppl'][-1]
    belief_ppl = results['belief']['eval_ppl'][-1]
    
    baseline_loss = results['baseline']['eval_loss'][-1]
    token_loss = results['token']['eval_loss'][-1]
    belief_loss = results['belief']['eval_loss'][-1]
    
    print(f"\nFinal Validation Results:")
    print(f"  Baseline:       loss={baseline_loss:.4f}, ppl={baseline_ppl:.2f}")
    print(f"  +Token History: loss={token_loss:.4f}, ppl={token_ppl:.2f}")
    print(f"  +Belief History: loss={belief_loss:.4f}, ppl={belief_ppl:.2f}")
    
    # Improvements
    token_improve = (baseline_ppl - token_ppl) / baseline_ppl * 100
    belief_improve = (baseline_ppl - belief_ppl) / baseline_ppl * 100
    belief_vs_token = (token_ppl - belief_ppl) / token_ppl * 100
    
    print(f"\nPerplexity Improvements:")
    print(f"  Token History vs Baseline:  {token_improve:+.2f}%")
    print(f"  Belief History vs Baseline: {belief_improve:+.2f}%")
    print(f"  Belief vs Token History:    {belief_vs_token:+.2f}%")
    
    # Winner
    print("\n" + "-"*50)
    if belief_ppl < baseline_ppl and belief_ppl < token_ppl:
        print(">>> BELIEF HISTORY WINS")
        print("    History attention improves standard transformer")
        print("    Storing beliefs > storing tokens (consistent with Exp 028)")
    elif token_ppl < baseline_ppl:
        print(">>> TOKEN HISTORY WINS")
        print("    History attention helps, but raw tokens work better than beliefs")
    else:
        print(">>> BASELINE WINS")
        print("    History attention does not improve standard transformer")
        print("    Further investigation needed")
    print("-"*50)
    
    print("\n" + "="*70)
    print("EXPERIMENT 029 COMPLETE")
    print("="*70)
    
    return results

# ==============================================================================
# CELL 12: RUN EXPERIMENT
# ==============================================================================

if __name__ == "__main__":
    results = run_experiment()
