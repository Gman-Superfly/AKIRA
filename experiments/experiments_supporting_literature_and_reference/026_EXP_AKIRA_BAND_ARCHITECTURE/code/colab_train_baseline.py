"""
Baseline Transformer - COLAB VERSION
====================================

Standard transformer for comparison with AKIRA.
Same parameter count, no band structure, uniform learning rate.

AKIRA Project - Experiment 026
Oscar Goldman - Shogu Research Group @ Datamutant.ai
"""

# ==============================================================================
# CELL 1: INSTALL (skip if already done)
# ==============================================================================

# !pip install transformers datasets tqdm

# ==============================================================================
# CELL 2: IMPORTS
# ==============================================================================

import os
import json
import math
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ==============================================================================
# CELL 3: BASELINE CONFIG
# ==============================================================================

@dataclass
class BaselineConfig:
    """Configuration for baseline transformer."""
    vocab_size: int = 50257
    num_layers: int = 6
    hidden_dim: int = 512
    num_heads: int = 8
    mlp_expansion: int = 4
    max_seq_length: int = 256
    dropout: float = 0.1
    learning_rate: float = 3e-4
    
    def validate(self) -> bool:
        assert self.hidden_dim % self.num_heads == 0
        return True

print("BaselineConfig defined")

# ==============================================================================
# CELL 4: MULTI-HEAD ATTENTION
# ==============================================================================

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
            scores = scores.masked_fill(mask == 0, torch.finfo(scores.dtype).min)
        
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        return self.out_proj(out)

print("MultiHeadAttention defined")

# ==============================================================================
# CELL 5: MLP
# ==============================================================================

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

print("MLP defined")

# ==============================================================================
# CELL 6: TRANSFORMER LAYER
# ==============================================================================

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

print("TransformerLayer defined")

# ==============================================================================
# CELL 7: BASELINE TRANSFORMER
# ==============================================================================

class BaselineTransformer(nn.Module):
    """Standard transformer model for baseline comparison."""
    
    def __init__(self, config: BaselineConfig):
        super().__init__()
        config.validate()
        self.config = config
        
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.position_embedding = nn.Embedding(config.max_seq_length, config.hidden_dim)
        self.dropout = nn.Dropout(config.dropout)
        
        self.layers = nn.ModuleList([TransformerLayer(config) for _ in range(config.num_layers)])
        self.final_norm = nn.LayerNorm(config.hidden_dim)
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)
        
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
        
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        x = self.dropout(x)
        
        # Causal mask
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
        causal_mask = ~causal_mask
        
        if attention_mask is not None:
            causal_mask = causal_mask & attention_mask.unsqueeze(1).unsqueeze(2)
        
        for layer in self.layers:
            x = layer(x, causal_mask)
        
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

print("BaselineTransformer defined")

# ==============================================================================
# CELL 8: TEST MODEL
# ==============================================================================

print("\n" + "="*60)
print("TESTING BASELINE MODEL")
print("="*60)

config = BaselineConfig()
model = BaselineTransformer(config)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Hidden dim: {config.hidden_dim}")
print(f"Layers: {config.num_layers}")
print(f"MLP expansion: {config.mlp_expansion}")

# Test forward pass
batch_size, seq_len = 2, 64
input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
labels = torch.randint(0, config.vocab_size, (batch_size, seq_len))

output = model(input_ids, labels=labels)
print(f"Output shape: {output['logits'].shape}")
print(f"Loss: {output['loss'].item():.4f}")

print("\nModel test complete!")

# ==============================================================================
# CELL 9: DATA LOADING (reuse if already loaded)
# ==============================================================================

def load_wikitext103(tokenizer, max_length: int = 256, split: str = "train"):
    """Load WikiText-103 dataset."""
    from datasets import load_dataset
    
    print(f"Loading WikiText-103 {split}...")
    dataset = load_dataset("wikitext", "wikitext-103-v1", split=split)
    
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
    tokenized = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    
    return tokenized

print("Data loading function defined")

# ==============================================================================
# CELL 10: TRAINING LOOP
# ==============================================================================

def train_baseline(
    model,
    train_loader,
    eval_loader,
    total_steps=5000,
    eval_every=500,
    log_every=50,
    save_path="./baseline_checkpoint"
):
    """Train baseline model with UNIFORM learning rate."""
    model = model.to(device)
    
    # Single learning rate for all parameters
    optimizer = AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-7)
    
    scaler = torch.amp.GradScaler('cuda') if device == "cuda" else None
    
    model.train()
    train_iter = iter(train_loader)
    train_losses = []
    best_eval_loss = float('inf')
    
    pbar = tqdm(range(total_steps), desc="Training Baseline")
    
    for step in pbar:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
        
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        
        optimizer.zero_grad()
        
        if scaler:
            with torch.amp.autocast('cuda'):
                output = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = output["loss"]
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            output = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = output["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        
        scheduler.step()
        train_losses.append(loss.item())
        
        if step % log_every == 0:
            avg_loss = sum(train_losses[-log_every:]) / len(train_losses[-log_every:])
            pbar.set_postfix({"loss": f"{avg_loss:.4f}", "lr": f"{scheduler.get_last_lr()[0]:.2e}"})
        
        if step > 0 and step % eval_every == 0:
            eval_loss = evaluate(model, eval_loader)
            print(f"\nStep {step}: eval_loss = {eval_loss:.4f}, ppl = {math.exp(eval_loss):.2f}")
            
            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                torch.save(model.state_dict(), f"{save_path}_best.pt")
                print(f"  Saved best model!")
            
            model.train()
    
    torch.save(model.state_dict(), f"{save_path}_final.pt")
    return model

def evaluate(model, eval_loader, max_batches=50):
    """Evaluate model."""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in eval_loader:
            if num_batches >= max_batches:
                break
            
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            with torch.amp.autocast('cuda') if device == "cuda" else torch.no_grad():
                output = model(input_ids, attention_mask=attention_mask, labels=labels)
            
            total_loss += output["loss"].item()
            num_batches += 1
    
    return total_loss / num_batches

print("Training functions defined")

# ==============================================================================
# CELL 11: LOAD DATA (skip if already loaded from AKIRA run)
# ==============================================================================

from transformers import GPT2Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# If you already have train_dataset and eval_dataset from AKIRA run, skip this
train_dataset = load_wikitext103(tokenizer, max_length=256, split="train")
eval_dataset = load_wikitext103(tokenizer, max_length=256, split="validation")

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2, pin_memory=True)
eval_loader = DataLoader(eval_dataset, batch_size=16, shuffle=False, num_workers=2, pin_memory=True)

print(f"Train samples: {len(train_dataset)}")
print(f"Eval samples: {len(eval_dataset)}")

# ==============================================================================
# CELL 12: START TRAINING
# ==============================================================================

# Create fresh baseline model
config = BaselineConfig(max_seq_length=256)
model = BaselineTransformer(config)

print(f"\nStarting BASELINE training on {device}...")
print(f"Model: {sum(p.numel() for p in model.parameters()):,} parameters")
print(f"Learning rate: {config.learning_rate} (uniform for all params)")

# Train with SAME settings as AKIRA
model = train_baseline(
    model,
    train_loader,
    eval_loader,
    total_steps=5000,
    eval_every=500,
    log_every=50,
    save_path="./baseline_model"
)

print("\nBaseline training complete!")
print("\n" + "="*60)
print("COMPARE WITH AKIRA RESULTS")
print("="*60)
print("AKIRA best loss: ~1.99 (from your run)")
print("Baseline best loss: [check above]")
print("="*60)
