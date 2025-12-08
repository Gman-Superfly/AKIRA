"""
AKIRA Band Transformer - COLAB VERSION
======================================

Complete self-contained script for Google Colab.
Run each section as a separate cell.

AKIRA Project - Experiment 026
Oscar Goldman - Shogu Research Group @ Datamutant.ai
"""

# ==============================================================================
# CELL 1: INSTALL DEPENDENCIES
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
# CELL 3: AKIRA CONFIG
# ==============================================================================

@dataclass
class AKIRAConfig:
    """AKIRA Band Transformer configuration."""
    vocab_size: int = 50257
    num_layers: int = 6
    num_heads: int = 8
    max_seq_length: int = 512
    dropout: float = 0.1
    num_bands: int = 7
    band_dims: Tuple[int, ...] = (128, 96, 80, 64, 64, 48, 32)
    band_learning_rates: Tuple[float, ...] = (1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2)
    mlp_expansion: int = 4
    use_wormhole: bool = True
    wormhole_threshold: float = 0.5
    
    @property
    def hidden_dim(self) -> int:
        return sum(self.band_dims)
    
    def validate(self) -> bool:
        assert len(self.band_dims) == self.num_bands
        assert len(self.band_learning_rates) == self.num_bands
        assert self.hidden_dim % self.num_heads == 0
        return True

print("AKIRAConfig defined")

# ==============================================================================
# CELL 4: BAND MLP
# ==============================================================================

class BandMLP(nn.Module):
    """MLP for a single frequency band."""
    
    def __init__(self, band_dim: int, expansion: int = 4, dropout: float = 0.1):
        super().__init__()
        hidden_dim = band_dim * expansion
        self.fc1 = nn.Linear(band_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, band_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

print("BandMLP defined")

# ==============================================================================
# CELL 5: WORMHOLE ATTENTION
# ==============================================================================

class WormholeAttention(nn.Module):
    """Cross-band communication via gated attention."""
    
    def __init__(self, band_dims: Tuple[int, ...], threshold: float = 0.5):
        super().__init__()
        self.band_dims = band_dims
        self.num_bands = len(band_dims)
        self.threshold = threshold
        self.cross_dim = min(band_dims) // 2
        
        self.cross_queries = nn.ModuleList([nn.Linear(dim, self.cross_dim) for dim in band_dims])
        self.cross_keys = nn.ModuleList([nn.Linear(dim, self.cross_dim) for dim in band_dims])
        self.cross_values = nn.ModuleList([nn.Linear(dim, self.cross_dim * 2) for dim in band_dims])
        self.out_projs = nn.ModuleList([nn.Linear(self.cross_dim * 2, dim) for dim in band_dims])
        self.gate = nn.ModuleList([nn.Linear(dim * 2, dim) for dim in band_dims])
    
    def forward(self, bands: List[torch.Tensor], mask: Optional[torch.Tensor] = None) -> List[torch.Tensor]:
        batch_size, seq_len = bands[0].shape[:2]
        updated_bands = []
        
        # Handle 4D mask from main attention (squeeze to 3D for our use)
        if mask is not None and mask.dim() == 4:
            mask = mask.squeeze(1)  # [batch, 1, seq, seq] -> [batch, seq, seq]
        
        for i, band in enumerate(bands):
            cross_info = torch.zeros(batch_size, seq_len, self.band_dims[i], device=band.device, dtype=band.dtype)
            
            for j, other_band in enumerate(bands):
                if i == j:
                    continue
                
                q = self.cross_queries[i](band)
                k = self.cross_keys[j](other_band)
                v = self.cross_values[j](other_band)
                
                scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.cross_dim)
                if mask is not None:
                    scores = scores.masked_fill(mask == 0, torch.finfo(scores.dtype).min)
                
                attn = F.softmax(scores, dim=-1)
                cross_output = torch.matmul(attn, v)
                cross_output = self.out_projs[i](cross_output)
                
                gate_input = torch.cat([band, cross_output], dim=-1)
                gate_value = torch.sigmoid(self.gate[i](gate_input))
                gate_value = (gate_value > self.threshold).float() * gate_value
                
                cross_info = cross_info + gate_value * cross_output
            
            updated_bands.append(band + cross_info / (self.num_bands - 1))
        
        return updated_bands

print("WormholeAttention defined")

# ==============================================================================
# CELL 6: MULTI-HEAD ATTENTION
# ==============================================================================

class AKIRAMultiHeadAttention(nn.Module):
    """Standard multi-head attention."""
    
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

print("AKIRAMultiHeadAttention defined")

# ==============================================================================
# CELL 7: AKIRA BAND LAYER
# ==============================================================================

class AKIRABandLayer(nn.Module):
    """Single AKIRA transformer layer with band structure."""
    
    def __init__(self, config: AKIRAConfig):
        super().__init__()
        self.config = config
        
        self.norm1 = nn.LayerNorm(config.hidden_dim)
        self.attention = AKIRAMultiHeadAttention(config)
        self.band_norms = nn.ModuleList([nn.LayerNorm(dim) for dim in config.band_dims])
        self.band_mlps = nn.ModuleList([BandMLP(dim, config.mlp_expansion, config.dropout) for dim in config.band_dims])
        
        if config.use_wormhole:
            self.wormhole = WormholeAttention(config.band_dims, config.wormhole_threshold)
        else:
            self.wormhole = None
        
        self.dropout = nn.Dropout(config.dropout)
    
    def split_bands(self, x: torch.Tensor) -> List[torch.Tensor]:
        bands = []
        start = 0
        for dim in self.config.band_dims:
            bands.append(x[..., start:start + dim])
            start += dim
        return bands
    
    def concat_bands(self, bands: List[torch.Tensor]) -> torch.Tensor:
        return torch.cat(bands, dim=-1)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Attention
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
        
        # Wormhole
        if self.wormhole is not None:
            processed_bands = self.wormhole(processed_bands, mask)
        
        return self.concat_bands(processed_bands)

print("AKIRABandLayer defined")

# ==============================================================================
# CELL 8: AKIRA BAND TRANSFORMER
# ==============================================================================

class AKIRABandTransformer(nn.Module):
    """Full AKIRA Band Transformer."""
    
    def __init__(self, config: AKIRAConfig):
        super().__init__()
        config.validate()
        self.config = config
        
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.position_embedding = nn.Embedding(config.max_seq_length, config.hidden_dim)
        self.dropout = nn.Dropout(config.dropout)
        
        self.layers = nn.ModuleList([AKIRABandLayer(config) for _ in range(config.num_layers)])
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
    
    def get_optimizer_param_groups(self) -> List[Dict]:
        """Get parameter groups with per-band learning rates."""
        band_params = {i: [] for i in range(self.config.num_bands)}
        
        for layer in self.layers:
            for i, mlp in enumerate(layer.band_mlps):
                band_params[i].extend(mlp.parameters())
            for i, norm in enumerate(layer.band_norms):
                band_params[i].extend(norm.parameters())
        
        param_groups = []
        for band_idx, params in band_params.items():
            param_groups.append({
                'params': params,
                'lr': self.config.band_learning_rates[band_idx],
                'name': f'band_{band_idx}'
            })
        
        # Non-band parameters
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
            'lr': self.config.band_learning_rates[3],
            'name': 'shared'
        })
        
        return param_groups
    
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

print("AKIRABandTransformer defined")

# ==============================================================================
# CELL 9: TEST MODEL
# ==============================================================================

print("\n" + "="*60)
print("TESTING AKIRA MODEL")
print("="*60)

config = AKIRAConfig()
model = AKIRABandTransformer(config)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Hidden dim: {config.hidden_dim}")
print(f"Band dims: {config.band_dims}")

# Test forward pass
batch_size, seq_len = 2, 64
input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
labels = torch.randint(0, config.vocab_size, (batch_size, seq_len))

output = model(input_ids, labels=labels)
print(f"Output shape: {output['logits'].shape}")
print(f"Loss: {output['loss'].item():.4f}")

# Show param groups
param_groups = model.get_optimizer_param_groups()
for pg in param_groups:
    print(f"  {pg['name']}: {sum(p.numel() for p in pg['params']):,} params, lr={pg['lr']}")

print("\nModel test complete!")

# ==============================================================================
# CELL 10: DATA LOADING
# ==============================================================================

def load_wikitext103(tokenizer, max_length: int = 512, split: str = "train"):
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
# CELL 11: TRAINING LOOP
# ==============================================================================

def train_akira(
    model,
    train_loader,
    eval_loader,
    total_steps=10000,
    eval_every=500,
    log_every=50,
    save_path="./akira_checkpoint"
):
    """Train AKIRA model."""
    model = model.to(device)
    
    # Optimizer with per-band LRs
    param_groups = model.get_optimizer_param_groups()
    optimizer = AdamW(param_groups, weight_decay=0.01)
    
    # Scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-7)
    
    # Mixed precision
    scaler = torch.amp.GradScaler('cuda') if device == "cuda" else None
    
    model.train()
    train_iter = iter(train_loader)
    train_losses = []
    best_eval_loss = float('inf')
    
    pbar = tqdm(range(total_steps), desc="Training")
    
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
# CELL 12: RUN TRAINING
# ==============================================================================

# Load tokenizer
from transformers import GPT2Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# Load data (this takes a few minutes)
train_dataset = load_wikitext103(tokenizer, max_length=256, split="train")
eval_dataset = load_wikitext103(tokenizer, max_length=256, split="validation")

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2, pin_memory=True)
eval_loader = DataLoader(eval_dataset, batch_size=16, shuffle=False, num_workers=2, pin_memory=True)

print(f"Train samples: {len(train_dataset)}")
print(f"Eval samples: {len(eval_dataset)}")

# ==============================================================================
# CELL 13: START TRAINING
# ==============================================================================

# Create fresh model
config = AKIRAConfig(max_seq_length=256)  # Match data length
model = AKIRABandTransformer(config)

print(f"\nStarting training on {device}...")
print(f"Model: {sum(p.numel() for p in model.parameters()):,} parameters")

# Train! (adjust steps as needed)
# For quick test: total_steps=1000
# For full training: total_steps=50000
model = train_akira(
    model,
    train_loader,
    eval_loader,
    total_steps=5000,      # Quick test - increase for real training
    eval_every=500,
    log_every=50,
    save_path="./akira_model"
)

print("\nTraining complete!")
