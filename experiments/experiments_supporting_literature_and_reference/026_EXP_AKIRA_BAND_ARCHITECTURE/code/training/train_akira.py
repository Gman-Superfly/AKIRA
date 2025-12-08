"""
AKIRA Band Transformer Training Script
======================================

Trains the AKIRA Band Transformer on WikiText-103 with:
- Per-band differential learning rates
- Per-band loss tracking
- Checkpoint saving
- Wandb logging (optional)

AKIRA Project - Experiment 026
Oscar Goldman - Shogu Research Group @ Datamutant.ai

USAGE:
------
# Basic training
python train_akira.py --output ./akira_model/

# With wandb logging
python train_akira.py --output ./akira_model/ --wandb --wandb_project akira_exp026

# Different seed for second model
python train_akira.py --output ./akira_model_seed43/ --seed 43
"""
# !pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# !pip install transformers datasets wandb tqdm


import os
import sys
import argparse
import json
import time
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from models.akira_band_transformer import AKIRABandTransformer, AKIRAConfig


# ==============================================================================
# CONFIGURATION
# ==============================================================================

class TrainingConfig:
    """Training configuration."""
    
    # Data
    dataset: str = "wikitext-103"
    max_seq_length: int = 512
    
    # Training
    batch_size: int = 32
    gradient_accumulation_steps: int = 2  # Effective batch = 64
    total_steps: int = 100_000
    warmup_steps: int = 1_000
    
    # Optimization
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    # Evaluation
    eval_every: int = 1_000
    save_every: int = 10_000
    log_every: int = 100
    
    # Hardware
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    mixed_precision: bool = True
    num_workers: int = 4
    
    # Reproducibility
    seed: int = 42


# ==============================================================================
# DATA LOADING
# ==============================================================================

def load_wikitext103(tokenizer, max_length: int = 512, split: str = "train"):
    """
    Load WikiText-103 dataset.
    
    Args:
        tokenizer: Tokenizer to use
        max_length: Maximum sequence length
        split: Dataset split ("train", "validation", "test")
    
    Returns:
        Dataset ready for DataLoader
    """
    from datasets import load_dataset
    
    print(f"Loading WikiText-103 {split} split...")
    dataset = load_dataset("wikitext", "wikitext-103-v1", split=split)
    
    def tokenize_function(examples):
        # Tokenize
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt"
        )
        # Labels are same as input for language modeling
        tokenized["labels"] = tokenized["input_ids"].clone()
        return tokenized
    
    # Filter empty strings
    dataset = dataset.filter(lambda x: len(x["text"].strip()) > 0)
    
    # Tokenize
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc=f"Tokenizing {split}"
    )
    
    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    
    return tokenized_dataset


class TextDataset(torch.utils.data.Dataset):
    """Simple dataset wrapper."""
    
    def __init__(self, encodings):
        self.encodings = encodings
    
    def __len__(self):
        return len(self.encodings["input_ids"])
    
    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": self.encodings["labels"][idx]
        }


# ==============================================================================
# TRAINING
# ==============================================================================

def train(
    model: AKIRABandTransformer,
    train_loader: DataLoader,
    eval_loader: DataLoader,
    config: TrainingConfig,
    output_dir: str,
    use_wandb: bool = False
):
    """
    Main training loop.
    
    Args:
        model: AKIRA model to train
        train_loader: Training data loader
        eval_loader: Evaluation data loader
        config: Training configuration
        output_dir: Directory to save checkpoints
        use_wandb: Whether to log to wandb
    """
    device = config.device
    model = model.to(device)
    
    # Setup optimizer with per-band learning rates
    param_groups = model.get_optimizer_param_groups()
    optimizer = AdamW(param_groups, weight_decay=config.weight_decay)
    
    # Learning rate scheduler
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config.total_steps - config.warmup_steps,
        eta_min=1e-7
    )
    
    # Mixed precision
    scaler = torch.cuda.amp.GradScaler() if config.mixed_precision and device == "cuda" else None
    
    # Training state
    global_step = 0
    best_eval_loss = float("inf")
    train_losses = []
    
    # Progress bar
    pbar = tqdm(total=config.total_steps, desc="Training")
    
    model.train()
    optimizer.zero_grad()
    
    train_iter = iter(train_loader)
    
    while global_step < config.total_steps:
        # Get batch
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
        
        # Move to device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        
        # Forward pass
        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs["loss"] / config.gradient_accumulation_steps
            scaler.scale(loss).backward()
        else:
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs["loss"] / config.gradient_accumulation_steps
            loss.backward()
        
        train_losses.append(loss.item() * config.gradient_accumulation_steps)
        
        # Gradient accumulation
        if (global_step + 1) % config.gradient_accumulation_steps == 0:
            # Gradient clipping
            if scaler is not None:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            
            # Optimizer step
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            
            optimizer.zero_grad()
            
            # Warmup
            if global_step < config.warmup_steps:
                # Linear warmup
                warmup_factor = (global_step + 1) / config.warmup_steps
                for pg in optimizer.param_groups:
                    pg["lr"] = pg["lr"] * warmup_factor
            else:
                scheduler.step()
        
        global_step += 1
        pbar.update(1)
        
        # Logging
        if global_step % config.log_every == 0:
            avg_loss = sum(train_losses[-config.log_every:]) / config.log_every
            current_lr = optimizer.param_groups[0]["lr"]
            
            pbar.set_postfix({
                "loss": f"{avg_loss:.4f}",
                "lr": f"{current_lr:.2e}"
            })
            
            if use_wandb:
                import wandb
                log_dict = {
                    "train/loss": avg_loss,
                    "train/step": global_step,
                }
                # Log per-band learning rates
                for pg in optimizer.param_groups:
                    log_dict[f"lr/{pg['name']}"] = pg["lr"]
                wandb.log(log_dict)
        
        # Evaluation
        if global_step % config.eval_every == 0:
            eval_loss = evaluate(model, eval_loader, device, scaler)
            
            print(f"\nStep {global_step}: eval_loss = {eval_loss:.4f}")
            
            if use_wandb:
                import wandb
                wandb.log({
                    "eval/loss": eval_loss,
                    "eval/perplexity": torch.exp(torch.tensor(eval_loss)).item(),
                    "train/step": global_step
                })
            
            # Save best model
            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                save_checkpoint(
                    model, optimizer, scheduler, global_step, eval_loss,
                    os.path.join(output_dir, "best_model")
                )
            
            model.train()
        
        # Save checkpoint
        if global_step % config.save_every == 0:
            save_checkpoint(
                model, optimizer, scheduler, global_step, train_losses[-1],
                os.path.join(output_dir, f"checkpoint_{global_step}")
            )
    
    pbar.close()
    
    # Save final model
    save_checkpoint(
        model, optimizer, scheduler, global_step, train_losses[-1],
        os.path.join(output_dir, "final_model")
    )
    
    return model


def evaluate(
    model: AKIRABandTransformer,
    eval_loader: DataLoader,
    device: str,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    max_batches: int = 100
) -> float:
    """
    Evaluate model on validation set.
    
    Args:
        model: Model to evaluate
        eval_loader: Evaluation data loader
        device: Device to use
        scaler: Gradient scaler for mixed precision
        max_batches: Maximum batches to evaluate
    
    Returns:
        Average loss
    """
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Evaluating", leave=False):
            if num_batches >= max_batches:
                break
            
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            else:
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            
            total_loss += outputs["loss"].item()
            num_batches += 1
    
    return total_loss / num_batches


def save_checkpoint(
    model: AKIRABandTransformer,
    optimizer: AdamW,
    scheduler: CosineAnnealingLR,
    step: int,
    loss: float,
    path: str
):
    """Save model checkpoint."""
    os.makedirs(path, exist_ok=True)
    
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "step": step,
        "loss": loss,
        "config": model.config.__dict__
    }, os.path.join(path, "checkpoint.pt"))
    
    print(f"Saved checkpoint to {path}")


def load_checkpoint(path: str, model: AKIRABandTransformer, optimizer: AdamW = None, scheduler = None):
    """Load model checkpoint."""
    checkpoint = torch.load(os.path.join(path, "checkpoint.pt"))
    
    model.load_state_dict(checkpoint["model_state_dict"])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    
    return checkpoint["step"], checkpoint["loss"]


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train AKIRA Band Transformer")
    
    # Output
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    
    # Model
    parser.add_argument("--num_layers", type=int, default=6, help="Number of layers")
    parser.add_argument("--hidden_dim", type=int, default=512, help="Hidden dimension")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--use_wormhole", action="store_true", default=True, help="Use wormhole attention")
    
    # Training
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--total_steps", type=int, default=100000, help="Total training steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Logging
    parser.add_argument("--wandb", action="store_true", help="Use wandb logging")
    parser.add_argument("--wandb_project", type=str, default="akira_exp026", help="Wandb project name")
    
    # Resume
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint to resume from")
    
    args = parser.parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Save args
    with open(os.path.join(args.output, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    
    # Initialize wandb
    if args.wandb:
        import wandb
        wandb.init(
            project=args.wandb_project,
            name=f"akira_seed{args.seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config=vars(args)
        )
    
    # Create model
    print("Creating AKIRA model...")
    config = AKIRAConfig(
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        use_wormhole=args.use_wormhole
    )
    model = AKIRABandTransformer(config)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Load tokenizer (use GPT-2 tokenizer)
    from transformers import GPT2Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load data
    train_dataset = load_wikitext103(tokenizer, config.max_seq_length, "train")
    eval_dataset = load_wikitext103(tokenizer, config.max_seq_length, "validation")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Training config
    train_config = TrainingConfig()
    train_config.batch_size = args.batch_size
    train_config.total_steps = args.total_steps
    train_config.seed = args.seed
    
    # Train
    print("Starting training...")
    model = train(
        model,
        train_loader,
        eval_loader,
        train_config,
        args.output,
        use_wandb=args.wandb
    )
    
    print("Training complete!")
    
    if args.wandb:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
