"""
Experiment B: Signal Domain - Attention History vs Signal History
=================================================================

Compares two approaches to temporal memory in array reconstruction:
- Model B1: History buffer stores raw signal samples
- Model B2: History buffer stores attention outputs (beliefs)

Tests with synthetic signals (sinusoids, chirps, transients).

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
import numpy as np

from belief_buffer import BeliefHistoryBuffer, TokenHistoryBuffer

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# ==============================================================================
# CONFIGURATION
# ==============================================================================

@dataclass
class SignalExperimentConfig:
    """Configuration for signal domain experiment."""
    # Signal dimensions
    signal_length: int = 256
    signal_dim: int = 1
    
    # Spectral bands
    num_bands: int = 7
    band_dim: int = 32
    
    # History (key parameter!)
    max_history: int = 128           # How far back to look
    decay_rate: float = 0.95
    
    # Architecture
    num_layers: int = 4
    dropout: float = 0.1
    
    # Training
    batch_size: int = 32
    learning_rate: float = 1e-3
    num_epochs: int = 20
    num_train_samples: int = 3000
    num_val_samples: int = 300

print("SignalExperimentConfig defined")

# ==============================================================================
# SYNTHETIC SIGNAL DATASET
# ==============================================================================

class SyntheticSignalDataset(Dataset):
    """Synthetic signals with known frequency content."""
    
    def __init__(
        self,
        num_samples: int,
        signal_length: int,
        max_frequency: float = 64.0,
        signal_type: str = "mixed"
    ):
        self.num_samples = num_samples
        self.signal_length = signal_length
        self.max_frequency = max_frequency
        self.signal_type = signal_type
        
        self.signals = []
        for _ in range(num_samples):
            signal = self._generate_signal()
            self.signals.append(signal)
    
    def _generate_signal(self) -> torch.Tensor:
        """Generate a single synthetic signal."""
        t = torch.linspace(0, 1, self.signal_length)
        
        if self.signal_type == "single":
            freq = torch.rand(1).item() * self.max_frequency
            amp = 0.5 + 0.5 * torch.rand(1).item()
            phase = 2 * math.pi * torch.rand(1).item()
            signal = amp * torch.sin(2 * math.pi * freq * t + phase)
            
        elif self.signal_type == "multi":
            num_freqs = torch.randint(2, 5, (1,)).item()
            freqs = torch.rand(num_freqs) * self.max_frequency
            amps = 0.3 + 0.4 * torch.rand(num_freqs)
            phases = 2 * math.pi * torch.rand(num_freqs)
            
            signal = torch.zeros(self.signal_length)
            for f, a, p in zip(freqs, amps, phases):
                signal += a * torch.sin(2 * math.pi * f * t + p)
            signal = signal / (signal.abs().max() + 1e-6)
            
        elif self.signal_type == "chirp":
            f0 = torch.rand(1).item() * 5
            f1 = 10 + torch.rand(1).item() * (self.max_frequency - 10)
            phase = f0 * t + (f1 - f0) * t ** 2 / 2
            signal = torch.sin(2 * math.pi * phase)
            
        elif self.signal_type == "transient":
            decay = 2 + torch.rand(1).item() * 8
            freq = 5 + torch.rand(1).item() * 30
            signal = torch.exp(-decay * t) * torch.sin(2 * math.pi * freq * t)
            
        else:  # mixed
            choice = torch.randint(0, 4, (1,)).item()
            self.signal_type = ["single", "multi", "chirp", "transient"][choice]
            signal = self._generate_signal()
            self.signal_type = "mixed"
        
        return signal
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        signal = self.signals[idx].unsqueeze(-1)  # [T, 1]
        return signal, signal.clone()

print("SyntheticSignalDataset defined")

# ==============================================================================
# SPECTRAL DECOMPOSER
# ==============================================================================

class SpectralDecomposer(nn.Module):
    """Decomposes signal into frequency bands."""
    
    def __init__(self, signal_dim: int, num_bands: int, band_dim: int):
        super().__init__()
        self.num_bands = num_bands
        self.band_dim = band_dim
        
        kernel_sizes = [31, 21, 15, 11, 7, 5, 3][:num_bands]
        
        self.band_convs = nn.ModuleList([
            nn.Conv1d(signal_dim, band_dim, kernel_size=k, padding=k//2)
            for k in kernel_sizes
        ])
        
        self.band_norms = nn.ModuleList([
            nn.LayerNorm(band_dim) for _ in range(num_bands)
        ])
    
    def forward(self, x: torch.Tensor) -> Dict[int, torch.Tensor]:
        B, T, D = x.shape
        x_conv = x.transpose(1, 2)
        
        bands = {}
        for band_idx, (conv, norm) in enumerate(zip(self.band_convs, self.band_norms)):
            h = conv(x_conv).transpose(1, 2)
            bands[band_idx] = norm(h)
        
        return bands

print("SpectralDecomposer defined")

# ==============================================================================
# BAND PROCESSOR WITH HISTORY TYPE SELECTION
# ==============================================================================

class BandProcessorWithHistory(nn.Module):
    """
    Processes a frequency band using history-based temporal attention.
    
    Can use either signal history or belief history.
    """
    
    def __init__(
        self,
        band_dim: int,
        max_history: int,
        decay_rate: float,
        history_type: str = "belief",
        dropout: float = 0.1
    ):
        super().__init__()
        self.band_dim = band_dim
        self.max_history = max_history
        self.history_type = history_type
        
        # History buffer
        if history_type == "belief":
            self.history_buffer = BeliefHistoryBuffer(max_history, band_dim)
        else:
            self.history_buffer = TokenHistoryBuffer(max_history, band_dim)
        
        # Attention projections
        self.q_proj = nn.Linear(band_dim, band_dim)
        self.k_proj = nn.Linear(band_dim, band_dim)
        self.v_proj = nn.Linear(band_dim, band_dim)
        self.out_proj = nn.Linear(band_dim, band_dim)
        
        # Feedforward
        self.ffn = nn.Sequential(
            nn.Linear(band_dim, band_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(band_dim * 2, band_dim)
        )
        
        # Norms
        self.norm1 = nn.LayerNorm(band_dim)
        self.norm2 = nn.LayerNorm(band_dim)
        
        # Decay weights
        decay_weights = torch.tensor([decay_rate ** i for i in range(max_history)])
        self.register_buffer('decay_weights', decay_weights)
    
    def forward(self, x: torch.Tensor, update_history: bool = True) -> torch.Tensor:
        """Process band with history-based attention."""
        B, T, D = x.shape
        
        # Normalize input
        x_norm = self.norm1(x)
        
        # Query from current input
        Q = self.q_proj(x_norm)  # [B, T, D]
        
        # Check if we have history
        if self.history_buffer.current_length == 0:
            # No history yet - just use FFN
            output = x + self.ffn(self.norm2(x))
            if update_history:
                if self.history_type == "belief":
                    self.history_buffer.update(output.detach())
                else:
                    self.history_buffer.update(x.detach())
            return output
        
        # Get history [T, T_hist, D] (positions x history x dim)
        history = self.history_buffer.get_all_positions_history()
        T_hist = min(history.shape[1], self.max_history)
        history = history[:, :T_hist, :]  # Truncate if needed
        
        # Handle shape mismatch (history positions vs current positions)
        if history.shape[0] != T:
            # Interpolate history to match current sequence length
            # [P_hist, T_hist, D] -> [T, T_hist, D]
            history = history.unsqueeze(0)  # [1, P_hist, T_hist, D]
            history = history.permute(0, 3, 2, 1)  # [1, D, T_hist, P_hist]
            history = F.interpolate(history, size=(T_hist, T), mode='bilinear', align_corners=False)
            history = history.permute(0, 3, 2, 1).squeeze(0)  # [T, T_hist, D]
        
        # Keys and values from history
        K = self.k_proj(history)  # [T, T_hist, D]
        V = self.v_proj(history)  # [T, T_hist, D]
        
        # Per-position attention
        # Q: [B, T, D] -> need to match with K: [T, T_hist, D]
        # Expand K, V for batch
        K = K.unsqueeze(0).expand(B, -1, -1, -1)  # [B, T, T_hist, D]
        V = V.unsqueeze(0).expand(B, -1, -1, -1)  # [B, T, T_hist, D]
        
        # Attention scores [B, T, T_hist]
        scores = torch.einsum('btd,bthd->bth', Q, K) / math.sqrt(D)
        
        # Apply temporal decay
        decay = self.decay_weights[:T_hist].flip(0)
        scores = scores + torch.log(decay.view(1, 1, -1) + 1e-10)
        
        # Softmax
        attn_weights = F.softmax(scores, dim=-1)  # [B, T, T_hist]
        
        # Attend to values
        attended = torch.einsum('bth,bthd->btd', attn_weights, V)  # [B, T, D]
        
        # Output with residual
        h = self.out_proj(attended)
        x = x + h
        
        # FFN
        output = x + self.ffn(self.norm2(x))
        
        # Update history
        if update_history:
            if self.history_type == "belief":
                self.history_buffer.update(output.detach())
            else:
                self.history_buffer.update(x_norm.detach())
        
        return output
    
    def reset_history(self):
        """Reset history buffer."""
        self.history_buffer.reset()

print("BandProcessorWithHistory defined")

# ==============================================================================
# SPECTRAL RECONSTRUCTOR
# ==============================================================================

class SpectralReconstructor(nn.Module):
    """Reconstructs signal from bands."""
    
    def __init__(self, num_bands: int, band_dim: int, signal_dim: int):
        super().__init__()
        
        self.band_projs = nn.ModuleList([
            nn.Linear(band_dim, signal_dim * 4) for _ in range(num_bands)
        ])
        
        self.mix = nn.Sequential(
            nn.Linear(signal_dim * 4 * num_bands, signal_dim * 8),
            nn.GELU(),
            nn.Linear(signal_dim * 8, signal_dim)
        )
    
    def forward(self, bands: Dict[int, torch.Tensor]) -> torch.Tensor:
        projected = [self.band_projs[i](bands[i]) for i in range(len(bands))]
        concat = torch.cat(projected, dim=-1)
        return self.mix(concat)

print("SpectralReconstructor defined")

# ==============================================================================
# FULL MODEL
# ==============================================================================

class ArrayDecoderWithHistory(nn.Module):
    """
    Array decoder with selectable history type.
    
    Compares:
    - history_type="signal": Uses raw signal samples as history
    - history_type="belief": Uses attention outputs as history
    """
    
    def __init__(self, config: SignalExperimentConfig, history_type: str = "belief"):
        super().__init__()
        self.config = config
        self.history_type = history_type
        
        # Decomposer
        self.decomposer = SpectralDecomposer(
            config.signal_dim, config.num_bands, config.band_dim
        )
        
        # Band processors with different decay rates
        decay_rates = [0.98, 0.95, 0.92, 0.88, 0.84, 0.80, 0.75][:config.num_bands]
        
        self.band_processors = nn.ModuleList([
            BandProcessorWithHistory(
                config.band_dim,
                config.max_history,
                decay_rates[i],
                history_type,
                config.dropout
            )
            for i in range(config.num_bands)
        ])
        
        # Reconstructor
        self.reconstructor = SpectralReconstructor(
            config.num_bands, config.band_dim, config.signal_dim
        )
    
    def forward(
        self,
        x: torch.Tensor,
        update_history: bool = True
    ) -> Dict[str, torch.Tensor]:
        """Forward pass."""
        # Decompose
        bands = self.decomposer(x)
        
        # Process each band
        processed = {}
        for i in range(self.config.num_bands):
            processed[i] = self.band_processors[i](bands[i], update_history)
        
        # Reconstruct
        output = self.reconstructor(processed)
        
        return {'output': output, 'bands': processed}
    
    def reset_history(self):
        """Reset all history buffers."""
        for proc in self.band_processors:
            proc.reset_history()

print("ArrayDecoderWithHistory defined")

# ==============================================================================
# TRAINING
# ==============================================================================

def train_model(
    model: ArrayDecoderWithHistory,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: SignalExperimentConfig
) -> Dict[str, List[float]]:
    """Train model and return history."""
    
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.01)
    
    history = {'train_mse': [], 'val_mse': []}
    
    for epoch in range(config.num_epochs):
        # Training
        model.train()
        model.reset_history()
        train_losses = []
        
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            output = model(inputs)
            loss = F.mse_loss(output['output'], targets)
            
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
                
                output = model(inputs, update_history=False)
                loss = F.mse_loss(output['output'], targets)
                val_losses.append(loss.item())
        
        # Record
        train_mse = sum(train_losses) / len(train_losses) if train_losses else float('inf')
        val_mse = sum(val_losses) / len(val_losses) if val_losses else float('inf')
        
        history['train_mse'].append(train_mse)
        history['val_mse'].append(val_mse)
        
        if epoch % 20 == 0 or epoch == config.num_epochs - 1:
            print(f"Epoch {epoch:3d}: train_mse={train_mse:.6f}, val_mse={val_mse:.6f}")
    
    return history

print("Training function defined")

# ==============================================================================
# MAIN EXPERIMENT
# ==============================================================================

def run_experiment():
    """Run comparison experiment."""
    print("\n" + "="*60)
    print("EXPERIMENT 028B: Signal Domain - History Type Comparison")
    print("="*60)
    
    config = SignalExperimentConfig()
    
    # Create datasets
    print("\nCreating datasets...")
    train_dataset = SyntheticSignalDataset(
        config.num_train_samples,
        config.signal_length,
        signal_type="mixed"
    )
    val_dataset = SyntheticSignalDataset(
        config.num_val_samples,
        config.signal_length,
        signal_type="mixed"
    )
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    results = {}
    
    # ==================================
    # Model B1: Signal History
    # ==================================
    print("\n" + "-"*40)
    print("Training Model B1: SIGNAL HISTORY (Raw Samples)")
    print("-"*40)
    
    model_signal = ArrayDecoderWithHistory(config, history_type="signal")
    num_params = sum(p.numel() for p in model_signal.parameters())
    print(f"Parameters: {num_params:,}")
    
    history_signal = train_model(model_signal, train_loader, val_loader, config)
    results['signal'] = history_signal
    
    # ==================================
    # Model B2: Belief History
    # ==================================
    print("\n" + "-"*40)
    print("Training Model B2: BELIEF HISTORY (Attention Output)")
    print("-"*40)
    
    model_belief = ArrayDecoderWithHistory(config, history_type="belief")
    print(f"Parameters: {num_params:,}")
    
    history_belief = train_model(model_belief, train_loader, val_loader, config)
    results['belief'] = history_belief
    
    # ==================================
    # Comparison
    # ==================================
    print("\n" + "="*60)
    print("RESULTS COMPARISON")
    print("="*60)
    
    signal_final_mse = results['signal']['val_mse'][-1]
    belief_final_mse = results['belief']['val_mse'][-1]
    
    print(f"\nFinal Validation MSE:")
    print(f"  Signal History: {signal_final_mse:.6f}")
    print(f"  Belief History: {belief_final_mse:.6f}")
    
    improvement = (signal_final_mse - belief_final_mse) / signal_final_mse * 100
    print(f"\nImprovement with Belief History: {improvement:+.2f}%")
    
    if belief_final_mse < signal_final_mse:
        print("\n>>> BELIEF HISTORY WINS - Belief propagation hypothesis supported")
    else:
        print("\n>>> SIGNAL HISTORY WINS - More investigation needed")
    
    print("\n" + "="*60)
    print("EXPERIMENT 028B COMPLETE")
    print("="*60)
    
    return results


if __name__ == "__main__":
    results = run_experiment()
