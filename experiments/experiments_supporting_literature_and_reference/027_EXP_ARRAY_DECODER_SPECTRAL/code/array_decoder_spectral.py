"""
Array Decoder with Spectral Bands - Experiment 027
===================================================

Validates spectral band separation using array reconstruction.
Each position has its own temporal memory (not global attention).

Key design:
- Fixed attention structure (exponential decay over history)
- Learnable decoder (reconstruction from bands)
- Synthetic signals with known frequency content

AKIRA Project - Experiment 027
Oscar Goldman - Shogu Research Group @ Datamutant.ai
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# ==============================================================================
# CONFIGURATION
# ==============================================================================

@dataclass
class ArrayDecoderConfig:
    """Configuration for Array Decoder."""
    # Signal dimensions
    signal_length: int = 256          # T: number of samples
    signal_dim: int = 1               # D: features per sample (1 for 1D signal)
    
    # Spectral bands
    num_bands: int = 7                # Number of frequency bands
    band_dim: int = 32                # Features per band
    
    # Temporal memory
    history_length: int = 16          # How many past frames each position sees
    decay_rate: float = 0.9           # Exponential decay for fixed attention
    
    # Architecture
    num_layers: int = 2               # Processing layers
    dropout: float = 0.1
    
    # Training
    batch_size: int = 32
    learning_rate: float = 1e-3
    num_epochs: int = 100
    
    # Synthetic signal params
    max_frequency: float = 64.0       # Max frequency in Hz
    sample_rate: float = 256.0        # Samples per second

print("ArrayDecoderConfig defined")

# ==============================================================================
# SYNTHETIC SIGNAL GENERATION
# ==============================================================================

class SyntheticSignalDataset(Dataset):
    """
    Generates synthetic signals with known frequency content.
    
    Signal types:
    - Single sinusoid (tests individual band)
    - Multi-frequency (tests band separation)
    - Chirp (frequency sweep)
    - Transient (impulse + decay)
    """
    
    def __init__(
        self,
        num_samples: int,
        signal_length: int,
        max_frequency: float = 64.0,
        sample_rate: float = 256.0,
        signal_type: str = "mixed"
    ):
        self.num_samples = num_samples
        self.signal_length = signal_length
        self.max_frequency = max_frequency
        self.sample_rate = sample_rate
        self.signal_type = signal_type
        
        # Pregenerate signals for consistency
        self.signals = []
        self.metadata = []
        
        for _ in range(num_samples):
            signal, meta = self._generate_signal()
            self.signals.append(signal)
            self.metadata.append(meta)
    
    def _generate_signal(self) -> Tuple[torch.Tensor, Dict]:
        """Generate a single synthetic signal."""
        t = torch.linspace(0, 1, self.signal_length)
        
        if self.signal_type == "single":
            # Single frequency sinusoid
            freq = torch.rand(1).item() * self.max_frequency
            amp = 0.5 + 0.5 * torch.rand(1).item()
            phase = 2 * math.pi * torch.rand(1).item()
            signal = amp * torch.sin(2 * math.pi * freq * t + phase)
            meta = {"type": "single", "frequencies": [freq], "amplitudes": [amp]}
            
        elif self.signal_type == "multi":
            # Multiple frequencies
            num_freqs = torch.randint(2, 5, (1,)).item()
            freqs = torch.rand(num_freqs) * self.max_frequency
            amps = 0.3 + 0.4 * torch.rand(num_freqs)
            phases = 2 * math.pi * torch.rand(num_freqs)
            
            signal = torch.zeros(self.signal_length)
            for f, a, p in zip(freqs, amps, phases):
                signal += a * torch.sin(2 * math.pi * f * t + p)
            signal = signal / (signal.abs().max() + 1e-6)  # Normalize
            meta = {"type": "multi", "frequencies": freqs.tolist(), "amplitudes": amps.tolist()}
            
        elif self.signal_type == "chirp":
            # Frequency sweep
            f0 = torch.rand(1).item() * 5  # Start frequency
            f1 = 10 + torch.rand(1).item() * (self.max_frequency - 10)  # End frequency
            phase = f0 * t + (f1 - f0) * t ** 2 / 2
            signal = torch.sin(2 * math.pi * phase)
            meta = {"type": "chirp", "f0": f0, "f1": f1}
            
        elif self.signal_type == "transient":
            # Impulse with exponential decay + oscillation
            decay = 2 + torch.rand(1).item() * 8
            freq = 5 + torch.rand(1).item() * 30
            signal = torch.exp(-decay * t) * torch.sin(2 * math.pi * freq * t)
            meta = {"type": "transient", "decay": decay, "frequency": freq}
            
        else:  # mixed
            # Random selection
            choice = torch.randint(0, 4, (1,)).item()
            self.signal_type = ["single", "multi", "chirp", "transient"][choice]
            signal, meta = self._generate_signal()
            self.signal_type = "mixed"
        
        return signal, meta
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (input_signal, target_signal) - same for reconstruction."""
        signal = self.signals[idx]
        # Add channel dimension: [T] -> [T, 1]
        signal = signal.unsqueeze(-1)
        return signal, signal.clone()

print("SyntheticSignalDataset defined")

# ==============================================================================
# SPECTRAL DECOMPOSER (Signal -> Bands)
# ==============================================================================

class SpectralDecomposer(nn.Module):
    """
    Decomposes 1D signal into frequency bands using learnable filters.
    
    Each band uses a different kernel size to capture different frequencies:
    - Large kernel = low frequency (smooth patterns)
    - Small kernel = high frequency (sharp details)
    """
    
    def __init__(self, signal_dim: int, num_bands: int, band_dim: int):
        super().__init__()
        self.signal_dim = signal_dim
        self.num_bands = num_bands
        self.band_dim = band_dim
        
        # Kernel sizes: decreasing for higher frequency bands
        kernel_sizes = [31, 21, 15, 11, 7, 5, 3][:num_bands]
        
        # Per-band convolutions
        self.band_convs = nn.ModuleList([
            nn.Conv1d(signal_dim, band_dim, kernel_size=k, padding=k//2)
            for k in kernel_sizes
        ])
        
        # Band normalization
        self.band_norms = nn.ModuleList([
            nn.LayerNorm(band_dim) for _ in range(num_bands)
        ])
    
    def forward(self, x: torch.Tensor) -> Dict[int, torch.Tensor]:
        """
        Decompose signal into bands.
        
        Args:
            x: [B, T, D] signal (batch, time, features)
            
        Returns:
            Dict of band tensors, each [B, T, band_dim]
        """
        B, T, D = x.shape
        
        # Transpose for conv1d: [B, T, D] -> [B, D, T]
        x_conv = x.transpose(1, 2)
        
        bands = {}
        for band_idx, (conv, norm) in enumerate(zip(self.band_convs, self.band_norms)):
            # Apply convolution
            h = conv(x_conv)  # [B, band_dim, T]
            h = h.transpose(1, 2)  # [B, T, band_dim]
            h = norm(h)
            bands[band_idx] = h
        
        return bands

print("SpectralDecomposer defined")

# ==============================================================================
# PER-POSITION TEMPORAL MEMORY (Fixed Attention)
# ==============================================================================

class PerPositionTemporalMemory(nn.Module):
    """
    Each position attends ONLY to its own history.
    
    This is NOT standard attention - position t sees [t at T-1, T-2, ..., T-H]
    where H is history_length.
    
    Attention weights are FIXED (exponential decay), not learned.
    This separates "structure" from "learning" - we test if the structure works.
    """
    
    def __init__(self, band_dim: int, history_length: int, decay_rate: float = 0.9):
        super().__init__()
        self.band_dim = band_dim
        self.history_length = history_length
        self.decay_rate = decay_rate
        
        # Fixed exponential decay weights (not learned)
        decay_weights = torch.tensor([decay_rate ** i for i in range(history_length)])
        decay_weights = decay_weights / decay_weights.sum()  # Normalize
        self.register_buffer("decay_weights", decay_weights)
        
        # Value projection (this IS learned - transforms history features)
        self.value_proj = nn.Linear(band_dim, band_dim)
        
        # Output projection
        self.out_proj = nn.Linear(band_dim, band_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply per-position temporal attention with fixed weights.
        
        Args:
            x: [B, T, D] band features
            
        Returns:
            [B, T, D] attended features
        """
        B, T, D = x.shape
        H = self.history_length
        
        # Pad for history: we need H frames before each position
        # Pad at the beginning with zeros
        x_padded = F.pad(x, (0, 0, H, 0), mode='constant', value=0)  # [B, T+H, D]
        
        # For each position t, gather its history [t-H, t-H+1, ..., t-1]
        # Use unfold to create sliding windows
        # x_padded: [B, T+H, D]
        # We want windows of size H ending at each position
        
        # Reshape for gathering: [B, T+H, D] -> [B, D, T+H]
        x_t = x_padded.transpose(1, 2)  # [B, D, T+H]
        
        # Unfold: creates [B, D, T, H] where last dim is history window
        x_windows = x_t.unfold(2, H, 1)  # [B, D, T, H]
        
        # Transpose to [B, T, H, D]
        x_windows = x_windows.permute(0, 2, 3, 1)  # [B, T, H, D]
        
        # Apply value projection to history
        v = self.value_proj(x_windows)  # [B, T, H, D]
        
        # Apply fixed attention weights (exponential decay)
        # decay_weights: [H] -> [1, 1, H, 1] for broadcasting
        weights = self.decay_weights.view(1, 1, H, 1)
        
        # Weighted sum over history dimension
        attended = (v * weights).sum(dim=2)  # [B, T, D]
        
        # Output projection
        output = self.out_proj(attended)
        
        return output

print("PerPositionTemporalMemory defined (fixed attention)")

# ==============================================================================
# NEIGHBOR ATTENTION (Local Spatial - adapted for 1D)
# ==============================================================================

class NeighborAttention1D(nn.Module):
    """
    Attends to adjacent positions (neighbors in 1D signal).
    
    Each position t sees [t-1, t, t+1] - local context.
    Uses fixed uniform weights initially.
    """
    
    def __init__(self, band_dim: int, kernel_size: int = 3):
        super().__init__()
        self.band_dim = band_dim
        self.kernel_size = kernel_size
        
        # Value projection
        self.value_proj = nn.Linear(band_dim, band_dim)
        
        # Fixed uniform weights for neighbors
        weights = torch.ones(kernel_size) / kernel_size
        self.register_buffer("neighbor_weights", weights)
        
        # Output projection
        self.out_proj = nn.Linear(band_dim, band_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply neighbor attention.
        
        Args:
            x: [B, T, D] band features
            
        Returns:
            [B, T, D] features with local context
        """
        B, T, D = x.shape
        K = self.kernel_size
        
        # Pad for neighbors
        pad = K // 2
        x_padded = F.pad(x, (0, 0, pad, pad), mode='replicate')  # [B, T+K-1, D]
        
        # Unfold to get neighbor windows
        x_t = x_padded.transpose(1, 2)  # [B, D, T+K-1]
        x_windows = x_t.unfold(2, K, 1)  # [B, D, T, K]
        x_windows = x_windows.permute(0, 2, 3, 1)  # [B, T, K, D]
        
        # Value projection
        v = self.value_proj(x_windows)  # [B, T, K, D]
        
        # Fixed neighbor weights
        weights = self.neighbor_weights.view(1, 1, K, 1)
        
        # Weighted sum
        attended = (v * weights).sum(dim=2)  # [B, T, D]
        
        # Output
        output = self.out_proj(attended)
        
        return output

print("NeighborAttention1D defined")

# ==============================================================================
# BAND PROCESSOR (Per-band processing layer)
# ==============================================================================

class BandProcessor(nn.Module):
    """
    Processes a single frequency band with:
    1. Per-position temporal memory
    2. Neighbor attention
    3. Feedforward
    """
    
    def __init__(self, band_dim: int, history_length: int, decay_rate: float, dropout: float = 0.1):
        super().__init__()
        
        # Temporal memory (each position sees its own history)
        self.temporal = PerPositionTemporalMemory(band_dim, history_length, decay_rate)
        
        # Neighbor attention (local context)
        self.neighbor = NeighborAttention1D(band_dim)
        
        # Feedforward
        self.ffn = nn.Sequential(
            nn.Linear(band_dim, band_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(band_dim * 2, band_dim),
            nn.Dropout(dropout)
        )
        
        # Layer norms
        self.norm1 = nn.LayerNorm(band_dim)
        self.norm2 = nn.LayerNorm(band_dim)
        self.norm3 = nn.LayerNorm(band_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process band features."""
        # Temporal memory
        x = x + self.temporal(self.norm1(x))
        
        # Neighbor attention
        x = x + self.neighbor(self.norm2(x))
        
        # Feedforward
        x = x + self.ffn(self.norm3(x))
        
        return x

print("BandProcessor defined")

# ==============================================================================
# SPECTRAL RECONSTRUCTOR (Bands -> Signal)
# ==============================================================================

class SpectralReconstructor(nn.Module):
    """
    Reconstructs signal from frequency bands.
    
    This is the LEARNABLE decoder - takes band representations
    and outputs reconstructed signal.
    """
    
    def __init__(self, num_bands: int, band_dim: int, signal_dim: int):
        super().__init__()
        self.num_bands = num_bands
        self.band_dim = band_dim
        self.signal_dim = signal_dim
        
        # Per-band projection back to signal space
        self.band_projs = nn.ModuleList([
            nn.Linear(band_dim, signal_dim * 4) for _ in range(num_bands)
        ])
        
        # Mixing layer (combines all bands)
        self.mix = nn.Sequential(
            nn.Linear(signal_dim * 4 * num_bands, signal_dim * 8),
            nn.GELU(),
            nn.Linear(signal_dim * 8, signal_dim * 4),
            nn.GELU(),
            nn.Linear(signal_dim * 4, signal_dim)
        )
    
    def forward(self, bands: Dict[int, torch.Tensor]) -> torch.Tensor:
        """
        Reconstruct signal from bands.
        
        Args:
            bands: Dict of [B, T, band_dim] tensors
            
        Returns:
            [B, T, signal_dim] reconstructed signal
        """
        # Project each band
        projected = []
        for band_idx in range(self.num_bands):
            if band_idx in bands:
                projected.append(self.band_projs[band_idx](bands[band_idx]))
        
        # Concatenate all bands
        concat = torch.cat(projected, dim=-1)  # [B, T, signal_dim * 4 * num_bands]
        
        # Mix and output
        output = self.mix(concat)  # [B, T, signal_dim]
        
        return output

print("SpectralReconstructor defined")

# ==============================================================================
# FULL ARRAY DECODER MODEL
# ==============================================================================

class ArrayDecoderSpectral(nn.Module):
    """
    Complete Array Decoder with Spectral Bands.
    
    Architecture:
    1. SpectralDecomposer: Signal -> 7 frequency bands
    2. Per-band BandProcessor: Temporal memory + neighbor attention
    3. SpectralReconstructor: Bands -> Signal
    
    Training:
    - Fixed attention weights (exponential decay)
    - Learnable decomposer, processors, and reconstructor
    - Loss: MSE + spectral loss
    """
    
    def __init__(self, config: ArrayDecoderConfig):
        super().__init__()
        self.config = config
        
        # Spectral decomposition
        self.decomposer = SpectralDecomposer(
            config.signal_dim,
            config.num_bands,
            config.band_dim
        )
        
        # Per-band processors with different decay rates
        # Low bands: slower decay (longer memory matters)
        # High bands: faster decay (recent memory matters)
        decay_rates = [0.95, 0.92, 0.88, 0.85, 0.80, 0.75, 0.70][:config.num_bands]
        
        self.band_processors = nn.ModuleList([
            BandProcessor(
                config.band_dim,
                config.history_length,
                decay_rates[i],
                config.dropout
            )
            for i in range(config.num_bands)
        ])
        
        # Reconstruction
        self.reconstructor = SpectralReconstructor(
            config.num_bands,
            config.band_dim,
            config.signal_dim
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: [B, T, D] input signal
            
        Returns:
            Dict with 'output' and 'bands'
        """
        # Decompose into bands
        bands = self.decomposer(x)
        
        # Process each band
        processed_bands = {}
        for band_idx in range(self.config.num_bands):
            processed_bands[band_idx] = self.band_processors[band_idx](bands[band_idx])
        
        # Reconstruct
        output = self.reconstructor(processed_bands)
        
        return {
            'output': output,
            'bands': processed_bands,
            'input_bands': bands
        }

print("ArrayDecoderSpectral defined")

# ==============================================================================
# LOSS FUNCTIONS
# ==============================================================================

class SpectralLoss(nn.Module):
    """
    Combined MSE + spectral domain loss.
    
    Spectral loss compares FFT magnitudes to ensure frequency content is preserved.
    """
    
    def __init__(self, spectral_weight: float = 0.1):
        super().__init__()
        self.spectral_weight = spectral_weight
        self.mse = nn.MSELoss()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss.
        
        Args:
            pred: [B, T, D] predicted signal
            target: [B, T, D] target signal
            
        Returns:
            Dict with 'total', 'mse', 'spectral' losses
        """
        # MSE loss
        mse_loss = self.mse(pred, target)
        
        # Spectral loss (compare FFT magnitudes)
        pred_fft = torch.fft.rfft(pred.squeeze(-1), dim=-1)
        target_fft = torch.fft.rfft(target.squeeze(-1), dim=-1)
        
        pred_mag = pred_fft.abs()
        target_mag = target_fft.abs()
        
        spectral_loss = self.mse(pred_mag, target_mag)
        
        # Total loss
        total_loss = mse_loss + self.spectral_weight * spectral_loss
        
        return {
            'total': total_loss,
            'mse': mse_loss,
            'spectral': spectral_loss
        }

print("SpectralLoss defined")

# ==============================================================================
# TRAINING LOOP
# ==============================================================================

def train_array_decoder(
    model: ArrayDecoderSpectral,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: ArrayDecoderConfig,
    num_epochs: int = 100
) -> Dict[str, List[float]]:
    """Train the array decoder."""
    
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    loss_fn = SpectralLoss(spectral_weight=0.1)
    
    history = {'train_loss': [], 'val_loss': [], 'train_mse': [], 'val_mse': []}
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_losses = []
        train_mses = []
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            losses = loss_fn(outputs['output'], targets)
            
            losses['total'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_losses.append(losses['total'].item())
            train_mses.append(losses['mse'].item())
        
        scheduler.step()
        
        # Validation
        model.eval()
        val_losses = []
        val_mses = []
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                outputs = model(inputs)
                losses = loss_fn(outputs['output'], targets)
                
                val_losses.append(losses['total'].item())
                val_mses.append(losses['mse'].item())
        
        # Record history
        train_loss = sum(train_losses) / len(train_losses)
        val_loss = sum(val_losses) / len(val_losses)
        train_mse = sum(train_mses) / len(train_mses)
        val_mse = sum(val_mses) / len(val_mses)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_mse'].append(train_mse)
        history['val_mse'].append(val_mse)
        
        # Logging
        if epoch % 10 == 0 or epoch == num_epochs - 1:
            print(f"Epoch {epoch:3d}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}, "
                  f"train_mse={train_mse:.6f}, val_mse={val_mse:.6f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
    
    print(f"\nBest validation loss: {best_val_loss:.6f}")
    return history

print("Training function defined")

# ==============================================================================
# VISUALIZATION
# ==============================================================================

def visualize_reconstruction(model: ArrayDecoderSpectral, dataset: SyntheticSignalDataset, num_samples: int = 4):
    """Visualize reconstruction quality."""
    model.eval()
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 3 * num_samples))
    
    for i in range(num_samples):
        input_signal, target = dataset[i]
        input_signal = input_signal.unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(input_signal)
        
        pred = output['output'].squeeze().cpu().numpy()
        target = target.squeeze().numpy()
        
        # Plot input
        axes[i, 0].plot(target, 'b-', label='Target')
        axes[i, 0].set_title(f'Sample {i}: Input Signal')
        axes[i, 0].set_ylabel('Amplitude')
        
        # Plot reconstruction
        axes[i, 1].plot(target, 'b-', alpha=0.5, label='Target')
        axes[i, 1].plot(pred, 'r-', label='Predicted')
        axes[i, 1].set_title('Reconstruction')
        axes[i, 1].legend()
        
        # Plot error
        error = target - pred
        axes[i, 2].plot(error, 'g-')
        axes[i, 2].set_title(f'Error (MSE={np.mean(error**2):.6f})')
        axes[i, 2].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('reconstruction_results.png', dpi=150)
    plt.show()
    print("Saved reconstruction_results.png")

def visualize_band_activations(model: ArrayDecoderSpectral, dataset: SyntheticSignalDataset):
    """Visualize which bands activate for different signals."""
    model.eval()
    
    # Get one sample
    input_signal, _ = dataset[0]
    input_signal = input_signal.unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(input_signal)
    
    bands = output['input_bands']
    
    fig, axes = plt.subplots(model.config.num_bands + 1, 1, figsize=(12, 2 * (model.config.num_bands + 1)))
    
    # Plot input
    axes[0].plot(input_signal.squeeze().cpu().numpy())
    axes[0].set_title('Input Signal')
    axes[0].set_ylabel('Amplitude')
    
    # Plot each band's energy
    for band_idx in range(model.config.num_bands):
        band_energy = bands[band_idx].squeeze().cpu().numpy()
        band_energy = np.mean(band_energy ** 2, axis=-1)  # Energy per time step
        axes[band_idx + 1].plot(band_energy)
        axes[band_idx + 1].set_title(f'Band {band_idx} Energy')
        axes[band_idx + 1].set_ylabel('Energy')
    
    plt.tight_layout()
    plt.savefig('band_activations.png', dpi=150)
    plt.show()
    print("Saved band_activations.png")

print("Visualization functions defined")

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("EXPERIMENT 027: Array Decoder with Spectral Bands")
    print("="*60)
    
    # Configuration
    config = ArrayDecoderConfig(
        signal_length=256,
        signal_dim=1,
        num_bands=7,
        band_dim=32,
        history_length=16,
        batch_size=32,
        learning_rate=1e-3,
        num_epochs=100
    )
    
    print(f"\nConfiguration:")
    print(f"  Signal length: {config.signal_length}")
    print(f"  Num bands: {config.num_bands}")
    print(f"  Band dim: {config.band_dim}")
    print(f"  History length: {config.history_length}")
    
    # Create datasets
    print("\nCreating datasets...")
    train_dataset = SyntheticSignalDataset(
        num_samples=5000,
        signal_length=config.signal_length,
        signal_type="mixed"
    )
    val_dataset = SyntheticSignalDataset(
        num_samples=500,
        signal_length=config.signal_length,
        signal_type="mixed"
    )
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Create model
    model = ArrayDecoderSpectral(config)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {num_params:,}")
    
    # Train
    print("\nStarting training...")
    history = train_array_decoder(model, train_loader, val_loader, config)
    
    # Visualize
    print("\nGenerating visualizations...")
    visualize_reconstruction(model, val_dataset)
    visualize_band_activations(model, val_dataset)
    
    print("\n" + "="*60)
    print("EXPERIMENT 027 COMPLETE")
    print("="*60)
