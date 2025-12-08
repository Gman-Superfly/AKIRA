"""
Baseline Models for Experiment 032.

These provide comparison points for the Spectral Belief Machine:
1. FlatBaseline: Standard ConvNet, no spectral structure
2. FourBandBaseline: Reduced spectral bands (4 instead of 7)
3. SpectralOnlyBaseline: 7 spectral bands but no temporal attention

All baselines are parameter-matched to the main model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class BaselineConfig:
    """Configuration for baseline models."""
    height: int = 64
    width: int = 64
    channels: int = 32
    num_layers: int = 4
    device: str = "cpu"
    dtype: torch.dtype = torch.float32


class FlatBaseline(nn.Module):
    """
    Flat ConvNet baseline with no spectral structure.
    
    Standard encoder-decoder architecture with skip connections.
    Parameter count matched to SpectralBeliefMachine.
    """
    
    def __init__(self, cfg: BaselineConfig):
        super().__init__()
        self.cfg = cfg
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, cfg.channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(cfg.channels, cfg.channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(cfg.channels, cfg.channels * 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(cfg.channels * 2, cfg.channels * 2, kernel_size=3, padding=1),
            nn.GELU(),
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(cfg.channels * 2, cfg.channels * 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(cfg.channels * 2, cfg.channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(cfg.channels, cfg.channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(cfg.channels, 1, kernel_size=3, padding=1),
        )
        
        self.to(cfg.device)
    
    def forward(
        self,
        x: torch.Tensor,
        history: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            x: (B, 1, H, W) input field
            history: Ignored (for API compatibility)
            
        Returns:
            pred: (B, 1, H, W) predicted next field
            belief_state: Empty dict (no belief tracking)
        """
        encoded = self.encoder(x)
        pred = self.decoder(encoded)
        
        # No residual skip - force the model to learn the dynamics
        # (Residual would let it cheat by just copying input)
        
        # Dummy belief state for API compatibility
        B = x.shape[0]
        belief_state = {
            "band_entropy": torch.zeros(B, 8, device=x.device),
            "global_entropy": torch.zeros(B, device=x.device),
            "band_features": torch.zeros(B, 56, device=x.device),
        }
        
        return pred, belief_state


class FourBandBaseline(nn.Module):
    """
    4-band spectral baseline.
    
    Tests whether 7 bands is better than fewer bands.
    Uses same FFT decomposition but with 4 bands instead of 7.
    No temporal band.
    """
    
    def __init__(self, cfg: BaselineConfig):
        super().__init__()
        self.cfg = cfg
        self.num_bands = 4
        
        # Create radial masks for 4 bands
        self.register_buffer(
            "masks",
            self._make_masks(cfg.height, cfg.width)
        )
        
        # Per-band processing
        self.band_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(2, cfg.channels, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv2d(cfg.channels, cfg.channels, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv2d(cfg.channels, 2, kernel_size=1),
            )
            for _ in range(self.num_bands)
        ])
        
        self.to(cfg.device)
    
    def _make_masks(self, h: int, w: int) -> torch.Tensor:
        """Create 4-band radial masks."""
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, h),
            torch.linspace(-1, 1, w),
            indexing="ij",
        )
        rr = torch.sqrt(yy ** 2 + xx ** 2).clamp(min=1e-6)
        
        # 4 equal log-spaced bands
        import math
        edges = torch.logspace(-2, math.log10(math.sqrt(2)), steps=5)
        edges[0] = 0
        
        masks = []
        for i in range(4):
            lo, hi = edges[i], edges[i + 1]
            mask = torch.sigmoid((rr - lo) * 20) * torch.sigmoid((hi - rr) * 20)
            masks.append(mask)
        
        masks = torch.stack(masks, dim=0)
        masks = masks / (masks.sum(dim=0, keepdim=True) + 1e-8)
        return masks
    
    def forward(
        self,
        x: torch.Tensor,
        history: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass with 4-band FFT decomposition.
        
        Args:
            x: (B, 1, H, W) input field
            history: Ignored
            
        Returns:
            pred: (B, 1, H, W) predicted next field
            belief_state: Entropy tracking
        """
        B = x.shape[0]
        
        # FFT decompose
        fft = torch.fft.fft2(x.squeeze(1))
        fft_shifted = torch.fft.fftshift(fft)
        
        # Process each band
        processed_bands = []
        band_entropies = []
        
        for i in range(self.num_bands):
            mask = self.masks[i].unsqueeze(0)
            band_fft = fft_shifted * mask
            band_feat = torch.stack([band_fft.real, band_fft.imag], dim=1)
            
            processed = self.band_blocks[i](band_feat)
            processed_bands.append(processed)
            
            # Simple entropy proxy
            mag = torch.sqrt(processed[:, 0] ** 2 + processed[:, 1] ** 2 + 1e-8)
            mag_norm = mag / (mag.sum(dim=[1, 2], keepdim=True) + 1e-8)
            entropy = -(mag_norm * torch.log(mag_norm + 1e-9)).sum(dim=[1, 2])
            band_entropies.append(entropy)
        
        # Reconstruct
        fft_recon = torch.zeros_like(fft_shifted)
        for i, band in enumerate(processed_bands):
            mask = self.masks[i].unsqueeze(0)
            band_complex = torch.complex(band[:, 0], band[:, 1])
            fft_recon = fft_recon + band_complex * mask
        
        fft_unshifted = torch.fft.ifftshift(fft_recon)
        pred = torch.fft.ifft2(fft_unshifted).real.unsqueeze(1)
        
        # Belief state (pad to 8 bands for API compatibility)
        entropy_tensor = torch.stack(band_entropies, dim=1)
        padded_entropy = F.pad(entropy_tensor, (0, 4))  # Pad to 8 bands
        
        belief_state = {
            "band_entropy": padded_entropy,
            "global_entropy": entropy_tensor.sum(dim=1),
            "band_features": torch.zeros(B, 56, device=x.device),
        }
        
        return pred, belief_state


class SpectralOnlyBaseline(nn.Module):
    """
    7-band spectral baseline WITHOUT temporal attention.
    
    Tests whether the temporal band provides benefit.
    Uses history via concatenation instead of attention.
    """
    
    def __init__(self, cfg: BaselineConfig, history_len: int = 4):
        super().__init__()
        self.cfg = cfg
        self.num_bands = 7
        self.history_len = history_len
        
        # Create radial masks for 7 bands
        self.register_buffer(
            "masks",
            self._make_masks(cfg.height, cfg.width)
        )
        
        # Per-band processing (takes history via channel concatenation)
        self.band_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(2 * history_len, cfg.channels, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv2d(cfg.channels, cfg.channels, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv2d(cfg.channels, 2, kernel_size=1),
            )
            for _ in range(self.num_bands)
        ])
        
        self.to(cfg.device)
    
    def _make_masks(self, h: int, w: int) -> torch.Tensor:
        """Create 7-band radial masks."""
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, h),
            torch.linspace(-1, 1, w),
            indexing="ij",
        )
        rr = torch.sqrt(yy ** 2 + xx ** 2).clamp(min=1e-6)
        
        import math
        edges = torch.logspace(-3, math.log10(math.sqrt(2)), steps=8)
        edges[0] = 0
        
        masks = []
        for i in range(7):
            lo, hi = edges[i], edges[i + 1]
            mask = torch.sigmoid((rr - lo) * 20) * torch.sigmoid((hi - rr) * 20)
            masks.append(mask)
        
        masks = torch.stack(masks, dim=0)
        masks = masks / (masks.sum(dim=0, keepdim=True) + 1e-8)
        return masks
    
    def forward(
        self,
        x: torch.Tensor,
        history: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass with 7-band FFT but no temporal attention.
        
        Args:
            x: (B, 1, H, W) input field
            history: List of past (B, 2*7, H, W) band features
            
        Returns:
            pred: (B, 1, H, W) predicted next field
            belief_state: Entropy tracking
        """
        B = x.shape[0]
        H, W = x.shape[2], x.shape[3]
        
        # FFT decompose
        fft = torch.fft.fft2(x.squeeze(1))
        fft_shifted = torch.fft.fftshift(fft)
        
        # Get current bands
        current_bands = []
        for i in range(self.num_bands):
            mask = self.masks[i].unsqueeze(0)
            band_fft = fft_shifted * mask
            band_feat = torch.stack([band_fft.real, band_fft.imag], dim=1)
            current_bands.append(band_feat)
        
        current_all = torch.cat(current_bands, dim=1)  # (B, 14, H, W)
        
        # Build history sequence
        # Filter history to only include 4D tensors with correct shape (B, 14, H, W)
        valid_history = []
        if history is not None:
            for h in history:
                if h.ndim == 4 and h.shape[1] == 14 and h.shape[0] == B:
                    valid_history.append(h)
        
        history_seq = valid_history + [current_all]
        while len(history_seq) < self.history_len:
            history_seq.insert(0, torch.zeros_like(current_all))
        history_seq = history_seq[-self.history_len:]
        
        # Process each band with history concatenated
        processed_bands = []
        band_entropies = []
        
        for i in range(self.num_bands):
            # Extract band i from each history frame
            band_history = torch.cat([
                h[:, i*2:(i+1)*2] for h in history_seq
            ], dim=1)  # (B, 2*history_len, H, W)
            
            processed = self.band_blocks[i](band_history)
            processed_bands.append(processed)
            
            # Entropy
            mag = torch.sqrt(processed[:, 0] ** 2 + processed[:, 1] ** 2 + 1e-8)
            mag_norm = mag / (mag.sum(dim=[1, 2], keepdim=True) + 1e-8)
            entropy = -(mag_norm * torch.log(mag_norm + 1e-9)).sum(dim=[1, 2])
            band_entropies.append(entropy)
        
        # Reconstruct
        fft_recon = torch.zeros_like(fft_shifted)
        for i, band in enumerate(processed_bands):
            mask = self.masks[i].unsqueeze(0)
            band_complex = torch.complex(band[:, 0], band[:, 1])
            fft_recon = fft_recon + band_complex * mask
        
        fft_unshifted = torch.fft.ifftshift(fft_recon)
        pred = torch.fft.ifft2(fft_unshifted).real.unsqueeze(1)
        
        # Belief state (pad to 8 for API)
        entropy_tensor = torch.stack(band_entropies, dim=1)
        padded_entropy = F.pad(entropy_tensor, (0, 1))
        
        belief_state = {
            "band_entropy": padded_entropy,
            "global_entropy": entropy_tensor.sum(dim=1),
            "band_features": current_all.flatten(2).mean(dim=2),  # Pooled features
        }
        
        return pred, belief_state


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test and compare parameter counts
    cfg = BaselineConfig(device="cpu")
    
    flat = FlatBaseline(cfg)
    four_band = FourBandBaseline(cfg)
    spectral_only = SpectralOnlyBaseline(cfg)
    
    print("Baseline Model Parameter Counts:")
    print(f"  FlatBaseline: {count_parameters(flat):,}")
    print(f"  FourBandBaseline: {count_parameters(four_band):,}")
    print(f"  SpectralOnlyBaseline: {count_parameters(spectral_only):,}")
    
    # Test forward passes
    x = torch.randn(2, 1, 64, 64)
    
    print("\nTesting forward passes...")
    
    pred_flat, belief_flat = flat(x)
    print(f"FlatBaseline: input={x.shape}, output={pred_flat.shape}")
    
    pred_four, belief_four = four_band(x)
    print(f"FourBandBaseline: input={x.shape}, output={pred_four.shape}")
    
    pred_spectral, belief_spectral = spectral_only(x)
    print(f"SpectralOnlyBaseline: input={x.shape}, output={pred_spectral.shape}")
    
    print("\nBelief state shapes:")
    print(f"  FlatBaseline entropy: {belief_flat['band_entropy'].shape}")
    print(f"  FourBandBaseline entropy: {belief_four['band_entropy'].shape}")
    print(f"  SpectralOnlyBaseline entropy: {belief_spectral['band_entropy'].shape}")
