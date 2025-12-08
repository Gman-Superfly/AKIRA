"""
Spectral Belief Machine (7+1 Architecture) for Experiment 032.

This implements the theory-aligned AKIRA architecture:
- 7 spectral bands with log-spaced radial masks
- 1 temporal band with causal attention
- Wormhole cross-band communication
- Explicit belief tracking (entropy per band)
- Differential learning rates per band

Reference: SPECTRAL_BELIEF_MACHINE.md, THE_SEVEN_PLUS_ONE_ARCHITECTURE.md
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SpectralConfig:
    """Configuration for the Spectral Belief Machine."""
    # Dimensions
    height: int = 64
    width: int = 64
    num_spectral_bands: int = 7
    channels_per_band: int = 16
    history_len: int = 4
    
    # Temporal attention
    num_heads: int = 4
    
    # Wormhole
    wormhole_top_k: int = 8
    wormhole_threshold: float = 0.5
    
    # FFT windowing
    # OFF by default - windowing causes edge artifacts in loss computation
    # Turn ON for WHAT/WHERE separation experiments where magnitude invariance matters
    use_windowing: bool = False
    
    # Learning rates (relative multipliers)
    # Band 0 (DC) = slowest, Band 6 = fastest
    # These multiply the base learning rate
    band_lr_multipliers: List[float] = field(default_factory=lambda: [
        0.001,   # Band 0: DC, identity - very slow
        0.01,    # Band 1: coarse structure
        0.03,    # Band 2: medium structure
        0.1,     # Band 3: transitions (bridge)
        0.3,     # Band 4: fine structure
        1.0,     # Band 5: textures
        3.0,     # Band 6: edges, details - fastest
        0.1,     # Band 7: temporal - medium
    ])
    
    # Device
    device: str = "cpu"
    dtype: torch.dtype = torch.float32


def make_hamming_window(h: int, w: int, device: torch.device) -> torch.Tensor:
    """
    Create 2D Hamming window for heresy resistance.
    
    Windowing reduces spectral leakage artifacts.
    """
    wy = torch.hamming_window(h, device=device)
    wx = torch.hamming_window(w, device=device)
    return wy.unsqueeze(1) * wx.unsqueeze(0)


def make_radial_masks(
    h: int, 
    w: int, 
    num_bands: int, 
    device: torch.device
) -> torch.Tensor:
    """
    Create log-spaced radial masks for FFT band splitting.
    
    Band 0: DC (very low frequencies) - identity, existence
    Band 1-5: Log-spaced middle frequencies
    Band 6: High frequencies up to Nyquist - edges, details
    
    Returns:
        Tensor of shape (num_bands, H, W) with soft masks
    """
    # Normalized frequency coordinates
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, h, device=device),
        torch.linspace(-1.0, 1.0, w, device=device),
        indexing="ij",
    )
    
    # Radial distance (0 at center, sqrt(2) at corners)
    rr = torch.sqrt(yy ** 2 + xx ** 2)
    rr = torch.clamp(rr, min=1e-6)  # Avoid log(0)
    
    # Log-spaced band edges from ~0 to sqrt(2)
    # Using log scale so low bands are wider (more frequencies)
    edges = torch.logspace(-3, math.log10(math.sqrt(2)), steps=num_bands + 1, device=device)
    edges[0] = 0.0  # Ensure DC is included
    
    masks = []
    for i in range(num_bands):
        lo, hi = edges[i], edges[i + 1]
        # Soft mask with smooth transitions
        mask = torch.sigmoid((rr - lo) * 20) * torch.sigmoid((hi - rr) * 20)
        masks.append(mask)
    
    # Normalize so masks sum to ~1 at each point
    masks = torch.stack(masks, dim=0)
    masks = masks / (masks.sum(dim=0, keepdim=True) + 1e-8)
    
    return masks


class PerBandBlock(nn.Module):
    """
    Processing block for a single spectral band.
    
    Takes real+imag as 2 input channels, processes through Conv2d,
    outputs real+imag as 2 channels.
    """
    
    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2, channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, 2, kernel_size=1),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 2, H, W) - real and imag channels
        Returns:
            (B, 2, H, W) - processed real and imag
        """
        return self.net(x)


class TemporalBand(nn.Module):
    """
    Band 7: Temporal attention over sequence history.
    
    Unlike spectral bands that process features at one timestep,
    this band processes across timesteps with causal masking.
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int,
        max_len: int,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        
        # Causal mask: lower triangular
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(max_len, max_len), diagonal=1).bool()
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, T, D) - sequence of features
            
        Returns:
            output: (B, T, D) - temporally contextualized features
            entropy: (B,) - attention entropy for belief tracking
        """
        B, T, D = x.shape
        
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        
        # Reshape for multi-head attention
        Q = Q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply causal mask
        scores = scores.masked_fill(self.causal_mask[:T, :T].unsqueeze(0).unsqueeze(0), float('-inf'))
        
        # Softmax
        attn = F.softmax(scores, dim=-1)
        
        # Compute entropy for belief tracking
        # Average over heads
        attn_avg = attn.mean(dim=1)  # (B, T, T)
        # Entropy per position, then average
        entropy_per_pos = -(attn_avg * torch.log(attn_avg + 1e-9)).sum(dim=-1)
        entropy = entropy_per_pos.mean(dim=1)  # (B,)
        
        # Apply attention
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        out = self.out_proj(out)
        
        return out, entropy


class WormholeInterconnect(nn.Module):
    """
    Cross-band communication via wormhole attention.
    
    Implements:
    - Complementary pairs: (0<->6), (1<->5), (2<->4)
    - Bridge band 3 -> all
    - Temporal band 7 -> all spectral
    
    Uses cosine similarity on hypersphere with top-k sparse selection.
    """
    
    def __init__(
        self,
        dim: int,
        top_k: int = 8,
        threshold: float = 0.5,
    ):
        super().__init__()
        self.dim = dim
        self.top_k = top_k
        self.threshold = threshold
        
        # Learnable projections for each band
        self.q_projs = nn.ModuleList([nn.Linear(dim, dim) for _ in range(8)])
        self.k_projs = nn.ModuleList([nn.Linear(dim, dim) for _ in range(8)])
        self.v_projs = nn.ModuleList([nn.Linear(dim, dim) for _ in range(8)])
        
        # Complementary pairs
        self.pairs = [(0, 6), (1, 5), (2, 4)]
        
    def _cross_attend(
        self,
        q_band: torch.Tensor,
        k_band: torch.Tensor,
        v_band: torch.Tensor,
        q_idx: int,
        k_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Cross-attention between two bands.
        
        Returns:
            output: Attended features
            entropy: Attention entropy
        """
        B, N, D = q_band.shape
        
        Q = self.q_projs[q_idx](q_band)
        K = self.k_projs[k_idx](k_band)
        V = self.v_projs[k_idx](v_band)
        
        # Normalize onto hypersphere (geometric belief)
        Q_norm = F.normalize(Q, p=2, dim=-1)
        K_norm = F.normalize(K, p=2, dim=-1)
        
        # Cosine similarity
        scores = torch.matmul(Q_norm, K_norm.transpose(-2, -1))
        
        # Top-k sparse selection
        if self.top_k < N:
            topk_vals, topk_idx = scores.topk(self.top_k, dim=-1)
            mask = torch.zeros_like(scores).scatter_(-1, topk_idx, 1.0)
            scores = scores * mask + (1 - mask) * float('-inf')
        
        # Softmax
        attn = F.softmax(scores, dim=-1)
        
        # Entropy
        entropy = -(attn * torch.log(attn + 1e-9)).sum(dim=-1).mean(dim=1)  # (B,)
        
        # Apply attention
        out = torch.matmul(attn, V)
        
        return out, entropy
    
    def forward(
        self,
        bands: Dict[int, torch.Tensor],
    ) -> Tuple[Dict[int, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Apply wormhole cross-band communication.
        
        Args:
            bands: Dict mapping band index to features (B, N, D)
            
        Returns:
            updated_bands: Dict with cross-attended features
            entropies: Dict of entropy values per connection
        """
        outputs = {k: v.clone() for k, v in bands.items()}
        entropies = {}
        
        # Complementary pairs (bidirectional)
        for lo, hi in self.pairs:
            if lo in bands and hi in bands:
                # Low -> High
                out_lo_to_hi, ent_lo_hi = self._cross_attend(
                    bands[hi], bands[lo], bands[lo], hi, lo
                )
                outputs[hi] = outputs[hi] + out_lo_to_hi
                entropies[f"wormhole_{lo}_{hi}"] = ent_lo_hi
                
                # High -> Low
                out_hi_to_lo, ent_hi_lo = self._cross_attend(
                    bands[lo], bands[hi], bands[hi], lo, hi
                )
                outputs[lo] = outputs[lo] + out_hi_to_lo
                entropies[f"wormhole_{hi}_{lo}"] = ent_hi_lo
        
        # Bridge band (3) -> all others
        if 3 in bands:
            for i in [0, 1, 2, 4, 5, 6]:
                if i in bands:
                    out_bridge, ent_bridge = self._cross_attend(
                        bands[i], bands[3], bands[3], i, 3
                    )
                    outputs[i] = outputs[i] + out_bridge * 0.5  # Scaled contribution
                    entropies[f"bridge_3_{i}"] = ent_bridge
        
        # Temporal band (7) -> all spectral
        if 7 in bands:
            for i in range(7):
                if i in bands:
                    out_temporal, ent_temporal = self._cross_attend(
                        bands[i], bands[7], bands[7], i, 7
                    )
                    outputs[i] = outputs[i] + out_temporal * 0.5
                    entropies[f"temporal_7_{i}"] = ent_temporal
        
        return outputs, entropies


class SpectralBeliefMachine(nn.Module):
    """
    Complete 7+1 Spectral Belief Machine.
    
    Architecture:
    1. Windowing (Hamming) for heresy resistance
    2. FFT2 -> 7 spectral bands via radial masks
    3. Per-band Conv processing with differential learning rates
    4. Temporal band with causal attention over history
    5. Wormhole cross-band communication
    6. Belief tracking (entropy per band)
    7. iFFT2 reconstruction
    """
    
    def __init__(self, cfg: SpectralConfig):
        super().__init__()
        self.cfg = cfg
        
        # Windowing (optional - OFF by default to avoid edge artifacts)
        if cfg.use_windowing:
            self.register_buffer(
                "window",
                make_hamming_window(cfg.height, cfg.width, torch.device(cfg.device))
            )
        else:
            # No windowing - use ones (identity)
            self.register_buffer(
                "window",
                torch.ones(cfg.height, cfg.width, device=torch.device(cfg.device))
            )
        
        # Radial masks for band splitting
        self.register_buffer(
            "masks",
            make_radial_masks(cfg.height, cfg.width, cfg.num_spectral_bands, torch.device(cfg.device))
        )
        
        # Per-band processing blocks
        self.band_blocks = nn.ModuleList([
            PerBandBlock(cfg.channels_per_band)
            for _ in range(cfg.num_spectral_bands)
        ])
        
        # Temporal band
        # Input: flattened band features from history
        temporal_dim = cfg.num_spectral_bands * 2 * 4  # bands * (real,imag) * pooled
        self.temporal_proj_in = nn.Linear(temporal_dim, cfg.channels_per_band * 8)
        self.temporal_band = TemporalBand(
            dim=cfg.channels_per_band * 8,
            num_heads=cfg.num_heads,
            max_len=cfg.history_len,
        )
        self.temporal_proj_out = nn.Linear(cfg.channels_per_band * 8, temporal_dim)
        
        # Wormhole interconnect (simplified version in forward pass)
        # Full wormhole with attention is too expensive for this toy experiment
        # We use direct cross-band mixing instead
        
        # Move to device
        self.to(cfg.device)
        
    def _fft_decompose(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Decompose input into spectral bands.
        
        Args:
            x: (B, 1, H, W) input field
            
        Returns:
            List of (B, 2, H, W) tensors, one per band (real, imag channels)
        """
        B = x.shape[0]
        
        # Apply window
        x_windowed = x.squeeze(1) * self.window  # (B, H, W)
        
        # FFT
        fft = torch.fft.fft2(x_windowed)
        fft_shifted = torch.fft.fftshift(fft)
        
        # Split into bands
        bands = []
        for i in range(self.cfg.num_spectral_bands):
            mask = self.masks[i].unsqueeze(0)  # (1, H, W)
            band_fft = fft_shifted * mask
            # Stack real and imag as channels
            band_feat = torch.stack([band_fft.real, band_fft.imag], dim=1)  # (B, 2, H, W)
            bands.append(band_feat)
            
        return bands
    
    def _fft_reconstruct(self, bands: List[torch.Tensor]) -> torch.Tensor:
        """
        Reconstruct field from spectral bands.
        
        Args:
            bands: List of (B, 2, H, W) tensors (real, imag channels)
            
        Returns:
            (B, 1, H, W) reconstructed field
        """
        B = bands[0].shape[0]
        
        # Sum bands in frequency domain
        fft_recon = torch.zeros(B, self.cfg.height, self.cfg.width, dtype=torch.complex64, device=bands[0].device)
        
        for i, band in enumerate(bands):
            mask = self.masks[i].unsqueeze(0)
            band_complex = torch.complex(band[:, 0], band[:, 1])
            fft_recon = fft_recon + band_complex * mask
        
        # Inverse shift and FFT
        fft_unshifted = torch.fft.ifftshift(fft_recon)
        recon = torch.fft.ifft2(fft_unshifted).real
        
        return recon.unsqueeze(1)  # (B, 1, H, W)
    
    def forward(
        self,
        x: torch.Tensor,
        history: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass with belief tracking.
        
        Args:
            x: (B, 1, H, W) current field
            history: Optional list of past band features for temporal attention
            
        Returns:
            pred: (B, 1, H, W) predicted next field
            belief_state: Dict containing:
                - band_entropy: (B, 8) entropy per band
                - global_entropy: (B,) sum of band entropies
                - band_features: List for history tracking
                - wormhole_entropy: Dict of cross-band entropies
        """
        B = x.shape[0]
        belief_state = {}
        
        # 1. Spectral decomposition
        bands = self._fft_decompose(x)
        
        # 2. Per-band processing
        processed_bands = []
        band_entropies = []
        
        for i, (band, block) in enumerate(zip(bands, self.band_blocks)):
            processed = block(band)
            processed_bands.append(processed)
            
            # Compute entropy of band magnitudes (simple proxy)
            mag = torch.sqrt(processed[:, 0] ** 2 + processed[:, 1] ** 2 + 1e-8)
            mag_norm = mag / (mag.sum(dim=[1, 2], keepdim=True) + 1e-8)
            entropy = -(mag_norm * torch.log(mag_norm + 1e-9)).sum(dim=[1, 2])
            band_entropies.append(entropy)
        
        # 3. Temporal band processing
        # Pool band features to vectors
        band_vecs = []
        for band in processed_bands:
            # Adaptive avg pool to 2x2, then flatten
            pooled = F.adaptive_avg_pool2d(band, (2, 2)).flatten(1)  # (B, 8)
            band_vecs.append(pooled)
        
        current_temporal_input = torch.cat(band_vecs, dim=1)  # (B, 7*8)
        
        # Build temporal sequence from history
        # Filter history to only include tensors with matching batch size
        if history is not None and len(history) > 0:
            valid_history = [h for h in history if h.shape[0] == B]
            history_seq = valid_history + [current_temporal_input]
            history_seq = history_seq[-self.cfg.history_len:]
        else:
            history_seq = [current_temporal_input]
        
        # Pad if needed (with zeros matching current batch size)
        while len(history_seq) < self.cfg.history_len:
            history_seq.insert(0, torch.zeros_like(current_temporal_input))
        
        temporal_seq = torch.stack(history_seq, dim=1)  # (B, T, D)
        temporal_seq = self.temporal_proj_in(temporal_seq)
        
        temporal_out, temporal_entropy = self.temporal_band(temporal_seq)
        temporal_out = self.temporal_proj_out(temporal_out[:, -1])  # Take last timestep
        
        band_entropies.append(temporal_entropy)
        
        # 4. Wormhole cross-band communication (simplified)
        # For this experiment, we use a lightweight version that operates on pooled features
        # Full wormhole with spatial attention is too expensive for this toy domain
        wormhole_entropy = {}
        
        # Pool each band to a vector for cross-band mixing
        pooled_bands = {}
        for i, band in enumerate(processed_bands):
            # Pool to (B, 2, 4, 4) then flatten to (B, 32)
            pooled = F.adaptive_avg_pool2d(band, (4, 4)).flatten(1)  # (B, 32)
            pooled_bands[i] = pooled
        
        # Add temporal as band 7 (project to same dim)
        pooled_bands[7] = temporal_out[:, :32]  # Take first 32 dims
        
        # Simple cross-band mixing via complementary pairs
        # Just add scaled information from partner band
        mixing_scale = 0.1
        
        # Pairs: (0,6), (1,5), (2,4)
        for lo, hi in [(0, 6), (1, 5), (2, 4)]:
            # Mix pooled representations
            lo_contrib = pooled_bands[lo] * mixing_scale
            hi_contrib = pooled_bands[hi] * mixing_scale
            
            # Add back to spatial bands (broadcast)
            lo_bias = lo_contrib.view(B, 2, 4, 4)
            hi_bias = hi_contrib.view(B, 2, 4, 4)
            
            # Upsample and add
            processed_bands[lo] = processed_bands[lo] + F.interpolate(hi_bias, size=(self.cfg.height, self.cfg.width), mode='bilinear', align_corners=False)
            processed_bands[hi] = processed_bands[hi] + F.interpolate(lo_bias, size=(self.cfg.height, self.cfg.width), mode='bilinear', align_corners=False)
        
        # Bridge band 3 contributes to all
        bridge_bias = pooled_bands[3].view(B, 2, 4, 4) * mixing_scale * 0.5
        bridge_up = F.interpolate(bridge_bias, size=(self.cfg.height, self.cfg.width), mode='bilinear', align_corners=False)
        for i in [0, 1, 2, 4, 5, 6]:
            processed_bands[i] = processed_bands[i] + bridge_up
        
        # 5. Reconstruction
        pred = self._fft_reconstruct(processed_bands)
        
        # 6. Collect belief state
        belief_state["band_entropy"] = torch.stack(band_entropies, dim=1)  # (B, 8)
        belief_state["global_entropy"] = belief_state["band_entropy"].sum(dim=1)  # (B,)
        belief_state["band_features"] = current_temporal_input  # For history tracking
        belief_state["wormhole_entropy"] = wormhole_entropy
        
        return pred, belief_state
    
    def get_lr_groups(self, base_lr: float) -> List[dict]:
        """
        Get parameter groups with differential learning rates.
        
        Args:
            base_lr: Base learning rate
            
        Returns:
            List of param group dicts for optimizer
        """
        groups = []
        
        # Per-band blocks
        for i, block in enumerate(self.band_blocks):
            groups.append({
                "params": block.parameters(),
                "lr": base_lr * self.cfg.band_lr_multipliers[i],
                "name": f"band_{i}",
            })
        
        # Temporal band
        groups.append({
            "params": list(self.temporal_proj_in.parameters()) + 
                      list(self.temporal_band.parameters()) + 
                      list(self.temporal_proj_out.parameters()),
            "lr": base_lr * self.cfg.band_lr_multipliers[7],
            "name": "temporal",
        })
        
        # Note: Simplified wormhole has no learnable parameters
        
        return groups


if __name__ == "__main__":
    # Quick test
    cfg = SpectralConfig(device="cpu")
    model = SpectralBeliefMachine(cfg)
    
    print(f"SpectralBeliefMachine: {sum(p.numel() for p in model.parameters())} parameters")
    
    x = torch.randn(2, 1, 64, 64)
    pred, belief = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {pred.shape}")
    print(f"Band entropy shape: {belief['band_entropy'].shape}")
    print(f"Global entropy: {belief['global_entropy']}")
    
    # Test with history
    history = [torch.randn(2, 7 * 8) for _ in range(3)]
    pred2, belief2 = model(x, history)
    print(f"With history - Global entropy: {belief2['global_entropy']}")
    
    # Test LR groups
    lr_groups = model.get_lr_groups(0.001)
    print(f"\nLearning rate groups:")
    for g in lr_groups:
        print(f"  {g['name']}: lr={g['lr']:.6f}")
