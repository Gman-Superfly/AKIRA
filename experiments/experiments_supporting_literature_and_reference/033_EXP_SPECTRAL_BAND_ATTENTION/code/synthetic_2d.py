"""
Synthetic 2D Sequence Learning

Simple experiments to validate spectral attention without physics overhead.
Tasks:
1. Moving blob tracking
2. Pattern reconstruction
3. Next-frame prediction

This is the main experiment script. Run with:
    python synthetic_2d.py --config fast --grid-size 32 --steps 300
"""

import argparse
import math
import time
from typing import Dict, Tuple
import os

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from spectral_attention import SpectralBandAttention
from configs import (
    get_default_config,
    get_fast_config,
    get_turbo_config,
    get_large_config,
    get_xlarge_config,
)

PATTERN_CHOICES = [
    'blob',
    'interference',
    'switching',
    'double_slit',
    'counter_rotate',
    'chirp',
    'phase_jump',
    'noisy_motion',
    'bifurcation',
    'wave_collision',
    'concentric',
]


def _maybe_save_snapshot(
    out_dir: str,
    step: int,
    current_frame: torch.Tensor,
    next_frame: torch.Tensor,
    pred: torch.Tensor,
    stats: Dict
):
    """Save a 2x2 snapshot (current, target, pred, error) with stats overlay if matplotlib is available."""
    try:
        import matplotlib.pyplot as plt  # Local import to avoid hard dependency at import time
    except Exception as e:
        # Matplotlib not available; skip silently but inform once per run
        if step == 0:
            print("Visualization skipped: matplotlib not available.")
        return
    
    os.makedirs(out_dir, exist_ok=True)
    
    cur = current_frame.detach().cpu().numpy()
    tgt = next_frame.detach().cpu().numpy()
    prd = pred.detach().cpu().numpy()
    err = (abs(prd - tgt)).clip(0.0, 1.0)
    
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    ax = axes.ravel()
    
    im0 = ax[0].imshow(cur, cmap='magma', vmin=0.0, vmax=1.0)
    ax[0].set_title('Current')
    fig.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.04)
    
    im1 = ax[1].imshow(tgt, cmap='magma', vmin=0.0, vmax=1.0)
    ax[1].set_title('Target (t+1)')
    fig.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04)
    
    im2 = ax[2].imshow(prd, cmap='magma', vmin=0.0, vmax=1.0)
    ax[2].set_title('Prediction')
    fig.colorbar(im2, ax=ax[2], fraction=0.046, pad=0.04)
    
    im3 = ax[3].imshow(err, cmap='viridis', vmin=0.0, vmax=1.0)
    ax[3].set_title('Abs Error')
    fig.colorbar(im3, ax=ax[3], fraction=0.046, pad=0.04)
    
    for a in ax:
        a.set_xticks([])
        a.set_yticks([])
    
    # Stats overlay
    te = stats.get('temporal_entropy', None)
    tmw = stats.get('temporal_mean_weight', None)
    nmw = stats.get('neighbor_mean_weight', None)
    ne = stats.get('neighbor_entropy', None)
    whn = stats.get('wormhole_num_connections', None)
    whp = stats.get('wormhole_positions_with_connections', None)
    whs = stats.get('wormhole_sparsity', None)
    
    subtitle = (
        f"Step {step} | "
        f"T(Ent)={te:.3f}  T(MeanW)={tmw:.3f} | "
        f"N(Ent)={('NA' if ne is None else f'{ne:.3f}')}  N(MeanW)={nmw:.3f} | "
        f"WH(conns)={whn} pos={whp} sp={whs:.2e}"
    )
    fig.suptitle(subtitle, fontsize=10)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    
    out_path = os.path.join(out_dir, f"step_{step:05d}.png")
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


class SyntheticDataGenerator:
    """Generate simple 2D sequences for testing."""
    
    def __init__(self, grid_size: int = 32, device: torch.device = torch.device('cpu')):
        self.grid_size = grid_size
        self.device = device
        
        # Create coordinate grids
        x = torch.linspace(0, 1, grid_size, device=device)
        y = torch.linspace(0, 1, grid_size, device=device)
        self.Y, self.X = torch.meshgrid(y, x, indexing='ij')
    
    @staticmethod
    def _scalar_noise(t: int, offset: float = 0.0) -> float:
        """Deterministic pseudo-random scalar in [0,1)."""
        value = math.sin((t + offset) * 12.9898) * 43758.5453
        return value - math.floor(value)
    
    def _gaussian(self, cx: float, cy: float, sigma: float = 0.1) -> torch.Tensor:
        return torch.exp(-((self.X - cx) ** 2 + (self.Y - cy) ** 2) / (2 * sigma ** 2))
    
    def _normalize(self, field: torch.Tensor) -> torch.Tensor:
        field_min = field.min()
        field_max = field.max()
        return (field - field_min) / (field_max - field_min + 1e-8)
    
    def moving_blob(self, t: int, speed: float = 0.02) -> torch.Tensor:
        """Generate a moving Gaussian blob."""
        angle = t * speed * 2 * math.pi
        cx = 0.5 + 0.3 * math.cos(angle)
        cy = 0.5 + 0.3 * math.sin(angle)
        blob = self._gaussian(cx, cy, sigma=0.1)
        return blob.to(self.device)
    
    def interference_pattern(self, t: int, freq: float = 0.1) -> torch.Tensor:
        """Generate an interference pattern."""
        k = 10.0  # wavenumber
        phase = t * freq
        
        wave1 = torch.sin(k * self.X + phase)
        wave2 = torch.sin(k * self.Y + phase * 0.7)
        
        return ((wave1 + wave2) / 2 + 1) / 2  # Normalize to [0, 1]
    
    def switching_frame(self, t: int, period: int = 50) -> torch.Tensor:
        """Alternate between blob and interference every 'period' steps."""
        phase_block = (t // period) % 2
        if phase_block == 0:
            return self.moving_blob(t)
        return self.interference_pattern(t)
    
    def double_slit(self, t: int) -> torch.Tensor:
        """Simulate interference emerging from two narrow slits."""
        separation = 0.18
        phase_speed = 0.12
        freq = 35.0
        phase = t * phase_speed
        centers = [
            (0.5 - separation / 2, 0.2),
            (0.5 + separation / 2, 0.2),
        ]
        wave = torch.zeros_like(self.X)
        for cx, cy in centers:
            dist = torch.sqrt((self.X - cx) ** 2 + (self.Y - cy) ** 2) + 1e-3
            wave += torch.sin(freq * dist - phase) / dist
        return self._normalize(wave)
    
    def counter_rotating_blobs(self, t: int) -> torch.Tensor:
        """Two blobs orbiting in opposite directions to test angular momentum handling."""
        speed = 0.02
        angle = t * speed * 2 * math.pi
        cx1 = 0.5 + 0.25 * math.cos(angle)
        cy1 = 0.5 + 0.25 * math.sin(angle)
        cx2 = 0.5 + 0.25 * math.cos(-angle)
        cy2 = 0.5 + 0.25 * math.sin(-angle)
        blobs = self._gaussian(cx1, cy1, sigma=0.08) + self._gaussian(cx2, cy2, sigma=0.08)
        return self._normalize(blobs)
    
    def frequency_sweep(self, t: int) -> torch.Tensor:
        """Chirped interference whose frequency ramps over time."""
        base_freq = 3.0
        sweep = 0.015
        inst_freq = base_freq + sweep * t
        phase = 0.1 * t
        wave = torch.sin(2 * math.pi * inst_freq * self.X + phase)
        wave += torch.sin(2 * math.pi * inst_freq * 0.6 * self.Y - phase)
        return self._normalize(wave)
    
    def phase_jump_pattern(self, t: int, jump_period: int = 40) -> torch.Tensor:
        """Interference with abrupt pi phase flips."""
        freq = 0.08
        phase = t * freq
        jump = math.pi if ((t // jump_period) % 2 == 1) else 0.0
        wave = torch.sin(8 * math.pi * self.X + phase + jump)
        wave += torch.sin(8 * math.pi * self.Y - phase - jump)
        return self._normalize(wave)
    
    def noise_injected_motion(self, t: int) -> torch.Tensor:
        """Blob trajectory with independent noise on position and velocity."""
        base_speed = 0.02
        velocity_jitter = (self._scalar_noise(t, 0.37) - 0.5) * 0.04
        angle = t * (base_speed + velocity_jitter) * 2 * math.pi
        pos_jitter_x = (self._scalar_noise(t, 0.11) - 0.5) * 0.15
        pos_jitter_y = (self._scalar_noise(t, 0.53) - 0.5) * 0.15
        cx = 0.5 + 0.28 * math.cos(angle) + pos_jitter_x
        cy = 0.5 + 0.28 * math.sin(angle) + pos_jitter_y
        
        blob = self._gaussian(cx, cy, sigma=0.1)
        # Trail captures uncertainty about previous momentum
        prev_angle = (t - 1) * (base_speed + velocity_jitter) * 2 * math.pi
        trail = self._gaussian(0.5 + 0.28 * math.cos(prev_angle),
                               0.5 + 0.28 * math.sin(prev_angle),
                               sigma=0.12)
        return self._normalize(blob + 0.4 * trail)
    
    def stochastic_bifurcation(self, t: int, split_period: int = 60) -> torch.Tensor:
        """Blob that occasionally bifurcates into two branches."""
        base = self.moving_blob(t)
        phase = (t % split_period) / split_period
        if phase < 0.5:
            return base
        
        orientation = (self._scalar_noise(t // split_period, 0.77) - 0.5)
        offset = 0.2 * orientation
        branch = self._gaussian(0.5 + offset, 0.5 - offset, sigma=0.08)
        combined = base + 0.9 * branch
        return self._normalize(combined)
    
    def wave_collision(self, t: int) -> torch.Tensor:
        """Opposing planar wavefronts creating standing nodes."""
        k = 12.0
        phase = t * 0.08 * 2 * math.pi
        left_wave = torch.sin(k * self.X - phase)
        right_wave = torch.sin(k * (1 - self.X) - phase)
        frame = left_wave + right_wave
        return self._normalize(frame)
    
    def concentric_blobs(self, t: int) -> torch.Tensor:
        """Two concentric ring-like blobs that expand and contract."""
        # Center of the pattern
        cx, cy = 0.5, 0.5
        
        # Distance from center
        dist = torch.sqrt((self.X - cx) ** 2 + (self.Y - cy) ** 2)
        
        # Two rings with different phases - one expanding, one contracting
        speed = 0.03
        
        # Inner ring: contracts then expands
        inner_radius = 0.15 + 0.1 * math.sin(t * speed * 2 * math.pi)
        inner_ring = torch.exp(-((dist - inner_radius) ** 2) / (2 * 0.04 ** 2))
        
        # Outer ring: expands then contracts (opposite phase)
        outer_radius = 0.35 + 0.15 * math.sin(t * speed * 2 * math.pi + math.pi)
        outer_ring = torch.exp(-((dist - outer_radius) ** 2) / (2 * 0.05 ** 2))
        
        # Combine with different intensities
        frame = inner_ring * 1.0 + outer_ring * 0.7
        return self._normalize(frame)
    
    def frame_for_pattern(self, t: int, pattern_type: str, switch_period: int = 50) -> torch.Tensor:
        """Dispatch helper to fetch frames by pattern name."""
        if pattern_type == 'blob':
            return self.moving_blob(t)
        if pattern_type == 'interference':
            return self.interference_pattern(t)
        if pattern_type == 'switching':
            return self.switching_frame(t, period=switch_period)
        if pattern_type == 'double_slit':
            return self.double_slit(t)
        if pattern_type == 'counter_rotate':
            return self.counter_rotating_blobs(t)
        if pattern_type == 'chirp':
            return self.frequency_sweep(t)
        if pattern_type == 'phase_jump':
            return self.phase_jump_pattern(t)
        if pattern_type == 'noisy_motion':
            return self.noise_injected_motion(t)
        if pattern_type == 'bifurcation':
            return self.stochastic_bifurcation(t)
        if pattern_type == 'wave_collision':
            return self.wave_collision(t)
        if pattern_type == 'concentric':
            return self.concentric_blobs(t)
        raise ValueError(f"Unknown pattern type: {pattern_type}")
    
    def generate_sequence(self, length: int, pattern_type: str = 'blob', switch_period: int = 50) -> torch.Tensor:
        """Generate a sequence of frames."""
        frames = [
            self.frame_for_pattern(t, pattern_type=pattern_type, switch_period=switch_period)
            for t in range(length)
        ]
        return torch.stack(frames)  # [T, H, W]


class MLPPredictor(nn.Module):
    """
    Simple MLP baseline predictor for comparison with SpectralPredictor.
    
    Architecture:
        - Flattens history buffer into a single vector
        - Passes through MLP layers
        - Outputs predicted next frame
    
    This is a control to test whether the "pump cycle" and error patterns
    are specific to attention mechanisms or emerge from any predictor.
    """
    
    def __init__(self, config: Dict, device: torch.device = torch.device('cpu')):
        super().__init__()
        
        self.config = config
        self.device = device
        self.grid_size = config['gridSize']
        self.time_depth = config['timeDepth']
        self.base_dim = config.get('baseDim', 64)
        
        # History buffer: [T, H, W]
        self.history_buffer = torch.zeros(
            self.time_depth,
            self.grid_size,
            self.grid_size,
            device=device
        )
        self.history_idx = 0
        self.history_initialized = False
        
        # MLP: flatten history -> hidden -> output frame
        input_size = self.time_depth * self.grid_size * self.grid_size
        hidden_size = min(512, input_size // 4)  # Reasonable hidden size
        output_size = self.grid_size * self.grid_size
        
        self.mlp = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.Sigmoid()
        )
        
        self.to(device)
    
    def reset(self):
        """Reset history buffer."""
        self.history_buffer.zero_()
        self.history_idx = 0
        self.history_initialized = False
    
    def forward(self, frame: torch.Tensor, step: int) -> Tuple[torch.Tensor, Dict]:
        """
        Process one frame and predict next.
        
        Args:
            frame: Input frame [H, W]
            step: Current step index
        
        Returns:
            prediction: Predicted next frame [H, W]
            stats: Empty stats dict (MLP has no attention stats)
        """
        H, W = frame.shape
        
        # Initialize history with noise if needed
        if not self.history_initialized:
            for t in range(self.time_depth):
                noise_scale = 0.01 * (1.0 - t / self.time_depth)
                self.history_buffer[t] = frame.detach() + torch.randn_like(frame) * noise_scale
            self.history_initialized = True
        
        # Update history (ring buffer)
        self.history_buffer[self.history_idx] = frame.detach()
        self.history_idx = (self.history_idx + 1) % self.time_depth
        
        # Reorder history so oldest is first
        history_ordered = torch.roll(self.history_buffer, shifts=-self.history_idx, dims=0)
        
        # Flatten and pass through MLP
        x = history_ordered.reshape(-1)  # [T * H * W]
        pred_flat = self.mlp(x)  # [H * W]
        pred = pred_flat.reshape(H, W)
        
        # Return empty stats (compatible with SpectralPredictor interface)
        stats = {
            'temporal': {'entropy': 0.0, 'mean_weight': 0.0, 'weights': None},
            'neighbor': {'mean_weight': 0.0, 'neighbor_weights': None},
            'wormhole': {
                'num_connections': 0,
                'sparsity': 0.0,
                'mean_similarity': 0.0,
                'positions_with_connections': 0,
                'per_position_counts': torch.zeros(H, W, device=self.device),
            },
        }
        
        return pred, stats


class SpectralPredictor(nn.Module):
    """
    Predictor using spectral attention with FFT-based band decomposition.
    
    Architecture:
        - Temporal & Neighbor attention operate on raw 'intensity' band
        - Wormhole attention uses spectral bands (low/high freq) for pattern matching
    
    Task: Given T frames, predict the next frame.
    """
    
    def __init__(self, config: Dict, device: torch.device = torch.device('cpu')):
        super().__init__()
        
        self.config = config
        self.device = device
        self.grid_size = config['gridSize']
        self.base_dim = config['baseDim']
        self.time_depth = config['timeDepth']
        
        # Three bands: intensity (raw), low_freq, high_freq
        self.bands = ['intensity', 'low_freq', 'high_freq']
        self.band_dims = {
            'intensity': self.base_dim,
            'low_freq': self.base_dim,
            'high_freq': self.base_dim,
        }
        
        # Input projection: [H, W, 1] -> [H, W, base_dim]
        self.input_proj = nn.Linear(1, self.base_dim)
        
        # Spectral band projections (from FFT magnitudes to features)
        self.low_freq_proj = nn.Linear(1, self.base_dim)
        self.high_freq_proj = nn.Linear(1, self.base_dim)
        
        # Frequency cutoff for low/high split (fraction of Nyquist)
        self.freq_cutoff = config.get('freq_cutoff', 0.25)
        
        # Build config with band assignments:
        # - Temporal/Neighbor: intensity -> intensity (local, same-band)
        # - Wormhole: 
        #   - similarity_band = low_freq (find similar structures by low-freq matching)
        #   - value_band = intensity (retrieve intensity values from matches)
        attn_config = config.copy()
        attn_config['temporal_query_band'] = 'intensity'
        attn_config['temporal_key_band'] = 'intensity'
        attn_config['neighbor_query_band'] = 'intensity'
        attn_config['neighbor_key_band'] = 'intensity'
        attn_config['wormhole_similarity_band'] = 'low_freq'   # Match by structure
        attn_config['wormhole_value_band'] = 'intensity'       # Retrieve intensity
        
        # Spectral attention with proper band assignments
        self.spectral_attn = SpectralBandAttention(
            bands=self.bands,
            band_dims=self.band_dims,
            attn_dim=config['attnDim'],
            config=attn_config,
            device=device
        )
        
        # Output projection: [H, W, attn_dim] -> [H, W, 1]
        self.output_proj = nn.Sequential(
            nn.Linear(config['attnDim'], self.base_dim),
            nn.ReLU(),
            nn.Linear(self.base_dim, 1),
            nn.Sigmoid()
        )
        
        # History buffers for each band
        self.history_buffer = {
            band: torch.zeros(
                self.time_depth,
                self.grid_size,
                self.grid_size,
                self.base_dim,
                device=device
            )
            for band in self.bands
        }
        self.history_idx = 0
        self.history_initialized = False
        
        self.to(device)
    
    def _compute_spectral_bands(self, frame: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decompose frame into low and high frequency bands using FFT.
        
        Args:
            frame: Input frame [H, W]
            
        Returns:
            low_freq: Low frequency magnitude [H, W, 1]
            high_freq: High frequency magnitude [H, W, 1]
        """
        H, W = frame.shape
        
        # 2D FFT
        fft = torch.fft.fft2(frame)
        fft_shifted = torch.fft.fftshift(fft)
        
        # Create frequency mask (circular cutoff)
        cy, cx = H // 2, W // 2
        y_coords = torch.arange(H, device=self.device).float() - cy
        x_coords = torch.arange(W, device=self.device).float() - cx
        Y, X = torch.meshgrid(y_coords, x_coords, indexing='ij')
        
        # Normalized distance from center (0 = DC, 1 = Nyquist)
        max_freq = min(cy, cx)
        freq_dist = torch.sqrt(X**2 + Y**2) / max_freq
        
        # Low freq mask: smooth falloff at cutoff
        low_mask = torch.sigmoid(10 * (self.freq_cutoff - freq_dist))
        high_mask = 1.0 - low_mask
        
        # Apply masks and inverse FFT
        low_fft = fft_shifted * low_mask
        high_fft = fft_shifted * high_mask
        
        low_spatial = torch.fft.ifft2(torch.fft.ifftshift(low_fft)).real
        high_spatial = torch.fft.ifft2(torch.fft.ifftshift(high_fft)).real
        
        # Normalize to [0, 1] range
        low_spatial = (low_spatial - low_spatial.min()) / (low_spatial.max() - low_spatial.min() + 1e-8)
        high_spatial = (high_spatial - high_spatial.min()) / (high_spatial.max() - high_spatial.min() + 1e-8)
        
        return low_spatial.unsqueeze(-1), high_spatial.unsqueeze(-1)
    
    def reset(self):
        """Reset history buffer."""
        for band in self.bands:
            self.history_buffer[band].zero_()
        self.history_idx = 0
        self.history_initialized = False
    
    def forward(self, frame: torch.Tensor, step: int) -> Tuple[torch.Tensor, Dict]:
        """
        Process one frame and predict next.
        
        Args:
            frame: Input frame [H, W]
            step: Current step index
        
        Returns:
            prediction: Predicted next frame [H, W]
            stats: Attention statistics
        """
        H, W = frame.shape
        
        # Project raw input to intensity band
        x_raw = frame.unsqueeze(-1)  # [H, W, 1]
        x_intensity = self.input_proj(x_raw)  # [H, W, base_dim]
        
        # Compute spectral decomposition
        low_freq_raw, high_freq_raw = self._compute_spectral_bands(frame)
        x_low = self.low_freq_proj(low_freq_raw)   # [H, W, base_dim]
        x_high = self.high_freq_proj(high_freq_raw)  # [H, W, base_dim]
        
        current_bands = {
            'intensity': x_intensity,
            'low_freq': x_low,
            'high_freq': x_high,
        }
        
        # CRITICAL: Seed history buffer on first frame to prevent zero-attention collapse
        # Add small noise per timestep to break symmetry and allow temporal differentiation
        if not self.history_initialized:
            for t in range(self.time_depth):
                for band in self.bands:
                    # Add decreasing noise for older timesteps (simulates temporal decay)
                    noise_scale = 0.01 * (1.0 - t / self.time_depth)
                    noise = torch.randn_like(current_bands[band]) * noise_scale
                    self.history_buffer[band][t] = current_bands[band].detach() + noise
            self.history_initialized = True
        
        # Reorder history for attention BEFORE updating with current frame
        # This ensures attention only sees past frames (t-1, t-2, ...), not current frame
        history_ordered = {
            band: torch.roll(
                self.history_buffer[band],
                shifts=-self.history_idx,
                dims=0
            )
            for band in self.bands
        }
        
        # Run spectral attention (with history that does NOT include current frame)
        attn_out, stats = self.spectral_attn(
            current_bands=current_bands,
            history_bands=history_ordered,
            current_step=step
        )
        
        # Project to output
        pred = self.output_proj(attn_out).squeeze(-1)  # [H, W]
        
        # Update history (ring buffer) AFTER attention - current frame becomes available for next step
        for band in self.bands:
            self.history_buffer[band][self.history_idx] = current_bands[band].detach()
        self.history_idx = (self.history_idx + 1) % self.time_depth
        
        return pred, stats


def train_single_model(model, model_name, data_gen, args, config, device):
    """Train a single model and return results."""
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    
    num_steps = args.steps
    errors = []
    
    print(f"\n{'='*60}")
    print(f"Training {model_name} for {num_steps} steps...")
    print(f"{'='*60}")
    start_time = time.time()
    
    model.reset()
    
    for step in tqdm(range(num_steps), desc=model_name):
        current_frame = data_gen.frame_for_pattern(
            step,
            pattern_type=args.pattern_train,
            switch_period=args.switch_period,
        )
        next_frame = data_gen.frame_for_pattern(
            step + 1,
            pattern_type=args.pattern_train,
            switch_period=args.switch_period,
        )
        
        pred, stats = model(current_frame, step)
        loss = criterion(pred, next_frame)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        errors.append(loss.item())
        
        if step % 100 == 0:
            print(f"  Step {step}: Loss = {loss.item():.6f}")
    
    elapsed = time.time() - start_time
    
    # Compute stats
    n_steps_done = len(errors)
    trend_window = min(50, max(10, n_steps_done // 3)) if n_steps_done > 0 else 0
    first_win_mean = (sum(errors[:trend_window]) / trend_window) if trend_window > 0 else float('nan')
    last_win_mean = (sum(errors[-trend_window:]) / trend_window) if trend_window > 0 else float('nan')
    
    # Held-out evaluation
    eval_errors = []
    if args.eval_steps > 0:
        model.eval()
        model.reset()
        with torch.no_grad():
            for step in range(args.eval_steps):
                cur = data_gen.frame_for_pattern(
                    step + 10_000,
                    pattern_type=args.pattern_eval,
                    switch_period=args.switch_period,
                )
                nxt = data_gen.frame_for_pattern(
                    step + 1 + 10_000,
                    pattern_type=args.pattern_eval,
                    switch_period=args.switch_period,
                )
                pred, _ = model(cur, step)
                eval_errors.append(criterion(pred, nxt).item())
    
    results = {
        'model_name': model_name,
        'elapsed': elapsed,
        'steps_per_sec': num_steps / elapsed,
        'final_loss': errors[-1],
        'min_loss': min(errors),
        'first_win_mean': first_win_mean,
        'last_win_mean': last_win_mean,
        'improvement_pct': (first_win_mean - last_win_mean) / first_win_mean * 100 if first_win_mean > 0 else 0,
        'eval_mean': sum(eval_errors) / len(eval_errors) if eval_errors else None,
        'eval_min': min(eval_errors) if eval_errors else None,
        'eval_final': eval_errors[-1] if eval_errors else None,
        'errors': errors,
        'eval_errors': eval_errors,
    }
    
    return results


def print_results(results: Dict):
    """Print results for a single model."""
    print(f"\n{results['model_name']} RESULTS:")
    print(f"  Total time: {results['elapsed']:.2f}s")
    print(f"  Steps/sec: {results['steps_per_sec']:.1f}")
    print(f"  Final loss: {results['final_loss']:.6f}")
    print(f"  Min loss: {results['min_loss']:.6f}")
    print(f"  First-window mean: {results['first_win_mean']:.6f}")
    print(f"  Last-window mean: {results['last_win_mean']:.6f}")
    print(f"  Trend: IMPROVING (+{results['improvement_pct']:.1f}%)" if results['improvement_pct'] > 0.5 else f"  Trend: {results['improvement_pct']:.1f}%")
    
    if results['eval_mean'] is not None:
        print(f"\n  HELD-OUT EVALUATION:")
        print(f"    Eval mean loss: {results['eval_mean']:.6f}")
        print(f"    Eval final loss: {results['eval_final']:.6f}")
        print(f"    Eval min loss: {results['eval_min']:.6f}")


def print_comparison(spectral_results: Dict, mlp_results: Dict):
    """Print side-by-side comparison."""
    print("\n" + "="*70)
    print("COMPARISON: SpectralPredictor vs MLPPredictor")
    print("="*70)
    
    print(f"\n{'Metric':<25} {'Spectral':>15} {'MLP':>15} {'Winner':>12}")
    print("-"*70)
    
    metrics = [
        ('Final Loss', 'final_loss', 'lower'),
        ('Min Loss', 'min_loss', 'lower'),
        ('Last-window Mean', 'last_win_mean', 'lower'),
        ('Improvement %', 'improvement_pct', 'higher'),
        ('Eval Mean Loss', 'eval_mean', 'lower'),
        ('Steps/sec', 'steps_per_sec', 'higher'),
    ]
    
    for label, key, better in metrics:
        s_val = spectral_results.get(key)
        m_val = mlp_results.get(key)
        
        if s_val is None or m_val is None:
            continue
        
        if better == 'lower':
            winner = 'Spectral' if s_val < m_val else ('MLP' if m_val < s_val else 'Tie')
        else:
            winner = 'Spectral' if s_val > m_val else ('MLP' if m_val > s_val else 'Tie')
        
        print(f"{label:<25} {s_val:>15.6f} {m_val:>15.6f} {winner:>12}")
    
    print("-"*70)
    
    # Summary
    s_eval = spectral_results.get('eval_mean', spectral_results['last_win_mean'])
    m_eval = mlp_results.get('eval_mean', mlp_results['last_win_mean'])
    diff_pct = (m_eval - s_eval) / m_eval * 100 if m_eval > 0 else 0
    
    if diff_pct > 5:
        print(f"\nSpectral is {diff_pct:.1f}% better than MLP on evaluation loss.")
    elif diff_pct < -5:
        print(f"\nMLP is {-diff_pct:.1f}% better than Spectral on evaluation loss.")
    else:
        print(f"\nBoth models perform similarly (within 5%).")


def run_experiment(args):
    """Run the synthetic 2D experiment."""
    
    # Select device
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Get config
    if args.config == 'default':
        config = get_default_config(args.grid_size)
    elif args.config == 'fast':
        config = get_fast_config(args.grid_size)
    elif args.config == 'turbo':
        config = get_turbo_config(args.grid_size)
    elif args.config == 'large':
        config = get_large_config(args.grid_size)
    elif args.config == 'xlarge':
        config = get_xlarge_config(args.grid_size)
    else:
        raise ValueError(f"Unknown config: {args.config}")
    
    # Optional override: Temporal top-k
    if args.attn_topk_temporal is not None:
        config['attnTopKTemporal'] = int(args.attn_topk_temporal)
    
    print(f"Config: {args.config}")
    print(f"Grid size: {args.grid_size}")
    print(f"Time depth: {config['timeDepth']}")
    print(f"Pattern: {args.pattern_train}")
    print(f"Model: {args.model}")
    
    # Create data generator
    data_gen = SyntheticDataGenerator(args.grid_size, device)
    
    # Run based on model selection
    if args.model == 'compare':
        # Run both models and compare
        spectral_model = SpectralPredictor(config, device)
        mlp_model = MLPPredictor(config, device)
        
        spectral_results = train_single_model(spectral_model, "SpectralPredictor", data_gen, args, config, device)
        mlp_results = train_single_model(mlp_model, "MLPPredictor", data_gen, args, config, device)
        
        print_results(spectral_results)
        print_results(mlp_results)
        print_comparison(spectral_results, mlp_results)
        return
    
    elif args.model == 'mlp':
        model = MLPPredictor(config, device)
        model_name = "MLPPredictor"
    else:
        model = SpectralPredictor(config, device)
        model_name = "SpectralPredictor"
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    
    # Training loop
    num_steps = args.steps
    errors = []
    attn_stats_records = []
    
    print(f"\nTraining for {num_steps} steps...")
    start_time = time.time()
    
    model.reset()
    
    for step in tqdm(range(num_steps)):
        # Generate current and next frame for chosen training pattern
        current_frame = data_gen.frame_for_pattern(
            step,
            pattern_type=args.pattern_train,
            switch_period=args.switch_period,
        )
        next_frame = data_gen.frame_for_pattern(
            step + 1,
            pattern_type=args.pattern_train,
            switch_period=args.switch_period,
        )
        
        # Forward pass
        pred, stats = model(current_frame, step)
        
        # Compute loss
        loss = criterion(pred, next_frame)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        errors.append(loss.item())
        
        # Collect attention stats (collapse vs structured use)
        temporal_entropy = stats['temporal'].get('entropy', 0.0)
        temporal_mean_w = stats['temporal'].get('mean_weight', 0.0)
        
        neighbor_mean_w = stats['neighbor'].get('mean_weight', 0.0)
        neighbor_entropy = None
        neighbor_weights = stats['neighbor'].get('neighbor_weights', None)
        if neighbor_weights is not None:
            w_safe = torch.clamp(neighbor_weights, min=1e-10)
            ent = -torch.sum(w_safe * torch.log(w_safe), dim=-1).mean().item()
            neighbor_entropy = ent
        
        wormhole = stats['wormhole']
        wh_num_conn = wormhole.get('num_connections', 0)
        wh_pos_with_conn = wormhole.get('positions_with_connections', 0)
        wh_mean_sim = wormhole.get('mean_similarity', 0.0)
        wh_sparsity = wormhole.get('sparsity', 0.0)
        wh_skipped = wormhole.get('skipped', False)
        
        attn_stats_records.append({
            'step': step,
            'train_loss': loss.item(),
            'temporal_entropy': float(temporal_entropy),
            'temporal_mean_weight': float(temporal_mean_w),
            'neighbor_mean_weight': float(neighbor_mean_w),
            'neighbor_entropy': float(neighbor_entropy) if neighbor_entropy is not None else None,
            'wormhole_num_connections': int(wh_num_conn),
            'wormhole_positions_with_connections': int(wh_pos_with_conn),
            'wormhole_mean_similarity': float(wh_mean_sim),
            'wormhole_sparsity': float(wh_sparsity),
            'wormhole_skipped': bool(wh_skipped),
        })
        
        if step % 50 == 0:
            print(f"  Step {step}: Loss = {loss.item():.6f}")
        
        # Visualization snapshots
        if args.viz_interval and args.viz_interval > 0 and (step % args.viz_interval == 0):
            _maybe_save_snapshot(
                out_dir=args.viz_dir,
                step=step,
                current_frame=current_frame,
                next_frame=next_frame,
                pred=pred.detach(),
                stats=attn_stats_records[-1]
            )
    
    elapsed = time.time() - start_time
    
    # Results
    n_steps_done = len(errors)
    # Dynamic window: up to 50, but at least 10 or n//3 for short runs
    trend_window = min(50, max(10, n_steps_done // 3)) if n_steps_done > 0 else 0
    first_win_mean = (sum(errors[:trend_window]) / trend_window) if trend_window > 0 else float('nan')
    last_win_mean = (sum(errors[-trend_window:]) / trend_window) if trend_window > 0 else float('nan')
    
    print()
    print("RESULTS:")
    print(f"  Total time: {elapsed:.2f}s")
    print(f"  Steps/sec: {num_steps / elapsed:.1f}")
    print(f"  Final loss: {errors[-1]:.6f}")
    print(f"  Min loss: {min(errors):.6f}")
    print(f"  Trend window: {trend_window}")
    print(f"  First-window mean: {first_win_mean:.6f}")
    print(f"  Last-window mean: {last_win_mean:.6f}")
    
    if trend_window > 0 and first_win_mean > 0:
        change_pct = (first_win_mean - last_win_mean) / first_win_mean * 100
        if abs(change_pct) < 0.5:
            print(f"  Trend: STABLE ({change_pct:+.1f}%)")
        elif change_pct > 0:
            print(f"  Trend: IMPROVING (+{change_pct:.1f}%)")
        else:
            print(f"  Trend: REGRESSING ({change_pct:.1f}%)")
    
    # Optionally write attention stats to CSV
    if args.log_stats_csv:
        import csv
        fieldnames = list(attn_stats_records[0].keys()) if attn_stats_records else []
        with open(args.log_stats_csv, mode='w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in attn_stats_records:
                writer.writerow(row)
        print(f"\nWrote attention stats to: {args.log_stats_csv}")
    
    # Held-out evaluation on a fresh sequence (reset history)
    if args.eval_steps > 0:
        model.eval()
        model.reset()
        with torch.no_grad():
            eval_errors = []
            for step in range(args.eval_steps):
                # Generate eval frames
                cur = data_gen.frame_for_pattern(
                    step + 10_000,
                    pattern_type=args.pattern_eval,
                    switch_period=args.switch_period,
                )
                nxt = data_gen.frame_for_pattern(
                    step + 1 + 10_000,
                    pattern_type=args.pattern_eval,
                    switch_period=args.switch_period,
                )
                
                pred, _ = model(cur, step)
                eval_loss = criterion(pred, nxt).item()
                eval_errors.append(eval_loss)
        
        eval_mean = sum(eval_errors) / len(eval_errors)
        eval_min = min(eval_errors)
        eval_last = eval_errors[-1]
        print("\nHELD-OUT EVALUATION:")
        print(f"  Eval steps: {args.eval_steps}")
        print(f"  Eval pattern: {args.pattern_eval}")
        print(f"  Eval mean loss: {eval_mean:.6f}")
        print(f"  Eval final loss: {eval_last:.6f}")
        print(f"  Eval min loss: {eval_min:.6f}")


def main():
    parser = argparse.ArgumentParser(description='Synthetic 2D Spectral Attention Experiment')
    
    parser.add_argument('--config', type=str, default='fast',
                        choices=['default', 'fast', 'turbo', 'large', 'xlarge'],
                        help='Configuration preset')
    parser.add_argument('--grid-size', type=int, default=32,
                        help='Grid size (default: 32)')
    parser.add_argument('--steps', type=int, default=300,
                        help='Number of training steps')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cpu', 'cuda', 'mps'],
                        help='Device to use')
    parser.add_argument('--pattern-train', type=str, default='blob',
                        choices=PATTERN_CHOICES,
                        help='Training pattern')
    parser.add_argument('--pattern-eval', type=str, default=None,
                        choices=PATTERN_CHOICES,
                        help='Evaluation pattern (defaults to training pattern)')
    parser.add_argument('--eval-steps', type=int, default=101,
                        help='Number of evaluation steps on held-out sequence')
    parser.add_argument('--switch-period', type=int, default=101,
                        help='Period for switching pattern')
    parser.add_argument('--log-stats-csv', type=str, default=None,
                        help='Optional path to write per-step attention stats CSV')
    parser.add_argument('--viz-interval', type=int, default=0,
                        help='If > 0, save a visualization PNG every N steps')
    parser.add_argument('--viz-dir', type=str, default='viz_out',
                        help='Directory to store visualization images')
    parser.add_argument('--attn-topk-temporal', type=int, default=None,
                        help='Override TemporalAttention top_k (default: from config)')
    parser.add_argument('--model', type=str, default='spectral',
                        choices=['spectral', 'mlp', 'compare'],
                        help='Model type: spectral (default), mlp (baseline), or compare (run both)')
    
    args = parser.parse_args()
    if args.pattern_eval is None:
        args.pattern_eval = args.pattern_train
    # Inject override into config by re-parsing inside run_experiment path
    # (we pass args through; config mutation happens after preset selection)
    run_experiment(args)


if __name__ == '__main__':
    main()
