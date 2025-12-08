"""
Mini Plasma Environment for Experiment 032.

A simple 2D continuous field with diffusion, advection, and actuator control.
Designed as a testbed for AKIRA's spectral belief architecture.

The environment is intentionally simple:
- Scalar field on a grid (H x W)
- Diffusion + mild advection
- External control from a small set of actuators (Gaussian bumps)
- Partial observability: controller sees field, not velocity/dynamics params
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional

import torch
import torch.nn.functional as F


@dataclass
class PlasmaConfig:
    """Configuration for the mini plasma environment.
    
    Difficulty modes:
    - easy (old defaults): diffusion=0.12, advection=0.02, noise_std=0.01
    - medium (new defaults): diffusion=0.25, advection=0.08, noise_std=0.02
    - hard: diffusion=0.4, advection=0.15, noise_std=0.05
    """
    height: int = 64
    width: int = 64
    # Increased defaults for harder prediction task
    diffusion: float = 0.25       # old: 0.12 - faster smoothing
    advection: float = 0.08       # old: 0.02 - stronger drift
    num_actuators: int = 6
    actuator_sigma: float = 5.0
    noise_std: float = 0.02       # old: 0.01 - more stochasticity
    # Random disturbance frequency and strength
    disturbance_prob: float = 0.1      # probability of random disturbance per step
    disturbance_strength: float = 0.15 # strength of random disturbances
    device: str = "cpu"
    dtype: torch.dtype = torch.float32
    
    @classmethod
    def easy(cls, device: str = "cpu") -> "PlasmaConfig":
        """Old easy defaults for comparison."""
        return cls(
            diffusion=0.12, advection=0.02, noise_std=0.01,
            disturbance_prob=0.0, disturbance_strength=0.0,
            device=device
        )
    
    @classmethod
    def medium(cls, device: str = "cpu") -> "PlasmaConfig":
        """Medium difficulty (new default)."""
        return cls(device=device)  # Uses default values
    
    @classmethod
    def hard(cls, device: str = "cpu") -> "PlasmaConfig":
        """Hard mode - significant dynamics."""
        return cls(
            diffusion=0.4, advection=0.15, noise_std=0.05,
            disturbance_prob=0.2, disturbance_strength=0.25,
            device=device
        )


class MiniPlasmaEnv:
    """
    A toy 2D plasma-like environment.
    
    The field evolves via:
    1. Diffusion (Laplacian smoothing)
    2. Advection (mild drift)
    3. Actuator forces (localized Gaussian pushes)
    4. Noise (small additive perturbation)
    
    Control task: Keep a blob centered at the target location.
    """
    
    def __init__(self, cfg: PlasmaConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.dtype = cfg.dtype
        self._actuator_maps = self._build_actuator_maps()
        
    def _build_actuator_maps(self) -> torch.Tensor:
        """
        Precompute actuator influence fields (Gaussian bumps).
        
        Actuators are placed in a grid pattern across the field.
        Each actuator has a localized Gaussian influence.
        
        Returns:
            Tensor of shape (num_actuators, H, W)
        """
        h, w = self.cfg.height, self.cfg.width
        
        # Create coordinate grids
        yy, xx = torch.meshgrid(
            torch.linspace(0, h - 1, h, device=self.device, dtype=self.dtype),
            torch.linspace(0, w - 1, w, device=self.device, dtype=self.dtype),
            indexing="ij",
        )
        
        # Place actuators in a grid pattern
        # Compute grid size to cover num_actuators
        grid_n = int((self.cfg.num_actuators) ** 0.5)
        if grid_n * grid_n < self.cfg.num_actuators:
            grid_n += 1
            
        # Centers at 20%-80% of field to avoid edges
        centers_y = torch.linspace(h * 0.2, h * 0.8, grid_n, device=self.device, dtype=self.dtype)
        centers_x = torch.linspace(w * 0.2, w * 0.8, grid_n, device=self.device, dtype=self.dtype)
        centers = torch.cartesian_prod(centers_y, centers_x)[:self.cfg.num_actuators]
        
        # Build Gaussian bumps
        sig2 = self.cfg.actuator_sigma ** 2
        bumps = []
        for cy, cx in centers:
            bump = torch.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sig2))
            bumps.append(bump)
            
        return torch.stack(bumps, dim=0)  # (A, H, W)
    
    def reset(self, batch_size: int = 1) -> torch.Tensor:
        """
        Reset environment with a random blob.
        
        Args:
            batch_size: Number of parallel environments
            
        Returns:
            Initial field of shape (B, 1, H, W)
        """
        h, w = self.cfg.height, self.cfg.width
        field = torch.zeros(
            batch_size, 1, h, w, 
            device=self.device, dtype=self.dtype
        )
        
        # Random blob position (within center region)
        cy = torch.randint(
            low=int(h * 0.3), high=int(h * 0.7), 
            size=(batch_size,), device=self.device
        )
        cx = torch.randint(
            low=int(w * 0.3), high=int(w * 0.7), 
            size=(batch_size,), device=self.device
        )
        
        # Coordinate grids
        yy, xx = torch.meshgrid(
            torch.arange(h, device=self.device, dtype=self.dtype),
            torch.arange(w, device=self.device, dtype=self.dtype),
            indexing="ij",
        )
        
        # Create blob for each batch element
        sig2 = (self.cfg.actuator_sigma * 1.5) ** 2
        for b in range(batch_size):
            blob = torch.exp(-((yy - cy[b].float()) ** 2 + (xx - cx[b].float()) ** 2) / (2 * sig2))
            field[b, 0] = blob
            
        return field
    
    def step(
        self, 
        field: torch.Tensor, 
        control: torch.Tensor, 
        noise: bool = True
    ) -> torch.Tensor:
        """
        Advance the field by one timestep.
        
        Args:
            field: Current field (B, 1, H, W)
            control: Actuator amplitudes (B, num_actuators), range [-1, 1]
            noise: Whether to add stochastic noise
            
        Returns:
            Next field (B, 1, H, W)
        """
        assert field.ndim == 4, f"Expected 4D tensor, got {field.ndim}D"
        assert control.shape[-1] == self.cfg.num_actuators
        
        B = field.shape[0]
        
        # 1. Diffusion via Laplacian (5-point stencil)
        # Laplacian = sum of neighbors - 4 * center
        lap = (
            -4 * field
            + F.pad(field, (0, 0, 1, 0))[:, :, :-1, :]  # shift down
            + F.pad(field, (0, 0, 0, 1))[:, :, 1:, :]   # shift up
            + F.pad(field, (1, 0, 0, 0))[:, :, :, :-1]  # shift right
            + F.pad(field, (0, 1, 0, 0))[:, :, :, 1:]   # shift left
        )
        diffused = field + self.cfg.diffusion * lap
        
        # 2. Advection: mild diagonal drift
        advected = (
            torch.roll(diffused, shifts=(1, -1), dims=(2, 3)) * self.cfg.advection 
            + diffused * (1 - self.cfg.advection)
        )
        
        # 3. Actuator forces
        # control: (B, A) -> (B, A, 1, 1)
        # bumps: (A, H, W) -> (1, A, H, W)
        bumps = self._actuator_maps.unsqueeze(0).expand(B, -1, -1, -1)
        control_expanded = control.unsqueeze(-1).unsqueeze(-1)
        force = torch.sum(control_expanded * bumps, dim=1, keepdim=True)
        
        next_field = advected + force
        
        # 4. Noise
        if noise and self.cfg.noise_std > 0:
            next_field = next_field + torch.randn_like(next_field) * self.cfg.noise_std
        
        # 5. Random disturbances (spatially localized)
        if self.cfg.disturbance_prob > 0:
            if torch.rand(1).item() < self.cfg.disturbance_prob:
                # Random Gaussian bump disturbance
                h, w = self.cfg.height, self.cfg.width
                cy = torch.randint(int(h * 0.2), int(h * 0.8), (1,)).item()
                cx = torch.randint(int(w * 0.2), int(w * 0.8), (1,)).item()
                yy, xx = torch.meshgrid(
                    torch.arange(h, device=self.device, dtype=self.dtype),
                    torch.arange(w, device=self.device, dtype=self.dtype),
                    indexing="ij",
                )
                sig2 = (self.cfg.actuator_sigma * 1.5) ** 2
                bump = torch.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sig2))
                sign = 2 * (torch.rand(1).item() > 0.5) - 1  # +1 or -1
                disturbance = sign * self.cfg.disturbance_strength * bump.unsqueeze(0).unsqueeze(0)
                next_field = next_field + disturbance.expand(B, -1, -1, -1)
        
        # 6. Clamp to valid range
        return torch.clamp(next_field, min=-1.0, max=1.0)
    
    def make_target(self, batch_size: int = 1) -> torch.Tensor:
        """
        Create target field: blob at center.
        
        Args:
            batch_size: Number of parallel targets
            
        Returns:
            Target field (B, 1, H, W)
        """
        h, w = self.cfg.height, self.cfg.width
        field = torch.zeros(
            batch_size, 1, h, w, 
            device=self.device, dtype=self.dtype
        )
        
        # Center coordinates
        cy = h / 2
        cx = w / 2
        
        # Coordinate grids
        yy, xx = torch.meshgrid(
            torch.linspace(0, h - 1, h, device=self.device, dtype=self.dtype),
            torch.linspace(0, w - 1, w, device=self.device, dtype=self.dtype),
            indexing="ij",
        )
        
        # Gaussian blob at center
        sig2 = (self.cfg.actuator_sigma * 1.3) ** 2
        blob = torch.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sig2))
        
        field[:, 0] = blob.unsqueeze(0).expand(batch_size, -1, -1)
        return field
    
    def compute_metrics(
        self, 
        field: torch.Tensor, 
        target: torch.Tensor
    ) -> dict:
        """
        Compute control metrics.
        
        Args:
            field: Current field (B, 1, H, W)
            target: Target field (B, 1, H, W)
            
        Returns:
            Dictionary with metrics:
            - field_mse: Mean squared error
            - centroid_error: Distance from blob center to target center
        """
        # Field MSE
        mse = F.mse_loss(field, target, reduction='none').mean(dim=[1, 2, 3])
        
        # Centroid error
        # Compute center of mass of field
        h, w = field.shape[2], field.shape[3]
        yy, xx = torch.meshgrid(
            torch.arange(h, device=field.device, dtype=field.dtype),
            torch.arange(w, device=field.device, dtype=field.dtype),
            indexing="ij",
        )
        
        # Normalize field to get probability distribution
        field_pos = F.relu(field.squeeze(1))  # (B, H, W), non-negative
        field_sum = field_pos.sum(dim=[1, 2], keepdim=True) + 1e-8
        field_prob = field_pos / field_sum
        
        # Compute centroids
        cy_field = (field_prob * yy.unsqueeze(0)).sum(dim=[1, 2])
        cx_field = (field_prob * xx.unsqueeze(0)).sum(dim=[1, 2])
        
        # Target centroid (should be center)
        cy_target = h / 2
        cx_target = w / 2
        
        # Euclidean distance
        centroid_err = torch.sqrt(
            (cy_field - cy_target) ** 2 + (cx_field - cx_target) ** 2
        )
        
        return {
            "field_mse": mse,
            "centroid_error": centroid_err,
        }


def generate_trajectories(
    env: MiniPlasmaEnv,
    num_trajectories: int,
    trajectory_length: int,
    control_scale: float = 0.1,
    prediction_horizon: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate random trajectories for training.
    
    Args:
        env: Environment instance
        num_trajectories: Number of trajectories to generate
        trajectory_length: Steps per trajectory
        control_scale: Scale of random control actions
        prediction_horizon: Steps ahead to predict (k in t+k).
                           Higher values create harder prediction tasks that
                           penalize identity copying more severely.
        
    Returns:
        Tuple of:
        - fields: (num_traj * traj_len, 1, H, W) all fields at time t
        - next_fields: (num_traj * traj_len, 1, H, W) fields at time t+k
        - controls: (num_traj * traj_len, num_actuators) control actions at t
    """
    all_fields = []
    all_next_fields = []
    all_controls = []
    
    for _ in range(num_trajectories):
        field = env.reset(batch_size=1)
        
        # Store trajectory for multi-step targets
        trajectory = [field.clone()]
        controls_traj = []
        
        # Generate full trajectory first
        for _ in range(trajectory_length + prediction_horizon):
            # Random control
            control = torch.randn(
                1, env.cfg.num_actuators, 
                device=env.device, dtype=env.dtype
            ) * control_scale
            control = torch.clamp(control, -1, 1)
            
            # Step
            next_field = env.step(field, control, noise=True)
            
            trajectory.append(next_field.clone())
            controls_traj.append(control)
            
            field = next_field.detach()
        
        # Create input-target pairs with prediction_horizon offset
        for t in range(trajectory_length):
            all_fields.append(trajectory[t])
            all_next_fields.append(trajectory[t + prediction_horizon])
            all_controls.append(controls_traj[t])
    
    return (
        torch.cat(all_fields, dim=0),
        torch.cat(all_next_fields, dim=0),
        torch.cat(all_controls, dim=0),
    )


if __name__ == "__main__":
    # Quick test
    cfg = PlasmaConfig(device="cpu")
    env = MiniPlasmaEnv(cfg)
    
    print(f"Environment: {cfg.height}x{cfg.width} grid, {cfg.num_actuators} actuators")
    
    field = env.reset(batch_size=2)
    print(f"Initial field shape: {field.shape}")
    
    target = env.make_target(batch_size=2)
    print(f"Target shape: {target.shape}")
    
    control = torch.randn(2, cfg.num_actuators)
    next_field = env.step(field, control)
    print(f"Next field shape: {next_field.shape}")
    
    metrics = env.compute_metrics(field, target)
    print(f"Initial metrics: MSE={metrics['field_mse'].mean():.4f}, Centroid={metrics['centroid_error'].mean():.4f}")
    
    # Generate some data
    fields, next_fields, controls = generate_trajectories(env, num_trajectories=10, trajectory_length=20)
    print(f"Generated data: {fields.shape[0]} samples")
