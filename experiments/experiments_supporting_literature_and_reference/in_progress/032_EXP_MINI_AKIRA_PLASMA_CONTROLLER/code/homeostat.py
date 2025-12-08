"""
Homeostat (PSON-Aligned Controller) for Experiment 032.

Implements a homeostatic controller based on:
- Precision-Scaled Orthogonal Noise (PSON) injection
- Setpoint maintenance
- Constrained relaxation dynamics
- Stability constraints

Reference: Ashby's homeostat concept, AKIRA's PSON theory

The homeostat maintains stability by:
1. Injecting noise orthogonal to the gradient (exploration without fighting optimization)
2. Maintaining a setpoint (target error level)
3. Damping oscillations
4. Adapting gain based on error dynamics
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim


@dataclass
class HomeostatConfig:
    """Configuration for the homeostat controller."""
    # PSON parameters
    noise_scale: float = 0.01  # Scale of orthogonal noise
    noise_decay: float = 0.99  # Decay noise over time
    
    # Setpoint maintenance
    setpoint: float = 0.1  # Target error level
    setpoint_gain: float = 0.1  # How strongly to correct toward setpoint
    
    # Stability
    max_step: float = 0.1  # Maximum parameter step size
    damping: float = 0.9  # Momentum damping factor
    oscillation_threshold: float = 0.5  # Detect oscillation
    
    # Adaptation
    gain_min: float = 0.1
    gain_max: float = 10.0
    gain_adapt_rate: float = 0.01


class PSONOptimizer(optim.Optimizer):
    """
    Precision-Scaled Orthogonal Noise Optimizer.
    
    Extends SGD with orthogonal noise injection:
    - Computes gradient g
    - Generates random noise n
    - Projects noise orthogonal to gradient: n_orth = n - proj(n, g)
    - Updates: theta -= lr * (g + alpha * n_orth)
    
    The orthogonal noise enables exploration without fighting the gradient.
    """
    
    def __init__(
        self,
        params,
        lr: float = 0.001,
        noise_scale: float = 0.01,
        max_step: float = 0.1,
        damping: float = 0.9,
    ):
        defaults = dict(
            lr=lr,
            noise_scale=noise_scale,
            max_step=max_step,
            damping=damping,
        )
        super().__init__(params, defaults)
        
        # Initialize state
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                state["momentum"] = torch.zeros_like(p.data)
    
    @torch.no_grad()
    def step(self, closure=None):
        """
        Perform a single optimization step with PSON.
        
        Args:
            closure: Optional closure for loss computation
            
        Returns:
            Loss value if closure provided
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            lr = group["lr"]
            noise_scale = group["noise_scale"]
            max_step = group["max_step"]
            damping = group["damping"]
            
            for p in group["params"]:
                if p.grad is None:
                    continue
                
                grad = p.grad
                state = self.state[p]
                momentum = state["momentum"]
                
                # Generate random noise
                noise = torch.randn_like(grad)
                
                # Project noise orthogonal to gradient
                # n_orth = n - (n . g / |g|^2) * g
                grad_norm_sq = (grad * grad).sum() + 1e-8
                proj_coeff = (noise * grad).sum() / grad_norm_sq
                noise_orth = noise - proj_coeff * grad
                
                # Normalize orthogonal noise
                noise_orth_norm = torch.sqrt((noise_orth * noise_orth).sum() + 1e-8)
                noise_orth = noise_orth / noise_orth_norm
                
                # Compute update: gradient + orthogonal noise
                update = grad + noise_scale * noise_orth
                
                # Apply damping (momentum)
                momentum.mul_(damping).add_(update)
                
                # Clip step size
                step_norm = torch.sqrt((momentum * momentum).sum())
                if step_norm > max_step:
                    momentum.mul_(max_step / step_norm)
                
                # Update parameters
                p.add_(momentum, alpha=-lr)
        
        return loss


class Homeostat:
    """
    Homeostatic controller for maintaining stability.
    
    The homeostat:
    1. Tracks error relative to a setpoint
    2. Adjusts gain to maintain the setpoint
    3. Detects and dampens oscillations
    4. Provides PSON-based parameter updates
    """
    
    def __init__(self, cfg: HomeostatConfig):
        self.cfg = cfg
        
        # State
        self.gain = 1.0
        self.error_history: List[float] = []
        self.step_count = 0
        
        # Oscillation detection
        self.sign_history: List[int] = []
        
    def reset(self):
        """Reset homeostat state."""
        self.gain = 1.0
        self.error_history.clear()
        self.sign_history.clear()
        self.step_count = 0
    
    def compute_control_signal(
        self,
        current_error: float,
        gradient: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """
        Compute homeostatic control signal.
        
        Args:
            current_error: Current error value
            gradient: Optional gradient for PSON
            
        Returns:
            Dict with:
            - gain: Current gain
            - correction: Setpoint correction signal
            - is_oscillating: Whether oscillation detected
        """
        self.error_history.append(current_error)
        self.step_count += 1
        
        # Compute error relative to setpoint
        error_from_setpoint = current_error - self.cfg.setpoint
        
        # Track sign changes for oscillation detection
        if len(self.error_history) >= 2:
            delta = current_error - self.error_history[-2]
            sign = 1 if delta > 0 else -1
            self.sign_history.append(sign)
            
            # Keep only recent history
            if len(self.sign_history) > 10:
                self.sign_history.pop(0)
        
        # Detect oscillation (many sign changes)
        is_oscillating = False
        if len(self.sign_history) >= 5:
            sign_changes = sum(
                1 for i in range(1, len(self.sign_history))
                if self.sign_history[i] != self.sign_history[i-1]
            )
            is_oscillating = sign_changes / len(self.sign_history) > self.cfg.oscillation_threshold
        
        # Adapt gain
        if error_from_setpoint > 0:
            # Error too high, increase gain
            self.gain = min(
                self.cfg.gain_max,
                self.gain * (1 + self.cfg.gain_adapt_rate)
            )
        else:
            # Error below setpoint, can relax gain
            self.gain = max(
                self.cfg.gain_min,
                self.gain * (1 - self.cfg.gain_adapt_rate * 0.5)
            )
        
        # If oscillating, reduce gain to dampen
        if is_oscillating:
            self.gain *= 0.9
        
        # Compute correction signal
        correction = self.cfg.setpoint_gain * error_from_setpoint * self.gain
        
        return {
            "gain": self.gain,
            "correction": correction,
            "is_oscillating": is_oscillating,
            "error_from_setpoint": error_from_setpoint,
        }
    
    def apply_pson_to_gradient(
        self,
        gradient: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply PSON to a gradient tensor.
        
        Args:
            gradient: Original gradient
            
        Returns:
            Modified gradient with orthogonal noise
        """
        # Generate noise
        noise = torch.randn_like(gradient)
        
        # Project orthogonal
        grad_norm_sq = (gradient * gradient).sum() + 1e-8
        proj_coeff = (noise * gradient).sum() / grad_norm_sq
        noise_orth = noise - proj_coeff * gradient
        
        # Scale noise (decay over time)
        noise_scale = self.cfg.noise_scale * (self.cfg.noise_decay ** self.step_count)
        
        # Combine
        modified = gradient + noise_scale * noise_orth * self.gain
        
        # Clip
        max_norm = self.cfg.max_step
        norm = torch.sqrt((modified * modified).sum())
        if norm > max_norm:
            modified = modified * (max_norm / norm)
        
        return modified
    
    def get_state(self) -> Dict[str, float]:
        """Get current homeostat state."""
        return {
            "gain": self.gain,
            "step_count": self.step_count,
            "recent_error": self.error_history[-1] if self.error_history else 0.0,
            "mean_error": sum(self.error_history[-10:]) / max(1, len(self.error_history[-10:])),
        }


class ControlHead(nn.Module):
    """
    Control head that produces actuator commands.
    
    Takes belief state as input to enable belief-informed control.
    """
    
    def __init__(
        self,
        input_dim: int,
        num_actuators: int,
        hidden_dim: int = 64,
        use_belief: bool = True,
        num_bands: int = 8,
    ):
        super().__init__()
        self.use_belief = use_belief
        self.num_bands = num_bands
        
        # Input: pooled field features + optional belief entropy
        belief_dim = num_bands if use_belief else 0
        total_input = input_dim + belief_dim
        
        self.net = nn.Sequential(
            nn.Linear(total_input, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_actuators),
            nn.Tanh(),  # Bounded output [-1, 1]
        )
    
    def forward(
        self,
        field_features: torch.Tensor,
        belief_entropy: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute control output.
        
        Args:
            field_features: (B, input_dim) pooled field features
            belief_entropy: (B, num_bands) optional entropy per band
            
        Returns:
            (B, num_actuators) control commands in [-1, 1]
        """
        if self.use_belief and belief_entropy is not None:
            x = torch.cat([field_features, belief_entropy], dim=1)
        else:
            x = field_features
        
        return self.net(x)


class HomeostaticController:
    """
    Complete homeostatic control system.
    
    Combines:
    - Control head (neural network)
    - Homeostat (stability maintenance)
    - PSON optimizer
    """
    
    def __init__(
        self,
        control_head: ControlHead,
        homeostat_cfg: HomeostatConfig,
        lr: float = 0.001,
    ):
        self.control_head = control_head
        self.homeostat = Homeostat(homeostat_cfg)
        
        # PSON optimizer for control head
        self.optimizer = PSONOptimizer(
            control_head.parameters(),
            lr=lr,
            noise_scale=homeostat_cfg.noise_scale,
            max_step=homeostat_cfg.max_step,
            damping=homeostat_cfg.damping,
        )
    
    def compute_control(
        self,
        field_features: torch.Tensor,
        belief_entropy: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute control output.
        
        Args:
            field_features: (B, D) pooled field features
            belief_entropy: (B, num_bands) optional belief state
            
        Returns:
            (B, num_actuators) control commands
        """
        return self.control_head(field_features, belief_entropy)
    
    def update(
        self,
        loss: torch.Tensor,
        current_error: float,
    ) -> Dict[str, float]:
        """
        Update controller with homeostatic adjustments.
        
        Args:
            loss: Differentiable loss
            current_error: Current control error (scalar)
            
        Returns:
            Dict with update statistics
        """
        # Compute homeostat signal
        homeo_signal = self.homeostat.compute_control_signal(current_error)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Apply PSON modification to gradients
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    p.grad = self.homeostat.apply_pson_to_gradient(p.grad)
        
        # Step
        self.optimizer.step()
        
        return {
            "homeo_gain": homeo_signal["gain"],
            "homeo_correction": homeo_signal["correction"],
            "is_oscillating": homeo_signal["is_oscillating"],
        }
    
    def reset(self):
        """Reset homeostat state."""
        self.homeostat.reset()


if __name__ == "__main__":
    # Quick test
    print("Testing PSON Optimizer...")
    
    # Simple quadratic optimization
    x = torch.randn(10, requires_grad=True)
    target = torch.zeros(10)
    
    optimizer = PSONOptimizer([x], lr=0.1, noise_scale=0.05)
    
    for step in range(50):
        optimizer.zero_grad()
        loss = ((x - target) ** 2).sum()
        loss.backward()
        optimizer.step()
        
        if step % 10 == 0:
            print(f"Step {step}: loss={loss.item():.4f}")
    
    print(f"\nFinal x: {x[:3].detach()}")
    
    # Test homeostat
    print("\n\nTesting Homeostat...")
    
    cfg = HomeostatConfig(setpoint=0.1)
    homeostat = Homeostat(cfg)
    
    # Simulate error dynamics
    errors = [0.5, 0.4, 0.3, 0.2, 0.15, 0.12, 0.11, 0.09, 0.08, 0.11, 0.13, 0.10]
    
    for i, err in enumerate(errors):
        signal = homeostat.compute_control_signal(err)
        print(f"Step {i}: error={err:.3f}, gain={signal['gain']:.3f}, correction={signal['correction']:.4f}")
    
    # Test control head
    print("\n\nTesting ControlHead...")
    
    head = ControlHead(input_dim=32, num_actuators=6, use_belief=True)
    
    field_feat = torch.randn(2, 32)
    belief = torch.rand(2, 8)
    
    control = head(field_feat, belief)
    print(f"Control output shape: {control.shape}")
    print(f"Control range: [{control.min().item():.3f}, {control.max().item():.3f}]")
    
    # Test full system
    print("\n\nTesting HomeostaticController...")
    
    controller = HomeostaticController(head, cfg, lr=0.01)
    
    for step in range(10):
        control = controller.compute_control(field_feat, belief)
        fake_loss = control.sum()  # Dummy loss
        stats = controller.update(fake_loss, 0.2 - step * 0.01)
        
        if step % 3 == 0:
            print(f"Step {step}: gain={stats['homeo_gain']:.3f}")
