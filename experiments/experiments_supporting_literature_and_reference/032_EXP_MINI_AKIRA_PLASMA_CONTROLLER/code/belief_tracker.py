"""
Belief Tracker for Experiment 032.

Implements explicit belief state tracking as required by AKIRA theory:
- Entropy per band
- Global entropy
- Entropy rate (dH/dt)
- Collapse detection (sharp entropy drops)
- Temperature parameter for phase transition control

Reference: SPECTRAL_BELIEF_MACHINE.md Section 8 (Collapse as Phase Transition)
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch


@dataclass
class BeliefConfig:
    """Configuration for belief tracking."""
    num_bands: int = 8  # 7 spectral + 1 temporal
    history_len: int = 20  # Steps to track for rate computation
    collapse_threshold: float = 0.3  # |dH/dt| threshold for collapse detection
    temperature_init: float = 1.0
    temperature_min: float = 0.1
    temperature_max: float = 5.0


class BeliefTracker:
    """
    Tracks belief state dynamics across training/inference.
    
    Key responsibilities:
    1. Track entropy per band over time
    2. Compute entropy rate (dH/dt)
    3. Detect collapse events (sharp entropy drops)
    4. Maintain temperature parameter
    5. Provide belief-based signals for control
    """
    
    def __init__(self, cfg: BeliefConfig):
        self.cfg = cfg
        
        # History buffers for each band
        self.entropy_history: List[deque] = [
            deque(maxlen=cfg.history_len) for _ in range(cfg.num_bands)
        ]
        self.global_entropy_history: deque = deque(maxlen=cfg.history_len)
        
        # Temperature (learnable or adaptive)
        self.temperature = cfg.temperature_init
        
        # Collapse event log
        self.collapse_events: List[Dict] = []
        
        # Running statistics
        self.step_count = 0
        self.total_collapses = 0
        
    def reset(self):
        """Reset tracker state (e.g., at episode start)."""
        for h in self.entropy_history:
            h.clear()
        self.global_entropy_history.clear()
        self.collapse_events.clear()
        self.step_count = 0
        
    def update(
        self,
        band_entropy: torch.Tensor,
        global_entropy: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Update belief tracker with new entropy observations.
        
        Args:
            band_entropy: (B, num_bands) entropy per band
            global_entropy: (B,) global entropy (computed if not provided)
            
        Returns:
            Dict with:
            - dH_dt: (B, num_bands) entropy rate per band
            - dH_dt_global: (B,) global entropy rate
            - is_collapsing: (B, num_bands) bool mask for collapse events
            - collapse_magnitude: (B,) magnitude of any collapse
        """
        B = band_entropy.shape[0]
        device = band_entropy.device
        
        if global_entropy is None:
            global_entropy = band_entropy.sum(dim=1)
        
        # Store current values (use mean across batch for history)
        for i in range(self.cfg.num_bands):
            self.entropy_history[i].append(band_entropy[:, i].mean().item())
        self.global_entropy_history.append(global_entropy.mean().item())
        
        # Compute entropy rates
        dH_dt = torch.zeros_like(band_entropy)
        if len(self.entropy_history[0]) >= 2:
            for i in range(self.cfg.num_bands):
                hist = list(self.entropy_history[i])
                # Simple finite difference
                dH_dt[:, i] = hist[-1] - hist[-2]
        
        dH_dt_global = torch.zeros(B, device=device)
        if len(self.global_entropy_history) >= 2:
            hist = list(self.global_entropy_history)
            dH_dt_global[:] = hist[-1] - hist[-2]
        
        # Detect collapse events (sharp negative entropy rate)
        is_collapsing = dH_dt < -self.cfg.collapse_threshold
        collapse_magnitude = (-dH_dt * is_collapsing.float()).sum(dim=1)
        
        # Log collapse events
        if is_collapsing.any():
            self.total_collapses += is_collapsing.sum().item()
            self.collapse_events.append({
                "step": self.step_count,
                "bands": is_collapsing[0].nonzero().flatten().tolist(),
                "magnitude": collapse_magnitude[0].item(),
            })
        
        self.step_count += 1
        
        return {
            "dH_dt": dH_dt,
            "dH_dt_global": dH_dt_global,
            "is_collapsing": is_collapsing,
            "collapse_magnitude": collapse_magnitude,
        }
    
    def get_belief_state(self) -> Dict[str, float]:
        """
        Get current belief state summary.
        
        Returns:
            Dict with current entropy values and rates.
        """
        state = {
            "temperature": self.temperature,
            "step_count": self.step_count,
            "total_collapses": self.total_collapses,
        }
        
        # Current entropy per band
        for i in range(self.cfg.num_bands):
            if len(self.entropy_history[i]) > 0:
                state[f"H_band_{i}"] = self.entropy_history[i][-1]
            else:
                state[f"H_band_{i}"] = 0.0
        
        # Global entropy
        if len(self.global_entropy_history) > 0:
            state["H_global"] = self.global_entropy_history[-1]
        else:
            state["H_global"] = 0.0
        
        # Entropy rate (smoothed over last few steps)
        if len(self.global_entropy_history) >= 3:
            recent = list(self.global_entropy_history)[-3:]
            state["dH_dt_smoothed"] = (recent[-1] - recent[0]) / 2
        else:
            state["dH_dt_smoothed"] = 0.0
        
        return state
    
    def adjust_temperature(self, target_entropy: float, current_entropy: float):
        """
        Adapt temperature based on entropy dynamics.
        
        If entropy is too high (diffuse), lower temperature to encourage collapse.
        If entropy is too low (stuck), raise temperature to encourage exploration.
        
        Args:
            target_entropy: Desired entropy level
            current_entropy: Current global entropy
        """
        error = current_entropy - target_entropy
        
        # Simple proportional control
        adjustment = 0.01 * error
        self.temperature = max(
            self.cfg.temperature_min,
            min(self.cfg.temperature_max, self.temperature + adjustment)
        )
    
    def get_collapse_summary(self) -> Dict[str, float]:
        """
        Get summary statistics about collapse events.
        
        Returns:
            Dict with collapse statistics.
        """
        if len(self.collapse_events) == 0:
            return {
                "num_collapses": 0,
                "mean_magnitude": 0.0,
                "collapse_rate": 0.0,
            }
        
        magnitudes = [e["magnitude"] for e in self.collapse_events]
        
        return {
            "num_collapses": len(self.collapse_events),
            "mean_magnitude": sum(magnitudes) / len(magnitudes),
            "collapse_rate": len(self.collapse_events) / max(1, self.step_count),
        }
    
    def compute_prediction_confidence(
        self,
        band_entropy: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute confidence score based on belief state.
        
        Low entropy = high confidence (belief has collapsed)
        High entropy = low confidence (belief is diffuse)
        
        Args:
            band_entropy: (B, num_bands) entropy per band
            
        Returns:
            (B,) confidence scores in [0, 1]
        """
        # Normalize entropy to [0, 1] range
        # Assuming max entropy is roughly log(H*W) for spatial bands
        max_entropy = 10.0  # Approximate
        normalized = band_entropy.sum(dim=1) / (self.cfg.num_bands * max_entropy)
        normalized = torch.clamp(normalized, 0, 1)
        
        # Confidence is inverse of normalized entropy
        confidence = 1.0 - normalized
        
        return confidence


class CollapseDetector:
    """
    Dedicated collapse event detection and analysis.
    
    Collapse = sharp transition from high to low entropy
    Corresponds to belief crystallization in AKIRA theory.
    """
    
    def __init__(
        self,
        threshold: float = 0.3,
        min_pre_entropy: float = 0.5,
        window_size: int = 5,
    ):
        self.threshold = threshold
        self.min_pre_entropy = min_pre_entropy
        self.window_size = window_size
        
        self.entropy_buffer: deque = deque(maxlen=window_size)
        
    def update(self, entropy: float) -> Tuple[bool, float]:
        """
        Check if current entropy indicates a collapse event.
        
        Args:
            entropy: Current entropy value
            
        Returns:
            Tuple of (is_collapse, collapse_magnitude)
        """
        self.entropy_buffer.append(entropy)
        
        if len(self.entropy_buffer) < self.window_size:
            return False, 0.0
        
        buffer = list(self.entropy_buffer)
        
        # Check if we had high entropy before
        pre_entropy = max(buffer[:-1])
        if pre_entropy < self.min_pre_entropy:
            return False, 0.0
        
        # Check for sharp drop
        drop = pre_entropy - entropy
        if drop > self.threshold:
            return True, drop
        
        return False, 0.0
    
    def reset(self):
        """Reset detector state."""
        self.entropy_buffer.clear()


def compute_attention_entropy(attention: torch.Tensor) -> torch.Tensor:
    """
    Compute entropy of attention distribution.
    
    Args:
        attention: (B, ..., N) attention weights (should sum to 1 along last dim)
        
    Returns:
        (B, ...) entropy values
    """
    # Ensure numerical stability
    attention = attention.clamp(min=1e-9)
    
    # Shannon entropy
    entropy = -(attention * torch.log(attention)).sum(dim=-1)
    
    return entropy


def compute_normalized_entropy(attention: torch.Tensor) -> torch.Tensor:
    """
    Compute normalized entropy (0 = deterministic, 1 = uniform).
    
    Args:
        attention: (B, ..., N) attention weights
        
    Returns:
        (B, ...) normalized entropy in [0, 1]
    """
    N = attention.shape[-1]
    max_entropy = torch.log(torch.tensor(N, dtype=attention.dtype, device=attention.device))
    
    entropy = compute_attention_entropy(attention)
    normalized = entropy / max_entropy
    
    return normalized


if __name__ == "__main__":
    # Quick test
    cfg = BeliefConfig()
    tracker = BeliefTracker(cfg)
    
    print("Testing BeliefTracker...")
    
    # Simulate some steps
    for step in range(30):
        # Simulated entropy that decreases (collapse)
        if step < 10:
            entropy = torch.rand(2, 8) * 2 + 1  # High entropy
        elif step < 15:
            entropy = torch.rand(2, 8) * 0.5 + 0.5  # Dropping
        else:
            entropy = torch.rand(2, 8) * 0.2  # Low entropy
        
        result = tracker.update(entropy)
        
        if result["is_collapsing"].any():
            print(f"Step {step}: COLLAPSE detected! Magnitude: {result['collapse_magnitude'][0]:.3f}")
    
    state = tracker.get_belief_state()
    print(f"\nFinal state: H_global={state['H_global']:.3f}, temperature={state['temperature']:.3f}")
    
    summary = tracker.get_collapse_summary()
    print(f"Collapse summary: {summary}")
    
    # Test confidence
    entropy = torch.rand(2, 8) * 0.5
    confidence = tracker.compute_prediction_confidence(entropy)
    print(f"\nConfidence from low entropy: {confidence}")
    
    entropy = torch.rand(2, 8) * 2
    confidence = tracker.compute_prediction_confidence(entropy)
    print(f"Confidence from high entropy: {confidence}")
