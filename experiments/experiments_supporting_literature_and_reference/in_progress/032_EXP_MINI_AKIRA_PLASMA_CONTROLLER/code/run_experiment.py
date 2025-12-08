"""
Run Experiment 032: Mini AKIRA Plasma Controller.

This script orchestrates the complete experiment:
1. Train predictors (7+1 SBM vs baselines)
2. Train control heads (Adam vs Homeostat)
3. Analyze belief-control correlations
4. Run ablation studies

Usage:
    python run_experiment.py                    # Full experiment
    python run_experiment.py --stage predictor  # Train predictors only
    python run_experiment.py --stage control    # Train control only
    python run_experiment.py --quick            # Quick test run
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from mini_plasma_env import MiniPlasmaEnv, PlasmaConfig, generate_trajectories
from spectral_belief_machine import SpectralBeliefMachine, SpectralConfig
from baselines import FlatBaseline, FourBandBaseline, SpectralOnlyBaseline, BaselineConfig
from belief_tracker import BeliefTracker, BeliefConfig
from homeostat import (
    Homeostat, HomeostatConfig, HomeostaticController, 
    ControlHead, PSONOptimizer
)


def train_predictor(
    model: nn.Module,
    train_data: Tuple[torch.Tensor, torch.Tensor],
    epochs: int,
    lr: float,
    device: str,
    use_differential_lr: bool = False,
    predict_delta: bool = True,
    prediction_horizon: int = 1,
) -> Dict[str, List[float]]:
    """
    Train a predictor on next-frame prediction.
    
    Args:
        model: Predictor model
        train_data: (fields, next_fields) tensors
        epochs: Number of training epochs
        lr: Base learning rate
        device: Device to train on
        use_differential_lr: Use per-band learning rates (for SBM)
        predict_delta: If True, train on predicting (y - x) instead of y directly.
                      This penalizes identity copying since the model must predict change.
        prediction_horizon: Steps ahead to predict (k in t+k). Higher = harder task.
                           Note: train_data must have appropriate targets if k > 1.
        
    Returns:
        Dict with training metrics per epoch
    """
    fields, next_fields = train_data
    fields = fields.to(device)
    next_fields = next_fields.to(device)
    
    # Setup optimizer
    if use_differential_lr and hasattr(model, 'get_lr_groups'):
        param_groups = model.get_lr_groups(lr)
        optimizer = optim.Adam(param_groups)
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Training loop
    metrics = {"loss": [], "epoch_time": [], "mode": "delta" if predict_delta else "absolute"}
    batch_size = 32
    num_samples = fields.shape[0]
    
    for epoch in range(epochs):
        epoch_start = time.time()
        epoch_losses = []
        history = []  # Reset history each epoch (shuffled data breaks temporal continuity)
        
        # Shuffle
        perm = torch.randperm(num_samples)
        
        for i in range(0, num_samples, batch_size):
            idx = perm[i:i + batch_size]
            x = fields[idx]
            y = next_fields[idx]
            
            # Forward
            pred, belief = model(x, history if history else None)
            
            if predict_delta:
                # Delta prediction mode:
                # Model outputs pred, but we compute loss on (pred - x) vs (y - x)
                # This forces the model to learn the CHANGE, not the absolute state
                # Identity copying (pred ≈ x) results in pred - x ≈ 0, which is wrong if y != x
                delta_target = y - x
                delta_pred = pred - x
                loss = F.mse_loss(delta_pred, delta_target)
            else:
                # Absolute prediction (old mode)
                loss = F.mse_loss(pred, y)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_losses.append(loss.item())
            
            # Update history (detached)
            if "band_features" in belief:
                history = (history + [belief["band_features"].detach()])[-4:]
        
        epoch_time = time.time() - epoch_start
        mean_loss = sum(epoch_losses) / len(epoch_losses)
        
        metrics["loss"].append(mean_loss)
        metrics["epoch_time"].append(epoch_time)
        
        print(f"  Epoch {epoch + 1}/{epochs}: loss={mean_loss:.6f}, time={epoch_time:.2f}s")
    
    return metrics


def train_controller(
    env: MiniPlasmaEnv,
    predictor: nn.Module,
    use_homeostat: bool,
    steps: int,
    lr: float,
    device: str,
) -> Dict[str, List[float]]:
    """
    Train a control head.
    
    Args:
        env: Plasma environment
        predictor: Trained predictor (frozen)
        use_homeostat: Use homeostatic controller vs Adam
        steps: Number of control steps
        lr: Learning rate
        device: Device
        
    Returns:
        Dict with control metrics
    """
    # Freeze predictor
    for p in predictor.parameters():
        p.requires_grad = False
    
    # Setup control head
    # field is (B, 1, 64, 64) -> pool to (4,4) -> flatten = 1*4*4 = 16
    control_head = ControlHead(
        input_dim=16,  # Pooled field features: 1 channel * 4 * 4
        num_actuators=env.cfg.num_actuators,
        use_belief=True,
        num_bands=8,
    ).to(device)
    
    # Setup optimizer/controller
    if use_homeostat:
        homeo_cfg = HomeostatConfig(setpoint=0.1, noise_scale=0.02)
        controller = HomeostaticController(control_head, homeo_cfg, lr=lr)
    else:
        optimizer = optim.Adam(control_head.parameters(), lr=lr)
        controller = None
    
    # Training loop
    metrics = {
        "control_error": [],
        "field_mse": [],
        "centroid_error": [],
    }
    
    field = env.reset(batch_size=1).to(device)
    target = env.make_target(batch_size=1).to(device)
    history = []
    
    for step in range(steps):
        # Get prediction and belief
        with torch.no_grad():
            pred, belief = predictor(field, history)
        
        # Pool field features
        field_feat = F.adaptive_avg_pool2d(field, (4, 4)).flatten(1)
        
        # Compute control
        control = control_head(field_feat, belief["band_entropy"])
        
        # Step environment (differentiable)
        next_field = env.step(field.detach(), control, noise=False)
        
        # Loss: distance to target
        loss = F.mse_loss(next_field, target)
        
        # Update
        if use_homeostat:
            stats = controller.update(loss, loss.item())
        else:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Compute metrics
        with torch.no_grad():
            env_metrics = env.compute_metrics(next_field, target)
        
        metrics["control_error"].append(loss.item())
        metrics["field_mse"].append(env_metrics["field_mse"].mean().item())
        metrics["centroid_error"].append(env_metrics["centroid_error"].mean().item())
        
        # Advance
        with torch.no_grad():
            field = env.step(field, control.detach(), noise=True)
        
        if "band_features" in belief:
            history = (history + [belief["band_features"].detach()])[-4:]
        
        if step % 50 == 0:
            print(f"  Step {step}: error={loss.item():.4f}, centroid={env_metrics['centroid_error'].mean().item():.2f}")
    
    return metrics


def analyze_belief_control_correlation(
    env: MiniPlasmaEnv,
    predictor: nn.Module,
    control_head: ControlHead,
    steps: int,
    device: str,
) -> Dict[str, float]:
    """
    Analyze correlation between belief entropy and control error.
    
    Tests H2: Does entropy predict control difficulty?
    """
    belief_tracker = BeliefTracker(BeliefConfig())
    
    field = env.reset(batch_size=1).to(device)
    target = env.make_target(batch_size=1).to(device)
    history = []
    
    entropies = []
    errors = []
    future_errors = []
    
    for step in range(steps):
        with torch.no_grad():
            pred, belief = predictor(field, history)
            
            field_feat = F.adaptive_avg_pool2d(field, (4, 4)).flatten(1)
            control = control_head(field_feat, belief["band_entropy"])
            
            next_field = env.step(field, control, noise=True)
            error = F.mse_loss(next_field, target).item()
        
        entropies.append(belief["global_entropy"].item())
        errors.append(error)
        
        # Track for future error correlation
        if step >= 3:
            future_errors.append(error)
        
        belief_tracker.update(belief["band_entropy"], belief["global_entropy"])
        
        field = next_field
        if "band_features" in belief:
            history = (history + [belief["band_features"]])[-4:]
    
    # Compute correlations
    import numpy as np
    
    entropies = np.array(entropies[:-3])  # Align with future errors
    future_errors = np.array(future_errors)
    
    if len(entropies) > 10:
        correlation = np.corrcoef(entropies, future_errors)[0, 1]
    else:
        correlation = 0.0
    
    collapse_summary = belief_tracker.get_collapse_summary()
    
    return {
        "entropy_error_correlation": correlation,
        "mean_entropy": float(np.mean(entropies)),
        "mean_error": float(np.mean(future_errors)),
        **collapse_summary,
    }


def run_full_experiment(args):
    """Run the complete experiment."""
    device = args.device
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    
    print("=" * 60)
    print("EXPERIMENT 032: Mini AKIRA Plasma Controller")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Quick mode: {args.quick}")
    print(f"Difficulty: {args.difficulty}")
    print(f"Prediction mode: {'delta' if args.predict_delta else 'absolute'}")
    print(f"Prediction horizon: t+{args.horizon}")
    print()
    
    # Configurations
    if args.quick:
        num_trajectories = 20
        trajectory_length = 50
        predictor_epochs = 2
        control_steps = 100
    else:
        num_trajectories = 100
        trajectory_length = 100
        predictor_epochs = 10
        control_steps = 500
    
    # Initialize environment with difficulty setting
    print("[1] Initializing environment...")
    if args.difficulty == "easy":
        plasma_cfg = PlasmaConfig.easy(device=device)
    elif args.difficulty == "hard":
        plasma_cfg = PlasmaConfig.hard(device=device)
    else:  # medium (default)
        plasma_cfg = PlasmaConfig.medium(device=device)
    
    print(f"    Diffusion: {plasma_cfg.diffusion}, Advection: {plasma_cfg.advection}")
    print(f"    Noise: {plasma_cfg.noise_std}, Disturbance: {plasma_cfg.disturbance_prob}@{plasma_cfg.disturbance_strength}")
    
    env = MiniPlasmaEnv(plasma_cfg)
    
    # Generate training data with prediction horizon
    print("[2] Generating training data...")
    fields, next_fields, controls = generate_trajectories(
        env, num_trajectories, trajectory_length, 
        control_scale=0.1,
        prediction_horizon=args.horizon
    )
    print(f"    Generated {fields.shape[0]} samples (predicting t+{args.horizon})")
    
    train_data = (fields, next_fields)
    
    # Initialize models
    print("\n[3] Initializing models...")
    
    spectral_cfg = SpectralConfig(device=device)
    sbm = SpectralBeliefMachine(spectral_cfg)
    print(f"    SpectralBeliefMachine: {sum(p.numel() for p in sbm.parameters()):,} params")
    
    baseline_cfg = BaselineConfig(device=device)
    flat = FlatBaseline(baseline_cfg)
    four_band = FourBandBaseline(baseline_cfg)
    spectral_only = SpectralOnlyBaseline(baseline_cfg)
    
    print(f"    FlatBaseline: {sum(p.numel() for p in flat.parameters()):,} params")
    print(f"    FourBandBaseline: {sum(p.numel() for p in four_band.parameters()):,} params")
    print(f"    SpectralOnlyBaseline: {sum(p.numel() for p in spectral_only.parameters()):,} params")
    
    results = {"models": {}, "control": {}, "analysis": {}}
    
    # Stage 1: Train predictors
    if args.stage in ["all", "predictor"]:
        print("\n" + "=" * 60)
        print("STAGE 1: PREDICTOR TRAINING")
        print("=" * 60)
        
        print("\n[3a] Training SpectralBeliefMachine (7+1)...")
        sbm_metrics = train_predictor(
            sbm, train_data, predictor_epochs, lr=0.001, 
            device=device, use_differential_lr=True,
            predict_delta=args.predict_delta,
            prediction_horizon=args.horizon,
        )
        results["models"]["sbm"] = {
            "final_loss": sbm_metrics["loss"][-1],
            "training_time": sum(sbm_metrics["epoch_time"]),
            "mode": sbm_metrics["mode"],
        }
        
        print("\n[3b] Training FlatBaseline...")
        flat_metrics = train_predictor(
            flat, train_data, predictor_epochs, lr=0.001, device=device,
            predict_delta=args.predict_delta,
            prediction_horizon=args.horizon,
        )
        results["models"]["flat"] = {
            "final_loss": flat_metrics["loss"][-1],
            "training_time": sum(flat_metrics["epoch_time"]),
            "mode": flat_metrics["mode"],
        }
        
        print("\n[3c] Training FourBandBaseline...")
        four_metrics = train_predictor(
            four_band, train_data, predictor_epochs, lr=0.001, device=device,
            predict_delta=args.predict_delta,
            prediction_horizon=args.horizon,
        )
        results["models"]["four_band"] = {
            "final_loss": four_metrics["loss"][-1],
            "training_time": sum(four_metrics["epoch_time"]),
            "mode": four_metrics["mode"],
        }
        
        print("\n[3d] Training SpectralOnlyBaseline...")
        spectral_metrics = train_predictor(
            spectral_only, train_data, predictor_epochs, lr=0.001, device=device,
            predict_delta=args.predict_delta,
            prediction_horizon=args.horizon,
        )
        results["models"]["spectral_only"] = {
            "final_loss": spectral_metrics["loss"][-1],
            "training_time": sum(spectral_metrics["epoch_time"]),
            "mode": spectral_metrics["mode"],
        }
        
        # Compare
        print("\n--- Predictor Results ---")
        print(f"SpectralBeliefMachine (7+1): {results['models']['sbm']['final_loss']:.6f}")
        print(f"FlatBaseline:                {results['models']['flat']['final_loss']:.6f}")
        print(f"FourBandBaseline:            {results['models']['four_band']['final_loss']:.6f}")
        print(f"SpectralOnlyBaseline:        {results['models']['spectral_only']['final_loss']:.6f}")
    
    # Stage 2: Train controllers
    if args.stage in ["all", "control"]:
        print("\n" + "=" * 60)
        print("STAGE 2: CONTROL TRAINING")
        print("=" * 60)
        
        print("\n[4a] Training controller with Adam...")
        adam_metrics = train_controller(
            env, sbm, use_homeostat=False, steps=control_steps, lr=0.001, device=device
        )
        results["control"]["adam"] = {
            "final_error": adam_metrics["control_error"][-1],
            "mean_error": sum(adam_metrics["control_error"]) / len(adam_metrics["control_error"]),
            "final_centroid": adam_metrics["centroid_error"][-1],
        }
        
        print("\n[4b] Training controller with Homeostat...")
        homeo_metrics = train_controller(
            env, sbm, use_homeostat=True, steps=control_steps, lr=0.001, device=device
        )
        results["control"]["homeostat"] = {
            "final_error": homeo_metrics["control_error"][-1],
            "mean_error": sum(homeo_metrics["control_error"]) / len(homeo_metrics["control_error"]),
            "final_centroid": homeo_metrics["centroid_error"][-1],
        }
        
        # Compare
        print("\n--- Control Results ---")
        print(f"Adam:      final_error={results['control']['adam']['final_error']:.4f}, mean={results['control']['adam']['mean_error']:.4f}")
        print(f"Homeostat: final_error={results['control']['homeostat']['final_error']:.4f}, mean={results['control']['homeostat']['mean_error']:.4f}")
    
    # Stage 3: Analysis
    if args.stage in ["all", "analysis"]:
        print("\n" + "=" * 60)
        print("STAGE 3: BELIEF-CONTROL ANALYSIS")
        print("=" * 60)
        
        # Create a fresh control head for analysis
        # field is (B, 1, 64, 64) -> pool to (4,4) -> flatten = 1*4*4 = 16
        control_head = ControlHead(
            input_dim=16, num_actuators=env.cfg.num_actuators,
            use_belief=True, num_bands=8
        ).to(device)
        
        print("\n[5] Analyzing belief-control correlation...")
        correlation_results = analyze_belief_control_correlation(
            env, sbm, control_head, steps=control_steps, device=device
        )
        results["analysis"] = correlation_results
        
        print(f"\n--- Analysis Results ---")
        print(f"Entropy-Error Correlation: r={correlation_results['entropy_error_correlation']:.4f}")
        print(f"Mean Entropy: {correlation_results['mean_entropy']:.4f}")
        print(f"Collapse Rate: {correlation_results['collapse_rate']:.4f}")
    
    # Save results
    print("\n" + "=" * 60)
    print("SAVING RESULTS")
    print("=" * 60)
    
    results_file = results_dir / f"results_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {results_file}")
    
    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Experiment 032: Mini AKIRA Plasma Controller"
    )
    parser.add_argument(
        "--device", default="cpu",
        help="Device to run on (cpu/cuda)"
    )
    parser.add_argument(
        "--stage", default="all",
        choices=["all", "predictor", "control", "analysis"],
        help="Which stage to run"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick test run with reduced data/epochs"
    )
    parser.add_argument(
        "--difficulty", default="medium",
        choices=["easy", "medium", "hard"],
        help="Environment difficulty (easy=old defaults, medium=new, hard=challenging)"
    )
    parser.add_argument(
        "--predict-delta", action="store_true", default=True,
        help="Train on delta prediction (y-x) instead of absolute (y). Default: True"
    )
    parser.add_argument(
        "--no-predict-delta", action="store_false", dest="predict_delta",
        help="Disable delta prediction, use absolute mode (old behavior)"
    )
    parser.add_argument(
        "--horizon", type=int, default=3,
        help="Prediction horizon: predict t+k steps ahead. Higher penalizes identity copying. Default: 3"
    )
    
    args = parser.parse_args()
    run_full_experiment(args)


if __name__ == "__main__":
    main()
