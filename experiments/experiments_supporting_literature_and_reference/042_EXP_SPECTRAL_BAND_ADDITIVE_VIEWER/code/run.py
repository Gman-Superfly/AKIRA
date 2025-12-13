"""
CLI entry point for Spectral Band Additive Viewer.

Usage:
    python run.py --mode web      # Start web server
    python run.py --mode cli      # Run training with CLI output
"""

import argparse
import math
import time

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from additive_wave_encoder import WaveConfig, AttentionWaveAnalyzer


class SyntheticDataGenerator:
    """Generate simple 2D sequences for testing."""
    
    def __init__(self, grid_size: int = 32, device: torch.device = torch.device('cpu')):
        self.grid_size = grid_size
        self.device = device
        x = torch.linspace(0, 1, grid_size, device=device)
        y = torch.linspace(0, 1, grid_size, device=device)
        self.Y, self.X = torch.meshgrid(y, x, indexing='ij')
    
    def _gaussian(self, cx: float, cy: float, sigma: float = 0.1) -> torch.Tensor:
        return torch.exp(-((self.X - cx) ** 2 + (self.Y - cy) ** 2) / (2 * sigma ** 2))
    
    def _normalize(self, field: torch.Tensor) -> torch.Tensor:
        field_min = field.min()
        field_max = field.max()
        return (field - field_min) / (field_max - field_min + 1e-8)
    
    def moving_blob(self, t: int, speed: float = 0.02) -> torch.Tensor:
        angle = t * speed * 2 * math.pi
        cx = 0.5 + 0.3 * math.cos(angle)
        cy = 0.5 + 0.3 * math.sin(angle)
        return self._gaussian(cx, cy, sigma=0.1)
    
    def interference_pattern(self, t: int, freq: float = 0.1) -> torch.Tensor:
        k = 10.0
        phase = t * freq
        wave1 = torch.sin(k * self.X + phase)
        wave2 = torch.sin(k * self.Y + phase * 0.7)
        return ((wave1 + wave2) / 2 + 1) / 2
    
    def switching_frame(self, t: int, period: int = 50) -> torch.Tensor:
        phase_block = (t // period) % 2
        if phase_block == 0:
            return self.moving_blob(t)
        return self.interference_pattern(t)
    
    def frame_for_pattern(self, t: int, pattern_type: str, switch_period: int = 50) -> torch.Tensor:
        if pattern_type == 'blob':
            return self.moving_blob(t)
        if pattern_type == 'interference':
            return self.interference_pattern(t)
        if pattern_type == 'switching':
            return self.switching_frame(t, period=switch_period)
        return self.moving_blob(t)


class SimpleAttentionPredictor(nn.Module):
    """Simple attention-based predictor."""
    
    def __init__(self, grid_size: int, hidden_dim: int = 64, n_heads: int = 4, n_layers: int = 3, device: torch.device = torch.device('cpu')):
        super().__init__()
        self.grid_size = grid_size
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.device = device
        
        self.input_proj = nn.Linear(1, hidden_dim)
        self.pos_embedding = nn.Parameter(torch.randn(grid_size * grid_size, hidden_dim) * 0.02)
        
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(hidden_dim, n_heads, batch_first=True)
            for _ in range(n_layers)
        ])
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(n_layers)
        ])
        self.ffn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.ReLU(),
                nn.Linear(hidden_dim * 2, hidden_dim)
            )
            for _ in range(n_layers)
        ])
        
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        self.to(device)
    
    def forward(self, frame: torch.Tensor) -> tuple:
        H, W = frame.shape
        x = frame.reshape(-1, 1)
        x = self.input_proj(x)
        x = x + self.pos_embedding
        x = x.unsqueeze(0)
        
        attention_weights = []
        
        for attn, ln, ffn in zip(self.attention_layers, self.layer_norms, self.ffn_layers):
            attn_out, attn_w = attn(x, x, x, need_weights=True, average_attn_weights=False)
            attention_weights.append(attn_w.detach().cpu().numpy()[0])
            x = ln(x + attn_out)
            x = x + ffn(x)
        
        x = x.squeeze(0)
        pred = self.output_proj(x).reshape(H, W)
        
        return pred, attention_weights


def select_device(device: str) -> torch.device:
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


def run_cli(args):
    """Run training with CLI output."""
    device = select_device(args.device)
    print(f"Using device: {device}")
    
    data_gen = SyntheticDataGenerator(args.grid_size, device)
    model = SimpleAttentionPredictor(args.grid_size, args.hidden_dim, args.n_heads, args.n_layers, device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    wave_analyzer = AttentionWaveAnalyzer(WaveConfig())
    
    print(f"\nTraining for {args.steps} steps on pattern '{args.pattern}'...")
    print("=" * 80)
    
    start_time = time.time()
    
    for step in range(args.steps):
        current_frame = data_gen.frame_for_pattern(step, args.pattern)
        next_frame = data_gen.frame_for_pattern(step + 1, args.pattern)
        
        pred, attention_weights = model(current_frame)
        loss = criterion(pred, next_frame)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if step % args.log_interval == 0:
            last_layer_attn = attention_weights[-1]
            analysis = wave_analyzer.analyze_layer(last_layer_attn, query_pos=-1)
            
            print(f"Step {step:5d} | Loss: {loss.item():.6f} | "
                  f"Entropy: {analysis['entropy']:.3f} | "
                  f"Coherence: {analysis['coherence']:.3f} | "
                  f"HeadSync: {analysis['head_sync']:.3f}")
    
    elapsed = time.time() - start_time
    print("=" * 80)
    print(f"Training complete in {elapsed:.2f}s ({args.steps / elapsed:.1f} steps/sec)")


def run_web(args):
    """Start web server."""
    import uvicorn
    from server import app
    print(f"Starting Spectral Band Additive Viewer on http://localhost:{args.port}")
    uvicorn.run(app, host="0.0.0.0", port=args.port)


def main():
    parser = argparse.ArgumentParser(description="Spectral Band Additive Viewer")
    parser.add_argument("--mode", type=str, default="web", choices=["web", "cli"],
                        help="Run mode: web (server) or cli (training)")
    parser.add_argument("--port", type=int, default=8042, help="Web server port")
    parser.add_argument("--grid-size", type=int, default=24, help="Grid size")
    parser.add_argument("--steps", type=int, default=500, help="Training steps")
    parser.add_argument("--pattern", type=str, default="blob",
                        choices=["blob", "interference", "switching"],
                        help="Training pattern")
    parser.add_argument("--hidden-dim", type=int, default=32, help="Hidden dimension")
    parser.add_argument("--n-heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--n-layers", type=int, default=3, help="Number of layers")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cpu", "cuda", "mps"],
                        help="Device to use")
    parser.add_argument("--log-interval", type=int, default=20, help="Log interval")
    
    args = parser.parse_args()
    
    if args.mode == "web":
        run_web(args)
    else:
        run_cli(args)


if __name__ == "__main__":
    main()

