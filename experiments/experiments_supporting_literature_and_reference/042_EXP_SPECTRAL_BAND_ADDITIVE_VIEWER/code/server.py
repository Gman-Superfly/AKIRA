"""
FastAPI Web Server for Spectral Band Additive Viewer

Real-time visualization of attention dynamics with additive wave representation.
"""

import io
import math
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from additive_wave_encoder import WaveConfig, AdditiveWaveEncoder, AttentionWaveAnalyzer
from configs import get_fast_config, get_default_config, get_turbo_config, PATTERN_CHOICES, SUPPORTED_CONFIGS


@dataclass
class TrainingState:
    is_running: bool = False
    should_stop: bool = False
    step: int = 0
    total_steps: int = 0
    last_loss: float = 0.0
    device_str: str = "cpu"
    
    latest_snapshot_png: Optional[bytes] = None
    latest_wave_png: Optional[bytes] = None
    latest_dynamics_png: Optional[bytes] = None
    
    entropy_history: List[float] = field(default_factory=list)
    coherence_history: List[float] = field(default_factory=list)
    head_sync_history: List[float] = field(default_factory=list)
    loss_history: List[float] = field(default_factory=list)
    
    current_superposed_wave: Optional[np.ndarray] = None
    current_individual_waves: Optional[np.ndarray] = None
    current_attention: Optional[np.ndarray] = None
    wave_t: Optional[np.ndarray] = None
    
    lock: threading.Lock = field(default_factory=threading.Lock)
    thread: Optional[threading.Thread] = None


state = TrainingState()
app = FastAPI(title="Spectral Band Additive Viewer")


def _join_thread_if_done():
    """Join finished thread and clear reference."""
    global state
    if state.thread and not state.thread.is_alive():
        try:
            state.thread.join(timeout=0.1)
        except Exception:
            pass
        state.thread = None


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
    
    def concentric_blobs(self, t: int) -> torch.Tensor:
        cx, cy = 0.5, 0.5
        dist = torch.sqrt((self.X - cx) ** 2 + (self.Y - cy) ** 2)
        speed = 0.03
        inner_radius = 0.15 + 0.1 * math.sin(t * speed * 2 * math.pi)
        inner_ring = torch.exp(-((dist - inner_radius) ** 2) / (2 * 0.04 ** 2))
        outer_radius = 0.35 + 0.15 * math.sin(t * speed * 2 * math.pi + math.pi)
        outer_ring = torch.exp(-((dist - outer_radius) ** 2) / (2 * 0.05 ** 2))
        return self._normalize(inner_ring * 1.0 + outer_ring * 0.7)
    
    def frame_for_pattern(self, t: int, pattern_type: str, switch_period: int = 50) -> torch.Tensor:
        if pattern_type == 'blob':
            return self.moving_blob(t)
        if pattern_type == 'interference':
            return self.interference_pattern(t)
        if pattern_type == 'switching':
            return self.switching_frame(t, period=switch_period)
        if pattern_type == 'concentric':
            return self.concentric_blobs(t)
        return self.moving_blob(t)


class SimpleAttentionPredictor(nn.Module):
    """
    Simple attention-based predictor that exposes attention weights for visualization.
    Uses multi-head self-attention on a flattened grid representation.
    """
    
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
            nn.LayerNorm(hidden_dim)
            for _ in range(n_layers)
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
        """
        Forward pass returning prediction and attention weights.
        
        Args:
            frame: Input frame [H, W]
        
        Returns:
            prediction: Predicted next frame [H, W]
            attention_weights: List of attention weights per layer [n_layers x (heads, seq, seq)]
        """
        H, W = frame.shape
        x = frame.reshape(-1, 1)
        x = self.input_proj(x)
        x = x + self.pos_embedding
        x = x.unsqueeze(0)
        
        attention_weights = []
        
        for i, (attn, ln, ffn) in enumerate(zip(self.attention_layers, self.layer_norms, self.ffn_layers)):
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


def make_snapshot_png(current_frame: torch.Tensor, next_frame: torch.Tensor, pred: torch.Tensor, step: int, loss: float) -> bytes:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return b""
    
    cur = current_frame.detach().cpu().numpy()
    tgt = next_frame.detach().cpu().numpy()
    prd = pred.detach().cpu().numpy()
    err = abs(prd - tgt).clip(0.0, 1.0)
    
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    ax = axes.ravel()
    
    im0 = ax[0].imshow(cur, cmap="magma", vmin=0.0, vmax=1.0)
    ax[0].set_title("Current")
    fig.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.04)
    
    im1 = ax[1].imshow(tgt, cmap="magma", vmin=0.0, vmax=1.0)
    ax[1].set_title("Target (t+1)")
    fig.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04)
    
    im2 = ax[2].imshow(prd, cmap="magma", vmin=0.0, vmax=1.0)
    ax[2].set_title("Prediction")
    fig.colorbar(im2, ax=ax[2], fraction=0.046, pad=0.04)
    
    im3 = ax[3].imshow(err, cmap="viridis", vmin=0.0, vmax=1.0)
    ax[3].set_title("Abs Error")
    fig.colorbar(im3, ax=ax[3], fraction=0.046, pad=0.04)
    
    for a in ax:
        a.set_xticks([])
        a.set_yticks([])
    
    fig.suptitle(f"Step {step} | Loss: {loss:.6f}", fontsize=12)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    plt_bytes = buf.getvalue()
    buf.close()
    plt.close(fig)
    return plt_bytes


def make_wave_png(superposed: np.ndarray, individual: np.ndarray, attention: np.ndarray, t: np.ndarray, step: int, coherence: float) -> bytes:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return b""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    ax1 = axes[0, 0]
    ax1.plot(t, superposed, 'b-', linewidth=2, label='Superposed')
    ax1.fill_between(t, -np.abs(superposed), np.abs(superposed), alpha=0.2, color='blue')
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Amplitude")
    ax1.set_title(f"Attention Superposition (Coherence R={coherence:.3f})")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    ax2 = axes[0, 1]
    seq_len = min(12, individual.shape[0])
    for i in range(seq_len):
        alpha = float(attention[i]) * 2 + 0.1
        ax2.plot(t, individual[i], alpha=min(1.0, alpha), linewidth=1, label=f"pos {i}")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Amplitude")
    ax2.set_title(f"Individual Waves (top {seq_len} positions)")
    ax2.grid(True, alpha=0.3)
    
    ax3 = axes[1, 0]
    ax3.bar(range(len(attention)), attention, color='steelblue')
    ax3.set_xlabel("Position")
    ax3.set_ylabel("Attention Weight")
    ax3.set_title("Attention Distribution")
    ax3.grid(True, alpha=0.3)
    
    ax4 = axes[1, 1]
    freqs = np.fft.fftfreq(len(superposed), d=t[1] - t[0])
    fft_vals = np.abs(np.fft.fft(superposed))
    pos_mask = freqs > 0
    ax4.plot(freqs[pos_mask], fft_vals[pos_mask], 'r-', linewidth=1)
    ax4.set_xlabel("Frequency")
    ax4.set_ylabel("Magnitude")
    ax4.set_title("Superposition Spectrum")
    ax4.grid(True, alpha=0.3)
    
    fig.suptitle(f"Additive Wave Analysis - Step {step}", fontsize=14)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    plt_bytes = buf.getvalue()
    buf.close()
    plt.close(fig)
    return plt_bytes


def make_dynamics_png(entropy_history: List[float], coherence_history: List[float], head_sync_history: List[float], loss_history: List[float]) -> bytes:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return b""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    ax1 = axes[0, 0]
    if entropy_history:
        ax1.plot(entropy_history[-200:], 'r-', linewidth=1)
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Entropy")
    ax1.set_title("Attention Entropy (last 200 steps)")
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[0, 1]
    if coherence_history:
        ax2.plot(coherence_history[-200:], 'g-', linewidth=1)
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Coherence R")
    ax2.set_title("Phase Coherence (last 200 steps)")
    ax2.grid(True, alpha=0.3)
    
    ax3 = axes[1, 0]
    if head_sync_history:
        ax3.plot(head_sync_history[-200:], 'b-', linewidth=1)
    ax3.set_xlabel("Step")
    ax3.set_ylabel("Head Sync R")
    ax3.set_title("Head Synchronization (last 200 steps)")
    ax3.grid(True, alpha=0.3)
    
    ax4 = axes[1, 1]
    if loss_history:
        ax4.plot(loss_history[-200:], 'm-', linewidth=1)
    ax4.set_xlabel("Step")
    ax4.set_ylabel("Loss")
    ax4.set_title("Training Loss (last 200 steps)")
    ax4.grid(True, alpha=0.3)
    
    fig.suptitle("Training Dynamics", fontsize=14)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    plt_bytes = buf.getvalue()
    buf.close()
    plt.close(fig)
    return plt_bytes


def training_loop(
    grid_size: int,
    steps: int,
    pattern: str,
    device_str: str,
    hidden_dim: int = 64,
    n_heads: int = 4,
    n_layers: int = 3,
    lr: float = 0.001,
):
    global state
    
    device = select_device(device_str)
    
    with state.lock:
        state.is_running = True
        state.should_stop = False
        state.step = 0
        state.total_steps = steps
        state.device_str = device_str
        state.entropy_history = []
        state.coherence_history = []
        state.head_sync_history = []
        state.loss_history = []
    
    data_gen = SyntheticDataGenerator(grid_size, device)
    model = SimpleAttentionPredictor(grid_size, hidden_dim, n_heads, n_layers, device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    wave_analyzer = AttentionWaveAnalyzer(WaveConfig())
    
    try:
        for step in range(steps):
            with state.lock:
                if state.should_stop:
                    break
            
            current_frame = data_gen.frame_for_pattern(step, pattern)
            next_frame = data_gen.frame_for_pattern(step + 1, pattern)
            
            pred, attention_weights = model(current_frame)
            loss = criterion(pred, next_frame)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            last_layer_attn = attention_weights[-1]
            analysis = wave_analyzer.analyze_layer(last_layer_attn, query_pos=-1)
            
            with state.lock:
                state.step = step
                state.last_loss = loss.item()
                
                state.entropy_history.append(analysis['entropy'])
                state.coherence_history.append(analysis['coherence'])
                state.head_sync_history.append(analysis['head_sync'])
                state.loss_history.append(loss.item())
                
                state.current_superposed_wave = analysis['superposed_wave']
                state.current_individual_waves = analysis['individual_waves']
                state.current_attention = analysis['mean_attention']
                state.wave_t = analysis['t']
                
                if step % 5 == 0:
                    state.latest_snapshot_png = make_snapshot_png(current_frame, next_frame, pred, step, loss.item())
                    state.latest_wave_png = make_wave_png(
                        analysis['superposed_wave'],
                        analysis['individual_waves'],
                        analysis['mean_attention'],
                        analysis['t'],
                        step,
                        analysis['coherence']
                    )
                    state.latest_dynamics_png = make_dynamics_png(
                        state.entropy_history,
                        state.coherence_history,
                        state.head_sync_history,
                        state.loss_history
                    )
    finally:
        with state.lock:
            state.is_running = False
            state.should_stop = False
            # Clear thread reference on exit; start_run will re-assign
            state.thread = None


INDEX_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Spectral Band Additive Viewer</title>
    <style>
        body { font-family: 'Segoe UI', Arial, sans-serif; margin: 0; padding: 20px; background: #1a1a2e; color: #eee; }
        h1 { color: #00d4ff; margin-bottom: 5px; }
        .subtitle { color: #888; margin-bottom: 20px; }
        .container { display: flex; flex-wrap: wrap; gap: 20px; }
        .panel { background: #16213e; border-radius: 8px; padding: 15px; flex: 1; min-width: 300px; }
        .panel h2 { color: #00d4ff; margin-top: 0; font-size: 16px; }
        .controls { display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 20px; }
        .control-group { display: flex; flex-direction: column; }
        label { font-size: 12px; color: #888; margin-bottom: 3px; }
        input, select { padding: 8px; border-radius: 4px; border: 1px solid #333; background: #0f0f23; color: #eee; }
        button { padding: 10px 20px; border-radius: 4px; border: none; cursor: pointer; font-weight: bold; }
        .btn-start { background: #00d4ff; color: #000; }
        .btn-stop { background: #ff4757; color: #fff; }
        .status { padding: 10px; background: #0f0f23; border-radius: 4px; margin-bottom: 10px; }
        .status-running { border-left: 3px solid #00d4ff; }
        .status-stopped { border-left: 3px solid #666; }
        img { max-width: 100%; border-radius: 4px; }
        .image-row { display: flex; flex-wrap: wrap; gap: 20px; }
        .image-panel { flex: 1; min-width: 400px; }
    </style>
</head>
<body>
    <h1>Spectral Band Additive Viewer</h1>
    <p class="subtitle">Real-time attention dynamics visualization with additive wave representation</p>
    
    <div class="controls">
        <div class="control-group">
            <label>Grid Size</label>
            <input type="number" id="gridSize" value="24" min="8" max="64">
        </div>
        <div class="control-group">
            <label>Steps</label>
            <input type="number" id="steps" value="500" min="10" max="5000">
        </div>
        <div class="control-group">
            <label>Pattern</label>
            <select id="pattern">
                <option value="blob">Moving Blob</option>
                <option value="interference">Interference</option>
                <option value="switching">Switching</option>
                <option value="concentric">Concentric Rings</option>
            </select>
        </div>
        <div class="control-group">
            <label>Hidden Dim</label>
            <input type="number" id="hiddenDim" value="32" min="8" max="256" step="8">
        </div>
        <div class="control-group">
            <label>Heads</label>
            <input type="number" id="nHeads" value="4" min="1" max="8">
        </div>
        <div class="control-group">
            <label>Layers</label>
            <input type="number" id="nLayers" value="3" min="1" max="6">
        </div>
        <div class="control-group" style="justify-content: flex-end;">
            <button class="btn-start" onclick="startTraining()">Start</button>
        </div>
        <div class="control-group" style="justify-content: flex-end;">
            <button class="btn-stop" onclick="stopTraining()">Stop</button>
        </div>
    </div>
    
    <div id="status" class="status status-stopped">Ready to start</div>
    
    <div class="image-row">
        <div class="image-panel">
            <h2>Prediction Snapshot</h2>
            <img id="snapshot" src="" alt="Waiting for training...">
        </div>
        <div class="image-panel">
            <h2>Additive Wave Analysis</h2>
            <img id="wave" src="" alt="Waiting for training...">
        </div>
    </div>
    
    <div style="margin-top: 20px;">
        <h2>Training Dynamics</h2>
        <img id="dynamics" src="" alt="Waiting for training..." style="max-width: 100%;">
    </div>
    
    <script>
        let refreshInterval = null;
        
        function adjustHiddenDimForHeads() {
            const headsEl = document.getElementById('nHeads');
            const hiddenEl = document.getElementById('hiddenDim');
            let heads = parseInt(headsEl.value) || 1;
            if (heads < 1) heads = 1;
            let hidden = parseInt(hiddenEl.value) || heads;
            if (hidden < heads) hidden = heads;
            if (hidden % heads !== 0) {
                hidden = Math.ceil(hidden / heads) * heads;
                hiddenEl.value = hidden;
            }
        }

        document.getElementById('nHeads').addEventListener('change', adjustHiddenDimForHeads);
        document.getElementById('hiddenDim').addEventListener('change', adjustHiddenDimForHeads);
        adjustHiddenDimForHeads();

        function startTraining() {
            const payload = {
                grid_size: parseInt(document.getElementById('gridSize').value),
                steps: parseInt(document.getElementById('steps').value),
                pattern: document.getElementById('pattern').value,
                hidden_dim: parseInt(document.getElementById('hiddenDim').value),
                n_heads: parseInt(document.getElementById('nHeads').value),
                n_layers: parseInt(document.getElementById('nLayers').value),
            };
            
            fetch('/run', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(payload)
            }).then(r => r.json()).then(data => {
                console.log('Started:', data);
                startRefreshing();
            });
        }
        
        function stopTraining() {
            fetch('/stop', {method: 'POST'}).then(r => r.json()).then(data => {
                console.log('Stopped:', data);
            });
        }
        
        function refreshStatus() {
            fetch('/status').then(r => r.json()).then(data => {
                const statusEl = document.getElementById('status');
                if (data.is_running) {
                    statusEl.className = 'status status-running';
                    statusEl.textContent = `Running: Step ${data.step}/${data.total_steps} | Loss: ${data.last_loss.toFixed(6)} | Coherence: ${data.coherence.toFixed(3)} | Entropy: ${data.entropy.toFixed(3)}`;
                } else {
                    statusEl.className = 'status status-stopped';
                    statusEl.textContent = data.step > 0 ? `Finished: ${data.step} steps` : 'Ready to start';
                }
                
                if (data.is_running || data.step > 0) {
                    const ts = Date.now();
                    document.getElementById('snapshot').src = '/snapshot.png?' + ts;
                    document.getElementById('wave').src = '/wave.png?' + ts;
                    document.getElementById('dynamics').src = '/dynamics.png?' + ts;
                }
            });
        }
        
        function startRefreshing() {
            if (refreshInterval) clearInterval(refreshInterval);
            refreshInterval = setInterval(refreshStatus, 500);
        }
        
        startRefreshing();
    </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index():
    return INDEX_HTML


@app.get("/status")
def get_status():
    with state.lock:
        return JSONResponse({
            "is_running": state.is_running,
            "step": state.step,
            "total_steps": state.total_steps,
            "last_loss": state.last_loss,
            "entropy": state.entropy_history[-1] if state.entropy_history else 0.0,
            "coherence": state.coherence_history[-1] if state.coherence_history else 0.0,
            "head_sync": state.head_sync_history[-1] if state.head_sync_history else 0.0,
        })


@app.get("/snapshot.png")
def get_snapshot():
    with state.lock:
        png_bytes = state.latest_snapshot_png
    if not png_bytes:
        return StreamingResponse(io.BytesIO(b""), media_type="image/png")
    return StreamingResponse(io.BytesIO(png_bytes), media_type="image/png")


@app.get("/wave.png")
def get_wave():
    with state.lock:
        png_bytes = state.latest_wave_png
    if not png_bytes:
        return StreamingResponse(io.BytesIO(b""), media_type="image/png")
    return StreamingResponse(io.BytesIO(png_bytes), media_type="image/png")


@app.get("/dynamics.png")
def get_dynamics():
    with state.lock:
        png_bytes = state.latest_dynamics_png
    if not png_bytes:
        return StreamingResponse(io.BytesIO(b""), media_type="image/png")
    return StreamingResponse(io.BytesIO(png_bytes), media_type="image/png")


@app.post("/run")
def start_run(payload: Dict):
    global state
    
    _join_thread_if_done()
    with state.lock:
        if state.is_running:
            return JSONResponse({"status": "already_running"})
        if state.thread and state.thread.is_alive():
            return JSONResponse({"status": "thread_busy"})
    
    grid_size = payload.get("grid_size", 24)
    steps = payload.get("steps", 500)
    pattern = payload.get("pattern", "blob")
    hidden_dim = int(payload.get("hidden_dim", 32))
    n_heads = int(payload.get("n_heads", 4))
    n_layers = int(payload.get("n_layers", 3))
    device_str = payload.get("device", "auto")
    
    # Ensure embed_dim divisible by num_heads
    if n_heads < 1:
        n_heads = 1
    if hidden_dim < 1:
        hidden_dim = n_heads
    if hidden_dim % n_heads != 0:
        # Round up to nearest multiple
        hidden_dim = ((hidden_dim + n_heads - 1) // n_heads) * n_heads
        print(f"[server] adjusted hidden_dim to {hidden_dim} to be divisible by n_heads={n_heads}")
    
    thread = threading.Thread(
        target=training_loop,
        kwargs={
            "grid_size": grid_size,
            "steps": steps,
            "pattern": pattern,
            "device_str": device_str,
            "hidden_dim": hidden_dim,
            "n_heads": n_heads,
            "n_layers": n_layers,
        }
    )
    thread.daemon = True
    thread.start()
    
    with state.lock:
        state.thread = thread
    
    return JSONResponse({"status": "started"})


@app.post("/stop")
def stop_run():
    global state
    with state.lock:
        state.should_stop = True
        thread = state.thread
    if thread and thread.is_alive():
        try:
            thread.join(timeout=1.0)
        except Exception:
            pass
    with state.lock:
        state.is_running = False
        state.thread = None
    return JSONResponse({"status": "stopping"})


@app.on_event("shutdown")
def _on_shutdown():
    """Ensure training thread is stopped on server shutdown to avoid hang."""
    global state
    with state.lock:
        state.should_stop = True
        thread = state.thread
    if thread and thread.is_alive():
        try:
            thread.join(timeout=1.0)
        except Exception:
            pass
    with state.lock:
        state.is_running = False
        state.thread = None


if __name__ == "__main__":
    import uvicorn
    print("Starting Spectral Band Additive Viewer on http://localhost:8042")
    uvicorn.run(app, host="0.0.0.0", port=8042)

