"""
FastAPI Web Server for Spectral Band Graph Viewer

Real-time visualization of a graph learner: predicts next node in a random walk.
Shows true vs predicted adjacency and training dynamics.
"""

import io
import math
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Dict, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse


@dataclass
class TrainingState:
    is_running: bool = False
    should_stop: bool = False
    step: int = 0
    total_steps: int = 0
    last_loss: float = 0.0
    device_str: str = "cpu"

    latest_graph_png: Optional[bytes] = None
    latest_dynamics_png: Optional[bytes] = None

    loss_history: List[float] = field(default_factory=list)
    acc_history: List[float] = field(default_factory=list)

    true_adj: Optional[np.ndarray] = None
    pred_adj: Optional[np.ndarray] = None

    correct: int = 0
    total: int = 0
    current_node: int = 0
    predicted_node: int = 0
    target_node: int = 0
    was_correct: bool = False
    pred_version: int = 0  # increments whenever belief map updates

    # Cached layout for JS viewer
    node_positions: Optional[np.ndarray] = None

    lock: threading.Lock = field(default_factory=threading.Lock)
    thread: Optional[threading.Thread] = None


state = TrainingState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan handler for startup/shutdown events."""
    yield
    # Shutdown: stop training thread
    global state
    with state.lock:
        state.should_stop = True
        state.is_running = False
        state.thread = None


app = FastAPI(title="Spectral Band Graph Viewer", lifespan=lifespan)


def _join_thread_if_done():
    """Join finished thread and clear reference."""
    global state
    if state.thread and not state.thread.is_alive():
        try:
            state.thread.join(timeout=0.1)
        except Exception:
            pass
        state.thread = None


# ---------------- Graph components ---------------- #

class GraphEnvironment:
    """Simple directed graph and random-walk generator."""

    def __init__(
        self,
        num_nodes: int,
        edge_prob: float = 0.2,
        graph_type: str = "random",
        seed: int = 1,
        device: torch.device = torch.device("cpu"),
    ):
        self.num_nodes = num_nodes
        self.edge_prob = edge_prob
        self.graph_type = graph_type
        self.device = device
        rng = torch.Generator().manual_seed(seed)

        def ensure_outgoing(adj: torch.Tensor):
            for i in range(adj.shape[0]):
                if adj[i].sum() == 0:
                    j = torch.randint(0, adj.shape[0], (1,), generator=rng).item()
                    if j == i:
                        j = (j + 1) % adj.shape[0]
                    adj[i, j] = 1.0
            return adj

        if graph_type == "cycle":
            adj = torch.zeros(num_nodes, num_nodes)
            for i in range(num_nodes):
                adj[i, (i + 1) % num_nodes] = 1.0
        elif graph_type == "tree":
            adj = torch.zeros(num_nodes, num_nodes)
            for i in range(num_nodes):
                left = 2 * i + 1
                right = 2 * i + 2
                if left < num_nodes:
                    adj[i, left] = 1.0
                if right < num_nodes:
                    adj[i, right] = 1.0
            adj = ensure_outgoing(adj)
        elif graph_type == "grid":
            adj = torch.zeros(num_nodes, num_nodes)
            side = int(math.sqrt(num_nodes))
            for i in range(num_nodes):
                row, col = divmod(i, side)
                if col < side - 1 and i + 1 < num_nodes:
                    adj[i, i + 1] = 1.0
                if row < side - 1 and i + side < num_nodes:
                    adj[i, i + side] = 1.0
            adj = ensure_outgoing(adj)
        elif graph_type == "hub":
            adj = torch.zeros(num_nodes, num_nodes)
            for i in range(1, num_nodes):
                adj[0, i] = 1.0
                adj[i, 0] = 1.0
                adj[i, (i % (num_nodes - 1)) + 1] = 1.0
        else:  # random
            adj = (torch.rand(num_nodes, num_nodes, generator=rng) < edge_prob).float()
            adj.fill_diagonal_(0.0)
            adj = ensure_outgoing(adj)

        self.adj = adj.to(device)

    def random_walk(self, length: int, rng: torch.Generator) -> torch.Tensor:
        """Generate a random walk of given length."""
        cur = torch.randint(0, self.num_nodes, (1,), generator=rng, device=self.device)
        walk = [cur.item()]
        for _ in range(length - 1):
            probs = self.adj[cur].squeeze(0)
            if probs.sum() == 0:
                probs = torch.ones_like(probs) / self.num_nodes
            else:
                probs = probs / probs.sum()
            cur = torch.multinomial(probs, num_samples=1, generator=rng)
            walk.append(cur.item())
        return torch.tensor(walk, device=self.device, dtype=torch.long)


class GraphPredictor(nn.Module):
    """Predict next node given current node (random-walk modeling)."""

    def __init__(self, num_nodes: int, hidden_dim: int = 64, n_heads: int = 4, n_layers: int = 2, device: torch.device = torch.device("cpu")):
        super().__init__()
        if hidden_dim % n_heads != 0:
            hidden_dim = ((hidden_dim + n_heads - 1) // n_heads) * n_heads

        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.n_layers = n_layers

        self.node_embed = nn.Embedding(num_nodes, hidden_dim)
        self.pos_embed = nn.Parameter(torch.zeros(2, hidden_dim))  # src, dst

        self.attn_layers = nn.ModuleList(
            [nn.MultiheadAttention(hidden_dim, n_heads, batch_first=True) for _ in range(n_layers)]
        )
        self.ln_layers = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(n_layers)])
        self.ff_layers = nn.ModuleList(
            [nn.Sequential(nn.Linear(hidden_dim, hidden_dim * 2), nn.ReLU(), nn.Linear(hidden_dim * 2, hidden_dim)) for _ in range(n_layers)]
        )

        self.out_proj = nn.Linear(hidden_dim, num_nodes)
        self.to(device)

    def forward(self, src_nodes: torch.Tensor) -> tuple:
        """
        Args:
            src_nodes: [B] current nodes
        Returns:
            logits: [B, num_nodes]
        """
        B = src_nodes.shape[0]
        src_emb = self.node_embed(src_nodes) + self.pos_embed[0]
        # Build a tiny sequence [src]
        x = src_emb.unsqueeze(1)  # [B, 1, D]

        attn_weights = []
        for attn, ln, ff in zip(self.attn_layers, self.ln_layers, self.ff_layers):
            out, w = attn(x, x, x, need_weights=True, average_attn_weights=False)
            attn_weights.append(w.detach().cpu().numpy())
            x = ln(x + out)
            x = x + ff(x)

        logits = self.out_proj(x.squeeze(1))
        return logits, attn_weights


def select_device(device: str) -> torch.device:
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


def make_graph_snapshot_png(
    true_adj: np.ndarray,
    belief_adj: np.ndarray,
    current_node: int,
    predicted_node: int,
    actual_next_node: int,
    was_correct: bool,
    step: int,
    loss: float,
    acc: float,
    correct: int,
    total: int,
) -> bytes:
    """Visualization similar to original graph demo: true graph, belief, metrics."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return b""

    num_nodes = true_adj.shape[0]
    import numpy as np

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=100)
    pos_angles = np.linspace(0, 2 * np.pi, num_nodes, endpoint=False)
    positions = np.column_stack([np.cos(pos_angles), np.sin(pos_angles)])

    # Colors
    node_colors = []
    for i in range(num_nodes):
        if i == current_node:
            node_colors.append("red")
        elif i == actual_next_node:
            node_colors.append("lime")
        elif i == predicted_node and not was_correct:
            node_colors.append("dodgerblue")
        else:
            node_colors.append("lightgray")

    # Left: true graph
    ax = axes[0]
    for i in range(num_nodes):
        for j in range(num_nodes):
            if true_adj[i, j] > 0:
                ax.annotate(
                    "",
                    xy=positions[j],
                    xytext=positions[i],
                    arrowprops=dict(arrowstyle="->", color="gray", alpha=0.4, lw=1.5),
                )
    ax.annotate(
        "",
        xy=positions[actual_next_node],
        xytext=positions[current_node],
        arrowprops=dict(arrowstyle="->", color="blue", alpha=1.0, lw=3),
    )
    ax.scatter(
        positions[:, 0],
        positions[:, 1],
        c=node_colors,
        s=500,
        zorder=5,
        edgecolors="black",
        linewidths=2,
    )
    for i in range(num_nodes):
        ax.annotate(str(i), positions[i], ha="center", va="center", fontsize=11, fontweight="bold")
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect("equal")
    ax.set_title(f"True graph (current {current_node} → target {actual_next_node})")
    ax.axis("off")

    # Middle: belief graph (top edges)
    ax2 = axes[1]
    for i in range(num_nodes):
        row = belief_adj[i]
        top_idx = np.argsort(row)[-2:]
        for j in top_idx:
            if row[j] > 0.15:
                ax2.annotate(
                    "",
                    xy=positions[j],
                    xytext=positions[i],
                    arrowprops=dict(
                        arrowstyle="->",
                        color="green",
                        alpha=min(1.0, row[j]),
                        lw=row[j] * 2.5,
                    ),
                )
    ax2.annotate(
        "",
        xy=positions[predicted_node],
        xytext=positions[current_node],
        arrowprops=dict(arrowstyle="->", color="orange", alpha=0.9, lw=4),
    )
    ax2.scatter(
        positions[:, 0],
        positions[:, 1],
        c=node_colors,
        s=500,
        zorder=5,
        edgecolors="black",
        linewidths=2,
    )
    for i in range(num_nodes):
        ax2.annotate(str(i), positions[i], ha="center", va="center", fontsize=11, fontweight="bold")
    ax2.set_xlim(-1.5, 1.5)
    ax2.set_ylim(-1.5, 1.5)
    ax2.set_aspect("equal")
    ax2.set_title("Belief graph (top edges)")
    ax2.axis("off")

    # Right: metrics text
    ax3 = axes[2]
    result_indicator = "✓ CORRECT" if was_correct else "✗ WRONG"
    cumulative_acc = (correct / total * 100) if total > 0 else 0.0
    metrics_text = f"""Step: {step}
Transition: {current_node} → {actual_next_node}
Predicted:  {predicted_node}
Result: {result_indicator}

Learning:
  Correct: {correct} / {total}
  Acc (running): {cumulative_acc:.1f}%
  Acc (last): {acc:.3f}
  Loss: {loss:.4f}
"""
    ax3.text(0.05, 0.95, metrics_text, ha="left", va="top", fontsize=10, transform=ax3.transAxes, family="monospace")
    ax3.set_title("Metrics")
    ax3.axis("off")

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110)
    plt_bytes = buf.getvalue()
    buf.close()
    try:
        plt.close(fig)
    except Exception:
        pass
    return plt_bytes


def make_dynamics_png(loss_hist: List[float], acc_hist: List[float]) -> bytes:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return b""

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    if loss_hist:
        axes[0].plot(loss_hist[-200:], 'm-', linewidth=1)
    axes[0].set_title("Loss (last 200)")
    axes[0].grid(True, alpha=0.3)

    if acc_hist:
        axes[1].plot(acc_hist[-200:], 'g-', linewidth=1)
    axes[1].set_title("Accuracy (last 200)")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110)
    plt_bytes = buf.getvalue()
    buf.close()
    try:
        plt.close(fig)
    except Exception:
        pass
    return plt_bytes


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


def make_dynamics_png(loss_hist: List[float], acc_hist: List[float]) -> bytes:
    """Plot loss and accuracy history."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return b""
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    if loss_hist:
        axes[0].plot(loss_hist[-200:], 'm-', linewidth=1)
    axes[0].set_title("Loss (last 200)")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)
    
    if acc_hist:
        axes[1].plot(acc_hist[-200:], 'g-', linewidth=1)
    axes[1].set_title("Accuracy (last 200)")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Acc")
    axes[1].grid(True, alpha=0.3)
    
    fig.tight_layout()
    
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    plt_bytes = buf.getvalue()
    buf.close()
    plt.close(fig)
    return plt_bytes


def training_loop(
    num_nodes: int,
    steps: int,
    device_str: str,
    hidden_dim: int = 64,
    n_heads: int = 4,
    n_layers: int = 2,
    lr: float = 0.001,
    edge_prob: float = 0.2,
    graph_type: str = "random",
):
    global state
    
    device = select_device(device_str)
    
    with state.lock:
        state.is_running = True
        state.should_stop = False
        state.step = 0
        state.total_steps = steps
        state.device_str = device_str
        state.loss_history = []
        state.acc_history = []
        state.correct = 0
        state.total = 0
        state.node_positions = None
    
    rng = torch.Generator(device=device).manual_seed(1234)
    env = GraphEnvironment(num_nodes, edge_prob=edge_prob, graph_type=graph_type, seed=1234, device=device)
    model = GraphPredictor(num_nodes, hidden_dim, n_heads, n_layers, device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    try:
        for step in range(steps):
            with state.lock:
                if state.should_stop:
                    break

            walk = env.random_walk(length=2, rng=rng)  # [src, dst]
            src = walk[0:1]
            dst = walk[1]

            logits, attn_weights = model(src)
            loss = criterion(logits, dst.unsqueeze(0))
            pred_idx = logits.argmax(dim=-1)
            acc = float((pred_idx == dst).float().item())
            was_correct = acc == 1.0
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            with state.lock:
                state.step = step
                state.last_loss = loss.item()
                state.loss_history.append(loss.item())
                state.acc_history.append(acc)
                state.total += 1
                if was_correct:
                    state.correct += 1
                state.current_node = src.item()
                state.predicted_node = pred_idx.item()
                state.target_node = dst.item()
                state.was_correct = was_correct

                with torch.no_grad():
                    # Compute belief matrix conditioned on each possible current node
                    logits_all = []
                    for node_idx in range(num_nodes):
                        node_tensor = torch.tensor([node_idx], device=device)
                        logits_i, _ = model(node_tensor)
                        logits_all.append(logits_i)
                    logits_all = torch.cat(logits_all, dim=0)  # [N, N]
                    probs = F.softmax(logits_all, dim=-1)  # [N, N]
                    pred_adj = probs.cpu().numpy()
                    true_adj = env.adj.cpu().numpy()
                    state.true_adj = true_adj
                    state.pred_adj = pred_adj
                    if state.node_positions is None or len(state.node_positions) != num_nodes:
                        angles = np.linspace(0, 2 * np.pi, num_nodes, endpoint=False)
                        state.node_positions = np.column_stack([np.cos(angles), np.sin(angles)])
                    state.pred_version += 1

                if step % 5 == 0:
                    state.latest_graph_png = make_graph_snapshot_png(
                        state.true_adj,
                        state.pred_adj,
                        current_node=src.item(),
                        predicted_node=pred_idx.item(),
                        actual_next_node=dst.item(),
                        was_correct=was_correct,
                        step=step,
                        loss=loss.item(),
                        acc=acc,
                        correct=state.correct,
                        total=state.total,
                    )
                    state.latest_dynamics_png = make_dynamics_png(
                        state.loss_history, state.acc_history
                    )
    finally:
        with state.lock:
            state.is_running = False
            state.should_stop = False
            # Clear thread reference on exit; start_run will re-assign
            state.thread = None


INDEX_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Spectral Band Graph Viewer</title>
  <style>
    :root{
      --bg:#0b0f14;
      --fg:#c9d1d9;
      --muted:#7d8590;
      --accent:#00ffc6;
      --accent2:#7aa2f7;
      --glass:rgba(255,255,255,0.06);
      --border:rgba(255,255,255,0.08);
    }
    *{box-sizing:border-box}
    html,body{height:100%}
    body{margin:0;background:var(--bg);color:var(--fg);font-family:ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,"Liberation Mono","Courier New",monospace;font-size:14px;}
    #app{display:flex;flex-direction:column;min-height:100vh;}
    .topbar{display:flex;align-items:center;justify-content:space-between;padding:10px 16px;border-bottom:1px solid var(--border);background:linear-gradient(180deg, rgba(255,255,255,0.04), rgba(255,255,255,0));backdrop-filter:saturate(120%) blur(6px);position:sticky;top:0;z-index:10;}
    .brand{font-weight:700;letter-spacing:.5px;color:var(--accent);}
    .muted{color:var(--muted);font-size:12px;}
    .layout{display:grid;grid-template-columns:1fr 1fr;grid-template-rows:auto auto 1fr;gap:16px;padding:16px;}
    @media(max-width:1100px){.layout{grid-template-columns:1fr;grid-template-rows:auto;}}
    .panel{border:1px solid var(--border);background:linear-gradient(180deg, rgba(255,255,255,0.04), rgba(255,255,255,0.01));border-radius:10px;padding:12px;position:relative;overflow:hidden;}
    .panel h2{margin:0 0 8px 0;color:var(--accent2);font-size:14px;}
    .glass{box-shadow:0 8px 30px rgba(0,0,0,0.35);backdrop-filter:blur(10px);}
    .controls{display:flex;flex-wrap:wrap;gap:12px;margin-bottom:12px;align-items:flex-end;}
    .control-group{display:flex;flex-direction:column;min-width:100px;}
    label{font-size:12px;color:var(--muted);margin-bottom:4px;}
    input,select{padding:6px 8px;border-radius:6px;border:1px dashed var(--border);background:#0a0f14;color:var(--fg);font-family:inherit;font-size:13px;}
    input:focus,select:focus{outline:none;border-color:var(--accent);box-shadow:0 0 0 3px rgba(0,255,198,0.12);}
    .btn{background:transparent;color:var(--accent);border:1px solid var(--accent);padding:6px 10px;border-radius:6px;cursor:pointer;font-family:inherit;font-weight:500;transition:background 0.15s, color 0.15s;}
    .btn:hover{background:var(--accent);color:#001f1a;}
    .btn-stop{color:var(--muted);border-color:var(--border);}
    .btn-stop:hover{background:rgba(255,255,255,0.08);color:var(--fg);border-color:var(--muted);}
    .chip{background:transparent;color:var(--fg);border:1px solid var(--border);padding:6px 10px;border-radius:999px;cursor:pointer;font-family:inherit;font-size:12px;}
    .chip:hover{border-color:var(--accent);color:var(--accent);}
    .chip.active{border-color:var(--accent);color:var(--accent);background:rgba(0,255,198,0.08);}
    .status{padding:8px 10px;border-radius:6px;border:1px solid var(--border);background:rgba(255,255,255,0.02);font-size:12px;}
    .status-running{border-left:3px solid var(--accent);}
    .status-stopped{border-left:3px solid var(--border);}
    .grid-2{display:grid;grid-template-columns:1fr 1fr;gap:16px;}
    @media(max-width:900px){.grid-2{grid-template-columns:1fr;}}
    canvas,img{max-width:100%;border-radius:6px;display:block;}
    .section-title{color:var(--muted);font-size:11px;margin-bottom:6px;text-transform:uppercase;letter-spacing:0.5px;}
    .helper{font-size:11px;color:var(--muted);margin-top:6px;}
    .actions{display:flex;gap:8px;flex-wrap:wrap;}
    .footer-fade{position:fixed;left:0;right:0;bottom:0;height:12px;max-height:12px;background:linear-gradient(180deg, rgba(11,15,20,0), rgba(11,15,20,0.85));pointer-events:none;z-index:20;}
  </style>
</head>
<body>
  <div id="app">
    <header class="topbar">
      <div class="brand">Spectral Band ▓▒░ Graph Live</div>
      <div class="muted">belief in motion</div>
    </header>

    <main class="layout">
      <section class="panel glass" style="grid-column:1 / -1;">
        <h2>Controls</h2>
        <div class="controls">
          <div class="control-group">
            <label>Nodes</label>
            <input type="number" id="numNodes" value="16" min="4" max="64" style="width:70px;">
          </div>
          <div class="control-group">
            <label>Steps</label>
            <input type="number" id="steps" value="500" min="10" max="5000" style="width:80px;">
          </div>
          <div class="control-group">
            <label>Edge Prob</label>
            <input type="number" id="edgeProb" value="0.2" min="0.05" max="0.9" step="0.05" style="width:70px;">
          </div>
          <div class="control-group">
            <label>Graph Type</label>
            <select id="graphType" style="width:100px;">
              <option value="random" selected>Random</option>
              <option value="cycle">Cycle</option>
              <option value="tree">Tree</option>
              <option value="grid">Grid</option>
              <option value="hub">Hub</option>
            </select>
          </div>
          <div class="control-group">
            <label>Hidden</label>
            <input type="number" id="hiddenDim" value="32" min="8" max="256" step="8" style="width:70px;">
          </div>
          <div class="control-group">
            <label>Heads</label>
            <input type="number" id="nHeads" value="32" min="1" max="32" style="width:60px;">
          </div>
          <div class="control-group">
            <label>Layers</label>
            <input type="number" id="nLayers" value="16" min="1" max="16" style="width:60px;">
          </div>
          <div class="actions" style="margin-left:auto;">
            <button class="btn" onclick="startTraining()">Start</button>
            <button class="btn btn-stop" onclick="stopTraining()">Stop</button>
          </div>
        </div>
        <div id="status" class="status status-stopped">Ready to start</div>
      </section>

      <section class="panel">
        <h2>Graph View</h2>
        <canvas id="graphCanvas" width="500" height="400" style="background:linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0)); border:1px dashed var(--border); width:100%;"></canvas>
        <div class="helper">Red = current · Green = target · Blue = predicted (wrong)</div>
      </section>
      <section class="panel">
        <h2>Training Dynamics</h2>
        <img id="dynamics" src="" alt="Waiting for training..." style="max-width:100%; border:1px dashed var(--border); min-height:200px; background:linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0));">
        <div class="helper">Loss and accuracy over training steps</div>
      </section>

      <section class="panel" style="grid-column:1 / -1;">
        <h2>Adjacency Maps</h2>
        <div class="grid-2">
          <div>
            <div class="section-title">True adjacency</div>
            <canvas id="adjTrue" width="240" height="240" style="background:#0a0f14; border:1px dashed var(--border);"></canvas>
          </div>
          <div>
            <div class="section-title">Belief (predicted)</div>
            <canvas id="adjBelief" width="240" height="240" style="background:#0a0f14; border:1px dashed var(--border);"></canvas>
          </div>
        </div>
        <div class="helper">Left: ground truth graph · Right: model's learned belief (updates live)</div>
      </section>
    </main>
    <div class="footer-fade" aria-hidden="true"></div>
  </div>

  <script>
        let refreshInterval = null;
        let graphInterval = null;
        // cache last drawn adjacency per canvas to show subtle flash on change
        const lastAdjCache = {};
        
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
                num_nodes: parseInt(document.getElementById('numNodes').value),
                steps: parseInt(document.getElementById('steps').value),
                edge_prob: parseFloat(document.getElementById('edgeProb').value),
                graph_type: document.getElementById('graphType').value,
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
                refreshGraph(); // kick off immediately for live belief map
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
                    const pct = data.total_steps > 0 ? ((data.step / data.total_steps) * 100).toFixed(0) : 0;
                    statusEl.textContent = `Running · ${data.step}/${data.total_steps} (${pct}%) · Loss ${data.last_loss.toFixed(4)} · Acc ${data.acc.toFixed(3)}`;
                } else {
                    statusEl.className = 'status status-stopped';
                    statusEl.textContent = data.step > 0 ? `Finished · ${data.step} steps` : 'Ready';
                }
                
                if (data.is_running || data.step > 0) {
                    const ts = Date.now();
                    document.getElementById('dynamics').src = '/dynamics.png?' + ts;
                }
            }).catch(e => {
                document.getElementById('status').textContent = 'Disconnected';
            });
        }

        async function refreshGraph() {
            try {
                const res = await fetch('/graph_data');
                const data = await res.json();
                if (!data.ready) return;
                drawGraph(data);
                if (data.adj_true && data.adj_belief) {
                    drawAdjacency(document.getElementById('adjTrue'), data.adj_true, data.version);
                    drawAdjacency(document.getElementById('adjBelief'), data.adj_belief, data.version);
                }
            } catch (e) {
                console.error('graph_data error', e);
            }
        }

        function drawGraph(data) {
            const canvas = document.getElementById('graphCanvas');
            const ctx = canvas.getContext('2d');
            const W = canvas.width;
            const H = canvas.height;
            ctx.clearRect(0, 0, W, H);
            // Subtle gradient background
            const grad = ctx.createLinearGradient(0, 0, 0, H);
            grad.addColorStop(0, '#0d1218');
            grad.addColorStop(1, '#0a0e13');
            ctx.fillStyle = grad;
            ctx.fillRect(0, 0, W, H);

            const pos = data.positions;
            const n = data.num_nodes;
            const scale = 140;
            const cx = W / 2;
            const cy = H / 2;

            // edges true (subtle)
            ctx.strokeStyle = 'rgba(125,133,144,0.15)';
            ctx.lineWidth = 1;
            for (const [i, j] of data.edges_true) {
                const [x1, y1] = pos[i];
                const [x2, y2] = pos[j];
                ctx.beginPath();
                ctx.moveTo(cx + x1 * scale, cy + y1 * scale);
                ctx.lineTo(cx + x2 * scale, cy + y2 * scale);
                ctx.stroke();
            }

            // belief edges (accent colored, weighted)
            for (const e of data.edges_belief) {
                const [x1, y1] = pos[e.from];
                const [x2, y2] = pos[e.to];
                // Use accent color with alpha based on probability
                const alpha = Math.min(0.85, 0.3 + e.p * 0.6);
                ctx.strokeStyle = `rgba(0,255,198,${alpha})`;
                ctx.lineWidth = 1 + 3 * e.p;
                ctx.beginPath();
                ctx.moveTo(cx + x1 * scale, cy + y1 * scale);
                ctx.lineTo(cx + x2 * scale, cy + y2 * scale);
                ctx.stroke();
            }

            // live guess: current -> predicted edge (strong yellow highlight)
            if (data.positions && Number.isInteger(data.current) && Number.isInteger(data.predicted)) {
                const [x1, y1] = pos[data.current];
                const [x2, y2] = pos[data.predicted];
                ctx.strokeStyle = 'rgba(255, 215, 71, 0.9)'; // strong yellow
                ctx.lineWidth = 2;
                ctx.beginPath();
                ctx.moveTo(cx + x1 * scale, cy + y1 * scale);
                ctx.lineTo(cx + x2 * scale, cy + y2 * scale);
                ctx.stroke();
            }

            // nodes
            for (let i = 0; i < n; i++) {
                const [x, y] = pos[i];
                let fillColor = '#3d4654';  // muted default
                let strokeColor = 'rgba(255,255,255,0.1)';
                if (i === data.current) {
                    fillColor = '#7aa2f7';  // accent2 blue for current
                    strokeColor = 'rgba(122,162,247,0.4)';
                } else if (i === data.target) {
                    fillColor = '#00ffc6';  // accent green for target
                    strokeColor = 'rgba(0,255,198,0.4)';
                } else if (i === data.predicted && !data.was_correct) {
                    fillColor = '#c9d1d9';  // light gray for wrong prediction
                    strokeColor = 'rgba(201,209,217,0.3)';
                }
                ctx.beginPath();
                ctx.fillStyle = fillColor;
                ctx.strokeStyle = strokeColor;
                ctx.lineWidth = 2;
                ctx.arc(cx + x * scale, cy + y * scale, 11, 0, Math.PI * 2);
                ctx.fill();
                ctx.stroke();
                // Node label
                ctx.fillStyle = i === data.current || i === data.target ? '#0b0f14' : '#c9d1d9';
                ctx.font = '10px ui-monospace, monospace';
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                ctx.fillText(i.toString(), cx + x * scale, cy + y * scale);
            }

            // header text
            ctx.fillStyle = '#7d8590';
            ctx.font = '11px ui-monospace, monospace';
            ctx.textAlign = 'left';
            ctx.fillText(`Step ${data.step}  Loss ${data.loss.toFixed(4)}  Acc ${(data.acc_running*100).toFixed(1)}%`, 10, 16);
        }

        function drawAdjacency(canvas, mat, version=0) {
            if (!canvas || !mat) return;
            const ctx = canvas.getContext('2d');
            const n = mat.length;
            const w = canvas.width;
            const h = canvas.height;
            ctx.clearRect(0, 0, w, h);
            ctx.fillStyle = '#0a0f14';
            ctx.fillRect(0, 0, w, h);
            
            // Contrast control: zero floor, percentile cap, mild gamma
            const values = [];
            for (let i = 0; i < n; i++) {
                for (let j = 0; j < n; j++) values.push(mat[i][j]);
            }
            values.sort((a,b)=>a-b);
            const pctIdx = Math.max(0, Math.min(values.length-1, Math.floor(values.length*0.95)));
            const maxv = Math.max(1e-6, values[pctIdx]); // 95th percentile as cap
            const gamma = 0.6; // <1 brightens peaks relative to floor
            
            // previous normalized values for flash detection
            const cacheKey = canvas.id || 'adj';
            let prevNorm = lastAdjCache[cacheKey];
            if (!prevNorm || prevNorm.length !== n || prevNorm[0].length !== n) {
                prevNorm = Array.from({length: n}, () => Array(n).fill(0));
            }

            const cellW = w / n;
            const cellH = h / n;
            for (let i = 0; i < n; i++) {
                for (let j = 0; j < n; j++) {
                    let v = mat[i][j] / maxv; // normalize to cap
                    v = Math.max(0, Math.min(1, v));
                    v = Math.pow(v, gamma); // gamma adjust to push low values darker
                    // Color: dark blue -> bright cyan
                    const r = Math.floor(10 + 40 * v);
                    const g = Math.floor(20 + 235 * v);
                    const b = Math.floor(40 + 158 * v);
                    ctx.fillStyle = `rgb(${r},${g},${b})`;
                    ctx.fillRect(j*cellW, i*cellH, cellW, cellH);

                    // Flash overlay if value changed noticeably (magenta, clearly visible)
                    const delta = Math.abs(v - prevNorm[i][j]);
                    if (delta > 0.02) {
                        const flashAlpha = Math.min(0.9, 0.15 + delta * 1.2);
                        ctx.fillStyle = `rgba(220,0,200,${flashAlpha})`;
                        ctx.fillRect(j*cellW, i*cellH, cellW, cellH);
                    }

                    // store normalized value
                    prevNorm[i][j] = v;
                }
            }
            // update cache
            lastAdjCache[cacheKey] = prevNorm;
            // Grid lines (subtle)
            if (n <= 32) {
                ctx.strokeStyle = 'rgba(0,0,0,0.3)';
                ctx.lineWidth = 0.5;
                for (let i = 0; i <= n; i++) {
                    ctx.beginPath();
                    ctx.moveTo(i * cellW, 0);
                    ctx.lineTo(i * cellW, h);
                    ctx.stroke();
                    ctx.beginPath();
                    ctx.moveTo(0, i * cellH);
                    ctx.lineTo(w, i * cellH);
                    ctx.stroke();
                }
            }
            // Version label
            ctx.fillStyle = '#7d8590';
            ctx.font = '10px ui-monospace, monospace';
            ctx.textAlign = 'right';
            ctx.fillText(`v${version}`, w-4, h-4);
        }
        
        function startRefreshing() {
            if (refreshInterval) clearInterval(refreshInterval);
            refreshInterval = setInterval(refreshStatus, 500);
            if (graphInterval) clearInterval(graphInterval);
            graphInterval = setInterval(refreshGraph, 100); // faster graph updates for smoother belief map
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
            "acc": state.acc_history[-1] if state.acc_history else 0.0,
            "correct": state.correct,
            "total": state.total,
        })


@app.get("/graph_data")
def get_graph_data():
    with state.lock:
        if state.true_adj is None or state.pred_adj is None:
            return JSONResponse({"ready": False})
        num_nodes = state.true_adj.shape[0]
        # Positions
        if state.node_positions is None or len(state.node_positions) != num_nodes:
            angles = np.linspace(0, 2 * np.pi, num_nodes, endpoint=False)
            positions = np.column_stack([np.cos(angles), np.sin(angles)])
        else:
            positions = state.node_positions
        # True edges list
        edges_true = []
        ta = state.true_adj
        for i in range(num_nodes):
            for j in range(num_nodes):
                if ta[i, j] > 0:
                    edges_true.append([int(i), int(j)])
        # Belief edges (top-2 per node, prob>0.1)
        edges_belief = []
        pa = state.pred_adj
        for i in range(num_nodes):
            row = pa[i]
            top_idx = np.argsort(row)[-2:]
            for j in top_idx:
                if row[j] > 0.1:
                    edges_belief.append({
                        "from": int(i),
                        "to": int(j),
                        "p": float(row[j]),
                    })
        return JSONResponse({
            "ready": True,
            "num_nodes": num_nodes,
            "positions": positions.tolist(),
            "edges_true": edges_true,
            "edges_belief": edges_belief,
            "adj_true": state.true_adj.tolist(),
            "adj_belief": state.pred_adj.tolist(),
            "current": int(state.current_node),
            "target": int(state.target_node),
            "predicted": int(state.predicted_node),
            "was_correct": bool(state.was_correct),
            "loss": float(state.last_loss),
            "acc_last": state.acc_history[-1] if state.acc_history else 0.0,
            "acc_running": (state.correct / state.total) if state.total > 0 else 0.0,
            "correct": state.correct,
            "total": state.total,
            "step": state.step,
            "version": state.pred_version,
        })


@app.get("/graph.png")
def get_graph():
    with state.lock:
        png_bytes = state.latest_graph_png
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
    
    num_nodes = payload.get("num_nodes", 16)
    steps = payload.get("steps", 500)
    edge_prob = float(payload.get("edge_prob", 0.2))
    graph_type = payload.get("graph_type", "random")
    hidden_dim = int(payload.get("hidden_dim", 32))
    n_heads = int(payload.get("n_heads", 4))
    n_layers = int(payload.get("n_layers", 2))
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
            "num_nodes": num_nodes,
            "steps": steps,
            "device_str": device_str,
            "hidden_dim": hidden_dim,
            "n_heads": n_heads,
            "n_layers": n_layers,
            "edge_prob": edge_prob,
            "graph_type": graph_type,
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
    return JSONResponse({"status": "stopping"})


# Entry point moved to run.py for unified CLI interface.
# Use: python run.py --mode web

