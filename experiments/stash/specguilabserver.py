"""
What/Where Spectral Attention Lab

A beautiful interactive demonstration of the what/where separation principle
in spectral attention. Features real-time JavaScript visualizations.

Run with: uvicorn spectral_gui_lab.server:app --reload --port 8765
"""

import io
import os
import json
import base64
import numpy as np
from typing import Optional, Dict, Any
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel


app = FastAPI(title="What/Where Spectral Lab")


# ============================================================================
# Core Spectral Functions
# ============================================================================

def create_pattern(
    H: int, W: int, 
    cx: float, cy: float, 
    pattern_type: str = 'ring',
    size: float = 0.12
) -> np.ndarray:
    """Create a pattern at position (cx, cy)."""
    x = np.linspace(0, 1, W)
    y = np.linspace(0, 1, H)
    X, Y = np.meshgrid(x, y)
    
    dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
    
    if pattern_type == 'ring':
        pattern = np.exp(-((dist - size)**2) / (2 * (size * 0.25)**2))
    elif pattern_type == 'blob':
        pattern = np.exp(-dist**2 / (2 * size**2))
    elif pattern_type == 'cross':
        dx = np.abs(X - cx)
        dy = np.abs(Y - cy)
        h_bar = (dy < size * 0.25) & (dx < size)
        v_bar = (dx < size * 0.25) & (dy < size)
        pattern = (h_bar | v_bar).astype(float)
        pattern = pattern * np.exp(-dist**2 / (2 * (size * 1.5)**2))
    elif pattern_type == 'star':
        angles = np.arctan2(Y - cy, X - cx)
        star = np.cos(5 * angles) * 0.5 + 0.5
        radial = np.exp(-dist**2 / (2 * size**2))
        pattern = star * radial
    elif pattern_type == 'spiral':
        angles = np.arctan2(Y - cy, X - cx)
        spiral = np.sin(angles * 3 + dist * 20)
        radial = np.exp(-dist**2 / (2 * size**2))
        pattern = (spiral * 0.5 + 0.5) * radial
    else:
        pattern = np.exp(-dist**2 / (2 * size**2))
    
    return pattern.astype(np.float32)


def compute_fft_components(frame: np.ndarray) -> Dict[str, np.ndarray]:
    """Compute FFT magnitude and phase."""
    fft = np.fft.fft2(frame)
    fft_shifted = np.fft.fftshift(fft)
    
    magnitude = np.abs(fft_shifted)
    phase = np.angle(fft_shifted)
    
    # Log magnitude for visualization
    magnitude_log = np.log1p(magnitude)
    magnitude_log = (magnitude_log - magnitude_log.min()) / (magnitude_log.max() - magnitude_log.min() + 1e-8)
    
    # Normalize phase to 0-1
    phase_norm = (phase + np.pi) / (2 * np.pi)
    
    return {
        'magnitude': magnitude_log.astype(np.float32),
        'phase': phase_norm.astype(np.float32),
        'magnitude_raw': np.abs(fft).astype(np.float32),
        'phase_raw': np.angle(fft).astype(np.float32),
    }


def compute_spectral_bands(
    frame: np.ndarray, 
    freq_cutoff: float = 0.2
) -> Dict[str, np.ndarray]:
    """Decompose into low and high frequency bands."""
    H, W = frame.shape
    
    fft = np.fft.fft2(frame)
    fft_shifted = np.fft.fftshift(fft)
    
    cy, cx = H // 2, W // 2
    y_coords = np.arange(H) - cy
    x_coords = np.arange(W) - cx
    X, Y = np.meshgrid(x_coords, y_coords)
    
    max_freq = min(cy, cx)
    freq_dist = np.sqrt(X**2 + Y**2) / max_freq
    
    # Smooth masks
    low_mask = 1.0 / (1.0 + np.exp(30 * (freq_dist - freq_cutoff)))
    high_mask = 1.0 - low_mask
    
    low_fft = fft_shifted * low_mask
    high_fft = fft_shifted * high_mask
    
    low_spatial = np.fft.ifft2(np.fft.ifftshift(low_fft)).real
    high_spatial = np.fft.ifft2(np.fft.ifftshift(high_fft)).real
    
    # Normalize
    def normalize(x):
        return ((x - x.min()) / (x.max() - x.min() + 1e-8)).astype(np.float32)
    
    return {
        'low_freq': normalize(low_spatial),
        'high_freq': normalize(high_spatial),
        'low_mask': low_mask.astype(np.float32),
        'high_mask': high_mask.astype(np.float32),
    }


def compute_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute normalized correlation."""
    a_flat = a.flatten()
    b_flat = b.flatten()
    
    a_norm = (a_flat - a_flat.mean()) / (a_flat.std() + 1e-8)
    b_norm = (b_flat - b_flat.mean()) / (b_flat.std() + 1e-8)
    
    return float(np.mean(a_norm * b_norm))


# ============================================================================
# API Models
# ============================================================================

class PatternRequest(BaseModel):
    pattern_type: str = 'ring'
    cx: float = 0.5
    cy: float = 0.5
    size: float = 0.12
    grid_size: int = 64
    freq_cutoff: float = 0.2


class ComparisonRequest(BaseModel):
    pattern_type: str = 'ring'
    pos1_x: float = 0.25
    pos1_y: float = 0.5
    pos2_x: float = 0.75
    pos2_y: float = 0.5
    size: float = 0.12
    grid_size: int = 64
    freq_cutoff: float = 0.2


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/", response_class=HTMLResponse)
async def get_ui():
    """Serve the main UI."""
    return get_html()


@app.post("/api/pattern")
async def generate_pattern(req: PatternRequest):
    """Generate a pattern and its spectral decomposition."""
    
    # Create pattern
    pattern = create_pattern(
        req.grid_size, req.grid_size,
        req.cx, req.cy,
        req.pattern_type,
        req.size
    )
    
    # Compute FFT
    fft_data = compute_fft_components(pattern)
    
    # Compute spectral bands
    bands = compute_spectral_bands(pattern, req.freq_cutoff)
    
    # Convert to lists for JSON
    return {
        'pattern': pattern.tolist(),
        'magnitude': fft_data['magnitude'].tolist(),
        'phase': fft_data['phase'].tolist(),
        'low_freq': bands['low_freq'].tolist(),
        'high_freq': bands['high_freq'].tolist(),
        'grid_size': req.grid_size,
    }


@app.post("/api/compare")
async def compare_positions(req: ComparisonRequest):
    """Compare patterns at two positions - the what/where test."""
    
    H = W = req.grid_size
    
    # Create patterns at two positions
    pattern1 = create_pattern(H, W, req.pos1_x, req.pos1_y, req.pattern_type, req.size)
    pattern2 = create_pattern(H, W, req.pos2_x, req.pos2_y, req.pattern_type, req.size)
    
    # Compute FFT for both
    fft1 = compute_fft_components(pattern1)
    fft2 = compute_fft_components(pattern2)
    
    # Compute spectral bands
    bands1 = compute_spectral_bands(pattern1, req.freq_cutoff)
    bands2 = compute_spectral_bands(pattern2, req.freq_cutoff)
    
    # Compute similarities
    mag_similarity = compute_similarity(fft1['magnitude_raw'], fft2['magnitude_raw'])
    phase_similarity = compute_similarity(fft1['phase_raw'], fft2['phase_raw'])
    low_similarity = compute_similarity(bands1['low_freq'], bands2['low_freq'])
    high_similarity = compute_similarity(bands1['high_freq'], bands2['high_freq'])
    spatial_similarity = compute_similarity(pattern1, pattern2)
    
    return {
        'pattern1': pattern1.tolist(),
        'pattern2': pattern2.tolist(),
        'magnitude1': fft1['magnitude'].tolist(),
        'magnitude2': fft2['magnitude'].tolist(),
        'phase1': fft1['phase'].tolist(),
        'phase2': fft2['phase'].tolist(),
        'low_freq1': bands1['low_freq'].tolist(),
        'low_freq2': bands2['low_freq'].tolist(),
        'high_freq1': bands1['high_freq'].tolist(),
        'high_freq2': bands2['high_freq'].tolist(),
        'similarities': {
            'magnitude': mag_similarity,
            'phase': phase_similarity,
            'low_freq': low_similarity,
            'high_freq': high_similarity,
            'spatial': spatial_similarity,
        },
        'grid_size': req.grid_size,
    }


@app.get("/api/demo_positions")
async def get_demo_positions():
    """Get multiple pattern positions for animation."""
    positions = []
    for i in range(36):
        angle = i * 10 * np.pi / 180
        cx = 0.5 + 0.3 * np.cos(angle)
        cy = 0.5 + 0.3 * np.sin(angle)
        positions.append({'cx': float(cx), 'cy': float(cy)})
    return {'positions': positions}


# ============================================================================
# HTML UI
# ============================================================================

def get_html():
    return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>What/Where Spectral Lab</title>
  <style>
    :root {
      --bg: #0a0d12;
      --surface: rgba(15, 20, 30, 0.85);
      --border: rgba(100, 120, 150, 0.2);
      --fg: #e4e8f0;
      --muted: #6b7894;
      --accent: #00ffaa;
      --accent2: #7aa2f7;
      --what: #22c55e;
      --where: #ef4444;
      --glow: rgba(0, 255, 170, 0.15);
    }
    
    * { margin: 0; padding: 0; box-sizing: border-box; }
    
    body {
      font-family: 'SF Mono', 'Fira Code', 'JetBrains Mono', monospace;
      background: var(--bg);
      color: var(--fg);
      min-height: 100vh;
      overflow-x: hidden;
    }
    
    /* Animated background */
    body::before {
      content: '';
      position: fixed;
      top: 0; left: 0; right: 0; bottom: 0;
      background: 
        radial-gradient(ellipse at 20% 20%, rgba(0, 255, 170, 0.05) 0%, transparent 50%),
        radial-gradient(ellipse at 80% 80%, rgba(122, 162, 247, 0.05) 0%, transparent 50%);
      pointer-events: none;
      z-index: -1;
    }
    
    .topbar {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 16px 24px;
      background: rgba(10, 13, 18, 0.9);
      border-bottom: 1px solid var(--border);
      backdrop-filter: blur(12px);
      position: sticky;
      top: 0;
      z-index: 100;
    }
    
    .brand {
      font-size: 18px;
      font-weight: 600;
      background: linear-gradient(135deg, var(--accent), var(--accent2));
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      background-clip: text;
    }
    
    .tagline {
      font-size: 12px;
      color: var(--muted);
      letter-spacing: 1px;
    }
    
    .container {
      max-width: 1600px;
      margin: 0 auto;
      padding: 24px;
    }
    
    .hero {
      text-align: center;
      padding: 40px 20px;
      margin-bottom: 24px;
    }
    
    .hero h1 {
      font-size: 36px;
      font-weight: 300;
      margin-bottom: 12px;
      background: linear-gradient(135deg, var(--what), var(--accent), var(--accent2), var(--where));
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      background-clip: text;
    }
    
    .hero p {
      font-size: 14px;
      color: var(--muted);
      max-width: 600px;
      margin: 0 auto;
      line-height: 1.6;
    }
    
    .insight-box {
      background: rgba(0, 255, 170, 0.08);
      border: 1px solid rgba(0, 255, 170, 0.3);
      border-radius: 12px;
      padding: 16px 24px;
      margin: 20px auto;
      max-width: 700px;
      font-size: 13px;
    }
    
    .insight-box code {
      color: var(--accent);
      background: rgba(0, 0, 0, 0.3);
      padding: 2px 6px;
      border-radius: 4px;
    }
    
    .layout {
      display: grid;
      grid-template-columns: 280px 1fr;
      gap: 24px;
    }
    
    @media (max-width: 1000px) {
      .layout { grid-template-columns: 1fr; }
    }
    
    .panel {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 20px;
      backdrop-filter: blur(16px);
    }
    
    .panel h2 {
      font-size: 14px;
      font-weight: 500;
      color: var(--accent2);
      text-transform: uppercase;
      letter-spacing: 1px;
      margin-bottom: 16px;
      padding-bottom: 8px;
      border-bottom: 1px solid var(--border);
    }
    
    .control {
      margin-bottom: 16px;
    }
    
    .control label {
      display: block;
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.5px;
      color: var(--muted);
      margin-bottom: 6px;
    }
    
    .control select,
    .control input[type="number"] {
      width: 100%;
      padding: 10px 12px;
      border-radius: 8px;
      border: 1px solid var(--border);
      background: rgba(0, 0, 0, 0.4);
      color: var(--fg);
      font-size: 13px;
      font-family: inherit;
    }
    
    .control select:focus,
    .control input:focus {
      outline: none;
      border-color: var(--accent);
      box-shadow: 0 0 0 3px var(--glow);
    }
    
    .slider-container {
      display: flex;
      align-items: center;
      gap: 12px;
    }
    
    .slider-container input[type="range"] {
      flex: 1;
      -webkit-appearance: none;
      height: 6px;
      border-radius: 3px;
      background: var(--border);
    }
    
    .slider-container input[type="range"]::-webkit-slider-thumb {
      -webkit-appearance: none;
      width: 18px;
      height: 18px;
      border-radius: 50%;
      background: var(--accent);
      cursor: pointer;
      box-shadow: 0 0 10px var(--glow);
    }
    
    .slider-value {
      font-size: 13px;
      color: var(--accent);
      min-width: 50px;
      text-align: right;
    }
    
    .btn {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 8px;
      padding: 12px 20px;
      border-radius: 8px;
      border: 1px solid var(--border);
      background: transparent;
      color: var(--fg);
      font-size: 13px;
      font-family: inherit;
      cursor: pointer;
      transition: all 0.2s;
    }
    
    .btn:hover {
      border-color: var(--accent);
      color: var(--accent);
    }
    
    .btn.primary {
      background: linear-gradient(135deg, var(--accent), #00cc88);
      border: none;
      color: #001a10;
      font-weight: 600;
    }
    
    .btn.primary:hover {
      filter: brightness(1.1);
      transform: translateY(-1px);
    }
    
    .btn-group {
      display: flex;
      gap: 8px;
      margin-top: 20px;
    }
    
    .btn-group .btn {
      flex: 1;
    }
    
    /* Canvas Grid */
    .canvas-grid {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 16px;
    }
    
    @media (max-width: 1200px) {
      .canvas-grid { grid-template-columns: repeat(2, 1fr); }
    }
    
    .canvas-card {
      background: rgba(0, 0, 0, 0.3);
      border: 1px solid var(--border);
      border-radius: 12px;
      overflow: hidden;
    }
    
    .canvas-card.what { border-color: rgba(34, 197, 94, 0.4); }
    .canvas-card.where { border-color: rgba(239, 68, 68, 0.4); }
    
    .canvas-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 12px 16px;
      background: rgba(0, 0, 0, 0.4);
      border-bottom: 1px solid var(--border);
    }
    
    .canvas-title {
      font-size: 12px;
      font-weight: 500;
      color: var(--fg);
    }
    
    .canvas-badge {
      font-size: 10px;
      padding: 3px 8px;
      border-radius: 999px;
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }
    
    .canvas-badge.what {
      background: rgba(34, 197, 94, 0.2);
      color: var(--what);
    }
    
    .canvas-badge.where {
      background: rgba(239, 68, 68, 0.2);
      color: var(--where);
    }
    
    .canvas-wrapper {
      padding: 16px;
      display: flex;
      justify-content: center;
    }
    
    .canvas-wrapper canvas {
      border-radius: 8px;
      image-rendering: pixelated;
      background: #000;
    }
    
    /* Comparison Section */
    .comparison-section {
      margin-top: 24px;
    }
    
    .comparison-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 16px;
    }
    
    .comparison-title {
      font-size: 16px;
      font-weight: 500;
    }
    
    .comparison-grid {
      display: grid;
      grid-template-columns: 1fr auto 1fr;
      gap: 24px;
      align-items: center;
    }
    
    .position-panel {
      background: rgba(0, 0, 0, 0.3);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 16px;
    }
    
    .position-panel h3 {
      font-size: 13px;
      color: var(--muted);
      margin-bottom: 12px;
    }
    
    .position-canvas {
      width: 100%;
      aspect-ratio: 1;
      border-radius: 8px;
      cursor: crosshair;
      background: #000;
    }
    
    .similarity-panel {
      min-width: 200px;
    }
    
    .similarity-bar {
      margin-bottom: 16px;
    }
    
    .similarity-label {
      display: flex;
      justify-content: space-between;
      font-size: 11px;
      margin-bottom: 6px;
    }
    
    .similarity-label span:first-child {
      color: var(--muted);
      text-transform: uppercase;
    }
    
    .similarity-label span:last-child {
      font-weight: 600;
    }
    
    .similarity-track {
      height: 8px;
      background: rgba(255, 255, 255, 0.1);
      border-radius: 4px;
      overflow: hidden;
    }
    
    .similarity-fill {
      height: 100%;
      border-radius: 4px;
      transition: width 0.3s ease;
    }
    
    .similarity-fill.what { background: var(--what); }
    .similarity-fill.where { background: var(--where); }
    .similarity-fill.neutral { background: var(--accent2); }
    
    /* Results */
    .result-box {
      background: rgba(0, 0, 0, 0.4);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 20px;
      margin-top: 24px;
      text-align: center;
    }
    
    .result-title {
      font-size: 14px;
      color: var(--muted);
      margin-bottom: 12px;
    }
    
    .result-value {
      font-size: 48px;
      font-weight: 300;
      background: linear-gradient(135deg, var(--what), var(--accent));
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      background-clip: text;
    }
    
    .result-unit {
      font-size: 14px;
      color: var(--muted);
      margin-top: 4px;
    }
    
    /* Animation */
    @keyframes pulse {
      0%, 100% { opacity: 1; }
      50% { opacity: 0.5; }
    }
    
    .animating .canvas-badge {
      animation: pulse 1s infinite;
    }
    
    /* Footer */
    .footer {
      text-align: center;
      padding: 40px 20px;
      color: var(--muted);
      font-size: 12px;
    }
    
    .footer a {
      color: var(--accent2);
      text-decoration: none;
    }
  </style>
</head>
<body>
  <div class="topbar">
    <div class="brand">What/Where Spectral Lab</div>
    <div class="tagline">magnitude · phase · attention</div>
  </div>
  
  <div class="container">
    <div class="hero">
      <h1>WHAT vs WHERE</h1>
      <p>
        When a pattern shifts in space, its FFT magnitude stays identical (WHAT) 
        while its phase changes completely (WHERE). This is the mathematical foundation 
        of position-invariant object recognition.
      </p>
      <div class="insight-box">
        <strong>The Mathematical Truth:</strong> 
        <code>f(x - Δ) → F(ω) · exp(-iωΔ)</code><br>
        Magnitude <code>|F(ω)|</code> unchanged → <strong style="color: var(--what)">WHAT</strong> &nbsp;|&nbsp;
        Phase <code>arg(F(ω))</code> changes → <strong style="color: var(--where)">WHERE</strong>
      </div>
    </div>
    
    <div class="layout">
      <!-- Controls Panel -->
      <div class="panel">
        <h2>Controls</h2>
        
        <div class="control">
          <label>Pattern Type</label>
          <select id="pattern-type">
            <option value="ring">Ring</option>
            <option value="blob">Blob</option>
            <option value="cross">Cross</option>
            <option value="star">Star</option>
            <option value="spiral">Spiral</option>
          </select>
        </div>
        
        <div class="control">
          <label>Grid Size</label>
          <select id="grid-size">
            <option value="32">32×32</option>
            <option value="64" selected>64×64</option>
            <option value="128">128×128</option>
          </select>
        </div>
        
        <div class="control">
          <label>Pattern Size</label>
          <div class="slider-container">
            <input type="range" id="pattern-size" min="0.05" max="0.25" step="0.01" value="0.12">
            <span class="slider-value" id="size-value">0.12</span>
          </div>
        </div>
        
        <div class="control">
          <label>Frequency Cutoff</label>
          <div class="slider-container">
            <input type="range" id="freq-cutoff" min="0.05" max="0.5" step="0.01" value="0.2">
            <span class="slider-value" id="cutoff-value">0.20</span>
          </div>
        </div>
        
        <div class="btn-group">
          <button class="btn primary" id="btn-animate">▶ Animate</button>
          <button class="btn" id="btn-compare">Compare</button>
        </div>
        
        <div class="result-box" id="result-box" style="display: none;">
          <div class="result-title">Magnitude/Phase Ratio</div>
          <div class="result-value" id="result-ratio">-</div>
          <div class="result-unit">× more invariant</div>
        </div>
      </div>
      
      <!-- Visualization Area -->
      <div class="panel" id="viz-panel">
        <h2>Spectral Decomposition</h2>
        
        <div class="canvas-grid">
          <div class="canvas-card">
            <div class="canvas-header">
              <span class="canvas-title">Original Pattern</span>
            </div>
            <div class="canvas-wrapper">
              <canvas id="canvas-pattern" width="192" height="192"></canvas>
            </div>
          </div>
          
          <div class="canvas-card what">
            <div class="canvas-header">
              <span class="canvas-title">FFT Magnitude</span>
              <span class="canvas-badge what">WHAT</span>
            </div>
            <div class="canvas-wrapper">
              <canvas id="canvas-magnitude" width="192" height="192"></canvas>
            </div>
          </div>
          
          <div class="canvas-card where">
            <div class="canvas-header">
              <span class="canvas-title">FFT Phase</span>
              <span class="canvas-badge where">WHERE</span>
            </div>
            <div class="canvas-wrapper">
              <canvas id="canvas-phase" width="192" height="192"></canvas>
            </div>
          </div>
          
          <div class="canvas-card what">
            <div class="canvas-header">
              <span class="canvas-title">Low Frequency</span>
              <span class="canvas-badge what">WHAT</span>
            </div>
            <div class="canvas-wrapper">
              <canvas id="canvas-lowfreq" width="192" height="192"></canvas>
            </div>
          </div>
          
          <div class="canvas-card where">
            <div class="canvas-header">
              <span class="canvas-title">High Frequency</span>
              <span class="canvas-badge where">WHERE</span>
            </div>
            <div class="canvas-wrapper">
              <canvas id="canvas-highfreq" width="192" height="192"></canvas>
            </div>
          </div>
          
          <div class="canvas-card">
            <div class="canvas-header">
              <span class="canvas-title">Frequency Mask</span>
            </div>
            <div class="canvas-wrapper">
              <canvas id="canvas-mask" width="192" height="192"></canvas>
            </div>
          </div>
        </div>
        
        <!-- Comparison Section -->
        <div class="comparison-section" id="comparison-section" style="display: none;">
          <div class="comparison-header">
            <span class="comparison-title">Position Comparison: Same Pattern, Different Locations</span>
          </div>
          
          <div class="comparison-grid">
            <div class="position-panel">
              <h3>Position A (Left)</h3>
              <canvas class="position-canvas" id="canvas-pos1"></canvas>
            </div>
            
            <div class="similarity-panel">
              <div class="similarity-bar">
                <div class="similarity-label">
                  <span>Magnitude</span>
                  <span id="sim-magnitude" style="color: var(--what)">-</span>
                </div>
                <div class="similarity-track">
                  <div class="similarity-fill what" id="bar-magnitude" style="width: 0%"></div>
                </div>
              </div>
              
              <div class="similarity-bar">
                <div class="similarity-label">
                  <span>Phase</span>
                  <span id="sim-phase" style="color: var(--where)">-</span>
                </div>
                <div class="similarity-track">
                  <div class="similarity-fill where" id="bar-phase" style="width: 0%"></div>
                </div>
              </div>
              
              <div class="similarity-bar">
                <div class="similarity-label">
                  <span>Low Freq</span>
                  <span id="sim-lowfreq" style="color: var(--what)">-</span>
                </div>
                <div class="similarity-track">
                  <div class="similarity-fill what" id="bar-lowfreq" style="width: 0%"></div>
                </div>
              </div>
              
              <div class="similarity-bar">
                <div class="similarity-label">
                  <span>High Freq</span>
                  <span id="sim-highfreq" style="color: var(--where)">-</span>
                </div>
                <div class="similarity-track">
                  <div class="similarity-fill where" id="bar-highfreq" style="width: 0%"></div>
                </div>
              </div>
            </div>
            
            <div class="position-panel">
              <h3>Position B (Right)</h3>
              <canvas class="position-canvas" id="canvas-pos2"></canvas>
            </div>
          </div>
        </div>
      </div>
    </div>
    
    <div class="footer">
      <p>Spectral Attention Lab · <a href="#">Documentation</a></p>
    </div>
  </div>
  
  <script>
    // ========================================================================
    // State
    // ========================================================================
    const state = {
      patternType: 'ring',
      gridSize: 64,
      patternSize: 0.12,
      freqCutoff: 0.2,
      cx: 0.5,
      cy: 0.5,
      animating: false,
      animationFrame: null,
      positions: [],
      positionIndex: 0,
    };
    
    // ========================================================================
    // Canvas Utilities
    // ========================================================================
    function getCanvasContext(id) {
      const canvas = document.getElementById(id);
      return canvas.getContext('2d');
    }
    
    function drawMatrix(ctx, data, colormap = 'magma', gridSize = 64) {
      const canvas = ctx.canvas;
      const cellW = canvas.width / gridSize;
      const cellH = canvas.height / gridSize;
      
      const imageData = ctx.createImageData(canvas.width, canvas.height);
      
      for (let y = 0; y < gridSize; y++) {
        for (let x = 0; x < gridSize; x++) {
          const value = data[y][x];
          const color = getColor(value, colormap);
          
          // Fill cell
          for (let dy = 0; dy < cellH; dy++) {
            for (let dx = 0; dx < cellW; dx++) {
              const px = Math.floor(x * cellW + dx);
              const py = Math.floor(y * cellH + dy);
              const idx = (py * canvas.width + px) * 4;
              imageData.data[idx] = color.r;
              imageData.data[idx + 1] = color.g;
              imageData.data[idx + 2] = color.b;
              imageData.data[idx + 3] = 255;
            }
          }
        }
      }
      
      ctx.putImageData(imageData, 0, 0);
    }
    
    function getColor(value, colormap) {
      value = Math.max(0, Math.min(1, value));
      
      if (colormap === 'magma') {
        // Magma-like colormap
        const colors = [
          {r: 0, g: 0, b: 4},
          {r: 40, g: 11, b: 84},
          {r: 101, g: 21, b: 110},
          {r: 159, g: 42, b: 99},
          {r: 212, g: 72, b: 66},
          {r: 245, g: 125, b: 21},
          {r: 250, g: 193, b: 39},
          {r: 252, g: 253, b: 191},
        ];
        return interpolateColors(colors, value);
      } else if (colormap === 'viridis') {
        // Viridis-like colormap (for WHAT)
        const colors = [
          {r: 68, g: 1, b: 84},
          {r: 72, g: 40, b: 120},
          {r: 62, g: 74, b: 137},
          {r: 49, g: 104, b: 142},
          {r: 38, g: 130, b: 142},
          {r: 31, g: 158, b: 137},
          {r: 53, g: 183, b: 121},
          {r: 109, g: 205, b: 89},
          {r: 180, g: 222, b: 44},
          {r: 253, g: 231, b: 37},
        ];
        return interpolateColors(colors, value);
      } else if (colormap === 'plasma') {
        // Plasma-like colormap (for WHERE)
        const colors = [
          {r: 13, g: 8, b: 135},
          {r: 75, g: 3, b: 161},
          {r: 125, g: 3, b: 168},
          {r: 168, g: 34, b: 150},
          {r: 203, g: 70, b: 121},
          {r: 229, g: 107, b: 93},
          {r: 248, g: 148, b: 65},
          {r: 253, g: 195, b: 40},
          {r: 240, g: 249, b: 33},
        ];
        return interpolateColors(colors, value);
      } else if (colormap === 'hsv') {
        // HSV for phase
        const hue = value * 360;
        return hsvToRgb(hue, 0.9, 0.9);
      } else {
        // Grayscale
        const v = Math.floor(value * 255);
        return {r: v, g: v, b: v};
      }
    }
    
    function interpolateColors(colors, value) {
      const n = colors.length - 1;
      const idx = value * n;
      const i = Math.floor(idx);
      const t = idx - i;
      
      if (i >= n) return colors[n];
      if (i < 0) return colors[0];
      
      const c1 = colors[i];
      const c2 = colors[i + 1];
      
      return {
        r: Math.round(c1.r + (c2.r - c1.r) * t),
        g: Math.round(c1.g + (c2.g - c1.g) * t),
        b: Math.round(c1.b + (c2.b - c1.b) * t),
      };
    }
    
    function hsvToRgb(h, s, v) {
      const c = v * s;
      const x = c * (1 - Math.abs((h / 60) % 2 - 1));
      const m = v - c;
      let r, g, b;
      
      if (h < 60) { r = c; g = x; b = 0; }
      else if (h < 120) { r = x; g = c; b = 0; }
      else if (h < 180) { r = 0; g = c; b = x; }
      else if (h < 240) { r = 0; g = x; b = c; }
      else if (h < 300) { r = x; g = 0; b = c; }
      else { r = c; g = 0; b = x; }
      
      return {
        r: Math.round((r + m) * 255),
        g: Math.round((g + m) * 255),
        b: Math.round((b + m) * 255),
      };
    }
    
    // ========================================================================
    // API Calls
    // ========================================================================
    async function fetchPattern(cx = 0.5, cy = 0.5) {
      const response = await fetch('/api/pattern', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
          pattern_type: state.patternType,
          cx: cx,
          cy: cy,
          size: state.patternSize,
          grid_size: state.gridSize,
          freq_cutoff: state.freqCutoff,
        }),
      });
      return response.json();
    }
    
    async function fetchComparison() {
      const response = await fetch('/api/compare', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
          pattern_type: state.patternType,
          pos1_x: 0.25,
          pos1_y: 0.5,
          pos2_x: 0.75,
          pos2_y: 0.5,
          size: state.patternSize,
          grid_size: state.gridSize,
          freq_cutoff: state.freqCutoff,
        }),
      });
      return response.json();
    }
    
    async function fetchDemoPositions() {
      const response = await fetch('/api/demo_positions');
      return response.json();
    }
    
    // ========================================================================
    // Rendering
    // ========================================================================
    async function renderPattern(cx = 0.5, cy = 0.5) {
      try {
        const data = await fetchPattern(cx, cy);
        const gridSize = data.grid_size;
        
        drawMatrix(getCanvasContext('canvas-pattern'), data.pattern, 'magma', gridSize);
        drawMatrix(getCanvasContext('canvas-magnitude'), data.magnitude, 'viridis', gridSize);
        drawMatrix(getCanvasContext('canvas-phase'), data.phase, 'hsv', gridSize);
        drawMatrix(getCanvasContext('canvas-lowfreq'), data.low_freq, 'viridis', gridSize);
        drawMatrix(getCanvasContext('canvas-highfreq'), data.high_freq, 'plasma', gridSize);
        
        // Draw frequency mask (circular)
        drawFrequencyMask(getCanvasContext('canvas-mask'), gridSize);
        
      } catch (e) {
        console.error('Error rendering pattern:', e);
      }
    }
    
    function drawFrequencyMask(ctx, gridSize) {
      const canvas = ctx.canvas;
      const centerX = canvas.width / 2;
      const centerY = canvas.height / 2;
      const maxRadius = Math.min(centerX, centerY);
      const cutoffRadius = maxRadius * state.freqCutoff * 2;
      
      // Clear
      ctx.fillStyle = '#000';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      
      // Draw low-freq region (WHAT)
      const gradient = ctx.createRadialGradient(centerX, centerY, 0, centerX, centerY, cutoffRadius);
      gradient.addColorStop(0, 'rgba(34, 197, 94, 0.8)');
      gradient.addColorStop(0.8, 'rgba(34, 197, 94, 0.4)');
      gradient.addColorStop(1, 'rgba(34, 197, 94, 0)');
      ctx.fillStyle = gradient;
      ctx.beginPath();
      ctx.arc(centerX, centerY, cutoffRadius, 0, Math.PI * 2);
      ctx.fill();
      
      // Draw high-freq region (WHERE)
      ctx.strokeStyle = 'rgba(239, 68, 68, 0.6)';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(centerX, centerY, cutoffRadius, 0, Math.PI * 2);
      ctx.stroke();
      
      // Labels
      ctx.font = '12px monospace';
      ctx.fillStyle = '#22c55e';
      ctx.textAlign = 'center';
      ctx.fillText('WHAT', centerX, centerY - 5);
      ctx.fillText('(low-freq)', centerX, centerY + 10);
      
      ctx.fillStyle = '#ef4444';
      ctx.fillText('WHERE', centerX, canvas.height - 20);
      ctx.fillText('(high-freq)', centerX, canvas.height - 5);
    }
    
    async function renderComparison() {
      try {
        const data = await fetchComparison();
        const gridSize = data.grid_size;
        
        // Show comparison section
        document.getElementById('comparison-section').style.display = 'block';
        
        // Resize comparison canvases
        const canvas1 = document.getElementById('canvas-pos1');
        const canvas2 = document.getElementById('canvas-pos2');
        canvas1.width = canvas1.offsetWidth;
        canvas1.height = canvas1.offsetWidth;
        canvas2.width = canvas2.offsetWidth;
        canvas2.height = canvas2.offsetWidth;
        
        // Draw patterns
        drawMatrix(canvas1.getContext('2d'), data.pattern1, 'magma', gridSize);
        drawMatrix(canvas2.getContext('2d'), data.pattern2, 'magma', gridSize);
        
        // Update similarity bars
        const sims = data.similarities;
        
        updateSimilarityBar('magnitude', sims.magnitude);
        updateSimilarityBar('phase', sims.phase);
        updateSimilarityBar('lowfreq', sims.low_freq);
        updateSimilarityBar('highfreq', sims.high_freq);
        
        // Show result box
        const ratio = Math.abs(sims.magnitude / (sims.phase + 0.001));
        document.getElementById('result-box').style.display = 'block';
        document.getElementById('result-ratio').textContent = ratio.toFixed(1);
        
      } catch (e) {
        console.error('Error rendering comparison:', e);
      }
    }
    
    function updateSimilarityBar(name, value) {
      const percentage = Math.abs(value) * 100;
      document.getElementById(`sim-${name}`).textContent = value.toFixed(3);
      document.getElementById(`bar-${name}`).style.width = `${Math.min(100, percentage)}%`;
    }
    
    // ========================================================================
    // Animation
    // ========================================================================
    async function startAnimation() {
      if (state.animating) {
        stopAnimation();
        return;
      }
      
      state.animating = true;
      document.getElementById('btn-animate').textContent = '⏹ Stop';
      document.getElementById('viz-panel').classList.add('animating');
      
      // Fetch positions
      const data = await fetchDemoPositions();
      state.positions = data.positions;
      state.positionIndex = 0;
      
      animationLoop();
    }
    
    function stopAnimation() {
      state.animating = false;
      document.getElementById('btn-animate').textContent = '▶ Animate';
      document.getElementById('viz-panel').classList.remove('animating');
      
      if (state.animationFrame) {
        cancelAnimationFrame(state.animationFrame);
      }
    }
    
    async function animationLoop() {
      if (!state.animating) return;
      
      const pos = state.positions[state.positionIndex];
      await renderPattern(pos.cx, pos.cy);
      
      state.positionIndex = (state.positionIndex + 1) % state.positions.length;
      
      // Wait a bit before next frame
      setTimeout(() => {
        if (state.animating) {
          state.animationFrame = requestAnimationFrame(animationLoop);
        }
      }, 100);
    }
    
    // ========================================================================
    // Event Handlers
    // ========================================================================
    function setupEventHandlers() {
      // Pattern type
      document.getElementById('pattern-type').addEventListener('change', (e) => {
        state.patternType = e.target.value;
        renderPattern(state.cx, state.cy);
      });
      
      // Grid size
      document.getElementById('grid-size').addEventListener('change', (e) => {
        state.gridSize = parseInt(e.target.value);
        renderPattern(state.cx, state.cy);
      });
      
      // Pattern size slider
      document.getElementById('pattern-size').addEventListener('input', (e) => {
        state.patternSize = parseFloat(e.target.value);
        document.getElementById('size-value').textContent = state.patternSize.toFixed(2);
        renderPattern(state.cx, state.cy);
      });
      
      // Frequency cutoff slider
      document.getElementById('freq-cutoff').addEventListener('input', (e) => {
        state.freqCutoff = parseFloat(e.target.value);
        document.getElementById('cutoff-value').textContent = state.freqCutoff.toFixed(2);
        renderPattern(state.cx, state.cy);
      });
      
      // Animate button
      document.getElementById('btn-animate').addEventListener('click', startAnimation);
      
      // Compare button
      document.getElementById('btn-compare').addEventListener('click', renderComparison);
      
      // Click on main pattern canvas to set position
      document.getElementById('canvas-pattern').addEventListener('click', (e) => {
        const rect = e.target.getBoundingClientRect();
        state.cx = (e.clientX - rect.left) / rect.width;
        state.cy = (e.clientY - rect.top) / rect.height;
        renderPattern(state.cx, state.cy);
      });
    }
    
    // ========================================================================
    // Initialize
    // ========================================================================
    document.addEventListener('DOMContentLoaded', () => {
      setupEventHandlers();
      renderPattern();
    });
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8765)

