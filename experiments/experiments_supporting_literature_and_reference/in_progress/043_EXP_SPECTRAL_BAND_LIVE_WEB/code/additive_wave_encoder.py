"""
Additive Wave Encoder

Converts attention patterns into wave representations for visualization.
Each position gets a wave with frequency based on its index (or rank).
The attention-weighted sum shows constructive/destructive interference.
"""

import math
from dataclasses import dataclass
from typing import Tuple, List, Dict

import torch
import numpy as np


@dataclass
class WaveConfig:
    """Configuration for wave generation."""
    n_harmonics: int = 7
    freq_min: float = 1.0
    freq_max: float = 10.0
    sample_rate: int = 200
    duration: float = 1.0


class AdditiveWaveEncoder:
    """
    Encode attention patterns as additive waves.
    
    Each position i in the sequence gets a base frequency f_i.
    The attention weights determine amplitude of each wave.
    Superposition reveals phase alignment.
    """
    
    def __init__(self, config: WaveConfig = None):
        self.config = config or WaveConfig()
        n_samples = int(self.config.sample_rate * self.config.duration)
        self.t = np.linspace(0, self.config.duration, n_samples)
    
    def position_to_frequency(self, pos: int, seq_len: int) -> float:
        """Map position to frequency. Earlier positions get higher freq (more common)."""
        if seq_len <= 1:
            return self.config.freq_min
        normalized = pos / (seq_len - 1)
        return self.config.freq_min + (self.config.freq_max - self.config.freq_min) * (1 - normalized)
    
    def generate_wave(self, freq: float, phase_offset: float = 0.0) -> np.ndarray:
        """Generate a wave with harmonics."""
        wave = np.zeros_like(self.t)
        for h in range(1, self.config.n_harmonics + 1):
            amplitude = 1.0 / h
            wave += amplitude * np.sin(2 * math.pi * freq * h * self.t + phase_offset * h)
        return wave
    
    def encode_attention(
        self,
        attention_weights: np.ndarray,
        query_pos: int = -1
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Encode attention pattern as superposed wave.
        
        Args:
            attention_weights: Attention weights [seq_len] or [heads, seq_len]
            query_pos: Which query position to analyze (-1 = last)
        
        Returns:
            superposed: The weighted sum of all waves [n_samples]
            individual_waves: Each position's wave [seq_len, n_samples]
            coherence: Phase coherence R (0-1)
        """
        if attention_weights.ndim == 2:
            weights = attention_weights.mean(axis=0)
        else:
            weights = attention_weights
        
        seq_len = len(weights)
        individual_waves = np.zeros((seq_len, len(self.t)))
        
        for i in range(seq_len):
            freq = self.position_to_frequency(i, seq_len)
            phase_offset = 2 * math.pi * i / seq_len
            individual_waves[i] = self.generate_wave(freq, phase_offset)
        
        superposed = np.zeros_like(self.t)
        for i in range(seq_len):
            superposed += weights[i] * individual_waves[i]
        
        coherence = self.compute_coherence(weights, seq_len)
        
        return superposed, individual_waves, coherence
    
    def compute_coherence(self, weights: np.ndarray, seq_len: int) -> float:
        """
        Compute phase coherence R for the attention pattern.
        Maps positions to angles, computes attention-weighted centroid magnitude.
        """
        angles = 2 * math.pi * np.arange(seq_len) / seq_len
        real_part = np.sum(weights * np.cos(angles))
        imag_part = np.sum(weights * np.sin(angles))
        R = np.sqrt(real_part**2 + imag_part**2)
        return float(R)
    
    def compute_interference_strength(
        self,
        attention_weights: np.ndarray
    ) -> float:
        """
        Compute interference strength: power of superposition vs sum of powers.
        > 1 = constructive, < 1 = destructive
        """
        superposed, individual, _ = self.encode_attention(attention_weights)
        power_super = np.mean(superposed**2)
        power_sum = np.sum(attention_weights**2 * np.mean(individual**2, axis=1))
        return float(power_super / (power_sum + 1e-10))


class AttentionWaveAnalyzer:
    """
    Analyze attention across layers using wave representation.
    Tracks coherence, entropy, and head synchronization.
    """
    
    def __init__(self, config: WaveConfig = None):
        self.encoder = AdditiveWaveEncoder(config)
    
    def analyze_layer(
        self,
        attention: np.ndarray,
        query_pos: int = -1
    ) -> Dict:
        """
        Analyze a single layer's attention.
        
        Args:
            attention: Attention weights [heads, query_seq, key_seq]
            query_pos: Which query position to analyze
        
        Returns:
            Dictionary with entropy, coherence, head_sync, waves
        """
        n_heads = attention.shape[0]
        
        per_head_attention = attention[:, query_pos, :]
        
        head_phases = []
        head_coherences = []
        head_entropies = []
        
        for h in range(n_heads):
            w = per_head_attention[h]
            
            _, _, coh = self.encoder.encode_attention(w)
            head_coherences.append(coh)
            
            angles = 2 * math.pi * np.arange(len(w)) / len(w)
            real_part = np.sum(w * np.cos(angles))
            imag_part = np.sum(w * np.sin(angles))
            head_phases.append(np.arctan2(imag_part, real_part))
            
            p = w.clip(min=1e-10)
            ent = -np.sum(p * np.log(p))
            head_entropies.append(ent)
        
        phases = np.array(head_phases)
        head_sync = float(np.abs(np.mean(np.exp(1j * phases))))
        
        mean_attention = per_head_attention.mean(axis=0)
        superposed, individual, mean_coherence = self.encoder.encode_attention(mean_attention)
        
        return {
            'entropy': float(np.mean(head_entropies)),
            'coherence': mean_coherence,
            'head_sync': head_sync,
            'head_phases': phases.tolist(),
            'head_coherences': head_coherences,
            'head_entropies': head_entropies,
            'superposed_wave': superposed,
            'individual_waves': individual,
            'mean_attention': mean_attention,
            't': self.encoder.t,
        }
    
    def analyze_all_layers(
        self,
        attentions: List[np.ndarray],
        query_pos: int = -1
    ) -> Dict:
        """
        Analyze attention across all layers.
        
        Args:
            attentions: List of attention tensors [heads, query_seq, key_seq]
            query_pos: Which query position to analyze
        
        Returns:
            Dictionary with per-layer analysis and summary
        """
        layer_results = []
        
        for layer_idx, attn in enumerate(attentions):
            result = self.analyze_layer(attn, query_pos)
            result['layer'] = layer_idx
            layer_results.append(result)
        
        entropies = [r['entropy'] for r in layer_results]
        coherences = [r['coherence'] for r in layer_results]
        head_syncs = [r['head_sync'] for r in layer_results]
        
        entropy_diff = np.diff(entropies)
        collapse_layer = int(np.argmin(entropy_diff)) if len(entropy_diff) > 0 else 0
        
        return {
            'layers': layer_results,
            'entropy_trajectory': entropies,
            'coherence_trajectory': coherences,
            'head_sync_trajectory': head_syncs,
            'collapse_layer': collapse_layer,
            'final_coherence': coherences[-1] if coherences else 0.0,
            'final_entropy': entropies[-1] if entropies else 0.0,
        }

