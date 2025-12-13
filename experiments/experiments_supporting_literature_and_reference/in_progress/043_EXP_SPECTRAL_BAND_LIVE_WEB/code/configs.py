"""
Configuration presets for the Spectral Band Additive Viewer.
"""

from typing import Dict


def get_fast_config(grid_size: int = 32) -> Dict:
    """Fast configuration for quick iteration."""
    return {
        'gridSize': grid_size,
        'baseDim': 32,
        'attnDim': 32,
        'timeDepth': 8,
        'attnTopKTemporal': 4,
        'decayRate': 0.95,
        'layerRange': 3,
        'wormholeThreshold': 0.92,
        'wormhole_min_temporal_distance': 1,
        'wormhole_max_connections': 8,
        'wormhole_skip_steps': 2,
        'wormhole_chunk_size': 2048,
        'temporal_causal': True,
        'neighbor_causal': True,
        'wormhole_causal': True,
        'neighbor_first_layer_only': False,
        'freq_cutoff': 0.25,
    }


def get_default_config(grid_size: int = 32) -> Dict:
    """Default balanced configuration."""
    return {
        'gridSize': grid_size,
        'baseDim': 64,
        'attnDim': 64,
        'timeDepth': 16,
        'attnTopKTemporal': 4,
        'decayRate': 0.95,
        'layerRange': 5,
        'wormholeThreshold': 0.92,
        'wormhole_min_temporal_distance': 1,
        'wormhole_max_connections': 16,
        'wormhole_skip_steps': 1,
        'wormhole_chunk_size': 4096,
        'temporal_causal': True,
        'neighbor_causal': True,
        'wormhole_causal': True,
        'neighbor_first_layer_only': False,
        'freq_cutoff': 0.25,
    }


def get_turbo_config(grid_size: int = 32) -> Dict:
    """Turbo configuration for maximum speed."""
    return {
        'gridSize': grid_size,
        'baseDim': 16,
        'attnDim': 16,
        'timeDepth': 4,
        'attnTopKTemporal': 2,
        'decayRate': 0.9,
        'layerRange': 2,
        'wormholeThreshold': 0.95,
        'wormhole_min_temporal_distance': 1,
        'wormhole_max_connections': 4,
        'wormhole_skip_steps': 4,
        'wormhole_chunk_size': 1024,
        'temporal_causal': True,
        'neighbor_causal': True,
        'wormhole_causal': True,
        'neighbor_first_layer_only': True,
        'freq_cutoff': 0.3,
    }


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


SUPPORTED_CONFIGS = {
    'fast': get_fast_config,
    'default': get_default_config,
    'turbo': get_turbo_config,
}

