"""Configuration presets for spectral attention experiments."""

from typing import Dict, Any


def get_default_config(grid_size: int = 32) -> Dict[str, Any]:
    """Get default configuration for 2D experiments."""
    return {
        'gridSize': grid_size,
        'baseDim': 64,
        'attnDim': 64,
        'timeDepth': 16,
        
        # Temporal attention
        'attnTopKTemporal': 4,
        'decayRate': 0.95,
        'temporal_causal': True,
        
        # Neighbor attention
        'layerRange': 3,
        'neighbor_first_layer_only': False,
        'neighbor_causal': True,
        
        # Wormhole attention
        # Note: with normalized features, cosine similarities cluster near 1.0,
        # so threshold must be very high (0.999+) to be selective.
        'wormholeThreshold': 0.9995,
        'wormhole_min_temporal_distance': 1,
        'wormhole_max_connections': 4,  # sparse: only top-4 non-local connections
        'wormhole_causal': True,
        'wormhole_skip_steps': 1,
        'wormhole_chunk_size': 4096,
    }


def get_fast_config(grid_size: int = 32) -> Dict[str, Any]:
    """Fast config for quick iteration."""
    config = get_default_config(grid_size)
    config.update({
        'baseDim': 32,
        'attnDim': 32,
        'timeDepth': 8,
        'wormhole_max_connections': 4,
        'wormholeThreshold': 0.9995,
        'wormhole_skip_steps': 2,
        'wormhole_chunk_size': 4096,
    })
    return config


def get_turbo_config(grid_size: int = 32) -> Dict[str, Any]:
    """Turbo config for maximum speed."""
    config = get_default_config(grid_size)
    config.update({
        'baseDim': 16,
        'attnDim': 16,
        'timeDepth': 4,
        'wormhole_max_connections': 2,
        'wormholeThreshold': 0.9998,
        'wormhole_skip_steps': 4,
        'attnTopKTemporal': 2,
        'layerRange': 1,
        'wormhole_chunk_size': 4096,
    })
    return config


def get_large_config(grid_size: int = 64) -> Dict[str, Any]:
    """Large grid config focused on stability and compute control."""
    config = get_default_config(grid_size)
    config.update({
        'baseDim': 32,
        'attnDim': 32,
        'timeDepth': 8,
        'wormhole_max_connections': 8,
        'wormholeThreshold': 0.9995,
        'wormhole_skip_steps': 2,
        'layerRange': 3,
        'attnTopKTemporal': 3,
        'wormhole_chunk_size': 8192,
    })
    return config


def get_xlarge_config(grid_size: int = 128) -> Dict[str, Any]:
    """Extra-large grid config with wormhole connections for non-local signal."""
    config = get_default_config(grid_size)
    config.update({
        'baseDim': 32,
        'attnDim': 32,
        'timeDepth': 8,
        'wormhole_max_connections': 8,
        'wormholeThreshold': 0.9995,
        'wormhole_skip_steps': 2,
        'layerRange': 2,
        'attnTopKTemporal': 3,
        'wormhole_chunk_size': 16384,
    })
    return config
