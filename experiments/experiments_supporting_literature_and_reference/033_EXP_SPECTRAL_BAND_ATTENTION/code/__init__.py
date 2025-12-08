"""
Experiment 033: Spectral Band Attention

A working reference implementation of three-attention-type architecture:
- Temporal: Per-position self-history with Top-K selection
- Neighbor: 8-connected local spatial attention
- Wormhole: Sparse similarity-gated non-local connections
"""

from .temporal_attention import TemporalAttention
from .neighbor_attention import NeighborAttention
from .wormhole_attention import WormholeAttention
from .spectral_attention import SpectralBandAttention

__all__ = [
    'TemporalAttention',
    'NeighborAttention',
    'WormholeAttention',
    'SpectralBandAttention',
]
