"""Streaming Zipf-grounded wave tokenizer.

Converts token ids to wave embeddings using Zipf-ranked frequencies with
lightweight streaming support (position accumulation) and multi-harmonic
signals projected into a fixed embedding dimension.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Union

import torch
import torch.nn as nn


@dataclass
class StreamZipfWaveConfig:
    """Configuration for the streaming Zipf wave tokenizer.

    Args:
        vocab_size: Size of the tokenizer vocabulary.
        embed_dim: Output embedding dimension.
        n_harmonics: Number of harmonic components per token.
        freq_min: Minimum normalized frequency (for most common tokens).
        freq_max: Maximum normalized frequency (for rare tokens).
        phase_mode: How to set phase ("position" is deterministic).
        cache_top_k: Reserved for future waveform caching (not used yet).
    """

    vocab_size: int = 50257
    embed_dim: int = 128
    n_harmonics: int = 2
    freq_min: float = 0.0
    freq_max: float = 1.0
    phase_mode: str = "position"
    cache_top_k: int = 512

    def validate(self) -> bool:
        assert self.vocab_size > 0, "vocab_size must be positive"
        assert self.embed_dim > 0, "embed_dim must be positive"
        assert self.n_harmonics > 0, "n_harmonics must be positive"
        assert 0.0 <= self.freq_min < self.freq_max <= 1.0, "freq_min/freq_max must satisfy 0 <= min < max <= 1"
        assert self.phase_mode in ["position"], "only position-driven phase is supported in streaming mode"
        assert self.cache_top_k >= 0, "cache_top_k must be non-negative"
        return True


class StreamingZipfWaveTokenizer(nn.Module):
    """Streaming tokenizer that maps token ids to wave embeddings.

    - Frequencies derive from Zipf ranks: f = freq_min + (freq_max - freq_min) * log(rank)/log(V)
    - Phases are position-driven for determinism in streaming.
    - Multi-harmonic sin/cos basis projected to embed_dim.
    - Maintains per-session position counters for incremental calls.
    """

    def __init__(
        self,
        rank_table: torch.Tensor,
        config: Optional[StreamZipfWaveConfig] = None,
    ) -> None:
        super().__init__()
        self.config = config or StreamZipfWaveConfig()
        self.config.validate()

        assert rank_table.numel() == self.config.vocab_size, "rank_table size must match vocab_size"
        self.register_buffer("rank_table", rank_table.float())

        harmonics = torch.arange(1, self.config.n_harmonics + 1, dtype=torch.float32)
        self.register_buffer("harmonics", harmonics)

        self.proj = nn.Linear(2 * self.config.n_harmonics, self.config.embed_dim, bias=False)

        # Session state: session_id -> current position offset
        self._session_pos: Dict[str, int] = {}

    @staticmethod
    def load_rank_table(path: Union[str, Path]) -> torch.Tensor:
        """Load ranks from a JSON file saved by the original tokenizer utilities."""
        import json

        with open(path, "r") as f:
            data = json.load(f)
        ranks = torch.tensor(data["ranks"], dtype=torch.float32)
        return ranks

    def reset_session(self, session_id: str) -> None:
        """Reset the position counter for a streaming session."""
        self._session_pos[session_id] = 0

    def _freq_from_rank(self, ranks: torch.Tensor) -> torch.Tensor:
        """Map ranks to normalized frequencies in [freq_min, freq_max]."""
        log_r = torch.log(ranks.clamp(min=1.0))
        log_v = torch.log(torch.tensor(float(self.config.vocab_size)))
        norm = log_r / log_v
        return self.config.freq_min + (self.config.freq_max - self.config.freq_min) * norm

    def _positions(self, seq_len: int, device: torch.device, session_id: Optional[str]) -> torch.Tensor:
        if session_id is None:
            return torch.arange(seq_len, device=device, dtype=torch.float32)
        start = self._session_pos.get(session_id, 0)
        pos = torch.arange(start, start + seq_len, device=device, dtype=torch.float32)
        self._session_pos[session_id] = start + seq_len
        return pos

    def forward(
        self,
        token_ids: torch.Tensor,
        session_id: Optional[str] = None,
        positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Convert token ids to wave embeddings.

        Args:
            token_ids: [T] or [B, T] tensor of token ids.
            session_id: Optional streaming session identifier.
            positions: Optional explicit positions; if provided, session_id is ignored.

        Returns:
            Wave embeddings shaped [T, D] or [B, T, D] matching input batch.
        """
        if token_ids.dim() == 1:
            return self._encode_sequence(token_ids, session_id=session_id, positions=positions)
        elif token_ids.dim() == 2:
            embeddings = []
            for b in range(token_ids.size(0)):
                emb = self._encode_sequence(
                    token_ids[b],
                    session_id=f"{session_id}:{b}" if session_id is not None else None,
                    positions=None if positions is None else positions[b],
                )
                embeddings.append(emb)
            return torch.stack(embeddings, dim=0)
        else:
            raise ValueError("token_ids must be 1D or 2D (T or B,T)")

    def _encode_sequence(
        self,
        token_ids: torch.Tensor,
        session_id: Optional[str],
        positions: Optional[torch.Tensor],
    ) -> torch.Tensor:
        device = token_ids.device
        seq_len = token_ids.size(0)

        pos = positions.float() if positions is not None else self._positions(seq_len, device, session_id)

        ranks = self.rank_table.to(device)[token_ids]
        freqs = self._freq_from_rank(ranks)  # [T]

        # Phase: position-driven, broadcast over harmonics
        base_phase = 2.0 * torch.pi * freqs[:, None] * self.harmonics[None, :] * pos[:, None]

        sin_part = torch.sin(base_phase)
        cos_part = torch.cos(base_phase)
        waves = torch.cat([sin_part, cos_part], dim=-1)  # [T, 2*n_harmonics]

        embeddings = self.proj(waves)
        return embeddings

