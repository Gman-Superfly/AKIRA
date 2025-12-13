"""Streaming decoder from Zipf-grounded wave embeddings back to token ids.

Assumes embeddings were produced by StreamingZipfWaveTokenizer (sine/cos
harmonics projected by a linear layer). Decoding is approximate:
- Invert the projection with a pseudo-inverse to recover sin/cos.
- Estimate normalized frequency from phase and position.
- Map frequency → rank → nearest token id via the rank table.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn as nn


@dataclass
class StreamZipfDecodeConfig:
    """Configuration for streaming decoder."""

    vocab_size: int = 50257
    n_harmonics: int = 2
    freq_min: float = 0.0
    freq_max: float = 1.0
    eps: float = 1e-6

    def validate(self) -> bool:
        assert self.vocab_size > 0, "vocab_size must be positive"
        assert self.n_harmonics > 0, "n_harmonics must be positive"
        assert 0.0 <= self.freq_min < self.freq_max <= 1.0, "freq_min/freq_max must satisfy 0 <= min < max <= 1"
        assert self.eps > 0, "eps must be positive"
        return True


class StreamingZipfWaveDecoder(nn.Module):
    """Approximate decoder from wave embeddings to token ids."""

    def __init__(
        self,
        rank_table: torch.Tensor,
        proj_weight: torch.Tensor,
        config: Optional[StreamZipfDecodeConfig] = None,
    ) -> None:
        """
        Args:
            rank_table: [V] tensor of ranks (1 = most frequent).
            proj_weight: [embed_dim, 2 * n_harmonics] weight matrix from encoder projection.
        """
        super().__init__()
        self.config = config or StreamZipfDecodeConfig()
        self.config.validate()

        assert rank_table.numel() == self.config.vocab_size, "rank_table size must match vocab_size"
        self.register_buffer("rank_table", rank_table.float())

        # Precompute sorted ranks for nearest-token lookup
        sorted_ranks, sorted_ids = torch.sort(self.rank_table)
        self.register_buffer("sorted_ranks", sorted_ranks)
        self.register_buffer("sorted_token_ids", sorted_ids)

        # Pseudo-inverse of projection to recover sin/cos
        # proj_weight: [D, 2H] -> pinv: [2H, D]
        pinv = torch.pinverse(proj_weight.float())
        self.register_buffer("proj_pinv", pinv)

        harmonics = torch.arange(1, self.config.n_harmonics + 1, dtype=torch.float32)
        self.register_buffer("harmonics", harmonics)

    @staticmethod
    def load_rank_table(path: Union[str, Path]) -> torch.Tensor:
        import json

        with open(path, "r") as f:
            data = json.load(f)
        return torch.tensor(data["ranks"], dtype=torch.float32)

    def _freq_to_rank(self, freq_norm: torch.Tensor) -> torch.Tensor:
        """Invert normalized frequency to rank (floating)."""
        freq_norm = freq_norm.clamp(self.config.freq_min, self.config.freq_max)
        span = self.config.freq_max - self.config.freq_min
        norm01 = (freq_norm - self.config.freq_min) / span
        log_v = torch.log(torch.tensor(float(self.config.vocab_size), device=freq_norm.device))
        rank_est = torch.exp(norm01 * log_v)
        return rank_est

    def _rank_to_token_id(self, rank_est: torch.Tensor) -> torch.Tensor:
        """Map estimated rank to nearest token id using sorted ranks."""
        idx = torch.searchsorted(self.sorted_ranks, rank_est)
        idx_clamped = idx.clamp(1, self.sorted_ranks.numel() - 1)

        lower_idx = idx_clamped - 1
        upper_idx = idx_clamped

        lower_rank = self.sorted_ranks[lower_idx]
        upper_rank = self.sorted_ranks[upper_idx]

        dist_lower = (rank_est - lower_rank).abs()
        dist_upper = (upper_rank - rank_est).abs()

        choose_upper = dist_upper < dist_lower
        chosen_idx = torch.where(choose_upper, upper_idx, lower_idx)

        token_ids = self.sorted_token_ids[chosen_idx]
        return token_ids

    def _estimate_freq(self, sin_part: torch.Tensor, cos_part: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Estimate normalized frequency per position from phase."""
        # phase: [T, H]
        phase = torch.atan2(sin_part, cos_part)  # [-pi, pi]
        positions = positions[:, None].clamp(min=self.config.eps)
        freq_est = phase / (2.0 * torch.pi * self.harmonics[None, :] * positions)
        # Average over harmonics for robustness
        freq_norm = freq_est.mean(dim=-1)
        return freq_norm

    def forward(
        self,
        embeddings: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Decode embeddings to token ids.

        Args:
            embeddings: [T, D] or [B, T, D] tensor.
            positions: Optional positions; if None, uses arange per sequence.

        Returns:
            token_ids: [T] or [B, T] tensor of decoded token ids.
        """
        if embeddings.dim() == 2:
            return self._decode_sequence(embeddings, positions)
        elif embeddings.dim() == 3:
            outputs = []
            for b in range(embeddings.size(0)):
                pos_b = None if positions is None else positions[b]
                outputs.append(self._decode_sequence(embeddings[b], pos_b))
            return torch.stack(outputs, dim=0)
        else:
            raise ValueError("embeddings must be 2D or 3D")

    def _decode_sequence(self, embeddings: torch.Tensor, positions: Optional[torch.Tensor]) -> torch.Tensor:
        device = embeddings.device
        T, D = embeddings.shape
        pos = positions
        if pos is None:
            pos = torch.arange(T, device=device, dtype=torch.float32)
        else:
            pos = pos.to(device).float()

        # Recover approx sin/cos components: [T, 2H]
        sincos = embeddings @ self.proj_pinv  # [T, 2H]
        sin_part, cos_part = torch.split(sincos, self.config.n_harmonics, dim=-1)

        freq_norm = self._estimate_freq(sin_part, cos_part, pos)
        rank_est = self._freq_to_rank(freq_norm)
        token_ids = self._rank_to_token_id(rank_est)
        return token_ids

