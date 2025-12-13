"""Word-aware frequency assigner.

Given text and a tokenizer, map each emitted token to a wave frequency
derived from its parent word's Zipf rank, optionally mixed with per-query
term frequency. Designed to be used before the streaming wave encoder.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch


@dataclass
class WordFreqConfig:
    vocab_size_words: int
    freq_min: float = 0.0
    freq_max: float = 1.0
    mix_weight: float = 0.7  # weight on global rank vs local rank

    def validate(self) -> bool:
        assert self.vocab_size_words > 0, "vocab_size_words must be positive"
        assert 0.0 <= self.freq_min < self.freq_max <= 1.0, "freq_min/freq_max must satisfy 0 <= min < max <= 1"
        assert 0.0 <= self.mix_weight <= 1.0, "mix_weight must be in [0,1]"
        return True


class WordFrequencyAssigner:
    """Assign per-token frequencies based on word-level ranks and local tf."""

    def __init__(
        self,
        tokenizer,
        word_rank_table: Dict[str, int],
        config: WordFreqConfig,
    ) -> None:
        """
        Args:
            tokenizer: HuggingFace-like tokenizer with encode_plus returning offsets.
            word_rank_table: map word string -> rank (1 = most frequent).
            config: configuration parameters.
        """
        self.tokenizer = tokenizer
        self.word_rank_table = word_rank_table
        self.config = config
        self.config.validate()

    def _word_ranks(self, words: List[str]) -> List[int]:
        ranks = []
        default_rank = self.config.vocab_size_words  # rarest fallback
        for w in words:
            ranks.append(self.word_rank_table.get(w, default_rank))
        return ranks

    def _local_ranks(self, words: List[str]) -> List[int]:
        # Simple local tf ordering: most frequent in this query gets rank 1 locally
        from collections import Counter

        counts = Counter(words)
        # Sort by count desc, then alpha to stabilize
        sorted_words = sorted(counts.items(), key=lambda x: (-x[1], x[0]))
        local_rank_map = {}
        for i, (w, _) in enumerate(sorted_words, start=1):
            local_rank_map[w] = i
        return [local_rank_map[w] for w in words]

    def _mix_ranks(self, global_rank: int, local_rank: int) -> float:
        # Geometric mean in log space: rank_eff = rank_g^a * rank_l^(1-a)
        a = self.config.mix_weight
        return (global_rank ** a) * (local_rank ** (1.0 - a))

    def _rank_to_freq(self, rank: torch.Tensor) -> torch.Tensor:
        # rank -> normalized [0,1] via log
        log_v = torch.log(torch.tensor(float(self.config.vocab_size_words)))
        norm = torch.log(rank) / log_v
        return self.config.freq_min + (self.config.freq_max - self.config.freq_min) * norm

    def assign(self, text: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Assign per-token frequencies for a given text.

        Returns:
            token_ids: [T] tensor
            freqs: [T] tensor of frequencies aligned with token_ids
        """
        # Tokenize with offsets to align tokens to words
        encoded = self.tokenizer.encode_plus(text, return_offsets_mapping=True, add_special_tokens=False)
        token_ids = torch.tensor(encoded["input_ids"], dtype=torch.long)
        offsets = encoded["offset_mapping"]
        # Build words and map token -> word index
        words = []
        token_to_word: List[int] = []
        prev_end = -1
        word_idx = -1
        for (start, end) in offsets:
            if start == prev_end:
                # continuation of current word
                token_to_word.append(word_idx)
            else:
                # new word boundary
                word_idx += 1
                token_to_word.append(word_idx)
                words.append(text[start:end])
            prev_end = end

        # Clean words (strip whitespace)
        words = [w for w in words]

        global_ranks = self._word_ranks(words)
        local_ranks = self._local_ranks(words)

        # Effective rank per word
        eff_ranks = [
            self._mix_ranks(g, l) for g, l in zip(global_ranks, local_ranks)
        ]

        # Map back to tokens
        rank_tensor = torch.tensor([eff_ranks[idx] for idx in token_to_word], dtype=torch.float32)
        freqs = self._rank_to_freq(rank_tensor)
        return token_ids, freqs

