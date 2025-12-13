# Word-aware Zipf frequencies for wave tokenization

## Goal
Assign wave frequencies to subword tokens based on the words they belong to, using word-level Zipf statistics and per-query context. This makes the spectral distribution query-dependent instead of fixed by global token ranks.

## Motivation
- Subword token IDs do not map cleanly to word frequency; the same token can appear in common and rare words.
- A query has its own frequency structure (local tf / tf-idf), which should influence band placement.
- We want band separation that reflects the actual words present, not just the global tokenizer distribution.

## Process
1) Input text → words → subword tokens with alignment (keep word boundaries).
2) For each word:
   - Look up global word rank (from a word frequency table). If unknown, fall back to token-level prior.
   - Optionally mix with per-query rank (local tf ordering or tf-idf) to get a query-specific effective rank.
3) For each token belonging to that word:
   - Assign the word’s effective rank as the token’s rank.
   - Option (split vs share): simplest is to share the same rank across all tokens of the word. If needed, add a small offset for trailing pieces to reduce overcounting.
4) Normalize ranks to [0, 1] for this sequence to preserve band spread:
   - norm_rank = log(rank) / log(V_word)  (use global vocab size for stability)
5) Map norm_rank → wave frequency: f = freq_min + (freq_max - freq_min) * norm_rank
6) Pass per-token frequencies to the wave encoder (phase still position-driven).

## Design choices
- Word frequency source: use a global word frequency table; cache-limited but stable. Optionally combine with query-local tf or tf-idf:
  rank_eff = mix(global_rank, local_rank_order)  (e.g., weighted geometric mean)
- Token sharing: by default, every subword of a word gets the same frequency; keeps simplicity and coherence.
- Fallback: if a word is unseen, fallback to global token rank; if tokens are unseen, assign high frequency (rarest band).
- Band consistency: keep Zipf-based 7-band split; per-query ranks should still cover low→high to avoid band collapse.

## Outputs
- For a given text: per-token frequencies aligned to the tokenizer output, ready for the streaming encoder.
- Optional diagnostics: band histogram per query, entropy of frequency allocation, fraction of tokens in each band.

## Next steps
- Implement a small helper in `code_002/word_frequency_assigner.py`:
  - Inputs: text string, tokenizer (to get tokens + word alignment), word_freq table, optional tf-idf weights.
  - Outputs: per-token frequency tensor aligned with token ids; diagnostics.
- Wire it to the streaming encoder: replace static rank lookup with query-specific frequencies when provided; fallback to static ranks otherwise.

