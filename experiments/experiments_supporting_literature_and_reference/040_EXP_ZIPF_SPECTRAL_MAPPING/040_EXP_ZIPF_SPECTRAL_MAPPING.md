# Experiment 040: Zipf-Spectral Mapping

## Token Frequency to Spectral Band Correspondence

**Tier:** ★ SUPPORTING  
**Status:** PLANNED  
**Depends On:** 003 (Spectral Band Dynamics), 034 (Zipf Wave Tokenizer)  
**References:** THE_LANGUAGE_OF_INFORMATION.md Section 2.4, LANGUAGE_ACTION_CONTEXT.md Section 10.4-10.5

---

## Motivation

### The Core Claim Being Tested

```
ZIPF'S LAW AND SPECTRAL STRUCTURE
─────────────────────────────────

From THE_LANGUAGE_OF_INFORMATION.md Section 2.4:

  "In natural language, symbol frequency follows Zipf's Law:
   f(r) ~ r^(-α)    where r = rank, α ≈ 1.0
   
   THE SPECTRAL CONNECTION:
   If we assign wave frequencies based on Zipf rank:
   
   Common tokens (the, is, a)     → LOW frequency (DC component)
   Rare tokens (quasar, mitochondria) → HIGH frequency (detail)"

From LANGUAGE_ACTION_CONTEXT.md Section 10.4:

  "ZIPF'S LAW AND ACTION QUANTA
   
   Shannon's insight: rare events carry more information.
   Common words: HIGH frequency, LOW information
   Rare words: LOW frequency, HIGH information
   
   This creates a NATURAL spectral mapping..."

THE HYPOTHESIS:
  Token embeddings naturally organize by Zipf rank in spectral space.
  When decomposed via FFT:
    - Common tokens have energy concentrated in LOW bands (DC-like)
    - Rare tokens have energy concentrated in HIGH bands (detail)
  This is NOT arbitrary - it reflects information density.
```

### Why This Matters

1. **Theoretical Foundation:** If Zipf rank naturally maps to spectral bands, then spectral decomposition is not arbitrary but grounded in information theory. AKIRA's bands correspond to information density tiers.

2. **Architecture Justification:** The 7-band structure isn't arbitrary - it separates tokens by information content. Low bands = structural glue, high bands = semantic content.

3. **Compression Insight:** Spectral compression naturally separates what to keep (high-info rare tokens) vs what to summarize (low-info common tokens).

4. **Training Guidance:** Wave embeddings (Exp 034) should be grounded in Zipf frequencies for inductive bias that matches language structure.

---

## Foundation

**Established Science:**

1. **Zipf's Law** (Zipf, 1949) - Word frequency in natural language follows a power law: f(r) ~ r^(-1). The most frequent word appears twice as often as the second, three times as often as the third, etc.

2. **Shannon Entropy** (Shannon, 1948) - Information content is inversely related to probability: I(x) = -log(P(x)). Rare events carry more information than common events.

3. **Mandelbrot's Extension** (Mandelbrot, 1953) - Information-theoretic derivation of Zipf's law. The distribution optimizes communication efficiency given finite symbol costs.

**Bridge to AKIRA:**

If information content is inversely related to frequency (Shannon), and word frequency follows Zipf's law, then:
- Common tokens (low rank) → Low information → Should map to low spectral bands
- Rare tokens (high rank) → High information → Should map to high spectral bands

This provides INDUCTIVE BIAS for spectral embeddings: not arbitrary wave frequencies, but frequencies grounded in information structure.

**Hypothesis:** Token embeddings, when decomposed into spectral bands, show a correlation between Zipf rank and band energy distribution. Common tokens have energy in low bands; rare tokens have energy in high bands. Correlation r > 0.5.

---

## Apparatus

### Required Infrastructure

```python
from typing import Dict, List, Tuple
import torch
import numpy as np
from collections import Counter
from scipy.fft import fft
from scipy.stats import spearmanr, pearsonr

class ZipfSpectralAnalyzer:
    """
    Analyze relationship between Zipf rank and spectral band distribution.
    """
    
    def __init__(self, model, tokenizer, corpus_path: str = None):
        self.model = model
        self.tokenizer = tokenizer
        self.zipf_ranks = {}  # token_id -> zipf_rank
        self.token_counts = Counter()
        
        if corpus_path:
            self.build_zipf_from_corpus(corpus_path)
        else:
            self.use_default_zipf()
    
    def build_zipf_from_corpus(self, corpus_path: str):
        """Build Zipf ranks from a text corpus."""
        with open(corpus_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Tokenize
        tokens = self.tokenizer.encode(text)
        self.token_counts = Counter(tokens)
        
        # Rank by frequency (most common = rank 1)
        sorted_tokens = self.token_counts.most_common()
        for rank, (token_id, count) in enumerate(sorted_tokens, start=1):
            self.zipf_ranks[token_id] = rank
    
    def use_default_zipf(self):
        """Use tokenizer vocabulary order as proxy for Zipf rank."""
        # Most tokenizers order vocabulary by frequency
        vocab_size = self.tokenizer.vocab_size
        for token_id in range(vocab_size):
            # Lower IDs are typically more common
            self.zipf_ranks[token_id] = token_id + 1
    
    def get_embedding(self, token_id: int) -> torch.Tensor:
        """Get embedding for a single token."""
        embeddings = self.model.get_input_embeddings()
        return embeddings.weight[token_id].detach()
    
    def spectral_decomposition(
        self, 
        embedding: torch.Tensor, 
        n_bands: int = 7
    ) -> Dict[int, float]:
        """
        Decompose embedding into spectral bands.
        
        Returns energy in each band.
        """
        emb_np = embedding.numpy()
        
        # FFT
        fft_result = fft(emb_np)
        magnitudes = np.abs(fft_result)
        
        # Split into bands
        n_freq = len(magnitudes) // 2  # Positive frequencies
        band_size = n_freq // n_bands
        
        band_energies = {}
        for band in range(n_bands):
            start = band * band_size
            end = (band + 1) * band_size if band < n_bands - 1 else n_freq
            band_energies[band] = np.sum(magnitudes[start:end] ** 2)
        
        # Normalize
        total_energy = sum(band_energies.values())
        if total_energy > 0:
            band_energies = {k: v / total_energy for k, v in band_energies.items()}
        
        return band_energies
    
    def compute_band_centroid(self, band_energies: Dict[int, float]) -> float:
        """
        Compute energy-weighted centroid of band distribution.
        
        Low centroid = energy in low bands
        High centroid = energy in high bands
        """
        total_energy = sum(band_energies.values())
        if total_energy == 0:
            return 0
        
        centroid = sum(band * energy for band, energy in band_energies.items())
        centroid /= total_energy
        
        return centroid
    
    def analyze_token(self, token_id: int) -> Dict:
        """Analyze a single token's spectral distribution."""
        embedding = self.get_embedding(token_id)
        band_energies = self.spectral_decomposition(embedding)
        centroid = self.compute_band_centroid(band_energies)
        zipf_rank = self.zipf_ranks.get(token_id, 0)
        log_rank = np.log10(zipf_rank) if zipf_rank > 0 else 0
        
        token_str = self.tokenizer.decode([token_id])
        
        return {
            'token_id': token_id,
            'token_str': token_str,
            'zipf_rank': zipf_rank,
            'log_rank': log_rank,
            'band_energies': band_energies,
            'centroid': centroid
        }
    
    def analyze_vocabulary(
        self, 
        n_tokens: int = 5000,
        sample_method: str = 'stratified'
    ) -> List[Dict]:
        """
        Analyze spectral distribution across vocabulary.
        
        Args:
            n_tokens: Number of tokens to analyze
            sample_method: 'all', 'random', or 'stratified'
        
        Returns:
            List of analysis results for each token
        """
        vocab_size = self.tokenizer.vocab_size
        
        if sample_method == 'all':
            token_ids = list(range(min(n_tokens, vocab_size)))
        elif sample_method == 'random':
            token_ids = np.random.choice(vocab_size, min(n_tokens, vocab_size), replace=False)
        elif sample_method == 'stratified':
            # Sample evenly across Zipf rank ranges
            n_per_bucket = n_tokens // 10
            token_ids = []
            sorted_by_rank = sorted(self.zipf_ranks.items(), key=lambda x: x[1])
            bucket_size = len(sorted_by_rank) // 10
            for i in range(10):
                start = i * bucket_size
                end = (i + 1) * bucket_size
                bucket_tokens = [t[0] for t in sorted_by_rank[start:end]]
                sampled = np.random.choice(bucket_tokens, min(n_per_bucket, len(bucket_tokens)), replace=False)
                token_ids.extend(sampled)
        
        results = []
        for token_id in token_ids:
            try:
                result = self.analyze_token(token_id)
                results.append(result)
            except Exception as e:
                continue  # Skip problematic tokens
        
        return results
    
    def compute_correlation(self, results: List[Dict]) -> Dict:
        """
        Compute correlation between Zipf rank and spectral centroid.
        """
        log_ranks = [r['log_rank'] for r in results if r['log_rank'] > 0]
        centroids = [r['centroid'] for r in results if r['log_rank'] > 0]
        
        # Pearson correlation
        pearson_r, pearson_p = pearsonr(log_ranks, centroids)
        
        # Spearman correlation (rank-based, more robust)
        spearman_r, spearman_p = spearmanr(log_ranks, centroids)
        
        return {
            'pearson_r': pearson_r,
            'pearson_p': pearson_p,
            'spearman_r': spearman_r,
            'spearman_p': spearman_p,
            'n_samples': len(log_ranks)
        }


class BandSpecificAnalysis:
    """
    Analyze specific bands for Zipf relationship.
    """
    
    def __init__(self, analyzer: ZipfSpectralAnalyzer):
        self.analyzer = analyzer
    
    def energy_by_rank_bucket(
        self, 
        results: List[Dict],
        n_buckets: int = 10
    ) -> Dict[int, Dict[int, float]]:
        """
        Compute mean band energies for each Zipf rank bucket.
        
        Returns:
            Dict[bucket_idx, Dict[band_idx, mean_energy]]
        """
        # Sort by Zipf rank
        sorted_results = sorted(results, key=lambda x: x['zipf_rank'])
        
        bucket_size = len(sorted_results) // n_buckets
        bucket_energies = {}
        
        for bucket in range(n_buckets):
            start = bucket * bucket_size
            end = (bucket + 1) * bucket_size if bucket < n_buckets - 1 else len(sorted_results)
            bucket_results = sorted_results[start:end]
            
            # Average band energies for this bucket
            mean_energies = {band: 0 for band in range(7)}
            for r in bucket_results:
                for band, energy in r['band_energies'].items():
                    mean_energies[band] += energy
            
            for band in mean_energies:
                mean_energies[band] /= len(bucket_results)
            
            bucket_energies[bucket] = mean_energies
        
        return bucket_energies
    
    def band_rank_correlation(
        self,
        results: List[Dict]
    ) -> Dict[int, float]:
        """
        Compute correlation between Zipf rank and energy for each band separately.
        
        Returns:
            Dict[band_idx, correlation]
        """
        band_correlations = {}
        
        for band in range(7):
            log_ranks = [r['log_rank'] for r in results if r['log_rank'] > 0]
            band_energies = [r['band_energies'][band] for r in results if r['log_rank'] > 0]
            
            corr, _ = spearmanr(log_ranks, band_energies)
            band_correlations[band] = corr
        
        return band_correlations
```

---

## Protocol

### Phase 1: Vocabulary-Wide Analysis

```
VOCABULARY ANALYSIS PROTOCOL:

1. Sample tokens from vocabulary:
   - 5000 tokens, stratified by Zipf rank
   - Cover full range: most common to very rare

2. For each token:
   a. Get embedding
   b. Decompose into 7 spectral bands
   c. Compute band energy distribution
   d. Compute spectral centroid
   e. Record Zipf rank

3. Compute correlation:
   - Spearman correlation: log(Zipf rank) vs spectral centroid
   - Prediction: r > 0.5 (higher rank → higher centroid)
```

### Phase 2: Band-Specific Analysis

```
BAND-SPECIFIC PROTOCOL:

1. For each spectral band (0-6):
   a. Compute correlation: log(Zipf rank) vs band energy
   
2. Predictions:
   - Band 0-1 (low): NEGATIVE correlation (common tokens have MORE energy here)
   - Band 5-6 (high): POSITIVE correlation (rare tokens have MORE energy here)
   - Band 2-4 (mid): Weak or no correlation

3. Create heatmap:
   - Rows: Zipf rank buckets (common → rare)
   - Columns: Spectral bands (0 → 6)
   - Values: Mean energy
   
   Expected pattern: Energy shifts from low bands to high bands as rank increases
```

### Phase 3: Extreme Token Analysis

```
EXTREME ANALYSIS PROTOCOL:

1. Analyze the 100 MOST common tokens:
   - "the", "a", "is", "of", "and", etc.
   - Expect: Energy concentrated in bands 0-2

2. Analyze 100 RARE tokens:
   - Technical terms, proper nouns, unusual words
   - Expect: Energy concentrated in bands 4-6

3. Compare distributions:
   - Should be clearly separable
   - Common tokens: centroid < 2
   - Rare tokens: centroid > 4
```

### Phase 4: Cross-Model Validation

```
CROSS-MODEL PROTOCOL:

1. Run analysis on multiple models:
   - GPT-2 (small)
   - GPT-2 (large)
   - LLaMA (if accessible)
   - Different model families

2. Test universality:
   - Is the Zipf-spectral relationship consistent?
   - Do all models show similar patterns?

3. Prediction:
   - Correlation should be consistent (r > 0.4 in all models)
   - The relationship is not model-specific but language-specific
```

---

## Predictions

### If Hypothesis is Correct

```
EXPECTED RESULTS:

┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│  SPECTRAL CENTROID vs ZIPF RANK:                                           │
│                                                                             │
│  Centroid │                                          ●                      │
│           │                                       ●     ●                   │
│     6     │                                    ●                            │
│           │                                 ●                               │
│     5     │                              ●                                  │
│           │                           ●                                     │
│     4     │                        ●                                        │
│           │                     ●                                           │
│     3     │                  ●                                              │
│           │               ●                                                 │
│     2     │            ●                                                    │
│           │         ●                                                       │
│     1     │      ●                                                          │
│           │   ●                                                             │
│     0     │●                                                                │
│           └──────────────────────────────────────────────────────►          │
│             1        10       100      1000     10000    100000             │
│                           Log(Zipf Rank)                                    │
│                                                                             │
│  Prediction: Strong positive correlation (r > 0.5)                         │
│  Higher rank (rarer) → Higher centroid (more high-band energy)             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│  BAND ENERGY HEATMAP:                                                       │
│                                                                             │
│             Band 0  Band 1  Band 2  Band 3  Band 4  Band 5  Band 6         │
│  ─────────────────────────────────────────────────────────────────         │
│  Common    │ ████ │ ███  │ ██   │ █    │      │      │      │             │
│  tokens    │ HIGH │ MED  │ LOW  │ LOW  │ ZERO │ ZERO │ ZERO │             │
│  ─────────────────────────────────────────────────────────────────         │
│  Mid-freq  │ ██   │ ██   │ ██   │ ██   │ █    │ █    │      │             │
│  tokens    │ MED  │ MED  │ MED  │ MED  │ LOW  │ LOW  │ ZERO │             │
│  ─────────────────────────────────────────────────────────────────         │
│  Rare      │      │ █    │ █    │ ██   │ ███  │ ███  │ ████ │             │
│  tokens    │ ZERO │ LOW  │ LOW  │ MED  │ MED  │ HIGH │ HIGH │             │
│  ─────────────────────────────────────────────────────────────────         │
│                                                                             │
│  Energy shifts from low bands (common) to high bands (rare)                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

QUANTITATIVE PREDICTIONS:

1. Overall correlation:
   - Spearman r > 0.5 (log rank vs centroid)
   - p-value < 0.001

2. Band-specific correlations:
   - Band 0: r < -0.3 (negative: common tokens have MORE)
   - Band 6: r > +0.3 (positive: rare tokens have MORE)
   - Band 3: |r| < 0.1 (no correlation: transition zone)

3. Extreme tokens:
   - 100 most common: mean centroid < 2.5
   - 100 rarest: mean centroid > 4.0
   - Separation clearly visible

4. Cross-model consistency:
   - Correlation r > 0.4 in all tested models
   - Direction consistent (always positive)
```

### Falsification Criteria

```
FALSIFICATION:

The hypothesis is FALSIFIED if:

1. STRONG: No correlation between rank and centroid
   - |r| < 0.2
   → Zipf rank does not map to spectral structure
   → Spectral bands are not information-grounded

2. STRONG: NEGATIVE correlation (opposite of prediction)
   - r < -0.2
   → Common tokens in HIGH bands, rare in LOW
   → Completely wrong model

3. STRONG: Band energies uniform across all ranks
   - All band correlations |r| < 0.1
   → No spectral differentiation by frequency
   → Bands are not meaningful

4. MODERATE: Cross-model inconsistency
   - Some models show correlation, others don't
   → Effect is model-specific, not language-specific
   → Not a fundamental property

5. MODERATE: Only correlation in extreme tails
   - Mid-frequency tokens show no pattern
   → Effect is artificial / due to outliers
```

---

## Analysis

### Primary Metrics

```python
def analyze_zipf_spectral(results: List[Dict]) -> Dict:
    """
    Full analysis of Zipf-spectral relationship.
    """
    # 1. Overall correlation
    log_ranks = [r['log_rank'] for r in results if r['log_rank'] > 0]
    centroids = [r['centroid'] for r in results if r['log_rank'] > 0]
    
    spearman_r, spearman_p = spearmanr(log_ranks, centroids)
    
    # 2. Band-specific correlations
    band_correlations = {}
    for band in range(7):
        band_energies = [r['band_energies'][band] for r in results if r['log_rank'] > 0]
        corr, _ = spearmanr(log_ranks, band_energies)
        band_correlations[band] = corr
    
    # 3. Extreme analysis
    sorted_by_rank = sorted(results, key=lambda x: x['zipf_rank'])
    common_100 = sorted_by_rank[:100]
    rare_100 = sorted_by_rank[-100:]
    
    common_centroid = np.mean([r['centroid'] for r in common_100])
    rare_centroid = np.mean([r['centroid'] for r in rare_100])
    
    # 4. Separation test
    from scipy.stats import mannwhitneyu
    common_centroids = [r['centroid'] for r in common_100]
    rare_centroids = [r['centroid'] for r in rare_100]
    u_stat, u_pval = mannwhitneyu(common_centroids, rare_centroids, alternative='less')
    
    return {
        'overall_correlation': spearman_r,
        'overall_pvalue': spearman_p,
        'band_correlations': band_correlations,
        'common_mean_centroid': common_centroid,
        'rare_mean_centroid': rare_centroid,
        'centroid_separation_pvalue': u_pval,
        
        # Verdict
        'hypothesis_supported': (
            spearman_r > 0.4 and
            spearman_p < 0.001 and
            band_correlations[0] < -0.2 and
            band_correlations[6] > 0.2 and
            rare_centroid > common_centroid + 1.0
        )
    }
```

---

## Expected Outcomes

### If Hypothesis Validated

```
IMPLICATIONS IF TRUE:

1. THEORETICAL GROUNDING:
   Spectral bands are NOT arbitrary
   They correspond to information density tiers
   Zipf's law provides the organizing principle

2. ARCHITECTURE JUSTIFICATION:
   7-band structure captures natural information hierarchy
   Band 0-2: Structural glue (common tokens)
   Band 3-4: Bridge zone
   Band 5-6: Semantic content (rare tokens)

3. COMPRESSION GUIDANCE:
   Know what each band contains:
   - Compress low bands → lose structure (bad)
   - Compress high bands → lose specifics (maybe OK)
   - Selective compression based on task

4. TRAINING INSIGHT:
   Wave embeddings (Exp 034) should use Zipf frequencies
   Grounded in language statistics, not arbitrary
   Inductive bias matches data structure

5. GENERALIZATION:
   Low bands generalize (common patterns)
   High bands specialize (rare distinctions)
   "Grokking" = learning to use low bands for structure
```

### If Hypothesis Falsified

```
IMPLICATIONS IF FALSE:

1. IF no correlation:
   Spectral structure is arbitrary
   Bands don't correspond to information
   Need different justification for architecture

2. IF model-specific:
   Effect is architectural, not linguistic
   Different training creates different structure
   Not a fundamental property of language

3. IMPLICATIONS FOR AKIRA:
   Band structure may still be useful
   But not grounded in Zipf/information theory
   Different interpretation needed
```

---

## Connection to Other Experiments

| Experiment | Relationship |
|------------|--------------|
| 003 (Spectral Bands) | This explains WHY bands have different dynamics |
| 034 (Zipf Wave Tokenizer) | Provides Zipf-grounded embeddings to test |
| 009 (Grokking) | Grokking may = learning Zipf-aligned structure |
| 011 (Prompt Spectral) | Prompts may have Zipf-predictable spectral content |
| 025 (Synergy-Redundancy) | High-Zipf tokens may carry synergy |

---

## Results

*[ TO BE FILLED AFTER EXPERIMENT ]*

### Overall Correlation

```
[ PLACEHOLDER FOR RESULTS ]

Spearman r = ____
p-value = ____
n samples = ____
```

### Band-Specific Correlations

```
[ PLACEHOLDER FOR RESULTS ]

Band 0: r = ____
Band 1: r = ____
Band 2: r = ____
Band 3: r = ____
Band 4: r = ____
Band 5: r = ____
Band 6: r = ____
```

### Extreme Token Analysis

```
[ PLACEHOLDER FOR RESULTS ]

100 most common tokens: mean centroid = ____
100 rarest tokens: mean centroid = ____
Separation p-value = ____
```

---

## Conclusion

*[ TO BE FILLED AFTER EXPERIMENT ]*

---

## References

1. **Zipf, G.K. (1949).** *Human Behavior and the Principle of Least Effort.* — Foundation of Zipf's law.

2. **Shannon, C.E. (1948).** *A Mathematical Theory of Communication.* — Information theory.

3. **Mandelbrot, B. (1953).** *An Informational Theory of the Statistical Structure of Language.* — Information-theoretic derivation of Zipf.

4. **AKIRA Internal:** `THE_LANGUAGE_OF_INFORMATION.md` Section 2.4 — Zipf and spectral structure.

5. **AKIRA Internal:** `LANGUAGE_ACTION_CONTEXT.md` Section 10.4-10.5 — Zipf and AQ.

6. **AKIRA Internal:** `034_EXP_ZIPF_WAVE_TOKENIZER` — Zipf-grounded wave embeddings.

---



*"Language has a frequency structure given by Zipf's law. Common words are DC components; rare words are high-frequency detail. The spectrum of language is the spectrum of meaning."*

*Oscar Goldman — Shogu Research Group @ Datamutant.ai subsidiary of 温心重工業*

