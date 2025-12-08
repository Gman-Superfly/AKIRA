"""
Experiment 040: Zipf-Spectral Mapping

Tests whether token embeddings organize by Zipf rank in spectral space.

Hypothesis: Common tokens have energy in low spectral bands; 
rare tokens have energy in high spectral bands.

Usage:
    python zipf_spectral_mapping.py
"""

from typing import Dict, List, Tuple, Optional
import torch
import numpy as np
from scipy.fft import fft
from scipy.stats import spearmanr, pearsonr
from dataclasses import dataclass
from collections import Counter
import json
import matplotlib.pyplot as plt


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class ExperimentConfig:
    """Configuration for Zipf-spectral mapping experiment."""
    model_name: str = "gpt2"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    n_tokens_to_analyze: int = 3000  # Number of tokens to analyze
    n_spectral_bands: int = 7  # Match AKIRA's band structure
    sample_method: str = "stratified"  # 'all', 'random', 'stratified'


# ============================================================================
# CORE ANALYZER
# ============================================================================

class ZipfSpectralAnalyzer:
    """Analyze relationship between Zipf rank and spectral distribution."""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.zipf_ranks: Dict[int, int] = {}  # token_id -> rank
        
    def load_model(self):
        """Load model and tokenizer."""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        print(f"Loading model: {self.config.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.config.model_name)
        self.model.to(self.config.device)
        self.model.eval()
        
        print(f"Model loaded. Vocabulary size: {self.tokenizer.vocab_size}")
        
        # Use token ID as proxy for Zipf rank (common tokens have lower IDs)
        # This is approximate but reasonable for BPE tokenizers
        for token_id in range(self.tokenizer.vocab_size):
            self.zipf_ranks[token_id] = token_id + 1
        
        print(f"Zipf ranks initialized (using token ID as proxy)")
    
    def get_embedding(self, token_id: int) -> torch.Tensor:
        """Get embedding for a single token."""
        embeddings = self.model.get_input_embeddings()
        return embeddings.weight[token_id].detach().cpu()
    
    def spectral_decomposition(
        self, 
        embedding: torch.Tensor
    ) -> Dict[int, float]:
        """Decompose embedding into spectral bands."""
        emb_np = embedding.numpy()
        
        # FFT
        fft_result = fft(emb_np)
        magnitudes = np.abs(fft_result)
        
        # Split into bands
        n_freq = len(magnitudes) // 2  # Positive frequencies
        band_size = n_freq // self.config.n_spectral_bands
        
        band_energies = {}
        for band in range(self.config.n_spectral_bands):
            start = band * band_size
            end = (band + 1) * band_size if band < self.config.n_spectral_bands - 1 else n_freq
            band_energies[band] = float(np.sum(magnitudes[start:end] ** 2))
        
        # Normalize
        total_energy = sum(band_energies.values())
        if total_energy > 0:
            band_energies = {k: v / total_energy for k, v in band_energies.items()}
        
        return band_energies
    
    def compute_spectral_centroid(self, band_energies: Dict[int, float]) -> float:
        """Compute energy-weighted centroid of band distribution."""
        total = sum(band_energies.values())
        if total == 0:
            return 0
        
        centroid = sum(band * energy for band, energy in band_energies.items())
        return centroid / total
    
    def analyze_token(self, token_id: int) -> Dict:
        """Analyze a single token's spectral distribution."""
        embedding = self.get_embedding(token_id)
        band_energies = self.spectral_decomposition(embedding)
        centroid = self.compute_spectral_centroid(band_energies)
        
        zipf_rank = self.zipf_ranks.get(token_id, token_id + 1)
        log_rank = np.log10(zipf_rank) if zipf_rank > 0 else 0
        
        # Get token string (handle special tokens)
        try:
            token_str = self.tokenizer.decode([token_id])
        except:
            token_str = f"[{token_id}]"
        
        return {
            'token_id': token_id,
            'token_str': token_str,
            'zipf_rank': zipf_rank,
            'log_rank': log_rank,
            'band_energies': band_energies,
            'centroid': centroid
        }
    
    def analyze_vocabulary(self) -> List[Dict]:
        """Analyze spectral distribution across vocabulary."""
        vocab_size = self.tokenizer.vocab_size
        n_tokens = min(self.config.n_tokens_to_analyze, vocab_size)
        
        # Select tokens based on sample method
        if self.config.sample_method == 'all':
            token_ids = list(range(n_tokens))
        elif self.config.sample_method == 'random':
            token_ids = np.random.choice(vocab_size, n_tokens, replace=False).tolist()
        elif self.config.sample_method == 'stratified':
            # Sample evenly across Zipf rank ranges
            n_per_bucket = n_tokens // 10
            token_ids = []
            bucket_size = vocab_size // 10
            for i in range(10):
                start = i * bucket_size
                end = min((i + 1) * bucket_size, vocab_size)
                bucket_size_actual = end - start
                n_sample = min(n_per_bucket, bucket_size_actual)
                sampled = np.random.choice(
                    range(start, end), 
                    n_sample, 
                    replace=False
                ).tolist()
                token_ids.extend(sampled)
        
        print(f"Analyzing {len(token_ids)} tokens...")
        
        results = []
        for i, token_id in enumerate(token_ids):
            if i % 500 == 0:
                print(f"  Progress: {i}/{len(token_ids)}")
            try:
                result = self.analyze_token(token_id)
                results.append(result)
            except Exception as e:
                continue
        
        print(f"  Analyzed {len(results)} tokens successfully")
        return results
    
    def compute_correlation(self, results: List[Dict]) -> Dict:
        """Compute correlation between Zipf rank and spectral centroid."""
        log_ranks = [r['log_rank'] for r in results if r['log_rank'] > 0]
        centroids = [r['centroid'] for r in results if r['log_rank'] > 0]
        
        if len(log_ranks) < 10:
            return {'error': 'Not enough valid samples'}
        
        # Pearson correlation
        pearson_r, pearson_p = pearsonr(log_ranks, centroids)
        
        # Spearman correlation (more robust)
        spearman_r, spearman_p = spearmanr(log_ranks, centroids)
        
        return {
            'pearson_r': pearson_r,
            'pearson_p': pearson_p,
            'spearman_r': spearman_r,
            'spearman_p': spearman_p,
            'n_samples': len(log_ranks)
        }
    
    def compute_band_correlations(self, results: List[Dict]) -> Dict[int, float]:
        """Compute correlation between Zipf rank and each band's energy."""
        log_ranks = [r['log_rank'] for r in results if r['log_rank'] > 0]
        
        band_correlations = {}
        for band in range(self.config.n_spectral_bands):
            band_energies = [r['band_energies'][band] for r in results if r['log_rank'] > 0]
            corr, _ = spearmanr(log_ranks, band_energies)
            band_correlations[band] = corr
        
        return band_correlations


# ============================================================================
# EXPERIMENT RUNNER
# ============================================================================

class ExperimentRunner:
    """Run the Zipf-spectral mapping experiment."""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.analyzer = ZipfSpectralAnalyzer(config)
        self.results = {}
        
    def run(self) -> Dict:
        """Run the complete experiment."""
        print("=" * 60)
        print("EXPERIMENT 040: Zipf-Spectral Mapping")
        print("=" * 60)
        
        # Load model
        self.analyzer.load_model()
        
        # Analyze vocabulary
        print("\n--- Analyzing Vocabulary ---")
        token_results = self.analyzer.analyze_vocabulary()
        
        # Compute correlations
        print("\n--- Computing Correlations ---")
        overall_correlation = self.analyzer.compute_correlation(token_results)
        band_correlations = self.analyzer.compute_band_correlations(token_results)
        
        # Analyze extremes
        print("\n--- Analyzing Extreme Tokens ---")
        extremes = self._analyze_extremes(token_results)
        
        # Compile results
        self.results = {
            'config': {
                'model_name': self.config.model_name,
                'n_tokens': len(token_results),
                'n_bands': self.config.n_spectral_bands
            },
            'token_results': token_results,
            'overall_correlation': overall_correlation,
            'band_correlations': band_correlations,
            'extremes': extremes
        }
        
        self._print_summary()
        
        return self.results
    
    def _analyze_extremes(self, results: List[Dict]) -> Dict:
        """Analyze most common and rarest tokens."""
        sorted_by_rank = sorted(results, key=lambda x: x['zipf_rank'])
        
        # Most common (lowest ranks)
        common_100 = sorted_by_rank[:100]
        common_mean_centroid = np.mean([r['centroid'] for r in common_100])
        common_band_means = {}
        for band in range(self.config.n_spectral_bands):
            common_band_means[band] = np.mean([r['band_energies'][band] for r in common_100])
        
        # Rarest (highest ranks)
        rare_100 = sorted_by_rank[-100:]
        rare_mean_centroid = np.mean([r['centroid'] for r in rare_100])
        rare_band_means = {}
        for band in range(self.config.n_spectral_bands):
            rare_band_means[band] = np.mean([r['band_energies'][band] for r in rare_100])
        
        return {
            'common_100': {
                'mean_centroid': common_mean_centroid,
                'band_means': common_band_means,
                'examples': [(r['token_str'], r['centroid']) for r in common_100[:10]]
            },
            'rare_100': {
                'mean_centroid': rare_mean_centroid,
                'band_means': rare_band_means,
                'examples': [(r['token_str'], r['centroid']) for r in rare_100[:10]]
            }
        }
    
    def _print_summary(self):
        """Print experiment summary."""
        print("\n" + "=" * 60)
        print("EXPERIMENT SUMMARY")
        print("=" * 60)
        
        corr = self.results['overall_correlation']
        band_corr = self.results['band_correlations']
        extremes = self.results['extremes']
        
        print("\n1. OVERALL CORRELATION (Log Zipf Rank vs Spectral Centroid):")
        print(f"   Spearman r = {corr['spearman_r']:.4f} (p = {corr['spearman_p']:.2e})")
        print(f"   Pearson r = {corr['pearson_r']:.4f} (p = {corr['pearson_p']:.2e})")
        
        print("\n2. BAND-SPECIFIC CORRELATIONS:")
        for band, r in band_corr.items():
            direction = "+" if r > 0 else "-"
            print(f"   Band {band}: r = {r:+.4f} ({direction})")
        
        print("\n3. EXTREME TOKEN ANALYSIS:")
        print(f"   Common tokens (top 100): mean centroid = {extremes['common_100']['mean_centroid']:.4f}")
        print(f"   Rare tokens (bottom 100): mean centroid = {extremes['rare_100']['mean_centroid']:.4f}")
        print(f"   Separation: {extremes['rare_100']['mean_centroid'] - extremes['common_100']['mean_centroid']:.4f}")
        
        print("\n4. VERDICT:")
        
        # Check predictions
        spearman_r = corr['spearman_r']
        p_value = corr['spearman_p']
        centroid_sep = extremes['rare_100']['mean_centroid'] - extremes['common_100']['mean_centroid']
        
        # Low bands should have negative correlation (common tokens have more)
        low_band_negative = band_corr[0] < -0.1 or band_corr[1] < -0.1
        # High bands should have positive correlation (rare tokens have more)
        high_band_positive = band_corr[5] > 0.1 or band_corr[6] > 0.1
        
        supported = (
            spearman_r > 0.3 and 
            p_value < 0.001 and 
            centroid_sep > 0.5
        )
        
        if supported:
            print("   HYPOTHESIS SUPPORTED")
            print(f"   - Strong positive correlation: r = {spearman_r:.3f}")
            print(f"   - Highly significant: p < 0.001")
            print(f"   - Clear centroid separation: {centroid_sep:.3f}")
            print("   - Zipf rank maps to spectral structure")
        else:
            print("   HYPOTHESIS NOT SUPPORTED")
            if spearman_r <= 0.3:
                print(f"   - Weak correlation: r = {spearman_r:.3f}")
            if p_value >= 0.001:
                print(f"   - Not significant: p = {p_value:.4f}")
            if centroid_sep <= 0.5:
                print(f"   - Small centroid separation: {centroid_sep:.3f}")
    
    def plot_results(self, save_path: str = None):
        """Plot experiment results."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        results = self.results['token_results']
        
        # Plot 1: Scatter of log rank vs centroid
        ax1 = axes[0, 0]
        log_ranks = [r['log_rank'] for r in results]
        centroids = [r['centroid'] for r in results]
        
        ax1.scatter(log_ranks, centroids, alpha=0.3, s=10)
        ax1.set_xlabel('Log10(Zipf Rank)', fontsize=12)
        ax1.set_ylabel('Spectral Centroid', fontsize=12)
        ax1.set_title('Zipf Rank vs Spectral Centroid', fontsize=14)
        
        # Add trend line
        z = np.polyfit(log_ranks, centroids, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(log_ranks), max(log_ranks), 100)
        ax1.plot(x_line, p(x_line), 'r-', linewidth=2, 
                 label=f'Trend (r={self.results["overall_correlation"]["spearman_r"]:.3f})')
        ax1.legend()
        
        # Plot 2: Band correlations
        ax2 = axes[0, 1]
        band_corr = self.results['band_correlations']
        bands = list(band_corr.keys())
        corrs = list(band_corr.values())
        colors = ['red' if c < 0 else 'green' for c in corrs]
        
        ax2.bar(bands, corrs, color=colors, alpha=0.7)
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Spectral Band', fontsize=12)
        ax2.set_ylabel('Correlation with Log(Zipf Rank)', fontsize=12)
        ax2.set_title('Band-Specific Correlations', fontsize=14)
        
        # Plot 3: Heatmap of band energy by rank bucket
        ax3 = axes[1, 0]
        sorted_results = sorted(results, key=lambda x: x['zipf_rank'])
        n_buckets = 10
        bucket_size = len(sorted_results) // n_buckets
        
        heatmap_data = []
        for i in range(n_buckets):
            start = i * bucket_size
            end = (i + 1) * bucket_size if i < n_buckets - 1 else len(sorted_results)
            bucket = sorted_results[start:end]
            
            band_means = []
            for band in range(self.config.n_spectral_bands):
                mean_energy = np.mean([r['band_energies'][band] for r in bucket])
                band_means.append(mean_energy)
            heatmap_data.append(band_means)
        
        heatmap_data = np.array(heatmap_data)
        im = ax3.imshow(heatmap_data, aspect='auto', cmap='viridis')
        ax3.set_xlabel('Spectral Band', fontsize=12)
        ax3.set_ylabel('Zipf Rank Bucket (Common -> Rare)', fontsize=12)
        ax3.set_title('Band Energy by Zipf Bucket', fontsize=14)
        ax3.set_xticks(range(self.config.n_spectral_bands))
        ax3.set_yticks(range(n_buckets))
        ax3.set_yticklabels([f'{i+1}' for i in range(n_buckets)])
        plt.colorbar(im, ax=ax3, label='Energy')
        
        # Plot 4: Centroid distribution by rank bucket
        ax4 = axes[1, 1]
        bucket_centroids = []
        for i in range(n_buckets):
            start = i * bucket_size
            end = (i + 1) * bucket_size if i < n_buckets - 1 else len(sorted_results)
            bucket = sorted_results[start:end]
            bucket_centroids.append([r['centroid'] for r in bucket])
        
        ax4.boxplot(bucket_centroids, labels=[f'{i+1}' for i in range(n_buckets)])
        ax4.set_xlabel('Zipf Rank Bucket (Common -> Rare)', fontsize=12)
        ax4.set_ylabel('Spectral Centroid', fontsize=12)
        ax4.set_title('Centroid Distribution by Rank Bucket', fontsize=14)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"Plot saved to {save_path}")
        
        plt.show()
    
    def save_results(self, path: str):
        """Save results to JSON."""
        # Create serializable version
        serializable = {
            'config': self.results['config'],
            'overall_correlation': self.results['overall_correlation'],
            'band_correlations': {str(k): v for k, v in self.results['band_correlations'].items()},
            'extremes': self.results['extremes']
        }
        
        with open(path, 'w') as f:
            json.dump(serializable, f, indent=2, default=str)
        print(f"Results saved to {path}")


# ============================================================================
# MAIN
# ============================================================================

def run_experiment(model_name: str = "gpt2") -> Dict:
    """Run the complete experiment."""
    config = ExperimentConfig(model_name=model_name)
    runner = ExperimentRunner(config)
    results = runner.run()
    runner.plot_results("040_zipf_spectral_plot.png")
    return results


if __name__ == "__main__":
    results = run_experiment("gpt2")
