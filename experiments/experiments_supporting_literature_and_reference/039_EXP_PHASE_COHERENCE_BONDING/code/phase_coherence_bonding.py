"""
Experiment 039: Phase Coherence Bonding

Tests whether semantically coherent concepts show phase alignment 
while conflicting concepts show phase opposition.

Hypothesis: AQ combine via superposition; phase alignment determines 
constructive vs destructive interference.

Usage:
    python phase_coherence_bonding.py
"""

from typing import Dict, List, Tuple
import torch
import numpy as np
from scipy.fft import fft
from scipy.stats import spearmanr, ttest_ind
from dataclasses import dataclass
import json
import matplotlib.pyplot as plt


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class ExperimentConfig:
    """Configuration for phase coherence experiment."""
    model_name: str = "gpt2"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    n_frequency_components: int = 100  # Top frequency components to analyze


# ============================================================================
# WORD PAIRS
# ============================================================================

COHERENT_PAIRS = [
    ("hot", "fire"),
    ("cold", "ice"),
    ("happy", "joy"),
    ("sad", "tears"),
    ("fast", "speed"),
    ("slow", "turtle"),
    ("big", "giant"),
    ("small", "tiny"),
    ("cat", "meow"),
    ("dog", "bark"),
    ("sun", "bright"),
    ("moon", "night"),
    ("water", "wet"),
    ("desert", "dry"),
    ("doctor", "hospital"),
    ("teacher", "school"),
    ("king", "throne"),
    ("ocean", "waves"),
    ("mountain", "peak"),
    ("forest", "trees"),
]

CONFLICTING_PAIRS = [
    ("hot", "cold"),
    ("big", "small"),
    ("fast", "slow"),
    ("happy", "sad"),
    ("true", "false"),
    ("up", "down"),
    ("left", "right"),
    ("good", "bad"),
    ("light", "dark"),
    ("alive", "dead"),
    ("open", "closed"),
    ("full", "empty"),
    ("wet", "dry"),
    ("young", "old"),
    ("love", "hate"),
    ("peace", "war"),
    ("success", "failure"),
    ("rich", "poor"),
    ("strong", "weak"),
    ("early", "late"),
]

NEUTRAL_PAIRS = [
    ("cat", "telephone"),
    ("mountain", "keyboard"),
    ("happy", "purple"),
    ("doctor", "banana"),
    ("ocean", "pencil"),
    ("king", "sandwich"),
    ("tree", "algorithm"),
    ("sun", "carpet"),
    ("music", "brick"),
    ("paper", "gravity"),
    ("book", "volcano"),
    ("chair", "dream"),
    ("window", "philosophy"),
    ("garden", "mathematics"),
    ("bridge", "poetry"),
    ("clock", "elephant"),
    ("mirror", "thunder"),
    ("ladder", "symphony"),
    ("candle", "democracy"),
    ("blanket", "equation"),
]


# ============================================================================
# CORE ANALYZER
# ============================================================================

class PhaseCoherenceAnalyzer:
    """Analyze phase relationships in embeddings."""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        
    def load_model(self):
        """Load model and tokenizer."""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        print(f"Loading model: {self.config.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.config.model_name)
        self.model.to(self.config.device)
        self.model.eval()
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print(f"Model loaded on {self.config.device}")
    
    def get_embedding(self, text: str) -> torch.Tensor:
        """Get embedding for a word or phrase."""
        inputs = self.tokenizer(text, return_tensors="pt").to(self.config.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
        
        # Get last hidden state, average over tokens
        hidden = outputs.hidden_states[-1]  # [1, seq_len, hidden_dim]
        return hidden.mean(dim=1).squeeze().cpu()  # [hidden_dim]
    
    def decompose_to_magnitude_phase(
        self, 
        embedding: torch.Tensor
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Decompose embedding into magnitude and phase via FFT."""
        emb_np = embedding.numpy()
        
        # FFT decomposition
        fft_result = fft(emb_np)
        
        magnitude = np.abs(fft_result)
        phase = np.angle(fft_result)
        
        return magnitude, phase
    
    def compute_phase_difference(
        self,
        embedding_1: torch.Tensor,
        embedding_2: torch.Tensor
    ) -> np.ndarray:
        """Compute phase difference between two embeddings."""
        _, phase_1 = self.decompose_to_magnitude_phase(embedding_1)
        _, phase_2 = self.decompose_to_magnitude_phase(embedding_2)
        
        # Phase difference (wrapped to [-pi, pi])
        delta_phase = phase_1 - phase_2
        delta_phase = np.arctan2(np.sin(delta_phase), np.cos(delta_phase))
        
        return delta_phase
    
    def compute_coherence_score(
        self,
        embedding_1: torch.Tensor,
        embedding_2: torch.Tensor
    ) -> float:
        """
        Compute phase coherence between two embeddings.
        
        Returns value from -1 (opposing) to +1 (aligned).
        """
        delta_phase = self.compute_phase_difference(embedding_1, embedding_2)
        
        # Use top frequency components
        n_components = min(self.config.n_frequency_components, len(delta_phase) // 2)
        
        # Coherence = mean of cos(delta_phase) for top components
        coherence = np.cos(delta_phase[:n_components]).mean()
        
        return float(coherence)
    
    def compute_combination_energy(
        self,
        embedding_1: torch.Tensor,
        embedding_2: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compute energy of individual embeddings vs their combination.
        """
        # Individual energies
        E_1 = float((embedding_1 ** 2).sum())
        E_2 = float((embedding_2 ** 2).sum())
        
        # Combined embedding (superposition)
        combined = embedding_1 + embedding_2
        E_combined = float((combined ** 2).sum())
        
        # Expected energy if no interference
        E_expected = E_1 + E_2
        
        # Interference ratio
        interference_ratio = E_combined / E_expected if E_expected > 0 else 1.0
        
        # Classify
        if interference_ratio > 1.05:
            interference_type = "constructive"
        elif interference_ratio < 0.95:
            interference_type = "destructive"
        else:
            interference_type = "neutral"
        
        return {
            'E_1': E_1,
            'E_2': E_2,
            'E_combined': E_combined,
            'E_expected': E_expected,
            'interference_ratio': interference_ratio,
            'interference_type': interference_type
        }
    
    def analyze_pair(self, word_1: str, word_2: str) -> Dict:
        """Full analysis of a word pair."""
        emb_1 = self.get_embedding(word_1)
        emb_2 = self.get_embedding(word_2)
        
        coherence = self.compute_coherence_score(emb_1, emb_2)
        energy = self.compute_combination_energy(emb_1, emb_2)
        
        # Cosine similarity (for comparison)
        cos_sim = float(torch.nn.functional.cosine_similarity(
            emb_1.unsqueeze(0), emb_2.unsqueeze(0)
        ))
        
        return {
            'word_1': word_1,
            'word_2': word_2,
            'phase_coherence': coherence,
            'interference_ratio': energy['interference_ratio'],
            'interference_type': energy['interference_type'],
            'cosine_similarity': cos_sim,
            'E_combined': energy['E_combined'],
            'E_expected': energy['E_expected']
        }


# ============================================================================
# EXPERIMENT RUNNER
# ============================================================================

class ExperimentRunner:
    """Run the phase coherence bonding experiment."""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.analyzer = PhaseCoherenceAnalyzer(config)
        self.results = {}
        
    def run(self) -> Dict:
        """Run the complete experiment."""
        print("=" * 60)
        print("EXPERIMENT 039: Phase Coherence Bonding")
        print("=" * 60)
        
        # Load model
        self.analyzer.load_model()
        
        # Analyze each pair type
        print("\n--- Analyzing Coherent Pairs ---")
        coherent_results = self._analyze_pairs(COHERENT_PAIRS, "coherent")
        
        print("\n--- Analyzing Conflicting Pairs ---")
        conflicting_results = self._analyze_pairs(CONFLICTING_PAIRS, "conflicting")
        
        print("\n--- Analyzing Neutral Pairs ---")
        neutral_results = self._analyze_pairs(NEUTRAL_PAIRS, "neutral")
        
        # Statistical analysis
        print("\n--- Statistical Analysis ---")
        stats = self._compute_statistics(
            coherent_results, conflicting_results, neutral_results
        )
        
        # Compile results
        self.results = {
            'coherent': coherent_results,
            'conflicting': conflicting_results,
            'neutral': neutral_results,
            'statistics': stats
        }
        
        self._print_summary()
        
        return self.results
    
    def _analyze_pairs(self, pairs: List[Tuple[str, str]], pair_type: str) -> List[Dict]:
        """Analyze a list of word pairs."""
        results = []
        
        for w1, w2 in pairs:
            result = self.analyzer.analyze_pair(w1, w2)
            result['pair_type'] = pair_type
            results.append(result)
            
            print(f"  {w1:12} + {w2:12} | "
                  f"coherence={result['phase_coherence']:+.3f} | "
                  f"ratio={result['interference_ratio']:.3f}")
        
        return results
    
    def _compute_statistics(
        self,
        coherent: List[Dict],
        conflicting: List[Dict],
        neutral: List[Dict]
    ) -> Dict:
        """Compute statistical comparisons."""
        
        # Extract metrics
        coh_coherences = [r['phase_coherence'] for r in coherent]
        con_coherences = [r['phase_coherence'] for r in conflicting]
        neu_coherences = [r['phase_coherence'] for r in neutral]
        
        coh_ratios = [r['interference_ratio'] for r in coherent]
        con_ratios = [r['interference_ratio'] for r in conflicting]
        neu_ratios = [r['interference_ratio'] for r in neutral]
        
        # Means
        coh_mean_coherence = np.mean(coh_coherences)
        con_mean_coherence = np.mean(con_coherences)
        neu_mean_coherence = np.mean(neu_coherences)
        
        coh_mean_ratio = np.mean(coh_ratios)
        con_mean_ratio = np.mean(con_ratios)
        neu_mean_ratio = np.mean(neu_ratios)
        
        # t-tests
        t_coherence, p_coherence = ttest_ind(coh_coherences, con_coherences)
        t_ratio, p_ratio = ttest_ind(coh_ratios, con_ratios)
        
        # Correlation between coherence and ratio
        all_coherences = coh_coherences + con_coherences + neu_coherences
        all_ratios = coh_ratios + con_ratios + neu_ratios
        correlation, corr_p = spearmanr(all_coherences, all_ratios)
        
        return {
            'coherent': {
                'mean_phase_coherence': coh_mean_coherence,
                'std_phase_coherence': np.std(coh_coherences),
                'mean_interference_ratio': coh_mean_ratio,
                'std_interference_ratio': np.std(coh_ratios)
            },
            'conflicting': {
                'mean_phase_coherence': con_mean_coherence,
                'std_phase_coherence': np.std(con_coherences),
                'mean_interference_ratio': con_mean_ratio,
                'std_interference_ratio': np.std(con_ratios)
            },
            'neutral': {
                'mean_phase_coherence': neu_mean_coherence,
                'std_phase_coherence': np.std(neu_coherences),
                'mean_interference_ratio': neu_mean_ratio,
                'std_interference_ratio': np.std(neu_ratios)
            },
            'coherence_ttest': {'t': t_coherence, 'p': p_coherence},
            'ratio_ttest': {'t': t_ratio, 'p': p_ratio},
            'coherence_ratio_correlation': {'r': correlation, 'p': corr_p},
            'coherence_separation': coh_mean_coherence - con_mean_coherence,
            'ratio_separation': coh_mean_ratio - con_mean_ratio
        }
    
    def _print_summary(self):
        """Print experiment summary."""
        stats = self.results['statistics']
        
        print("\n" + "=" * 60)
        print("EXPERIMENT SUMMARY")
        print("=" * 60)
        
        print("\n1. PHASE COHERENCE BY PAIR TYPE:")
        print(f"   Coherent pairs:   {stats['coherent']['mean_phase_coherence']:+.4f} "
              f"(+/- {stats['coherent']['std_phase_coherence']:.4f})")
        print(f"   Conflicting pairs: {stats['conflicting']['mean_phase_coherence']:+.4f} "
              f"(+/- {stats['conflicting']['std_phase_coherence']:.4f})")
        print(f"   Neutral pairs:    {stats['neutral']['mean_phase_coherence']:+.4f} "
              f"(+/- {stats['neutral']['std_phase_coherence']:.4f})")
        print(f"   Separation: {stats['coherence_separation']:.4f}")
        
        print("\n2. INTERFERENCE RATIO BY PAIR TYPE:")
        print(f"   Coherent pairs:   {stats['coherent']['mean_interference_ratio']:.4f}")
        print(f"   Conflicting pairs: {stats['conflicting']['mean_interference_ratio']:.4f}")
        print(f"   Neutral pairs:    {stats['neutral']['mean_interference_ratio']:.4f}")
        
        print("\n3. STATISTICAL TESTS:")
        print(f"   Coherence t-test: t={stats['coherence_ttest']['t']:.2f}, "
              f"p={stats['coherence_ttest']['p']:.4f}")
        print(f"   Ratio t-test: t={stats['ratio_ttest']['t']:.2f}, "
              f"p={stats['ratio_ttest']['p']:.4f}")
        print(f"   Coherence-Ratio correlation: r={stats['coherence_ratio_correlation']['r']:.3f}")
        
        print("\n4. VERDICT:")
        
        # Check predictions
        coherence_sep = stats['coherence_separation']
        p_coherence = stats['coherence_ttest']['p']
        correlation = stats['coherence_ratio_correlation']['r']
        
        supported = (
            coherence_sep > 0.1 and 
            p_coherence < 0.05 and 
            correlation > 0.2
        )
        
        if supported:
            print("   HYPOTHESIS SUPPORTED")
            print("   - Coherent pairs show higher phase coherence")
            print("   - Difference is statistically significant")
            print("   - Coherence correlates with interference ratio")
        else:
            print("   HYPOTHESIS NOT SUPPORTED")
            if coherence_sep <= 0.1:
                print(f"   - Coherence separation too small: {coherence_sep:.3f}")
            if p_coherence >= 0.05:
                print(f"   - Not statistically significant: p={p_coherence:.4f}")
            if correlation <= 0.2:
                print(f"   - Weak correlation: r={correlation:.3f}")
    
    def plot_results(self, save_path: str = None):
        """Plot experiment results."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        stats = self.results['statistics']
        
        # Plot 1: Phase coherence by pair type
        ax1 = axes[0]
        types = ['Coherent', 'Conflicting', 'Neutral']
        means = [
            stats['coherent']['mean_phase_coherence'],
            stats['conflicting']['mean_phase_coherence'],
            stats['neutral']['mean_phase_coherence']
        ]
        stds = [
            stats['coherent']['std_phase_coherence'],
            stats['conflicting']['std_phase_coherence'],
            stats['neutral']['std_phase_coherence']
        ]
        colors = ['green', 'red', 'gray']
        
        ax1.bar(types, means, yerr=stds, color=colors, alpha=0.7, capsize=5)
        ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax1.set_ylabel('Phase Coherence')
        ax1.set_title('Phase Coherence by Pair Type')
        ax1.set_ylim(-0.5, 0.5)
        
        # Plot 2: Interference ratio by pair type
        ax2 = axes[1]
        ratios = [
            stats['coherent']['mean_interference_ratio'],
            stats['conflicting']['mean_interference_ratio'],
            stats['neutral']['mean_interference_ratio']
        ]
        ratio_stds = [
            stats['coherent']['std_interference_ratio'],
            stats['conflicting']['std_interference_ratio'],
            stats['neutral']['std_interference_ratio']
        ]
        
        ax2.bar(types, ratios, yerr=ratio_stds, color=colors, alpha=0.7, capsize=5)
        ax2.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label='No interference')
        ax2.set_ylabel('Interference Ratio')
        ax2.set_title('Interference Ratio by Pair Type')
        ax2.legend()
        
        # Plot 3: Scatter of coherence vs ratio
        ax3 = axes[2]
        
        for pair_type, color, label in [
            ('coherent', 'green', 'Coherent'),
            ('conflicting', 'red', 'Conflicting'),
            ('neutral', 'gray', 'Neutral')
        ]:
            data = self.results[pair_type]
            coherences = [r['phase_coherence'] for r in data]
            ratios = [r['interference_ratio'] for r in data]
            ax3.scatter(coherences, ratios, c=color, label=label, alpha=0.6, s=60)
        
        ax3.axhline(y=1.0, color='black', linestyle='--', alpha=0.3)
        ax3.axvline(x=0, color='black', linestyle='--', alpha=0.3)
        ax3.set_xlabel('Phase Coherence')
        ax3.set_ylabel('Interference Ratio')
        ax3.set_title('Coherence vs Interference')
        ax3.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"Plot saved to {save_path}")
        
        plt.show()
    
    def save_results(self, path: str):
        """Save results to JSON."""
        with open(path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"Results saved to {path}")


# ============================================================================
# MAIN
# ============================================================================

def run_experiment(model_name: str = "gpt2") -> Dict:
    """Run the complete experiment."""
    config = ExperimentConfig(model_name=model_name)
    runner = ExperimentRunner(config)
    results = runner.run()
    runner.plot_results("039_phase_coherence_plot.png")
    return results


if __name__ == "__main__":
    results = run_experiment("gpt2")
