"""
EXPERIMENT 000 - PHASE 5: Spectral Analysis of Action Quanta
=============================================================

AKIRA Project - Foundational Experiment
Oscar Goldman - Shogu Research Group @ Datamutant.ai

QUESTION: What frequency band do Action Quanta live in?

AKIRA PREDICTION:
- Universal features should concentrate in LOW frequency bands
- Model-specific features should be in HIGH frequency bands
- The 78 universal AQ should be predominantly low-frequency

METHOD:
1. Extract activations across sequence positions
2. Apply FFT to each neuron's activation pattern
3. Compute spectral centroid (dominant frequency)
4. Compare: Universal AQ vs non-AQ neurons

TO RUN IN GOOGLE COLAB:
-----------------------
1. Run after 001_action_quanta_extraction_aligned.py
2. Requires results_random and results_text from previous run
3. Or run standalone with fresh extraction
"""

import torch
import numpy as np
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Dict, List, Tuple, Set
import warnings
warnings.filterwarnings('ignore')

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


# ==============================================================================
# CONFIGURATION
# ==============================================================================

class Config:
    MODEL_A = "gpt2"
    MODEL_B = "gpt2-medium"
    LAYERS_TO_ANALYZE = [3, 5, 7]
    NUM_SAMPLES = 500
    SEQ_LENGTH = 64
    BATCH_SIZE = 16
    SEED = 42
    
    # Spectral bands (following AKIRA's 7-band structure conceptually)
    # We'll divide the frequency spectrum into bands
    NUM_BANDS = 7


# ==============================================================================
# SPECTRAL ANALYSIS FUNCTIONS
# ==============================================================================

def compute_neuron_spectrum(activations: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute frequency spectrum for each neuron.
    
    Args:
        activations: [num_samples, seq_len, num_neurons]
    
    Returns:
        spectra: [num_neurons, num_freqs] power spectrum per neuron
        freqs: [num_freqs] frequency bins
    """
    num_samples, seq_len, num_neurons = activations.shape
    
    # Flatten samples: [num_samples * seq_len, num_neurons] won't work
    # Instead, compute FFT along sequence dimension for each sample, then average
    
    # FFT along sequence dimension
    # Shape: [num_samples, seq_len, num_neurons] -> FFT -> [num_samples, seq_len, num_neurons]
    
    spectra_list = []
    
    for sample_idx in range(num_samples):
        sample_acts = activations[sample_idx]  # [seq_len, num_neurons]
        
        # FFT along sequence (time) dimension for each neuron
        fft_result = fft(sample_acts, axis=0)  # [seq_len, num_neurons]
        
        # Power spectrum (magnitude squared)
        power = np.abs(fft_result) ** 2
        
        # Only keep positive frequencies (first half)
        power = power[:seq_len // 2]
        
        spectra_list.append(power)
    
    # Average across samples
    spectra = np.mean(spectra_list, axis=0)  # [seq_len//2, num_neurons]
    
    # Transpose to [num_neurons, num_freqs]
    spectra = spectra.T
    
    # Frequency bins
    freqs = fftfreq(seq_len, d=1.0)[:seq_len // 2]
    
    return spectra, freqs


def compute_spectral_centroid(spectra: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    """
    Compute spectral centroid for each neuron.
    
    Spectral centroid = weighted average of frequencies by power
    Lower centroid = more low-frequency dominant
    Higher centroid = more high-frequency dominant
    
    Args:
        spectra: [num_neurons, num_freqs]
        freqs: [num_freqs]
    
    Returns:
        centroids: [num_neurons] spectral centroid per neuron
    """
    # Normalize to get rid of negative frequencies in calculation
    freqs_pos = np.abs(freqs)
    
    # Weighted average: sum(freq * power) / sum(power)
    total_power = np.sum(spectra, axis=1, keepdims=True)
    total_power = np.maximum(total_power, 1e-10)  # Avoid division by zero
    
    centroids = np.sum(spectra * freqs_pos, axis=1) / total_power.squeeze()
    
    return centroids


def assign_frequency_bands(centroids: np.ndarray, num_bands: int = 7) -> np.ndarray:
    """
    Assign each neuron to a frequency band based on its spectral centroid.
    
    Args:
        centroids: [num_neurons] spectral centroid values
        num_bands: Number of bands to divide spectrum into
    
    Returns:
        bands: [num_neurons] band assignment (0 = lowest freq, num_bands-1 = highest)
    """
    # Use percentiles to create equal-sized bands
    percentiles = np.linspace(0, 100, num_bands + 1)
    thresholds = np.percentile(centroids, percentiles)
    
    bands = np.digitize(centroids, thresholds[1:-1])
    
    return bands


def compute_band_distribution(
    bands: np.ndarray,
    aq_indices: Set[int],
    num_bands: int = 7
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute distribution of neurons across frequency bands.
    
    Args:
        bands: [num_neurons] band assignment
        aq_indices: Set of AQ neuron indices
        num_bands: Number of bands
    
    Returns:
        aq_dist: [num_bands] distribution of AQ across bands
        non_aq_dist: [num_bands] distribution of non-AQ across bands
    """
    all_indices = set(range(len(bands)))
    non_aq_indices = all_indices - aq_indices
    
    aq_bands = bands[list(aq_indices)] if aq_indices else np.array([])
    non_aq_bands = bands[list(non_aq_indices)]
    
    aq_dist = np.zeros(num_bands)
    non_aq_dist = np.zeros(num_bands)
    
    for b in range(num_bands):
        if len(aq_bands) > 0:
            aq_dist[b] = np.sum(aq_bands == b) / len(aq_bands)
        non_aq_dist[b] = np.sum(non_aq_bands == b) / len(non_aq_bands)
    
    return aq_dist, non_aq_dist


# ==============================================================================
# VISUALIZATION
# ==============================================================================

def plot_spectral_analysis(
    centroids: np.ndarray,
    bands: np.ndarray,
    aq_indices: Set[int],
    layer: int,
    aq_type: str = "Universal"
):
    """Plot spectral analysis results."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Spectral Analysis: Layer {layer} - {aq_type} AQ', fontsize=14, fontweight='bold')
    
    all_indices = set(range(len(centroids)))
    non_aq_indices = list(all_indices - aq_indices)
    aq_indices_list = list(aq_indices)
    
    # 1. Histogram of spectral centroids
    ax1 = axes[0, 0]
    ax1.hist(centroids[non_aq_indices], bins=50, alpha=0.5, label='Non-AQ', color='gray')
    if aq_indices_list:
        ax1.hist(centroids[aq_indices_list], bins=50, alpha=0.7, label=f'{aq_type} AQ', color='green')
    ax1.set_xlabel('Spectral Centroid (lower = more low-freq)')
    ax1.set_ylabel('Count')
    ax1.set_title('Distribution of Spectral Centroids')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Band distribution comparison
    ax2 = axes[0, 1]
    aq_dist, non_aq_dist = compute_band_distribution(bands, aq_indices, Config.NUM_BANDS)
    
    x = np.arange(Config.NUM_BANDS)
    width = 0.35
    ax2.bar(x - width/2, non_aq_dist, width, label='Non-AQ', color='gray', alpha=0.7)
    ax2.bar(x + width/2, aq_dist, width, label=f'{aq_type} AQ', color='green', alpha=0.7)
    ax2.set_xlabel('Frequency Band (0=lowest, 6=highest)')
    ax2.set_ylabel('Proportion')
    ax2.set_title('Distribution Across Frequency Bands')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'Band {i}' for i in range(Config.NUM_BANDS)])
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Scatter: neuron index vs centroid
    ax3 = axes[1, 0]
    ax3.scatter(non_aq_indices, centroids[non_aq_indices], alpha=0.3, s=5, c='gray', label='Non-AQ')
    if aq_indices_list:
        ax3.scatter(aq_indices_list, centroids[aq_indices_list], alpha=0.7, s=20, c='green', label=f'{aq_type} AQ')
    ax3.set_xlabel('Neuron Index')
    ax3.set_ylabel('Spectral Centroid')
    ax3.set_title(f'{aq_type} AQ Highlighted')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Summary statistics
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    aq_centroids = centroids[aq_indices_list] if aq_indices_list else np.array([0])
    non_aq_centroids = centroids[non_aq_indices]
    
    # Compute low-freq concentration (bands 0-2)
    low_freq_aq = np.sum(aq_dist[:3]) if len(aq_dist) > 2 else 0
    low_freq_non_aq = np.sum(non_aq_dist[:3]) if len(non_aq_dist) > 2 else 0
    
    stats_text = f"""
    LAYER {layer} SPECTRAL ANALYSIS
    ================================
    
    {aq_type} AQ neurons: {len(aq_indices_list)}
    Non-AQ neurons: {len(non_aq_indices)}
    
    SPECTRAL CENTROID (lower = more low-freq):
    ------------------------------------------
    {aq_type} AQ mean:  {np.mean(aq_centroids):.4f}
    Non-AQ mean:        {np.mean(non_aq_centroids):.4f}
    Difference:         {np.mean(aq_centroids) - np.mean(non_aq_centroids):.4f}
    
    LOW-FREQUENCY CONCENTRATION (Bands 0-2):
    ----------------------------------------
    {aq_type} AQ:  {100*low_freq_aq:.1f}%
    Non-AQ:       {100*low_freq_non_aq:.1f}%
    
    AKIRA PREDICTION:
    -----------------
    Universal AQ should be MORE low-frequency
    {'CONFIRMED' if low_freq_aq > low_freq_non_aq else 'NOT CONFIRMED'}
    """
    
    ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.show()
    
    return aq_dist, non_aq_dist


# ==============================================================================
# MAIN ANALYSIS
# ==============================================================================

def run_spectral_analysis(
    acts_a: Dict[int, np.ndarray],
    universal_aq: Dict[int, Set[int]],
    structural_aq: Dict[int, Set[int]] = None,
    semantic_aq: Dict[int, Set[int]] = None
):
    """
    Run spectral analysis on extracted activations.
    
    Args:
        acts_a: Activations from model A {layer: [samples, seq, dim]}
        universal_aq: Universal AQ indices {layer: set of indices}
        structural_aq: Structural-only AQ indices (optional)
        semantic_aq: Semantic-only AQ indices (optional)
    """
    print("=" * 70)
    print("PHASE 5: SPECTRAL ANALYSIS OF ACTION QUANTA")
    print("=" * 70)
    print("\nAKIRA PREDICTION: Universal AQ should concentrate in LOW frequency bands")
    print()
    
    results = {}
    
    for layer in Config.LAYERS_TO_ANALYZE:
        if layer not in acts_a:
            continue
            
        print(f"\n{'='*50}")
        print(f"ANALYZING LAYER {layer}")
        print(f"{'='*50}")
        
        activations = acts_a[layer]
        print(f"  Activations shape: {activations.shape}")
        
        # Compute spectra
        print("  Computing frequency spectra...")
        spectra, freqs = compute_neuron_spectrum(activations)
        print(f"  Spectra shape: {spectra.shape}")
        
        # Compute spectral centroids
        print("  Computing spectral centroids...")
        centroids = compute_spectral_centroid(spectra, freqs)
        
        # Assign frequency bands
        bands = assign_frequency_bands(centroids, Config.NUM_BANDS)
        
        # Analyze Universal AQ
        if layer in universal_aq and universal_aq[layer]:
            print(f"\n  Analyzing {len(universal_aq[layer])} Universal AQ...")
            aq_dist, non_aq_dist = plot_spectral_analysis(
                centroids, bands, universal_aq[layer], layer, "Universal"
            )
            
            # Statistical test
            aq_centroids = centroids[list(universal_aq[layer])]
            non_aq_indices = list(set(range(len(centroids))) - universal_aq[layer])
            non_aq_centroids = centroids[non_aq_indices]
            
            print(f"\n  RESULTS:")
            print(f"    Universal AQ mean centroid: {np.mean(aq_centroids):.4f}")
            print(f"    Non-AQ mean centroid: {np.mean(non_aq_centroids):.4f}")
            
            low_freq_aq = np.sum(aq_dist[:3])
            low_freq_non_aq = np.sum(non_aq_dist[:3])
            print(f"    Universal AQ in low-freq bands (0-2): {100*low_freq_aq:.1f}%")
            print(f"    Non-AQ in low-freq bands (0-2): {100*low_freq_non_aq:.1f}%")
            
            if low_freq_aq > low_freq_non_aq:
                print(f"    -> AKIRA PREDICTION CONFIRMED: AQ are more low-frequency")
            else:
                print(f"    -> AKIRA prediction not confirmed at this layer")
        
        results[layer] = {
            'centroids': centroids,
            'bands': bands,
            'spectra': spectra,
            'freqs': freqs
        }
    
    return results


def run_standalone_spectral_analysis():
    """
    Run complete spectral analysis from scratch.
    """
    print("=" * 70)
    print("SPECTRAL ANALYSIS - STANDALONE MODE")
    print("=" * 70)
    print("\nThis will extract activations and compute spectral properties.")
    print("For best results, run after 001_action_quanta_extraction_aligned.py")
    print("and pass the universal_aq indices from that analysis.")
    print()
    
    torch.manual_seed(Config.SEED)
    np.random.seed(Config.SEED)
    
    from transformer_lens import HookedTransformer
    
    print(f"Loading {Config.MODEL_A}...")
    model_a = HookedTransformer.from_pretrained(Config.MODEL_A, device=device)
    
    # Generate random tokens
    print(f"Generating {Config.NUM_SAMPLES} random token sequences...")
    vocab_size = model_a.tokenizer.vocab_size
    tokens = torch.randint(1000, vocab_size - 1000, (Config.NUM_SAMPLES, Config.SEQ_LENGTH))
    
    # Extract activations
    print("Extracting activations...")
    activations = {}
    num_batches = (len(tokens) + Config.BATCH_SIZE - 1) // Config.BATCH_SIZE
    
    for layer in Config.LAYERS_TO_ANALYZE:
        activations[layer] = []
    
    for batch_idx in tqdm(range(num_batches), desc="Extracting"):
        start = batch_idx * Config.BATCH_SIZE
        end = min(start + Config.BATCH_SIZE, len(tokens))
        batch_tokens = tokens[start:end].to(device)
        
        with torch.no_grad():
            _, cache = model_a.run_with_cache(batch_tokens)
        
        for layer in Config.LAYERS_TO_ANALYZE:
            hook_name = f"blocks.{layer}.mlp.hook_post"
            if hook_name in cache:
                activations[layer].append(cache[hook_name].cpu().numpy())
        
        del cache
        torch.cuda.empty_cache() if device == "cuda" else None
    
    for layer in Config.LAYERS_TO_ANALYZE:
        activations[layer] = np.concatenate(activations[layer], axis=0)
    
    del model_a
    torch.cuda.empty_cache() if device == "cuda" else None
    
    # Without AQ indices, just show overall spectral distribution
    print("\n" + "=" * 70)
    print("SPECTRAL DISTRIBUTION (No AQ indices provided)")
    print("=" * 70)
    
    for layer in Config.LAYERS_TO_ANALYZE:
        print(f"\nLayer {layer}:")
        spectra, freqs = compute_neuron_spectrum(activations[layer])
        centroids = compute_spectral_centroid(spectra, freqs)
        bands = assign_frequency_bands(centroids, Config.NUM_BANDS)
        
        print(f"  Mean centroid: {np.mean(centroids):.4f}")
        print(f"  Std centroid: {np.std(centroids):.4f}")
        
        band_counts = [np.sum(bands == b) for b in range(Config.NUM_BANDS)]
        print(f"  Band distribution: {band_counts}")
    
    return activations


# ==============================================================================
# USAGE EXAMPLE
# ==============================================================================

"""
USAGE:

After running 001_action_quanta_extraction_aligned.py with both random and text:

# Get universal AQ indices (overlap between random and text)
universal_aq = {}
for layer in [3, 5, 7]:
    random_aq = set(np.where(results_random['procrustes'][layer]['aq_mask'])[0])
    text_aq = set(np.where(results_text['procrustes'][layer]['aq_mask'])[0])
    universal_aq[layer] = random_aq & text_aq
    print(f"Layer {layer}: {len(universal_aq[layer])} universal AQ")

# Run spectral analysis
# Need activations from model A (already extracted in previous run)
spectral_results = run_spectral_analysis(acts_a, universal_aq)
"""

if __name__ == "__main__":
    # Standalone mode - just shows spectral distribution
    activations = run_standalone_spectral_analysis()
