"""
EXPERIMENT 000: Action Quanta Extraction (ALIGNED VERSION)
===========================================================

AKIRA Project - Foundational Experiment
Oscar Goldman - Shogu Research Group @ Datamutant.ai

This version adds PROPER ALIGNMENT METHODS:
1. CKA (Centered Kernel Alignment) - measures representation similarity
2. Optimal Transport - finds best neuron-to-neuron matching
3. Linear Mapping - learns transformation between spaces

These address the key weakness of the basic version: assuming neurons
align by index across models (they don't).

TO RUN IN GOOGLE COLAB:
-----------------------
1. Create new Colab notebook
2. Set Runtime > Change runtime type > T4 GPU
3. Install: !pip install transformer-lens torch numpy scipy matplotlib tqdm pot --quiet
4. Copy this file and run

Reference: Kornblith et al. (2019) - Similarity of Neural Network Representations
"""

# ==============================================================================
# SETUP
# ==============================================================================

# !pip install transformer-lens torch numpy scipy matplotlib tqdm pot --quiet

import torch
import numpy as np
from scipy.stats import pearsonr
from scipy.linalg import orthogonal_procrustes
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Tuple, List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

# Check GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


# ==============================================================================
# CONFIGURATION
# ==============================================================================

class Config:
    """Experiment configuration."""
    
    MODEL_A = "gpt2"
    MODEL_B = "gpt2-medium"
    
    LAYERS_TO_ANALYZE = [3, 5, 7]
    
    NUM_SAMPLES = 500
    SEQ_LENGTH = 64
    BATCH_SIZE = 16
    
    # Threshold methods: "statistical", "percentile", or "fixed"
    THRESHOLD_METHOD = "statistical"  # Recommended
    FIXED_THRESHOLD = 0.3  # Only used if THRESHOLD_METHOD = "fixed"
    PERCENTILE = 95  # Only used if THRESHOLD_METHOD = "percentile"
    
    INPUT_TYPE = "random"
    SEED = 42


# ==============================================================================
# ALIGNMENT METHOD 1: CKA (Centered Kernel Alignment)
# ==============================================================================

def centering_matrix(n: int) -> np.ndarray:
    """Create centering matrix H = I - (1/n) * 1 * 1^T"""
    return np.eye(n) - np.ones((n, n)) / n


def hsic(K: np.ndarray, L: np.ndarray) -> float:
    """
    Hilbert-Schmidt Independence Criterion.
    Measures statistical dependence between two kernel matrices.
    """
    n = K.shape[0]
    H = centering_matrix(n)
    return np.trace(K @ H @ L @ H) / ((n - 1) ** 2)


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Linear Centered Kernel Alignment.
    
    Measures similarity between two representations independent of
    dimensionality or neuron ordering.
    
    CKA = 1.0 means identical representations (up to linear transform)
    CKA = 0.0 means completely unrelated
    
    Reference: Kornblith et al. (2019)
    
    Args:
        X: [n_samples, dim_x] activations from model A
        Y: [n_samples, dim_y] activations from model B
    
    Returns:
        CKA similarity score in [0, 1]
    """
    # Center the data
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)
    
    # Compute linear kernels
    K = X @ X.T  # [n_samples, n_samples]
    L = Y @ Y.T
    
    # Compute CKA
    hsic_kl = hsic(K, L)
    hsic_kk = hsic(K, K)
    hsic_ll = hsic(L, L)
    
    if hsic_kk * hsic_ll == 0:
        return 0.0
    
    cka = hsic_kl / np.sqrt(hsic_kk * hsic_ll)
    return float(cka)


def compute_cka_per_layer(
    acts_a: Dict[int, np.ndarray],
    acts_b: Dict[int, np.ndarray],
    layers: List[int],
    subsample: int = 5000
) -> Dict[int, float]:
    """
    Compute CKA similarity for each layer.
    
    Args:
        acts_a: Activations from model A {layer: [samples, seq, dim]}
        acts_b: Activations from model B
        layers: Which layers to analyze
        subsample: Max samples for CKA (memory constraint)
    
    Returns:
        Dict mapping layer to CKA score
    """
    cka_scores = {}
    
    for layer in layers:
        print(f"\n  Computing CKA for layer {layer}...")
        
        # Flatten: [samples, seq, dim] -> [samples*seq, dim]
        flat_a = acts_a[layer].reshape(-1, acts_a[layer].shape[-1])
        flat_b = acts_b[layer].reshape(-1, acts_b[layer].shape[-1])
        
        # Subsample if too large (CKA is O(n^2))
        if len(flat_a) > subsample:
            idx = np.random.choice(len(flat_a), subsample, replace=False)
            flat_a = flat_a[idx]
            flat_b = flat_b[idx]
        
        cka = linear_cka(flat_a, flat_b)
        cka_scores[layer] = cka
        print(f"    CKA = {cka:.4f}")
    
    return cka_scores


# ==============================================================================
# ALIGNMENT METHOD 2: Optimal Transport
# ==============================================================================

def compute_optimal_transport_matching(
    acts_a: np.ndarray,
    acts_b: np.ndarray,
    n_features: int = 100
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Use optimal transport to find best matching between neurons.
    
    This solves: which neuron in model B best corresponds to each neuron in model A?
    
    Args:
        acts_a: [n_samples, dim_a] flattened activations
        acts_b: [n_samples, dim_b] flattened activations
        n_features: Number of top features to match (for speed)
    
    Returns:
        matching_a: indices in A
        matching_b: corresponding best-match indices in B
        transport_cost: total transport cost (lower = better alignment)
    """
    try:
        import ot  # Python Optimal Transport library
    except ImportError:
        print("  WARNING: 'pot' not installed. Run: pip install pot")
        print("  Falling back to correlation-based matching...")
        return correlation_based_matching(acts_a, acts_b, n_features)
    
    dim_a = acts_a.shape[1]
    dim_b = acts_b.shape[1]
    
    # Use subset of features for speed
    use_a = min(n_features, dim_a)
    use_b = min(n_features, dim_b)
    
    # Select most variable features
    var_a = np.var(acts_a, axis=0)
    var_b = np.var(acts_b, axis=0)
    top_a = np.argsort(var_a)[-use_a:]
    top_b = np.argsort(var_b)[-use_b:]
    
    subset_a = acts_a[:, top_a]  # [n_samples, use_a]
    subset_b = acts_b[:, top_b]  # [n_samples, use_b]
    
    # Compute cost matrix: distance between each pair of neurons
    # Cost[i,j] = 1 - |correlation(neuron_i_A, neuron_j_B)|
    print(f"    Computing cost matrix ({use_a} x {use_b})...")
    cost_matrix = np.zeros((use_a, use_b))
    
    for i in tqdm(range(use_a), desc="    OT cost matrix", leave=False):
        for j in range(use_b):
            corr = np.corrcoef(subset_a[:, i], subset_b[:, j])[0, 1]
            if np.isnan(corr):
                corr = 0
            cost_matrix[i, j] = 1 - abs(corr)  # Lower cost = higher correlation
    
    # Uniform distributions (each neuron has equal mass)
    a_dist = np.ones(use_a) / use_a
    b_dist = np.ones(use_b) / use_b
    
    # Solve optimal transport
    print("    Solving optimal transport...")
    transport_plan = ot.emd(a_dist, b_dist, cost_matrix)
    
    # Extract matching: for each neuron in A, find best match in B
    matching_a = top_a
    matching_b = top_b[np.argmax(transport_plan, axis=1)]
    
    # Compute transport cost
    transport_cost = np.sum(transport_plan * cost_matrix)
    
    return matching_a, matching_b, transport_cost


def correlation_based_matching(
    acts_a: np.ndarray,
    acts_b: np.ndarray,
    n_features: int = 100
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Fallback: simple correlation-based matching."""
    dim_a = acts_a.shape[1]
    dim_b = acts_b.shape[1]
    
    use_a = min(n_features, dim_a)
    
    var_a = np.var(acts_a, axis=0)
    top_a = np.argsort(var_a)[-use_a:]
    
    matching_a = []
    matching_b = []
    total_corr = 0
    
    for i in tqdm(top_a, desc="    Correlation matching", leave=False):
        best_corr = -1
        best_j = 0
        for j in range(dim_b):
            corr = abs(np.corrcoef(acts_a[:, i], acts_b[:, j])[0, 1])
            if not np.isnan(corr) and corr > best_corr:
                best_corr = corr
                best_j = j
        matching_a.append(i)
        matching_b.append(best_j)
        total_corr += best_corr
    
    return np.array(matching_a), np.array(matching_b), use_a - total_corr


# ==============================================================================
# ALIGNMENT METHOD 3: Linear Mapping (Procrustes)
# ==============================================================================

def learn_linear_mapping(
    acts_a: np.ndarray,
    acts_b: np.ndarray
) -> Tuple[np.ndarray, float]:
    """
    Learn optimal linear transformation from space A to space B.
    
    Solves: min ||A @ R - B||_F  subject to R^T @ R = I
    
    This is the Orthogonal Procrustes problem.
    
    Args:
        acts_a: [n_samples, dim] activations from model A
        acts_b: [n_samples, dim] activations from model B (same dim required)
    
    Returns:
        R: Orthogonal transformation matrix
        error: Procrustes error (lower = better alignment)
    """
    # Procrustes requires same dimensions
    min_dim = min(acts_a.shape[1], acts_b.shape[1])
    A = acts_a[:, :min_dim]
    B = acts_b[:, :min_dim]
    
    # Center the data
    A = A - A.mean(axis=0)
    B = B - B.mean(axis=0)
    
    # Solve Procrustes
    R, scale = orthogonal_procrustes(A, B)
    
    # Compute alignment error
    aligned_A = A @ R
    error = np.linalg.norm(aligned_A - B, 'fro') / np.linalg.norm(B, 'fro')
    
    return R, error


def compute_aligned_correlations(
    acts_a: np.ndarray,
    acts_b: np.ndarray,
    R: np.ndarray
) -> np.ndarray:
    """
    Compute correlations after applying learned alignment.
    
    Returns:
        correlations: [min_dim] correlation per aligned dimension
    """
    min_dim = R.shape[0]
    A = acts_a[:, :min_dim]
    B = acts_b[:, :min_dim]
    
    # Apply alignment
    aligned_A = A @ R
    
    # Compute per-dimension correlations
    correlations = np.array([
        pearsonr(aligned_A[:, i], B[:, i])[0]
        for i in range(min_dim)
    ])
    
    return np.nan_to_num(correlations, 0)


# ==============================================================================
# STATISTICAL THRESHOLD COMPUTATION
# ==============================================================================

def compute_statistical_threshold(
    excess_corr: np.ndarray,
    method: str = "statistical",
    percentile: int = 95,
    fixed: float = 0.3
) -> Tuple[float, str]:
    """
    Compute threshold using principled methods.
    
    Args:
        excess_corr: Array of excess correlations
        method: "statistical", "percentile", or "fixed"
        percentile: Percentile for percentile method
        fixed: Fixed threshold value
    
    Returns:
        threshold: Computed threshold
        explanation: How it was computed
    """
    if method == "statistical":
        # Mean + 2*std (outliers beyond 2 standard deviations)
        mean = np.mean(excess_corr)
        std = np.std(excess_corr)
        threshold = mean + 2 * std
        explanation = f"mean + 2*std = {mean:.4f} + 2*{std:.4f}"
        
    elif method == "percentile":
        threshold = np.percentile(excess_corr, percentile)
        explanation = f"{percentile}th percentile"
        
    else:  # fixed
        threshold = fixed
        explanation = f"fixed value (arbitrary)"
    
    return threshold, explanation


# ==============================================================================
# DATA GENERATION (same as basic version)
# ==============================================================================

def generate_random_text_tokens(tokenizer, num_samples: int, seq_length: int) -> torch.Tensor:
    vocab_size = tokenizer.vocab_size
    random_tokens = torch.randint(
        low=1000,
        high=vocab_size - 1000,
        size=(num_samples, seq_length)
    )
    return random_tokens


# ==============================================================================
# ACTIVATION EXTRACTION (same as basic version)
# ==============================================================================

def extract_mlp_activations(
    model,
    tokens: torch.Tensor,
    layer_indices: List[int]
) -> Dict[int, np.ndarray]:
    activations = {layer: [] for layer in layer_indices}
    num_batches = (len(tokens) + Config.BATCH_SIZE - 1) // Config.BATCH_SIZE
    
    for batch_idx in tqdm(range(num_batches), desc="Extracting activations"):
        start = batch_idx * Config.BATCH_SIZE
        end = min(start + Config.BATCH_SIZE, len(tokens))
        batch_tokens = tokens[start:end].to(device)
        
        with torch.no_grad():
            _, cache = model.run_with_cache(batch_tokens)
        
        for layer in layer_indices:
            hook_name = f"blocks.{layer}.mlp.hook_post"
            if hook_name in cache:
                layer_acts = cache[hook_name].cpu().numpy()
                activations[layer].append(layer_acts)
        
        del cache
        torch.cuda.empty_cache() if device == "cuda" else None
    
    for layer in layer_indices:
        activations[layer] = np.concatenate(activations[layer], axis=0)
    
    return activations


# ==============================================================================
# MAIN EXPERIMENT
# ==============================================================================

def run_aligned_experiment():
    """
    Run experiment with proper alignment methods.
    """
    print("=" * 70)
    print("EXPERIMENT 000: ACTION QUANTA EXTRACTION (ALIGNED)")
    print("=" * 70)
    print(f"\nModel A: {Config.MODEL_A}")
    print(f"Model B: {Config.MODEL_B}")
    print(f"Layers: {Config.LAYERS_TO_ANALYZE}")
    print(f"Threshold method: {Config.THRESHOLD_METHOD}")
    print(f"Input type: {Config.INPUT_TYPE}")
    print()
    
    torch.manual_seed(Config.SEED)
    np.random.seed(Config.SEED)
    
    print("Loading transformer_lens...")
    from transformer_lens import HookedTransformer
    
    print(f"\nLoading {Config.MODEL_A}...")
    model_a = HookedTransformer.from_pretrained(Config.MODEL_A, device=device)
    
    print(f"Loading {Config.MODEL_B}...")
    model_b = HookedTransformer.from_pretrained(Config.MODEL_B, device=device)
    
    # Generate input tokens based on INPUT_TYPE setting
    if Config.INPUT_TYPE == "random":
        print(f"\nGenerating {Config.NUM_SAMPLES} random token sequences...")
        print("  (Random tokens test STRUCTURAL universality)")
        tokens = generate_random_text_tokens(model_a.tokenizer, Config.NUM_SAMPLES, Config.SEQ_LENGTH)
    else:
        print(f"\nLoading {Config.NUM_SAMPLES} real text samples...")
        print("  (Real text tests SEMANTIC universality)")
        from datasets import load_dataset
        try:
            dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
            texts = [t for t in dataset["text"] if len(t.strip()) > 50][:Config.NUM_SAMPLES]
            print(f"  Loaded {len(texts)} samples from wikitext")
        except:
            print("  Using fallback text samples")
            texts = [
                "The quantum mechanics of electrons determines electrical properties.",
                "Climate change affects ocean currents and temperature patterns.",
                "Machine learning algorithms learn patterns from data.",
                "The industrial revolution transformed manufacturing worldwide.",
            ] * (Config.NUM_SAMPLES // 4 + 1)
            texts = texts[:Config.NUM_SAMPLES]
        
        encoded = model_a.tokenizer(
            texts,
            padding='max_length',
            truncation=True,
            max_length=Config.SEQ_LENGTH,
            return_tensors='pt'
        )
        tokens = encoded['input_ids']
        print(f"  Tokenized {len(tokens)} text samples")
    
    model_a_layers = model_a.cfg.n_layers
    model_b_layers = model_b.cfg.n_layers
    min_layers = min(model_a_layers, model_b_layers)
    layers_to_use = [l for l in Config.LAYERS_TO_ANALYZE if l < min_layers]
    
    print(f"\nExtracting activations from Model A...")
    acts_a = extract_mlp_activations(model_a, tokens, layers_to_use)
    
    print(f"\nExtracting activations from Model B...")
    acts_b = extract_mlp_activations(model_b, tokens, layers_to_use)
    
    del model_a, model_b
    torch.cuda.empty_cache() if device == "cuda" else None
    
    results = {}
    
    # ==========================================================================
    # ANALYSIS 1: CKA (Global representation similarity)
    # ==========================================================================
    print("\n" + "=" * 70)
    print("ANALYSIS 1: CKA (Centered Kernel Alignment)")
    print("=" * 70)
    print("Measures: Overall representation similarity (invariant to neuron ordering)")
    
    cka_scores = compute_cka_per_layer(acts_a, acts_b, layers_to_use)
    
    print("\nCKA Results:")
    for layer, cka in cka_scores.items():
        interpretation = "HIGH" if cka > 0.7 else "MODERATE" if cka > 0.4 else "LOW"
        print(f"  Layer {layer}: CKA = {cka:.4f} ({interpretation} similarity)")
    
    results['cka'] = cka_scores
    
    # ==========================================================================
    # ANALYSIS 2: Optimal Transport Matching
    # ==========================================================================
    print("\n" + "=" * 70)
    print("ANALYSIS 2: Optimal Transport Neuron Matching")
    print("=" * 70)
    print("Finds: Best neuron-to-neuron correspondence across models")
    
    ot_results = {}
    for layer in layers_to_use:
        print(f"\n  Layer {layer}:")
        flat_a = acts_a[layer].reshape(-1, acts_a[layer].shape[-1])
        flat_b = acts_b[layer].reshape(-1, acts_b[layer].shape[-1])
        
        matching_a, matching_b, cost = compute_optimal_transport_matching(
            flat_a, flat_b, n_features=100
        )
        
        # Compute correlations for matched pairs
        matched_corrs = []
        for i, (idx_a, idx_b) in enumerate(zip(matching_a, matching_b)):
            corr = np.corrcoef(flat_a[:, idx_a], flat_b[:, idx_b])[0, 1]
            if not np.isnan(corr):
                matched_corrs.append(abs(corr))
        
        mean_corr = np.mean(matched_corrs) if matched_corrs else 0
        print(f"    Transport cost: {cost:.4f}")
        print(f"    Mean matched correlation: {mean_corr:.4f}")
        print(f"    High-corr matches (>0.5): {sum(c > 0.5 for c in matched_corrs)}/{len(matched_corrs)}")
        
        ot_results[layer] = {
            'matching_a': matching_a,
            'matching_b': matching_b,
            'cost': cost,
            'matched_correlations': matched_corrs
        }
    
    results['optimal_transport'] = ot_results
    
    # ==========================================================================
    # ANALYSIS 3: Linear Mapping (Procrustes)
    # ==========================================================================
    print("\n" + "=" * 70)
    print("ANALYSIS 3: Linear Mapping (Procrustes Alignment)")
    print("=" * 70)
    print("Learns: Optimal rotation from space A to space B")
    
    procrustes_results = {}
    for layer in layers_to_use:
        print(f"\n  Layer {layer}:")
        flat_a = acts_a[layer].reshape(-1, acts_a[layer].shape[-1])
        flat_b = acts_b[layer].reshape(-1, acts_b[layer].shape[-1])
        
        R, error = learn_linear_mapping(flat_a, flat_b)
        aligned_corrs = compute_aligned_correlations(flat_a, flat_b, R)
        
        # Compute threshold
        threshold, explanation = compute_statistical_threshold(
            aligned_corrs,
            Config.THRESHOLD_METHOD,
            Config.PERCENTILE,
            Config.FIXED_THRESHOLD
        )
        
        aq_mask = aligned_corrs > threshold
        num_aq = np.sum(aq_mask)
        pct_aq = 100 * num_aq / len(aligned_corrs)
        
        print(f"    Procrustes error: {error:.4f}")
        print(f"    Threshold ({explanation}): {threshold:.4f}")
        print(f"    AQ candidates: {num_aq} ({pct_aq:.1f}%)")
        print(f"    Mean aligned correlation: {np.mean(aligned_corrs):.4f}")
        print(f"    Max aligned correlation: {np.max(aligned_corrs):.4f}")
        
        procrustes_results[layer] = {
            'R': R,
            'error': error,
            'aligned_correlations': aligned_corrs,
            'threshold': threshold,
            'threshold_method': explanation,
            'aq_mask': aq_mask,
            'num_aq': num_aq,
            'pct_aq': pct_aq
        }
    
    results['procrustes'] = procrustes_results
    
    # ==========================================================================
    # SUMMARY
    # ==========================================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT 000 SUMMARY (ALIGNED VERSION)")
    print("=" * 70)
    
    print("\n1. CKA (Representation Similarity):")
    avg_cka = np.mean(list(cka_scores.values()))
    print(f"   Average CKA: {avg_cka:.4f}")
    if avg_cka > 0.5:
        print("   -> Models share significant representational structure")
    else:
        print("   -> Models have different representational structure")
    
    print("\n2. Optimal Transport (Neuron Matching):")
    for layer, ot in ot_results.items():
        high_matches = sum(c > 0.5 for c in ot['matched_correlations'])
        print(f"   Layer {layer}: {high_matches} high-correlation neuron pairs found")
    
    print("\n3. Procrustes (Aligned AQ Detection):")
    total_aq = sum(p['num_aq'] for p in procrustes_results.values())
    avg_pct = np.mean([p['pct_aq'] for p in procrustes_results.values()])
    print(f"   Total AQ candidates: {total_aq}")
    print(f"   Average: {avg_pct:.1f}%")
    
    print("\n" + "-" * 70)
    print("INTERPRETATION:")
    print("-" * 70)
    if avg_cka > 0.5 and avg_pct > 5:
        print("STRONG evidence for universal Action Quanta")
        print("Both global similarity (CKA) and specific features (Procrustes) align")
    elif avg_cka > 0.3 or avg_pct > 2:
        print("MODERATE evidence for Action Quanta")
        print("Some universal structure detected")
    else:
        print("WEAK evidence for discrete Action Quanta")
        print("Models may have different internal organization")
    
    return results


# ==============================================================================
# RUN
# ==============================================================================

if __name__ == "__main__":
    results = run_aligned_experiment()

# ==============================================================================

# Run both experiments and compare
# (commented out for now as it an extra 
# this will run overlap analysis
# it will re-run both experiments, save the neuron indices
# and compare the results

# import numpy as np

# Config.INPUT_TYPE = "random"
# print("=" * 70)
# print("RUNNING RANDOM...")
# print("=" * 70)
# results_random = run_aligned_experiment()

# Config.INPUT_TYPE = "text" 
# print("\n" + "=" * 70)
# print("RUNNING TEXT...")
# print("=" * 70)
# results_text = run_aligned_experiment()



# print("=" * 70)
# print("OVERLAP ANALYSIS: Random vs Text AQ")
# print("=" * 70)
# for layer in [3, 5, 7]:
#     random_aq = set(np.where(results_random['procrustes'][layer]['aq_mask'])[0])
#     text_aq = set(np.where(results_text['procrustes'][layer]['aq_mask'])[0])
#     overlap = random_aq & text_aq
#    
#     print(f"\nLayer {layer}:")
#     print(f"  Random AQ: {len(random_aq)}")
#     print(f"  Text AQ: {len(text_aq)}")
#     print(f"  OVERLAP: {len(overlap)} ({100*len(overlap)/max(len(random_aq),1):.1f}% of random)")

# HAVE FUN NOW :D