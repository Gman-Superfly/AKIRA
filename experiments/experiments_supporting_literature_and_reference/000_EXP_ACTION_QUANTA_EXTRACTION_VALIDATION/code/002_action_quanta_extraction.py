"""
EXPERIMENT 000: Action Quanta Extraction
=========================================

AKIRA Project - Foundational Experiment
Oscar Goldman - Shogu Research Group @ Datamutant.ai

This script identifies Action Quanta (AQ) candidates across independently 
trained LLMs using the Excess Correlation Method.

Action Quanta are defined as: "Irreducible units of actionable information 
that emerge from collapse - quasiparticles of the belief field"

This experiment tests whether such units exist as discrete, universal features
that transfer across different model architectures.

TO RUN IN GOOGLE COLAB:
-----------------------
1. Create new Colab notebook
2. Set Runtime > Change runtime type > T4 GPU
3. Copy this entire file into a cell
4. Run

We found this method to be very useful and it works well for our initial purposes.
Reference: Shuyang (2025) - What is Universality in LLMs?
https://towardsdatascience.com/what-is-universality-in-llm-and-how-to-find-universal-neurons/
"""

# note we chose rather arbitrary numbers for threshold 
# and we were actually sirprised at the results
# so we actually keep this colab notebook as it is 
# as it's the genesys of the framework

# ==============================================================================
# SETUP - Run this first in Colab
# ==============================================================================

# !pip install transformer-lens torch numpy scipy matplotlib tqdm --quiet
# you might get an error just restart the runtime
# and run !pip install transformer-lens again

import torch
import numpy as np
from scipy.stats import pearsonr, ortho_group
from scipy.linalg import orth
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Tuple, List, Dict
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
    """Experiment configuration - adjust as needed."""
    
    # Models to compare (choose pairs that fit in your GPU memory)
    # Option 1: Same architecture, different scale
    MODEL_A = "gpt2"           # 124M params, ~500MB
    MODEL_B = "gpt2-medium"    # 355M params, ~1.5GB
    
    # Option 2: Different architectures (uncomment to use)
    # MODEL_A = "EleutherAI/pythia-70m"
    # MODEL_B = "EleutherAI/pythia-160m"
    
    # Layers to analyze (middle layers often most interesting)
    LAYERS_TO_ANALYZE = [3, 5, 7]  # Adjust based on model depth
    
    # Data settings
    NUM_SAMPLES = 500          # Number of text samples
    SEQ_LENGTH = 64            # Tokens per sample
    BATCH_SIZE = 16            # Reduce if OOM
    
    # AQ detection threshold (excess correlation above this = AQ candidate)
    AQ_THRESHOLD = 0.3  # Start lower, can raise if too many candidates
    
    # Input type: "random" for random tokens, "text" for real text samples
    # Random tokens test structural universality (recommended for initial experiment)
    # Real text tests semantic universality (use for follow-up experiments)
    INPUT_TYPE = "random"  # "random" or "text"
    
    # Random seed for reproducibility
    SEED = 42


# ==============================================================================
# DATA GENERATION
# ==============================================================================

def generate_random_text_tokens(tokenizer, num_samples: int, seq_length: int) -> torch.Tensor:
    """
    Generate random token sequences for activation extraction.
    
    Using random tokens ensures we're not biasing toward any particular
    semantic content - we want to find structure that emerges regardless
    of input.
    """
    vocab_size = tokenizer.vocab_size
    
    # Generate random tokens (avoiding special tokens)
    random_tokens = torch.randint(
        low=1000,  # Skip special tokens at start of vocab
        high=vocab_size - 1000,  # Skip special tokens at end
        size=(num_samples, seq_length)
    )
    
    return random_tokens


def get_sample_texts() -> List[str]:
    """
    Alternative: Use actual text samples for more realistic activations.
    Downloads a small subset of a public dataset for diverse text.
    
    For production experiments, consider using:
    - Hugging Face datasets: load_dataset("wikitext", "wikitext-2-raw-v1")
    - Or: load_dataset("openwebtext", split="train[:5000]")
    """
    try:
        # Try to load from Hugging Face datasets (best option)
        from datasets import load_dataset
        print("  Loading text samples from Hugging Face datasets...")
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        # Filter out empty lines and get sufficient samples
        texts = [t for t in dataset["text"] if len(t.strip()) > 50][:Config.NUM_SAMPLES * 2]
        print(f"  Loaded {len(texts)} text samples from wikitext")
        return texts[:Config.NUM_SAMPLES]
    except ImportError:
        print("  datasets library not available, using fallback texts")
    except Exception as e:
        print(f"  Could not load dataset: {e}, using fallback texts")
    
    # Fallback: diverse manually curated sentences across domains
    texts = [
        # Science
        "The quantum mechanics of electrons in semiconductors determines their electrical properties.",
        "Climate change affects ocean currents and global temperature patterns significantly.",
        "DNA replication occurs through a complex process involving multiple enzymes and proteins.",
        "Black holes emit radiation due to quantum effects near the event horizon.",
        "Photosynthesis converts carbon dioxide and water into glucose using sunlight energy.",
        # Technology
        "Machine learning algorithms learn patterns from data without explicit programming.",
        "The internet protocol stack consists of multiple layers handling different functions.",
        "Cryptographic hash functions are essential for secure digital communications.",
        "Neural networks process information through interconnected layers of artificial neurons.",
        "Cloud computing enables scalable on-demand access to computing resources.",
        # History
        "The industrial revolution transformed manufacturing and urbanization patterns worldwide.",
        "Ancient civilizations developed writing systems to record trade and religious texts.",
        "World War II resulted in significant geopolitical changes across continents.",
        "The Renaissance marked a period of renewed interest in classical art and learning.",
        "Colonial powers extracted resources from territories across Africa and Asia.",
        # Literature/Philosophy
        "Existentialist philosophy emphasizes individual freedom and personal responsibility.",
        "Shakespeare's plays explore themes of power, love, and human nature.",
        "The Enlightenment promoted reason and scientific inquiry over tradition.",
        "Mythology across cultures shares common archetypes and narrative structures.",
        "Postmodern literature often questions objective truth and grand narratives.",
        # Daily life
        "The morning commute through the city takes approximately forty five minutes.",
        "Cooking requires careful attention to timing, temperature, and ingredient proportions.",
        "Exercise and proper nutrition are essential for maintaining good health.",
        "Financial planning helps individuals prepare for retirement and unexpected expenses.",
        "Education systems vary significantly across different countries and cultures.",
        # Nature
        "Migration patterns of birds span thousands of miles across multiple continents.",
        "Coral reefs support incredible biodiversity despite covering small ocean areas.",
        "Mountain ecosystems host unique species adapted to high altitude conditions.",
        "Forests play a crucial role in carbon sequestration and oxygen production.",
        "Desert organisms have evolved remarkable adaptations for water conservation.",
        # Abstract/Reasoning
        "The relationship between cause and effect forms the basis of scientific inquiry.",
        "Mathematical proofs require rigorous logical reasoning from axioms to conclusions.",
        "Economic models attempt to predict market behavior under various conditions.",
        "Language acquisition in children follows predictable developmental stages.",
        "Decision making under uncertainty involves weighing probabilities and outcomes.",
        # Social
        "Social media platforms have transformed how people communicate and share information.",
        "Urban planning must balance housing, transportation, and environmental concerns.",
        "Healthcare systems face challenges of access, cost, and quality worldwide.",
        "Education reform debates center on curriculum, testing, and teaching methods.",
        "International cooperation is essential for addressing global challenges.",
    ]
    
    # Extend by combining sentences for more variety
    extended = texts.copy()
    for i in range(len(texts)):
        for j in range(i+1, min(i+5, len(texts))):
            extended.append(f"{texts[i]} {texts[j]}")
    
    # Shuffle for variety
    np.random.shuffle(extended)
    
    return extended[:Config.NUM_SAMPLES]


# ==============================================================================
# ACTIVATION EXTRACTION
# ==============================================================================

def extract_mlp_activations(
    model,
    tokens: torch.Tensor,
    layer_indices: List[int]
) -> Dict[int, np.ndarray]:
    """
    Extract MLP activations from specified layers.
    
    Args:
        model: HookedTransformer model
        tokens: Input tokens [batch, seq_len]
        layer_indices: Which layers to extract from
    
    Returns:
        Dict mapping layer index to activations [num_samples, seq_len, mlp_dim]
    """
    activations = {layer: [] for layer in layer_indices}
    
    # Process in batches
    num_batches = (len(tokens) + Config.BATCH_SIZE - 1) // Config.BATCH_SIZE
    
    for batch_idx in tqdm(range(num_batches), desc="Extracting activations"):
        start = batch_idx * Config.BATCH_SIZE
        end = min(start + Config.BATCH_SIZE, len(tokens))
        batch_tokens = tokens[start:end].to(device)
        
        # Run model and cache activations
        with torch.no_grad():
            _, cache = model.run_with_cache(batch_tokens)
        
        # Extract MLP post-activations (after nonlinearity)
        for layer in layer_indices:
            # Hook name format: "blocks.{layer}.mlp.hook_post"
            hook_name = f"blocks.{layer}.mlp.hook_post"
            if hook_name in cache:
                layer_acts = cache[hook_name].cpu().numpy()
                activations[layer].append(layer_acts)
        
        # Clear cache to save memory
        del cache
        torch.cuda.empty_cache() if device == "cuda" else None
    
    # Concatenate batches
    for layer in layer_indices:
        activations[layer] = np.concatenate(activations[layer], axis=0)
    
    return activations


# ==============================================================================
# EXCESS CORRELATION COMPUTATION
# ==============================================================================

def compute_excess_correlation(
    activations_a: np.ndarray,
    activations_b: np.ndarray,
    num_random_baselines: int = 5
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute excess correlation between two models' activations.
    
    The key insight: random rotation destroys genuine alignment but
    preserves the distribution. Excess = actual - baseline tells us
    which correlations are real vs chance.
    
    Args:
        activations_a: [num_samples, seq_len, dim_a] from model A
        activations_b: [num_samples, seq_len, dim_b] from model B
        num_random_baselines: Number of random rotations to average
    
    Returns:
        actual_corr: Actual correlations per neuron
        baseline_corr: Baseline correlations (averaged)
        excess_corr: Excess = actual - baseline
    """
    # Flatten to [num_samples * seq_len, dim]
    flat_a = activations_a.reshape(-1, activations_a.shape[-1])
    flat_b = activations_b.reshape(-1, activations_b.shape[-1])
    
    # Handle dimension mismatch by using smaller dimension
    min_dim = min(flat_a.shape[1], flat_b.shape[1])
    flat_a = flat_a[:, :min_dim]
    flat_b = flat_b[:, :min_dim]
    
    print(f"  Computing correlations for {min_dim} neurons...")
    
    # Compute actual correlation per neuron
    actual_corr = np.zeros(min_dim)
    for i in tqdm(range(min_dim), desc="  Actual correlation", leave=False):
        corr, _ = pearsonr(flat_a[:, i], flat_b[:, i])
        actual_corr[i] = corr if not np.isnan(corr) else 0
    
    # Compute baseline with random rotations
    baseline_corrs = []
    for _ in tqdm(range(num_random_baselines), desc="  Baseline rotations", leave=False):
        # Generate random orthogonal matrix
        random_rotation = ortho_group.rvs(min_dim)
        rotated_b = flat_b @ random_rotation
        
        baseline = np.zeros(min_dim)
        for i in range(min_dim):
            corr, _ = pearsonr(flat_a[:, i], rotated_b[:, i])
            baseline[i] = corr if not np.isnan(corr) else 0
        baseline_corrs.append(baseline)
    
    baseline_corr = np.mean(baseline_corrs, axis=0)
    excess_corr = actual_corr - baseline_corr
    
    return actual_corr, baseline_corr, excess_corr


def identify_action_quanta(
    excess_corr: np.ndarray,
    threshold: float = Config.AQ_THRESHOLD
) -> Tuple[np.ndarray, int]:
    """
    Identify Action Quanta candidates based on excess correlation threshold.
    
    Features with high excess correlation transfer across models, suggesting
    they represent universal structure - these are AQ candidates.
    
    Returns:
        aq_mask: Boolean mask for AQ candidates
        num_aq: Count of AQ candidates
    """
    aq_mask = excess_corr > threshold
    num_aq = np.sum(aq_mask)
    
    return aq_mask, num_aq


# ==============================================================================
# VISUALIZATION
# ==============================================================================

def plot_aq_analysis(
    actual_corr: np.ndarray,
    baseline_corr: np.ndarray,
    excess_corr: np.ndarray,
    aq_mask: np.ndarray,
    layer: int,
    save_path: str = None
):
    """Generate visualization of Action Quanta analysis."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('AKIRA Experiment 000: Action Quanta Extraction', fontsize=14, fontweight='bold')
    
    # 1. Actual vs Baseline scatter
    ax1 = axes[0, 0]
    ax1.scatter(baseline_corr, actual_corr, alpha=0.5, s=10)
    ax1.plot([-1, 1], [-1, 1], 'r--', label='y=x')
    ax1.set_xlabel('Baseline Correlation')
    ax1.set_ylabel('Actual Correlation')
    ax1.set_title(f'Layer {layer}: Actual vs Baseline Correlation')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Excess correlation distribution
    ax2 = axes[0, 1]
    ax2.hist(excess_corr, bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax2.axvline(x=Config.AQ_THRESHOLD, color='red', 
                linestyle='--', label=f'AQ Threshold ({Config.AQ_THRESHOLD})')
    ax2.set_xlabel('Excess Correlation')
    ax2.set_ylabel('Count')
    ax2.set_title(f'Layer {layer}: Excess Correlation Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Excess correlation with AQ candidates highlighted
    ax3 = axes[1, 0]
    neuron_indices = np.arange(len(excess_corr))
    colors = ['green' if aq else 'gray' for aq in aq_mask]
    ax3.scatter(neuron_indices, excess_corr, c=colors, alpha=0.6, s=10)
    ax3.axhline(y=Config.AQ_THRESHOLD, color='red', 
                linestyle='--', label=f'AQ Threshold')
    ax3.set_xlabel('Neuron Index')
    ax3.set_ylabel('Excess Correlation')
    ax3.set_title(f'Layer {layer}: Action Quanta Candidates (green)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Summary statistics
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    num_aq = np.sum(aq_mask)
    total_neurons = len(excess_corr)
    pct_aq = 100 * num_aq / total_neurons
    
    stats_text = f"""
    LAYER {layer} RESULTS
    =====================
    
    Total neurons analyzed: {total_neurons}
    Action Quanta candidates: {num_aq} ({pct_aq:.1f}%)
    
    Excess Correlation Stats:
      Mean: {np.mean(excess_corr):.4f}
      Std:  {np.std(excess_corr):.4f}
      Max:  {np.max(excess_corr):.4f}
      Min:  {np.min(excess_corr):.4f}
    
    AQ Threshold: {Config.AQ_THRESHOLD}
    
    INTERPRETATION (AKIRA Framework):
    ---------------------------------
    AQ candidates = features that TRANSFER across models
    
    If {pct_aq:.1f}% > 10%: Strong evidence for universal AQ
    If {pct_aq:.1f}% < 5%:  Weak evidence for discrete AQ
    
    NEXT: Test irreducibility and actionability
    """
    
    ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved plot to: {save_path}")
    
    plt.show()


# ==============================================================================
# MAIN EXPERIMENT
# ==============================================================================

def run_experiment():
    """
    Main experiment: Extract Action Quanta candidates from LLMs.
    
    Action Quanta (AKIRA definition):
    "Irreducible units of actionable information that emerge from collapse"
    
    This experiment tests Phase 2 of Experiment 000:
    - Do discrete, universal features exist across models?
    - If yes, these are AQ CANDIDATES (need further validation)
    """
    print("=" * 70)
    print("EXPERIMENT 000: ACTION QUANTA EXTRACTION")
    print("AKIRA Project - Foundational Experiment")
    print("=" * 70)
    print(f"\nModel A: {Config.MODEL_A}")
    print(f"Model B: {Config.MODEL_B}")
    print(f"Layers to analyze: {Config.LAYERS_TO_ANALYZE}")
    print(f"AQ Threshold: {Config.AQ_THRESHOLD}")
    print(f"Input type: {Config.INPUT_TYPE} ({'structural' if Config.INPUT_TYPE == 'random' else 'semantic'} universality test)")
    print()
    
    # Set seed
    torch.manual_seed(Config.SEED)
    np.random.seed(Config.SEED)
    
    # Import transformer_lens here (slow import)
    print("Loading transformer_lens...")
    from transformer_lens import HookedTransformer
    
    # Load models
    print(f"\nLoading {Config.MODEL_A}...")
    model_a = HookedTransformer.from_pretrained(Config.MODEL_A, device=device)
    
    print(f"Loading {Config.MODEL_B}...")
    model_b = HookedTransformer.from_pretrained(Config.MODEL_B, device=device)
    
    # Generate input tokens based on INPUT_TYPE setting
    if Config.INPUT_TYPE == "random":
        print(f"\nGenerating {Config.NUM_SAMPLES} random token sequences...")
        print("  (Random tokens test STRUCTURAL universality - independent of semantics)")
        tokens = generate_random_text_tokens(model_a.tokenizer, Config.NUM_SAMPLES, Config.SEQ_LENGTH)
    else:
        print(f"\nLoading {Config.NUM_SAMPLES} real text samples...")
        print("  (Real text tests SEMANTIC universality)")
        texts = get_sample_texts()
        # Tokenize with padding/truncation
        encoded = model_a.tokenizer(
            texts, 
            padding='max_length', 
            truncation=True, 
            max_length=Config.SEQ_LENGTH,
            return_tensors='pt'
        )
        tokens = encoded['input_ids']
        print(f"  Tokenized {len(tokens)} text samples")
    
    # Adjust layers based on actual model depth
    model_a_layers = model_a.cfg.n_layers
    model_b_layers = model_b.cfg.n_layers
    min_layers = min(model_a_layers, model_b_layers)
    
    layers_to_use = [l for l in Config.LAYERS_TO_ANALYZE if l < min_layers]
    if not layers_to_use:
        layers_to_use = [min_layers // 4, min_layers // 2, 3 * min_layers // 4]
    
    print(f"\nModel A has {model_a_layers} layers, Model B has {model_b_layers} layers")
    print(f"Analyzing layers: {layers_to_use}")
    
    # Extract activations
    print(f"\nExtracting activations from Model A ({Config.MODEL_A})...")
    acts_a = extract_mlp_activations(model_a, tokens, layers_to_use)
    
    print(f"\nExtracting activations from Model B ({Config.MODEL_B})...")
    acts_b = extract_mlp_activations(model_b, tokens, layers_to_use)
    
    # Free model memory
    del model_a, model_b
    torch.cuda.empty_cache() if device == "cuda" else None
    
    # Analyze each layer
    results = {}
    
    for layer in layers_to_use:
        print(f"\n{'='*50}")
        print(f"ANALYZING LAYER {layer}")
        print(f"{'='*50}")
        
        # Compute excess correlation
        actual, baseline, excess = compute_excess_correlation(
            acts_a[layer], acts_b[layer]
        )
        
        # Identify Action Quanta candidates
        aq_mask, num_aq = identify_action_quanta(excess)
        
        total = len(excess)
        pct = 100 * num_aq / total
        
        print(f"\n  RESULTS:")
        print(f"  - Total neurons: {total}")
        print(f"  - Action Quanta candidates: {num_aq} ({pct:.1f}%)")
        print(f"  - Mean excess correlation: {np.mean(excess):.4f}")
        print(f"  - Max excess correlation: {np.max(excess):.4f}")
        
        # Store results
        results[layer] = {
            'actual_corr': actual,
            'baseline_corr': baseline,
            'excess_corr': excess,
            'aq_mask': aq_mask,
            'num_aq': num_aq,
            'pct_aq': pct,
            'aq_indices': np.where(aq_mask)[0]  # Store which neurons are AQ candidates
        }
        
        # Plot
        plot_aq_analysis(
            actual, baseline, excess, aq_mask, layer
        )
    
    # Summary
    print("\n" + "=" * 70)
    print("EXPERIMENT 000 SUMMARY: ACTION QUANTA EXTRACTION")
    print("=" * 70)
    
    total_aq_candidates = 0
    for layer, r in results.items():
        print(f"Layer {layer}: {r['num_aq']} AQ candidates ({r['pct_aq']:.1f}%)")
        total_aq_candidates += r['num_aq']
    
    avg_pct = np.mean([r['pct_aq'] for r in results.values()])
    print(f"\nAverage across layers: {avg_pct:.1f}%")
    print(f"Total AQ candidates found: {total_aq_candidates}")
    
    print("\n" + "-" * 70)
    print("VERDICT:")
    print("-" * 70)
    
    if avg_pct > 10:
        print("STRONG evidence for universal Action Quanta")
        print("These features transfer across architectures - AQ likely EXIST")
    elif avg_pct > 5:
        print("MODERATE evidence for Action Quanta")
        print("Some universal structure detected - needs further validation")
    else:
        print("WEAK evidence for discrete Action Quanta")
        print("Consider: lower threshold, different model pairs, or")
        print("AQ may be emergent (not discrete) - see AKIRA framework notes")
    
    print("\n" + "=" * 70)
    print("NEXT STEPS (Experiment 000 Phases 3-6):")
    print("=" * 70)
    print("1. IRREDUCIBILITY TEST: Can AQ candidates be decomposed further?")
    print("2. ACTIONABILITY TEST: Are they load-bearing for downstream tasks?")
    print("3. SPECTRAL ANALYSIS: Do they live in low-frequency bands?")
    print("4. CROSS-ARCHITECTURE: Do they transfer to Llama/Mistral?")
    print("5. BIOLOGICAL VALIDATION: Correlate with brain imaging data")
    
    return results


# ==============================================================================
# RUN
# ==============================================================================

if __name__ == "__main__":
    results = run_experiment()
