"""
EXPERIMENT 000 - PHASE 5: Spectral Analysis of Action Quanta
=============================================================

AKIRA Project - Foundational Experiment
Oscar Goldman - Shogu Research Group @ Datamutant.ai

PURPOSE:
--------
This experiment tests a core AKIRA prediction: that universal features (Action Quanta)
should be concentrated in LOW frequency bands. The reasoning is:

1. Low-frequency features represent slow-changing, stable patterns
2. Universal features should be robust across different inputs
3. High-frequency features are more input-specific and variable

If AQ are truly fundamental "quasiparticles" of the belief field (as AKIRA proposes),
they should exhibit this low-frequency concentration.

METHOD:
-------
1. Extract activations from two different models (GPT-2 and GPT-2-medium)
2. Identify Action Quanta using Procrustes alignment and correlation analysis
3. Compute spectral centroid for each neuron (lower = more low-frequency dominant)
4. Compare: Do AQ neurons have lower spectral centroids than non-AQ neurons?

THREE TYPES OF AQ ARE ANALYZED:
- Structural AQ: Found with random token input (input-independent patterns)
- Semantic AQ: Found with real text input (meaning-dependent patterns)
- Universal AQ: Found in BOTH conditions (the most robust candidates)

TO RUN IN GOOGLE COLAB:
-----------------------
1. Create a new notebook
2. Paste this entire script into a cell
3. Run the cell (takes approximately 3-5 minutes on T4 GPU)
4. Results will print to console

EXPECTED RUNTIME: ~15-20 minutes on Colab T4 GPU (ALL 12 LAYERS)
EXPECTED RAM: ~16-20 GB
"""

import torch
import numpy as np
from scipy.fft import fft, fftfreq
from scipy.linalg import orthogonal_procrustes
from tqdm import tqdm
from typing import Dict, List, Set, Tuple

# ==============================================================================
# CONFIGURATION
# ==============================================================================

class Config:
    """
    Experiment configuration parameters.
    
    HIGH PRECISION VERSION: 1000 samples for robust statistics.
    Expected runtime: ~8-12 minutes on Colab T4 GPU.
    """
    # Models to compare
    MODEL_A = "gpt2"           # 124M parameters, 12 layers
    MODEL_B = "gpt2-medium"    # 355M parameters, 24 layers
    
    # Layers to analyze - ALL 12 layers of GPT-2 (0-11)
    LAYERS_TO_ANALYZE = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    
    # Data parameters - HIGH PRECISION
    NUM_SAMPLES = 1000         # 1000 sequences for robust statistics
    SEQ_LENGTH = 64            # Tokens per sequence
    BATCH_SIZE = 32            # Larger batch for efficiency
    
    # Procrustes subsampling
    # Full data: 1000 * 64 = 64,000 rows
    # We use 15,000 for good approximation while staying tractable
    SUBSAMPLE = 15000
    
    # Spectral analysis parameters
    NUM_SPECTRAL_SAMPLES = 200 # More samples for accurate FFT
    NUM_FREQUENCY_BANDS = 7    # Following AKIRA's 7-band architecture
    
    # Random seed for reproducibility
    SEED = 42


# ==============================================================================
# SETUP
# ==============================================================================

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# Set random seeds for reproducibility
torch.manual_seed(Config.SEED)
np.random.seed(Config.SEED)

# Import transformer_lens (must be installed: pip install transformer-lens)
from transformer_lens import HookedTransformer

print("=" * 70)
print("EXPERIMENT 000 - PHASE 5: SPECTRAL ANALYSIS OF ACTION QUANTA")
print("=" * 70)
print(f"Model A: {Config.MODEL_A}")
print(f"Model B: {Config.MODEL_B}")
print(f"Layers: {Config.LAYERS_TO_ANALYZE}")
print(f"Samples: {Config.NUM_SAMPLES}")
print()


# ==============================================================================
# MODEL LOADING
# ==============================================================================

print("Loading models...")
print(f"  Loading {Config.MODEL_A}...")
model_a = HookedTransformer.from_pretrained(Config.MODEL_A, device=device)

print(f"  Loading {Config.MODEL_B}...")
model_b = HookedTransformer.from_pretrained(Config.MODEL_B, device=device)

print("  Models loaded successfully.")
print()


# ==============================================================================
# ACTIVATION EXTRACTION
# ==============================================================================

def extract_activations(
    model: HookedTransformer,
    tokens: torch.Tensor,
    layers: List[int]
) -> Dict[int, np.ndarray]:
    """
    Extract MLP post-activation values from specified layers.
    
    Args:
        model: The transformer model to extract from
        tokens: Input token tensor of shape [num_samples, seq_length]
        layers: List of layer indices to extract from
    
    Returns:
        Dictionary mapping layer index to activation array of shape
        [num_samples, seq_length, hidden_dim]
    
    The MLP post-activations are chosen because they represent the
    "output" of each layer's feedforward computation, which is where
    interpretable features are most likely to emerge.
    """
    activations = {layer: [] for layer in layers}
    num_batches = (len(tokens) + Config.BATCH_SIZE - 1) // Config.BATCH_SIZE
    
    for batch_idx in tqdm(range(num_batches), desc="Extracting"):
        start_idx = batch_idx * Config.BATCH_SIZE
        end_idx = min(start_idx + Config.BATCH_SIZE, len(tokens))
        batch_tokens = tokens[start_idx:end_idx].to(device)
        
        with torch.no_grad():
            _, cache = model.run_with_cache(batch_tokens)
        
        for layer in layers:
            hook_name = f"blocks.{layer}.mlp.hook_post"
            activations[layer].append(cache[hook_name].cpu().numpy())
        
        # Clean up to prevent memory accumulation
        del cache
        if device == "cuda":
            torch.cuda.empty_cache()
    
    # Concatenate all batches
    return {layer: np.concatenate(activations[layer], axis=0) for layer in layers}


# ==============================================================================
# ACTION QUANTA DETECTION
# ==============================================================================

def detect_action_quanta(
    acts_a: Dict[int, np.ndarray],
    acts_b: Dict[int, np.ndarray],
    layer: int
) -> Tuple[Set[int], np.ndarray, float]:
    """
    Detect Action Quanta using Procrustes alignment and correlation analysis.
    
    This method:
    1. Flattens activations across samples and sequence positions
    2. Subsamples for computational tractability
    3. Finds optimal orthogonal transformation (Procrustes) aligning A to B
    4. Computes per-neuron correlation after alignment
    5. Identifies neurons with correlation > mean + 2*std as AQ candidates
    
    Args:
        acts_a: Activations from model A
        acts_b: Activations from model B
        layer: Layer index to analyze
    
    Returns:
        aq_indices: Set of neuron indices identified as AQ
        correlations: Array of per-neuron correlations
        threshold: The statistical threshold used
    """
    # Flatten: [samples, seq, dim] -> [samples*seq, dim]
    a = acts_a[layer].reshape(-1, acts_a[layer].shape[-1])
    b = acts_b[layer].reshape(-1, acts_b[layer].shape[-1])
    
    # Match dimensions (GPT-2 has 3072, GPT-2-medium has 4096)
    min_dim = min(a.shape[1], b.shape[1])
    a = a[:, :min_dim]
    b = b[:, :min_dim]
    
    # Subsample rows for Procrustes (key speedup)
    # Without this, the computation would be intractable
    n_rows = a.shape[0]
    subsample_indices = np.random.choice(
        n_rows, 
        min(Config.SUBSAMPLE, n_rows), 
        replace=False
    )
    a_subsample = a[subsample_indices]
    b_subsample = b[subsample_indices]
    
    # Z-score normalize the subsampled data
    a_subsample_norm = (a_subsample - a_subsample.mean(axis=0)) / (a_subsample.std(axis=0) + 1e-8)
    b_subsample_norm = (b_subsample - b_subsample.mean(axis=0)) / (b_subsample.std(axis=0) + 1e-8)
    
    # Orthogonal Procrustes: find rotation R such that A @ R approximates B
    # This aligns the representation spaces of the two models
    rotation_matrix, _ = orthogonal_procrustes(a_subsample_norm, b_subsample_norm)
    
    # Apply rotation to full normalized data
    a_normalized = (a - a.mean(axis=0)) / (a.std(axis=0) + 1e-8)
    b_normalized = (b - b.mean(axis=0)) / (b.std(axis=0) + 1e-8)
    a_aligned = a_normalized @ rotation_matrix
    
    # Compute per-neuron correlation (vectorized for speed)
    # Correlation = sum(A * B) / (norm(A) * norm(B))
    a_centered = a_aligned - a_aligned.mean(axis=0)
    b_centered = b_normalized - b_normalized.mean(axis=0)
    a_unit = a_centered / (np.linalg.norm(a_centered, axis=0) + 1e-8)
    b_unit = b_centered / (np.linalg.norm(b_centered, axis=0) + 1e-8)
    correlations = np.sum(a_unit * b_unit, axis=0)
    
    # Statistical threshold: mean + 2*std
    # This identifies neurons that are significantly more correlated than average
    threshold = np.mean(correlations) + 2 * np.std(correlations)
    aq_mask = correlations > threshold
    aq_indices = set(np.where(aq_mask)[0])
    
    return aq_indices, correlations, threshold


# ==============================================================================
# SPECTRAL ANALYSIS
# ==============================================================================

def compute_spectral_centroids(activations: np.ndarray) -> np.ndarray:
    """
    Compute spectral centroid for each neuron.
    
    The spectral centroid is the "center of mass" of the frequency spectrum,
    weighted by power. A lower centroid means the neuron's activation pattern
    is dominated by low frequencies (slow-changing, stable patterns).
    
    Method:
    1. For each sample, compute FFT of each neuron's activation across sequence
    2. Compute power spectrum (magnitude squared)
    3. Calculate centroid = sum(freq * power) / sum(power)
    4. Average across samples
    
    Args:
        activations: Array of shape [num_samples, seq_length, num_neurons]
    
    Returns:
        centroids: Array of shape [num_neurons] with spectral centroid per neuron
    """
    num_samples, seq_length, num_neurons = activations.shape
    
    # Use subset of samples for computational efficiency
    num_samples_to_use = min(Config.NUM_SPECTRAL_SAMPLES, num_samples)
    
    spectra_accumulator = []
    
    for sample_idx in range(num_samples_to_use):
        # Get activations for this sample: [seq_length, num_neurons]
        sample_activations = activations[sample_idx]
        
        # FFT along sequence dimension
        fft_result = fft(sample_activations, axis=0)
        
        # Power spectrum (only positive frequencies)
        power_spectrum = np.abs(fft_result[:seq_length // 2]) ** 2
        
        spectra_accumulator.append(power_spectrum)
    
    # Average power spectrum across samples: [seq_length//2, num_neurons]
    mean_spectrum = np.mean(spectra_accumulator, axis=0)
    
    # Transpose to [num_neurons, num_freqs]
    mean_spectrum = mean_spectrum.T
    
    # Frequency bins (positive frequencies only)
    frequencies = np.abs(fftfreq(seq_length, d=1.0)[:seq_length // 2])
    
    # Compute spectral centroid for each neuron
    # Centroid = sum(freq * power) / sum(power)
    total_power = np.sum(mean_spectrum, axis=1, keepdims=True)
    total_power = np.maximum(total_power, 1e-10)  # Avoid division by zero
    
    centroids = np.sum(mean_spectrum * frequencies, axis=1) / total_power.squeeze()
    
    return centroids


def assign_frequency_bands(centroids: np.ndarray, num_bands: int = 7) -> np.ndarray:
    """
    Assign each neuron to a frequency band based on its spectral centroid.
    
    Uses percentile-based binning to create bands of roughly equal size.
    Band 0 = lowest frequency, Band (num_bands-1) = highest frequency.
    
    Args:
        centroids: Array of spectral centroid values
        num_bands: Number of bands to create
    
    Returns:
        bands: Array of band assignments (0 to num_bands-1)
    """
    percentiles = np.linspace(0, 100, num_bands + 1)
    thresholds = np.percentile(centroids, percentiles)
    bands = np.digitize(centroids, thresholds[1:-1])
    return bands


# ==============================================================================
# TEXT GENERATION
# ==============================================================================

def generate_diverse_text_samples(num_samples: int) -> List[str]:
    """
    Generate diverse text samples for semantic AQ detection.
    
    Unlike repeated short sentences (which cause Procrustes to fail due to
    degenerate data), this function creates genuinely varied text by combining
    different topic prefixes with varied completions.
    
    The topics span multiple domains (science, economics, arts, technology, etc.)
    to ensure broad semantic coverage.
    
    Args:
        num_samples: Number of text samples to generate
    
    Returns:
        List of diverse text strings
    """
    topic_prefixes = [
        # Science
        "The scientist discovered that the fundamental particles exhibit",
        "In the laboratory experiment researchers observed unexpected",
        "The chemical reaction produced a previously unknown compound with",
        "Quantum mechanics predicts that entangled particles will show",
        "The protein structure reveals important information about cellular",
        "Genetic analysis of the sample indicated ancestral connections to",
        "The enzyme catalyzes reactions by lowering the activation energy of",
        "Thermodynamic principles govern the efficiency of heat engines through",
        
        # Technology
        "The algorithm computed an optimal solution for the complex problem of",
        "Machine learning models can identify subtle patterns in datasets containing",
        "The neural network architecture was designed to process sequential data with",
        "Database optimization techniques reduced query execution time for",
        "The compiler generates efficient machine code by analyzing the structure of",
        "Cryptographic protocols ensure secure communication between parties using",
        "The operating system manages memory allocation to prevent conflicts between",
        "Network protocols define the rules for data transmission across distributed",
        
        # Economics and Finance
        "The stock market experienced significant volatility following announcements about",
        "Economic indicators suggest potential growth in sectors related to",
        "Market analysis reveals correlation patterns between commodity prices and",
        "The central bank adjusted interest rates to address concerns about",
        "Investment strategies should consider risk diversification across multiple",
        "Supply chain disruptions affected manufacturing output in industries producing",
        
        # Nature and Environment
        "Climate data collected over decades indicates significant changes in",
        "The ecosystem depends on complex interactions between species including",
        "Geological surveys of the region revealed mineral deposits containing",
        "Ocean currents influence weather patterns by distributing thermal energy across",
        "Forest conservation efforts aim to protect biodiversity by preserving habitats for",
        
        # History and Society
        "Archaeological evidence suggests ancient civilizations developed sophisticated",
        "Historical records from the period describe significant cultural exchanges between",
        "The political landscape shifted dramatically following events that occurred in",
        "Social movements throughout history have challenged established norms regarding",
        "Legal precedents established in landmark cases continue to influence decisions about",
        
        # Arts and Culture
        "The painting depicts a scene that reflects the artistic movement known as",
        "Musical composition techniques evolved during the period to incorporate elements of",
        "Literary analysis reveals themes of identity and belonging in works exploring",
        "Architectural styles from the era demonstrate influences from traditions including",
        "The theatrical performance incorporated innovative staging techniques that emphasized",
        
        # Space and Astronomy
        "Astronomical observations using advanced telescopes revealed new information about",
        "The spacecraft mission collected data that will help scientists understand",
        "Stellar evolution models predict that stars of this mass will eventually become",
        "Exoplanet detection methods have identified potentially habitable worlds orbiting",
        
        # Medicine and Health
        "Clinical trials demonstrated the effectiveness of new treatments for patients with",
        "Medical imaging techniques allow doctors to visualize internal structures including",
        "Public health initiatives focus on prevention strategies for diseases caused by",
        "Pharmaceutical research aims to develop targeted therapies that minimize side effects of",
        
        # Philosophy and Cognition
        "Philosophical arguments have long debated the nature of consciousness and its relationship to",
        "Cognitive research suggests that human memory operates through processes involving",
        "Ethical considerations must guide decisions about emerging technologies that affect",
        "Linguistic analysis demonstrates how language shapes thought patterns related to",
    ]
    
    # Generate samples by cycling through prefixes with varied completions
    samples = []
    completion_variants = [
        "various interconnected phenomena observed in nature",
        "fundamental aspects of the underlying system architecture",
        "complex relationships that emerge from simple interactions",
        "previously unexplored regions of the parameter space",
        "mechanisms that operate across multiple scales simultaneously",
    ]
    
    for i in range(num_samples):
        prefix_idx = i % len(topic_prefixes)
        variant_idx = i % len(completion_variants)
        sample = f"{topic_prefixes[prefix_idx]} {completion_variants[variant_idx]}."
        samples.append(sample)
    
    return samples


# ==============================================================================
# MAIN EXPERIMENT
# ==============================================================================

# ------------------------------------------------------------------------------
# PHASE 1: Random Token Analysis (Structural AQ)
# ------------------------------------------------------------------------------

print("=" * 70)
print("PHASE 1: STRUCTURAL AQ DETECTION (Random Tokens)")
print("=" * 70)
print()
print("Random tokens test for INPUT-INDEPENDENT universal features.")
print("These are structural patterns like positional encoding, layer norms, etc.")
print()

# Generate random tokens (avoiding special tokens at edges of vocabulary)
vocab_size = model_a.tokenizer.vocab_size
random_tokens = torch.randint(
    1000,                    # Avoid special tokens at start
    vocab_size - 1000,       # Avoid special tokens at end
    (Config.NUM_SAMPLES, Config.SEQ_LENGTH)
)

print(f"Generated {Config.NUM_SAMPLES} random token sequences of length {Config.SEQ_LENGTH}")
print()

# Extract activations
print("Extracting activations from Model A (GPT-2)...")
acts_a_random = extract_activations(model_a, random_tokens, Config.LAYERS_TO_ANALYZE)

print("Extracting activations from Model B (GPT-2-medium)...")
acts_b_random = extract_activations(model_b, random_tokens, Config.LAYERS_TO_ANALYZE)

# Detect structural AQ
print()
print("Detecting structural Action Quanta...")
structural_aq = {}
structural_correlations = {}

for layer in Config.LAYERS_TO_ANALYZE:
    aq_indices, correlations, threshold = detect_action_quanta(acts_a_random, acts_b_random, layer)
    structural_aq[layer] = aq_indices
    structural_correlations[layer] = correlations
    print(f"  Layer {layer}: {len(aq_indices)} structural AQ (threshold={threshold:.4f})")

print()

# ------------------------------------------------------------------------------
# PHASE 2: Text Analysis (Semantic AQ)
# ------------------------------------------------------------------------------

print("=" * 70)
print("PHASE 2: SEMANTIC AQ DETECTION (Diverse Text)")
print("=" * 70)
print()
print("Real text tests for MEANING-DEPENDENT universal features.")
print("These are semantic patterns that emerge from language understanding.")
print()

# Generate diverse text samples
text_samples = generate_diverse_text_samples(Config.NUM_SAMPLES)
print(f"Generated {len(text_samples)} diverse text samples")
print(f"Sample topics: science, technology, economics, nature, history, arts, etc.")
print()

# Tokenize
text_tokens = model_a.tokenizer(
    text_samples,
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length=Config.SEQ_LENGTH
)["input_ids"]

# Extract activations
print("Extracting activations from Model A (GPT-2)...")
acts_a_text = extract_activations(model_a, text_tokens, Config.LAYERS_TO_ANALYZE)

print("Extracting activations from Model B (GPT-2-medium)...")
acts_b_text = extract_activations(model_b, text_tokens, Config.LAYERS_TO_ANALYZE)

# Detect semantic AQ
print()
print("Detecting semantic Action Quanta...")
semantic_aq = {}
semantic_correlations = {}

for layer in Config.LAYERS_TO_ANALYZE:
    aq_indices, correlations, threshold = detect_action_quanta(acts_a_text, acts_b_text, layer)
    semantic_aq[layer] = aq_indices
    semantic_correlations[layer] = correlations
    print(f"  Layer {layer}: {len(aq_indices)} semantic AQ (threshold={threshold:.4f})")

print()

# ------------------------------------------------------------------------------
# PHASE 3: Universal AQ (Overlap Analysis)
# ------------------------------------------------------------------------------

print("=" * 70)
print("PHASE 3: UNIVERSAL AQ IDENTIFICATION (Overlap)")
print("=" * 70)
print()
print("Universal AQ are neurons that appear in BOTH structural AND semantic sets.")
print("These are the most robust candidates for fundamental features.")
print()

universal_aq = {}
for layer in Config.LAYERS_TO_ANALYZE:
    universal_aq[layer] = structural_aq[layer] & semantic_aq[layer]
    
    struct_count = len(structural_aq[layer])
    sem_count = len(semantic_aq[layer])
    univ_count = len(universal_aq[layer])
    
    print(f"Layer {layer}:")
    print(f"  Structural AQ: {struct_count}")
    print(f"  Semantic AQ:   {sem_count}")
    print(f"  Universal AQ:  {univ_count} ({100*univ_count/max(struct_count,1):.1f}% of structural)")
    print()

total_universal = sum(len(universal_aq[l]) for l in Config.LAYERS_TO_ANALYZE)
print(f"TOTAL UNIVERSAL AQ: {total_universal}")
print()

# ------------------------------------------------------------------------------
# PHASE 4: Spectral Analysis
# ------------------------------------------------------------------------------

print("=" * 70)
print("PHASE 4: SPECTRAL ANALYSIS")
print("=" * 70)
print()
print("AKIRA PREDICTION: Universal AQ should be concentrated in LOW frequency bands")
print()
print("Spectral centroid measures the dominant frequency of each neuron's activation")
print("pattern across the sequence. Lower centroid = more low-frequency dominant.")
print()

# Compute spectral centroids using random activations (for structural analysis)
spectral_results = {}

for layer in Config.LAYERS_TO_ANALYZE:
    print(f"Analyzing Layer {layer}...")
    
    # Compute centroids for all neurons
    centroids = compute_spectral_centroids(acts_a_random[layer])
    bands = assign_frequency_bands(centroids, Config.NUM_FREQUENCY_BANDS)
    
    # Separate AQ and non-AQ neurons
    structural_aq_indices = list(structural_aq[layer])
    semantic_aq_indices = list(semantic_aq[layer])
    universal_aq_indices = list(universal_aq[layer])
    all_indices = set(range(len(centroids)))
    non_aq_indices = list(all_indices - structural_aq[layer] - semantic_aq[layer])
    
    # Compute statistics
    results = {
        'centroids': centroids,
        'bands': bands,
        'structural_aq': {
            'indices': structural_aq_indices,
            'mean_centroid': np.mean(centroids[structural_aq_indices]) if structural_aq_indices else 0,
        },
        'semantic_aq': {
            'indices': semantic_aq_indices,
            'mean_centroid': np.mean(centroids[semantic_aq_indices]) if semantic_aq_indices else 0,
        },
        'universal_aq': {
            'indices': universal_aq_indices,
            'mean_centroid': np.mean(centroids[universal_aq_indices]) if universal_aq_indices else 0,
        },
        'non_aq': {
            'indices': non_aq_indices,
            'mean_centroid': np.mean(centroids[non_aq_indices]) if non_aq_indices else 0,
        }
    }
    
    spectral_results[layer] = results

print()

# ------------------------------------------------------------------------------
# RESULTS SUMMARY
# ------------------------------------------------------------------------------

print("=" * 70)
print("SPECTRAL ANALYSIS RESULTS")
print("=" * 70)
print()

for layer in Config.LAYERS_TO_ANALYZE:
    results = spectral_results[layer]
    
    struct_mean = results['structural_aq']['mean_centroid']
    sem_mean = results['semantic_aq']['mean_centroid']
    univ_mean = results['universal_aq']['mean_centroid']
    non_aq_mean = results['non_aq']['mean_centroid']
    
    struct_count = len(results['structural_aq']['indices'])
    sem_count = len(results['semantic_aq']['indices'])
    univ_count = len(results['universal_aq']['indices'])
    
    print(f"LAYER {layer}")
    print("-" * 40)
    print(f"  Structural AQ ({struct_count} neurons):")
    print(f"    Mean centroid: {struct_mean:.4f}")
    print(f"    vs Non-AQ:     {struct_mean - non_aq_mean:+.4f}")
    
    print(f"  Semantic AQ ({sem_count} neurons):")
    print(f"    Mean centroid: {sem_mean:.4f}")
    print(f"    vs Non-AQ:     {sem_mean - non_aq_mean:+.4f}")
    
    if univ_count > 0:
        print(f"  Universal AQ ({univ_count} neurons):")
        print(f"    Mean centroid: {univ_mean:.4f}")
        print(f"    vs Non-AQ:     {univ_mean - non_aq_mean:+.4f}")
    else:
        print(f"  Universal AQ: None found")
    
    print(f"  Non-AQ baseline: {non_aq_mean:.4f}")
    print()

# ------------------------------------------------------------------------------
# VERDICT
# ------------------------------------------------------------------------------

print("=" * 70)
print("VERDICT: AKIRA SPECTRAL HYPOTHESIS")
print("=" * 70)
print()

confirmed_layers = 0
total_layers = len(Config.LAYERS_TO_ANALYZE)

for layer in Config.LAYERS_TO_ANALYZE:
    results = spectral_results[layer]
    struct_mean = results['structural_aq']['mean_centroid']
    non_aq_mean = results['non_aq']['mean_centroid']
    struct_count = len(results['structural_aq']['indices'])
    
    if struct_count > 0:
        diff = struct_mean - non_aq_mean
        if diff < 0:
            status = "CONFIRMED"
            confirmed_layers += 1
            pct = abs(diff) / non_aq_mean * 100
            detail = f"AQ are {pct:.1f}% more low-frequency"
        else:
            status = "NOT CONFIRMED"
            detail = "AQ are higher frequency than non-AQ"
    else:
        status = "INSUFFICIENT DATA"
        detail = "No AQ found for comparison"
    
    print(f"Layer {layer}: {status}")
    print(f"  {detail}")
    print()

print("-" * 40)
print(f"Overall: {confirmed_layers}/{total_layers} layers confirm the hypothesis")
print()

if confirmed_layers == total_layers:
    print("STRONG SUPPORT: All layers show AQ concentrated in low frequencies")
elif confirmed_layers > 0:
    print("PARTIAL SUPPORT: Some layers show the predicted pattern")
    print("This may indicate layer-specific AQ properties worthy of further study")
else:
    print("NOT SUPPORTED: No layers show the predicted low-frequency concentration")
    print("The spectral hypothesis may need refinement")

print()
print("=" * 70)
print("EXPERIMENT COMPLETE")
print("=" * 70)
