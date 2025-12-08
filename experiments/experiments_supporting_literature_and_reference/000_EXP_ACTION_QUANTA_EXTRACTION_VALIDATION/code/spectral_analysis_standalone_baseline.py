# ============================================================
# SPECTRAL ANALYSIS - COMPLETE STANDALONE
# ============================================================

import torch
import numpy as np
from scipy.fft import fft, fftfreq
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Config
LAYERS = [3, 5, 7]
NUM_SAMPLES = 500
SEQ_LENGTH = 64
BATCH_SIZE = 16
NUM_BANDS = 7

torch.manual_seed(42)
np.random.seed(42)

# Load model
from transformer_lens import HookedTransformer
print("Loading GPT-2...")
model = HookedTransformer.from_pretrained("gpt2", device=device)

# Generate random tokens
print(f"Generating {NUM_SAMPLES} random token sequences...")
vocab_size = model.tokenizer.vocab_size
tokens = torch.randint(1000, vocab_size - 1000, (NUM_SAMPLES, SEQ_LENGTH))

# Extract activations
print("Extracting activations...")
activations = {layer: [] for layer in LAYERS}
num_batches = (len(tokens) + BATCH_SIZE - 1) // BATCH_SIZE

for batch_idx in tqdm(range(num_batches), desc="Extracting"):
    start = batch_idx * BATCH_SIZE
    end = min(start + BATCH_SIZE, len(tokens))
    batch_tokens = tokens[start:end].to(device)
    
    with torch.no_grad():
        _, cache = model.run_with_cache(batch_tokens)
    
    for layer in LAYERS:
        hook_name = f"blocks.{layer}.mlp.hook_post"
        if hook_name in cache:
            activations[layer].append(cache[hook_name].cpu().numpy())
    
    del cache
    if device == "cuda":
        torch.cuda.empty_cache()

for layer in LAYERS:
    activations[layer] = np.concatenate(activations[layer], axis=0)
    print(f"Layer {layer} shape: {activations[layer].shape}")

# Spectral analysis functions
def compute_neuron_spectrum(acts):
    num_samples, seq_len, num_neurons = acts.shape
    spectra_list = []
    for i in range(num_samples):
        fft_result = fft(acts[i], axis=0)
        power = np.abs(fft_result[:seq_len//2]) ** 2
        spectra_list.append(power)
    return np.mean(spectra_list, axis=0).T, fftfreq(seq_len, d=1.0)[:seq_len//2]

def compute_spectral_centroid(spectra, freqs):
    freqs_pos = np.abs(freqs)
    total_power = np.maximum(np.sum(spectra, axis=1, keepdims=True), 1e-10)
    return np.sum(spectra * freqs_pos, axis=1) / total_power.squeeze()

def assign_bands(centroids, num_bands=7):
    thresholds = np.percentile(centroids, np.linspace(0, 100, num_bands + 1))
    return np.digitize(centroids, thresholds[1:-1])

# Run spectral analysis
print("\n" + "="*70)
print("SPECTRAL ANALYSIS: Frequency distribution of neurons")
print("="*70)
print("Lower centroid = more low-frequency dominant")
print("AKIRA predicts: Universal features live in low-frequency bands\n")

for layer in LAYERS:
    acts = activations[layer]
    spectra, freqs = compute_neuron_spectrum(acts)
    centroids = compute_spectral_centroid(spectra, freqs)
    bands = assign_bands(centroids, NUM_BANDS)
    
    print(f"Layer {layer}:")
    print(f"  Mean centroid: {np.mean(centroids):.4f}")
    print(f"  Std centroid:  {np.std(centroids):.4f}")
    print(f"  Min centroid:  {np.min(centroids):.4f}")
    print(f"  Max centroid:  {np.max(centroids):.4f}")
    
    band_counts = [np.sum(bands == b) for b in range(NUM_BANDS)]
    low_freq_pct = sum(band_counts[:3]) / len(bands) * 100
    high_freq_pct = sum(band_counts[4:]) / len(bands) * 100
    
    print(f"  Band distribution: {band_counts}")
    print(f"  Low-freq (0-2):  {low_freq_pct:.1f}%")
    print(f"  High-freq (4-6): {high_freq_pct:.1f}%")
    print()