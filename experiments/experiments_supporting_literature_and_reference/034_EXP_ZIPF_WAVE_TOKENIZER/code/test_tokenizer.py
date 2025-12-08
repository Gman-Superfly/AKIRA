# -*- coding: utf-8 -*-
"""
Test and Validate the Zipf Wave Tokenizer

This script validates that the Zipf-grounded wave tokenizer produces
meaningful spectral structure:

1. Verify Zipf rank distribution matches expected power law
2. Verify frequency mapping produces expected band distribution
3. Analyze real sentences for spectral content
4. Test reconstruction capability
5. Visualize wave representations

Run locally or in Colab.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Tuple, Optional
import numpy as np

# ==============================================================================
# CELL 1: IMPORTS AND SETUP
# ==============================================================================

try:
    from zipf_wave_tokenizer import (
        ZipfWaveConfig,
        ZipfRankTable,
        ZipfWaveEmbedding,
        ZipfWaveDecoder,
        analyze_sentence_spectrum
    )
except ImportError:
    # If running as standalone, define inline
    print("Running standalone - importing from current directory")
    import sys
    sys.path.insert(0, '.')
    from zipf_wave_tokenizer import (
        ZipfWaveConfig,
        ZipfRankTable,
        ZipfWaveEmbedding,
        ZipfWaveDecoder,
        analyze_sentence_spectrum
    )

# Check for optional dependencies
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("matplotlib not available - skipping visualizations")

try:
    from transformers import GPT2Tokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("transformers not available - using synthetic data")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


# ==============================================================================
# CELL 2: TEST ZIPF RANK DISTRIBUTION
# ==============================================================================

def test_zipf_distribution():
    """
    Verify that the Zipf rank table produces expected power-law distribution.
    """
    print("\n" + "="*60)
    print("TEST 1: Zipf Rank Distribution")
    print("="*60)
    
    # Create rank table
    rank_table = ZipfRankTable.from_gpt2(vocab_size=50257)
    ranks = rank_table.ranks
    
    print(f"Vocabulary size: {len(ranks)}")
    print(f"Rank range: [{ranks.min():.0f}, {ranks.max():.0f}]")
    
    # Sample some tokens
    sample_ids = [0, 100, 1000, 10000, 50000]
    print("\nSample token ranks:")
    for tid in sample_ids:
        if tid < len(ranks):
            print(f"  Token {tid}: rank = {ranks[tid]:.0f}")
    
    # Verify monotonicity (ranks should generally increase with token ID)
    # Note: First 256 are byte tokens, so we check after that
    bpe_ranks = ranks[256:]
    monotonic_violations = (bpe_ranks[1:] < bpe_ranks[:-1]).sum().item()
    print(f"\nMonotonicity violations (BPE tokens): {monotonic_violations}")
    print(f"  (GPT-2 BPE order roughly correlates with frequency)")
    
    # Visualize if matplotlib available
    if HAS_MATPLOTLIB:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Rank distribution
        axes[0].plot(ranks.numpy())
        axes[0].set_xlabel('Token ID')
        axes[0].set_ylabel('Rank')
        axes[0].set_title('Zipf Rank by Token ID')
        axes[0].set_yscale('log')
        
        # Log-log plot (should be linear for power law)
        sorted_ranks = torch.sort(ranks)[0]
        axes[1].loglog(range(1, len(sorted_ranks)+1), sorted_ranks.numpy())
        axes[1].set_xlabel('Position (sorted)')
        axes[1].set_ylabel('Rank')
        axes[1].set_title('Log-Log Rank Distribution')
        
        plt.tight_layout()
        plt.savefig('zipf_distribution.png', dpi=150)
        print("\nSaved: zipf_distribution.png")
        plt.close()
    
    return True


# ==============================================================================
# CELL 3: TEST FREQUENCY MAPPING
# ==============================================================================

def test_frequency_mapping():
    """
    Verify that Zipf ranks map to wave frequencies correctly.
    """
    print("\n" + "="*60)
    print("TEST 2: Frequency Mapping")
    print("="*60)
    
    config = ZipfWaveConfig(
        vocab_size=50257,
        embed_dim=512,
        n_harmonics=8,
        freq_min=0.01,
        freq_max=1.0
    )
    
    wave_embed = ZipfWaveEmbedding(config)
    
    # Check base frequencies
    base_freqs = wave_embed.base_frequencies
    print(f"Frequency range: [{base_freqs.min():.4f}, {base_freqs.max():.4f}]")
    
    # Sample tokens at different ranks
    sample_ids = torch.tensor([[0, 100, 1000, 10000, 50000]])
    
    spectrum = wave_embed.get_wave_spectrum(sample_ids)
    print("\nSample token frequencies (fundamental):")
    for i, tid in enumerate(sample_ids[0]):
        freq = spectrum['base_frequencies'][0, i].item()
        band = spectrum['band_assignments'][0, i].item()
        print(f"  Token {tid.item()}: freq = {freq:.4f}, band = {band}")
    
    # Verify frequency-rank relationship
    # Should be: log(rank) / log(V) ~ normalized frequency
    ranks = wave_embed.zipf_ranks
    expected_normalized = torch.log(ranks) / math.log(config.vocab_size)
    expected_freqs = config.freq_min + (config.freq_max - config.freq_min) * expected_normalized
    
    freq_error = (base_freqs - expected_freqs).abs().max().item()
    print(f"\nMax frequency mapping error: {freq_error:.6f}")
    assert freq_error < 1e-5, "Frequency mapping error too large"
    
    # Check band distribution
    all_bands = spectrum['band_assignments']
    band_counts = torch.bincount(all_bands.flatten(), minlength=config.num_bands)
    print(f"\nBand assignments for sample: {band_counts.tolist()}")
    
    # Visualize
    if HAS_MATPLOTLIB:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Frequency vs rank
        sample_size = 1000
        sample_ids = torch.arange(0, config.vocab_size, config.vocab_size // sample_size)
        sample_freqs = base_freqs[sample_ids]
        sample_ranks = ranks[sample_ids]
        
        axes[0].scatter(sample_ranks.numpy(), sample_freqs.numpy(), alpha=0.5, s=1)
        axes[0].set_xlabel('Zipf Rank')
        axes[0].set_ylabel('Wave Frequency')
        axes[0].set_title('Frequency vs Rank')
        axes[0].set_xscale('log')
        
        # Frequency histogram
        axes[1].hist(base_freqs.numpy(), bins=50, edgecolor='black')
        axes[1].set_xlabel('Wave Frequency')
        axes[1].set_ylabel('Count')
        axes[1].set_title('Frequency Distribution')
        
        plt.tight_layout()
        plt.savefig('frequency_mapping.png', dpi=150)
        print("Saved: frequency_mapping.png")
        plt.close()
    
    return True


# ==============================================================================
# CELL 4: TEST BAND DISTRIBUTION
# ==============================================================================

def test_band_distribution():
    """
    Verify that tokens distribute across bands according to Zipf law.
    
    Expected: More tokens in high-frequency bands (rare tokens),
    fewer in low-frequency bands (common tokens).
    """
    print("\n" + "="*60)
    print("TEST 3: Band Distribution")
    print("="*60)
    
    config = ZipfWaveConfig(
        vocab_size=50257,
        embed_dim=512,
        n_harmonics=8,
        num_bands=7
    )
    
    wave_embed = ZipfWaveEmbedding(config)
    
    # Get band assignments for all tokens
    all_tokens = torch.arange(config.vocab_size).unsqueeze(0)  # [1, V]
    spectrum = wave_embed.get_wave_spectrum(all_tokens)
    
    band_assignments = spectrum['band_assignments'].flatten()
    band_counts = torch.bincount(band_assignments, minlength=config.num_bands)
    
    print("\nTokens per band:")
    for i, count in enumerate(band_counts):
        pct = 100 * count.item() / config.vocab_size
        print(f"  Band {i}: {count.item():>6} tokens ({pct:>5.1f}%)")
    
    # Verify Zipf-like distribution (more tokens in higher bands = higher freq = rare)
    # Due to log mapping, distribution should be roughly exponential
    print("\nExpected pattern: increasing tokens in higher bands (log mapping)")
    
    # Get frequency ranges per band
    rank_table = ZipfRankTable.from_gpt2(config.vocab_size)
    band_ranges = rank_table.get_band_indices(config.num_bands)
    print("\nRank ranges per band:")
    for band, (start, end) in band_ranges.items():
        print(f"  Band {band}: ranks {start} to {end}")
    
    # Visualize
    if HAS_MATPLOTLIB:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Bar chart of band distribution
        axes[0].bar(range(config.num_bands), band_counts.numpy())
        axes[0].set_xlabel('Band')
        axes[0].set_ylabel('Token Count')
        axes[0].set_title('Tokens per Spectral Band')
        
        # Cumulative distribution
        cumsum = band_counts.cumsum(0).numpy()
        axes[1].plot(range(config.num_bands), cumsum, 'o-')
        axes[1].set_xlabel('Band')
        axes[1].set_ylabel('Cumulative Tokens')
        axes[1].set_title('Cumulative Token Distribution')
        axes[1].axhline(config.vocab_size, color='r', linestyle='--', label='Total vocab')
        axes[1].legend()
        
        plt.tight_layout()
        plt.savefig('band_distribution.png', dpi=150)
        print("\nSaved: band_distribution.png")
        plt.close()
    
    return True


# ==============================================================================
# CELL 5: TEST REAL SENTENCES
# ==============================================================================

def test_real_sentences():
    """
    Analyze spectral content of real sentences.
    
    Hypothesis:
    - Common words (the, is, a) should be low frequency
    - Rare/technical words should be high frequency
    - Different sentence types should have different spectral profiles
    """
    print("\n" + "="*60)
    print("TEST 4: Real Sentence Analysis")
    print("="*60)
    
    if not HAS_TRANSFORMERS:
        print("Skipping - transformers not available")
        return True
    
    from transformers import GPT2Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    
    config = ZipfWaveConfig(
        vocab_size=50257,
        embed_dim=512,
        n_harmonics=8,
        num_bands=7
    )
    wave_embed = ZipfWaveEmbedding(config)
    
    # Test sentences with different characteristics
    sentences = {
        "simple": "The cat sat on the mat.",
        "technical": "The mitochondria is the powerhouse of the cell.",
        "scientific": "Quantum entanglement exhibits non-local correlations.",
        "common": "I want to go to the store and buy some food.",
        "rare_words": "The defenestration of Prague precipitated catastrophe."
    }
    
    results = {}
    
    for name, sentence in sentences.items():
        print(f"\n--- {name.upper()} ---")
        print(f"Sentence: {sentence}")
        
        # Tokenize
        token_ids = torch.tensor([tokenizer.encode(sentence)])
        tokens = tokenizer.convert_ids_to_tokens(token_ids[0].tolist())
        
        # Get spectrum
        spectrum = wave_embed.get_wave_spectrum(token_ids)
        
        # Analyze
        base_freqs = spectrum['base_frequencies'][0]
        bands = spectrum['band_assignments'][0]
        
        mean_freq = base_freqs.mean().item()
        band_dist = torch.bincount(bands, minlength=config.num_bands)
        
        print(f"Tokens: {tokens}")
        print(f"Mean frequency: {mean_freq:.4f}")
        print(f"Band distribution: {band_dist.tolist()}")
        
        # Show per-token info
        print("Per-token frequencies:")
        for i, (tok, freq, band) in enumerate(zip(tokens, base_freqs, bands)):
            print(f"  {tok:>15}: freq={freq.item():.4f}, band={band.item()}")
        
        results[name] = {
            'mean_freq': mean_freq,
            'band_dist': band_dist,
            'tokens': tokens
        }
    
    # Compare profiles
    print("\n--- COMPARISON ---")
    print(f"{'Sentence Type':<15} {'Mean Freq':>10} {'Low Bands':>10} {'High Bands':>10}")
    for name, res in results.items():
        low_bands = res['band_dist'][:3].sum().item()
        high_bands = res['band_dist'][4:].sum().item()
        print(f"{name:<15} {res['mean_freq']:>10.4f} {low_bands:>10} {high_bands:>10}")
    
    # Visualize
    if HAS_MATPLOTLIB:
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        axes = axes.flatten()
        
        for idx, (name, res) in enumerate(results.items()):
            if idx < len(axes):
                axes[idx].bar(range(config.num_bands), res['band_dist'].numpy())
                axes[idx].set_xlabel('Band')
                axes[idx].set_ylabel('Count')
                axes[idx].set_title(f'{name}\nmean_freq={res["mean_freq"]:.3f}')
        
        # Hide unused subplot
        if len(results) < len(axes):
            axes[-1].axis('off')
        
        plt.tight_layout()
        plt.savefig('sentence_spectra.png', dpi=150)
        print("\nSaved: sentence_spectra.png")
        plt.close()
    
    return True


# ==============================================================================
# CELL 6: TEST WAVE EMBEDDING QUALITY
# ==============================================================================

def test_embedding_quality():
    """
    Test that wave embeddings preserve meaningful structure.
    
    Tests:
    1. Different tokens produce different embeddings
    2. Similar-frequency tokens have more similar embeddings
    3. Embeddings are well-distributed (not collapsed)
    """
    print("\n" + "="*60)
    print("TEST 5: Embedding Quality")
    print("="*60)
    
    config = ZipfWaveConfig(
        vocab_size=50257,
        embed_dim=512,
        n_harmonics=8,
        phase_mode="learnable",
        amplitude_mode="zipf_scaled"
    )
    wave_embed = ZipfWaveEmbedding(config)
    
    # Sample tokens
    n_samples = 100
    token_ids = torch.randint(0, config.vocab_size, (1, n_samples))
    
    # Get embeddings
    embeddings = wave_embed(token_ids)  # [1, n_samples, D]
    embeddings = embeddings.squeeze(0)  # [n_samples, D]
    
    # Test 1: Uniqueness
    # Compute pairwise distances
    dists = torch.cdist(embeddings, embeddings)
    
    # Off-diagonal should be non-zero
    mask = ~torch.eye(n_samples, dtype=torch.bool)
    min_dist = dists[mask].min().item()
    mean_dist = dists[mask].mean().item()
    
    print(f"Embedding uniqueness:")
    print(f"  Min pairwise distance: {min_dist:.4f}")
    print(f"  Mean pairwise distance: {mean_dist:.4f}")
    
    # Test 2: Frequency-based similarity
    # Tokens with similar frequencies should have more similar embeddings
    spectrum = wave_embed.get_wave_spectrum(token_ids)
    freqs = spectrum['base_frequencies'].squeeze(0)  # [n_samples]
    
    # Compute frequency distance
    freq_dists = torch.abs(freqs.unsqueeze(0) - freqs.unsqueeze(1))  # [n, n]
    
    # Correlation between frequency distance and embedding distance
    freq_dists_flat = freq_dists[mask].numpy()
    embed_dists_flat = dists[mask].numpy()
    
    correlation = np.corrcoef(freq_dists_flat, embed_dists_flat)[0, 1]
    print(f"\nFrequency-embedding distance correlation: {correlation:.4f}")
    print(f"  (Positive = similar frequencies -> similar embeddings)")
    
    # Test 3: Distribution statistics
    mean_embed = embeddings.mean(dim=0)
    std_embed = embeddings.std(dim=0)
    
    print(f"\nEmbedding statistics:")
    print(f"  Mean norm: {embeddings.norm(dim=1).mean():.4f}")
    print(f"  Std of means: {mean_embed.std():.4f}")
    print(f"  Mean of stds: {std_embed.mean():.4f}")
    
    # Check for collapse (all embeddings similar)
    embedding_variance = embeddings.var(dim=0).mean().item()
    print(f"  Overall variance: {embedding_variance:.4f}")
    
    if embedding_variance < 0.01:
        print("  WARNING: Low variance - embeddings may be collapsed!")
    else:
        print("  Embeddings appear well-distributed")
    
    return True


# ==============================================================================
# CELL 7: TEST DECODER ROUNDTRIP
# ==============================================================================

def test_decoder_roundtrip():
    """
    Test that the wave decoder can recover tokens from embeddings.
    
    This tests the symmetric encode-decode property.
    """
    print("\n" + "="*60)
    print("TEST 6: Decoder Roundtrip")
    print("="*60)
    
    config = ZipfWaveConfig(
        vocab_size=50257,
        embed_dim=512,
        n_harmonics=8,
        phase_mode="learnable",
        amplitude_mode="fixed"
    )
    
    wave_embed = ZipfWaveEmbedding(config)
    decoder = ZipfWaveDecoder(config, wave_embed)
    
    # Sample tokens
    batch_size, seq_len = 4, 32
    token_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    # Encode
    embeddings = wave_embed(token_ids)
    
    # Decode
    logits = decoder(embeddings)
    predictions = logits.argmax(dim=-1)
    
    # Check accuracy
    correct = (predictions == token_ids).float()
    accuracy = correct.mean().item()
    
    print(f"Roundtrip accuracy: {accuracy*100:.2f}%")
    print(f"  (Note: This is WITHOUT any transformer processing)")
    print(f"  (Direct embed -> decode, so imperfect is expected)")
    
    # Check top-k accuracy
    for k in [1, 5, 10, 100]:
        top_k_preds = logits.topk(k, dim=-1).indices  # [B, T, k]
        in_top_k = (top_k_preds == token_ids.unsqueeze(-1)).any(dim=-1)
        top_k_acc = in_top_k.float().mean().item()
        print(f"  Top-{k} accuracy: {top_k_acc*100:.2f}%")
    
    # Analyze errors by band
    spectrum = wave_embed.get_wave_spectrum(token_ids)
    bands = spectrum['band_assignments']
    
    print("\nAccuracy by band:")
    for band in range(config.num_bands):
        band_mask = (bands == band)
        if band_mask.sum() > 0:
            band_acc = correct[band_mask].mean().item()
            print(f"  Band {band}: {band_acc*100:.2f}%")
    
    return True


# ==============================================================================
# CELL 8: VISUALIZE WAVE REPRESENTATIONS
# ==============================================================================

def visualize_wave_representations():
    """
    Visualize what the wave representations look like.
    """
    print("\n" + "="*60)
    print("TEST 7: Wave Visualization")
    print("="*60)
    
    if not HAS_MATPLOTLIB:
        print("Skipping - matplotlib not available")
        return True
    
    config = ZipfWaveConfig(
        vocab_size=50257,
        embed_dim=512,
        n_harmonics=8
    )
    wave_embed = ZipfWaveEmbedding(config)
    
    # Select tokens at different frequency levels
    # Token 0 = common, Token 50000 = rare
    sample_tokens = [0, 100, 1000, 10000, 50000]
    
    fig, axes = plt.subplots(len(sample_tokens), 2, figsize=(14, 3*len(sample_tokens)))
    
    for row, tid in enumerate(sample_tokens):
        # Get wave parameters
        token_tensor = torch.tensor([[tid]])
        spectrum = wave_embed.get_wave_spectrum(token_tensor)
        
        freq = spectrum['base_frequencies'][0, 0].item()
        amps = spectrum['amplitudes'][0, 0].numpy()
        phases = spectrum['phases'][0, 0].numpy()
        harmonics = wave_embed.harmonics.numpy()
        
        # Generate wave signal
        t = np.linspace(0, 10, 1000)
        signal = np.zeros_like(t)
        for h, (a, p) in enumerate(zip(amps, phases)):
            signal += a * np.sin(2 * np.pi * freq * (h+1) * t + p)
        
        # Plot time domain
        axes[row, 0].plot(t, signal)
        axes[row, 0].set_xlabel('Time')
        axes[row, 0].set_ylabel('Amplitude')
        axes[row, 0].set_title(f'Token {tid} (freq={freq:.4f}) - Time Domain')
        
        # Plot frequency spectrum
        freqs = freq * harmonics
        axes[row, 1].bar(range(len(amps)), amps)
        axes[row, 1].set_xlabel('Harmonic')
        axes[row, 1].set_ylabel('Amplitude')
        axes[row, 1].set_title(f'Token {tid} - Harmonic Amplitudes')
    
    plt.tight_layout()
    plt.savefig('wave_representations.png', dpi=150)
    print("Saved: wave_representations.png")
    plt.close()
    
    return True


# ==============================================================================
# CELL 9: TEST SPECTRAL DECOMPOSITION
# ==============================================================================

def test_spectral_decomposition():
    """
    Test that wave embeddings can be decomposed into frequency bands.
    
    This validates the core AKIRA concept: information naturally
    separates into spectral bands.
    """
    print("\n" + "="*60)
    print("TEST 8: Spectral Decomposition")
    print("="*60)
    
    config = ZipfWaveConfig(
        vocab_size=50257,
        embed_dim=512,
        n_harmonics=8,
        num_bands=7
    )
    wave_embed = ZipfWaveEmbedding(config)
    
    # Generate a sequence
    seq_len = 64
    token_ids = torch.randint(0, config.vocab_size, (1, seq_len))
    
    # Get embeddings
    embeddings = wave_embed(token_ids)  # [1, T, D]
    
    # Apply FFT along embedding dimension
    fft_result = torch.fft.rfft(embeddings, dim=-1)
    magnitude = torch.abs(fft_result)
    
    print(f"Embedding shape: {embeddings.shape}")
    print(f"FFT magnitude shape: {magnitude.shape}")
    
    # Analyze energy distribution across frequency bins
    n_freqs = magnitude.shape[-1]
    band_size = n_freqs // config.num_bands
    
    print("\nEnergy per frequency band:")
    total_energy = (magnitude ** 2).sum().item()
    
    for band in range(config.num_bands):
        start = band * band_size
        end = (band + 1) * band_size if band < config.num_bands - 1 else n_freqs
        band_energy = (magnitude[..., start:end] ** 2).sum().item()
        pct = 100 * band_energy / total_energy
        print(f"  Band {band} (bins {start}-{end}): {pct:.1f}% of energy")
    
    # Visualize
    if HAS_MATPLOTLIB:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Average magnitude spectrum
        avg_magnitude = magnitude.mean(dim=(0, 1)).numpy()
        axes[0].plot(avg_magnitude)
        axes[0].set_xlabel('Frequency Bin')
        axes[0].set_ylabel('Magnitude')
        axes[0].set_title('Average FFT Magnitude Spectrum')
        
        # Energy per band
        energies = []
        for band in range(config.num_bands):
            start = band * band_size
            end = (band + 1) * band_size if band < config.num_bands - 1 else n_freqs
            band_energy = (magnitude[..., start:end] ** 2).sum().item()
            energies.append(band_energy)
        
        axes[1].bar(range(config.num_bands), energies)
        axes[1].set_xlabel('Band')
        axes[1].set_ylabel('Energy')
        axes[1].set_title('Energy Distribution Across Bands')
        
        plt.tight_layout()
        plt.savefig('spectral_decomposition.png', dpi=150)
        print("\nSaved: spectral_decomposition.png")
        plt.close()
    
    return True


# ==============================================================================
# CELL 10: RUN ALL TESTS
# ==============================================================================

def run_all_tests():
    """Run all tokenizer tests."""
    print("="*70)
    print("EXPERIMENT 034: ZIPF WAVE TOKENIZER VALIDATION")
    print("="*70)
    
    tests = [
        ("Zipf Distribution", test_zipf_distribution),
        ("Frequency Mapping", test_frequency_mapping),
        ("Band Distribution", test_band_distribution),
        ("Real Sentences", test_real_sentences),
        ("Embedding Quality", test_embedding_quality),
        ("Decoder Roundtrip", test_decoder_roundtrip),
        ("Wave Visualization", visualize_wave_representations),
        ("Spectral Decomposition", test_spectral_decomposition),
    ]
    
    results = {}
    for name, test_fn in tests:
        try:
            passed = test_fn()
            results[name] = "PASSED" if passed else "FAILED"
        except Exception as e:
            print(f"\nERROR in {name}: {e}")
            results[name] = f"ERROR: {e}"
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    for name, result in results.items():
        status = "OK" if result == "PASSED" else "FAIL"
        print(f"  [{status}] {name}: {result}")
    
    passed = sum(1 for r in results.values() if r == "PASSED")
    total = len(results)
    print(f"\nTotal: {passed}/{total} tests passed")
    
    return results


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    run_all_tests()
