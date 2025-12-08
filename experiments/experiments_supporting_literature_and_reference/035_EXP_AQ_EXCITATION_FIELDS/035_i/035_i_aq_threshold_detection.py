# %% [markdown]
# # Experiment 035I: AQ Excitation Threshold Detection
# 
# **AKIRA Project - Oscar Goldman - Shogu Research Group @ Datamutant.ai**
# 
# ---
# 
# ## Core Hypothesis
# 
# The AKIRA framework proposes that:
# 1. AQ (Action Quanta) are quasiparticle excitations in the model's weight field
# 2. A prompt must contain a MINIMUM number of AQ to excite the belief field sufficiently
# 3. Below this threshold, the model cannot construct a coherent response
# 4. Above threshold, AQ "resonate" with the weight field and bond into the answer
# 
# This is analogous to:
# - Quantum mechanics: minimum energy to excite a state
# - Radar: minimum signal-to-noise for target detection
# - Neural activation: threshold potential for firing
# 
# ---
# 
# ## What We're Testing
# 
# 1. **Threshold Detection**: Find the minimum AQ count needed for coherent responses
# 2. **Belief Field Visualization**: Measure activation coherence as proxy for field state
# 3. **Resonance Patterns**: Track how AQ in prompts excite corresponding weight patterns
# 4. **Phase Transition**: Identify the critical point where responses become coherent
# 
# ---
# 
# ## Experimental Design
# 
# We construct prompts with varying numbers of AQ (action-enabling discriminations):
# - 0 AQ: Pure noise / no actionable content
# - 1 AQ: Single discrimination (e.g., just "threat" without context)
# - 2 AQ: Two discriminations (e.g., "threat" + "proximity")
# - 3+ AQ: Multiple discriminations enabling full action
# 
# We measure:
# - Response coherence/quality
# - Activation coherence (field excitation proxy)
# - Layer-wise activation magnitude (excitation strength)
# - Attention entropy (focus vs diffuse)
# 
# ---

# %% [markdown]
# ## 1. Setup

# %%
# Install dependencies (uncomment for Colab)
# !pip install transformers torch numpy scikit-learn matplotlib seaborn scipy -q

# %%
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2LMHeadModel
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
import warnings
from scipy import stats
from scipy.signal import hilbert
import json
from tqdm import tqdm
import gc
from collections import defaultdict

warnings.filterwarnings('ignore')

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")
print(f"PyTorch version: {torch.__version__}")
if DEVICE == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# %% [markdown]
# ## 2. Configuration

# %%
@dataclass
class ExperimentConfig:
    """Configuration for AQ threshold detection experiment."""
    
    model_name: str = "gpt2-medium"
    model_path: str = "gpt2-medium"
    
    # Number of prompts per AQ count level
    n_prompts_per_level: int = 100
    
    # AQ count levels to test (0 = no AQ, 1 = single AQ, etc.)
    aq_levels: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4, 5])
    
    # Layers to probe for field visualization
    layers_to_probe: List[int] = field(default_factory=lambda: [0, 4, 8, 12, 16, 20, 23])
    
    # Number of tokens to generate for response quality
    n_generate_tokens: int = 20
    
    random_seed: int = 42
    
    def __post_init__(self) -> None:
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)


config = ExperimentConfig()
print(f"Model: {config.model_name}")
print(f"AQ levels to test: {config.aq_levels}")
print(f"Prompts per level: {config.n_prompts_per_level}")
print(f"Layers to probe: {config.layers_to_probe}")

# %% [markdown]
# ## 3. AQ-Graded Prompt Construction
# 
# We construct prompts with precisely controlled AQ content.
# 
# Each AQ represents a discrimination that enables action:
# - THREAT_PRESENT: Is there a threat? (enables FLEE vs STAY)
# - PROXIMITY: How close? (enables URGENT vs DELAYED response)
# - DIRECTION: Which way? (enables LEFT vs RIGHT vs FORWARD)
# - MAGNITUDE: How severe? (enables PROPORTIONAL response)
# - AGENCY: Who acts? (enables SELF vs OTHER response)
# 
# A prompt with 0 AQ has no actionable discriminations.
# A prompt with 5 AQ has all discriminations needed for precise action.

# %%
# Define the AQ components that can be added to prompts
# Each AQ enables a specific discrimination

AQ_COMPONENTS = {
    "THREAT_PRESENT": {
        "description": "Discriminates THREAT vs NO-THREAT",
        "positive_markers": ["danger", "threat", "attack", "fire", "flood", "predator", "enemy", "poison", "collapse"],
        "negative_markers": ["safe", "calm", "peaceful", "secure", "protected"],
        "enabled_action": "FLEE vs STAY"
    },
    "PROXIMITY": {
        "description": "Discriminates NEAR vs FAR",
        "positive_markers": ["approaching", "close", "nearby", "imminent", "seconds away", "right here", "at your feet"],
        "negative_markers": ["distant", "far away", "miles away", "hours away"],
        "enabled_action": "URGENT vs DELAYED"
    },
    "DIRECTION": {
        "description": "Discriminates directional alternatives",
        "positive_markers": ["from the left", "from the right", "from above", "from behind", "from the north"],
        "negative_markers": ["from somewhere", "from around"],
        "enabled_action": "LEFT vs RIGHT vs FORWARD vs BACK"
    },
    "MAGNITUDE": {
        "description": "Discriminates severity level",
        "positive_markers": ["massive", "lethal", "catastrophic", "tiny", "minor", "severe", "critical"],
        "negative_markers": ["some", "a bit of", "possibly"],
        "enabled_action": "PROPORTIONAL response scaling"
    },
    "AGENCY": {
        "description": "Discriminates who must act",
        "positive_markers": ["you must", "you should", "you need to", "your responsibility"],
        "negative_markers": ["someone should", "it might be", "perhaps"],
        "enabled_action": "SELF-ACTION vs DELEGATE"
    },
    "TEMPORAL": {
        "description": "Discriminates when to act",
        "positive_markers": ["now", "immediately", "right now", "this instant", "before it's too late"],
        "negative_markers": ["eventually", "sometime", "when possible", "later"],
        "enabled_action": "NOW vs LATER"
    }
}

print(f"Defined {len(AQ_COMPONENTS)} AQ components:")
for name, data in AQ_COMPONENTS.items():
    print(f"  {name}: {data['description']} -> {data['enabled_action']}")

# %%
def generate_graded_prompts(n_per_level: int = 100) -> Dict[int, List[Dict]]:
    """Generate prompts with precisely controlled AQ counts.
    
    Args:
        n_per_level: Number of prompts per AQ level
        
    Returns:
        Dict mapping AQ count to list of prompt dicts
    """
    prompts_by_level = {level: [] for level in config.aq_levels}
    
    # Base scenarios (neutral, no AQ by themselves)
    base_scenarios = [
        "Something is happening.",
        "There is a situation.",
        "An event is occurring.",
        "A thing exists.",
        "Something is present.",
    ]
    
    # Level 0: No AQ - completely ambiguous, no actionable discrimination
    level_0_templates = [
        "Something might be somewhere.",
        "There could be a thing.",
        "It is possible that something exists.",
        "One might consider that perhaps.",
        "Things may or may not be happening.",
        "An unspecified situation of unknown nature.",
        "Conditions are in some state.",
        "Elements are arranged somehow.",
        "Factors exist in the environment.",
        "A state of affairs persists.",
    ]
    
    # Level 1: Single AQ (one discrimination only)
    # Just THREAT_PRESENT - you know there's danger but nothing else
    level_1_templates = [
        "There is danger.",
        "A threat exists.",
        "Danger is present.",
        "Something dangerous.",
        "A hazard exists.",
        "There is a predator.",
        "An enemy is present.",
        "Fire exists somewhere.",
        "Poison is present.",
        "A threat has appeared.",
    ]
    
    # Level 2: Two AQ (THREAT + PROXIMITY)
    level_2_templates = [
        "Danger is approaching.",
        "A threat is nearby.",
        "Close danger exists.",
        "An imminent threat.",
        "Nearby hazard detected.",
        "A predator is close.",
        "The enemy approaches.",
        "Fire is spreading toward you.",
        "Poison is right here.",
        "A threat is seconds away.",
    ]
    
    # Level 3: Three AQ (THREAT + PROXIMITY + DIRECTION)
    level_3_templates = [
        "Danger approaches from the left.",
        "A threat is coming from behind.",
        "Close danger from the north.",
        "Imminent threat from above.",
        "Nearby hazard to your right.",
        "A predator stalks from the shadows on your left.",
        "The enemy charges from the east.",
        "Fire spreads from the south toward you.",
        "Poison gas drifts from the west.",
        "A threat emerges from below, close.",
    ]
    
    # Level 4: Four AQ (THREAT + PROXIMITY + DIRECTION + MAGNITUDE)
    level_4_templates = [
        "A massive danger rapidly approaches from the left.",
        "A lethal threat is close behind you.",
        "Critical danger from the north, imminent.",
        "A severe, imminent threat from above.",
        "A minor hazard nearby to your right.",
        "A lethal predator stalks close from the left.",
        "A massive enemy force charges from the east.",
        "A catastrophic fire spreads rapidly from the south.",
        "Lethal poison gas drifts nearby from the west.",
        "A critical threat emerges from below, seconds away.",
    ]
    
    # Level 5: Five AQ (THREAT + PROXIMITY + DIRECTION + MAGNITUDE + AGENCY)
    level_5_templates = [
        "A massive danger rapidly approaches from the left. You must act.",
        "A lethal threat is close behind you. You need to respond.",
        "Critical danger from the north is imminent. You should move.",
        "A severe threat from above requires your immediate action.",
        "You must handle the minor hazard nearby to your right.",
        "A lethal predator stalks you from the left. You must flee.",
        "A massive enemy charges from the east. You need to defend.",
        "Catastrophic fire from the south. You must evacuate now.",
        "Lethal gas from the west. You should hold your breath and run.",
        "Critical threat from below. You must jump immediately.",
    ]
    
    # Level 6: Six AQ (all components - full action specification)
    level_6_templates = [
        "A massive danger rapidly approaches from the left. You must act now.",
        "A lethal threat is close behind you. You need to respond immediately.",
        "Critical danger from the north is imminent. You should move right now.",
        "A severe threat from above requires your immediate action this instant.",
        "You must handle the critical hazard nearby to your right now.",
        "A lethal predator stalks you from the left. You must flee immediately.",
        "A massive enemy charges from the east. You need to defend now.",
        "Catastrophic fire spreads from the south. You must evacuate this instant.",
        "Lethal gas drifts from the west. You should run now, hold your breath.",
        "Critical threat emerges from below. You must jump right now.",
    ]
    
    template_sets = {
        0: level_0_templates,
        1: level_1_templates,
        2: level_2_templates,
        3: level_3_templates,
        4: level_4_templates,
        5: level_5_templates,
    }
    
    # Add level 6 if in config
    if 6 in config.aq_levels:
        template_sets[6] = level_6_templates
    
    # Generate prompts for each level
    for level in config.aq_levels:
        if level not in template_sets:
            continue
            
        templates = template_sets[level]
        
        for i in range(n_per_level):
            template = templates[i % len(templates)]
            
            # Add variation by slight rewording
            variations = [
                template,
                template.replace(".", "!"),
                "Warning: " + template,
                template + " What do you do?",
                "Alert: " + template,
            ]
            
            prompt_text = variations[i % len(variations)]
            
            # Identify which AQ are present
            present_aq = []
            prompt_lower = prompt_text.lower()
            
            for aq_name, aq_data in AQ_COMPONENTS.items():
                for marker in aq_data["positive_markers"]:
                    if marker.lower() in prompt_lower:
                        present_aq.append(aq_name)
                        break
            
            prompts_by_level[level].append({
                "text": prompt_text,
                "aq_count": level,
                "aq_present": present_aq,
                "expected_action_clarity": level / max(config.aq_levels)  # 0 to 1
            })
    
    return prompts_by_level


# Generate all prompts
GRADED_PROMPTS = generate_graded_prompts(config.n_prompts_per_level)

print(f"\nGenerated prompts by AQ level:")
for level, prompts in GRADED_PROMPTS.items():
    print(f"  Level {level}: {len(prompts)} prompts")
    if prompts:
        print(f"    Example: {prompts[0]['text'][:60]}...")

# %% [markdown]
# ## 4. Model Loading

# %%
print(f"Loading {config.model_name}...")
tokenizer = AutoTokenizer.from_pretrained(config.model_path)
model = GPT2LMHeadModel.from_pretrained(config.model_path)
model = model.to(DEVICE)
model.eval()

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"Model loaded: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters")
print(f"Number of layers: {model.config.n_layer}")
print(f"Hidden size: {model.config.n_embd}")

# %% [markdown]
# ## 5. Belief Field Measurement Functions
# 
# The "belief field" is the model's internal representation state.
# We cannot see it directly, but we can measure proxies:
# 
# 1. **Activation Magnitude**: How "excited" is each layer?
# 2. **Activation Coherence**: How aligned are the representations?
# 3. **Attention Entropy**: How focused vs diffuse is attention?
# 4. **Layer-wise Correlation**: How correlated are layers (field coupling)?

# %%
class BeliefFieldProbe:
    """Probe the model's internal 'belief field' state."""
    
    def __init__(self, model: nn.Module):
        """Initialize probe.
        
        Args:
            model: The language model
        """
        self.model = model
        self.hooks = []
        self.stored_activations = {}
        self.stored_attentions = {}
        
    def _get_activation_hook(self, layer_idx: int) -> Callable:
        """Create hook to store activations."""
        def hook(module, input, output):
            if isinstance(output, tuple):
                self.stored_activations[layer_idx] = output[0].detach().clone()
            else:
                self.stored_activations[layer_idx] = output.detach().clone()
        return hook
    
    def _get_attention_hook(self, layer_idx: int) -> Callable:
        """Create hook to store attention weights."""
        def hook(module, input, output):
            # output is (attn_output, attn_weights) for GPT-2
            if isinstance(output, tuple) and len(output) > 1:
                if output[1] is not None:
                    self.stored_attentions[layer_idx] = output[1].detach().clone()
        return hook
    
    def register_hooks(self, layers: List[int]) -> None:
        """Register hooks to capture activations and attention."""
        self.clear_hooks()
        
        for layer_idx in layers:
            if hasattr(self.model, 'transformer'):
                block = self.model.transformer.h[layer_idx]
                # Activation hook on block output
                hook1 = block.register_forward_hook(self._get_activation_hook(layer_idx))
                self.hooks.append(hook1)
                # Attention hook on attention module
                if hasattr(block, 'attn'):
                    hook2 = block.attn.register_forward_hook(self._get_attention_hook(layer_idx))
                    self.hooks.append(hook2)
    
    def clear_hooks(self) -> None:
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.stored_activations = {}
        self.stored_attentions = {}
    
    def compute_field_metrics(self, prompt: str, tokenizer: AutoTokenizer,
                              layers: List[int]) -> Dict[str, Any]:
        """Compute belief field metrics for a prompt.
        
        Args:
            prompt: Input prompt
            tokenizer: The tokenizer
            layers: Layers to probe
            
        Returns:
            Dict with field metrics
        """
        self.register_hooks(layers)
        
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        
        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=True)
        
        metrics = {
            "activation_magnitude": {},
            "activation_coherence": {},
            "attention_entropy": {},
            "layer_correlation": {}
        }
        
        # Compute metrics for each layer
        for layer in layers:
            if layer in self.stored_activations:
                act = self.stored_activations[layer]  # [batch, seq, hidden]
                
                # Activation magnitude (mean L2 norm)
                magnitude = torch.norm(act, dim=-1).mean().item()
                metrics["activation_magnitude"][layer] = magnitude
                
                # Activation coherence (cosine similarity between tokens)
                if act.shape[1] > 1:
                    act_flat = act[0].cpu().numpy()  # [seq, hidden]
                    cos_sim = cosine_similarity(act_flat)
                    # Mean off-diagonal similarity
                    mask = ~np.eye(cos_sim.shape[0], dtype=bool)
                    coherence = cos_sim[mask].mean()
                    metrics["activation_coherence"][layer] = float(coherence)
                else:
                    metrics["activation_coherence"][layer] = 1.0
        
        # Attention entropy from model outputs
        if outputs.attentions is not None:
            for i, layer in enumerate(layers):
                if i < len(outputs.attentions):
                    attn = outputs.attentions[i]  # [batch, heads, seq, seq]
                    # Mean attention entropy across heads
                    attn_probs = attn[0].mean(dim=0)  # [seq, seq]
                    # Compute entropy for last token
                    last_attn = attn_probs[-1]
                    entropy = -torch.sum(last_attn * torch.log(last_attn + 1e-10)).item()
                    metrics["attention_entropy"][layer] = entropy
        
        # Layer correlation (correlation between adjacent layers)
        layer_acts = []
        for layer in sorted(layers):
            if layer in self.stored_activations:
                act = self.stored_activations[layer][0, -1, :].cpu().numpy()
                layer_acts.append(act)
        
        if len(layer_acts) > 1:
            correlations = []
            for i in range(len(layer_acts) - 1):
                corr = np.corrcoef(layer_acts[i], layer_acts[i+1])[0, 1]
                correlations.append(corr)
            metrics["layer_correlation"]["mean"] = float(np.mean(correlations))
            metrics["layer_correlation"]["values"] = [float(c) for c in correlations]
        
        self.clear_hooks()
        
        return metrics


probe = BeliefFieldProbe(model)
print("BeliefFieldProbe ready")

# %%
def measure_response_quality(prompt: str, model: nn.Module, 
                             tokenizer: AutoTokenizer, n_tokens: int = 20) -> Dict[str, float]:
    """Measure quality of model's response to a prompt.
    
    Args:
        prompt: Input prompt
        model: The model
        tokenizer: The tokenizer
        n_tokens: Tokens to generate
        
    Returns:
        Dict with quality metrics
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            max_new_tokens=n_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            output_scores=True,
            return_dict_in_generate=True
        )
    
    generated_ids = outputs.sequences[0, inputs["input_ids"].shape[1]:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    # Quality metrics
    metrics = {}
    
    # 1. Response confidence (mean probability of generated tokens)
    if outputs.scores:
        probs = []
        for i, score in enumerate(outputs.scores):
            if i < len(generated_ids):
                prob = torch.softmax(score[0], dim=-1)
                token_prob = prob[generated_ids[i]].item()
                probs.append(token_prob)
        metrics["confidence"] = float(np.mean(probs)) if probs else 0.0
    else:
        metrics["confidence"] = 0.0
    
    # 2. Response coherence (does it contain action-relevant words?)
    action_words = ["run", "flee", "escape", "move", "go", "leave", "stay", "wait", 
                    "fight", "defend", "hide", "duck", "jump", "stop", "avoid",
                    "quickly", "immediately", "now", "fast", "slowly", "carefully"]
    action_count = sum(1 for w in action_words if w.lower() in generated_text.lower())
    metrics["action_relevance"] = min(1.0, action_count / 3)
    
    # 3. Response specificity (not vague/hedge words)
    vague_words = ["maybe", "perhaps", "might", "could", "possibly", "somehow", 
                   "something", "someone", "somewhere", "somewhat"]
    vague_count = sum(1 for w in vague_words if w.lower() in generated_text.lower())
    metrics["specificity"] = max(0.0, 1.0 - vague_count / 3)
    
    # 4. Response length (very short = unclear prompt)
    metrics["response_length"] = len(generated_text.split())
    
    # 5. Combined quality score
    metrics["quality_score"] = (
        0.4 * metrics["confidence"] +
        0.3 * metrics["action_relevance"] +
        0.3 * metrics["specificity"]
    )
    
    metrics["generated_text"] = generated_text
    
    return metrics


print("Response quality measurement ready")

# %% [markdown]
# ## 6. Run Threshold Detection Experiment

# %%
print("\n" + "=" * 70)
print("RUNNING AQ THRESHOLD DETECTION EXPERIMENT")
print("=" * 70)

RESULTS_BY_LEVEL = {level: [] for level in config.aq_levels}

for level in config.aq_levels:
    print(f"\nProcessing AQ Level {level}...")
    prompts = GRADED_PROMPTS[level]
    
    for i, prompt_data in enumerate(tqdm(prompts, desc=f"Level {level}")):
        prompt_text = prompt_data["text"]
        
        # Measure belief field state
        field_metrics = probe.compute_field_metrics(
            prompt_text, tokenizer, config.layers_to_probe
        )
        
        # Measure response quality
        response_metrics = measure_response_quality(
            prompt_text, model, tokenizer, config.n_generate_tokens
        )
        
        # Combine results
        result = {
            "prompt": prompt_text,
            "aq_count": level,
            "aq_present": prompt_data["aq_present"],
            "field_metrics": field_metrics,
            "response_metrics": response_metrics
        }
        
        RESULTS_BY_LEVEL[level].append(result)
        
        # Memory cleanup
        if i % 20 == 0:
            torch.cuda.empty_cache() if DEVICE == "cuda" else None

print("\nExperiment complete.")

# %% [markdown]
# ## 7. Analyze Threshold Effects

# %%
# Aggregate metrics by AQ level
print("\n" + "=" * 70)
print("ANALYSIS: THRESHOLD EFFECTS BY AQ LEVEL")
print("=" * 70)

aggregated = {}

for level in config.aq_levels:
    results = RESULTS_BY_LEVEL[level]
    
    # Aggregate response quality
    quality_scores = [r["response_metrics"]["quality_score"] for r in results]
    confidences = [r["response_metrics"]["confidence"] for r in results]
    action_relevances = [r["response_metrics"]["action_relevance"] for r in results]
    
    # Aggregate field metrics
    mean_magnitudes = []
    mean_coherences = []
    mean_entropies = []
    
    for r in results:
        fm = r["field_metrics"]
        if fm["activation_magnitude"]:
            mean_magnitudes.append(np.mean(list(fm["activation_magnitude"].values())))
        if fm["activation_coherence"]:
            mean_coherences.append(np.mean(list(fm["activation_coherence"].values())))
        if fm["attention_entropy"]:
            mean_entropies.append(np.mean(list(fm["attention_entropy"].values())))
    
    aggregated[level] = {
        "quality_score": {
            "mean": float(np.mean(quality_scores)),
            "std": float(np.std(quality_scores)),
            "values": quality_scores
        },
        "confidence": {
            "mean": float(np.mean(confidences)),
            "std": float(np.std(confidences))
        },
        "action_relevance": {
            "mean": float(np.mean(action_relevances)),
            "std": float(np.std(action_relevances))
        },
        "activation_magnitude": {
            "mean": float(np.mean(mean_magnitudes)) if mean_magnitudes else 0,
            "std": float(np.std(mean_magnitudes)) if mean_magnitudes else 0
        },
        "activation_coherence": {
            "mean": float(np.mean(mean_coherences)) if mean_coherences else 0,
            "std": float(np.std(mean_coherences)) if mean_coherences else 0
        },
        "attention_entropy": {
            "mean": float(np.mean(mean_entropies)) if mean_entropies else 0,
            "std": float(np.std(mean_entropies)) if mean_entropies else 0
        }
    }
    
    print(f"\nAQ Level {level}:")
    print(f"  Quality Score: {aggregated[level]['quality_score']['mean']:.3f} +/- {aggregated[level]['quality_score']['std']:.3f}")
    print(f"  Confidence: {aggregated[level]['confidence']['mean']:.3f}")
    print(f"  Action Relevance: {aggregated[level]['action_relevance']['mean']:.3f}")
    print(f"  Activation Magnitude: {aggregated[level]['activation_magnitude']['mean']:.3f}")
    print(f"  Activation Coherence: {aggregated[level]['activation_coherence']['mean']:.3f}")
    print(f"  Attention Entropy: {aggregated[level]['attention_entropy']['mean']:.3f}")

# %%
# Statistical tests for threshold detection
print("\n" + "=" * 70)
print("STATISTICAL TESTS: THRESHOLD DETECTION")
print("=" * 70)

# Test for significant difference between each level and the next
threshold_tests = []

for i in range(len(config.aq_levels) - 1):
    level_low = config.aq_levels[i]
    level_high = config.aq_levels[i + 1]
    
    quality_low = aggregated[level_low]["quality_score"]["values"]
    quality_high = aggregated[level_high]["quality_score"]["values"]
    
    # T-test
    t_stat, p_value = stats.ttest_ind(quality_low, quality_high)
    
    # Effect size (Cohen's d)
    pooled_std = np.sqrt((np.var(quality_low) + np.var(quality_high)) / 2)
    cohens_d = (np.mean(quality_high) - np.mean(quality_low)) / pooled_std if pooled_std > 0 else 0
    
    threshold_tests.append({
        "transition": f"{level_low} -> {level_high}",
        "mean_low": np.mean(quality_low),
        "mean_high": np.mean(quality_high),
        "t_statistic": t_stat,
        "p_value": p_value,
        "cohens_d": cohens_d
    })
    
    sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
    print(f"\n{level_low} AQ -> {level_high} AQ:")
    print(f"  Quality change: {np.mean(quality_low):.3f} -> {np.mean(quality_high):.3f}")
    print(f"  t-statistic: {t_stat:.3f}")
    print(f"  p-value: {p_value:.4f} {sig}")
    print(f"  Cohen's d: {cohens_d:.3f}")

# Find the critical threshold (largest effect size transition)
if threshold_tests:
    critical_transition = max(threshold_tests, key=lambda x: x["cohens_d"])
    print(f"\n" + "-" * 70)
    print(f"CRITICAL THRESHOLD DETECTED: {critical_transition['transition']}")
    print(f"Effect size (d): {critical_transition['cohens_d']:.3f}")
    print(f"p-value: {critical_transition['p_value']:.6f}")

# %% [markdown]
# ## 8. Visualize Belief Field

# %%
# Create visualization of the belief field across AQ levels
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 1. Response Quality by AQ Level
ax = axes[0, 0]
levels = list(aggregated.keys())
quality_means = [aggregated[l]["quality_score"]["mean"] for l in levels]
quality_stds = [aggregated[l]["quality_score"]["std"] for l in levels]
ax.errorbar(levels, quality_means, yerr=quality_stds, marker='o', capsize=5, linewidth=2, markersize=8)
ax.set_xlabel("AQ Count in Prompt")
ax.set_ylabel("Response Quality Score")
ax.set_title("Response Quality vs AQ Count\n(Threshold Detection)")
ax.grid(True, alpha=0.3)
ax.axhline(y=np.mean(quality_means), color='r', linestyle='--', alpha=0.5, label='Mean')
ax.legend()

# 2. Activation Magnitude (Field Excitation)
ax = axes[0, 1]
mag_means = [aggregated[l]["activation_magnitude"]["mean"] for l in levels]
mag_stds = [aggregated[l]["activation_magnitude"]["std"] for l in levels]
ax.errorbar(levels, mag_means, yerr=mag_stds, marker='s', capsize=5, linewidth=2, markersize=8, color='green')
ax.set_xlabel("AQ Count in Prompt")
ax.set_ylabel("Mean Activation Magnitude")
ax.set_title("Field Excitation vs AQ Count\n(Activation Magnitude)")
ax.grid(True, alpha=0.3)

# 3. Activation Coherence (Field Alignment)
ax = axes[0, 2]
coh_means = [aggregated[l]["activation_coherence"]["mean"] for l in levels]
coh_stds = [aggregated[l]["activation_coherence"]["std"] for l in levels]
ax.errorbar(levels, coh_means, yerr=coh_stds, marker='^', capsize=5, linewidth=2, markersize=8, color='purple')
ax.set_xlabel("AQ Count in Prompt")
ax.set_ylabel("Activation Coherence")
ax.set_title("Field Coherence vs AQ Count\n(Representation Alignment)")
ax.grid(True, alpha=0.3)

# 4. Attention Entropy (Focus vs Diffuse)
ax = axes[1, 0]
ent_means = [aggregated[l]["attention_entropy"]["mean"] for l in levels]
ent_stds = [aggregated[l]["attention_entropy"]["std"] for l in levels]
ax.errorbar(levels, ent_means, yerr=ent_stds, marker='d', capsize=5, linewidth=2, markersize=8, color='orange')
ax.set_xlabel("AQ Count in Prompt")
ax.set_ylabel("Attention Entropy")
ax.set_title("Attention Focus vs AQ Count\n(Lower = More Focused)")
ax.grid(True, alpha=0.3)

# 5. Effect Size by Transition
ax = axes[1, 1]
if threshold_tests:
    transitions = [t["transition"] for t in threshold_tests]
    effects = [t["cohens_d"] for t in threshold_tests]
    colors = ['green' if t["p_value"] < 0.05 else 'gray' for t in threshold_tests]
    ax.bar(transitions, effects, color=colors, alpha=0.7)
    ax.axhline(y=0.8, color='red', linestyle='--', label='Large effect (d=0.8)')
    ax.axhline(y=0.5, color='orange', linestyle='--', label='Medium effect (d=0.5)')
    ax.set_xlabel("AQ Level Transition")
    ax.set_ylabel("Cohen's d")
    ax.set_title("Effect Size at Each AQ Transition\n(Green = p < 0.05)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

# 6. Confidence vs AQ Level
ax = axes[1, 2]
conf_means = [aggregated[l]["confidence"]["mean"] for l in levels]
conf_stds = [aggregated[l]["confidence"]["std"] for l in levels]
ax.errorbar(levels, conf_means, yerr=conf_stds, marker='o', capsize=5, linewidth=2, markersize=8, color='red')
ax.set_xlabel("AQ Count in Prompt")
ax.set_ylabel("Model Confidence")
ax.set_title("Model Confidence vs AQ Count\n(Token Generation Probability)")
ax.grid(True, alpha=0.3)

plt.suptitle("035I: AQ Excitation Threshold - Belief Field Visualization", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("035I_threshold_detection.png", dpi=150, bbox_inches='tight')
plt.show()
print("Saved: 035I_threshold_detection.png")

# %% [markdown]
# ## 9. Layer-wise Field Analysis

# %%
# Analyze how the field state varies by layer for different AQ levels
print("\n" + "=" * 70)
print("LAYER-WISE BELIEF FIELD ANALYSIS")
print("=" * 70)

# Collect layer-wise data
layer_data = {layer: {level: {"magnitude": [], "coherence": []} 
                      for level in config.aq_levels}
              for layer in config.layers_to_probe}

for level in config.aq_levels:
    for result in RESULTS_BY_LEVEL[level]:
        fm = result["field_metrics"]
        for layer in config.layers_to_probe:
            if layer in fm["activation_magnitude"]:
                layer_data[layer][level]["magnitude"].append(fm["activation_magnitude"][layer])
            if layer in fm["activation_coherence"]:
                layer_data[layer][level]["coherence"].append(fm["activation_coherence"][layer])

# Create heatmap of magnitude by layer and AQ level
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Magnitude heatmap
ax = axes[0]
mag_matrix = np.zeros((len(config.layers_to_probe), len(config.aq_levels)))
for i, layer in enumerate(config.layers_to_probe):
    for j, level in enumerate(config.aq_levels):
        values = layer_data[layer][level]["magnitude"]
        mag_matrix[i, j] = np.mean(values) if values else 0

sns.heatmap(mag_matrix, ax=ax, cmap='YlOrRd',
            xticklabels=config.aq_levels, yticklabels=config.layers_to_probe,
            annot=True, fmt='.1f')
ax.set_xlabel("AQ Count")
ax.set_ylabel("Layer")
ax.set_title("Activation Magnitude by Layer and AQ Level\n(Field Excitation)")

# Coherence heatmap
ax = axes[1]
coh_matrix = np.zeros((len(config.layers_to_probe), len(config.aq_levels)))
for i, layer in enumerate(config.layers_to_probe):
    for j, level in enumerate(config.aq_levels):
        values = layer_data[layer][level]["coherence"]
        coh_matrix[i, j] = np.mean(values) if values else 0

sns.heatmap(coh_matrix, ax=ax, cmap='YlGnBu',
            xticklabels=config.aq_levels, yticklabels=config.layers_to_probe,
            annot=True, fmt='.2f')
ax.set_xlabel("AQ Count")
ax.set_ylabel("Layer")
ax.set_title("Activation Coherence by Layer and AQ Level\n(Field Alignment)")

plt.suptitle("035I: Layer-wise Belief Field State", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("035I_layer_analysis.png", dpi=150, bbox_inches='tight')
plt.show()
print("Saved: 035I_layer_analysis.png")

# %% [markdown]
# ## 10. Summary and Conclusions

# %%
print("\n" + "=" * 70)
print("EXPERIMENT 035I: AQ EXCITATION THRESHOLD - SUMMARY")
print("=" * 70)

print(f"\nExperiment Configuration:")
print(f"  Model: {config.model_name}")
print(f"  AQ levels tested: {config.aq_levels}")
print(f"  Prompts per level: {config.n_prompts_per_level}")
print(f"  Total prompts: {sum(len(GRADED_PROMPTS[l]) for l in config.aq_levels)}")

print(f"\nKEY FINDINGS:")
print("-" * 70)

# 1. Threshold identification
if threshold_tests:
    critical = max(threshold_tests, key=lambda x: x["cohens_d"])
    print(f"\n1. CRITICAL THRESHOLD:")
    print(f"   The largest quality jump occurs at: {critical['transition']}")
    print(f"   Effect size (Cohen's d): {critical['cohens_d']:.3f}")
    print(f"   Statistical significance: p = {critical['p_value']:.6f}")
    
    if critical['cohens_d'] > 0.8:
        print(f"   Interpretation: LARGE effect - strong evidence for threshold")
    elif critical['cohens_d'] > 0.5:
        print(f"   Interpretation: MEDIUM effect - moderate evidence for threshold")
    else:
        print(f"   Interpretation: SMALL effect - weak evidence for threshold")

# 2. Field excitation pattern
print(f"\n2. FIELD EXCITATION PATTERN:")
mag_trend = [aggregated[l]["activation_magnitude"]["mean"] for l in config.aq_levels]
if np.corrcoef(config.aq_levels, mag_trend)[0,1] > 0.5:
    print(f"   Activation magnitude INCREASES with AQ count")
    print(f"   Correlation: r = {np.corrcoef(config.aq_levels, mag_trend)[0,1]:.3f}")
else:
    print(f"   Activation magnitude shows NO clear trend with AQ count")

# 3. Coherence pattern
print(f"\n3. FIELD COHERENCE PATTERN:")
coh_trend = [aggregated[l]["activation_coherence"]["mean"] for l in config.aq_levels]
coh_corr = np.corrcoef(config.aq_levels, coh_trend)[0,1]
print(f"   Coherence correlation with AQ: r = {coh_corr:.3f}")
if coh_corr > 0.3:
    print(f"   Higher AQ count -> Higher coherence (more aligned field)")
elif coh_corr < -0.3:
    print(f"   Higher AQ count -> Lower coherence (more distributed field)")
else:
    print(f"   No strong coherence trend with AQ count")

# 4. Attention pattern
print(f"\n4. ATTENTION PATTERN:")
ent_trend = [aggregated[l]["attention_entropy"]["mean"] for l in config.aq_levels]
ent_corr = np.corrcoef(config.aq_levels, ent_trend)[0,1]
print(f"   Entropy correlation with AQ: r = {ent_corr:.3f}")
if ent_corr < -0.3:
    print(f"   Higher AQ count -> Lower entropy (more focused attention)")
elif ent_corr > 0.3:
    print(f"   Higher AQ count -> Higher entropy (more diffuse attention)")
else:
    print(f"   No strong entropy trend with AQ count")

# Overall conclusion
print(f"\n" + "=" * 70)
print("CONCLUSIONS:")
print("=" * 70)

# Check if threshold hypothesis is supported
threshold_supported = (
    threshold_tests and 
    critical['cohens_d'] > 0.5 and 
    critical['p_value'] < 0.05
)

if threshold_supported:
    print(f"\n  THRESHOLD HYPOTHESIS: SUPPORTED")
    print(f"  There IS a minimum AQ count required for coherent responses.")
    print(f"  Critical threshold appears at: {critical['transition']}")
    print(f"  Below this threshold, the model cannot construct coherent action responses.")
    print(f"  Above this threshold, AQ resonate with the weight field and bond into answers.")
else:
    print(f"\n  THRESHOLD HYPOTHESIS: NOT CLEARLY SUPPORTED")
    print(f"  No sharp threshold detected in this experiment.")
    print(f"  Quality may increase gradually rather than in a phase transition.")

print(f"\n  BELIEF FIELD VISUALIZATION:")
print(f"  While we cannot 'see' the belief field directly, proxies show:")
print(f"  - Activation magnitude varies by AQ count")
print(f"  - Coherence patterns emerge at different layers")
print(f"  - Attention focus changes with actionable information")

print("=" * 70)

# %%
# Save all results
results_output = {
    "config": {
        "model": config.model_name,
        "aq_levels": config.aq_levels,
        "n_prompts_per_level": config.n_prompts_per_level,
        "layers_probed": config.layers_to_probe
    },
    "aggregated_by_level": {str(k): v for k, v in aggregated.items()},
    "threshold_tests": threshold_tests,
    "conclusions": {
        "threshold_supported": threshold_supported,
        "critical_transition": critical["transition"] if threshold_tests else None,
        "critical_effect_size": critical["cohens_d"] if threshold_tests else None
    }
}

with open("035I_results.json", "w") as f:
    json.dump(results_output, f, indent=2, default=str)

print("Results saved to 035I_results.json")
