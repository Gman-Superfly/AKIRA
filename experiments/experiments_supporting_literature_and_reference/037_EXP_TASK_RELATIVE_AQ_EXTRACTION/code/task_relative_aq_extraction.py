"""
Experiment 037: Task-Relative AQ Extraction

Tests whether the same input produces different Action Quanta under different task framings.

Hypothesis: AQ are emergent from signal-task interaction, not intrinsic properties of signals.

Usage:
    # Run on Google Colab or local with GPU
    python task_relative_aq_extraction.py
    
    # Or import and use components
    from task_relative_aq_extraction import TaskRelativeAQExtractor, run_experiment
"""

from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass
from collections import defaultdict
import json
import matplotlib.pyplot as plt


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class ExperimentConfig:
    """Configuration for task-relative AQ extraction experiment."""
    model_name: str = "gpt2"  # Or "gpt2-medium", "gpt2-large"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    layers_to_analyze: List[int] = None  # Will be set based on model
    n_ablation_samples: int = 50
    importance_threshold: float = 0.1  # 10% degradation = load-bearing
    
    def __post_init__(self):
        if self.layers_to_analyze is None:
            # Default layers for GPT-2 (12 layers)
            self.layers_to_analyze = [0, 2, 4, 6, 8, 10, 11]


# ============================================================================
# TASK DEFINITIONS
# ============================================================================

TASK_SCENARIOS = {
    'visual_scene': {
        'input': "A red ball on a blue table, next to a green cup.",
        'tasks': {
            'color_query': "What color is the ball?",
            'location_query': "Where is the ball?",
            'existence_query': "Is there a ball?",
            'relation_query': "What is the ball next to?",
            'count_query': "How many objects are on the table?"
        }
    },
    'temporal_event': {
        'input': "John picked up the key and walked to the door.",
        'tasks': {
            'agent_query': "Who performed the action?",
            'object_query': "What did John pick up?",
            'sequence_query': "What did John do after picking up the key?",
            'location_query': "Where did John go?"
        }
    },
    'ambiguous_signal': {
        'input': "Fire!",
        'tasks': {
            'emergency_context': "You are in a crowded building. Someone shouts:",
            'military_context': "You are a soldier with weapon ready. The commander says:",
            'campfire_context': "You are camping and someone points at wood saying:",
            'pottery_context': "You are in a pottery studio. The instructor says:"
        }
    },
    'numeric_reasoning': {
        'input': "There are 5 apples and 3 oranges in the basket.",
        'tasks': {
            'total_query': "How many fruits are there in total?",
            'difference_query': "How many more apples than oranges are there?",
            'type_query': "What types of fruit are in the basket?",
            'container_query': "Where are the fruits?"
        }
    }
}


# ============================================================================
# CORE EXTRACTOR CLASS
# ============================================================================

class TaskRelativeAQExtractor:
    """
    Extract and compare AQ candidates under different task framings.
    """
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.hooks = {}
        self.activations = {}
        
    def load_model(self):
        """Load model and tokenizer."""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        print(f"Loading model: {self.config.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.config.model_name)
        self.model.to(self.config.device)
        self.model.eval()
        
        # Set pad token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print(f"Model loaded on {self.config.device}")
        
    def _register_hooks(self, layer_indices: List[int]):
        """Register forward hooks to capture activations."""
        self._clear_hooks()
        
        for idx in layer_indices:
            layer = self.model.transformer.h[idx]
            
            def hook_fn(module, inp, out, layer_idx=idx):
                # out is tuple, first element is hidden states
                self.activations[layer_idx] = out[0].detach().cpu()
            
            self.hooks[idx] = layer.register_forward_hook(hook_fn)
    
    def _clear_hooks(self):
        """Remove all hooks."""
        for handle in self.hooks.values():
            handle.remove()
        self.hooks = {}
        self.activations = {}
    
    def extract_under_task(
        self, 
        input_text: str, 
        task_prompt: str
    ) -> Dict:
        """
        Extract activations when processing input under a specific task framing.
        
        Args:
            input_text: The signal to process
            task_prompt: The task framing
        
        Returns:
            Dict with activations, attention patterns, and output distribution
        """
        assert self.model is not None, "Model not loaded. Call load_model() first."
        
        # Format prompt
        full_prompt = f"{task_prompt}\n\nContext: {input_text}\n\nAnswer:"
        
        # Tokenize
        inputs = self.tokenizer(
            full_prompt, 
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.config.device)
        
        # Register hooks
        self._register_hooks(self.config.layers_to_analyze)
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(
                **inputs, 
                output_attentions=True,
                output_hidden_states=True
            )
        
        # Collect results
        results = {
            'prompt': full_prompt,
            'input_ids': inputs['input_ids'].cpu(),
            'activations': dict(self.activations),
            'attention_patterns': [a.detach().cpu() for a in outputs.attentions],
            'logits': outputs.logits.detach().cpu(),
            'output_distribution': torch.softmax(outputs.logits[0, -1, :], dim=-1).cpu()
        }
        
        # Cleanup
        self._clear_hooks()
        
        return results
    
    def compute_activation_similarity(
        self,
        results_a: Dict,
        results_b: Dict
    ) -> Dict[int, float]:
        """
        Compute cosine similarity between activations under two tasks.
        
        Returns:
            Dict mapping layer_idx to similarity score
        """
        similarities = {}
        
        for layer_idx in self.config.layers_to_analyze:
            if layer_idx in results_a['activations'] and layer_idx in results_b['activations']:
                act_a = results_a['activations'][layer_idx].flatten()
                act_b = results_b['activations'][layer_idx].flatten()
                
                # Handle different lengths by truncating to shorter
                min_len = min(len(act_a), len(act_b))
                act_a = act_a[:min_len]
                act_b = act_b[:min_len]
                
                # Cosine similarity
                sim = torch.nn.functional.cosine_similarity(
                    act_a.unsqueeze(0), 
                    act_b.unsqueeze(0)
                ).item()
                
                similarities[layer_idx] = sim
        
        return similarities
    
    def compute_attention_overlap(
        self,
        results_a: Dict,
        results_b: Dict,
        top_k: int = 10
    ) -> Dict[int, float]:
        """
        Compute overlap in top-attended tokens between two tasks.
        
        Returns:
            Dict mapping layer_idx to overlap ratio (0 to 1)
        """
        overlaps = {}
        
        for layer_idx in range(len(results_a['attention_patterns'])):
            # Get attention from last token (query) to all previous tokens
            attn_a = results_a['attention_patterns'][layer_idx][0, :, -1, :].mean(dim=0)
            attn_b = results_b['attention_patterns'][layer_idx][0, :, -1, :].mean(dim=0)
            
            # Get top-k attended positions
            min_len = min(len(attn_a), len(attn_b))
            top_a = set(attn_a[:min_len].topk(min(top_k, min_len)).indices.tolist())
            top_b = set(attn_b[:min_len].topk(min(top_k, min_len)).indices.tolist())
            
            # Compute overlap
            overlap = len(top_a & top_b) / len(top_a | top_b) if top_a | top_b else 0
            overlaps[layer_idx] = overlap
        
        return overlaps
    
    def compute_feature_importance(
        self,
        input_text: str,
        task_prompt: str,
        layer_idx: int
    ) -> torch.Tensor:
        """
        Compute feature importance via gradient-based attribution.
        
        Returns importance scores for each feature dimension.
        """
        assert self.model is not None, "Model not loaded"
        
        full_prompt = f"{task_prompt}\n\nContext: {input_text}\n\nAnswer:"
        inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.config.device)
        
        # Get embeddings with gradients
        embeddings = self.model.get_input_embeddings()(inputs['input_ids'])
        embeddings.requires_grad = True
        
        # Forward pass
        outputs = self.model(inputs_embeds=embeddings)
        
        # Get predicted token and compute gradient
        pred_logit = outputs.logits[0, -1, :].max()
        pred_logit.backward()
        
        # Feature importance = gradient magnitude
        importance = embeddings.grad.abs().mean(dim=(0, 1))
        
        return importance.detach().cpu()


# ============================================================================
# EXPERIMENT RUNNER
# ============================================================================

class ExperimentRunner:
    """Run the full task-relative AQ extraction experiment."""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.extractor = TaskRelativeAQExtractor(config)
        self.results = {}
        
    def run(self) -> Dict:
        """Run the complete experiment."""
        print("=" * 60)
        print("EXPERIMENT 037: Task-Relative AQ Extraction")
        print("=" * 60)
        
        # Load model
        self.extractor.load_model()
        
        # Phase 1: Activation comparison
        print("\n--- Phase 1: Activation Comparison ---")
        activation_results = self._run_activation_comparison()
        
        # Phase 2: Attention overlap
        print("\n--- Phase 2: Attention Overlap ---")
        attention_results = self._run_attention_analysis()
        
        # Phase 3: Feature importance
        print("\n--- Phase 3: Feature Importance ---")
        importance_results = self._run_importance_analysis()
        
        # Compile results
        self.results = {
            'config': {
                'model_name': self.config.model_name,
                'layers_analyzed': self.config.layers_to_analyze
            },
            'activation_comparison': activation_results,
            'attention_overlap': attention_results,
            'feature_importance': importance_results
        }
        
        # Analyze and print summary
        self._print_summary()
        
        return self.results
    
    def _run_activation_comparison(self) -> Dict:
        """Compare activations across tasks for each scenario."""
        results = {}
        
        for scenario_name, scenario in TASK_SCENARIOS.items():
            print(f"\nScenario: {scenario_name}")
            input_text = scenario['input']
            tasks = scenario['tasks']
            
            # Extract for each task
            task_extractions = {}
            for task_name, task_prompt in tasks.items():
                extraction = self.extractor.extract_under_task(input_text, task_prompt)
                task_extractions[task_name] = extraction
            
            # Compare all pairs
            task_names = list(tasks.keys())
            similarities = {}
            
            for i, task_a in enumerate(task_names):
                for task_b in task_names[i+1:]:
                    pair_name = f"{task_a}_vs_{task_b}"
                    sim = self.extractor.compute_activation_similarity(
                        task_extractions[task_a],
                        task_extractions[task_b]
                    )
                    similarities[pair_name] = sim
                    
                    # Print average similarity
                    avg_sim = np.mean(list(sim.values()))
                    print(f"  {pair_name}: avg similarity = {avg_sim:.3f}")
            
            results[scenario_name] = {
                'input': input_text,
                'similarities': similarities
            }
        
        return results
    
    def _run_attention_analysis(self) -> Dict:
        """Analyze attention pattern overlap across tasks."""
        results = {}
        
        for scenario_name, scenario in TASK_SCENARIOS.items():
            print(f"\nScenario: {scenario_name}")
            input_text = scenario['input']
            tasks = scenario['tasks']
            
            # Extract for each task
            task_extractions = {}
            for task_name, task_prompt in tasks.items():
                extraction = self.extractor.extract_under_task(input_text, task_prompt)
                task_extractions[task_name] = extraction
            
            # Compare attention overlap
            task_names = list(tasks.keys())
            overlaps = {}
            
            for i, task_a in enumerate(task_names):
                for task_b in task_names[i+1:]:
                    pair_name = f"{task_a}_vs_{task_b}"
                    overlap = self.extractor.compute_attention_overlap(
                        task_extractions[task_a],
                        task_extractions[task_b]
                    )
                    overlaps[pair_name] = overlap
                    
                    avg_overlap = np.mean(list(overlap.values()))
                    print(f"  {pair_name}: avg attention overlap = {avg_overlap:.3f}")
            
            results[scenario_name] = {'overlaps': overlaps}
        
        return results
    
    def _run_importance_analysis(self) -> Dict:
        """Analyze feature importance across tasks."""
        results = {}
        
        # Use first scenario for detailed importance analysis
        scenario_name = 'visual_scene'
        scenario = TASK_SCENARIOS[scenario_name]
        input_text = scenario['input']
        tasks = scenario['tasks']
        
        print(f"\nDetailed importance analysis for: {scenario_name}")
        
        importances = {}
        for task_name, task_prompt in tasks.items():
            importance = self.extractor.compute_feature_importance(
                input_text, task_prompt, layer_idx=6
            )
            importances[task_name] = importance
            
            # Top features
            top_k = 10
            top_indices = importance.topk(top_k).indices.tolist()
            print(f"  {task_name}: top feature indices = {top_indices}")
        
        # Compute overlap in important features
        task_names = list(tasks.keys())
        importance_overlaps = {}
        
        for i, task_a in enumerate(task_names):
            for task_b in task_names[i+1:]:
                top_a = set(importances[task_a].topk(50).indices.tolist())
                top_b = set(importances[task_b].topk(50).indices.tolist())
                
                jaccard = len(top_a & top_b) / len(top_a | top_b)
                pair_name = f"{task_a}_vs_{task_b}"
                importance_overlaps[pair_name] = jaccard
        
        results[scenario_name] = {
            'importance_overlaps': importance_overlaps
        }
        
        return results
    
    def _print_summary(self):
        """Print experiment summary."""
        print("\n" + "=" * 60)
        print("EXPERIMENT SUMMARY")
        print("=" * 60)
        
        # Activation divergence
        print("\n1. ACTIVATION DIVERGENCE")
        for scenario, data in self.results['activation_comparison'].items():
            sims = []
            for pair, layer_sims in data['similarities'].items():
                avg = np.mean(list(layer_sims.values()))
                sims.append(avg)
            overall_avg = np.mean(sims) if sims else 0
            print(f"   {scenario}: mean similarity = {overall_avg:.3f}")
        
        # Attention overlap
        print("\n2. ATTENTION OVERLAP")
        for scenario, data in self.results['attention_overlap'].items():
            overlaps = []
            for pair, layer_overlaps in data['overlaps'].items():
                avg = np.mean(list(layer_overlaps.values()))
                overlaps.append(avg)
            overall_avg = np.mean(overlaps) if overlaps else 0
            print(f"   {scenario}: mean overlap = {overall_avg:.3f}")
        
        # Verdict
        print("\n3. HYPOTHESIS TEST")
        all_sims = []
        for scenario, data in self.results['activation_comparison'].items():
            for pair, layer_sims in data['similarities'].items():
                all_sims.extend(layer_sims.values())
        
        mean_sim = np.mean(all_sims) if all_sims else 1.0
        
        if mean_sim < 0.9:
            print(f"   SUPPORTED: Mean similarity {mean_sim:.3f} < 0.9")
            print("   Different tasks produce divergent activations")
        else:
            print(f"   NOT SUPPORTED: Mean similarity {mean_sim:.3f} >= 0.9")
            print("   Tasks do not significantly alter activations")
    
    def save_results(self, path: str):
        """Save results to JSON file."""
        # Convert tensors to lists for JSON serialization
        def convert(obj):
            if isinstance(obj, torch.Tensor):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(v) for v in obj]
            else:
                return obj
        
        serializable = convert(self.results)
        
        with open(path, 'w') as f:
            json.dump(serializable, f, indent=2)
        
        print(f"Results saved to {path}")
    
    def plot_divergence_by_layer(self, save_path: Optional[str] = None):
        """Plot activation divergence by layer."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        for idx, (scenario, data) in enumerate(self.results['activation_comparison'].items()):
            if idx >= 4:
                break
            
            ax = axes[idx]
            
            for pair, layer_sims in data['similarities'].items():
                layers = sorted(layer_sims.keys())
                sims = [layer_sims[l] for l in layers]
                ax.plot(layers, sims, marker='o', label=pair[:20], alpha=0.7)
            
            ax.set_xlabel('Layer')
            ax.set_ylabel('Cosine Similarity')
            ax.set_title(f'{scenario}')
            ax.set_ylim(0, 1.1)
            ax.legend(fontsize=6)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"Plot saved to {save_path}")
        
        plt.show()


# ============================================================================
# MAIN
# ============================================================================

def run_experiment(model_name: str = "gpt2") -> Dict:
    """
    Run the complete task-relative AQ extraction experiment.
    
    Args:
        model_name: HuggingFace model name
    
    Returns:
        Experiment results dictionary
    """
    config = ExperimentConfig(model_name=model_name)
    runner = ExperimentRunner(config)
    results = runner.run()
    return results


if __name__ == "__main__":
    # Run experiment
    results = run_experiment("gpt2")
    
    # Save results
    runner = ExperimentRunner(ExperimentConfig())
    runner.results = results
    runner.save_results("037_results.json")
    runner.plot_divergence_by_layer("037_divergence_plot.png")
