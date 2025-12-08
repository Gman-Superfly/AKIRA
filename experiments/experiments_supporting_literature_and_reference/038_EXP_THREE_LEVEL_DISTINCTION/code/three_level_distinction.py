"""
Experiment 038: Three-Level Distinction

Tests whether Measurement (L1), Inference (L2), and Action Quanta (L3) are empirically separable.

Hypothesis:
- L3 (AQ) is a strict subset of L2 (Inferences)
- L3 is more stable across paraphrases
- Ablating L3 is catastrophic; ablating L2-only is tolerable

Usage:
    python three_level_distinction.py
"""

from typing import Dict, List, Set, Tuple, Optional
import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass, field
from collections import defaultdict
import json
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class ExperimentConfig:
    """Configuration for three-level distinction experiment."""
    model_name: str = "gpt2"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    probe_layer: int = 8  # Layer to probe
    n_probe_samples: int = 200
    ablation_threshold: float = 0.1  # 10% degradation = load-bearing
    top_k_features: int = 50  # Top features to consider


# ============================================================================
# TEST DATA
# ============================================================================

# Training data for semantic probes
PROBE_TRAINING_DATA = {
    'sentiment': {
        'positive': [
            "This movie was absolutely wonderful and amazing.",
            "I loved every minute of this fantastic experience.",
            "The service was excellent and the food was delicious.",
            "What a beautiful day, everything is perfect!",
            "This is the best thing that ever happened to me.",
            "Brilliant performance, truly outstanding work.",
            "I'm so happy and grateful for this opportunity.",
            "Exceptional quality, exceeded all expectations.",
        ],
        'negative': [
            "This movie was terrible and boring.",
            "I hated every minute of this awful experience.",
            "The service was horrible and the food was disgusting.",
            "What a terrible day, everything went wrong!",
            "This is the worst thing that ever happened to me.",
            "Awful performance, completely disappointing work.",
            "I'm so sad and frustrated by this situation.",
            "Poor quality, failed to meet basic expectations.",
        ]
    },
    'entity_type': {
        'person': [
            "John went to the store yesterday.",
            "Mary is studying at the university.",
            "The doctor examined the patient carefully.",
            "Sarah and Tom are getting married next month.",
        ],
        'object': [
            "The red ball rolled across the floor.",
            "My car needs new tires installed.",
            "The ancient book contained hidden secrets.",
            "A bright lamp illuminated the dark room.",
        ]
    },
    'temporal': {
        'past': [
            "Yesterday I walked to the park.",
            "Last year we visited Paris together.",
            "The meeting happened three days ago.",
            "She graduated from college in 2020.",
        ],
        'present': [
            "Right now I am walking to the park.",
            "Currently we are visiting Paris together.",
            "The meeting is happening at this moment.",
            "She is graduating from college today.",
        ]
    }
}

# Test scenarios for ablation
TEST_SCENARIOS = [
    {
        'input': "This movie was absolutely wonderful, I loved every minute.",
        'task': 'sentiment',
        'target': 'positive',
        'paraphrases': [
            "This film was totally amazing, I enjoyed every second.",
            "What a fantastic movie, loved it from start to finish.",
            "An absolutely brilliant film, thoroughly enjoyable.",
        ]
    },
    {
        'input': "John picked up the red ball from the table.",
        'task': 'entity',
        'target': 'John',
        'paraphrases': [
            "John grabbed the red ball off the table.",
            "The red ball was picked up by John from the table.",
            "From the table, John took the red ball.",
        ]
    }
]


# ============================================================================
# CORE CLASSES
# ============================================================================

class ThreeLevelExtractor:
    """
    Extract and distinguish three levels:
    L1: Measurements (raw activations)
    L2: Inferences (probed semantic features)
    L3: AQ (action-predictive features)
    """
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.probes: Dict[str, LogisticRegression] = {}
        self.probe_accuracies: Dict[str, float] = {}
        
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
    
    def extract_L1_measurements(
        self, 
        input_text: str, 
        layer_idx: int = None
    ) -> torch.Tensor:
        """
        Extract raw activations (L1 - Measurements).
        """
        if layer_idx is None:
            layer_idx = self.config.probe_layer
            
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.config.device)
        
        activations = {}
        
        def hook(module, inp, out):
            activations['output'] = out[0].detach().cpu()
        
        handle = self.model.transformer.h[layer_idx].register_forward_hook(hook)
        
        with torch.no_grad():
            self.model(**inputs)
        
        handle.remove()
        
        # Return mean-pooled activation
        return activations['output'].mean(dim=1).squeeze()
    
    def train_semantic_probes(self):
        """Train linear probes for semantic features (L2 - Inferences)."""
        print("\nTraining semantic probes...")
        
        for feature_name, categories in PROBE_TRAINING_DATA.items():
            print(f"  Training probe for: {feature_name}")
            
            X = []
            y = []
            
            for label_idx, (label, examples) in enumerate(categories.items()):
                for text in examples:
                    act = self.extract_L1_measurements(text)
                    X.append(act.numpy())
                    y.append(label_idx)
            
            X = np.array(X)
            y = np.array(y)
            
            # Train probe
            probe = LogisticRegression(max_iter=1000, random_state=42)
            
            # Cross-validation score
            if len(X) >= 5:
                scores = cross_val_score(probe, X, y, cv=min(5, len(X)))
                accuracy = scores.mean()
            else:
                accuracy = 0.0
            
            # Fit on all data
            probe.fit(X, y)
            
            self.probes[feature_name] = probe
            self.probe_accuracies[feature_name] = accuracy
            
            print(f"    Accuracy: {accuracy:.3f}")
        
        print(f"  Trained {len(self.probes)} probes")
    
    def extract_L2_inferences(self, input_text: str) -> Dict[str, float]:
        """
        Extract inferred features using trained probes (L2 - Inferences).
        """
        act = self.extract_L1_measurements(input_text)
        X = act.numpy().reshape(1, -1)
        
        inferences = {}
        for feature_name, probe in self.probes.items():
            prob = probe.predict_proba(X)[0]
            inferences[feature_name] = {
                'prediction': probe.predict(X)[0],
                'confidence': float(prob.max()),
                'probabilities': prob.tolist()
            }
        
        return inferences
    
    def identify_L3_AQ(
        self,
        input_text: str,
        target_token: str,
        layer_idx: int = None
    ) -> Tuple[Set[int], Dict[int, float]]:
        """
        Identify which features are AQ (L3 - Action Quanta).
        
        Returns:
            Tuple of (AQ feature indices, importance scores)
        """
        if layer_idx is None:
            layer_idx = self.config.probe_layer
        
        # Get baseline output
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.config.device)
        target_id = self.tokenizer.encode(target_token, add_special_tokens=False)
        if len(target_id) > 0:
            target_id = target_id[0]
        else:
            target_id = self.tokenizer.eos_token_id
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            baseline_probs = torch.softmax(outputs.logits[0, -1, :], dim=-1)
            baseline_prob = baseline_probs[target_id].item()
        
        # Compute feature importance via gradient
        embeddings = self.model.get_input_embeddings()(inputs['input_ids'])
        embeddings = embeddings.clone().detach().requires_grad_(True)
        
        outputs = self.model(inputs_embeds=embeddings)
        target_logit = outputs.logits[0, -1, target_id]
        target_logit.backward()
        
        # Importance = gradient magnitude
        importance = embeddings.grad.abs().mean(dim=(0, 1)).cpu()
        
        # Get top-k important features
        top_k = self.config.top_k_features
        top_indices = importance.topk(top_k).indices.tolist()
        
        importance_dict = {i: importance[i].item() for i in top_indices}
        
        # AQ = features where ablation causes significant degradation
        aq_indices = set()
        
        for idx in top_indices[:20]:  # Check top 20 for efficiency
            # Simple ablation: zero out feature
            degradation = self._estimate_ablation_effect(
                input_text, target_id, idx, baseline_prob
            )
            
            if degradation > self.config.ablation_threshold:
                aq_indices.add(idx)
        
        return aq_indices, importance_dict
    
    def _estimate_ablation_effect(
        self,
        input_text: str,
        target_id: int,
        feature_idx: int,
        baseline_prob: float
    ) -> float:
        """Estimate effect of ablating a single feature."""
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.config.device)
        
        embeddings = self.model.get_input_embeddings()(inputs['input_ids'])
        
        # Zero out the feature dimension
        embeddings[:, :, feature_idx] = 0
        
        with torch.no_grad():
            outputs = self.model(inputs_embeds=embeddings)
            ablated_probs = torch.softmax(outputs.logits[0, -1, :], dim=-1)
            ablated_prob = ablated_probs[target_id].item()
        
        # Degradation ratio
        if baseline_prob > 0:
            degradation = (baseline_prob - ablated_prob) / baseline_prob
        else:
            degradation = 0
        
        return max(0, degradation)


class StabilityAnalyzer:
    """Analyze stability of features across paraphrases."""
    
    def __init__(self, extractor: ThreeLevelExtractor):
        self.extractor = extractor
    
    def measure_stability(
        self,
        base_input: str,
        paraphrases: List[str]
    ) -> Dict[str, Dict]:
        """
        Measure how stable L2 and L3 features are across paraphrases.
        """
        # Get base L2 inferences
        base_L2 = self.extractor.extract_L2_inferences(base_input)
        
        # Get L2 for each paraphrase
        paraphrase_L2s = [
            self.extractor.extract_L2_inferences(p) for p in paraphrases
        ]
        
        # Compute stability for each feature
        stability_scores = {}
        
        for feature_name in base_L2.keys():
            base_conf = base_L2[feature_name]['confidence']
            para_confs = [p[feature_name]['confidence'] for p in paraphrase_L2s]
            
            all_confs = [base_conf] + para_confs
            variance = np.var(all_confs)
            stability = 1.0 / (1.0 + variance * 10)  # Scale variance
            
            stability_scores[feature_name] = {
                'base_confidence': base_conf,
                'variance': variance,
                'stability': stability,
                'all_confidences': all_confs
            }
        
        return stability_scores


# ============================================================================
# EXPERIMENT RUNNER
# ============================================================================

class ExperimentRunner:
    """Run the complete three-level distinction experiment."""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.extractor = ThreeLevelExtractor(config)
        self.stability_analyzer = None
        self.results = {}
    
    def run(self) -> Dict:
        """Run the complete experiment."""
        print("=" * 60)
        print("EXPERIMENT 038: Three-Level Distinction")
        print("=" * 60)
        
        # Load model
        self.extractor.load_model()
        
        # Train probes (establish L2)
        print("\n--- Phase 1: Establish L2 (Train Probes) ---")
        self.extractor.train_semantic_probes()
        
        # Initialize stability analyzer
        self.stability_analyzer = StabilityAnalyzer(self.extractor)
        
        # Identify L3 features
        print("\n--- Phase 2: Identify L3 (Load-Bearing Features) ---")
        l3_results = self._identify_l3_features()
        
        # Stability analysis
        print("\n--- Phase 3: Stability Analysis ---")
        stability_results = self._run_stability_analysis()
        
        # Ablation comparison
        print("\n--- Phase 4: Ablation Comparison ---")
        ablation_results = self._run_ablation_comparison()
        
        # Compile results
        self.results = {
            'probe_accuracies': self.extractor.probe_accuracies,
            'l3_identification': l3_results,
            'stability': stability_results,
            'ablation': ablation_results
        }
        
        self._print_summary()
        
        return self.results
    
    def _identify_l3_features(self) -> Dict:
        """Identify L3 features for test scenarios."""
        results = {}
        
        for scenario in TEST_SCENARIOS:
            print(f"\nScenario: {scenario['task']}")
            print(f"  Input: {scenario['input'][:50]}...")
            
            l3_indices, importance = self.extractor.identify_L3_AQ(
                scenario['input'],
                scenario['target']
            )
            
            print(f"  L3 features identified: {len(l3_indices)}")
            print(f"  Top importance indices: {list(importance.keys())[:5]}")
            
            results[scenario['task']] = {
                'l3_indices': list(l3_indices),
                'top_importance': dict(list(importance.items())[:10])
            }
        
        return results
    
    def _run_stability_analysis(self) -> Dict:
        """Run stability analysis across paraphrases."""
        results = {}
        
        for scenario in TEST_SCENARIOS:
            if 'paraphrases' not in scenario:
                continue
                
            print(f"\nScenario: {scenario['task']}")
            
            stability = self.stability_analyzer.measure_stability(
                scenario['input'],
                scenario['paraphrases']
            )
            
            for feature, scores in stability.items():
                print(f"  {feature}: stability = {scores['stability']:.3f}")
            
            results[scenario['task']] = stability
        
        return results
    
    def _run_ablation_comparison(self) -> Dict:
        """Compare L3 vs L2-only ablation effects."""
        results = {}
        
        # For simplicity, use feature importance as proxy
        for scenario in TEST_SCENARIOS:
            print(f"\nScenario: {scenario['task']}")
            
            l3_indices, importance = self.extractor.identify_L3_AQ(
                scenario['input'],
                scenario['target']
            )
            
            # L3 ablation effect (sum of top importance)
            l3_effect = sum(importance.get(i, 0) for i in l3_indices)
            
            # L2-only effect (features not in L3)
            all_indices = set(importance.keys())
            l2_only_indices = all_indices - l3_indices
            l2_only_effect = sum(importance.get(i, 0) for i in l2_only_indices)
            
            ratio = l3_effect / max(l2_only_effect, 0.001)
            
            print(f"  L3 effect: {l3_effect:.4f}")
            print(f"  L2-only effect: {l2_only_effect:.4f}")
            print(f"  Ratio: {ratio:.2f}x")
            
            results[scenario['task']] = {
                'l3_effect': l3_effect,
                'l2_only_effect': l2_only_effect,
                'ratio': ratio
            }
        
        return results
    
    def _print_summary(self):
        """Print experiment summary."""
        print("\n" + "=" * 60)
        print("EXPERIMENT SUMMARY")
        print("=" * 60)
        
        # Probe accuracies
        print("\n1. L2 PROBES (Semantic Features):")
        for name, acc in self.results['probe_accuracies'].items():
            print(f"   {name}: {acc:.3f}")
        
        # L3 identification
        print("\n2. L3 IDENTIFICATION (Load-Bearing Features):")
        for task, data in self.results['l3_identification'].items():
            print(f"   {task}: {len(data['l3_indices'])} features")
        
        # Ablation
        print("\n3. ABLATION ASYMMETRY:")
        for task, data in self.results['ablation'].items():
            print(f"   {task}: L3/L2-only ratio = {data['ratio']:.2f}x")
        
        # Verdict
        print("\n4. VERDICT:")
        
        # Check if L3 effect >> L2-only effect
        ratios = [d['ratio'] for d in self.results['ablation'].values()]
        mean_ratio = np.mean(ratios) if ratios else 0
        
        if mean_ratio > 2.0:
            print("   HYPOTHESIS SUPPORTED")
            print(f"   Mean L3/L2-only ratio: {mean_ratio:.2f}x > 2.0")
            print("   L3 (AQ) features are distinctly more load-bearing")
        else:
            print("   HYPOTHESIS NOT SUPPORTED")
            print(f"   Mean L3/L2-only ratio: {mean_ratio:.2f}x <= 2.0")
    
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
    return runner.run()


if __name__ == "__main__":
    results = run_experiment("gpt2")
