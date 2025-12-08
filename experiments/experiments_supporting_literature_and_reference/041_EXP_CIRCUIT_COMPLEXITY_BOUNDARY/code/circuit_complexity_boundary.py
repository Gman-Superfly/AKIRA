"""
Experiment 041: Circuit Complexity Boundary

Tests whether fixed-width networks have hard performance boundaries 
determined by SOS width.

Hypothesis: For breadth B=8 and predicate arity beta=2, problems 
with SOS width k > 3 should systematically fail.

Usage:
    python circuit_complexity_boundary.py
"""

from typing import Dict, List, Tuple, Optional
import torch
import numpy as np
from dataclasses import dataclass
from enum import Enum
import json
import matplotlib.pyplot as plt


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class ExperimentConfig:
    """Configuration for circuit complexity experiment."""
    model_name: str = "gpt2"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    n_samples_per_k: int = 50  # Samples per SOS width level
    max_k: int = 5  # Maximum SOS width to test
    breadth: int = 8  # Network breadth (AKIRA-like)
    predicate_arity: int = 2  # Beta value


# ============================================================================
# TASK DEFINITIONS
# ============================================================================

class TaskGenerator:
    """
    Generate tasks with controlled SOS width.
    
    SOS width k = maximum number of constraints that must be tracked
    simultaneously for an optimal solution.
    """
    
    def __init__(self):
        self.objects = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    
    def generate_k1_tasks(self, n: int = 50) -> List[Dict]:
        """Generate k=1 tasks: single object tracking."""
        tasks = []
        for i in range(n):
            obj = self.objects[i % len(self.objects)]
            tasks.append({
                'k': 1,
                'prompt': f"A {obj} is on the table. Where is {obj}?",
                'answer': 'table',
                'description': f"Track single object {obj}"
            })
        return tasks
    
    def generate_k2_tasks(self, n: int = 50) -> List[Dict]:
        """Generate k=2 tasks: object + one relationship."""
        tasks = []
        for i in range(n):
            obj1 = self.objects[i % len(self.objects)]
            obj2 = self.objects[(i + 1) % len(self.objects)]
            tasks.append({
                'k': 2,
                'prompt': f"{obj1} is on the table. {obj2} is on {obj1}. Where is {obj2}?",
                'answer': obj1,
                'description': f"Track {obj2} on {obj1}"
            })
        return tasks
    
    def generate_k3_tasks(self, n: int = 50) -> List[Dict]:
        """Generate k=3 tasks: object + two relationships."""
        tasks = []
        for i in range(n):
            obj1 = self.objects[i % len(self.objects)]
            obj2 = self.objects[(i + 1) % len(self.objects)]
            obj3 = self.objects[(i + 2) % len(self.objects)]
            tasks.append({
                'k': 3,
                'prompt': f"{obj1} is on the table. {obj2} is on {obj1}. {obj3} is on {obj2}. "
                         f"What is on top of the stack?",
                'answer': obj3,
                'description': f"Track 3-object stack"
            })
        return tasks
    
    def generate_k4_tasks(self, n: int = 50) -> List[Dict]:
        """Generate k=4 tasks: object + three relationships (beyond B=8 capacity)."""
        tasks = []
        for i in range(n):
            objs = [self.objects[(i + j) % len(self.objects)] for j in range(4)]
            tasks.append({
                'k': 4,
                'prompt': f"{objs[0]} is on the table. {objs[1]} is on {objs[0]}. "
                         f"{objs[2]} is on {objs[1]}. {objs[3]} is on {objs[2]}. "
                         f"If we remove {objs[1]}, where does {objs[3]} fall?",
                'answer': 'table',  # Simplified expected answer
                'description': f"Track 4-object stack with removal"
            })
        return tasks
    
    def generate_k5_tasks(self, n: int = 50) -> List[Dict]:
        """Generate k=5 tasks: well beyond B=8 capacity."""
        tasks = []
        for i in range(n):
            objs = [self.objects[(i + j) % len(self.objects)] for j in range(5)]
            tasks.append({
                'k': 5,
                'prompt': f"{objs[0]} is on the table. {objs[1]} is on {objs[0]}. "
                         f"{objs[2]} is on {objs[1]}. {objs[3]} is on {objs[2]}. "
                         f"{objs[4]} is on {objs[3]}. "
                         f"What is the order from bottom to top?",
                'answer': f"{objs[0]}, {objs[1]}, {objs[2]}, {objs[3]}, {objs[4]}",
                'description': f"Track 5-object stack order"
            })
        return tasks
    
    def generate_all_tasks(self, n_per_k: int = 50) -> Dict[int, List[Dict]]:
        """Generate tasks for all k values."""
        return {
            1: self.generate_k1_tasks(n_per_k),
            2: self.generate_k2_tasks(n_per_k),
            3: self.generate_k3_tasks(n_per_k),
            4: self.generate_k4_tasks(n_per_k),
            5: self.generate_k5_tasks(n_per_k)
        }


# ============================================================================
# VISUAL TRACKING TASKS (More Relevant to AKIRA)
# ============================================================================

class VisualTrackingTaskGenerator:
    """Generate visual tracking tasks with controlled complexity."""
    
    def generate_k1_visual(self, n: int = 50) -> List[Dict]:
        """k=1: Track one object's property."""
        colors = ['red', 'blue', 'green', 'yellow']
        objects = ['ball', 'box', 'cup', 'block']
        tasks = []
        
        for i in range(n):
            color = colors[i % len(colors)]
            obj = objects[i % len(objects)]
            tasks.append({
                'k': 1,
                'prompt': f"There is a {color} {obj}. What color is the {obj}?",
                'answer': color,
                'description': f"Track single property"
            })
        return tasks
    
    def generate_k2_visual(self, n: int = 50) -> List[Dict]:
        """k=2: Track object + one relationship."""
        tasks = []
        positions = ['left', 'right', 'center']
        
        for i in range(n):
            pos = positions[i % len(positions)]
            tasks.append({
                'k': 2,
                'prompt': f"A ball starts on the {pos}. It moves to the opposite side. "
                         f"Where is the ball now?",
                'answer': 'right' if pos == 'left' else 'left' if pos == 'right' else 'center',
                'description': f"Track position change"
            })
        return tasks
    
    def generate_k3_visual(self, n: int = 50) -> List[Dict]:
        """k=3: Track object + two relationships."""
        tasks = []
        
        for i in range(n):
            tasks.append({
                'k': 3,
                'prompt': f"Ball A is red and on the left. Ball B is blue and on the right. "
                         f"They swap positions. What color is the ball on the left now?",
                'answer': 'blue',
                'description': f"Track multiple objects with swap"
            })
        return tasks
    
    def generate_k4_visual(self, n: int = 50) -> List[Dict]:
        """k=4: Beyond boundary - multiple objects with interactions."""
        tasks = []
        
        for i in range(n):
            tasks.append({
                'k': 4,
                'prompt': f"Ball A is red at position 1. Ball B is blue at position 2. "
                         f"Ball C is green at position 3. Ball D is yellow at position 4. "
                         f"A and C swap. B and D swap. What color is at position 3?",
                'answer': 'red',  # A moved to 3
                'description': f"Track 4 objects with swaps"
            })
        return tasks
    
    def generate_all_tasks(self, n_per_k: int = 50) -> Dict[int, List[Dict]]:
        """Generate visual tasks for k=1 to 4."""
        return {
            1: self.generate_k1_visual(n_per_k),
            2: self.generate_k2_visual(n_per_k),
            3: self.generate_k3_visual(n_per_k),
            4: self.generate_k4_visual(n_per_k)
        }


# ============================================================================
# EVALUATOR
# ============================================================================

class ComplexityBoundaryEvaluator:
    """Evaluate model performance across SOS width boundary."""
    
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
    
    def evaluate_task(self, task: Dict) -> Dict:
        """Evaluate model on a single task."""
        prompt = task['prompt']
        expected = task['answer'].lower()
        
        # Generate response
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.config.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=20,
                temperature=0.1,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated = response[len(prompt):].strip().lower()
        
        # Check if answer is in response
        correct = expected in generated or generated in expected
        
        # More lenient: check if key words match
        expected_words = set(expected.split())
        generated_words = set(generated.split())
        partial_match = len(expected_words & generated_words) / max(len(expected_words), 1)
        
        return {
            'prompt': prompt,
            'expected': expected,
            'generated': generated,
            'correct': correct,
            'partial_match': partial_match
        }
    
    def evaluate_k_level(self, tasks: List[Dict]) -> Dict:
        """Evaluate all tasks at a given k level."""
        results = []
        correct = 0
        
        for task in tasks:
            result = self.evaluate_task(task)
            results.append(result)
            if result['correct']:
                correct += 1
        
        accuracy = correct / len(tasks) if tasks else 0
        mean_partial = np.mean([r['partial_match'] for r in results])
        
        return {
            'n_tasks': len(tasks),
            'correct': correct,
            'accuracy': accuracy,
            'mean_partial_match': mean_partial,
            'results': results
        }


# ============================================================================
# EXPERIMENT RUNNER
# ============================================================================

class ExperimentRunner:
    """Run the circuit complexity boundary experiment."""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.evaluator = ComplexityBoundaryEvaluator(config)
        self.results = {}
        
    def run(self) -> Dict:
        """Run the complete experiment."""
        print("=" * 60)
        print("EXPERIMENT 041: Circuit Complexity Boundary")
        print("=" * 60)
        
        # Load model
        self.evaluator.load_model()
        
        # Generate tasks
        print("\n--- Generating Tasks ---")
        generator = TaskGenerator()
        all_tasks = generator.generate_all_tasks(self.config.n_samples_per_k)
        
        # Evaluate each k level
        print("\n--- Evaluating by SOS Width ---")
        k_results = {}
        
        for k in range(1, self.config.max_k + 1):
            print(f"\nk = {k}:")
            if k in all_tasks:
                result = self.evaluator.evaluate_k_level(all_tasks[k])
                k_results[k] = result
                print(f"  Accuracy: {result['accuracy']:.1%}")
                print(f"  Partial match: {result['mean_partial_match']:.3f}")
        
        # Compute boundary
        print("\n--- Analyzing Boundary ---")
        boundary_analysis = self._analyze_boundary(k_results)
        
        # Compile results
        self.results = {
            'config': {
                'model_name': self.config.model_name,
                'breadth': self.config.breadth,
                'predicate_arity': self.config.predicate_arity,
                'predicted_max_k': (self.config.breadth // self.config.predicate_arity) - 1
            },
            'k_results': {k: {'accuracy': r['accuracy'], 'n_tasks': r['n_tasks']} 
                         for k, r in k_results.items()},
            'boundary_analysis': boundary_analysis
        }
        
        self._print_summary()
        
        return self.results
    
    def _analyze_boundary(self, k_results: Dict) -> Dict:
        """Analyze where the performance boundary occurs."""
        k_values = sorted(k_results.keys())
        accuracies = [k_results[k]['accuracy'] for k in k_values]
        
        # Find largest drop
        drops = []
        for i in range(len(accuracies) - 1):
            drop = accuracies[i] - accuracies[i + 1]
            drops.append({
                'from_k': k_values[i],
                'to_k': k_values[i + 1],
                'drop': drop
            })
        
        if drops:
            max_drop = max(drops, key=lambda x: x['drop'])
        else:
            max_drop = {'from_k': 0, 'to_k': 0, 'drop': 0}
        
        # Predicted boundary
        predicted_boundary = (self.config.breadth // self.config.predicate_arity) - 1 + 1
        
        # Check if boundary matches prediction
        observed_boundary = max_drop['to_k'] if max_drop['drop'] > 0.2 else None
        
        return {
            'predicted_boundary_k': predicted_boundary,
            'observed_boundary_k': observed_boundary,
            'max_drop': max_drop,
            'drops': drops,
            'boundary_matches': observed_boundary == predicted_boundary if observed_boundary else False
        }
    
    def _print_summary(self):
        """Print experiment summary."""
        print("\n" + "=" * 60)
        print("EXPERIMENT SUMMARY")
        print("=" * 60)
        
        config = self.results['config']
        k_results = self.results['k_results']
        boundary = self.results['boundary_analysis']
        
        print(f"\n1. CONFIGURATION:")
        print(f"   Breadth B = {config['breadth']}")
        print(f"   Predicate arity beta = {config['predicate_arity']}")
        print(f"   Predicted max k = (B/beta) - 1 = {config['predicted_max_k']}")
        print(f"   Predicted boundary at k = {boundary['predicted_boundary_k']}")
        
        print(f"\n2. ACCURACY BY SOS WIDTH:")
        for k, data in k_results.items():
            status = "OK" if data['accuracy'] > 0.7 else "FAIL" if data['accuracy'] < 0.5 else "WEAK"
            print(f"   k = {k}: {data['accuracy']:.1%} [{status}]")
        
        print(f"\n3. BOUNDARY ANALYSIS:")
        print(f"   Max drop: k={boundary['max_drop']['from_k']} -> k={boundary['max_drop']['to_k']} "
              f"({boundary['max_drop']['drop']:.1%})")
        print(f"   Observed boundary: k = {boundary['observed_boundary_k']}")
        print(f"   Predicted boundary: k = {boundary['predicted_boundary_k']}")
        
        print(f"\n4. VERDICT:")
        
        # Check if results match theory
        acc_k3 = k_results.get(3, {}).get('accuracy', 0)
        acc_k4 = k_results.get(4, {}).get('accuracy', 1)
        sharp_drop = acc_k3 - acc_k4 > 0.3
        
        if sharp_drop and boundary['boundary_matches']:
            print("   HYPOTHESIS SUPPORTED")
            print(f"   - Sharp drop at predicted boundary (k={boundary['predicted_boundary_k']})")
            print(f"   - k <= {config['predicted_max_k']}: High accuracy")
            print(f"   - k > {config['predicted_max_k']}: Low accuracy")
            print("   - Circuit complexity theory validated")
        else:
            print("   HYPOTHESIS NOT SUPPORTED")
            if not sharp_drop:
                print(f"   - No sharp drop: k=3 ({acc_k3:.1%}) vs k=4 ({acc_k4:.1%})")
            if not boundary['boundary_matches']:
                print(f"   - Boundary mismatch: observed={boundary['observed_boundary_k']}, "
                      f"predicted={boundary['predicted_boundary_k']}")
    
    def plot_results(self, save_path: str = None):
        """Plot experiment results."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        k_results = self.results['k_results']
        boundary = self.results['boundary_analysis']
        
        # Plot 1: Accuracy by k
        ax1 = axes[0]
        k_values = sorted(k_results.keys())
        accuracies = [k_results[k]['accuracy'] for k in k_values]
        
        colors = ['green' if acc > 0.7 else 'orange' if acc > 0.4 else 'red' for acc in accuracies]
        ax1.bar(k_values, accuracies, color=colors, alpha=0.7, edgecolor='black')
        ax1.axvline(x=boundary['predicted_boundary_k'] - 0.5, 
                    color='red', linestyle='--', linewidth=2, label='Predicted Boundary')
        ax1.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
        ax1.set_xlabel('SOS Width (k)', fontsize=12)
        ax1.set_ylabel('Accuracy', fontsize=12)
        ax1.set_title('Accuracy by SOS Width', fontsize=14)
        ax1.set_ylim(0, 1.1)
        ax1.legend()
        ax1.set_xticks(k_values)
        
        # Plot 2: Drop magnitude
        ax2 = axes[1]
        drops = boundary['drops']
        if drops:
            labels = [f"k={d['from_k']}->k={d['to_k']}" for d in drops]
            drop_values = [d['drop'] for d in drops]
            colors = ['red' if d > 0.3 else 'orange' if d > 0.1 else 'green' for d in drop_values]
            ax2.bar(range(len(drops)), drop_values, color=colors, alpha=0.7, edgecolor='black')
            ax2.set_xticks(range(len(drops)))
            ax2.set_xticklabels(labels)
            ax2.axhline(y=0.3, color='red', linestyle='--', label='Significant drop threshold')
            ax2.set_xlabel('Transition', fontsize=12)
            ax2.set_ylabel('Accuracy Drop', fontsize=12)
            ax2.set_title('Accuracy Drop at Each Transition', fontsize=14)
            ax2.legend()
        
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
    runner.plot_results("041_circuit_complexity_plot.png")
    return results


if __name__ == "__main__":
    results = run_experiment("gpt2")
