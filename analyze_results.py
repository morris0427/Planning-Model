"""
Analyze and visualize experiment results.

Usage:
    python analyze_results.py --sweep-file results/sweep_summary.json
    python analyze_results.py --compare exp1 exp2 exp3
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict
import sys


class ResultsAnalyzer:
    """Analyze and compare experiment results."""
    
    def __init__(self, results_dir: str = "./results"):
        self.results_dir = Path(results_dir)
    
    def load_sweep(self, sweep_file: str) -> List[Dict]:
        """Load sweep results from JSON."""
        with open(sweep_file, 'r') as f:
            return json.load(f)
    
    def load_experiment(self, exp_name: str) -> Dict:
        """Load single experiment results."""
        exp_dir = self.results_dir / exp_name
        results_file = exp_dir / "results.json"
        
        if not results_file.exists():
            raise FileNotFoundError(f"Results not found: {results_file}")
        
        with open(results_file, 'r') as f:
            return json.load(f)
    
    def compare_model_sizes(self, results: List[Dict]):
        """Compare performance across model sizes."""
        print("\n" + "=" * 80)
        print("MODEL SIZE COMPARISON")
        print("=" * 80)
        
        # Check what metric we have (prioritize solve_rate)
        has_solve_rate = 'solve_rate' in results[0] if results else False
        has_test_loss = 'test_loss' in results[0] if results else False
        
        if has_solve_rate:
            metric_key = 'solve_rate'
            metric_name = "Solve Rate"
        elif has_test_loss:
            metric_key = 'test_loss'
            metric_name = "Test Loss"
        else:
            metric_key = 'test_accuracy'
            metric_name = "Accuracy"
        
        # Group by domain and model type
        grouped = {}
        for r in results:
            domain = r['domain']
            model = r['model']
            
            # Extract size and type
            parts = model.split('_')
            size = parts[0]  # tiny, small, medium, large
            model_type = 'WM' if 'WM' in model else 'Base'
            
            key = (domain, model_type)
            if key not in grouped:
                grouped[key] = []
            grouped[key].append((size, r[metric_key]))
        
        # Print tables
        for (domain, model_type), data in sorted(grouped.items()):
            print(f"\n{domain.upper()} - {model_type}")
            print("-" * 50)
            print(f"{'Size':<12} {metric_name:<15} {'Params':<12}")
            print("-" * 50)
            
            # Sort by size order
            size_order = {'tiny': 0, 'small': 1, 'medium': 2, 'large': 3}
            sorted_data = sorted(data, key=lambda x: size_order.get(x[0], 99))
            
            for size, value in sorted_data:
                param_est = self._estimate_params(size)
                if has_solve_rate:
                    print(f"{size:<12} {value:<15.1%} {param_est:<12}")
                elif metric_key == 'test_loss':
                    print(f"{size:<12} {value:<15.4f} {param_est:<12}")
                else:
                    print(f"{size:<12} {value:<15.1%} {param_est:<12}")
        
        # Print WM advantage by size
        print("\n" + "=" * 80)
        print("WORLD MODEL ADVANTAGE BY SIZE")
        print("=" * 80)
        
        for domain in set(r['domain'] for r in results):
            print(f"\n{domain.upper()}")
            print("-" * 60)
            
            if has_solve_rate:
                print(f"{'Size':<12} {'Base Rate':<15} {'WM Rate':<15} {'Δ':<12}")
            elif metric_key == 'test_loss':
                print(f"{'Size':<12} {'Base Loss':<15} {'WM Loss':<15} {'Δ':<12}")
            else:
                print(f"{'Size':<12} {'Base Acc':<15} {'WM Acc':<15} {'Δ':<12}")
            print("-" * 60)
            
            # Get baseline and WM results by size
            base_results = {size: val for size, val in grouped.get((domain, 'Base'), [])}
            wm_results = {size: val for size, val in grouped.get((domain, 'WM'), [])}
            
            for size in ['tiny', 'small', 'medium', 'large']:
                if size in base_results and size in wm_results:
                    base_val = base_results[size]
                    wm_val = wm_results[size]
                    
                    if has_solve_rate or metric_key == 'test_accuracy':
                        # For solve rate/accuracy, higher is better
                        delta = wm_val - base_val
                        print(f"{size:<12} {base_val:<15.1%} {wm_val:<15.1%} {delta:+.1%}")
                    else:
                        # For loss, lower is better, so delta is base - wm
                        delta = base_val - wm_val
                        print(f"{size:<12} {base_val:<15.4f} {wm_val:<15.4f} {delta:+.4f}")
                        if delta > 0:
                            improvement = delta / base_val * 100
                            print(f"             (WM is {improvement:.1f}% better)")
    
    def compare_weight_sharing(self, results: List[Dict]):
        """Compare standard vs weight-shared models."""
        print("\n" + "=" * 80)
        print("WEIGHT SHARING COMPARISON")
        print("=" * 80)
        
        # Check what metric we have (prioritize solve_rate)
        has_solve_rate = 'solve_rate' in results[0] if results else False
        has_test_loss = 'test_loss' in results[0] if results else False
        
        if has_solve_rate:
            metric_key = 'solve_rate'
            metric_name = "Solve Rate"
        elif has_test_loss:
            metric_key = 'test_loss'
            metric_name = "Test Loss"
        else:
            metric_key = 'test_accuracy'
            metric_name = "Accuracy"
        
        # Group by domain, model type, and sharing
        grouped = {}
        for r in results:
            domain = r['domain']
            model = r['model']
            
            model_type = 'WM' if 'WM' in model else 'Base'
            sharing = 'Shared' if 'Shared' in model else 'Standard'
            
            key = (domain, model_type, sharing)
            grouped[key] = r[metric_key]
        
        # Print comparison
        for domain in set(r['domain'] for r in results):
            print(f"\n{domain.upper()}")
            print("-" * 70)
            
            if has_solve_rate:
                print(f"{'Model':<20} {'Standard':<15} {'Shared':<15} {'Δ':<15}")
            elif metric_key == 'test_loss':
                print(f"{'Model':<20} {'Standard':<15} {'Shared':<15} {'Δ':<15}")
            else:
                print(f"{'Model':<20} {'Standard':<15} {'Shared':<15} {'Δ':<15}")
            print("-" * 70)
            
            for model_type in ['Base', 'WM']:
                std = grouped.get((domain, model_type, 'Standard'), None)
                shared = grouped.get((domain, model_type, 'Shared'), None)
                
                if std is not None and shared is not None:
                    if has_solve_rate or metric_key == 'test_accuracy':
                        # For solve rate/accuracy, higher is better
                        delta = shared - std
                        print(f"{model_type:<20} {std:<15.1%} {shared:<15.1%} {delta:+.1%}")
                    else:
                        # For loss, lower is better
                        delta = std - shared
                        print(f"{model_type:<20} {std:<15.4f} {shared:<15.4f} {delta:+.4f}")
                        if delta > 0:
                            improvement = abs(delta) / std * 100
                            print(f"{'':20} (Shared is {improvement:.1f}% better)")
    
    def compare_experiments(self, exp_names: List[str]):
        """Compare multiple named experiments."""
        print("\n" + "=" * 80)
        print(f"COMPARING {len(exp_names)} EXPERIMENTS")
        print("=" * 80)
        
        results = []
        for name in exp_names:
            try:
                result = self.load_experiment(name)
                results.append(result)
            except FileNotFoundError:
                print(f"Warning: Could not load {name}")
        
        if not results:
            print("No valid experiments found")
            return
        
        # Check what metric we have (prioritize solve_rate)
        has_solve_rate = 'solve_rate' in results[0] if results else False
        has_test_loss = 'test_loss' in results[0] if results else False
        
        if has_solve_rate:
            metric_key = 'solve_rate'
            metric_name = "Solve Rate"
        elif has_test_loss:
            metric_key = 'test_loss'
            metric_name = "Test Loss"
        else:
            metric_key = 'test_accuracy'
            metric_name = "Accuracy"
        
        # Print comparison table
        print(f"\n{'Experiment':<40} {'Domain':<15} {'Model':<15} {metric_name:<15}")
        print("-" * 85)
        
        # Sort (descending for solve rate/accuracy, ascending for loss)
        if has_solve_rate or metric_key == 'test_accuracy':
            sorted_results = sorted(results, key=lambda x: x[metric_key], reverse=True)
        else:
            sorted_results = sorted(results, key=lambda x: x[metric_key])
        
        for r in sorted_results:
            name = r['experiment_name']
            if len(name) > 39:
                name = name[:36] + "..."
            
            if has_solve_rate or metric_key == 'test_accuracy':
                print(f"{name:<40} {r['domain']:<15} {r['model']:<15} {r[metric_key]:<15.1%}")
            else:
                print(f"{name:<40} {r['domain']:<15} {r['model']:<15} {r[metric_key]:<15.4f}")
    
    def _estimate_params(self, size: str) -> str:
        """Estimate parameter count from size name."""
        estimates = {
            'tiny': '60K',
            'small': '240K',
            'medium': '900K',
            'large': '2M'
        }
        return estimates.get(size, 'Unknown')
    
    def print_best_configs(self, results: List[Dict], top_k: int = 5):
        """Print top-k best configurations."""
        print("\n" + "=" * 80)
        print(f"TOP {top_k} CONFIGURATIONS")
        print("=" * 80)
        
        # Check what metric we have (prioritize solve_rate)
        has_solve_rate = 'solve_rate' in results[0] if results else False
        has_test_loss = 'test_loss' in results[0] if results else False
        
        if has_solve_rate:
            metric_key = 'solve_rate'
            metric_header = "Solve Rate"
        elif has_test_loss:
            metric_key = 'test_loss'
            metric_header = "Test Loss"
        else:
            metric_key = 'test_accuracy'
            metric_header = "Accuracy"
        
        # Sort (descending for solve rate/accuracy, ascending for loss)
        if has_solve_rate or metric_key == 'test_accuracy':
            sorted_results = sorted(results, key=lambda x: x[metric_key], reverse=True)
        else:
            sorted_results = sorted(results, key=lambda x: x[metric_key])
        
        print(f"\n{'Rank':<6} {'Experiment':<35} {metric_header:<15} {'Config':<25}")
        print("-" * 85)
        
        for i, r in enumerate(sorted_results[:top_k], 1):
            name = r['experiment_name']
            if len(name) > 34:
                name = name[:31] + "..."
            
            config = r.get('config', r)  # Handle both nested and flat config
            if 'model' in config:
                model_config = config['model']
            else:
                model_config = config
            
            config_str = f"d={model_config.get('d_model', 'N/A')}, l={model_config.get('n_layers', 'N/A')}"
            if model_config.get('weight_sharing'):
                config_str += ", WS"
            
            if has_solve_rate or metric_key == 'test_accuracy':
                print(f"{i:<6} {name:<35} {r[metric_key]:<15.1%} {config_str:<25}")
            else:
                print(f"{i:<6} {name:<35} {r[metric_key]:<15.4f} {config_str:<25}")
    
    def analyze_split_comparison(self, results: List[Dict]):
        """Compare performance across different splits."""
        print("\n" + "=" * 80)
        print("SPLIT TYPE COMPARISON")
        print("=" * 80)
        
        # Check what metric we have (prioritize solve_rate)
        has_solve_rate = 'solve_rate' in results[0] if results else False
        has_test_loss = 'test_loss' in results[0] if results else False
        
        if has_solve_rate:
            metric_key = 'solve_rate'
        elif has_test_loss:
            metric_key = 'test_loss'
        else:
            metric_key = 'test_accuracy'
        
        # Group by domain, model, and split type
        grouped = {}
        for r in results:
            domain = r['domain']
            model = r['model']
            split = r.get('config', {}).get('data', {}).get('split_type', 
                          r.get('split_type', 'in_distribution'))
            
            key = (domain, model)
            if key not in grouped:
                grouped[key] = {}
            grouped[key][split] = r[metric_key]
        
        # Print comparison
        for (domain, model), splits in sorted(grouped.items()):
            print(f"\n{domain} - {model}")
            print("-" * 60)
            
            for split_type, value in sorted(splits.items()):
                if has_solve_rate or metric_key == 'test_accuracy':
                    print(f"  {split_type:<30} {value:.1%}")
                else:
                    print(f"  {split_type:<30} {value:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Analyze experiment results")
    
    parser.add_argument(
        '--sweep-file',
        type=str,
        help='Path to sweep summary JSON file'
    )
    
    parser.add_argument(
        '--compare',
        nargs='+',
        help='Compare specific experiments by name'
    )
    
    parser.add_argument(
        '--results-dir',
        type=str,
        default='./results',
        help='Results directory'
    )
    
    parser.add_argument(
        '--analysis',
        type=str,
        choices=['model_size', 'weight_sharing', 'splits', 'best', 'all'],
        default='all',
        help='Type of analysis to perform'
    )
    
    args = parser.parse_args()
    
    analyzer = ResultsAnalyzer(results_dir=args.results_dir)
    
    if args.sweep_file:
        # Analyze sweep results
        results = analyzer.load_sweep(args.sweep_file)
        
        if args.analysis in ['model_size', 'all']:
            analyzer.compare_model_sizes(results)
        
        if args.analysis in ['weight_sharing', 'all']:
            analyzer.compare_weight_sharing(results)
        
        if args.analysis in ['splits', 'all']:
            analyzer.analyze_split_comparison(results)
        
        if args.analysis in ['best', 'all']:
            analyzer.print_best_configs(results)
    
    elif args.compare:
        # Compare specific experiments
        analyzer.compare_experiments(args.compare)
    
    else:
        print("Please specify --sweep-file or --compare")
        print("\nExamples:")
        print("  python analyze_results.py --sweep-file results/sweep_summary.json")
        print("  python analyze_results.py --compare exp1 exp2 exp3")


if __name__ == "__main__":
    main()
