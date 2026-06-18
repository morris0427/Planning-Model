"""
Main experiment runner.

Usage:
    # Run single experiment
    python run_experiments.py --experiment blocks_world_standard
    
    # Run model size sweep (all sizes)
    python run_experiments.py --sweep model_size --domain blocks_world
    
    # Run ONLY large models from model_size sweep
    python run_experiments.py --sweep model_size --domain blocks_world --sizes large
    
    # Run only small and medium models
    python run_experiments.py --sweep model_size --domain eight_puzzle --sizes small medium
    
    # Run full ablation but only large models
    python run_experiments.py --sweep full --domain blocks_world --sizes large
    
    # Run weight sharing ablation
    python run_experiments.py --sweep weight_sharing --domain eight_puzzle
"""

import argparse
import json
import os
from pathlib import Path
from typing import List
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from config import (
    ExperimentConfig,
    ModelPresets,
    DataPresets,
    SplitType,
    create_model_size_sweep,
    create_weight_sharing_sweep,
    create_full_ablation,
)


class ExperimentRunner:
    """Runs and tracks experiments."""
    
    def __init__(self, results_dir: str = "./results", force_regenerate: bool = False):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.force_regenerate = force_regenerate
        
    def run_experiment(self, config: ExperimentConfig) -> dict:
        """
        Run a single experiment.
        
        Args:
            config: Experiment configuration
            
        Returns:
            Dictionary with results
        """
        print("=" * 70)
        print(f"RUNNING EXPERIMENT: {config.experiment_name}")
        print("=" * 70)
        
        # Create experiment directory
        exp_dir = self.results_dir / config.experiment_name
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Update config to save to experiment directory
        config.save_dir = str(self.results_dir)
        
        # Save configuration
        config_path = exp_dir / "config.json"
        self._save_config(config, config_path)
        
        # Import trainer (done here to avoid import errors if trainer.py doesn't exist yet)
        try:
            from trainer import train
            
            # Run actual training
            results = train(config, force_regenerate=self.force_regenerate)
            
            # Add config to results
            results['config'] = self._config_to_dict(config)
            
        except ImportError as e:
            print(f"\n⚠️  WARNING: Could not import trainer")
            print(f"   {e}")
            print(f"\n   Using placeholder results instead.")
            print(f"   Create trainer.py to run real experiments.")
            
            # Placeholder results if trainer doesn't exist
            results = {
                'experiment_name': config.experiment_name,
                'domain': config.data.domain,
                'model': config.model.name,
                'split_type': config.data.split_type.value,
                'train_loss_final': 0.35,  # Placeholder
                'test_loss': 0.42,   # Placeholder
                'config': self._config_to_dict(config),
                'note': 'PLACEHOLDER - trainer.py not found'
            }
        
        # Save results
        results_path = exp_dir / "results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✓ Results saved to {results_path}")
        
        return results
    
    def run_sweep(self, experiments: List[ExperimentConfig]) -> List[dict]:
        """
        Run multiple experiments.
        
        Args:
            experiments: List of experiment configs
            
        Returns:
            List of result dictionaries
        """
        print(f"\n{'=' * 70}")
        print(f"RUNNING SWEEP: {len(experiments)} experiments")
        print(f"{'=' * 70}\n")
        
        all_results = []
        
        for i, config in enumerate(experiments):
            print(f"\n[{i+1}/{len(experiments)}] {config.experiment_name}")
            results = self.run_experiment(config)
            all_results.append(results)
        
        # Save summary
        summary_path = self.results_dir / "sweep_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print(f"\n{'=' * 70}")
        print(f"SWEEP COMPLETE")
        print(f"{'=' * 70}")
        print(f"Summary saved to: {summary_path}")
        
        # Print comparison table
        self._print_comparison(all_results)
        
        return all_results
    
    def _save_config(self, config: ExperimentConfig, filepath: Path):
        """Save config to JSON."""
        config_dict = self._config_to_dict(config)
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    def _config_to_dict(self, config: ExperimentConfig) -> dict:
        """Convert config to dictionary."""
        return {
            'experiment_name': config.experiment_name,
            'seed': config.seed,
            'model': {
                'name': config.model.name,
                'd_model': config.model.d_model,
                'n_heads': config.model.n_heads,
                'd_ff': config.model.d_ff,
                'n_layers': config.model.n_layers,
                'weight_sharing': config.model.weight_sharing,
                'use_world_model': config.model.use_world_model,
                'learning_rate': config.model.learning_rate,
                'batch_size': config.model.batch_size,
                'max_epochs': config.model.max_epochs,
            },
            'data': {
                'domain': config.data.domain,
                'split_type': config.data.split_type.value,
                'num_train_samples': config.data.num_train_samples,
                'num_test_samples': config.data.num_test_samples,
                'train_difficulty_range': config.data.train_difficulty_range,
                'test_difficulty_range': config.data.test_difficulty_range,
            }
        }
    
    def _print_comparison(self, results: List[dict]):
        """Print comparison table of results."""
        print("\n" + "=" * 70)
        print("RESULTS COMPARISON")
        print("=" * 70)
        
        # Check what metrics we have
        has_solve_rate = 'solve_rate' in results[0] if results else False
        
        # Print header
        if has_solve_rate:
            print(f"{'Experiment':<35} {'Test Loss':<12} {'Solve Rate':<12} {'Converged':<10}")
        else:
            print(f"{'Experiment':<40} {'Test Loss':<12} {'Converged':<12}")
        print("-" * 70)
        
        # Sort by solve rate (descending - higher is better), fallback to test loss
        if has_solve_rate:
            sorted_results = sorted(results, key=lambda x: x.get('solve_rate', 0), reverse=True)
        else:
            sorted_results = sorted(results, key=lambda x: x.get('test_loss', float('inf')))
        
        for result in sorted_results:
            name = result['experiment_name']
            if len(name) > (34 if has_solve_rate else 39):
                name = name[:(31 if has_solve_rate else 36)] + "..."
            
            converged = "✓" if result.get('converged', False) else "✗"
            
            if has_solve_rate:
                solve_rate = result.get('solve_rate', 0)
                print(f"{name:<35} {result['test_loss']:<12.3f} {solve_rate:<12.1%} {converged:<10}")
            else:
                print(f"{name:<40} {result['test_loss']:<12.3f} {converged:<12}")
        
        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Run compositional generalization experiments")
    parser.add_argument(
        "--force-regenerate",
        action="store_true",
        help="Ignore any cached data and regenerate fresh datasets.",
    )
    
    # Experiment selection
    parser.add_argument(
        '--experiment',
        type=str,
        help='Run single named experiment'
    )
    
    parser.add_argument(
        '--sweep',
        type=str,
        choices=['model_size', 'weight_sharing', 'full'],
        help='Run experiment sweep'
    )
    
    parser.add_argument(
        '--domain',
        type=str,
        choices=['blocks_world', 'eight_puzzle'],
        default='blocks_world',
        help='Planning domain'
    )
    
    parser.add_argument(
        '--sizes',
        type=str,
        nargs='+',
        choices=['tiny', 'small', 'medium', 'large'],
        help='Filter experiments to specific model sizes (e.g., --sizes large, or --sizes small medium)'
    )
    
    parser.add_argument(
        '--split',
        type=str,
        choices=['in_distribution', 'productivity', 'systematicity'],
        default='in_distribution',
        help='Train/test split type'
    )
    
    parser.add_argument(
        '--results-dir',
        type=str,
        default='./results',
        help='Directory to save results'
    )
    
    args = parser.parse_args()
    
    # Initialize runner
    runner = ExperimentRunner(results_dir=args.results_dir, force_regenerate=args.force_regenerate)
    
    # Map split string to enum
    split_map = {
        'in_distribution': SplitType.IN_DISTRIBUTION,
        'productivity': SplitType.PRODUCTIVITY,
        'systematicity': SplitType.SYSTEMATICITY,
    }
    split_type = split_map[args.split]
    
    if args.sweep:
        # Run sweep
        if args.sweep == 'model_size':
            experiments = create_model_size_sweep(args.domain, split_type)
        elif args.sweep == 'weight_sharing':
            experiments = create_weight_sharing_sweep(args.domain)
        elif args.sweep == 'full':
            experiments = create_full_ablation(args.domain)
        
        # Filter by model sizes if specified
        if args.sizes:
            print(f"\n📊 Filtering to model sizes: {', '.join(args.sizes)}")
            original_count = len(experiments)
            
            # Filter experiments by checking if model name contains size keyword
            filtered_experiments = []
            for exp in experiments:
                model_name_lower = exp.model.name.lower()
                if any(size in model_name_lower for size in args.sizes):
                    filtered_experiments.append(exp)
            
            experiments = filtered_experiments
            
            print(f"   Filtered: {original_count} → {len(experiments)} experiments")
            
            if len(experiments) == 0:
                print(f"\n⚠️  No experiments match size filter: {args.sizes}")
                print(f"   Available experiment names: {[e.experiment_name for e in experiments[:3]]}")
                return
        
        runner.run_sweep(experiments)
    
    elif args.experiment:
        # Run single experiment
        # Parse experiment name to create config
        # Format: {domain}_{size}_{base|wm}_{shared|std}
        parts = args.experiment.split('_')
        
        # Simple example configs
        if args.experiment == 'blocks_world_standard':
            config = ExperimentConfig(
                model=ModelPresets.small(use_world_model=True),
                data=DataPresets.blocks_world_standard(),
                experiment_name=args.experiment
            )
        elif args.experiment == '8_puzzle_standard':
            config = ExperimentConfig(
                model=ModelPresets.small(use_world_model=False),
                data=DataPresets.puzzle_standard(),
                experiment_name=args.experiment
            )
        else:
            print(f"Unknown experiment: {args.experiment}")
            print("Try: blocks_world_standard or 8_puzzle_standard")
            return
        
        runner.run_experiment(config)
    
    else:
        # Show usage
        print("Please specify either --experiment or --sweep")
        print("\nExamples:")
        print("  python run_experiments.py --experiment blocks_world_standard")
        print("  python run_experiments.py --sweep model_size --domain blocks_world")
        print("  python run_experiments.py --sweep weight_sharing --domain 8_puzzle")
        print("  python run_experiments.py --sweep full --domain blocks_world")


if __name__ == "__main__":
    main()
