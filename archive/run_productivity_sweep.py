"""
Run productivity (length generalization) experiments across both domains.

Usage:
    python3 run_productivity_sweep.py --domain blocks_world
    python3 run_productivity_sweep.py --domain eight_puzzle
    python3 run_productivity_sweep.py --domain both
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import ExperimentConfig, ModelPresets, DataPresets, SplitType
from run_experiments import ExperimentRunner


def create_productivity_sweep(domain, sizes=None):
    """Create productivity experiments for a domain.

    Args:
        domain: 'blocks_world' or 'eight_puzzle'
        sizes: optional list of size names to include (e.g. ['medium']).
               If None, includes all of small/medium/large.
    """
    
    experiments = []
    
    # Get appropriate data config
    if domain == "blocks_world":
        data_config = DataPresets.blocks_world_productivity()
    elif domain == "eight_puzzle":
        data_config = DataPresets.eight_puzzle_productivity()
    else:
        raise ValueError(f"Unknown domain: {domain}")
    
    # Test 3 model sizes × 2 model types (baseline + WM)
    all_sizes = [
        ("small", ModelPresets.small),
        ("medium", ModelPresets.medium),
        ("large", ModelPresets.large),
    ]
    if sizes is not None:
        size_set = set(sizes)
        all_sizes = [(n, fn) for (n, fn) in all_sizes if n in size_set]
        if not all_sizes:
            raise ValueError(
                f"No matching sizes after filtering. Requested: {sorted(size_set)}. "
                f"Available: small, medium, large."
            )
    for size_name, size_fn in all_sizes:
        for use_wm in [False, True]:
            model_type = "wm" if use_wm else "base"
            
            config = ExperimentConfig(
                model=size_fn(use_world_model=use_wm),
                data=data_config,
                experiment_name=f"{domain}_{size_name}_{model_type}_productivity"
            )
            
            experiments.append(config)
    
    return experiments


def main():
    parser = argparse.ArgumentParser(
        description='Run productivity (length generalization) experiments'
    )
    parser.add_argument(
        '--domain',
        choices=['blocks_world', 'eight_puzzle', 'both'],
        default='both',
        help='Which domain(s) to run'
    )
    parser.add_argument(
        '--results-dir',
        default='results',
        help='Results directory'
    )
    parser.add_argument(
        '--sizes',
        nargs='+',
        choices=['small', 'medium', 'large'],
        default=None,
        help='Restrict to specific model sizes (default: all three).'
    )
    parser.add_argument(
        '--force-regenerate',
        action='store_true',
        help='Ignore any cached data and regenerate fresh datasets.',
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("PRODUCTIVITY (LENGTH GENERALIZATION) EXPERIMENTS")
    print("=" * 80)
    if args.sizes:
        print(f"Filtering to sizes: {', '.join(args.sizes)}")
    
    # Create experiments
    all_experiments = []
    
    if args.domain in ['blocks_world', 'both']:
        print("\nBlocks World Productivity:")
        print("  Train: 1-4 moves")
        print("  Test:  5-8 moves")
        bw_experiments = create_productivity_sweep('blocks_world', sizes=args.sizes)
        all_experiments.extend(bw_experiments)
        print(f"  Experiments: {len(bw_experiments)}")
    
    if args.domain in ['eight_puzzle', 'both']:
        print("\n8-Puzzle Productivity:")
        print("  Train: 10-12 moves")
        print("  Test:  13-18 moves")
        ep_experiments = create_productivity_sweep('eight_puzzle', sizes=args.sizes)
        all_experiments.extend(ep_experiments)
        print(f"  Experiments: {len(ep_experiments)}")
    
    print(f"\nTotal experiments: {len(all_experiments)}")
    
    # Run experiments
    runner = ExperimentRunner(results_dir=args.results_dir, force_regenerate=args.force_regenerate)
    runner.run_sweep(all_experiments)
    
    print("\n" + "=" * 80)
    print("PRODUCTIVITY EXPERIMENTS COMPLETE")
    print("=" * 80)
    
    print("\nResults saved to:")
    print(f"  {args.results_dir}/sweep_summary.json")
    
    print("\nAnalyze results:")
    print(f"  python3 analyze_results.py --sweep-file {args.results_dir}/sweep_summary.json")


if __name__ == "__main__":
    main()
