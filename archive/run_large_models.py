"""
Run experiments for LARGE models only.

⚠️  DEPRECATED: Use run_experiments.py with --sizes flag instead!

RECOMMENDED USAGE:
    python3 run_experiments.py --sweep model_size --domain eight_puzzle --sizes large

This new approach:
  ✅ Uses same code path as all other experiments (consistency!)
  ✅ Supports any size combination (--sizes small medium, etc.)
  ✅ Clearer intent with explicit --sizes flag
  ✅ No need for separate script

This script (run_large_models.py) is kept for backwards compatibility only.

Old usage (deprecated, but still works):
    python3 run_large_models.py --domain eight_puzzle
    python3 run_large_models.py --domain blocks_world
    python3 run_large_models.py --domain both
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import ExperimentConfig, ModelPresets, DataPresets
from run_experiments import ExperimentRunner


def create_large_model_sweep(domain):
    """Create experiments for large models only."""
    
    experiments = []
    
    # Get appropriate data config
    if domain == "blocks_world":
        data_config = DataPresets.blocks_world_standard()
    elif domain == "eight_puzzle":
        data_config = DataPresets.eight_puzzle_standard()
    else:
        raise ValueError(f"Unknown domain: {domain}")
    
    # Large models: baseline + WM × standard + shared = 4 configs
    for use_wm in [False, True]:
        for weight_sharing in [False, True]:
            model_type = "wm" if use_wm else "base"
            ws_suffix = "shared" if weight_sharing else "std"
            
            config = ExperimentConfig(
                model=ModelPresets.large(
                    use_world_model=use_wm,
                    weight_sharing=weight_sharing
                ),
                data=data_config,
                experiment_name=f"{domain}_large_{model_type}_{ws_suffix}"
            )
            
            experiments.append(config)
    
    return experiments


def main():
    parser = argparse.ArgumentParser(
        description='⚠️  DEPRECATED: Use run_experiments.py --sizes large instead'
    )
    parser.add_argument(
        '--domain',
        choices=['blocks_world', 'eight_puzzle', 'both'],
        default='eight_puzzle',
        help='Which domain(s) to run'
    )
    parser.add_argument(
        '--results-dir',
        default='results',
        help='Results directory (appends to existing sweep_summary.json)'
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("⚠️  DEPRECATED SCRIPT")
    print("=" * 80)
    print("\nThis script is deprecated. Please use the new unified approach:")
    print("\n  RECOMMENDED:")
    if args.domain == 'both':
        print("    python3 run_experiments.py --sweep model_size --domain blocks_world --sizes large")
        print("    python3 run_experiments.py --sweep model_size --domain eight_puzzle --sizes large")
    else:
        print(f"    python3 run_experiments.py --sweep model_size --domain {args.domain} --sizes large")
    print("\n  Benefits:")
    print("    ✅ Same code path as all experiments (consistency)")
    print("    ✅ No separate script to maintain")
    print("    ✅ Clearer intent with --sizes flag")
    print("\n  Continuing with legacy script...")
    print("=" * 80)
    
    # Check if existing results exist
    sweep_file = Path(args.results_dir) / "sweep_summary.json"
    if sweep_file.exists():
        print(f"\n✓ Found existing results: {sweep_file}")
        print("  Large model results will be APPENDED (no overwrites)")
    else:
        print(f"\n✓ No existing results found")
        print("  Will create new sweep_summary.json")
    
    # Create experiments
    all_experiments = []
    
    if args.domain in ['blocks_world', 'both']:
        print("\nBlocks World - Large Models:")
        bw_experiments = create_large_model_sweep('blocks_world')
        for exp in bw_experiments:
            print(f"  - {exp.experiment_name}")
        all_experiments.extend(bw_experiments)
    
    if args.domain in ['eight_puzzle', 'both']:
        print("\n8-Puzzle - Large Models:")
        ep_experiments = create_large_model_sweep('eight_puzzle')
        for exp in ep_experiments:
            print(f"  - {exp.experiment_name}")
        all_experiments.extend(ep_experiments)
    
    print(f"\nTotal experiments: {len(all_experiments)}")
    
    # Estimate time
    minutes_per_exp = 60  # Large models take ~1 hour each
    total_hours = (len(all_experiments) * minutes_per_exp) / 60
    
    print(f"\nEstimated time: ~{total_hours:.1f} hours")
    print("  (Large models take longer due to size)")
    
    # Run experiments
    print("\n" + "=" * 80)
    input("Press Enter to start, or Ctrl+C to cancel...")
    print("=" * 80)
    
    runner = ExperimentRunner(results_dir=args.results_dir)
    runner.run_sweep(all_experiments)
    
    print("\n" + "=" * 80)
    print("LARGE MODEL EXPERIMENTS COMPLETE")
    print("=" * 80)
    
    print(f"\nResults appended to: {sweep_file}")
    
    print("\nAnalyze all results (including previous tiny/small/medium):")
    print(f"  python3 analyze_results.py --sweep-file {sweep_file}")


if __name__ == "__main__":
    main()
