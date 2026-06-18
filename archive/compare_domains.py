"""
Compare results across domains (e.g., Blocks World vs 8-Puzzle).

Usage:
    python3 compare_domains.py results/sweep_summary_blocks.json results/sweep_summary_8puzzle.json
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict


def load_sweep_summary(filepath):
    """Load a sweep summary file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def extract_domain_name(filepath):
    """Extract domain name from filepath or data."""
    # Try to infer from filename
    path = Path(filepath)
    name = path.stem
    
    if '8puzzle' in name.lower() or '8p' in name.lower():
        return 'eight_puzzle'
    elif 'blocks' in name.lower() or 'bw' in name.lower():
        return 'blocks_world'
    else:
        return name


def organize_by_config(experiments):
    """Organize experiments by model configuration."""
    by_config = defaultdict(lambda: {'baseline': None, 'wm': None})
    
    for exp in experiments:
        name = exp['experiment_name']
        
        # Extract size and config
        if 'tiny' in name:
            size = 'tiny'
        elif 'small' in name:
            size = 'small'
        elif 'medium' in name:
            size = 'medium'
        elif 'large' in name:
            size = 'large'
        else:
            continue
        
        # Extract model type and weight sharing
        if '_wm_' in name:
            model_type = 'wm'
        elif '_base_' in name:
            model_type = 'base'
        else:
            continue
        
        if '_shared' in name:
            weight_sharing = 'shared'
        else:
            weight_sharing = 'std'
        
        key = (size, weight_sharing)
        by_config[key][model_type] = exp
    
    return by_config


def compare_domains(domain1_file, domain2_file):
    """Compare results from two domains."""
    
    # Load data
    domain1_data = load_sweep_summary(domain1_file)
    domain2_data = load_sweep_summary(domain2_file)
    
    domain1_name = extract_domain_name(domain1_file)
    domain2_name = extract_domain_name(domain2_file)
    
    print("=" * 80)
    print("CROSS-DOMAIN COMPARISON")
    print("=" * 80)
    print(f"\nDomain 1: {domain1_name} ({domain1_file})")
    print(f"Domain 2: {domain2_name} ({domain2_file})")
    
    # Handle both formats: list of experiments or dict with 'experiments' key
    if isinstance(domain1_data, list):
        d1_experiments = domain1_data
    else:
        d1_experiments = domain1_data.get('experiments', [])
    
    if isinstance(domain2_data, list):
        d2_experiments = domain2_data
    else:
        d2_experiments = domain2_data.get('experiments', [])
    
    # Organize experiments
    d1_configs = organize_by_config(d1_experiments)
    d2_configs = organize_by_config(d2_experiments)
    
    # Find common configurations
    common_configs = set(d1_configs.keys()) & set(d2_configs.keys())
    
    if not common_configs:
        print("\n⚠️  No common configurations found!")
        print("Make sure both domains ran the same model sizes and configurations.")
        return
    
    print(f"\nFound {len(common_configs)} common configurations")
    
    # Compare by model size
    print("\n" + "=" * 80)
    print("WORLD MODEL BENEFIT BY SIZE")
    print("=" * 80)
    
    sizes = ['tiny', 'small', 'medium', 'large']
    
    print(f"\n{'Size':<12} {domain1_name:<20} {domain2_name:<20} {'Gap':<15}")
    print("-" * 80)
    
    for size in sizes:
        # Try standard (non-shared) first
        for weight_sharing in ['std', 'shared']:
            key = (size, weight_sharing)
            
            if key not in common_configs:
                continue
            
            d1_base = d1_configs[key]['base']
            d1_wm = d1_configs[key]['wm']
            d2_base = d2_configs[key]['base']
            d2_wm = d2_configs[key]['wm']
            
            if not all([d1_base, d1_wm, d2_base, d2_wm]):
                continue
            
            # Calculate improvements
            d1_base_sr = d1_base.get('solve_rate', 0) * 100
            d1_wm_sr = d1_wm.get('solve_rate', 0) * 100
            d1_improvement = d1_wm_sr - d1_base_sr
            
            d2_base_sr = d2_base.get('solve_rate', 0) * 100
            d2_wm_sr = d2_wm.get('solve_rate', 0) * 100
            d2_improvement = d2_wm_sr - d2_base_sr
            
            gap = d1_improvement - d2_improvement
            
            ws_label = " (shared)" if weight_sharing == 'shared' else ""
            
            print(f"{size}{ws_label:<12} "
                  f"+{d1_improvement:>5.1f}pp ({d1_base_sr:.0f}→{d1_wm_sr:.0f}%)  "
                  f"+{d2_improvement:>5.1f}pp ({d2_base_sr:.0f}→{d2_wm_sr:.0f}%)  "
                  f"{gap:>+6.1f}pp")
    
    # Overall summary
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    
    # Calculate average improvements
    d1_improvements = []
    d2_improvements = []
    
    for key in common_configs:
        d1_base = d1_configs[key]['base']
        d1_wm = d1_configs[key]['wm']
        d2_base = d2_configs[key]['base']
        d2_wm = d2_configs[key]['wm']
        
        if all([d1_base, d1_wm, d2_base, d2_wm]):
            d1_improvements.append(
                (d1_wm.get('solve_rate', 0) - d1_base.get('solve_rate', 0)) * 100
            )
            d2_improvements.append(
                (d2_wm.get('solve_rate', 0) - d2_base.get('solve_rate', 0)) * 100
            )
    
    if d1_improvements and d2_improvements:
        avg_d1 = sum(d1_improvements) / len(d1_improvements)
        avg_d2 = sum(d2_improvements) / len(d2_improvements)
        
        print(f"\n{domain1_name}:")
        print(f"  Average WM improvement: +{avg_d1:.1f}pp")
        print(f"  Min: +{min(d1_improvements):.1f}pp, Max: +{max(d1_improvements):.1f}pp")
        
        print(f"\n{domain2_name}:")
        print(f"  Average WM improvement: +{avg_d2:.1f}pp")
        print(f"  Min: +{min(d2_improvements):.1f}pp, Max: +{max(d2_improvements):.1f}pp")
        
        print(f"\nCross-domain consistency:")
        diff = abs(avg_d1 - avg_d2)
        if diff < 5:
            print(f"  ✓ Very consistent ({diff:.1f}pp difference)")
        elif diff < 10:
            print(f"  ✓ Reasonably consistent ({diff:.1f}pp difference)")
        else:
            print(f"  ⚠️  Large difference ({diff:.1f}pp)")
    
    # Detailed comparison table
    print("\n" + "=" * 80)
    print("DETAILED COMPARISON")
    print("=" * 80)
    
    print(f"\n{domain1_name.upper()}")
    print("-" * 80)
    print(f"{'Config':<20} {'Base SR':<12} {'WM SR':<12} {'Improvement':<15}")
    print("-" * 80)
    
    for key in sorted(common_configs):
        size, ws = key
        d1_base = d1_configs[key]['base']
        d1_wm = d1_configs[key]['wm']
        
        if d1_base and d1_wm:
            config_name = f"{size}_{ws}"
            base_sr = d1_base.get('solve_rate', 0) * 100
            wm_sr = d1_wm.get('solve_rate', 0) * 100
            improvement = wm_sr - base_sr
            
            print(f"{config_name:<20} {base_sr:>6.1f}%    {wm_sr:>6.1f}%    +{improvement:>5.1f}pp")
    
    print(f"\n{domain2_name.upper()}")
    print("-" * 80)
    print(f"{'Config':<20} {'Base SR':<12} {'WM SR':<12} {'Improvement':<15}")
    print("-" * 80)
    
    for key in sorted(common_configs):
        size, ws = key
        d2_base = d2_configs[key]['base']
        d2_wm = d2_configs[key]['wm']
        
        if d2_base and d2_wm:
            config_name = f"{size}_{ws}"
            base_sr = d2_base.get('solve_rate', 0) * 100
            wm_sr = d2_wm.get('solve_rate', 0) * 100
            improvement = wm_sr - base_sr
            
            print(f"{config_name:<20} {base_sr:>6.1f}%    {wm_sr:>6.1f}%    +{improvement:>5.1f}pp")
    
    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(description='Compare results across domains')
    parser.add_argument('domain1', help='First domain sweep summary file')
    parser.add_argument('domain2', help='Second domain sweep summary file')
    
    args = parser.parse_args()
    
    compare_domains(args.domain1, args.domain2)


if __name__ == "__main__":
    main()
