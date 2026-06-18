"""
Create publication-quality plots from experiment results.

Usage:
    python3 plot_results.py --sweep-file results/sweep_summary.json
"""

import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import defaultdict


def load_results(sweep_file):
    """Load results from JSON."""
    with open(sweep_file) as f:
        return json.load(f)


def plot_model_size_comparison(results, save_dir):
    """Plot test loss or solve rate vs model size for baseline and WM."""
    
    # Check what metric we have
    has_solve_rate = 'solve_rate' in results[0] if results else False
    metric_key = 'solve_rate' if has_solve_rate else 'test_loss'
    metric_name = "Solve Rate" if has_solve_rate else "Test Loss"
    
    # Group by model type and size
    data = defaultdict(lambda: defaultdict(list))
    
    for r in results:
        model = r['model']
        
        # Extract size and type
        parts = model.split('_')
        size = parts[0]  # tiny, small, medium
        model_type = 'World Model' if 'WM' in model else 'Baseline'
        
        # Skip shared weight models for clarity
        if 'Shared' in model:
            continue
        
        value = r.get(metric_key, r.get('test_loss', 0))
        data[model_type][size].append(value)
    
    # Average if multiple runs
    for model_type in data:
        for size in data[model_type]:
            data[model_type][size] = np.mean(data[model_type][size])
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sizes = ['tiny', 'small', 'medium','large']
    size_labels = ['Tiny\n(60K)', 'Small\n(240K)', 'Medium\n(900K)','Large\n(1400K)']
    x = np.arange(len(sizes))
    width = 0.35
    
    baseline_vals = [data['Baseline'].get(s, 0) for s in sizes]
    wm_vals = [data['World Model'].get(s, 0) for s in sizes]
    
    ax.bar(x - width/2, baseline_vals, width, label='Baseline', 
           color='#E74C3C', alpha=0.8)
    ax.bar(x + width/2, wm_vals, width, label='World Model',
           color='#3498DB', alpha=0.8)
    
    ax.set_xlabel('Model Size (Parameters)', fontsize=12, fontweight='bold')
    ax.set_ylabel(metric_name, fontsize=12, fontweight='bold')
    
    if has_solve_rate:
        ax.set_title('Model Size Ablation: World Model Doubles Solve Rate', 
                     fontsize=14, fontweight='bold', pad=20)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
        
        # Add improvement percentages
        for i, (baseline, wm) in enumerate(zip(baseline_vals, wm_vals)):
            if baseline > 0 and wm > 0:
                improvement = (wm - baseline) / baseline * 100
                ax.text(i, max(baseline, wm) + 0.02, f'+{improvement:.0f}%',
                       ha='center', fontsize=9, fontweight='bold', color='green')
    else:
        ax.set_title('Model Size Ablation: World Model Benefit is Consistent', 
                     fontsize=14, fontweight='bold', pad=20)
        
        # Add improvement percentages
        for i, (baseline, wm) in enumerate(zip(baseline_vals, wm_vals)):
            if baseline > 0 and wm > 0:
                improvement = (baseline - wm) / baseline * 100
                ax.text(i, max(baseline, wm) + 0.01, f'{improvement:.1f}%',
                       ha='center', fontsize=9, fontweight='bold', color='green')
    
    ax.set_xticks(x)
    ax.set_xticklabels(size_labels)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'model_size_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(save_dir / 'model_size_comparison.pdf', bbox_inches='tight')
    print(f"✓ Saved: {save_dir / 'model_size_comparison.png'}")
    
    return fig


def plot_weight_sharing_comparison(results, save_dir):
    """Plot effect of weight sharing on baseline vs WM."""
    
    # Check what metric we have
    has_solve_rate = 'solve_rate' in results[0] if results else False
    metric_key = 'solve_rate' if has_solve_rate else 'test_loss'
    metric_name = "Solve Rate" if has_solve_rate else "Test Loss"
    
    # Group by model type and sharing
    data = defaultdict(lambda: defaultdict(list))
    
    for r in results:
        model = r['model']
        
        # Only use 'small' models for clarity
        if 'small' not in model.lower():
            continue
        
        model_type = 'World Model' if 'WM' in model else 'Baseline'
        sharing = 'Shared' if 'Shared' in model else 'Standard'
        
        value = r.get(metric_key, r.get('test_loss', 0))
        data[model_type][sharing].append(value)
    
    # Average if multiple runs
    for model_type in data:
        for sharing in data[model_type]:
            data[model_type][sharing] = np.mean(data[model_type][sharing])
    
    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    
    x = np.arange(2)
    width = 0.35
    
    baseline_vals = [data['Baseline'].get('Standard', 0), 
                     data['Baseline'].get('Shared', 0)]
    wm_vals = [data['World Model'].get('Standard', 0),
               data['World Model'].get('Shared', 0)]
    
    ax.bar(x - width/2, baseline_vals, width, label='Baseline',
           color='#E74C3C', alpha=0.8)
    ax.bar(x + width/2, wm_vals, width, label='World Model',
           color='#3498DB', alpha=0.8)
    
    ax.set_xlabel('Architecture', fontsize=12, fontweight='bold')
    ax.set_ylabel(metric_name, fontsize=12, fontweight='bold')
    ax.set_title('Weight Sharing: Benefits Both Baseline and World Model',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(['Standard', 'Weight Shared'])
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    if has_solve_rate:
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    
    plt.tight_layout()
    plt.savefig(save_dir / 'weight_sharing_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(save_dir / 'weight_sharing_comparison.pdf', bbox_inches='tight')
    print(f"✓ Saved: {save_dir / 'weight_sharing_comparison.png'}")
    
    return fig


def plot_training_curves(results, save_dir):
    """Plot training curves if history files are available."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Find a baseline and WM example
    baseline_exp = None
    wm_exp = None
    
    for r in results:
        if 'small' in r['model'].lower():
            if 'WM' in r['model'] and wm_exp is None:
                wm_exp = r['experiment_name']
            elif 'WM' not in r['model'] and baseline_exp is None:
                baseline_exp = r['experiment_name']
    
    # Try to load training histories
    for exp_name, label, color, ax in [
        (baseline_exp, 'Baseline', '#E74C3C', ax1),
        (wm_exp, 'World Model', '#3498DB', ax2)
    ]:
        if exp_name:
            history_file = Path('results') / exp_name / 'training_history.json'
            if history_file.exists():
                with open(history_file) as f:
                    history = json.load(f)
                
                losses = history.get('train_losses', [])
                if losses:
                    ax.plot(losses, color=color, linewidth=2)
                    ax.set_xlabel('Epoch', fontsize=11)
                    ax.set_ylabel('Training Loss', fontsize=11)
                    ax.set_title(f'{label} Training Curve', fontsize=12, fontweight='bold')
                    ax.grid(alpha=0.3)
                    
                    # Add final loss annotation
                    final_loss = losses[-1]
                    ax.text(len(losses)-1, final_loss, f'{final_loss:.3f}',
                           ha='right', va='bottom', fontsize=9,
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_dir / 'training_curves.png'}")
    
    return fig


def create_summary_table(results, save_dir):
    """Create a LaTeX table summarizing results."""
    
    # Check what metrics we have
    has_solve_rate = 'solve_rate' in results[0] if results else False
    
    # Group by model type and size
    table_data = defaultdict(lambda: defaultdict(dict))
    
    for r in results:
        model = r['model']
        parts = model.split('_')
        size = parts[0]
        model_type = 'WM' if 'WM' in model else 'Base'
        sharing = 'Shared' if 'Shared' in model else 'Std'
        
        test_loss = r.get('test_loss', r.get('test_loss_final', 0))
        solve_rate = r.get('solve_rate', 0)
        converged = r.get('converged', False)
        
        key = f"{model_type}_{sharing}"
        table_data[size][key] = {
            'loss': test_loss,
            'solve_rate': solve_rate,
            'converged': converged
        }
    
    # Generate LaTeX
    latex = []
    latex.append("\\begin{table}[t]")
    latex.append("\\centering")
    
    if has_solve_rate:
        latex.append("\\caption{Test Loss and Solve Rate for Different Configurations}")
        latex.append("\\begin{tabular}{l|cc|cc|cc|cc}")
        latex.append("\\hline")
        latex.append("& \\multicolumn{2}{c|}{Base (Std)} & \\multicolumn{2}{c|}{Base (Shr)} & "
                    "\\multicolumn{2}{c|}{WM (Std)} & \\multicolumn{2}{c}{WM (Shr)} \\\\")
        latex.append("Size & Loss & Solve & Loss & Solve & Loss & Solve & Loss & Solve \\\\")
    else:
        latex.append("\\caption{Test Loss for Different Configurations}")
        latex.append("\\begin{tabular}{l|cc|cc}")
        latex.append("\\hline")
        latex.append("Size & Base (Std) & Base (Shared) & WM (Std) & WM (Shared) \\\\")
    
    latex.append("\\hline")
    
    for size in ['tiny', 'small', 'medium']:
        if size in table_data:
            row = f"{size.capitalize()}"
            
            if has_solve_rate:
                for key in ['Base_Std', 'Base_Shared', 'WM_Std', 'WM_Shared']:
                    if key in table_data[size]:
                        loss = table_data[size][key]['loss']
                        solve = table_data[size][key]['solve_rate']
                        row += f" & {loss:.4f} & {solve:.1%}"
                    else:
                        row += " & -- & --"
            else:
                for key in ['Base_Std', 'Base_Shared', 'WM_Std', 'WM_Shared']:
                    if key in table_data[size]:
                        loss = table_data[size][key]['loss']
                        row += f" & {loss:.4f}"
                    else:
                        row += " & --"
            
            row += " \\\\"
            latex.append(row)
    
    latex.append("\\hline")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")
    
    # Save
    table_file = save_dir / 'results_table.tex'
    with open(table_file, 'w') as f:
        f.write('\n'.join(latex))
    
    print(f"✓ Saved: {table_file}")
    
    return '\n'.join(latex)


def main():
    parser = argparse.ArgumentParser(description="Plot experiment results")
    parser.add_argument('--sweep-file', type=str, default='results/sweep_summary.json',
                       help='Path to sweep summary JSON')
    parser.add_argument('--output-dir', type=str, default='results/plots',
                       help='Directory to save plots')
    
    args = parser.parse_args()
    
    # Load results
    print(f"Loading results from {args.sweep_file}...")
    results = load_results(args.sweep_file)
    print(f"✓ Loaded {len(results)} experiments")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate plots
    print("\nGenerating plots...")
    plot_model_size_comparison(results, output_dir)
    plot_weight_sharing_comparison(results, output_dir)
    plot_training_curves(results, output_dir)
    
    # Generate table
    print("\nGenerating LaTeX table...")
    create_summary_table(results, output_dir)
    
    print("\n" + "="*70)
    print("PLOTS COMPLETE")
    print("="*70)
    print(f"All plots saved to: {output_dir}")
    print("\nFiles created:")
    print("  - model_size_comparison.png (and .pdf)")
    print("  - weight_sharing_comparison.png (and .pdf)")
    print("  - training_curves.png")
    print("  - results_table.tex")
    print("\nUse these in your paper!")


if __name__ == "__main__":
    main()
