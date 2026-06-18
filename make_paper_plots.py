"""
make_paper_plots.py

Read the JSON files produced by make_paper_results.py and produce the
paper's figures and tables.

Outputs in results/paper/:
  - fig_headline.png/.pdf  -- 2x3 bar chart, solve rates across domains
                              and splits
  - fig_state_validity.png/.pdf -- mechanism figure: fraction of valid
                                    state-tokens emitted by step, both
                                    domains, WM with state_source='model'
  - fig_plan_length.png/.pdf -- diagnostic: mean generated actions by
                                 reference length, by model and domain
  - results_table.tex      -- LaTeX table mirroring fig_headline

Run from the experiments/ directory:
    python3 make_paper_plots.py
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# Color scheme: distinct, colorblind-friendly. Reds for baseline, blues
# for WM-oracle, greens for WM-model. The exact hex values follow common
# matplotlib qualitative palettes.
COLOR_BASELINE  = "#d95f02"   # orange
COLOR_WM_ORACLE = "#7570b3"   # purple
COLOR_WM_MODEL  = "#1b9e77"   # teal-green


def load_data(out_dir):
    with open(out_dir / "paper_results.json") as f:
        results = json.load(f)
    with open(out_dir / "paper_diagnostics.json") as f:
        diagnostics = json.load(f)
    return results, diagnostics


# ============================================================
# Figure 1: headline bar chart
# ============================================================

DOMAIN_LABELS = {
    "blocks_world": "Blocks World",
    "eight_puzzle": "8-Puzzle",
}

SPLIT_LABELS = {
    "in_distribution": "In-distribution",
    "productivity":    "Productivity\n(unfiltered)",
    "truly_ood":       "Productivity\n(truly OOD)",
}


def make_headline_figure(results, out_dir):
    """A 2x3 grid: rows = domains, columns = splits. Each cell: three bars."""
    domains = ["blocks_world", "eight_puzzle"]
    splits = ["in_distribution", "productivity", "truly_ood"]
    conditions = [("baseline", "Baseline", COLOR_BASELINE),
                  ("wm_oracle", "WM (oracle)", COLOR_WM_ORACLE),
                  ("wm_model",  "WM (model)",  COLOR_WM_MODEL)]

    fig, axes = plt.subplots(len(domains), len(splits),
                              figsize=(11, 6), sharey=True)

    for i, domain in enumerate(domains):
        for j, split in enumerate(splits):
            ax = axes[i, j]
            cell = results.get(domain, {}).get(split, {})
            n = cell.get("n", 0)

            vals = []
            labels = []
            colors = []
            for key, label, color in conditions:
                v = cell.get(key)
                vals.append(v if v is not None else 0.0)
                labels.append(label)
                colors.append(color)

            x = np.arange(len(conditions))
            bars = ax.bar(x, vals, color=colors, alpha=0.85, edgecolor='black', linewidth=0.5)

            # Annotate bar tops with values
            for bar, v in zip(bars, vals):
                if v > 0:
                    label_text = f"{v:.2f}"
                else:
                    label_text = "0"
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.02,
                        label_text,
                        ha='center', va='bottom', fontsize=9)

            ax.set_xticks(x)
            if i == len(domains) - 1:
                ax.set_xticklabels([c[1] for c in conditions], rotation=20, ha='right', fontsize=9)
            else:
                ax.set_xticklabels([])

            ax.set_ylim(0, 1.10)
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            ax.set_axisbelow(True)

            if j == 0:
                ax.set_ylabel(f"{DOMAIN_LABELS[domain]}\nsolve rate",
                              fontsize=11, fontweight='bold')
            if i == 0:
                ax.set_title(f"{SPLIT_LABELS[split]}\n(n={n})", fontsize=10)
            else:
                ax.set_xlabel(f"(n={n})", fontsize=9)

    fig.suptitle("Solve rate across domains and splits, under corrected check",
                 fontsize=13, fontweight='bold', y=1.00)
    plt.tight_layout()

    out_png = out_dir / "fig_headline.png"
    out_pdf = out_dir / "fig_headline.pdf"
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.savefig(out_pdf, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ {out_png}")
    print(f"✓ {out_pdf}")


# ============================================================
# Figure 2: state validity by step (mechanism)
# ============================================================

def make_state_validity_figure(diagnostics, out_dir):
    """Line plot: fraction of valid state-blocks emitted by step,
    one line per domain. WM productivity checkpoints, state_source='model'.
    """
    bw_data = diagnostics["state_validity_by_step"].get("blocks_world", [])
    ep_data = diagnostics["state_validity_by_step"].get("eight_puzzle", [])

    fig, ax = plt.subplots(figsize=(8, 5))

    for data, label, color in [
        (bw_data, "Blocks World", "#1f77b4"),
        (ep_data, "8-Puzzle",     "#d62728"),
    ]:
        if not data:
            continue
        steps = [d["step"] for d in data]
        frac = [d["valid"] / d["total"] if d["total"] > 0 else 0.0 for d in data]
        # Also annotate sample size at first step (typically the largest)
        ax.plot(steps, frac, marker='o', linewidth=2, label=label, color=color)

    ax.set_xlabel("Step in generated trajectory (state-block index)", fontsize=11, fontweight='bold')
    ax.set_ylabel("Fraction of state-blocks that are valid", fontsize=11, fontweight='bold')
    ax.set_title("State-prediction validity collapse, by domain\n"
                 "(WM productivity checkpoints, state_source='model')",
                 fontsize=12, fontweight='bold')
    ax.set_ylim(-0.05, 1.05)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.grid(alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    ax.legend(fontsize=11, loc='best')

    plt.tight_layout()
    out_png = out_dir / "fig_state_validity.png"
    out_pdf = out_dir / "fig_state_validity.pdf"
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.savefig(out_pdf, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ {out_png}")
    print(f"✓ {out_pdf}")


# ============================================================
# Figure 3: plan length by reference length
# ============================================================

def make_plan_length_figure(diagnostics, out_dir):
    """Bar chart: mean generated action count vs reference-num-moves,
    one panel per domain, two bar colors per ref (baseline vs WM-oracle).
    """
    pl = diagnostics["plan_length"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for ax_idx, (domain, label) in enumerate([
        ("blocks_world", "Blocks World"),
        ("eight_puzzle", "8-Puzzle"),
    ]):
        ax = axes[ax_idx]
        base = pl.get(f"{domain}_baseline", [])
        wm = pl.get(f"{domain}_wm_oracle", [])

        # Aggregate by ref length
        by_ref_base = defaultdict(list)
        for r in base:
            by_ref_base[r["ref_num_moves"]].append(r["n_actions"])
        by_ref_wm = defaultdict(list)
        for r in wm:
            by_ref_wm[r["ref_num_moves"]].append(r["n_actions"])

        refs = sorted(set(by_ref_base.keys()) | set(by_ref_wm.keys()))
        if not refs:
            ax.set_title(f"{label} (no data)")
            continue

        x = np.arange(len(refs))
        width = 0.35

        base_means = [np.mean(by_ref_base[r]) if by_ref_base[r] else 0 for r in refs]
        wm_means = [np.mean(by_ref_wm[r]) if by_ref_wm[r] else 0 for r in refs]

        ax.bar(x - width/2, base_means, width, label="Baseline",
               color=COLOR_BASELINE, alpha=0.85, edgecolor='black', linewidth=0.5)
        ax.bar(x + width/2, wm_means, width, label="WM (oracle)",
               color=COLOR_WM_ORACLE, alpha=0.85, edgecolor='black', linewidth=0.5)

        # Reference: y=x line
        ax.plot(x, refs, color='black', linestyle=':', linewidth=1.5,
                label="Reference\nlength", alpha=0.7)

        ax.set_xticks(x)
        ax.set_xticklabels([str(r) for r in refs])
        ax.set_xlabel("Reference plan length", fontsize=11, fontweight='bold')
        if ax_idx == 0:
            ax.set_ylabel("Mean generated plan length", fontsize=11, fontweight='bold')
        ax.set_title(label, fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        ax.legend(fontsize=9, loc='upper left')

    fig.suptitle("Length-locking: models produce training-distribution length plans\n"
                 "regardless of problem difficulty",
                 fontsize=12, y=1.02)
    plt.tight_layout()

    out_png = out_dir / "fig_plan_length.png"
    out_pdf = out_dir / "fig_plan_length.pdf"
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.savefig(out_pdf, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ {out_png}")
    print(f"✓ {out_pdf}")


# ============================================================
# LaTeX table
# ============================================================

def make_latex_table(results, out_dir):
    """Mirrors the data in fig_headline.png as a publishable LaTeX table."""
    rows = []
    rows.append("% Auto-generated by make_paper_plots.py")
    rows.append("\\begin{table}[t]")
    rows.append("\\centering")
    rows.append("\\caption{Solve rates by domain, split, and model type. "
                "WM (oracle) injects ground-truth states between actions during "
                "inference; WM (model) lets the model emit state tokens "
                "autoregressively. Truly-OOD subsets are filtered to problems "
                "whose BFS shortest path exceeds the training distribution.}")
    rows.append("\\label{tab:headline}")
    rows.append("\\begin{tabular}{llcccc}")
    rows.append("\\toprule")
    rows.append("Domain & Split & $n$ & Baseline & WM (oracle) & WM (model) \\\\")
    rows.append("\\midrule")

    for domain, dlabel in [("blocks_world", "Blocks World"),
                           ("eight_puzzle", "8-Puzzle")]:
        for split, slabel in [("in_distribution", "in-distribution"),
                              ("productivity",    "productivity (unfiltered)"),
                              ("truly_ood",       "productivity (truly OOD)")]:
            cell = results.get(domain, {}).get(split, {})
            n = cell.get("n", 0)
            def fmt(v):
                if v is None:
                    return "--"
                return f"{v:.3f}"
            row = (
                f"{dlabel} & {slabel} & {n} & "
                f"{fmt(cell.get('baseline'))} & "
                f"{fmt(cell.get('wm_oracle'))} & "
                f"{fmt(cell.get('wm_model'))} \\\\"
            )
            rows.append(row)
        rows.append("\\midrule" if domain == "blocks_world" else "")

    if rows[-1] == "":
        rows.pop()
    rows.append("\\bottomrule")
    rows.append("\\end{tabular}")
    rows.append("\\end{table}")

    out_tex = out_dir / "results_table.tex"
    with open(out_tex, "w") as f:
        f.write("\n".join(rows) + "\n")
    print(f"✓ {out_tex}")


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--paper-dir', default='results/paper',
                    help='Directory containing paper_results.json and paper_diagnostics.json')
    args = ap.parse_args()

    out_dir = Path(args.paper_dir)
    if not out_dir.exists():
        print(f"ERROR: {out_dir} not found. Run make_paper_results.py first.")
        raise SystemExit(1)
    if not (out_dir / "paper_results.json").exists():
        print(f"ERROR: paper_results.json not found in {out_dir}.")
        raise SystemExit(1)

    results, diagnostics = load_data(out_dir)
    print(f"Loaded {len(results)} domain(s) of results from {out_dir}")

    print("\nGenerating figures and table...")
    make_headline_figure(results, out_dir)
    make_state_validity_figure(diagnostics, out_dir)
    make_plan_length_figure(diagnostics, out_dir)
    make_latex_table(results, out_dir)

    print()
    print("=" * 60)
    print("DONE")
    print("=" * 60)
    print(f"Outputs in: {out_dir}")
    print("  - fig_headline.{png,pdf}        (main result figure)")
    print("  - fig_state_validity.{png,pdf}  (mechanism figure)")
    print("  - fig_plan_length.{png,pdf}     (length-locking diagnostic)")
    print("  - results_table.tex             (publishable table)")


if __name__ == "__main__":
    main()
