"""
make_tf_vs_freegen_artifacts.py

Read tf_vs_freegen.json (produced by measure_tf_vs_freegen.py) and
generate a LaTeX table and a figure suitable for the "domains matter"
section of the paper.

The artifacts highlight the cross-domain asymmetry: only 8-Puzzle WM
shows compound-error decay (FG legal drops sharply with position).
The other three cells (Blocks World baseline, Blocks World WM, 8-Puzzle
baseline) maintain high FG legality regardless of position.

Reads from results/paper/:
  - tf_vs_freegen.json

Writes to results/paper/:
  - table_tf_vs_freegen.tex
  - fig_tf_vs_freegen.{png,pdf}

Run from the experiments/ directory:
    python3 make_tf_vs_freegen_artifacts.py
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def load_data(paper_dir):
    path = paper_dir / "tf_vs_freegen.json"
    if not path.exists():
        print(f"ERROR: {path} not found. Run measure_tf_vs_freegen.py first.")
        raise SystemExit(1)
    with open(path) as f:
        return json.load(f)


# ============================================================
# Figure: FG legal by position, 2x2 panels
# ============================================================

def make_figure(data, out_dir):
    """Plot FG legal by position, one line per (domain, condition).

    The visual story: only 8-Puzzle WM degrades; the others stay high.
    """
    fig, ax = plt.subplots(figsize=(8.5, 5))

    cells = [
        ("blocks_world", "baseline", "Blocks World, Baseline", "#d95f02", "-", "o"),
        ("blocks_world", "wm",       "Blocks World, WM",        "#d95f02", "--", "s"),
        ("eight_puzzle", "baseline", "8-Puzzle, Baseline",      "#1f77b4", "-", "o"),
        ("eight_puzzle", "wm",       "8-Puzzle, WM",            "#1f77b4", "--", "s"),
    ]

    for domain, cond, label, color, ls, marker in cells:
        cell = data.get(domain, {}).get(cond)
        if cell is None:
            continue

        positions = sorted(int(k) for k in cell["fg_legal"].keys())
        # Restrict to positions with reasonable sample size to avoid noise
        positions = [p for p in positions if cell["fg_legal"][str(p)]["n"] >= 20]
        if not positions:
            continue

        legal_rates = [cell["fg_legal"][str(p)]["mean"] * 100 for p in positions]

        ax.plot(positions, legal_rates,
                color=color, linestyle=ls, marker=marker, markersize=6,
                linewidth=2, label=label, alpha=0.85)

    ax.set_xlabel("Step in generated plan", fontsize=12, fontweight='bold')
    ax.set_ylabel("Fraction of actions legal in true state (%)",
                  fontsize=12, fontweight='bold')
    ax.set_title("Compound-error signature: only 8-Puzzle WM degrades with step",
                 fontsize=12, fontweight='bold')
    ax.set_ylim(0, 105)
    ax.grid(alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    ax.legend(loc='lower left', fontsize=10, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(out_dir / "fig_tf_vs_freegen.png", dpi=300, bbox_inches='tight')
    plt.savefig(out_dir / "fig_tf_vs_freegen.pdf", bbox_inches='tight')
    plt.close(fig)
    print(f"\u2713 {out_dir / 'fig_tf_vs_freegen.png'}")
    print(f"\u2713 {out_dir / 'fig_tf_vs_freegen.pdf'}")


# ============================================================
# Table: aggregate TF loss + FG metrics at early and late positions
# ============================================================

def aggregate_at_positions(cell_data, metric, positions):
    """Return mean of metric across the given positions, weighted by n."""
    vals_n = []
    for p in positions:
        d = cell_data.get(metric, {}).get(str(p))
        if d is not None and d["n"] > 0:
            vals_n.append((d["mean"], d["n"]))
    if not vals_n:
        return None
    total_n = sum(n for _, n in vals_n)
    return sum(v * n for v, n in vals_n) / total_n


def make_table(data, out_dir):
    """Compact table showing the key cross-domain pattern.

    For each (domain, condition), report:
      - TF loss at step 1 and step 5 (or last available)
      - FG legal at step 1 and step 5 (or last available)

    The contrast: 8-Puzzle WM has TF loss going to near-zero (model
    learns the conditional well) but FG legal dropping (rollout fails).
    """
    bw_base = data.get("blocks_world", {}).get("baseline")
    bw_wm = data.get("blocks_world", {}).get("wm")
    ep_base = data.get("eight_puzzle", {}).get("baseline")
    ep_wm = data.get("eight_puzzle", {}).get("wm")

    # Use a range of positions for "early" and "late"; weighted average.
    # For Blocks World, "late" means step 4 (plans are 1-4); for 8-Puzzle,
    # step 10 (plans are 10-12 with some longer test items).
    bw_early_positions = [1, 2]
    bw_late_positions = [3, 4]
    ep_early_positions = [1, 2]
    ep_late_positions = [9, 10, 11]

    def row(label, cell, early_pos, late_pos):
        if cell is None:
            return f"{label} & -- & -- & -- & -- \\\\"
        tf_early = aggregate_at_positions(cell, "tf_loss", early_pos)
        tf_late = aggregate_at_positions(cell, "tf_loss", late_pos)
        leg_early = aggregate_at_positions(cell, "fg_legal", early_pos)
        leg_late = aggregate_at_positions(cell, "fg_legal", late_pos)

        def fmt_loss(x): return f"{x:.2f}" if x is not None else "--"
        def fmt_pct(x): return f"{100*x:.1f}\\%" if x is not None else "--"

        return (
            f"{label} & "
            f"{fmt_loss(tf_early)} $\\to$ {fmt_loss(tf_late)} & "
            f"{fmt_pct(leg_early)} $\\to$ {fmt_pct(leg_late)} \\\\"
        )

    rows = [
        "% Teacher-forced vs free-generation comparison.",
        "% Highlights the compound-error signature in 8-Puzzle WM only.",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Teacher-forced loss versus free-generation legality, by "
        "position within the generated plan. Each cell shows the change "
        "from early positions (steps 1--2) to later positions (steps 3--4 "
        "for Blocks World; 9--11 for 8-Puzzle). Teacher-forced loss "
        "decreases with position in all cells (the model has learned the "
        "conditional distribution well), consistent with the general "
        "pattern that later tokens are easier to predict from earlier "
        "context. However, free-generation legality $-$ whether the "
        "model's action at step $k$ is legal in the true state at "
        "step $k$ $-$ tells a different story. Only 8-Puzzle WM "
        "degrades sharply: state-prediction errors corrupt the rollout "
        "context, and subsequent action predictions become invalid. "
        "Blocks World maintains legality regardless of condition "
        "because state predictions are robust; 8-Puzzle Baseline "
        "maintains legality because there are no state predictions to "
        "err on. The compound-error mechanism only manifests in the "
        "combination of a constraint-rich domain (8-Puzzle) and the "
        "world-model auxiliary task.}",
        "\\label{tab:tf-vs-fg}",
        "\\begin{tabular}{lcc}",
        "\\toprule",
        "Cell & TF loss & FG legal \\\\",
        "     & (early $\\to$ late) & (early $\\to$ late) \\\\",
        "\\midrule",
        row("Blocks World, Baseline", bw_base, bw_early_positions, bw_late_positions),
        row("Blocks World, WM", bw_wm, bw_early_positions, bw_late_positions),
        row("8-Puzzle, Baseline", ep_base, ep_early_positions, ep_late_positions),
        row("8-Puzzle, WM", ep_wm, ep_early_positions, ep_late_positions),
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ]
    out_path = out_dir / "table_tf_vs_freegen.tex"
    out_path.write_text("\n".join(rows) + "\n")
    print(f"\u2713 {out_path}")


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper-dir", default="results/paper")
    args = ap.parse_args()

    out_dir = Path(args.paper_dir)
    if not out_dir.exists():
        print(f"ERROR: {out_dir} not found.")
        raise SystemExit(1)

    data = load_data(out_dir)

    print("Generating artifacts...")
    make_figure(data, out_dir)
    make_table(data, out_dir)

    print()
    print("Done. Outputs in", out_dir)


if __name__ == "__main__":
    main()
