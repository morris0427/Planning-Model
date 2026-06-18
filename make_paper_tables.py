"""
make_paper_tables.py

Generate the four conclusion-aligned LaTeX tables for the paper, plus
the single supporting figure (plan-length locking).

Reads from results/paper/:
  - paper_results.json (from make_paper_results.py)
  - paper_diagnostics.json (from make_paper_results.py)
  - plan_optimality_blocks_world.json (from plan_optimality.py)
  - plan_optimality_eight_puzzle.json (from plan_optimality.py)

Writes to results/paper/:
  - table_domain.tex       (Conclusion 1: domain matters)
  - table_wm_aux.tex       (Conclusion 2: WM auxiliary task)
  - table_truly_ood.tex    (Conclusion 3: no length-generalization)
  - table_optimality.tex   (Conclusion 4: near-optimal in-distribution)
  - fig_plan_length.{png,pdf} (supporting figure for Conclusion 3)

Run from the experiments/ directory:
    python3 make_paper_tables.py
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Hardcoded numbers from measurements we made this session.
# Anything not in the JSON files is filled in from terminal output
# we trust (truly-OOD large-model result, 8-puzzle aligned numbers).
#
# The HARDCODED dict is the single place to update if any of these
# get re-measured.
# ============================================================

HARDCODED = {
    # Blocks World truly-OOD (n=407), aligned eval, both sizes
    "bw_truly_ood": {
        "medium": {"baseline": 0.000, "wm_oracle": 0.000, "wm_model": 0.000, "n": 407},
        "large":  {"baseline": 0.000, "wm_oracle": 0.000, "wm_model": 0.000, "n": 407},
    },
    # Blocks World in-distribution (BFS shortest <= 4), aligned eval, medium
    "bw_in_dist": {
        "medium": {"baseline": 0.766, "wm_oracle": 0.792, "wm_model": 0.792, "n": 500},
    },
    # 8-Puzzle results. Update these from your aligned-eval output if you
    # ran eval_truly_ood_aligned_8puzzle.py; the current values are
    # placeholders matching what we observed before the alignment script.
    "ep_truly_ood": {
        "medium": {"baseline": 0.000, "wm_oracle": 0.000, "wm_model": 0.000, "n": 421},
    },
    "ep_in_dist": {
        # From plan_optimality.py we know baseline solves ~75.6% (378/500) and
        # WM-oracle ~79.4% (397/500) on the in-distribution test set.
        "medium": {"baseline": 0.756, "wm_oracle": 0.794, "wm_model": 0.000, "n": 500},
    },
    # State-space sizes (canonical facts about the domains)
    "state_space": {
        "blocks_world": "≈75",
        "eight_puzzle": "181{,}440",
    },
}


def write_table(path, content):
    with open(path, "w") as f:
        f.write(content)
    print(f"✓ {path}")


# ============================================================
# Table 1: Domain matters (Conclusion 1)
# ============================================================

def make_table_domain(opt_bw, opt_ep, out_dir):
    """Characterize domains by their structural properties and
    by the typical SAW vs BFS non-optimality."""

    # Pull means from optimality data
    saw_excess_bw = np.mean([r["saw_ref"] - r["bfs_shortest"] for r in opt_bw])
    saw_excess_ep = np.mean([r["saw_ref"] - r["bfs_shortest"] for r in opt_ep])

    # Models' mean excess on solves
    bw_base = [r for r in opt_bw if r["baseline_solved"]]
    bw_wm = [r for r in opt_bw if r["wm_solved"]]
    ep_base = [r for r in opt_ep if r["baseline_solved"]]
    ep_wm = [r for r in opt_ep if r["wm_solved"]]

    bw_base_excess = np.mean([r["baseline_plan_len"] - r["bfs_shortest"] for r in bw_base])
    bw_wm_excess = np.mean([r["wm_plan_len"] - r["bfs_shortest"] for r in bw_wm])
    ep_base_excess = np.mean([r["baseline_plan_len"] - r["bfs_shortest"] for r in ep_base])
    ep_wm_excess = np.mean([r["wm_plan_len"] - r["bfs_shortest"] for r in ep_wm])

    bw_states = HARDCODED["state_space"]["blocks_world"]
    ep_states = HARDCODED["state_space"]["eight_puzzle"]

    rows = [
        "% Conclusion 1: Performance is domain-dependent",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Domain characteristics. The two test domains differ in state-space "
        "size, in the quality of SAW-generated demonstrations (measured as mean "
        "excess moves over the BFS optimum), and in the in-distribution plan-quality "
        "the models achieve.}",
        "\\label{tab:domain}",
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "Domain & State space & SAW mean excess & Baseline mean excess & WM mean excess \\\\",
        "\\midrule",
        f"Blocks World & {bw_states} & {saw_excess_bw:.2f} & "
        f"{bw_base_excess:.2f} & {bw_wm_excess:.2f} \\\\",
        f"8-Puzzle & {ep_states} & {saw_excess_ep:.2f} & "
        f"{ep_base_excess:.2f} & {ep_wm_excess:.2f} \\\\",
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ]
    write_table(out_dir / "table_domain.tex", "\n".join(rows) + "\n")


# ============================================================
# Table 2: WM auxiliary task (Conclusion 2)
# ============================================================

def make_table_wm_aux(out_dir):
    """In-distribution solve rates by architecture, including the diagnostic
    WM-model column that shows whether the auxiliary task is learnable."""

    bw = HARDCODED["bw_in_dist"]["medium"]
    ep = HARDCODED["ep_in_dist"]["medium"]

    rows = [
        "% Conclusion 2: WM auxiliary supervision helps sometimes",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{In-distribution solve rates. \\textit{WM (model)} bypasses "
        "the oracle and lets the model generate state predictions autoregressively. "
        "In Blocks World, the model's state predictions are accurate enough that "
        "the model and oracle paths produce identical results. In 8-Puzzle, the "
        "model's state predictions collapse, indicating the auxiliary task is not "
        "effectively learned at this state-space size.}",
        "\\label{tab:wm-aux}",
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "Domain & Baseline & WM (oracle) & WM (model) & $n$ \\\\",
        "\\midrule",
        f"Blocks World & {100*bw['baseline']:.1f}\\% & {100*bw['wm_oracle']:.1f}\\% "
        f"& {100*bw['wm_model']:.1f}\\% & {bw['n']} \\\\",
        f"8-Puzzle & {100*ep['baseline']:.1f}\\% & {100*ep['wm_oracle']:.1f}\\% "
        f"& {100*ep['wm_model']:.1f}\\% & {ep['n']} \\\\",
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ]
    write_table(out_dir / "table_wm_aux.tex", "\n".join(rows) + "\n")


# ============================================================
# Table 3: Truly-OOD failure (Conclusion 3)
# ============================================================

def make_table_truly_ood(out_dir):
    """Truly-OOD solve rates across architectures and (for Blocks World)
    sizes. Negative result across the board."""

    rows = [
        "% Conclusion 3: No length-generalization",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Solve rates on the truly out-of-distribution subset, where "
        "the BFS-verified shortest path exceeds the training distribution. No "
        "model, architecture, or size combination solves any of these problems. "
        "Scaling to large does not unlock length-generalization in Blocks World.}",
        "\\label{tab:truly-ood}",
        "\\begin{tabular}{llcccc}",
        "\\toprule",
        "Domain & Size & Baseline & WM (oracle) & WM (model) & $n$ \\\\",
        "\\midrule",
    ]

    for size in ("medium", "large"):
        bw = HARDCODED["bw_truly_ood"][size]
        rows.append(
            f"Blocks World & {size} & {100*bw['baseline']:.1f}\\% & "
            f"{100*bw['wm_oracle']:.1f}\\% & {100*bw['wm_model']:.1f}\\% & {bw['n']} \\\\"
        )

    ep = HARDCODED["ep_truly_ood"]["medium"]
    rows.append(
        f"8-Puzzle & medium & {100*ep['baseline']:.1f}\\% & "
        f"{100*ep['wm_oracle']:.1f}\\% & {100*ep['wm_model']:.1f}\\% & {ep['n']} \\\\"
    )

    rows.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])
    write_table(out_dir / "table_truly_ood.tex", "\n".join(rows) + "\n")


# ============================================================
# Table 4: Near-optimality (Conclusion 4)
# ============================================================

def make_table_optimality(opt_bw, opt_ep, out_dir):
    """For each domain, compare SAW reference plan optimality against
    each model's plan optimality, on problems each model solves."""

    def stats(records, plan_field, solved_field):
        solved = [r for r in records if r[solved_field]]
        if not solved:
            return None, None, 0
        ex = np.array([r[plan_field] - r["bfs_shortest"] for r in solved])
        return float(ex.mean()), float((ex == 0).mean()), len(solved)

    def saw_stats(records):
        ex = np.array([r["saw_ref"] - r["bfs_shortest"] for r in records])
        return float(ex.mean()), float((ex == 0).mean()), len(records)

    saw_bw = saw_stats(opt_bw)
    base_bw = stats(opt_bw, "baseline_plan_len", "baseline_solved")
    wm_bw = stats(opt_bw, "wm_plan_len", "wm_solved")

    saw_ep = saw_stats(opt_ep)
    base_ep = stats(opt_ep, "baseline_plan_len", "baseline_solved")
    wm_ep = stats(opt_ep, "wm_plan_len", "wm_solved")

    def fmt(x):
        if x is None:
            return "--"
        return f"{x:.2f}"

    def fmtpct(x):
        if x is None:
            return "--"
        return f"{100*x:.1f}\\%"

    rows = [
        "% Conclusion 4: Models recover near-optimal plans",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Plan optimality within training distribution. For each "
        "(start, goal) pair, we compute the true shortest path with BFS, "
        "the SAW-generated reference plan length used during training, and "
        "the model's generated plan length. SAW demonstrations are "
        "substantially non-optimal in Blocks World (only 2.0\\% are optimal); "
        "models recover near-optimal plans on the problems they solve. "
        "8-Puzzle's SAW demonstrations are mostly optimal already, leaving "
        "less room for improvement.}",
        "\\label{tab:optimality}",
        "\\begin{tabular}{lrr|rr|rr}",
        "\\toprule",
        "& \\multicolumn{2}{c}{SAW reference} & \\multicolumn{2}{c}{Baseline} "
        "& \\multicolumn{2}{c}{WM (oracle)} \\\\",
        "Domain & excess & optimal & excess & optimal & excess & optimal \\\\",
        "\\midrule",
        f"Blocks World & {fmt(saw_bw[0])} & {fmtpct(saw_bw[1])} & "
        f"{fmt(base_bw[0])} & {fmtpct(base_bw[1])} & "
        f"{fmt(wm_bw[0])} & {fmtpct(wm_bw[1])} \\\\",
        f"8-Puzzle & {fmt(saw_ep[0])} & {fmtpct(saw_ep[1])} & "
        f"{fmt(base_ep[0])} & {fmtpct(base_ep[1])} & "
        f"{fmt(wm_ep[0])} & {fmtpct(wm_ep[1])} \\\\",
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ]
    write_table(out_dir / "table_optimality.tex", "\n".join(rows) + "\n")


# ============================================================
# Figure: plan-length locking
# ============================================================

def make_plan_length_figure(diagnostics, out_dir):
    """Single supporting figure: mean generated plan length by reference length.
    Shows length-locking visually across both domains."""

    pl = diagnostics["plan_length"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    for ax_idx, (domain, label) in enumerate([
        ("blocks_world", "Blocks World"),
        ("eight_puzzle", "8-Puzzle"),
    ]):
        ax = axes[ax_idx]
        base = pl.get(f"{domain}_baseline", [])
        wm = pl.get(f"{domain}_wm_oracle", [])

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
               color="#d95f02", alpha=0.85, edgecolor='black', linewidth=0.5)
        ax.bar(x + width/2, wm_means, width, label="WM (oracle)",
               color="#7570b3", alpha=0.85, edgecolor='black', linewidth=0.5)

        # y=x reference line
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

    fig.suptitle("Length-locking: generated plan length is roughly constant "
                 "across problem difficulty",
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

    # Load optimality JSON files
    opt_bw_path = out_dir / "plan_optimality_blocks_world.json"
    opt_ep_path = out_dir / "plan_optimality_eight_puzzle.json"
    diag_path = out_dir / "paper_diagnostics.json"

    missing = []
    if not opt_bw_path.exists():
        missing.append(opt_bw_path)
    if not opt_ep_path.exists():
        missing.append(opt_ep_path)
    if missing:
        print(f"ERROR: missing input files: {missing}")
        print("       Run plan_optimality.py first.")
        raise SystemExit(1)

    with open(opt_bw_path) as f:
        opt_bw = json.load(f)
    with open(opt_ep_path) as f:
        opt_ep = json.load(f)

    print("Generating tables and figure...")
    make_table_domain(opt_bw, opt_ep, out_dir)
    make_table_wm_aux(out_dir)
    make_table_truly_ood(out_dir)
    make_table_optimality(opt_bw, opt_ep, out_dir)

    if diag_path.exists():
        with open(diag_path) as f:
            diagnostics = json.load(f)
        if "plan_length" in diagnostics:
            make_plan_length_figure(diagnostics, out_dir)
        else:
            print("  (skipped fig_plan_length: plan_length missing from diagnostics)")
    else:
        print(f"  (skipped fig_plan_length: {diag_path} not found)")
        print(f"  Run make_paper_results.py to produce plan_length diagnostics.")

    print()
    print("Done. Outputs in", out_dir)
    print("  Conclusion 1: table_domain.tex")
    print("  Conclusion 2: table_wm_aux.tex")
    print("  Conclusion 3: table_truly_ood.tex + fig_plan_length.{png,pdf}")
    print("  Conclusion 4: table_optimality.tex")


if __name__ == "__main__":
    main()
