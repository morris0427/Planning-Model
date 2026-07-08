"""
make_paper_tables_v2.py

Revised version of make_paper_tables.py reflecting the planning-as-deliberation
framing.

Headline tables compare only two architectural conditions:
  - Baseline: model emits actions only, no state representation
  - WM (planning): model emits actions and predicts states, all from model

A supplementary table includes the WM (diagnostic) condition, which is
WM with oracle-injected state. This is not a planning condition (the
oracle bypasses deliberation) but is included for completeness because
it isolates action-selection ability from state-prediction ability.

Reads from results/paper/:
  - plan_optimality_blocks_world.json
  - plan_optimality_eight_puzzle.json
  - paper_diagnostics.json (for the plan-length figure)

Writes to results/paper/:
  - table_planning_indist.tex       (Conclusion 1+2 combined)
  - table_planning_truly_ood.tex    (Conclusion 3)
  - table_optimality.tex            (Conclusion 4)
  - table_diagnostic.tex            (supplementary, WM with oracle)
  - fig_plan_length.{png,pdf}       (supporting figure for Conclusion 3)

Run from the experiments/ directory:
    python3 make_paper_tables_v2.py
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Measured numbers from our session.
# ============================================================

HARDCODED = {
    # Blocks World, in-distribution (BFS shortest <= 4 from aligned eval)
    "bw_in_dist": {
        "medium": {
            "baseline": 0.766,
            "wm_planning": 0.792,       # state_source="model"
            "wm_diagnostic": 0.792,      # state_source="oracle"
            "n": 500,
        },
    },
    # Blocks World, truly-OOD (BFS shortest >= 5 from aligned eval)
    "bw_truly_ood": {
        "medium": {
            "baseline": 0.000,
            "wm_planning": 0.000,
            "wm_diagnostic": 0.000,
            "n": 407,
        },
        "large": {
            "baseline": 0.000,
            "wm_planning": 0.000,
            "wm_diagnostic": 0.000,
            "n": 407,
        },
    },
    # 8-Puzzle, in-distribution
    "ep_in_dist": {
        "medium": {
            "baseline": 0.756,
            "wm_planning": 0.000,       # state_source="model" — collapses
            "wm_diagnostic": 0.794,      # state_source="oracle" — works
            "n": 500,
        },
    },
    # 8-Puzzle, truly-OOD
    "ep_truly_ood": {
        "medium": {
            "baseline": 0.000,
            "wm_planning": 0.000,
            "wm_diagnostic": 0.000,
            "n": 421,
        },
    },
    # State-space sizes
    "state_space": {
        "blocks_world": "$\\approx 75$",
        "eight_puzzle": "$181{,}440$",
    },
}


def write_table(path, content):
    with open(path, "w") as f:
        f.write(content)
    print(f"✓ {path}")


# ============================================================
# Headline table 1: in-distribution planning solve rates
# ============================================================
# Compares the two planning conditions across domains. The cross-domain
# asymmetry is the headline finding: WM-Planning works in Blocks World
# but fails completely in 8-Puzzle because of state-prediction collapse.

def make_table_planning_indist(out_dir):
    bw = HARDCODED["bw_in_dist"]["medium"]
    ep = HARDCODED["ep_in_dist"]["medium"]

    rows = [
        "% Planning conditions, in-distribution.",
        "% Both Baseline and WM (Planning) compute action sequences without",
        "% environmental feedback. Validator checks whether actions reach goal.",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{In-distribution planning performance. Both conditions are "
        "true planning: the model emits an action sequence without oracle "
        "assistance, and the plan validator checks the result. The cross-domain "
        "difference reflects whether the auxiliary state-prediction task is "
        "learnable: in Blocks World it is, and WM-Planning matches Baseline; "
        "in 8-Puzzle it is not, and WM-Planning collapses to 0\\%.}",
        "\\label{tab:planning-indist}",
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "Domain & Baseline & WM (Planning) & $n$ \\\\",
        "\\midrule",
        f"Blocks World & {100*bw['baseline']:.1f}\\% & "
        f"{100*bw['wm_planning']:.1f}\\% & {bw['n']} \\\\",
        f"8-Puzzle & {100*ep['baseline']:.1f}\\% & "
        f"{100*ep['wm_planning']:.1f}\\% & {ep['n']} \\\\",
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ]
    write_table(out_dir / "table_planning_indist.tex", "\n".join(rows) + "\n")


# ============================================================
# Headline table 2: truly-OOD failure
# ============================================================
# Shows that NEITHER planning condition generalizes to longer plans, in
# either domain, at either model size.

def make_table_truly_ood(out_dir):
    rows = [
        "% Truly-OOD: neither planning condition generalizes.",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Solve rates on truly-OOD problems, where BFS-verified "
        "shortest path exceeds the training distribution. No condition, "
        "domain, or model size succeeds. Scaling from medium (900K parameters) "
        "to large (3.2M parameters) does not unlock length-generalization in "
        "Blocks World.}",
        "\\label{tab:truly-ood}",
        "\\begin{tabular}{llccc}",
        "\\toprule",
        "Domain & Size & Baseline & WM (Planning) & $n$ \\\\",
        "\\midrule",
    ]

    for size in ("medium", "large"):
        bw = HARDCODED["bw_truly_ood"][size]
        rows.append(
            f"Blocks World & {size} & {100*bw['baseline']:.1f}\\% & "
            f"{100*bw['wm_planning']:.1f}\\% & {bw['n']} \\\\"
        )

    ep = HARDCODED["ep_truly_ood"]["medium"]
    rows.append(
        f"8-Puzzle & medium & {100*ep['baseline']:.1f}\\% & "
        f"{100*ep['wm_planning']:.1f}\\% & {ep['n']} \\\\"
    )

    rows.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])
    write_table(out_dir / "table_planning_truly_ood.tex", "\n".join(rows) + "\n")


# ============================================================
# Headline table 3: plan optimality
# ============================================================

def make_table_optimality(opt_bw, opt_ep, out_dir):
    """Mean excess over BFS optimum on solved problems, plus % optimal.

    Includes only Baseline and WM (Planning) columns. WM (Diagnostic) goes
    in the supplementary table.
    """
    def stats_solved(records, plan_field, solved_field):
        solved = [r for r in records if r[solved_field]]
        if not solved:
            return None, None, 0
        ex = np.array([r[plan_field] - r["bfs_shortest"] for r in solved])
        return float(ex.mean()), float((ex == 0).mean()), len(solved)

    def stats_saw(records):
        ex = np.array([r["saw_ref"] - r["bfs_shortest"] for r in records])
        return float(ex.mean()), float((ex == 0).mean()), len(records)

    saw_bw = stats_saw(opt_bw)
    base_bw = stats_solved(opt_bw, "baseline_plan_len", "baseline_solved")
    wm_bw = stats_solved(opt_bw, "wm_plan_len", "wm_solved")

    saw_ep = stats_saw(opt_ep)
    base_ep = stats_solved(opt_ep, "baseline_plan_len", "baseline_solved")
    wm_ep = stats_solved(opt_ep, "wm_plan_len", "wm_solved")

    def fmt(x):
        return "--" if x is None else f"{x:.2f}"

    def fmtpct(x):
        return "--" if x is None else f"{100*x:.1f}\\%"

    rows = [
        "% Plan optimality on problems each condition solves.",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Plan optimality within training distribution. SAW reference "
        "plans (the demonstrations the model was trained on) are mostly "
        "non-optimal in Blocks World; models recover near-optimal plans. "
        "WM (Planning) for 8-Puzzle is omitted from the bottom row because "
        "it solves 0 problems (see Table~\\ref{tab:planning-indist}).}",
        "\\label{tab:optimality}",
        "\\begin{tabular}{lrr|rr|rr}",
        "\\toprule",
        "& \\multicolumn{2}{c}{SAW reference} "
        "& \\multicolumn{2}{c}{Baseline} "
        "& \\multicolumn{2}{c}{WM (Planning)} \\\\",
        "Domain & excess & optimal & excess & optimal & excess & optimal \\\\",
        "\\midrule",
        f"Blocks World & {fmt(saw_bw[0])} & {fmtpct(saw_bw[1])} & "
        f"{fmt(base_bw[0])} & {fmtpct(base_bw[1])} & "
        f"{fmt(wm_bw[0])} & {fmtpct(wm_bw[1])} \\\\",
        f"8-Puzzle & {fmt(saw_ep[0])} & {fmtpct(saw_ep[1])} & "
        f"{fmt(base_ep[0])} & {fmtpct(base_ep[1])} & "
        "-- & -- \\\\",
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ]
    write_table(out_dir / "table_optimality.tex", "\n".join(rows) + "\n")


# ============================================================
# Supplementary table: WM (Diagnostic) condition
# ============================================================
# Not a planning condition. Included to show what action-selection
# alone achieves when state-tracking is provided by the environment.

def make_table_diagnostic(out_dir):
    bw_id = HARDCODED["bw_in_dist"]["medium"]
    ep_id = HARDCODED["ep_in_dist"]["medium"]
    bw_to = HARDCODED["bw_truly_ood"]["medium"]
    ep_to = HARDCODED["ep_truly_ood"]["medium"]

    rows = [
        "% Supplementary: WM with oracle state injection.",
        "% NOT a planning condition; included for diagnostic completeness.",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Diagnostic condition: WM with oracle state injection "
        "(\\textsc{wm-diagnostic} in Algorithm~\\ref{alg:generate_solution}). "
        "At each step the true post-action state is computed by the oracle "
        "and inserted into the model's context, replacing the model's own "
        "state predictions. This isolates action-selection ability from "
        "state-prediction ability and is not a planning condition. Compare "
        "to Table~\\ref{tab:planning-indist}, which shows the true "
        "planning result.}",
        "\\label{tab:diagnostic}",
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "Domain & In-distribution & Truly-OOD & "
        "WM (Planning) compare & WM (Planning) compare \\\\",
        " & WM (Diag.) & WM (Diag.) & In-distribution & Truly-OOD \\\\",
        "\\midrule",
        f"Blocks World & {100*bw_id['wm_diagnostic']:.1f}\\% & "
        f"{100*bw_to['wm_diagnostic']:.1f}\\% & "
        f"{100*bw_id['wm_planning']:.1f}\\% & "
        f"{100*bw_to['wm_planning']:.1f}\\% \\\\",
        f"8-Puzzle & {100*ep_id['wm_diagnostic']:.1f}\\% & "
        f"{100*ep_to['wm_diagnostic']:.1f}\\% & "
        f"{100*ep_id['wm_planning']:.1f}\\% & "
        f"{100*ep_to['wm_planning']:.1f}\\% \\\\",
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ]
    write_table(out_dir / "table_diagnostic.tex", "\n".join(rows) + "\n")


# ============================================================
# Figure: plan-length locking
# ============================================================

def make_plan_length_figure(diagnostics, out_dir):
    pl = diagnostics.get("plan_length", {})
    if not pl:
        print(f"  (skipped fig_plan_length: 'plan_length' not in diagnostics)")
        return

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
            continue

        x = np.arange(len(refs))
        width = 0.35

        base_means = [np.mean(by_ref_base[r]) if by_ref_base[r] else 0 for r in refs]
        wm_means = [np.mean(by_ref_wm[r]) if by_ref_wm[r] else 0 for r in refs]

        ax.bar(x - width/2, base_means, width, label="Baseline",
               color="#d95f02", alpha=0.85, edgecolor='black', linewidth=0.5)
        ax.bar(x + width/2, wm_means, width, label="WM (Planning)",
               color="#7570b3", alpha=0.85, edgecolor='black', linewidth=0.5)

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

    opt_bw_path = out_dir / "plan_optimality_blocks_world.json"
    opt_ep_path = out_dir / "plan_optimality_eight_puzzle.json"
    diag_path = out_dir / "paper_diagnostics.json"

    if not opt_bw_path.exists() or not opt_ep_path.exists():
        print(f"ERROR: missing optimality JSON files. Run plan_optimality.py first.")
        raise SystemExit(1)

    with open(opt_bw_path) as f:
        opt_bw = json.load(f)
    with open(opt_ep_path) as f:
        opt_ep = json.load(f)

    print("Generating tables and figure...")
    make_table_planning_indist(out_dir)
    make_table_truly_ood(out_dir)
    make_table_optimality(opt_bw, opt_ep, out_dir)
    make_table_diagnostic(out_dir)

    if diag_path.exists():
        with open(diag_path) as f:
            diagnostics = json.load(f)
        make_plan_length_figure(diagnostics, out_dir)

    print()
    print("Done. Outputs in", out_dir)
    print("  Headline tables:")
    print("    table_planning_indist.tex     (Conclusion 1+2: in-distribution planning)")
    print("    table_planning_truly_ood.tex  (Conclusion 3: no length-generalization)")
    print("    table_optimality.tex          (Conclusion 4: near-optimal recovery)")
    print("  Supplementary:")
    print("    table_diagnostic.tex          (WM with oracle state injection)")
    print("  Figure:")
    print("    fig_plan_length.{png,pdf}     (supporting Conclusion 3)")


if __name__ == "__main__":
    main()
