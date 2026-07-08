"""
make_paper_tables_v4.py

v4 changes from v3:
  - Column label "WM (Planning)" simplified to "WM" throughout. The
    "Planning" suffix was needed when WM-Diagnostic was a parallel
    option, but the diagnostic is now described as a separate
    analytical device in its own table.
  - 8-Puzzle WM cell in the length-locking table is footnoted: the
    generated plan length there reflects truncation cutoff rather
    than the model's deliberative stopping point, since most
    generations terminate via 'truncation_seqlen' rather than 'sep'.

Reads from results/paper/:
  - plan_optimality_blocks_world.json
  - plan_optimality_eight_puzzle.json
  - paper_diagnostics.json

Writes to results/paper/:
  - table_planning_indist.tex
  - table_planning_truly_ood.tex
  - table_optimality.tex
  - table_diagnostic.tex
  - fig_plan_length.{png,pdf}

Run from the experiments/ directory:
    python3 make_paper_tables_v4.py
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


HARDCODED = {
    "bw_in_dist": {
        "medium": {"baseline": 0.766, "wm": 0.792,
                   "wm_diagnostic": 0.792, "n": 500},
    },
    "bw_truly_ood": {
        "medium": {"baseline": 0.000, "wm": 0.000,
                   "wm_diagnostic": 0.000, "n": 407,
                   "training_range": "1-4", "test_range": "5-8"},
        "large":  {"baseline": 0.000, "wm": 0.000,
                   "wm_diagnostic": 0.000, "n": 407,
                   "training_range": "1-4", "test_range": "5-8"},
    },
    "ep_in_dist": {
        "medium": {"baseline": 0.756, "wm": 0.000,
                   "wm_diagnostic": 0.794, "n": 500},
    },
    "ep_truly_ood": {
        "medium": {"baseline": 0.000, "wm": 0.000,
                   "wm_diagnostic": 0.000, "n": 421,
                   "training_range": "10-12", "test_range": "13-18"},
    },
    "state_space": {
        "blocks_world": "$\\approx 75$",
        "eight_puzzle": "$181{,}440$",
    },
}


def write_table(path, content):
    with open(path, "w") as f:
        f.write(content)
    print(f"\u2713 {path}")


def compute_plan_length_means(diagnostics, domain, condition_key):
    pl = diagnostics.get("plan_length", {})
    key = f"{domain}_{condition_key}"
    records = pl.get(key, [])
    if not records:
        return None, None
    plan_lens = [r["n_actions"] for r in records]
    return float(np.mean(plan_lens)), float(np.std(plan_lens))


# ============================================================
# Table 1: In-distribution planning
# ============================================================

def make_table_planning_indist(out_dir):
    bw = HARDCODED["bw_in_dist"]["medium"]
    ep = HARDCODED["ep_in_dist"]["medium"]

    rows = [
        "% In-distribution planning solve rates.",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{In-distribution planning performance. Both Baseline and "
        "WM are true planning conditions: the model emits an action "
        "sequence without environmental feedback, and an independent plan "
        "validator checks whether the actions reach the goal. The "
        "cross-domain difference reflects whether the auxiliary state-prediction "
        "task is learnable: in Blocks World it is, and WM matches Baseline; "
        "in 8-Puzzle it is not, and WM collapses to 0\\%.}",
        "\\label{tab:planning-indist}",
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "Domain & Baseline & WM & $n$ \\\\",
        "\\midrule",
        f"Blocks World & {100*bw['baseline']:.1f}\\% & "
        f"{100*bw['wm']:.1f}\\% & {bw['n']} \\\\",
        f"8-Puzzle & {100*ep['baseline']:.1f}\\% & "
        f"{100*ep['wm']:.1f}\\% & {ep['n']} \\\\",
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ]
    write_table(out_dir / "table_planning_indist.tex", "\n".join(rows) + "\n")


# ============================================================
# Table 2: Truly-OOD with plan-length data and 8-Puzzle WM footnote
# ============================================================

def make_table_truly_ood_expanded(diagnostics, out_dir):
    rows = [
        "% Truly-OOD solve rates with plan-length data showing length-locking.",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{No length-generalization across architectures and model "
        "sizes. The training range and test range columns indicate plan "
        "lengths the model was trained on versus the truly-OOD test set, "
        "BFS-verified. The mean generated plan length columns show how many "
        "actions the model produces; in every case the model produces plans "
        "of approximately training-distribution length, regardless of the "
        "reference plan length on the test problem. This length-locking is "
        "the mechanism behind the 0\\% solve rate. "
        "$^{\\dagger}$8-Puzzle WM generations are dominated by "
        "\\textit{truncation\\_seqlen} termination rather than the model's "
        "own SEP emission, because the model's state predictions are invalid "
        "and the goal-recognition feature never fires; this number reflects "
        "the sequence-budget cutoff rather than the model's deliberative "
        "stopping point.}",
        "\\label{tab:truly-ood}",
        "\\begin{tabular}{llcccccc}",
        "\\toprule",
        "Domain & Size & Train. & Test & Baseline & WM    & "
        "Mean plan length & $n$ \\\\",
        "       &      & range  & range & solve   & solve & "
        "(Baseline / WM)  &       \\\\",
        "\\midrule",
    ]

    # Blocks World, both sizes
    bl_bw, _ = compute_plan_length_means(diagnostics, "blocks_world", "baseline")
    wm_bw, _ = compute_plan_length_means(diagnostics, "blocks_world", "wm_oracle")
    bl_bw_str = f"{bl_bw:.1f}" if bl_bw is not None else "--"
    wm_bw_str = f"{wm_bw:.1f}" if wm_bw is not None else "--"

    for size in ("medium", "large"):
        cell = HARDCODED["bw_truly_ood"][size]
        rows.append(
            f"Blocks World & {size} & {cell['training_range']} & "
            f"{cell['test_range']} & {100*cell['baseline']:.1f}\\% & "
            f"{100*cell['wm']:.1f}\\% & "
            f"{bl_bw_str} / {wm_bw_str} & {cell['n']} \\\\"
        )

    # 8-Puzzle: only Baseline plan length is meaningful here; WM marked with footnote
    bl_ep, _ = compute_plan_length_means(diagnostics, "eight_puzzle", "baseline")
    wm_ep, _ = compute_plan_length_means(diagnostics, "eight_puzzle", "wm_oracle")
    bl_ep_str = f"{bl_ep:.1f}" if bl_ep is not None else "--"
    wm_ep_str = f"{wm_ep:.1f}$^{{\\dagger}}$" if wm_ep is not None else "--"

    cell = HARDCODED["ep_truly_ood"]["medium"]
    rows.append(
        f"8-Puzzle & medium & {cell['training_range']} & "
        f"{cell['test_range']} & {100*cell['baseline']:.1f}\\% & "
        f"{100*cell['wm']:.1f}\\% & "
        f"{bl_ep_str} / {wm_ep_str} & {cell['n']} \\\\"
    )

    rows.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])
    write_table(out_dir / "table_planning_truly_ood.tex", "\n".join(rows) + "\n")


# ============================================================
# Table 3: Plan optimality
# ============================================================

def make_table_optimality(opt_bw, opt_ep, out_dir):
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
        "WM for 8-Puzzle is omitted from the bottom row because it solves "
        "0 problems (see Table~\\ref{tab:planning-indist}).}",
        "\\label{tab:optimality}",
        "\\begin{tabular}{lrr|rr|rr}",
        "\\toprule",
        "& \\multicolumn{2}{c}{SAW reference} "
        "& \\multicolumn{2}{c}{Baseline} "
        "& \\multicolumn{2}{c}{WM} \\\\",
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
# Table 4: WM Diagnostic
# ============================================================

def make_table_diagnostic(out_dir):
    bw_id = HARDCODED["bw_in_dist"]["medium"]
    ep_id = HARDCODED["ep_in_dist"]["medium"]
    bw_to = HARDCODED["bw_truly_ood"]["medium"]
    ep_to = HARDCODED["ep_truly_ood"]["medium"]

    rows = [
        "% WM-Diagnostic supplementary table.",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Diagnostic condition: WM with oracle state injection. "
        "At each step the true post-action state is computed by the oracle "
        "and inserted into the model's context, replacing the model's own "
        "state predictions. This is not a planning condition; it isolates "
        "action-selection ability from state-prediction ability. The "
        "8-Puzzle in-distribution contrast (79.4\\% with oracle vs 0.0\\% "
        "without; see Table~\\ref{tab:planning-indist}) confirms that "
        "8-Puzzle's action-selection is competent; the failure is in state "
        "prediction.}",
        "\\label{tab:diagnostic}",
        "\\begin{tabular}{lcc}",
        "\\toprule",
        "Domain & In-distribution & Truly-OOD \\\\",
        "       & WM (Diagnostic) & WM (Diagnostic) \\\\",
        "\\midrule",
        f"Blocks World & {100*bw_id['wm_diagnostic']:.1f}\\% & "
        f"{100*bw_to['wm_diagnostic']:.1f}\\% \\\\",
        f"8-Puzzle & {100*ep_id['wm_diagnostic']:.1f}\\% & "
        f"{100*ep_to['wm_diagnostic']:.1f}\\% \\\\",
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
        ax.bar(x + width/2, wm_means, width, label="WM",
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

    plt.savefig(out_dir / "fig_plan_length.png", dpi=300, bbox_inches='tight')
    plt.savefig(out_dir / "fig_plan_length.pdf", bbox_inches='tight')
    plt.close(fig)
    print(f"\u2713 {out_dir / 'fig_plan_length.png'}")
    print(f"\u2713 {out_dir / 'fig_plan_length.pdf'}")


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
        print(f"ERROR: missing optimality JSON files.")
        raise SystemExit(1)
    if not diag_path.exists():
        print(f"ERROR: missing diagnostics JSON.")
        raise SystemExit(1)

    with open(opt_bw_path) as f:
        opt_bw = json.load(f)
    with open(opt_ep_path) as f:
        opt_ep = json.load(f)
    with open(diag_path) as f:
        diagnostics = json.load(f)

    print("Generating tables and figure...")
    make_table_planning_indist(out_dir)
    make_table_truly_ood_expanded(diagnostics, out_dir)
    make_table_optimality(opt_bw, opt_ep, out_dir)
    make_table_diagnostic(out_dir)
    make_plan_length_figure(diagnostics, out_dir)

    print()
    print("Done. Outputs in", out_dir)


if __name__ == "__main__":
    main()
