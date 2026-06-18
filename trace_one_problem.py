"""
Single-problem trace for 8-Puzzle WM evaluation.

Purpose
-------
We have a measurement contradiction: the same medium-WM checkpoint scored 94%
solve in an earlier table, but 6% under the current `evaluate_solve_rate`.
Teacher-forced action accuracy on that checkpoint is 95.6%, which is
incompatible with 6% solve unless the free-generation path is misreading the
model's outputs.

This script picks a few test problems and prints, side by side:
  - the reference (ground-truth) sequence, token by token with labels
  - what `generate_solution` produces, token by token with labels
  - what `check_solution_correctness` does with the generated tokens
  - the oracle-applied board after each generated move

If there's a token-convention mismatch, an off-by-one in state-token offsets,
or a SEP-detection bug, it will be visible in the trace.

Usage
-----
    python3 trace_one_problem.py --domain eight_puzzle --size medium --wm \
        --num-problems 2 \
        --indices 0 5 12       # or omit to take the first N

    # baseline:
    python3 trace_one_problem.py --domain eight_puzzle --size large
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from config import ModelPresets, DataPresets
from data.base import DatasetFactory  # noqa: F401  (registers datasets)
import trainer as T


# ----------------------------- token labeling ----------------------------- #

EIGHT_PUZZLE_MOVE_NAME = {10: "up", 11: "down", 12: "left", 13: "right"}
EIGHT_PUZZLE_SPECIAL = {14: "SEP", 15: "PAD"}


def label_token(tok: int) -> str:
    if tok in EIGHT_PUZZLE_SPECIAL:
        return EIGHT_PUZZLE_SPECIAL[tok]
    if tok in EIGHT_PUZZLE_MOVE_NAME:
        return f"{EIGHT_PUZZLE_MOVE_NAME[tok]}({tok})"
    if 0 <= tok <= 9:
        return f"tile{tok}"
    return f"?({tok})"


def fmt_token_row(tokens, width=12):
    """Pretty-print a list of tokens with index headers."""
    lines = []
    for chunk_start in range(0, len(tokens), width):
        chunk = tokens[chunk_start:chunk_start + width]
        idx_row = " ".join(f"{chunk_start+i:>5}" for i in range(len(chunk)))
        tok_row = " ".join(f"{label_token(t):>5}" for t in chunk)
        lines.append("    idx: " + idx_row)
        lines.append("    tok: " + tok_row)
        lines.append("")
    return "\n".join(lines)


def board_str(board: np.ndarray) -> str:
    """3x3 board as a small block."""
    rows = []
    for r in range(3):
        rows.append("  " + " ".join(
            "_" if board[r, c] == 0 else str(int(board[r, c])) for c in range(3)
        ))
    return "\n".join(rows)


# ----------------------------- model loading ----------------------------- #

def load_checkpoint(domain, size, use_wm, results_dir="results"):
    """Locate and load a checkpoint, returning (model, test_problems, gen, device)."""
    device = torch.device("cpu")

    suffix = "wm" if use_wm else "base"
    candidates = [
        f"{domain}_in_distribution_{size}_{suffix}",
        f"{domain}_{size}_{suffix}_std",
        f"{domain}_{size}_{suffix}",
        f"{domain}_{size}_{suffix}_shared",
    ]
    ckpt = None
    for name in candidates:
        p = Path(results_dir) / name / "best_model.pth"
        if p.exists():
            ckpt = p
            break
    if ckpt is None:
        raise FileNotFoundError(
            f"No checkpoint for {domain}/{size}/{suffix}. Tried: {candidates}"
        )
    print(f"checkpoint: {ckpt}")

    state = torch.load(ckpt, map_location=device)
    ckpt_max_seq, ckpt_d_model = state["pos_encoder.weight"].shape
    ckpt_vocab = state["embedding.weight"].shape[0]

    model_cfg = {
        "tiny": ModelPresets.tiny, "small": ModelPresets.small,
        "medium": ModelPresets.medium, "large": ModelPresets.large,
    }[size](use_world_model=use_wm)

    model = T.PlanningTransformer(
        vocab_size=ckpt_vocab,
        d_model=ckpt_d_model,
        nhead=model_cfg.n_heads,
        num_layers=model_cfg.n_layers,
        dim_feedforward=model_cfg.d_ff,
        max_seq_length=ckpt_max_seq,
    ).to(device)
    model.load_state_dict(state)
    model.eval()

    # Load cached test data + dataset generator (for use_world_model flag)
    if domain == "eight_puzzle":
        data_cfg = DataPresets.eight_puzzle_standard()
    else:
        data_cfg = DataPresets.blocks_world_standard()

    class _Shim: pass
    shim = _Shim(); shim.data = data_cfg
    _, test_problems = T.load_cached_data(shim, use_wm)
    if test_problems is None:
        raise FileNotFoundError(f"No cached test data for {domain} (wm={use_wm}).")

    test_gen = DatasetFactory.create(
        domain=domain,
        difficulty_range=data_cfg.test_difficulty_range,
        num_samples=len(test_problems),
        use_world_model=use_wm,
    )
    test_gen.problems = test_problems

    print(f"model: d_model={ckpt_d_model}, max_seq_length={ckpt_max_seq}, "
          f"params={sum(p.numel() for p in model.parameters()):,}")
    print(f"test set: {len(test_problems)} problems")
    return model, test_problems, test_gen, device


# ----------------------------- the trace ----------------------------- #

def trace_problem(model, problem, dataset_generator, device, problem_idx=None):
    """Print everything we know about one problem's reference + generation + scoring."""
    print("\n" + "=" * 76)
    if problem_idx is not None:
        print(f"PROBLEM #{problem_idx}")
    print("=" * 76)

    seq = problem["sequence"]
    num_moves = problem["num_moves"]
    use_wm = dataset_generator.use_world_model

    # --- 1. Reference sequence ---
    print(f"\nReference: num_moves={num_moves}, use_wm={use_wm}, "
          f"sequence length={len(seq)}")
    start = np.array(seq[1:10]).reshape(3, 3)
    goal = np.array(seq[11:20]).reshape(3, 3)
    print("\n  START board:")
    print(board_str(start))
    print("\n  GOAL board:")
    print(board_str(goal))

    # Reference move tokens (between the goal block and SEP)
    ref_moves_section = seq[20:]
    if ref_moves_section and ref_moves_section[-1] == 14:
        ref_moves_section = ref_moves_section[:-1]
    # In WM encoding moves alternate with 9 state tokens; in baseline they're contiguous
    ref_move_tokens = [t for t in ref_moves_section if t in EIGHT_PUZZLE_MOVE_NAME]
    print(f"\n  Reference moves ({len(ref_move_tokens)}): "
          + " ".join(EIGHT_PUZZLE_MOVE_NAME[t] for t in ref_move_tokens))

    # --- 2. Free generation (full sequence) ---
    print("\n" + "-" * 76)
    print("FREE GENERATION (greedy, oracle states for WM)")
    print("-" * 76)
    generated, info = T.generate_solution(
        model, problem, dataset_generator, device,
        max_length=100, return_info=True
    )
    print(f"\n  termination = {info['termination']}, "
          f"moves_generated = {info['moves_generated']}, "
          f"final_len = {info['final_len']} "
          f"(model_max_seq_length = {info['model_max_seq_length']})")

    print("\n  Full generated token sequence (first 20 = context, rest = output):")
    print(fmt_token_row(generated))

    # Extract the moves the model actually emitted (positions >=20, by mapping)
    gen_after_context = generated[20:]
    gen_move_tokens = [t for t in gen_after_context if t in EIGHT_PUZZLE_MOVE_NAME]
    gen_sep_positions = [i for i, t in enumerate(generated) if t == 14]
    print(f"\n  Generated moves ({len(gen_move_tokens)}): "
          + " ".join(EIGHT_PUZZLE_MOVE_NAME[t] for t in gen_move_tokens))
    print(f"  SEP tokens emitted at positions: {gen_sep_positions}")

    # --- 3. Apply generated moves to start board (independent oracle replay) ---
    print("\n" + "-" * 76)
    print("INDEPENDENT REPLAY: apply generated moves to START with oracle")
    print("-" * 76)
    current = start.copy()
    invalid_at = None
    for i, t in enumerate(gen_move_tokens):
        mv = EIGHT_PUZZLE_MOVE_NAME[t]
        nxt = T.apply_move_8puzzle(current, mv)
        if nxt is None:
            invalid_at = (i, mv, current.copy())
            break
        current = nxt
    reached_goal = np.array_equal(current, goal)
    print(f"  Reached goal after replay: {reached_goal}")
    if invalid_at is not None:
        i, mv, bd = invalid_at
        print(f"  Replay STOPPED at move #{i} ({mv}): invalid from board")
        print(board_str(bd))
    print("\n  Board after replaying all generated moves:")
    print(board_str(current))

    # --- 4. What check_solution_correctness says ---
    print("\n" + "-" * 76)
    print("EVALUATOR VERDICT (check_solution_correctness)")
    print("-" * 76)
    verdict = T.check_solution_correctness(generated, problem, dataset_generator)
    print(f"  check_solution_correctness returned: {verdict}")

    # --- 5. Cross-check: do (3) and (4) agree? ---
    print("\n  Cross-check:")
    print(f"    independent replay reached goal? {reached_goal}")
    print(f"    evaluator says solved?          {verdict}")
    if reached_goal != verdict:
        print("  ⚠️  DISAGREEMENT between independent replay and evaluator.")
        print("      The evaluator is reading the generated tokens differently")
        print("      than a straight 'apply each move and check goal' would.")
        print("      This is the kind of bug we are looking for.")
    else:
        print("  (replay and evaluator agree)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="eight_puzzle",
                    choices=["eight_puzzle", "blocks_world"])
    ap.add_argument("--size", default="medium",
                    choices=["tiny", "small", "medium", "large"])
    ap.add_argument("--wm", action="store_true")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--num-problems", type=int, default=2,
                    help="How many problems to trace if --indices is not given")
    ap.add_argument("--indices", type=int, nargs="+", default=None,
                    help="Specific test-set indices to trace")
    args = ap.parse_args()

    if args.domain == "blocks_world":
        print("ERROR: trace_one_problem.py is currently 8-puzzle-only.")
        print("       Every token-labeling and layout constant in this script")
        print("       (EIGHT_PUZZLE_MOVE_NAME, context_end=20, state_len=9, etc.)")
        print("       assumes 8-puzzle. Running with --domain blocks_world would")
        print("       silently mis-parse sequences. A dedicated BW trace tool")
        print("       would be a separate piece of work.")
        sys.exit(2)

    print("=" * 76)
    print(f"SINGLE-PROBLEM TRACE  ({args.domain}, {args.size}, "
          f"{'WM' if args.wm else 'Baseline'})")
    print("=" * 76)

    model, test_problems, test_gen, device = load_checkpoint(
        args.domain, args.size, args.wm, args.results_dir
    )

    if args.indices is not None:
        indices = args.indices
    else:
        indices = list(range(min(args.num_problems, len(test_problems))))

    for idx in indices:
        if idx >= len(test_problems):
            print(f"\n(skipping idx={idx}: only {len(test_problems)} problems)")
            continue
        trace_problem(model, test_problems[idx], test_gen, device, problem_idx=idx)


if __name__ == "__main__":
    main()
