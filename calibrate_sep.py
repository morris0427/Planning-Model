"""
SEP-emission calibration diagnostic for World Model checkpoints.

Question
--------
When the WM model predicts whether the next token is the end-of-sequence (SEP/END)
marker, is its prediction well-calibrated to the actual fact "we are at the goal"?

We measure, on the reference sequences under TEACHER FORCING:
  - P(SEP at this position | the current state in context equals the goal state)
  - P(SEP at this position | the current state in context does NOT equal the goal state)
A well-calibrated model has the first probability close to 1 and the second close to 0.
A poorly-calibrated one has them close together.

The hypothesis under test: 8-puzzle has a FIXED goal across all problems, allowing
the WM to learn SEP-emission via shortcuts that ignore the goal-comparison entirely.
Blocks World has a VARYING goal per problem, forcing the model to actually compare
current state to goal state to predict SEP. Therefore:
  - Blocks World WM: sharp calibration (high P(SEP|at-goal), low P(SEP|not-at-goal))
  - 8-Puzzle WM: dull calibration (the two probabilities are close)

Why teacher forcing, not free generation
----------------------------------------
Under free generation the WM rarely visits the actual goal state (that's why it
fails). We'd have near-zero positive samples. Teacher forcing on the reference
sequence guarantees the final state IS the goal, giving one positive sample per
problem. This tests "did the model learn to emit SEP when context says we're at
the goal," which is the proximal mechanism we care about.

Usage
-----
    python3 calibrate_sep.py --domain eight_puzzle --size medium --wm
    python3 calibrate_sep.py --domain blocks_world --size medium --wm
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from config import ModelPresets, DataPresets
from data.base import DatasetFactory  # noqa: F401  (registers datasets)
import trainer as T


# ----- Domain-specific layout describers -----
#
# We need to know, for a given encoded reference sequence:
#   (a) which token positions are SEP-decision positions (i.e. positions where the
#       NEXT token's target is either an action-block start or the END/SEP token);
#       at each, we read P(target = SEP) from the model's logits
#   (b) the encoded goal block, as a fixed-length token tuple
#   (c) the encoded current-state block immediately preceding that decision
#
# In both domains the layout after the goal block is repeating: [action_block,
# state_block]. After the FINAL state_block we expect SEP. So the SEP-decision
# position is the position whose target token is the one immediately following a
# state block.

def layout_eight_puzzle():
    """Return the layout describer for 8-puzzle.

    Sequence: [dummy=13, start(9), PAD=15, goal(9), then for each move:
               move(1) + state(9), then SEP=14].
    Indices:  0,        1..9,       10,     11..19, 20..(20+10k-1),                 last.
    """
    return {
        "name": "eight_puzzle",
        "sep_token": 14,
        "context_end": 20,  # First token AFTER the [dummy + start + PAD + goal] block
        "goal_start": 11,
        "goal_end": 20,     # exclusive
        "action_len": 1,
        "state_len": 9,
    }


def layout_blocks_world():
    """Return the layout describer for Blocks World.

    Sequence: [START=0, start(4), goal(4), then for each move:
               action(2) + state(4), then END=1].
    Indices:  0,        1..4,      5..8,   9..(9+6k-1),                              last.
    """
    return {
        "name": "blocks_world",
        "sep_token": 1,     # END
        "context_end": 17,  # START + start_state(8) + goal_state(8)
        "goal_start": 9,    # immediately after start_state(8)
        "goal_end": 17,     # exclusive; matches context_end
        "action_len": 2,
        "state_len": 8,     # uniform encoding: 4 blocks + 4 POS_k separators
    }


def sep_decision_positions(seq, layout):
    """Yield (decision_pos, current_state_tuple) pairs.

    decision_pos = the LAST token position whose target (seq[decision_pos+1]) is
    either the start of an action block (next move) or the SEP/END token.
    Equivalently: the position just after each state block (or just after the
    goal block, for the very first action).

    current_state_tuple = the state-block that immediately PRECEDES this
    decision (for the first decision, that's the goal block, which is also the
    'current state' in a teacher-forced sequence that hasn't started moving yet
    -- but we skip the first decision because there's been no action yet).
    """
    context_end = layout["context_end"]
    action_len = layout["action_len"]
    state_len = layout["state_len"]
    sep = layout["sep_token"]
    step = action_len + state_len

    # First step starts at context_end. The state block of step k starts at
    #   context_end + action_len + k*step
    # and the decision position FOLLOWING that state block is the position
    # whose target is the next token, which is at index:
    #   (context_end + action_len + k*step + state_len) - 1
    # i.e. the last token of the state block. Its target seq[that+1] is the
    # start of the next action OR the SEP token.

    k = 0
    while True:
        state_block_start = context_end + action_len + k * step
        state_block_end = state_block_start + state_len  # exclusive
        decision_pos = state_block_end - 1  # we read logits[decision_pos]
        target_pos = decision_pos + 1

        if target_pos >= len(seq):
            return  # ran past end
        # current state block sits at [state_block_start, state_block_end)
        if state_block_end > len(seq):
            return
        current_state = tuple(seq[state_block_start:state_block_end])
        target_tok = seq[target_pos]

        yield decision_pos, current_state, target_tok, (target_tok == sep)

        if target_tok == sep:
            return  # done with this sequence
        k += 1


def calibrate(model, test_problems, layout, device, max_samples=None):
    """Walk through reference sequences and accumulate P(SEP) bucketed by
    whether current_state == goal."""
    model.eval()
    sep = layout["sep_token"]
    goal_lo, goal_hi = layout["goal_start"], layout["goal_end"]

    sep_probs_at_goal = []      # P(SEP) when current_state == goal_state
    sep_probs_not_at_goal = []  # P(SEP) when current_state != goal_state
    targets_at_goal = []        # was the target actually SEP at these positions?
    targets_not_at_goal = []

    n = len(test_problems) if max_samples is None else min(len(test_problems), max_samples)

    with torch.no_grad():
        for prob in test_problems[:n]:
            seq = prob["sequence"]
            if len(seq) < layout["context_end"] + layout["action_len"] + layout["state_len"]:
                continue
            goal_state = tuple(seq[goal_lo:goal_hi])

            # Single forward pass over the full reference sequence (teacher forcing)
            x = torch.tensor([seq], dtype=torch.long, device=device)
            # Truncate if longer than model can take
            if x.shape[1] > model.max_seq_length:
                x = x[:, : model.max_seq_length]
            logits = model(x)[0]  # [seq_len, vocab]
            probs = F.softmax(logits, dim=-1)

            for decision_pos, current_state, target_tok, target_is_sep in sep_decision_positions(seq, layout):
                if decision_pos >= logits.shape[0]:
                    break
                p_sep = float(probs[decision_pos, sep].item())
                if current_state == goal_state:
                    sep_probs_at_goal.append(p_sep)
                    targets_at_goal.append(int(target_is_sep))
                else:
                    sep_probs_not_at_goal.append(p_sep)
                    targets_not_at_goal.append(int(target_is_sep))

    return {
        "n_at_goal": len(sep_probs_at_goal),
        "n_not_at_goal": len(sep_probs_not_at_goal),
        "P_sep_when_at_goal_mean": (np.mean(sep_probs_at_goal) if sep_probs_at_goal else None),
        "P_sep_when_not_at_goal_mean": (np.mean(sep_probs_not_at_goal) if sep_probs_not_at_goal else None),
        "P_sep_when_at_goal_median": (float(np.median(sep_probs_at_goal)) if sep_probs_at_goal else None),
        "P_sep_when_not_at_goal_median": (float(np.median(sep_probs_not_at_goal)) if sep_probs_not_at_goal else None),
        # Sanity: at the "at-goal" positions, what fraction of targets are actually SEP?
        # (Should be very close to 1 -- those ARE the end-of-sequence positions in the reference.)
        "target_sep_rate_at_goal": (float(np.mean(targets_at_goal)) if targets_at_goal else None),
        "target_sep_rate_not_at_goal": (float(np.mean(targets_not_at_goal)) if targets_not_at_goal else None),
    }


# --- Checkpoint loading (mirrors trace_one_problem.py) ---

def load_checkpoint(domain, size, use_wm, results_dir="results"):
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

    if domain == "eight_puzzle":
        data_cfg = DataPresets.eight_puzzle_standard()
    else:
        data_cfg = DataPresets.blocks_world_standard()

    class _Shim: pass
    shim = _Shim(); shim.data = data_cfg
    _, test_problems = T.load_cached_data(shim, use_wm)
    if test_problems is None:
        raise FileNotFoundError(f"No cached test data for {domain} (wm={use_wm}).")

    print(f"checkpoint: {ckpt}")
    print(f"model: d_model={ckpt_d_model}, max_seq_length={ckpt_max_seq}, "
          f"params={sum(p.numel() for p in model.parameters()):,}")
    print(f"test set: {len(test_problems)} problems")
    return model, test_problems, device


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="eight_puzzle",
                    choices=["eight_puzzle", "blocks_world"])
    ap.add_argument("--size", default="medium",
                    choices=["tiny", "small", "medium", "large"])
    ap.add_argument("--wm", action="store_true",
                    help="Evaluate WM checkpoint (default: baseline)")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--max-samples", type=int, default=None,
                    help="Cap how many test problems to use (default: all)")
    args = ap.parse_args()

    print("=" * 76)
    print(f"SEP CALIBRATION  ({args.domain}, {args.size}, "
          f"{'WM' if args.wm else 'Baseline'})")
    print("=" * 76)

    if args.domain == "eight_puzzle":
        layout = layout_eight_puzzle()
    else:
        layout = layout_blocks_world()

    if not args.wm:
        # Baseline layout has no per-move state blocks, so "P(SEP | current==goal)"
        # is not directly observable. We still run something useful: count the
        # P(SEP) at each post-action position regardless of state-equality, just
        # to see what baseline emits.
        print("(note: baseline has no current-state in context; the bucket")
        print(" P(SEP|at-goal) is not measurable. Use --wm for the real test.)")
        print()

    model, test_problems, device = load_checkpoint(
        args.domain, args.size, args.wm, args.results_dir
    )

    res = calibrate(model, test_problems, layout, device, max_samples=args.max_samples)

    print()
    print("=" * 76)
    print("RESULTS")
    print("=" * 76)
    print(f"  positions where current_state == goal_state: n = {res['n_at_goal']}")
    print(f"  positions where current_state != goal_state: n = {res['n_not_at_goal']}")
    print()
    if res["P_sep_when_at_goal_mean"] is None or res["P_sep_when_not_at_goal_mean"] is None:
        print("  Insufficient data in one or both buckets.")
        return

    print(f"  P(SEP | at goal):     mean = {res['P_sep_when_at_goal_mean']:.4f}   "
          f"median = {res['P_sep_when_at_goal_median']:.4f}")
    print(f"  P(SEP | not at goal): mean = {res['P_sep_when_not_at_goal_mean']:.4f}   "
          f"median = {res['P_sep_when_not_at_goal_median']:.4f}")
    ratio = res["P_sep_when_at_goal_mean"] / max(res["P_sep_when_not_at_goal_mean"], 1e-12)
    print(f"  ratio (at-goal / not-at-goal): {ratio:.2f}x")
    print()
    print(f"  sanity check -- fraction of 'at-goal' targets that ARE the SEP token: "
          f"{res['target_sep_rate_at_goal']:.3f}")
    print(f"  fraction of 'not-at-goal' targets that ARE the SEP token:           "
          f"{res['target_sep_rate_not_at_goal']:.3f}")
    print()
    print("Interpretation guide:")
    print("  Well-calibrated SEP-emission:  ratio >> 10x, P(at-goal) ~ 0.5-1.0,")
    print("                                  P(not-at-goal) ~ 0.0-0.05.")
    print("  Shortcut-fitted SEP-emission:  ratio close to 1x, both probs similar.")


if __name__ == "__main__":
    main()
