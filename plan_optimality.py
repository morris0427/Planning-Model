"""
plan_optimality.py

Measure plan optimality on solved in-distribution problems. For each
test problem the model solves, we record three quantities:

  - bfs_shortest: the true optimal plan length (via BFS).
  - saw_ref:      the SAW-generated reference plan length used during
                  training generation (recorded in `num_moves`).
  - model_plan:   the action count in the model's actual generation.

These three let us answer:

  - How non-optimal are the training demonstrations?
       saw_ref - bfs_shortest
  - How non-optimal are the model's plans?
       model_plan - bfs_shortest
  - Did the model learn to do better than its training demonstrations?
       Compare (model_plan - bfs_shortest) to (saw_ref - bfs_shortest)

Runs on the medium productivity checkpoints (since those are the ones
with the cleanest aligned-eval comparisons). Restricts analysis to
problems solved correctly by the model — non-solved problems don't have
a meaningful plan-length to discuss.

Run from the experiments/ directory:
    python3 plan_optimality.py [--n N] [--domain blocks_world|eight_puzzle|both]
"""

import sys
sys.path.insert(0, ".")

import argparse
import json
import time
from collections import Counter, defaultdict, deque
from pathlib import Path

import numpy as np
import torch

for mod in list(sys.modules):
    if mod.startswith("data") or mod == "trainer" or mod.startswith("trainer."):
        del sys.modules[mod]

from config import ModelPresets, DataPresets
from data.base import DatasetFactory
from data.blocks_world import BlocksWorldDataset
import trainer as T


# ============================================================
# BFS helpers
# ============================================================

# --- Blocks World ---

_bw_helper = BlocksWorldDataset(
    difficulty_range=(3, 3), num_samples=1,
    use_world_model=False, seed=0,
)


def bw_state_key(s):
    return tuple(tuple(t) for t in s)


def bw_actions(state):
    out = []
    for i, t in enumerate(state):
        if not t:
            continue
        for d in range(4):
            if d != i:
                out.append((t[-1], d))
    return out


def bw_bfs(start, goal, max_depth=12):
    if start == goal:
        return 0
    visited = {bw_state_key(start)}
    queue = deque([(start, 0)])
    while queue:
        state, depth = queue.popleft()
        if depth >= max_depth:
            continue
        for a in bw_actions(state):
            nxt = _bw_helper.apply_action(state, a)
            if nxt is None:
                continue
            k = bw_state_key(nxt)
            if k in visited:
                continue
            if nxt == goal:
                return depth + 1
            visited.add(k)
            queue.append((nxt, depth + 1))
    return None


# --- 8-puzzle ---

def ep_state_key(s):
    return bytes(s.flatten().tolist())


def ep_bfs(start, goal, max_depth=18):
    """8-puzzle BFS. We use max_depth=18 here (not 12) because in-distribution
    has plans of length 10-12, and we want the true shortest path which
    could be shorter.
    """
    sk = ep_state_key(start)
    gk = ep_state_key(goal)
    if sk == gk:
        return 0
    visited = {sk}
    queue = deque([(start, 0)])
    while queue:
        state, depth = queue.popleft()
        if depth >= max_depth:
            continue
        for mv in ('up', 'down', 'left', 'right'):
            nxt = T.apply_move_8puzzle(state, mv)
            if nxt is None:
                continue
            k = ep_state_key(nxt)
            if k in visited:
                continue
            if k == gk:
                return depth + 1
            visited.add(k)
            queue.append((nxt, depth + 1))
    return None


# ============================================================
# Action counting in generated trajectories
# ============================================================

def count_actions_bw(gen, use_wm):
    """Count action pairs in a Blocks World generation."""
    CONTEXT_LEN = 17
    END_TOK = 1
    BLOCK_TOKS = {2, 3, 4, 5}
    POS_TOKS = {6, 7, 8, 9}
    STRIDE = 10 if use_wm else 2

    pos = CONTEXT_LEN
    n = 0
    while pos + 1 < len(gen):
        if gen[pos] == END_TOK:
            break
        if gen[pos] in BLOCK_TOKS and gen[pos + 1] in POS_TOKS:
            n += 1
            pos += STRIDE
        else:
            break
    return n


def count_actions_ep(gen):
    """Count move tokens in an 8-puzzle generation."""
    CONTEXT_LEN = 20
    SEP_TOK = 14
    MOVE_TOKS = {10, 11, 12, 13}

    n = 0
    # For WM the moves are at positions 20, 30, 40, ... (1 action then 9 state)
    # For baseline, the moves are at positions 20, 21, 22, ...
    # Easiest robust approach: count move tokens that appear, stopping at SEP
    # but we need to skip state tokens for WM.
    #
    # Walk forward: at position p, expect either MOVE or SEP. If MOVE, count
    # it and advance past the following state-block (if any) or just by 1.
    # We can't know whether we're in baseline or WM from the gen alone, but
    # we can detect by what's at position 21: if it's a tile (0-8), we're in
    # WM (state block follows the action); if it's a move token or SEP, baseline.

    if len(gen) <= CONTEXT_LEN:
        return 0

    # Detect mode by inspecting what follows the first action
    if len(gen) > CONTEXT_LEN + 1 and gen[CONTEXT_LEN + 1] in MOVE_TOKS:
        # Baseline: moves consecutively
        for t in gen[CONTEXT_LEN:]:
            if t == SEP_TOK:
                break
            if t in MOVE_TOKS:
                n += 1
            else:
                # Unexpected token; stop
                break
    else:
        # WM mode: action + 9-token state, repeating
        pos = CONTEXT_LEN
        while pos < len(gen):
            if gen[pos] == SEP_TOK:
                break
            if gen[pos] in MOVE_TOKS:
                n += 1
                pos += 10  # 1 action + 9 state tokens
            else:
                break
    return n


# ============================================================
# Model loading
# ============================================================

def load_model(ckpt_path, use_world_model):
    state = torch.load(ckpt_path, map_location="cpu")
    max_seq, d_model = state["pos_encoder.weight"].shape
    vocab = state["embedding.weight"].shape[0]
    m_cfg = ModelPresets.medium(use_world_model=use_world_model)
    model = T.PlanningTransformer(
        vocab_size=vocab, d_model=d_model, nhead=m_cfg.n_heads,
        num_layers=m_cfg.n_layers, dim_feedforward=m_cfg.d_ff,
        max_seq_length=max_seq,
    )
    model.load_state_dict(state)
    model.eval()
    return model


# ============================================================
# Analysis driver
# ============================================================

def analyze_blocks_world(n_problems, out_dir):
    """Analyze plan optimality on Blocks World using the WM productivity
    cache as the canonical problem set (where we have the lossless
    encoding to decode start/goal).
    """
    print("=" * 70)
    print("Blocks World plan-optimality analysis")
    print("=" * 70)

    with open("cached_data/blocks_world_test_wm_productivity.json") as f:
        wm_cache = json.load(f)
    with open("cached_data/blocks_world_test_baseline_productivity.json") as f:
        base_cache = json.load(f)

    # Use first n_problems of the WM cache. We'll compute BFS, then
    # re-encode start/goal for the baseline model.
    n = min(n_problems, len(wm_cache))
    print(f"Analyzing first {n} problems")

    model_base = load_model(
        Path("results/blocks_world_medium_base_productivity/best_model.pth"),
        use_world_model=False,
    )
    model_wm = load_model(
        Path("results/blocks_world_medium_wm_productivity/best_model.pth"),
        use_world_model=True,
    )

    data_cfg = DataPresets.blocks_world_productivity()
    test_gen_base = DatasetFactory.create(
        domain="blocks_world",
        difficulty_range=data_cfg.test_difficulty_range,
        num_samples=1, use_world_model=False,
    )
    test_gen_wm = DatasetFactory.create(
        domain="blocks_world",
        difficulty_range=data_cfg.test_difficulty_range,
        num_samples=1, use_world_model=True,
    )

    # For each canonical problem, decode start/goal, compute BFS, then
    # run both models on it.
    records = []
    t0 = time.time()
    for i in range(n):
        p_wm = wm_cache[i]
        start = _bw_helper._decode_state(p_wm["sequence"][1:9])
        goal = _bw_helper._decode_state(p_wm["sequence"][9:17])
        saw_ref = p_wm["num_moves"]

        bfs_short = bw_bfs(start, goal, max_depth=12)
        if bfs_short is None:
            # Skip problems whose true shortest > 12 (productivity-region)
            continue

        # Baseline: re-encode and run
        b_ds = BlocksWorldDataset(difficulty_range=(3, 3), num_samples=1,
                                   use_world_model=False, seed=0)
        b_seq = [b_ds.vocab["START"]]
        b_seq.extend(b_ds._encode_state(start))
        b_seq.extend(b_ds._encode_state(goal))
        b_seq.append(b_ds.vocab["END"])
        b_prob = {"sequence": b_seq, "num_moves": saw_ref,
                  "start_state": start, "goal_state": goal}
        b_gen, _ = T.generate_solution(
            model_base, b_prob, test_gen_base, torch.device("cpu"),
            max_length=100, return_info=True,
        )
        b_solved = T.check_solution_correctness(b_gen, b_prob, test_gen_base)
        b_plan_len = count_actions_bw(b_gen, use_wm=False)

        # WM: re-encode and run with oracle state source
        w_ds = BlocksWorldDataset(difficulty_range=(3, 3), num_samples=1,
                                   use_world_model=True, seed=0)
        w_seq = [w_ds.vocab["START"]]
        w_seq.extend(w_ds._encode_state(start))
        w_seq.extend(w_ds._encode_state(goal))
        w_seq.append(w_ds.vocab["END"])
        w_prob = {"sequence": w_seq, "num_moves": saw_ref,
                  "start_state": start, "goal_state": goal}
        w_gen, _ = T.generate_solution(
            model_wm, w_prob, test_gen_wm, torch.device("cpu"),
            max_length=100, return_info=True, state_source="oracle",
        )
        w_solved = T.check_solution_correctness(w_gen, w_prob, test_gen_wm)
        w_plan_len = count_actions_bw(w_gen, use_wm=True)

        records.append({
            "i": i,
            "bfs_shortest": bfs_short,
            "saw_ref": saw_ref,
            "baseline_plan_len": b_plan_len,
            "baseline_solved": b_solved,
            "wm_plan_len": w_plan_len,
            "wm_solved": w_solved,
        })

        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{n} done ({time.time()-t0:.1f}s)")

    print(f"  Done. Analyzed {len(records)} problems (BFS shortest <= 12).")

    # Save records
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "plan_optimality_blocks_world.json", "w") as f:
        json.dump(records, f, indent=2)
    print(f"  Wrote {out_dir / 'plan_optimality_blocks_world.json'}")

    summarize("Blocks World", records)


def analyze_8puzzle(n_problems, out_dir):
    """Analyze plan optimality on 8-puzzle.

    For 8-puzzle, BFS is expensive at the in-distribution depths (10-12).
    We use max_depth=15 to be sure we find the true shortest for problems
    in the training range.
    """
    print()
    print("=" * 70)
    print("8-puzzle plan-optimality analysis")
    print("=" * 70)

    # Use the in-distribution test set (training range was 10-12, test
    # was 10-15 per the trainer output we saw earlier)
    with open("cached_data/eight_puzzle_test_wm.json") as f:
        wm_cache = json.load(f)
    with open("cached_data/eight_puzzle_test_baseline.json") as f:
        base_cache = json.load(f)

    n = min(n_problems, len(wm_cache))
    print(f"Analyzing first {n} problems from in-distribution test set")
    print(f"(BFS to depth 15 — may take time)")

    model_base = load_model(
        Path("results/eight_puzzle_in_distribution_medium_base/best_model.pth"),
        use_world_model=False,
    )
    model_wm = load_model(
        Path("results/eight_puzzle_in_distribution_medium_wm/best_model.pth"),
        use_world_model=True,
    )

    data_cfg = DataPresets.eight_puzzle_standard()
    test_gen_base = DatasetFactory.create(
        domain="eight_puzzle",
        difficulty_range=data_cfg.test_difficulty_range,
        num_samples=1, use_world_model=False,
    )
    test_gen_wm = DatasetFactory.create(
        domain="eight_puzzle",
        difficulty_range=data_cfg.test_difficulty_range,
        num_samples=1, use_world_model=True,
    )

    records = []
    t0 = time.time()
    for i in range(n):
        # Use whichever cache as canonical
        p_wm = wm_cache[i]
        seq = p_wm["sequence"]
        start = np.array(seq[1:10]).reshape(3, 3)
        goal = np.array(seq[11:20]).reshape(3, 3)
        saw_ref = p_wm["num_moves"]

        bfs_short = ep_bfs(start, goal, max_depth=15)
        if bfs_short is None:
            # Shortest > 15; rare for in-dist but skip if so
            continue

        # Construct baseline-format and WM-format problems on the fly.
        # The first 20 tokens are the only ones generate_solution reads.
        from data.eight_puzzle import EightPuzzleDataset
        b_ds_ep = EightPuzzleDataset(difficulty_range=(10, 12), num_samples=1,
                                      use_world_model=False, seed=0)
        w_ds_ep = EightPuzzleDataset(difficulty_range=(10, 12), num_samples=1,
                                      use_world_model=True, seed=0)

        b_seq = ([13] + start.flatten().tolist() + [b_ds_ep.vocab.get("PAD", 15)]
                 + goal.flatten().tolist() + [b_ds_ep.vocab.get("SEP", 14)])
        b_prob = {"sequence": b_seq, "num_moves": saw_ref,
                  "start_state": start.tolist(), "goal_state": goal.tolist()}
        b_gen, _ = T.generate_solution(
            model_base, b_prob, test_gen_base, torch.device("cpu"),
            max_length=100, return_info=True,
        )
        b_solved = T.check_solution_correctness(b_gen, b_prob, test_gen_base)
        b_plan_len = count_actions_ep(b_gen)

        w_seq = ([13] + start.flatten().tolist() + [w_ds_ep.vocab.get("PAD", 15)]
                 + goal.flatten().tolist() + [w_ds_ep.vocab.get("SEP", 14)])
        w_prob = {"sequence": w_seq, "num_moves": saw_ref,
                  "start_state": start.tolist(), "goal_state": goal.tolist()}
        w_gen, _ = T.generate_solution(
            model_wm, w_prob, test_gen_wm, torch.device("cpu"),
            max_length=100, return_info=True, state_source="oracle",
        )
        w_solved = T.check_solution_correctness(w_gen, w_prob, test_gen_wm)
        w_plan_len = count_actions_ep(w_gen)

        records.append({
            "i": i,
            "bfs_shortest": bfs_short,
            "saw_ref": saw_ref,
            "baseline_plan_len": b_plan_len,
            "baseline_solved": b_solved,
            "wm_plan_len": w_plan_len,
            "wm_solved": w_solved,
        })

        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{n} done ({time.time()-t0:.1f}s)")

    print(f"  Done. Analyzed {len(records)} problems.")

    with open(out_dir / "plan_optimality_eight_puzzle.json", "w") as f:
        json.dump(records, f, indent=2)
    print(f"  Wrote {out_dir / 'plan_optimality_eight_puzzle.json'}")

    summarize("8-Puzzle", records)


def summarize(domain_label, records):
    """Print summary statistics on optimality."""
    print()
    print(f"--- {domain_label} optimality summary ---")
    print(f"  n = {len(records)} problems analyzed")
    print()

    # SAW excess over BFS (training data quality)
    saw_excess = [r["saw_ref"] - r["bfs_shortest"] for r in records]

    # Filter to solved-by-baseline and solved-by-WM separately
    base_solved = [r for r in records if r["baseline_solved"]]
    wm_solved = [r for r in records if r["wm_solved"]]

    base_excess = [r["baseline_plan_len"] - r["bfs_shortest"] for r in base_solved]
    wm_excess = [r["wm_plan_len"] - r["bfs_shortest"] for r in wm_solved]

    def stats(label, data):
        if not data:
            print(f"  {label}: (no data)")
            return
        arr = np.array(data)
        print(f"  {label}:")
        print(f"    n           = {len(data)}")
        print(f"    mean excess = {arr.mean():.2f}")
        print(f"    median      = {int(np.median(arr))}")
        print(f"    min         = {arr.min()}")
        print(f"    max         = {arr.max()}")
        opt_rate = (arr == 0).mean()
        print(f"    optimal     = {opt_rate:.1%}")

    print(f"  Reference (SAW) plans vs BFS (training data quality):")
    stats("SAW - BFS", saw_excess)
    print()
    print(f"  Baseline-model plans vs BFS (on problems baseline solves):")
    stats("baseline - BFS", base_excess)
    print()
    print(f"  WM-model plans vs BFS (on problems WM solves):")
    stats("WM - BFS", wm_excess)

    # Compare: are models better than SAW?
    if base_solved and saw_excess:
        base_saw_excess = [r["saw_ref"] - r["bfs_shortest"] for r in base_solved]
        gap = np.mean(base_excess) - np.mean(base_saw_excess)
        print()
        print(f"  Baseline does {abs(gap):.2f} moves "
              f"{'WORSE' if gap > 0 else 'BETTER'} than SAW on average")
    if wm_solved and saw_excess:
        wm_saw_excess = [r["saw_ref"] - r["bfs_shortest"] for r in wm_solved]
        gap = np.mean(wm_excess) - np.mean(wm_saw_excess)
        print(f"  WM does {abs(gap):.2f} moves "
              f"{'WORSE' if gap > 0 else 'BETTER'} than SAW on average")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=500)
    ap.add_argument('--domain', choices=['blocks_world', 'eight_puzzle', 'both'],
                    default='both')
    args = ap.parse_args()

    out_dir = Path("results/paper")
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.domain in ('blocks_world', 'both'):
        analyze_blocks_world(args.n, out_dir)
    if args.domain in ('eight_puzzle', 'both'):
        analyze_8puzzle(args.n, out_dir)

    print()
    print("Done.")


if __name__ == "__main__":
    main()
