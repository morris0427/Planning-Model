"""
8-puzzle parallel of eval_truly_ood.py.

For each problem in the 8-puzzle productivity test set (SAW reference
lengths 13-18), use bidirectional BFS to determine whether the TRUE
shortest path is <= 12 (within training distribution) or > 12 (truly
out-of-distribution relative to training range 10-12).

Then evaluate the baseline and WM productivity checkpoints on:
  - the truly-OOD subset (shortest > 12)
  - the within-distribution subset (shortest <= 12), for context

Run from the experiments/ directory:
    python3 eval_truly_ood_8puzzle.py

BFS is bidirectional to half the per-side depth (~6 each), keeping
memory and time manageable. Expected runtime: a few minutes for
BFS + a few minutes for inference.
"""

import sys
sys.path.insert(0, ".")

import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

for mod in list(sys.modules):
    if mod.startswith("data") or mod == "trainer" or mod.startswith("trainer."):
        del sys.modules[mod]

from config import ModelPresets, DataPresets
from data.base import DatasetFactory
import trainer as T


# Reuse trainer's apply_move_8puzzle helper (we know it exists, we read its source earlier)
apply_move = T.apply_move_8puzzle


def state_key(state):
    """Hashable key for an 8-puzzle state. 9 bytes, one per tile value (0-8)."""
    return bytes(state.flatten().tolist())


def state_from_key(state_b):
    """Inverse of state_key: turn 9-byte buffer back into a 3x3 numpy array."""
    return np.array(list(state_b), dtype=np.int64).reshape(3, 3)


def expand_layer(frontier, visited_self, visited_other, max_layer_size=200_000):
    """One BFS step: expand all states in frontier, return next-frontier and
    any state that's also in visited_other (meaning the two searches met).

    Returns (next_frontier_dict, meet_distance) where meet_distance is the
    sum of depths if the searches met, else None.
    """
    next_frontier = {}
    for state_b, depth in frontier.items():
        state = state_from_key(state_b)
        for move in ('up', 'down', 'left', 'right'):
            nxt = apply_move(state, move)
            if nxt is None:
                continue
            k = state_key(nxt)
            if k in visited_self:
                continue
            visited_self[k] = depth + 1
            if k in visited_other:
                # Meeting point found
                return next_frontier, depth + 1 + visited_other[k]
            next_frontier[k] = depth + 1
            if len(next_frontier) > max_layer_size:
                # Refuse to expand further to avoid blowing memory; treat as
                # "not within reach" for this depth
                return next_frontier, None
    return next_frontier, None


def bidirectional_bfs(start, goal, max_total_depth=12):
    """Bidirectional BFS for shortest path between start and goal.
    Returns the path length if <= max_total_depth, else None.
    """
    start_k = state_key(start)
    goal_k = state_key(goal)
    if start_k == goal_k:
        return 0

    visited_fwd = {start_k: 0}
    visited_bwd = {goal_k: 0}
    frontier_fwd = {start_k: 0}
    frontier_bwd = {goal_k: 0}

    per_side_max = max_total_depth // 2 + 1

    for layer in range(per_side_max):
        # Expand the smaller frontier first (more efficient)
        if len(frontier_fwd) <= len(frontier_bwd):
            frontier_fwd, meet = expand_layer(frontier_fwd, visited_fwd, visited_bwd)
            if meet is not None and meet <= max_total_depth:
                return meet
        else:
            frontier_bwd, meet = expand_layer(frontier_bwd, visited_bwd, visited_fwd)
            if meet is not None and meet <= max_total_depth:
                return meet
        if not frontier_fwd or not frontier_bwd:
            return None
    return None  # not found within max_total_depth


# -- Compute shortest paths for the test set ----------------------------

print("=" * 70)
print("Computing true shortest paths for 8-puzzle productivity test problems")
print("Bidirectional BFS with max total depth 12")
print("=" * 70)

with open("cached_data/eight_puzzle_test_baseline_productivity.json") as f:
    base_test = json.load(f)

n_problems = len(base_test)
print(f"Total test problems: {n_problems}")

shortest_paths = []
t0 = time.time()
for i, p in enumerate(base_test):
    seq = p["sequence"]
    start_state = np.array(seq[1:10]).reshape(3, 3)
    goal_state = np.array(seq[11:20]).reshape(3, 3)
    sp = bidirectional_bfs(start_state, goal_state, max_total_depth=12)
    shortest_paths.append(sp)
    if (i + 1) % 50 == 0:
        elapsed = time.time() - t0
        print(f"  {i+1}/{n_problems} done ({elapsed:.1f}s)")

print(f"  Done in {time.time() - t0:.1f}s total")

# Categorize
truly_ood_idxs = [i for i, sp in enumerate(shortest_paths) if sp is None]
in_dist_idxs = [i for i, sp in enumerate(shortest_paths) if sp is not None]

print()
print(f"Shortest path <= 12 (within training-or-easy):  {len(in_dist_idxs)}")
print(f"Shortest path  > 12 (truly OOD):                {len(truly_ood_idxs)}")

# Distribution of resolved shortest paths
print()
print("Distribution of resolved shortest-path lengths:")
dist = Counter(sp for sp in shortest_paths if sp is not None)
for L in sorted(dist):
    print(f"  shortest = {L:>2}: {dist[L]}")
print(f"  shortest >  12: {len(truly_ood_idxs)}")


# -- Load models -------------------------------------------------------

print()
print("=" * 70)
print("Loading 8-puzzle checkpoints...")
print("=" * 70)

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

model_baseline = load_model(
    Path("results/eight_puzzle_medium_base_productivity/best_model.pth"),
    use_world_model=False,
)
model_wm = load_model(
    Path("results/eight_puzzle_medium_wm_productivity/best_model.pth"),
    use_world_model=True,
)
print("  baseline and WM both loaded")


# -- Build test generators ---------------------------------------------

with open("cached_data/eight_puzzle_test_wm_productivity.json") as f:
    wm_test = json.load(f)

data_cfg = DataPresets.eight_puzzle_productivity()

test_gen_base = DatasetFactory.create(
    domain="eight_puzzle",
    difficulty_range=data_cfg.test_difficulty_range,
    num_samples=len(base_test),
    use_world_model=False,
)
test_gen_base.problems = base_test

test_gen_wm = DatasetFactory.create(
    domain="eight_puzzle",
    difficulty_range=data_cfg.test_difficulty_range,
    num_samples=len(wm_test),
    use_world_model=True,
)
test_gen_wm.problems = wm_test


def evaluate(model, problems, test_gen, idxs, state_source=None):
    solved = 0
    for i in idxs:
        p = problems[i]
        kwargs = {"max_length": 100, "return_info": True}
        if state_source is not None:
            kwargs["state_source"] = state_source
        gen, info = T.generate_solution(model, p, test_gen, torch.device("cpu"), **kwargs)
        if T.check_solution_correctness(gen, p, test_gen):
            solved += 1
    return solved


# -- Truly-OOD subset --------------------------------------------------

print()
print("=" * 70)
print("Solve rates on the TRULY-OOD subset (true shortest path > 12)")
print(f"Subset size: {len(truly_ood_idxs)} problems")
print("=" * 70)

if len(truly_ood_idxs) == 0:
    print("  No truly-OOD problems found. Cannot test length-generalization.")
else:
    s = evaluate(model_baseline, base_test, test_gen_base, truly_ood_idxs)
    print(f"  Baseline:                   {s}/{len(truly_ood_idxs)}  ({100*s/len(truly_ood_idxs):.1f}%)")

    s = evaluate(model_wm, wm_test, test_gen_wm, truly_ood_idxs, state_source="oracle")
    print(f"  WM (state_source=oracle):   {s}/{len(truly_ood_idxs)}  ({100*s/len(truly_ood_idxs):.1f}%)")

    s = evaluate(model_wm, wm_test, test_gen_wm, truly_ood_idxs, state_source="model")
    print(f"  WM (state_source=model):    {s}/{len(truly_ood_idxs)}  ({100*s/len(truly_ood_idxs):.1f}%)")


# -- Within-distribution subset ---------------------------------------

# Cap at 200 problems for speed; the subset is potentially much larger
in_dist_subset = in_dist_idxs[:200]

print()
print("=" * 70)
print("Solve rates on the WITHIN-TRAINING-OR-EASY subset (shortest <= 12)")
print(f"Subset size: {len(in_dist_subset)} of {len(in_dist_idxs)} (capped for speed)")
print("=" * 70)

if in_dist_subset:
    s = evaluate(model_baseline, base_test, test_gen_base, in_dist_subset)
    print(f"  Baseline:                   {s}/{len(in_dist_subset)}  ({100*s/len(in_dist_subset):.1f}%)")

    s = evaluate(model_wm, wm_test, test_gen_wm, in_dist_subset, state_source="oracle")
    print(f"  WM (state_source=oracle):   {s}/{len(in_dist_subset)}  ({100*s/len(in_dist_subset):.1f}%)")

    s = evaluate(model_wm, wm_test, test_gen_wm, in_dist_subset, state_source="model")
    print(f"  WM (state_source=model):    {s}/{len(in_dist_subset)}  ({100*s/len(in_dist_subset):.1f}%)")
