"""
Re-evaluate Blocks World baseline and WM on the truly-OOD subset of
productivity test problems: those whose TRUE shortest path is >= 5.

The SAW-generated productivity test set contains many problems labeled
"length 5-8" whose true shortest path is much shorter (in our sample
~80% had shortest <= 4). Those problems are solvable within the training
distribution, so a model that produces 3-4 move plans can succeed on
them without genuine length-generalization. To test length-generalization
properly, we filter to the ~20% that truly require >= 5 moves.

Run from the experiments/ directory:
    python3 eval_truly_ood.py

Output: solve rates on the filtered subset for:
  - Baseline (productivity checkpoint)
  - WM with state_source='oracle'  (productivity checkpoint)
  - WM with state_source='model'   (productivity checkpoint)
Plus, for context, solve rates on the WITHIN-training-distribution
subset (shortest <= 4), where we expect models to perform well.
"""

import sys
sys.path.insert(0, ".")

import json
import time
from collections import Counter, deque
from pathlib import Path

import torch

for mod in list(sys.modules):
    if mod.startswith("data") or mod == "trainer" or mod.startswith("trainer."):
        del sys.modules[mod]

from config import ModelPresets, DataPresets
from data.base import DatasetFactory
from data.blocks_world import BlocksWorldDataset
import trainer as T


# -- BFS helpers ----------------------------------------------------------

ds_helper = BlocksWorldDataset(
    difficulty_range=(3, 3), num_samples=1,
    use_world_model=False, seed=0,
)


def state_to_tuple(state):
    return tuple(tuple(t) for t in state)


def all_valid_actions(state):
    actions = []
    for pos_idx, tower in enumerate(state):
        if not tower:
            continue
        block = tower[-1]
        for dest in range(4):
            if dest != pos_idx:
                actions.append((block, dest))
    return actions


def bfs_shortest(start, goal, max_depth=12):
    if start == goal:
        return 0
    visited = {state_to_tuple(start)}
    queue = deque([(start, 0)])
    while queue:
        state, depth = queue.popleft()
        if depth >= max_depth:
            continue
        for action in all_valid_actions(state):
            nxt = ds_helper.apply_action(state, action)
            if nxt is None:
                continue
            nxt_key = state_to_tuple(nxt)
            if nxt_key in visited:
                continue
            if nxt == goal:
                return depth + 1
            visited.add(nxt_key)
            queue.append((nxt, depth + 1))
    return None


# -- Compute shortest paths for the full test set -------------------------

print("=" * 70)
print("Computing true shortest paths for all productivity test problems...")
print("=" * 70)

# Both baseline and WM productivity test caches contain the same problems
# (modulo encoding). Use the baseline cache to compute shortest paths.
with open("cached_data/blocks_world_test_baseline_productivity.json") as f:
    base_test_problems = json.load(f)

print(f"Total test problems: {len(base_test_problems)}")
t0 = time.time()
shortest_paths = []
for i, p in enumerate(base_test_problems):
    seq = p["sequence"]
    start = ds_helper._decode_state(seq[1:9])
    goal = ds_helper._decode_state(seq[9:17])
    sp = bfs_shortest(start, goal, max_depth=12)
    shortest_paths.append(sp)
    if (i + 1) % 500 == 0:
        elapsed = time.time() - t0
        print(f"  {i+1}/{len(base_test_problems)} done ({elapsed:.1f}s)")

# Index sets for filtering
truly_ood_idxs = [i for i, sp in enumerate(shortest_paths) if sp is not None and sp >= 5]
in_dist_idxs = [i for i, sp in enumerate(shortest_paths) if sp is not None and sp <= 4]
not_found_idxs = [i for i, sp in enumerate(shortest_paths) if sp is None]

print()
print(f"Truly OOD (shortest >= 5):       {len(truly_ood_idxs)}")
print(f"Within training dist (<= 4):     {len(in_dist_idxs)}")
print(f"Not resolved within BFS depth:   {len(not_found_idxs)}")

dist = Counter(shortest_paths)
print()
print("Shortest-path distribution across the full 2000-problem test set:")
for L in sorted(d for d in dist if d is not None):
    print(f"  shortest = {L}: {dist[L]}")
if None in dist:
    print(f"  shortest = not_found: {dist[None]}")


# -- Load models ----------------------------------------------------------

def load_model(ckpt_path, use_world_model):
    state = torch.load(ckpt_path, map_location="cpu")
    max_seq, d_model = state["pos_encoder.weight"].shape
    vocab = state["embedding.weight"].shape[0]
    m_cfg = ModelPresets.medium(use_world_model=use_world_model)
    model = T.PlanningTransformer(
        vocab_size=vocab,
        d_model=d_model,
        nhead=m_cfg.n_heads,
        num_layers=m_cfg.n_layers,
        dim_feedforward=m_cfg.d_ff,
        max_seq_length=max_seq,
    )
    model.load_state_dict(state)
    model.eval()
    return model


print()
print("=" * 70)
print("Loading checkpoints...")
print("=" * 70)
model_baseline = load_model(
    Path("results/blocks_world_medium_base_productivity/best_model.pth"),
    use_world_model=False,
)
model_wm = load_model(
    Path("results/blocks_world_medium_wm_productivity/best_model.pth"),
    use_world_model=True,
)
print("  baseline and WM both loaded")


# -- Build the test generators -------------------------------------------

# The baseline and WM cache files have the SAME (start, goal) pairs but
# different encodings (WM cache has state blocks; baseline cache does not).
# We need a test_gen for each.

with open("cached_data/blocks_world_test_wm_productivity.json") as f:
    wm_test_problems = json.load(f)

data_cfg = DataPresets.blocks_world_productivity()
test_gen_base = DatasetFactory.create(
    domain="blocks_world",
    difficulty_range=data_cfg.test_difficulty_range,
    num_samples=len(base_test_problems),
    use_world_model=False,
)
test_gen_base.problems = base_test_problems

test_gen_wm = DatasetFactory.create(
    domain="blocks_world",
    difficulty_range=data_cfg.test_difficulty_range,
    num_samples=len(wm_test_problems),
    use_world_model=True,
)
test_gen_wm.problems = wm_test_problems


def evaluate_on_idxs(model, problems, test_gen, idxs, state_source=None):
    """Run generate_solution + semantic check on the problems at the given indices."""
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


# -- Evaluate on truly-OOD subset -----------------------------------------

print()
print("=" * 70)
print("Solve rates on the TRULY-OOD subset (true shortest path >= 5)")
print(f"Subset size: {len(truly_ood_idxs)} problems")
print("=" * 70)

s_base = evaluate_on_idxs(
    model_baseline, base_test_problems, test_gen_base, truly_ood_idxs
)
print(f"  Baseline:                   {s_base}/{len(truly_ood_idxs)}  ({100*s_base/len(truly_ood_idxs):.1f}%)")

s_wm_oracle = evaluate_on_idxs(
    model_wm, wm_test_problems, test_gen_wm, truly_ood_idxs, state_source="oracle"
)
print(f"  WM (state_source=oracle):   {s_wm_oracle}/{len(truly_ood_idxs)}  ({100*s_wm_oracle/len(truly_ood_idxs):.1f}%)")

s_wm_model = evaluate_on_idxs(
    model_wm, wm_test_problems, test_gen_wm, truly_ood_idxs, state_source="model"
)
print(f"  WM (state_source=model):    {s_wm_model}/{len(truly_ood_idxs)}  ({100*s_wm_model/len(truly_ood_idxs):.1f}%)")


# -- Evaluate on within-training-distribution subset for comparison ------

print()
print("=" * 70)
print("Solve rates on the WITHIN-training-distribution subset")
print("(true shortest path <= 4 -- expected to be easy)")
print(f"Subset size: {len(in_dist_idxs)} problems")
print("=" * 70)

# Use first 500 to keep wall-clock manageable; the subset is ~1600 problems
in_dist_subset = in_dist_idxs[:500]
print(f"  (evaluating on first 500 of {len(in_dist_idxs)} for speed)")

s_base = evaluate_on_idxs(
    model_baseline, base_test_problems, test_gen_base, in_dist_subset
)
print(f"  Baseline:                   {s_base}/{len(in_dist_subset)}  ({100*s_base/len(in_dist_subset):.1f}%)")

s_wm_oracle = evaluate_on_idxs(
    model_wm, wm_test_problems, test_gen_wm, in_dist_subset, state_source="oracle"
)
print(f"  WM (state_source=oracle):   {s_wm_oracle}/{len(in_dist_subset)}  ({100*s_wm_oracle/len(in_dist_subset):.1f}%)")

s_wm_model = evaluate_on_idxs(
    model_wm, wm_test_problems, test_gen_wm, in_dist_subset, state_source="model"
)
print(f"  WM (state_source=model):    {s_wm_model}/{len(in_dist_subset)}  ({100*s_wm_model/len(in_dist_subset):.1f}%)")
