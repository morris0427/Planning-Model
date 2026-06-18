"""
Re-evaluate 8-puzzle baseline and WM productivity checkpoints, using
shortest paths computed by corrected (unidirectional) BFS.

Prereq: /tmp/8puzzle_shortest_paths.json must exist (run redo_bfs_8puzzle.py
first to produce it).

Run from the experiments/ directory:
    python3 eval_truly_ood_8puzzle_v2.py

Outputs solve rates on:
  - Truly-OOD subset (shortest > 12)
  - Within-or-easy subset (shortest <= 12)
For each, prints baseline solve rate and WM solve rate (oracle and model
state sources).
"""

import sys
sys.path.insert(0, ".")

import json
import time
from collections import Counter
from pathlib import Path

import torch

for mod in list(sys.modules):
    if mod.startswith("data") or mod == "trainer" or mod.startswith("trainer."):
        del sys.modules[mod]

from config import ModelPresets, DataPresets
from data.base import DatasetFactory
import trainer as T


# Load the shortest-path file produced by redo_bfs_8puzzle.py
sp_path = Path("/tmp/8puzzle_shortest_paths.json")
if not sp_path.exists():
    print(f"ERROR: {sp_path} not found.")
    print("       Run redo_bfs_8puzzle.py first to produce it.")
    sys.exit(1)

with open(sp_path) as f:
    shortest_paths = json.load(f)

print(f"Loaded {len(shortest_paths)} shortest paths from {sp_path}")

# Load test problems (baseline cache is the simpler format)
with open("cached_data/eight_puzzle_test_baseline_productivity.json") as f:
    base_test = json.load(f)
with open("cached_data/eight_puzzle_test_wm_productivity.json") as f:
    wm_test = json.load(f)

assert len(shortest_paths) == len(base_test) == len(wm_test), (
    f"Length mismatch: paths={len(shortest_paths)}, "
    f"base_test={len(base_test)}, wm_test={len(wm_test)}"
)

# Build index sets
truly_ood_idxs = [i for i, sp in enumerate(shortest_paths) if sp is None]
in_dist_idxs = [i for i, sp in enumerate(shortest_paths) if sp is not None]

print(f"Truly OOD (shortest > 12): {len(truly_ood_idxs)}")
print(f"Within or easy (<= 12):    {len(in_dist_idxs)}")

dist = Counter(shortest_paths)
print()
print("Shortest-path distribution (recap):")
for L in sorted(d for d in dist if d is not None):
    print(f"  shortest = {L:>2}: {dist[L]}")
print(f"  shortest >  12: {dist.get(None, 0)}")


# Load both productivity checkpoints
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


print()
print("Loading checkpoints...")
model_baseline = load_model(
    Path("results/eight_puzzle_medium_base_productivity/best_model.pth"),
    use_world_model=False,
)
model_wm = load_model(
    Path("results/eight_puzzle_medium_wm_productivity/best_model.pth"),
    use_world_model=True,
)
print("  baseline and WM loaded")


# Build test_gen objects
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
        gen, info = T.generate_solution(
            model, p, test_gen, torch.device("cpu"), **kwargs
        )
        if T.check_solution_correctness(gen, p, test_gen):
            solved += 1
    return solved


print()
print("=" * 70)
print(f"Truly-OOD subset: solve rates ({len(truly_ood_idxs)} problems)")
print("=" * 70)

if len(truly_ood_idxs) == 0:
    print("  No truly-OOD problems found.")
else:
    t0 = time.time()
    s = evaluate(model_baseline, base_test, test_gen_base, truly_ood_idxs)
    print(f"  Baseline:                   {s}/{len(truly_ood_idxs)}  "
          f"({100*s/len(truly_ood_idxs):.1f}%)  [{time.time()-t0:.1f}s]")

    t0 = time.time()
    s = evaluate(model_wm, wm_test, test_gen_wm, truly_ood_idxs, state_source="oracle")
    print(f"  WM (state_source=oracle):   {s}/{len(truly_ood_idxs)}  "
          f"({100*s/len(truly_ood_idxs):.1f}%)  [{time.time()-t0:.1f}s]")

    t0 = time.time()
    s = evaluate(model_wm, wm_test, test_gen_wm, truly_ood_idxs, state_source="model")
    print(f"  WM (state_source=model):    {s}/{len(truly_ood_idxs)}  "
          f"({100*s/len(truly_ood_idxs):.1f}%)  [{time.time()-t0:.1f}s]")


print()
print("=" * 70)
print(f"Within-or-easy subset: solve rates ({len(in_dist_idxs)} problems)")
print("=" * 70)

# Cap at 500 for runtime if needed
in_dist_subset = in_dist_idxs[:500]
print(f"(evaluating on first {len(in_dist_subset)} of {len(in_dist_idxs)})")

if not in_dist_subset:
    print("  No within-or-easy problems found.")
else:
    t0 = time.time()
    s = evaluate(model_baseline, base_test, test_gen_base, in_dist_subset)
    print(f"  Baseline:                   {s}/{len(in_dist_subset)}  "
          f"({100*s/len(in_dist_subset):.1f}%)  [{time.time()-t0:.1f}s]")

    t0 = time.time()
    s = evaluate(model_wm, wm_test, test_gen_wm, in_dist_subset, state_source="oracle")
    print(f"  WM (state_source=oracle):   {s}/{len(in_dist_subset)}  "
          f"({100*s/len(in_dist_subset):.1f}%)  [{time.time()-t0:.1f}s]")

    t0 = time.time()
    s = evaluate(model_wm, wm_test, test_gen_wm, in_dist_subset, state_source="model")
    print(f"  WM (state_source=model):    {s}/{len(in_dist_subset)}  "
          f"({100*s/len(in_dist_subset):.1f}%)  [{time.time()-t0:.1f}s]")
