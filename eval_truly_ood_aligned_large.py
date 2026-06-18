"""
eval_truly_ood_aligned_large.py

Aligned truly-OOD evaluation for LARGE Blocks World checkpoints.
Mirrors eval_truly_ood_aligned.py exactly except for the checkpoint
paths and the ModelPresets.large(...) call. The alignment trick is
identical: use one cache as canonical, decode start/goal, re-encode
on-the-fly for the other format.

Run from the experiments/ directory:
    python3 eval_truly_ood_aligned_large.py
"""

import sys
sys.path.insert(0, ".")

import json
import time
from collections import deque, Counter
from pathlib import Path

import torch

for mod in list(sys.modules):
    if mod.startswith("data") or mod == "trainer" or mod.startswith("trainer."):
        del sys.modules[mod]

from config import ModelPresets, DataPresets
from data.base import DatasetFactory
from data.blocks_world import BlocksWorldDataset
import trainer as T


# -- BFS shortest path on Blocks World ---------------------------------

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


# -- Re-encoders ------------------------------------------------------

def reencode_baseline(start, goal, num_moves):
    ds_b = BlocksWorldDataset(
        difficulty_range=(3, 3), num_samples=1,
        use_world_model=False, seed=0,
    )
    tokens = [ds_b.vocab["START"]]
    tokens.extend(ds_b._encode_state(start))
    tokens.extend(ds_b._encode_state(goal))
    return {
        "sequence": tokens + [ds_b.vocab["END"]],
        "num_moves": num_moves,
        "start_state": start,
        "goal_state": goal,
    }


def reencode_wm(start, goal, num_moves):
    ds_w = BlocksWorldDataset(
        difficulty_range=(3, 3), num_samples=1,
        use_world_model=True, seed=0,
    )
    tokens = [ds_w.vocab["START"]]
    tokens.extend(ds_w._encode_state(start))
    tokens.extend(ds_w._encode_state(goal))
    return {
        "sequence": tokens + [ds_w.vocab["END"]],
        "num_moves": num_moves,
        "start_state": start,
        "goal_state": goal,
    }


# -- Load canonical problems and compute BFS --------------------------

print("=" * 70)
print("Loading canonical problem set and computing BFS shortest paths")
print("=" * 70)

with open("cached_data/blocks_world_test_wm_productivity.json") as f:
    canonical = json.load(f)

print(f"Loaded {len(canonical)} canonical problems")

problems_with_sp = []
t0 = time.time()
for i, p in enumerate(canonical):
    seq = p["sequence"]
    start = ds_helper._decode_state(seq[1:9])
    goal = ds_helper._decode_state(seq[9:17])
    sp = bfs_shortest(start, goal, max_depth=12)
    problems_with_sp.append({
        "start": start,
        "goal": goal,
        "saw_num_moves": p["num_moves"],
        "bfs_shortest": sp,
    })
print(f"BFS done in {time.time()-t0:.1f}s")

truly_ood = [p for p in problems_with_sp if p["bfs_shortest"] is not None
             and p["bfs_shortest"] >= 5]
in_dist = [p for p in problems_with_sp if p["bfs_shortest"] is not None
           and p["bfs_shortest"] <= 4]

print()
print(f"Truly OOD (BFS shortest >= 5):    {len(truly_ood)}")
print(f"Within training dist (<= 4):      {len(in_dist)}")


# -- Load LARGE checkpoints ------------------------------------------

print()
print("=" * 70)
print("Loading LARGE checkpoints")
print("=" * 70)


def load_model(ckpt_path, use_world_model):
    state = torch.load(ckpt_path, map_location="cpu")
    max_seq, d_model = state["pos_encoder.weight"].shape
    vocab = state["embedding.weight"].shape[0]
    m_cfg = ModelPresets.large(use_world_model=use_world_model)
    model = T.PlanningTransformer(
        vocab_size=vocab, d_model=d_model, nhead=m_cfg.n_heads,
        num_layers=m_cfg.n_layers, dim_feedforward=m_cfg.d_ff,
        max_seq_length=max_seq,
    )
    model.load_state_dict(state)
    model.eval()
    return model


ckpt_base = Path("results/blocks_world_productivity_large_base/best_model.pth")
ckpt_wm = Path("results/blocks_world_productivity_large_wm/best_model.pth")
if not ckpt_base.exists():
    print(f"  ERROR: {ckpt_base} not found")
    raise SystemExit(1)
if not ckpt_wm.exists():
    print(f"  ERROR: {ckpt_wm} not found")
    raise SystemExit(1)

model_baseline = load_model(ckpt_base, use_world_model=False)
model_wm = load_model(ckpt_wm, use_world_model=True)
print(f"  baseline ({ckpt_base.name}) and WM ({ckpt_wm.name}) loaded")


# -- Test gens --------------------------------------------------------

data_cfg = DataPresets.blocks_world_productivity()

test_gen_baseline = DatasetFactory.create(
    domain="blocks_world",
    difficulty_range=data_cfg.test_difficulty_range,
    num_samples=1,
    use_world_model=False,
)

test_gen_wm = DatasetFactory.create(
    domain="blocks_world",
    difficulty_range=data_cfg.test_difficulty_range,
    num_samples=1,
    use_world_model=True,
)


def evaluate_aligned(problems_meta, model, test_gen, reencode_fn, state_source=None,
                     max_length=100):
    solved = 0
    for p in problems_meta:
        problem = reencode_fn(p["start"], p["goal"], p["saw_num_moves"])
        kwargs = {"max_length": max_length, "return_info": True}
        if state_source is not None:
            kwargs["state_source"] = state_source
        gen, info = T.generate_solution(model, problem, test_gen,
                                         torch.device("cpu"), **kwargs)
        if T.check_solution_correctness(gen, problem, test_gen):
            solved += 1
    return solved


# -- Evaluate ---------------------------------------------------------

print()
print("=" * 70)
print(f"LARGE: Truly-OOD subset solve rates ({len(truly_ood)} problems, BFS shortest >= 5)")
print("=" * 70)

if len(truly_ood) > 0:
    t0 = time.time()
    s = evaluate_aligned(truly_ood, model_baseline, test_gen_baseline, reencode_baseline)
    print(f"  Baseline:                   {s}/{len(truly_ood)}  "
          f"({100*s/len(truly_ood):.1f}%)  [{time.time()-t0:.1f}s]")

    t0 = time.time()
    s = evaluate_aligned(truly_ood, model_wm, test_gen_wm, reencode_wm,
                         state_source="oracle")
    print(f"  WM (state_source=oracle):   {s}/{len(truly_ood)}  "
          f"({100*s/len(truly_ood):.1f}%)  [{time.time()-t0:.1f}s]")

    t0 = time.time()
    s = evaluate_aligned(truly_ood, model_wm, test_gen_wm, reencode_wm,
                         state_source="model")
    print(f"  WM (state_source=model):    {s}/{len(truly_ood)}  "
          f"({100*s/len(truly_ood):.1f}%)  [{time.time()-t0:.1f}s]")


# Same in-dist subset cap as the medium-model eval, for comparability
in_dist_subset = in_dist[:500]
print()
print("=" * 70)
print(f"LARGE: Within-training-dist subset (BFS shortest <= 4)")
print(f"Evaluating on first {len(in_dist_subset)} of {len(in_dist)}")
print("=" * 70)

if in_dist_subset:
    t0 = time.time()
    s = evaluate_aligned(in_dist_subset, model_baseline, test_gen_baseline, reencode_baseline)
    print(f"  Baseline:                   {s}/{len(in_dist_subset)}  "
          f"({100*s/len(in_dist_subset):.1f}%)  [{time.time()-t0:.1f}s]")

    t0 = time.time()
    s = evaluate_aligned(in_dist_subset, model_wm, test_gen_wm, reencode_wm,
                         state_source="oracle")
    print(f"  WM (state_source=oracle):   {s}/{len(in_dist_subset)}  "
          f"({100*s/len(in_dist_subset):.1f}%)  [{time.time()-t0:.1f}s]")

    t0 = time.time()
    s = evaluate_aligned(in_dist_subset, model_wm, test_gen_wm, reencode_wm,
                         state_source="model")
    print(f"  WM (state_source=model):    {s}/{len(in_dist_subset)}  "
          f"({100*s/len(in_dist_subset):.1f}%)  [{time.time()-t0:.1f}s]")


print()
print("Comparison context (medium-model aligned eval):")
print("  truly-OOD: 0/407, 0/407, 0/407 (baseline, WM-oracle, WM-model)")
print("  within-training-dist: 383/500 (77%), 396/500 (79%), 396/500 (79%)")
