"""
eval_truly_ood_aligned.py

Properly-aligned truly-OOD evaluation for Blocks World.

Previously, eval_truly_ood.py computed BFS shortest paths on
cached_data/blocks_world_test_baseline_productivity.json and indexed
into cached_data/blocks_world_test_wm_productivity.json. These two
caches were generated independently and contain DIFFERENT problems at
the same index, so the BFS classification did not correspond to the WM
problems being evaluated. The 60.5% headline result was based on a
random subset of the WM cache, not the BFS-filtered truly-OOD subset
we claimed.

This script fixes that by:

  1. Treating ONE cache as canonical (we use the WM cache, since it has
     the lossless state encoding that lets us decode start/goal cleanly).
  2. Decoding start/goal from each canonical problem.
  3. Computing BFS shortest path on the canonical problem.
  4. Re-encoding the SAME (start, goal) pair in baseline format for the
     baseline model's eval.
  5. Filtering to truly-OOD by BFS, then evaluating both models on the
     SAME problem set.

Run from the experiments/ directory:
    python3 eval_truly_ood_aligned.py
"""

import sys
sys.path.insert(0, ".")

import json
import time
from collections import deque
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


# -- Helper: re-encode a (start, goal, num_moves) problem in baseline format ----

def reencode_baseline(start, goal, num_moves):
    """Reconstruct a baseline-format problem from (start, goal, num_moves).

    Returns a problem dict with 'sequence' and 'num_moves' suitable for
    baseline model inference. The 'sequence' contains only the context
    (START + start_state + goal_state) plus a placeholder for the moves
    section. Since we only use the context for generation seeding, this
    is sufficient.

    NB: We can't reconstruct the original SAW solution_moves without
    re-running SAW with the same seed. But generate_solution only needs
    the context (the first state_length tokens) to seed generation. The
    rest of the sequence is what the model generates.
    """
    # Make a tiny dataset to access its _encode_state and vocab
    ds_b = BlocksWorldDataset(
        difficulty_range=(3, 3), num_samples=1,
        use_world_model=False, seed=0,
    )

    # Context layout: START(1) + start_state(8) + goal_state(8) = 17 tokens
    tokens = [ds_b.vocab["START"]]
    tokens.extend(ds_b._encode_state(start))
    tokens.extend(ds_b._encode_state(goal))

    # Pad the rest with placeholder END so the sequence has reasonable
    # length. generate_solution only reads `sequence[:state_length]` to
    # seed; the rest is generated.
    return {
        "sequence": tokens + [ds_b.vocab["END"]],
        "num_moves": num_moves,
        "start_state": start,
        "goal_state": goal,
    }


def reencode_wm(start, goal, num_moves):
    """Reconstruct a WM-format problem from (start, goal, num_moves).
    Same idea as reencode_baseline but with use_world_model=True context.
    """
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

# Use the WM cache as canonical (it has the lossless state encoding that
# we can roundtrip cleanly). Baseline encoding under uniform_8 is
# actually also lossless, but pick one and stick with it.
with open("cached_data/blocks_world_test_wm_productivity.json") as f:
    canonical = json.load(f)

print(f"Loaded {len(canonical)} canonical problems")

# Compute BFS shortest path for each, by decoding start/goal from
# the canonical (WM-encoded) sequence
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

# Filter
truly_ood = [p for p in problems_with_sp if p["bfs_shortest"] is not None
             and p["bfs_shortest"] >= 5]
in_dist = [p for p in problems_with_sp if p["bfs_shortest"] is not None
           and p["bfs_shortest"] <= 4]
unresolved = [p for p in problems_with_sp if p["bfs_shortest"] is None]

print()
print(f"Truly OOD (BFS shortest >= 5):    {len(truly_ood)}")
print(f"Within training dist (<= 4):      {len(in_dist)}")
print(f"Unresolved within BFS depth 12:   {len(unresolved)}")

# Also report distribution of (saw_num_moves, bfs_shortest)
from collections import Counter
discrepancies = Counter()
for p in problems_with_sp:
    if p["bfs_shortest"] is not None:
        discrepancies[p["saw_num_moves"] - p["bfs_shortest"]] += 1
print()
print("Distribution of (SAW ref length - true shortest path):")
for d in sorted(discrepancies):
    print(f"  diff={d:>2}: {discrepancies[d]}")


# -- Load both productivity models ------------------------------------

print()
print("=" * 70)
print("Loading checkpoints")
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
    Path("results/blocks_world_medium_base_productivity/best_model.pth"),
    use_world_model=False,
)
model_wm = load_model(
    Path("results/blocks_world_medium_wm_productivity/best_model.pth"),
    use_world_model=True,
)
print("  baseline and WM loaded")


# -- Build test_gens for both encodings -------------------------------

# We need DatasetFactory test_gen objects with .problems attribute populated
# Their use_world_model and apply_action behavior is what generate_solution
# and check_solution_correctness rely on.

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
    """For each (start, goal, num_moves) problem, re-encode in the requested
    format and run model + check.
    """
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
print(f"Truly-OOD subset: solve rates ({len(truly_ood)} problems, BFS shortest >= 5)")
print("=" * 70)

if len(truly_ood) == 0:
    print("  No truly-OOD problems found. Cannot test length-generalization.")
else:
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


# Subset of in-distribution (BFS shortest <= 4) for context
in_dist_subset = in_dist[:500]
print()
print("=" * 70)
print(f"Within-training-dist subset (BFS shortest <= 4)")
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
