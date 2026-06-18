import sys
sys.path.insert(0, ".")
import json
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

# Recompute the truly-OOD subset (same logic as eval_truly_ood.py)
ds_helper = BlocksWorldDataset(difficulty_range=(3, 3), num_samples=1,
                                use_world_model=False, seed=0)

def state_to_tuple(s): return tuple(tuple(t) for t in s)
def all_valid_actions(state):
    out = []
    for i, t in enumerate(state):
        if not t: continue
        for d in range(4):
            if d != i: out.append((t[-1], d))
    return out
def bfs_shortest(start, goal, max_depth=12):
    if start == goal: return 0
    visited = {state_to_tuple(start)}
    queue = deque([(start, 0)])
    while queue:
        state, depth = queue.popleft()
        if depth >= max_depth: continue
        for a in all_valid_actions(state):
            nxt = ds_helper.apply_action(state, a)
            if nxt is None: continue
            k = state_to_tuple(nxt)
            if k in visited: continue
            if nxt == goal: return depth + 1
            visited.add(k)
            queue.append((nxt, depth + 1))
    return None

with open("cached_data/blocks_world_test_baseline_productivity.json") as f:
    base_test = json.load(f)
with open("cached_data/blocks_world_test_wm_productivity.json") as f:
    wm_test = json.load(f)

print("Computing shortest paths and finding truly-OOD subset...")
truly_ood_idxs = []
for i, p in enumerate(base_test):
    seq = p["sequence"]
    start = ds_helper._decode_state(seq[1:9])
    goal = ds_helper._decode_state(seq[9:17])
    sp = bfs_shortest(start, goal, max_depth=12)
    if sp is not None and sp >= 5:
        truly_ood_idxs.append(i)
print(f"Truly-OOD subset size: {len(truly_ood_idxs)}")

# Load WM productivity model
state = torch.load("results/blocks_world_medium_wm_productivity/best_model.pth",
                   map_location="cpu")
max_seq, d_model = state["pos_encoder.weight"].shape
vocab = state["embedding.weight"].shape[0]
m_cfg = ModelPresets.medium(use_world_model=True)
model = T.PlanningTransformer(
    vocab_size=vocab, d_model=d_model, nhead=m_cfg.n_heads,
    num_layers=m_cfg.n_layers, dim_feedforward=m_cfg.d_ff,
    max_seq_length=max_seq,
)
model.load_state_dict(state); model.eval()

data_cfg = DataPresets.blocks_world_productivity()
test_gen = DatasetFactory.create(
    domain="blocks_world",
    difficulty_range=data_cfg.test_difficulty_range,
    num_samples=len(wm_test),
    use_world_model=True,
)
test_gen.problems = wm_test

# For each problem in the truly-OOD subset, run BOTH paths and record:
#   - whether each path solves it
#   - whether the two generated sequences are byte-identical
n_solved_oracle = 0
n_solved_model = 0
n_solved_both = 0
n_solved_oracle_only = 0
n_solved_model_only = 0
n_identical_gens = 0
n_diff_gens = 0
diff_but_both_solve = []
diff_examples = []

for idx in truly_ood_idxs:
    p = wm_test[idx]
    gen_o, info_o = T.generate_solution(
        model, p, test_gen, torch.device("cpu"),
        max_length=100, return_info=True, state_source="oracle"
    )
    gen_m, info_m = T.generate_solution(
        model, p, test_gen, torch.device("cpu"),
        max_length=100, return_info=True, state_source="model"
    )
    solved_o = T.check_solution_correctness(gen_o, p, test_gen)
    solved_m = T.check_solution_correctness(gen_m, p, test_gen)
    n_solved_oracle += solved_o
    n_solved_model += solved_m
    n_solved_both += (solved_o and solved_m)
    n_solved_oracle_only += (solved_o and not solved_m)
    n_solved_model_only += (not solved_o and solved_m)
    if gen_o == gen_m:
        n_identical_gens += 1
    else:
        n_diff_gens += 1
        if solved_o and solved_m and len(diff_but_both_solve) < 3:
            diff_but_both_solve.append((idx, gen_o, gen_m))
        if len(diff_examples) < 3:
            diff_examples.append((idx, gen_o, gen_m, solved_o, solved_m))

print()
print("=" * 70)
print("Diagnostic results on truly-OOD subset")
print("=" * 70)
print(f"  Total problems:               {len(truly_ood_idxs)}")
print(f"  Solved by oracle path:        {n_solved_oracle}")
print(f"  Solved by model path:         {n_solved_model}")
print(f"  Solved by BOTH:               {n_solved_both}")
print(f"  Solved by oracle ONLY:        {n_solved_oracle_only}")
print(f"  Solved by model ONLY:         {n_solved_model_only}")
print()
print(f"  Generations BYTE-IDENTICAL:   {n_identical_gens}")
print(f"  Generations differ:           {n_diff_gens}")

if diff_but_both_solve:
    print()
    print("Examples of problems where the two paths produced DIFFERENT generations")
    print("but BOTH still solved:")
    for idx, gen_o, gen_m in diff_but_both_solve[:1]:
        print(f"  Problem {idx}:")
        print(f"    oracle gen ({len(gen_o)} toks): {gen_o[:25]}...")
        print(f"    model  gen ({len(gen_m)} toks): {gen_m[:25]}...")

if diff_examples:
    print()
    print("Examples of problems where the two paths differ:")
    for idx, gen_o, gen_m, so, sm in diff_examples[:2]:
        print(f"  Problem {idx}: oracle_solved={so}, model_solved={sm}")
        print(f"    oracle gen ({len(gen_o)} toks): {gen_o}")
        print(f"    model  gen ({len(gen_m)} toks): {gen_m}")
