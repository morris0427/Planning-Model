"""
make_paper_results.py

Produce the complete results table for the paper, computed under the
corrected check_solution_correctness and using the state_source flag.

Outputs two JSON files in results/paper/:
  - paper_results.json: solve rates for the paper's headline table
  - paper_diagnostics.json: state-validity-by-step and plan-length-by-ref
                            data for the mechanism figures

Headline table structure:
  {
    "<domain>": {
      "<split>": {
        "n":           int,
        "baseline":    float,
        "wm_oracle":   float,
        "wm_model":    float,    # null if the cell is N/A
      },
      ...
    },
    ...
  }

The "splits" for each domain are:
  - "in_distribution":  full in-distribution test set, first N problems
  - "productivity":     unfiltered productivity test set, first N problems
                        (kept for transparency about the SAW-confound effect)
  - "truly_ood":        filtered to problems whose BFS shortest path
                        exceeds the training distribution

Run from the experiments/ directory:
    python3 make_paper_results.py [--n-eval N] [--bfs-depth D]

Default N is 200 for in-distribution and productivity (fast), and the
full truly-OOD subset (we already have its size determined by BFS).
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
# BFS helpers (per domain)
# ============================================================

# Blocks World BFS (small state space, unidirectional BFS fine)

_bw_helper = BlocksWorldDataset(
    difficulty_range=(3, 3), num_samples=1, use_world_model=False, seed=0,
)

def _bw_state_key(s):
    return tuple(tuple(t) for t in s)

def _bw_actions(state):
    out = []
    for i, t in enumerate(state):
        if not t:
            continue
        for d in range(4):
            if d != i:
                out.append((t[-1], d))
    return out

def bw_bfs_shortest(start, goal, max_depth=12):
    if start == goal:
        return 0
    visited = {_bw_state_key(start)}
    queue = deque([(start, 0)])
    while queue:
        state, depth = queue.popleft()
        if depth >= max_depth:
            continue
        for a in _bw_actions(state):
            nxt = _bw_helper.apply_action(state, a)
            if nxt is None:
                continue
            k = _bw_state_key(nxt)
            if k in visited:
                continue
            if nxt == goal:
                return depth + 1
            visited.add(k)
            queue.append((nxt, depth + 1))
    return None


# 8-puzzle BFS (uses trainer's apply_move_8puzzle; simple unidirectional)

def _ep_state_key(s):
    return bytes(s.flatten().tolist())

def ep_bfs_shortest(start, goal, max_depth=12):
    sk = _ep_state_key(start)
    gk = _ep_state_key(goal)
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
            k = _ep_state_key(nxt)
            if k in visited:
                continue
            if k == gk:
                return depth + 1
            visited.add(k)
            queue.append((nxt, depth + 1))
    return None


# ============================================================
# Model loading and evaluation
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


def evaluate(model, problems, test_gen, idxs, state_source=None, max_length=100):
    solved = 0
    for i in idxs:
        p = problems[i]
        kwargs = {"max_length": max_length, "return_info": True}
        if state_source is not None:
            kwargs["state_source"] = state_source
        gen, info = T.generate_solution(model, p, test_gen, torch.device("cpu"), **kwargs)
        if T.check_solution_correctness(gen, p, test_gen):
            solved += 1
    return solved


# ============================================================
# State-validity diagnostic
# ============================================================

def bw_state_is_valid(state_tokens):
    """A Blocks World 8-token state block is valid iff it decodes to a state
    where each of A,B,C,D appears exactly once across all towers.
    """
    if len(state_tokens) != 8:
        return False
    try:
        decoded = _bw_helper._decode_state(state_tokens)
    except Exception:
        return False
    blocks = []
    for tower in decoded:
        blocks.extend(tower)
    return sorted(blocks) == ['A', 'B', 'C', 'D']


def ep_state_is_valid(state_tokens):
    """An 8-puzzle 9-token state block is valid iff it is a permutation
    of {0..8}.
    """
    if len(state_tokens) != 9:
        return False
    return sorted(state_tokens) == list(range(9))


def measure_state_validity(model, problems, test_gen, idxs, domain,
                           max_length=100, max_steps_to_track=20):
    """For each of N problems, generate with state_source='model' and record,
    for each step, whether the model's emitted state block is syntactically valid.
    Returns: list of length max_steps_to_track, each a (valid_count, total_count) pair.
    """
    if domain == 'blocks_world':
        action_len = 2
        state_len = 8
        context_len = 17
        validity_check = bw_state_is_valid
        end_tok = 1
    else:
        action_len = 1
        state_len = 9
        context_len = 20
        validity_check = ep_state_is_valid
        end_tok = 14

    counts = [[0, 0] for _ in range(max_steps_to_track)]  # [valid, total] per step

    for i in idxs:
        p = problems[i]
        gen, info = T.generate_solution(
            model, p, test_gen, torch.device("cpu"),
            max_length=max_length, return_info=True, state_source="model",
        )
        # Walk through gen tokens past context, extracting state blocks
        pos = context_len + action_len  # first state block starts here
        step = 0
        while pos + state_len <= len(gen) and step < max_steps_to_track:
            state_block = gen[pos:pos + state_len]
            # Stop if we hit END inside the state block
            if end_tok in state_block:
                break
            counts[step][1] += 1
            if validity_check(state_block):
                counts[step][0] += 1
            pos += state_len + action_len
            step += 1

    return [(v, n) for (v, n) in counts]


# ============================================================
# Plan-length-by-reference-length diagnostic
# ============================================================

def measure_plan_lengths(model, problems, test_gen, idxs, domain,
                          state_source=None, max_length=100):
    """For each problem in idxs, generate a plan and record (ref_num_moves, n_actions, solved).
    Returns a list of those triples.
    """
    if domain == 'blocks_world':
        action_len = 2
        context_len = 17
        end_tok = 1
        block_toks = {2, 3, 4, 5}
        pos_toks = {6, 7, 8, 9}
        def count_actions(gen):
            n = 0
            p = context_len
            while p + 1 < len(gen):
                if gen[p] == end_tok:
                    break
                if gen[p] in block_toks and gen[p + 1] in pos_toks:
                    n += 1
                    p += action_len
                else:
                    break
            return n
    else:
        context_len = 20
        end_tok = 14
        move_toks = {10, 11, 12, 13}
        def count_actions(gen):
            n = 0
            for t in gen[context_len:]:
                if t == end_tok:
                    break
                if t in move_toks:
                    n += 1
                else:
                    break
            return n

    results = []
    for i in idxs:
        p = problems[i]
        kwargs = {"max_length": max_length, "return_info": True}
        if state_source is not None:
            kwargs["state_source"] = state_source
        gen, info = T.generate_solution(model, p, test_gen, torch.device("cpu"), **kwargs)
        n_actions = count_actions(gen)
        solved = T.check_solution_correctness(gen, p, test_gen)
        results.append({"ref_num_moves": p['num_moves'],
                        "n_actions": n_actions, "solved": bool(solved)})
    return results


# ============================================================
# Main pipeline
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-eval', type=int, default=200,
                    help='Problems to evaluate per cell (default 200)')
    ap.add_argument('--bfs-depth-bw', type=int, default=12,
                    help='Max BFS depth for Blocks World shortest-path computation')
    ap.add_argument('--bfs-depth-ep', type=int, default=12,
                    help='Max BFS depth for 8-puzzle shortest-path computation')
    ap.add_argument('--state-validity-n', type=int, default=200,
                    help='Problems to use for state-validity diagnostic')
    args = ap.parse_args()

    out_dir = Path("results/paper")
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}      # headline table
    diagnostics = {}  # mechanism data

    # ===================================================
    # Blocks World
    # ===================================================
    print("=" * 70)
    print("BLOCKS WORLD")
    print("=" * 70)

    bw_in_dist_cfg = DataPresets.blocks_world_standard()
    bw_prod_cfg = DataPresets.blocks_world_productivity()

    # Load problems
    class Shim: pass
    shim = Shim()

    shim.data = bw_in_dist_cfg
    _, bw_id_test_base = T.load_cached_data(shim, use_wm=False)
    _, bw_id_test_wm = T.load_cached_data(shim, use_wm=True)

    shim.data = bw_prod_cfg
    _, bw_prod_test_base = T.load_cached_data(shim, use_wm=False)
    _, bw_prod_test_wm = T.load_cached_data(shim, use_wm=True)

    bw_id_gen_base = DatasetFactory.create(
        domain='blocks_world', difficulty_range=bw_in_dist_cfg.test_difficulty_range,
        num_samples=len(bw_id_test_base), use_world_model=False,
    )
    bw_id_gen_base.problems = bw_id_test_base
    bw_id_gen_wm = DatasetFactory.create(
        domain='blocks_world', difficulty_range=bw_in_dist_cfg.test_difficulty_range,
        num_samples=len(bw_id_test_wm), use_world_model=True,
    )
    bw_id_gen_wm.problems = bw_id_test_wm
    bw_prod_gen_base = DatasetFactory.create(
        domain='blocks_world', difficulty_range=bw_prod_cfg.test_difficulty_range,
        num_samples=len(bw_prod_test_base), use_world_model=False,
    )
    bw_prod_gen_base.problems = bw_prod_test_base
    bw_prod_gen_wm = DatasetFactory.create(
        domain='blocks_world', difficulty_range=bw_prod_cfg.test_difficulty_range,
        num_samples=len(bw_prod_test_wm), use_world_model=True,
    )
    bw_prod_gen_wm.problems = bw_prod_test_wm

    # Load checkpoints
    print("Loading Blocks World checkpoints...")
    bw_base_id = load_model(
        Path("results/blocks_world_in_distribution_medium_base/best_model.pth"),
        use_world_model=False,
    )
    bw_wm_id = load_model(
        Path("results/blocks_world_in_distribution_medium_wm/best_model.pth"),
        use_world_model=True,
    )
    bw_base_prod = load_model(
        Path("results/blocks_world_medium_base_productivity/best_model.pth"),
        use_world_model=False,
    )
    bw_wm_prod = load_model(
        Path("results/blocks_world_medium_wm_productivity/best_model.pth"),
        use_world_model=True,
    )

    # ---- BW in-distribution ----
    print("\n[BW in-distribution]")
    n = min(args.n_eval, len(bw_id_test_base))
    idxs = list(range(n))
    t0 = time.time()
    bw_id_cell = {"n": n}
    bw_id_cell["baseline"] = evaluate(bw_base_id, bw_id_test_base, bw_id_gen_base, idxs) / n
    print(f"  baseline:   {bw_id_cell['baseline']:.3f}  [{time.time()-t0:.1f}s]")
    t0 = time.time()
    bw_id_cell["wm_oracle"] = evaluate(bw_wm_id, bw_id_test_wm, bw_id_gen_wm, idxs, state_source="oracle") / n
    print(f"  wm_oracle:  {bw_id_cell['wm_oracle']:.3f}  [{time.time()-t0:.1f}s]")
    t0 = time.time()
    bw_id_cell["wm_model"] = evaluate(bw_wm_id, bw_id_test_wm, bw_id_gen_wm, idxs, state_source="model") / n
    print(f"  wm_model:   {bw_id_cell['wm_model']:.3f}  [{time.time()-t0:.1f}s]")

    # ---- BW productivity (unfiltered) ----
    print("\n[BW productivity, unfiltered]")
    n = min(args.n_eval, len(bw_prod_test_base))
    idxs = list(range(n))
    t0 = time.time()
    bw_pu_cell = {"n": n}
    bw_pu_cell["baseline"] = evaluate(bw_base_prod, bw_prod_test_base, bw_prod_gen_base, idxs) / n
    print(f"  baseline:   {bw_pu_cell['baseline']:.3f}  [{time.time()-t0:.1f}s]")
    t0 = time.time()
    bw_pu_cell["wm_oracle"] = evaluate(bw_wm_prod, bw_prod_test_wm, bw_prod_gen_wm, idxs, state_source="oracle") / n
    print(f"  wm_oracle:  {bw_pu_cell['wm_oracle']:.3f}  [{time.time()-t0:.1f}s]")
    t0 = time.time()
    bw_pu_cell["wm_model"] = evaluate(bw_wm_prod, bw_prod_test_wm, bw_prod_gen_wm, idxs, state_source="model") / n
    print(f"  wm_model:   {bw_pu_cell['wm_model']:.3f}  [{time.time()-t0:.1f}s]")

    # ---- BW productivity (truly-OOD: BFS shortest > 4) ----
    print("\n[BW productivity, truly-OOD via BFS]")
    print(f"  computing BFS shortest paths on {len(bw_prod_test_base)} problems...")
    bw_shortest = []
    for i, p in enumerate(bw_prod_test_base):
        seq = p["sequence"]
        start = _bw_helper._decode_state(seq[1:9])
        goal = _bw_helper._decode_state(seq[9:17])
        bw_shortest.append(bw_bfs_shortest(start, goal, max_depth=args.bfs_depth_bw))
    bw_truly_ood_idxs = [i for i, sp in enumerate(bw_shortest) if sp is not None and sp >= 5]
    n_ood = len(bw_truly_ood_idxs)
    print(f"  truly-OOD subset: {n_ood} problems")

    bw_to_cell = {"n": n_ood}
    if n_ood > 0:
        t0 = time.time()
        bw_to_cell["baseline"] = evaluate(bw_base_prod, bw_prod_test_base, bw_prod_gen_base, bw_truly_ood_idxs) / n_ood
        print(f"  baseline:   {bw_to_cell['baseline']:.3f}  [{time.time()-t0:.1f}s]")
        t0 = time.time()
        bw_to_cell["wm_oracle"] = evaluate(bw_wm_prod, bw_prod_test_wm, bw_prod_gen_wm, bw_truly_ood_idxs, state_source="oracle") / n_ood
        print(f"  wm_oracle:  {bw_to_cell['wm_oracle']:.3f}  [{time.time()-t0:.1f}s]")
        t0 = time.time()
        bw_to_cell["wm_model"] = evaluate(bw_wm_prod, bw_prod_test_wm, bw_prod_gen_wm, bw_truly_ood_idxs, state_source="model") / n_ood
        print(f"  wm_model:   {bw_to_cell['wm_model']:.3f}  [{time.time()-t0:.1f}s]")

    results["blocks_world"] = {
        "in_distribution": bw_id_cell,
        "productivity": bw_pu_cell,
        "truly_ood": bw_to_cell,
    }

    # ===================================================
    # 8-puzzle
    # ===================================================
    print()
    print("=" * 70)
    print("8-PUZZLE")
    print("=" * 70)

    ep_in_dist_cfg = DataPresets.eight_puzzle_standard()
    ep_prod_cfg = DataPresets.eight_puzzle_productivity()

    shim.data = ep_in_dist_cfg
    _, ep_id_test_base = T.load_cached_data(shim, use_wm=False)
    _, ep_id_test_wm = T.load_cached_data(shim, use_wm=True)

    shim.data = ep_prod_cfg
    _, ep_prod_test_base = T.load_cached_data(shim, use_wm=False)
    _, ep_prod_test_wm = T.load_cached_data(shim, use_wm=True)

    ep_id_gen_base = DatasetFactory.create(
        domain='eight_puzzle', difficulty_range=ep_in_dist_cfg.test_difficulty_range,
        num_samples=len(ep_id_test_base), use_world_model=False,
    )
    ep_id_gen_base.problems = ep_id_test_base
    ep_id_gen_wm = DatasetFactory.create(
        domain='eight_puzzle', difficulty_range=ep_in_dist_cfg.test_difficulty_range,
        num_samples=len(ep_id_test_wm), use_world_model=True,
    )
    ep_id_gen_wm.problems = ep_id_test_wm
    ep_prod_gen_base = DatasetFactory.create(
        domain='eight_puzzle', difficulty_range=ep_prod_cfg.test_difficulty_range,
        num_samples=len(ep_prod_test_base), use_world_model=False,
    )
    ep_prod_gen_base.problems = ep_prod_test_base
    ep_prod_gen_wm = DatasetFactory.create(
        domain='eight_puzzle', difficulty_range=ep_prod_cfg.test_difficulty_range,
        num_samples=len(ep_prod_test_wm), use_world_model=True,
    )
    ep_prod_gen_wm.problems = ep_prod_test_wm

    print("Loading 8-puzzle checkpoints...")
    ep_base_id = load_model(
        Path("results/eight_puzzle_in_distribution_medium_base/best_model.pth"),
        use_world_model=False,
    )
    ep_wm_id = load_model(
        Path("results/eight_puzzle_in_distribution_medium_wm/best_model.pth"),
        use_world_model=True,
    )
    ep_base_prod = load_model(
        Path("results/eight_puzzle_medium_base_productivity/best_model.pth"),
        use_world_model=False,
    )
    ep_wm_prod = load_model(
        Path("results/eight_puzzle_medium_wm_productivity/best_model.pth"),
        use_world_model=True,
    )

    # ---- 8-puzzle in-distribution ----
    print("\n[8-puzzle in-distribution]")
    n = min(args.n_eval, len(ep_id_test_base))
    idxs = list(range(n))
    ep_id_cell = {"n": n}
    t0 = time.time()
    ep_id_cell["baseline"] = evaluate(ep_base_id, ep_id_test_base, ep_id_gen_base, idxs) / n
    print(f"  baseline:   {ep_id_cell['baseline']:.3f}  [{time.time()-t0:.1f}s]")
    t0 = time.time()
    ep_id_cell["wm_oracle"] = evaluate(ep_wm_id, ep_id_test_wm, ep_id_gen_wm, idxs, state_source="oracle") / n
    print(f"  wm_oracle:  {ep_id_cell['wm_oracle']:.3f}  [{time.time()-t0:.1f}s]")
    t0 = time.time()
    ep_id_cell["wm_model"] = evaluate(ep_wm_id, ep_id_test_wm, ep_id_gen_wm, idxs, state_source="model") / n
    print(f"  wm_model:   {ep_id_cell['wm_model']:.3f}  [{time.time()-t0:.1f}s]")

    # ---- 8-puzzle productivity (unfiltered) ----
    print("\n[8-puzzle productivity, unfiltered]")
    n = min(args.n_eval, len(ep_prod_test_base))
    idxs = list(range(n))
    ep_pu_cell = {"n": n}
    t0 = time.time()
    ep_pu_cell["baseline"] = evaluate(ep_base_prod, ep_prod_test_base, ep_prod_gen_base, idxs) / n
    print(f"  baseline:   {ep_pu_cell['baseline']:.3f}  [{time.time()-t0:.1f}s]")
    t0 = time.time()
    ep_pu_cell["wm_oracle"] = evaluate(ep_wm_prod, ep_prod_test_wm, ep_prod_gen_wm, idxs, state_source="oracle") / n
    print(f"  wm_oracle:  {ep_pu_cell['wm_oracle']:.3f}  [{time.time()-t0:.1f}s]")
    t0 = time.time()
    ep_pu_cell["wm_model"] = evaluate(ep_wm_prod, ep_prod_test_wm, ep_prod_gen_wm, idxs, state_source="model") / n
    print(f"  wm_model:   {ep_pu_cell['wm_model']:.3f}  [{time.time()-t0:.1f}s]")

    # ---- 8-puzzle productivity (truly-OOD: BFS shortest > 12) ----
    print("\n[8-puzzle productivity, truly-OOD via BFS]")
    print(f"  computing BFS shortest paths on {len(ep_prod_test_base)} problems...")
    ep_shortest = []
    for i, p in enumerate(ep_prod_test_base):
        seq = p["sequence"]
        start = np.array(seq[1:10]).reshape(3, 3)
        goal = np.array(seq[11:20]).reshape(3, 3)
        ep_shortest.append(ep_bfs_shortest(start, goal, max_depth=args.bfs_depth_ep))
    ep_truly_ood_idxs = [i for i, sp in enumerate(ep_shortest) if sp is None]
    n_ood = len(ep_truly_ood_idxs)
    print(f"  truly-OOD subset: {n_ood} problems")

    ep_to_cell = {"n": n_ood}
    if n_ood > 0:
        t0 = time.time()
        ep_to_cell["baseline"] = evaluate(ep_base_prod, ep_prod_test_base, ep_prod_gen_base, ep_truly_ood_idxs) / n_ood
        print(f"  baseline:   {ep_to_cell['baseline']:.3f}  [{time.time()-t0:.1f}s]")
        t0 = time.time()
        ep_to_cell["wm_oracle"] = evaluate(ep_wm_prod, ep_prod_test_wm, ep_prod_gen_wm, ep_truly_ood_idxs, state_source="oracle") / n_ood
        print(f"  wm_oracle:  {ep_to_cell['wm_oracle']:.3f}  [{time.time()-t0:.1f}s]")
        t0 = time.time()
        ep_to_cell["wm_model"] = evaluate(ep_wm_prod, ep_prod_test_wm, ep_prod_gen_wm, ep_truly_ood_idxs, state_source="model") / n_ood
        print(f"  wm_model:   {ep_to_cell['wm_model']:.3f}  [{time.time()-t0:.1f}s]")

    results["eight_puzzle"] = {
        "in_distribution": ep_id_cell,
        "productivity": ep_pu_cell,
        "truly_ood": ep_to_cell,
    }

    # ===================================================
    # Save headline results
    # ===================================================
    headline_path = out_dir / "paper_results.json"
    with open(headline_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Wrote headline results to {headline_path}")

    # ===================================================
    # Diagnostics: state validity by step
    # ===================================================
    print()
    print("=" * 70)
    print("DIAGNOSTICS: state validity by step")
    print("=" * 70)

    n_validity = min(args.state_validity_n, len(bw_prod_test_wm))
    validity_idxs = list(range(n_validity))

    print(f"\n[Blocks World, productivity checkpoint, n={n_validity}]")
    bw_validity = measure_state_validity(
        bw_wm_prod, bw_prod_test_wm, bw_prod_gen_wm, validity_idxs,
        domain='blocks_world',
    )
    for step, (v, n) in enumerate(bw_validity):
        if n > 0:
            print(f"  step {step+1:2d}: {v}/{n} valid ({100*v/n:.0f}%)")

    print(f"\n[8-puzzle, productivity checkpoint, n={n_validity}]")
    n_validity_ep = min(args.state_validity_n, len(ep_prod_test_wm))
    validity_idxs_ep = list(range(n_validity_ep))
    ep_validity = measure_state_validity(
        ep_wm_prod, ep_prod_test_wm, ep_prod_gen_wm, validity_idxs_ep,
        domain='eight_puzzle',
    )
    for step, (v, n) in enumerate(ep_validity):
        if n > 0:
            print(f"  step {step+1:2d}: {v}/{n} valid ({100*v/n:.0f}%)")

    diagnostics["state_validity_by_step"] = {
        "blocks_world": [{"step": i+1, "valid": v, "total": n}
                         for i, (v, n) in enumerate(bw_validity) if n > 0],
        "eight_puzzle": [{"step": i+1, "valid": v, "total": n}
                         for i, (v, n) in enumerate(ep_validity) if n > 0],
    }

    # ===================================================
    # Diagnostics: plan length by reference length
    # ===================================================
    print()
    print("=" * 70)
    print("DIAGNOSTICS: plan length by reference length")
    print("=" * 70)

    # Use the productivity (OOD) test set for plan-length diagnostics
    n_pl = min(200, len(bw_prod_test_base))
    pl_idxs = list(range(n_pl))

    bw_pl_base = measure_plan_lengths(
        bw_base_prod, bw_prod_test_base, bw_prod_gen_base, pl_idxs, 'blocks_world'
    )
    bw_pl_wm = measure_plan_lengths(
        bw_wm_prod, bw_prod_test_wm, bw_prod_gen_wm, pl_idxs, 'blocks_world',
        state_source="oracle",
    )
    ep_pl_base = measure_plan_lengths(
        ep_base_prod, ep_prod_test_base, ep_prod_gen_base, pl_idxs, 'eight_puzzle'
    )
    ep_pl_wm = measure_plan_lengths(
        ep_wm_prod, ep_prod_test_wm, ep_prod_gen_wm, pl_idxs, 'eight_puzzle',
        state_source="oracle",
    )

    diagnostics["plan_length"] = {
        "blocks_world_baseline": bw_pl_base,
        "blocks_world_wm_oracle": bw_pl_wm,
        "eight_puzzle_baseline": ep_pl_base,
        "eight_puzzle_wm_oracle": ep_pl_wm,
    }

    # Print summary
    for label, data in [
        ("BW baseline", bw_pl_base),
        ("BW WM(oracle)", bw_pl_wm),
        ("8P baseline", ep_pl_base),
        ("8P WM(oracle)", ep_pl_wm),
    ]:
        by_ref = defaultdict(list)
        for r in data:
            by_ref[r['ref_num_moves']].append(r['n_actions'])
        print(f"\n[{label}]")
        for ref in sorted(by_ref):
            vals = by_ref[ref]
            print(f"  ref={ref}: n={len(vals)} mean_actions={np.mean(vals):.1f}")

    # Save diagnostics
    diag_path = out_dir / "paper_diagnostics.json"
    with open(diag_path, "w") as f:
        json.dump(diagnostics, f, indent=2)
    print(f"\n✓ Wrote diagnostics to {diag_path}")

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)
    print(f"Results:     {headline_path}")
    print(f"Diagnostics: {diag_path}")
    print()
    print("Next: python3 make_paper_plots.py")


if __name__ == "__main__":
    main()
