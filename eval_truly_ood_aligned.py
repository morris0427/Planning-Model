"""
eval_truly_ood_aligned.py

Canonical aligned truly-OOD evaluation. Replaces:
  - eval_truly_ood.py                    (misaligned, superseded)
  - eval_truly_ood_8puzzle.py            (misaligned, superseded)
  - eval_truly_ood_8puzzle_v2.py         (misaligned, superseded)
  - eval_truly_ood_aligned_large.py      (subsumed by --size large)
  - eval_truly_ood_aligned_8puzzle.py    (subsumed by --domain eight_puzzle)

The aligned procedure: use one cache as canonical, decode (start, goal)
from each problem, re-encode on the fly for both baseline and WM
formats, and evaluate both models on the SAME underlying problems. This
is necessary because the baseline and WM caches were generated
independently and contain different problems at the same index.

For each problem, BFS shortest path determines whether it belongs to
the within-training-distribution subset (shortest <= training-max) or
the truly-OOD subset (shortest > training-max).

Usage
-----
    python3 eval_truly_ood_aligned.py --domain blocks_world --size medium
    python3 eval_truly_ood_aligned.py --domain blocks_world --size large
    python3 eval_truly_ood_aligned.py --domain eight_puzzle --size medium

Optional:
    --n-cap 500          Max problems per subset (default 500)
    --bfs-depth 12       Max BFS depth (default 12)
    --state-sources oracle,model    Which WM state sources to run
    --output PATH        Write results JSON to PATH (default:
                         results/paper/aligned_eval_{domain}_{size}.json)
"""

import argparse
import json
import sys
import time
from collections import Counter, deque
from pathlib import Path

sys.path.insert(0, ".")

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
# Domain-specific configuration
# ============================================================

# Blocks World

_bw_helper = BlocksWorldDataset(
    difficulty_range=(3, 3), num_samples=1,
    use_world_model=False, seed=0,
)


def bw_bfs_shortest(start, goal, max_depth=12):
    if start == goal:
        return 0
    def key(s): return tuple(tuple(t) for t in s)
    def actions(state):
        out = []
        for i, t in enumerate(state):
            if not t: continue
            for d in range(4):
                if d != i:
                    out.append((t[-1], d))
        return out
    visited = {key(start)}
    queue = deque([(start, 0)])
    while queue:
        state, depth = queue.popleft()
        if depth >= max_depth: continue
        for a in actions(state):
            nxt = _bw_helper.apply_action(state, a)
            if nxt is None: continue
            k = key(nxt)
            if k in visited: continue
            if nxt == goal: return depth + 1
            visited.add(k)
            queue.append((nxt, depth + 1))
    return None


def bw_decode_from_seq(seq):
    """Extract start and goal from a Blocks World test cache sequence."""
    return _bw_helper._decode_state(seq[1:9]), _bw_helper._decode_state(seq[9:17])


def bw_reencode(start, goal, num_moves, use_wm):
    """Reconstruct a problem dict for baseline or WM format."""
    ds = BlocksWorldDataset(difficulty_range=(3, 3), num_samples=1,
                             use_world_model=use_wm, seed=0)
    tokens = [ds.vocab["START"]]
    tokens.extend(ds._encode_state(start))
    tokens.extend(ds._encode_state(goal))
    tokens.append(ds.vocab["END"])
    return {
        "sequence": tokens,
        "num_moves": num_moves,
        "start_state": start,
        "goal_state": goal,
    }


# 8-Puzzle

def ep_bfs_shortest(start, goal, max_depth=12):
    def key(s): return bytes(s.flatten().tolist())
    sk, gk = key(start), key(goal)
    if sk == gk: return 0
    visited = {sk}
    queue = deque([(start, 0)])
    while queue:
        state, depth = queue.popleft()
        if depth >= max_depth: continue
        for mv in ('up', 'down', 'left', 'right'):
            nxt = T.apply_move_8puzzle(state, mv)
            if nxt is None: continue
            k = key(nxt)
            if k in visited: continue
            if k == gk: return depth + 1
            visited.add(k)
            queue.append((nxt, depth + 1))
    return None


def ep_decode_from_seq(seq):
    """Extract start and goal from an 8-Puzzle test cache sequence."""
    start = np.array(seq[1:10]).reshape(3, 3)
    goal = np.array(seq[11:20]).reshape(3, 3)
    return start, goal


def ep_reencode(start, goal, num_moves, use_wm):
    """Reconstruct a problem dict. Position 0 is a fixed anchor token (13)."""
    from data.eight_puzzle import EightPuzzleDataset
    ds = EightPuzzleDataset(difficulty_range=(10, 12), num_samples=1,
                             use_world_model=use_wm, seed=0)
    seq = (
        [13]                                # anchor at position 0
        + start.flatten().tolist()          # 9 tokens: start state
        + [ds.vocab["PAD"]]                 # separator
        + goal.flatten().tolist()           # 9 tokens: goal state
        + [ds.vocab["SEP"]]                 # trailing SEP
    )
    return {
        "sequence": seq,
        "num_moves": num_moves,
        "start_state": start.tolist(),
        "goal_state": goal.tolist(),
    }


# ============================================================
# Domain adapter table
# ============================================================

DOMAIN_CONFIG = {
    "blocks_world": {
        "training_max_shortest": 4,
        "decode_from_seq": bw_decode_from_seq,
        "bfs_shortest": bw_bfs_shortest,
        "reencode": bw_reencode,
        "canonical_cache": "cached_data/blocks_world_test_wm_productivity.json",
        "data_preset": lambda: DataPresets.blocks_world_productivity(),
    },
    "eight_puzzle": {
        "training_max_shortest": 12,
        "decode_from_seq": ep_decode_from_seq,
        "bfs_shortest": ep_bfs_shortest,
        "reencode": ep_reencode,
        "canonical_cache": "cached_data/eight_puzzle_test_wm_productivity.json",
        "data_preset": lambda: DataPresets.eight_puzzle_productivity(),
    },
}


# ============================================================
# Checkpoint path derivation
# ============================================================

def checkpoint_path(domain, size, use_wm):
    """Derive the expected checkpoint path from (domain, size, use_wm).

    Convention used in run_experiments.py output:
      - productivity: results/{domain}_{size}_{base|wm}_productivity/best_model.pth
    """
    kind = "wm" if use_wm else "base"
    return Path(f"results/{domain}_productivity_{size}_{kind}/best_model.pth")

def load_model(ckpt_path, size, use_wm):
    state = torch.load(ckpt_path, map_location="cpu")
    max_seq, d_model = state["pos_encoder.weight"].shape
    vocab = state["embedding.weight"].shape[0]
    if size == "medium":
        m_cfg = ModelPresets.medium(use_world_model=use_wm)
    elif size == "large":
        m_cfg = ModelPresets.large(use_world_model=use_wm)
    else:
        raise ValueError(f"unknown size {size!r}")
    model = T.PlanningTransformer(
        vocab_size=vocab, d_model=d_model, nhead=m_cfg.n_heads,
        num_layers=m_cfg.n_layers, dim_feedforward=m_cfg.d_ff,
        max_seq_length=max_seq,
    )
    model.load_state_dict(state)
    model.eval()
    return model


# ============================================================
# Evaluation
# ============================================================

def evaluate_cell(problems_meta, model, test_gen, reencode_fn, use_wm,
                  state_source=None, max_length=100):
    solved = 0
    for p in problems_meta:
        problem = reencode_fn(p["start"], p["goal"], p["saw_num_moves"], use_wm)
        kwargs = {"max_length": max_length, "return_info": True}
        if state_source is not None:
            kwargs["state_source"] = state_source
        gen, _ = T.generate_solution(model, problem, test_gen,
                                       torch.device("cpu"), **kwargs)
        if T.check_solution_correctness(gen, problem, test_gen):
            solved += 1
    return solved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True,
                    choices=list(DOMAIN_CONFIG.keys()))
    ap.add_argument("--size", required=True, choices=["medium", "large"])
    ap.add_argument("--n-cap", type=int, default=500,
                    help="Max problems per subset (default 500)")
    ap.add_argument("--bfs-depth", type=int, default=12)
    ap.add_argument("--state-sources", default="oracle,model",
                    help="Comma-separated WM state sources to test.")
    ap.add_argument("--output", default=None,
                    help="Path for results JSON.")
    args = ap.parse_args()

    domain = args.domain
    size = args.size
    cfg = DOMAIN_CONFIG[domain]
    training_max = cfg["training_max_shortest"]

    state_sources = [s.strip() for s in args.state_sources.split(",") if s.strip()]

    # -- Load canonical problems and compute BFS shortest ------------
    print("=" * 70)
    print(f"Aligned truly-OOD eval | domain={domain} size={size}")
    print("=" * 70)
    print(f"Canonical cache: {cfg['canonical_cache']}")
    with open(cfg["canonical_cache"]) as f:
        canonical = json.load(f)
    print(f"Loaded {len(canonical)} canonical problems")

    print(f"Computing BFS shortest paths (max_depth={args.bfs_depth})...")
    problems_meta = []
    t0 = time.time()
    for p in canonical:
        start, goal = cfg["decode_from_seq"](p["sequence"])
        sp = cfg["bfs_shortest"](start, goal, max_depth=args.bfs_depth)
        problems_meta.append({
            "start": start,
            "goal": goal,
            "saw_num_moves": p["num_moves"],
            "bfs_shortest": sp,
        })
    print(f"BFS done in {time.time()-t0:.1f}s")

    in_dist = [p for p in problems_meta
               if p["bfs_shortest"] is not None and p["bfs_shortest"] <= training_max]
    truly_ood = [p for p in problems_meta
                 if p["bfs_shortest"] is None or p["bfs_shortest"] > training_max]

    print(f"Within training dist (BFS shortest <= {training_max}): {len(in_dist)}")
    print(f"Truly-OOD (BFS shortest > {training_max}):              {len(truly_ood)}")

    # Cap
    in_dist = in_dist[:args.n_cap]
    truly_ood = truly_ood[:args.n_cap]
    print(f"After cap n={args.n_cap}: in_dist={len(in_dist)} truly_ood={len(truly_ood)}")

    # BFS discrepancy stats (for the paper's methodology section)
    disc = Counter()
    for p in problems_meta:
        if p["bfs_shortest"] is not None:
            disc[p["saw_num_moves"] - p["bfs_shortest"]] += 1
    print()
    print("SAW ref length - BFS shortest (measures SAW non-optimality):")
    for d in sorted(disc):
        print(f"  diff={d:>2}: {disc[d]}")

    # -- Load checkpoints ------------------------------------------
    print()
    print("=" * 70)
    print("Loading checkpoints")
    print("=" * 70)

    ckpt_base = checkpoint_path(domain, size, use_wm=False)
    ckpt_wm = checkpoint_path(domain, size, use_wm=True)
    if not ckpt_base.exists():
        print(f"ERROR: baseline checkpoint {ckpt_base} not found")
        sys.exit(1)
    if not ckpt_wm.exists():
        print(f"ERROR: WM checkpoint {ckpt_wm} not found")
        sys.exit(1)

    model_baseline = load_model(ckpt_base, size, use_wm=False)
    model_wm = load_model(ckpt_wm, size, use_wm=True)
    print(f"  baseline: {ckpt_base}")
    print(f"  WM:       {ckpt_wm}")

    # -- Test gens for on-the-fly re-encoding ----------------------
    data_cfg = cfg["data_preset"]()
    test_gen_base = DatasetFactory.create(
        domain=domain,
        difficulty_range=data_cfg.test_difficulty_range,
        num_samples=1, use_world_model=False,
    )
    test_gen_wm = DatasetFactory.create(
        domain=domain,
        difficulty_range=data_cfg.test_difficulty_range,
        num_samples=1, use_world_model=True,
    )

    # -- Evaluate --------------------------------------------------
    results = {
        "domain": domain,
        "size": size,
        "training_max_shortest": training_max,
        "canonical_cache": cfg["canonical_cache"],
        "cells": {},
    }

    def run_and_print(subset_label, subset):
        print()
        print("=" * 70)
        print(f"{subset_label}: {len(subset)} problems")
        print("=" * 70)
        cell_results = {"n": len(subset)}
        if not subset:
            return cell_results

        # Baseline
        t0 = time.time()
        s = evaluate_cell(subset, model_baseline, test_gen_base,
                         cfg["reencode"], use_wm=False)
        cell_results["baseline"] = s / len(subset)
        print(f"  Baseline:                   {s}/{len(subset)}  "
              f"({100*s/len(subset):.1f}%)  [{time.time()-t0:.1f}s]")

        # WM with each state source
        for ss in state_sources:
            t0 = time.time()
            s = evaluate_cell(subset, model_wm, test_gen_wm,
                             cfg["reencode"], use_wm=True, state_source=ss)
            cell_results[f"wm_{ss}"] = s / len(subset)
            print(f"  WM (state_source={ss:<6}):  {s}/{len(subset)}  "
                  f"({100*s/len(subset):.1f}%)  [{time.time()-t0:.1f}s]")

        return cell_results

    results["cells"]["truly_ood"] = run_and_print("Truly-OOD", truly_ood)
    results["cells"]["within_training_dist"] = run_and_print(
        "Within training distribution", in_dist)

    # -- Save results ------------------------------------------------
    if args.output:
        out_path = Path(args.output)
    else:
        out_dir = Path("results/paper")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"aligned_eval_{domain}_{size}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print()
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
