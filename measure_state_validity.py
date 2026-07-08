"""
measure_state_validity.py

Quantify how well the WM auxiliary task learns global constraints
across state tokens, in both domains.

For each (domain, checkpoint) cell:

  - Run free generation (state_source='model') on N in-distribution
    test problems.
  - Extract every state block the model emits.
  - Classify each state block:
      * VALID: decodes to a state satisfying the domain's global
        constraint (Blocks World: block multiset = {A,B,C,D}; 8-Puzzle:
        token multiset = {0..8}).
      * INVALID: violates the constraint. For 8-Puzzle, we additionally
        record whether the block contains 0 or not.

  - Report distributions.

The prediction: in Blocks World, nearly all state blocks are valid
(the constraint is learned). In 8-Puzzle, few state blocks are valid,
and specifically the 0 token (blank tile) is rarely present -
quantitative evidence that the "exactly one blank" constraint is not
learned.

Run from the experiments/ directory:
    python3 measure_state_validity.py [--n 500]
"""

import sys
sys.path.insert(0, ".")

import argparse
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
from data.blocks_world import BlocksWorldDataset
import trainer as T


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


BW_CONTEXT_LEN = 17
BW_END = 1
BW_BLOCK_TOKS = {2, 3, 4, 5}
BW_POS_TOKS = {6, 7, 8, 9}

EP_CONTEXT_LEN = 20
EP_SEP = 14
EP_MOVE_TOKS = {10, 11, 12, 13}

_bw_helper = BlocksWorldDataset(
    difficulty_range=(3, 3), num_samples=1,
    use_world_model=True, seed=0,
)


def extract_bw_state_blocks(gen):
    """Walk a Blocks World WM generated sequence and extract state blocks."""
    blocks = []
    pos = BW_CONTEXT_LEN
    action_len = 2
    state_len = 8
    while pos < len(gen):
        if gen[pos] == BW_END:
            break
        if gen[pos] not in BW_BLOCK_TOKS:
            break
        if pos + 1 >= len(gen) or gen[pos + 1] not in BW_POS_TOKS:
            break
        pos += action_len
        if pos + state_len > len(gen):
            break
        blocks.append(gen[pos:pos + state_len])
        pos += state_len
    return blocks


def extract_ep_state_blocks(gen):
    """Walk an 8-Puzzle WM generated sequence and extract state blocks."""
    blocks = []
    pos = EP_CONTEXT_LEN
    action_len = 1
    state_len = 9
    while pos < len(gen):
        if gen[pos] == EP_SEP:
            break
        if gen[pos] not in EP_MOVE_TOKS:
            break
        pos += action_len
        if pos + state_len > len(gen):
            break
        blocks.append(gen[pos:pos + state_len])
        pos += state_len
    return blocks


def classify_bw_block(block):
    """Return category for a Blocks World state block."""
    if len(block) != 8:
        return "wrong_length"
    n_pos = sum(1 for t in block if t in BW_POS_TOKS)
    n_blk = sum(1 for t in block if t in BW_BLOCK_TOKS)
    if n_pos != 4 or n_blk != 4:
        return "wrong_token_counts"
    try:
        decoded = _bw_helper._decode_state(block)
        all_blocks = []
        for tower in decoded:
            all_blocks.extend(tower)
        if sorted(all_blocks) == ['A', 'B', 'C', 'D']:
            return "valid"
        else:
            return "wrong_multiset"
    except Exception:
        return "decode_error"


def classify_ep_block(block):
    """Return classification info for an 8-Puzzle state block."""
    info = {"length": len(block)}
    if len(block) != 9:
        info["category"] = "wrong_length"
        info["n_zeros"] = None
        return info
    counter = Counter(block)
    n_zeros = counter.get(0, 0)
    info["n_zeros"] = n_zeros
    out_of_range = [t for t in block if t not in set(range(9))]
    if out_of_range:
        info["category"] = "out_of_range_tokens"
        return info
    if sorted(block) == list(range(9)):
        info["category"] = "valid"
    elif n_zeros == 0:
        info["category"] = "missing_blank"
    elif n_zeros > 1:
        info["category"] = "multiple_blanks"
    else:
        info["category"] = "wrong_multiset_with_blank"
    return info


def analyze_bw(ckpt_path, test_cache_path, data_cfg, n_problems, label):
    print(f"\n[{label}]")
    if not Path(ckpt_path).exists():
        print(f"  Skipping: {ckpt_path} not found")
        return None
    model = load_model(ckpt_path, use_world_model=True)
    with open(test_cache_path) as f:
        problems = json.load(f)
    problems = problems[:n_problems]

    test_gen = DatasetFactory.create(
        domain="blocks_world",
        difficulty_range=data_cfg.test_difficulty_range,
        num_samples=len(problems),
        use_world_model=True,
    )
    test_gen.problems = problems

    counts = Counter()
    total = 0
    t0 = time.time()
    for i, p in enumerate(problems):
        gen, _ = T.generate_solution(
            model, p, test_gen, torch.device("cpu"),
            max_length=100, return_info=True, state_source="model",
        )
        blocks = extract_bw_state_blocks(gen)
        for b in blocks:
            counts[classify_bw_block(b)] += 1
            total += 1
        if (i + 1) % 100 == 0:
            print(f"    {i+1}/{len(problems)} ({time.time()-t0:.1f}s)  "
                  f"blocks so far: {total}")

    print(f"  Analyzed {total} state blocks from {len(problems)} problems")
    print(f"  Categories:")
    for cat in sorted(counts.keys()):
        pct = 100 * counts[cat] / total if total else 0
        print(f"    {cat:>25}: {counts[cat]:>5}  ({pct:.1f}%)")
    return {"total": total, "counts": dict(counts)}


def analyze_ep(ckpt_path, test_cache_path, data_cfg, n_problems, label):
    print(f"\n[{label}]")
    if not Path(ckpt_path).exists():
        print(f"  Skipping: {ckpt_path} not found")
        return None
    model = load_model(ckpt_path, use_world_model=True)
    with open(test_cache_path) as f:
        problems = json.load(f)
    problems = problems[:n_problems]

    test_gen = DatasetFactory.create(
        domain="eight_puzzle",
        difficulty_range=data_cfg.test_difficulty_range,
        num_samples=len(problems),
        use_world_model=True,
    )
    test_gen.problems = problems

    counts = Counter()
    zero_hist = Counter()  # distribution of number of 0s per block
    total = 0
    t0 = time.time()
    for i, p in enumerate(problems):
        gen, _ = T.generate_solution(
            model, p, test_gen, torch.device("cpu"),
            max_length=100, return_info=True, state_source="model",
        )
        blocks = extract_ep_state_blocks(gen)
        for b in blocks:
            info = classify_ep_block(b)
            counts[info["category"]] += 1
            if info["n_zeros"] is not None:
                zero_hist[info["n_zeros"]] += 1
            total += 1
        if (i + 1) % 20 == 0:
            print(f"    {i+1}/{len(problems)} ({time.time()-t0:.1f}s)  "
                  f"blocks so far: {total}")

    print(f"  Analyzed {total} state blocks from {len(problems)} problems")
    print(f"  Categories:")
    for cat in sorted(counts.keys()):
        pct = 100 * counts[cat] / total if total else 0
        print(f"    {cat:>28}: {counts[cat]:>5}  ({pct:.1f}%)")

    print(f"  Distribution of blanks (0-tokens) per state block:")
    for n0 in sorted(zero_hist.keys()):
        pct = 100 * zero_hist[n0] / total if total else 0
        print(f"    {n0} blanks: {zero_hist[n0]:>5}  ({pct:.1f}%)")

    print(f"  Random baseline: if 9 tokens drawn uniformly from {{0..8}},")
    print(f"    P(exactly one 0) = 9 * (1/9) * (8/9)^8 ≈ {9 * (1/9) * (8/9)**8:.3f}")
    print(f"    P(no 0)          = (8/9)^9 ≈ {(8/9)**9:.3f}")

    return {"total": total, "counts": dict(counts), "zero_hist": dict(zero_hist)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    args = ap.parse_args()

    out_dir = Path("results/paper")
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    # Blocks World
    print("=" * 70)
    print("BLOCKS WORLD state validity under free generation")
    print("=" * 70)
    for label, ckpt, cache_fn, data_cfg_fn in [
        ("BW WM in-distribution",
         "results/blocks_world_in_distribution_medium_wm/best_model.pth",
         "cached_data/blocks_world_test_wm.json",
         DataPresets.blocks_world_standard),
        ("BW WM productivity",
         "results/blocks_world_medium_wm_productivity/best_model.pth",
         "cached_data/blocks_world_test_wm_productivity.json",
         DataPresets.blocks_world_productivity),
    ]:
        r = analyze_bw(ckpt, cache_fn, data_cfg_fn(), args.n, label)
        if r is not None:
            results[label] = r

    # 8-Puzzle
    print()
    print("=" * 70)
    print("8-PUZZLE state validity under free generation")
    print("=" * 70)
    for label, ckpt, cache_fn, data_cfg_fn in [
        ("8P WM in-distribution",
         "results/eight_puzzle_in_distribution_medium_wm/best_model.pth",
         "cached_data/eight_puzzle_test_wm.json",
         DataPresets.eight_puzzle_standard),
        ("8P WM productivity",
         #"results/eight_puzzle_medium_wm_productivity/best_model.pth",
         "results/eight_puzzle_productivity_medium_wm/best_model.pth",
         "cached_data/eight_puzzle_test_wm_productivity.json",
         DataPresets.eight_puzzle_productivity),
    ]:
        r = analyze_ep(ckpt, cache_fn, data_cfg_fn(), args.n, label)
        if r is not None:
            results[label] = r

    with open(out_dir / "state_validity.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out_dir / 'state_validity.json'}")


if __name__ == "__main__":
    main()
