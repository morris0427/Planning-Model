"""
generate_bfs_test_sets.py

Generate test sets with BFS-verified difficulty. For each candidate
(start, goal) pair, runs BFS once and routes the pair to the appropriate
bucket:
  - in_dist  if shortest path falls in the in-distribution range
  - prod     if shortest path exceeds the in-distribution range

Outputs four files (one per domain x bucket) in cached_data/:
  - blocks_world_test_indist_bfs.json
  - blocks_world_test_productivity_bfs.json
  - eight_puzzle_test_indist_bfs.json
  - eight_puzzle_test_productivity_bfs.json

The output JSON matches the format of the existing test caches so the
eval scripts work without modification. Each problem also includes a
`bfs_shortest` field (None for productivity bucket if shortest > max
BFS depth).

Run from the experiments/ directory:
    python3 generate_bfs_test_sets.py
        [--n-per-bucket 500]
        [--domain blocks_world|eight_puzzle|both]
        [--max-iterations 100000]
"""

import sys
sys.path.insert(0, ".")

import argparse
import json
import random
import time
from collections import deque
from pathlib import Path

import numpy as np

for mod in list(sys.modules):
    if mod.startswith("data") or mod == "trainer" or mod.startswith("trainer."):
        del sys.modules[mod]

from data.blocks_world import BlocksWorldDataset
import trainer as T


# ===================================================================
# Blocks World: state sampler, BFS, encoders
# ===================================================================

_bw = BlocksWorldDataset(
    difficulty_range=(3, 3), num_samples=1,
    use_world_model=False, seed=0,
)

_bw_wm = BlocksWorldDataset(
    difficulty_range=(3, 3), num_samples=1,
    use_world_model=True, seed=0,
)

BLOCKS = ['A', 'B', 'C', 'D']
N_POS = 4


def sample_random_bw_state(rng):
    """Sample a random valid Blocks World state: each block placed on one of 4 positions,
    in some random order within each tower.
    """
    blocks_shuffled = list(BLOCKS)
    rng.shuffle(blocks_shuffled)
    towers = [[] for _ in range(N_POS)]
    for block in blocks_shuffled:
        pos = rng.randrange(N_POS)
        towers[pos].append(block)
    return towers


def _bw_state_key(s):
    return tuple(tuple(t) for t in s)


def _bw_actions(state):
    out = []
    for i, t in enumerate(state):
        if not t:
            continue
        for d in range(N_POS):
            if d != i:
                out.append((t[-1], d))
    return out


def bw_bfs(start, goal, max_depth=12):
    """BFS for shortest path between Blocks World states. Returns shortest length,
    or None if not reachable within max_depth.
    """
    if start == goal:
        return 0
    visited = {_bw_state_key(start)}
    queue = deque([(start, 0)])
    while queue:
        state, depth = queue.popleft()
        if depth >= max_depth:
            continue
        for a in _bw_actions(state):
            nxt = _bw.apply_action(state, a)
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


def make_bw_problem(start, goal, num_moves, use_wm, bfs_shortest):
    """Build a problem dict in the format the eval scripts expect.

    The 'sequence' contains the context (START + start_state + goal_state)
    plus a placeholder END. generate_solution only reads up to state_length
    (17 tokens for Blocks World) to seed inference.
    """
    ds = _bw_wm if use_wm else _bw
    tokens = [ds.vocab["START"]]
    tokens.extend(ds._encode_state(start))
    tokens.extend(ds._encode_state(goal))
    tokens.append(ds.vocab["END"])
    return {
        "sequence": tokens,
        "num_moves": num_moves,
        "start_state": start,
        "goal_state": goal,
        "bfs_shortest": bfs_shortest,
    }


# ===================================================================
# 8-puzzle: state sampler, BFS, encoders
# ===================================================================

# 8-puzzle state parity invariant: only half of the 9! permutations are
# reachable from a given goal. To ensure sampled (start, goal) pairs are
# solvable, we sample by walking from the goal a random number of steps.
# That gives us a state in the same parity class as the goal.

GOAL_8P = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 0]])


def sample_random_8p_state(rng, n_walk=200):
    """Sample a random 8-puzzle state by walking from the canonical goal
    n_walk random legal moves. Guarantees solvability and gives a roughly
    uniform sample over the reachable state space at saturated walk length.
    """
    state = GOAL_8P.copy()
    moves = ['up', 'down', 'left', 'right']
    for _ in range(n_walk):
        valid = []
        for m in moves:
            nxt = T.apply_move_8puzzle(state, m)
            if nxt is not None:
                valid.append(nxt)
        if not valid:
            break
        state = valid[rng.randrange(len(valid))]
    return state


def _8p_state_key(s):
    return bytes(s.flatten().tolist())


def ep_bfs(start, goal, max_depth=12):
    """BFS for shortest path between 8-puzzle states. Returns shortest length,
    or None if not reachable within max_depth.
    """
    sk = _8p_state_key(start)
    gk = _8p_state_key(goal)
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
            k = _8p_state_key(nxt)
            if k in visited:
                continue
            if k == gk:
                return depth + 1
            visited.add(k)
            queue.append((nxt, depth + 1))
    return None


# We need an 8-puzzle dataset instance to construct problem dicts
# matching the existing cache format. Build a tiny one to access its
# encoding helpers.
from data.eight_puzzle import EightPuzzleDataset
_ep = EightPuzzleDataset(difficulty_range=(10, 12), num_samples=1,
                          use_world_model=False, seed=0)
_ep_wm = EightPuzzleDataset(difficulty_range=(10, 12), num_samples=1,
                             use_world_model=True, seed=0)


def make_ep_problem(start, goal, num_moves, use_wm, bfs_shortest):
    """Build an 8-puzzle problem dict matching the format the eval scripts expect.

    Layout: [dummy(1), start_state(9), PAD(1), goal_state(9), placeholder, SEP(1)]
    The dummy and PAD tokens follow the existing dataset's convention.
    """
    ds = _ep_wm if use_wm else _ep
    # Use the dataset's vocab keys; they match the format in cached test data
    sequence = (
        [ds.vocab.get("DUMMY", 0)]
        + start.flatten().tolist()
        + [ds.vocab.get("PAD", 15)]
        + goal.flatten().tolist()
        + [ds.vocab.get("SEP", 14)]
    )
    return {
        "sequence": sequence,
        "num_moves": num_moves,
        "start_state": start.tolist(),
        "goal_state": goal.tolist(),
        "bfs_shortest": bfs_shortest,
    }


# ===================================================================
# Generator driver
# ===================================================================

def generate_bw(n_per_bucket, max_iters, seed=0):
    """Generate Blocks World in-distribution and productivity test sets."""
    rng = random.Random(seed)
    in_dist = []  # shortest in [1, 4]
    prod = []     # shortest in [5, 8]

    iters = 0
    t0 = time.time()
    while (len(in_dist) < n_per_bucket or len(prod) < n_per_bucket) and iters < max_iters:
        iters += 1
        start = sample_random_bw_state(rng)
        goal = sample_random_bw_state(rng)
        if start == goal:
            continue

        # BFS with max_depth=12 catches everything in the small Blocks World state space
        sp = bw_bfs(start, goal, max_depth=12)
        if sp is None:
            continue

        if 1 <= sp <= 4 and len(in_dist) < n_per_bucket:
            in_dist.append((start, goal, sp))
        elif 5 <= sp <= 8 and len(prod) < n_per_bucket:
            prod.append((start, goal, sp))

        if iters % 500 == 0:
            print(f"  [Blocks World] iters={iters} in_dist={len(in_dist)} "
                  f"prod={len(prod)} elapsed={time.time()-t0:.1f}s")

    print(f"  [Blocks World] done. iters={iters} in_dist={len(in_dist)} "
          f"prod={len(prod)} elapsed={time.time()-t0:.1f}s")
    return in_dist, prod


def generate_8p(n_per_bucket, max_iters, seed=0):
    """Generate 8-puzzle in-distribution and productivity test sets."""
    rng = random.Random(seed)
    in_dist = []  # shortest in [10, 12]
    prod = []     # shortest > 12 (sp=None at max_depth=12)

    iters = 0
    t0 = time.time()
    while (len(in_dist) < n_per_bucket or len(prod) < n_per_bucket) and iters < max_iters:
        iters += 1
        start = sample_random_8p_state(rng, n_walk=200)
        goal = sample_random_8p_state(rng, n_walk=200)
        if np.array_equal(start, goal):
            continue

        sp = ep_bfs(start, goal, max_depth=12)
        if sp is None:
            if len(prod) < n_per_bucket:
                prod.append((start, goal, None))
        elif 10 <= sp <= 12 and len(in_dist) < n_per_bucket:
            in_dist.append((start, goal, sp))

        if iters % 100 == 0:
            elapsed = time.time() - t0
            rate = iters / elapsed if elapsed > 0 else 0
            print(f"  [8-puzzle] iters={iters} in_dist={len(in_dist)} "
                  f"prod={len(prod)} elapsed={elapsed:.1f}s ({rate:.1f} iter/s)")

    print(f"  [8-puzzle] done. iters={iters} in_dist={len(in_dist)} "
          f"prod={len(prod)} elapsed={time.time()-t0:.1f}s")
    return in_dist, prod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-per-bucket', type=int, default=500,
                    help='Target problems per bucket (per domain).')
    ap.add_argument('--domain', choices=['blocks_world', 'eight_puzzle', 'both'],
                    default='both')
    ap.add_argument('--max-iterations', type=int, default=200000,
                    help='Safety cap on candidates per domain.')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    cache_dir = Path("cached_data")
    cache_dir.mkdir(exist_ok=True)

    # ---- Blocks World ----
    if args.domain in ('blocks_world', 'both'):
        print("=" * 70)
        print("Generating Blocks World BFS test sets")
        print("=" * 70)
        bw_in, bw_prod = generate_bw(args.n_per_bucket, args.max_iterations, args.seed)

        # Save in both baseline and WM encodings (different cache files, same problems)
        for use_wm, suffix in [(False, "baseline"), (True, "wm")]:
            indist_problems = [
                make_bw_problem(s, g, sp, use_wm=use_wm, bfs_shortest=sp)
                for (s, g, sp) in bw_in
            ]
            prod_problems = [
                make_bw_problem(s, g, sp, use_wm=use_wm, bfs_shortest=sp)
                for (s, g, sp) in bw_prod
            ]
            in_path = cache_dir / f"blocks_world_test_indist_bfs_{suffix}.json"
            pr_path = cache_dir / f"blocks_world_test_productivity_bfs_{suffix}.json"
            with open(in_path, "w") as f:
                json.dump(indist_problems, f)
            with open(pr_path, "w") as f:
                json.dump(prod_problems, f)
            print(f"  wrote {in_path} ({len(indist_problems)} problems)")
            print(f"  wrote {pr_path} ({len(prod_problems)} problems)")

    # ---- 8-puzzle ----
    if args.domain in ('eight_puzzle', 'both'):
        print()
        print("=" * 70)
        print("Generating 8-puzzle BFS test sets")
        print("=" * 70)
        ep_in, ep_prod = generate_8p(args.n_per_bucket, args.max_iterations, args.seed)

        for use_wm, suffix in [(False, "baseline"), (True, "wm")]:
            indist_problems = [
                make_ep_problem(s, g, sp if sp is not None else 11, use_wm=use_wm, bfs_shortest=sp)
                for (s, g, sp) in ep_in
            ]
            prod_problems = [
                make_ep_problem(s, g, sp if sp is not None else 13, use_wm=use_wm, bfs_shortest=sp)
                for (s, g, sp) in ep_prod
            ]
            in_path = cache_dir / f"eight_puzzle_test_indist_bfs_{suffix}.json"
            pr_path = cache_dir / f"eight_puzzle_test_productivity_bfs_{suffix}.json"
            with open(in_path, "w") as f:
                json.dump(indist_problems, f)
            with open(pr_path, "w") as f:
                json.dump(prod_problems, f)
            print(f"  wrote {in_path} ({len(indist_problems)} problems)")
            print(f"  wrote {pr_path} ({len(prod_problems)} problems)")

    print()
    print("Done.")


if __name__ == "__main__":
    main()
