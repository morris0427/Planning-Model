"""
Compute true shortest-path lengths for problems in the Blocks World
productivity test set, via BFS. Compare to the SAW-reference lengths
(num_moves) to see how often the reference is non-minimal.

This diagnoses how confounded the "productivity" test actually is. If
most problems labeled "length 7" have true shortest paths of 3-4, then
a model that produces ~3-move plans can succeed on most of the test set
without actually length-generalizing.

Run from the experiments/ directory:
    python3 check_saw_optimality.py

Outputs:
  - Per-reference-length, the distribution of true shortest-path lengths.
  - Headline summary: fraction of OOD test problems whose true shortest
    path is within the training distribution (<= 4).
"""

import sys
sys.path.insert(0, ".")

import json
from collections import Counter, deque

from data.blocks_world import BlocksWorldDataset

# Load the productivity test set
with open("cached_data/blocks_world_test_baseline_productivity.json") as f:
    test_problems = json.load(f)

# Need a dataset instance for apply_action and _decode_state
ds = BlocksWorldDataset(
    difficulty_range=(3, 3), num_samples=1,
    use_world_model=False, seed=0,
)


def state_to_tuple(state):
    """Hashable representation of a state."""
    return tuple(tuple(t) for t in state)


def all_valid_actions(state):
    """Enumerate all (block, dest_pos) pairs that are legal from this state.

    A block can be moved iff it is on top of a tower. It can be moved
    to any position that is not its current position.
    """
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
    """BFS for the shortest plan that transforms `start` into `goal`.
    Returns the integer length, or None if no path is found within max_depth.
    """
    if start == goal:
        return 0
    visited = {state_to_tuple(start)}
    queue = deque([(start, 0)])
    while queue:
        state, depth = queue.popleft()
        if depth >= max_depth:
            continue
        for action in all_valid_actions(state):
            nxt = ds.apply_action(state, action)
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


# For each problem, compute the true shortest path length
# Limit to first 500 for speed; 500 is plenty for stable statistics
N_SAMPLES = min(500, len(test_problems))
print(f"Running BFS on {N_SAMPLES} problems from the productivity test set...")
print()

shortest_by_ref = []
for i, p in enumerate(test_problems[:N_SAMPLES]):
    seq = p["sequence"]
    start = ds._decode_state(seq[1:9])
    goal = ds._decode_state(seq[9:17])
    ref = p["num_moves"]
    sp = bfs_shortest(start, goal, max_depth=12)
    shortest_by_ref.append((ref, sp))
    if (i + 1) % 100 == 0:
        print(f"  ...{i+1}/{N_SAMPLES} done")

print()
print("=" * 70)
print("True shortest-path length distribution, by SAW reference length")
print("=" * 70)
print()

# Header
print("ref_len  count   true shortest path length")
header_lengths = list(range(0, 9))
header_str = "                     " + "  ".join(f"{L:>3}" for L in header_lengths) + "    n/f"
print(header_str)

by_ref = {}
for ref, sp in shortest_by_ref:
    by_ref.setdefault(ref, []).append(sp)

for ref in sorted(by_ref):
    counts = Counter(by_ref[ref])
    cells = [counts.get(L, 0) for L in header_lengths]
    not_found = counts.get(None, 0)
    cell_str = "  ".join(f"{c:>3}" for c in cells)
    print(f"  {ref}     {len(by_ref[ref]):4d}   {cell_str}    {not_found:>3}")

# Headline numbers
n_resolved = sum(1 for _, sp in shortest_by_ref if sp is not None)
n_in_training_dist = sum(1 for _, sp in shortest_by_ref if sp is not None and sp <= 4)
n_truly_ood = sum(1 for _, sp in shortest_by_ref if sp is not None and sp >= 5)

print()
print("=" * 70)
print("Headline numbers")
print("=" * 70)
print(f"Problems with shortest path resolved (within BFS depth):  {n_resolved}/{N_SAMPLES}")
print(f"Problems whose TRUE shortest path is <= 4 (in training):  {n_in_training_dist}/{N_SAMPLES}  ({100*n_in_training_dist/N_SAMPLES:.1f}%)")
print(f"Problems whose TRUE shortest path is >= 5 (truly OOD):    {n_truly_ood}/{N_SAMPLES}  ({100*n_truly_ood/N_SAMPLES:.1f}%)")

# Mean discrepancy
discrepancies = []
for ref, sp in shortest_by_ref:
    if sp is not None:
        discrepancies.append(ref - sp)
if discrepancies:
    print()
    print(f"Mean (ref - shortest): {sum(discrepancies)/len(discrepancies):.2f}")
    print(f"Median (ref - shortest): {sorted(discrepancies)[len(discrepancies)//2]}")
    print(f"Distribution of ref-shortest:")
    discrepancy_counts = Counter(discrepancies)
    for d in sorted(discrepancy_counts):
        print(f"  ref - shortest = {d}: {discrepancy_counts[d]}")
