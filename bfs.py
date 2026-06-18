import sys; sys.path.insert(0, ".")
import json
import time
from collections import Counter, deque
from pathlib import Path
import numpy as np

for mod in list(sys.modules):
    if mod.startswith("data") or mod == "trainer" or mod.startswith("trainer."):
        del sys.modules[mod]

import trainer as T
apply_move = T.apply_move_8puzzle


def state_key(s):
    return bytes(s.flatten().tolist())


def unidir_bfs(start, goal, max_depth=12):
    sk = state_key(start)
    gk = state_key(goal)
    if sk == gk:
        return 0
    visited = {sk}
    queue = deque([(start, 0)])
    while queue:
        state, depth = queue.popleft()
        if depth >= max_depth:
            continue
        for mv in ('up', 'down', 'left', 'right'):
            nxt = apply_move(state, mv)
            if nxt is None:
                continue
            k = state_key(nxt)
            if k in visited:
                continue
            if k == gk:
                return depth + 1
            visited.add(k)
            queue.append((nxt, depth + 1))
    return None


with open("cached_data/eight_puzzle_test_baseline_productivity.json") as f:
    test_problems = json.load(f)

print(f"Running unidirectional BFS (max depth 12) on {len(test_problems)} problems...")
t0 = time.time()
shortest_paths = []
for i, p in enumerate(test_problems):
    start = np.array(p["sequence"][1:10]).reshape(3, 3)
    goal = np.array(p["sequence"][11:20]).reshape(3, 3)
    sp = unidir_bfs(start, goal, max_depth=12)
    shortest_paths.append(sp)
    if (i + 1) % 50 == 0:
        print(f"  {i+1}/{len(test_problems)} done ({time.time()-t0:.1f}s)")

print(f"  Done in {time.time()-t0:.1f}s total")
print()
dist = Counter(shortest_paths)
print("Shortest-path distribution:")
for L in sorted(d for d in dist if d is not None):
    print(f"  shortest = {L:>2}: {dist[L]}")
print(f"  shortest >  12: {dist.get(None, 0)}")

truly_ood = sum(1 for sp in shortest_paths if sp is None)
in_dist = len(test_problems) - truly_ood
print()
print(f"Truly OOD (shortest > 12): {truly_ood}/{len(test_problems)}")
print(f"Within or easy (<= 12):    {in_dist}/{len(test_problems)}")

with open("/tmp/8puzzle_shortest_paths.json", "w") as f:
    json.dump(shortest_paths, f)
print()
print("Saved shortest paths to /tmp/8puzzle_shortest_paths.json")
