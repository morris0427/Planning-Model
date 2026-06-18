#!/usr/bin/env python3
"""
Combined patch for data/eight_puzzle.py: fixes BOTH the data generator and
the encoder, which were independently wrong in a coordinated way.

DIAGNOSIS
---------
1. Generator: generate_problem() builds the full trajectory `states` (length
   N+1, ending at the goal), then returns `states[:-1]` (length N, ending
   one move from goal). The goal state is silently dropped.

2. Encoder: encode_sequence() pairs move i with solution_states[i] (the
   pre-move state), so SEP fires after a state that's one move from goal.

Either bug alone would be visible; combined they round-tripped consistently
(N moves paired with N pre-move states) and made the model learn to stop one
move early. Free generation feeds the model post-move states via the oracle,
which is the convention it was NOT trained on -- which explains the 5%
solve rate.

FIX
---
1. Generator: `'solution_states': states[:-1]` -> `'solution_states': states`.
   Now solution_states has length N+1, ending at the goal.

2. Encoder: `solution_states[i]` -> `solution_states[i + 1]`, with
   `i < len(...)` -> `i + 1 < len(...)`. Move i now pairs with its
   post-move state, the last move's state is the goal, SEP follows the goal.

After both fixes, the encoder matches the working Blocks World convention.

USAGE
-----
    python3 fix_eight_puzzle_encoding.py --dry-run
    python3 fix_eight_puzzle_encoding.py
"""

import argparse
import shutil
import sys
import time
from pathlib import Path

TARGET_PATH = Path("data/eight_puzzle.py")

# --- Patch 1: encoder loop in encode_sequence ---
ENCODER_BUGGY = (
    "                if i < len(prob['solution_states']):\n"
    "                    seq.extend(prob['solution_states'][i].flatten().tolist())\n"
)
ENCODER_FIXED = (
    "                if i + 1 < len(prob['solution_states']):\n"
    "                    seq.extend(prob['solution_states'][i + 1].flatten().tolist())\n"
)

# --- Patch 2: generator return statement in generate_problem ---
GENERATOR_BUGGY = "                    'solution_states': states[:-1],\n"
GENERATOR_FIXED = "                    'solution_states': states,\n"


def apply_patch(src, name, buggy, fixed):
    """Returns (new_src, status_string)."""
    n_buggy = src.count(buggy)
    n_fixed = src.count(fixed)
    if n_buggy == 0 and n_fixed > 0:
        return src, f"  ⚪ {name}: already patched"
    if n_buggy == 0:
        return src, f"  ✗ {name}: expected buggy text NOT FOUND"
    if n_buggy > 1:
        return src, f"  ✗ {name}: {n_buggy} occurrences (expected 1)"
    return src.replace(buggy, fixed), f"  ✓ {name}: applied"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not TARGET_PATH.exists():
        print(f"ERROR: {TARGET_PATH} not found. Run from the experiments/ directory.")
        sys.exit(1)

    src_orig = TARGET_PATH.read_text()
    src = src_orig
    src, msg1 = apply_patch(src, "encoder loop", ENCODER_BUGGY, ENCODER_FIXED)
    src, msg2 = apply_patch(src, "generator slice", GENERATOR_BUGGY, GENERATOR_FIXED)

    print("Patch status:")
    print(msg1)
    print(msg2)

    any_missing = "NOT FOUND" in msg1 or "occurrences" in msg1 \
        or "NOT FOUND" in msg2 or "occurrences" in msg2
    any_applied = "applied" in msg1 or "applied" in msg2

    if any_missing:
        print()
        print("ERROR: at least one patch could not be located cleanly. Refusing to write.")
        sys.exit(2)

    if not any_applied:
        print()
        print("Nothing to do (both patches already applied).")
        sys.exit(0)

    print()
    print("--- patch 1 (encoder loop) ---")
    print("BEFORE:\n" + ENCODER_BUGGY)
    print("AFTER:\n" + ENCODER_FIXED)
    print("--- patch 2 (generator slice) ---")
    print("BEFORE: " + repr(GENERATOR_BUGGY))
    print("AFTER:  " + repr(GENERATOR_FIXED))
    print()

    if args.dry_run:
        print("(dry run; not modifying the file)")
        sys.exit(0)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    backup = TARGET_PATH.with_suffix(f".py.bak-{timestamp}")
    shutil.copy2(TARGET_PATH, backup)
    print(f"Backed up original to {backup}")

    TARGET_PATH.write_text(src)
    print(f"Wrote patched {TARGET_PATH}.")

    # Verify by reimporting
    print()
    print("Verifying by reimporting and generating one problem...")
    for mod in list(sys.modules):
        if mod.startswith("data.eight_puzzle") or mod == "data":
            del sys.modules[mod]
    sys.path.insert(0, ".")
    try:
        import numpy as np
        from data.eight_puzzle import EightPuzzleDataset

        ds = EightPuzzleDataset(difficulty_range=(5, 5), num_samples=1,
                                 use_world_model=True, seed=0)
        p = ds.generate_problem(difficulty=5)

        # Check 1: solution_states length is N+1
        if len(p['solution_states']) == p['num_moves'] + 1:
            print(f"  ✓ solution_states has {len(p['solution_states'])} entries "
                  f"(N+1 = {p['num_moves']+1}, correct)")
        else:
            print(f"  ✗ solution_states has {len(p['solution_states'])} entries; "
                  f"expected {p['num_moves']+1}")

        # Check 2: solution_states[-1] equals the goal
        if np.array_equal(p['solution_states'][-1], p['goal_state']):
            print(f"  ✓ solution_states[-1] == goal_state")
        else:
            print(f"  ✗ solution_states[-1] != goal_state")
            print(f"    last: {p['solution_states'][-1].flatten().tolist()}")
            print(f"    goal: {p['goal_state'].flatten().tolist()}")

        # Check 3: encoded sequence's last state-block equals the goal
        seq = ds.encode_sequence(p)
        last_state_block = seq[-10:-1]
        goal_flat = p['goal_state'].flatten().tolist()
        if last_state_block == goal_flat:
            print(f"  ✓ encoded sequence ends with goal-state + SEP")
        else:
            print(f"  ✗ encoded sequence's final state-block != goal")
            print(f"    final state-block: {last_state_block}")
            print(f"    goal:              {goal_flat}")
    except Exception as e:
        print(f"  ⚠️  verification raised: {e}")

    print()
    print("Next steps:")
    print("  1. Run the round-trip test:")
    print("       python3 test_encoding_roundtrip.py --domain eight_puzzle --skip-cache")
    print("  2. Regenerate the WM cache:")
    print("       python3 regen_eight_puzzle_wm_cache.py")
    print("  3. Re-run the test with cache verification:")
    print("       python3 test_encoding_roundtrip.py --domain eight_puzzle")
    print("  4. Retrain medium WM:")
    print("       python3 run_experiments.py --sweep model_size \\")
    print("           --domain eight_puzzle --sizes medium")


if __name__ == "__main__":
    main()
