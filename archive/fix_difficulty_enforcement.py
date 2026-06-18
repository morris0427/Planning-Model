#!/usr/bin/env python3
"""
Patch data/base.py to enforce the requested difficulty in generate_dataset().

WHY
---
The current generate_dataset() samples a target difficulty, calls
generate_problem(difficulty), and records whatever it gets -- including
problems whose actual num_moves is less than the requested difficulty.
For Blocks World this happens often, because the SAW backward walk gets
stuck (no unvisited next states) and returns short. At requested
difficulty 8 only 19% of generated problems actually have 8 moves; the
rest have fewer.

The consequence: cached datasets contain problems whose num_moves don't
match the configured difficulty range. Productivity test sets configured
for (5, 8) actually contain a substantial number of problems at lengths
2, 3, and 4 -- inside the training range. This silently contaminates
out-of-distribution evaluation.

FIX
---
Inside generate_dataset(), retry generate_problem() up to MAX_ATTEMPTS
times until problem['num_moves'] matches the requested difficulty. If
retries are exhausted, log a one-time warning (so failures are visible)
and record whatever was produced.

This is a single-function patch in data/base.py. It affects both
Blocks World and 8-puzzle uniformly.

USAGE
-----
    python3 fix_difficulty_enforcement.py --dry-run
    python3 fix_difficulty_enforcement.py
"""

import argparse
import shutil
import sys
import time
from pathlib import Path

TARGET_PATH = Path("data/base.py")

# Exact text of the current generate_dataset loop body, byte-for-byte
BUGGY_TEXT = '''        for _ in range(self.num_samples):
            # Sample difficulty uniformly
            difficulty = np.random.randint(min_diff, max_diff + 1)
            
            # Generate problem
            problem = self.generate_problem(difficulty)
            
            # Encode to sequence
            sequence = self.encode_sequence(problem)
            
            # Store
            self.problems.append({
                'sequence': sequence,
                'length': len(sequence),
                'num_moves': problem['num_moves'],
                'problem_idx': len(self.problems)
            })
'''

FIXED_TEXT = '''        # Bounded retry budget. Some domains (notably Blocks World) often
        # fail to reach the requested difficulty because the SAW backward
        # walk gets stuck. Without retries, the dataset silently contains
        # problems whose num_moves is below the requested difficulty,
        # which contaminates productivity tests.
        MAX_ATTEMPTS = 100
        n_short_after_retries = 0
        for _ in range(self.num_samples):
            # Sample difficulty uniformly
            difficulty = np.random.randint(min_diff, max_diff + 1)
            
            # Retry until we hit the requested difficulty exactly, or run out
            problem = None
            for _attempt in range(MAX_ATTEMPTS):
                problem = self.generate_problem(difficulty)
                if problem['num_moves'] == difficulty:
                    break
            else:
                # Retries exhausted -- record whatever was produced
                n_short_after_retries += 1
            
            # Encode to sequence
            sequence = self.encode_sequence(problem)
            
            # Store
            self.problems.append({
                'sequence': sequence,
                'length': len(sequence),
                'num_moves': problem['num_moves'],
                'problem_idx': len(self.problems)
            })
        
        if n_short_after_retries > 0:
            print(f"warning: {n_short_after_retries}/{self.num_samples} "
                  f"problems had num_moves != requested difficulty even after "
                  f"{MAX_ATTEMPTS} retries (domain may be too constrained for "
                  f"the requested difficulty)")
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not TARGET_PATH.exists():
        print(f"ERROR: {TARGET_PATH} not found. Run from the experiments/ directory.")
        sys.exit(1)

    src = TARGET_PATH.read_text()
    n_buggy = src.count(BUGGY_TEXT)
    n_fixed = src.count(FIXED_TEXT)

    if n_buggy == 0 and n_fixed > 0:
        print("Already patched: data/base.py contains the retry loop.")
        sys.exit(0)
    if n_buggy == 0:
        print("ERROR: did not find the expected buggy text in", TARGET_PATH)
        print("       Verify with: python3 -c \\")
        print("         \"import inspect; from data.base import BasePuzzleDataset; \\")
        print("          print(inspect.getsource(BasePuzzleDataset.generate_dataset))\"")
        sys.exit(2)
    if n_buggy > 1:
        print(f"ERROR: found {n_buggy} occurrences (expected 1).")
        sys.exit(3)

    print("Found 1 occurrence of the unsafe generate_dataset loop.")
    print()
    print("--- BEFORE ---")
    print(BUGGY_TEXT)
    print("--- AFTER ---")
    print(FIXED_TEXT)

    if args.dry_run:
        print("(dry run; not modifying the file)")
        sys.exit(0)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    backup = TARGET_PATH.with_suffix(f".py.bak-{timestamp}")
    shutil.copy2(TARGET_PATH, backup)
    print(f"Backed up original to {backup}")

    new_src = src.replace(BUGGY_TEXT, FIXED_TEXT)
    TARGET_PATH.write_text(new_src)
    print(f"Patched {TARGET_PATH}.")

    # Verification: import and check that the new source contains the retry logic
    print()
    print("Verifying by reimporting...")
    for mod in list(sys.modules):
        if mod.startswith("data") or mod == "data":
            del sys.modules[mod]
    sys.path.insert(0, ".")
    try:
        import inspect
        from data.base import PlanningDataset
        src_now = inspect.getsource(PlanningDataset.generate_dataset)
        if "MAX_ATTEMPTS" in src_now and "n_short_after_retries" in src_now:
            print("  ✓ generate_dataset now contains the bounded-retry logic")
        else:
            print("  ⚠️  reimport succeeded but the retry logic is not visible.")
            print("  Inspect the file manually.")
    except ImportError as e:
        # Class name may be different on user's side; surface the error
        print(f"  ⚠️  could not import the base class: {e}")
        print("     The patch is still on disk; verify manually with inspect.getsource.")
    except Exception as e:
        print(f"  ⚠️  verification raised: {e}")

    print()
    print("Next steps:")
    print("  1. Verify the fix with an empirical distribution check (same one we ran):")
    print("       python3 -c \\")
    print("         \"import sys; sys.path.insert(0, '.'); \\")
    print("          from data.blocks_world import BlocksWorldDataset; \\")
    print("          from collections import Counter; \\")
    print("          ds = BlocksWorldDataset(difficulty_range=(8,8), num_samples=100, \\")
    print("                                  use_world_model=False, seed=99); \\")
    print("          print(Counter(p['num_moves'] for p in ds.generate_dataset()))\"")
    print("     Expected: Counter({8: 100}) -- 100% at the requested difficulty.")
    print("  2. Regenerate productivity caches:")
    print("       python3 generate_productivity_data.py --domain blocks_world")
    print("  3. Re-run the productivity sweep:")
    print("       python3 run_productivity_sweep.py --domain blocks_world --sizes medium")


if __name__ == "__main__":
    main()
