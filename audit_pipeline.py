#!/usr/bin/env python3
"""
Audit the current state of the data pipeline.

Prints the live source (via inspect.getsource) of every critical function
across the data pipeline. The point is to see what's actually running on
this machine, since the patch scripts and my-sandbox source files have
been the cause of several rounds of confusion this session.

Run this from the experiments/ directory. It prints one big report.

For each function it also tries to flag, based on simple substring checks,
whether the expected patches have landed. These flags are heuristic --
they can be wrong if the surrounding code has been rewritten. The actual
authoritative information is the printed source itself; the flags are a
quick summary.
"""

import sys
import inspect
from pathlib import Path

sys.path.insert(0, ".")


def banner(title):
    line = "=" * 76
    print()
    print(line)
    print(title)
    print(line)


def show(obj, expected_substrings=None, unexpected_substrings=None):
    """Print source and run simple substring checks."""
    try:
        src = inspect.getsource(obj)
    except Exception as e:
        print(f"  ⚠️  could not get source: {e}")
        return
    src_file = "unknown"
    try:
        src_file = inspect.getsourcefile(obj)
    except Exception:
        pass
    print(f"file: {src_file}")
    print()
    print(src)
    if expected_substrings or unexpected_substrings:
        print("--- patch checks ---")
        if expected_substrings:
            for s in expected_substrings:
                hit = s in src
                tick = "✓" if hit else "✗"
                print(f"  {tick} should contain: {s!r}  -- {'found' if hit else 'NOT FOUND'}")
        if unexpected_substrings:
            for s in unexpected_substrings:
                hit = s in src
                tick = "✗" if hit else "✓"
                print(f"  {tick} should NOT contain: {s!r}  -- {'found (BAD)' if hit else 'absent (good)'}")


def main():
    # ----------------------------------------------------------------
    # 8-puzzle
    # ----------------------------------------------------------------
    banner("EightPuzzleDataset.generate_problem")
    try:
        from data.eight_puzzle import EightPuzzleDataset
        show(
            EightPuzzleDataset.generate_problem,
            expected_substrings=[
                "'solution_states': states,",  # post-fix: no [:-1] slice
            ],
            unexpected_substrings=[
                "states[:-1]",
                "solve_puzzle_bfs",  # we should be using SAW, not BFS
            ],
        )
    except Exception as e:
        print(f"could not load EightPuzzleDataset: {e}")

    banner("EightPuzzleDataset.encode_sequence")
    try:
        show(
            EightPuzzleDataset.encode_sequence,
            expected_substrings=[
                "prob['solution_states'][i + 1]",   # post-fix: post-move state
                "i + 1 < len(prob['solution_states'])",
            ],
            unexpected_substrings=[
                "prob['solution_states'][i].flatten",  # the original buggy indexing
            ],
        )
    except Exception:
        pass

    # ----------------------------------------------------------------
    # Blocks World
    # ----------------------------------------------------------------
    banner("BlocksWorldDataset._encode_state")
    try:
        from data.blocks_world import BlocksWorldDataset
        show(
            BlocksWorldDataset._encode_state,
            expected_substrings=[
                "reversed(tower)",  # post-fix: top-to-bottom within tower
                "POS_",             # POS_k separators emitted
            ],
            unexpected_substrings=[
                "tower[-1]",         # old lossy "top block only" pattern
            ],
        )
    except Exception as e:
        print(f"could not load BlocksWorldDataset: {e}")

    banner("BlocksWorldDataset._decode_state  (added by encoding patch)")
    try:
        if hasattr(BlocksWorldDataset, "_decode_state"):
            show(BlocksWorldDataset._decode_state)
        else:
            print("  ✗ _decode_state is NOT present on BlocksWorldDataset")
            print("    The encoding patch may not have landed.")
    except Exception:
        pass

    banner("BlocksWorldDataset.encode_sequence")
    try:
        show(BlocksWorldDataset.encode_sequence)
    except Exception:
        pass

    banner("BlocksWorldDataset.generate_problem")
    try:
        show(BlocksWorldDataset.generate_problem)
    except Exception:
        pass

    banner("BlocksWorldDataset._estimate_state_tokens")
    try:
        show(
            BlocksWorldDataset._estimate_state_tokens,
            expected_substrings=[
                "self.blocks",        # post-fix: parameterized
                "self.num_positions",
            ],
            unexpected_substrings=[
                "return 4",  # the old hardcoded value
            ],
        )
    except Exception:
        pass

    # ----------------------------------------------------------------
    # Base class
    # ----------------------------------------------------------------
    banner("PlanningDataset.generate_dataset  (base class)")
    try:
        from data.base import PlanningDataset
        show(
            PlanningDataset.generate_dataset,
            expected_substrings=[
                "MAX_ATTEMPTS",
                "n_short_after_retries",
            ],
        )
    except Exception as e:
        print(f"could not load PlanningDataset: {e}")

    # ----------------------------------------------------------------
    # Trainer
    # ----------------------------------------------------------------
    banner("trainer.save_generated_data")
    try:
        import trainer as T
        show(
            T.save_generated_data,
            expected_substrings=[
                "split_suffix",
                "_productivity",
            ],
        )
    except Exception as e:
        print(f"could not load trainer module: {e}")

    banner("trainer.load_cached_data")
    try:
        show(
            T.load_cached_data,
            expected_substrings=[
                "split_suffix",
                "_productivity",
            ],
        )
    except Exception:
        pass

    # ----------------------------------------------------------------
    # Downstream tools
    # ----------------------------------------------------------------
    banner("calibrate_sep.py  (Blocks World layout constants)")
    p = Path("calibrate_sep.py")
    if p.exists():
        src = p.read_text()
        snippets = []
        # Heuristically extract the BW layout dict
        idx = src.find('"name": "blocks_world"')
        if idx >= 0:
            start = src.rfind("{", 0, idx)
            end = src.find("}", idx) + 1
            if start >= 0 and end > start:
                snippets.append(src[start:end])
        if snippets:
            for s in snippets:
                print(s)
                print()
            print("--- patch checks ---")
            ok = '"state_len": 8' in src and '"context_end": 17' in src
            print(f"  {'✓' if ok else '✗'} BW layout uses state_len=8, context_end=17")
        else:
            print("  could not locate the BW layout dict; inspect manually")
    else:
        print("  calibrate_sep.py not present")

    banner("trace_one_problem.py  (BW guard)")
    p = Path("trace_one_problem.py")
    if p.exists():
        src = p.read_text()
        ok = 'args.domain == "blocks_world"' in src and "8-puzzle-only" in src
        print(f"  {'✓' if ok else '✗'} trace_one_problem.py has the BW guard")
        if not ok:
            print("    (the guard exits with an error message if --domain blocks_world is passed)")
    else:
        print("  trace_one_problem.py not present")

    # ----------------------------------------------------------------
    # File inventory
    # ----------------------------------------------------------------
    banner("file inventory")
    print("Patch scripts (should be deletable once everything is confirmed):")
    for name in [
        "fix_eight_puzzle_encoding.py",
        "fix_blocks_world_encoding.py",
        "fix_downstream_for_bw_encoding.py",
        "fix_trainer_cache_split.py",
        "fix_difficulty_enforcement.py",
    ]:
        marker = "present" if Path(name).exists() else "absent"
        print(f"  {name}: {marker}")

    print()
    print("Cache regenerators (deletable once caches are stable):")
    for name in [
        "regen_eight_puzzle_wm_cache.py",
        "regen_blocks_world_cache.py",
    ]:
        marker = "present" if Path(name).exists() else "absent"
        print(f"  {name}: {marker}")

    print()
    print("Backup files in data/ (artifacts from patches):")
    for p in Path("data").glob("*.bak-*"):
        print(f"  {p}")
    for p in Path(".").glob("*.bak-*"):
        print(f"  {p}")

    print()
    print("Cached data files:")
    for p in sorted(Path("cached_data").glob("*.json")):
        size_kb = p.stat().st_size // 1024
        print(f"  {p} ({size_kb} KB)")


if __name__ == "__main__":
    main()
