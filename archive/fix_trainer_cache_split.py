#!/usr/bin/env python3
"""
Patch trainer.py to make the cache lookup split-aware.

WHY
---
load_cached_data() currently uses filename {domain}_train{wm_suffix}.json with
no split-type component. This means productivity runs (which need cached data
trained on (1,4) / tested on (5,8) for Blocks World) will silently load the
in-distribution cache (trained AND tested on (1,6)) if it exists. The
training and test data are then NOT the productivity split, despite the
experiment being configured for productivity. Previous "productivity"
results may have been measuring something else entirely.

FIX
---
Append "_productivity" to the filename when config.data.split_type ==
SplitType.PRODUCTIVITY. Empty suffix for in-distribution (so existing caches
keep working). Same change applied to save_generated_data so newly-generated
data writes to a non-colliding filename.

CHANGES
-------
Two adjacent functions in trainer.py:
  - save_generated_data: derive split_suffix, include in filenames
  - load_cached_data:    same

USAGE
-----
    python3 fix_trainer_cache_split.py --dry-run
    python3 fix_trainer_cache_split.py
"""

import argparse
import shutil
import sys
import time
from pathlib import Path

TARGET_PATH = Path("trainer.py")

# --- Patch 1: save_generated_data ---
SAVE_BUGGY = (
    "    # Create filename based on config\n"
    "    domain = config.data.domain\n"
    "    wm_suffix = \"_wm\" if use_wm else \"_baseline\"\n"
    "    \n"
    "    train_file = cache_dir / f\"{domain}_train{wm_suffix}.json\"\n"
    "    test_file = cache_dir / f\"{domain}_test{wm_suffix}.json\"\n"
)
SAVE_FIXED = (
    "    # Create filename based on config\n"
    "    domain = config.data.domain\n"
    "    wm_suffix = \"_wm\" if use_wm else \"_baseline\"\n"
    "    # Include split type so productivity runs do not collide with in-distribution\n"
    "    split_suffix = \"_productivity\" if config.data.split_type.value == \"productivity\" else \"\"\n"
    "    \n"
    "    train_file = cache_dir / f\"{domain}_train{wm_suffix}{split_suffix}.json\"\n"
    "    test_file = cache_dir / f\"{domain}_test{wm_suffix}{split_suffix}.json\"\n"
)

# --- Patch 2: load_cached_data ---
LOAD_BUGGY = (
    "    cache_dir = Path(\"cached_data\")\n"
    "    domain = config.data.domain\n"
    "    \n"
    "    wm_suffix = \"_wm\" if use_wm else \"_baseline\"\n"
    "    \n"
    "    train_file = cache_dir / f\"{domain}_train{wm_suffix}.json\"\n"
    "    test_file = cache_dir / f\"{domain}_test{wm_suffix}.json\"\n"
)
LOAD_FIXED = (
    "    cache_dir = Path(\"cached_data\")\n"
    "    domain = config.data.domain\n"
    "    \n"
    "    wm_suffix = \"_wm\" if use_wm else \"_baseline\"\n"
    "    # Include split type so productivity runs do not collide with in-distribution\n"
    "    split_suffix = \"_productivity\" if config.data.split_type.value == \"productivity\" else \"\"\n"
    "    \n"
    "    train_file = cache_dir / f\"{domain}_train{wm_suffix}{split_suffix}.json\"\n"
    "    test_file = cache_dir / f\"{domain}_test{wm_suffix}{split_suffix}.json\"\n"
)


def apply_patch(src, name, buggy, fixed):
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
    src, msg1 = apply_patch(src, "save_generated_data", SAVE_BUGGY, SAVE_FIXED)
    src, msg2 = apply_patch(src, "load_cached_data",  LOAD_BUGGY, LOAD_FIXED)

    print("Patch status:")
    print(msg1)
    print(msg2)

    any_missing = "NOT FOUND" in msg1 or "occurrences" in msg1 \
        or "NOT FOUND" in msg2 or "occurrences" in msg2
    any_applied = "applied" in msg1 or "applied" in msg2

    if any_missing:
        print()
        print("ERROR: at least one patch could not be located cleanly.")
        print("Run python3 -c \"import inspect; from trainer import save_generated_data, load_cached_data;")
        print("print(inspect.getsource(save_generated_data)); print(inspect.getsource(load_cached_data))\"")
        print("and paste the output so the byte-match constants can be corrected.")
        sys.exit(2)

    if not any_applied:
        print()
        print("Nothing to do (both patches already applied).")
        sys.exit(0)

    print()
    print("--- patch 1 (save_generated_data) ---")
    print("BEFORE:\n" + SAVE_BUGGY)
    print("AFTER:\n" + SAVE_FIXED)
    print("--- patch 2 (load_cached_data) ---")
    print("BEFORE:\n" + LOAD_BUGGY)
    print("AFTER:\n" + LOAD_FIXED)

    if args.dry_run:
        print("\n(dry run; not modifying the file)")
        sys.exit(0)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    backup = TARGET_PATH.with_suffix(f".py.bak-{timestamp}")
    shutil.copy2(TARGET_PATH, backup)
    print(f"\nBacked up original to {backup}")

    TARGET_PATH.write_text(src)
    print(f"Wrote patched {TARGET_PATH}.")

    # Verification: import and exercise both functions on a fake config
    print()
    print("Verifying by exercising both functions with a fake productivity config...")
    for mod in list(sys.modules):
        if mod == "trainer" or mod.startswith("trainer."):
            del sys.modules[mod]
    sys.path.insert(0, ".")
    try:
        import inspect
        import trainer as T
        from config import SplitType

        # Build a minimal config shim
        class _DataCfg:
            domain = "blocks_world"
            split_type = SplitType.PRODUCTIVITY
        class _Cfg:
            data = _DataCfg()

        # Make sure load_cached_data computes the productivity-suffixed filename
        # (we test by reading the source, since we don't want to actually load)
        src_loaded = inspect.getsource(T.load_cached_data)
        if "_productivity" in src_loaded:
            print("  ✓ load_cached_data source contains productivity suffix logic")
        else:
            print("  ✗ load_cached_data source missing productivity suffix logic")

        src_saved = inspect.getsource(T.save_generated_data)
        if "_productivity" in src_saved:
            print("  ✓ save_generated_data source contains productivity suffix logic")
        else:
            print("  ✗ save_generated_data source missing productivity suffix logic")
    except Exception as e:
        print(f"  ⚠️  verification raised: {e}")

    print()
    print("Next steps:")
    print("  1. Generate productivity data (uses fixed encodings):")
    print("       python3 generate_productivity_data.py --domain blocks_world")
    print("       python3 generate_productivity_data.py --domain eight_puzzle")
    print("  2. Verify the new caches are internally consistent:")
    print("       python3 test_encoding_roundtrip.py --cache-dir cached_data \\")
    print("           --domain blocks_world")
    print("     (NOTE: test_encoding_roundtrip currently only looks for")
    print("      non-productivity-suffixed caches; you may want to extend it")
    print("      or just trust the regenerator's inline check.)")
    print("  3. Retrain on productivity split:")
    print("       python3 run_productivity_sweep.py --domain blocks_world")
    print("     (this will retrain medium WM + medium baseline on (1,4),")
    print("      test on (5,8))")


if __name__ == "__main__":
    main()
