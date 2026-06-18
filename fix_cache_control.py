#!/usr/bin/env python3
"""
Add --force-regenerate flag and cache metadata sidecar.

WHY
---
Cache loading is currently silent: if cached_data/ has the right filenames,
the trainer loads them with no in-band way for the user to override or see
what they're loading. This was the root cause of two distinct footguns this
session:
  - Pre-fix productivity caches contained difficulty-contaminated data and
    runs used them silently.
  - Pre-split-aware lookup loaded in-distribution caches for productivity
    runs, silently substituting one distribution for another.

WHAT THIS DOES
--------------
Six edits across three files. All anchored on byte-exact text we've
verified against the live source.

  trainer.py
    save_generated_data: also write a {filename}.meta.json sidecar
                         (created timestamp, domain, split type,
                          difficulty range, sample count, num_moves
                          min/max for both train and test).
    load_cached_data:    take force_regenerate=False kwarg, return
                         (None, None) when set. On hit, read the
                         metadata sidecar and print its contents;
                         warn if split_type or difficulty range
                         disagree with the current config.
    train:               take force_regenerate=False kwarg, pass it
                         to load_cached_data.

  run_experiments.py
    ExperimentRunner.__init__: take force_regenerate=False kwarg, store.
    ExperimentRunner.run_experiment: pass self.force_regenerate to train.
    main(): add --force-regenerate, pass to ExperimentRunner(...).

  run_productivity_sweep.py
    main(): add --force-regenerate, pass to ExperimentRunner(...).

Backward compatible: flag defaults False; caches without sidecars still
load with a one-line note.

USAGE
-----
    python3 fix_cache_control.py --dry-run
    python3 fix_cache_control.py
"""

import argparse
import shutil
import sys
import time
from pathlib import Path


# =====================================================================
# trainer.py edits
# =====================================================================

SAVE_BUGGY = '''def save_generated_data(train_problems, test_problems, config, use_wm):
    """Save generated data to disk for reuse across experiments."""
    cache_dir = Path("cached_data")
    cache_dir.mkdir(exist_ok=True)
    
    # Create filename based on config
    domain = config.data.domain
    wm_suffix = "_wm" if use_wm else "_baseline"
    # Include split type so productivity runs do not collide with in-distribution
    split_suffix = "_productivity" if config.data.split_type.value == "productivity" else ""
    
    train_file = cache_dir / f"{domain}_train{wm_suffix}{split_suffix}.json"
    test_file = cache_dir / f"{domain}_test{wm_suffix}{split_suffix}.json"
    
    # Save
    with open(train_file, 'w') as f:
        json.dump(train_problems, f)
    
    with open(test_file, 'w') as f:
        json.dump(test_problems, f)
    
    print(f"\\n{'='*70}")
    print(f"💾 SAVED DATA FOR REUSE")
    print(f"{'='*70}")
    print(f"Train: {train_file}")
    print(f"Test:  {test_file}")
    print(f"This data will be reused for remaining experiments!")
    print(f"{'='*70}\\n")
    
    return train_file, test_file'''

SAVE_FIXED = '''def save_generated_data(train_problems, test_problems, config, use_wm):
    """Save generated data to disk for reuse across experiments.

    Also writes a {filename}.meta.json sidecar so the next load can show
    the user exactly what is in the cache.
    """
    cache_dir = Path("cached_data")
    cache_dir.mkdir(exist_ok=True)
    
    # Create filename based on config
    domain = config.data.domain
    wm_suffix = "_wm" if use_wm else "_baseline"
    # Include split type so productivity runs do not collide with in-distribution
    split_suffix = "_productivity" if config.data.split_type.value == "productivity" else ""
    
    train_file = cache_dir / f"{domain}_train{wm_suffix}{split_suffix}.json"
    test_file = cache_dir / f"{domain}_test{wm_suffix}{split_suffix}.json"
    
    # Save
    with open(train_file, 'w') as f:
        json.dump(train_problems, f)
    
    with open(test_file, 'w') as f:
        json.dump(test_problems, f)
    
    # Metadata sidecar (purely informational; trainer never reads this
    # to decide whether to use the cache)
    import time as _time
    train_ranges, test_ranges = config.data.get_split_ranges()
    meta = {
        "created_utc": _time.strftime("%Y-%m-%dT%H:%M:%SZ", _time.gmtime()),
        "domain": domain,
        "use_world_model": use_wm,
        "split_type": config.data.split_type.value,
        "train": {
            "difficulty_range": list(train_ranges),
            "num_samples": len(train_problems),
            "num_moves_min": min(p["num_moves"] for p in train_problems),
            "num_moves_max": max(p["num_moves"] for p in train_problems),
        },
        "test": {
            "difficulty_range": list(test_ranges),
            "num_samples": len(test_problems),
            "num_moves_min": min(p["num_moves"] for p in test_problems),
            "num_moves_max": max(p["num_moves"] for p in test_problems),
        },
    }
    for path in (train_file, test_file):
        meta_path = path.with_suffix(".meta.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
    
    print(f"\\n{'='*70}")
    print(f"💾 SAVED DATA FOR REUSE")
    print(f"{'='*70}")
    print(f"Train: {train_file}")
    print(f"Test:  {test_file}")
    print(f"Metadata sidecars written alongside.")
    print(f"This data will be reused for remaining experiments!")
    print(f"{'='*70}\\n")
    
    return train_file, test_file'''


LOAD_BUGGY = '''def load_cached_data(config, use_wm):
    """Try to load cached data if it exists."""
    cache_dir = Path("cached_data")
    domain = config.data.domain
    
    wm_suffix = "_wm" if use_wm else "_baseline"
    # Include split type so productivity runs do not collide with in-distribution
    split_suffix = "_productivity" if config.data.split_type.value == "productivity" else ""
    
    train_file = cache_dir / f"{domain}_train{wm_suffix}{split_suffix}.json"
    test_file = cache_dir / f"{domain}_test{wm_suffix}{split_suffix}.json"
    
    if train_file.exists() and test_file.exists():
        print(f"\\n{'='*70}")
        print(f"📂 LOADING CACHED DATA (Skipping generation!)")
        print(f"{'='*70}")
        print(f"Train: {train_file}")
        print(f"Test:  {test_file}")
        print(f"{'='*70}\\n")
        
        with open(train_file, 'r') as f:
            train_problems = json.load(f)
        
        with open(test_file, 'r') as f:
            test_problems = json.load(f)
        
        return train_problems, test_problems
    
    return None, None'''

LOAD_FIXED = '''def load_cached_data(config, use_wm, force_regenerate=False):
    """Try to load cached data if it exists.

    If force_regenerate is True, skip the lookup and return (None, None)
    so the caller regenerates from scratch.

    On a cache hit, reads the {filename}.meta.json sidecar (if present)
    and prints its contents. Warns if recorded split_type or difficulty
    range disagrees with the current config.
    """
    if force_regenerate:
        print(f"\\n{'='*70}")
        print(f"🔁 FORCE REGENERATE: skipping any cached data")
        print(f"{'='*70}\\n")
        return None, None
    
    cache_dir = Path("cached_data")
    domain = config.data.domain
    
    wm_suffix = "_wm" if use_wm else "_baseline"
    # Include split type so productivity runs do not collide with in-distribution
    split_suffix = "_productivity" if config.data.split_type.value == "productivity" else ""
    
    train_file = cache_dir / f"{domain}_train{wm_suffix}{split_suffix}.json"
    test_file = cache_dir / f"{domain}_test{wm_suffix}{split_suffix}.json"
    
    if train_file.exists() and test_file.exists():
        print(f"\\n{'='*70}")
        print(f"📂 LOADING CACHED DATA (Skipping generation!)")
        print(f"{'='*70}")
        print(f"Train: {train_file}")
        print(f"Test:  {test_file}")
        meta_path = train_file.with_suffix(".meta.json")
        if meta_path.exists():
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
                print(f"Cache metadata:")
                print(f"  created:       {meta.get('created_utc', '?')}")
                print(f"  domain:        {meta.get('domain', '?')}")
                print(f"  split_type:    {meta.get('split_type', '?')}")
                print(f"  world_model:   {meta.get('use_world_model', '?')}")
                tr = meta.get("train", {})
                te = meta.get("test", {})
                print(f"  train: range={tr.get('difficulty_range','?')} "
                      f"samples={tr.get('num_samples','?')} "
                      f"num_moves=[{tr.get('num_moves_min','?')}..{tr.get('num_moves_max','?')}]")
                print(f"  test:  range={te.get('difficulty_range','?')} "
                      f"samples={te.get('num_samples','?')} "
                      f"num_moves=[{te.get('num_moves_min','?')}..{te.get('num_moves_max','?')}]")
                cur_split = config.data.split_type.value
                if meta.get("split_type") and meta["split_type"] != cur_split:
                    print(f"  ⚠️  cache split_type ({meta['split_type']}) "
                          f"differs from current config ({cur_split})")
                req_train_range, req_test_range = config.data.get_split_ranges()
                if tr.get("difficulty_range") and list(tr["difficulty_range"]) != list(req_train_range):
                    print(f"  ⚠️  cached train range {tr['difficulty_range']} "
                          f"differs from current {list(req_train_range)}")
                if te.get("difficulty_range") and list(te["difficulty_range"]) != list(req_test_range):
                    print(f"  ⚠️  cached test range {te['difficulty_range']} "
                          f"differs from current {list(req_test_range)}")
            except Exception as e:
                print(f"  (could not read metadata sidecar: {e})")
        else:
            print(f"  (no metadata sidecar; cache predates the metadata feature)")
        print(f"{'='*70}\\n")
        
        with open(train_file, 'r') as f:
            train_problems = json.load(f)
        
        with open(test_file, 'r') as f:
            test_problems = json.load(f)
        
        return train_problems, test_problems
    
    return None, None'''


TRAIN_SIG_BUGGY = '''def train(config: ExperimentConfig) -> Dict[str, Any]:
    """
    Main training function.
    
    Extracted from user's train_blocks_world.py and adapted
    to use the experiment framework configuration.
    
    CRITICAL: Uses learning_rate=0.0001 (not 0.001) to prevent divergence.
    
    Args:
        config: Experiment configuration
    
    Returns:
        Dictionary with results
    """'''

TRAIN_SIG_FIXED = '''def train(config: ExperimentConfig, force_regenerate: bool = False) -> Dict[str, Any]:
    """
    Main training function.
    
    Extracted from user's train_blocks_world.py and adapted
    to use the experiment framework configuration.
    
    CRITICAL: Uses learning_rate=0.0001 (not 0.001) to prevent divergence.
    
    Args:
        config: Experiment configuration
        force_regenerate: If True, skip any cached data and regenerate.
            Default False preserves the cache-when-present behavior.
    
    Returns:
        Dictionary with results
    """'''

TRAIN_CALL_BUGGY = '''    train_problems, test_problems = load_cached_data(config, use_wm)'''
TRAIN_CALL_FIXED = '''    train_problems, test_problems = load_cached_data(config, use_wm, force_regenerate=force_regenerate)'''


# =====================================================================
# run_experiments.py edits
# =====================================================================

RUNNER_INIT_BUGGY = '''    def __init__(self, results_dir: str = "./results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)'''

RUNNER_INIT_FIXED = '''    def __init__(self, results_dir: str = "./results", force_regenerate: bool = False):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.force_regenerate = force_regenerate'''

RUNNER_CALL_BUGGY = '''            # Run actual training
            results = train(config)'''
RUNNER_CALL_FIXED = '''            # Run actual training
            results = train(config, force_regenerate=self.force_regenerate)'''

RE_ARGPARSE_BUGGY = '''    parser = argparse.ArgumentParser(description="Run compositional generalization experiments")'''
RE_ARGPARSE_FIXED = '''    parser = argparse.ArgumentParser(description="Run compositional generalization experiments")
    parser.add_argument(
        "--force-regenerate",
        action="store_true",
        help="Ignore any cached data and regenerate fresh datasets.",
    )'''

RE_INSTANTIATE_BUGGY = '''    runner = ExperimentRunner(results_dir=args.results_dir)'''
RE_INSTANTIATE_FIXED = '''    runner = ExperimentRunner(results_dir=args.results_dir, force_regenerate=args.force_regenerate)'''


# =====================================================================
# run_productivity_sweep.py edits
# =====================================================================

# This one we add the argparse argument right before `args = parser.parse_args()`
# (the existing --sizes block; we'll anchor on a unique line).
# Since we've previously added --sizes to this script, we anchor on a
# unique stable line in the argparse section.
PROD_ARGPARSE_BUGGY = '''    parser.add_argument(
        '--sizes',
        nargs='+',
        choices=['small', 'medium', 'large'],
        default=None,
        help='Restrict to specific model sizes (default: all three).'
    )
    
    args = parser.parse_args()'''

PROD_ARGPARSE_FIXED = '''    parser.add_argument(
        '--sizes',
        nargs='+',
        choices=['small', 'medium', 'large'],
        default=None,
        help='Restrict to specific model sizes (default: all three).'
    )
    parser.add_argument(
        '--force-regenerate',
        action='store_true',
        help='Ignore any cached data and regenerate fresh datasets.',
    )
    
    args = parser.parse_args()'''

PROD_INSTANTIATE_BUGGY = '''    runner = ExperimentRunner(results_dir=args.results_dir)'''
PROD_INSTANTIATE_FIXED = '''    runner = ExperimentRunner(results_dir=args.results_dir, force_regenerate=args.force_regenerate)'''


# =====================================================================
# Driver
# =====================================================================

def apply_patch(src, name, buggy, fixed):
    n_buggy = src.count(buggy)
    n_fixed = src.count(fixed)
    if n_buggy == 0 and n_fixed > 0:
        return src, f"  ⚪ {name}: already patched", False
    if n_buggy == 0:
        return src, f"  ✗ {name}: expected text NOT FOUND", True
    if n_buggy > 1:
        return src, f"  ✗ {name}: {n_buggy} occurrences (expected 1)", True
    return src.replace(buggy, fixed), f"  ✓ {name}: applied", False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    trainer_path = Path("trainer.py")
    runner_path  = Path("run_experiments.py")
    prod_path    = Path("run_productivity_sweep.py")

    for p in (trainer_path, runner_path, prod_path):
        if not p.exists():
            print(f"ERROR: {p} not found. Run from the experiments/ directory.")
            sys.exit(1)

    any_error = False
    all_msgs = []

    # ----- trainer.py -----
    s = trainer_path.read_text()
    orig_trainer = s
    s, m, err = apply_patch(s, "trainer.save_generated_data (metadata sidecar)", SAVE_BUGGY, SAVE_FIXED)
    all_msgs.append(m); any_error = any_error or err
    s, m, err = apply_patch(s, "trainer.load_cached_data (force_regenerate + metadata)", LOAD_BUGGY, LOAD_FIXED)
    all_msgs.append(m); any_error = any_error or err
    s, m, err = apply_patch(s, "trainer.train signature", TRAIN_SIG_BUGGY, TRAIN_SIG_FIXED)
    all_msgs.append(m); any_error = any_error or err
    s, m, err = apply_patch(s, "trainer.train: pass force_regenerate to load_cached_data", TRAIN_CALL_BUGGY, TRAIN_CALL_FIXED)
    all_msgs.append(m); any_error = any_error or err
    new_trainer = s

    # ----- run_experiments.py -----
    s = runner_path.read_text()
    orig_runner = s
    s, m, err = apply_patch(s, "run_experiments: ExperimentRunner.__init__", RUNNER_INIT_BUGGY, RUNNER_INIT_FIXED)
    all_msgs.append(m); any_error = any_error or err
    s, m, err = apply_patch(s, "run_experiments: pass force_regenerate to train", RUNNER_CALL_BUGGY, RUNNER_CALL_FIXED)
    all_msgs.append(m); any_error = any_error or err
    s, m, err = apply_patch(s, "run_experiments: argparse --force-regenerate", RE_ARGPARSE_BUGGY, RE_ARGPARSE_FIXED)
    all_msgs.append(m); any_error = any_error or err
    s, m, err = apply_patch(s, "run_experiments: instantiate runner with flag", RE_INSTANTIATE_BUGGY, RE_INSTANTIATE_FIXED)
    all_msgs.append(m); any_error = any_error or err
    new_runner = s

    # ----- run_productivity_sweep.py -----
    s = prod_path.read_text()
    orig_prod = s
    s, m, err = apply_patch(s, "run_productivity_sweep: argparse --force-regenerate", PROD_ARGPARSE_BUGGY, PROD_ARGPARSE_FIXED)
    all_msgs.append(m); any_error = any_error or err
    s, m, err = apply_patch(s, "run_productivity_sweep: instantiate runner with flag", PROD_INSTANTIATE_BUGGY, PROD_INSTANTIATE_FIXED)
    all_msgs.append(m); any_error = any_error or err
    new_prod = s

    print("Patch status:")
    for m in all_msgs:
        print(m)

    if any_error:
        print()
        print("ERROR: at least one patch could not be located cleanly.")
        print("Refusing to write. Run audit_pipeline.py and verify the relevant source")
        print("blocks match the constants in this script (especially indentation).")
        sys.exit(2)

    print()
    if args.dry_run:
        print("(dry run; not modifying any files)")
        return

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    for orig, new, path in [
        (orig_trainer, new_trainer, trainer_path),
        (orig_runner,  new_runner,  runner_path),
        (orig_prod,    new_prod,    prod_path),
    ]:
        if new != orig:
            backup = path.with_suffix(f".py.bak-{timestamp}")
            shutil.copy2(path, backup)
            path.write_text(new)
            print(f"Patched {path}  (backup: {backup})")
        else:
            print(f"No change to {path} (all relevant patches already applied)")

    print()
    print("Verifying by reimporting modules and inspecting key functions...")
    for mod in list(sys.modules):
        if mod == "trainer" or mod.startswith("trainer."):
            del sys.modules[mod]
    sys.path.insert(0, ".")
    try:
        import inspect
        import trainer as T
        src_load = inspect.getsource(T.load_cached_data)
        src_save = inspect.getsource(T.save_generated_data)
        src_train = inspect.getsource(T.train)
        ok = (
            "force_regenerate" in src_load
            and "meta_path" in src_save
            and "force_regenerate" in src_train
        )
        if ok:
            print("  ✓ trainer.py: force_regenerate and metadata-sidecar logic in place")
        else:
            print("  ⚠️  trainer.py: reimport succeeded but expected substrings missing")
    except Exception as e:
        print(f"  ⚠️  verification raised: {e}")

    print()
    print("Done. Usage:")
    print("  python3 run_experiments.py --sweep model_size --domain blocks_world \\")
    print("      --sizes medium --force-regenerate")
    print("  python3 run_productivity_sweep.py --domain blocks_world \\")
    print("      --sizes medium --force-regenerate")
    print()
    print("Without the flag, default behavior is unchanged: cache loaded when present,")
    print("now with metadata sidecar shown in the cache-hit message.")


if __name__ == "__main__":
    main()
