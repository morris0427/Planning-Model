#!/usr/bin/env python3
"""
Clean up the working directory after all patches and regenerations are
confirmed to be correctly applied (per audit_pipeline.py).

What this does:
  1. Creates an archive/ directory.
  2. Moves all fix_*.py and regen_*.py scripts into archive/.
  3. Deletes the *.bak-* backup files left by the patches.
  4. Writes a README.md documenting the data pipeline.

Why each step:
  - The patch scripts have already done their job. Keeping them in the
    working directory creates confusion about which files are part of
    the live pipeline vs. one-shot tools. Archive lets us refer to them
    if needed without cluttering the top level.
  - The regen scripts are also one-shot tools at this point. Caches are
    stable.
  - The .bak files are timestamped snapshots from when patches were
    applied. They're no longer needed once we trust the current code,
    and they confuse file listings.
  - A README documents the (now-correct) pipeline so future you (or a
    reader) can find their way around.

Run from the experiments/ directory:
    python3 cleanup_after_audit.py --dry-run
    python3 cleanup_after_audit.py

Nothing is destructive without --dry-run preview.
"""

import argparse
import shutil
from pathlib import Path
from datetime import datetime

PATCH_SCRIPTS = [
    "fix_eight_puzzle_encoding.py",
    "fix_blocks_world_encoding.py",
    "fix_downstream_for_bw_encoding.py",
    "fix_trainer_cache_split.py",
    "fix_difficulty_enforcement.py",
]

REGEN_SCRIPTS = [
    "regen_eight_puzzle_wm_cache.py",
    "regen_blocks_world_cache.py",
]


README_CONTENT = """# experiments/ data pipeline

This directory contains the planning-with-transformers experiments
(Blocks World and 8-puzzle, baseline vs world-model auxiliary task).

## Where things live

### Domain data generation and encoding

  data/base.py
    PlanningDataset (abstract)
      generate_dataset()        -- samples difficulties and calls
                                   generate_problem() with bounded retry
                                   to enforce the requested difficulty
      get_max_sequence_length() -- sizing the positional embedding budget
      _estimate_state_tokens()  -- per-domain hook

  data/blocks_world.py
    BlocksWorldDataset
      generate_problem(diff)    -- SAW backward walk from a random goal,
                                   forward-reconstruct moves
      _encode_state(state)      -- uniform 8-token state encoding
                                   (top-to-bottom per tower, POS_k separator
                                    always emitted)
      _decode_state(tokens)     -- inverse of _encode_state
      encode_sequence(problem)  -- full token sequence
      _estimate_state_tokens()  -- returns num_blocks + num_positions

  data/eight_puzzle.py
    EightPuzzleDataset
      generate_problem(diff)    -- SAW backward walk from goal, then
                                   reverse moves to get start -> goal
      encode_sequence(problem)  -- full token sequence; WM mode pairs
                                   each move with its post-move state
                                   via solution_states[i+1]

### Sequence layouts

Blocks World (4 blocks A-D, 4 positions, vocab size 11):
  Baseline:
    [START(1), start_state(8), goal_state(8), m_1(2), m_2(2), ..., m_N(2), END(1)]
  World Model:
    [START(1), start_state(8), goal_state(8),
     m_1(2), state_after_1(8), m_2(2), state_after_2(8), ...,
     m_N(2), state_after_N=goal(8), END(1)]
  An action is [block, POS_k] (2 tokens).

8-puzzle (vocab size 16; goal is fixed):
  Baseline:
    [dummy_move(1), start_state(9), PAD(1), goal_state(9),
     m_1(1), m_2(1), ..., m_N(1), SEP(1)]
  World Model:
    [dummy_move(1), start_state(9), PAD(1), goal_state(9),
     m_1(1), state_after_1(9), m_2(1), state_after_2(9), ...,
     m_N(1), state_after_N=goal(9), SEP(1)]
  An action is a single move token (10=up, 11=down, 12=left, 13=right).

### Training pipeline

  trainer.py
    train(config, use_wm)       -- main entrypoint
    load_cached_data(...)       -- looks up cached_data/{domain}_train|test{_wm|_baseline}{_productivity?}.json
    save_generated_data(...)    -- symmetric writer
    generate_solution(...)      -- inference, with oracle state injection for WM

### Configuration

  config.py
    DataPresets.{blocks_world|eight_puzzle}                  -- in-distribution
    DataPresets.{blocks_world|eight_puzzle}_productivity     -- OOD-length split
    ModelPresets.{tiny|small|medium|large}                   -- transformer sizes

### Caches

  cached_data/
    {domain}_train{_wm|_baseline}.json                       -- in-distribution
    {domain}_test{_wm|_baseline}.json                        -- in-distribution
    {domain}_train{_wm|_baseline}_productivity.json          -- OOD-length
    {domain}_test{_wm|_baseline}_productivity.json           -- OOD-length

### Verification

  test_encoding_roundtrip.py
    Layer 1: strict round-trip on fresh problems
    Layer 3: dataset's encode_sequence agrees with an independent reference
    Layer 4: cached data is internally consistent
    Run: python3 test_encoding_roundtrip.py [--domain DOMAIN]

  audit_pipeline.py
    Prints live source of every critical function. Use to verify state.

### Diagnostics

  calibrate_sep.py        -- measure P(SEP | at goal) vs P(SEP | not at goal)
                              for a trained model
  trace_one_problem.py    -- per-problem trace (8-puzzle only)
  diagnose_tf_vs_freegen  -- teacher-forced vs free-generation comparison

## How to run experiments

In-distribution sweep, medium only:
    python3 run_experiments.py --sweep model_size --domain blocks_world --sizes medium
    python3 run_experiments.py --sweep model_size --domain eight_puzzle --sizes medium

Productivity sweep, medium only:
    python3 run_productivity_sweep.py --domain blocks_world --sizes medium
    python3 run_productivity_sweep.py --domain eight_puzzle --sizes medium

Calibration on a trained checkpoint:
    python3 calibrate_sep.py --domain blocks_world --size medium --wm

## archive/

One-shot tools used to bring the pipeline to its current state. Kept for
reference but not part of the live pipeline. See archive/README.md.

## Notes on what is *not* in the pipeline anymore

  - The lossy "top-block-only" Blocks World state encoding (replaced by
    uniform 8-token encoding).
  - The off-by-one in 8-puzzle's WM encoder pairing move i with the
    pre-move state (fixed; now pairs with i+1, the post-move state).
  - The states[:-1] slice in 8-puzzle's generate_problem that dropped
    the goal from solution_states (fixed; full N+1 entries returned).
  - The split-agnostic cache lookup that caused productivity runs to
    silently load in-distribution caches (fixed; split_suffix appended).
  - The unenforced difficulty range in generate_dataset that let SAW
    failures contaminate test sets with shorter problems (fixed;
    bounded retry).
"""


ARCHIVE_README = """# archive/

One-shot tools used to bring the experiments/ pipeline to its current
state. These have already been applied. They are kept here for
reference but are not part of the live pipeline.

## Patch scripts

Each of these modified a specific file in the codebase:

  fix_eight_puzzle_encoding.py
    Modified data/eight_puzzle.py:
      generate_problem: removed states[:-1] slice so solution_states
                        has N+1 entries (start through goal).
      encode_sequence: changed solution_states[i] to [i+1] (and the
                       bounds check correspondingly) so the WM training
                       data pairs each move with its post-move state.

  fix_blocks_world_encoding.py
    Modified data/blocks_world.py:
      _encode_state: replaced lossy top-blocks-only encoding with the
                     uniform 8-token encoding (top-to-bottom per tower,
                     POS_k separator always emitted).
      _decode_state: added as the inverse.

  fix_downstream_for_bw_encoding.py
    Modified data/blocks_world.py and calibrate_sep.py and
    trace_one_problem.py for downstream consequences of the new
    Blocks World state encoding:
      _estimate_state_tokens: parameterized as blocks + positions
      calibrate_sep.py: BW layout uses state_len=8, context_end=17
      trace_one_problem.py: guard preventing accidental BW usage
                            (script is 8-puzzle-only)

  fix_trainer_cache_split.py
    Modified trainer.py:
      load_cached_data, save_generated_data: split-aware filenames
      so productivity runs don't collide with in-distribution caches.

  fix_difficulty_enforcement.py
    Modified data/base.py:
      generate_dataset: bounded retry to enforce the requested
      difficulty exactly. Prevents SAW failures from contaminating
      test sets with shorter problems.

## Cache regenerators

  regen_eight_puzzle_wm_cache.py
  regen_blocks_world_cache.py
    Regenerated the cache files after the encoding fixes landed.
    Run once. The caches in cached_data/ are now the regenerated
    versions and should remain stable.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    archive = Path("archive")
    actions = []

    # Plan: move patch + regen scripts into archive/
    scripts_to_move = []
    for name in PATCH_SCRIPTS + REGEN_SCRIPTS:
        p = Path(name)
        if p.exists():
            scripts_to_move.append(p)
            actions.append(f"move  {p}  ->  archive/{p.name}")

    # Plan: delete .bak-* files in data/ and top-level
    backups_to_delete = []
    for p in Path("data").glob("*.bak-*"):
        backups_to_delete.append(p)
        actions.append(f"delete  {p}")
    for p in Path(".").glob("*.bak-*"):
        backups_to_delete.append(p)
        actions.append(f"delete  {p}")

    # Plan: write README and archive/README
    actions.append("write  README.md")
    actions.append("write  archive/README.md")

    print("Planned actions:")
    for a in actions:
        print(f"  {a}")

    if args.dry_run:
        print()
        print("(dry run; nothing modified)")
        return

    print()
    if not scripts_to_move and not backups_to_delete and Path("README.md").exists():
        print("Nothing to do.")
        return

    archive.mkdir(exist_ok=True)
    (archive / "README.md").write_text(ARCHIVE_README)
    print(f"Wrote archive/README.md")

    for p in scripts_to_move:
        dest = archive / p.name
        shutil.move(str(p), str(dest))
        print(f"Moved {p} -> {dest}")

    for p in backups_to_delete:
        p.unlink()
        print(f"Deleted {p}")

    Path("README.md").write_text(README_CONTENT)
    print(f"Wrote README.md")

    print()
    print("Cleanup complete. Suggested verification:")
    print("  python3 audit_pipeline.py     # confirm pipeline still in good state")
    print("  python3 test_encoding_roundtrip.py    # full round-trip test")


if __name__ == "__main__":
    main()
