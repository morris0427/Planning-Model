#!/usr/bin/env python3
"""
Update downstream tooling for the new Blocks World uniform 8-token state encoding.

CHANGES
-------
1. data/blocks_world.py: _estimate_state_tokens() returns 8, not 4.
   This is used by get_max_sequence_length() in data/base.py to budget the
   transformer's positional embedding. With the old value of 4, the budget
   would be undersized and training would crash or silently truncate.

2. calibrate_sep.py: BW layout dict updated to state_len=8, context_end=17.
   This is what makes calibrate_sep.py's state-block extraction match the
   actual encoded format.

3. trace_one_problem.py: Documented as 8-puzzle-only.
   The current script's "Blocks World support" is just a CLI flag; every code
   path uses 8-puzzle constants (EIGHT_PUZZLE_MOVE_NAME, context_end=20, etc).
   Rather than pretend it works, we make it fail loudly with a clear message
   when --domain blocks_world is passed. A real BW trace tool is a separate
   piece of work.

USAGE
-----
    python3 fix_downstream_for_bw_encoding.py --dry-run
    python3 fix_downstream_for_bw_encoding.py
"""

import argparse
import shutil
import sys
import time
from pathlib import Path


PATCHES = [
    {
        'path': Path('data/blocks_world.py'),
        'description': '_estimate_state_tokens: 4 -> 8',
        'old': '''    def _estimate_state_tokens(self) -> int:
        """Estimate tokens per state."""
        return 4  # 4 positions in Blocks World
''',
        'new': '''    def _estimate_state_tokens(self) -> int:
        """Estimate tokens per state.

        Under the uniform encoding (top-to-bottom within each tower, POS_k
        separator always emitted): num_blocks + num_positions tokens per state.
        For the standard 4-block 4-position setup that is 8.
        """
        return len(self.blocks) + self.num_positions
''',
        'verify_after': lambda: _verify_estimate_state_tokens(),
    },
    {
        'path': Path('calibrate_sep.py'),
        'description': 'BW layout dict: state_len 4->8, context_end 9->17',
        'old': '''    return {
        "name": "blocks_world",
        "sep_token": 1,     # END
        "context_end": 9,
        "goal_start": 5,
        "goal_end": 9,
        "action_len": 2,
        "state_len": 4,
    }''',
        'new': '''    return {
        "name": "blocks_world",
        "sep_token": 1,     # END
        "context_end": 17,  # START + start_state(8) + goal_state(8)
        "goal_start": 9,    # immediately after start_state(8)
        "goal_end": 17,     # exclusive; matches context_end
        "action_len": 2,
        "state_len": 8,     # uniform encoding: 4 blocks + 4 POS_k separators
    }''',
        'verify_after': None,
    },
    {
        'path': Path('trace_one_problem.py'),
        'description': 'add Blocks World guard so it fails loudly',
        # We inject a check after the argparse line that says
        # "if args.domain == 'blocks_world': raise SystemExit(...)".
        # We anchor on the existing `args = ap.parse_args()` line.
        'old': '''    args = ap.parse_args()

    print("=" * 76)
    print(f"SINGLE-PROBLEM TRACE  ({args.domain}, {args.size}, "
          f"{'WM' if args.wm else 'Baseline'})")
    print("=" * 76)''',
        'new': '''    args = ap.parse_args()

    if args.domain == "blocks_world":
        print("ERROR: trace_one_problem.py is currently 8-puzzle-only.")
        print("       Every token-labeling and layout constant in this script")
        print("       (EIGHT_PUZZLE_MOVE_NAME, context_end=20, state_len=9, etc.)")
        print("       assumes 8-puzzle. Running with --domain blocks_world would")
        print("       silently mis-parse sequences. A dedicated BW trace tool")
        print("       would be a separate piece of work.")
        sys.exit(2)

    print("=" * 76)
    print(f"SINGLE-PROBLEM TRACE  ({args.domain}, {args.size}, "
          f"{'WM' if args.wm else 'Baseline'})")
    print("=" * 76)''',
        'verify_after': None,
    },
]


def _verify_estimate_state_tokens():
    """Reimport blocks_world and check the new value."""
    for mod in list(sys.modules):
        if mod.startswith("data.blocks_world") or mod == "data":
            del sys.modules[mod]
    sys.path.insert(0, ".")
    try:
        from data.blocks_world import BlocksWorldDataset
        ds = BlocksWorldDataset(difficulty_range=(3, 3), num_samples=1,
                                 use_world_model=False, seed=0)
        n = ds._estimate_state_tokens()
        if n == 8:
            print(f"  ✓ _estimate_state_tokens() now returns {n}")
            # Also exercise get_max_sequence_length with the new value
            ds.problems = []  # force estimation path
            max_len = ds.get_max_sequence_length()
            print(f"  ✓ get_max_sequence_length() with no cached problems: {max_len}")
            return True
        else:
            print(f"  ✗ _estimate_state_tokens() returned {n}, expected 8")
            return False
    except Exception as e:
        print(f"  ⚠️  verification raised: {e}")
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    # Pre-flight: every target file exists
    for p in PATCHES:
        if not p['path'].exists():
            print(f"ERROR: {p['path']} not found. Run from the experiments/ directory.")
            sys.exit(1)

    # Pre-flight: every old-text occurs exactly once
    print("Pre-flight: locating buggy text in each file...")
    for p in PATCHES:
        src = p['path'].read_text()
        n = src.count(p['old'])
        n_new = src.count(p['new'])
        if n == 0 and n_new > 0:
            print(f"  ⚪ {p['path']}: already patched ({p['description']})")
            p['_skip'] = True
        elif n == 0:
            print(f"  ✗ {p['path']}: expected text NOT FOUND ({p['description']})")
            print(f"    The file has been modified in a way I didn't expect.")
            print(f"    Refusing to patch. Inspect manually.")
            sys.exit(2)
        elif n > 1:
            print(f"  ✗ {p['path']}: {n} occurrences (expected 1) ({p['description']})")
            sys.exit(3)
        else:
            print(f"  ✓ {p['path']}: 1 occurrence ({p['description']})")
            p['_skip'] = False

    print()
    print("Patches to apply:")
    for p in PATCHES:
        marker = "(skip - already patched)" if p.get('_skip') else ""
        print(f"  - {p['path']}: {p['description']} {marker}")
    print()

    if args.dry_run:
        for p in PATCHES:
            if p.get('_skip'):
                continue
            print(f"--- diff for {p['path']} ---")
            print("BEFORE:")
            print(p['old'])
            print("AFTER:")
            print(p['new'])
            print()
        print("(dry run; not modifying any files)")
        sys.exit(0)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    for p in PATCHES:
        if p.get('_skip'):
            continue
        backup = p['path'].with_suffix(p['path'].suffix + f".bak-{timestamp}")
        shutil.copy2(p['path'], backup)
        print(f"Backed up {p['path']} -> {backup}")
        src = p['path'].read_text()
        p['path'].write_text(src.replace(p['old'], p['new']))
        print(f"Patched {p['path']} ({p['description']})")

    # Post-apply verifications
    print()
    print("Verifying patches...")
    all_ok = True
    for p in PATCHES:
        if p.get('_skip'):
            continue
        if p.get('verify_after'):
            ok = p['verify_after']()
            if not ok:
                all_ok = False

    print()
    if all_ok:
        print("All patches applied and verified.")
    else:
        print("⚠️  Some verifications failed; inspect the files manually.")

    print()
    print("Next steps:")
    print("  1. Re-run the round-trip test once more to confirm everything still")
    print("     passes after the downstream changes:")
    print("       python3 test_encoding_roundtrip.py")
    print("  2. Sanity-check calibrate_sep.py on the medium WM BW checkpoint")
    print("     (this re-measures with the new layout; numbers may shift slightly):")
    print("       python3 calibrate_sep.py --domain blocks_world --size medium --wm")
    print("     (NOTE: this only works on the OLD checkpoint until you retrain")
    print("      on the new cache; the layout for the OLD checkpoint differs.")
    print("      For meaningful results, run AFTER retraining on the new cache.)")
    print("  3. Retrain: python3 run_experiments.py --sweep model_size \\")
    print("                     --domain blocks_world --sizes medium")


if __name__ == "__main__":
    main()
