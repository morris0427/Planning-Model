"""
Fix generate_solution truncation asymmetry between state_source paths.

WHY
---
After our state_source patch, the WM-model path used headroom=1 for the
seq-length truncation check, while the WM-oracle path used
headroom=action_tok_size+per_state_tokens+1. This gave the model path
more sequence budget on long trajectories, so on borderline-length
problems the model path could finish where the oracle path truncated.

Also, when the oracle path saw a non-action-start token at the first
action slot, it did a half-hearted recovery (appending the bad token and
breaking out of the inner loop, hoping the next outer iteration would
proceed). That doesn't match the model path which just runs straight.

These two implementation differences made the two state_source paths
produce slightly different results even on in-distribution problems
where the model's state predictions are accurate (and the two paths
should agree exactly by construction).

WHAT THIS DOES
--------------
1. Aligns the headroom check. Both paths now use the same total budget
   in the truncation guard. The oracle path's headroom was correct in
   intent (leave room for one full action + state + END); we just need
   the model path to use the same definition.

2. Removes the half-hearted non-action-token recovery in the oracle
   path. If the model emits a non-action-start token at an action slot,
   we treat it as a termination (it's almost certainly an END token or
   garbage; the model has nothing useful to say).

After this patch, on problems where the model's state predictions are
accurate (verified by byte-identical generation), both state_source
paths should solve exactly the same set of problems.

USAGE
-----
    python3 fix_generate_solution_truncation_symmetry.py --dry-run
    python3 fix_generate_solution_truncation_symmetry.py
"""

import argparse
import shutil
import sys
import time
from pathlib import Path


# We'll do TWO surgical replacements, each anchored on byte-exact text
# from the current generate_solution.


# ---------- Fix 1: oracle path non-action-token recovery ----------
ORACLE_BAD_RECOVERY = '''                    if not action_pieces and not is_action_start_tok(next_tok):
                        # Skip non-action token; appending it lets the loop
                        # re-evaluate context at next iteration. Bounded by max_steps.
                        generated.append(next_tok)
                        break'''

ORACLE_GOOD_RECOVERY = '''                    if not action_pieces and not is_action_start_tok(next_tok):
                        # Model emitted neither END nor a valid action-start
                        # token. Treat as garbage termination; do not append.
                        premature_sep = True
                        break'''


# ---------- Fix 2: model path headroom check ----------
# The original model-path block:
#
#     for _ in range(max_length * max(1, stride + 1)):
#         if len(generated) >= model_max_seq_length - 1:
#             termination = 'truncation_seqlen'
#             break
#
# We change `- 1` to match the oracle path's headroom of
# `action_tok_size + per_state_tokens + 1`.

MODEL_BAD_HEADROOM = '''            for _ in range(max_length * max(1, stride + 1)):
                if len(generated) >= model_max_seq_length - 1:
                    termination = 'truncation_seqlen'
                    break'''

MODEL_GOOD_HEADROOM = '''            # Same headroom as the oracle path so the two state_source modes
            # truncate at the same effective boundary
            headroom = action_tok_size + (per_state_tokens if use_world_model else 0) + 1
            for _ in range(max_length * max(1, stride + 1)):
                if len(generated) >= model_max_seq_length - headroom:
                    termination = 'truncation_seqlen'
                    break'''


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

    target = Path("trainer.py")
    if not target.exists():
        print(f"ERROR: {target} not found. Run from the experiments/ directory.")
        sys.exit(1)

    src = target.read_text()
    new_src = src

    print("Applying truncation-symmetry patches to trainer.generate_solution:")
    print()

    new_src, msg, err1 = apply_patch(
        new_src,
        "oracle path non-action-token recovery",
        ORACLE_BAD_RECOVERY,
        ORACLE_GOOD_RECOVERY,
    )
    print(msg)

    new_src, msg, err2 = apply_patch(
        new_src,
        "model path headroom alignment",
        MODEL_BAD_HEADROOM,
        MODEL_GOOD_HEADROOM,
    )
    print(msg)

    any_error = err1 or err2

    if any_error:
        print()
        print("ERROR: at least one anchor not found cleanly. Refusing to patch.")
        print("       Verify the live source matches the constants in this script.")
        print()
        print("To inspect what's there:")
        print('  python3 -c "import inspect, trainer; '
              'print(inspect.getsource(trainer.generate_solution))"')
        sys.exit(2)

    if new_src == src:
        print()
        print("No change needed (patches already applied).")
        sys.exit(0)

    print()
    if args.dry_run:
        print(f"trainer.py would change from {len(src)} to {len(new_src)} bytes.")
        print("(dry run; not modifying)")
        sys.exit(0)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    backup = target.with_suffix(f".py.bak-{timestamp}")
    shutil.copy2(target, backup)
    print(f"Backed up to {backup}")

    target.write_text(new_src)
    print(f"Patched {target}.")

    # Verification: reimport and inspect
    print()
    print("Verifying by reimporting...")
    sys.path.insert(0, ".")
    for mod in list(sys.modules):
        if mod == "trainer" or mod.startswith("trainer."):
            del sys.modules[mod]
    try:
        import inspect
        import trainer as T
        src_now = inspect.getsource(T.generate_solution)
        has_aligned_headroom = (
            "headroom = action_tok_size + (per_state_tokens if use_world_model else 0) + 1" in src_now
        )
        has_premature_sep_fix = (
            "Model emitted neither END nor a valid action-start" in src_now
        )
        if has_aligned_headroom and has_premature_sep_fix:
            print("  ✓ both fixes confirmed in patched source")
        else:
            print("  ⚠️  patches applied to disk but expected substrings missing on reimport")
            print(f"     aligned headroom: {has_aligned_headroom}")
            print(f"     premature sep fix: {has_premature_sep_fix}")
    except Exception as e:
        print(f"  ⚠️  verification raised: {e}")
        print(f"  Revert from {backup} if needed.")

    print()
    print("Suggested next step: re-run the four-cell Blocks World in-distribution")
    print("evaluation to confirm WM-oracle and WM-model now agree (within sampling")
    print("noise; ideally exactly when state predictions are accurate).")


if __name__ == "__main__":
    main()
