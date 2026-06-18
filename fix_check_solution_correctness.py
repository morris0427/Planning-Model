#!/usr/bin/env python3
"""
Fix check_solution_correctness for Blocks World.

WHY
---
The current Blocks World branch of check_solution_correctness has two
problems:

  1. Its arithmetic for `state_length` assumes the BASELINE sequence layout
     (only START + start_state + goal_state in the prefix). For WM
     sequences, which interleave 8-token state blocks with 2-token actions,
     the arithmetic computes a meaningless offset and the function ends up
     comparing arbitrary slices of state tokens / END markers.

  2. Even if the arithmetic were right, the check is byte-equality against
     the reference plan. That fails for valid plans that don't match the
     reference token-for-token, e.g. plans that use a different shortest
     path, or plans whose state predictions differ slightly from the
     reference's state blocks (which doesn't affect semantic correctness).

WHAT THIS DOES
--------------
Replaces the Blocks World branch with a semantic check that mirrors the
8-puzzle branch:
  - Decode start and goal from the reference sequence's first two state
    blocks (the lossless 8-token encoding).
  - Walk the generated tokens past the 17-token context, extracting
    action pairs (block, position). Skip state blocks under WM layout.
  - Apply each action via dataset_generator.apply_action(). An illegal
    action returns False.
  - After consuming all actions, return whether the resulting state
    equals the goal.

This means a generated trajectory is counted as solving the problem if
and only if its action sequence, applied from the start state, reaches
the goal state.

The 8-puzzle branch is unchanged.

USAGE
-----
    python3 fix_check_solution_correctness.py --dry-run
    python3 fix_check_solution_correctness.py
"""

import argparse
import shutil
import sys
import time
from pathlib import Path


NEW_TEXT = '''def check_solution_correctness(
    generated_tokens: list[int],
    problem: Dict[str, Any],
    dataset_generator
) -> bool:
    """
    Check if generated solution actually solves the problem.

    For 8-Puzzle: applies moves to the start state and checks goal reach.
    For Blocks World: applies actions to the start state and checks goal reach
                      (semantic correctness, not byte-equality with reference).

    A generated trajectory is correct iff its action sequence is legal at
    every step and the final state equals the goal. State predictions in
    the generated tokens (under WM) are NOT required to match the reference;
    only the actions must lead to the goal.

    Args:
        generated_tokens: Generated token sequence
        problem: Original problem with ground truth sequence
        dataset_generator: Dataset for decoding/applying actions

    Returns:
        True if solution is correct
    """
    try:
        gt_sequence = problem['sequence']
        num_moves = problem['num_moves']
        domain_name = dataset_generator.__class__.__name__

        if 'EightPuzzle' in domain_name:
            # 8-puzzle layout: [dummy(1), start_state(9), PAD(1), goal_state(9), moves, SEP]
            start_state = np.array(gt_sequence[1:10]).reshape(3, 3)
            goal_state = np.array(gt_sequence[11:20]).reshape(3, 3)

            move_tokens = {10: 'up', 11: 'down', 12: 'left', 13: 'right'}
            gen_moves = []
            for token in generated_tokens[20:]:
                if token == 14:  # SEP
                    break
                if token in move_tokens:
                    gen_moves.append(move_tokens[token])

            current_state = start_state.copy()
            for move in gen_moves:
                blank_pos = np.argwhere(current_state == 0)
                if len(blank_pos) == 0:
                    return False
                row, col = blank_pos[0]
                if move == 'up' and row > 0:
                    current_state[row, col], current_state[row-1, col] = \\
                        current_state[row-1, col], current_state[row, col]
                elif move == 'down' and row < 2:
                    current_state[row, col], current_state[row+1, col] = \\
                        current_state[row+1, col], current_state[row, col]
                elif move == 'left' and col > 0:
                    current_state[row, col], current_state[row, col-1] = \\
                        current_state[row, col-1], current_state[row, col]
                elif move == 'right' and col < 2:
                    current_state[row, col], current_state[row, col+1] = \\
                        current_state[row, col+1], current_state[row, col]
                else:
                    return False
            return np.array_equal(current_state, goal_state)

        else:
            # Blocks World layout (uniform encoding):
            #   [START(1), start_state(8), goal_state(8), then per-step blocks, END(1)]
            #   Baseline per-step: action(2 tokens)
            #   WM per-step:       action(2 tokens) + state(8 tokens)
            CONTEXT_LEN = 17     # START + start_state + goal_state
            STATE_LEN = 8
            ACTION_LEN = 2
            END_TOKEN = 1
            BLOCK_TOKS = {2: 'A', 3: 'B', 4: 'C', 5: 'D'}
            POS_TOKS = {6: 0, 7: 1, 8: 2, 9: 3}

            if len(generated_tokens) < CONTEXT_LEN + ACTION_LEN:
                # Didn't generate even one action
                return False

            # Decode start and goal from the reference sequence's lossless
            # state blocks. Note: we use the reference (ground truth)
            # sequence's context, not the generated tokens', since the
            # context tokens are always set from the problem.
            try:
                start_state = dataset_generator._decode_state(
                    gt_sequence[1:1 + STATE_LEN]
                )
                goal_state = dataset_generator._decode_state(
                    gt_sequence[1 + STATE_LEN:CONTEXT_LEN]
                )
            except Exception:
                # If decoding fails, fall back to declaring incorrect
                return False

            use_wm = dataset_generator.use_world_model
            stride = ACTION_LEN + (STATE_LEN if use_wm else 0)

            current_state = [tower[:] for tower in start_state]
            pos = CONTEXT_LEN

            while pos < len(generated_tokens):
                if generated_tokens[pos] == END_TOKEN:
                    break
                # Need at least an action pair from here
                if pos + ACTION_LEN > len(generated_tokens):
                    return False
                block_tok = generated_tokens[pos]
                pos_tok = generated_tokens[pos + 1]
                if block_tok not in BLOCK_TOKS or pos_tok not in POS_TOKS:
                    # Malformed action (state tokens in an action slot, etc.)
                    return False
                action = (BLOCK_TOKS[block_tok], POS_TOKS[pos_tok])
                next_state = dataset_generator.apply_action(current_state, action)
                if next_state is None:
                    # Illegal action (block not on top of any tower)
                    return False
                current_state = next_state
                pos += stride

            return current_state == goal_state

    except Exception:
        # If any error occurs in checking, declare incorrect rather than
        # raising. This matches the previous function's behavior.
        return False
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    target = Path("trainer.py")
    if not target.exists():
        print(f"ERROR: {target} not found. Run from the experiments/ directory.")
        sys.exit(1)

    # Read current function byte-exactly from the live module
    sys.path.insert(0, ".")
    for mod in list(sys.modules):
        if mod == "trainer" or mod.startswith("trainer."):
            del sys.modules[mod]
    try:
        import inspect
        import trainer as T
        OLD_TEXT = inspect.getsource(T.check_solution_correctness)
    except Exception as e:
        print(f"ERROR: could not read current function via inspect: {e}")
        sys.exit(2)

    # Idempotency check: did the new logic already land?
    if "_decode_state" in OLD_TEXT and "apply_action" in OLD_TEXT:
        print("Already patched: current check_solution_correctness uses _decode_state and apply_action.")
        sys.exit(0)

    src = target.read_text()
    n = src.count(OLD_TEXT)
    if n == 0:
        print("ERROR: byte-exact text from inspect.getsource was NOT found in trainer.py.")
        print(f"       inspect length: {len(OLD_TEXT)}")
        print(f"       trainer length: {len(src)}")
        sys.exit(3)
    if n > 1:
        print(f"ERROR: {n} occurrences found (expected 1). Refusing to patch ambiguously.")
        sys.exit(4)

    print(f"Found 1 occurrence (length {len(OLD_TEXT)} bytes).")
    print(f"Replacing with new version (length {len(NEW_TEXT)} bytes).")

    new_src = src.replace(OLD_TEXT, NEW_TEXT)

    if args.dry_run:
        print()
        print("=== first 40 lines of new function ===")
        for line in NEW_TEXT.splitlines()[:40]:
            print(line)
        print("...")
        print()
        print(f"trainer.py would change from {len(src)} to {len(new_src)} bytes.")
        print("(dry run; not modifying)")
        sys.exit(0)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    backup = target.with_suffix(f".py.bak-{timestamp}")
    shutil.copy2(target, backup)
    print(f"Backed up to {backup}")

    target.write_text(new_src)
    print(f"Patched {target}.")

    # Verify by reimporting
    print()
    print("Verifying by reimporting...")
    for mod in list(sys.modules):
        if mod == "trainer" or mod.startswith("trainer."):
            del sys.modules[mod]
    try:
        import inspect
        import trainer as T2
        new_src_check = inspect.getsource(T2.check_solution_correctness)
        ok = ("_decode_state" in new_src_check
              and "apply_action" in new_src_check
              and "current_state == goal_state" in new_src_check)
        if ok:
            print("  ✓ check_solution_correctness now uses semantic correctness")
        else:
            print("  ⚠️  reimport succeeded but expected substrings missing")
    except Exception as e:
        print(f"  ⚠️  verification raised: {e}")
        print(f"  Revert from {backup} if needed.")

    print()
    print("Next: re-run evaluation on existing Blocks World WM checkpoints to see")
    print("how much of the prior solve-rate numbers was reference-reproduction vs.")
    print("actual goal-reaching:")
    print()
    print("  python3 - <<'PY'")
    print("  # ... build model, load test_problems ...")
    print("  # for state_source in ['oracle', 'model']:")
    print("  #     solved = sum(check_solution_correctness(")
    print("  #         generate_solution(model, p, ..., state_source=state_source),")
    print("  #         p, dataset_generator)")
    print("  #     for p in test_problems[:100])")
    print("  #     print(f'{state_source}: {solved}/100')")
    print("  PY")


if __name__ == "__main__":
    main()
