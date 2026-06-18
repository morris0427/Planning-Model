#!/usr/bin/env python3
"""
Patch trainer.py to add state_source flag to generate_solution.

This patch does NOT walk the file or parse Python. It uses inspect.getsource()
to obtain the byte-exact current text of generate_solution from the live
module, then does a single literal string replace in trainer.py. This avoids
the bugs that a function-locator helper introduced in an earlier attempt.

USAGE
-----
    python3 fix_generate_solution_state_source.py --dry-run
    python3 fix_generate_solution_state_source.py
"""

import argparse
import shutil
import sys
import time
from pathlib import Path


NEW_TEXT = '''def generate_solution(
    model: nn.Module,
    problem: Dict[str, Any],
    dataset_generator,
    device: torch.device,
    max_length: int = 50,
    return_info: bool = False,
    state_source: str = "oracle",
):
    """
    Generate a solution for a problem using the model.

    For baseline models: generates action tokens autoregressively. The
    state_source parameter is ignored (baseline sequences contain no
    per-step state tokens).

    For WM models:
      - state_source="oracle": after each action, the oracle computes
        the true post-action state and appends its encoded form to the
        context. The model's own state-token predictions are bypassed.
        Keeps the context grounded throughout the trajectory.
      - state_source="model": the model emits action AND state tokens
        autoregressively. The model's state predictions become its own
        input context for subsequent predictions.

    Args:
        model: Trained model
        problem: Problem dictionary with 'sequence' and 'num_moves'
        dataset_generator: Dataset instance for vocab/oracle access
        device: Device
        max_length: Maximum number of moves to generate
        return_info: If True, also return diagnostics dict
        state_source: "oracle" or "model" (WM only; ignored for baselines)

    Returns:
        Generated token list, or (tokens, info) if return_info=True.
        info_dict termination values:
          'solved_sep'        - reached goal, emitted SEP/END
          'sep'               - emitted SEP/END without solving
          'truncation_seqlen' - hit positional embedding capacity
          'invalid_move'      - generated a move illegal in current state
          'max_steps'         - WM wasted-prediction safety cap
          'max_length'        - hit max_length move cap / loop end
    """
    if state_source not in ("oracle", "model"):
        raise ValueError(
            f"state_source must be 'oracle' or 'model', got {state_source!r}"
        )

    model.eval()
    model_max_seq_length = model.max_seq_length

    sequence = problem['sequence']
    num_moves = problem['num_moves']
    domain_name = dataset_generator.__class__.__name__
    use_world_model = dataset_generator.use_world_model

    termination = 'max_length'
    moves_generated = 0

    is_eight_puzzle = 'EightPuzzle' in domain_name

    # ----- Domain-specific setup -----
    if is_eight_puzzle:
        state_length = 20  # 1 + 9 + 1 + 9
        end_token = 14     # SEP
        per_state_tokens = 9
        start_state = np.array(sequence[1:10]).reshape(3, 3)
        goal_state = np.array(sequence[11:20]).reshape(3, 3)
        move_tokens = {10: 'up', 11: 'down', 12: 'left', 13: 'right'}

        def oracle_step(cur_state, action_pieces):
            return apply_move_8puzzle(cur_state, move_tokens[action_pieces[0]])

        def encode_state(cur_state):
            return cur_state.flatten().tolist()

        def states_equal(a, b):
            return np.array_equal(a, b)

        def is_action_start_tok(tok):
            return tok in move_tokens

        action_tok_size = 1  # one token per move
    else:
        # Blocks World
        state_length = 17  # START(1) + start_state(8) + goal_state(8)
        end_token = 1      # END
        per_state_tokens = 8
        start_state = dataset_generator._decode_state(sequence[1:9])
        goal_state = dataset_generator._decode_state(sequence[9:17])

        BW_BLOCK_TOKENS = {2: 'A', 3: 'B', 4: 'C', 5: 'D'}
        BW_POS_TOKENS = {6: 0, 7: 1, 8: 2, 9: 3}

        def oracle_step(cur_state, action_pieces):
            block_tok, pos_tok = action_pieces
            if block_tok not in BW_BLOCK_TOKENS or pos_tok not in BW_POS_TOKENS:
                return None
            return dataset_generator.apply_action(
                cur_state, (BW_BLOCK_TOKENS[block_tok], BW_POS_TOKENS[pos_tok])
            )

        def encode_state(cur_state):
            return dataset_generator._encode_state(cur_state)

        def states_equal(a, b):
            return a == b

        def is_action_start_tok(tok):
            return tok in BW_BLOCK_TOKENS

        action_tok_size = 2  # block + pos

    # Initial context (start + goal)
    generated = sequence[:state_length].copy()

    # ----- WM + oracle path -----
    if use_world_model and state_source == "oracle":
        if is_eight_puzzle:
            current_state = start_state.copy()
        else:
            current_state = [tower[:] for tower in start_state]

        steps = 0
        max_steps = max_length * 20
        termination = 'max_length'

        with torch.no_grad():
            while moves_generated < max_length and steps < max_steps:
                steps += 1

                headroom = action_tok_size + per_state_tokens + 1
                if len(generated) >= model_max_seq_length - headroom:
                    termination = 'truncation_seqlen'
                    break

                # Collect a complete action (1 token for 8-puzzle, 2 for BW)
                action_pieces = []
                premature_sep = False
                while len(action_pieces) < action_tok_size:
                    input_seq = torch.tensor([generated], dtype=torch.long).to(device)
                    logits = model(input_seq)
                    next_tok = int(torch.argmax(logits[0, -1, :]).item())

                    if not action_pieces and next_tok == end_token:
                        # Model decided to stop right before producing an action
                        premature_sep = True
                        break

                    if not action_pieces and not is_action_start_tok(next_tok):
                        # Skip non-action token; appending it lets the loop
                        # re-evaluate context at next iteration. Bounded by max_steps.
                        generated.append(next_tok)
                        break

                    generated.append(next_tok)
                    action_pieces.append(next_tok)

                if premature_sep:
                    termination = 'sep'
                    break
                if len(action_pieces) < action_tok_size:
                    # Didn't finish a full action this pass; iterate again
                    continue

                # Oracle step
                next_state = oracle_step(current_state, action_pieces)
                if next_state is None:
                    termination = 'invalid_move'
                    break

                current_state = next_state
                moves_generated += 1

                # Inject the oracle-encoded post-action state into context.
                # This replaces whatever state tokens the model would have
                # produced.
                generated.extend(encode_state(current_state))

                if states_equal(current_state, goal_state):
                    generated.append(end_token)
                    termination = 'solved_sep'
                    break
            else:
                termination = 'max_steps' if steps >= max_steps else 'max_length'

    # ----- Pure-model path (baseline OR WM with state_source="model") -----
    else:
        termination = 'max_length'
        stride = action_tok_size + (per_state_tokens if use_world_model else 0)
        with torch.no_grad():
            # Bound the per-token loop so a WM emitting nothing useful still terminates
            for _ in range(max_length * max(1, stride + 1)):
                if len(generated) >= model_max_seq_length - 1:
                    termination = 'truncation_seqlen'
                    break

                input_seq = torch.tensor([generated], dtype=torch.long).to(device)
                logits = model(input_seq)
                next_tok = int(torch.argmax(logits[0, -1, :]).item())
                generated.append(next_tok)

                if next_tok == end_token:
                    termination = 'sep'
                    break

                approx_moves = max(0, (len(generated) - state_length)) // max(1, stride)
                if approx_moves >= max_length:
                    moves_generated = approx_moves
                    termination = 'max_length'
                    break
                moves_generated = approx_moves

    if return_info:
        info = {
            'termination': termination,
            'moves_generated': moves_generated,
            'final_len': len(generated),
            'model_max_seq_length': model_max_seq_length,
            'state_source': state_source if use_world_model else 'n/a',
        }
        return generated, info
    return generated
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    target = Path("trainer.py")
    if not target.exists():
        print(f"ERROR: {target} not found. Run from the experiments/ directory.")
        sys.exit(1)

    # Read current generate_solution byte-exactly from the live module
    sys.path.insert(0, ".")
    for mod in list(sys.modules):
        if mod == "trainer" or mod.startswith("trainer."):
            del sys.modules[mod]
    try:
        import inspect
        import trainer as T
        OLD_TEXT = inspect.getsource(T.generate_solution)
    except Exception as e:
        print(f"ERROR: could not read current generate_solution via inspect: {e}")
        sys.exit(2)

    if "state_source" in OLD_TEXT:
        print("Already patched: current generate_solution contains state_source.")
        sys.exit(0)

    src = target.read_text()
    n = src.count(OLD_TEXT)
    if n == 0:
        print("ERROR: the byte-exact text from inspect.getsource was NOT found")
        print("       as a substring of trainer.py. This is unexpected.")
        print(f"       inspect.getsource length: {len(OLD_TEXT)}")
        print(f"       trainer.py length:        {len(src)}")
        sys.exit(3)
    if n > 1:
        print(f"ERROR: found {n} occurrences of generate_solution body in trainer.py.")
        print("       Refusing to patch ambiguously.")
        sys.exit(4)

    print(f"Found 1 occurrence of generate_solution (length {len(OLD_TEXT)} bytes).")
    print(f"Replacing with new version (length {len(NEW_TEXT)} bytes).")

    new_src = src.replace(OLD_TEXT, NEW_TEXT)

    if args.dry_run:
        print()
        print("=== first 30 lines of NEW function ===")
        for line in NEW_TEXT.splitlines()[:30]:
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

    # Verify the new function parses
    print()
    print("Verifying by reimporting...")
    for mod in list(sys.modules):
        if mod == "trainer" or mod.startswith("trainer."):
            del sys.modules[mod]
    try:
        import inspect
        import trainer as T2
        new_src_check = inspect.getsource(T2.generate_solution)
        ok = "state_source" in new_src_check and "oracle_step" in new_src_check
        if ok:
            print("  ✓ new generate_solution contains state_source and oracle_step logic")
        else:
            print("  ⚠️  reimport succeeded but expected substrings missing")
    except Exception as e:
        print(f"  ⚠️  verification raised: {e}")
        print(f"  The patch is on disk. If the import error is in the new function,")
        print(f"  revert from {backup}.")

    print()
    print("Next steps (inference only, no retraining):")
    print("  1. Sanity check: BW WM in-distribution with state_source='oracle'")
    print("     should match the previous 100% (model's state predictions were")
    print("     already correct in-distribution).")
    print("  2. Re-run: BW WM productivity with state_source='oracle'")
    print("     This is the experiment that tests the validity-collapse hypothesis.")
    print()
    print("  These require a small script that calls generate_solution directly")
    print("  with state_source='oracle' (evaluate_solve_rate doesn't pass it yet).")


if __name__ == "__main__":
    main()
