"""
diagnose_8p_wm_model.py

Diagnostic for the 8-puzzle WM-model state-source = 0% result.

We want to determine whether the 0% solve rate reflects:
  (a) Genuine model failure: state predictions are invalid (e.g., a
      permutation of 9 tile values with duplicates or missing tiles),
      so the model's autoregressive rollout drifts off the reachable
      state manifold and never reaches the goal.
  (b) Implementation artifact: state predictions are actually valid, but
      something in check_solution_correctness or the state_source='model'
      branch of generate_solution treats them as failures incorrectly.

The diagnostic: take the in-distribution 8-puzzle WM checkpoint, run 5
problems with state_source='model', and for each problem print:
  - The full generated token sequence
  - Every model-emitted state block, with a verdict (valid permutation
    of {0..8}, or describing the malformation)
  - The model-emitted action tokens
  - Whether check_solution_correctness says it solved
  - Whether the actions, applied to start via apply_move_8puzzle, reach
    the goal (independent of check_solution_correctness)

This separates "model produces invalid output" from "model produces
valid output but our eval code calls it wrong."

Run from the experiments/ directory:
    python3 diagnose_8p_wm_model.py
"""

import sys
sys.path.insert(0, ".")

import json
from pathlib import Path

import numpy as np
import torch

for mod in list(sys.modules):
    if mod.startswith("data") or mod == "trainer" or mod.startswith("trainer."):
        del sys.modules[mod]

from config import ModelPresets, DataPresets
from data.base import DatasetFactory
import trainer as T


# ============================================================
# Setup
# ============================================================

# Load 8-puzzle in-distribution WM checkpoint
ckpt_path = Path("results/eight_puzzle_in_distribution_medium_wm/best_model.pth")
if not ckpt_path.exists():
    # Fall back to productivity checkpoint if in-distribution is not present
    ckpt_path = Path("results/eight_puzzle_medium_wm_productivity/best_model.pth")
    print(f"In-distribution checkpoint not found; using productivity: {ckpt_path}")

state = torch.load(ckpt_path, map_location="cpu")
max_seq, d_model = state["pos_encoder.weight"].shape
vocab = state["embedding.weight"].shape[0]
m_cfg = ModelPresets.medium(use_world_model=True)
model = T.PlanningTransformer(
    vocab_size=vocab, d_model=d_model, nhead=m_cfg.n_heads,
    num_layers=m_cfg.n_layers, dim_feedforward=m_cfg.d_ff,
    max_seq_length=max_seq,
)
model.load_state_dict(state)
model.eval()

print(f"Loaded checkpoint: {ckpt_path}")
print(f"  vocab={vocab}, d_model={d_model}, max_seq={max_seq}")

# Load test cache
test_cache_path = "cached_data/eight_puzzle_test_wm.json"
if not Path(test_cache_path).exists():
    test_cache_path = "cached_data/eight_puzzle_test_wm_productivity.json"
    print(f"In-distribution cache not found; using productivity: {test_cache_path}")

with open(test_cache_path) as f:
    test_problems = json.load(f)

print(f"Loaded test cache: {test_cache_path} ({len(test_problems)} problems)")

data_cfg = DataPresets.eight_puzzle_standard()
test_gen = DatasetFactory.create(
    domain="eight_puzzle",
    difficulty_range=data_cfg.test_difficulty_range,
    num_samples=len(test_problems),
    use_world_model=True,
)
test_gen.problems = test_problems


# ============================================================
# Diagnostic logic
# ============================================================

CONTEXT_LEN = 20      # dummy(1) + start(9) + PAD(1) + goal(9)
ACTION_LEN = 1
STATE_LEN = 9
SEP_TOKEN = 14
MOVE_TOKENS = {10: "up", 11: "down", 12: "left", 13: "right"}
TILE_RANGE = set(range(9))  # valid state tokens are {0..8}


def classify_state_block(block):
    """Return (is_valid, description) for a 9-token state block.

    A valid 8-puzzle state is a permutation of {0..8}. We check:
      - Length is exactly 9
      - All tokens are in {0..8}
      - Each token appears exactly once
    """
    if len(block) != 9:
        return False, f"length {len(block)} (expected 9)"
    out_of_range = [t for t in block if t not in TILE_RANGE]
    if out_of_range:
        return False, f"contains out-of-range tokens {out_of_range}"
    from collections import Counter
    counts = Counter(block)
    duplicates = [v for v, c in counts.items() if c > 1]
    missing = [v for v in TILE_RANGE if v not in counts]
    if duplicates or missing:
        parts = []
        if duplicates:
            parts.append(f"duplicates {duplicates}")
        if missing:
            parts.append(f"missing {missing}")
        return False, "; ".join(parts)
    return True, "valid permutation"


def walk_actions_independently(start_state, action_moves, goal_state):
    """Apply the action sequence to start, using apply_move_8puzzle independently
    of the check_solution_correctness path. Return (final_state, reached_goal,
    n_valid_steps, illegal_at_step).
    """
    cur = start_state.copy()
    n_valid = 0
    illegal_at = None
    for i, move in enumerate(action_moves):
        nxt = T.apply_move_8puzzle(cur, move)
        if nxt is None:
            illegal_at = i
            break
        cur = nxt
        n_valid += 1
    return cur, np.array_equal(cur, goal_state), n_valid, illegal_at


def diagnose_problem(idx):
    p = test_problems[idx]
    seq = p["sequence"]

    # Extract start/goal from the test sequence
    start = np.array(seq[1:10]).reshape(3, 3)
    goal = np.array(seq[11:20]).reshape(3, 3)

    print()
    print("=" * 78)
    print(f"Problem {idx}, ref num_moves = {p['num_moves']}")
    print("=" * 78)
    print(f"  Start state:")
    print(f"    {start[0].tolist()}")
    print(f"    {start[1].tolist()}")
    print(f"    {start[2].tolist()}")
    print(f"  Goal state:")
    print(f"    {goal[0].tolist()}")
    print(f"    {goal[1].tolist()}")
    print(f"    {goal[2].tolist()}")

    # Run with state_source="model"
    gen, info = T.generate_solution(
        model, p, test_gen, torch.device("cpu"),
        max_length=100, return_info=True, state_source="model",
    )

    print(f"\n  Generated sequence ({len(gen)} tokens), termination={info['termination']}:")
    print(f"    context (first 20): {gen[:20]}")
    print(f"    rest:              {gen[20:]}")

    # Walk through gen tokens past context, extracting actions and state blocks
    pos = CONTEXT_LEN
    step = 0
    actions = []
    state_blocks = []
    issues = []

    while pos < len(gen):
        if gen[pos] == SEP_TOKEN:
            issues.append(f"step {step+1}: SEP at action slot (model wants to terminate)")
            break
        if gen[pos] in MOVE_TOKENS:
            actions.append((MOVE_TOKENS[gen[pos]], gen[pos]))
            # After action, expect 9 state tokens
            if pos + 1 + STATE_LEN <= len(gen):
                block = gen[pos + 1:pos + 1 + STATE_LEN]
                state_blocks.append(block)
            else:
                issues.append(f"step {step+1}: truncated before full state block")
                break
            pos += 1 + STATE_LEN
            step += 1
        else:
            issues.append(f"step {step+1}: unexpected token {gen[pos]} at action slot")
            break

    # Report actions and states
    print(f"\n  Decomposed into {len(actions)} action+state pairs:")
    for i, ((move_name, move_tok), block) in enumerate(zip(actions, state_blocks)):
        valid, desc = classify_state_block(block)
        verdict = "✓ VALID" if valid else "✗ INVALID"
        print(f"    step {i+1}: action={move_name} (tok={move_tok})")
        print(f"             state_block={block}  --  {verdict}: {desc}")

    if issues:
        print(f"\n  Termination/parsing issues:")
        for s in issues:
            print(f"    {s}")

    # Apply actions independently and see what happens
    move_list = [name for name, tok in actions]
    final, reached, n_valid, illegal_at = walk_actions_independently(start, move_list, goal)

    print(f"\n  Independent walk through actions (using apply_move_8puzzle):")
    print(f"    final state after {n_valid} valid steps:")
    print(f"      {final[0].tolist()}")
    print(f"      {final[1].tolist()}")
    print(f"      {final[2].tolist()}")
    if illegal_at is not None:
        print(f"    -> action {illegal_at+1} was ILLEGAL (move not applicable to state)")
    print(f"    reached goal? {reached}")

    # Compare against check_solution_correctness
    cs_solved = T.check_solution_correctness(gen, p, test_gen)
    print(f"\n  check_solution_correctness verdict: {cs_solved}")

    # Reconcile
    print(f"\n  Reconciliation:")
    print(f"    independent walk says reached_goal = {reached}")
    print(f"    check_solution_correctness says   = {cs_solved}")
    if reached != cs_solved:
        print(f"    >>> DISAGREEMENT — bug in either walk or check.")
    else:
        print(f"    agreement.")

    # Final verdict on validity of states
    invalid_count = sum(1 for b in state_blocks if not classify_state_block(b)[0])
    if invalid_count > 0:
        print(f"\n  STATE VALIDITY: {invalid_count}/{len(state_blocks)} state blocks invalid.")
    elif state_blocks:
        print(f"\n  STATE VALIDITY: all {len(state_blocks)} state blocks valid.")
    else:
        print(f"\n  STATE VALIDITY: no state blocks emitted (model terminated immediately).")


# ============================================================
# Run diagnostic
# ============================================================

print()
print("=" * 78)
print("Running 8-puzzle WM-model state_source diagnostic on 5 in-distribution")
print("problems")
print("=" * 78)

for idx in [0, 1, 2, 3, 4]:
    diagnose_problem(idx)

# ============================================================
# Summary
# ============================================================

print()
print("=" * 78)
print("SUMMARY")
print("=" * 78)
print("""
Interpreting the results:

  - If ALL state blocks across all 5 problems are invalid (duplicates,
    missing tiles, etc.), then the 8-puzzle WM-model 0% solve rate is
    GENUINELY due to the model failing to learn the auxiliary state-
    prediction task. The claim that the auxiliary task is unlearnable
    in 8-puzzle is supported.

  - If ALL state blocks are valid permutations of {0..8} but
    check_solution_correctness still calls these failures, the 0% is an
    EVAL BUG — the model is producing reasonable outputs but our scoring
    is wrong.

  - If state blocks are MIXED (some valid, some not), the model has
    partially learned but is unreliable. The claim shifts from "task is
    unlearnable" to "task is partially learnable but model-state-source
    failures dominate."

  - If 'independent walk' and 'check_solution_correctness' DISAGREE
    on the same problem, there's a bug in one of them and we need to
    investigate.

If the conclusion is "task is genuinely not learned" we can keep WM-model
as a column in the paper. If it's "eval bug" we need to fix it. If it's
"partially learned" we need to update the paper's framing.
""")
