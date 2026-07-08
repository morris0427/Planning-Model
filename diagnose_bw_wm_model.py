"""
diagnose_bw_wm_model.py

Positive control for the WM-model diagnostic: same as
diagnose_8p_wm_model.py but for Blocks World, where we expect the
state predictions to be valid and the solve rate to be ~79%.

If 8-puzzle WM-model shows all-invalid state blocks and 0% solve rate,
this script should show all-valid state blocks and ~5/5 problems solved.
The asymmetry confirms that the 8-puzzle failure is a real
auxiliary-task-learnability finding, not an artifact of the eval
pipeline.

Run from the experiments/ directory:
    python3 diagnose_bw_wm_model.py
"""

import sys
sys.path.insert(0, ".")

import json
from pathlib import Path

import torch

for mod in list(sys.modules):
    if mod.startswith("data") or mod == "trainer" or mod.startswith("trainer."):
        del sys.modules[mod]

from config import ModelPresets, DataPresets
from data.base import DatasetFactory
from data.blocks_world import BlocksWorldDataset
import trainer as T


# ============================================================
# Setup
# ============================================================

# Prefer the in-distribution WM checkpoint for consistency with the
# 8-puzzle diagnostic; fall back to productivity if absent.
ckpt_path = Path("results/blocks_world_in_distribution_medium_wm/best_model.pth")
if not ckpt_path.exists():
    ckpt_path = Path("results/blocks_world_medium_wm_productivity/best_model.pth")
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

# Load test cache matching the checkpoint
if "in_distribution" in str(ckpt_path):
    test_cache_path = "cached_data/blocks_world_test_wm.json"
else:
    test_cache_path = "cached_data/blocks_world_test_wm_productivity.json"

with open(test_cache_path) as f:
    test_problems = json.load(f)

print(f"Loaded test cache: {test_cache_path} ({len(test_problems)} problems)")

# Determine difficulty range from path
if "in_distribution" in str(ckpt_path):
    data_cfg = DataPresets.blocks_world_standard()
else:
    data_cfg = DataPresets.blocks_world_productivity()

test_gen = DatasetFactory.create(
    domain="blocks_world",
    difficulty_range=data_cfg.test_difficulty_range,
    num_samples=len(test_problems),
    use_world_model=True,
)
test_gen.problems = test_problems

# Helper dataset for decoding states
ds_helper = BlocksWorldDataset(
    difficulty_range=(3, 3), num_samples=1, use_world_model=True, seed=0,
)


# ============================================================
# Diagnostic logic
# ============================================================

CONTEXT_LEN = 17  # START(1) + start_state(8) + goal_state(8)
ACTION_LEN = 2
STATE_LEN = 8
END_TOKEN = 1
BLOCK_TOKENS = {2: "A", 3: "B", 4: "C", 5: "D"}
POS_TOKENS = {6: 0, 7: 1, 8: 2, 9: 3}


def classify_state_block(block):
    """Return (is_valid, description) for an 8-token Blocks World state block.

    Valid encoding: tokens at positions 0, 2, 4, 6 are block tokens (2-5)
    and tokens at positions 1, 3, 5, 7 are POS tokens (6-9). Additionally,
    when decoded, each block letter (A, B, C, D) must appear exactly once.
    """
    if len(block) != 8:
        return False, f"length {len(block)} (expected 8)"

    # Structural check: alternating block/POS tokens
    for i, t in enumerate(block):
        if i % 2 == 0 and t not in BLOCK_TOKENS:
            return False, f"position {i} is {t}, expected block token (2-5)"
        if i % 2 == 1 and t not in POS_TOKENS:
            return False, f"position {i} is {t}, expected POS token (6-9)"

    # Semantic check: decode and verify all blocks appear exactly once
    try:
        decoded = ds_helper._decode_state(block)
    except Exception as e:
        return False, f"_decode_state raised: {e}"

    all_blocks = []
    for tower in decoded:
        all_blocks.extend(tower)
    if sorted(all_blocks) != ['A', 'B', 'C', 'D']:
        return False, f"decoded blocks {sorted(all_blocks)} (expected ['A','B','C','D'])"

    return True, f"valid; decodes to {decoded}"


def walk_actions_independently(start_state, actions, goal_state):
    """Apply actions to start using BlocksWorldDataset.apply_action,
    independently of check_solution_correctness."""
    cur = [t[:] for t in start_state]
    n_valid = 0
    illegal_at = None
    for i, (block_letter, pos_idx) in enumerate(actions):
        nxt = ds_helper.apply_action(cur, (block_letter, pos_idx))
        if nxt is None:
            illegal_at = i
            break
        cur = nxt
        n_valid += 1
    return cur, cur == goal_state, n_valid, illegal_at


def diagnose_problem(idx):
    p = test_problems[idx]
    seq = p["sequence"]

    # Decode start/goal from the test sequence
    start = ds_helper._decode_state(seq[1:9])
    goal = ds_helper._decode_state(seq[9:17])

    print()
    print("=" * 78)
    print(f"Problem {idx}, ref num_moves = {p['num_moves']}")
    print("=" * 78)
    print(f"  Start state: {start}")
    print(f"  Goal state:  {goal}")

    # Run with state_source="model"
    gen, info = T.generate_solution(
        model, p, test_gen, torch.device("cpu"),
        max_length=100, return_info=True, state_source="model",
    )

    print(f"\n  Generated sequence ({len(gen)} tokens), termination={info['termination']}:")
    print(f"    context (first 17): {gen[:17]}")
    print(f"    rest:               {gen[17:]}")

    # Walk through gen tokens past context, extracting actions and state blocks
    pos = CONTEXT_LEN
    step = 0
    actions = []
    state_blocks = []
    issues = []

    while pos < len(gen):
        if gen[pos] == END_TOKEN:
            issues.append(f"step {step+1}: END at action slot (model terminating)")
            break

        # Expect 2-token action
        if pos + 1 >= len(gen):
            issues.append(f"step {step+1}: truncated mid-action")
            break

        if gen[pos] not in BLOCK_TOKENS:
            issues.append(f"step {step+1}: unexpected token {gen[pos]} at action slot")
            break

        if gen[pos + 1] not in POS_TOKENS:
            issues.append(f"step {step+1}: action {gen[pos]} but no valid POS token")
            break

        block_letter = BLOCK_TOKENS[gen[pos]]
        pos_idx = POS_TOKENS[gen[pos + 1]]
        actions.append((block_letter, pos_idx))

        # After action, expect 8 state tokens
        if pos + 2 + STATE_LEN <= len(gen):
            block = gen[pos + 2:pos + 2 + STATE_LEN]
            state_blocks.append(block)
        else:
            issues.append(f"step {step+1}: truncated before full state block")
            break

        pos += 2 + STATE_LEN
        step += 1

    # Report actions and states
    print(f"\n  Decomposed into {len(actions)} action+state pairs:")
    for i, ((block_letter, pos_idx), block) in enumerate(zip(actions, state_blocks)):
        valid, desc = classify_state_block(block)
        verdict = "✓ VALID" if valid else "✗ INVALID"
        print(f"    step {i+1}: action=({block_letter}, {pos_idx})")
        print(f"             state_block={block}  --  {verdict}: {desc}")

    if issues:
        print(f"\n  Termination/parsing issues:")
        for s in issues:
            print(f"    {s}")

    # Apply actions independently
    final, reached, n_valid, illegal_at = walk_actions_independently(start, actions, goal)

    print(f"\n  Independent walk through actions (using apply_action):")
    print(f"    final state after {n_valid} valid steps: {final}")
    if illegal_at is not None:
        print(f"    -> action {illegal_at+1} was ILLEGAL")
    print(f"    reached goal? {reached}")

    # check_solution_correctness verdict
    cs_solved = T.check_solution_correctness(gen, p, test_gen)
    print(f"\n  check_solution_correctness verdict: {cs_solved}")

    # Reconciliation
    print(f"\n  Reconciliation:")
    print(f"    independent walk says reached_goal = {reached}")
    print(f"    check_solution_correctness says   = {cs_solved}")
    if reached != cs_solved:
        print(f"    >>> DISAGREEMENT — bug in either walk or check.")
    else:
        print(f"    agreement.")

    invalid_count = sum(1 for b in state_blocks if not classify_state_block(b)[0])
    if state_blocks:
        if invalid_count == 0:
            print(f"\n  STATE VALIDITY: all {len(state_blocks)} state blocks valid.")
        else:
            print(f"\n  STATE VALIDITY: {invalid_count}/{len(state_blocks)} state blocks invalid.")
    else:
        print(f"\n  STATE VALIDITY: no state blocks emitted.")

    return cs_solved


# ============================================================
# Run diagnostic
# ============================================================

print()
print("=" * 78)
print("Blocks World WM-model state_source diagnostic")
print("(positive control: expect valid state blocks and ~79% solve rate)")
print("=" * 78)

results = []
for idx in [0, 1, 2, 3, 4]:
    solved = diagnose_problem(idx)
    results.append(solved)

print()
print("=" * 78)
print("SUMMARY (5 problems, in-distribution Blocks World WM-model)")
print("=" * 78)
print(f"  Solved: {sum(results)}/5")
print()
print("Expected: ~4/5 solved, all state blocks valid. If we instead see")
print("~0/5 or many invalid state blocks, the Blocks World WM-model")
print("result is not what we thought, and the asymmetry claim between")
print("Blocks World and 8-puzzle would need revisiting.")
print()
print("To compare: rerun diagnose_8p_wm_model.py and check whether 8-puzzle")
print("state blocks are invalid (expected) and solve rate is ~0/5 (expected).")
