"""
Re-evaluate the Blocks World BASELINE checkpoints under the corrected
check_solution_correctness, mirroring the WM re-evaluation.

Run from the experiments/ directory:
    python3 /tmp/eval_bw_baseline.py
(or wherever you save this)

Reports:
    Blocks World baseline, in-distribution: semantic solve rate /100
    Blocks World baseline, productivity:    semantic solve rate /100

Notes:
  - Baselines have no state tokens in their sequences. The state_source
    parameter is ignored for baselines; we don't pass it.
  - The DatasetFactory is constructed with use_world_model=False so the
    test_gen object has the correct flag, which check_solution_correctness
    reads to know which sequence layout to expect.
"""

import sys
sys.path.insert(0, ".")

import torch
from pathlib import Path

for mod in list(sys.modules):
    if mod.startswith("data") or mod == "trainer" or mod.startswith("trainer."):
        del sys.modules[mod]

from config import ModelPresets, DataPresets
from data.base import DatasetFactory
import trainer as T


def load_baseline_model(ckpt_path):
    """Load a baseline checkpoint. ModelPresets.medium(use_world_model=False)
    produces the architectural config; the state dict on disk is the
    trained weights."""
    state = torch.load(ckpt_path, map_location='cpu')
    max_seq, d_model = state["pos_encoder.weight"].shape
    vocab = state["embedding.weight"].shape[0]
    m_cfg = ModelPresets.medium(use_world_model=False)
    model = T.PlanningTransformer(
        vocab_size=vocab,
        d_model=d_model,
        nhead=m_cfg.n_heads,
        num_layers=m_cfg.n_layers,
        dim_feedforward=m_cfg.d_ff,
        max_seq_length=max_seq,
    )
    model.load_state_dict(state)
    model.eval()
    return model


def evaluate(model, test_problems, test_gen, n=100, max_length=100):
    """Run generation and semantic correctness check on n test problems."""
    solved = 0
    for p in test_problems[:n]:
        gen, info = T.generate_solution(
            model, p, test_gen, torch.device('cpu'),
            max_length=max_length, return_info=True,
        )
        if T.check_solution_correctness(gen, p, test_gen):
            solved += 1
    return solved


print("=" * 60)
print("Blocks World BASELINE re-evaluation")
print("=" * 60)
print()
print("=== IN-DISTRIBUTION ===")
data_cfg = DataPresets.blocks_world_standard()
class _Shim: pass
shim = _Shim()
shim.data = data_cfg
_, test_problems_id = T.load_cached_data(shim, use_wm=False)

test_gen_id = DatasetFactory.create(
    domain='blocks_world',
    difficulty_range=data_cfg.test_difficulty_range,
    num_samples=len(test_problems_id),
    use_world_model=False,
)
test_gen_id.problems = test_problems_id

ckpt_id = Path("results/blocks_world_in_distribution_medium_base/best_model.pth")
if not ckpt_id.exists():
    print("  NOT FOUND:", ckpt_id)
else:
    model_id = load_baseline_model(ckpt_id)
    solved = evaluate(model_id, test_problems_id, test_gen_id)
    print("  solve rate:", solved, "/100")

print()
print("=== PRODUCTIVITY ===")
data_cfg = DataPresets.blocks_world_productivity()
shim.data = data_cfg
_, test_problems_p = T.load_cached_data(shim, use_wm=False)

test_gen_p = DatasetFactory.create(
    domain='blocks_world',
    difficulty_range=data_cfg.test_difficulty_range,
    num_samples=len(test_problems_p),
    use_world_model=False,
)
test_gen_p.problems = test_problems_p

ckpt_p = Path("results/blocks_world_medium_base_productivity/best_model.pth")
if not ckpt_p.exists():
    print("  NOT FOUND:", ckpt_p)
else:
    model_p = load_baseline_model(ckpt_p)
    solved = evaluate(model_p, test_problems_p, test_gen_p)
    print("  solve rate:", solved, "/100")

print()
print("For comparison, the WM numbers from the same checks:")
print("  in-distribution: oracle 93/100, model 96/100")
print("  productivity:    oracle 55/100, model 55/100")
