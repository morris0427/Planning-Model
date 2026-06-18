"""
Evaluate 8-puzzle baseline and WM on the IN-DISTRIBUTION test set,
under the corrected check_solution_correctness.

This fills the missing cell in our cross-domain comparison table.
The 8-puzzle 'standard' (in-distribution) preset uses difficulty range
(10, 12) for both train and test.

Run from the experiments/ directory:
    python3 eval_8puzzle_indist.py
"""

import sys
sys.path.insert(0, ".")

import time
from pathlib import Path

import torch

for mod in list(sys.modules):
    if mod.startswith("data") or mod == "trainer" or mod.startswith("trainer."):
        del sys.modules[mod]

from config import ModelPresets, DataPresets
from data.base import DatasetFactory
import trainer as T


def load_model(ckpt_path, use_world_model):
    state = torch.load(ckpt_path, map_location="cpu")
    max_seq, d_model = state["pos_encoder.weight"].shape
    vocab = state["embedding.weight"].shape[0]
    m_cfg = ModelPresets.medium(use_world_model=use_world_model)
    model = T.PlanningTransformer(
        vocab_size=vocab, d_model=d_model, nhead=m_cfg.n_heads,
        num_layers=m_cfg.n_layers, dim_feedforward=m_cfg.d_ff,
        max_seq_length=max_seq,
    )
    model.load_state_dict(state)
    model.eval()
    return model


def evaluate(model, test_problems, test_gen, state_source=None, n=200):
    """Run generation + semantic check on the first n test problems."""
    solved = 0
    n = min(n, len(test_problems))
    for p in test_problems[:n]:
        kwargs = {"max_length": 100, "return_info": True}
        if state_source is not None:
            kwargs["state_source"] = state_source
        gen, info = T.generate_solution(
            model, p, test_gen, torch.device("cpu"), **kwargs
        )
        if T.check_solution_correctness(gen, p, test_gen):
            solved += 1
    return solved, n


# In-distribution 8-puzzle config
data_cfg = DataPresets.eight_puzzle_standard()
print(f"In-distribution 8-puzzle setup")
print(f"  test difficulty range: {data_cfg.test_difficulty_range}")
print()


# ---- Baseline ----
print("=" * 70)
print("Loading baseline (8-puzzle in-distribution medium)...")
print("=" * 70)

class _Shim: pass
shim = _Shim()
shim.data = data_cfg
_, base_test = T.load_cached_data(shim, use_wm=False)

base_test_gen = DatasetFactory.create(
    domain="eight_puzzle",
    difficulty_range=data_cfg.test_difficulty_range,
    num_samples=len(base_test),
    use_world_model=False,
)
base_test_gen.problems = base_test

ckpt_base = Path("results/eight_puzzle_in_distribution_medium_base/best_model.pth")
if not ckpt_base.exists():
    print(f"  NOT FOUND: {ckpt_base}")
    print(f"  Listing results/ to find correct path:")
    for p in Path("results").glob("eight_puzzle*base*/best_model.pth"):
        print(f"    {p}")
else:
    model_base = load_model(ckpt_base, use_world_model=False)
    t0 = time.time()
    s, n = evaluate(model_base, base_test, base_test_gen)
    print(f"  Baseline solve rate: {s}/{n}  ({100*s/n:.1f}%)  [{time.time()-t0:.1f}s]")


# ---- WM ----
print()
print("=" * 70)
print("Loading WM (8-puzzle in-distribution medium)...")
print("=" * 70)

shim.data = data_cfg
_, wm_test = T.load_cached_data(shim, use_wm=True)

wm_test_gen = DatasetFactory.create(
    domain="eight_puzzle",
    difficulty_range=data_cfg.test_difficulty_range,
    num_samples=len(wm_test),
    use_world_model=True,
)
wm_test_gen.problems = wm_test

ckpt_wm = Path("results/eight_puzzle_in_distribution_medium_wm/best_model.pth")
if not ckpt_wm.exists():
    print(f"  NOT FOUND: {ckpt_wm}")
    print(f"  Listing results/ to find correct path:")
    for p in Path("results").glob("eight_puzzle*wm*/best_model.pth"):
        print(f"    {p}")
else:
    model_wm = load_model(ckpt_wm, use_world_model=True)

    t0 = time.time()
    s, n = evaluate(model_wm, wm_test, wm_test_gen, state_source="oracle")
    print(f"  WM (state_source=oracle): {s}/{n}  ({100*s/n:.1f}%)  [{time.time()-t0:.1f}s]")

    t0 = time.time()
    s, n = evaluate(model_wm, wm_test, wm_test_gen, state_source="model")
    print(f"  WM (state_source=model):  {s}/{n}  ({100*s/n:.1f}%)  [{time.time()-t0:.1f}s]")


print()
print("Comparison context (Blocks World, same check):")
print("  in-distribution: baseline 94, WM-oracle 93, WM-model 96 (/100)")
print("  productivity truly-OOD: baseline 0, WM-oracle 254, WM-model 254 (/420) = 60.5%")
