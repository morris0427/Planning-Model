#cat > /tmp/eval_bw.py <<'PYEOF'
import sys; sys.path.insert(0, '.')
import torch
from pathlib import Path

for mod in list(sys.modules):
    if mod.startswith("data") or mod == "trainer" or mod.startswith("trainer."):
        del sys.modules[mod]

from config import ModelPresets, DataPresets
from data.base import DatasetFactory
import trainer as T

def load_model(ckpt_path):
    state = torch.load(ckpt_path, map_location='cpu')
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
    return model

def evaluate(model, test_problems, test_gen, state_source, n=100):
    solved = 0
    for p in test_problems[:n]:
        gen, info = T.generate_solution(
            model, p, test_gen, torch.device('cpu'),
            max_length=100, return_info=True, state_source=state_source,
        )
        if T.check_solution_correctness(gen, p, test_gen):
            solved += 1
    return solved

print("=== IN-DISTRIBUTION ===")
data_cfg = DataPresets.blocks_world_standard()
class _Shim: pass
shim = _Shim()
shim.data = data_cfg
_, test_problems_id = T.load_cached_data(shim, use_wm=True)
test_gen_id = DatasetFactory.create(
    domain='blocks_world',
    difficulty_range=data_cfg.test_difficulty_range,
    num_samples=len(test_problems_id),
    use_world_model=True,
)
test_gen_id.problems = test_problems_id
model_id = load_model(Path("results/blocks_world_in_distribution_medium_wm/best_model.pth"))
oracle_id = evaluate(model_id, test_problems_id, test_gen_id, 'oracle')
model_path_id = evaluate(model_id, test_problems_id, test_gen_id, 'model')
print("  oracle:", oracle_id, "/100")
print("  model: ", model_path_id, "/100")

print()
print("=== PRODUCTIVITY ===")
data_cfg = DataPresets.blocks_world_productivity()
shim.data = data_cfg
_, test_problems_p = T.load_cached_data(shim, use_wm=True)
test_gen_p = DatasetFactory.create(
    domain='blocks_world',
    difficulty_range=data_cfg.test_difficulty_range,
    num_samples=len(test_problems_p),
    use_world_model=True,
)
test_gen_p.problems = test_problems_p
model_p = load_model(Path("results/blocks_world_medium_wm_productivity/best_model.pth"))
oracle_p = evaluate(model_p, test_problems_p, test_gen_p, 'oracle')
model_path_p = evaluate(model_p, test_problems_p, test_gen_p, 'model')
print("  oracle:", oracle_p, "/100")
print("  model: ", model_path_p, "/100")
#PYEOF

#python3 /tmp/eval_bw.py
