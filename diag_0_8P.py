import sys; sys.path.insert(0, '.')
import json
import numpy as np
import torch
import torch.nn.functional as F

for mod in list(sys.modules):
    if mod.startswith("data") or mod == "trainer" or mod.startswith("trainer."):
        del sys.modules[mod]

from config import ModelPresets, DataPresets
import trainer as T

# Load the in-distribution 8-puzzle WM checkpoint
state = torch.load("results/eight_puzzle_in_distribution_medium_wm/best_model.pth",
                   map_location='cpu')
max_seq, d_model = state["pos_encoder.weight"].shape
vocab = state["embedding.weight"].shape[0]
m_cfg = ModelPresets.medium(use_world_model=True)
model = T.PlanningTransformer(
    vocab_size=vocab, d_model=d_model, nhead=m_cfg.n_heads,
    num_layers=m_cfg.n_layers, dim_feedforward=m_cfg.d_ff,
    max_seq_length=max_seq,
)
model.load_state_dict(state); model.eval()

with open("cached_data/eight_puzzle_test_wm.json") as f:
    problems = json.load(f)

# For each state-token position in each test problem, record:
#   - the true target
#   - the model's log-prob for the true target
#   - the model's argmax
loss_by_target = {i: [] for i in range(9)}
argmax_by_target = {i: [] for i in range(9)}

for p in problems[:100]:
    seq = p["sequence"]
    input_tensor = torch.tensor([seq], dtype=torch.long)
    with torch.no_grad():
        logits = model(input_tensor)  # [1, L, V]
    log_probs = F.log_softmax(logits[0], dim=-1)  # [L, V]

    # Find state-token positions (each is 9 tokens after each action)
    pos = 21  # first state-token position (after context+first action)
    while pos + 9 <= len(seq):
        if seq[pos - 1] not in {10, 11, 12, 13}:
            break
        # Positions pos..pos+8 are state tokens; the model predicts them
        # from context [0..pos-1], [0..pos], ..., [0..pos+7]
        for i in range(9):
            target = seq[pos + i]
            if target > 8:  # not a tile value; skip
                continue
            # The prediction for position pos+i uses context up to pos+i-1
            log_p = log_probs[pos + i - 1, target].item()
            argmax = int(log_probs[pos + i - 1].argmax().item())
            loss_by_target[target].append(-log_p)
            argmax_by_target[target].append(argmax)
        pos += 10  # advance by action(1) + state(9)

print("Teacher-forced NLL by target token (8-puzzle WM in-distribution):")
for tok in range(9):
    losses = loss_by_target[tok]
    argmaxes = argmax_by_target[tok]
    if losses:
        n_correct = sum(1 for a in argmaxes if a == tok)
        print(f"  Target={tok}: n={len(losses)}, "
              f"mean NLL={np.mean(losses):.3f}, "
              f"argmax correct: {n_correct}/{len(argmaxes)} "
              f"({100*n_correct/len(argmaxes):.1f}%)")
