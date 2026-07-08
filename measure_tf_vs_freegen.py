"""
measure_tf_vs_freegen.py

Per-position measurement of teacher-forced loss vs free-generation
behavior, to test whether the natural-language analogy (later tokens
are easier to predict from earlier context) holds for planning, and
whether it depends on the domain's state-prediction robustness.

For each (domain, architecture) cell, we measure on the within-training-
distribution test set:

  TF_loss[N]:   cross-entropy of the model's prediction for the
                ground-truth token at position N, given the ground-truth
                context up to position N-1. Measures how well the model
                has learned the conditional distribution.

  FG_match[N]:  whether the model's argmax at position N matches the
                ground-truth demonstration token at position N, given
                free-generated context up to N-1. Measures whether the
                conditional distribution survives rollout.

  FG_legal[N]:  whether the action the model emits at position N is
                legal in the TRUE current state (computed by applying
                ground-truth actions 0..N-1 to start). Measures when
                rollout produces actions that the environment couldn't
                accept.

The hypothesis: in Blocks World all three curves stay good as N
increases. In 8-Puzzle, TF_loss stays good but FG_match and FG_legal
degrade with N because compound state-prediction error corrupts the
rollout context.

Run from experiments/ directory:
    python3 measure_tf_vs_freegen.py [--n 200] [--domain DOMAIN]
"""

import sys
sys.path.insert(0, ".")

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

for mod in list(sys.modules):
    if mod.startswith("data") or mod == "trainer" or mod.startswith("trainer."):
        del sys.modules[mod]

from config import ModelPresets, DataPresets
from data.base import DatasetFactory
from data.blocks_world import BlocksWorldDataset
import trainer as T


# ============================================================
# Model loading
# ============================================================

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


# ============================================================
# Domain-specific configuration
# ============================================================

# Blocks World
BW_CONTEXT_LEN = 17
BW_END = 1
BW_BLOCK_TOKS = {2: 'A', 3: 'B', 4: 'C', 5: 'D'}
BW_POS_TOKS = {6: 0, 7: 1, 8: 2, 9: 3}

_bw_helper = BlocksWorldDataset(
    difficulty_range=(3, 3), num_samples=1,
    use_world_model=False, seed=0,
)


def bw_decode_state(tokens):
    return _bw_helper._decode_state(tokens)


def bw_apply_action_from_tokens(state, action_tokens):
    """action_tokens is a list of 2 tokens [block_tok, pos_tok]."""
    if len(action_tokens) != 2:
        return None
    b, p = action_tokens
    if b not in BW_BLOCK_TOKS or p not in BW_POS_TOKS:
        return None
    return _bw_helper.apply_action(state, (BW_BLOCK_TOKS[b], BW_POS_TOKS[p]))


# 8-Puzzle
EP_CONTEXT_LEN = 20
EP_SEP = 14
EP_MOVE_TOKS = {10: 'up', 11: 'down', 12: 'left', 13: 'right'}


def ep_apply_action_from_tokens(state, action_tokens):
    if len(action_tokens) != 1:
        return None
    t = action_tokens[0]
    if t not in EP_MOVE_TOKS:
        return None
    return T.apply_move_8puzzle(state, EP_MOVE_TOKS[t])


# ============================================================
# Teacher-forced loss per position
# ============================================================

def teacher_forced_loss_per_position(model, problem, action_token_positions):
    """For each position p in action_token_positions, compute the
    cross-entropy loss of the model's prediction at p given the
    ground-truth context up to p-1.

    Returns a dict {position: loss}.
    """
    seq = problem["sequence"]
    input_tensor = torch.tensor([seq], dtype=torch.long)

    with torch.no_grad():
        logits = model(input_tensor)  # [1, L, V]

    losses = {}
    for p in action_token_positions:
        if p >= len(seq) or p == 0:
            continue
        # The model's prediction for position p uses context [0..p-1],
        # which is logits[0, p-1, :]. Target is seq[p].
        log_probs = F.log_softmax(logits[0, p - 1, :], dim=-1)
        target = seq[p]
        loss = -log_probs[target].item()
        losses[p] = loss
    return losses


# ============================================================
# Free-generation match and legality per position
# ============================================================

def free_generation_per_position(model, problem, dataset_generator, domain,
                                    state_source=None):
    """Run free generation (with state_source if WM) and measure, at each
    action position, whether the model's argmax matches the ground-truth
    demonstration and whether the emitted action is legal in the true
    current state.

    Returns dict {position: {'match': bool, 'legal': bool}}.

    'Position' here is the position-within-action-sequence (1-indexed).
    For Blocks World: action at position k occupies tokens 17+2(k-1) and
                      17+2(k-1)+1 in baseline; for WM, k occupies the
                      same indices in the action slots.
    For 8-Puzzle: action at position k is a single token at varying offset.
    """
    seq = problem["sequence"]
    use_wm = dataset_generator.use_world_model

    # Run generation
    gen_kwargs = {"max_length": 100, "return_info": True}
    if state_source is not None:
        gen_kwargs["state_source"] = state_source
    gen, info = T.generate_solution(
        model, problem, dataset_generator, torch.device("cpu"), **gen_kwargs
    )

    # Decode start state and walk ground-truth actions to get true states
    if domain == "blocks_world":
        start_state = bw_decode_state(seq[1:9])
        context_len = BW_CONTEXT_LEN
        action_len = 2
        state_len_per = 8 if use_wm else 0
        gt_action_starts = []
        # Walk ground truth sequence to find action token positions
        p = context_len
        while p < len(seq):
            if seq[p] == BW_END:
                break
            if seq[p] in BW_BLOCK_TOKS and p + 1 < len(seq) and seq[p + 1] in BW_POS_TOKS:
                gt_action_starts.append(p)
                p += action_len + (state_len_per if use_wm else 0)
            else:
                break
        apply_fn = bw_apply_action_from_tokens

    else:  # eight_puzzle
        start_state = np.array(seq[1:10]).reshape(3, 3)
        context_len = EP_CONTEXT_LEN
        action_len = 1
        state_len_per = 9 if use_wm else 0
        gt_action_starts = []
        p = context_len
        while p < len(seq):
            if seq[p] == EP_SEP:
                break
            if seq[p] in EP_MOVE_TOKS:
                gt_action_starts.append(p)
                p += action_len + (state_len_per if use_wm else 0)
            else:
                break
        apply_fn = ep_apply_action_from_tokens

    # Walk gen sequence to find action token positions in gen
    gen_action_starts = []
    p = context_len
    while p < len(gen):
        if domain == "blocks_world":
            if gen[p] == BW_END: break
            if gen[p] in BW_BLOCK_TOKS and p + 1 < len(gen) and gen[p + 1] in BW_POS_TOKS:
                gen_action_starts.append(p)
                p += action_len + (state_len_per if use_wm else 0)
            else:
                break
        else:
            if gen[p] == EP_SEP: break
            if gen[p] in EP_MOVE_TOKS:
                gen_action_starts.append(p)
                p += action_len + (state_len_per if use_wm else 0)
            else:
                break

    # For each position, check match (gen matches GT) and legality
    # (action would be legal in the TRUE state at that position)
    true_state = [t[:] for t in start_state] if domain == "blocks_world" else start_state.copy()
    results = {}
    n_positions = min(len(gt_action_starts), len(gen_action_starts))
    # Also keep the larger of the two so we can record cases where gen
    # ran out before the demonstration did
    max_positions = max(len(gt_action_starts), len(gen_action_starts))

    for k in range(max_positions):
        gen_tokens = None
        gt_tokens = None

        if k < len(gen_action_starts):
            ps = gen_action_starts[k]
            gen_tokens = gen[ps:ps + action_len]
        if k < len(gt_action_starts):
            ps = gt_action_starts[k]
            gt_tokens = seq[ps:ps + action_len]

        # Match: do gen and gt action tokens agree?
        if gen_tokens is not None and gt_tokens is not None:
            match = (gen_tokens == gt_tokens)
        else:
            match = False  # one side ran out

        # Legality: is gen's action legal in the true state at this step?
        if gen_tokens is not None and true_state is not None:
            next_state = apply_fn(true_state, gen_tokens)
            legal = next_state is not None
        else:
            legal = False

        results[k + 1] = {  # 1-indexed positions
            "match": match,
            "legal": legal,
            "gen_present": gen_tokens is not None,
            "gt_present": gt_tokens is not None,
        }

        # Advance true_state by applying the GROUND-TRUTH action
        # (this is the trajectory the model was supposed to follow).
        # We do this so 'legal' at position k+1 is checked against the
        # true state that would obtain if the model had been correct.
        if gt_tokens is not None:
            nxt = apply_fn(true_state, gt_tokens)
            if nxt is not None:
                true_state = nxt

    return results


# ============================================================
# Aggregation
# ============================================================

def analyze_domain(domain, condition, n_problems):
    """Run measurements on n_problems for a domain x condition cell.
    Returns aggregated per-position statistics.
    """
    print(f"\n[{domain} / {condition}]")

    # Load checkpoint and test data
    if domain == "blocks_world":
        if condition == "baseline":
            ckpt = "results/blocks_world_in_distribution_medium_base/best_model.pth"
            use_wm = False
            state_source = None
        else:  # wm
            ckpt = "results/blocks_world_in_distribution_medium_wm/best_model.pth"
            use_wm = True
            state_source = "model"
        data_cfg = DataPresets.blocks_world_standard()
        cache_suffix = "_wm" if use_wm else "_baseline"
        test_cache = f"cached_data/blocks_world_test{cache_suffix}.json"
    else:
        if condition == "baseline":
            ckpt = "results/eight_puzzle_in_distribution_medium_base/best_model.pth"
            use_wm = False
            state_source = None
        else:
            ckpt = "results/eight_puzzle_in_distribution_medium_wm/best_model.pth"
            use_wm = True
            state_source = "model"
        data_cfg = DataPresets.eight_puzzle_standard()
        cache_suffix = "_wm" if use_wm else "_baseline"
        test_cache = f"cached_data/eight_puzzle_test{cache_suffix}.json"

    if not Path(ckpt).exists():
        print(f"  Skipping: checkpoint {ckpt} not found")
        return None

    model = load_model(ckpt, use_world_model=use_wm)
    with open(test_cache) as f:
        problems = json.load(f)
    problems = problems[:n_problems]

    test_gen = DatasetFactory.create(
        domain=domain,
        difficulty_range=data_cfg.test_difficulty_range,
        num_samples=len(problems),
        use_world_model=use_wm,
    )
    test_gen.problems = problems

    # Identify action positions in each ground-truth sequence
    context_len = BW_CONTEXT_LEN if domain == "blocks_world" else EP_CONTEXT_LEN
    action_len = 2 if domain == "blocks_world" else 1
    state_len_per = (8 if domain == "blocks_world" else 9) if use_wm else 0

    # Aggregators
    tf_loss_by_pos = defaultdict(list)
    fg_match_by_pos = defaultdict(list)
    fg_legal_by_pos = defaultdict(list)

    t0 = time.time()
    for i, p in enumerate(problems):
        seq = p["sequence"]

        # Find action positions in the ground-truth sequence
        action_positions_per_step = []  # list of lists; each inner list is the tokens of one action
        pos = context_len
        step = 0
        while pos < len(seq):
            if domain == "blocks_world":
                if seq[pos] == BW_END: break
                if seq[pos] in BW_BLOCK_TOKS:
                    action_positions_per_step.append(list(range(pos, pos + action_len)))
                    pos += action_len + state_len_per
                    step += 1
                else:
                    break
            else:
                if seq[pos] == EP_SEP: break
                if seq[pos] in EP_MOVE_TOKS:
                    action_positions_per_step.append([pos])
                    pos += action_len + state_len_per
                    step += 1
                else:
                    break

        # Teacher-forced loss: average loss over the action tokens of each step
        flat_positions = [p for step_positions in action_positions_per_step for p in step_positions]
        tf_losses_flat = teacher_forced_loss_per_position(model, p, flat_positions)

        # Aggregate to per-step loss (mean over tokens within each step)
        for step_idx, step_positions in enumerate(action_positions_per_step):
            step_losses = [tf_losses_flat[pp] for pp in step_positions if pp in tf_losses_flat]
            if step_losses:
                tf_loss_by_pos[step_idx + 1].append(np.mean(step_losses))

        # Free generation
        fg = free_generation_per_position(
            model, p, test_gen, domain, state_source=state_source
        )
        for step_idx, data in fg.items():
            if data["gen_present"] and data["gt_present"]:
                fg_match_by_pos[step_idx].append(1.0 if data["match"] else 0.0)
            if data["gen_present"]:
                fg_legal_by_pos[step_idx].append(1.0 if data["legal"] else 0.0)

        if (i + 1) % 50 == 0:
            print(f"    {i+1}/{len(problems)} done ({time.time()-t0:.1f}s)")

    # Aggregate to means per step (with sample sizes)
    summary = {
        "n_problems": len(problems),
        "tf_loss": {k: {"mean": float(np.mean(v)), "n": len(v)}
                    for k, v in tf_loss_by_pos.items()},
        "fg_match": {k: {"mean": float(np.mean(v)), "n": len(v)}
                     for k, v in fg_match_by_pos.items()},
        "fg_legal": {k: {"mean": float(np.mean(v)), "n": len(v)}
                     for k, v in fg_legal_by_pos.items()},
    }

    # Print summary
    max_pos = max(
        max(summary["tf_loss"].keys(), default=0),
        max(summary["fg_match"].keys(), default=0),
        max(summary["fg_legal"].keys(), default=0),
    )
    print(f"  Pos |   TF loss   | FG match  | FG legal  | n")
    print(f"  ----+-------------+-----------+-----------+----")
    for pos in range(1, min(max_pos + 1, 20)):
        tf = summary["tf_loss"].get(pos, {"mean": None, "n": 0})
        m = summary["fg_match"].get(pos, {"mean": None, "n": 0})
        l = summary["fg_legal"].get(pos, {"mean": None, "n": 0})
        tf_str = f"{tf['mean']:.3f}" if tf['mean'] is not None else "  -- "
        m_str = f"{100*m['mean']:.1f}%" if m['mean'] is not None else "  -- "
        l_str = f"{100*l['mean']:.1f}%" if l['mean'] is not None else "  -- "
        n = max(tf['n'], m['n'], l['n'])
        print(f"  {pos:>3} |   {tf_str:>7}   |  {m_str:>6}  |  {l_str:>6}  | {n}")

    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--domain", choices=["blocks_world", "eight_puzzle", "both"],
                    default="both")
    args = ap.parse_args()

    out_dir = Path("results/paper")
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    domains = ["blocks_world", "eight_puzzle"] if args.domain == "both" else [args.domain]

    for domain in domains:
        results[domain] = {}
        for condition in ["baseline", "wm"]:
            r = analyze_domain(domain, condition, args.n)
            if r is not None:
                results[domain][condition] = r

    with open(out_dir / "tf_vs_freegen.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out_dir / 'tf_vs_freegen.json'}")


if __name__ == "__main__":
    main()
