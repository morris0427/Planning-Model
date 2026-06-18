# experiments/ — Planning with Transformers

Source code for studying length-generalization in Transformer planners
across Blocks World and 8-puzzle, comparing baseline (action-prediction
only) against a world-model auxiliary task (action + state prediction).

## What this code does

We train small Transformers (medium: ~900K params, large: ~3.2M) to
produce action sequences that take a start state to a goal state, in
two planning domains. We compare:

  - Baseline vs world-model (WM) architectures
  -   Baseline: trained over the space of possible action sequences only
  -   WM (Wordl Model-based): trained over the space of action sequence + world dynamics (how actions change the world state)
  - In-distribution test problems vs truly-out-of-distribution (BFS-verified
    longer-than-training) problems
  - Plan optimality of demonstrations vs models
  - Multiple model sizes

The headline findings from these experiments:

  1. Neither architecture generalizes to plans longer than training
     distribution. Scaling from medium (900K) to large (3.2M parameters)
     does not change this.
  2. World-model auxiliary supervision yields no significant in-distribution
     improvement over baseline. Differences (~2-3 points) are within
     sampling noise.
  3. The auxiliary state-prediction task is learnable in Blocks World
     (model and oracle state-sources yield byte-identical results) but
     not in 8-puzzle (model state predictions are invalid from step 1).
  4. Models trained on non-optimal SAW demonstrations recover near-optimal
     plans on problems they solve. In Blocks World, demonstrations are
     ~98% non-optimal but model plans are ~88% optimal — a substantial
     gap-closing effect.

The experimental design takes advantage of small-model affordances: the
training distribution is exactly specifiable, internal features
(action/state/goal-recognition) can be measured independently, and
intervention experiments are tractable. We refer to this as a
detect-isolate-recover loop, and argue it is the right experimental
setup for the question of whether Transformers can learn to plan.

## Quick start

To run one in-distribution training and evaluation cycle from scratch
(verifies the pipeline; ~30 minutes on CPU, less on GPU):

    python3 run_experiments.py --sweep model_size --domain blocks_world --sizes medium
    python3 eval_truly_ood_aligned.py

You should see ~77% in-distribution solve rate (baseline) and 0/407
on truly-OOD. If you get something very different, something is wrong;
see the "Known pitfalls" section below.

To reproduce the paper's headline numbers (~hours on CPU):

    python3 run_experiments.py --sweep model_size --domain blocks_world --sizes medium
    python3 run_experiments.py --sweep model_size --domain eight_puzzle --sizes medium
    python3 run_experiments.py --sweep model_size --domain blocks_world --sizes medium --split productivity
    python3 run_experiments.py --sweep model_size --domain eight_puzzle --sizes medium --split productivity
    python3 eval_truly_ood_aligned.py            # corrected Blocks World eval
    python3 eval_truly_ood_aligned_8puzzle.py    # corrected 8-puzzle eval
    python3 plan_optimality.py                   # Conclusion 4 numbers
    python3 make_paper_tables.py                 # produce LaTeX tables

The `eval_truly_ood_aligned*` scripts are required — the per-cache solve
rates printed by `run_experiments.py` reflect the SAW-non-optimality
confound and are not the paper numbers. See "Methodology decisions"
below for why.

## File layout

### Domain data generation and encoding

    data/base.py            PlanningDataset (abstract base class)
    data/blocks_world.py    BlocksWorldDataset
                              _encode_state: uniform 8-token
                                top-to-bottom per tower with POS_k separator
                              _decode_state: lossless inverse
    data/eight_puzzle.py    EightPuzzleDataset
                              encode_sequence: pairs each move with the
                                post-move state (in WM mode)

### Sequence layouts

Blocks World (4 blocks, 4 positions, vocab 11):

    Baseline:
      START, start_state(8), goal_state(8), m_1(2), ..., m_N(2), END
    World Model:
      START, start_state(8), goal_state(8),
      m_1(2), state_after_1(8), m_2(2), state_after_2(8), ...,
      m_N(2), goal(8), END
    An action is [block_token, POS_token] = 2 tokens.

8-puzzle (vocab 16, goal is fixed):

    Baseline:
      dummy_move(1), start_state(9), PAD(1), goal_state(9),
      m_1(1), ..., m_N(1), SEP(1)
    World Model:
      dummy_move(1), start_state(9), PAD(1), goal_state(9),
      m_1(1), state_after_1(9), ..., m_N(1), goal(9), SEP(1)
    An action is one move token (10=up, 11=down, 12=left, 13=right).
    Note: position 0 always holds token 13. It is a context anchor,
    not a meaningful "first move."

### Training pipeline

    trainer.py
      train(config, use_wm)               -- main training entrypoint
      load_cached_data(...)               -- cache lookup with split-aware suffix
      save_generated_data(...)
      generate_solution(model, problem, ..., state_source=None)
                                          -- inference; the state_source
                                             parameter controls WM rollout
                                             (oracle vs model autoregressive)
      check_solution_correctness(gen, problem, dataset)
                                          -- decodes start/goal from sequence,
                                             walks gen tokens via apply_action,
                                             returns True iff final == goal

### Configuration

    config.py
      DataPresets.blocks_world_standard()         -- in-distribution (length 1-4)
      DataPresets.blocks_world_productivity()     -- length 1-4 train, 5-8 test
      DataPresets.eight_puzzle_standard()         -- in-distribution (length 10-12)
      DataPresets.eight_puzzle_productivity()     -- length 10-12 train, 13-18 test
      ModelPresets.{tiny|small|medium|large}      -- transformer sizes

### Caches

    cached_data/
      {domain}_train{_wm|_baseline}.json                  in-distribution
      {domain}_test{_wm|_baseline}.json                   in-distribution
      {domain}_train{_wm|_baseline}_productivity.json     productivity (OOD)
      {domain}_test{_wm|_baseline}_productivity.json      productivity (OOD)
      {domain}_*.meta.json                                metadata sidecars

    Each cache file is a list of problem dicts. Run with
    --force-regenerate to rebuild from scratch (slow; SAW is single-threaded).

### Evaluation pipeline (use these for paper-grade numbers)

    eval_truly_ood_aligned.py          Aligned Blocks World eval, medium.
                                       Uses the WM cache as canonical,
                                       decodes (start, goal), re-encodes
                                       on-the-fly for baseline. Filters to
                                       truly-OOD via BFS.

    eval_truly_ood_aligned_large.py    Same, large checkpoints.

    eval_truly_ood_aligned_8puzzle.py  8-puzzle parallel.

    plan_optimality.py                 Computes (BFS-shortest, SAW-reference,
                                       model-plan) triples per problem.
                                       Saves to results/paper/plan_optimality_*.json.
                                       This is the source for Conclusion 4.

    make_paper_results.py              Single-shot computation of all paper
                                       cells. Writes results/paper/paper_results.json
                                       and results/paper/paper_diagnostics.json.

    make_paper_tables.py               Reads paper JSONs, writes LaTeX tables
                                       and the plan-length figure.

### Verification

    test_encoding_roundtrip.py
      Layer 1: strict round-trip on freshly generated problems.
      Layer 3: encode_sequence agrees with independent reference.
      Layer 4: cached data is internally consistent.
      Run: python3 test_encoding_roundtrip.py [--domain DOMAIN]
      Run this after modifying anything in data/.

    audit_pipeline.py
      Prints live source of every critical function. Use to verify the
      patches are applied if anything seems off.

### Diagnostics

    calibrate_sep.py        P(SEP|at goal) vs P(SEP|not at goal) for a
                            trained model. Shows the model has learned
                            goal-recognition under teacher forcing
                            (~1.0 vs ~0.0).
    trace_one_problem.py    Per-problem trace (8-puzzle).
    diagnose_tf_vs_freegen  Teacher-forced vs free-generation comparison.

### archive/

One-shot tools and historical patches. Not part of the live pipeline.
See archive/README.md for what each was for.

## Methodology decisions (important — read before modifying)

### Why aligned-eval is required

The baseline and WM test caches are generated independently with
different random seeds. Problem `i` in `*_test_baseline_productivity.json`
is *not* the same problem as `i` in `*_test_wm_productivity.json`.
Cross-architecture comparisons that index both caches by `i` are
meaningless. The aligned-eval scripts work by treating one cache as
canonical and re-encoding (start, goal) on-the-fly for the other
architecture.

If you write a new comparison script, you must either use the aligned
pattern or run a BFS-generated test set that's identical across formats.

### Why BFS-filtering is required for productivity

SAW (the data generator) walks the state space backward from a goal
and reports the walk length as the problem's difficulty. But the walk
may not be optimal — it can revisit nearby states, producing a
non-optimal "reference plan." In Blocks World, ~98% of SAW reference
plans are non-optimal, with mean excess of 2.72 moves over the BFS
shortest. In 8-puzzle the gap is smaller (mean excess 0.62) but still
present.

This means the SAW productivity test set (length 5-8 reference plans)
contains many problems whose *true* shortest path is in the training
distribution. Filtering by BFS shortest >= 5 isolates the actually-OOD
subset.

For a future paper that wants productivity as a primary measurement,
the right design is to generate test data with BFS from the start
(see `generate_bfs_test_sets.py` for the prototype). SAW remains
appropriate for training data because the training distribution is
specified by length range, not by optimality.

### Why state_source matters for WM eval

The WM model is trained on sequences where state tokens follow each
action. At inference time, you can either:

  - state_source="oracle": after each model-emitted action, inject the
    true post-action state into the context. Measures the model's
    action-selection ability conditional on accurate state observation.

  - state_source="model": let the model emit state tokens
    autoregressively. Measures the model's joint action+state prediction
    capability end-to-end.

In Blocks World, these are byte-identical on solved problems — the
model's state predictions are accurate. In 8-puzzle, model state
predictions are invalid by step 1, so the model path fails. This
difference is itself a finding (auxiliary task learnability differs
across domains).

### Why we use small Transformers

Small from-scratch models offer experimental control that pretrained
LLMs do not:

  - Training distribution is specifiable exactly.
  - Subcomponent losses (action, state, goal-recognition) can be probed.
  - Intervention experiments fit in hours, not weeks.

If you extend this work, preserve these affordances. Going to LLMs
would make the methodology less rigorous, not more.

## Known pitfalls (have hit before; document if you hit them again)

If your in-distribution solve rate drops dramatically when running a
new eval script, suspect a re-encoder bug. Check:

  - 8-puzzle: position 0 of the sequence must be `13`, not `0`. The
    "dummy_move" position is a context anchor that is always token 13.
    Setting it to 0 (which is a valid tile value, the blank) causes
    confusion.
  - Blocks World: the state encoding uses one token per block plus a
    position separator (POS_k), always emitted (not omitted for empty
    towers). Check `_encode_state` against the round-trip test.

If WM-model and WM-oracle results disagree on problems where state
predictions are accurate, suspect a truncation asymmetry in
`generate_solution`. Both paths should use the same headroom in the
sequence-length truncation guard:

    headroom = action_tok_size + (per_state_tokens if use_world_model else 0) + 1

If a productivity eval reports unexpectedly high WM solve rates (say,
above 30%), suspect cache misalignment. Run a sanity check:

    import json
    with open("cached_data/{domain}_test_baseline_productivity.json") as f: b = json.load(f)
    with open("cached_data/{domain}_test_wm_productivity.json") as f: w = json.load(f)
    # decode start/goal from each at the same index; compare. They will differ.

The fix is to use eval_truly_ood_aligned*.py, not to regenerate caches.

If 8-puzzle WM-model state-source gives nonzero solve rate, double-check
by inspecting the actual generated state tokens. Earlier diagnostics
showed model-emitted states like [4, 1, 8, 6, 3, 5, 7, 3, 2] — note the
duplicate 3 and missing 0. A "successful" generation with such invalid
state tokens may indicate a bug in `check_solution_correctness`.

## Compute expectations

On CPU (M-series Mac):

    Medium model, in-distribution, Blocks World:       ~30 min
    Medium model, in-distribution, 8-puzzle:           ~2-3 hours
    Medium model, productivity, Blocks World:          ~1 hour
    Medium model, productivity, 8-puzzle:              ~3-4 hours
    Large model, productivity, Blocks World, baseline: ~3.5 hours
    Large model, productivity, Blocks World, WM:       ~18 hours
    Large model, productivity, 8-puzzle:               not measured;
                                                       likely overnight+

On GPU these are 5-20x faster depending on memory bandwidth.

Evaluation scripts are inference-only:

    eval_truly_ood_aligned.py (407 problems x 3 conditions):       ~1 min
    plan_optimality.py (500 problems x 2 conditions x 2 domains):  ~1 min
    make_paper_results.py (all cells):                             ~5 min

BFS computations are essentially free for Blocks World (small state
space) but ~1 sec/problem for 8-puzzle (depth 12-15 search).

## What we have not done

Things that are tempting but were deferred for paper-1:

  - Reward-based training as a recover step (currently the paper's
    detect-isolate-recover loop covers detect and isolate; recover is
    sketched in the discussion).
  - Data augmentation with longer-plan exemplars as a cheaper alternative
    to RL.
  - Architectural variants (different positional encodings, different
    state-prediction objectives).
  - Larger model sweep on 8-puzzle.
  - Other domains.

If you want to take any of these on, they all build naturally on the
existing pipeline.
