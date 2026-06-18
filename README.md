# experiments/ data pipeline

This directory contains the planning-with-transformers experiments
(Blocks World and 8-puzzle, baseline vs world-model auxiliary task).

## Where things live

### Domain data generation and encoding

  data/base.py
    PlanningDataset (abstract)
      generate_dataset()        -- samples difficulties and calls
                                   generate_problem() with bounded retry
                                   to enforce the requested difficulty
      get_max_sequence_length() -- sizing the positional embedding budget
      _estimate_state_tokens()  -- per-domain hook

  data/blocks_world.py
    BlocksWorldDataset
      generate_problem(diff)    -- SAW backward walk from a random goal,
                                   forward-reconstruct moves
      _encode_state(state)      -- uniform 8-token state encoding
                                   (top-to-bottom per tower, POS_k separator
                                    always emitted)
      _decode_state(tokens)     -- inverse of _encode_state
      encode_sequence(problem)  -- full token sequence
      _estimate_state_tokens()  -- returns num_blocks + num_positions

  data/eight_puzzle.py
    EightPuzzleDataset
      generate_problem(diff)    -- SAW backward walk from goal, then
                                   reverse moves to get start -> goal
      encode_sequence(problem)  -- full token sequence; WM mode pairs
                                   each move with its post-move state
                                   via solution_states[i+1]

### Sequence layouts

Blocks World (4 blocks A-D, 4 positions, vocab size 11):
  Baseline:
    [START(1), start_state(8), goal_state(8), m_1(2), m_2(2), ..., m_N(2), END(1)]
  World Model:
    [START(1), start_state(8), goal_state(8),
     m_1(2), state_after_1(8), m_2(2), state_after_2(8), ...,
     m_N(2), state_after_N=goal(8), END(1)]
  An action is [block, POS_k] (2 tokens).

8-puzzle (vocab size 16; goal is fixed):
  Baseline:
    [dummy_move(1), start_state(9), PAD(1), goal_state(9),
     m_1(1), m_2(1), ..., m_N(1), SEP(1)]
  World Model:
    [dummy_move(1), start_state(9), PAD(1), goal_state(9),
     m_1(1), state_after_1(9), m_2(1), state_after_2(9), ...,
     m_N(1), state_after_N=goal(9), SEP(1)]
  An action is a single move token (10=up, 11=down, 12=left, 13=right).

### Training pipeline

  trainer.py
    train(config, use_wm)       -- main entrypoint
    load_cached_data(...)       -- looks up cached_data/{domain}_train|test{_wm|_baseline}{_productivity?}.json
    save_generated_data(...)    -- symmetric writer
    generate_solution(...)      -- inference, with oracle state injection for WM

### Configuration

  config.py
    DataPresets.{blocks_world|eight_puzzle}                  -- in-distribution
    DataPresets.{blocks_world|eight_puzzle}_productivity     -- OOD-length split
    ModelPresets.{tiny|small|medium|large}                   -- transformer sizes

### Caches

  cached_data/
    {domain}_train{_wm|_baseline}.json                       -- in-distribution
    {domain}_test{_wm|_baseline}.json                        -- in-distribution
    {domain}_train{_wm|_baseline}_productivity.json          -- OOD-length
    {domain}_test{_wm|_baseline}_productivity.json           -- OOD-length

### Verification

  test_encoding_roundtrip.py
    Layer 1: strict round-trip on fresh problems
    Layer 3: dataset's encode_sequence agrees with an independent reference
    Layer 4: cached data is internally consistent
    Run: python3 test_encoding_roundtrip.py [--domain DOMAIN]

  audit_pipeline.py
    Prints live source of every critical function. Use to verify state.

### Diagnostics

  calibrate_sep.py        -- measure P(SEP | at goal) vs P(SEP | not at goal)
                              for a trained model
  trace_one_problem.py    -- per-problem trace (8-puzzle only)
  diagnose_tf_vs_freegen  -- teacher-forced vs free-generation comparison

## How to run experiments

In-distribution sweep, medium only:
    python3 run_experiments.py --sweep model_size --domain blocks_world --sizes medium
    python3 run_experiments.py --sweep model_size --domain eight_puzzle --sizes medium

Productivity sweep, medium only:
    python3 run_productivity_sweep.py --domain blocks_world --sizes medium
    python3 run_productivity_sweep.py --domain eight_puzzle --sizes medium

Calibration on a trained checkpoint:
    python3 calibrate_sep.py --domain blocks_world --size medium --wm

## archive/

One-shot tools used to bring the pipeline to its current state. Kept for
reference but not part of the live pipeline. See archive/README.md.

## Notes on what is *not* in the pipeline anymore

  - The lossy "top-block-only" Blocks World state encoding (replaced by
    uniform 8-token encoding).
  - The off-by-one in 8-puzzle's WM encoder pairing move i with the
    pre-move state (fixed; now pairs with i+1, the post-move state).
  - The states[:-1] slice in 8-puzzle's generate_problem that dropped
    the goal from solution_states (fixed; full N+1 entries returned).
  - The split-agnostic cache lookup that caused productivity runs to
    silently load in-distribution caches (fixed; split_suffix appended).
  - The unenforced difficulty range in generate_dataset that let SAW
    failures contaminate test sets with shorter problems (fixed;
    bounded retry).
