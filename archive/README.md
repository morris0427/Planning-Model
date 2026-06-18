# archive/

One-shot tools used to bring the experiments/ pipeline to its current
state. These have already been applied. They are kept here for
reference but are not part of the live pipeline.

## Patch scripts

Each of these modified a specific file in the codebase:

  fix_eight_puzzle_encoding.py
    Modified data/eight_puzzle.py:
      generate_problem: removed states[:-1] slice so solution_states
                        has N+1 entries (start through goal).
      encode_sequence: changed solution_states[i] to [i+1] (and the
                       bounds check correspondingly) so the WM training
                       data pairs each move with its post-move state.

  fix_blocks_world_encoding.py
    Modified data/blocks_world.py:
      _encode_state: replaced lossy top-blocks-only encoding with the
                     uniform 8-token encoding (top-to-bottom per tower,
                     POS_k separator always emitted).
      _decode_state: added as the inverse.

  fix_downstream_for_bw_encoding.py
    Modified data/blocks_world.py and calibrate_sep.py and
    trace_one_problem.py for downstream consequences of the new
    Blocks World state encoding:
      _estimate_state_tokens: parameterized as blocks + positions
      calibrate_sep.py: BW layout uses state_len=8, context_end=17
      trace_one_problem.py: guard preventing accidental BW usage
                            (script is 8-puzzle-only)

  fix_trainer_cache_split.py
    Modified trainer.py:
      load_cached_data, save_generated_data: split-aware filenames
      so productivity runs don't collide with in-distribution caches.

  fix_difficulty_enforcement.py
    Modified data/base.py:
      generate_dataset: bounded retry to enforce the requested
      difficulty exactly. Prevents SAW failures from contaminating
      test sets with shorter problems.

## Cache regenerators

  regen_eight_puzzle_wm_cache.py
  regen_blocks_world_cache.py
    Regenerated the cache files after the encoding fixes landed.
    Run once. The caches in cached_data/ are now the regenerated
    versions and should remain stable.
