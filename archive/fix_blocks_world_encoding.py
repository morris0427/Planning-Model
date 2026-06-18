#!/usr/bin/env python3
"""
Fix the lossy state encoding in data/blocks_world.py.

DESIGN (CONFIRMED)
------------------
- Uniform 8-token state encoding (every state-block is always 8 tokens).
- Within each position's tower: top block FIRST, bottom block LAST.
- Position separator POS_k is ALWAYS emitted at the end of each position's
  contents, even if the position is empty.
- An empty position contributes just its POS_k separator (1 token).
- 4 blocks + 4 separators = exactly 8 tokens per state.

EXAMPLES
--------
  state [['A','B','C'], [], ['D'], []]    (top blocks: C, D)
    -> [C, B, A, POS_0, POS_1, D, POS_2, POS_3]

  state [['A'], ['B'], ['C'], ['D']]      (4 separate towers)
    -> [A, POS_0, B, POS_1, C, POS_2, D, POS_3]

  state [['A','B','C','D'], [], [], []]   (all stacked at pos 0)
    -> [D, C, B, A, POS_0, POS_1, POS_2, POS_3]

ACTION SEMANTICS (unchanged)
----------------------------
  [block, POS_k] means "make `block` the top of position k." Source position
  is implicit (the tower where `block` is currently on top). Top-to-bottom
  encoding within towers puts movable blocks first within each tower so the
  model can read action-legality directly.

USAGE
-----
    python3 fix_blocks_world_encoding.py --dry-run
    python3 fix_blocks_world_encoding.py
"""

import argparse
import shutil
import sys
import time
from pathlib import Path

TARGET_PATH = Path("data/blocks_world.py")

BUGGY_ENCODE_STATE = '''    def _encode_state(self, state: List[List[str]]) -> List[int]:
        """Encode a state as tokens."""
        tokens = []
        for tower in state:
            if tower:
                # Encode top block
                tokens.append(self.vocab[tower[-1]])
            else:
                # Encode position if empty (or skip)
                pass
        
        # Pad to fixed length if needed
        while len(tokens) < self.num_positions:
            tokens.append(self.vocab['POS_0'])  # Placeholder
        
        return tokens[:self.num_positions]
'''

FIXED_ENCODE_STATE = '''    def _encode_state(self, state: List[List[str]]) -> List[int]:
        """Encode a state as tokens, losslessly with uniform fixed length.

        Each position contributes its tower contents from top to bottom,
        followed by its POS_k separator. Empty positions contribute just the
        separator. For a 4-block, 4-position domain every state encodes to
        exactly num_blocks + num_positions = 8 tokens.

        Example: state [['A','B','C'], [], ['D'], []]
            -> [C, B, A, POS_0, POS_1, D, POS_2, POS_3]
        """
        tokens = []
        for pos_idx, tower in enumerate(state):
            for block in reversed(tower):  # top-to-bottom
                tokens.append(self.vocab[block])
            tokens.append(self.vocab[f'POS_{pos_idx}'])
        return tokens

    def _decode_state(self, tokens: List[int]) -> List[List[str]]:
        """Inverse of _encode_state. Returns a state in the bottom-to-top
        raw format that generate_problem produces.
        """
        state = [[] for _ in range(self.num_positions)]
        buf = []
        block_token_set = {self.vocab[b] for b in self.blocks}
        pos_token_to_idx = {self.vocab[f'POS_{k}']: k for k in range(self.num_positions)}
        for tok in tokens:
            if tok in block_token_set:
                buf.append(self.inv_vocab[tok])
            elif tok in pos_token_to_idx:
                state[pos_token_to_idx[tok]] = list(reversed(buf))
                buf = []
            else:
                raise ValueError(
                    f"_decode_state: unexpected token {tok} "
                    f"({self.inv_vocab.get(tok, '?')}); state encoding "
                    f"contains only block tokens and POS_k separators."
                )
        if buf:
            raise ValueError(
                f"_decode_state: trailing block tokens with no closing "
                f"POS_k separator: {buf}"
            )
        return state
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not TARGET_PATH.exists():
        print(f"ERROR: {TARGET_PATH} not found. Run from the experiments/ directory.")
        sys.exit(1)

    src = TARGET_PATH.read_text()
    occurrences = src.count(BUGGY_ENCODE_STATE)

    if "_decode_state" in src and occurrences == 0:
        print("Already patched: file contains _decode_state and the buggy text is absent.")
        sys.exit(0)

    if occurrences == 0:
        print("ERROR: did not find the expected buggy _encode_state text in", TARGET_PATH)
        sys.exit(2)

    if occurrences > 1:
        print(f"ERROR: found {occurrences} occurrences of the buggy text; expected 1.")
        sys.exit(3)

    print("Found 1 occurrence of buggy _encode_state.")
    print()
    print("--- BEFORE ---")
    print(BUGGY_ENCODE_STATE)
    print("--- AFTER ---")
    print(FIXED_ENCODE_STATE)

    if args.dry_run:
        print("(dry run; not modifying the file)")
        sys.exit(0)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    backup = TARGET_PATH.with_suffix(f".py.bak-{timestamp}")
    shutil.copy2(TARGET_PATH, backup)
    print(f"Backed up original to {backup}")

    new_src = src.replace(BUGGY_ENCODE_STATE, FIXED_ENCODE_STATE)
    TARGET_PATH.write_text(new_src)
    print(f"Patched {TARGET_PATH}.")

    print()
    print("Verifying by reimporting and round-tripping test states...")
    for mod in list(sys.modules):
        if mod.startswith("data.blocks_world") or mod == "data":
            del sys.modules[mod]
    sys.path.insert(0, ".")
    try:
        from data.blocks_world import BlocksWorldDataset
        ds = BlocksWorldDataset(difficulty_range=(3, 3), num_samples=1,
                                 use_world_model=False, seed=0)

        test_states = [
            [['A', 'B', 'C'], [], ['D'], []],
            [['A'], ['B'], ['C'], ['D']],
            [['A', 'B', 'C', 'D'], [], [], []],
            [[], [], [], ['A', 'B', 'C', 'D']],
            [['A', 'B'], ['C'], [], ['D']],
        ]
        all_ok = True
        for st in test_states:
            enc = ds._encode_state(st)
            dec = ds._decode_state(enc)
            ok = (dec == st) and (len(enc) == 8)
            if not ok:
                all_ok = False
            print(f"  {'✓' if ok else '✗'} {st}")
            print(f"      encoded ({len(enc)} tokens): {enc}")
            print(f"        names: {[ds.inv_vocab[t] for t in enc]}")
            print(f"      decoded: {dec}")
        if all_ok:
            print()
            print("All round-trips passed; every encoding is exactly 8 tokens.")
        else:
            print()
            print("⚠️  Some round-trips FAILED; encoder/decoder disagree.")
            sys.exit(4)
    except Exception as e:
        print(f"⚠️  reimport raised: {e}")
        sys.exit(5)

    print()
    print("Next steps:")
    print("  1. Update test_encoding_roundtrip.py: state blocks are now 8 tokens")
    print("     (BW_STATE_LEN = 8, not 4).")
    print("  2. Run the round-trip test to confirm everything matches:")
    print("       python3 test_encoding_roundtrip.py --domain blocks_world --skip-cache")
    print("  3. Regenerate the Blocks World cache:")
    print("       python3 regen_blocks_world_cache.py")
    print("  4. Update calibrate_sep.py & trace_one_problem.py for stride 10.")
    print("  5. Retrain medium WM + medium baseline; compare to previous numbers.")


if __name__ == "__main__":
    main()
