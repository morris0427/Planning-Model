"""Blocks World dataset implementation."""

from typing import List, Dict, Any, Tuple
import random
from copy import deepcopy
from data.base import PlanningDataset, DatasetFactory


@DatasetFactory.register("blocks_world")
class BlocksWorldDataset(PlanningDataset):
    """
    Blocks World dataset with SAW generation.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Vocabulary
        self.vocab = {
            'START': 0, 'END': 1,
            'A': 2, 'B': 3, 'C': 4, 'D': 5,
            'POS_0': 6, 'POS_1': 7, 'POS_2': 8, 'POS_3': 9,
            'PAD': 10
        }
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        
        # Domain configuration
        self.blocks = ['A', 'B', 'C', 'D']
        self.num_positions = 4
        self.same_size = True  # All blocks same size

    def generate_problem(self, difficulty: int) -> Dict[str, Any]:
        """
        Generate a problem using SAW (Sequential Adversarial Walking).
        
        Args:
            difficulty: Number of moves
            
        Returns:
            Problem dictionary
        """
        num_moves = difficulty
        
        # Generate random goal state
        goal_state = self.generate_random_state()
        current_state = deepcopy(goal_state)
        moves_made = []
        states = [deepcopy(current_state)]

        # Walk backwards from goal
        for _ in range(num_moves):
            valid_actions = self.get_all_valid_actions(current_state)
            if not valid_actions:
                break

            action = random.choice(valid_actions)
            new_state = self.apply_action(current_state, action)

            if new_state is not None:
                # Check for cycles
                is_duplicate = False
                for state in states:
                    if state == new_state:
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    moves_made.append(action)
                    current_state = new_state
                    states.append(deepcopy(current_state))
        
        # Reverse to get start → goal
        reversed_states = list(reversed(states))
        
        # Compute forward actions
        forward_moves = []
        for i in range(len(reversed_states) - 1):
            state_before = reversed_states[i]
            state_after = reversed_states[i + 1]
            
            # Find which block moved
            moved_block = None
            dest_position = None
            
            for block in self.blocks:
                loc_before = self.find_block(state_before, block)
                loc_after = self.find_block(state_after, block)
                if loc_before and loc_after:
                    pos_before, _ = loc_before
                    pos_after, _ = loc_after
                    if pos_before != pos_after:
                        moved_block = block
                        dest_position = pos_after
                        break
            
            if moved_block is not None and dest_position is not None:
                forward_moves.append([moved_block, dest_position])

        return {
            'start_state': reversed_states[0],
            'goal_state': goal_state,
            'solution_moves': forward_moves,
            'solution_states': reversed_states,
            'num_moves': len(forward_moves)
        }

    def encode_sequence(self, problem: Dict[str, Any]) -> List[int]:
        """
        Encode problem as token sequence.
        
        For baseline: [START, start, goal, action1, action2, ..., END]
        For WM: [START, start, goal, action1, state1, action2, state2, ..., END]
        """
        tokens = [self.vocab['START']]
        
        # Encode start state
        tokens.extend(self._encode_state(problem['start_state']))
        
        # Encode goal state
        tokens.extend(self._encode_state(problem['goal_state']))
        
        # Encode solution
        if self.use_world_model:
            # Include intermediate states
            for i, move in enumerate(problem['solution_moves']):
                tokens.extend(self._encode_move(move))
                # Add intermediate state
                if i + 1 < len(problem['solution_states']):
                    tokens.extend(self._encode_state(problem['solution_states'][i + 1]))
        else:
            # Just moves
            for move in problem['solution_moves']:
                tokens.extend(self._encode_move(move))
        
        tokens.append(self.vocab['END'])
        return tokens

    def decode_sequence(self, token_ids: List[int]) -> Dict[str, Any]:
        """Decode tokens back to problem."""
        tokens = [self.inv_vocab.get(tid, 'UNK') for tid in token_ids]
        return {
            'tokens': tokens,
            'raw_ids': token_ids
        }

    def _estimate_state_tokens(self) -> int:
        """Estimate tokens per state.

        Under the uniform encoding (top-to-bottom within each tower, POS_k
        separator always emitted): num_blocks + num_positions tokens per state.
        For the standard 4-block 4-position setup that is 8.
        """
        return len(self.blocks) + self.num_positions
    
    # Helper methods for SAW generation
    
    def generate_random_state(self) -> List[List[str]]:
        """Generate a random valid state."""
        state = [[] for _ in range(self.num_positions)]
        blocks = self.blocks[:]
        random.shuffle(blocks)
        
        for block in blocks:
            pos = random.randint(0, self.num_positions - 1)
            state[pos].append(block)
        
        return state
    
    def get_all_valid_actions(self, state: List[List[str]]) -> List[Tuple[str, int]]:
        """Get all valid actions from current state."""
        actions = []
        
        for pos_idx, tower in enumerate(state):
            if tower:  # If tower has blocks
                top_block = tower[-1]
                # Can move to any other position
                for dest_pos in range(self.num_positions):
                    if dest_pos != pos_idx:
                        actions.append((top_block, dest_pos))
        
        return actions
    
    def apply_action(self, state: List[List[str]], action: Tuple[str, int]) -> List[List[str]]:
        """Apply an action to a state."""
        block, dest_pos = action
        
        # Find block
        source_pos = None
        for pos_idx, tower in enumerate(state):
            if tower and tower[-1] == block:
                source_pos = pos_idx
                break
        
        if source_pos is None:
            return None  # Invalid action
        
        # Create new state
        new_state = [tower[:] for tower in state]
        new_state[source_pos].pop()
        new_state[dest_pos].append(block)
        
        return new_state
    
    def find_block(self, state: List[List[str]], block: str) -> Tuple[int, int]:
        """Find block position in state. Returns (position, height)."""
        for pos_idx, tower in enumerate(state):
            for height, b in enumerate(tower):
                if b == block:
                    return (pos_idx, height)
        return None
    
    def _encode_state(self, state: List[List[str]]) -> List[int]:
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
    
    def _encode_move(self, move: List) -> List[int]:
        """Encode a move [block, position]."""
        block, position = move
        return [self.vocab[block], self.vocab[f'POS_{position}']]


# Print confirmation
print("✓ BlocksWorldDataset registered successfully")
