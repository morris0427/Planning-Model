"""
8-Puzzle Dataset Generator - Self-Avoiding Random Walk (SAW)
Generates exact N-move solutions with uniform odd/even distribution
"""

import numpy as np
import random
from typing import List, Dict, Any, Tuple
from collections import Counter
from data.base import PlanningDataset, DatasetFactory


@DatasetFactory.register("eight_puzzle")
class EightPuzzleDataset(PlanningDataset):
    """8-Puzzle dataset using self-avoiding random walk."""
    
    GOAL_STATE = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 0]])
    PAD_TOKEN = 15
    SEP_TOKEN = 14
    
    def __init__(self, difficulty_range, num_samples, use_world_model=False, seed=None):
        self.difficulty_range = difficulty_range
        self.num_samples = num_samples
        self.use_world_model = use_world_model
        
        if seed:
            random.seed(seed)
            np.random.seed(seed)
        
        self.vocab = {
            'PAD': 15, 'SEP': 14,
            'up': 10, 'down': 11, 'left': 12, 'right': 13
        }
        
        self.id_to_token = {v: k for k, v in self.vocab.items()}
        self.id_to_token.update({i: str(i) for i in range(10)})
    
    def get_vocab_size(self):
        return 16
    
    def get_max_sequence_length(self):
        max_moves = self.difficulty_range[1]
        if self.use_world_model:
            return 1 + 9 + 1 + 9 + (max_moves * 10) + 1
        else:
            return 1 + 9 + 1 + 9 + max_moves + 1
    
    def reverse_move(self, move):
        """Reverse a move direction."""
        return {
            'up': 'down', 'down': 'up',
            'left': 'right', 'right': 'left'
        }[move]
    
    def apply_move(self, state, move):
        """Apply a move to the puzzle state."""
        s = state.copy()
        pos = np.argwhere(s == 0)
        if len(pos) == 0:
            return None
        r, c = pos[0]
        
        if move == 'up' and r > 0:
            s[r, c], s[r-1, c] = s[r-1, c], s[r, c]
        elif move == 'down' and r < 2:
            s[r, c], s[r+1, c] = s[r+1, c], s[r, c]
        elif move == 'left' and c > 0:
            s[r, c], s[r, c-1] = s[r, c-1], s[r, c]
        elif move == 'right' and c < 2:
            s[r, c], s[r, c+1] = s[r, c+1], s[r, c]
        else:
            return None
        return s
    
    def get_valid_moves(self, state):
        """Get list of valid moves from current state."""
        pos = np.argwhere(state == 0)
        if len(pos) == 0:
            return []
        r, c = pos[0]
        moves = []
        if r > 0: moves.append('up')
        if r < 2: moves.append('down')
        if c > 0: moves.append('left')
        if c < 2: moves.append('right')
        return moves
    
    def generate_problem(self, difficulty):
        """
        Generate exact N-move puzzle using self-avoiding random walk (SAW).
        
        Walks N steps from goal without revisiting states, then reverses moves.
        This guarantees an N-move solution that's near-optimal.
        """
        goal = self.GOAL_STATE
        max_attempts = 100
        
        for _ in range(max_attempts):
            curr = goal.copy()
            moves = []
            visited = {tuple(goal.flatten())}
            
            # Self-avoiding walk: N steps without revisiting
            for _ in range(difficulty):
                valid = self.get_valid_moves(curr)
                
                # Only consider unvisited states (SAW constraint)
                unvisited = []
                for m in valid:
                    nxt = self.apply_move(curr, m)
                    if nxt is not None and tuple(nxt.flatten()) not in visited:
                        unvisited.append(m)
                
                if not unvisited:
                    break  # Stuck, restart
                
                # Random choice among unvisited
                m = random.choice(unvisited)
                curr = self.apply_move(curr, m)
                visited.add(tuple(curr.flatten()))
                moves.append(m)
            
            # Success if we made exactly N moves
            if len(moves) == difficulty:
                # Reverse moves to get solution
                sol = [self.reverse_move(m) for m in reversed(moves)]
                
                # Generate intermediate states
                states = [curr.copy()]
                tmp = curr.copy()
                for m in sol:
                    tmp = self.apply_move(tmp, m)
                    states.append(tmp.copy())
                
                return {
                    'start_state': curr,
                    'goal_state': goal,
                    'solution_moves': sol,
                    'solution_states': states,
                    'num_moves': difficulty
                }
        
        raise RuntimeError(f"Could not generate {difficulty}-move puzzle after {max_attempts} attempts")
    
    def encode_sequence(self, prob):
        """Encode problem as token sequence."""
        seq = []
        
        # Dummy move
        seq.append(self.vocab['right'])
        
        # Start state (flattened)
        seq.extend(prob['start_state'].flatten().tolist())
        
        # PAD
        seq.append(self.PAD_TOKEN)
        
        # Goal state (flattened)
        seq.extend(prob['goal_state'].flatten().tolist())
        
        # Moves (and states for world model)
        if self.use_world_model:
            for i, m in enumerate(prob['solution_moves']):
                seq.append(self.vocab[m])
                if i + 1 < len(prob['solution_states']):
                    seq.extend(prob['solution_states'][i + 1].flatten().tolist())
        else:
            for m in prob['solution_moves']:
                seq.append(self.vocab[m])
        
        # SEP
        seq.append(self.SEP_TOKEN)
        
        return seq
    
    def generate_dataset(self):
        """Generate dataset of 8-puzzle problems."""
        print(f"\nGenerating {self.num_samples} 8-Puzzle problems...")
        print(f"  Difficulty: {self.difficulty_range[0]}-{self.difficulty_range[1]} moves")
        print(f"  Method: Self-Avoiding Random Walk (SAW)")
        print(f"  World model: {self.use_world_model}")
        
        problems = []
        
        for i in range(self.num_samples):
            # Uniform difficulty distribution
            diff = random.randint(*self.difficulty_range)
            prob_data = self.generate_problem(diff)
            seq = self.encode_sequence(prob_data)
            
            problems.append({
                'sequence': seq,
                'length': len(seq),
                'num_moves': prob_data['num_moves'],
                'problem_idx': i
            })
            
            if (i+1) % 100 == 0 or (i+1) == self.num_samples:
                print(f"  Generated {i+1}/{self.num_samples}...")
        
        # Report distribution
        counts = Counter([p['num_moves'] for p in problems])
        print(f"  Distribution: {dict(sorted(counts.items()))}")
        
        odd = sum(c for m, c in counts.items() if m % 2 == 1)
        even = sum(c for m, c in counts.items() if m % 2 == 0)
        print(f"  Odd-numbered: {odd}, Even-numbered: {even}")
        
        return problems
    
    def decode_sequence(self, seq):
        """Decode sequence to human-readable format."""
        return " ".join([self.id_to_token.get(t, f"UNK{t}") for t in seq])
    
    def _estimate_state_tokens(self):
        """Estimate tokens per state."""
        return 9


def apply_move_8puzzle(state, move):
    """Standalone apply_move function for use in trainer.py."""
    s = state.copy()
    pos = np.argwhere(s == 0)
    if len(pos) == 0:
        return None
    r, c = pos[0]
    
    if move == 'up' and r > 0:
        s[r, c], s[r-1, c] = s[r-1, c], s[r, c]
    elif move == 'down' and r < 2:
        s[r, c], s[r+1, c] = s[r+1, c], s[r, c]
    elif move == 'left' and c > 0:
        s[r, c], s[r, c-1] = s[r, c-1], s[r, c]
    elif move == 'right' and c < 2:
        s[r, c], s[r, c+1] = s[r, c+1], s[r, c]
    else:
        return None
    return s
