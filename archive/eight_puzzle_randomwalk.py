"""
8-Puzzle Dataset Generator - Random Walk Approach
IMPROVED VERSION: Guaranteed N-move solutions, supports odd/even
"""

import numpy as np
import random
from typing import List, Dict, Any, Tuple
from collections import Counter


class EightPuzzleDataset:
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
        
        self.vocab = {'PAD': 15, 'SEP': 14, 'up': 10, 'down': 11, 'left': 12, 'right': 13}
        self.id_to_token = {v: k for k, v in self.vocab.items()}
        self.id_to_token.update({i: str(i) for i in range(10)})
    
    def get_vocab_size(self):
        return 16
    
    def get_max_sequence_length(self):
        max_moves = self.difficulty_range[1]
        return 1 + 9 + 1 + 9 + (max_moves * 10 if self.use_world_model else max_moves) + 1
    
    def reverse_move(self, move):
        return {'up': 'down', 'down': 'up', 'left': 'right', 'right': 'left'}[move]
    
    def apply_move(self, state, move):
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
        goal = self.GOAL_STATE
        for _ in range(100):
            curr = goal.copy()
            moves = []
            visited = {tuple(goal.flatten())}
            
            for _ in range(difficulty):
                valid = self.get_valid_moves(curr)
                unvisited = []
                for m in valid:
                    nxt = self.apply_move(curr, m)
                    if nxt is not None and tuple(nxt.flatten()) not in visited:
                        unvisited.append(m)
                
                if not unvisited:
                    break
                
                m = random.choice(unvisited)
                curr = self.apply_move(curr, m)
                visited.add(tuple(curr.flatten()))
                moves.append(m)
            
            if len(moves) == difficulty:
                sol = [self.reverse_move(m) for m in reversed(moves)]
                states = [curr.copy()]
                tmp = curr.copy()
                for m in sol:
                    tmp = self.apply_move(tmp, m)
                    states.append(tmp.copy())
                
                return {
                    'start_state': curr,
                    'goal_state': goal,
                    'solution_moves': sol,
                    'solution_states': states[:-1],
                    'num_moves': difficulty
                }
        
        raise RuntimeError(f"Failed to generate {difficulty}-move puzzle")
    
    def encode_sequence(self, prob):
        seq = [self.vocab['right']]
        seq.extend(prob['start_state'].flatten().tolist())
        seq.append(15)
        seq.extend(prob['goal_state'].flatten().tolist())
        
        if self.use_world_model:
            for i, m in enumerate(prob['solution_moves']):
                seq.append(self.vocab[m])
                if i < len(prob['solution_states']):
                    seq.extend(prob['solution_states'][i].flatten().tolist())
        else:
            for m in prob['solution_moves']:
                seq.append(self.vocab[m])
        
        seq.append(14)
        return seq
    
    def generate_dataset(self):
        print(f"\nGenerating {self.num_samples} problems (random walk method)...")
        print(f"  Range: {self.difficulty_range[0]}-{self.difficulty_range[1]} moves")
        
        problems = []
        for i in range(self.num_samples):
            diff = random.randint(*self.difficulty_range)
            prob_data = self.generate_problem(diff)
            seq = self.encode_sequence(prob_data)
            
            problems.append({
                'sequence': seq,
                'length': len(seq),
                'num_moves': prob_data['num_moves'],
                'problem_idx': i
            })
            
            if (i+1) % 100 == 0:
                print(f"  {i+1}/{self.num_samples}...")
        
        counts = Counter([p['num_moves'] for p in problems])
        odd = sum(c for m, c in counts.items() if m % 2 == 1)
        even = sum(c for m, c in counts.items() if m % 2 == 0)
        print(f"  ✓ Done. Odd: {odd}, Even: {even}")
        return problems
    
    def decode_sequence(self, seq):
        return " ".join([self.id_to_token.get(t, f"?{t}") for t in seq])
    
    def _estimate_state_tokens(self):
        return 9


def apply_move_8puzzle(state, move):
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
