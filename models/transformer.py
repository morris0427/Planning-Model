import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import copy

class BaselineTransformerDataset(Dataset):
    """
    Dataset for baseline transformer WITHOUT chain of thought
    Format: [start_state] → [move_1, move_2, ..., move_n]
    Does NOT include intermediate states
    """
    def __init__(self, training_data, max_seq_length=50):
        self.samples = []
        self.max_seq_length = max_seq_length
        
        # Vocabulary
        self.move_to_token = {'up': 9, 'down': 10, 'left': 11, 'right': 12}
        self.token_to_move = {9: 'up', 10: 'down', 11: 'left', 12: 'right'}
        self.PAD_TOKEN = 13
        self.SEP_TOKEN = 14
        self.vocab_size = 15
        
        for trajectory in training_data:
            # Handle both 'start' and 'start_state' keys
            if 'start_state' in trajectory:
                start = np.array(trajectory['start_state'])
            elif 'start' in trajectory:
                start = np.array(trajectory['start'])
            else:
                print(f"⚠️  Skipping sample - no 'start' or 'start_state' field")
                continue
            
            if 'moves' not in trajectory:
                print(f"⚠️  Skipping sample - no 'moves' field")
                continue
                
            moves = trajectory['moves']
            
            # Optionally include goal state in the sequence
            # Format: [start_state, SEP, goal_state, SEP, moves]
            sequence = []
            sequence.extend(start.flatten().tolist())
            
            # Add goal state if present (helps model know the target)
            if 'goal_state' in trajectory or 'goal' in trajectory:
                goal = np.array(trajectory.get('goal_state', trajectory.get('goal')))
                sequence.append(self.SEP_TOKEN)
                sequence.extend(goal.flatten().tolist())
            
            sequence.append(self.SEP_TOKEN)
            sequence.extend([self.move_to_token[m] for m in moves])
            
            if len(sequence) <= max_seq_length:
                self.samples.append(sequence)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sequence = self.samples[idx]
        
        # Pad sequence
        padded = sequence + [self.PAD_TOKEN] * (self.max_seq_length - len(sequence))
        padded = padded[:self.max_seq_length]
        
        # Input: all tokens except last, Target: all tokens except first
        x = torch.LongTensor(padded[:-1])
        y = torch.LongTensor(padded[1:])
        
        return x, y

class CoTTransformerDataset(Dataset):
    """
    Dataset for Chain-of-Thought transformer
    Format: [start_state] → [move_1, state_1, move_2, state_2, ...]
    INCLUDES intermediate states
    """
    def __init__(self, training_data, max_seq_length=1000):
        self.samples = []
        self.max_seq_length = max_seq_length
        
        # Same vocabulary as baseline
        self.move_to_token = {'up': 9, 'down': 10, 'left': 11, 'right': 12}
        self.token_to_move = {9: 'up', 10: 'down', 11: 'left', 12: 'right'}
        self.PAD_TOKEN = 13
        self.SEP_TOKEN = 14
        self.vocab_size = 15
        
        skipped_no_states = 0
        skipped_too_long = 0
        
        for trajectory in training_data:
            # Handle both 'start' and 'start_state' keys
            start = np.array(trajectory.get('start', trajectory.get('start_state')))
            moves = trajectory['moves']
            
            # Build CoT sequence with intermediate states
            # Format: [start_state, SEP, goal_state, SEP, move_1, SEP, state_1, SEP, ...]
            sequence = []
            sequence.extend(start.flatten().tolist())
            
            # Add goal state if present
            if 'goal_state' in trajectory or 'goal' in trajectory:
                goal = np.array(trajectory.get('goal_state', trajectory.get('goal')))
                sequence.append(self.SEP_TOKEN)
                sequence.extend(goal.flatten().tolist())
            
            sequence.append(self.SEP_TOKEN)
            
            # Add intermediate states if available
            if 'intermediate_states' in trajectory:
                states = trajectory['intermediate_states']
                for i, move in enumerate(moves):
                    sequence.append(self.SEP_TOKEN)
                    sequence.append(self.move_to_token[move])
                    if i + 1 < len(states):
                        sequence.append(self.SEP_TOKEN)
                        sequence.extend(np.array(states[i + 1]).flatten().tolist())
            else:
                # Fallback: just moves (like baseline)
                skipped_no_states += 1
                sequence.append(self.SEP_TOKEN)
                sequence.extend([self.move_to_token[m] for m in moves])
            
            if len(sequence) <= max_seq_length:
                self.samples.append(sequence)
            else:
                skipped_too_long += 1
        
        print(f"  Loaded {len(self.samples)} sequences")
        if skipped_no_states > 0:
            print(f"  ⚠️  Skipped {skipped_no_states} samples without intermediate_states")
        if skipped_too_long > 0:
            print(f"  ⚠️  Skipped {skipped_too_long} samples exceeding max_seq_length={max_seq_length}")
        
        if len(self.samples) == 0:
            print("\n❌ ERROR: No valid CoT samples created!")
            print("   This usually means:")
            print("   1. Data missing 'intermediate_states' field")
            print("   2. All sequences are too long (try increasing max_seq_length)")
            print("\n   Your data should look like:")
            print("   {")
            print("     'start_state': [[1,2,3],[4,5,6],[7,8,0]],")
            print("     'moves': ['up', 'right'],")
            print("     'intermediate_states': [[[...]], [[...]], [[...]]]")
            print("   }")
            raise ValueError("No valid CoT training samples!")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sequence = self.samples[idx]
        
        # Pad sequence
        padded = sequence + [self.PAD_TOKEN] * (self.max_seq_length - len(sequence))
        padded = padded[:self.max_seq_length]
        
        x = torch.LongTensor(padded[:-1])
        y = torch.LongTensor(padded[1:])
        
        return x, y

class PlanningTransformer(nn.Module):
    """
    Transformer model for 8-puzzle solving
    Can be trained with or without chain-of-thought
    """
    def __init__(self, vocab_size=15, d_model=128, nhead=4, num_layers=4, 
                 dim_feedforward=512, max_seq_length=100):
        super(PlanningTransformer, self).__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_length = max_seq_length
        
        # Token embedding
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional encoding
        self.pos_encoder = nn.Embedding(max_seq_length, d_model)
        
        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # Output projection
        self.output_layer = nn.Linear(d_model, vocab_size)
        
        # Move mapping
        self.move_to_token = {'up': 9, 'down': 10, 'left': 11, 'right': 12}
        self.token_to_move = {9: 'up', 10: 'down', 11: 'left', 12: 'right'}
        self.PAD_TOKEN = 13
        self.SEP_TOKEN = 14
    
    def forward(self, x, mask=None):
        batch_size, seq_len = x.shape
        
        # Token embeddings
        token_emb = self.embedding(x)
        
        # Positional embeddings
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.pos_encoder(positions)
        
        # Combine embeddings
        emb = token_emb + pos_emb
        
        # Create causal mask
        if mask is None:
            mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(x.device)
        
        # Transformer
        output = self.transformer_decoder(emb, emb, tgt_mask=mask, memory_mask=mask)
        
        # Project to vocabulary
        logits = self.output_layer(output)
        
        return logits
    
    def generate_solution(self, start_state, goal_state=None, max_length=50, 
                         temperature=0.1, use_cot=False):
        """
        Generate solution moves (and optionally states with CoT)
        
        FIXED: Properly checks sequence length against model's max_seq_length
        
        Args:
            start_state: 3x3 numpy array
            goal_state: Not used but kept for compatibility
            max_length: Maximum number of moves to generate
            temperature: Sampling temperature
            use_cot: Whether to generate intermediate states
        
        Returns:
            moves: List of moves
            states: List of intermediate states (if use_cot=True)
            success: Whether puzzle was solved
        """
        self.eval()
        
        # Start with initial state tokens
        sequence = start_state.flatten().tolist()
        sequence.append(self.SEP_TOKEN)
        
        moves = []
        states = [start_state.copy()]
        current_state = start_state.copy()
        
        with torch.no_grad():
            for step in range(max_length):
                # CRITICAL FIX: Check sequence length BEFORE generating
                # Leave room for: current sequence + some buffer
                if len(sequence) >= self.max_seq_length - 15:
                    # Sequence is getting too long for the model
                    break
                
                # Prepare input - truncate if needed (safety check)
                seq_to_use = sequence[-(self.max_seq_length-1):] if len(sequence) > self.max_seq_length - 1 else sequence
                x = torch.LongTensor([seq_to_use]).to(next(self.parameters()).device)
                
                # Generate next token
                logits = self.forward(x)
                next_token_logits = logits[0, -1, :] / temperature
                
                # Sample next token
                probs = torch.softmax(next_token_logits, dim=0)
                next_token = torch.multinomial(probs, 1).item()
                
                # Add to sequence
                sequence.append(next_token)
                
                # Parse token
                if next_token == self.SEP_TOKEN:
                    continue
                elif next_token in self.token_to_move:
                    # This is a move
                    move = self.token_to_move[next_token]
                    
                    # Try to apply move
                    new_state = self._try_apply_move(current_state, move)
                    if new_state is not None:
                        moves.append(move)
                        current_state = new_state
                        states.append(current_state.copy())
                        
                        # Check if solved (assume standard goal)
                        goal = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 0]])
                        if np.array_equal(current_state, goal):
                            return moves, states, True
                        
                        # If using CoT, add state tokens
                        if use_cot:
                            sequence.append(self.SEP_TOKEN)
                            sequence.extend(current_state.flatten().tolist())
                    else:
                        return moves, states, False
        
        return moves, states, False
    
    def _try_apply_move(self, board, move):
        """Try to apply a move, return new board or None if invalid"""
        board = board.copy()
        blank_pos = tuple(np.argwhere(board == 0)[0])
        row, col = blank_pos
        
        if move == 'up' and row > 0:
            board[row, col], board[row-1, col] = board[row-1, col], board[row, col]
            return board
        elif move == 'down' and row < 2:
            board[row, col], board[row+1, col] = board[row+1, col], board[row, col]
            return board
        elif move == 'left' and col > 0:
            board[row, col], board[row, col-1] = board[row, col-1], board[row, col]
            return board
        elif move == 'right' and col < 2:
            board[row, col], board[row, col+1] = board[row, col+1], board[row, col]
            return board
        return None
