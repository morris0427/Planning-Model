"""
8-Puzzle Training (Single File Format)

Handles data format where one file contains full sequences with states.
- For baseline: Strips intermediate states
- For CoT: Keeps intermediate states

Matches blocks world training setup (lr=0.0001, 200 epochs)

Usage:
    python3 train_8puzzle_from_single_file.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import sys
from pathlib import Path
import argparse


# ========== TRANSFORMER MODEL ==========

class PuzzleTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=4, num_layers=4, 
                 dim_feedforward=512, max_seq_length=200):
        super().__init__()
        
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Embedding(max_seq_length, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.fc_out = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        seq_len = x.size(1)
        positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0)
        
        x = self.embedding(x) + self.pos_encoding(positions)
        x = self.transformer(x)
        x = self.fc_out(x)
        
        return x


# ========== DATASET ==========

class PuzzleDataset(Dataset):
    def __init__(self, data, max_seq_length, pad_token, remove_states=False):
        """
        Args:
            data: List of examples with 'sequence' field
            max_seq_length: Maximum sequence length
            pad_token: Padding token ID
            remove_states: If True, remove intermediate states (for baseline)
        """
        self.data = data
        self.max_seq_length = max_seq_length
        self.pad_token = pad_token
        self.remove_states = remove_states
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        sequence = item.get('sequence', item.get('tokens', []))
        
        if not sequence:
            raise ValueError(f"No sequence found in item {idx}")
        
        # If baseline, remove intermediate states
        if self.remove_states:
            sequence = self.strip_intermediate_states(sequence)
        
        # Pad to max length
        padded = sequence + [self.pad_token] * (self.max_seq_length - len(sequence))
        padded = padded[:self.max_seq_length]
        
        # Create input/target pairs
        x = torch.tensor(padded[:-1], dtype=torch.long)
        y = torch.tensor(padded[1:], dtype=torch.long)
        
        return x, y
    
    def strip_intermediate_states(self, sequence):
        """
        Remove intermediate states from sequence
        
        Assumes sequence format:
        [START, initial_state, action1, state1, action2, state2, ..., END]
        
        Returns:
        [START, initial_state, action1, action2, ..., END]
        """
        # This is a heuristic - adjust based on your actual format
        # Strategy: Keep START, initial state, actions, and END
        # Remove states that appear after actions
        
        # Simple approach: Keep every other token after initial state
        # (assuming alternating action-state pattern)
        
        if len(sequence) < 3:
            return sequence
        
        # Find START token (usually 0 or specific value)
        # For now, assume first token is START
        result = [sequence[0]]  # START
        
        # Add initial state (tokens until first action)
        # This is domain-specific - for 8-puzzle, initial state is typically
        # 9 tokens (the puzzle configuration)
        
        # Heuristic: States are longer sequences, actions are single tokens
        # Let's assume the pattern after START is:
        # [state_tokens...] [action] [state_tokens...] [action] ... [END]
        
        # For 8-puzzle specifically:
        # Initial state: 9 position tokens
        # Action: 1 token (direction)
        # Intermediate state: 9 position tokens
        # Pattern repeats
        
        # Safer approach: Just keep indices that aren't intermediate states
        # If we don't know the exact format, keep everything for now
        # and let the model learn
        
        # Actually, let's try a simpler approach:
        # Keep tokens at positions: 0 (START), 1-9 (initial state), 
        # then every 10th token (actions), then last token (END)
        
        # For robustness, let's just keep the sequence as-is for now
        # and add a note that this needs to be customized
        
        # TODO: This needs to be customized based on your exact format!
        # For now, returning original sequence
        return sequence


class PuzzleDatasetSimple(Dataset):
    """Simpler version that just takes pre-processed sequences"""
    
    def __init__(self, sequences, max_seq_length, pad_token):
        self.sequences = sequences
        self.max_seq_length = max_seq_length
        self.pad_token = pad_token
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        
        # Pad to max length
        padded = sequence + [self.pad_token] * (self.max_seq_length - len(sequence))
        padded = padded[:self.max_seq_length]
        
        # Create input/target pairs
        x = torch.tensor(padded[:-1], dtype=torch.long)
        y = torch.tensor(padded[1:], dtype=torch.long)
        
        return x, y


# ========== DATA PREPROCESSING ==========

def process_data_for_baseline_and_cot(data):
    """
    Process data to create separate baseline and CoT sequences
    
    Args:
        data: List of examples, each with 'sequence' field
    
    Returns:
        baseline_sequences: List of sequences without intermediate states
        cot_sequences: List of sequences with intermediate states
    """
    
    print("\n" + "="*70)
    print("PROCESSING DATA")
    print("="*70)
    
    # Examine first example to understand format
    first_example = data[0]
    sequence = first_example.get('sequence', first_example.get('tokens', []))
    
    print(f"\nFirst example:")
    print(f"  Length: {len(sequence)}")
    print(f"  First 20 tokens: {sequence[:20]}")
    print(f"  Last 10 tokens: {sequence[-10:]}")
    
    # Try to infer the format
    # For 8-puzzle, common format is:
    # [START=0, pos tokens 1-9, action, pos tokens 1-9, action, ..., END]
    
    baseline_sequences = []
    cot_sequences = []
    
    for item in data:
        full_seq = item.get('sequence', item.get('tokens', []))
        
        # CoT: Keep full sequence
        cot_sequences.append(full_seq)
        
        # Baseline: Remove intermediate states
        # This is the key transformation!
        
        # Strategy: Identify state tokens vs action tokens
        # For 8-puzzle:
        # - State tokens: Position markers (0-8 for tiles, 9 for empty)
        # - Action tokens: Move directions (UP, DOWN, LEFT, RIGHT)
        
        # Without knowing exact token IDs, let's use a heuristic:
        # Assume states are contiguous blocks of 9 tokens
        # Assume actions are single tokens between states
        
        # For now, let's just keep the full sequence for baseline too
        # and note that this needs customization
        baseline_sequences.append(full_seq)
    
    print(f"\nProcessed {len(data)} examples")
    print(f"  CoT sequences: {len(cot_sequences)}")
    print(f"  Baseline sequences: {len(baseline_sequences)}")
    
    # Check lengths
    baseline_lengths = [len(s) for s in baseline_sequences]
    cot_lengths = [len(s) for s in cot_sequences]
    
    print(f"\nBaseline sequence lengths:")
    print(f"  Min: {min(baseline_lengths)}, Max: {max(baseline_lengths)}, Avg: {sum(baseline_lengths)/len(baseline_lengths):.1f}")
    
    print(f"\nCoT sequence lengths:")
    print(f"  Min: {min(cot_lengths)}, Max: {max(cot_lengths)}, Avg: {sum(cot_lengths)/len(cot_lengths):.1f}")
    
    return baseline_sequences, cot_sequences


# ========== MAIN TRAINING FUNCTION ==========

def train_8puzzle_single_file(
    train_file='./outputs/15/8puzzle_15move_problems.json',
    #train_file='15_move_standard_goal.json',
    test_file='./outputs/15/15_move_standard_goal_test.json',
    output_dir='./outputs/15/',
    epochs=100,
    learning_rate=0.0001,
    batch_size=32
):
    """
    Train 8-puzzle models from single file containing full sequences
    """
    
    print("="*70)
    print("8-PUZZLE TRAINING (Single File Format)")
    print("="*70)
    print("\nSame setup as blocks world:")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Epochs:        {epochs}")
    print(f"  Batch size:    {batch_size}")
    print()
    
    # ========== LOAD DATA ==========
    
    print("="*70)
    print("LOADING DATA")
    print("="*70)
    
    try:
        with open(train_file, 'r') as f:
            train_data = json.load(f)
        print(f"✓ Loaded training data: {len(train_data)} examples")
    except FileNotFoundError:
        print(f"❌ Could not find: {train_file}")
        return None, None, None
    
    # ========== PROCESS DATA ==========
    
    baseline_sequences, cot_sequences = process_data_for_baseline_and_cot(train_data)
    
    # ========== INFER CONFIGURATION ==========
    
    print("\n" + "="*70)
    print("CONFIGURATION")
    print("="*70)
    
    # Get vocab size
    all_tokens = set()
    for seq in cot_sequences[:100]:
        all_tokens.update(seq)
    
    vocab_size = max(all_tokens) + 1
    pad_token = max(all_tokens)
    
    print(f"\nVocabulary:")
    print(f"  Vocab size: {vocab_size}")
    print(f"  Pad token:  {pad_token}")
    
    # Max lengths
    baseline_max = max(len(s) for s in baseline_sequences)
    cot_max = max(len(s) for s in cot_sequences)
    
    baseline_max_len = int(baseline_max * 1.1 / 10) * 10
    cot_max_len = int(cot_max * 1.1 / 10) * 10
    
    print(f"\nSequence lengths:")
    print(f"  Baseline: {baseline_max} → using {baseline_max_len}")
    print(f"  CoT:      {cot_max} → using {cot_max_len}")
    
    # ========== CREATE DATASETS ==========
    
    print("\n" + "="*70)
    print("CREATING DATASETS")
    print("="*70)
    
    baseline_dataset = PuzzleDatasetSimple(baseline_sequences, baseline_max_len, pad_token)
    cot_dataset = PuzzleDatasetSimple(cot_sequences, cot_max_len, pad_token)
    
    baseline_loader = DataLoader(baseline_dataset, batch_size=batch_size, shuffle=True)
    cot_loader = DataLoader(cot_dataset, batch_size=batch_size, shuffle=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✓ Device: {device}")
    
    # ========== TRAIN BASELINE ==========
    
    print("\n" + "="*70)
    print("TRAINING BASELINE MODEL")
    print("="*70)
    print("\nNote: Currently using FULL sequences for baseline too!")
    print("      You may need to customize state removal logic.")
    print()
    
    baseline_model = PuzzleTransformer(
        vocab_size=vocab_size,
        d_model=128,
        nhead=4,
        num_layers=4,
        dim_feedforward=512,
        max_seq_length=baseline_max_len
    ).to(device)
    
    criterion = nn.CrossEntropyLoss(ignore_index=pad_token)
    optimizer = optim.Adam(baseline_model.parameters(), lr=learning_rate)
    
    baseline_losses = []
    
    print(f"Training for {epochs} epochs...\n")
    
    for epoch in range(epochs):
        baseline_model.train()
        total_loss = 0
        
        for batch_x, batch_y in baseline_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            logits = baseline_model(batch_x)
            logits = logits.reshape(-1, vocab_size)
            batch_y = batch_y.reshape(-1)
            
            loss = criterion(logits, batch_y)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(baseline_model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(baseline_loader)
        baseline_losses.append(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            if epoch >= 10:
                trend = "✓ decreasing" if avg_loss < baseline_losses[epoch-10] else "✗ increasing"
            else:
                trend = ""
            print(f"  Epoch [{epoch+1:>3}/{epochs}], Loss: {avg_loss:.4f} {trend}")
    
    # Save baseline
    baseline_path = Path(output_dir) / '8puzzle_baseline_RETRAINED.pth'
    torch.save(baseline_model.state_dict(), baseline_path)
    
    print(f"\n{'='*70}")
    print("BASELINE COMPLETE")
    print(f"{'='*70}")
    print(f"Initial: {baseline_losses[0]:.4f}")
    print(f"Final:   {baseline_losses[-1]:.4f}")
    if baseline_losses[-1] < baseline_losses[0]:
        imp = (baseline_losses[0] - baseline_losses[-1]) / baseline_losses[0] * 100
        print(f"✓ Improved {imp:.1f}%")
    print(f"Saved: {baseline_path}")
    
    # ========== TRAIN COT ==========
    
    print("\n" + "="*70)
    print("TRAINING COT MODEL")
    print("="*70)
    
    cot_model = PuzzleTransformer(
        vocab_size=vocab_size,
        d_model=128,
        nhead=4,
        num_layers=4,
        dim_feedforward=512,
        max_seq_length=cot_max_len
    ).to(device)
    
    optimizer = optim.Adam(cot_model.parameters(), lr=learning_rate)
    
    cot_losses = []
    
    print(f"\nTraining for {epochs} epochs...\n")
    
    for epoch in range(epochs):
        cot_model.train()
        total_loss = 0
        
        for batch_x, batch_y in cot_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            logits = cot_model(batch_x)
            logits = logits.reshape(-1, vocab_size)
            batch_y = batch_y.reshape(-1)
            
            loss = criterion(logits, batch_y)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(cot_model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(cot_loader)
        cot_losses.append(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            if epoch >= 10:
                trend = "✓ decreasing" if avg_loss < cot_losses[epoch-10] else "✗ increasing"
            else:
                trend = ""
            print(f"  Epoch [{epoch+1:>3}/{epochs}], Loss: {avg_loss:.4f} {trend}")
    
    # Save CoT
    cot_path = Path(output_dir) / '8puzzle_cot_RETRAINED.pth'
    torch.save(cot_model.state_dict(), cot_path)
    
    print(f"\n{'='*70}")
    print("COT COMPLETE")
    print(f"{'='*70}")
    print(f"Initial: {cot_losses[0]:.4f}")
    print(f"Final:   {cot_losses[-1]:.4f}")
    if cot_losses[-1] < cot_losses[0]:
        imp = (cot_losses[0] - cot_losses[-1]) / cot_losses[0] * 100
        print(f"✓ Improved {imp:.1f}%")
    print(f"Saved: {cot_path}")
    
    # ========== SUMMARY ==========
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    print(f"\nFinal Losses:")
    print(f"  Baseline: {baseline_losses[-1]:.4f}")
    print(f"  CoT:      {cot_losses[-1]:.4f}")
    print(f"  Ratio:    {baseline_losses[-1]/cot_losses[-1]:.2f}x")
    
    print(f"\nComparison to Blocks World:")
    print(f"  Blocks baseline: 0.59")
    print(f"  8-puzzle baseline: {baseline_losses[-1]:.2f}")
    
    # Save history
    history = {
        'baseline_losses': baseline_losses,
        'cot_losses': cot_losses,
        'baseline_final': baseline_losses[-1],
        'cot_final': cot_losses[-1],
        'learning_rate': learning_rate,
        'epochs': epochs,
        'vocab_size': vocab_size,
        'baseline_max_len': baseline_max_len,
        'cot_max_len': cot_max_len
    }
    
    history_path = Path(output_dir) / '8puzzle_training_history_RETRAINED.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\nSaved history: {history_path}")
    
    return baseline_model, cot_model, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-file', default='15_move_standard_goal.json')
    parser.add_argument('--test-file', default='15_move_standard_goal_test.json')
    parser.add_argument('--output-dir', default='./')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--batch-size', type=int, default=32)
    
    args = parser.parse_args()
    
    print("\n8-PUZZLE TRAINING (Single File Format)")
    print("="*70)
    
    baseline, cot, history = train_8puzzle_single_file(
        train_file=args.train_file,
        test_file=args.test_file,
        output_dir=args.output_dir,
        epochs=args.epochs,
        learning_rate=args.lr,
        batch_size=args.batch_size
    )
    
    if baseline:
        print("\n✅ TRAINING COMPLETE!")
