"""
FIXED Blocks World Training Script

CRITICAL FIX: Learning rate changed from 0.001 → 0.0001

The original lr=0.001 caused:
- Training loss to INCREASE (divergence)  
- Baseline to predict POS tokens instead of blocks (80% confidence on wrong category!)
- Models to not learn basic action structure

With lr=0.0001:
- Loss should DECREASE consistently
- Should converge to <0.15
- Models should learn correct token categories
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import sys

sys.path.insert(0, '/Users/bmorris/Blocks/Working/Output')
from cot_comparison_experiment import PuzzleTransformer


class BlocksWorldDataset(Dataset):
    def __init__(self, data, max_seq_length, pad_token=9):
        self.data = data
        self.max_seq_length = max_seq_length
        self.pad_token = pad_token
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sequence = self.data[idx]['sequence']
        padded = sequence + [self.pad_token] * (self.max_seq_length - len(sequence))
        padded = padded[:self.max_seq_length]
        
        x = torch.tensor(padded[:-1], dtype=torch.long)
        y = torch.tensor(padded[1:], dtype=torch.long)
        
        return x, y


def train_blocks_world_models(
    #baseline_train_file="blocks_world_varied1-10_samesize_baseline_train.json",
    baseline_train_file="blocks_world_baseline_train.json",
    #baseline_test_file='blocks_world_varied1-10_samesize_baseline_test.json',
    baseline_test_file='blocks_world_baseline_test.json',
    #cot_train_file='blocks_world_varied1-10_samesize_cot_train.json',
    cot_train_file='blocks_world_cot_train.json',
    #cot_test_file='blocks_world_varied1-10_samesize_cot_test.json',
    cot_test_file='blocks_world_cot_test.json',
    #metadata_file='blocks_world_varied1-10_samesize_metadata.json',
    metadata_file='blocks_world_metadata.json',
    epochs=200,
    learning_rate=0.00001,  # ✓ FIXED: Was 0.001 (too high!)
    output_dir='./Output'
):
    """
    Train baseline and CoT models with CORRECT learning rate
    
    CRITICAL FIX:
    - Changed lr from 0.001 → 0.0001
    - This prevents divergence
    """
    print("="*70)
    print("BLOCKS WORLD: FIXED TRAINING")
    print("="*70)
    print("\n⚠️  CRITICAL FIX APPLIED:")
    print(f"   Learning rate: {learning_rate} (was 0.001 - too high!)")
    print("   Previous training had INCREASING loss (divergence)")
    print("   This training should have DECREASING loss")
    print()
    
    # Load metadata
    with open(f"{output_dir}/{metadata_file}", 'r') as f:
        metadata = json.load(f)
    
    vocab_size = metadata['vocab_size']
    baseline_max_len = metadata['recommended_max_seq_length']['baseline']
    cot_max_len = metadata['recommended_max_seq_length']['cot']
    
    print(f"Configuration:")
    print(f"  Vocabulary size: {vocab_size}")
    print(f"  Baseline max_seq_length: {baseline_max_len}")
    print(f"  CoT max_seq_length: {cot_max_len}")
    print(f"  Epochs: {epochs}")
    print(f"  Learning rate: {learning_rate} ✓")
    print()
    
    # Load data
    print("Loading data...")
    try:
        with open(f"{output_dir}/{baseline_train_file}", 'r') as f:
            baseline_train = json.load(f)
        with open(f"{output_dir}/{baseline_test_file}", 'r') as f:
            baseline_test = json.load(f)
        with open(f"{output_dir}/{cot_train_file}", 'r') as f:
            cot_train = json.load(f)
        with open(f"{output_dir}/{cot_test_file}", 'r') as f:
            cot_test = json.load(f)
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: Could not find data files!")
        print(f"   {e}")
        print(f"\n   Make sure you're in the correct directory:")
        print(f"   cd /Users/bmorris/Blocks/Working/Output")
        print(f"\n   And that these files exist:")
        print(f"   - {baseline_train_file}")
        print(f"   - {cot_train_file}")
        return None, None, None
    
    print(f"✓ Baseline train: {len(baseline_train)} samples")
    print(f"✓ Baseline test:  {len(baseline_test)} samples")
    print(f"✓ CoT train:      {len(cot_train)} samples")
    print(f"✓ CoT test:       {len(cot_test)} samples")
    
    # Verify same problems
    same_problems = all(
        baseline_train[i]['problem_idx'] == cot_train[i]['problem_idx']
        for i in range(len(baseline_train))
    )
    print(f"✓ Same problems verified: {same_problems}\n")
    
    # Create datasets
    baseline_dataset = BlocksWorldDataset(baseline_train, baseline_max_len)
    cot_dataset = BlocksWorldDataset(cot_train, cot_max_len)
    
    baseline_loader = DataLoader(baseline_dataset, batch_size=32, shuffle=True)
    cot_loader = DataLoader(cot_dataset, batch_size=32, shuffle=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    # ========== TRAIN BASELINE ==========
    print("="*70)
    print("TRAINING BASELINE (No CoT)")
    print("="*70)
    
    baseline_model = PuzzleTransformer(
        vocab_size=vocab_size,
        d_model=128,
        nhead=4,
        num_layers=4,
        dim_feedforward=512,
        max_seq_length=baseline_max_len
    ).to(device)
    
    criterion = nn.CrossEntropyLoss(ignore_index=9)
    optimizer = optim.Adam(baseline_model.parameters(), lr=learning_rate)
    
    baseline_losses = []
    print(f"\nTraining for {epochs} epochs...")
    print(f"Expected: Loss should DECREASE consistently\n")
    
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
            if epoch == 0:
                trend = ""
            elif avg_loss < baseline_losses[epoch - 10]:
                trend = "✓ decreasing (GOOD!)"
            else:
                trend = "✗ INCREASING (BAD - still broken!)"
            print(f"  Epoch [{epoch+1:>3}/{epochs}], Loss: {avg_loss:.4f} {trend}")
    
    # Save baseline
    baseline_save_path = f"{output_dir}/blocks_world_baseline_FIXED.pth"
    torch.save(baseline_model.state_dict(), baseline_save_path)
    
    print(f"\n{'='*70}")
    print("BASELINE TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"Initial loss: {baseline_losses[0]:.4f}")
    print(f"Final loss:   {baseline_losses[-1]:.4f}")
    
    if baseline_losses[-1] < baseline_losses[0]:
        improvement = (baseline_losses[0] - baseline_losses[-1]) / baseline_losses[0] * 100
        print(f"✓ Loss decreased by {improvement:.1f}% (GOOD!)")
        print(f"✓ Model learned successfully")
    else:
        print(f"✗ Loss INCREASED by {(baseline_losses[-1] - baseline_losses[0]) / baseline_losses[0] * 100:.1f}%")
        print(f"✗ Training still broken - check configuration!")
    
    print(f"\nModel saved: {baseline_save_path}")
    
    # ========== TRAIN COT ==========
    print("\n" + "="*70)
    print("TRAINING COT (With Intermediate States)")
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
    print(f"\nTraining for {epochs} epochs...")
    print(f"Expected: Loss should DECREASE consistently\n")
    
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
            if epoch == 0:
                trend = ""
            elif avg_loss < cot_losses[epoch - 10]:
                trend = "✓ decreasing (GOOD!)"
            else:
                trend = "✗ INCREASING (BAD - still broken!)"
            print(f"  Epoch [{epoch+1:>3}/{epochs}], Loss: {avg_loss:.4f} {trend}")
    
    # Save CoT
    cot_save_path = f"{output_dir}/blocks_world_cot_FIXED.pth"
    torch.save(cot_model.state_dict(), cot_save_path)
    
    print(f"\n{'='*70}")
    print("COT TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"Initial loss: {cot_losses[0]:.4f}")
    print(f"Final loss:   {cot_losses[-1]:.4f}")
    
    if cot_losses[-1] < cot_losses[0]:
        improvement = (cot_losses[0] - cot_losses[-1]) / cot_losses[0] * 100
        print(f"✓ Loss decreased by {improvement:.1f}% (GOOD!)")
        print(f"✓ Model learned successfully")
    else:
        print(f"✗ Loss INCREASED - training broken!")
    
    print(f"\nModel saved: {cot_save_path}")
    
    # ========== SUMMARY ==========
    print("\n" + "="*70)
    print("TRAINING SUMMARY")
    print("="*70)
    
    baseline_final = baseline_losses[-1]
    cot_final = cot_losses[-1]
    
    print(f"\nFinal Training Loss:")
    print(f"  Baseline: {baseline_final:.4f}")
    print(f"  CoT:      {cot_final:.4f}")
    
    if cot_final < baseline_final:
        loss_improvement = (baseline_final - cot_final) / baseline_final * 100
        print(f"  CoT lower by: {loss_improvement:.1f}%")
    
    print(f"\nExpected with proper training:")
    print(f"  ✓ Both losses < 0.2 (converged)")
    print(f"  ✓ CoT loss 20-40% lower than baseline")
    print(f"  ✓ Baseline predicts blocks (not POS tokens!)")
    print(f"  ✓ Both models solve >40% of problems")
    
    print(f"\nModels saved:")
    print(f"  Baseline: {baseline_save_path}")
    print(f"  CoT:      {cot_save_path}")
    
    # Save history
    history = {
        'baseline_losses': baseline_losses,
        'cot_losses': cot_losses,
        'baseline_final': baseline_final,
        'cot_final': cot_final,
        'learning_rate': learning_rate,
        'epochs': epochs,
        'vocab_size': vocab_size,
        'fix_applied': 'Changed lr from 0.001 to 0.0001 to prevent divergence'
    }
    
    history_file = f"{output_dir}/blocks_world_training_history_FIXED.json"
    with open(history_file, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"  History: {history_file}")
    
    print("\n" + "="*70)
    print("NEXT STEPS")
    print("="*70)
    print("\n1. Verify models work:")
    print("   python3 test_blocks.py")
    print("   → Should predict blocks (not POS tokens!)")
    print()
    print("2. Re-run full evaluation:")
    print("   python3 run_decay_analysis.py")
    print("   → Get valid results with properly trained models")
    
    return baseline_model, cot_model, history


if __name__ == "__main__":
    print("\n" + "="*70)
    print("RETRAINING WITH FIXED LEARNING RATE")
    print("="*70)
    print("\nThis fixes the baseline model that was predicting")
    print("POS tokens with 80% confidence instead of blocks.")
    print()
    print("Root cause: lr=0.001 was too high (caused divergence)")
    print("Fix: lr=0.0001 (10x lower)")
    print()
    
    baseline_model, cot_model, history = train_blocks_world_models(
        epochs=200,
        learning_rate=0.0001,
        output_dir='/Users/bmorris/Blocks/Working/Output'
    )
    
    if baseline_model is not None:
        print("\n✅ TRAINING COMPLETE")
        print("\nTest the models and verify they work properly!")
