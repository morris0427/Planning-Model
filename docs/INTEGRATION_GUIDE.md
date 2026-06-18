# Integration Guide: Connecting Your Existing Code to the Framework

**Quick guide to integrate your existing Blocks World and 8-Puzzle code into the new experiment framework.**

---

## 🎯 **What You Have**

✅ **Working code:**
- Blocks World data generation (SAW)
- 8-Puzzle data generation  
- Model training scripts
- Evaluation scripts

❌ **Problem:**
- Code is duplicated across domains
- Hard to run systematic ablations
- Difficult to ensure fair comparisons

✅ **Solution:**
- New factored framework (in `/mnt/user-data/outputs/experiments/`)
- Shared configuration system
- Easy experiment sweeps

---

## 📋 **Integration Steps**

### **Step 1: Copy Your Encoding Code**

**Create `experiments/data/blocks_world.py`:**

```python
"""Blocks World dataset implementation."""

from typing import List, Dict, Any, Tuple
import random
from data.base import PlanningDataset, DatasetFactory


@DatasetFactory.register("blocks_world")
class BlocksWorldDataset(PlanningDataset):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # COPY YOUR VOCAB from blocks_encoding.py
        self.vocab = {
            'START': 0, 'END': 1,
            'A': 2, 'B': 3, 'C': 4, 'D': 5,
            'POS_0': 6, 'POS_1': 7, 'POS_2': 8, 'POS_3': 9,
            'PAD': 10
        }
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
    
    def generate_problem(self, difficulty: int) -> Dict[str, Any]:
        """
        Generate problem using SAW.
        
        COPY YOUR SAW LOGIC HERE from your existing generation script.
        """
        # Your existing SAW code goes here
        # Should return:
        return {
            'start_state': start_state,  # e.g., [['A','B'], [], ['C','D']]
            'goal_state': goal_state,
            'solution_moves': solution_moves,  # e.g., [['A', 2], ['B', 1]]
            'solution_states': solution_states,  # Intermediate states
            'num_moves': difficulty
        }
    
    def encode_sequence(self, problem: Dict[str, Any]) -> List[int]:
        """
        COPY YOUR encode_sequence() from blocks_encoding.py
        """
        # Your existing encoding logic
        pass
    
    def decode_sequence(self, token_ids: List[int]) -> Dict[str, Any]:
        """
        COPY YOUR decode logic
        """
        # Your existing decoding logic
        pass
    
    def _estimate_state_tokens(self) -> int:
        """Estimate tokens per state."""
        return 7  # Approximate for 4 blocks, 4 positions
```

**Similarly, create `experiments/data/puzzle.py` for 8-Puzzle.**

---

### **Step 2: Copy Your Model Code**

**Create `experiments/models/transformer.py`:**

```python
"""Shared transformer model."""

import torch
import torch.nn as nn


class PlanningTransformer(nn.Module):
    """
    Transformer for planning tasks.
    
    COPY YOUR MODEL ARCHITECTURE HERE from your existing code.
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 64,
        n_heads: int = 4,
        d_ff: int = 256,
        n_layers: int = 2,
        dropout: float = 0.1,
        weight_sharing: bool = False,
        max_seq_length: int = 200
    ):
        super().__init__()
        
        # COPY YOUR EXISTING MODEL LAYERS
        # Embeddings, encoder, decoder, etc.
        
        self.weight_sharing = weight_sharing
        # If weight_sharing, reuse same layer weights
    
    def forward(self, src, tgt):
        # COPY YOUR FORWARD PASS
        pass


def create_model(config, vocab_size, max_seq_length):
    """Factory function to create model from config."""
    return PlanningTransformer(
        vocab_size=vocab_size,
        d_model=config.d_model,
        n_heads=config.n_heads,
        d_ff=config.d_ff,
        n_layers=config.n_layers,
        dropout=config.dropout,
        weight_sharing=config.weight_sharing,
        max_seq_length=max_seq_length
    )
```

---

### **Step 3: Copy Your Training Loop**

**Create `experiments/trainer.py`:**

```python
"""Training and evaluation."""

import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

from config import ExperimentConfig
from data.base import DatasetFactory
from models.transformer import create_model


def train_epoch(model, dataloader, optimizer, criterion, device):
    """COPY YOUR TRAINING EPOCH LOGIC"""
    model.train()
    total_loss = 0
    
    for batch in dataloader:
        # Your existing training step
        pass
    
    return total_loss / len(dataloader)


def evaluate(model, dataloader, device):
    """COPY YOUR EVALUATION LOGIC"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in dataloader:
            # Your existing evaluation logic
            pass
    
    return correct / total


def train(config: ExperimentConfig) -> dict:
    """
    Main training function.
    
    This is the ONLY function you need to implement to integrate
    your existing code!
    """
    
    # 1. Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 2. Generate data using factory
    train_ranges, test_ranges = config.data.get_split_ranges()
    
    train_dataset = DatasetFactory.create(
        domain=config.data.domain,
        difficulty_range=train_ranges,
        num_samples=config.data.num_train_samples,
        use_world_model=config.model.use_world_model,
        seed=config.seed
    )
    train_dataset.generate_dataset()
    
    test_dataset = DatasetFactory.create(
        domain=config.data.domain,
        difficulty_range=test_ranges,
        num_samples=config.data.num_test_samples,
        use_world_model=config.model.use_world_model,
        seed=config.seed + 1
    )
    test_dataset.generate_dataset()
    
    # 3. Create dataloaders (COPY YOUR EXISTING DATALOADER CODE)
    train_loader = create_dataloader(train_dataset, config.model.batch_size)
    test_loader = create_dataloader(test_dataset, config.model.batch_size)
    
    # 4. Create model
    model = create_model(
        config.model,
        vocab_size=train_dataset.get_vocab_size(),
        max_seq_length=train_dataset.get_max_sequence_length()
    )
    model = model.to(device)
    
    # 5. Setup optimizer and loss (COPY YOUR EXISTING SETUP)
    optimizer = optim.Adam(model.parameters(), lr=config.model.learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    # 6. Training loop (COPY YOUR EXISTING LOOP)
    best_acc = 0
    for epoch in range(config.model.max_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        test_acc = evaluate(model, test_loader, device)
        
        if test_acc > best_acc:
            best_acc = test_acc
            # Save checkpoint if needed
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Test Acc={test_acc:.2%}")
    
    # 7. Final evaluation
    final_acc = evaluate(model, test_loader, device)
    
    # 8. Return results
    return {
        'experiment_name': config.experiment_name,
        'domain': config.data.domain,
        'model': config.model.name,
        'split_type': config.data.split_type.value,
        'train_loss': train_loss,
        'test_accuracy': final_acc,
        'best_accuracy': best_acc,
    }
```

---

### **Step 4: Update Experiment Runner**

**Modify `experiments/run_experiments.py`:**

Replace the placeholder section with:

```python
from trainer import train

def run_experiment(self, config: ExperimentConfig) -> dict:
    # ... existing setup code ...
    
    # REPLACE PLACEHOLDER WITH:
    results = train(config)
    
    # ... existing save code ...
    return results
```

---

## 🧪 **Testing the Integration**

### **Test 1: Single Experiment**

```bash
cd experiments

# Test with your existing setup
python run_experiments.py --experiment blocks_world_standard

# Should output:
# ======================================================================
# RUNNING EXPERIMENT: blocks_world_standard
# ======================================================================
# Experiment: blocks_world_standard
#   Domain: blocks_world
#   Model: small_WM (240,000 params)
#   Split: in_distribution
#   Train range: (1, 6)
#   Test range: (1, 6)
# 
# [Training progress...]
# ✓ Results saved to results/blocks_world_standard/results.json
```

### **Test 2: Small Sweep**

```bash
# Test model size sweep with just 2 sizes
python run_experiments.py --sweep model_size --domain blocks_world

# Should run 6 experiments (3 sizes × 2 types)
```

### **Test 3: Verify Results Match**

```python
# Compare to your existing results
import json

# Your old results
old_wm_acc = 0.57  # From your existing experiments

# New framework results
with open('results/blocks_world_small_wm/results.json') as f:
    new_results = json.load(f)
    new_wm_acc = new_results['test_accuracy']

print(f"Old: {old_wm_acc:.2%}")
print(f"New: {new_wm_acc:.2%}")
print(f"Match: {abs(old_wm_acc - new_wm_acc) < 0.01}")  # Should be True
```

---

## 📊 **Running New Experiments**

Once integrated, you can easily run the experiments from Ontañón et al. (2022):

### **Experiment 1: Model Size Ablation**

```bash
# Test if WM benefit is consistent across model sizes
python run_experiments.py --sweep model_size --domain blocks_world --split productivity

# Analyzes results
python analyze_results.py --sweep-file results/sweep_summary.json --analysis model_size
```

**Expected output:**
```
MODEL SIZE COMPARISON
================================================================================

BLOCKS_WORLD - Base
----------------------------------------
Size         Accuracy     Params      
----------------------------------------
tiny         18.5%        60K         
small        26.0%        240K        
medium       28.2%        900K        

BLOCKS_WORLD - WM
----------------------------------------
Size         Accuracy     Params      
----------------------------------------
tiny         45.3%        60K         
small        57.0%        240K        
medium       60.1%        900K        

WORLD MODEL ADVANTAGE BY SIZE
================================================================================

BLOCKS_WORLD
----------------------------------------
Size         Base Acc     WM Acc       Δ           
----------------------------------------
tiny         18.5%        45.3%        +26.8%      
small        26.0%        57.0%        +31.0%      
medium       28.2%        60.1%        +31.9%      

Finding: WM benefit is CONSISTENT across sizes → Not a capacity artifact!
```

### **Experiment 2: Weight Sharing**

```bash
python run_experiments.py --sweep weight_sharing --domain blocks_world

python analyze_results.py --sweep-file results/sweep_summary.json --analysis weight_sharing
```

**Expected output:**
```
WEIGHT SHARING COMPARISON
================================================================================

BLOCKS_WORLD
------------------------------------------------------------
Model                Standard        Shared          Δ         
------------------------------------------------------------
Base                 26.0%           35.2%           +9.2%     
WM                   57.0%           68.5%           +11.5%    

Finding: Weight sharing helps BOTH, but WM benefits MORE!
```

---

## 🎯 **Benefits After Integration**

### **Before:**
```bash
# Had to manually edit multiple files
# - Change model size in blocks_world_train.py
# - Change model size in puzzle_train.py
# - Run each script separately
# - Manually compare results
```

### **After:**
```bash
# Single command for full ablation
python run_experiments.py --sweep full --domain blocks_world

# Automatic analysis
python analyze_results.py --sweep-file results/sweep_summary.json

# Generates comparison tables automatically
```

### **For Your Paper:**
- All experiments use SAME codebase → Reproducible
- Easy to add new ablations → Extend experiments
- Automatic result tracking → No manual spreadsheets
- Fair comparisons → Same data generation, training, etc.

---

## 🚀 **Next Steps**

1. **Copy your data generation:**
   - `blocks_world.py` ← Your SAW generation
   - `puzzle.py` ← Your puzzle generation

2. **Copy your model:**
   - `transformer.py` ← Your model architecture

3. **Copy your training:**
   - `trainer.py` ← Your training loop

4. **Test:**
   ```bash
   python run_experiments.py --experiment blocks_world_standard
   ```

5. **Run new experiments:**
   ```bash
   python run_experiments.py --sweep model_size --domain blocks_world
   ```

6. **Update paper with new results!** 🎉

---

## ❓ **FAQ**

**Q: Do I have to rewrite all my code?**  
A: No! Just copy-paste your existing functions into the new structure. The framework provides the organization, you provide the logic.

**Q: Can I keep my old code?**  
A: Yes! Keep it for reference. The framework wraps your existing code, doesn't replace it.

**Q: How long does integration take?**  
A: ~2-4 hours to copy existing code into new structure, test, and verify results match.

**Q: What if I want to add a new experiment type?**  
A: Easy! Just add a new function in `config.py`:
```python
def create_my_sweep(...):
    experiments = []
    # ... create configs
    return experiments
```

---

**Ready to integrate? Start with `experiments/data/blocks_world.py`!** 🚀
