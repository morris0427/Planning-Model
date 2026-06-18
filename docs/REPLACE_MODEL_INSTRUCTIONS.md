# IMPORTANT: Replace PlanningTransformer Placeholder

The `trainer.py` file currently has a **placeholder PlanningTransformer** implementation.

## ⚠️ Action Required

You need to replace it with your actual `PuzzleTransformer` from `cot_comparison_experiment.py`.

---

## 🔧 How to Do This

### **Step 1: Find Your PuzzleTransformer**

Your model is defined in `/Users/bmorris/Blocks/Working/Output/cot_comparison_experiment.py`.

### **Step 2: Copy the Class**

1. Open `cot_comparison_experiment.py`
2. Find the `class PuzzleTransformer(nn.Module):` definition
3. Copy the entire class (typically 50-100 lines)

### **Step 3: Replace in trainer.py**

1. Open `experiments/trainer.py`
2. Find the `class PlanningTransformer(nn.Module):` section (around line 60)
3. **Delete the entire placeholder class**
4. **Paste your `PuzzleTransformer` class**
5. **Rename the class from `PuzzleTransformer` to `PlanningTransformer`**

### **Step 4: Add Weight Sharing Support (Optional)**

If your current model doesn't support weight sharing, you can add it later. For now, just make sure the `__init__` signature includes:

```python
def __init__(
    self,
    vocab_size: int,
    d_model: int = 128,
    nhead: int = 4,
    num_layers: int = 4,
    dim_feedforward: int = 512,
    max_seq_length: int = 200,
    dropout: float = 0.1,
    weight_sharing: bool = False  # Add this parameter
):
```

If `weight_sharing=True`, you can just ignore it for now (implement later).

---

## 📋 Example Structure

After replacement, `trainer.py` should look like:

```python
# ... imports ...

class SequenceDataset(Dataset):
    # ... existing code ...
    pass

class PlanningTransformer(nn.Module):  # <-- YOUR CODE HERE
    """
    Your actual PuzzleTransformer implementation.
    Renamed to PlanningTransformer.
    """
    
    def __init__(self, vocab_size, d_model=128, ...):
        super().__init__()
        
        # YOUR MODEL ARCHITECTURE
        # Copy from PuzzleTransformer
        
        self.embedding = ...
        self.transformer = ...
        self.fc_out = ...
    
    def forward(self, x):
        # YOUR FORWARD PASS
        # Copy from PuzzleTransformer
        
        return logits

# ... rest of trainer.py (train_epoch, evaluate, train functions) ...
```

---

## ✅ Verification

After replacing, test that it works:

```bash
cd experiments
python trainer.py
```

Should output:
```
Testing trainer...
Running quick test with 100 samples, 5 epochs...
[Training progress...]
✓ Trainer test complete!
```

---

## 🚨 Common Issues

### **Issue 1: Import Errors**
```python
# If your PuzzleTransformer imports other modules, add them at top:
import torch.nn.functional as F
# etc.
```

### **Issue 2: Different Parameter Names**
Make sure your constructor parameters match what `train()` function expects:
- `vocab_size` (required)
- `d_model` (maps to config.model.d_model)
- `nhead` (maps to config.model.n_heads)
- `num_layers` (maps to config.model.n_layers)
- `dim_feedforward` (maps to config.model.d_ff)

### **Issue 3: Device Handling**
Your model should handle `.to(device)` correctly. The trainer calls:
```python
model = PlanningTransformer(...).to(device)
```

---

## 💡 Quick Checklist

- [ ] Found PuzzleTransformer in cot_comparison_experiment.py
- [ ] Copied entire class definition
- [ ] Pasted into trainer.py, replacing placeholder
- [ ] Renamed class to PlanningTransformer
- [ ] Verified __init__ parameters match
- [ ] Tested with `python trainer.py`
- [ ] No import errors
- [ ] Model trains successfully

---

**Once complete, you'll have a fully working trainer integrated with the framework!** 🎉

The framework will then be able to run experiments like:
```bash
python run_experiments.py --experiment blocks_world_standard
python run_experiments.py --sweep model_size --domain blocks_world
```
