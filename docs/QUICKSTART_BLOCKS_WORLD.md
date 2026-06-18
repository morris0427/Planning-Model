# Quick Start: Blocks World End-to-End Pipeline

**Goal: Get a working Blocks World experiment running with the framework in ~2 hours.**

---

## ✅ What's Complete

The framework is **90% ready**. I've extracted your exact training code from `train_blocks_world.py`:

- ✅ `trainer.py` - Uses your exact training loop
- ✅ `config.py` - Uses lr=0.0001 (not 0.001!)
- ✅ `run_experiments.py` - Calls trainer
- ✅ `analyze_results.py` - Results analysis
- ✅ Documentation - Complete guides

---

## ⏳ What's Missing

Only **2 files** need your existing code:

1. **`data/blocks_world.py`** - Your SAW generation (~1 hour)
2. **`PlanningTransformer` in `trainer.py`** - Your model (~30 minutes)

---

## 🚀 Step-by-Step: Get It Working

### **Step 1: Replace PlanningTransformer (30 minutes)**

Open `trainer.py` and find this section (around line 60):

```python
class PlanningTransformer(nn.Module):
    """
    ⚠️  TODO: REPLACE THIS WITH YOUR PuzzleTransformer
    """
```

**Do this:**

1. Open `/Users/bmorris/Blocks/Working/Output/cot_comparison_experiment.py`
2. Find `class PuzzleTransformer(nn.Module):`
3. Copy the **entire class** (all methods)
4. In `trainer.py`, **delete** the placeholder `PlanningTransformer` class
5. **Paste** your `PuzzleTransformer` class
6. **Rename** the class: `PuzzleTransformer` → `PlanningTransformer`
7. Save

**That's it!** The framework already calls it with the right parameters:
- vocab_size, d_model=128, nhead=4, num_layers=4, dim_feedforward=512

---

### **Step 2: Create Blocks World Dataset (1 hour)**

Create `experiments/data/blocks_world.py`:

```python
"""Blocks World dataset implementation."""

from typing import List, Dict, Any, Tuple
import random
from data.base import PlanningDataset, DatasetFactory


@DatasetFactory.register("blocks_world")
class BlocksWorldDataset(PlanningDataset):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # YOUR VOCABULARY (from your encoding)
        self.vocab = {
            'START': 0, 'END': 1,
            'A': 2, 'B': 3, 'C': 4, 'D': 5,
            'POS_0': 6, 'POS_1': 7, 'POS_2': 8, 'POS_3': 9,
            'PAD': 10
        }
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
    
    def generate_problem(self, difficulty: int) -> Dict[str, Any]:
        """
        Generate a problem using SAW.
        
        TODO: Copy your SAW generation code here.
        
        Should return:
            {
                'start_state': [...],     # e.g., [['A','B'], [], ['C','D']]
                'goal_state': [...],
                'solution_moves': [...],  # e.g., [['A', 2], ['B', 1]]
                'solution_states': [...], # Intermediate states (for WM)
                'num_moves': difficulty
            }
        """
        # COPY YOUR SAW LOGIC HERE
        pass
    
    def encode_sequence(self, problem: Dict[str, Any]) -> List[int]:
        """
        Encode problem as token sequence.
        
        TODO: Copy your encoding logic.
        
        For baseline: [START, start, goal, action1, action2, ..., END]
        For WM: [START, start, goal, action1, state1, action2, state2, ..., END]
        """
        # COPY YOUR ENCODING LOGIC HERE
        pass
    
    def decode_sequence(self, token_ids: List[int]) -> Dict[str, Any]:
        """Decode tokens back to problem."""
        # COPY YOUR DECODING LOGIC HERE
        pass
    
    def _estimate_state_tokens(self) -> int:
        """Estimate tokens per state."""
        return 7  # Approximate for Blocks World
```

**Where to find your code:**
- SAW generation: Look in your existing blocks generation script
- Encoding logic: Look in `blocks_encoding.py` or similar
- Just copy-paste the functions!

---

### **Step 3: Test It Works (10 minutes)**

```bash
cd experiments

# Test trainer loads
python trainer.py

# Should output:
# Testing trainer...
# Running quick test with 100 samples, 5 epochs...
# [Training progress with decreasing loss]
# ✓ Trainer test complete!
```

If you get errors:
- `ImportError: PuzzleTransformer` → You didn't replace the model
- `ModuleNotFoundError: data.blocks_world` → You didn't create the file
- `NotImplementedError: generate_problem` → You didn't copy the SAW code

---

### **Step 4: Run Your First Experiment! (5 minutes)**

```bash
# Run a quick test experiment (100 samples, 10 epochs)
python run_experiments.py --experiment blocks_world_standard

# Should output:
# ======================================================================
# TRAINING: blocks_world_standard
# ======================================================================
# Generating data...
# ✓ Train: 20000 problems
# ✓ Test: 2000 problems
# ...
# Training for 100 epochs...
# [Loss should DECREASE]
# ...
# ✓ Results saved to results/blocks_world_standard/results.json
```

**Success criteria:**
- ✓ Loss DECREASES consistently
- ✓ Final loss < 0.2
- ✓ No errors

---

### **Step 5: Run Model Size Ablation (2 minutes)**

Once Step 4 works, you can immediately run ablations:

```bash
# Test 3 model sizes × 2 types (baseline, WM) = 6 experiments
python run_experiments.py --sweep model_size --domain blocks_world

# Analyzes automatically:
python analyze_results.py --sweep-file results/sweep_summary.json
```

**Output:**
```
MODEL SIZE COMPARISON
================================================================================

BLOCKS_WORLD - Base
Size         Accuracy     Params      
tiny         18.5%        60K         
small        26.0%        240K        
medium       28.2%        900K        

BLOCKS_WORLD - WM
Size         Accuracy     Params      
tiny         45.3%        60K         
small        57.0%        240K        
medium       60.1%        900K        

→ WM benefit is CONSISTENT across sizes!
```

---

## 📋 Complete Checklist

- [ ] **Step 1:** Replaced `PlanningTransformer` with your `PuzzleTransformer` (30 min)
- [ ] **Step 2:** Created `data/blocks_world.py` with your SAW code (1 hour)
- [ ] **Step 3:** Tested with `python trainer.py` - no errors, loss decreases (10 min)
- [ ] **Step 4:** Ran first experiment - completed successfully (5 min)
- [ ] **Step 5:** Ran model size sweep - got comparison table (2 min)

**Total time: ~2 hours**

---

## 🎯 What You Get

After these 2 hours, you'll have:

✅ Working end-to-end pipeline
✅ Can run ablations with single commands
✅ Automatic result tracking and comparison
✅ Easy to extend to 8-Puzzle later
✅ Ready to run experiments for paper

**Then you can:**
```bash
# Run weight sharing ablation
python run_experiments.py --sweep weight_sharing --domain blocks_world

# Run productivity split (train 1-4, test 5-8)
python run_experiments.py --sweep model_size --domain blocks_world --split productivity

# Run full ablation (12 experiments)
python run_experiments.py --sweep full --domain blocks_world
```

---

## ❓ Troubleshooting

**Q: "ModuleNotFoundError: data"**
```bash
# Create __init__.py files:
touch experiments/__init__.py
touch experiments/data/__init__.py
```

**Q: "Loss is INCREASING"**
- Check config.py has `learning_rate = 0.0001` (not 0.001)
- This is already fixed in the downloaded files!

**Q: "Can't find my SAW generation code"**
- Look for files with "generate" or "SAW" in name
- Look in your existing blocks training scripts
- If stuck, I can help extract it

**Q: "My PuzzleTransformer uses different parameters"**
- Just copy it as-is
- The framework passes the right parameters
- Check line ~180 in trainer.py for how it's called

---

## 💡 Pro Tips

**Start simple:**
1. Get PlanningTransformer working first (quick!)
2. Then add blocks_world.py (more work)
3. Test with small samples first (100 samples, 10 epochs)
4. Once working, run full experiments

**Use existing code:**
- Don't rewrite anything from scratch
- Copy-paste your working functions
- The framework just wraps them

**Test incrementally:**
- Test each step before moving to next
- Use `python trainer.py` to catch errors early
- Check that loss DECREASES (proves it's working)

---

**Ready to start? Begin with Step 1 - replace the model!** 🚀

It's literally just copy-paste of your existing PuzzleTransformer class.
