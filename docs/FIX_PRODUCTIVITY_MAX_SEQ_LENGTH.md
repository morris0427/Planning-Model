# FIX: Position Embedding Error in Productivity Sweeps

## 🐛 The Problem

**Error during Blocks World productivity evaluation:**
```
IndexError: index out of range in self
  at pos_emb = self.pos_encoder(positions)
```

**Root cause:**
- Model created with `max_seq_length=34` (from TRAIN data: 1-4 moves)
- But TEST data has 5-8 moves → sequences up to ~50 tokens
- During generation, sequences exceed 34 → crash

---

## ✅ The Fix

**Updated `trainer.py` line 699:**

**Before:**
```python
max_seq_length = train_dataset_generator.get_max_sequence_length()
```

**After:**
```python
# Use MAXIMUM of train and test seq lengths (for productivity splits)
train_max_seq = train_dataset_generator.get_max_sequence_length()
test_max_seq = test_dataset_generator.get_max_sequence_length()
max_seq_length = max(train_max_seq, test_max_seq)
```

**Now the model has enough positional embeddings for BOTH train and test sequences!**

---

## 🚀 How to Apply

### **Option 1: Just download updated trainer.py (above)**

```bash
cd experiments
# Replace trainer.py with the fixed version
# Then restart your productivity sweep
```

### **Option 2: If you want to salvage the trained model**

The model trained successfully but can't evaluate. You could:
1. Save the trained weights
2. Recreate model with larger max_seq_length
3. Load weights back
4. Evaluate

**But it's easier to just retrain** (will use correct max_seq_length from start)

---

## 🔄 Restart Blocks World Productivity

```bash
cd experiments

# Clear any partial results
rm -rf results/blocks_world_*productivity*

# Re-run with fixed trainer.py
python3 run_productivity_sweep.py --domain blocks_world
```

**This time it will:**
- Detect: train_max=34, test_max=50
- Use: max_seq_length=50
- ✓ No position embedding errors!

---

## ⏱️ Time Estimate

- Training already completed: ~27 minutes (you did this already!)
- Just need to rerun since it crashed during evaluation
- Total: ~30 minutes to restart and complete

---

## 📊 Expected Output After Fix

```
✓ Max sequence length: 50  ← Fixed! (was 34)
  Note: Train max_seq=34, Test max_seq=50
        Using max=50 for model capacity

Training for 100 epochs...
  [training completes successfully]

Evaluating solve rate...
  ✓ 100/100 problems evaluated  ← No crash!
  
Solve rate: XX%  ← Actual results!
```

---

## 💡 Why This Happened

**Productivity splits have asymmetric difficulty:**
- Train: Easy problems (1-4 moves) → short sequences
- Test: Hard problems (5-8 moves) → long sequences

**The old code only checked TRAIN max → model too small for TEST data.**

**The fix checks BOTH and uses the larger → model handles all data.**

---

## ✅ After Applying Fix

This fix ensures productivity sweeps work for:
- ✅ Blocks World productivity (1-4 → 5-8)
- ✅ 8-Puzzle productivity (10-12 → 13-18)
- ✅ Any future productivity splits with asymmetric ranges

---

**Download updated `trainer.py` above and restart the Blocks World productivity sweep!**
