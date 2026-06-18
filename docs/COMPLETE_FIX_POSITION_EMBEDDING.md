# COMPLETE FIX: Position Embedding Overflow in Productivity

## 🐛 The Problem

**Position embedding crash during Blocks World productivity:**
```
IndexError: index out of range in self
  at pos_emb = self.pos_encoder(positions)
```

**Root cause had TWO parts:**

### **Part 1:** Model created with wrong max_seq_length
- Used TRAIN max (1-4 moves → 34 tokens)
- But TEST has longer sequences (5-8 moves → ~50 tokens)

### **Part 2:** Generation can exceed ANY fixed max_seq_length
- Even if model has max_seq_length=50
- Generation loop with max_length=100 can create 120+ token sequences
- Problem context (20) + 100 generated moves = 120 tokens > 50

---

## ✅ Complete Fix (TWO changes required)

### **Fix #1: Use max of train AND test seq lengths**

**In `trainer.py` line ~699:**

```python
# OLD (broken):
max_seq_length = train_dataset_generator.get_max_sequence_length()

# NEW (fixed):
train_max_seq = train_dataset_generator.get_max_sequence_length()
test_max_seq = test_dataset_generator.get_max_sequence_length()
max_seq_length = max(train_max_seq, test_max_seq)
```

**This ensures model has capacity for BOTH train and test data.**

---

### **Fix #2: Stop generation before exceeding model capacity**

**In `generate_solution()` function:**

**Added at line ~183:**
```python
# Get model's maximum sequence length
model_max_seq_length = model.max_seq_length
```

**Added at line ~245 (WM generation):**
```python
# Safety: stop if sequence would exceed model capacity
if len(generated) >= model_max_seq_length - 10:  # Leave room
    break
```

**Added at line ~289 (Baseline generation):**
```python
# Safety: stop if sequence would exceed model capacity  
if len(generated) >= model_max_seq_length - 1:
    break
```

**This prevents generation from creating sequences longer than model can handle.**

---

## 🚀 How to Apply

```bash
cd experiments

# Download the COMPLETE fixed trainer.py (above)

# Clear partial results
rm -rf results/blocks_world_*productivity*

# Restart productivity sweep
python3 run_productivity_sweep.py --domain blocks_world
```

**Now it will:**
1. ✓ Create model with max_seq_length = max(train_max, test_max)
2. ✓ Stop generation before exceeding model capacity
3. ✓ Complete evaluation without crashes

---

## 📊 Expected Output

```
✓ Max sequence length: 50
  Note: Train max_seq=34, Test max_seq=50
        Using max=50 for model capacity

Training for 100 epochs...
  [completes successfully]

Evaluating solve rate...
  Evaluating 100/100 problems...
  ✓ Solve rate: XX%  [actual results!]
```

---

## 🎯 Why Both Fixes Are Needed

### **Fix #1 alone** (use max of train/test):
- ❌ Still fails if generation exceeds test_max
- Example: test_max=50 but generation creates 120 tokens

### **Fix #2 alone** (stop at model capacity):
- ❌ Still fails if model created with train_max only
- Example: model has max_seq_length=34, test needs 50

### **Both together:**
- ✅ Model has enough capacity (max of train/test)
- ✅ Generation never exceeds model capacity
- ✅ Works for all productivity splits!

---

## ⏱️ Timeline

- Clear results: 1 second
- Training: ~30-40 minutes (6 configs × 5-7 min each)
- Evaluation: ~10 minutes
- **Total: ~45-50 minutes**

---

## ✅ This Fix Also Prevents Future Issues

**Will work for:**
- ✅ Any productivity split (asymmetric train/test ranges)
- ✅ Long generation attempts (model stops before overflow)
- ✅ Both domains (Blocks World and 8-Puzzle)
- ✅ All model sizes (tiny to large)

---

**Download the complete fixed `trainer.py` and restart - should work now!** 🎯
