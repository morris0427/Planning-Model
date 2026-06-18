# CRITICAL: Configuration Mismatch for 8-Puzzle

## 🚨 **The Problem You Caught**

**Previous runs (tiny/small/medium):**
```
Train: (10, 15) moves, 5000 samples
Test:  (10, 15) moves, 500 samples
```

**Current config.py was using:**
```
Train: (8, 18) moves, 2000 samples  ← WRONG!
Test:  (8, 18) moves, 200 samples   ← WRONG!
```

**Impact:**
- ❌ Large model results NOT comparable to tiny/small/medium
- ❌ Different difficulty (8-18 is wider range)
- ❌ Different data amount (2000 vs 5000 samples)
- ❌ All comparisons are invalid!

---

## ✅ **I Fixed config.py**

**Updated eight_puzzle_standard() to match previous runs:**
```python
train_difficulty_range=(10, 15),  # Now matches!
test_difficulty_range=(10, 15),   # Now matches!
num_train_samples=5000,           # Now matches!
num_test_samples=500              # Now matches!
```

---

## 🔄 **What You Need to Do**

### **If you already started running large models:**

**STOP the current run** (Ctrl+C) because it's using wrong configuration!

Then:

```bash
cd experiments

# 1. Download the fixed config.py (above)

# 2. Clear 8-Puzzle cache (has wrong difficulty range)
rm cached_data/eight_puzzle_*

# 3. Regenerate with CORRECT (10, 15) range
python3 << 'EOF'
from data import DatasetFactory
import json

print("Regenerating 8-Puzzle data with (10, 15) range...")

for use_wm in [False, True]:
    suffix = "wm" if use_wm else "baseline"
    
    # Generate with (10, 15) to match previous runs
    train_gen = DatasetFactory.create(
        domain="eight_puzzle",
        difficulty_range=(10, 15),  # Match previous!
        num_samples=5000,           # Match previous!
        use_world_model=use_wm
    )
    train_data = train_gen.generate_dataset()
    
    test_gen = DatasetFactory.create(
        domain="eight_puzzle",
        difficulty_range=(10, 15),  # Match previous!
        num_samples=500,            # Match previous!
        use_world_model=use_wm
    )
    test_data = test_gen.generate_dataset()
    
    # Save
    with open(f'cached_data/eight_puzzle_train_{suffix}.json', 'w') as f:
        json.dump(train_data, f)
    
    with open(f'cached_data/eight_puzzle_test_{suffix}.json', 'w') as f:
        json.dump(test_data, f)
    
    print(f"  ✓ Generated {suffix} data")

print("\n✓ Data regenerated with correct (10, 15) range!")
EOF

# 4. Now run large models with correct data
python3 run_large_models.py --domain eight_puzzle
```

---

### **If you haven't started yet:**

**Great! Just follow these steps:**

```bash
cd experiments

# 1. Download fixed config.py

# 2. Make sure cache has correct data
rm cached_data/eight_puzzle_*

# 3. Regenerate data (use script above)

# 4. Run large models
python3 run_large_models.py --domain eight_puzzle
```

---

## ✅ **Verification**

**After regenerating data, verify it's correct:**

```bash
python3 << 'EOF'
import json

with open('cached_data/eight_puzzle_train_baseline.json') as f:
    train = json.load(f)

with open('cached_data/eight_puzzle_test_baseline.json') as f:
    test = json.load(f)

print("8-Puzzle Data Verification:")
print(f"  Train samples: {len(train)}")
print(f"  Train difficulty: {min(p['num_moves'] for p in train)}-{max(p['num_moves'] for p in train)}")
print(f"  Test samples: {len(test)}")
print(f"  Test difficulty: {min(p['num_moves'] for p in test)}-{max(p['num_moves'] for p in test)}")

# Should show:
# Train samples: 5000
# Train difficulty: 10-15
# Test samples: 500
# Test difficulty: 10-15

expected = {
    'train_samples': 5000,
    'train_min': 10,
    'train_max': 15,
    'test_samples': 500,
    'test_min': 10,
    'test_max': 15
}

actual = {
    'train_samples': len(train),
    'train_min': min(p['num_moves'] for p in train),
    'train_max': max(p['num_moves'] for p in train),
    'test_samples': len(test),
    'test_min': min(p['num_moves'] for p in test),
    'test_max': max(p['num_moves'] for p in test)
}

if actual == expected:
    print("\n✓ CORRECT! Data matches previous runs.")
else:
    print(f"\n✗ MISMATCH!")
    print(f"  Expected: {expected}")
    print(f"  Actual: {actual}")
EOF
```

---

## 📊 **Why This Matters**

**Comparing (10,15) vs (8,18) results:**

| Config | Difficulty | Learning | Result Quality |
|--------|----------|----------|----------------|
| (10, 15) | Medium, narrow | 5000 samples | Better |
| (8, 18) | Mixed (easy+hard) | 2000 samples | Worse |

**If you compared these:**
- Large model would look worse (harder data, less training)
- Comparisons would be meaningless
- Paper conclusions would be wrong!

**Good catch preventing a major mistake!**

---

## ✅ **Summary**

1. ✓ config.py is now fixed (10, 15) to match previous runs
2. ⚠️ Must regenerate 8-Puzzle cache data with correct range
3. ⚠️ Must re-run large models with correct data
4. ✓ Blocks World config is fine (already correct)

**After fixing, your large model results will be comparable to tiny/small/medium!** 🎯
