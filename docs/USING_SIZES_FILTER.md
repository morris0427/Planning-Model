# Using --sizes Filter in run_experiments.py

## ✅ **NEW: Single Script for All Experiments**

You now have a unified way to run experiments with optional size filtering!

---

## 🎯 **Basic Usage**

### **Run ONLY large models:**

```bash
# Model size sweep - large only
python3 run_experiments.py --sweep model_size --domain eight_puzzle --sizes large

# Full ablation - large only
python3 run_experiments.py --sweep full --domain blocks_world --sizes large
```

**Expected experiments:**
```
model_size + large:
  - eight_puzzle_large_base
  - eight_puzzle_large_wm

full + large:
  - blocks_world_large_base_std
  - blocks_world_large_base_shared
  - blocks_world_large_wm_std
  - blocks_world_large_wm_shared
```

---

### **Run multiple sizes:**

```bash
# Small and medium only (skip tiny and large)
python3 run_experiments.py --sweep model_size --domain blocks_world --sizes small medium

# All except large
python3 run_experiments.py --sweep full --domain eight_puzzle --sizes tiny small medium
```

---

### **Run all sizes (default behavior):**

```bash
# No --sizes argument = all sizes
python3 run_experiments.py --sweep model_size --domain blocks_world

# Runs: tiny, small, medium, large (all 8 experiments)
```

---

## 📊 **Comparison: Old vs New Approach**

### **OLD (Two Scripts):**

```bash
# Step 1: Run tiny/small/medium
python3 run_experiments.py --sweep model_size --domain eight_puzzle
# Creates: sweep_summary.json with 6 experiments

# Step 2: Run large separately
python3 run_large_models.py --domain eight_puzzle
# Appends: 2 more experiments
# Total: 8 experiments

# Problems:
# - Two different scripts
# - Results appended (order matters)
# - Easy to forget which step you're on
# - Different code paths might have bugs
```

---

### **NEW (Single Script with Filter):**

```bash
# One step: Run all sizes
python3 run_experiments.py --sweep model_size --domain eight_puzzle

# OR run only large if you already have tiny/small/medium
python3 run_experiments.py --sweep model_size --domain eight_puzzle --sizes large

# Benefits:
# ✅ Single script
# ✅ Consistent code path
# ✅ Clear what's being run
# ✅ No appending confusion
```

---

## 🔧 **How It Works**

**The filter checks model names:**

```python
# If you specify --sizes large
# It keeps experiments where model.name contains "large"

Examples:
  "large_Base"      ✓ matches "large"
  "large_WM"        ✓ matches "large"
  "small_Base"      ✗ doesn't match "large"
  "medium_WM"       ✗ doesn't match "large"
```

**Case-insensitive matching:**
```bash
--sizes Large    # works
--sizes large    # works
--sizes LARGE    # works
```

---

## 🎯 **Your Use Case: Large Models Only**

**You want to run large models without re-running tiny/small/medium.**

### **Option 1: Start fresh (everything from one script)**

```bash
cd experiments

# Clear old results
rm results/sweep_summary.json

# Run ALL sizes from single script
python3 run_experiments.py --sweep model_size --domain eight_puzzle

# Time: ~12-16 hours (includes all sizes)
# Benefit: Everything from single consistent code path
```

---

### **Option 2: Add large to existing results (faster)**

```bash
cd experiments

# You already have tiny/small/medium results

# Run ONLY large models
python3 run_experiments.py --sweep model_size --domain eight_puzzle --sizes large

# Time: ~4-6 hours (large only)
# Benefit: Saves time, builds on existing work
```

**Note:** This still appends to sweep_summary.json, but at least it's the same script!

---

### **Option 3: Separate files for clarity (recommended)**

```bash
cd experiments

# Run large models to separate file
python3 run_experiments.py \
  --sweep model_size \
  --domain eight_puzzle \
  --sizes large \
  --results-dir results_large

# Creates: results_large/sweep_summary.json
# Original results/sweep_summary.json unchanged

# Then merge manually if needed:
python3 << 'EOF'
import json

# Load both
with open('results/sweep_summary.json') as f:
    small_results = json.load(f)

with open('results_large/sweep_summary.json') as f:
    large_results = json.load(f)

# Merge
combined = small_results + large_results

# Save
with open('results/sweep_summary_combined.json', 'w') as f:
    json.dump(combined, f, indent=2)

print(f"Combined: {len(small_results)} + {len(large_results)} = {len(combined)} experiments")
EOF
```

---

## 📋 **Complete Workflow Example**

**Goal:** Run large models for both domains, verify consistency

```bash
cd experiments

# 1. Verify cache is correct
python3 << 'EOF'
import json

for domain in ['blocks_world', 'eight_puzzle']:
    try:
        with open(f'cached_data/{domain}_train_baseline.json') as f:
            train = json.load(f)
        moves = [p['num_moves'] for p in train]
        print(f"{domain}: {min(moves)}-{max(moves)} moves, {len(train)} samples")
    except FileNotFoundError:
        print(f"{domain}: NO CACHE - must generate!")
EOF

# Expected:
# blocks_world: 1-6 moves, 20000 samples
# eight_puzzle: 10-15 moves, 5000 samples

# 2. If cache is wrong, regenerate (see previous guides)

# 3. Run large models for both domains
python3 run_experiments.py --sweep model_size --domain blocks_world --sizes large
python3 run_experiments.py --sweep model_size --domain eight_puzzle --sizes large

# 4. Check results
python3 << 'EOF'
import json

with open('results/sweep_summary.json') as f:
    results = json.load(f)

print(f"Total experiments: {len(results)}")
print("\nLarge model results:")
for exp in results:
    if 'large' in exp['experiment_name']:
        print(f"  {exp['experiment_name']}: {exp['solve_rate']:.1%}")
EOF
```

---

## ⚠️ **Important Notes**

### **1. Appending behavior:**
```bash
# First run creates file
python3 run_experiments.py --sweep model_size --domain blocks_world --sizes large
# Creates: results/sweep_summary.json with 2 experiments

# Second run APPENDS
python3 run_experiments.py --sweep model_size --domain eight_puzzle --sizes large
# Appends: 2 more experiments
# Total: 4 experiments in sweep_summary.json
```

**If you want separate files, use `--results-dir`:**
```bash
python3 run_experiments.py --sweep model_size --domain blocks_world --sizes large --results-dir results_blocks_large
python3 run_experiments.py --sweep model_size --domain eight_puzzle --sizes large --results-dir results_eight_large
```

---

### **2. Size matching is flexible:**
```bash
# These all work:
--sizes large
--sizes Large
--sizes small medium large
--sizes tiny  # Even though it's not "large"
```

---

### **3. Works with any sweep type:**
```bash
# Model size sweep - large only
--sweep model_size --sizes large

# Full ablation - large only
--sweep full --sizes large

# Weight sharing - medium only
--sweep weight_sharing --sizes medium
```

---

## ✅ **Benefits of This Approach**

1. **Single script** - No need for run_large_models.py
2. **Consistent code path** - Same trainer, same evaluation, same everything
3. **Flexible** - Run any combination of sizes
4. **Clear intent** - `--sizes large` is explicit
5. **Easy to verify** - One script to check, not two

---

## 🎯 **Recommended: Fresh Run for Peace of Mind**

**If you want to be 100% sure results are consistent:**

```bash
cd experiments

# 1. Clear everything
rm -rf results/*
rm cached_data/*

# 2. Regenerate cache with correct configs
python3 << 'EOF'
from data import DatasetFactory
import json

# Blocks World (1-6 moves)
for wm in [False, True]:
    suffix = "wm" if wm else "baseline"
    train = DatasetFactory.create("blocks_world", (1,6), 20000, wm).generate_dataset()
    test = DatasetFactory.create("blocks_world", (1,6), 2000, wm).generate_dataset()
    with open(f'cached_data/blocks_world_train_{suffix}.json', 'w') as f:
        json.dump(train, f)
    with open(f'cached_data/blocks_world_test_{suffix}.json', 'w') as f:
        json.dump(test, f)

# 8-Puzzle (10-15 moves)
for wm in [False, True]:
    suffix = "wm" if wm else "baseline"
    train = DatasetFactory.create("eight_puzzle", (10,15), 5000, wm).generate_dataset()
    test = DatasetFactory.create("eight_puzzle", (10,15), 500, wm).generate_dataset()
    with open(f'cached_data/eight_puzzle_train_{suffix}.json', 'w') as f:
        json.dump(train, f)
    with open(f'cached_data/eight_puzzle_test_{suffix}.json', 'w') as f:
        json.dump(test, f)

print("✓ Cache regenerated")
EOF

# 3. Run ALL sizes from single consistent script
python3 run_experiments.py --sweep model_size --domain blocks_world
python3 run_experiments.py --sweep model_size --domain eight_puzzle

# Time: ~16-20 hours for everything
# Benefit: 100% confidence in consistency
```

**OR just run large if you trust your existing small/medium results:**

```bash
# Just add large to existing
python3 run_experiments.py --sweep model_size --domain blocks_world --sizes large
python3 run_experiments.py --sweep model_size --domain eight_puzzle --sizes large

# Time: ~4-6 hours
# Benefit: Faster, builds on existing work
```

---

**Your call: fresh run for confidence, or incremental for speed!** 🎯
