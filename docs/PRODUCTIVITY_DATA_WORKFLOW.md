# Productivity Data Generation Workflow

## 🐛 The Problem

**Current cache doesn't distinguish splits:**
```
cached_data/
├── eight_puzzle_train_baseline.json  ← Standard (8-18)? Productivity (10-12)?
└── eight_puzzle_train_wm.json        ← Which split?
```

**Result:** Running productivity after standard loads WRONG data!

---

## ✅ Solution: Split-Specific Cache

### **Option 1: Manual Generation (Recommended)**

**Step 1: Generate productivity data**
```bash
python3 generate_productivity_data.py --domain eight_puzzle
```

**This creates:**
```
cached_data/
├── eight_puzzle_train_baseline.json              ← Standard (keep this!)
├── eight_puzzle_train_wm.json                    ← Standard (keep this!)
├── eight_puzzle_train_baseline_productivity.json ← New!
├── eight_puzzle_test_baseline_productivity.json  ← New!
├── eight_puzzle_train_wm_productivity.json       ← New!
└── eight_puzzle_test_wm_productivity.json        ← New!
```

**Step 2: Temporarily rename for productivity runs**
```bash
cd cached_data

# Backup standard data
mv eight_puzzle_train_baseline.json eight_puzzle_train_baseline_STANDARD.json
mv eight_puzzle_test_baseline.json eight_puzzle_test_baseline_STANDARD.json
mv eight_puzzle_train_wm.json eight_puzzle_train_wm_STANDARD.json
mv eight_puzzle_test_wm.json eight_puzzle_test_wm_STANDARD.json

# Use productivity data
cp eight_puzzle_train_baseline_productivity.json eight_puzzle_train_baseline.json
cp eight_puzzle_test_baseline_productivity.json eight_puzzle_test_baseline.json
cp eight_puzzle_train_wm_productivity.json eight_puzzle_train_wm.json
cp eight_puzzle_test_wm_productivity.json eight_puzzle_test_wm.json
```

**Step 3: Run productivity experiments**
```bash
cd ..
python3 run_productivity_sweep.py --domain eight_puzzle
```

**Step 4: Save results and restore standard data**
```bash
# Save productivity results
mv results/sweep_summary.json results/sweep_summary_8puzzle_productivity.json

# Restore standard data
cd cached_data
mv eight_puzzle_train_baseline_STANDARD.json eight_puzzle_train_baseline.json
mv eight_puzzle_test_baseline_STANDARD.json eight_puzzle_test_baseline.json
mv eight_puzzle_train_wm_STANDARD.json eight_puzzle_train_wm.json
mv eight_puzzle_test_wm_STANDARD.json eight_puzzle_test_wm.json
```

---

### **Option 2: Clear Cache Each Time (Simple but Slow)**

```bash
# Run standard experiments
python3 run_experiments.py --sweep full --domain eight_puzzle
mv results/sweep_summary.json results/sweep_summary_8puzzle_standard.json

# Clear cache
rm cached_data/eight_puzzle_*

# Run productivity experiments (will regenerate)
python3 run_productivity_sweep.py --domain eight_puzzle
mv results/sweep_summary.json results/sweep_summary_8puzzle_productivity.json
```

**Time cost:** ~30 min × 2 = 1 hour of regeneration

---

### **Option 3: Fix Caching in trainer.py (Best Long-term)**

Update `save_generated_data()` and `load_cached_data()` in trainer.py to include split type:

```python
def save_generated_data(train_problems, test_problems, config, use_wm):
    """Save with split-specific names."""
    cache_dir = Path("cached_data")
    cache_dir.mkdir(exist_ok=True)
    
    domain = config.data.domain
    split = config.data.split_type.value  # ← Add this!
    wm_suffix = "_wm" if use_wm else "_baseline"
    
    # Include split in filename
    train_file = cache_dir / f"{domain}_train{wm_suffix}_{split}.json"
    test_file = cache_dir / f"{domain}_test{wm_suffix}_{split}.json"
    
    # ... rest of save code
```

**Then filenames become:**
```
eight_puzzle_train_baseline_in_distribution.json
eight_puzzle_train_baseline_productivity.json
eight_puzzle_train_wm_in_distribution.json
eight_puzzle_train_wm_productivity.json
```

**Benefit:** No manual file juggling needed!

---

## 📋 Recommended Workflow

**For your current situation:**

1. **Use Option 1** (manual generation with file swapping)
2. **Generate productivity data once:** `python3 generate_productivity_data.py --domain eight_puzzle`
3. **Swap files** before/after productivity runs
4. **Keep both sets of results** for comparison

**Total time:**
- Generate productivity data: ~30-60 minutes (once)
- Run productivity experiments: ~2-3 hours
- **Total:** ~3-4 hours

---

## 🎯 What You'll Have

After following Option 1:

```
cached_data/
├── eight_puzzle_train_baseline.json              ← Standard (8-18)
├── eight_puzzle_train_wm.json                    ← Standard (8-18)
├── eight_puzzle_train_baseline_productivity.json ← Productivity (10-12, 13-18)
└── eight_puzzle_train_wm_productivity.json       ← Productivity (10-12, 13-18)

results/
├── sweep_summary_8puzzle_standard.json           ← In-distribution
└── sweep_summary_8puzzle_productivity.json       ← Length generalization
```

**Then compare:**
```bash
python3 compare_generalization.py \
    results/sweep_summary_8puzzle_standard.json \
    results/sweep_summary_8puzzle_productivity.json
```

---

## ⚙️ Generator Support

**Yes, the generator fully supports this!**

The `EightPuzzleDataset` takes `difficulty_range` as a parameter:
```python
DatasetFactory.create(
    domain="eight_puzzle",
    difficulty_range=(10, 12),  # ← Any range works!
    num_samples=5000,
    use_world_model=False,
    seed=42
)
```

**So you can generate any difficulty range:**
- Standard: (8, 18)
- Productivity: (10, 12) for train, (13, 18) for test
- Custom: (5, 25), (15, 20), whatever you want!

---

## 🚀 Quick Start

**Just run this:**
```bash
# Generate productivity data (takes 30-60 min)
python3 generate_productivity_data.py --domain eight_puzzle

# Follow Option 1 file swapping instructions above
```

**The generator works perfectly for this!** The only issue is the cache naming, which the manual workflow solves.
