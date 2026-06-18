# Running Large Models Only

## 📊 Model Sizes Overview

| Size | Params | d_model | n_heads | d_ff | n_layers |
|------|--------|---------|---------|------|----------|
| Tiny | ~60K | 32 | 2 | 128 | 2 |
| Small | ~240K | 64 | 4 | 256 | 2 |
| Medium | ~900K | 128 | 8 | 512 | 4 |
| **Large** | **~2M** | **256** | **8** | **1024** | **4** |

---

## 🚀 How to Run Large Models Only

### **Option 1: Use Custom Script (Recommended)**

```bash
# Run all 4 large configs for 8-Puzzle
python3 run_large_models.py --domain eight_puzzle

# Or for Blocks World
python3 run_large_models.py --domain blocks_world

# Or both domains
python3 run_large_models.py --domain both
```

**Configs tested:**
- `large_base_std` (baseline, standard)
- `large_base_shared` (baseline, weight sharing)
- `large_wm_std` (world model, standard)
- `large_wm_shared` (world model, weight sharing)

---

### **Option 2: Manual Individual Runs**

```bash
# 8-Puzzle - Large Baseline
python3 run_experiments.py \
    --model large_base \
    --data eight_puzzle_standard \
    --name eight_puzzle_large_base_std

# 8-Puzzle - Large WM
python3 run_experiments.py \
    --model large_wm \
    --data eight_puzzle_standard \
    --name eight_puzzle_large_wm_std

# Repeat for _shared variants...
```

---

## ⏱️ **Time Estimates**

**Per large model experiment:**
- Data loading: ~1 min (uses cached data)
- Training: ~45-90 minutes (larger model = slower)
- Testing: ~5 minutes
- **Total: ~1 hour per experiment**

**For all 4 large configs (one domain):**
- Total time: ~4 hours

**For both domains (8 experiments):**
- Total time: ~8 hours

---

## 💾 **Data Caching**

**Good news:** Your existing cached data will be reused!

- ✅ No regeneration needed
- ✅ Just loads from `cached_data/`
- ✅ Saves hours of time

---

## 📈 **Results Integration**

**Important:** Results APPEND to existing `sweep_summary.json`

**Before running large models:**
```json
{
  "experiments": [
    { "name": "eight_puzzle_tiny_base_std", ... },
    { "name": "eight_puzzle_small_base_std", ... },
    { "name": "eight_puzzle_medium_base_std", ... },
    // 12 experiments total
  ]
}
```

**After running large models:**
```json
{
  "experiments": [
    { "name": "eight_puzzle_tiny_base_std", ... },
    { "name": "eight_puzzle_small_base_std", ... },
    { "name": "eight_puzzle_medium_base_std", ... },
    { "name": "eight_puzzle_large_base_std", ... },    ← NEW
    { "name": "eight_puzzle_large_base_shared", ... }, ← NEW
    { "name": "eight_puzzle_large_wm_std", ... },      ← NEW
    { "name": "eight_puzzle_large_wm_shared", ... },   ← NEW
    // 16 experiments total
  ]
}
```

**No overwrites!** All previous results preserved.

---

## 🔍 **Analyzing Results**

After running large models:

```bash
# View all results (tiny + small + medium + large)
python3 analyze_results.py --sweep-file results/sweep_summary.json
```

**You'll see a complete size comparison:**
```
EIGHT_PUZZLE - Base
--------------------------------------------------
Size         Solve Rate      Params      
--------------------------------------------------
tiny         27.0%           60K         
small        78.0%           240K        
medium       93.0%           900K        
large        97.0%           2M          ← NEW!

EIGHT_PUZZLE - WM
--------------------------------------------------
Size         Solve Rate      Params      
--------------------------------------------------
tiny         30.0%           60K         
small        85.0%           240K        
medium       95.0%           900K        
large        98.0%           2M          ← NEW!
```

---

## 📊 **Expected Large Model Performance**

**Hypothesis:** Larger models should perform better

**Expected pattern:**
```
Size      Baseline    WM         Gap
Tiny      27%         30%        +11%
Small     78%         85%        +9%
Medium    93%         95%        +2%
Large     97%         98%        +1%   ← Diminishing returns?
```

**Why test large?**
1. Check if WM benefit persists at scale
2. See if models saturate on this task
3. Understand compute vs. performance tradeoff

---

## 💡 **What You Might Find**

### **Scenario 1: Performance Saturates**
```
Medium: 93% → 95%  (+2%)
Large:  97% → 98%  (+1%)
```
**Interpretation:** Task is too easy for large models; both approaches near ceiling

### **Scenario 2: WM Advantage Persists**
```
Medium: 93% → 95%  (+2%)
Large:  95% → 99%  (+4%)
```
**Interpretation:** WM benefit grows with scale

### **Scenario 3: WM Advantage Disappears**
```
Medium: 93% → 95%  (+2%)
Large:  97% → 97%  (0%)
```
**Interpretation:** Baseline catches up with enough capacity

---

## 🎯 **For Your Paper**

**If large models show interesting patterns, you can report:**

> "We tested model sizes from 60K to 2M parameters. The world model advantage was most pronounced at medium scale (900K parameters, +X% improvement), with [increasing/decreasing/stable] benefits at 2M parameters (+Y% improvement). This suggests that [interpretation based on results]."

---

## 📋 **Quick Start Checklist**

- [ ] Ensure cached data exists (from previous runs)
- [ ] Run `python3 run_large_models.py --domain eight_puzzle`
- [ ] Wait ~4 hours (grab coffee, read papers, etc.)
- [ ] Check results with `analyze_results.py`
- [ ] Compare all sizes (tiny → small → medium → large)
- [ ] Update paper with findings

---

## ⚠️ **Important Notes**

1. **No conflicts:** Results append, no overwrites
2. **Uses cache:** No data regeneration (fast!)
3. **Takes time:** ~1 hour per experiment
4. **Memory:** Large models use ~4-6GB RAM during training
5. **Convergence:** May need more epochs for large models

---

## 🔧 **If Training is Slow**

Large models take longer. If you want to speed up:

**Option A: Reduce epochs**
```python
# In config.py, for large models:
training_epochs=50  # Instead of 100
```

**Option B: Increase batch size** (if you have GPU memory)
```python
batch_size=64  # Instead of 32
```

**Option C: Use early stopping**
```python
# Stop if no improvement for 10 epochs
patience=10
```

---

**TL;DR:** Just run `python3 run_large_models.py --domain eight_puzzle` and wait ~4 hours. Results will append to existing sweep!
