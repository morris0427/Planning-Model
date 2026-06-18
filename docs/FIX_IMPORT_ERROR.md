# QUICK FIX: "Unknown domain: blocks_world" Error

## The Error

```
ValueError: Unknown domain: blocks_world. Available: []
```

## The Problem

The `DatasetFactory` doesn't know about `blocks_world` because the module hasn't been imported yet.

## The Solution

I've created **3 new files** to fix this:

---

## ✅ Download These Files

1. **`data/blocks_world.py`** - Minimal working placeholder
2. **`data/__init__.py`** - Makes data a proper Python package  
3. **`data/base.py`** - Updated with auto-import
4. **`test_setup.py`** - Verification script

---

## 🚀 How to Fix

### **Step 1: Download the new files**

Download all 4 files above and place them:
- `data/blocks_world.py` → `experiments/data/blocks_world.py`
- `data/__init__.py` → `experiments/data/__init__.py`
- `data/base.py` → `experiments/data/base.py` (replace existing)
- `test_setup.py` → `experiments/test_setup.py`

### **Step 2: Verify setup**

```bash
cd experiments
python test_setup.py
```

**Expected output:**
```
======================================================================
FRAMEWORK SETUP VERIFICATION
======================================================================

1. Checking directory structure...
   ✓ Found: config.py
   ✓ Found: run_experiments.py
   ...

2. Testing imports...
   ✓ config.py imports successfully
   ✓ data.base imports successfully

3. Testing dataset creation...
✓ Blocks World dataset registered (PLACEHOLDER VERSION)
   ✓ DatasetFactory.create() works
   ✓ Generated 10 problems
   ...

======================================================================
VERIFICATION COMPLETE
======================================================================

✅ All core components working!
```

### **Step 3: Test trainer.py again**

```bash
python trainer.py
```

**Should now work!** You'll see:
```
Testing trainer...
✓ Blocks World dataset registered (PLACEHOLDER VERSION)
Running quick test with 100 samples, 5 epochs...
======================================================================
TRAINING: trainer_test
======================================================================
...
[Training will run and loss should DECREASE]
```

---

## 📝 About the Placeholder

The `blocks_world.py` file I created is a **minimal placeholder** that:

✅ **Works immediately** - Gets you past the import error
✅ **Generates simple random problems** - Not real SAW, but good for testing
✅ **Has basic encoding** - Simplified version
✅ **Lets you test the framework** - Training will run

⚠️ **But you should replace it** with your real SAW generation:
- Copy your SAW generation logic
- Copy your state encoding from `blocks_encoding.py`
- This placeholder is just to get you started!

---

## 🔧 Next Steps After Fix

1. **Verify it works:**
   ```bash
   python test_setup.py  # Should pass all checks
   python trainer.py     # Should train (loss decreases)
   ```

2. **Replace placeholder:**
   - Open `data/blocks_world.py`
   - Find `generate_problem()` function
   - Replace with your SAW logic
   - Find `encode_sequence()` function
   - Replace with your encoding logic

3. **Replace model:**
   - Open `trainer.py`
   - Find `class PlanningTransformer`
   - Replace with your `PuzzleTransformer`

4. **Run real experiments:**
   ```bash
   python run_experiments.py --experiment blocks_world_standard
   ```

---

## ❓ Troubleshooting

**Q: Still getting "Unknown domain" error?**
- Make sure you downloaded the **updated `data/base.py`** (with auto-import)
- Check that `data/__init__.py` exists
- Check that `data/blocks_world.py` exists

**Q: Import error: "No module named 'data'"?**
- Make sure `data/__init__.py` exists
- Run from the `experiments/` directory
- Try: `export PYTHONPATH="${PYTHONPATH}:$(pwd)"`

**Q: "NameError: name 'random' is not defined"?**
- The placeholder uses Python's random module
- It should be imported automatically
- If error persists, add `import random` at top of blocks_world.py

---

## 📦 Your Directory Should Look Like

```
experiments/
├── config.py
├── run_experiments.py
├── trainer.py
├── analyze_results.py
├── test_setup.py          # ← NEW
├── data/
│   ├── __init__.py        # ← NEW  
│   ├── base.py            # ← UPDATED
│   └── blocks_world.py    # ← NEW
└── results/               # Created when you run experiments
```

---

**Download the 4 new files above and run `test_setup.py` to verify everything works!** ✅
