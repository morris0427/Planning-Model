# INFINITE LOOP BUG FIX: WM Testing Hangs

## 🐛 The Problem

**WM testing hangs forever at:**
```
Evaluating solve rate on 100 test problems...
```

After 12+ hours, no progress.

---

## 🔍 Root Cause

**Infinite loop in WM generation:**

```python
# OLD (BROKEN):
while moves_generated < max_length:
    next_token = model.predict()
    
    if next_token not in move_tokens:  # Not a move (e.g., state token 0-9)
        continue  # ← INFINITE LOOP!
        # moves_generated never increments!
        # Loop never exits!
```

**What happens:**
1. Model predicts token `5` (a state token, not a move)
2. Code skips it with `continue`
3. Back to step 1
4. **Infinite loop** - `moves_generated` never changes!

**Why this happens:**
- WM was trained on `[move][state][move][state]...`
- During generation, we only want moves
- But model sometimes predicts state tokens (0-9)
- Each non-move token causes a wasted iteration
- If model keeps predicting non-moves: **infinite loop**

---

## ✅ The Fix

**Add a step counter:**

```python
# NEW (FIXED):
steps = 0
max_steps = max_length * 20  # Safety limit

while moves_generated < max_length and steps < max_steps:
    steps += 1  # Always increment!
    
    next_token = model.predict()
    
    if next_token not in move_tokens:
        continue  # ← Now safe - steps will eventually hit max_steps
```

**Now:**
- Every iteration increments `steps`
- If model predicts 1000 non-move tokens: loop exits after `max_steps`
- No infinite loop possible!

---

## 🚀 What To Do

### **1. Download fixed trainer.py above**

### **2. Kill the hung process:**
```bash
# Press Ctrl+C on the running experiment
```

### **3. Restart:**
```bash
python3 run_experiments.py --sweep full --domain eight_puzzle
```

**Should now complete!**

---

## ⏱️ **Expected Timing**

**With fix:**
- Each test problem: <1 second
- 100 test problems: ~1-2 minutes
- Full WM experiment: ~30-40 minutes (training + testing)

**Without fix:**
- Each test problem: FOREVER (infinite loop)
- Never completes

---

## 🔍 **Optional: Diagnose Before Re-running**

If you want to verify the fix works:

```bash
python3 diagnose_wm_hang.py
```

**Expected output:**
```
✓ Generation completed in 0.5s
✓ PASS: Generation is fast enough
```

---

## 📊 **Why This Happened**

**WM models are trained to predict both:**
- Move tokens (10-13): The actions
- State tokens (0-9): The puzzle states

**During test-time generation:**
- We ONLY want moves
- We provide states via oracle
- But model sometimes predicts state tokens anyway (confusion)
- Old code: infinite loop when this happens
- New code: skips them, but with safety limit

---

## ⚠️ **Important Notes**

1. **This doesn't affect training** - Only test-time generation
2. **Your trained models are fine** - Just need to re-evaluate them
3. **Fix is backward compatible** - Works for both WM and baseline
4. **Baseline was unaffected** - Only WM had this issue

---

## 📋 **Checklist**

- [ ] Download fixed `trainer.py`
- [ ] Kill hung experiment (Ctrl+C)
- [ ] Restart experiments
- [ ] Verify WM testing completes in ~2 minutes (not hours!)
- [ ] Check solve rate results are reasonable

---

## 🎯 **Expected Results After Fix**

```
Training: 100 epochs (~30 min)
Evaluating on test set... (~30 sec)
Evaluating solve rate... (~2 min for 100 problems)
✓ Complete!
```

**Total per WM experiment:** ~30-40 minutes (not 12+ hours!)

---

**Download the fixed trainer.py and restart - should complete quickly now!**
