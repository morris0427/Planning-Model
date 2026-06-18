# Planning Experiment Framework

A unified, factored codebase for running systematic experiments across planning domains (Blocks World, 8-Puzzle, etc.).

## 🎯 Design Goals

1. **Shared Infrastructure**: Single codebase for all domains
2. **Easy Configuration**: Presets + command-line overrides
3. **Systematic Sweeps**: Ablation studies with one command
4. **Reproducibility**: All configs saved automatically

---

## 🚀 Quick Start

### Run Single Experiment
```bash
python run_experiment.py --preset in_distribution --domain blocks_world --use_wm
```

### Run Systematic Sweep
```bash
python sweep_experiments.py --sweep model_size --domain blocks_world
```

### Analyze Results
```bash
python analyze_results.py --sweep_dir sweep_results
```

---

## ⚙️ Available Presets

**Experiment Presets:**
- `in_distribution` - Random split (1-6 moves)
- `productivity` - Train 1-4, test 5-8 moves
- `length_extrapolation` - Train 1-6, test 7-10

**Model Presets:**
- `tiny` - 32d, 2 layers
- `small` - 64d, 2 layers (current baseline)
- `medium` - 128d, 4 layers
- `large` - 256d, 6 layers

---

## 📊 Example: Model Size Ablation

```bash
# Runs 6 experiments: tiny/small/medium × baseline/WM
python sweep_experiments.py --sweep model_size --domain blocks_world
```

---

See full documentation in comments within each file.
