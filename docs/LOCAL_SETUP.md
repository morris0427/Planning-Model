# Local Setup Instructions

After downloading the framework files, here's how to set up your local directory:

## 📁 Directory Structure

Create this structure on your local machine:

```
your_project/
├── experiments/
│   ├── config.py                  # Downloaded
│   ├── run_experiments.py         # Downloaded
│   ├── analyze_results.py         # Downloaded
│   ├── README.md                  # Downloaded
│   ├── INTEGRATION_GUIDE.md       # Downloaded
│   ├── DIRECTORY_STRUCTURE.txt    # Downloaded
│   │
│   ├── data/
│   │   ├── __init__.py            # Create: empty file
│   │   ├── base.py                # Downloaded
│   │   ├── blocks_world.py        # Create: copy your generation code
│   │   └── puzzle.py              # Create: copy your generation code
│   │
│   ├── models/
│   │   ├── __init__.py            # Create: empty file
│   │   └── transformer.py         # Create: copy your model
│   │
│   ├── trainer.py                 # Create: copy your training code
│   │
│   └── results/                   # Will be created automatically
│       └── (experiment results will go here)
│
└── (your existing code files can stay here)
```

## 🚀 Quick Setup Commands

```bash
# 1. Create directory structure
mkdir -p experiments/data
mkdir -p experiments/models
mkdir -p experiments/results

# 2. Create __init__.py files
touch experiments/__init__.py
touch experiments/data/__init__.py
touch experiments/models/__init__.py

# 3. Move downloaded files
# Place config.py, run_experiments.py, analyze_results.py in experiments/
# Place base.py in experiments/data/

# 4. Test the framework is set up correctly
cd experiments
python config.py  # Should print example configurations
```

## ✅ Verify Setup

Run this to verify everything is in the right place:

```bash
cd experiments
python -c "
import sys
from pathlib import Path

required_files = [
    'config.py',
    'run_experiments.py',
    'analyze_results.py',
    'data/base.py'
]

missing = []
for f in required_files:
    if not Path(f).exists():
        missing.append(f)

if missing:
    print('❌ Missing files:')
    for f in missing:
        print(f'   - {f}')
else:
    print('✅ All core files present!')
    
# Try importing
try:
    from config import ExperimentConfig, ModelPresets
    print('✅ config.py imports successfully!')
except Exception as e:
    print(f'❌ Import error: {e}')
"
```

## 📝 Next Steps After Setup

1. **Read the documentation:**
   ```bash
   cat README.md
   cat INTEGRATION_GUIDE.md
   ```

2. **Create your domain files:**
   - Copy your Blocks World code into `data/blocks_world.py`
   - Copy your 8-Puzzle code into `data/puzzle.py`
   - See INTEGRATION_GUIDE.md for templates

3. **Create your model file:**
   - Copy your transformer into `models/transformer.py`

4. **Create your trainer:**
   - Copy your training loop into `trainer.py`

5. **Test it works:**
   ```bash
   python run_experiments.py --experiment blocks_world_standard
   ```

## 🔧 If You Get Import Errors

If you see `ModuleNotFoundError`, make sure:

1. All `__init__.py` files are created
2. You're running commands from the `experiments/` directory
3. Python can find the modules:
   ```bash
   export PYTHONPATH="${PYTHONPATH}:$(pwd)"
   ```

Or add this to the top of scripts:
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
```

## 📦 Alternative: Create as Package

You can also set it up as a proper Python package:

```bash
# Create setup.py
cat > setup.py << 'EOF'
from setuptools import setup, find_packages

setup(
    name="planning_experiments",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "numpy",
    ]
)
EOF

# Install in development mode
pip install -e .

# Now you can import from anywhere
python -c "from experiments.config import ModelPresets; print('Works!')"
```

## 💡 Tips

- Keep your old code in a separate directory as backup
- Use version control (git) to track changes
- Test each component as you integrate it
- Start with a small experiment before running full sweeps

## ❓ Common Issues

**Issue:** `ModuleNotFoundError: No module named 'data'`
**Fix:** Make sure you're in the `experiments/` directory and `__init__.py` files exist

**Issue:** `Cannot find config.py`
**Fix:** Check you're running from the right directory: `cd experiments/`

**Issue:** `Torch not found`
**Fix:** Install PyTorch: `pip install torch numpy`

---

**Ready to start? Begin with step 1 - create the directory structure!** 🚀
