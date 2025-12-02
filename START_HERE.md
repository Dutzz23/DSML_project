# 🚀 START HERE - Quick Launch Guide

## Installation Complete! ✅

Your DMSL project environment is now ready with all dependencies installed.

## Launch in 3 Simple Steps

### Step 1: Activate Virtual Environment

Open your terminal in this folder and run:

```bash
venv\Scripts\activate
```

You should see `(venv)` appear at the beginning of your command prompt.

### Step 2: Launch Jupyter Notebook

```bash
jupyter notebook
```

Your browser will open automatically showing the Jupyter interface.

### Step 3: Open a Notebook

In the Jupyter browser interface:
1. Click on the `notebooks/` folder
2. Start with `01_lorenz_system.ipynb`
3. Press `Shift+Enter` to run each cell

## 📚 Recommended Order

1. **[01_lorenz_system.ipynb](notebooks/01_lorenz_system.ipynb)** ← START HERE
   - Most comprehensive introduction
   - Trains both FNN and LSTM models
   - Beautiful 3D visualizations

2. **[03_van_der_pol.ipynb](notebooks/03_van_der_pol.ipynb)**
   - Simpler 2D system
   - Understand oscillatory vs chaotic behavior

3. **[05_comparative_analysis.ipynb](notebooks/05_comparative_analysis.ipynb)**
   - Compare all systems
   - Final conclusions

4. **[02_rossler_system.ipynb](notebooks/02_rossler_system.ipynb)** & **[04_duffing_oscillator.ipynb](notebooks/04_duffing_oscillator.ipynb)**
   - Additional systems for complete analysis

## ⌨️ Helpful Jupyter Shortcuts

- `Shift + Enter`: Run current cell and move to next
- `Ctrl + Enter`: Run current cell and stay
- `A`: Insert cell above
- `B`: Insert cell below
- `DD`: Delete cell (press D twice)

## 📖 Documentation

- **[README.md](README.md)**: Project overview
- **[GETTING_STARTED.md](GETTING_STARTED.md)**: Detailed tutorial
- **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)**: Code snippets and tips

## ❓ Need Help?

If you see any errors:
1. Make sure virtual environment is activated (you should see `(venv)`)
2. Check [GETTING_STARTED.md](GETTING_STARTED.md) for troubleshooting
3. All source code is in the `src/` folder with detailed docstrings

## 🎯 What You'll Learn

- How to implement 4 dynamical systems (Lorenz, Rössler, Van der Pol, Duffing)
- Train Feed-Forward and LSTM neural networks
- Predict chaotic and oscillatory behavior
- Visualize 2D/3D trajectories and phase spaces
- Compare model performance across systems

---

**Ready to begin?** Run: `venv\Scripts\activate` then `jupyter notebook`

Have fun exploring nonlinear dynamics! 🌊
