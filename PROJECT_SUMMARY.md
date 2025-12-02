# Project Summary: Neural Networks for Nonlinear Dynamical Systems

## Overview

This project implements neural network models to predict trajectories of nonlinear dynamical systems using only observed time-series data, based on the research paper by Team 19.

**Team Members**:
- Vlad-Flavius Misăilă
- Robert-Daniel Man
- Sebastian-Adrian Mărginean

## What Has Been Created

### 📁 Project Structure

```
DMSL PROJECT/
│
├── src/                                    # Core Python modules (reusable)
│   ├── __init__.py                         # Package initialization
│   ├── dynamical_systems.py                # 4 dynamical systems implementations
│   ├── data_preparation.py                 # Data preprocessing utilities
│   ├── neural_models.py                    # Neural network architectures
│   └── evaluation.py                       # Metrics and visualization tools
│
├── notebooks/                              # Jupyter notebooks (experiments)
│   ├── 01_lorenz_system.ipynb              # Lorenz chaotic system
│   ├── 02_rossler_system.ipynb             # Rössler chaotic system
│   ├── 03_van_der_pol.ipynb                # Van der Pol oscillator
│   ├── 04_duffing_oscillator.ipynb         # Duffing oscillator
│   └── 05_comparative_analysis.ipynb       # Cross-system comparison
│
├── requirements.txt                        # Python dependencies
├── setup_venv.py                           # Automated setup script
├── README.md                               # Main project documentation
├── GETTING_STARTED.md                      # Quick start guide
├── PROJECT_SUMMARY.md                      # This file
└── .gitignore                              # Git ignore rules
```

### 🔬 Implemented Dynamical Systems

1. **Lorenz System** (Chaotic, 3D)
   - Classic chaotic attractor
   - Parameters: σ, ρ, β
   - Exhibits sensitive dependence on initial conditions

2. **Rössler System** (Chaotic, 3D)
   - Simpler chaotic attractor
   - Single continuous band structure
   - Parameters: a, b, c

3. **Van der Pol Oscillator** (Oscillatory, 2D)
   - Nonlinear oscillator with limit cycles
   - Relaxation oscillations
   - Parameter: μ (damping)

4. **Duffing Oscillator** (Chaotic/Periodic, 2D)
   - Forced nonlinear oscillator
   - Double-well potential
   - Can exhibit both periodic and chaotic behavior

### 🧠 Neural Network Models

#### Feed-Forward Networks (FNN)
- Multiple fully-connected layers
- ReLU activation
- Dropout regularization
- Best for: Quick prototyping, short-term prediction

#### LSTM Networks
- Recurrent architecture
- Memory cells for temporal dependencies
- Better long-term prediction
- Best for: Production use, temporal sequences

#### GRU Networks
- Simplified recurrent architecture
- Faster training than LSTM
- Similar performance to LSTM

### 📊 Features Implemented

#### Data Preparation
- ✅ Trajectory generation using 4th-order Runge-Kutta
- ✅ Sliding window sequence creation
- ✅ Data normalization (standard/minmax)
- ✅ Train-test splitting (sequential)
- ✅ Noise injection for robustness testing
- ✅ Multi-step prediction data preparation

#### Model Training
- ✅ PyTorch implementation
- ✅ Automatic device selection (CPU/GPU)
- ✅ Progress tracking with tqdm
- ✅ Training history logging
- ✅ Validation during training
- ✅ Batch processing

#### Evaluation & Visualization
- ✅ RMSE, MAE, Normalized RMSE metrics
- ✅ Per-dimension error analysis
- ✅ 2D/3D trajectory plots
- ✅ Time series visualization
- ✅ Phase space portraits
- ✅ Prediction horizon analysis
- ✅ Training history plots
- ✅ Error growth curves

### 📓 Jupyter Notebooks

Each notebook includes:
- ✅ Comprehensive documentation
- ✅ Mathematical equations in LaTeX
- ✅ System-specific visualizations
- ✅ Model training with progress bars
- ✅ One-step and multi-step predictions
- ✅ Performance metrics
- ✅ Prediction horizon analysis
- ✅ Noise robustness testing
- ✅ Conclusions and insights

#### Notebook 1: Lorenz System
- 3D attractor visualization
- FNN vs LSTM comparison
- Multi-step iterative prediction
- Prediction error growth analysis
- Sensitivity to noise
- ~12 executable sections

#### Notebook 2: Rössler System
- Characteristic band structure visualization
- Parameter effects
- Attractor comparison
- Cross-validation with Lorenz
- ~7 executable sections

#### Notebook 3: Van der Pol
- Limit cycle behavior
- Phase portraits for different μ values
- FNN vs LSTM vs GRU comparison
- Long-term limit cycle preservation
- Applications to biological systems
- ~8 executable sections

#### Notebook 4: Duffing Oscillator
- Periodic vs chaotic regimes
- Forcing amplitude effects
- Strange attractor reconstruction
- Non-autonomous system analysis
- ~8 executable sections

#### Notebook 5: Comparative Analysis
- Cross-system performance comparison
- Window size optimization
- Noise robustness across all systems
- Chaotic vs oscillatory predictability
- Comprehensive summary statistics
- Research question answered
- ~6 major analysis sections

## 🎯 Key Features

### Code Quality
- ✅ Fully documented with docstrings
- ✅ Type hints throughout
- ✅ Modular and reusable design
- ✅ Object-oriented architecture
- ✅ Example usage in each module
- ✅ Error handling

### Reproducibility
- ✅ Fixed random seeds
- ✅ Automated setup script
- ✅ Requirements.txt for dependencies
- ✅ Detailed documentation
- ✅ Step-by-step notebooks

### Educational Value
- ✅ Clear explanations
- ✅ Mathematical background
- ✅ Physical insights
- ✅ Comparative analysis
- ✅ Best practices demonstrated

## 📈 Experiments Covered

### Core Experiments (As per paper requirements)
1. ✅ Train neural networks on chaotic systems (Lorenz, Rössler)
2. ✅ Train on oscillatory systems (Van der Pol, Duffing)
3. ✅ Evaluate short-term prediction accuracy
4. ✅ Analyze medium-term prediction
5. ✅ Test effects of noise
6. ✅ Study effects of window length
7. ✅ Compare FNN vs RNN/LSTM architectures
8. ✅ Characterize prediction degradation in chaotic systems

### Additional Analysis
9. ✅ Prediction horizon analysis
10. ✅ Cross-system comparison
11. ✅ Hyperparameter sensitivity
12. ✅ GRU architecture testing
13. ✅ Phase space preservation
14. ✅ Regime-dependent behavior (Duffing)

## 🚀 How to Use

### Quick Start
```bash
# 1. Run setup
python setup_venv.py

# 2. Activate environment
venv\Scripts\activate          # Windows
source venv/bin/activate       # Linux/Mac

# 3. Launch Jupyter
jupyter notebook

# 4. Open any notebook in notebooks/
```

### Using Modules Directly
```python
from src.dynamical_systems import LorenzSystem
from src.data_preparation import generate_trajectory, create_sequences
from src.neural_models import LSTMPredictor, NeuralPredictor
from src.evaluation import evaluate_prediction, plot_trajectory_3d

# Your code here...
```

## 📊 Expected Results

### Performance Metrics (typical values)

**Lorenz System**:
- FNN RMSE: ~0.02-0.05 (normalized)
- LSTM RMSE: ~0.01-0.03 (normalized)
- LSTM Improvement: ~30-50%

**Van der Pol Oscillator**:
- FNN RMSE: ~0.01-0.02
- LSTM RMSE: ~0.005-0.015
- LSTM Improvement: ~20-40%

**Key Finding**: LSTM consistently outperforms FNN, especially for chaotic systems

## 🎓 Learning Outcomes

After completing this project, you will understand:

1. **Dynamical Systems Theory**
   - Chaotic vs oscillatory behavior
   - Attractors and limit cycles
   - Lyapunov divergence
   - Sensitivity to initial conditions

2. **Machine Learning**
   - Time series prediction
   - Feed-forward vs recurrent architectures
   - Hyperparameter tuning
   - Model evaluation

3. **Data Science**
   - Data preprocessing
   - Normalization techniques
   - Train-test splitting
   - Performance metrics

4. **Python Programming**
   - NumPy for numerical computing
   - PyTorch for deep learning
   - Matplotlib for visualization
   - Object-oriented design

## 📝 Documentation Quality

- ✅ README.md: Project overview and setup
- ✅ GETTING_STARTED.md: Detailed guide for beginners
- ✅ PROJECT_SUMMARY.md: This comprehensive summary
- ✅ Inline comments: Throughout all code
- ✅ Docstrings: Every function and class
- ✅ Markdown cells: Extensive notebook documentation
- ✅ Mathematical notation: LaTeX equations

## 🔧 Technical Specifications

### Dependencies
- Python ≥ 3.8
- NumPy ≥ 1.24.0
- SciPy ≥ 1.10.0
- PyTorch ≥ 2.0.0
- TensorFlow ≥ 2.13.0
- Matplotlib ≥ 3.7.0
- Pandas ≥ 2.0.0
- Jupyter ≥ 1.0.0
- scikit-learn ≥ 1.3.0

### Code Statistics (approximate)
- Total Python files: 5 modules + 1 setup script
- Total Jupyter notebooks: 5 comprehensive notebooks
- Lines of Python code: ~2,500
- Lines of documentation: ~1,500
- Number of functions: ~50+
- Number of classes: ~10+

## ✅ Project Completeness Checklist

### Required Components
- ✅ Virtual environment setup
- ✅ All 4 dynamical systems implemented
- ✅ Data preparation pipeline
- ✅ Multiple neural network architectures
- ✅ Comprehensive evaluation metrics
- ✅ Visualization tools
- ✅ Jupyter notebooks for each system
- ✅ Comparative analysis
- ✅ Documentation
- ✅ Requirements file

### Code Quality
- ✅ Modular design
- ✅ Reusable components
- ✅ Well-documented
- ✅ Type hints
- ✅ Error handling
- ✅ Example usage

### Experiments
- ✅ Chaotic systems (Lorenz, Rössler)
- ✅ Oscillatory systems (Van der Pol, Duffing)
- ✅ One-step prediction
- ✅ Multi-step prediction
- ✅ Noise robustness
- ✅ Window size effects
- ✅ Architecture comparison

## 🎉 Summary

This project provides a **complete, production-ready framework** for studying neural network predictions of nonlinear dynamical systems. It includes:

- ✅ **4 fully implemented dynamical systems** with physical insights
- ✅ **3 neural network architectures** (FNN, LSTM, GRU)
- ✅ **5 comprehensive Jupyter notebooks** with experiments
- ✅ **Reusable Python modules** for all functionality
- ✅ **Extensive documentation** for easy understanding
- ✅ **Automated setup** for quick start
- ✅ **All experiments from the paper** plus additional analysis

The code is:
- **Well-structured** for easy reuse and extension
- **Thoroughly documented** for educational purposes
- **Scientifically rigorous** with proper methodology
- **Practically useful** for real research applications

## 🔜 Next Steps

Potential extensions:
1. Add physics-informed neural networks (PINNs)
2. Implement reservoir computing / echo state networks
3. Add uncertainty quantification
4. Test on real experimental data
5. Implement ensemble methods
6. Add attention mechanisms
7. Compare with traditional numerical methods

---

**Project Status**: ✅ COMPLETE

All requirements from the research paper have been implemented and documented.

**Date**: 2025-11-26
**Team 19**: West University of Timișoara, Department of Computer Science
