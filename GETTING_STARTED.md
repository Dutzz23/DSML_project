# Getting Started Guide

## Quick Start

Follow these steps to get the project up and running:

### 1. Setup Virtual Environment

Run the automated setup script:

```bash
python setup_venv.py
```

This will:
- Create a Python virtual environment
- Install all required dependencies
- Verify the installation

### 2. Activate Virtual Environment

**Windows (CMD):**
```bash
venv\Scripts\activate.bat
```

**Windows (PowerShell):**
```bash
venv\Scripts\Activate.ps1
```

**Linux/Mac:**
```bash
source venv/bin/activate
```

### 3. Launch Jupyter Notebook

```bash
jupyter notebook
```

Your browser will open automatically. Navigate to the `notebooks/` folder.

## Recommended Learning Path

### Beginner Path

1. **Start with Lorenz System** ([01_lorenz_system.ipynb](notebooks/01_lorenz_system.ipynb))
   - Classic chaotic system
   - Comprehensive introduction to the workflow
   - Train both FNN and LSTM models
   - Visualize 3D attractors

2. **Van der Pol Oscillator** ([03_van_der_pol.ipynb](notebooks/03_van_der_pol.ipynb))
   - Simpler 2D system
   - Understand limit cycles
   - Compare with chaotic systems

3. **Comparative Analysis** ([05_comparative_analysis.ipynb](notebooks/05_comparative_analysis.ipynb))
   - Cross-system comparison
   - Key insights and conclusions

### Advanced Path

4. **Rössler System** ([02_rossler_system.ipynb](notebooks/02_rossler_system.ipynb))
   - Alternative chaotic attractor
   - Compare with Lorenz

5. **Duffing Oscillator** ([04_duffing_oscillator.ipynb](notebooks/04_duffing_oscillator.ipynb))
   - Non-autonomous system
   - Periodic and chaotic regimes

## Project Structure Overview

```
DMSL PROJECT/
│
├── src/                              # Reusable Python modules
│   ├── dynamical_systems.py          # System implementations
│   ├── data_preparation.py           # Data processing utilities
│   ├── neural_models.py              # Neural network architectures
│   └── evaluation.py                 # Metrics and visualization
│
├── notebooks/                        # Jupyter notebooks
│   ├── 01_lorenz_system.ipynb        # Lorenz system analysis
│   ├── 02_rossler_system.ipynb       # Rössler system analysis
│   ├── 03_van_der_pol.ipynb          # Van der Pol analysis
│   ├── 04_duffing_oscillator.ipynb   # Duffing oscillator analysis
│   └── 05_comparative_analysis.ipynb # Cross-system comparison
│
├── requirements.txt                  # Python dependencies
├── setup_venv.py                     # Automated setup script
├── README.md                         # Project overview
└── GETTING_STARTED.md               # This file
```

## Basic Usage Example

Here's a minimal example of using the modules directly (without notebooks):

```python
from src.dynamical_systems import LorenzSystem
from src.data_preparation import generate_trajectory, create_sequences
from src.neural_models import LSTMPredictor, NeuralPredictor
from src.evaluation import evaluate_prediction

# 1. Generate data
system = LorenzSystem()
t, trajectory = generate_trajectory(system, t_span=(0, 50), dt=0.01)

# 2. Prepare sequences
X_train, y_train, X_test, y_test, scaler = create_sequences(
    trajectory, window_size=50, train_ratio=0.8
)

# 3. Train model
model = LSTMPredictor(input_dim=3, hidden_dim=64, num_layers=2, output_dim=3)
predictor = NeuralPredictor(model, learning_rate=0.001)
predictor.train(X_train, y_train, epochs=100)

# 4. Evaluate
predictions = predictor.predict(X_test)
metrics = evaluate_prediction(y_test, predictions)
print(f"RMSE: {metrics['rmse']:.6f}")
```

## Understanding the Systems

### Chaotic Systems

**Lorenz System** and **Rössler System** exhibit:
- Sensitive dependence on initial conditions
- Bounded but non-periodic trajectories
- Strange attractors
- Limited prediction horizon

**Key Insight**: Error grows exponentially with prediction horizon

### Oscillatory Systems

**Van der Pol Oscillator** exhibits:
- Stable limit cycles
- Relaxation oscillations
- Predictable periodic behavior
- Long-term stability

**Duffing Oscillator** can exhibit:
- Both periodic and chaotic behavior
- Depends on forcing amplitude
- Double-well potential dynamics

**Key Insight**: Periodic systems are fundamentally more predictable

## Neural Network Models

### Feed-Forward Network (FNN)
- Flattens input window
- Multiple dense layers
- Good for short-term prediction
- Faster training

### LSTM Network
- Processes sequential data
- Memory cells for long-term dependencies
- Better for temporal patterns
- Superior long-term prediction

### When to Use Each?
- **FNN**: Quick prototyping, very short horizons
- **LSTM**: Production use, better accuracy, temporal data

## Common Tasks

### Change System Parameters

```python
# Lorenz with different parameters
lorenz_custom = LorenzSystem(sigma=15.0, rho=30.0, beta=2.5)

# Van der Pol with higher nonlinearity
vdp_strong = VanDerPolOscillator(mu=5.0)
```

### Add Noise to Data

```python
t, noisy_trajectory = generate_trajectory(
    system,
    t_span=(0, 50),
    noise_std=0.05  # Add 5% noise
)
```

### Adjust Model Architecture

```python
# Deeper LSTM
deep_lstm = LSTMPredictor(
    input_dim=3,
    hidden_dim=128,
    num_layers=3,  # More layers
    dropout=0.2    # Regularization
)

# Wider FNN
wide_fnn = FeedForwardPredictor(
    input_dim=150,
    hidden_dims=[256, 128, 64],  # Wider layers
    output_dim=3
)
```

### Multi-Step Prediction

```python
# Predict 100 steps into future
initial_window = X_test[0]
future_trajectory = predictor.iterative_predict(
    initial_window,
    n_steps=100
)
```

## Troubleshooting

### Import Errors
```bash
# Make sure you're in the virtual environment
# Check with:
which python  # Linux/Mac
where python  # Windows

# Should point to venv/bin/python or venv\Scripts\python.exe
```

### CUDA/GPU Issues
```python
# Force CPU usage if GPU issues occur
predictor = NeuralPredictor(model, device='cpu')
```

### Memory Errors
```python
# Reduce batch size
predictor.train(X_train, y_train, batch_size=16)  # Default is 32

# Or reduce trajectory length
t, trajectory = generate_trajectory(system, t_span=(0, 20))  # Shorter
```

### Slow Training
- Reduce epochs for quick experiments
- Use smaller models (fewer layers/units)
- Enable GPU if available
- Reduce window size

## Further Exploration

### Experiment Ideas

1. **Different Parameter Regimes**
   - Vary Lorenz ρ to change chaos level
   - Test Van der Pol with μ from 0.1 to 10

2. **Architecture Comparison**
   - Try GRU vs LSTM
   - Test different hidden dimensions
   - Compare 1 vs 2 vs 3 layers

3. **Data Augmentation**
   - Multiple initial conditions
   - Various noise levels
   - Different time steps (dt)

4. **Advanced Techniques**
   - Ensemble methods
   - Bayesian neural networks
   - Physics-informed neural networks

## Getting Help

- **Documentation**: Check docstrings in source files
- **Examples**: All notebooks contain detailed examples
- **Issues**: Review error messages carefully

## Tips for Success

1. **Start Simple**: Begin with provided examples
2. **Visualize**: Always plot your results
3. **Validate**: Check predictions make physical sense
4. **Iterate**: Experiment with hyperparameters
5. **Document**: Keep notes on what works

## Next Steps

After completing the notebooks:

1. **Modify Systems**: Change parameters and observe effects
2. **Custom Models**: Design your own architectures
3. **New Systems**: Implement other dynamical systems
4. **Real Data**: Apply techniques to experimental data
5. **Research**: Explore physics-informed ML methods

## References

- Roy & Rana (2020): *Machine Learning in Nonlinear Dynamical Systems*
- Project Paper: See PDF document for theoretical background

---

**Happy Exploring! 🚀**

For questions or issues, contact:
- Vlad-Flavius Misăilă: vlad.misaila02@e-uvt.ro
- Robert-Daniel Man: robert.man01@e-uvt.ro
- Sebastian-Adrian Mărginean: sebastian.marginean02@e-uvt.ro
