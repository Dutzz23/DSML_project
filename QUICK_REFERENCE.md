# Quick Reference Guide

## Installation (One-Time Setup)

```bash
# Run the automated setup
python setup_venv.py

# Activate virtual environment
venv\Scripts\activate          # Windows
source venv/bin/activate       # Linux/Mac

# Launch Jupyter
jupyter notebook
```

## Module Import Cheatsheet

```python
# Dynamical Systems
from src.dynamical_systems import (
    LorenzSystem,
    RosslerSystem,
    VanDerPolOscillator,
    DuffingOscillator
)

# Data Preparation
from src.data_preparation import (
    generate_trajectory,
    create_sequences,
    normalize_data,
    add_noise
)

# Neural Models
from src.neural_models import (
    FeedForwardPredictor,
    LSTMPredictor,
    GRUPredictor,
    NeuralPredictor
)

# Evaluation
from src.evaluation import (
    evaluate_prediction,
    plot_trajectory_2d,
    plot_trajectory_3d,
    plot_time_series,
    plot_training_history
)
```

## Common Code Snippets

### Generate System Data

```python
# Lorenz System
lorenz = LorenzSystem(sigma=10, rho=28, beta=8/3)
t, traj = lorenz.integrate(
    initial_state=np.array([1, 1, 1]),
    t_span=(0, 50),
    dt=0.01
)

# Van der Pol
vdp = VanDerPolOscillator(mu=2.0)
t, traj = vdp.integrate(
    initial_state=np.array([2, 0]),
    t_span=(0, 50),
    dt=0.01
)
```

### Prepare Training Data

```python
# Complete pipeline
X_train, y_train, X_test, y_test, scaler = create_sequences(
    trajectory,
    window_size=50,        # Look back 50 steps
    prediction_horizon=1,  # Predict 1 step ahead
    train_ratio=0.8,       # 80% training
    normalize=True
)
```

### Train a Model

```python
# LSTM (recommended)
lstm = LSTMPredictor(input_dim=3, hidden_dim=64, num_layers=2, output_dim=3)
predictor = NeuralPredictor(lstm, learning_rate=0.001)

history = predictor.train(
    X_train, y_train,
    X_val=X_test,
    y_val=y_test,
    epochs=100,
    batch_size=32
)

# Feed-Forward
fnn = FeedForwardPredictor(
    input_dim=50*3,  # window_size * n_features
    hidden_dims=[128, 64, 32],
    output_dim=3
)
predictor = NeuralPredictor(fnn, learning_rate=0.001)
predictor.train(X_train, y_train, epochs=100)
```

### Make Predictions

```python
# One-step prediction
predictions = predictor.predict(X_test)

# Evaluate
metrics = evaluate_prediction(y_test, predictions)
print(f"RMSE: {metrics['rmse']:.6f}")

# Multi-step iterative prediction
future = predictor.iterative_predict(
    initial_window=X_test[0],
    n_steps=100
)
```

### Visualize Results

```python
# 3D trajectory
plot_trajectory_3d(y_test, predictions, title="Predictions")

# Time series
plot_time_series(t, trajectory, feature_names=['X', 'Y', 'Z'])

# Training history
plot_training_history(history)
```

## System Parameters

### Lorenz System
```python
LorenzSystem(
    sigma=10.0,   # Prandtl number
    rho=28.0,     # Rayleigh number
    beta=8.0/3.0  # Geometric factor
)
```

### Rössler System
```python
RosslerSystem(
    a=0.2,  # Default parameters
    b=0.2,
    c=5.7
)
```

### Van der Pol Oscillator
```python
VanDerPolOscillator(
    mu=1.0  # Nonlinearity parameter
    # mu=0: harmonic oscillator
    # mu>0: self-excited oscillations
)
```

### Duffing Oscillator
```python
DuffingOscillator(
    alpha=-1.0,   # Linear stiffness
    beta=1.0,     # Nonlinear stiffness
    delta=0.3,    # Damping
    gamma=0.5,    # Forcing amplitude
    omega=1.2     # Forcing frequency
)
```

## Model Architectures

### Small Model (Fast Training)
```python
FeedForwardPredictor(input_dim=150, hidden_dims=[32, 16], output_dim=3)
LSTMPredictor(input_dim=3, hidden_dim=32, num_layers=1, output_dim=3)
```

### Medium Model (Balanced)
```python
FeedForwardPredictor(input_dim=150, hidden_dims=[64, 32], output_dim=3)
LSTMPredictor(input_dim=3, hidden_dim=64, num_layers=2, output_dim=3)
```

### Large Model (Best Accuracy)
```python
FeedForwardPredictor(input_dim=150, hidden_dims=[128, 64, 32], output_dim=3)
LSTMPredictor(input_dim=3, hidden_dim=128, num_layers=3, output_dim=3)
```

## Hyperparameter Guidelines

### Window Size
- **Small (20-30)**: Fast, less context
- **Medium (50-75)**: Balanced (recommended)
- **Large (100+)**: More context, slower

### Learning Rate
- **0.001**: Safe default
- **0.0001**: Slower, more stable
- **0.01**: Faster, may be unstable

### Batch Size
- **16**: Small datasets, more updates
- **32**: Standard (recommended)
- **64**: Larger datasets, faster epochs

### Epochs
- **Quick test**: 20-30
- **Standard**: 50-100
- **Thorough**: 100-200

## Evaluation Metrics

```python
metrics = evaluate_prediction(y_true, y_pred)

# Available metrics:
metrics['rmse']           # Root Mean Square Error
metrics['mae']            # Mean Absolute Error
metrics['nrmse']          # Normalized RMSE
metrics['rmse_per_dim']   # RMSE for each dimension
metrics['mae_per_dim']    # MAE for each dimension
```

## Common Issues & Solutions

### Out of Memory
```python
# Reduce batch size
predictor.train(X_train, y_train, batch_size=16)

# Or use shorter trajectory
t, traj = system.integrate(t_span=(0, 20))  # Instead of (0, 100)
```

### Poor Predictions
```python
# Try:
1. Increase epochs (e.g., 100 → 200)
2. Use LSTM instead of FNN
3. Increase hidden dimensions
4. Adjust window size
5. Check data normalization
```

### Slow Training
```python
# Speed up:
1. Reduce epochs for testing
2. Use smaller model
3. Increase batch size
4. Use GPU if available
```

### GPU Usage
```python
# Check if GPU available
import torch
print(torch.cuda.is_available())

# Force CPU
predictor = NeuralPredictor(model, device='cpu')

# Force GPU
predictor = NeuralPredictor(model, device='cuda')
```

## File Locations

### Notebooks
- [notebooks/01_lorenz_system.ipynb](notebooks/01_lorenz_system.ipynb) - Start here!
- [notebooks/03_van_der_pol.ipynb](notebooks/03_van_der_pol.ipynb) - Oscillatory system
- [notebooks/05_comparative_analysis.ipynb](notebooks/05_comparative_analysis.ipynb) - All systems

### Source Code
- [src/dynamical_systems.py](src/dynamical_systems.py) - System implementations
- [src/neural_models.py](src/neural_models.py) - Model architectures
- [src/evaluation.py](src/evaluation.py) - Plotting and metrics

### Documentation
- [README.md](README.md) - Project overview
- [GETTING_STARTED.md](GETTING_STARTED.md) - Detailed tutorial
- [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) - Complete summary

## Notebook Workflow

```
1. Import modules
2. Generate/load data
3. Visualize raw data
4. Prepare sequences
5. Train model(s)
6. Evaluate predictions
7. Visualize results
8. Analyze errors
9. Draw conclusions
```

## Typical Analysis Pipeline

```python
# 1. Setup
from src.dynamical_systems import LorenzSystem
from src.data_preparation import generate_trajectory, create_sequences
from src.neural_models import LSTMPredictor, NeuralPredictor
from src.evaluation import evaluate_prediction, plot_trajectory_3d

# 2. Generate data
system = LorenzSystem()
t, traj = system.integrate([1,1,1], (0,50), dt=0.01)

# 3. Prepare
X_train, y_train, X_test, y_test, scaler = create_sequences(
    traj, window_size=50, train_ratio=0.8
)

# 4. Train
model = LSTMPredictor(3, 64, 2, 3)
predictor = NeuralPredictor(model, learning_rate=0.001)
predictor.train(X_train, y_train, epochs=100)

# 5. Evaluate
pred = predictor.predict(X_test)
metrics = evaluate_prediction(y_test, pred)
print(f"RMSE: {metrics['rmse']:.6f}")

# 6. Visualize
plot_trajectory_3d(
    scaler.inverse_transform(y_test[:200]),
    scaler.inverse_transform(pred[:200])
)
```

## Keyboard Shortcuts (Jupyter)

- `Shift + Enter`: Run cell and move to next
- `Ctrl + Enter`: Run cell and stay
- `A`: Insert cell above
- `B`: Insert cell below
- `DD`: Delete cell
- `M`: Convert to markdown
- `Y`: Convert to code

## Tips for Success

1. ✅ **Start simple**: Run provided examples first
2. ✅ **Visualize often**: Always plot your data and results
3. ✅ **Check shapes**: Print array shapes to debug
4. ✅ **Save work**: Export important results
5. ✅ **Document**: Add comments to modified code

## Getting Help

1. Check docstrings: `help(function_name)`
2. Read error messages carefully
3. Review notebook examples
4. Check [GETTING_STARTED.md](GETTING_STARTED.md)

## Next Steps

After running the notebooks:
1. Modify parameters and observe changes
2. Try different model architectures
3. Implement custom systems
4. Experiment with real data
5. Explore advanced techniques

---

**Pro Tip**: Keep this file open as reference while working with the notebooks!
