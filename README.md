# Learning Predictive Models for Nonlinear Dynamical Systems Using Neural Networks

This project implements neural network models to predict trajectories of nonlinear dynamical systems using only observed time-series data.

## Team 19
- Vlad-Flavius Misăilă (vlad.misaila02@e-uvt.ro)
- Robert-Daniel Man (robert.man01@e-uvt.ro)
- Sebastian-Adrian Mărginean (sebastian.marginean02@e-uvt.ro)

## Research Question
How accurately can neural network models predict future trajectories of nonlinear dynamical systems—specifically chaotic and oscillatory systems—using only observed time-series data, without explicit knowledge of the governing differential equations?

## Project Structure

```
DMSL PROJECT/
├── src/                               # Source code modules
│   ├── dynamical_systems.py           # Implementations of dynamical systems
│   ├── data_preparation.py            # Data generation and preprocessing utilities
│   ├── neural_models.py               # Neural network architectures
│   └── evaluation.py                  # Evaluation metrics and visualization
├── notebooks/                         # Jupyter notebooks
│   ├── 01_lorenz_system.ipynb         # Lorenz attractor experiments
│   ├── 02_rossler_system.ipynb        # Rössler attractor experiments
│   ├── 03_van_der_pol.ipynb           # Van der Pol oscillator experiments
│   ├── 04_duffing_oscillator.ipynb    # Duffing oscillator experiments
│   └── 05_comparative_analysis.ipynb  # Cross-system comparison
├── requirements.txt                   # Python dependencies
├── setup_venv.py                      # Virtual environment setup script
└── README.md                          # This file

## Setup Instructions

### 1. Create and activate virtual environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/Mac:**
```bash
python -m venv venv
source venv/bin/activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch Jupyter
```bash
jupyter notebook
```

Navigate to the `notebooks/` folder and start with any notebook.

## Systems Implemented

### Chaotic Systems
1. **Lorenz System**: Classic chaotic system exhibiting sensitive dependence on initial conditions
2. **Rössler System**: Simpler chaotic attractor with continuous band of oscillations

### Oscillatory Systems
3. **Van der Pol Oscillator**: Nonlinear oscillator with limit cycles
4. **Duffing Oscillator**: Forced oscillator exhibiting periodic and chaotic behavior

## Methodology

1. **Data Preparation**:
   - Numerical simulation using 4th-order Runge-Kutta
   - Sliding-window embedding for sequence generation
   - Normalization and train-test split
   - Optional noise injection

2. **Modeling**:
   - Feed-forward neural networks (FNN)
   - Recurrent neural networks (RNN/LSTM)
   - Training via stochastic gradient descent

3. **Evaluation**:
   - RMSE (Root Mean Square Error)
   - Prediction horizon analysis
   - Sensitivity to initial conditions
   - Robustness under noise

## Usage Example

```python
from src.dynamical_systems import LorenzSystem
from src.data_preparation import generate_trajectory, create_sequences
from src.neural_models import FeedForwardPredictor
from src.evaluation import evaluate_prediction, plot_trajectories

# Generate training data
system = LorenzSystem()
t, trajectory = generate_trajectory(system, t_span=(0, 50), dt=0.01)

# Prepare sequences
X_train, y_train, X_test, y_test = create_sequences(
    trajectory, window_size=50, train_ratio=0.8
)

# Train model
model = FeedForwardPredictor(input_dim=3, hidden_dims=[64, 64])
model.train(X_train, y_train, epochs=100)

# Evaluate
predictions = model.predict(X_test)
metrics = evaluate_prediction(y_test, predictions)
plot_trajectories(y_test, predictions)
```

## References

S. Roy and D. Rana, *Machine Learning in Nonlinear Dynamical Systems*, 2020.
