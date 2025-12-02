"""
Data Preparation Module

This module provides utilities for generating and preprocessing time-series data
from dynamical systems for neural network training.

Key functionalities:
- Trajectory generation from dynamical systems
- Sliding window sequence creation
- Data normalization and standardization
- Train-test splitting
- Noise injection for robustness testing
"""

import numpy as np
from typing import Tuple, Optional, List
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from .dynamical_systems import DynamicalSystem


def generate_trajectory(system: DynamicalSystem,
                        initial_state: Optional[np.ndarray] = None,
                        t_span: Tuple[float, float] = (0, 100),
                        dt: float = 0.01,
                        noise_std: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a trajectory from a dynamical system.

    Parameters:
    -----------
    system : DynamicalSystem
        The dynamical system to simulate
    initial_state : np.ndarray, optional
        Initial conditions. If None, uses default for the system
    t_span : tuple
        (t_start, t_end) time span for simulation
    dt : float
        Time step for output (default: 0.01)
    noise_std : float
        Standard deviation of Gaussian noise to add (default: 0.0 - no noise)

    Returns:
    --------
    t : np.ndarray
        Time points, shape (n_points,)
    trajectory : np.ndarray
        State trajectory, shape (n_points, n_dimensions)
    """
    # Default initial conditions if not provided
    if initial_state is None:
        if system.name == "Lorenz System":
            initial_state = np.array([1.0, 1.0, 1.0])
        elif system.name == "Rössler System":
            initial_state = np.array([1.0, 1.0, 1.0])
        elif system.name == "Van der Pol Oscillator":
            initial_state = np.array([1.0, 0.0])
        elif system.name == "Duffing Oscillator":
            initial_state = np.array([0.1, 0.1])
        else:
            raise ValueError(f"Unknown system: {system.name}")

    # Integrate the system
    t, trajectory = system.integrate(initial_state, t_span, dt)

    # Add noise if requested
    if noise_std > 0:
        noise = np.random.normal(0, noise_std, trajectory.shape)
        trajectory = trajectory + noise

    return t, trajectory


def create_sliding_windows(data: np.ndarray,
                           window_size: int,
                           prediction_horizon: int = 1,
                           stride: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sliding window sequences for time series prediction.

    Parameters:
    -----------
    data : np.ndarray
        Time series data, shape (n_points, n_features)
    window_size : int
        Number of time steps in input window
    prediction_horizon : int
        Number of time steps ahead to predict (default: 1)
    stride : int
        Step size between consecutive windows (default: 1)

    Returns:
    --------
    X : np.ndarray
        Input sequences, shape (n_samples, window_size, n_features)
    y : np.ndarray
        Target values, shape (n_samples, n_features)
    """
    n_points, n_features = data.shape
    X, y = [], []

    for i in range(0, n_points - window_size - prediction_horizon + 1, stride):
        # Input window
        X.append(data[i:i + window_size])
        # Target (value at prediction_horizon steps ahead)
        y.append(data[i + window_size + prediction_horizon - 1])

    return np.array(X), np.array(y)


def normalize_data(data: np.ndarray,
                   method: str = 'standard',
                   scaler: Optional[object] = None) -> Tuple[np.ndarray, object]:
    """
    Normalize time series data.

    Parameters:
    -----------
    data : np.ndarray
        Data to normalize, shape (n_points, n_features)
    method : str
        Normalization method: 'standard' (z-score) or 'minmax' (default: 'standard')
    scaler : object, optional
        Pre-fitted scaler to use. If None, fits new scaler

    Returns:
    --------
    normalized_data : np.ndarray
        Normalized data
    scaler : object
        Fitted scaler object (can be used to inverse transform)
    """
    if scaler is None:
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        scaler.fit(data)

    normalized_data = scaler.transform(data)
    return normalized_data, scaler


def train_test_split_sequential(X: np.ndarray,
                                 y: np.ndarray,
                                 train_ratio: float = 0.8) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split sequential data into train and test sets.

    Note: Uses sequential split (not random) to preserve temporal order.

    Parameters:
    -----------
    X : np.ndarray
        Input sequences
    y : np.ndarray
        Target values
    train_ratio : float
        Fraction of data to use for training (default: 0.8)

    Returns:
    --------
    X_train, y_train, X_test, y_test : np.ndarray
        Training and testing splits
    """
    n_samples = len(X)
    split_idx = int(n_samples * train_ratio)

    X_train = X[:split_idx]
    y_train = y[:split_idx]
    X_test = X[split_idx:]
    y_test = y[split_idx:]

    return X_train, y_train, X_test, y_test


def create_sequences(trajectory: np.ndarray,
                     window_size: int = 50,
                     prediction_horizon: int = 1,
                     train_ratio: float = 0.8,
                     normalize: bool = True,
                     normalization_method: str = 'standard') -> Tuple:
    """
    Complete pipeline: create sequences, normalize, and split data.

    This is a convenience function that combines all preprocessing steps.

    Parameters:
    -----------
    trajectory : np.ndarray
        Raw trajectory data, shape (n_points, n_features)
    window_size : int
        Size of input window (default: 50)
    prediction_horizon : int
        Steps ahead to predict (default: 1)
    train_ratio : float
        Fraction for training (default: 0.8)
    normalize : bool
        Whether to normalize data (default: True)
    normalization_method : str
        'standard' or 'minmax' (default: 'standard')

    Returns:
    --------
    X_train : np.ndarray
        Training input sequences
    y_train : np.ndarray
        Training targets
    X_test : np.ndarray
        Testing input sequences
    y_test : np.ndarray
        Testing targets
    scaler : object
        Fitted scaler (None if normalize=False)
    """
    # Normalize if requested
    scaler = None
    if normalize:
        trajectory, scaler = normalize_data(trajectory, method=normalization_method)

    # Create sliding windows
    X, y = create_sliding_windows(trajectory, window_size, prediction_horizon)

    # Split into train/test
    X_train, y_train, X_test, y_test = train_test_split_sequential(X, y, train_ratio)

    return X_train, y_train, X_test, y_test, scaler


def add_noise(data: np.ndarray, noise_std: float) -> np.ndarray:
    """
    Add Gaussian noise to data.

    Parameters:
    -----------
    data : np.ndarray
        Data to add noise to
    noise_std : float
        Standard deviation of Gaussian noise

    Returns:
    --------
    noisy_data : np.ndarray
        Data with added noise
    """
    noise = np.random.normal(0, noise_std, data.shape)
    return data + noise


def multi_step_prediction_data(trajectory: np.ndarray,
                                window_size: int = 50,
                                prediction_steps: int = 10,
                                train_ratio: float = 0.8) -> Tuple:
    """
    Prepare data for multi-step ahead prediction.

    Parameters:
    -----------
    trajectory : np.ndarray
        Raw trajectory data
    window_size : int
        Size of input window
    prediction_steps : int
        Number of steps to predict ahead
    train_ratio : float
        Fraction for training

    Returns:
    --------
    X_train, y_train, X_test, y_test : np.ndarray
        Training and testing data where y contains multiple future steps
    """
    n_points, n_features = trajectory.shape
    X, y = [], []

    for i in range(n_points - window_size - prediction_steps):
        X.append(trajectory[i:i + window_size])
        y.append(trajectory[i + window_size:i + window_size + prediction_steps])

    X = np.array(X)
    y = np.array(y)

    # Split
    split_idx = int(len(X) * train_ratio)
    X_train = X[:split_idx]
    y_train = y[:split_idx]
    X_test = X[split_idx:]
    y_test = y[split_idx:]

    return X_train, y_train, X_test, y_test


# Example usage and testing
if __name__ == "__main__":
    from dynamical_systems import LorenzSystem

    print("Testing Data Preparation Module")
    print("=" * 50)

    # Generate trajectory
    system = LorenzSystem()
    t, trajectory = generate_trajectory(
        system,
        t_span=(0, 50),
        dt=0.01,
        noise_std=0.0
    )
    print(f"✓ Generated trajectory: {trajectory.shape}")

    # Create sequences
    X_train, y_train, X_test, y_test, scaler = create_sequences(
        trajectory,
        window_size=50,
        prediction_horizon=1,
        train_ratio=0.8,
        normalize=True
    )
    print(f"✓ Training data: X={X_train.shape}, y={y_train.shape}")
    print(f"✓ Testing data: X={X_test.shape}, y={y_test.shape}")

    # Test multi-step prediction data
    X_train_ms, y_train_ms, X_test_ms, y_test_ms = multi_step_prediction_data(
        trajectory,
        window_size=50,
        prediction_steps=10,
        train_ratio=0.8
    )
    print(f"✓ Multi-step data: y_train={y_train_ms.shape}, y_test={y_test_ms.shape}")

    print("\n" + "=" * 50)
    print("All data preparation functions tested successfully!")
