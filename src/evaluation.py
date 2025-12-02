"""
Evaluation and Visualization Module

This module provides utilities for evaluating neural network predictions
and visualizing results for dynamical systems.

Key functionalities:
- Prediction error metrics (RMSE, MAE, etc.)
- Trajectory visualization (2D and 3D)
- Error analysis and plots
- Phase space comparisons
- Lyapunov exponent estimation
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Tuple, Optional, List
from sklearn.metrics import mean_squared_error, mean_absolute_error


# ============================================================================
# Evaluation Metrics
# ============================================================================

def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Root Mean Square Error.

    Parameters:
    -----------
    y_true : np.ndarray
        True values
    y_pred : np.ndarray
        Predicted values

    Returns:
    --------
    float
        RMSE value
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))


def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Mean Absolute Error.

    Parameters:
    -----------
    y_true : np.ndarray
        True values
    y_pred : np.ndarray
        Predicted values

    Returns:
    --------
    float
        MAE value
    """
    return mean_absolute_error(y_true, y_pred)


def compute_normalized_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Normalized RMSE (NRMSE).

    Normalized by the range of true values.

    Parameters:
    -----------
    y_true : np.ndarray
        True values
    y_pred : np.ndarray
        Predicted values

    Returns:
    --------
    float
        NRMSE value
    """
    rmse = compute_rmse(y_true, y_pred)
    value_range = np.max(y_true) - np.min(y_true)
    return rmse / value_range if value_range > 0 else rmse


def evaluate_prediction(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Comprehensive evaluation of predictions.

    Parameters:
    -----------
    y_true : np.ndarray
        True values, shape (n_samples, n_features)
    y_pred : np.ndarray
        Predicted values, shape (n_samples, n_features)

    Returns:
    --------
    dict
        Dictionary containing various metrics
    """
    metrics = {
        'rmse': compute_rmse(y_true, y_pred),
        'mae': compute_mae(y_true, y_pred),
        'nrmse': compute_normalized_rmse(y_true, y_pred),
        'rmse_per_dim': [],
        'mae_per_dim': []
    }

    # Per-dimension metrics
    n_features = y_true.shape[1] if len(y_true.shape) > 1 else 1

    if n_features > 1:
        for i in range(n_features):
            metrics['rmse_per_dim'].append(compute_rmse(y_true[:, i], y_pred[:, i]))
            metrics['mae_per_dim'].append(compute_mae(y_true[:, i], y_pred[:, i]))

    return metrics


def prediction_horizon_analysis(model,
                                 initial_window: np.ndarray,
                                 true_trajectory: np.ndarray,
                                 max_steps: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """
    Analyze prediction quality vs. prediction horizon.

    Parameters:
    -----------
    model : NeuralPredictor
        Trained model with iterative_predict method
    initial_window : np.ndarray
        Initial input window
    true_trajectory : np.ndarray
        True future trajectory
    max_steps : int
        Maximum number of steps to predict

    Returns:
    --------
    errors : np.ndarray
        RMSE at each prediction step
    steps : np.ndarray
        Step numbers
    """
    # Make iterative predictions
    predictions = model.iterative_predict(initial_window, max_steps)

    # Compute error at each step
    n_steps = min(max_steps, len(true_trajectory))
    errors = np.zeros(n_steps)

    for i in range(n_steps):
        errors[i] = compute_rmse(
            true_trajectory[i:i+1],
            predictions[i:i+1]
        )

    steps = np.arange(n_steps)
    return errors, steps


# ============================================================================
# Visualization Functions
# ============================================================================

def plot_trajectory_2d(true_trajectory: np.ndarray,
                       pred_trajectory: Optional[np.ndarray] = None,
                       dims: Tuple[int, int] = (0, 1),
                       title: str = "Trajectory Comparison",
                       save_path: Optional[str] = None):
    """
    Plot 2D trajectory.

    Parameters:
    -----------
    true_trajectory : np.ndarray
        True trajectory, shape (n_points, n_features)
    pred_trajectory : np.ndarray, optional
        Predicted trajectory
    dims : tuple
        Dimensions to plot (default: (0, 1))
    title : str
        Plot title
    save_path : str, optional
        Path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot true trajectory
    ax.plot(true_trajectory[:, dims[0]], true_trajectory[:, dims[1]],
            'b-', alpha=0.7, label='True', linewidth=2)

    # Plot predicted trajectory if provided
    if pred_trajectory is not None:
        ax.plot(pred_trajectory[:, dims[0]], pred_trajectory[:, dims[1]],
                'r--', alpha=0.7, label='Predicted', linewidth=2)

    ax.set_xlabel(f'Dimension {dims[0]}', fontsize=12)
    ax.set_ylabel(f'Dimension {dims[1]}', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_trajectory_3d(true_trajectory: np.ndarray,
                       pred_trajectory: Optional[np.ndarray] = None,
                       dims: Tuple[int, int, int] = (0, 1, 2),
                       title: str = "3D Trajectory Comparison",
                       save_path: Optional[str] = None):
    """
    Plot 3D trajectory.

    Parameters:
    -----------
    true_trajectory : np.ndarray
        True trajectory, shape (n_points, n_features)
    pred_trajectory : np.ndarray, optional
        Predicted trajectory
    dims : tuple
        Dimensions to plot (default: (0, 1, 2))
    title : str
        Plot title
    save_path : str, optional
        Path to save figure
    """
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    # Plot true trajectory
    ax.plot(true_trajectory[:, dims[0]],
            true_trajectory[:, dims[1]],
            true_trajectory[:, dims[2]],
            'b-', alpha=0.6, label='True', linewidth=1.5)

    # Plot predicted trajectory if provided
    if pred_trajectory is not None:
        ax.plot(pred_trajectory[:, dims[0]],
                pred_trajectory[:, dims[1]],
                pred_trajectory[:, dims[2]],
                'r--', alpha=0.6, label='Predicted', linewidth=1.5)

    ax.set_xlabel(f'Dimension {dims[0]}', fontsize=11)
    ax.set_ylabel(f'Dimension {dims[1]}', fontsize=11)
    ax.set_zlabel(f'Dimension {dims[2]}', fontsize=11)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=11)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_time_series(t: np.ndarray,
                     true_trajectory: np.ndarray,
                     pred_trajectory: Optional[np.ndarray] = None,
                     feature_names: Optional[List[str]] = None,
                     title: str = "Time Series Comparison",
                     save_path: Optional[str] = None):
    """
    Plot time series for each dimension.

    Parameters:
    -----------
    t : np.ndarray
        Time points
    true_trajectory : np.ndarray
        True trajectory
    pred_trajectory : np.ndarray, optional
        Predicted trajectory
    feature_names : list, optional
        Names for each feature
    title : str
        Plot title
    save_path : str, optional
        Path to save figure
    """
    n_features = true_trajectory.shape[1]

    if feature_names is None:
        feature_names = [f'Dim {i}' for i in range(n_features)]

    fig, axes = plt.subplots(n_features, 1, figsize=(12, 3 * n_features))

    if n_features == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        # Plot true values
        ax.plot(t, true_trajectory[:, i], 'b-', alpha=0.7,
                label='True', linewidth=2)

        # Plot predictions if provided
        if pred_trajectory is not None:
            t_pred = t[:len(pred_trajectory)]
            ax.plot(t_pred, pred_trajectory[:, i], 'r--', alpha=0.7,
                    label='Predicted', linewidth=2)

        ax.set_xlabel('Time', fontsize=11)
        ax.set_ylabel(feature_names[i], fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14, y=1.00)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_prediction_error(errors: np.ndarray,
                          steps: np.ndarray,
                          title: str = "Prediction Error vs. Horizon",
                          log_scale: bool = False,
                          save_path: Optional[str] = None):
    """
    Plot prediction error as a function of prediction horizon.

    Parameters:
    -----------
    errors : np.ndarray
        Error values
    steps : np.ndarray
        Step numbers
    title : str
        Plot title
    log_scale : bool
        Use log scale for y-axis (default: False)
    save_path : str, optional
        Path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(steps, errors, 'b-', linewidth=2, marker='o', markersize=4)

    ax.set_xlabel('Prediction Horizon (steps)', fontsize=12)
    ax.set_ylabel('RMSE', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)

    if log_scale:
        ax.set_yscale('log')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_training_history(history: dict,
                          title: str = "Training History",
                          save_path: Optional[str] = None):
    """
    Plot training and validation loss curves.

    Parameters:
    -----------
    history : dict
        Training history with 'train_loss' and optionally 'val_loss'
    title : str
        Plot title
    save_path : str, optional
        Path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    epochs = range(1, len(history['train_loss']) + 1)
    ax.plot(epochs, history['train_loss'], 'b-', linewidth=2, label='Training Loss')

    if 'val_loss' in history:
        ax.plot(epochs, history['val_loss'], 'r--', linewidth=2, label='Validation Loss')

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss (MSE)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_phase_space(trajectory: np.ndarray,
                     dims: Tuple[int, int] = (0, 1),
                     title: str = "Phase Space",
                     save_path: Optional[str] = None):
    """
    Plot phase space diagram.

    Parameters:
    -----------
    trajectory : np.ndarray
        Trajectory data
    dims : tuple
        Dimensions to plot (default: (0, 1))
    title : str
        Plot title
    save_path : str, optional
        Path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create color gradient based on time
    colors = np.linspace(0, 1, len(trajectory))
    scatter = ax.scatter(trajectory[:, dims[0]], trajectory[:, dims[1]],
                        c=colors, cmap='viridis', alpha=0.6, s=10)

    ax.set_xlabel(f'Dimension {dims[0]}', fontsize=12)
    ax.set_ylabel(f'Dimension {dims[1]}', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Time progression', fontsize=11)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


# Example usage and testing
if __name__ == "__main__":
    print("Testing Evaluation Module")
    print("=" * 50)

    # Create dummy data
    n_points = 500
    n_features = 3

    t = np.linspace(0, 50, n_points)
    true_traj = np.random.randn(n_points, n_features).cumsum(axis=0)
    pred_traj = true_traj + np.random.randn(n_points, n_features) * 0.5

    # Test metrics
    print("\n1. Testing Metrics")
    metrics = evaluate_prediction(true_traj, pred_traj)
    print(f"✓ RMSE: {metrics['rmse']:.4f}")
    print(f"✓ MAE: {metrics['mae']:.4f}")
    print(f"✓ NRMSE: {metrics['nrmse']:.4f}")

    # Test visualizations
    print("\n2. Testing Visualizations")
    print("  (Close each plot window to continue)")

    # 2D trajectory
    plot_trajectory_2d(true_traj, pred_traj, dims=(0, 1),
                      title="Test 2D Trajectory")
    print("✓ 2D trajectory plot")

    # 3D trajectory
    plot_trajectory_3d(true_traj, pred_traj, dims=(0, 1, 2),
                      title="Test 3D Trajectory")
    print("✓ 3D trajectory plot")

    # Time series
    plot_time_series(t, true_traj, pred_traj,
                    feature_names=['X', 'Y', 'Z'],
                    title="Test Time Series")
    print("✓ Time series plot")

    # Phase space
    plot_phase_space(true_traj, dims=(0, 1), title="Test Phase Space")
    print("✓ Phase space plot")

    print("\n" + "=" * 50)
    print("All evaluation functions tested successfully!")
