"""
Neural Network Predictive Models for Nonlinear Dynamical Systems

This package provides tools for learning and predicting trajectories
of nonlinear dynamical systems using neural networks.
"""

__version__ = "1.0.0"
__authors__ = [
    "Vlad-Flavius Misăilă",
    "Robert-Daniel Man",
    "Sebastian-Adrian Mărginean"
]

from .dynamical_systems import (
    DynamicalSystem,
    LorenzSystem,
    RosslerSystem,
    VanDerPolOscillator,
    DuffingOscillator
)

from .data_preparation import (
    generate_trajectory,
    create_sliding_windows,
    normalize_data,
    train_test_split_sequential,
    create_sequences,
    add_noise,
    multi_step_prediction_data
)

from .neural_models import (
    FeedForwardPredictor,
    LSTMPredictor,
    GRUPredictor,
    NeuralPredictor
)

from .evaluation import (
    compute_rmse,
    compute_mae,
    compute_normalized_rmse,
    evaluate_prediction,
    prediction_horizon_analysis,
    plot_trajectory_2d,
    plot_trajectory_3d,
    plot_time_series,
    plot_prediction_error,
    plot_training_history,
    plot_phase_space
)

__all__ = [
    # Dynamical Systems
    'DynamicalSystem',
    'LorenzSystem',
    'RosslerSystem',
    'VanDerPolOscillator',
    'DuffingOscillator',

    # Data Preparation
    'generate_trajectory',
    'create_sliding_windows',
    'normalize_data',
    'train_test_split_sequential',
    'create_sequences',
    'add_noise',
    'multi_step_prediction_data',

    # Neural Models
    'FeedForwardPredictor',
    'LSTMPredictor',
    'GRUPredictor',
    'NeuralPredictor',

    # Evaluation
    'compute_rmse',
    'compute_mae',
    'compute_normalized_rmse',
    'evaluate_prediction',
    'prediction_horizon_analysis',
    'plot_trajectory_2d',
    'plot_trajectory_3d',
    'plot_time_series',
    'plot_prediction_error',
    'plot_training_history',
    'plot_phase_space'
]
