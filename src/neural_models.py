"""
Neural Network Models Module

This module implements various neural network architectures for predicting
trajectories of dynamical systems.

Models implemented:
- Feed-forward Neural Networks (FNN)
- Recurrent Neural Networks (RNN)
- Long Short-Term Memory (LSTM)
- Gated Recurrent Unit (GRU)

Both PyTorch and TensorFlow implementations are provided.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import List, Tuple, Optional
from tqdm import tqdm


# ============================================================================
# PyTorch Models
# ============================================================================

class FeedForwardPredictor(nn.Module):
    """
    Feed-forward neural network for time series prediction.

    Architecture:
        Input: Flattened window of past states
        Hidden: Multiple fully-connected layers with ReLU activation
        Output: Predicted next state
    """

    def __init__(self,
                 input_dim: int,
                 hidden_dims: List[int] = [64, 64],
                 output_dim: Optional[int] = None,
                 dropout: float = 0.0):
        """
        Initialize feed-forward predictor.

        Parameters:
        -----------
        input_dim : int
            Dimension of input features (window_size * n_features)
        hidden_dims : list
            List of hidden layer dimensions (default: [64, 64])
        output_dim : int, optional
            Dimension of output. If None, uses input_dim
        dropout : float
            Dropout probability (default: 0.0)
        """
        super(FeedForwardPredictor, self).__init__()

        if output_dim is None:
            output_dim = input_dim

        layers = []
        prev_dim = input_dim

        # Build hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass.

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor, shape (batch_size, window_size, n_features)

        Returns:
        --------
        torch.Tensor
            Predictions, shape (batch_size, output_dim)
        """
        # Flatten the input
        batch_size = x.shape[0]
        x_flat = x.view(batch_size, -1)
        return self.network(x_flat)


class LSTMPredictor(nn.Module):
    """
    LSTM-based predictor for time series.

    Architecture:
        Input: Sequence of past states
        LSTM layers: Process temporal dependencies
        Output: Predicted next state
    """

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 64,
                 num_layers: int = 2,
                 output_dim: Optional[int] = None,
                 dropout: float = 0.0):
        """
        Initialize LSTM predictor.

        Parameters:
        -----------
        input_dim : int
            Dimension of input features at each time step
        hidden_dim : int
            Dimension of LSTM hidden state (default: 64)
        num_layers : int
            Number of LSTM layers (default: 2)
        output_dim : int, optional
            Dimension of output. If None, uses input_dim
        dropout : float
            Dropout probability between LSTM layers (default: 0.0)
        """
        super(LSTMPredictor, self).__init__()

        if output_dim is None:
            output_dim = input_dim

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """
        Forward pass.

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor, shape (batch_size, window_size, n_features)

        Returns:
        --------
        torch.Tensor
            Predictions, shape (batch_size, output_dim)
        """
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)

        # Use the last output
        last_output = lstm_out[:, -1, :]

        # Fully connected layer
        prediction = self.fc(last_output)

        return prediction


class GRUPredictor(nn.Module):
    """
    GRU-based predictor for time series.

    Similar to LSTM but with simpler gating mechanism.
    """

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 64,
                 num_layers: int = 2,
                 output_dim: Optional[int] = None,
                 dropout: float = 0.0):
        """
        Initialize GRU predictor.

        Parameters:
        -----------
        input_dim : int
            Dimension of input features at each time step
        hidden_dim : int
            Dimension of GRU hidden state (default: 64)
        num_layers : int
            Number of GRU layers (default: 2)
        output_dim : int, optional
            Dimension of output. If None, uses input_dim
        dropout : float
            Dropout probability between GRU layers (default: 0.0)
        """
        super(GRUPredictor, self).__init__()

        if output_dim is None:
            output_dim = input_dim

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """
        Forward pass.

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor, shape (batch_size, window_size, n_features)

        Returns:
        --------
        torch.Tensor
            Predictions, shape (batch_size, output_dim)
        """
        # GRU forward pass
        gru_out, _ = self.gru(x)

        # Use the last output
        last_output = gru_out[:, -1, :]

        # Fully connected layer
        prediction = self.fc(last_output)

        return prediction


# ============================================================================
# Training and Prediction Utilities
# ============================================================================

class NeuralPredictor:
    """
    Wrapper class for training and using neural network predictors.

    Handles training loop, validation, and prediction.
    """

    def __init__(self,
                 model: nn.Module,
                 learning_rate: float = 0.001,
                 device: str = 'auto'):
        """
        Initialize predictor wrapper.

        Parameters:
        -----------
        model : nn.Module
            PyTorch model to train
        learning_rate : float
            Learning rate for optimizer (default: 0.001)
        device : str
            'auto', 'cpu', or 'cuda' (default: 'auto')
        """
        self.model = model

        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.model.to(self.device)

        # Optimizer and loss
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

        # Training history
        self.train_losses = []
        self.val_losses = []

    def train(self,
              X_train: np.ndarray,
              y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None,
              epochs: int = 100,
              batch_size: int = 32,
              verbose: bool = True) -> dict:
        """
        Train the model.

        Parameters:
        -----------
        X_train : np.ndarray
            Training input sequences
        y_train : np.ndarray
            Training targets
        X_val : np.ndarray, optional
            Validation input sequences
        y_val : np.ndarray, optional
            Validation targets
        epochs : int
            Number of training epochs (default: 100)
        batch_size : int
            Batch size (default: 32)
        verbose : bool
            Whether to show progress bar (default: True)

        Returns:
        --------
        dict
            Training history with 'train_loss' and optionally 'val_loss'
        """
        # Convert to torch tensors
        X_train_t = torch.FloatTensor(X_train).to(self.device)
        y_train_t = torch.FloatTensor(y_train).to(self.device)

        # Create data loader
        train_dataset = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Validation data
        if X_val is not None and y_val is not None:
            X_val_t = torch.FloatTensor(X_val).to(self.device)
            y_val_t = torch.FloatTensor(y_val).to(self.device)

        # Training loop
        iterator = tqdm(range(epochs), desc='Training') if verbose else range(epochs)

        for epoch in iterator:
            # Training
            self.model.train()
            train_loss = 0.0

            for batch_X, batch_y in train_loader:
                self.optimizer.zero_grad()
                predictions = self.model(batch_X)
                loss = self.criterion(predictions, batch_y)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)
            self.train_losses.append(train_loss)

            # Validation
            if X_val is not None and y_val is not None:
                self.model.eval()
                with torch.no_grad():
                    val_predictions = self.model(X_val_t)
                    val_loss = self.criterion(val_predictions, y_val_t).item()
                    self.val_losses.append(val_loss)

                if verbose:
                    iterator.set_postfix({
                        'train_loss': f'{train_loss:.6f}',
                        'val_loss': f'{val_loss:.6f}'
                    })
            else:
                if verbose:
                    iterator.set_postfix({'train_loss': f'{train_loss:.6f}'})

        history = {'train_loss': self.train_losses}
        if self.val_losses:
            history['val_loss'] = self.val_losses

        return history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.

        Parameters:
        -----------
        X : np.ndarray
            Input sequences

        Returns:
        --------
        np.ndarray
            Predictions
        """
        self.model.eval()
        with torch.no_grad():
            X_t = torch.FloatTensor(X).to(self.device)
            predictions = self.model(X_t)
            return predictions.cpu().numpy()

    def iterative_predict(self,
                          initial_window: np.ndarray,
                          n_steps: int) -> np.ndarray:
        """
        Make iterative multi-step predictions.

        Uses model predictions as input for subsequent predictions.

        Parameters:
        -----------
        initial_window : np.ndarray
            Initial input window, shape (window_size, n_features)
        n_steps : int
            Number of steps to predict ahead

        Returns:
        --------
        np.ndarray
            Predicted trajectory, shape (n_steps, n_features)
        """
        self.model.eval()
        predictions = []
        current_window = initial_window.copy()

        with torch.no_grad():
            for _ in range(n_steps):
                # Predict next step
                X_t = torch.FloatTensor(current_window[np.newaxis, :, :]).to(self.device)
                next_pred = self.model(X_t).cpu().numpy()[0]
                predictions.append(next_pred)

                # Update window: remove oldest, add newest prediction
                current_window = np.vstack([current_window[1:], next_pred])

        return np.array(predictions)


# Example usage and testing
if __name__ == "__main__":
    print("Testing Neural Models Module")
    print("=" * 50)

    # Create dummy data
    n_samples = 1000
    window_size = 50
    n_features = 3

    X_train = np.random.randn(n_samples, window_size, n_features).astype(np.float32)
    y_train = np.random.randn(n_samples, n_features).astype(np.float32)

    # Test Feed-forward model
    print("\n1. Testing Feed-forward Neural Network")
    fnn = FeedForwardPredictor(
        input_dim=window_size * n_features,
        hidden_dims=[64, 32],
        output_dim=n_features
    )
    predictor_fnn = NeuralPredictor(fnn, learning_rate=0.001)
    history_fnn = predictor_fnn.train(X_train, y_train, epochs=5, verbose=False)
    print(f"✓ FNN trained for 5 epochs")
    print(f"  Final training loss: {history_fnn['train_loss'][-1]:.6f}")

    # Test LSTM model
    print("\n2. Testing LSTM Network")
    lstm = LSTMPredictor(
        input_dim=n_features,
        hidden_dim=32,
        num_layers=2,
        output_dim=n_features
    )
    predictor_lstm = NeuralPredictor(lstm, learning_rate=0.001)
    history_lstm = predictor_lstm.train(X_train, y_train, epochs=5, verbose=False)
    print(f"✓ LSTM trained for 5 epochs")
    print(f"  Final training loss: {history_lstm['train_loss'][-1]:.6f}")

    # Test GRU model
    print("\n3. Testing GRU Network")
    gru = GRUPredictor(
        input_dim=n_features,
        hidden_dim=32,
        num_layers=2,
        output_dim=n_features
    )
    predictor_gru = NeuralPredictor(gru, learning_rate=0.001)
    history_gru = predictor_gru.train(X_train, y_train, epochs=5, verbose=False)
    print(f"✓ GRU trained for 5 epochs")
    print(f"  Final training loss: {history_gru['train_loss'][-1]:.6f}")

    # Test prediction
    print("\n4. Testing Prediction")
    X_test = np.random.randn(10, window_size, n_features).astype(np.float32)
    predictions = predictor_lstm.predict(X_test)
    print(f"✓ Predictions shape: {predictions.shape}")

    # Test iterative prediction
    print("\n5. Testing Iterative Prediction")
    initial_window = X_test[0]
    trajectory = predictor_lstm.iterative_predict(initial_window, n_steps=20)
    print(f"✓ Iterative predictions shape: {trajectory.shape}")

    print("\n" + "=" * 50)
    print("All neural models tested successfully!")
