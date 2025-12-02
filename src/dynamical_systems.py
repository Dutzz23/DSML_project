"""
Dynamical Systems Module

This module implements various nonlinear dynamical systems including
chaotic and oscillatory systems. Each system is implemented as a class
with methods to compute derivatives for numerical integration.

Systems implemented:
- Lorenz System (chaotic)
- Rössler System (chaotic)
- Van der Pol Oscillator (oscillatory)
- Duffing Oscillator (oscillatory)
"""

import numpy as np
from typing import Tuple, Callable
from scipy.integrate import solve_ivp


class DynamicalSystem:
    """
    Base class for dynamical systems.

    All systems should inherit from this class and implement the
    derivatives() method which computes the time derivatives of state variables.
    """

    def __init__(self, name: str):
        """
        Initialize a dynamical system.

        Parameters:
        -----------
        name : str
            Name of the dynamical system
        """
        self.name = name

    def derivatives(self, t: float, state: np.ndarray) -> np.ndarray:
        """
        Compute time derivatives of state variables.

        Parameters:
        -----------
        t : float
            Current time
        state : np.ndarray
            Current state vector

        Returns:
        --------
        np.ndarray
            Time derivatives of state variables
        """
        raise NotImplementedError("Subclasses must implement derivatives()")

    def integrate(self,
                  initial_state: np.ndarray,
                  t_span: Tuple[float, float],
                  dt: float = 0.01,
                  method: str = 'RK45') -> Tuple[np.ndarray, np.ndarray]:
        """
        Integrate the system using scipy's solve_ivp.

        Parameters:
        -----------
        initial_state : np.ndarray
            Initial conditions
        t_span : tuple
            (t_start, t_end) time span for integration
        dt : float
            Time step for output
        method : str
            Integration method (default: 'RK45' - 4th order Runge-Kutta)

        Returns:
        --------
        t : np.ndarray
            Time points
        trajectory : np.ndarray
            State trajectory, shape (n_points, n_dimensions)
        """
        t_eval = np.arange(t_span[0], t_span[1], dt)
        solution = solve_ivp(
            self.derivatives,
            t_span,
            initial_state,
            method=method,
            t_eval=t_eval,
            dense_output=False
        )
        return solution.t, solution.y.T


class LorenzSystem(DynamicalSystem):
    """
    Lorenz System - a chaotic system exhibiting sensitive dependence on initial conditions.

    Equations:
        dx/dt = σ(y - x)
        dy/dt = x(ρ - z) - y
        dz/dt = xy - βz

    The classic Lorenz attractor exhibits chaotic behavior for standard parameter values.
    """

    def __init__(self, sigma: float = 10.0, rho: float = 28.0, beta: float = 8.0/3.0):
        """
        Initialize Lorenz system.

        Parameters:
        -----------
        sigma : float
            Prandtl number (default: 10.0)
        rho : float
            Rayleigh number (default: 28.0)
        beta : float
            Geometric factor (default: 8/3)
        """
        super().__init__("Lorenz System")
        self.sigma = sigma
        self.rho = rho
        self.beta = beta

    def derivatives(self, t: float, state: np.ndarray) -> np.ndarray:
        """
        Compute Lorenz system derivatives.

        Parameters:
        -----------
        t : float
            Current time (not used, system is autonomous)
        state : np.ndarray
            Current state [x, y, z]

        Returns:
        --------
        np.ndarray
            Derivatives [dx/dt, dy/dt, dz/dt]
        """
        x, y, z = state
        dx = self.sigma * (y - x)
        dy = x * (self.rho - z) - y
        dz = x * y - self.beta * z
        return np.array([dx, dy, dz])


class RosslerSystem(DynamicalSystem):
    """
    Rössler System - a simpler chaotic attractor with a single continuous band.

    Equations:
        dx/dt = -y - z
        dy/dt = x + ay
        dz/dt = b + z(x - c)

    Exhibits chaotic behavior for standard parameter values.
    """

    def __init__(self, a: float = 0.2, b: float = 0.2, c: float = 5.7):
        """
        Initialize Rössler system.

        Parameters:
        -----------
        a : float
            Parameter a (default: 0.2)
        b : float
            Parameter b (default: 0.2)
        c : float
            Parameter c (default: 5.7)
        """
        super().__init__("Rössler System")
        self.a = a
        self.b = b
        self.c = c

    def derivatives(self, t: float, state: np.ndarray) -> np.ndarray:
        """
        Compute Rössler system derivatives.

        Parameters:
        -----------
        t : float
            Current time (not used, system is autonomous)
        state : np.ndarray
            Current state [x, y, z]

        Returns:
        --------
        np.ndarray
            Derivatives [dx/dt, dy/dt, dz/dt]
        """
        x, y, z = state
        dx = -y - z
        dy = x + self.a * y
        dz = self.b + z * (x - self.c)
        return np.array([dx, dy, dz])


class VanDerPolOscillator(DynamicalSystem):
    """
    Van der Pol Oscillator - a nonlinear oscillator with limit cycles.

    Equations (2nd order ODE converted to 1st order system):
        dx/dt = y
        dy/dt = μ(1 - x²)y - x

    Exhibits relaxation oscillations for large μ.
    """

    def __init__(self, mu: float = 1.0):
        """
        Initialize Van der Pol oscillator.

        Parameters:
        -----------
        mu : float
            Damping parameter (default: 1.0)
            μ > 0: self-excited oscillations
            μ = 0: simple harmonic oscillator
        """
        super().__init__("Van der Pol Oscillator")
        self.mu = mu

    def derivatives(self, t: float, state: np.ndarray) -> np.ndarray:
        """
        Compute Van der Pol oscillator derivatives.

        Parameters:
        -----------
        t : float
            Current time (not used, system is autonomous)
        state : np.ndarray
            Current state [x, y] where y = dx/dt

        Returns:
        --------
        np.ndarray
            Derivatives [dx/dt, dy/dt]
        """
        x, y = state
        dx = y
        dy = self.mu * (1 - x**2) * y - x
        return np.array([dx, dy])


class DuffingOscillator(DynamicalSystem):
    """
    Duffing Oscillator - a forced oscillator exhibiting periodic and chaotic behavior.

    Equations (2nd order ODE converted to 1st order system):
        dx/dt = y
        dy/dt = -δy - αx - βx³ + γ*cos(ωt)

    Can exhibit chaotic behavior depending on parameters and forcing.
    """

    def __init__(self,
                 alpha: float = -1.0,
                 beta: float = 1.0,
                 delta: float = 0.3,
                 gamma: float = 0.5,
                 omega: float = 1.2):
        """
        Initialize Duffing oscillator.

        Parameters:
        -----------
        alpha : float
            Linear stiffness (default: -1.0, negative for double-well potential)
        beta : float
            Nonlinear stiffness (default: 1.0)
        delta : float
            Damping coefficient (default: 0.3)
        gamma : float
            Forcing amplitude (default: 0.5)
        omega : float
            Forcing frequency (default: 1.2)
        """
        super().__init__("Duffing Oscillator")
        self.alpha = alpha
        self.beta = beta
        self.delta = delta
        self.gamma = gamma
        self.omega = omega

    def derivatives(self, t: float, state: np.ndarray) -> np.ndarray:
        """
        Compute Duffing oscillator derivatives.

        Parameters:
        -----------
        t : float
            Current time (system is non-autonomous due to forcing)
        state : np.ndarray
            Current state [x, y] where y = dx/dt

        Returns:
        --------
        np.ndarray
            Derivatives [dx/dt, dy/dt]
        """
        x, y = state
        dx = y
        dy = (-self.delta * y - self.alpha * x - self.beta * x**3 +
              self.gamma * np.cos(self.omega * t))
        return np.array([dx, dy])


# Example usage and testing
if __name__ == "__main__":
    print("Testing Dynamical Systems Module")
    print("=" * 50)

    # Test Lorenz system
    lorenz = LorenzSystem()
    t, traj = lorenz.integrate(
        initial_state=np.array([1.0, 1.0, 1.0]),
        t_span=(0, 50),
        dt=0.01
    )
    print(f"✓ {lorenz.name}: Generated {len(t)} time points")
    print(f"  Trajectory shape: {traj.shape}")

    # Test Rössler system
    rossler = RosslerSystem()
    t, traj = rossler.integrate(
        initial_state=np.array([1.0, 1.0, 1.0]),
        t_span=(0, 100),
        dt=0.01
    )
    print(f"✓ {rossler.name}: Generated {len(t)} time points")
    print(f"  Trajectory shape: {traj.shape}")

    # Test Van der Pol oscillator
    vdp = VanDerPolOscillator(mu=2.0)
    t, traj = vdp.integrate(
        initial_state=np.array([1.0, 0.0]),
        t_span=(0, 50),
        dt=0.01
    )
    print(f"✓ {vdp.name}: Generated {len(t)} time points")
    print(f"  Trajectory shape: {traj.shape}")

    # Test Duffing oscillator
    duffing = DuffingOscillator()
    t, traj = duffing.integrate(
        initial_state=np.array([0.1, 0.1]),
        t_span=(0, 100),
        dt=0.01
    )
    print(f"✓ {duffing.name}: Generated {len(t)} time points")
    print(f"  Trajectory shape: {traj.shape}")

    print("\n" + "=" * 50)
    print("All systems tested successfully!")
