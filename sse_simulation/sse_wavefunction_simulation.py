"""
sse_wavefunction_simulation.py
===============================

This module implements a wavefunction-based simulator for the stochastic 
Schrödinger equation (SSE) with discrete measurements following the framework 
in EP_Production.md. 

The simulator evolves quantum states using the discrete SSE:
    |ψ_{i+1}⟩ = |ψ_i⟩ + |δψ⟩
where 
    |δψ⟩ = ξ_i ε (σ_z - ⟨σ_z⟩)/2 |ψ⟩ - (ε²/2) (σ_z - ⟨σ_z⟩)²/4 |ψ⟩

The entropy production Q is calculated using the exact discrete formula:
    Q = 2ε Σ r_i (z_{i-1} + z_i)/2  (Stratonovich discretization)

This implementation supports:
- Single trajectory and ensemble simulations
- Configurable initial states (default: Bloch equator)
- Hamiltonian coupling parameter J (default: J=0, no coupling)
- Direct comparison with theoretical predictions
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Optional


@dataclass
class SSEWavefunctionSimulator:
    """Wavefunction-based SSE simulator with discrete measurements.

    Parameters
    ----------
    epsilon : float
        Measurement strength parameter ε. Smaller values correspond
        to weaker measurements.
    N_steps : int
        Number of discrete measurement steps.
    J : float
        Hamiltonian coupling parameter (default: 0, no Hamiltonian evolution).
        When J≠0, adds Hamiltonian H = J σ_x to the evolution.
    initial_state : str
        Initial state specification:
        - 'bloch_equator': |+x⟩ = (|0⟩ + |1⟩)/√2 (default)
        - 'up': |0⟩ 
        - 'down': |1⟩
        - 'plus_y': |+y⟩ = (|0⟩ + i|1⟩)/√2
        - 'minus_y': |-y⟩ = (|0⟩ - i|1⟩)/√2
    theta : float
        For custom initial state: polar angle θ in Bloch sphere
    phi : float  
        For custom initial state: azimuthal angle φ in Bloch sphere
    rng : np.random.Generator | None
        Optional NumPy random number generator.
    """

    epsilon: float = 0.1
    N_steps: int = 100
    J: float = 0.0
    initial_state: str = 'bloch_equator'
    theta: float = np.pi/2  # For custom state
    phi: float = 0.0        # For custom state  
    rng: np.random.Generator | None = field(default=None, repr=False)
    
    def __post_init__(self) -> None:
        if self.rng is None:
            self.rng = np.random.default_rng()
        
        # Pre-compute Pauli matrices
        self.sigma_z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
        self.sigma_x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
        self.sigma_y = np.array([[0.0, -1j], [1j, 0.0]], dtype=complex)
        self.identity = np.eye(2, dtype=complex)
        
        # Set initial state
        self.psi_initial = self._prepare_initial_state()
        
    def _prepare_initial_state(self) -> np.ndarray:
        """Prepare the initial quantum state based on configuration."""
        if self.initial_state == 'bloch_equator':
            # |+x⟩ = (|0⟩ + |1⟩)/√2
            return np.array([1.0, 1.0], dtype=complex) / np.sqrt(2)
        elif self.initial_state == 'up':
            # |0⟩
            return np.array([1.0, 0.0], dtype=complex)
        elif self.initial_state == 'down':
            # |1⟩  
            return np.array([0.0, 1.0], dtype=complex)
        elif self.initial_state == 'plus_y':
            # |+y⟩ = (|0⟩ + i|1⟩)/√2
            return np.array([1.0, 1j], dtype=complex) / np.sqrt(2)
        elif self.initial_state == 'minus_y':
            # |-y⟩ = (|0⟩ - i|1⟩)/√2
            return np.array([1.0, -1j], dtype=complex) / np.sqrt(2)
        elif self.initial_state == 'custom':
            # General Bloch sphere state: cos(θ/2)|0⟩ + e^(iφ)sin(θ/2)|1⟩
            return np.array([
                np.cos(self.theta/2),
                np.exp(1j * self.phi) * np.sin(self.theta/2)
            ], dtype=complex)
        else:
            raise ValueError(f"Unknown initial state: {self.initial_state}")
    
    def _expectation_value_z(self, psi: np.ndarray) -> float:
        """Calculate ⟨ψ|σ_z|ψ⟩."""
        return np.real(np.conj(psi) @ self.sigma_z @ psi)
    
    def _apply_hamiltonian_evolution(self, psi: np.ndarray, dt: float) -> np.ndarray:
        """Apply Hamiltonian evolution for time step dt."""
        if abs(self.J) < 1e-15:  # No Hamiltonian evolution
            return psi
        
        # H = J σ_x, so U = exp(-i J σ_x dt)
        # For σ_x: exp(-i θ σ_x) = cos(θ)I - i sin(θ)σ_x  
        theta = self.J * dt
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        
        U = cos_theta * self.identity - 1j * sin_theta * self.sigma_x
        return U @ psi
    
    def _measurement_update(self, psi: np.ndarray) -> Tuple[np.ndarray, int, float]:
        """Apply single discrete measurement step.
        
        Returns:
            (new_psi, measurement_result, z_before)
        """
        # Calculate z expectation value before measurement
        z_before = self._expectation_value_z(psi)
        
        # Measurement probabilities  
        p_plus1 = 0.5 * (1.0 + self.epsilon * z_before)
        p_minus1 = 0.5 * (1.0 - self.epsilon * z_before)
        
        # Sample measurement outcome
        xi = 1 if self.rng.random() < p_plus1 else -1
        
        # Apply SSE update: |δψ⟩ = ξε(σ_z - z)/2|ψ⟩ - (ε²/2)(σ_z - z)²/4|ψ⟩
        sigma_z_minus_z = self.sigma_z - z_before * self.identity
        
        # First order term: ξε(σ_z - z)/2
        first_order = xi * self.epsilon * 0.5 * sigma_z_minus_z @ psi
        
        # Second order term: -(ε²/2)(σ_z - z)²/4  
        second_order_op = -0.5 * (self.epsilon**2) * 0.25 * (sigma_z_minus_z @ sigma_z_minus_z)
        second_order = second_order_op @ psi
        
        # Update wavefunction
        psi_new = psi + first_order + second_order
        
        # Normalize to maintain unit norm
        norm = np.linalg.norm(psi_new)
        if norm > 1e-15:
            psi_new = psi_new / norm
        else:
            # Fallback to prevent numerical issues
            psi_new = psi / np.linalg.norm(psi)
            
        return psi_new, xi, z_before
    
    def simulate_trajectory(self) -> Tuple[float, np.ndarray, np.ndarray]:
        """Simulate a single stochastic trajectory.
        
        Returns:
            (Q, z_trajectory, measurement_results)
        where:
            Q: Entropy production using discrete formula
            z_trajectory: Array of z expectation values at each step
            measurement_results: Array of measurement outcomes ξ_i = ±1
        """
        psi = self.psi_initial.copy()
        z_trajectory = np.zeros(self.N_steps + 1)
        measurement_results = np.zeros(self.N_steps, dtype=int)
        
        # Initial z value
        z_trajectory[0] = self._expectation_value_z(psi)
        
        # Time step for Hamiltonian evolution (if any)
        dt = 1.0 / self.N_steps  # Normalize total time to 1
        
        Q = 0.0  # Entropy production accumulator
        
        for i in range(self.N_steps):
            # Apply Hamiltonian evolution (if J ≠ 0)
            if abs(self.J) > 1e-15:
                psi = self._apply_hamiltonian_evolution(psi, dt)
            
            # Apply measurement
            psi, xi, z_before = self._measurement_update(psi)
            measurement_results[i] = xi
            
            # Calculate z after measurement
            z_after = self._expectation_value_z(psi)
            z_trajectory[i + 1] = z_after
            
            # Accumulate entropy production using discrete Stratonovich formula:
            # Q = 2ε Σ r_i (z_{i-1} + z_i)/2
            Q += 2.0 * self.epsilon * xi * (z_before + z_after) / 2.0
        
        return Q, z_trajectory, measurement_results
    
    def simulate_ensemble(self, n_trajectories: int, progress: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Simulate an ensemble of trajectories.
        
        Parameters:
        -----------
        n_trajectories : int
            Number of independent trajectories to simulate
        progress : bool
            Whether to use tqdm progress bar
            
        Returns:
        --------
        (Q_values, z_trajectories, measurement_results)
        where:
            Q_values: Array of entropy production values (n_trajectories,)
            z_trajectories: Array of z expectation trajectories (n_trajectories, N_steps+1)
            measurement_results: Array of measurement outcomes (n_trajectories, N_steps)
        """
        Q_values = np.zeros(n_trajectories)
        z_trajectories = np.zeros((n_trajectories, self.N_steps + 1))
        measurement_results = np.zeros((n_trajectories, self.N_steps), dtype=int)
        
        # Set up iterator with optional tqdm progress bar
        if progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(range(n_trajectories), desc="Simulating trajectories")
            except ImportError:
                # Fallback if tqdm not available
                iterator = range(n_trajectories)
        else:
            iterator = range(n_trajectories)
        
        for i in iterator:
            Q, z_traj, meas_results = self.simulate_trajectory()
            Q_values[i] = Q
            z_trajectories[i] = z_traj
            measurement_results[i] = meas_results
            
        return Q_values, z_trajectories, measurement_results
    
    def theoretical_mean_variance(self) -> Tuple[float, float]:
        """Calculate theoretical mean and variance of Q for comparison.
        
        Based on the connection to Dressel et al.: T/τ = N ε²
        For initial state on Bloch equator (z_i = 0), we expect:
        ⟨Q⟩ ≈ (3/2) * N * ε²  
        Var(Q) ≈ 2 * N * ε²
        
        Returns:
        --------
        (theoretical_mean, theoretical_variance)
        """
        # For z_i = 0 (Bloch equator initial state)
        if self.initial_state == 'bloch_equator':
            T_over_tau = self.N_steps * (self.epsilon**2)
            theoretical_mean = 1.5 * T_over_tau  # (3/2) * T/τ
            theoretical_variance = 2.0 * T_over_tau  # 2 * T/τ
        else:
            # For general initial states, the formulas are more complex
            # This is a simplified approximation
            T_over_tau = self.N_steps * (self.epsilon**2)
            theoretical_mean = 1.5 * T_over_tau
            theoretical_variance = 2.0 * T_over_tau
            
        return theoretical_mean, theoretical_variance


def compute_histogram(data: np.ndarray, bins: int = 50, density: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Compute histogram of data.
    
    Parameters:
    -----------
    data : np.ndarray
        Input samples
    bins : int
        Number of histogram bins  
    density : bool
        If True, normalize to form probability density
        
    Returns:
    --------
    (bin_centers, counts)
    """
    counts, edges = np.histogram(data, bins=bins, density=density)
    bin_centers = 0.5 * (edges[:-1] + edges[1:])
    return bin_centers, counts


if __name__ == "__main__":
    # Example usage and validation
    print("SSE Wavefunction Simulator - Validation Test")
    print("=" * 50)
    
    # Test with different parameters
    sim = SSEWavefunctionSimulator(
        epsilon=0.1,
        N_steps=100, 
        J=0.0,
        initial_state='bloch_equator'
    )
    
    print(f"Parameters: ε={sim.epsilon}, N={sim.N_steps}, J={sim.J}")
    print(f"Initial state: {sim.initial_state}")
    
    # Single trajectory test
    Q, z_traj, meas_results = sim.simulate_trajectory()
    print(f"\nSingle trajectory:")
    print(f"Q = {Q:.3f}")
    print(f"Initial z = {z_traj[0]:.3f}")
    print(f"Final z = {z_traj[-1]:.3f}")
    
    # Ensemble test
    n_traj = 1000
    print(f"\nRunning ensemble simulation with {n_traj} trajectories...")
    Q_values, _, _ = sim.simulate_ensemble(n_traj, progress=True)
    
    mean_Q = np.mean(Q_values)
    var_Q = np.var(Q_values)
    theoretical_mean, theoretical_var = sim.theoretical_mean_variance()
    
    print(f"\nEnsemble results (n={n_traj}):")
    print(f"Observed: ⟨Q⟩ = {mean_Q:.3f}, Var(Q) = {var_Q:.3f}")
    print(f"Theory:   ⟨Q⟩ = {theoretical_mean:.3f}, Var(Q) = {theoretical_var:.3f}")
    print(f"Ratio:    ⟨Q⟩ = {mean_Q/theoretical_mean:.3f}, Var(Q) = {var_Q/theoretical_var:.3f}")