"""
multi_qubit_correlation_simulator.py
=====================================

Generalized L-qubit simulation using correlation matrix approach with:
1. Hamiltonian evolution: H = J Σᵢ σᵢˣ σᵢ₊₁ˣ (nearest-neighbor XX coupling)
2. Stochastic measurement: independent measurements on each qubit in σᶻ basis
3. Entropy production calculation following EP_Production.md

The correlation matrix G evolves according to:
    dG = -2i dt [G, h] + ε(G ξ̂ + ξ̂ G - 2G ξ̂ G) - ε²(G - G^diag)

where:
- h is the single-particle Hamiltonian in Nambu (Bogoliubov) space
- ξ̂ = diag(ξ₁, ..., ξₗ, -ξ₁, ..., -ξₗ) with ξᵢ = ±1 random
- G is a (2L)×(2L) correlation matrix for L qubits

Initial state: |↑↑...↑⟩ corresponds to G = diag(1, 1, ..., 1, 0, 0, ..., 0)
"""

from __future__ import annotations
import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class MultiQubitCorrelationSimulator:
    """
    Correlation matrix simulator for L qubits with Hamiltonian + measurements.
    
    Parameters
    ----------
    L : int
        Number of qubits in the chain
    J : float
        Coupling constant for H = J Σᵢ σᵢˣ σᵢ₊₁ˣ
    epsilon : float
        Measurement strength parameter
    N_steps : int
        Number of time steps
    T : float
        Total evolution time (used to compute dt = T / N_steps)
    periodic : bool
        If True, use periodic boundary conditions (σₗˣ σ₁ˣ term)
        If False, use open boundary conditions
    rng : np.random.Generator | None
        Random number generator
    """
    L: int = 2
    J: float = 1.0
    epsilon: float = 0.1
    N_steps: int = 1000
    T: float = 1.0
    periodic: bool = False
    rng: np.random.Generator | None = field(default=None, repr=False)
    
    def __post_init__(self):
        if self.L < 2:
            raise ValueError("L must be at least 2")
        
        if self.rng is None:
            self.rng = np.random.default_rng()
        self.dt = self.T / self.N_steps
        
        # Build the Hamiltonian matrix h in Nambu space (2L × 2L)
        self.h = self._build_hamiltonian()
        
        # Initial correlation matrix for |↑↑...↑⟩ state
        # In fermion language: all sites occupied
        # G = diag(1, 1, ..., 1, 0, 0, ..., 0) with L ones and L zeros
        self.G_initial = np.diag(
            [1.0] * self.L + [0.0] * self.L
        ).astype(complex)
    
    def _build_hamiltonian(self) -> np.ndarray:
        """
        Build the (2L)×(2L) Hamiltonian matrix h for H = J Σᵢ σᵢˣ σᵢ₊₁ˣ.
        
        The Hamiltonian matrix has block structure:
        h = [[h₁₁, h₁₂],
             [h₂₁, h₂₂]]
        
        Each block is L×L. For nearest-neighbor XX coupling:
        - h₁₁ and h₁₂ have -J on the off-diagonals
        - h₂₁ and h₂₂ have +J on the off-diagonals
        
        From the Jordan-Wigner transformation:
        σᵢˣ σᵢ₊₁ˣ → (cᵢ† + cᵢ) Kᵢ (cᵢ₊₁† + cᵢ₊₁)
        where Kᵢ = ∏ⱼ₌₁ⁱ (2nⱼ - 1)
        
        After working through the algebra, for nearest neighbors:
        h₁₁[i, i+1] = h₁₂[i, i+1] = -J
        h₂₁[i, i+1] = h₂₂[i, i+1] = +J
        (and transpose for [i+1, i])
        """
        # Initialize blocks
        h11 = np.zeros((self.L, self.L), dtype=complex)
        h12 = np.zeros((self.L, self.L), dtype=complex)
        h21 = np.zeros((self.L, self.L), dtype=complex)
        h22 = np.zeros((self.L, self.L), dtype=complex)
        
        # Fill nearest-neighbor terms
        for i in range(self.L - 1):
            h11[i, i+1] = -self.J
            h11[i+1, i] = -self.J
            
            h12[i, i+1] = -self.J
            h12[i+1, i] = -self.J
            
            h21[i, i+1] = +self.J
            h21[i+1, i] = +self.J
            
            h22[i, i+1] = +self.J
            h22[i+1, i] = +self.J
        
        # Periodic boundary conditions
        if self.periodic:
            h11[0, self.L-1] = -self.J
            h11[self.L-1, 0] = -self.J
            
            h12[0, self.L-1] = -self.J
            h12[self.L-1, 0] = -self.J
            
            h21[0, self.L-1] = +self.J
            h21[self.L-1, 0] = +self.J
            
            h22[0, self.L-1] = +self.J
            h22[self.L-1, 0] = +self.J
        
        # Assemble block matrix
        top = np.hstack([h11, h12])
        bottom = np.hstack([h21, h22])
        h = np.vstack([top, bottom])
        return h
    
    def _compute_z_values(self, G: np.ndarray) -> np.ndarray:
        """
        Compute ⟨σᵢᶻ⟩ for all qubits from correlation matrix.
        
        From Jordan-Wigner: σᵢᶻ = 2nᵢ - 1 = 2c†ᵢcᵢ - 1
        So ⟨σᵢᶻ⟩ = 2⟨c†ᵢcᵢ⟩ - 1 = 2G[i, i] - 1
        
        Returns:
            z_values: array of shape (L,) with ⟨σᵢᶻ⟩ for i = 1, ..., L
        """
        z_values = np.zeros(self.L)
        for i in range(self.L):
            z_values[i] = 2.0 * np.real(G[i, i]) - 1.0
        return z_values
    
    def _hamiltonian_step(self, G: np.ndarray) -> np.ndarray:
        """
        Apply Hamiltonian evolution: dG = -2i dt [G, h]
        """
        commutator = G @ self.h - self.h @ G
        dG = -2.0j * self.dt * commutator
        return G + dG
    
    def _measurement_step(self, G: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply measurement step with random outcomes.
        
        dG = ε(G ξ̂ + ξ̂ G - 2G ξ̂ G) - ε²(G - G^diag)
        
        Returns:
            (G_new, xi_array) where xi_array = [ξ₁, ..., ξₗ]
        """
        # Sample random measurement outcomes for all qubits
        xi_array = np.array([
            1 if self.rng.random() < 0.5 else -1 
            for _ in range(self.L)
        ])
        
        # Build ξ̂ = diag(ξ₁, ..., ξₗ, -ξ₁, ..., -ξₗ)
        xi_hat = np.diag(
            list(xi_array) + list(-xi_array)
        ).astype(complex)
        
        # Measurement evolution (stochastic term)
        stochastic = self.epsilon * (G @ xi_hat + xi_hat @ G - 2.0 * G @ xi_hat @ G)
        
        # Damping term (deterministic)
        G_diag = np.diag(np.diag(G))
        damping = -self.epsilon**2 * (G - G_diag)
        
        dG = stochastic + damping
        G_new = G + dG
        
        # Ensure Hermiticity (numerical stability)
        G_new = 0.5 * (G_new + G_new.conj().T)
        
        # Clip diagonal elements to physical range [0, 1]
        # The diagonal of G represents occupation probabilities
        for i in range(len(G_new)):
            G_new[i, i] = np.clip(np.real(G_new[i, i]), 0.0, 1.0) + 0.0j
        
        return G_new, xi_array
    
    def simulate_trajectory(self) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Simulate a single trajectory.
        
        Returns:
            (Q, z_trajectory, xi_trajectory)
        where:
            Q: entropy production
            z_trajectory: array of shape (N_steps+1, L) with zᵢ at each step
            xi_trajectory: array of shape (N_steps, L) with ξᵢ at each step
        """
        G = self.G_initial.copy()
        z_trajectory = np.zeros((self.N_steps + 1, self.L))
        xi_trajectory = np.zeros((self.N_steps, self.L), dtype=int)
        
        # Initial z values
        z_trajectory[0] = self._compute_z_values(G)
        
        Q = 0.0
        
        for i in range(self.N_steps):
            # Store z before this step
            z_before = z_trajectory[i]
            
            # Apply Hamiltonian evolution
            G = self._hamiltonian_step(G)
            
            # Apply measurement
            G, xi_array = self._measurement_step(G)
            xi_trajectory[i] = xi_array
            
            # Compute z after this step
            z_after = self._compute_z_values(G)
            z_trajectory[i + 1] = z_after
            
            # Accumulate entropy production (Stratonovich discretization)
            # Q += Σₖ Σᵢ [2ε² z_{i-1,k} (z_{i-1,k}+z_{i,k})/2 + 2ε ξ_{i,k} (z_{i-1,k}+z_{i,k})/2]
            for k in range(self.L):
                z_before_k = z_before[k]
                z_after_k = z_after[k]
                xi_k = xi_array[k]
                
                # First term: 2ε² z_{i-1,k} (z_{i-1,k}+z_{i,k})/2
                term1 = 2.0 * self.epsilon**2 * z_before_k * 0.5 * (z_before_k + z_after_k)
                
                # Second term: 2ε ξ_{i,k} (z_{i-1,k}+z_{i,k})/2
                term2 = 2.0 * self.epsilon * xi_k * 0.5 * (z_before_k + z_after_k)
                
                Q += term1 + term2
        
        return Q, z_trajectory, xi_trajectory
    
    def simulate_ensemble(
        self, 
        n_trajectories: int,
        progress: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Simulate an ensemble of trajectories.
        
        Parameters
        ----------
        n_trajectories : int
            Number of independent trajectories
        progress : bool
            Show progress bar (requires tqdm)
            
        Returns
        -------
        (Q_values, z_trajectories, xi_trajectories)
        where:
            Q_values: shape (n_trajectories,)
            z_trajectories: shape (n_trajectories, N_steps+1, L)
            xi_trajectories: shape (n_trajectories, N_steps, L)
        """
        Q_values = np.zeros(n_trajectories)
        z_trajectories = np.zeros((n_trajectories, self.N_steps + 1, self.L))
        xi_trajectories = np.zeros((n_trajectories, self.N_steps, self.L), dtype=int)
        
        # Set up iterator with optional progress bar
        if progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(range(n_trajectories), desc="Simulating trajectories")
            except ImportError:
                iterator = range(n_trajectories)
        else:
            iterator = range(n_trajectories)
        
        for i in iterator:
            Q, z_traj, xi_traj = self.simulate_trajectory()
            Q_values[i] = Q
            z_trajectories[i] = z_traj
            xi_trajectories[i] = xi_traj
        
        return Q_values, z_trajectories, xi_trajectories
    
    def theoretical_prediction(self) -> float:
        """
        Calculate theoretical prediction for ⟨Q⟩ in the limit J >> 1/τ.
        
        From EP_Production.md:
        Q = L × (2T/τ × A + T/τ × (1-A)) = L × T/τ × (A+1)
        
        where:
        - T/τ = N × ε² (measurement strength)
        - A = ⟨z²⟩ averaged over time and qubits
        
        For ergodic systems, A ≈ 1/2, so ⟨Q⟩ = L × T/τ × 3/2
        """
        T_over_tau = self.N_steps * self.epsilon**2
        A = 0.5  # Assumption from theory
        return self.L * T_over_tau * (A + 1.0)


if __name__ == "__main__":
    # Quick validation test
    print("Multi-Qubit Correlation Matrix Simulator")
    print("=" * 60)
    
    # Test with L=3 qubits
    L_test = 3
    sim = MultiQubitCorrelationSimulator(
        L=L_test,
        J=10.0,      # Strong coupling limit
        epsilon=0.05,  # Measurement strength
        N_steps=1000,
        T=1.0,
        periodic=False
    )
    
    print(f"Parameters:")
    print(f"  L = {sim.L} qubits")
    print(f"  J = {sim.J}")
    print(f"  epsilon = {sim.epsilon}")
    print(f"  N = {sim.N_steps}")
    print(f"  T = {sim.T}")
    print(f"  dt = {sim.dt:.6f}")
    print(f"  T/tau = N*epsilon^2 = {sim.N_steps * sim.epsilon**2:.3f}")
    print(f"  Boundary conditions: {'periodic' if sim.periodic else 'open'}")
    
    # Single trajectory
    Q, z_traj, xi_traj = sim.simulate_trajectory()
    print(f"\nSingle trajectory:")
    print(f"  Q = {Q:.3f}")
    print(f"  Initial z = {z_traj[0]}")
    print(f"  Final z = {z_traj[-1]}")
    
    # Ensemble
    n_traj = 500
    print(f"\nRunning ensemble ({n_traj} trajectories)...")
    Q_values, z_trajs, _ = sim.simulate_ensemble(n_traj, progress=True)
    
    mean_Q = np.mean(Q_values)
    std_Q = np.std(Q_values)
    theoretical_Q = sim.theoretical_prediction()
    
    # Compute average A = <z^2> from trajectories
    z_squared = z_trajs**2  # shape (n_traj, N_steps+1, L)
    A_observed = np.mean(z_squared)
    A_per_qubit = np.mean(z_squared, axis=(0, 1))
    
    print(f"\nResults:")
    print(f"  Observed <Q> = {mean_Q:.3f} +/- {std_Q:.3f}")
    print(f"  Theoretical <Q> = {theoretical_Q:.3f} (assuming A=0.5)")
    print(f"  Ratio: {mean_Q/theoretical_Q:.3f}")
    print(f"  Observed A = <z^2> = {A_observed:.3f}")
    print(f"  A per qubit: {A_per_qubit}")
    
    # Compare with L=2 case
    print(f"\n{'='*60}")
    print("Comparison with L=2 case:")
    sim2 = MultiQubitCorrelationSimulator(
        L=2,
        J=10.0,
        epsilon=0.05,
        N_steps=1000,
        T=1.0
    )
    Q_values_2, z_trajs_2, _ = sim2.simulate_ensemble(n_traj, progress=True)
    mean_Q_2 = np.mean(Q_values_2)
    theoretical_Q_2 = sim2.theoretical_prediction()
    
    print(f"  L=2: <Q> = {mean_Q_2:.3f}, Theory = {theoretical_Q_2:.3f}, Ratio = {mean_Q_2/theoretical_Q_2:.3f}")
    print(f"  L=3: <Q> = {mean_Q:.3f}, Theory = {theoretical_Q:.3f}, Ratio = {mean_Q/theoretical_Q:.3f}")
    print(f"  Scaling ratio (L=3)/(L=2): Observed = {mean_Q/mean_Q_2:.3f}, Theory = {theoretical_Q/theoretical_Q_2:.3f}")
