"""
two_qubit_correlation_simulator.py
===================================

Implements 2-qubit simulation using correlation matrix approach with:
1. Hamiltonian evolution: H = J σ₁ˣ σ₂ˣ
2. Stochastic measurement: independent measurements on each qubit in σᶻ basis
3. Entropy production calculation following EP_Production.md

The correlation matrix G evolves according to:
    dG = -2i dt [G, h] + ε(G ξ̂ + ξ̂ G - 2G ξ̂ G) - ε²(G - G^diag)

where:
- h is the single-particle Hamiltonian in Nambu (Bogoliubov) space
- ξ̂ = diag(ξ₁, ξ₂, -ξ₁, -ξ₂) with ξᵢ = ±1 random
- G is a 4×4 correlation matrix for L=2

Initial state: |↑↑⟩ corresponds to G = diag(1, 1, 0, 0)
"""

from __future__ import annotations
import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class TwoQubitCorrelationSimulator:
    """
    Correlation matrix simulator for 2 qubits with Hamiltonian + measurements.
    
    Parameters
    ----------
    J : float
        Coupling constant for H = J σ₁ˣ σ₂ˣ
    epsilon : float
        Measurement strength parameter
    N_steps : int
        Number of time steps
    T : float
        Total evolution time (used to compute dt = T / N_steps)
    rng : np.random.Generator | None
        Random number generator
    """
    J: float = 1.0
    epsilon: float = 0.1
    N_steps: int = 1000
    T: float = 1.0
    rng: np.random.Generator | None = field(default=None, repr=False)
    
    def __post_init__(self):
        if self.rng is None:
            self.rng = np.random.default_rng()
        self.dt = self.T / self.N_steps
        self.L = 2  # Two qubits
        
        # Build the Hamiltonian matrix h in Nambu space (4×4)
        self.h = self._build_hamiltonian()
        
        # Initial correlation matrix for |↑↑⟩ state
        # In fermion language: both sites occupied
        self.G_initial = np.diag([1.0, 1.0, 0.0, 0.0]).astype(complex)
    
    def _build_hamiltonian(self) -> np.ndarray:
        """
        Build the 4×4 Hamiltonian matrix h for H = J σ₁ˣ σ₂ˣ.
        
        From free_fermion.md Example (L=2):
        h₁₁ = [[0, -J], [0, 0]]
        h₂₂ = [[0, +J], [0, 0]]
        h₁₂ = [[0, -J], [0, 0]]
        h₂₁ = [[0, +J], [0, 0]]
        
        where J here is actually -1 in the docs example, but we parameterize it.
        """
        # Each block is 2×2 for L=2
        h11 = np.array([[0.0, -self.J], [0.0, 0.0]], dtype=complex)
        h12 = np.array([[0.0, -self.J], [0.0, 0.0]], dtype=complex)
        h21 = np.array([[0.0, +self.J], [0.0, 0.0]], dtype=complex)
        h22 = np.array([[0.0, +self.J], [0.0, 0.0]], dtype=complex)
        
        # Assemble block matrix
        top = np.hstack([h11, h12])
        bottom = np.hstack([h21, h22])
        h = np.vstack([top, bottom])
        return h
    
    def _compute_z_values(self, G: np.ndarray) -> Tuple[float, float]:
        """
        Compute ⟨σ₁ᶻ⟩ and ⟨σ₂ᶻ⟩ from correlation matrix.
        
        From Jordan-Wigner: σᵢᶻ = 2nᵢ - 1 = 2c†ᵢcᵢ - 1
        So ⟨σᵢᶻ⟩ = 2⟨c†ᵢcᵢ⟩ - 1 = 2G[i-1, i-1] - 1
        
        For L=2:
        - G[0,0] = ⟨c₁†c₁⟩, so z₁ = 2*Re(G[0,0]) - 1
        - G[1,1] = ⟨c₂†c₂⟩, so z₂ = 2*Re(G[1,1]) - 1
        """
        z1 = 2.0 * np.real(G[0, 0]) - 1.0
        z2 = 2.0 * np.real(G[1, 1]) - 1.0
        return float(z1), float(z2)
    
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
            (G_new, xi_array) where xi_array = [ξ₁, ξ₂]
        """
        # Sample random measurement outcomes
        xi1 = 1 if self.rng.random() < 0.5 else -1
        xi2 = 1 if self.rng.random() < 0.5 else -1
        xi_array = np.array([xi1, xi2])
        
        # Build ξ̂ = diag(ξ₁, ξ₂, -ξ₁, -ξ₂)
        xi_hat = np.diag([xi1, xi2, -xi1, -xi2]).astype(complex)
        
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
            z_trajectory: array of shape (N_steps+1, 2) with z₁, z₂ at each step
            xi_trajectory: array of shape (N_steps, 2) with ξ₁, ξ₂ at each step
        """
        G = self.G_initial.copy()
        z_trajectory = np.zeros((self.N_steps + 1, 2))
        xi_trajectory = np.zeros((self.N_steps, 2), dtype=int)
        
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
            for k in range(2):  # k = 0, 1 for qubits 1, 2
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
            z_trajectories: shape (n_trajectories, N_steps+1, 2)
            xi_trajectories: shape (n_trajectories, N_steps, 2)
        """
        Q_values = np.zeros(n_trajectories)
        z_trajectories = np.zeros((n_trajectories, self.N_steps + 1, 2))
        xi_trajectories = np.zeros((n_trajectories, self.N_steps, 2), dtype=int)
        
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
        Q = 2 × (2T/τ × A + T/τ × (1-A)) = 2 × T/τ × (A+1)
        
        where:
        - T/τ = N × ε² (measurement strength)
        - A = ⟨z²⟩ averaged over time
        
        For single qubit, A = 1/2. For 2 qubits, we also expect A ≈ 1/2.
        If A = 1/2, then ⟨Q⟩ = 2 × T/τ × (1/2 + 1) = 3T/τ
        """
        T_over_tau = self.N_steps * self.epsilon**2
        A = 0.5  # Assumption from theory
        return 2.0 * T_over_tau * (A + 1.0)
    
    def compute_susceptibility(
        self,
        z_trajectory: np.ndarray,
        site_i: int = 0,
        site_j: Optional[int] = None,
    ) -> float:
        """Compute susceptibility from a z trajectory.
        
        Parameters
        ----------
        z_trajectory : np.ndarray
            Trajectory of shape (N_steps+1, 2) containing [z₁, z₂] at each step
        site_i : int
            First site index (0 or 1)
        site_j : int, optional
            Second site index. If None, use site_i (autocorrelation)
        
        Returns
        -------
        float
            Static susceptibility χ_ij
        
        Examples
        --------
        >>> sim = TwoQubitCorrelationSimulator()
        >>> Q, z_traj, _ = sim.simulate_trajectory()
        >>> chi = sim.compute_susceptibility(z_traj, site_i=0)
        """
        from quantum_measurement.susceptibility import compute_static_susceptibility
        
        if site_j is None:
            site_j = site_i
        
        times = np.linspace(0, self.T, self.N_steps + 1)
        z_i = z_trajectory[:, site_i]
        z_j = z_trajectory[:, site_j]
        
        chi = compute_static_susceptibility(times, z_i, z_j)
        return chi
    
    def compute_susceptibility_ensemble(
        self,
        z_trajectories: np.ndarray,
        site_i: int = 0,
        site_j: Optional[int] = None,
    ) -> Tuple[float, float]:
        """Compute average susceptibility over an ensemble of trajectories.
        
        Parameters
        ----------
        z_trajectories : np.ndarray
            Ensemble of trajectories, shape (n_trajectories, N_steps+1, 2)
        site_i : int
            First site index (0 or 1)
        site_j : int, optional
            Second site index. If None, use site_i
        
        Returns
        -------
        Tuple[float, float]
            (mean_chi, std_chi) - mean and standard deviation of susceptibility
        
        Examples
        --------
        >>> sim = TwoQubitCorrelationSimulator()
        >>> Q_vals, z_trajs, _ = sim.simulate_ensemble(100)
        >>> mean_chi, std_chi = sim.compute_susceptibility_ensemble(z_trajs)
        """
        n_traj = z_trajectories.shape[0]
        chi_values = np.zeros(n_traj)
        
        for i in range(n_traj):
            chi_values[i] = self.compute_susceptibility(
                z_trajectories[i], site_i, site_j
            )
        
        return np.mean(chi_values), np.std(chi_values)


if __name__ == "__main__":
    # Quick validation test
    print("Two-Qubit Correlation Matrix Simulator")
    print("=" * 50)
    
    sim = TwoQubitCorrelationSimulator(
        J=10.0,      # Strong coupling limit
        epsilon=0.05,  # Smaller epsilon for stability
        N_steps=1000,
        T=1.0
    )
    
    print(f"Parameters:")
    print(f"  J = {sim.J}")
    print(f"  epsilon = {sim.epsilon}")
    print(f"  N = {sim.N_steps}")
    print(f"  T = {sim.T}")
    print(f"  dt = {sim.dt:.6f}")
    print(f"  T/tau = N*epsilon^2 = {sim.N_steps * sim.epsilon**2:.3f}")
    
    # Single trajectory
    Q, z_traj, xi_traj = sim.simulate_trajectory()
    print(f"\nSingle trajectory:")
    print(f"  Q = {Q:.3f}")
    print(f"  Initial z1, z2 = {z_traj[0]}")
    print(f"  Final z1, z2 = {z_traj[-1]}")
    
    # Ensemble
    n_traj = 1000
    print(f"\nRunning ensemble ({n_traj} trajectories)...")
    Q_values, z_trajs, _ = sim.simulate_ensemble(n_traj, progress=True)
    
    mean_Q = np.mean(Q_values)
    std_Q = np.std(Q_values)
    theoretical_Q = sim.theoretical_prediction()
    
    # Compute average A = <z^2> from trajectories
    z_squared = z_trajs[:, :, :]**2  # shape (n_traj, N_steps+1, 2)
    A_observed = np.mean(z_squared)
    
    print(f"\nResults:")
    print(f"  Observed <Q> = {mean_Q:.3f} +/- {std_Q:.3f}")
    print(f"  Theoretical <Q> = {theoretical_Q:.3f} (assuming A=0.5)")
    print(f"  Ratio: {mean_Q/theoretical_Q:.3f}")
    print(f"  Observed A = <z^2> = {A_observed:.3f}")
