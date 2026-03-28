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
from typing import Any, Tuple
from dataclasses import dataclass, field

from quantum_measurement.backends import Backend, get_backend
from quantum_measurement.utilities.gpu_utils import estimate_trajectory_batch_size


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
    device: str = "cpu"
    backend: Backend | None = field(default=None, repr=False)
    rng: np.random.Generator | None = field(default=None, repr=False)
    
    def __post_init__(self):
        if self.backend is None:
            self.backend = get_backend(self.device)
        if self.rng is None:
            self.rng = np.random.default_rng()
        self.dt = self.T / self.N_steps
        self.L = 2  # Two qubits
        
        # Build the Hamiltonian matrix h in Nambu space (4×4)
        self.h = self._build_hamiltonian()
        
        # Initial correlation matrix for |↑↑⟩ state
        # In fermion language: both sites occupied
        self.G_initial = self.backend.array(self.backend.diag([1.0, 1.0, 0.0, 0.0]), dtype=complex)
    
    def _build_hamiltonian(self) -> Any:
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
        h11 = self.backend.array([[0.0, -self.J], [0.0, 0.0]], dtype=complex)
        h12 = self.backend.array([[0.0, -self.J], [0.0, 0.0]], dtype=complex)
        h21 = self.backend.array([[0.0, +self.J], [0.0, 0.0]], dtype=complex)
        h22 = self.backend.array([[0.0, +self.J], [0.0, 0.0]], dtype=complex)
        
        # Assemble block matrix
        top = self.backend.hstack([h11, h12])
        bottom = self.backend.hstack([h21, h22])
        h = self.backend.vstack([top, bottom])
        return h
    
    def _compute_z_values(self, G: Any) -> Tuple[float, float]:
        """
        Compute ⟨σ₁ᶻ⟩ and ⟨σ₂ᶻ⟩ from correlation matrix.
        
        From Jordan-Wigner: σᵢᶻ = 2nᵢ - 1 = 2c†ᵢcᵢ - 1
        So ⟨σᵢᶻ⟩ = 2⟨c†ᵢcᵢ⟩ - 1 = 2G[i-1, i-1] - 1
        
        For L=2:
        - G[0,0] = ⟨c₁†c₁⟩, so z₁ = 2*Re(G[0,0]) - 1
        - G[1,1] = ⟨c₂†c₂⟩, so z₂ = 2*Re(G[1,1]) - 1
        """
        diag_real = self.backend.asnumpy(self.backend.real(self.backend.diag(G)))
        z1 = 2.0 * float(diag_real[0]) - 1.0
        z2 = 2.0 * float(diag_real[1]) - 1.0
        return float(z1), float(z2)
    
    def _hamiltonian_step(self, G: Any) -> Any:
        """
        Apply Hamiltonian evolution: dG = -2i dt [G, h]
        """
        commutator = self.backend.matmul(G, self.h) - self.backend.matmul(self.h, G)
        dG = -2.0j * self.dt * commutator
        return G + dG
    
    def _measurement_step(self, G: Any) -> Tuple[Any, np.ndarray]:
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
        xi_hat = self.backend.array(self.backend.diag([xi1, xi2, -xi1, -xi2]), dtype=complex)
        
        # Measurement evolution (stochastic term)
        stochastic = self.epsilon * (
            self.backend.matmul(G, xi_hat)
            + self.backend.matmul(xi_hat, G)
            - 2.0 * self.backend.matmul(self.backend.matmul(G, xi_hat), G)
        )
        
        # Damping term (deterministic)
        G_diag = self.backend.diag(self.backend.diag(G))
        damping = -self.epsilon**2 * (G - G_diag)
        
        dG = stochastic + damping
        G_new = G + dG
        
        # Ensure Hermiticity (numerical stability)
        G_new = 0.5 * (G_new + self.backend.conj(self.backend.transpose(G_new)))
        
        # Clip diagonal elements to physical range [0, 1]
        # The diagonal of G represents occupation probabilities
        diag_real = self.backend.real(self.backend.diag(G_new))
        diag_clipped = self.backend.asnumpy(self.backend.clip(diag_real, 0.0, 1.0))
        for i, value in enumerate(diag_clipped):
            G_new[i, i] = float(value) + 0.0j
        
        return G_new, xi_array

    def _compute_z_values_batch(self, G_batch: Any) -> np.ndarray:
        """Compute z-values for batched correlation matrices.

        Parameters
        ----------
        G_batch : backend array
            Shape (n_batch, 4, 4)
        """
        diag0 = self.backend.real(G_batch[:, 0, 0])
        diag1 = self.backend.real(G_batch[:, 1, 1])
        z1 = 2.0 * diag0 - 1.0
        z2 = 2.0 * diag1 - 1.0
        z = self.backend.asnumpy(self.backend.vstack([z1, z2])).T
        return z.astype(float)

    def _hamiltonian_step_batch(self, G_batch: Any) -> Any:
        """Apply Hamiltonian evolution to a batch of trajectories."""
        return self.backend.batched_commutator_update(G_batch, self.h, self.dt)

    def _measurement_step_batch(self, G_batch: Any, xi_step: Any) -> Any:
        """Apply a single measurement step to a batch of trajectories."""
        n_batch = int(G_batch.shape[0])
        xi_hat = self.backend.get_workspace("twoq_xi_hat", (n_batch, 4, 4), complex)

        xi1 = xi_step[:, 0]
        xi2 = xi_step[:, 1]
        xi_hat[:, 0, 0] = xi1
        xi_hat[:, 1, 1] = xi2
        xi_hat[:, 2, 2] = -xi1
        xi_hat[:, 3, 3] = -xi2

        stochastic = self.epsilon * (
            self.backend.matmul(G_batch, xi_hat)
            + self.backend.matmul(xi_hat, G_batch)
            - 2.0 * self.backend.matmul(self.backend.matmul(G_batch, xi_hat), G_batch)
        )

        damping = -(self.epsilon**2) * G_batch

        G_new = G_batch + stochastic + damping
        for idx in range(4):
            G_new[:, idx, idx] += (self.epsilon**2) * G_batch[:, idx, idx]

        G_new = self.backend.symmetrize_clip_diag_inplace(G_new)

        return G_new

    def simulate_trajectory_batch(
        self,
        n_batch: int,
        xi_batch: Any | None = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Simulate a batch of trajectories in parallel.

        Parameters
        ----------
        n_batch : int
            Number of trajectories in this batch.
        xi_batch : backend/NumPy array | None
            Optional pre-generated measurement outcomes with shape
            (n_batch, N_steps, 2) and values in {-1, +1}.
        """
        if n_batch < 1:
            raise ValueError("n_batch must be at least 1")

        if xi_batch is None:
            xi_batch_dev = self.backend.choice_pm1((n_batch, self.N_steps, 2))
        else:
            xi_batch_dev = self.backend.array(xi_batch)

        G_batch = self.backend.zeros((n_batch, 4, 4), dtype=complex)
        for idx in range(n_batch):
            G_batch[idx] = self.G_initial

        z_trajectory = np.zeros((n_batch, self.N_steps + 1, 2), dtype=float)
        xi_trajectory = np.zeros((n_batch, self.N_steps, 2), dtype=int)
        Q_values = np.zeros(n_batch, dtype=float)

        z_trajectory[:, 0, :] = self._compute_z_values_batch(G_batch)

        for i in range(self.N_steps):
            z_before = z_trajectory[:, i, :]

            G_batch = self._hamiltonian_step_batch(G_batch)

            xi_step = xi_batch_dev[:, i, :]
            G_batch = self._measurement_step_batch(G_batch, xi_step)

            xi_step_np = self.backend.asnumpy(xi_step).astype(int)
            xi_trajectory[:, i, :] = xi_step_np

            z_after = self._compute_z_values_batch(G_batch)
            z_trajectory[:, i + 1, :] = z_after

            for k in range(2):
                z_before_k = z_before[:, k]
                z_after_k = z_after[:, k]
                xi_k = xi_step_np[:, k]
                term1 = 2.0 * self.epsilon**2 * z_before_k * 0.5 * (z_before_k + z_after_k)
                term2 = 2.0 * self.epsilon * xi_k * 0.5 * (z_before_k + z_after_k)
                Q_values += term1 + term2

        return Q_values, z_trajectory, xi_trajectory
    
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
        G = self.backend.copy(self.G_initial)
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
        progress: bool = False,
        batch_size: int | None = None,
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

        if batch_size is None:
            if self.backend.is_gpu:
                batch_size = min(n_trajectories, estimate_trajectory_batch_size(self.L))
            else:
                batch_size = 1
        batch_size = max(1, int(batch_size))
        
        # Set up iterator with optional progress bar
        if progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(range(0, n_trajectories, batch_size), desc="Simulating trajectories")
            except ImportError:
                iterator = range(0, n_trajectories, batch_size)
        else:
            iterator = range(0, n_trajectories, batch_size)
        
        for start_idx in iterator:
            end_idx = min(start_idx + batch_size, n_trajectories)
            current_batch = end_idx - start_idx
            Q_batch, z_batch, xi_batch = self.simulate_trajectory_batch(current_batch)
            Q_values[start_idx:end_idx] = Q_batch
            z_trajectories[start_idx:end_idx] = z_batch
            xi_trajectories[start_idx:end_idx] = xi_batch
        
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
