#!/usr/bin/env python3
"""
Test that the updated parameter range (γ ≥ 0.01) works properly
"""
import numpy as np
import sys
from pathlib import Path
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

from quantum_measurement.jw_expansion.non_hermitian_hat import NonHermitianHatSimulator
from quantum_measurement.jw_expansion.n_infty import sum_pbc

def get_adaptive_max_steps(L, gamma, dt=0.001):
    """NEW formula from run_ninf_scan.py"""
    base_time = 20.0 / gamma
    base_steps = int(base_time / dt)
    size_factor = 1.0 + 0.2 * np.log(max(L / 9.0, 1.0))
    max_steps = int(base_steps * size_factor)
    max_steps = max(30000, min(max_steps, 500000))
    return max_steps

# Test the NEW minimum gamma
L = 9
gamma = 0.01  # NEW minimum (was 0.001)
J = 1.0
g = gamma / (4 * J)
dt = 0.001
tolerance = 1e-4
window_size = 1000

max_steps = get_adaptive_max_steps(L, gamma, dt)
n_exact = sum_pbc(g, L)

print("="*60)
print("TESTING NEW PARAMETER RANGE (γ ≥ 0.01)")
print("="*60)
print(f"L={L}, γ={gamma}, g={g}")
print(f"max_steps={max_steps}")
print(f"Expected n_∞: {n_exact:.6f}")
print()

print("Running simulation...")
start = time.time()

sim = NonHermitianHatSimulator(
    L=L, J=J, gamma=gamma, dt=dt, N_steps=max_steps,
    closed_boundary=True
)

Q_total, n_traj, G_final = sim.simulate_trajectory(return_G_final=True)

# Post-hoc convergence check
converged = False
convergence_step = max_steps

for step in range(window_size, len(n_traj)):
    n_recent = n_traj[step - window_size//2:step, :].mean(axis=0)
    n_previous = n_traj[step - window_size:step - window_size//2, :].mean(axis=0)
    max_diff = np.max(np.abs(n_recent - n_previous))
    
    if max_diff < tolerance:
        converged = True
        convergence_step = step
        break

n_infinity = n_traj[-window_size:, :].mean()
runtime = time.time() - start

# Check stability
times = np.arange(len(n_traj)) * dt
n_mean = n_traj.mean(axis=1)

# Sample at different time points
samples = [
    (1000, times[1000], n_mean[1000]),
    (10000, times[10000], n_mean[10000]),
    (50000, times[50000], n_mean[50000]),
    (-1, times[-1], n_mean[-1])
]

print(f"✓ Simulation complete!")
print(f"  Runtime: {runtime:.1f}s")
print(f"  Converged: {converged}")
if converged:
    print(f"  Convergence step: {convergence_step}/{max_steps}")
print(f"\n  n_∞^sim: {n_infinity:.6f}")
print(f"  n_∞^exact: {n_exact:.6f}")
print(f"  Error: {abs(n_infinity - n_exact):.2e}")

print(f"\nStability check (occupation at different times):")
for step, t, n in samples:
    print(f"  t={t:6.1f}: n={n:.6f}")

# Check for wild oscillations
n_std = np.std(n_mean[-10000:])  # Std dev of last 10k steps
print(f"\nLate-time stability: σ(n) = {n_std:.2e}")

if n_std < 0.01 and abs(n_infinity - n_exact) < 0.05:
    print("\n✓ TEST PASSED: Stable evolution, reasonable accuracy")
elif n_std < 0.1:
    print("\n⚠ TEST ACCEPTABLE: Some oscillations but converging")
else:
    print("\n❌ TEST FAILED: Still showing large oscillations")

print("="*60)
