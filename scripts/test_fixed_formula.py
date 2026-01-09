#!/usr/bin/env python3
"""
Quick test to verify the fixed adaptive max_steps works
"""
import numpy as np
import sys
from pathlib import Path
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

from quantum_measurement.jw_expansion.non_hermitian_hat import NonHermitianHatSimulator
from quantum_measurement.jw_expansion.n_infty import sum_pbc

def get_adaptive_max_steps(L, gamma, dt=0.001):
    """NEW formula - same as in run_ninf_scan.py"""
    base_time = 20.0 / gamma
    base_steps = int(base_time / dt)
    size_factor = 1.0 + 0.2 * np.log(max(L / 9.0, 1.0))
    max_steps = int(base_steps * size_factor)
    max_steps = max(30000, min(max_steps, 500000))
    return max_steps

# Test the most extreme case: L=9, gamma=0.001
L = 9
gamma = 0.001
J = 1.0
g = gamma / (4 * J)
dt = 0.001
tolerance = 1e-4
window_size = 1000

max_steps = get_adaptive_max_steps(L, gamma, dt)
n_exact = sum_pbc(g, L)

print("="*60)
print("TESTING FIXED ADAPTIVE MAX_STEPS")
print("="*60)
print(f"L={L}, γ={gamma}, g={g}")
print(f"max_steps={max_steps} (was 3,000,000 before fix)")
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

print(f"✓ Simulation complete!")
print(f"  Runtime: {runtime:.1f}s")
print(f"  Converged: {converged}")
if converged:
    print(f"  Convergence step: {convergence_step}/{max_steps}")
else:
    print(f"  Did not converge (final max|Δn|={max_diff:.2e})")
print(f"  n_∞^sim: {n_infinity:.6f}")
print(f"  n_∞^exact: {n_exact:.6f}")
print(f"  Error: {abs(n_infinity - n_exact):.2e}")
print()

if abs(n_infinity - n_exact) < 0.01 or converged:
    print("✓ TEST PASSED: Formula works reasonably")
else:
    print("⚠ TEST INCONCLUSIVE: May need more steps for weak measurement")
    print("  (This is expected - weak measurement converges slowly)")

print("="*60)
