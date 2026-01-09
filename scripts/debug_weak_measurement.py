#!/usr/bin/env python3
"""
Investigate weak measurement convergence behavior
"""
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from quantum_measurement.jw_expansion.non_hermitian_hat import NonHermitianHatSimulator
from quantum_measurement.jw_expansion.n_infty import sum_pbc

L = 9
gamma = 0.001
J = 1.0
g = gamma / (4 * J)
dt = 0.001
max_steps = 100000

n_exact = sum_pbc(g, L)

print(f"Investigating weak measurement: L={L}, γ={gamma}, g={g}")
print(f"Expected n_∞: {n_exact:.6f}\n")

sim = NonHermitianHatSimulator(
    L=L, J=J, gamma=gamma, dt=dt, N_steps=max_steps,
    closed_boundary=True
)

Q_total, n_traj, G_final = sim.simulate_trajectory(return_G_final=True)

# Analyze trajectory
times = np.arange(len(n_traj)) * dt
n_mean = n_traj.mean(axis=1)

print(f"Trajectory shape: {n_traj.shape}")
print(f"Final n: {n_mean[-1]:.6f}")
print(f"Error: {abs(n_mean[-1] - n_exact):.2e}\n")

# Check for oscillations
print("Checking behavior at different time points:")
checkpoints = [100, 500, 1000, 5000, 10000, 50000, 99000]
for step in checkpoints:
    if step < len(n_mean):
        print(f"  Step {step:6d} (t={times[step]:6.1f}): n={n_mean[step]:.6f}")

# Plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

# Full trajectory
ax1.plot(times, n_mean, 'b-', linewidth=0.8, alpha=0.7)
ax1.axhline(n_exact, color='red', linestyle='--', linewidth=2, label=f'Exact: {n_exact:.6f}')
ax1.set_xlabel('Time')
ax1.set_ylabel('⟨n⟩')
ax1.set_title(f'Weak Measurement: L={L}, γ={gamma}, g={g}')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Convergence metric
window = 1000
conv_metric = []
conv_times = []
for i in range(window, len(n_traj)):
    n_recent = n_traj[i - window//2:i, :].mean(axis=0)
    n_previous = n_traj[i - window:i - window//2, :].mean(axis=0)
    conv_metric.append(np.max(np.abs(n_recent - n_previous)))
    conv_times.append(times[i])

ax2.semilogy(conv_times, conv_metric, 'g-', linewidth=1)
ax2.axhline(1e-4, color='red', linestyle='--', label='Tolerance=1e-4')
ax2.set_xlabel('Time')
ax2.set_ylabel('max|Δn|')
ax2.set_title('Convergence Metric')
ax2.legend()
ax2.grid(True, alpha=0.3, which='both')

plt.tight_layout()
plt.savefig('results/ninf_scan/weak_measurement_debug.png', dpi=150)
print(f"\n✓ Plot saved to results/ninf_scan/weak_measurement_debug.png")
