#!/usr/bin/env python3
"""
Diagnostic script to reproduce the NaN issue in run_ninf_scan.py
"""

import numpy as np
import sys
from pathlib import Path
import warnings

warnings.filterwarnings('error', category=RuntimeWarning)

sys.path.insert(0, str(Path(__file__).parent.parent))

from quantum_measurement.jw_expansion.non_hermitian_hat import NonHermitianHatSimulator
from quantum_measurement.jw_expansion.n_infty import sum_pbc

# Test with the FIRST parameter from the grid
L = 9
gamma = 0.001  # Smallest gamma in log grid
J = 1.0
g = gamma / (4 * J)
dt = 0.001

print("="*60)
print("DIAGNOSTIC: Testing first parameter combination")
print("="*60)
print(f"L={L}, γ={gamma}, g={g}")
print(f"Expected n_∞: {sum_pbc(g, L):.6f}")
print()

# Try to reproduce the issue
try:
    print("Step 1: Initialize simulator...")
    sim = NonHermitianHatSimulator(
        L=L, J=J, gamma=gamma, dt=dt, N_steps=1000,
        closed_boundary=True
    )
    print("  ✓ Simulator initialized")
    
    print("\nStep 2: Check initial state...")
    print(f"  G_initial shape: {sim.G_initial.shape}")
    print(f"  G_initial has NaN: {np.isnan(sim.G_initial).any()}")
    print(f"  G_initial has Inf: {np.isinf(sim.G_initial).any()}")
    print(f"  G_initial min/max: {sim.G_initial.real.min():.6f} / {sim.G_initial.real.max():.6f}")
    
    print("\nStep 3: Run trajectory (1000 steps)...")
    Q_total, n_traj, G_final = sim.simulate_trajectory(return_G_final=True)
    
    print(f"  ✓ Simulation completed")
    print(f"\nStep 4: Check results...")
    print(f"  Trajectory shape: {n_traj.shape}")
    print(f"  Final n: {n_traj[-1].mean():.6f}")
    print(f"  Expected: {sum_pbc(g, L):.6f}")
    print(f"  Error: {abs(n_traj[-1].mean() - sum_pbc(g, L)):.2e}")
    print(f"  G_final has NaN: {np.isnan(G_final).any()}")
    print(f"  G_final has Inf: {np.isinf(G_final).any()}")
    
    print("\n" + "="*60)
    print("✓ SUCCESS: No NaN/Inf issues detected")
    print("="*60)
    
except RuntimeWarning as e:
    print(f"\n❌ RuntimeWarning caught!")
    print(f"Message: {e}")
    
    import traceback
    traceback.print_exc()
    
except Exception as e:
    print(f"\n❌ Error: {type(e).__name__}")
    print(f"Message: {e}")
    
    import traceback
    traceback.print_exc()
