# Explanation of the Numerical Solver

This short repository accompanies a numerical experiment exploring the
fermionic representation of the one‑dimensional Ising model.  The core
piece is a matrix differential equation

$$\frac{\mathrm{d}}{\mathrm{d}t} G(t) = -2\mathrm{i}\,(G h - h G)$$

where `G` and `h` are `2L×2L` complex matrices and
`[G, h] = G h - h G` denotes the matrix commutator.  For the
Jordan–Wigner (JW) fermionised Ising chain this commutator
equation governs the evolution of the correlation matrix, and the
observables of interest can be read off from its diagonal elements.

## Matrix construction

The module `matrix_commutator_solver.py` defines two helper functions:

* **`build_h(L, J)`** constructs the Hamiltonian‐like matrix `h`.
  `h` is divided into four `L×L` blocks, `h11`, `h12`, `h21` and
  `h22`.  The blocks `h11` and `h12` contain a single non‑zero band
  immediately above the main diagonal with value `−J`.  Conversely
  `h21` and `h22` have a band above the diagonal with value `+J`.

* **`initial_G(L)`** returns the initial condition
  `G(0)`: a diagonal matrix with `L` ones followed by `L` zeros.

## Time integration

The solver uses a forward Euler discretisation for the matrix ODE.  For a
time step `dt` the update reads

$$G_{n+1} = G_n + dt\;\bigl(-2\mathrm{i}\,(G_n h - h G_n)\bigr)$$

To mitigate numerical drift and preserve Hermiticity, each update
symmetrises `G` via `(G + G.conj().T)/2`.  A helper function
`compute_time_series(L, J, T, steps)` performs the integration, collects
the time grid and the quantity `1 + 2*G[0, 0]` at each step, and
returns both arrays.  The value `1 + 2*G[0, 0]` corresponds to
$\langle\sigma_1^z(t)\rangle$ in the JW picture.

## Jupyter notebook

The notebook `run_simulations.ipynb` demonstrates the solver for
`L = 2` and `L = 3` with `J = 1`, integrating up to `T = 10.0`
using `2000` steps.  It imports the helper function, runs the
simulation for each system size, and plots
$1 + 2\,G_{0,0}(t)$ as a function of time.  The plots exhibit
clear oscillations matching the analytic result
$\cos(2Jt)$, demonstrating that the numerical discretisation
captures the expected behaviour even for small chains.

You can adjust the chain length `L`, the coupling `J`, the final time
`T`, and the number of integration steps by editing the parameters
passed to `compute_time_series` and `run_and_plot` in the notebook.