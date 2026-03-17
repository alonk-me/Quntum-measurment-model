# Non‑Hermitian Measurement Simulators

This repository contains a collection of small Python modules and a
demonstration notebook implementing free‑fermion simulations of a
spin–1/2 chain subject to continuous local measurements.  The goal is to
reproduce and extend the non–Hermitian dynamics discussed in
Turkeshi *et al.* (Eq. 10 of the referenced notes) and to explore the
resulting entropy production.

## Background

Continuous monitoring of a quantum observable can be modelled in
stochastic Schrödinger equation (SSE) language by a Hamiltonian
interrupted by quantum jumps.  When the measurement record is
post‑selected to contain **no clicks** the average state evolves under
an *effective* non‑Hermitian Hamiltonian.  For a one–dimensional chain
with nearest‑neighbour transverse coupling the Hamiltonian adopted in
Turkeshi’s notes reads

\begin{aligned}
H_{\mathrm{eff}} &= J \sum_{j=0}^{L-2} \sigma^x_j\,\sigma^x_{j+1}
  \;\;\;\; -\frac{\mathrm{i}\,\gamma}{2} \sum_{j=0}^{L-1} \hat{n}_j,\\
\hat{n}_j &= |1_j\rangle\langle 1_j| = \frac{\sigma^z_j + 1}{2}
\end{aligned}

The first term is the usual nearest‑neighbour exchange while the second
term is purely imaginary and suppresses any occupation of the state
`|1⟩`.  Evolving under this non‑Hermitian Hamiltonian causes the
probability amplitudes of **both** creation and annihilation operators
to decay at rate `γ/2`.  When combined with a Jordan–Wigner
transformation the problem maps to free fermions with a quadratic
Hamiltonian.  The dynamics of the `2L×2L` covariance matrix

``\nG_{ab} = \langle \Psi_a \Psi_b^\dagger \rangle,\n``

obey the simple differential equation

\[\dot{G} = -2\mathrm{i}[G,h] - \gamma G,\]

where `h` is the Bogoliubov–de Gennes (BdG) single–particle
Hamiltonian.  The last term generates exponential decay of every matrix
element.  The state is initialised in the vacuum `|↓↓…↓⟩`, which in
fermionic language means no particles: \(\langle c_i^\dagger c_i \rangle
= 0\).  The standard magnetisation is obtained from the diagonal of `G`
via \(\langle \sigma^z_i\rangle = 2\,\langle n_i\rangle - 1\).

The entropy production associated with the imaginary potential can be
computed from forward and backward trajectory probabilities.  For a
product initial state one finds that the instantaneous entropy
production rate is

\[\frac{\mathrm{d}S}{\mathrm{d}t}
    = \gamma\sum_{j=0}^{L-1} \bigl(1 - \langle n_j\rangle\bigr),\]

which reflects the fact that an empty site contributes maximally to the
entropy budget while an occupied site is stable.  When divided by the
number of discretisation steps the total accumulated entropy `Q`
approaches the linear growth displayed in Figs. 1 and 2 of the attached
PDF.  Those figures plot an *adjusted* entropy production rate
\(\dot{S} - \gamma L\) so as to isolate the subleading corrections from
the leading extensive term `γ L`.

## Files

* **`non_hermitian_hat.py`** – Implements the basic correlation matrix
  simulator with number‑operator monitoring.  The class
  `NonHermitianHatSimulator` updates the covariance matrix according to
  \(\dot{G} = -2\mathrm{i}[G,h] - \gamma G\) using a simple Euler
  discretisation and accumulates the entropy production via the
  Stratonovich prescription described above.  It returns occupation
  trajectories `⟨n_i⟩` and the total entropy `Q`.

* **`non_hermitian_spin.py`** – Builds on the hat simulator but
  converts the occupation on each site to a magnetisation
  \(\langle\sigma^z\rangle = 2\langle n\rangle - 1\).  The entropy is
  accumulated in exactly the same way; only the reported observables
  differ.  The class `NonHermitianSpinSimulator` is a drop–in
  replacement when one is interested in spin expectation values rather
  than fermion occupations.

* **`non_hermitian_adjusted.py`** – Wraps the hat simulator and
  post‑processes its output by subtracting the extensive term `γ L` from
  the accumulated entropy.  This adjusted entropy corresponds to the
  quantity plotted in the PDF figures, where the leading linear
  contribution has been removed to emphasise subleading decay rates.

* **`demo_non_hermitian.ipynb`** – A Jupyter notebook illustrating the
  usage of the three simulators.  It runs a short simulation on a
  periodic six‑site chain for several measurement rates `γ` and plots
  the entropy production per time step for each implementation.  The
  notebook uses periodic boundary conditions (set via
  `closed_boundary=True`) as requested.

## Differences between the implementations

1. **Hat vs spin operators** – The underlying free‑fermion dynamics
   (Hamiltonian update and imaginary potential) are identical in both
   implementations.  The hat version reports the occupations
   `⟨n_i⟩` directly, whereas the spin version converts these to
   magnetisations via `σ^z = 2 n - 1`.  Consequently, the entropy
   increment at each time step can be expressed as either
   \(\gamma\sum (1 - \bar{n})\) or \((\gamma/2)\sum (1 - \bar{\sigma}^z)\),
   but the accumulated `Q` is the same for both classes.  In the
   spin version one typically visualises `⟨σ^z⟩` rather than
   `⟨n⟩`.

2. **Adjusted simulator** – The hat simulator returns the full entropy
   production, which grows approximately at rate `γ L` for strong
   measurements.  To compare with the analytic spectra derived via the
   Jordan–Wigner transformation one subtracts this leading term.  The
   adjusted simulator performs this subtraction automatically, returning
   `Q - γ L` as the entropy.  When divided by the total time (unity in
   these simulations) and plotted against `γ` the results reproduce the
   trends shown in Figs. 1 and 2 of the PDF.  In particular, at large
   `γ` the adjusted rate tends towards zero, indicating that the
   spectrum is dominated by the trivial decay and the system stabilises
   in the state `|↓↓…↓⟩`.

## Running the notebook

To run the demonstration notebook, open `demo_non_hermitian.ipynb` in
Jupyter or any compatible notebook interface.  Execute the cells in
order.  The notebook will import the simulators, sweep over a few
values of the measurement rate `γ` and generate a plot of the entropy
production per time step for each implementation.  Feel free to adjust
the chain length `L`, the coupling `J`, the number of time steps
`N_steps` or switch between open and periodic boundary conditions by
modifying the parameters passed to the simulator constructors.
