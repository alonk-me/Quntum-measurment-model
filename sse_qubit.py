
"""
sse_qubit.py

Stochastic Schrödinger Equation (SSE) for a single qubit continuously
measured in σ_z with rate gamma. Discretized with Euler–Maruyama (Itô).
Optionally includes a Hamiltonian H = (omega/2) σ_x to generate precession.

State is represented as a complex vector psi ∈ C^2 with ||psi||=1.
Measurement operator: L = sqrt(gamma) * sigma_z.
d|psi> = [ -i H dt - 0.5 L^\dagger L dt + ( <L+L^\dagger>/2 * I - L ) dW ] |psi>   (common forms vary)
Here we use the familiar weak-measurement Itô SSE for Hermitian measured operator A=σ_z:

d|psi> = [ -i H dt - (gamma/2)(sigma_z^2 - <sigma_z> sigma_z) dt + sqrt(gamma) (sigma_z - <sigma_z>) dW ] |psi>

Because sigma_z^2 = I, the deterministic term simplifies.

We also accumulate an additive functional Q suggested by the user's notes:
Q = sum_t (avg_{midpoint} <sigma_z>) * xi_t
where dW_t = sqrt(dt) * xi_t and xi_t ~ N(0,1).
This matches a “symmetric operators / midpoint” convention for entropy production-like observables.

Edit map_epsilon_to_gamma(epsilon, dt) if you have an exact relation from your notes.

Author: ChatGPT
"""
import numpy as np

SIGMA_X = np.array([[0., 1.],[1., 0.]], dtype=complex)
SIGMA_Y = np.array([[0., -1j],[1j, 0.]], dtype=complex)
SIGMA_Z = np.array([[1., 0.],[0., -1.]], dtype=complex)
IDENT   = np.eye(2, dtype=complex)

def normalize(psi):
    n2 = np.vdot(psi, psi).real
    if n2 <= 0:
        raise ValueError("State norm vanished.")
    return psi / np.sqrt(n2)

def expval(psi, A):
    return np.vdot(psi, A @ psi).real

def map_epsilon_to_gamma(epsilon, dt):
    """
    Placeholder mapping between a discretization parameter epsilon and the
    continuous measurement rate gamma.

    A common weak-measurement scaling is epsilon ~ sqrt(gamma * dt),
    hence gamma ≈ (epsilon**2) / dt. If your notes introduce a factor,
    adjust here, e.g., gamma = (epsilon**2) / (c * dt).

    Replace 'c = 1.0' below per your derivation.
    """
    c = 1.0
    return (epsilon**2) / (c * dt)

def step_sse(psi, dt, gamma, omega=0.0, rng=None):
    """
    One Euler–Maruyama step of the Itô SSE for measurement in sigma_z.
    Optionally include H = (omega/2) sigma_x.

    Returns:
      psi_next, xi, m_before, m_after
    where xi ~ N(0,1), m_* are <sigma_z> before/after the step.
    """
    if rng is None:
        rng = np.random.default_rng()
    m_before = expval(psi, SIGMA_Z)
    xi = rng.normal(0.0, 1.0)
    dW = np.sqrt(dt) * xi

    # Hamiltonian
    H = 0.5 * omega * SIGMA_X

    # Drift + diffusion operators
    # Deterministic drift from measurement: -(gamma/2)(I - <σ_z> σ_z)
    drift_meas = -0.5 * gamma * (IDENT - m_before * SIGMA_Z)
    # Stochastic term: sqrt(gamma) (σ_z - <σ_z>)
    diff_op = np.sqrt(gamma) * (SIGMA_Z - m_before * IDENT)

    # Update: |ψ> + [-i H |ψ> + drift_meas |ψ>] dt + diff_op |ψ> dW
    psi_next = psi + (-1j * (H @ psi) + drift_meas @ psi) * dt + (diff_op @ psi) * dW
    psi_next = normalize(psi_next)
    m_after  = expval(psi_next, SIGMA_Z)
    return psi_next, xi, m_before, m_after

def run_trajectory(T=1.0, N=1000, psi0=None, gamma=1.0, omega=0.0, midpoint_Q=True, rng=None):
    """
    Simulate one trajectory.
    Returns dict with time grid, magnetization m_t, states, and Q.
    """
    if rng is None:
        rng = np.random.default_rng()
    dt = T / N
    if psi0 is None:
        # default: +x state (equal superposition) for symmetry
        psi0 = (1/np.sqrt(2)) * np.array([1., 1.], dtype=complex)

    psi = normalize(psi0.astype(complex))
    tgrid = np.linspace(0., T, N+1)
    m_list = [expval(psi, SIGMA_Z)]
    Q = 0.0

    for k in range(N):
        psi_next, xi, m_before, m_after = step_sse(psi, dt, gamma, omega=omega, rng=rng)
        m_mid = 0.5*(m_before + m_after)
        # Entropy-production-like increment with symmetric (midpoint) average
        if midpoint_Q:
            Q += m_mid * xi
        else:
            Q += m_before * xi
        psi = psi_next
        m_list.append(m_after)

    return {
        "t": tgrid,
        "m": np.array(m_list),
        "Q": float(Q)
    }

def run_ensemble(num_traj=1000, T=1.0, N=1000, psi0=None, gamma=1.0, omega=0.0, midpoint_Q=True, seed=None):
    rng = np.random.default_rng(seed)
    Qs = []
    ms = []
    for j in range(num_traj):
        out = run_trajectory(T=T, N=N, psi0=psi0, gamma=gamma, omega=omega, midpoint_Q=midpoint_Q, rng=rng)
        Qs.append(out["Q"])
        ms.append(out["m"])
    return np.array(Qs), np.array(ms)
