use ndarray::{s, Array1, Array2, Array3};
use num_complex::Complex64;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArray3, PyReadonlyArray2, ToPyArray};
use pyo3::prelude::*;
use rand::prelude::*;
use rand::rngs::StdRng;
use rayon::prelude::*;

use crate::measurement::{hermitianise_and_clip, measurement_step_fused};

/// Rust implementation of LQubitCorrelationSimulator.
#[pyclass]
pub struct RustLQubitSimulator {
    #[pyo3(get)]
    pub l: usize,
    #[pyo3(get)]
    pub j: f64,
    #[pyo3(get)]
    pub epsilon: f64,
    #[pyo3(get)]
    pub n_steps: usize,
    #[pyo3(get)]
    pub t: f64,
    #[pyo3(get)]
    pub closed_boundary: bool,
    pub dt: f64,
    pub h: Array2<Complex64>,
    pub g_initial: Array2<Complex64>,
    pub seed: Option<u64>,
}

#[pymethods]
impl RustLQubitSimulator {
    #[new]
    #[pyo3(signature = (l=2, j=1.0, epsilon=0.1, n_steps=1000, t=1.0, closed_boundary=false, seed=None))]
    pub fn new(
        l: usize,
        j: f64,
        epsilon: f64,
        n_steps: usize,
        t: f64,
        closed_boundary: bool,
        seed: Option<u64>,
    ) -> PyResult<Self> {
        if l < 1 {
            return Err(pyo3::exceptions::PyValueError::new_err("L must be at least 1"));
        }
        if n_steps < 1 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "N_steps must be at least 1",
            ));
        }
        if t <= 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err("T must be positive"));
        }

        let dt = t / n_steps as f64;
        let h = build_hamiltonian(l, j, closed_boundary);
        let g_initial = build_g_initial(l);

        Ok(Self {
            l,
            j,
            epsilon,
            n_steps,
            t,
            closed_boundary,
            dt,
            h,
            g_initial,
            seed,
        })
    }

    /// Simulate a single trajectory.
    /// Returns (Q, z_trajectory, xi_trajectory) as numpy arrays.
    pub fn simulate_trajectory<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<(f64, &'py PyArray2<f64>, &'py PyArray2<i32>)> {
        let rng: StdRng = match self.seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_entropy(),
        };
        let (q, z_traj, xi_traj) = simulate_trajectory_internal(self, rng);
        Ok((
            q,
            z_traj.into_pyarray(py),
            xi_traj.into_pyarray(py),
        ))
    }

    /// Simulate an ensemble of trajectories and return mean z^2 over all.
    /// Uses Rayon for parallelism across trajectories.
    pub fn simulate_z2_mean(&self, num_trajectories: usize) -> PyResult<f64> {
        let seed_base: u64 = self.seed.unwrap_or(42);

        let total_sum: f64 = (0..num_trajectories)
            .into_par_iter()
            .map(|i| {
                let rng = StdRng::seed_from_u64(seed_base.wrapping_add(i as u64));
                let (_, z_traj, _) = simulate_trajectory_internal(self, rng);
                z_traj.iter().map(|&z| z * z).sum::<f64>()
            })
            .sum();

        let total_samples = num_trajectories * (self.n_steps + 1) * self.l;
        Ok(total_sum / total_samples as f64)
    }

    /// Simulate an ensemble and return (Q_values, z_trajectories, xi_trajectories).
    pub fn simulate_ensemble<'py>(
        &self,
        py: Python<'py>,
        n_trajectories: usize,
    ) -> PyResult<(&'py PyArray1<f64>, &'py PyArray3<f64>, &'py PyArray3<i32>)> {
        let seed_base: u64 = self.seed.unwrap_or(42);

        let results: Vec<(f64, Array2<f64>, Array2<i32>)> = (0..n_trajectories)
            .into_par_iter()
            .map(|i| {
                let rng = StdRng::seed_from_u64(seed_base.wrapping_add(i as u64));
                simulate_trajectory_internal(self, rng)
            })
            .collect();

        // Allocate 3D arrays: (n_traj, n_steps+1, L) and (n_traj, n_steps, L)
        let mut q_vals = Array1::<f64>::zeros(n_trajectories);
        let mut z_all = Array3::<f64>::zeros((n_trajectories, self.n_steps + 1, self.l));
        let mut xi_all = Array3::<i32>::zeros((n_trajectories, self.n_steps, self.l));
        
        for (i, (q, z, xi)) in results.into_iter().enumerate() {
            q_vals[i] = q;
            z_all.slice_mut(s![i, .., ..]).assign(&z);
            xi_all.slice_mut(s![i, .., ..]).assign(&xi);
        }
        
        Ok((q_vals.to_pyarray(py), z_all.to_pyarray(py), xi_all.to_pyarray(py)))
    }

    /// Set the initial correlation matrix from a Python array.
    pub fn set_g_initial(&mut self, g: PyReadonlyArray2<Complex64>) -> PyResult<()> {
        self.g_initial = g.as_array().to_owned();
        Ok(())
    }
}

// ─── Internal helpers ────────────────────────────────────────────────────────

fn build_hamiltonian(l: usize, j: f64, closed_boundary: bool) -> Array2<Complex64> {
    let n = 2 * l;
    let mut h = Array2::<Complex64>::zeros((n, n));

    for i in 0..(l - 1) {
        let ni = i;
        let ni1 = i + 1;
        // h11: top-left L×L block, -J on super-diagonal
        h[[ni, ni1]] = Complex64::new(-j, 0.0);
        // h12: top-right L×L block (offset by L), -J
        h[[ni, l + ni1]] = Complex64::new(-j, 0.0);
        // h21: bottom-left L×L block (offset by L), +J
        h[[l + ni, ni1]] = Complex64::new(j, 0.0);
        // h22: bottom-right L×L block, +J
        h[[l + ni, l + ni1]] = Complex64::new(j, 0.0);
    }

    if closed_boundary && l > 1 {
        let i = l - 1;
        let wrap_idx = 0_usize; // site wrapping back to the first site
        h[[i, wrap_idx]] = Complex64::new(-j, 0.0);
        h[[i, l + wrap_idx]] = Complex64::new(-j, 0.0);
        h[[l + i, wrap_idx]] = Complex64::new(j, 0.0);
        h[[l + i, l + wrap_idx]] = Complex64::new(j, 0.0);
    }

    h
}

fn build_g_initial(l: usize) -> Array2<Complex64> {
    let n = 2 * l;
    let mut g = Array2::<Complex64>::zeros((n, n));
    for i in 0..l {
        g[[i, i]] = Complex64::new(1.0, 0.0);
    }
    g
}

fn compute_z_values(g: &Array2<Complex64>, l: usize) -> Array1<f64> {
    Array1::from_iter((0..l).map(|i| {
        let g_ii_re = g[[i, i]].re.clamp(0.0, 1.0);
        2.0 * g_ii_re - 1.0
    }))
}

fn hamiltonian_step(g: &mut Array2<Complex64>, h: &Array2<Complex64>, dt: f64) {
    // G += -2i * dt * (G@h - h@G)
    let gh = g.dot(h);
    let hg = h.dot(&*g);
    let factor = Complex64::new(0.0, -2.0 * dt);
    let n = g.nrows();
    for i in 0..n {
        for j in 0..n {
            g[[i, j]] += factor * (gh[[i, j]] - hg[[i, j]]);
        }
    }
}

/// Core simulation; returns (Q, z_traj (N+1)×L, xi_traj N×L).
pub fn simulate_trajectory_internal(
    sim: &RustLQubitSimulator,
    mut rng: StdRng,
) -> (f64, Array2<f64>, Array2<i32>) {
    let l = sim.l;
    let n_steps = sim.n_steps;

    let mut g = sim.g_initial.clone();
    let mut z_traj = Array2::<f64>::zeros((n_steps + 1, l));
    let mut xi_traj = Array2::<i32>::zeros((n_steps, l));

    // Record initial z
    let z0 = compute_z_values(&g, l);
    z_traj.row_mut(0).assign(&z0);

    let mut q = 0.0_f64;

    for step in 0..n_steps {
        // Hamiltonian step
        hamiltonian_step(&mut g, &sim.h, sim.dt);

        // Sample xi: ±1 for each site
        let xi: Vec<i32> = (0..l)
            .map(|_| if rng.gen::<bool>() { 1 } else { -1 })
            .collect();

        // Always symmetrise after measurement (match Python behavior)
        measurement_step_fused(&mut g, &xi, sim.epsilon, true);

        for (k, &xk) in xi.iter().enumerate() {
            xi_traj[[step, k]] = xk;
        }

        let z_after = compute_z_values(&g, l);

        // Accumulate entropy production (Stratonovich)
        let z_before = z_traj.row(step);
        for i in 0..l {
            let avg_z = 0.5 * (z_before[i] + z_after[i]);
            q += 2.0 * sim.epsilon * sim.epsilon * z_before[i] * avg_z;
            q += 2.0 * sim.epsilon * xi[i] as f64 * avg_z;
        }

        z_traj.row_mut(step + 1).assign(&z_after);
    }

    (q, z_traj, xi_traj)
}
