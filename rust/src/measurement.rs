use ndarray::{Array2, ArrayView2};
use num_complex::Complex64;

/// Compute `G * Xi * G` where Xi = diag(xi_diag) exploiting diagonal sparsity.
/// result[i,j] = sum_k G[i,k] * xi_diag[k] * G[k,j]
///
/// This is equivalent to (G * xi_row_broadcast) @ G, exploiting the fact that
/// Xi is diagonal.  We scale columns of G by xi_diag, then multiply by G.
#[inline]
pub fn triple_product_diagonal(
    g: &ArrayView2<Complex64>,
    xi_diag: &[Complex64],
) -> Array2<Complex64> {
    let n = g.nrows();
    // g_xi[i,j] = g[i,j] * xi_diag[j]  (scale columns)
    let mut g_xi = g.to_owned();
    for j in 0..n {
        let xj = xi_diag[j];
        for i in 0..n {
            g_xi[[i, j]] *= xj;
        }
    }
    // result = g_xi @ g
    g_xi.dot(g)
}

/// Apply the fused measurement step in-place on G.
///
/// Updates:
///   stochastic = epsilon * (G*Xi + Xi*G - 2*G*Xi*G)
///   damping    = -epsilon^2 * (G - diag(G))
///   G += stochastic + damping
///
/// Then Hermitianise and clip diagonal when `symmetrise` is true.
pub fn measurement_step_fused(
    g: &mut Array2<Complex64>,
    xi: &[i32],
    epsilon: f64,
    symmetrise: bool,
) {
    let n = g.nrows();
    let l = xi.len();

    // Build xi_diag: [+xi_0..+xi_{L-1}, -xi_0..-xi_{L-1}]
    let xi_diag: Vec<Complex64> = (0..n)
        .map(|k| {
            if k < l {
                Complex64::new(xi[k] as f64, 0.0)
            } else {
                Complex64::new(-(xi[k - l] as f64), 0.0)
            }
        })
        .collect();

    // Compute G*Xi: column j gets multiplied by xi_diag[j]
    let mut g_xi = g.to_owned();
    for j in 0..n {
        let xj = xi_diag[j];
        for i in 0..n {
            g_xi[[i, j]] *= xj;
        }
    }

    // Compute Xi*G: row i gets multiplied by xi_diag[i]
    let mut xi_g = g.to_owned();
    for i in 0..n {
        let xi_i = xi_diag[i];
        for j in 0..n {
            xi_g[[i, j]] *= xi_i;
        }
    }

    // Compute G*Xi*G using the optimized function
    let g_xi_g = g_xi.dot(&*g);

    let eps = Complex64::new(epsilon, 0.0);
    let eps2 = Complex64::new(epsilon * epsilon, 0.0);
    let two = Complex64::new(2.0, 0.0);

    // Apply update in-place: G += eps*(G_Xi + Xi_G - 2*G_Xi_G) - eps^2*(G - diag(G))
    for i in 0..n {
        for j in 0..n {
            let stoch = eps * (g_xi[[i, j]] + xi_g[[i, j]] - two * g_xi_g[[i, j]]);
            let damp = if i == j {
                Complex64::new(0.0, 0.0)
            } else {
                -eps2 * g[[i, j]]
            };
            g[[i, j]] += stoch + damp;
        }
    }

    if symmetrise {
        hermitianise_and_clip(g);
    }
}

/// Symmetrise G = 0.5*(G + G†) and clip diagonal Re(G[i,i]) ∈ [0,1].
pub fn hermitianise_and_clip(g: &mut Array2<Complex64>) {
    let n = g.nrows();
    // Hermitianise
    for i in 0..n {
        for j in (i + 1)..n {
            let sym = 0.5 * (g[[i, j]] + g[[j, i]].conj());
            g[[i, j]] = sym;
            g[[j, i]] = sym.conj();
        }
        // Force diagonal to be real and in [0,1]
        let re = g[[i, i]].re.clamp(0.0, 1.0);
        g[[i, i]] = Complex64::new(re, 0.0);
    }
}
