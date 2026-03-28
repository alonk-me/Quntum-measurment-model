## [0.1.2] - 2026-03-22
### Fixed
- Disabled projector eigenvalue snap by default in stable integrator mode; snap was designed for Euler-drift repair and erases physical interior eigenvalue dynamics when applied on the exact unitary path.

## [0.1.1] - 2026-03-22
### Fixed
- Reduced stable-integrator BdG enforcement from three unconditional applications per step to a single post-measurement application, and added an optional `bdg_enforce_threshold` lazy-enforcement gate (off by default) to avoid over-regularization at larger system sizes.
