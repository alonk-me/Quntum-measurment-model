# Work Breakdown Structure: Susceptibility Analysis System

**Project**: Compute χₙ(γ,L) and estimate γc for monitored free-fermion chains  
**Date**: January 10, 2026  
**Priority Order**: Refactor → Susceptibility → Checkpointing → Peak Finding → Visualization → Documentation Notebook → Validation → Execution

---

## 1. REFACTOR EXISTING CODE

### 1.1 Extract Core Functions
- [ ] **1.1.1** Create `quantum_measurement/analysis/` directory
  - [ ] Add `__init__.py`
  - [ ] Add module docstrings
- [ ] **1.1.2** Create `quantum_measurement/analysis/steady_state.py`
  - [ ] Extract logic from `run_ninf_scan.py`
  - [ ] Implement `compute_n_inf(gamma, L, **params)` wrapper
  - [ ] Return value: `float` or `(float, dict)` with diagnostics
  - [ ] Add comprehensive docstrings with parameter descriptions
- [ ] **1.1.3** Enhance convergence diagnostics
  - [ ] Add `t_sat` (saturation time) calculation
  - [ ] Add `final_norm` from correlation matrix trace
  - [ ] Add `final_hermiticity_error` metric
  - [ ] Collect warnings list (numerical instability, etc.)
- [ ] **1.1.4** Refactor `run_ninf_scan.py` to use new API
  - [ ] Replace inline simulation code with `compute_n_inf()` calls
  - [ ] Maintain backward compatibility
  - [ ] Update tests to verify equivalence

**Duration**: 2 days  
**Dependencies**: None  
**Deliverables**: 
- `quantum_measurement/analysis/steady_state.py`
- Updated `scripts/run_ninf_scan.py`

---

## 2. IMPLEMENT SUSCEPTIBILITY

### 2.1 Core Susceptibility Module
- [ ] **2.1.1** Create `quantum_measurement/jw_expansion/susceptibility.py`
  - [ ] Add module-level documentation
  - [ ] Import required dependencies (numpy, pandas, scipy)
- [ ] **2.1.2** Implement `compute_chi_n()` function
  - [ ] Central difference implementation: `(n⁺ - n⁻)/(2*dg)`
  - [ ] Forward difference fallback for boundaries
  - [ ] Adaptive dg selection logic
  - [ ] Error estimation: `σ_χ ≈ √(σ₊² + σ₋²)/(2*dg)`
  - [ ] Return `(chi_n, chi_n_error)` tuple
- [ ] **2.1.3** Implement dg selection strategies
  - [ ] **Strategy 1**: Fixed relative `dg = max(1e-3, 0.01*gamma)`
  - [ ] **Strategy 2**: Adaptive with dg/2 validation
  - [ ] Parameter: `method='fixed'|'adaptive'`
- [ ] **2.1.4** Create unit tests
  - [ ] Test against analytical derivatives for small L
  - [ ] Test dg sensitivity (compare dg vs dg/2)
  - [ ] Test error propagation formulas
  - [ ] Test boundary cases (gamma → 0, gamma → ∞)

### 2.2 Batch Susceptibility Computation
- [ ] **2.2.1** Implement `compute_chi_n_scan()`
  - [ ] Accept `gamma_grid`, `L_list` arrays
  - [ ] Accept optional `dg_dict` for per-L step sizes
  - [ ] Return pandas DataFrame with columns:
    - `L, gamma, g, n_inf, chi_n, chi_n_error, dg_used, converged_all`
- [ ] **2.2.2** Integrate with result cache (see Section 3)
  - [ ] Query cache before computing new n_inf values
  - [ ] Store computed values for reuse
- [ ] **2.2.3** Add progress reporting
  - [ ] Log completion percentage
  - [ ] Estimated time remaining
  - [ ] Current (gamma, L) being processed

**Duration**: 3 days  
**Dependencies**: Section 1 (steady_state.py)  
**Deliverables**: 
- `quantum_measurement/jw_expansion/susceptibility.py`
- `tests/test_susceptibility.py`

---

## 3. CHECKPOINTING & CACHING

### 3.1 Result Cache Infrastructure
- [ ] **3.1.1** Create `quantum_measurement/utilities/cache.py`
  - [ ] Implement `ResultCache` class using HDF5
  - [ ] Methods: `get_n_inf()`, `store_n_inf()`, `has_key()`
  - [ ] Hierarchical structure: `/L{L}/gamma{gamma:.6e}`
  - [ ] Store diagnostics as HDF5 attributes
- [ ] **3.1.2** Add cache validation
  - [ ] Check cache integrity on load
  - [ ] Verify schema compatibility
  - [ ] Handle corrupted cache gracefully
- [ ] **3.1.3** Implement metadata tracking
  - [ ] Store run parameters (dt, tolerance, etc.)
  - [ ] Store creation timestamp
  - [ ] Store code version/git commit hash

### 3.2 CSV Checkpointing
- [ ] **3.2.1** Enhance CSV output format
  - [ ] Add chi_n-specific columns to header
  - [ ] Include dg_used, chi_n_error columns
- [ ] **3.2.2** Implement incremental append
  - [ ] Write after each (gamma, L) chi_n computation
  - [ ] Atomic write to prevent corruption
- [ ] **3.2.3** Resume capability
  - [ ] Load existing CSV on startup
  - [ ] Build set of completed (gamma, L) pairs
  - [ ] Skip redundant computations

### 3.3 Output Directory Structure
- [ ] **3.3.1** Create organized results hierarchy
  ```
  results/susceptibility_scan/
  ├── n_inf_cache.h5
  ├── chi_n_results_{timestamp}.csv
  ├── metadata.json
  └── checkpoints/
  ```
- [ ] **3.3.2** Implement automatic backup
  - [ ] Copy cache file periodically
  - [ ] Retain last 3 backups

**Duration**: 2 days  
**Dependencies**: Section 2.1 (core susceptibility)  
**Deliverables**: 
- `quantum_measurement/utilities/cache.py`
- Enhanced CSV workflow
- `tests/test_cache.py`

---

## 4. PEAK FINDING & CRITICAL POINT ANALYSIS

### 4.1 Peak Finding Module
- [ ] **4.1.1** Create `quantum_measurement/analysis/critical_point.py`
  - [ ] Module documentation on finite-size scaling
- [ ] **4.1.2** Implement `find_chi_peak()` function
  - [ ] **Method 1**: Simple maximum of |χₙ(γ)|
  - [ ] **Method 2**: Gaussian fit around maximum
  - [ ] **Method 3**: Spline interpolation + root finding
  - [ ] Return `(gamma_peak, gamma_peak_error)`
- [ ] **4.1.3** Add smoothing preprocessing
  - [ ] Gaussian filter with adaptive sigma
  - [ ] Outlier removal (3-sigma clipping)
  - [ ] Handle noisy data gracefully

### 4.2 Finite-Size Extrapolation
- [ ] **4.2.1** Implement `estimate_gamma_c()` function
  - [ ] Accept dict/DataFrame of {L: gamma_peak}
  - [ ] Filter L ≥ L_min (default 33)
  - [ ] Fit model: `gamma_peak(L) = gamma_c - a/L^(1/ν)`
- [ ] **4.2.2** Support multiple extrapolation models
  - [ ] Linear: `gamma_peak = gamma_c - a/L`
  - [ ] Power law: `gamma_peak = gamma_c - a/L^b`
  - [ ] Custom user function
- [ ] **4.2.3** Error propagation
  - [ ] Bootstrap resampling for confidence intervals
  - [ ] Covariance matrix from curve_fit
  - [ ] Report gamma_c ± error
- [ ] **4.2.4** Goodness-of-fit metrics
  - [ ] R² value
  - [ ] Residual analysis
  - [ ] Chi-squared statistic

### 4.3 Command-Line Tool
- [ ] **4.3.1** Create `scripts/estimate_gamma_c.py`
  - [ ] Accept CSV input path
  - [ ] Command-line arguments for L_min, extrapolation method
  - [ ] Print summary report to stdout
  - [ ] Save detailed results to JSON
- [ ] **4.3.2** Generate diagnostic plots
  - [ ] gamma_peak vs 1/L scatter + fit line
  - [ ] Residuals plot
  - [ ] Confidence bands

**Duration**: 2 days  
**Dependencies**: Section 2 (susceptibility data)  
**Deliverables**: 
- `quantum_measurement/analysis/critical_point.py`
- `scripts/estimate_gamma_c.py`
- `tests/test_critical_analysis.py`

---

## 5. VISUALIZATION

### 5.1 Susceptibility Plotting
- [ ] **5.1.1** Create `scripts/plot_susceptibility.py`
  - [ ] CLI arguments for input CSV, output directory
  - [ ] Publication-quality settings (300 DPI, proper fonts)
- [ ] **5.1.2** Implement chi_n vs gamma plot
  - [ ] Overlay curves for all L values
  - [ ] Color scheme: viridis or colorblind-friendly
  - [ ] Highlight peak positions with markers
  - [ ] Include error bars (chi_n_error)
  - [ ] Legend with L values
- [ ] **5.1.3** Critical region zoom plot
  - [ ] Focus on g ∈ [0.6, 1.4] (gamma ∈ [2.4, 5.6])
  - [ ] Vertical line at suspected gamma_c
  - [ ] Dense grid for smooth curves
- [ ] **5.1.4** Finite-size scaling plot
  - [ ] gamma_peak vs 1/L scatter
  - [ ] Fitted line/curve
  - [ ] Extrapolation to L → ∞
  - [ ] Shaded confidence region
  - [ ] Annotate gamma_c estimate

### 5.2 Live Progress Monitoring
- [ ] **5.2.1** Enhance `scripts/plot_progress.py`
  - [ ] Detect chi_n scan CSV format
  - [ ] Plot partial chi_n curves as data arrives
  - [ ] Show current peak position estimates
  - [ ] Display completion matrix (L vs gamma grid)
- [ ] **5.2.2** Add summary statistics panel
  - [ ] Total points computed / total needed
  - [ ] Average runtime per point
  - [ ] ETA for completion
  - [ ] Convergence success rate

### 5.3 Comparison Plots
- [ ] **5.3.1** chi_n vs analytical derivatives
  - [ ] Compute d/dg of sum_pbc numerically
  - [ ] Plot simulation vs analytical for small L
  - [ ] Quantify agreement (correlation, RMSE)
- [ ] **5.3.2** Error analysis plots
  - [ ] chi_n_error vs gamma
  - [ ] dg sensitivity plot (chi vs dg step size)
  - [ ] Convergence diagnostics distribution

**Duration**: 2 days  
**Dependencies**: Section 2 (susceptibility data), Section 4 (peak finding)  
**Deliverables**: 
- `scripts/plot_susceptibility.py`
- Enhanced `scripts/plot_progress.py`
- Gallery of publication-ready figures

---

## 6. DOCUMENTATION NOTEBOOK

### 6.1 Tutorial Notebook Creation
- [ ] **6.1.1** Create `notebooks/susceptibility_tutorial.ipynb`
  - [ ] Clear title and overview section
  - [ ] Table of contents
  - [ ] Learning objectives listed
- [ ] **6.1.2** Section: Introduction to Susceptibility
  - [ ] Physics motivation (measurement-induced transitions)
  - [ ] Mathematical definition: χₙ = ∂⟨n⟩∞/∂γ
  - [ ] Why χₙ locates γc better than other observables
  - [ ] Finite-size scaling theory
- [ ] **6.1.3** Section: Code Architecture Overview
  - [ ] Diagram of module dependencies
  - [ ] Explain refactored structure
  - [ ] Key functions and their roles
  - [ ] Cache and checkpointing workflow

### 6.2 Hands-On Examples
- [ ] **6.2.1** Example 1: Single χₙ calculation
  - [ ] Import necessary modules
  - [ ] Set L=17, gamma=4.0
  - [ ] Call `compute_chi_n()` with different dg values
  - [ ] Plot convergence of chi_n vs dg
  - [ ] Interpret results
- [ ] **6.2.2** Example 2: Small parameter scan
  - [ ] Define gamma_grid (20 points, g ∈ [0.5, 1.5])
  - [ ] Set L_list = [9, 17, 33]
  - [ ] Run `compute_chi_n_scan()`
  - [ ] Plot chi_n(gamma) curves
  - [ ] Find peaks for each L
- [ ] **6.2.3** Example 3: Using cache
  - [ ] Initialize ResultCache
  - [ ] Pre-populate with n_inf values
  - [ ] Compute chi_n using cached values
  - [ ] Demonstrate speed improvement
  - [ ] Inspect cache contents

### 6.3 Advanced Topics
- [ ] **6.3.1** Adaptive dg selection deep-dive
  - [ ] Compare fixed vs adaptive methods
  - [ ] Show when adaptive is necessary
  - [ ] Trade-offs: accuracy vs computational cost
- [ ] **6.3.2** Finite-size extrapolation walkthrough
  - [ ] Load example chi_n data
  - [ ] Extract peaks for L ∈ [17, 33, 65, 129]
  - [ ] Fit gamma_peak(L) with multiple models
  - [ ] Compare extrapolations
  - [ ] Estimate gamma_c with error bars
- [ ] **6.3.3** Error analysis
  - [ ] Propagation of n_inf errors to chi_n
  - [ ] Bootstrap uncertainty quantification
  - [ ] Sensitivity to convergence tolerance

### 6.4 Production Workflow Guide
- [ ] **6.4.1** Step-by-step production run
  - [ ] Configure experiment grid
  - [ ] Set up tmux session
  - [ ] Launch parallel computation
  - [ ] Monitor with live plots
  - [ ] Detach and resume
- [ ] **6.4.2** Post-processing pipeline
  - [ ] Load results CSV
  - [ ] Estimate gamma_c
  - [ ] Generate publication plots
  - [ ] Export data for external analysis

### 6.5 Troubleshooting & FAQ
- [ ] **6.5.1** Common issues
  - [ ] Convergence failures at specific (gamma, L)
  - [ ] Cache corruption recovery
  - [ ] Memory errors for large L
  - [ ] Numerical instabilities
- [ ] **6.5.2** Performance optimization tips
  - [ ] Choosing n_workers
  - [ ] Balancing cache size vs recomputation
  - [ ] When to use SLURM vs multiprocessing

**Duration**: 3 days  
**Dependencies**: Sections 1-5 (all core functionality)  
**Deliverables**: 
- `notebooks/susceptibility_tutorial.ipynb`
- Rendered HTML/PDF version for sharing
- Example output plots embedded in notebook

---

## 7. VALIDATION

### 7.1 Unit Tests
- [ ] **7.1.1** Test `compute_n_inf()`
  - [ ] Small L (L=5,9) vs analytical sum_pbc
  - [ ] Verify diagnostics dict structure
  - [ ] Test convergence detection logic
  - [ ] Boundary conditions (open vs periodic)
- [ ] **7.1.2** Test `compute_chi_n()`
  - [ ] Analytical derivative comparison (small L)
  - [ ] Central vs forward difference agreement
  - [ ] dg sensitivity (chi(dg) ≈ chi(dg/2))
  - [ ] Error estimate reasonableness
  - [ ] Sign check: chi_n < 0 (monotonic decrease)
- [ ] **7.1.3** Test `ResultCache`
  - [ ] Store and retrieve n_inf values
  - [ ] Handle missing keys gracefully
  - [ ] Concurrent access safety
  - [ ] Cache invalidation on parameter change
- [ ] **7.1.4** Test `find_chi_peak()`
  - [ ] Synthetic Gaussian peak recovery
  - [ ] Multiple peaks (should return global max)
  - [ ] Noisy data robustness
  - [ ] Edge cases (peak at boundary)
- [ ] **7.1.5** Test `estimate_gamma_c()`
  - [ ] Known linear scaling: gamma_peak = 4 - 1/L
  - [ ] Recover gamma_c = 4 with small error
  - [ ] Test with different fit models
  - [ ] Insufficient data handling (< 3 L values)

### 7.2 Integration Tests
- [ ] **7.2.1** End-to-end small scan
  - [ ] gamma_grid: 10 points
  - [ ] L_list: [9, 17]
  - [ ] Run compute_chi_n_scan()
  - [ ] Verify output DataFrame structure
  - [ ] Check all expected columns present
- [ ] **7.2.2** Cache persistence test
  - [ ] Run scan, populate cache
  - [ ] Close and reopen cache
  - [ ] Verify all data retained
  - [ ] Run same scan again (should be instant)
- [ ] **7.2.3** Checkpoint resume test
  - [ ] Start scan with 20 points
  - [ ] Interrupt after 10 points
  - [ ] Resume scan
  - [ ] Verify exactly 10 new computations
  - [ ] Check final CSV has 20 rows

### 7.3 Regression Tests
- [ ] **7.3.1** Create reference dataset
  - [ ] Run validated small scan (L=[9,17,33], 20 gamma points)
  - [ ] Store results as JSON fixture
  - [ ] Include n_inf, chi_n, gamma_peak values
- [ ] **7.3.2** Implement regression test
  - [ ] Load reference fixture
  - [ ] Recompute with current code
  - [ ] Assert values match within tolerance (1%)
  - [ ] Fail test if discrepancy detected
- [ ] **7.3.3** Physics sanity checks
  - [ ] 0 ≤ n_inf ≤ 0.5 always
  - [ ] chi_n < 0 for all gamma > 0
  - [ ] |chi_n| decreases for large gamma
  - [ ] Peaks exist for all L in critical region

### 7.4 Performance Benchmarks
- [ ] **7.4.1** Profile compute_n_inf()
  - [ ] Measure time vs L (log-log plot)
  - [ ] Verify O(L²) scaling per step
  - [ ] Identify bottleneck functions
- [ ] **7.4.2** Profile compute_chi_n_scan()
  - [ ] Measure speedup vs n_workers
  - [ ] Check for I/O bottlenecks
  - [ ] Verify cache effectiveness (hit rate)
- [ ] **7.4.3** Memory profiling
  - [ ] Peak memory vs L
  - [ ] Check for memory leaks in long scans
  - [ ] Validate fallback logic (L=256 → L=129)

### 7.5 Cross-Validation with Literature
- [ ] **7.5.1** Compare n_inf(gamma) curves
  - [ ] Find reference data from papers (if available)
  - [ ] Overlay our results
  - [ ] Quantify agreement
- [ ] **7.5.2** Compare gamma_c estimate
  - [ ] Check literature value for XX chain with monitoring
  - [ ] Compare our extrapolation
  - [ ] Discuss discrepancies in documentation

**Duration**: 3 days  
**Dependencies**: Sections 1-6 (all implementation complete)  
**Deliverables**: 
- `tests/test_susceptibility.py`
- `tests/test_critical_analysis.py`
- `tests/test_integration.py`
- `tests/fixtures/reference_chi_n.json`
- Test coverage report (>80% target)

---

## 8. SIMULATION EXECUTION

### 8.1 Production CLI Script
- [ ] **8.1.1** Create `scripts/run_susceptibility_scan.py`
  - [ ] Argparse interface for all parameters
  - [ ] Arguments: --L, --gamma-range, --n-gamma, --workers
  - [ ] Arguments: --resume, --output-dir, --cache-file
  - [ ] Validate inputs before starting
- [ ] **8.1.2** Implement main execution loop
  - [ ] Load or create cache
  - [ ] Build gamma_grid and L_list
  - [ ] Call compute_chi_n_scan() with parallelization
  - [ ] Write results incrementally to CSV
  - [ ] Handle KeyboardInterrupt gracefully
- [ ] **8.1.3** Add logging
  - [ ] Dual output: stdout + file
  - [ ] Separate error log
  - [ ] Timestamp all messages
  - [ ] Log level control (--verbose flag)
- [ ] **8.1.4** Help and documentation
  - [ ] Comprehensive --help text
  - [ ] Usage examples in docstring
  - [ ] Link to tutorial notebook

### 8.2 Experiment Configuration
- [ ] **8.2.1** Define production parameter grid
  ```python
  L_list = [9, 17, 33, 65, 129, 257]  # or fallback to 129
  gamma_critical = np.linspace(2.4, 5.6, 200)  # dense near g=1
  gamma_flanks = np.concatenate([
      np.linspace(0.5, 2.4, 30),
      np.linspace(5.6, 10.0, 40)
  ])
  gamma_grid = np.sort(np.concatenate([gamma_critical, gamma_flanks]))
  ```
- [ ] **8.2.2** Estimate computational cost
  - [ ] Total (gamma, L) pairs: ~1,620
  - [ ] With chi_n (3 points each): ~4,860 n_inf calls
  - [ ] Deduplication: ~1,500 unique
  - [ ] Time estimate: 5-6 hours @ 8 workers
- [ ] **8.2.3** Create configuration file
  - [ ] YAML or JSON with all parameters
  - [ ] Load via --config argument
  - [ ] Allow override with CLI flags

### 8.3 Remote Execution Setup
- [ ] **8.3.1** Test run (small grid)
  - [ ] L_list = [9, 17, 33]
  - [ ] 20 gamma points
  - [ ] Verify end-to-end workflow
  - [ ] Check output quality
- [ ] **8.3.2** tmux session script
  ```bash
  #!/bin/bash
  # create_tmux_session.sh
  tmux new-session -d -s chi_scan
  tmux send-keys -t chi_scan 'cd /path/to/project' C-m
  tmux send-keys -t chi_scan 'python scripts/run_susceptibility_scan.py --workers 8' C-m
  tmux split-window -t chi_scan -h
  tmux send-keys -t chi_scan.1 'python scripts/plot_progress.py' C-m
  tmux attach -t chi_scan
  ```
- [ ] **8.3.3** Alternative: nohup execution
  ```bash
  nohup python scripts/run_susceptibility_scan.py --workers 8 \
    > logs/chi_scan_$(date +%Y%m%d_%H%M%S).log 2>&1 &
  echo $! > logs/chi_scan.pid
  ```

### 8.4 Production Run Execution
- [ ] **8.4.1** Pre-flight checks
  - [ ] Verify Python environment activated
  - [ ] Check available disk space (>10 GB)
  - [ ] Check available memory (>8 GB)
  - [ ] Test cache I/O permissions
- [ ] **8.4.2** Launch production scan
  - [ ] Start tmux session or nohup
  - [ ] Monitor initial progress (first 10 points)
  - [ ] Verify CSV output format
  - [ ] Check cache file growth
- [ ] **8.4.3** Monitoring during run
  - [ ] Check plot_progress.py updates (every 30s)
  - [ ] Monitor system resources (htop, free -h)
  - [ ] Inspect logs for errors
  - [ ] Verify convergence success rate >90%
- [ ] **8.4.4** Post-completion
  - [ ] Verify total rows in CSV matches expected
  - [ ] Check for missing (L, gamma) combinations
  - [ ] Backup results and cache files
  - [ ] Run estimate_gamma_c.py
  - [ ] Generate all plots with plot_susceptibility.py

### 8.5 Data Analysis & Reporting
- [ ] **8.5.1** Load and inspect results
  - [ ] Read CSV into pandas DataFrame
  - [ ] Check for NaN or inf values
  - [ ] Verify chi_n signs (should be negative)
  - [ ] Compute summary statistics
- [ ] **8.5.2** Extract gamma_c
  - [ ] Run scripts/estimate_gamma_c.py
  - [ ] Review fit quality (R², residuals)
  - [ ] Compare linear vs power-law extrapolation
  - [ ] Document final gamma_c ± error
- [ ] **8.5.3** Generate publication figures
  - [ ] chi_n vs gamma for all L
  - [ ] Critical region zoom
  - [ ] Finite-size scaling collapse
  - [ ] gamma_peak vs 1/L extrapolation
- [ ] **8.5.4** Write results summary
  - [ ] Key findings (gamma_c value, exponents)
  - [ ] Comparison with analytical predictions
  - [ ] Numerical convergence quality
  - [ ] Recommendations for future work

**Duration**: 1 day setup + 5-6 hours compute + 1 day analysis = ~3 days total  
**Dependencies**: All previous sections  
**Deliverables**: 
- `scripts/run_susceptibility_scan.py`
- Production results CSV
- Cached n_inf database (HDF5)
- gamma_c estimate report
- Publication-quality plots
- Results summary document

---

## PROJECT TIMELINE SUMMARY

| Phase | Duration | Dependencies | Key Deliverables |
|-------|----------|--------------|------------------|
| 1. Refactor | 2 days | None | steady_state.py, refactored scripts |
| 2. Susceptibility | 3 days | Phase 1 | susceptibility.py, unit tests |
| 3. Checkpointing | 2 days | Phase 2 | cache.py, enhanced CSV |
| 4. Peak Finding | 2 days | Phase 2 | critical_point.py, estimate_gamma_c.py |
| 5. Visualization | 2 days | Phases 2,4 | plot_susceptibility.py, enhanced monitoring |
| 6. Documentation | 3 days | Phases 1-5 | susceptibility_tutorial.ipynb |
| 7. Validation | 3 days | Phases 1-6 | Comprehensive test suite |
| 8. Execution | 3 days | All | Production results, gamma_c report |

**Total Estimated Duration**: 20 working days (~4 weeks)

---

## RISK MITIGATION

### Technical Risks
- **Risk**: Numerical instabilities at extreme gamma values
  - **Mitigation**: Extensive validation, graceful error handling, parameter bounds
- **Risk**: Memory overflow for L=257
  - **Mitigation**: Automatic fallback to L=129, streaming trajectories
- **Risk**: Poor convergence for some (gamma, L) pairs
  - **Mitigation**: Adaptive max_steps, multiple tolerance levels, retry logic

### Schedule Risks
- **Risk**: Longer compute times than estimated
  - **Mitigation**: Start with small test grid, optimize bottlenecks first, parallel execution
- **Risk**: Complex bugs in cache system
  - **Mitigation**: Thorough unit tests early, simple fallback to CSV-only mode

### Data Quality Risks
- **Risk**: Insufficient resolution in critical region
  - **Mitigation**: Dense gamma_grid near g=1, adaptive refinement if needed
- **Risk**: Finite-size effects too strong
  - **Mitigation**: Include large L (129, 257), multiple extrapolation models

---

## SUCCESS CRITERIA

### Technical Success
- [ ] All unit tests pass (>80% coverage)
- [ ] Integration tests pass end-to-end
- [ ] Regression tests confirm backward compatibility
- [ ] Production run completes without intervention
- [ ] Chi_n curves are smooth and physical

### Scientific Success
- [ ] Gamma_c extracted with <5% error bars
- [ ] Finite-size scaling collapse achieved
- [ ] Results consistent with analytical predictions (where available)
- [ ] Publication-quality figures generated

### Documentation Success
- [ ] Tutorial notebook runs without errors
- [ ] All functions have comprehensive docstrings
- [ ] User can reproduce results from README
- [ ] Code is maintainable by others

---

## NOTES & ASSUMPTIONS

1. **Computational Resources**: Assumes 8-core CPU, 16GB RAM, sufficient disk space
2. **Python Environment**: Python 3.8+, standard scientific stack (numpy, scipy, pandas, matplotlib)
3. **Dependencies**: h5py for caching, joblib optional for parallelization
4. **Domain Knowledge**: User familiar with correlation matrix evolution, Jordan-Wigner transformation
5. **Validation Data**: May need to generate own "ground truth" if literature data unavailable

---

**Last Updated**: January 10, 2026  
**Status**: Planning Complete - Ready for Implementation
