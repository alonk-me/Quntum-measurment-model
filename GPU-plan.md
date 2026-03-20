# Plan: GPU-Accelerated Quantum Trajectories with PyTorch ML Pipeline

This plan modernizes the quantum measurement simulation codebase by adding GPU acceleration (CuPy backend) for trajectory parallelization, then establishes a PyTorch workflow for phase transition classification.

**Context**: Research shows pure NumPy code with zero parallelization. Main bottlenecks are: (1) time evolution loops (30k-500k matrix ops), (2) serial ensemble loops (100-10k trajectories), (3) serial parameter sweeps (1080 combinations). Expected 50-100× speedup from GPU + parallel sweeps.

**Key decisions**:
- CuPy as opt-in backend, preserving CPU functionality
- Single GPU optimization (not multi-GPU)
- GPU work first, PyTorch ML integration deferred to Phase 2
- Backend abstraction allows future JAX/PyTorch if needed
- PyTorch preferred over PhysicsNeMo/Modulus: task is classification of simulator outputs, not PDE-constrained surrogate modeling; PyTorch/CuPy GPU interop via DLPack enables zero-copy pipeline
- Staged ML approach: MLP on handcrafted features → 1D CNN on correlation profiles → sequence models only if needed

---

## Steps

### Phase 1: Backend Abstraction Layer ✅

1. Create `quantum_measurement/backends/` module with `__init__.py`, `base.py`, `numpy_backend.py`, `cupy_backend.py`
   - `base.py`: Abstract `Backend` protocol defining array ops (`matmul`, `conj`, `transpose`, `zeros`, `random`, etc.)
   - `numpy_backend.py`: Wrapper class implementing protocol with NumPy
   - `cupy_backend.py`: Parallel implementation using CuPy (with graceful import fallback)
   - Factory function `get_backend(device='cpu')` returns appropriate backend

2. Add `backend` parameter to simulator classes in `sse.py`, `non_hermitian_hat.py`, `l_qubit_correlation_simulator.py`, `two_qubit_correlation_simulator.py`
   - Dataclass field: `backend: Backend = field(default_factory=lambda: get_backend('cpu'))`
   - Replace direct NumPy calls (`np.matmul`) with backend calls (`self.backend.matmul`)

3. Update initialization to accept `device='cpu'|'gpu'` parameter that selects backend

4. Add GPU memory profiling utility in `quantum_measurement/utilities/gpu_utils.py`
   - Functions: `check_gpu_available()`, `get_gpu_memory_info()`, `estimate_trajectory_batch_size(L, max_vram_gb)`
   - Batch size calculator for L=256: ~4MB per trajectory, target **60% VRAM usage** (conservative to account for CuPy memory pool overhead and CUDA runtime)
   - This utility is a hard prerequisite before any batching code is written

### Phase 2: Batch Trajectory Execution 🟡

5. Refactor `simulate_trajectory()` methods to support batched execution
   - Current: single trajectory returns `(Q_value, trajectory_data, measurements)`
   - New: `simulate_trajectory_batch(n_batch)` processes `n_batch` trajectories in parallel
   - Key change in evolution loops: expand dims to `(n_batch, 2L, 2L)` for correlation matrices
   - Batch matrix operations: `backend.matmul(G_batch, h)` operates on all trajectories simultaneously
   - Random number generation: batch RNG calls via `backend.random.normal((n_batch, ...))`

6. Update `simulate_ensemble()` in all simulator classes
   - Replace serial `for i in range(n_trajectories)` with batched calls
   - Auto-determine batch size: `batch_size = min(n_trajectories, estimate_trajectory_batch_size(L))` on GPU, else 1 on CPU
   - Loop over batches: `for batch_idx in range(0, n_trajectories, batch_size)`
   - Transfer results from GPU to CPU after each batch (`.get()` for CuPy arrays)
   - Preserve tqdm progress bar functionality

🟡
Note: `non_hermitian_hat.py` currently exposes batched trajectory execution only.
Adding a `simulate_ensemble()` batching wrapper for this simulator was intentionally
omitted from this upgrade and is out of scope for this Phase 2 pass.
🟡

7. Add GPU-specific optimizations in CuPy backend:
   - Fuse operations: commutator + symmetrization in single kernel where possible
   - Pre-allocate device arrays for reuse across time steps
   - Use CuPy's memory pool (`cupy.get_default_memory_pool()`) to reduce allocation overhead
   - Implement in-place ops where mathematically valid

### Phase 3: Parallel Parameter Sweeps ✅

8. Create `quantum_measurement/parallel/` module with `sweep_executor.py`
   - Class `ParameterSweepExecutor` managing parallel execution of (L,γ) grid
   - Support backends: `'multiprocessing'`, `'ray'`, `'sequential'` (default for compatibility)
   - Method `run_sweep(L_values, gamma_grid, simulator_factory, backend_device='cpu')`
   - Checkpoint handling: load existing CSV, skip completed (L,γ) pairs, append new results
    - **Phase 3 scope note**: keep `'ray'` in the public interface, but defer actual Ray execution backend implementation to a follow-up phase.

9. Refactor `run_ninf_scan.py` to use `ParameterSweepExecutor`
   - Replace nested `for L` / `for gamma` loops with executor call
   - Allow CLI args: `--device cpu|gpu`, `--parallel-backend multiprocessing|ray|sequential`, `--n-workers N`, `--n-trajectories N`
   - **GPU worker model: one worker per GPU** — multi-worker GPU sharing is avoided due to VRAM contention; CPU workers handle parallelism and the GPU is reserved for the simulator
   - Workers write to shared results queue, main process handles CSV I/O to avoid race conditions

10. Add dynamic resource allocation:
    - If GPU available: limit concurrent workers to `min(n_workers, gpu_count)` to avoid VRAM overload
    - CPU workers: use `os.cpu_count()` workers via `multiprocessing.Pool`, with L-aware concurrency limits (e.g. `cpu_count // 2` for large L) to avoid RAM exhaustion
    - Fallback: if GPU OOM detected, apply a simple retry policy that reruns the failed (L,γ) task on CPU

### Phase 4: Testing & Validation

11. Create GPU-specific tests in `tests/test_gpu_backend.py`
    - Test backend abstraction: verify NumPy and CuPy return identical results (within numerical tolerance)
    - Test trajectory batching: compare single vs batch execution for consistency
    - Test reproducibility: seeded RNG produces identical results across backends
    - Test memory: verify no memory leaks over 1000 time steps

12. Update existing tests in `test_non_hermitian_simulators.py`, `test_steady_state_integration.py`
    - Parameterize with `device=['cpu', 'gpu']` to run all tests on both backends
    - Add `@pytest.mark.gpu` decorator for GPU-only tests (skipped if CuPy unavailable)

13. Add benchmarking script `scripts/benchmark_gpu_speedup.py`
    - Compare runtimes: CPU vs GPU for varying L values and n_trajectories
    - Generate speedup plots: trajectories/second vs batch size
    - Report optimal batch sizes for different system sizes
    - Include end-to-end timing: parameter sweep with/without parallelization
    - Run profiling pass after Phase 2 before committing to Phase 3 parallelization strategy — bottlenecks may differ from expectations

### Phase 5: Documentation & Config

14. Add GPU installation instructions to `README.md`
    - CuPy installation: `pip install cupy-cuda12x` (adjust for CUDA version)
    - NVIDIA driver requirements, CUDA toolkit version check
    - Verification: `python -c "import cupy; print(cupy.cuda.runtime.getDeviceCount())"`

15. Create `docs/GPU_OPTIMIZATION_GUIDE.md`
    - Overview of backend architecture
    - Usage examples: enable GPU in scripts and notebooks
    - Performance tuning: batch sizes, memory management, mixed precision
    - Troubleshooting: common errors (OOM, CUDA version mismatch)

16. Add configuration file `config.yaml` for simulation defaults using **Hydra** (`pip install hydra-core`)
    - Fields: `device` (cpu/gpu), `parallel_backend`, `n_workers`, `n_trajectories`, `batch_size_override`
    - Hydra handles CLI override precedence cleanly and integrates naturally with Dagster orchestration
    - Load in scripts via Hydra's `@hydra.main` decorator; CLI args take precedence over config file

### Phase 6: ML Data Pipeline

17. Create `quantum_measurement/ml_pipeline/` module
    - `data_generator.py`: Run parameter sweeps, collect trajectory data + labels
    - `feature_extraction.py`: Extract physics-motivated features from trajectories — correlation functions `C(r)`, entanglement entropy proxies, fluctuation statistics of `n_∞(γ)`, correlation length
    - `dataset.py`: Format data as PyTorch `Dataset` subclass; use HDF5 or Zarr for large datasets

18. Define phase transition labeling strategy in `scripts/generate_ml_dataset.py`
    - **Labels must be derived from simulation results, not hardcoded thresholds**: use crossing points of Binder cumulants or peaks of fluctuations per (L,γ) — a fixed γ_c ≈ 1.0 boundary will produce noisy labels near the critical point for small L due to finite-size effects
    - Classes: "subcritical" (γ << γ_c(L)), "critical" (γ ≈ γ_c(L)), "supercritical" (γ >> γ_c(L))
    - Balanced sampling across phase boundaries
    - Output: labeled dataset with train/val/test splits

19. Implement trajectory preprocessing pipeline
    - Subsample time series if needed (500k steps → 10k for model input)
    - Normalize features (z-score or min-max)
    - Augmentation strategies: time-reversal, L-site permutations (if physics-valid)

### Phase 7: PyTorch Model — Staged Architecture

**Philosophy**: start simple, escalate only if needed. A simpler model with interpretable weights is preferable to a sequence model that is data-hungry and opaque near the critical point.

20. **Stage 1 — MLP on handcrafted features** (implement first)
    - Input: physics-motivated feature vector — variance of `n_∞(γ)`, extracted correlation length, entanglement entropy proxy, peak fluctuation magnitude
    - Architecture: 3–4 layer MLP with BatchNorm and dropout
    - Training script: `quantum_measurement/ml_pipeline/train_mlp.py`
    - Target: >90% validation accuracy on held-out (L,γ) pairs
    - Benefit: fast iteration, interpretable weights reveal which features are physically informative
    - If Stage 1 achieves target accuracy, **stop here**

21. **Stage 2 — 1D CNN on spatial correlation profile** (only if Stage 1 underperforms)
    - Input: steady-state correlation function `C(r)` as 1D spatial signal of length L
    - Architecture: 3–4 layer 1D CNN with global average pooling, then MLP head
    - Captures translational structure in `C(r)` without requiring full sequence modeling
    - Training script: `quantum_measurement/ml_pipeline/train_cnn.py`

22. **Stage 3 — Sequence model on trajectory time series** (last resort only)
    - Input: `(n_steps, L)` correlation/occupation data per trajectory
    - Architecture: Transformer encoder or CNN-LSTM
    - Use only if Stages 1–2 fail — sequence models are data-hungry and training signal near γ_c is inherently noisy
    - Training script: `quantum_measurement/ml_pipeline/train_seq.py`

23. CuPy → PyTorch zero-copy pipeline (implement alongside Stage 1+):
    - Use `torch.utils.dlpack` to convert CuPy arrays to PyTorch tensors with no CPU round-trips
    - **CUDA stream synchronization required**: CuPy simulation and PyTorch model may run on different streams; use `cupy.cuda.stream.get_current_stream().synchronize()` before DLPack transfer
    - Implement in `quantum_measurement/ml_pipeline/gpu_pipeline.py` as an optional fast path; CPU fallback always available

### Phase 8: Inference & Integration

24. Create inference workflow in `quantum_measurement/ml_pipeline/inference.py`
    - Method: given new (L,γ), run short simulation (fewer trajectories), extract features, classify phase
    - Uncertainty quantification: MC Dropout or ensemble of models for confidence near γ_c
    - Integration with `run_ninf_scan.py`: `--ml-surrogate` flag uses trained model for initial phase identification, then runs full simulation only near predicted γ_c

25. Create `docs/ML_PIPELINE_GUIDE.md`
    - Model architecture rationale and staged escalation logic
    - Feature extraction details and physics motivation
    - Training procedure, hyperparameter choices
    - Inference integration with parameter sweep scripts

---

## Verification

After Phase 1–3 (GPU acceleration):
- Readiness gate: verify `ParameterSweepExecutor` GPU worker/device detection and OOM fallback paths are healthy before long runs
- Run `pytest tests/test_gpu_backend.py tests/test_sweep_executor.py`: all tests pass
- Execute `python scripts/benchmark_gpu_speedup.py --simulator l_qubit --L 32 --n-steps 200 --n-trajectories 128 --batch-sizes 1 2 4 8 16 32 --repeats 3`
- Run small parameter sweep using existing CLI flags:
    `python scripts/run_ninf_scan.py --device gpu --parallel-backend sequential --l-values 9 17 --gamma-values 0.4 1.0 4.0 --skip-plots`
- Run CPU baseline with the same grid:
    `python scripts/run_ninf_scan.py --device cpu --parallel-backend sequential --l-values 9 17 --gamma-values 0.4 1.0 4.0 --skip-plots`
- Compare GPU vs CPU outputs on shared `(L,γ)` points; require |Δn_∞|/max(|n_∞|,1e-12) ≤ 1%
- Dockerized option (recommended for environment reproducibility):
    `docker compose -f docker-compose.gpu.yml build`
    `docker compose -f docker-compose.gpu.yml run --rm qm-gpu-verify`


## Post-Phase-3 Report Card ✅

- Correctness threshold (<=1%): **PASS**
- Overlap points compared: **6**
- Max relative difference: **3.367e-15**
- Mean relative difference: **1.021e-15**
- Small-sweep runtime ratio (GPU/CPU): **10.69x**
- Best benchmark speedup: **1.14x** at batch size **32**


## Post-Phase-3 Overnight Benchmark + Visual Monitoring Plan

Goal: complete a predictable one-night benchmark run (8-12h target) while making progress visible via terminal and live plot outputs.

1. Overnight default profile (GPU-first, one night)
        - `device_scope = gpu` (default)
        - `L_values = [64]`
        - `n_trajectories = [512, 1024]`
        - `batch_sizes = [16, 32, 64]`
        - `gammas = [0.4, 4.0]` (weak + critical)
        - **`n_steps` starts at `10000`** (fixed overnight baseline)
        - `repeats = 3`
        - Expected rows before pruning/skips: `12`

2. ETA guardrails
        - After first 2-3 completed rows, estimate runtime from observed `campaign_runtime_sec/rows`
        - If projected finish is above 12 hours, apply fallback before reducing physics coverage:
            - fallback `n_trajectories = [256, 512]`
        - Keep VRAM guard active (`estimate_trajectory_batch_size`) and skip oversized batch rows rather than crashing.

3. Disconnect-safe execution
        - Start in background:
            - `./scripts/monitor_long_gpu_benchmark.sh start`
        - Check process states:
            - `./scripts/monitor_long_gpu_benchmark.sh status`
        - Stop campaign cleanly:
            - `./scripts/monitor_long_gpu_benchmark.sh stop`

4. Visual monitoring controls (hybrid mode)
        - Terminal progress card (rows, rows/hour, ETA, latest tuple):
            - `./scripts/monitor_long_gpu_benchmark.sh view`
        - Start live PNG refresher (default 30s):
            - `./scripts/monitor_long_gpu_benchmark.sh plot --interval 30`
        - Stop live PNG refresher:
            - `./scripts/monitor_long_gpu_benchmark.sh plot-stop`
        - PNG output path:
            - `results/test_scan/gpu_benchmark_live_progress.png`
        - Optional log stream:
            - `./scripts/monitor_long_gpu_benchmark.sh logs`
            - `./scripts/monitor_long_gpu_benchmark.sh log-on`
            - `./scripts/monitor_long_gpu_benchmark.sh log-off`

5. Troubleshooting notes
        - Empty or delayed log output can happen due stdout buffering in background mode.
        - Treat CSV growth and visual monitor updates as source-of-truth for progress.

6. Extended profile (non-default, <=24h target)
        - Use only when overnight baseline is stable.
        - Expand to `L_values = [64, 128]` and/or add `n_steps = 30000` with reduced trajectory counts.

7. Acceptance criteria
        - Campaign remains detached and stable after terminal disconnect.
        - ETA remains within overnight target after early-row projection.
        - No OOM crashes for retained rows (skip-pruning is acceptable).
        - Live PNG and terminal progress both refresh during run.


After Phase 6 (ML data pipeline):
- Generate test dataset: 1000 trajectories with balanced class labels
- Verify HDF5 file structure and feature dimensions
- Confirm labels are derived from Binder cumulant crossings, not fixed γ threshold
- Visualize feature distributions per phase class

After Phase 7 (PyTorch models):
- Stage 1 MLP: train on synthetic dataset, achieve >90% validation accuracy
- Test on held-out (L,γ) combinations not in training set
- Compare ML-predicted phase boundaries with known physics (γ_c ≈ 1, L-dependent)
- Only proceed to Stage 2 if Stage 1 fails to meet accuracy target

---

## Decisions

- **Chose CuPy over JAX**: Simpler drop-in replacement, no need for autodiff in simulation phase
- **PyTorch over PhysicsNeMo/Modulus**: Task is classification of simulator outputs, not PDE-constrained surrogate modeling; PhysicsNeMo's PINN framework adds friction without value here. PyTorch gives autodiff for free (enables gradient-based inference later), clean CuPy interop via DLPack, and standard supervised learning workflows
- **Staged ML architecture**: MLP → CNN → sequence model; avoids data-hungry sequence models unless strictly necessary; interpretable MLP weights have scientific value
- **Physics-derived labels**: Binder cumulant crossings per L, not hardcoded γ_c; critical for label quality near the phase boundary under finite-size scaling
- **Hydra for config**: Handles CLI override precedence cleanly; integrates with Dagster orchestration
- **Backend abstraction**: Future-proofs for JAX/PyTorch if gradients needed in simulation
- **Batching over kernel fusion**: Easier implementation, still captures most GPU gains
- **Conservative VRAM target (60%)**: Accounts for CuPy memory pool and CUDA runtime overhead
- **One GPU worker model**: Multi-worker GPU sharing avoided; CPU handles parallelism, GPU reserved for simulator
- **Checkpoint preservation**: Maintain CSV-based checkpointing for reliability
- **Deferred multi-GPU**: Single GPU sufficient for now, avoids complexity of data distribution
