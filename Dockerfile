# ── Stage 1: Build the Rust wheel ─────────────────────────────────────────────
# Uses the same Python version as the runtime stage so the compiled extension
# is ABI-compatible with the interpreter that will load it.
FROM python:3.12-slim AS builder

# System packages needed to compile the Rust extension
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install the stable Rust toolchain
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | \
        sh -s -- -y --default-toolchain stable --no-modify-path
ENV PATH="/root/.cargo/bin:${PATH}"

# Install maturin (the Rust↔Python build bridge used by pyproject.toml)
RUN pip install --no-cache-dir "maturin>=1.0"

WORKDIR /build

# Copy Cargo manifests first so dependency downloads are cached as a layer
# that is only invalidated when Cargo.toml / Cargo.lock change.
COPY rust/Cargo.toml rust/Cargo.lock rust/
COPY rust/src/                        rust/src/
COPY rust/.cargo/                     rust/.cargo/
COPY pyproject.toml                   .

# Build an optimised wheel.
# RUSTFLAGS overrides the `target-cpu=native` flag in rust/.cargo/config.toml
# so the resulting shared library is portable across all x86-64 hosts.
# To opt into native CPU optimisations (faster but less portable), pass:
#   docker build --build-arg RUSTFLAGS="" -t quantum-measurements .
ARG RUSTFLAGS="-C target-cpu=x86-64"
RUN RUSTFLAGS="${RUSTFLAGS}" maturin build --release --out /build/dist/


# ── Stage 2: Runtime image ─────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

WORKDIR /app

# Install Python dependencies declared in pyproject.toml
RUN pip install --no-cache-dir \
        numpy \
        matplotlib \
        scipy \
        pandas \
        tqdm

# Install the wheel produced in Stage 1.
# This places the quantum_measurement package (including the compiled Rust
# extension _rust.cpython-*.so) into site-packages.
COPY --from=builder /build/dist/*.whl /tmp/
RUN pip install --no-cache-dir /tmp/*.whl && rm -f /tmp/*.whl

# Copy the full project source into the working directory.
# Scripts and tests resolve quantum_measurement via the local source tree
# (e.g. tests do sys.path.insert(0, .../quantum_measurement/jw_expansion)).
COPY . .

# The compiled Rust extension (.so) is excluded from git (via *cpython* in
# .gitignore) so it is not present in the COPY above.  Copy it from
# site-packages back into the source tree so that imports resolve correctly
# regardless of which entry in sys.path Python resolves the package from.
RUN find /usr/local/lib -name "_rust*.so" -path "*/quantum_measurement/*" \
        -exec cp {} /app/quantum_measurement/ \;

# Smoke-test: confirm the Rust extension loads at image build time
RUN python -c \
    "from quantum_measurement.rust_simulator import FastLQubitSimulator; \
     sim = FastLQubitSimulator(L=3, J=1.0, epsilon=0.1, N_steps=100, T=1.0); \
     Q, z, xi = sim.simulate_trajectory(); \
     print(f'Build verification OK – Q={Q:.3f}, z.shape={z.shape}')"

# Default command: show a brief usage message and run a quick self-test.
# Override with any command when running the container, e.g.:
#   docker run <image> python scripts/benchmark_rust_vs_python.py
#   docker run <image> python -m pytest tests/ -v
CMD ["python", "-c", \
     "from quantum_measurement.rust_simulator import FastLQubitSimulator; \
      print('quantum-measurements ready (Rust-accelerated simulator active)'); \
      sim = FastLQubitSimulator(L=3, J=1.0, epsilon=0.1, N_steps=500, T=1.0); \
      Q, z, xi = sim.simulate_trajectory(); \
      print(f'  Quick test – Q={Q:.3f}, z.shape={z.shape}, xi.shape={xi.shape}')"]
