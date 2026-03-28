FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_NO_CACHE_DIR=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    build-essential \
    ca-certificates \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1

WORKDIR /workspace

COPY pyproject.toml README.md pytest.ini ./
COPY quantum_measurement ./quantum_measurement
COPY scripts ./scripts
COPY tests ./tests

RUN python -m pip install --upgrade pip setuptools wheel
RUN python -m pip install -e ".[dev,gpu,progress]"

CMD ["bash"]
