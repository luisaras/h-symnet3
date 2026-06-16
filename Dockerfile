# Use an older Debian Buster layer to natively support the legacy .so memory alignment
FROM docker.io/library/python:3.10-slim-buster

# CRITICAL FIX: Rewrite the old EOL repositories to use the Debian archive servers for Buster
RUN sed -i 's/deb.debian.org/archive.debian.org/g' /etc/apt/sources.list && \
    sed -i 's|security.debian.org/debian-security|archive.debian.org/debian-security|g' /etc/apt/sources.list && \
    sed -i '/buster-updates/d' /etc/apt/sources.list

# Install the matching legacy background packages using the archive mirror, ignoring expired security flags
RUN apt-get update -o Acquire::Check-Valid-Until=false && \
    apt-get install -y --allow-unauthenticated \
    libbdd-dev \
    && rm -rf /var/lib/apt/lists/*

# Install uv inside the image for package management
RUN pip install --no-cache-dir uv

WORKDIR /app

# Copy your requirements file
COPY requirements.txt .

# Pre-install all your requirements and NVIDIA CUDA runtimes into the image system layer
RUN uv pip install --system -r requirements.txt && \
    uv pip install --system nvidia-cuda-runtime-cu11 nvidia-cudnn-cu11 nvidia-cublas-cu11

# Replicate your script's LD_LIBRARY_PATH environment settings globally inside the image
ENV NVIDIA_LIBS=/usr/local/lib/python3.10/site-packages/nvidia
ENV LD_LIBRARY_PATH=${NVIDIA_LIBS}/cuda_runtime/lib:${NVIDIA_LIBS}/cublas/lib:${NVIDIA_LIBS}/cudnn/lib:${LD_LIBRARY_PATH}

ENTRYPOINT ["uv", "run"]
