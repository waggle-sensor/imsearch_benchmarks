# NVIDIA PyTorch base image (includes CUDA + a CUDA-enabled torch build)
FROM nvcr.io/nvidia/pytorch:24.01-py3

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# keep pip fresh
RUN python -m pip install --upgrade pip

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Sanity check: make sure we have CUDA-enabled torch in the image
RUN python -c "import torch; print('torch', torch.__version__); print('cuda', torch.version.cuda); print('is_available', torch.cuda.is_available())"

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Default command (can be overridden)
CMD ["/bin/bash"]

