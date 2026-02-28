FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/app/hf_cache

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    git \
    wget \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip first
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel

# Clone the repo
RUN git clone https://github.com/fashn-AI/fashn-vton-1.5.git /app

# Step 1: Install PyTorch explicitly first (prevents version conflicts)
RUN pip3 install --no-cache-dir \
    torch==2.1.0 \
    torchvision==0.16.0 \
    --index-url https://download.pytorch.org/whl/cu121

# Step 2: Install the fashn-vton package and its dependencies
RUN pip3 install --no-cache-dir -e .

# Step 3: Swap CPU onnxruntime for GPU version
RUN pip3 uninstall -y onnxruntime || true
RUN pip3 install --no-cache-dir onnxruntime-gpu

# Step 4: Install runpod
RUN pip3 install --no-cache-dir runpod

# Copy handler
COPY handler.py /app/handler.py

CMD ["python3", "-u", "/app/handler.py"]