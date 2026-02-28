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
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Clone the repo
RUN git clone https://github.com/fashn-AI/fashn-vton-1.5.git /app

# Install Python dependencies
RUN pip3 install --no-cache-dir -e . && \
    pip3 uninstall -y onnxruntime && \
    pip3 install --no-cache-dir onnxruntime-gpu runpod

# Copy handler
COPY handler.py /app/handler.py

CMD ["python3", "-u", "/app/handler.py"]