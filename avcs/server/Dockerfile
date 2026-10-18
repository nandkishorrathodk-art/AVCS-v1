# AVCS Cloud Inference Server Dockerfile (CPU-Optimized, Zero-GPU Required)
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY pyproject.toml .
RUN pip install --no-cache-dir \
    torch==2.3.0 \
    numpy \
    pillow \
    fastapi \
    uvicorn \
    pydantic \
    requests

# Copy project source and pre-trained weights
COPY avcs /app/avcs
COPY forge /app/forge
COPY runs/weights /app/runs/weights

ENV PYTHONPATH=/app
ENV AVCS_WEIGHTS_PATH=/app/runs/weights/avcs_cortex_weights.pt
ENV AVCS_HUMAN_EYE_WEIGHTS_PATH=/app/runs/weights/human_eye_weights.pt
ENV OMP_NUM_THREADS=1

EXPOSE 8000

CMD ["uvicorn", "avcs.server.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]

