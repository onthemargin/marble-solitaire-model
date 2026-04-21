FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# Install minimal deps
RUN pip install --no-cache-dir numpy onnx onnxruntime google-cloud-storage

# Copy source
COPY src/ ./src/

# Training entrypoint
COPY vertex-training/run.py ./run.py

ENTRYPOINT ["python", "run.py"]
