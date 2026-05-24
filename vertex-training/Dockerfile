FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

WORKDIR /app

# Install pinned deps
RUN pip install --no-cache-dir \
    numpy==1.26.4 \
    onnx==1.16.1 \
    onnxruntime==1.18.0 \
    google-cloud-storage==2.16.0

# Copy source
COPY src/ ./src/

# Training entrypoint
COPY vertex-training/run.py ./run.py

# Run as non-root
RUN useradd -m trainer
USER trainer

ENTRYPOINT ["python", "run.py"]
