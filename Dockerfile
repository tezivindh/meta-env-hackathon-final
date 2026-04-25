FROM python:3.10-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install runtime dependencies only (no torch/cuda/unsloth needed)
RUN pip install --no-cache-dir fastapi uvicorn pydantic numpy

# Copy project files
COPY . /app

# Install the package
RUN pip install --no-cache-dir -e .

EXPOSE 7860

ENV ENABLE_WEB_INTERFACE=true

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
