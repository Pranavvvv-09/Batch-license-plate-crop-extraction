# =========================================================================
#  sang_det - 24/7 Cloud Background Worker & FastAPI Server
#  Runs license plate detection + NVIDIA NIM OCR + Supabase synchronization
# =========================================================================

FROM python:3.11-slim

# Prevent Python from buffering stdout/stderr and writing bytecode
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000 \
    HOST=0.0.0.0

# Install required system packages: ffmpeg, OpenCV libs, and PostgreSQL driver libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libpq-dev \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install CPU-optimized PyTorch first, then requirements
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY . .

# Create persistent output and data directories
RUN mkdir -p data/output data/logs

EXPOSE 8000

# Start FastAPI server + persistent background worker
CMD ["python", "run.py", "serve"]
