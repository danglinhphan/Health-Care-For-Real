FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git wget curl build-essential \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .

# Install PyTorch CPU version and other dependencies
RUN pip install --upgrade pip \
    && pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu \
    && pip install transformers>=4.35.0 peft>=0.6.0 datasets>=2.14.0 \
       fastapi>=0.104.0 uvicorn[standard]>=0.24.0 pydantic>=2.4.0 \
       accelerate>=0.24.0 pyyaml>=6.0 tqdm>=4.66.0 numpy>=1.24.0

# Copy application code
COPY src/ ./src/
COPY config/ ./config/
COPY models/ ./models/

EXPOSE 3001

CMD ["python", "-m", "uvicorn", "src.api_server:app", "--host", "0.0.0.0", "--port", "3001"]