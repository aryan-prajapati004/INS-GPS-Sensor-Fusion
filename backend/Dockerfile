FROM python:3.11-slim

WORKDIR /app

# System deps for scientific python wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only torch first
RUN pip install --no-cache-dir torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu

COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/models.py backend/main.py ./
COPY backend/artifacts/ ./artifacts/

# Copy the default dataset so main.py (which looks in ../data) can find it at /data
COPY data/v1.csv /data/v1.csv

ENV ARTIFACT_DIR=/app/artifacts
ENV PORT=8000
EXPOSE 8000

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
