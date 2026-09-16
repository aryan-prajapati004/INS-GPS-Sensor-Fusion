FROM python:3.11-slim

WORKDIR /app

# System deps for scientific python wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only torch first (the default PyPI wheel pulls in ~2GB of
# CUDA libraries that are useless on Render and will blow the free-tier
# disk/RAM budget). The CPU wheel is ~200MB.
RUN pip install --no-cache-dir torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY models.py main.py ./
# Trained model artifacts (.pkl files) — see README for how to get these
# from your training notebook onto Render (Git LFS, object storage, or
# baked directly into the image).
COPY artifacts/ ./artifacts/

ENV ARTIFACT_DIR=/app/artifacts
ENV PORT=8000
EXPOSE 8000

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
