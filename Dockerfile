# ── Dockerfile ────────────────────────────────────────────────────────────────
# Builds the production image for SEC Filings Analyst.
#
# Runtime stack: Streamlit + FAISS-CPU + OpenAI API
# No GPU required — all ML inference is via OpenAI API.
# No sentence-transformers, torch, or CUDA packages.
#
# Build:   docker build -t sec-filings-analyst .
# Run:     docker run -p 8501:8501 -e OPENAI_API_KEY=sk-... sec-filings-analyst
# Compose: docker compose up -d

FROM python:3.11-slim

WORKDIR /app

# libgomp1 is required by faiss-cpu
# curl is required by the docker-compose healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── Dependencies ──────────────────────────────────────────────────────────────
# Copy requirements first — Docker layer cache skips this on code-only rebuilds.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Source files ──────────────────────────────────────────────────────────────
# Only the three files needed to serve queries at runtime.
# Entry point and the rag package are all the container needs at runtime.
COPY app.py ./
COPY rag/ ./rag/

# ── Data files ────────────────────────────────────────────────────────────────
# FAISS index (index.faiss) and chunk metadata (chunks.json) are built
# locally by the pipeline and baked into the image here.
# They are not in git — see .gitignore.
#
# IMPORTANT: use data/index/chunks.json — it matches the FAISS index built
# by index.py. data/processed/chunks.json is an intermediate pipeline file
# and may have been updated independently of the index.
COPY data/index/ ./data/index/

# ── Runtime ───────────────────────────────────────────────────────────────────
EXPOSE 8501

# OPENAI_API_KEY is injected at container start via -e or docker-compose.yml.
# It is never written into the image.
ENTRYPOINT ["streamlit", "run", "app.py", \
            "--server.port=8501", \
            "--server.headless=true", \
            "--server.address=0.0.0.0"]
