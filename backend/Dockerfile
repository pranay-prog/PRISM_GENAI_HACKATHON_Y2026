FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HF_HOME=/models DATA_DIR=/app/data
WORKDIR /app/backend

COPY requirements.txt .
# CPU-only torch first so sentence-transformers does not pull the CUDA build
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt

# Download embedding + reranker weights at build time so the container runs offline.
# If the download fails, the app falls back to the LSA embedder / lexical reranker.
ARG PRELOAD_MODELS=true
ARG EMBEDDING_MODEL=all-MiniLM-L6-v2
ARG RERANKER_MODEL=BAAI/bge-reranker-base
RUN if [ "$PRELOAD_MODELS" = "true" ]; then \
      python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; SentenceTransformer('${EMBEDDING_MODEL}'); CrossEncoder('${RERANKER_MODEL}')" \
      || echo 'WARNING: model preload failed; runtime will use fallbacks'; \
    fi
ENV HF_HUB_OFFLINE=1 EMBEDDING_MODEL=${EMBEDDING_MODEL} RERANKER_MODEL=${RERANKER_MODEL}

COPY . .
RUN python scripts/seed_corpus.py && python scripts/build_index.py

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
