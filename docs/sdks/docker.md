# Docker — Containerising a RAG API

Docker packages your RAG API and all its dependencies into a portable image that runs identically on any machine. This page covers writing a slim Dockerfile, building and running the image, and composing the API alongside a vector database and Ollama.

## What you'll learn

- How to write a production-ready Dockerfile for a Python RAG API
- How to build and run the image with `docker build` / `docker run`
- How to compose the API, a vector database, and Ollama with Docker Compose
- How to keep images small with `.dockerignore` and multi-stage builds

## Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (or Docker Engine on Linux)
- A working FastAPI app — see [FastAPI](fastapi.md) for the API code

## Dockerfile

```dockerfile
# syntax=docker/dockerfile:1
FROM python:3.12-slim

# Prevent .pyc files and enable unbuffered stdout
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies first (layer is cached unless requirements.txt changes)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose the port Uvicorn will listen on
EXPOSE 8000

# Run the FastAPI app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

!!! info "python:3.12-slim"
    The `-slim` variant omits documentation, test tools, and dev headers. It is about 50 MB versus ~900 MB for the full image. Always prefer it unless you need to compile C extensions at runtime.

## .dockerignore

Create `.dockerignore` at the project root to exclude files that should not enter the build context. This speeds up `docker build` and prevents secrets leaking into the image.

```
# .dockerignore
__pycache__/
*.pyc
*.pyo
.env
.env.*
*.env
venv/
.venv/
.git/
.gitignore
*.md
tests/
data/
*.faiss
*.index
chroma_db/
notebooks/
```

!!! warning "Never copy .env into an image"
    Secrets baked into a Docker image are readable by anyone with access to the image. Pass secrets at runtime via environment variables or Docker secrets instead.

## Building and running

```bash
# Build the image and tag it
docker build -t rag-api:latest .

# Run the container, forwarding port 8000
docker run --rm -p 8000:8000 rag-api:latest

# Pass environment variables at runtime
docker run --rm -p 8000:8000 \
  -e OPENAI_API_KEY=sk-... \
  rag-api:latest
```

Test the running API:

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is RAG?"}'
```

## Docker Compose with a vector database and Ollama

A typical local RAG stack needs three services:

| Service | Role |
|---|---|
| `api` | The FastAPI RAG application |
| `chromadb` | Persistent vector store |
| `ollama` | Local LLM inference server |

```yaml
# docker-compose.yml
services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - CHROMA_HOST=chromadb
      - CHROMA_PORT=8000
      - OLLAMA_HOST=http://ollama:11434
    depends_on:
      chromadb:
        condition: service_healthy
      ollama:
        condition: service_started
    volumes:
      - ./data:/app/data  # mount local documents read-only
    restart: unless-stopped

  chromadb:
    image: chromadb/chroma:latest
    ports:
      - "8001:8000"
    volumes:
      - chroma_data:/chroma/chroma
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/api/v1/heartbeat"]
      interval: 10s
      timeout: 5s
      retries: 5

  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
    volumes:
      - ollama_models:/root/.ollama
    # Pull a model on first start (remove after first run to speed up restarts)
    # entrypoint: ["/bin/sh", "-c", "ollama serve & sleep 5 && ollama pull llama3 && wait"]

volumes:
  chroma_data:
  ollama_models:
```

```bash
# Start all services
docker compose up --build

# Start in the background
docker compose up --build -d

# Tail logs from the API only
docker compose logs -f api

# Stop and remove containers (volumes are preserved)
docker compose down
```

### Ollama: host vs container

You have two options for running Ollama:

**Option A — Ollama in its own container** (shown above): everything is self-contained; the API reaches Ollama at `http://ollama:11434`. GPU passthrough requires extra compose configuration (`deploy.resources.reservations.devices`).

**Option B — Ollama on the host**: if you already run Ollama locally and want to avoid duplicating model downloads, point the API at the host network. On Docker Desktop for Mac/Windows use the special DNS name `host.docker.internal`:

```yaml
# docker-compose.yml (option B)
services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OLLAMA_HOST=http://host.docker.internal:11434
```

On Linux, add `network_mode: host` to the `api` service, or use `--add-host=host.docker.internal:host-gateway` in a bridge network.

## Keeping images slim

!!! tip "Reduce image size"
    A smaller image builds faster, transfers faster, and has a smaller attack surface.

- **Use `-slim` or `-alpine` base images.** `python:3.12-slim` is ~50 MB; `python:3.12-alpine` is ~20 MB (but needs `gcc` for some packages).
- **Pin dependency versions** in `requirements.txt` so rebuilds are reproducible.
- **Install only what you need.** Split `requirements.txt` into `requirements.txt` (runtime) and `requirements-dev.txt` (tests, linting).
- **Use multi-stage builds** to keep build-time tools out of the final image.

```dockerfile
# Multi-stage build — compile in one stage, copy artefacts to a clean image
FROM python:3.12-slim AS builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

FROM python:3.12-slim
WORKDIR /app
COPY --from=builder /install /usr/local
COPY . .
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

## Mounting a persistent index

Vector indexes built at ingest time should live on a volume, not be baked into the image. Mount the index directory at runtime:

```bash
docker run --rm -p 8000:8000 \
  -v $(pwd)/chroma_db:/app/chroma_db \
  rag-api:latest
```

Or in compose:

```yaml
volumes:
  - ./chroma_db:/app/chroma_db
```

## Common requirements.txt for a RAG API

```
fastapi>=0.115.0
uvicorn[standard]>=0.30.0
pydantic>=2.0.0
chromadb>=0.5.0
sentence-transformers>=3.0.0
langchain>=0.3.0
```

## Next steps

- [FastAPI](fastapi.md) — the API code this Dockerfile wraps
- [Production](../advanced/production.md) — authentication, HTTPS, secrets management, and Kubernetes
- [Typing and Pydantic](../python/ai-python/packaging.md) — packaging Python projects for distribution
