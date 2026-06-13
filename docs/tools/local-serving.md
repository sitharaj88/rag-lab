# Local Serving Engines

Run LLMs and embedding models entirely on your own hardware — no API keys, no data leaving your machine, full control over model choice and versioning.

## What you'll learn

- The landscape of local serving engines and how they differ
- When to use each tool (dev vs. production, CPU vs. GPU, GUI vs. CLI)
- Minimal startup commands for each engine
- GGUF and quantization basics
- A clear path: **learn on Ollama, scale with vLLM or TEI**

---

## Quick-comparison table

| Engine | Ease of start | Throughput | GPU required | OpenAI-compatible API | Best for |
|---|---|---|---|---|---|
| **Ollama** | Very easy (single binary) | Moderate | Optional (CPU works fine) | Yes (`/v1/`) | Learning, local dev, quick experiments |
| **vLLM** | Moderate (Python + pip) | Very high | Yes (NVIDIA recommended) | Yes (`/v1/`) | High-throughput production inference |
| **LM Studio** | Very easy (desktop GUI) | Moderate | Optional | Yes (local server tab) | Non-CLI users, model discovery/download |
| **llama.cpp** | Moderate (build from source or pre-built) | Good on CPU | No (CPU-first, optional GPU offload) | Yes (`llama-server`) | Low-resource machines, quantized GGUF, embedded use |
| **HF TEI** | Moderate (Docker) | Very high | Yes (optimised) | Partial (embeddings endpoint) | Production embedding & rerank serving |

---

## Engines in depth

### Ollama

Ollama is the recommended starting point for this site. A single binary download gives you a model library, a CLI, and an OpenAI-compatible REST server. Pull a model and start serving in two commands.

```bash
# Pull a model
ollama pull llama3.2

# The server starts automatically; test it
curl http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"llama3.2","messages":[{"role":"user","content":"Hello"}]}'
```

!!! tip "Start here"
    If you are new to local LLMs, start with Ollama. See the dedicated [Ollama tool page](ollama.md) and the [Ollama Python SDK](../sdks/ollama-sdk.md) page for deeper coverage.

---

### vLLM

vLLM is a high-throughput inference server built around **PagedAttention** — an algorithm that manages KV-cache memory efficiently, enabling much higher request concurrency than naive implementations. It exposes an OpenAI-compatible server, making it a drop-in replacement for any client code that already targets the OpenAI API.

```bash
pip install vllm

# Serve a model (requires a CUDA-capable GPU)
python -m vllm.entrypoints.openai.api_server \
  --model mistralai/Mistral-7B-Instruct-v0.3 \
  --port 8000
```

!!! warning "GPU requirement"
    vLLM is GPU-first. CPU-only operation is possible but unsupported for production use. Plan for at least one NVIDIA GPU with sufficient VRAM for your chosen model.

!!! note "Check official docs for current install instructions"
    vLLM releases frequently. Always verify the install command and supported model list at [https://docs.vllm.ai](https://docs.vllm.ai).

---

### LM Studio

LM Studio is a desktop application (macOS, Windows, Linux) that provides a graphical interface for searching, downloading, and running GGUF models from Hugging Face. It includes a built-in local server that speaks the OpenAI Chat Completions API, so any OpenAI-compatible client works against it.

```text
# No CLI install — download from https://lmstudio.ai
# 1. Open LM Studio → Search tab → find and download a model
# 2. Local Server tab → select model → Start Server
# Server defaults to http://localhost:1234/v1
```

!!! tip "Who should use LM Studio"
    LM Studio is ideal if you prefer a GUI workflow or want to browse available models without writing any code. Once you need automation or scripting, consider migrating to Ollama or vLLM.

---

### llama.cpp + llama-server

llama.cpp is the C++ inference engine that underpins many of the tools above (including Ollama internally). Running it directly gives you the most control: quantization levels, thread counts, GPU layer offloading, and a built-in HTTP server (`llama-server`).

```bash
# Using a pre-built release binary (see github.com/ggerganov/llama.cpp/releases)
# Download llama-server for your platform, then:

llama-server \
  --model ./models/mistral-7b-instruct-q4_k_m.gguf \
  --port 8080 \
  --n-gpu-layers 35        # offload 35 layers to GPU; 0 = CPU-only
```

!!! note "GGUF and quantization"
    llama.cpp uses the **GGUF** file format. Models are distributed in quantized variants (e.g. `Q4_K_M`, `Q8_0`, `F16`). Lower-bit quantizations use less RAM and run faster at a small quality cost. `Q4_K_M` is a widely used balance point. Always verify the model source and check the project's README for current quantization recommendations.

---

### Hugging Face Text Embeddings Inference (TEI)

TEI is a production-grade Docker service from Hugging Face for serving embedding and reranking models at high throughput. It supports Flash Attention, tensor parallelism, and a REST API that returns embeddings compatible with most vector store clients.

```bash
# Pull and run a TEI container (GPU example)
docker run --gpus all \
  -p 8080:80 \
  ghcr.io/huggingface/text-embeddings-inference:latest \
  --model-id BAAI/bge-m3
```

```bash
# Embed a text
curl http://localhost:8080/embed \
  -X POST \
  -d '{"inputs":"RAG is retrieval-augmented generation"}' \
  -H 'Content-Type: application/json'
```

!!! note "CPU images available"
    TEI ships CPU-only images (tagged `-cpu`). Throughput is lower but suitable for low-traffic or development embedding workloads.

!!! tip "Pair with vLLM"
    A common production pattern is TEI for embeddings + vLLM for generation, both behind a simple FastAPI router. See [production deployment](../advanced/production.md) for architecture guidance.

---

## Choosing between engines

```
Are you learning or prototyping?
  └─ YES → Use Ollama. It is the path of least resistance.

Do you need a desktop GUI to discover and run models?
  └─ YES → Try LM Studio alongside Ollama.

Do you need CPU-only inference or direct GGUF control?
  └─ YES → llama.cpp / llama-server.

Do you need high-throughput generation in production?
  └─ YES → vLLM (GPU).

Do you need high-throughput embedding or reranking in production?
  └─ YES → HF TEI (GPU or CPU).
```

!!! info "The site's default path"
    All tutorials and building-block examples on this site use **Ollama** as the local LLM server. When a tutorial reaches production scale-out, it will explicitly note where vLLM or TEI is the better fit.

---

## GGUF and quantization primer

**GGUF** is the container format used by llama.cpp and consumed by Ollama and LM Studio. A GGUF file bundles model weights, tokenizer data, and metadata in a single file.

**Quantization** reduces weight precision from 32-bit or 16-bit floats to lower-bit integers, shrinking the model and speeding up inference:

| Quantization | Approx. size vs. F16 | Quality loss | Typical use |
|---|---|---|---|
| F16 | 1× (baseline) | None | Max quality, high VRAM |
| Q8_0 | ~0.5× | Minimal | Good balance on large GPUs |
| Q4_K_M | ~0.28× | Small | Most popular general-purpose choice |
| Q2_K | ~0.16× | Noticeable | Very constrained hardware |

!!! warning "Verify current recommendations"
    Quantization naming schemes and quality tradeoffs evolve as llama.cpp is updated. Always check the model card and the [llama.cpp releases page](https://github.com/ggerganov/llama.cpp/releases) for the latest guidance.

---

## Next steps

- Follow the hands-on [Ollama setup guide](../getting-started/local-llm-ollama.md) to get your first model running.
- Explore the [Ollama Python SDK](../sdks/ollama-sdk.md) to call your local server from Python.
- Learn about picking the right embedding model in [Choosing models](choosing-models.md).
- When you are ready to deploy, read [Production deployment](../advanced/production.md).
