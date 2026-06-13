# Why Run a Local LLM?

Running a language model on your own machine keeps your data private, your costs near zero, and your learning environment fully under your control.

## What you'll learn

- The five key reasons to prefer local inference during development and learning
- An honest trade-off comparison between local models and hosted APIs
- Hardware requirements and what to expect on different machines
- How Ollama makes local model management simple

## Five reasons to go local

### 1 — Cost

Hosted APIs charge per token. A RAG system that retrieves 2 000 tokens of context and generates 300 tokens of answer can cost several cents per query. At hundreds of queries a day during development, that adds up. Local inference runs at the cost of electricity.

### 2 — Privacy

Documents you index often contain sensitive information — internal reports, personal records, proprietary code. Sending that data to a third-party API may violate data-handling policies or simply feel wrong. Local models never leave your network.

### 3 — Transparency and learning

You control the model weights, the prompt, the sampling parameters, and the server. Nothing is hidden behind a rate-limited HTTP endpoint. This makes debugging, experimenting, and understanding the system much easier.

### 4 — Offline operation

Once a model is downloaded you can work without any internet connection. This matters for air-gapped environments, unreliable networks, or simply focusing without distractions.

### 5 — No rate limits

Local inference has no request-per-minute caps. Run a batch evaluation of 1 000 queries overnight without worrying about throttling or quota errors.

## Local vs hosted API trade-offs

| Factor | Local model | Hosted API |
|---|---|---|
| Output quality | Good (small models) to excellent (13B+) | Generally excellent |
| Latency | Slower on CPU; fast on GPU/Apple Silicon | Usually fast; varies by load |
| Hardware required | 4 GB RAM minimum; 8–16 GB comfortable | None on client |
| Cost per query | ~$0 | $0.0001 – $0.03+ |
| Privacy | Data stays local | Data sent to provider |
| Offline use | Yes | No |
| Setup effort | 10–15 minutes | API key only |
| Model variety | Many open-weights models | Provider's catalogue |

!!! tip
    For learning and prototyping, local wins on every axis except raw quality on very hard tasks. Start local, switch to a hosted API only when you need to benchmark top-tier quality.

## Hardware guidance

| Hardware | Expected experience |
|---|---|
| Any modern CPU (8 GB RAM) | Works for 1B–3B models; ~5–15 tokens/sec |
| CPU with 16 GB RAM | Comfortable for 7B–8B models; ~3–8 tokens/sec |
| NVIDIA GPU (8 GB VRAM) | Fast for 7B–8B; ~30–60 tokens/sec |
| Apple Silicon (M1/M2/M3) | Excellent; unified memory handles 13B+ smoothly |
| CPU only, 4 GB RAM | Possible for 1B models; expect slow responses |

!!! note
    "Tokens per second" varies with prompt length, context size, and quantisation level. 4-bit quantised models (the default in Ollama) use roughly half the RAM of full-precision weights.

!!! warning
    Do not try to run a 7B model if you have less than 8 GB of total RAM. The system will swap to disk and become unusably slow. Stick to 1B or 3B models in constrained environments.

## Ollama: local model management in one tool

Ollama wraps model download, quantisation, and a local HTTP server into a single CLI. You run `ollama pull llama3.2` and get a ready-to-use model with no Python packaging complexity.

```bash
# Pull a model
ollama pull llama3.2

# Chat interactively
ollama run llama3.2

# Check what you have installed
ollama list
```

The lab defaults to `llama3.2` (3B parameters, ~2 GB download) because it runs acceptably on CPU and produces sensible answers for RAG tasks.

## Next steps

- [Setting up Ollama](local-llm-ollama.md) — install Ollama and run your first local model.
- [Environment setup](environment-setup.md) — create a Python project with all dependencies installed.
