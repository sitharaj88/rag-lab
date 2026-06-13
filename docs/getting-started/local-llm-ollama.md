# Setting Up Ollama

Ollama is a single-binary tool that downloads, manages, and serves open-weight language models locally with a simple CLI and a REST/Python API.

## What you'll learn

- How to install Ollama on Windows, macOS, and Linux
- The essential CLI commands: `pull`, `run`, `serve`, `list`
- How to call Ollama from Python for chat and embeddings
- How to choose a model size that fits your RAM
- How to fix the two most common errors

## Install Ollama

Download the installer from **[ollama.com/download](https://ollama.com/download)** and follow the platform instructions.

### Windows

Run the `.exe` installer. Ollama installs as a background service and starts automatically. You can confirm it is running from the system tray icon.

### macOS

Download the `.zip`, unzip, and drag **Ollama.app** to Applications. Launch it once — it runs as a menu-bar app and starts the server on `http://localhost:11434`.

### Linux

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

The script installs the binary and registers a systemd service. Start it with:

```bash
sudo systemctl start ollama
```

## Essential CLI commands

### Pull a model

```bash
ollama pull llama3.2
```

This downloads the 3B-parameter `llama3.2` model (~2 GB). Models are stored in `~/.ollama/models` (macOS/Linux) or `%USERPROFILE%\.ollama\models` (Windows).

### Chat interactively

```bash
ollama run llama3.2
```

Type your message and press Enter. Type `/bye` to exit.

### Start the server manually

```bash
ollama serve
```

Normally the server starts automatically. Use this command if you installed without the service or want to see verbose logs.

### List installed models

```bash
ollama list
```

```text
NAME            ID              SIZE    MODIFIED
llama3.2:latest a80c4f17acd5    2.0 GB  2 minutes ago
```

## Python client

Install the client with `pip install ollama` (already in `requirements.txt`).

### Chat completion

```python
import ollama

response = ollama.chat(
    model="llama3.2",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is retrieval-augmented generation?"},
    ],
)

print(response["message"]["content"])
```

!!! tip
    For streaming output — useful in chat UIs — add `stream=True` and iterate over the response:

    ```python
    for chunk in ollama.chat(model="llama3.2", messages=[...], stream=True):
        print(chunk["message"]["content"], end="", flush=True)
    ```

### Embeddings

```python
import ollama

result = ollama.embeddings(
    model="llama3.2",
    prompt="Retrieval-augmented generation grounds LLM answers in documents.",
)

vector = result["embedding"]   # list of floats
print(f"Embedding dimension: {len(vector)}")
```

!!! note
    For RAG pipelines this lab uses `sentence-transformers` (`all-MiniLM-L6-v2`) for embeddings rather than the LLM itself. It is faster and cheaper. Use `ollama.embeddings()` only when you want the same model for both embedding and generation.

## Choosing a model size

| Model | Parameters | Download size | Min RAM | Notes |
|---|---|---|---|---|
| `llama3.2:1b` | 1B | ~0.7 GB | 4 GB | Fastest; basic reasoning |
| `llama3.2` | 3B | ~2 GB | 6 GB | Good balance; lab default |
| `llama3.1:8b` | 8B | ~4.7 GB | 10 GB | Noticeably better quality |
| `llama3.1:70b` | 70B | ~40 GB | 48 GB | Excellent; needs a powerful GPU |

!!! warning
    Always leave headroom. If your system has 8 GB of RAM, choose the 1B or 3B model. Running a model that barely fits will cause swapping and make responses painfully slow.

## Troubleshooting

### Error: `connection refused` / `could not connect to ollama`

The server is not running. Fix:

```powershell
# Windows — restart from the system tray, or:
ollama serve

# macOS — reopen Ollama.app from Applications

# Linux
sudo systemctl start ollama
```

Verify the server is up:

```bash
curl http://localhost:11434
# Expected: "Ollama is running"
```

### Error: `model 'llama3.2' not found`

You have not pulled the model yet, or you mistyped the name.

```bash
ollama pull llama3.2
ollama list          # confirm it appears
```

Model names are case-sensitive. Use the exact name shown in `ollama list`.

## Next steps

- [Ollama tool reference](../tools/ollama.md) — full parameter reference, custom model files, and advanced options.
- [LLM basics](../foundations/llm-basics.md) — understand what the model is actually doing when it generates text.
