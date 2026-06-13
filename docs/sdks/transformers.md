# Hugging Face Transformers SDK

The Hugging Face Transformers library gives you direct access to thousands of pretrained models for text generation, feature extraction, classification, and more — all runnable locally without an API key.

## What you'll learn

- How to load any model with `AutoTokenizer` and `AutoModelForCausalLM`
- Using `pipeline()` for a zero-boilerplate quickstart
- Correct use of `dtype=` for mixed-precision loading (v5+)
- Running local text-generation and feature-extraction models
- How Transformers relates to `sentence-transformers` and Ollama

## Install

```bash
pip install transformers torch accelerate
```

For GPU support, install the appropriate CUDA-enabled PyTorch build from [pytorch.org](https://pytorch.org) before installing the rest.

## pipeline() quickstart

The `pipeline()` high-level API is the fastest way to run a model. It handles tokenisation, inference, and decoding for you.

```python
from transformers import pipeline

# Text generation — downloads model on first run
gen = pipeline("text-generation", model="HuggingFaceTB/SmolLM2-360M-Instruct", device_map="auto")
result = gen("Explain retrieval-augmented generation in one sentence:", max_new_tokens=80)
print(result[0]["generated_text"])
```

```python
from transformers import pipeline

# Feature extraction (dense embeddings from a small encoder)
extractor = pipeline("feature-extraction", model="sentence-transformers/all-MiniLM-L6-v2", device_map="auto")
vectors = extractor("RAG combines retrieval with generation.", return_tensors=False)
print(len(vectors[0][0]))  # embedding dimension
```

`device_map="auto"` lets the `accelerate` library place model layers across available devices (CPU, GPU, multi-GPU) automatically.

## AutoTokenizer and AutoModelFor*

Use the `Auto*` classes when you need fine-grained control over tokenisation or want to inspect logits, hidden states, or attention weights.

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "HuggingFaceTB/SmolLM2-360M-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype=torch.bfloat16,   # half-precision — see warning below
    device_map="auto",
)

inputs = tokenizer("The capital of France is", return_tensors="pt")
with torch.no_grad():
    output_ids = model.generate(**inputs, max_new_tokens=30)

print(tokenizer.decode(output_ids[0], skip_special_tokens=True))
```

### Loading an encoder for feature extraction

```python
import torch
from transformers import AutoTokenizer, AutoModel

model_id = "sentence-transformers/all-MiniLM-L6-v2"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModel.from_pretrained(model_id, dtype=torch.float32, device_map="auto")

encoded = tokenizer("RAG reduces hallucinations.", return_tensors="pt", truncation=True)
with torch.no_grad():
    outputs = model(**encoded)

# Mean-pool the last hidden state to get a sentence vector
embedding = outputs.last_hidden_state.mean(dim=1)
print(embedding.shape)  # torch.Size([1, 384])
```

!!! warning "Use `dtype=`, not `torch_dtype=`"
    Transformers v5 renamed the precision parameter from `torch_dtype=` to `dtype=`.
    Passing `torch_dtype=` still works but raises a `DeprecationWarning` and will be
    removed in a future release. Update any existing code now.

    ```python
    # Deprecated (v4 style)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16)

    # Correct (v5+)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float16)
    ```

    See [versions and deprecations](versions-and-deprecations.md) for the full migration timeline.

## Choosing a task class

| Task | Class |
|---|---|
| Causal (decoder) generation | `AutoModelForCausalLM` |
| Masked language modelling | `AutoModelForMaskedLM` |
| Sequence classification | `AutoModelForSequenceClassification` |
| Token classification / NER | `AutoModelForTokenClassification` |
| Generic encoder embeddings | `AutoModel` |

## Relationship to sentence-transformers and Ollama

- **sentence-transformers** wraps Transformers encoders with a clean embedding API (`encode()`, cosine similarity helpers, cross-encoders). Prefer it when your goal is producing embeddings for RAG rather than running raw inference. See [sentence-transformers.md](sentence-transformers.md).
- **Ollama** serves quantised GGUF models through a local REST server. It is easier to install on laptops (no CUDA required) but offers less programmatic access to model internals. See the [embedding models guide](../tools/embedding-models.md).
- Use Transformers directly when you need custom fine-tuning, access to hidden states, or a model not yet available in Ollama's library.

## Next steps

- Produce production-quality embeddings with [sentence-transformers](sentence-transformers.md)
- Compare model formats and version compatibility in [versions and deprecations](versions-and-deprecations.md)
- Browse available embedding models in the [embedding models guide](../tools/embedding-models.md)
