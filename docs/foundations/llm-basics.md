# LLM Basics

Large Language Models (LLMs) are neural networks trained to predict the next token in a sequence — understanding this core mechanic explains both their power and their limitations, and motivates why RAG exists.

## What you'll learn

- What tokens and context windows are
- How temperature and parameters shape model output
- What next-token prediction means in practice
- Why LLMs hallucinate and lack live knowledge
- How to make a minimal `ollama.chat` call

---

## Tokens and context windows

LLMs don't read words — they read **tokens**, which are sub-word chunks produced by a tokenizer (e.g., `"embeddings"` → `["embed", "dings"]`). A rough rule of thumb: 1 token ≈ 0.75 English words.

The **context window** is the maximum number of tokens the model can attend to at once — both your input *and* the model's output count toward this limit. `llama3.2` (3B) has a 128 k-token context window; exceeding it silently truncates the oldest tokens.

| Concept | Typical range | Effect if exceeded |
|---|---|---|
| Context window | 4 k – 128 k tokens | Early tokens dropped |
| Output limit | 512 – 8 k tokens | Generation stops mid-response |

---

## Parameters and temperature

**Parameters** are the billions of learned weights inside the model (e.g., 3 B for `llama3.2`). More parameters generally means stronger reasoning but higher RAM usage.

**Temperature** controls randomness in token sampling:

- `0.0` — deterministic, always picks the highest-probability token (good for factual Q&A)
- `0.7` — balanced creativity and coherence (good for writing)
- `1.0+` — highly creative but prone to drift

---

## Next-token prediction

Every LLM response is produced one token at a time. At each step the model scores its entire vocabulary and samples from that distribution. The process is **autoregressive**: each generated token is fed back as input for the next step. This means the model has no memory between separate calls and no access to the internet or a clock.

!!! warning "LLMs can't do these things by default"
    - Access real-time or post-training information
    - Recall previous conversations (stateless API)
    - Guarantee factual accuracy — they will **hallucinate** plausible-sounding but wrong answers

---

## Why this motivates RAG

Because the model's knowledge is frozen at training time, you must supply relevant facts at inference time. **Retrieval-Augmented Generation** retrieves those facts from your own documents and injects them into the prompt, turning a closed-book test into an open-book one.

!!! note "Hallucination vs. ignorance"
    An LLM that *knows* it doesn't know a fact will say so. One that lacks that knowledge will often confabulate. Grounding the model with retrieved context dramatically reduces confabulation.

---

## Minimal `ollama.chat` call

```python
import ollama

response = ollama.chat(
    model="llama3.2",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user",   "content": "What is next-token prediction?"},
    ],
    options={"temperature": 0.0},
)

print(response["message"]["content"])
```

```bash
# Make sure Ollama is running first
ollama serve &
ollama pull llama3.2
```

!!! tip "Keep temperature at 0 for RAG"
    Factual retrieval tasks benefit from deterministic output. Set `temperature: 0` in your options dict to reduce hallucination risk.

---

## Next steps

- [What is RAG?](../getting-started/what-is-rag.md) — see how retrieval plugs into the LLM pipeline
- [Embeddings](embeddings.md) — turn text into vectors so you can retrieve it
