"""
rag.py — Grounded answer generation using the hybrid retrieval pipeline.

Flow:
    user question
        -> HybridRetriever.search()     (BM25 + dense + RRF + rerank)
        -> build grounded prompt        (context + citations)
        -> ollama.chat()                (local LLM, no API key needed)
        -> print answer with sources

Usage:
    python rag.py                       # interactive REPL
    python rag.py --question "What is RRF?"   # single question, then exit
"""

import argparse
import textwrap

import ollama

from config import OLLAMA_MODEL, OLLAMA_TEMPERATURE, TOP_K
from retrieval import HybridRetriever

# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = textwrap.dedent("""\
    You are a helpful assistant that answers questions using ONLY the provided
    context passages.

    Rules:
    - Base every factual claim on the context below.
    - After each claim, cite the source in brackets, e.g. [rag-techniques.md].
    - If the context does not contain enough information to answer the question,
      say "I don't have enough information in the provided context to answer that."
    - Do not invent facts or use knowledge outside the context.
    - Be concise and clear.
""")


def build_context_block(
    reranked: list[tuple[str, float]],
    retriever: HybridRetriever,
) -> str:
    """
    Format the reranked chunks into a numbered context block for the prompt.

    Each passage is prefixed with its source filename so the LLM can cite it.
    """
    lines = []
    for rank, (chunk_id, score) in enumerate(reranked, start=1):
        source = retriever.get_source(chunk_id)
        text = retriever.get_text(chunk_id)
        lines.append(f"[{rank}] Source: {source}\n{text}")
    return "\n\n---\n\n".join(lines)


def answer(
    question: str,
    retriever: HybridRetriever,
    verbose: bool = False,
) -> str:
    """
    Retrieve relevant chunks and generate a grounded answer via Ollama.

    Args:
        question:   The user's question.
        retriever:  A loaded HybridRetriever instance.
        verbose:    If True, print intermediate retrieval debug info.

    Returns:
        The LLM's answer string.
    """
    # -- 1. Hybrid retrieval ------------------------------------------------
    reranked, debug = retriever.search(question, top_k=TOP_K)

    if verbose:
        print("\n--- Retrieval debug ---")
        print(f"BM25 top-5:   {debug['bm25_top_ids'][:5]}")
        print(f"Dense top-5:  {debug['dense_top_ids'][:5]}")
        print(f"Fused top-5:  {debug['fused_top_ids'][:5]}")
        print(f"Reranked top: {[(cid, round(s, 2)) for cid, s in debug['reranked_top']]}")
        print("-----------------------\n")

    if not reranked:
        return "No relevant documents were found for your question."

    # -- 2. Build prompt ----------------------------------------------------
    context_block = build_context_block(reranked, retriever)

    user_message = (
        f"Context passages:\n\n{context_block}\n\n"
        f"Question: {question}"
    )

    # -- 3. Generate answer with Ollama -------------------------------------
    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        options={"temperature": OLLAMA_TEMPERATURE},
    )

    return response["message"]["content"]


# ---------------------------------------------------------------------------
# Interactive REPL
# ---------------------------------------------------------------------------

def run_repl(retriever: HybridRetriever) -> None:
    """Run an interactive question-answer loop."""
    print("=" * 60)
    print("Hybrid RAG — Local Q&A (type 'quit' or 'exit' to stop)")
    print(f"Model: {OLLAMA_MODEL}  |  Top-K: {TOP_K}")
    print("=" * 60)

    while True:
        try:
            question = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not question:
            continue
        if question.lower() in {"quit", "exit", "q"}:
            print("Goodbye!")
            break

        print("\nThinking …")
        try:
            response = answer(question, retriever, verbose=False)
            print(f"\nAssistant:\n{response}")
        except ollama.ResponseError as exc:
            print(f"[ERROR] Ollama error: {exc}")
            print(f"Make sure Ollama is running and the model '{OLLAMA_MODEL}' is pulled.")
            print(f"  ollama pull {OLLAMA_MODEL}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ask questions against the hybrid RAG pipeline."
    )
    parser.add_argument(
        "--question", "-q",
        type=str,
        default=None,
        help="A single question to answer, then exit.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print per-stage retrieval debug info.",
    )
    args = parser.parse_args()

    print("[rag] Loading retriever …")
    retriever = HybridRetriever()

    if args.question:
        print(f"\nQuestion: {args.question}\n")
        result = answer(args.question, retriever, verbose=args.verbose)
        print(f"Answer:\n{result}")
    else:
        run_repl(retriever)
