"""
compare.py — Side-by-side comparison of retrieval strategies.

Prints the top results for a query under four different retrieval strategies:
  1. BM25-only
  2. Dense-only
  3. Hybrid (BM25 + Dense, RRF fusion)
  4. Hybrid + Cross-encoder rerank  (the full pipeline)

This is the key TEACHING tool in the project.  Run it with different queries
to see how each strategy behaves and when hybrid+rerank wins.

Usage:
    python compare.py "What is reciprocal rank fusion?"
    python compare.py "BM25 term frequency saturation"
    python compare.py "cosine similarity distance metric"
    python compare.py --top 3 "how does cross-encoder reranking work?"
"""

import argparse
import textwrap

from config import CANDIDATES_K, TOP_K
from retrieval import HybridRetriever


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

SEPARATOR = "-" * 70


def print_stage_header(title: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


def print_ranked_ids(
    ranked_ids: list[str],
    retriever: HybridRetriever,
    top: int,
    scores: list[float] | None = None,
) -> None:
    """Print up to *top* results with a snippet of the chunk text."""
    for rank, chunk_id in enumerate(ranked_ids[:top], start=1):
        source = retriever.get_source(chunk_id)
        text = retriever.get_text(chunk_id)
        snippet = textwrap.shorten(text, width=120, placeholder=" …")
        score_str = ""
        if scores is not None and rank - 1 < len(scores):
            score_str = f"  score={scores[rank - 1]:.4f}"
        print(f"\n  [{rank}] {chunk_id}  ({source}){score_str}")
        print(f"       {snippet}")


def print_reranked(
    reranked: list[tuple[str, float]],
    retriever: HybridRetriever,
    top: int,
) -> None:
    """Print reranked (id, score) pairs with snippets."""
    ids = [cid for cid, _ in reranked[:top]]
    scores = [s for _, s in reranked[:top]]
    print_ranked_ids(ids, retriever, top, scores=scores)


# ---------------------------------------------------------------------------
# Main comparison
# ---------------------------------------------------------------------------

def run_comparison(query: str, top: int) -> None:
    print(f"\nQuery: \"{query}\"")
    print(f"Candidates per retriever: {CANDIDATES_K}  |  Display top: {top}\n")

    retriever = HybridRetriever()

    # ---- Stage 1a: BM25 only -----------------------------------------------
    bm25_ids = retriever.bm25_search(query, k=CANDIDATES_K)
    print_stage_header("STAGE 1a — BM25 Only  (keyword / sparse retrieval)")
    print(f"  Returns chunks that contain the query's exact keywords.")
    print(f"  Great for: rare terms, product codes, exact phrases.")
    print(SEPARATOR)
    print_ranked_ids(bm25_ids, retriever, top)

    # ---- Stage 1b: Dense only ----------------------------------------------
    dense_ids = retriever.dense_search(query, k=CANDIDATES_K)
    print_stage_header("STAGE 1b — Dense Only  (semantic / embedding retrieval)")
    print(f"  Returns chunks semantically close to the query vector.")
    print(f"  Great for: paraphrased questions, synonym matching.")
    print(SEPARATOR)
    print_ranked_ids(dense_ids, retriever, top)

    # ---- Stage 2: Hybrid (RRF) ---------------------------------------------
    fused_ids = retriever.reciprocal_rank_fusion([bm25_ids, dense_ids], k=60)
    print_stage_header("STAGE 2 — Hybrid: BM25 + Dense fused with RRF")
    print(f"  Chunks that rank high in BOTH lists get boosted.")
    print(f"  Uses Reciprocal Rank Fusion (k=60) — no score normalisation needed.")
    print(SEPARATOR)
    print_ranked_ids(fused_ids, retriever, top)

    # ---- Stage 3: Hybrid + Rerank ------------------------------------------
    reranked = retriever.rerank(query, fused_ids, k=top)
    print_stage_header("STAGE 3 — Hybrid + Cross-encoder Rerank  (full pipeline)")
    print(f"  Cross-encoder jointly reads (query, passage) — most accurate.")
    print(f"  Scores are raw logits; higher = more relevant.")
    print(SEPARATOR)
    print_reranked(reranked, retriever, top)

    # ---- Summary table -----------------------------------------------------
    print(f"\n{'=' * 70}")
    print("  SUMMARY — top-1 chunk per strategy")
    print(f"{'=' * 70}")
    strategies = {
        "BM25-only    ": bm25_ids[0] if bm25_ids else "—",
        "Dense-only   ": dense_ids[0] if dense_ids else "—",
        "Hybrid (RRF) ": fused_ids[0] if fused_ids else "—",
        "Hybrid+Rerank": reranked[0][0] if reranked else "—",
    }
    for name, chunk_id in strategies.items():
        source = retriever.get_source(chunk_id) if chunk_id != "—" else "—"
        print(f"  {name}  ->  {chunk_id}  ({source})")
    print()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare BM25, dense, hybrid-RRF, and hybrid+rerank retrieval."
    )
    parser.add_argument(
        "query",
        nargs="?",
        default="What is reciprocal rank fusion and how does it work?",
        help="The search query to compare strategies on.",
    )
    parser.add_argument(
        "--top", "-k",
        type=int,
        default=TOP_K,
        help=f"Number of results to display per strategy (default: {TOP_K}).",
    )
    args = parser.parse_args()
    run_comparison(args.query, args.top)
