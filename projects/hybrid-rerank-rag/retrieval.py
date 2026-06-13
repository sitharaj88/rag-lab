"""
retrieval.py — Hybrid retrieval: BM25 + Dense + RRF fusion + Cross-encoder reranking.

Architecture of HybridRetriever.search():

    query
      |
      +---> bm25_search(q, k=CANDIDATES_K)  --> [id, id, ...]   (ranked by BM25 score)
      |
      +---> dense_search(q, k=CANDIDATES_K) --> [id, id, ...]   (ranked by cosine sim)
      |
      +---> reciprocal_rank_fusion([bm25_ids, dense_ids], k=RRF_K)
      |         --> {id: rrf_score, ...}  (sorted, de-duplicated)
      |
      +---> rerank(q, fused_ids, k=TOP_K)
                --> [(id, cross_encoder_score), ...]  (sorted desc, top TOP_K)

The search() method ALSO returns a debug dict with each stage's top IDs so that
compare.py and the demo can print the stage-by-stage differences.
"""

import json
from typing import Any

import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

from config import (
    CANDIDATES_K,
    CHROMA_DIR,
    CHUNKS_JSON,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    RERANK_MODEL,
    RRF_K,
    TOP_K,
)


class HybridRetriever:
    """
    Combines BM25 sparse retrieval, dense vector retrieval, RRF fusion, and
    cross-encoder reranking into a single, inspectable retrieval pipeline.
    """

    def __init__(self) -> None:
        # -- Load chunk corpus (text + source) from the JSON written by ingest.py
        if not CHUNKS_JSON.exists():
            raise FileNotFoundError(
                f"chunks.json not found at {CHUNKS_JSON}. "
                "Run `python ingest.py` first."
            )
        with open(CHUNKS_JSON, encoding="utf-8") as f:
            self.corpus: dict[str, dict] = json.load(f)

        # Ordered list of IDs so BM25 index positions align with self.ids
        self.ids: list[str] = list(self.corpus.keys())
        self.texts: list[str] = [self.corpus[i]["text"] for i in self.ids]

        # -- Build BM25 index over tokenised chunks -------------------------
        # BM25Okapi expects a list of token lists.  Simple whitespace tokenisation
        # is sufficient here; for production use an NLTK or spaCy tokeniser.
        print("[retrieval] Building BM25 index …")
        tokenised = [text.lower().split() for text in self.texts]
        self.bm25 = BM25Okapi(tokenised)

        # -- Load embedding model (dense retrieval) -------------------------
        print(f"[retrieval] Loading embedding model: {EMBEDDING_MODEL}")
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)

        # -- Connect to ChromaDB collection ---------------------------------
        if not CHROMA_DIR.exists():
            raise FileNotFoundError(
                f"ChromaDB directory not found at {CHROMA_DIR}. "
                "Run `python ingest.py` first."
            )
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        self.collection = client.get_collection(name=COLLECTION_NAME)

        # -- Load cross-encoder reranker (downloaded on first use) ----------
        print(f"[retrieval] Loading cross-encoder: {RERANK_MODEL}")
        self.cross_encoder = CrossEncoder(RERANK_MODEL)

        print("[retrieval] HybridRetriever ready.\n")

    # -----------------------------------------------------------------------
    # Stage 1a: BM25 sparse retrieval
    # -----------------------------------------------------------------------

    def bm25_search(self, query: str, k: int = CANDIDATES_K) -> list[str]:
        """
        Return up to *k* chunk IDs ranked by BM25Okapi score.

        BM25 excels at exact keyword matches, rare terms, and product codes.
        It is completely unsupervised — no training or embeddings needed.
        """
        query_tokens = query.lower().split()
        scores = self.bm25.get_scores(query_tokens)   # ndarray of length = corpus size

        # argsort descending and take top k
        top_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:k]
        return [self.ids[i] for i in top_indices]

    # -----------------------------------------------------------------------
    # Stage 1b: Dense vector retrieval
    # -----------------------------------------------------------------------

    def dense_search(self, query: str, k: int = CANDIDATES_K) -> list[str]:
        """
        Return up to *k* chunk IDs ranked by cosine similarity to the query embedding.

        Dense retrieval captures semantic meaning even when surface words differ.
        """
        query_vec = self.embedder.encode(query).tolist()
        results = self.collection.query(
            query_embeddings=[query_vec],
            n_results=min(k, self.collection.count()),
        )
        # results["ids"] is [[id1, id2, ...]] — unwrap the outer list
        return results["ids"][0]

    # -----------------------------------------------------------------------
    # Stage 2: Reciprocal Rank Fusion
    # -----------------------------------------------------------------------

    def reciprocal_rank_fusion(
        self,
        list_of_ranked_id_lists: list[list[str]],
        k: int = RRF_K,
    ) -> list[str]:
        """
        Fuse multiple ranked lists using Reciprocal Rank Fusion (RRF).

        Formula: RRF(doc) = sum_over_lists  1 / (k + rank_in_list)
        where rank is 1-based.  Documents not present in a list contribute 0.

        Returns a list of IDs sorted by descending RRF score.

        Args:
            list_of_ranked_id_lists: e.g. [bm25_ids, dense_ids]
            k: smoothing constant (default 60, the original paper's value)

        Why RRF beats score normalisation:
        - BM25 scores and cosine similarities live on incompatible scales.
        - Normalising to [0,1] is sensitive to outliers.
        - RRF uses only *rank*, which is scale-free.
        """
        rrf_scores: dict[str, float] = {}
        for ranked_list in list_of_ranked_id_lists:
            for rank_0based, doc_id in enumerate(ranked_list):
                rank_1based = rank_0based + 1
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (
                    1.0 / (k + rank_1based)
                )

        # Sort descending by RRF score
        fused = sorted(rrf_scores.keys(), key=lambda d: rrf_scores[d], reverse=True)
        return fused

    # -----------------------------------------------------------------------
    # Stage 3: Cross-encoder reranking
    # -----------------------------------------------------------------------

    def rerank(
        self,
        query: str,
        candidate_ids: list[str],
        k: int = TOP_K,
    ) -> list[tuple[str, float]]:
        """
        Score each (query, candidate_text) pair with the cross-encoder and
        return the top *k* as [(id, score), ...] sorted by descending score.

        The cross-encoder sees both the query and passage jointly, enabling
        it to model fine-grained relevance signals that a bi-encoder misses.
        """
        if not candidate_ids:
            return []

        pairs = [
            (query, self.corpus[cid]["text"])
            for cid in candidate_ids
            if cid in self.corpus
        ]
        scores = self.cross_encoder.predict(pairs)

        # Pair each candidate ID with its score, sort descending
        scored = sorted(
            zip(candidate_ids, scores),
            key=lambda x: x[1],
            reverse=True,
        )
        return scored[:k]

    # -----------------------------------------------------------------------
    # Full pipeline — returns final results + debug info
    # -----------------------------------------------------------------------

    def search(
        self,
        query: str,
        candidates_k: int = CANDIDATES_K,
        top_k: int = TOP_K,
    ) -> tuple[list[tuple[str, float]], dict[str, Any]]:
        """
        Run the full hybrid retrieval pipeline and return:

            (reranked_results, debug)

        where:
          - reranked_results: [(chunk_id, cross_encoder_score), ...]  top_k items
          - debug: {
                "bm25_top_ids":    [id, ...],   # top candidates_k from BM25
                "dense_top_ids":   [id, ...],   # top candidates_k from dense
                "fused_top_ids":   [id, ...],   # all fused IDs (sorted by RRF score)
                "reranked_top":    [(id, score), ...],  # top_k after cross-encoder
            }
        """
        # Stage 1: independent retrieval
        bm25_ids = self.bm25_search(query, k=candidates_k)
        dense_ids = self.dense_search(query, k=candidates_k)

        # Stage 2: fuse with RRF
        fused_ids = self.reciprocal_rank_fusion([bm25_ids, dense_ids], k=RRF_K)

        # Stage 3: rerank the fused candidates
        reranked = self.rerank(query, fused_ids, k=top_k)

        debug: dict[str, Any] = {
            "bm25_top_ids": bm25_ids,
            "dense_top_ids": dense_ids,
            "fused_top_ids": fused_ids,
            "reranked_top": reranked,
        }
        return reranked, debug

    # -----------------------------------------------------------------------
    # Convenience helpers
    # -----------------------------------------------------------------------

    def get_text(self, chunk_id: str) -> str:
        """Return the text for a chunk ID, or a placeholder if not found."""
        return self.corpus.get(chunk_id, {}).get("text", "[chunk not found]")

    def get_source(self, chunk_id: str) -> str:
        """Return the source filename for a chunk ID."""
        return self.corpus.get(chunk_id, {}).get("source", "unknown")
