"""Lexical and hybrid retrieval methods for the Day 7 comparison."""
from collections import Counter
from typing import Any

from baseline import TfidfIndex, tokenize


class LexicalRetriever:
    """TF-IDF retrieval over the normalized Day 7 chunk contract."""

    def __init__(self, chunks: list[dict[str, Any]]) -> None:
        self.index = TfidfIndex(chunks)

    def search(
        self,
        query: str,
        *,
        filters: dict[str, Any] | None = None,
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        filters = filters or {}
        query_vector = self.index._vector(Counter(tokenize(query)))
        results = []
        for chunk, vector in zip(self.index.chunks, self.index.vectors):
            if any(str(chunk["metadata"].get(key)) != str(value) for key, value in filters.items()):
                continue
            score = sum(value * vector.get(term, 0) for term, value in query_vector.items())
            if score > 0:
                results.append({**chunk, "score": score, "strategy": "tfidf"})
        return sorted(results, key=lambda item: (-item["score"], item["id"]))[:top_k]


class HybridRetriever:
    """Fuse dense and lexical rankings using Reciprocal Rank Fusion."""

    def __init__(
        self,
        dense_retriever: Any,
        lexical_retriever: LexicalRetriever,
        *,
        rrf_k: int = 60,
        candidate_k: int = 10,
    ) -> None:
        if rrf_k <= 0 or candidate_k <= 0:
            raise ValueError("rrf_k and candidate_k must be greater than zero")
        self.dense = dense_retriever
        self.lexical = lexical_retriever
        self.rrf_k = rrf_k
        self.candidate_k = candidate_k

    def search(
        self,
        query: str,
        *,
        filters: dict[str, Any] | None = None,
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        dense_hits = self.dense.search(
            query, filters=filters, top_k=max(top_k, self.candidate_k)
        )
        lexical_hits = self.lexical.search(
            query, filters=filters, top_k=max(top_k, self.candidate_k)
        )
        by_id: dict[str, dict[str, Any]] = {}
        for rank, hit in enumerate(dense_hits, start=1):
            by_id.setdefault(hit["id"], {**hit, "dense_rank": rank})
            by_id[hit["id"]]["dense_score"] = hit["score"]
        for rank, hit in enumerate(lexical_hits, start=1):
            by_id.setdefault(hit["id"], {**hit, "lexical_rank": rank})
            by_id[hit["id"]]["lexical_rank"] = rank
            by_id[hit["id"]]["lexical_score"] = hit["score"]

        for hit in by_id.values():
            hit["rrf_score"] = sum(
                1 / (self.rrf_k + rank)
                for rank in (hit.get("dense_rank"), hit.get("lexical_rank"))
                if rank is not None
            )
            hit["score"] = hit["rrf_score"]
            hit["strategy"] = "hybrid_rrf"
        return sorted(
            by_id.values(),
            key=lambda item: (-item["rrf_score"], item["id"]),
        )[:top_k]
