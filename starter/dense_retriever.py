"""Dense retrieval adapter for the Day 6 PostgreSQL + pgvector index."""
import os
from typing import Any


DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
PERMITTED_FILTERS = {
    "section",
    "language",
    "year",
    "version",
    "category",
    "audience",
    "document_type",
}


class Day6DenseRetriever:
    """Embed a query and retrieve traceable chunks from a Day 6 index."""

    def __init__(
        self,
        database_url: str | None = None,
        *,
        strategy: str = "B",
        model_name: str | None = None,
    ) -> None:
        self.database_url = database_url or os.getenv("DATABASE_URL")
        if not self.database_url:
            raise ValueError("DATABASE_URL is required")
        if not strategy.strip():
            raise ValueError("strategy must not be empty")
        self.strategy = strategy
        self.model_name = model_name or os.getenv("EMBEDDING_MODEL") or DEFAULT_MODEL

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise RuntimeError(
                "sentence-transformers is required for dense retrieval"
            ) from error

        self._model = SentenceTransformer(self.model_name)
        self.dimensions = self._model.get_embedding_dimension()
        if not isinstance(self.dimensions, int) or self.dimensions <= 0:
            raise RuntimeError(
                f"Embedding model returned invalid dimensions: {self.dimensions}"
            )

        try:
            import psycopg
        except ImportError as error:
            raise RuntimeError("psycopg is required for PostgreSQL retrieval") from error
        self._psycopg = psycopg

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
        unsupported = set(filters) - PERMITTED_FILTERS
        if unsupported:
            raise ValueError(
                "Unsupported metadata filters: " + ", ".join(sorted(unsupported))
            )

        query_vector = self._embed_query(query)
        vector_literal = self._vector_literal(query_vector)
        conditions = ["strategy = %s"]
        parameters: list[Any] = [self.strategy]
        for key, value in filters.items():
            conditions.append("metadata ->> %s = %s")
            parameters.extend([key, str(value)])
        parameters.extend([vector_literal, top_k])

        statement = f"""
            SELECT chunk_id, document_id, text, source, page, metadata,
                   1 - (embedding <=> %s::vector) AS score
            FROM ingestion_chunks
            WHERE {" AND ".join(conditions)}
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """
        # The query vector is used once for score and once for ordering.
        parameters.insert(0, vector_literal)

        with self._psycopg.connect(self.database_url) as connection:
            with connection.cursor() as cursor:
                cursor.execute(statement, parameters)
                rows = cursor.fetchall()

        return [
            {
                "id": row[0],
                "text": row[2],
                "source_id": row[1],
                "source": row[3],
                "page": row[4],
                "metadata": row[5],
                "score": float(row[6]),
            }
            for row in rows
        ]

    def _embed_query(self, query: str) -> list[float]:
        vectors = self._model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        vector = vectors[0].tolist()
        if len(vector) != self.dimensions:
            raise RuntimeError(
                f"Query embedding dimension mismatch: expected {self.dimensions}, "
                f"got {len(vector)}"
            )
        return vector

    @staticmethod
    def _vector_literal(vector: list[float]) -> str:
        return "[" + ",".join(str(float(value)) for value in vector) + "]"


def create_day6_retriever(
    *,
    database_url: str | None = None,
    strategy: str = "B",
    model_name: str | None = None,
) -> Day6DenseRetriever:
    """Create the configured Day 6 dense retriever for ``run_agent``."""
    return Day6DenseRetriever(
        database_url,
        strategy=strategy,
        model_name=model_name,
    )
