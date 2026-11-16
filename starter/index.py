"""Create the Day 7 pgvector schema and index the bootstrapped chunks."""
import argparse
import json
import os
from pathlib import Path

import psycopg
from sentence_transformers import SentenceTransformer


MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def main() -> None:
    parser = argparse.ArgumentParser(description="Index Day 7 chunks in pgvector")
    parser.add_argument("--chunks", type=Path, default=Path("starter/data/day7/chunks.json"))
    parser.add_argument("--database-url", default=os.getenv("DATABASE_URL"))
    parser.add_argument("--strategy", default="B")
    parser.add_argument("--model", default=os.getenv("EMBEDDING_MODEL") or MODEL)
    args = parser.parse_args()
    if not args.database_url:
        raise ValueError("--database-url or DATABASE_URL is required")
    chunks = json.loads(args.chunks.read_text(encoding="utf-8"))
    model = SentenceTransformer(args.model)
    vectors = model.encode(
        [chunk["text"] for chunk in chunks],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    dimensions = int(model.get_embedding_dimension())
    with psycopg.connect(args.database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cursor.execute(
                f"""
                CREATE TABLE IF NOT EXISTS ingestion_chunks (
                    strategy TEXT NOT NULL,
                    chunk_id TEXT NOT NULL,
                    document_id TEXT NOT NULL,
                    source TEXT NOT NULL,
                    page INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                    embedding vector({dimensions}) NOT NULL,
                    PRIMARY KEY (strategy, chunk_id)
                )
                """
            )
            cursor.execute("DELETE FROM ingestion_chunks WHERE strategy = %s", (args.strategy,))
            cursor.executemany(
                """
                INSERT INTO ingestion_chunks
                    (strategy, chunk_id, document_id, source, page, text, metadata, embedding)
                VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, %s::vector)
                """,
                [
                    (
                        args.strategy,
                        chunk["id"],
                        chunk["source_id"],
                        chunk["source"],
                        chunk["page"],
                        chunk["text"],
                        json.dumps(chunk["metadata"]),
                        "[" + ",".join(str(float(value)) for value in vector) + "]",
                    )
                    for chunk, vector in zip(chunks, vectors)
                ],
            )
        connection.commit()
    print(f"Indexed {len(chunks)} chunks with {dimensions}-dimensional embeddings.")


if __name__ == "__main__":
    main()
