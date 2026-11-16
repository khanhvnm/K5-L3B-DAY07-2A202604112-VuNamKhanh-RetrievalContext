import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "starter"))
from retrieval_methods import HybridRetriever, LexicalRetriever


CHUNKS = [
    {
        "id": "c1",
        "text": "Docker volume retains data after a container is removed.",
        "source_id": "doc",
        "source": "doc.txt",
        "page": 1,
        "metadata": {"section": "storage"},
    },
    {
        "id": "c2",
        "text": "Docker bridge networks connect containers.",
        "source_id": "doc",
        "source": "doc.txt",
        "page": 2,
        "metadata": {"section": "network"},
    },
]


class FakeDense:
    def search(self, query, *, filters, top_k):
        return [{**CHUNKS[0], "score": 0.9}]


class RetrievalMethodTests(unittest.TestCase):
    def test_lexical_filter_is_applied(self):
        retriever = LexicalRetriever(CHUNKS)
        results = retriever.search("Docker volume", filters={"section": "storage"})
        self.assertEqual([result["id"] for result in results], ["c1"])

    def test_hybrid_contains_rrf_score_and_ranks(self):
        retriever = HybridRetriever(FakeDense(), LexicalRetriever(CHUNKS))
        results = retriever.search("Docker volume", filters={"section": "storage"})
        self.assertEqual(results[0]["id"], "c1")
        self.assertIn("rrf_score", results[0])
        self.assertIn("dense_rank", results[0])
        self.assertIn("lexical_rank", results[0])


if __name__ == "__main__":
    unittest.main()
