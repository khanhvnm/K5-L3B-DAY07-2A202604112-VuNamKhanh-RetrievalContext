import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "starter"))
from dense_retriever import Day6DenseRetriever


class FakeCursor:
    def __init__(self):
        self.parameters = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def execute(self, statement, parameters):
        self.parameters = parameters

    def fetchall(self):
        return [
            (
                "chunk-1",
                "doc-1",
                "Evidence text",
                "https://example.test/doc",
                1,
                {"section": "build"},
                0.91,
            )
        ]


class FakeConnection:
    def __init__(self, cursor):
        self.cursor_value = cursor

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def cursor(self):
        return self.cursor_value


class FakePsycopg:
    def __init__(self, connection):
        self.connection = connection

    def connect(self, database_url):
        return self.connection


class FakeModel:
    def encode(self, texts, **kwargs):
        class FakeVector:
            def __getitem__(self, index):
                return self

            def tolist(self):
                return [0.5, 0.5]

        return FakeVector()


class DenseRetrieverTests(unittest.TestCase):
    def make_retriever(self):
        cursor = FakeCursor()
        retriever = Day6DenseRetriever.__new__(Day6DenseRetriever)
        retriever.database_url = "postgresql://test"
        retriever.strategy = "B"
        retriever.dimensions = 2
        retriever._model = FakeModel()
        retriever._psycopg = FakePsycopg(FakeConnection(cursor))
        return retriever, cursor

    def test_search_maps_day6_rows_to_agent_contract(self):
        retriever, cursor = self.make_retriever()

        results = retriever.search(
            "build",
            filters={"section": "build"},
            top_k=2,
        )

        self.assertEqual(results[0]["id"], "chunk-1")
        self.assertEqual(results[0]["source_id"], "doc-1")
        self.assertEqual(results[0]["score"], 0.91)
        self.assertEqual(
            cursor.parameters,
            ["[0.5,0.5]", "B", "section", "build", "[0.5,0.5]", 2],
        )

    def test_search_rejects_unsupported_filter(self):
        retriever, _ = self.make_retriever()

        with self.assertRaises(ValueError):
            retriever.search("build", filters={"unknown": "value"})


if __name__ == "__main__":
    unittest.main()
