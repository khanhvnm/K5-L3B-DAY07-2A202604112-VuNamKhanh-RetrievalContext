import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "starter"))
from baseline import TfidfIndex, evaluate, validate


class BaselineUnanswerableTests(unittest.TestCase):
    def test_validation_and_evaluation_allow_unanswerable_case(self):
        chunks = [
            {
                "id": "c1",
                "text": "Docker volumes persist data.",
                "source_id": "doc",
                "source": "doc.txt",
                "page": 1,
                "metadata": {},
            }
        ]
        questions = [
            {
                "id": "q1",
                "split": "dev",
                "question": "What is the exact AWS region?",
                "answerable": False,
                "expected_behavior": "search_and_abstain",
            }
        ]
        validate(chunks, questions)
        trace = evaluate(TfidfIndex(chunks), questions)[0]
        self.assertFalse(trace["hit3"])


if __name__ == "__main__":
    unittest.main()
