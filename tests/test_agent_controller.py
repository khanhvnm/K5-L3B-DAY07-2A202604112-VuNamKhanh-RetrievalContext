import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "starter"))
from agent_controller import OpenAICompatibleController


class FakeController(OpenAICompatibleController):
    def __init__(self, content):
        super().__init__(api_base_url="http://test", model="test-model")
        self.content = content

    def _post_json(self, payload):
        return {
            "choices": [
                {"message": {"content": self.content}}
            ]
        }


class AgentControllerTests(unittest.TestCase):
    def test_parses_search_action(self):
        controller = FakeController('{"action":"search","query":"Docker volume"}')
        self.assertEqual(
            controller.decide("How does volume work?", []),
            {"action": "search", "query": "Docker volume"},
        )

    def test_parses_answer_action_with_citations(self):
        controller = FakeController(
            '{"action":"answer","answer":"7 days [c1]","citations":["c1"]}'
        )
        self.assertEqual(
            controller.decide("How long?", []),
            {
                "action": "answer",
                "answer": "7 days [c1]",
                "citations": ["c1"],
            },
        )

    def test_rejects_malformed_action(self):
        controller = FakeController('{"action":"search","query":""}')
        with self.assertRaises(ValueError):
            controller.decide("Question", [])


if __name__ == "__main__":
    unittest.main()
