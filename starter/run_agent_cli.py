"""Run the real Day 7 controller and dense retriever for one question."""
import argparse
import json
import os

from agent_controller import OpenAICompatibleController
from agentic import run_agent
from dense_retriever import Day6DenseRetriever


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Day 7 agent")
    parser.add_argument("--question", required=True)
    parser.add_argument("--database-url", default=os.getenv("DATABASE_URL"))
    parser.add_argument("--strategy", default="B")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-searches", type=int, default=4)
    parser.add_argument("--budget", type=int, default=2400)
    parser.add_argument("--filter-json")
    parser.add_argument(
        "--filter",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Metadata filter; repeat for multiple fields",
    )
    args = parser.parse_args()
    if args.filter_json and args.filter:
        raise ValueError("use either --filter-json or --filter, not both")
    if args.filter_json:
        filters = json.loads(args.filter_json)
    else:
        filters = {}
        for item in args.filter:
            if "=" not in item:
                raise ValueError(f"invalid filter, expected KEY=VALUE: {item}")
            key, value = item.split("=", 1)
            if not key or not value:
                raise ValueError(f"invalid filter, expected KEY=VALUE: {item}")
            filters[key] = value
        if not filters:
            filters = None
    controller = OpenAICompatibleController()
    retriever = Day6DenseRetriever(args.database_url, strategy=args.strategy)
    result = run_agent(
        args.question,
        controller,
        retriever,
        filters=filters,
        max_searches=args.max_searches,
        top_k=args.top_k,
        budget=args.budget,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
