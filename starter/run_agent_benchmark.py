"""Run the real agent on one benchmark split and save auditable traces."""
import argparse
import json
import time
from pathlib import Path

from agent_controller import OpenAICompatibleController
from agentic import run_agent
from dense_retriever import Day6DenseRetriever
from retrieval_methods import HybridRetriever, LexicalRetriever


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Day 7 agent benchmark")
    parser.add_argument("--chunks", required=True, type=Path)
    parser.add_argument("--questions", required=True, type=Path)
    parser.add_argument("--database-url", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--split", choices=("dev", "test"), required=True)
    parser.add_argument("--retriever", choices=("dense", "hybrid"), default="hybrid")
    parser.add_argument("--strategy", default="B")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-searches", type=int, default=4)
    parser.add_argument("--budget", type=int, default=2400)
    args = parser.parse_args()

    chunks = json.loads(args.chunks.read_text(encoding="utf-8"))
    questions = [
        question
        for question in json.loads(args.questions.read_text(encoding="utf-8"))
        if question.get("split") == args.split
    ]
    dense = Day6DenseRetriever(args.database_url, strategy=args.strategy)
    retriever = dense
    if args.retriever == "hybrid":
        retriever = HybridRetriever(dense, LexicalRetriever(chunks))
    controller = OpenAICompatibleController()

    cases = [
        {"id": q["id"], "split": q["split"], "answerable": q["answerable"]}
        for q in questions
    ]
    traces = []
    for question in questions:
        started = time.perf_counter()
        try:
            result = run_agent(
                question["question"],
                controller,
                retriever,
                filters=question.get("filter"),
                max_searches=args.max_searches,
                top_k=args.top_k,
                budget=args.budget,
            )
            traces.append(
                {
                    "query_id": question["id"],
                    "split": question["split"],
                    "question": question["question"],
                    "elapsed_ms": (time.perf_counter() - started) * 1000,
                    "service_cost_usd": None,
                    **result,
                }
            )
        except (RuntimeError, ConnectionError, TimeoutError, ValueError) as error:
            traces.append(
                {
                    "query_id": question["id"],
                    "split": question["split"],
                    "question": question["question"],
                    "status": "error",
                    "elapsed_ms": (time.perf_counter() - started) * 1000,
                    "service_cost_usd": None,
                    "answer": "",
                    "citations": [],
                    "events": [],
                    "error": {"type": type(error).__name__, "message": str(error)},
                }
            )

    report = {
        "split": args.split,
        "retriever": args.retriever,
        "strategy": args.strategy,
        "top_k": args.top_k,
        "max_searches": args.max_searches,
        "budget": args.budget,
        "service_cost_usd": None,
        "cases": cases,
        "traces": traces,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "split": args.split,
                "cases": len(cases),
                "answered": sum(t["status"] == "answered" for t in traces),
                "abstain": sum(t["status"] == "abstain" for t in traces),
                "errors": sum(t["status"] == "error" for t in traces),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
