"""Compare dense, lexical and hybrid retrieval on a fixed Day 7 split."""
import argparse
import json
import re
import statistics
import sys
import time
from pathlib import Path

from dense_retriever import Day6DenseRetriever
from retrieval_methods import HybridRetriever, LexicalRetriever


def is_evidence_hit(chunk: dict, question: dict) -> bool:
    chunk_text = re.sub(r"\s+", " ", chunk["text"]).strip()
    return any(
        chunk["source_id"] == gold["source_id"]
        and str(chunk["page"]) == str(gold["page"])
        and re.sub(r"\s+", " ", gold["evidence"]).strip() in chunk_text
        for gold in question.get("gold", [])
    )


def evaluate(method, questions: list[dict]) -> tuple[list[dict], dict]:
    traces = []
    answerable = [question for question in questions if question["answerable"]]
    for question in questions:
        started = time.perf_counter()
        hits = method.search(
            question["question"],
            filters=question.get("filter"),
            top_k=3,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000
        marked = [{**hit, "evidence_hit": is_evidence_hit(hit, question)} for hit in hits]
        traces.append(
            {
                "query_id": question["id"],
                "split": question["split"],
                "answerable": question["answerable"],
                "results": marked,
                "elapsed_ms": elapsed_ms,
            }
        )

    answerable_traces = [
        trace for trace in traces if trace["answerable"]
    ]
    summary = {
        "n_answerable": len(answerable),
        "hit_at_1": sum(
            bool(trace["results"] and trace["results"][0]["evidence_hit"])
            for trace in answerable_traces
        ),
        "evidence_hit_at_3": sum(
            any(hit["evidence_hit"] for hit in trace["results"][:3])
            for trace in answerable_traces
        ),
        "median_ms": statistics.median(trace["elapsed_ms"] for trace in traces),
        "n_total": len(traces),
    }
    return traces, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Day 7 retrieval methods")
    parser.add_argument("--chunks", required=True, type=Path)
    parser.add_argument("--questions", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--database-url", required=True)
    parser.add_argument("--strategy", default="B")
    args = parser.parse_args()

    chunks = json.loads(args.chunks.read_text(encoding="utf-8"))
    questions = json.loads(args.questions.read_text(encoding="utf-8"))
    selected = [question for question in questions if question["split"] == "dev"]
    lexical = LexicalRetriever(chunks)
    dense = Day6DenseRetriever(args.database_url, strategy=args.strategy)
    methods = {
        "dense": dense,
        "lexical_tfidf": lexical,
        "hybrid_rrf": HybridRetriever(dense, lexical),
    }
    report = {"split": "dev", "methods": {}}
    for name, method in methods.items():
        traces, summary = evaluate(method, selected)
        report["methods"][name] = {"summary": summary, "traces": traces}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    for name, result in report["methods"].items():
        print(name, json.dumps(result["summary"], ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        print(f"retrieval benchmark failed: {error}", file=sys.stderr)
        raise SystemExit(1)
