"""Reproduce the Day 6 corpus and create the Day 7 benchmark contract.

This script reads the permitted Day 6 text corpus and benchmark. It does not
read the Day 6 database, embedding vectors, or hidden evaluation labels.
"""
import argparse
import json
import re
import unicodedata
from pathlib import Path


UNANSWERABLE_CASES = [
    {
        "id": "q21",
        "split": "dev",
        "question": "Docker Engine mã hóa dữ liệu trong volume bằng thuật toán nào?",
        "answerable": False,
        "expected_behavior": "search_and_abstain",
        "absence_reason": "Corpus không nêu thuật toán mã hóa dữ liệu volume cụ thể.",
    },
    {
        "id": "q22",
        "split": "dev",
        "question": "Docker Compose tự động tạo chính xác bao nhiêu bản sao của mỗi service khi chạy mặc định?",
        "answerable": False,
        "expected_behavior": "search_and_abstain",
        "absence_reason": "Corpus không quy định số bản sao mặc định cho mỗi service.",
    },
    {
        "id": "q23",
        "split": "test",
        "question": "Docker image được lưu mặc định ở vùng địa lý nào của AWS?",
        "answerable": False,
        "expected_behavior": "search_and_abstain",
        "absence_reason": "Corpus không chứa thông tin về vùng AWS mặc định cho Docker image.",
    },
    {
        "id": "q24",
        "split": "test",
        "question": "Docker Engine giới hạn chính xác bao nhiêu container trên một host?",
        "answerable": False,
        "expected_behavior": "search_and_abstain",
        "absence_reason": "Corpus không đưa ra một giới hạn số container cố định trên host.",
    },
]


def normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    normalized = normalized.replace("\u00a0", " ")
    normalized = re.sub(r"[ \t\f\v]+", " ", normalized)
    normalized = re.sub(r" *\n *", "\n", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def contains_evidence(text: str, evidence: str) -> bool:
    compact_text = re.sub(r"\s+", " ", text).strip()
    compact_evidence = re.sub(r"\s+", " ", evidence).strip()
    return compact_evidence in compact_text


def paragraph_chunks(unit: dict, text: str, max_size: int = 1000) -> list[dict]:
    paragraphs = [part.strip() for part in text.split("\n\n") if part.strip()]
    chunk_texts: list[str] = []
    current: list[str] = []
    current_size = 0

    def flush() -> None:
        nonlocal current, current_size
        if current:
            chunk_texts.append("\n\n".join(current))
            current = []
            current_size = 0

    for paragraph in paragraphs:
        if len(paragraph) <= max_size:
            separator = 2 if current else 0
            if current and current_size + separator + len(paragraph) > max_size:
                flush()
            current.append(paragraph)
            current_size += (2 if len(current) > 1 else 0) + len(paragraph)
            continue

        flush()
        for start in range(0, len(paragraph), max_size):
            part = paragraph[start : start + max_size].strip()
            if part:
                chunk_texts.append(part)

    flush()
    return [
        {
            "id": f"paragraph-{unit['document_id']}-p1-c{index}",
            "text": text_value,
            "source_id": unit["document_id"],
            "source": unit["source_path_or_url"],
            "page": 1,
            "metadata": {"section": unit["section"], "language": "en"},
        }
        for index, text_value in enumerate(chunk_texts)
    ]


def load_json(path: Path) -> dict | list:
    return json.loads(path.read_text(encoding="utf-8"))


def prepare(day6_root: Path) -> tuple[list[dict], list[dict]]:
    day6_starter = day6_root / "starter"
    manifest = load_json(day6_starter / "data" / "manifest.json")
    source_questions = load_json(day6_starter / "benchmark" / "questions.json")
    chunks: list[dict] = []
    for unit in manifest["units"]:
        text_path = day6_starter / "data" / "docker" / f"{unit['document_id']}.txt"
        if not text_path.is_file():
            raise FileNotFoundError(f"missing corpus unit: {text_path}")
        chunks.extend(paragraph_chunks(unit, normalize_text(text_path.read_text(encoding="utf-8"))))

    questions = []
    for question in source_questions["questions"]:
        questions.append(
            {
                "id": question["id"],
                "split": question["split"],
                "question": question["question"],
                "answerable": True,
                "expected_answer": question["expected_answer"],
                "gold": [
                    {
                        "source_id": question["gold_source_id"],
                        "page": question["gold_page"],
                        "evidence": question["evidence_quote"],
                    }
                ],
                "filter": question.get("filter"),
            }
        )
    questions.extend(UNANSWERABLE_CASES)
    return chunks, questions


def validate(chunks: list[dict], questions: list[dict]) -> None:
    chunk_ids = {chunk["id"] for chunk in chunks}
    if len(chunk_ids) != len(chunks):
        raise ValueError("chunk IDs must be unique")
    if len(questions) != 24:
        raise ValueError(f"expected 24 questions, got {len(questions)}")
    question_ids = {question["id"] for question in questions}
    if len(question_ids) != len(questions):
        raise ValueError("question IDs must be unique")
    for question in questions:
        if question["answerable"]:
            if not question["gold"] or not question["expected_answer"].strip():
                raise ValueError(f"{question['id']}: missing answerable evidence")
            for gold in question["gold"]:
                matching = [
                    chunk
                    for chunk in chunks
                    if chunk["source_id"] == gold["source_id"]
                    and chunk["page"] == gold["page"]
                ]
                if not any(
                    contains_evidence(chunk["text"], gold["evidence"])
                    for chunk in matching
                ):
                    raise ValueError(f"{question['id']}: evidence is not in a chunk")
        elif question["expected_behavior"] != "search_and_abstain":
            raise ValueError(f"{question['id']}: invalid unanswerable behavior")
    counts = {
        split: {
            answerable: sum(
                question["split"] == split and question["answerable"] == answerable
                for question in questions
            )
            for answerable in (True, False)
        }
        for split in ("dev", "test")
    }
    if counts != {"dev": {True: 15, False: 2}, "test": {True: 5, False: 2}}:
        raise ValueError(f"invalid split counts: {counts}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Day 7 data from Day 6")
    parser.add_argument("--day6-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    chunks, questions = prepare(args.day6_root.resolve())
    validate(chunks, questions)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "chunks.json").write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output / "questions.json").write_text(
        json.dumps(questions, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Prepared {len(chunks)} chunks and {len(questions)} questions.")
    print("dev: 15 answerable, 2 unanswerable; test: 5 answerable, 2 unanswerable")


if __name__ == "__main__":
    main()
