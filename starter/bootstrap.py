"""Download the permitted Docker docs and recreate the Day 7 corpus contract."""
import argparse
import json
from html.parser import HTMLParser
from pathlib import Path
from urllib.request import Request, urlopen

from prepare_data import (
    UNANSWERABLE_CASES,
    normalize_text,
    paragraph_chunks,
    validate,
)


BLOCK_TAGS = {
    "article", "br", "div", "h1", "h2", "h3", "h4", "li", "p", "pre",
    "section", "table", "tr",
}


class MainTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.depth = 0
        self.in_main = False
        self.skip_depth = 0
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "noscript", "svg"}:
            self.skip_depth += 1
            return
        if tag == "main":
            self.in_main = True
            self.depth += 1
        elif self.in_main:
            self.depth += 1
        if self.in_main and tag in BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if self.skip_depth and tag in {"script", "style", "noscript", "svg"}:
            self.skip_depth -= 1
            return
        if self.in_main and tag in BLOCK_TAGS:
            self.parts.append("\n")
        if self.in_main:
            self.depth -= 1
            if tag == "main" and self.depth == 0:
                self.in_main = False

    def handle_data(self, data: str) -> None:
        if self.in_main and not self.skip_depth:
            self.parts.append(data)


def download_text(url: str, destination: Path) -> str:
    if destination.is_file():
        return destination.read_text(encoding="utf-8")
    request = Request(url, headers={"User-Agent": "day7-rag-bootstrap/1.0"})
    with urlopen(request, timeout=60) as response:
        html = response.read().decode("utf-8")
    parser = MainTextParser()
    parser.feed(html)
    text = normalize_text("".join(parser.parts))
    if len(text) < 1500:
        raise RuntimeError(f"Downloaded source is unexpectedly short: {url}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text, encoding="utf-8")
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap Day 7 without Day 6")
    parser.add_argument("--output", type=Path, default=Path("starter/data/day7"))
    parser.add_argument("--cache", type=Path, default=Path("starter/source/cache"))
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    manifest = json.loads((root / "source" / "manifest.json").read_text(encoding="utf-8"))
    source_units = []
    chunks = []
    for unit in manifest["units"]:
        cache_path = args.cache / f"{unit['document_id']}.txt"
        text = download_text(unit["source_path_or_url"], cache_path)
        source_units.append(unit)
        chunks.extend(paragraph_chunks(unit, text))

    source_questions = json.loads(
        (root / "benchmark" / "questions.json").read_text(encoding="utf-8")
    )["questions"]
    questions = []
    for question in source_questions:
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
                        "evidence": normalize_text(question["evidence_quote"]),
                    }
                ],
                "filter": question.get("filter"),
            }
        )
    questions.extend(UNANSWERABLE_CASES)
    validate(chunks, questions)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "chunks.json").write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output / "questions.json").write_text(
        json.dumps(questions, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Downloaded {len(source_units)} sources, created {len(chunks)} chunks and 24 questions.")


if __name__ == "__main__":
    main()
