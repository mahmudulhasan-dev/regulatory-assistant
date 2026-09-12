from __future__ import annotations
import json
import re
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path
import fitz


DOCS = [
    {
        "doc_id": "nist_ai_rmf",
        "path": "data/raw/nist_ai_rmf_1.0.pdf",
        "title": "NIST AI RMF 1.0",
        "heading_pattern": re.compile(
            r"(?P<label>Part\s+\d+|(?:\d+\.)+\d*|\d+)\s+"
            r"(?P<title>[A-Z][A-Za-z0-9 ,\-&]{2,80})(?=\s{2,}|$)"
        ),
    },
    {
        "doc_id": "nist_genai_profile",
        "path": "data/raw/nist_genai_profile_600-1.pdf",
        "title": "NIST Generative AI Profile (600-1)",
        "heading_pattern": re.compile(
            r"(?P<label>(?:\d+\.)+\d*|\d+)\s+"
            r"(?P<title>[A-Z][A-Za-z0-9 ,\-&]{2,80})(?=\s{2,}|$)"
        ),
    },
    {
        "doc_id": "eu_ai_act",
        "path": "data/raw/eu_ai_act_2024_1689.pdf",
        "title": "EU AI Act (Regulation (EU) 2024/1689)",
        "heading_pattern": re.compile(
            r"(?P<label>Article\s+\d+|CHAPTER\s+[IVXLC]+|ANNEX\s+[IVXLC]+)"
            r"\s*(?P<title>[A-Z][A-Za-z0-9 ,\-&]{2,80})?"
        ),
    },
    {
        "doc_id": "uk_ai_whitepaper",
        "path": "data/raw/uk_ai_whitepaper_2023.pdf",
        "title": "UK White Paper: A pro-innovation approach to AI regulation",
        "heading_pattern": None,  # not yet inspected - fixed-size fallback only
    },
    {
        "doc_id": "us_ai_bill_of_rights",
        "path": "data/raw/us_ai_bill_of_rights_2022.pdf",
        "title": "Blueprint for an AI Bill of Rights",
        "heading_pattern": None,  # not yet inspected - fixed-size fallback only
    },
]

WORDS_PER_CHUNK = 300
OVERLAP_WORDS = 50
MIN_BOILERPLATE_PAGE_FRACTION = 0.4   # a line must repeat on >=40% of pages
MAX_BOILERPLATE_LINE_LENGTH = 80      # headers/footers are short lines as per inspection so far

PAGE_NUMBER_LINE = re.compile(r"^(Page\s+)?\d{1,4}(/\d{1,4})?$", re.IGNORECASE)


@dataclass
class Chunk:
    doc_id: str
    doc_title: str
    section_label: str | None
    page: int
    chunk_index: int
    text: str


def extract_pages(pdf_path: Path) -> list[str]:
    """Raw extracted text per page, in order."""
    doc = fitz.open(pdf_path)
    pages = [page.get_text() for page in doc]
    doc.close()
    return pages


def detect_boilerplate_lines(pages: list[str]) -> set[str]:
    """Lines that repeat across many pages of THIS document - running
    headers/footers - found by frequency rather than hardcoded per doc."""
    line_counts: Counter[str] = Counter()
    for page_text in pages:
        seen_this_page = set()
        for line in page_text.splitlines():
            stripped = line.strip()
            if stripped and len(stripped) <= MAX_BOILERPLATE_LINE_LENGTH:
                seen_this_page.add(stripped)
        for line in seen_this_page:
            line_counts[line] += 1

    threshold = max(2, int(len(pages) * MIN_BOILERPLATE_PAGE_FRACTION))
    return {line for line, count in line_counts.items() if count >= threshold}


def clean_pages(pages: list[str], boilerplate: set[str]) -> list[str]:
    """Strip boilerplate lines and bare page-number lines from every page."""
    cleaned = []
    for page_text in pages:
        kept_lines = []
        for line in page_text.splitlines():
            stripped = line.strip()
            if not stripped or stripped in boilerplate:
                continue
            if PAGE_NUMBER_LINE.match(stripped):
                continue
            kept_lines.append(stripped)
        cleaned.append(" ".join(kept_lines))
    return cleaned


def build_offset_map(cleaned_pages: list[str]) -> tuple[str, list[tuple[int, int]]]:
    """Join all pages into one string; record each page's (start, end)
    span in that string so any regex match position maps back to a page."""
    parts = []
    spans = []
    offset = 0
    for page_text in cleaned_pages:
        start = offset
        parts.append(page_text)
        offset += len(page_text)
        spans.append((start, offset))
        offset += 1  # the joining space added by " ".join below
    return " ".join(parts), spans


def offset_to_page(offset: int, spans: list[tuple[int, int]]) -> int:
    for i, (start, end) in enumerate(spans):
        if start <= offset <= end:
            return i + 1
    return len(spans)


def split_fixed_size(text: str) -> list[str]:
    """Fixed word-count chunks with overlap. Used as the fallback for
    unstructured documents, and for any single section that's too long
    to stand as one chunk."""
    words = text.split()
    if not words:
        return []
    out = []
    start = 0
    while start < len(words):
        end = start + WORDS_PER_CHUNK
        out.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        start = end - OVERLAP_WORDS
    return out


def chunk_document(doc_config: dict) -> list[Chunk]:
    pdf_path = Path(doc_config["path"])
    pages = extract_pages(pdf_path)
    boilerplate = detect_boilerplate_lines(pages)
    cleaned_pages = clean_pages(pages, boilerplate)
    full_text, spans = build_offset_map(cleaned_pages)

    pattern = doc_config["heading_pattern"]
    chunks: list[Chunk] = []
    idx = 0

    def emit(text: str, label: str | None, page: int) -> None:
        nonlocal idx
        chunks.append(Chunk(
            doc_id=doc_config["doc_id"],
            doc_title=doc_config["title"],
            section_label=label,
            page=page,
            chunk_index=idx,
            text=text,
        ))
        idx += 1

    if pattern is None:
        for page_num, (start, end) in enumerate(spans, start=1):
            for chunk_text in split_fixed_size(full_text[start:end]):
                emit(chunk_text, None, page_num)
        return chunks

    matches = list(pattern.finditer(full_text))
    boundaries = [m.start() for m in matches] + [len(full_text)]
    section_starts = [0] + boundaries[:-1]
    labels = [None] + [
        f"{m.group('label')} {m.group('title') or ''}".strip() for m in matches
    ]

    for section_start, section_end, label in zip(section_starts, boundaries, labels):
        section_text = full_text[section_start:section_end].strip()
        if not section_text:
            continue
        page = offset_to_page(section_start, spans)

        if len(section_text.split()) <= WORDS_PER_CHUNK:
            emit(section_text, label, page)
        else:
            for chunk_text in split_fixed_size(section_text):
                emit(chunk_text, label, page)

    return chunks


def main() -> None:
    all_chunks: list[Chunk] = []
    for doc_config in DOCS:
        path = Path(doc_config["path"])
        if not path.exists():
            print(f"[skip] {path} not found")
            continue
        doc_chunks = chunk_document(doc_config)
        print(f"{doc_config['doc_id']}: {len(doc_chunks)} chunks")
        all_chunks.extend(doc_chunks)

    out_path = Path("data/processed/chunks.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")

    print(f"\nWrote {len(all_chunks)} total chunks to {out_path}")


if __name__ == "__main__":
    main()