"""
src/ingest.py

Ingestion pipeline: PDF -> chunked text with citation metadata.

- Strips repeated header/footer lines (detected by frequency across a
  document's own pages).
- Detects document-specific structural headings by matching whole lines
  (not substrings within a joined blob) - this is what lets us tell a
  real heading ("Article 5" alone on its own line) apart from a mid-
  sentence cross-reference ("...as referred to in Article 42(1)...").
- Falls back to fixed-size chunking with overlap - tagged by page number -
  for documents with no known heading pattern yet, or for oversized
  sections.
- Every chunk carries: doc_id, section_label, page, chunk_index, text.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, asdict, field
from pathlib import Path

import fitz  # PyMuPDF


@dataclass
class HeadingRule:
    pattern: re.Pattern
    # "same_line": pattern has named groups label + title, both on one line.
    # "next_line": pattern has named group label only; title is taken from
    #              the following non-empty line (if that line isn't itself
    #              a heading).
    mode: str


# ---------------------------------------------------------------------------
# One entry per source document. `heading_rules` is empty for documents we
# haven't inspected closely yet - they get the fixed-size fallback
# everywhere until we look at their structure and add real rules.
#
# NOTE (known v1 limitation): NIST rules only catch "Part N" and bare
# top-level "N." section numbers, confirmed present in body text. We have
# NOT confirmed how "5.1"-style subsections look in body text (only seen
# in the Table of Contents, a different layout) - so subsection-level
# citations aren't attempted yet.
# ---------------------------------------------------------------------------

DOCS = [
    {
        "doc_id": "nist_ai_rmf",
        "path": "data/raw/nist_ai_rmf_1.0.pdf",
        "title": "NIST AI RMF 1.0",
        "heading_rules": [
            HeadingRule(re.compile(r"Part\s+\d+:\s*(?P<title>.+)".replace(
                "(?P<title>.+)", "(?P<title>.+)")), "same_line"),
        ],
    },
    {
        "doc_id": "nist_genai_profile",
        "path": "data/raw/nist_genai_profile_600-1.pdf",
        "title": "NIST Generative AI Profile (600-1)",
        "heading_rules": [],  # not yet confirmed - fixed-size fallback only
    },
    {
        "doc_id": "eu_ai_act",
        "path": "data/raw/eu_ai_act_2024_1689.pdf",
        "title": "EU AI Act (Regulation (EU) 2024/1689)",
        "heading_rules": [
            HeadingRule(re.compile(r"(?P<label>Article\s+\d+)\.?"), "next_line"),
            HeadingRule(re.compile(r"(?P<label>CHAPTER\s+[IVXLC]+)"), "next_line"),
            HeadingRule(re.compile(r"(?P<label>ANNEX\s+[IVXLC]+)"), "next_line"),
        ],
    },
    {
        "doc_id": "uk_ai_whitepaper",
        "path": "data/raw/uk_ai_whitepaper_2023.pdf",
        "title": "UK White Paper: A pro-innovation approach to AI regulation",
        "heading_rules": [],  # not yet inspected - fixed-size fallback only
    },
    {
        "doc_id": "us_ai_bill_of_rights",
        "path": "data/raw/us_ai_bill_of_rights_2022.pdf",
        "title": "Blueprint for an AI Bill of Rights",
        "heading_rules": [],  # not yet inspected - fixed-size fallback only
    },
]

# Fix for the NIST "Part 2:" style rule accidentally double-escaped above -
# defined cleanly here and used directly.
PART_RULE = HeadingRule(re.compile(r"Part\s+\d+:\s*(?P<title>.+)"), "same_line")
SUBSECTION_RULE = HeadingRule(
    re.compile(r"(?P<label>\d+\.\d+(?:\.\d+)?)"),
    "next_line",
)
DOCS[0]["heading_rules"] = [
    PART_RULE,
    SUBSECTION_RULE,
    HeadingRule(re.compile(r"(?P<label>\d+\.)"), "next_line"),
]

WORDS_PER_CHUNK = 300
OVERLAP_WORDS = 50
MIN_BOILERPLATE_PAGE_FRACTION = 0.4   # a line must repeat on >=40% of pages
MAX_BOILERPLATE_LINE_LENGTH = 80      # headers/footers are short lines

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


def clean_pages_to_lines(
    pages: list[str], boilerplate: set[str]
) -> list[tuple[str, int]]:
    """Flatten the whole document into (line, page_number) pairs, in
    order, with boilerplate and bare page-number lines removed. This is
    the representation heading detection runs against - line boundaries
    are the signal that lets a real heading be told apart from the same
    words appearing mid-sentence."""
    result: list[tuple[str, int]] = []
    for page_num, page_text in enumerate(pages, start=1):
        for line in page_text.splitlines():
            stripped = line.strip()
            if not stripped or stripped in boilerplate:
                continue
            if PAGE_NUMBER_LINE.match(stripped):
                continue
            result.append((stripped, page_num))
    return result


def detect_headings(
    lines: list[tuple[str, int]], rules: list[HeadingRule]
) -> list[dict]:
    """Find heading lines by requiring a FULL match against an entire
    line - not a search anywhere within it. This is what excludes
    cross-references like '...referred to in Article 42(1)...', since
    that text is never the entirety of its own line."""
    if not rules:
        return []

    def is_any_heading_line(text: str) -> bool:
        return any(r.pattern.fullmatch(text) for r in rules)

    headings = []
    n = len(lines)
    for i, (line, page) in enumerate(lines):
        for rule in rules:
            m = rule.pattern.fullmatch(line)
            if not m:
                continue
            label = m.group("label") if "label" in m.groupdict() else m.group(0)

            if rule.mode == "same_line":
                title = m.group("title")
                body_start = i + 1
            else:
                title = None
                body_start = i + 1
                if i + 1 < n:
                    candidate = lines[i + 1][0]
                    if candidate and not is_any_heading_line(candidate):
                        title = candidate
                        body_start = i + 2

            headings.append({
                "label": label,
                "title": title,
                "page": page,
                "body_start": body_start,
            })
            break  # first matching rule wins; don't double-count a line

    return headings


def build_sections(
    lines: list[tuple[str, int]], headings: list[dict]
) -> list[dict]:
    """Slice the document into (start, end, label, page) sections using
    each heading's line position as a boundary and its body_start as
    where that section's own text begins (skipping the heading's label
    line, and title line if one was consumed)."""
    sections = []
    prev_start = 0
    prev_label = None
    prev_page = lines[0][1] if lines else 1

    heading_line_indices = {
        i for i, (line, _p) in enumerate(lines)
        for h in headings if False  # placeholder, replaced below
    }

    # Recompute heading line indices properly (we need the index each
    # heading was found at, not just its body_start).
    idx_by_heading = []
    scan_pos = 0
    for h in headings:
        # body_start - 1 (or -2) tells us where the heading line itself was;
        # since headings are produced in document order and non-overlapping,
        # walk forward to find it precisely.
        while scan_pos < len(lines):
            if lines[scan_pos][1] == h["page"] and (
                lines[scan_pos][0] == h["label"]
                or lines[scan_pos][0].rstrip(".") == h["label"].rstrip(".")
            ):
                break
            scan_pos += 1
        idx_by_heading.append(scan_pos)
        scan_pos += 1

    for heading_idx, h in zip(idx_by_heading, headings):
        sections.append({
            "start": prev_start,
            "end": heading_idx,
            "label": prev_label,
            "page": prev_page,
        })
        prev_label = f"{h['label']} {h['title']}".strip() if h["title"] else h["label"]
        prev_page = h["page"]
        prev_start = h["body_start"]

    sections.append({
        "start": prev_start,
        "end": len(lines),
        "label": prev_label,
        "page": prev_page,
    })
    return sections


def split_fixed_size(text: str) -> list[str]:
    """Fixed word-count chunks with overlap. Used for unstructured
    documents, and for any single section too long to stand as one chunk."""
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
    lines = clean_pages_to_lines(pages, boilerplate)

    headings = detect_headings(lines, doc_config["heading_rules"])
    sections = build_sections(lines, headings)

    chunks: list[Chunk] = []
    idx = 0
    for sec in sections:
        text = " ".join(line for line, _p in lines[sec["start"]:sec["end"]]).strip()
        if not text:
            continue

        pieces = (
            [text] if len(text.split()) <= WORDS_PER_CHUNK else split_fixed_size(text)
        )
        for piece in pieces:
            chunks.append(Chunk(
                doc_id=doc_config["doc_id"],
                doc_title=doc_config["title"],
                section_label=sec["label"],
                page=sec["page"],
                chunk_index=idx,
                text=piece,
            ))
            idx += 1

    return chunks


def main() -> None:
    all_chunks: list[Chunk] = []
    for doc_config in DOCS:
        path = Path(doc_config["path"])
        if not path.exists():
            print(f"[skip] {path} not found")
            continue
        doc_chunks = chunk_document(doc_config)
        labeled = sum(1 for c in doc_chunks if c.section_label)
        print(f"{doc_config['doc_id']}: {len(doc_chunks)} chunks "
              f"({labeled} labeled)")
        all_chunks.extend(doc_chunks)

    out_path = Path("data/processed/chunks.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")

    print(f"\nWrote {len(all_chunks)} total chunks to {out_path}")


if __name__ == "__main__":
    main()