"""
src/peek_chunks.py

Quick diagnostic over data/processed/chunks.jsonl - sample a few chunks
per document and report basic size stats, so we can sanity-check the
chunker's output without reading all 1200+ lines by hand.
"""

import json
import sys
import statistics
from collections import defaultdict
from pathlib import Path

CHUNKS_PATH = Path("data/processed/chunks.jsonl")
SAMPLES_PER_DOC = 4


def show_all(doc_id: str, page: int | None, page_range: tuple[int, int] | None = None) -> None:
    """Print every chunk matching a given doc_id and/or page (or page
    range), in full - not a sample. Pass doc_id='' / page=None /
    page_range=None to leave that filter off. page and page_range are
    mutually exclusive; page_range wins if both are somehow passed."""
    with CHUNKS_PATH.open(encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            if doc_id and c["doc_id"] != doc_id:
                continue
            if page_range is not None and not (page_range[0] <= c["page"] <= page_range[1]):
                continue
            if page_range is None and page is not None and c["page"] != page:
                continue
            print(f"\n[{c['chunk_index']}] {c['doc_id']}  page {c['page']}  "
                  f"label={c['section_label']!r}")
            print(c["text"])
            print("-" * 70)


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "--find":
        doc_id = ""
        page = None
        page_range = None
        for arg in sys.argv[2:]:
            if arg.startswith("doc="):
                doc_id = arg.split("=", 1)[1]
            elif arg.startswith("page="):
                page = int(arg.split("=", 1)[1])
            elif arg.startswith("page_range="):
                lo, hi = arg.split("=", 1)[1].split("-")
                page_range = (int(lo), int(hi))
        show_all(doc_id, page, page_range)
        return

    by_doc: dict[str, list[dict]] = defaultdict(list)
    with CHUNKS_PATH.open(encoding="utf-8") as f:
        for line in f:
            chunk = json.loads(line)
            by_doc[chunk["doc_id"]].append(chunk)

    for doc_id, chunks in by_doc.items():
        word_counts = [len(c["text"].split()) for c in chunks]
        labeled = sum(1 for c in chunks if c["section_label"])

        print(f"\n{'=' * 70}\n{doc_id}  ({len(chunks)} chunks)")
        print(f"  words/chunk: min={min(word_counts)} "
              f"median={statistics.median(word_counts):.0f} "
              f"max={max(word_counts)}")
        print(f"  chunks with a section_label: {labeled}/{len(chunks)}")

        # spread the sample across the document, not just the first few
        step = max(1, len(chunks) // SAMPLES_PER_DOC)
        for c in chunks[::step][:SAMPLES_PER_DOC]:
            preview = c["text"][:150].replace("\n", " ")
            print(f"\n  [{c['chunk_index']}] page {c['page']} "
                  f"label={c['section_label']!r}")
            print(f"  {preview}...")


if __name__ == "__main__":
    main()