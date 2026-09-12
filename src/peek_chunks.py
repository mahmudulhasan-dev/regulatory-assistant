"""
src/peek_chunks.py

Quick diagnostic over data/processed/chunks.jsonl - sample a few chunks
per document and report basic size stats, so we can sanity-check the
chunker's output without reading all 1200+ lines by hand.
"""

import json
import statistics
from collections import defaultdict
from pathlib import Path

CHUNKS_PATH = Path("data/processed/chunks.jsonl")
SAMPLES_PER_DOC = 4


def main() -> None:
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