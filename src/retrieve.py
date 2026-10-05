"""
Loads the FAISS index + metadata and runs similarity
search for a query. Returns the top-k chunks ranked by cosine similarity
(via inner product over normalized vectors - see embed.py).

Importable as a module (search()) for use by eval and generation later;
also runnable directly for quick manual testing.
"""

import json
import sys
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

INDEX_PATH = Path("data/processed/faiss.index")
METADATA_PATH = Path("data/processed/chunk_metadata.jsonl")
MODEL_NAME = "all-MiniLM-L6-v2"  # must match embed.py exactly

_model: SentenceTransformer | None = None
_index: faiss.Index | None = None
_metadata: list[dict] | None = None


def _load() -> None:
    """Lazy-load the model, index, and metadata once per process."""
    global _model, _index, _metadata
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    if _index is None:
        _index = faiss.read_index(str(INDEX_PATH))
    if _metadata is None:
        with METADATA_PATH.open(encoding="utf-8") as f:
            _metadata = [json.loads(line) for line in f]


def search(query: str, k: int = 5) -> list[dict]:
    """Return the top-k chunks for a query, ranked by similarity.

    Each result is the chunk's metadata (doc_id, doc_title, section_label,
    page, chunk_index, text) plus a 'score' field - cosine similarity,
    roughly in [-1, 1], higher is more similar.
    """
    _load()

    query_vec = _model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,  # must match embed.py - see note above
    ).astype(np.float32)

    scores, indices = _index.search(query_vec, k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:  # FAISS pads with -1 if k > number of vectors
            continue
        chunk = dict(_metadata[idx])
        chunk["score"] = float(score)
        results.append(chunk)
    return results


def _print_results(query: str, results: list[dict]) -> None:
    print(f"\nQuery: {query}\n{'=' * 70}")
    for rank, r in enumerate(results, start=1):
        label = r["section_label"] or f"page {r['page']}"
        print(f"\n[{rank}] score={r['score']:.3f}  {r['doc_id']} — {label}")
        print(r["text"][:250].replace("\n", " ") + "...")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python src/retrieve.py "your question" [k]')
        sys.exit(1)

    query = sys.argv[1]
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    _print_results(query, search(query, k))