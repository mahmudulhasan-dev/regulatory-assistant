"""
Embeds every chunk in data/processed/chunks.jsonl and builds a FAISS index 
for similarity search.

Architecture Details:
- Model: all-MiniLM-L6-v2 (sentence-transformers) - 384-dim, CPU-friendly,
  trained for semantic search. 
- Index: FAISS IndexFlatIP (exact search) over L2-normalized vectors, which 
  makes inner product equivalent to cosine similarity. 
- Rationale: Exact search is appropriate here because the corpus is small
  (~1,100 chunks) - approximate indexing (IVF/HNSW) exists to trade accuracy
  for speed at a scale this corpus doesn't reach.
"""

import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

CHUNKS_PATH = Path("data/processed/chunks.jsonl")
INDEX_PATH = Path("data/processed/faiss.index")
METADATA_PATH = Path("data/processed/chunk_metadata.jsonl")
MODEL_NAME = "all-MiniLM-L6-v2"


def load_chunks() -> list[dict]:
    with CHUNKS_PATH.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def main() -> None:
    chunks = load_chunks()
    print(f"Loaded {len(chunks)} chunks")

    model = SentenceTransformer(MODEL_NAME)
    texts = [c["text"] for c in chunks]

    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # required for inner product = cosine similarity
    )
    embeddings = embeddings.astype(np.float32)  # FAISS requires float32

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_PATH))

    # Metadata written separately, same order as the index, so a FAISS
    # result position (an integer) maps back to doc_id/section/page/text.
    with METADATA_PATH.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    print(f"Wrote index ({index.ntotal} vectors, dim={dim}) to {INDEX_PATH}")
    print(f"Wrote metadata to {METADATA_PATH}")


if __name__ == "__main__":
    main()