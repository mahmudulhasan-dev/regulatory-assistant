"""
Given a query, retrieves supporting chunks and asks Gemini to
produce a grounded answer with citations - or explicitly declines when the
corpus doesn't support one.

Two-layer "don't answer what you can't support" design, based on scores
measured in retrieve.py testing:
  1. A hard score floor (SCORE_FLOOR) on the top retrieved result - below
     it, decline without calling the LLM at all. Measured gap: nonsense
     queries scored <=0.21, topical queries >=0.56.
  2. The model itself judges sufficiency from the retrieved text - this
     catches the harder case a score floor can't: a topically-related but
     wrong-jurisdiction question (e.g. asking about China inside an
     EU-only document) scores in the SAME range as a real match.

Every citation the model returns is verified against the actual set of
retrieved chunk indices before being shown - the model cannot make a
citation "real" just by asserting one. This is the mechanism that
prevents hallucinated citations, not a prompt instruction asking nicely.
"""

import json
import os
import sys
from dataclasses import dataclass

from dotenv import load_dotenv
from google import genai
from google.genai import types

import retrieve

load_dotenv()

MODEL_NAME = "gemini-3.1-flash-lite"
SCORE_FLOOR = 0.3  # see module docstring - gap measured between nonsense and topical queries
TOP_K = 5

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "sufficient": {
            "type": "boolean",
            "description": "True only if the retrieved chunks actually answer the question asked.",
        },
        "answer": {
            "type": "string",
            "description": "The answer, grounded only in the retrieved chunks. Empty if sufficient is false.",
        },
        "citations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "chunk_index": {
                        "type": "integer",
                        "description": "The [N] index of the retrieved chunk this claim comes from.",
                    },
                    "claim": {
                        "type": "string",
                        "description": "The specific claim from the answer this citation supports.",
                    },
                },
                "required": ["chunk_index", "claim"],
            },
        },
    },
    "required": ["sufficient", "answer", "citations"],
}

SYSTEM_PROMPT = """You answer questions ONLY using the numbered source \
chunks provided. Each chunk is labeled [N] with its document and section.

Rules:
- Set "sufficient" to false if the chunks do not actually answer the \
question - being topically related is not enough; they must address the \
specific question asked (same jurisdiction, same subject, same scope).
- If sufficient is false, "answer" must be empty and "citations" must be \
an empty list. Do not guess or fill in gaps from general knowledge.
- If sufficient is true, every factual claim in "answer" must have a \
matching entry in "citations" referencing the [N] index it came from.
- Never cite a chunk index that was not provided.
- Do not combine information from outside the provided chunks, even if \
you know it from general knowledge."""


@dataclass
class GenerationResult:
    sufficient: bool
    answer: str
    citations: list[dict]  # each: {chunk_index, claim, chunk} after verification
    retrieved: list[dict]
    decline_reason: str | None = None


def _build_prompt(query: str, chunks: list[dict]) -> str:
    def label(c: dict) -> str:
        return c["section_label"] or f"page {c['page']}"

    sources = "\n\n".join(
        f"[{i}] ({c['doc_id']} \u2014 {label(c)})\n{c['text']}"
        for i, c in enumerate(chunks)
    )
    return f"Question: {query}\n\nSource chunks:\n\n{sources}"


def generate(query: str, k: int = TOP_K) -> GenerationResult:
    retrieved = retrieve.search(query, k)

    if not retrieved or retrieved[0]["score"] < SCORE_FLOOR:
        return GenerationResult(
            sufficient=False,
            answer="",
            citations=[],
            retrieved=retrieved,
            decline_reason=f"top retrieval score {retrieved[0]['score']:.3f} "
                            f"below floor {SCORE_FLOOR}" if retrieved else "no chunks retrieved",
        )

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=_build_prompt(query, retrieved),
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_schema=RESPONSE_SCHEMA,
        ),
    )
    parsed = json.loads(response.text)

    if not parsed["sufficient"]:
        return GenerationResult(
            sufficient=False,
            answer="",
            citations=[],
            retrieved=retrieved,
            decline_reason="model judged retrieved chunks insufficient",
        )

    # Verification: a citation is only kept if its chunk_index is within
    # the range we actually retrieved. This is the check that makes the
    # citation mechanism real rather than trust-based.
    verified_citations = []
    for cite in parsed["citations"]:
        idx = cite["chunk_index"]
        if 0 <= idx < len(retrieved):
            verified_citations.append({**cite, "chunk": retrieved[idx]})
        else:
            print(f"[WARNING] model cited out-of-range chunk_index {idx} "
                  f"(only 0-{len(retrieved) - 1} were retrieved) - dropped", file=sys.stderr)

    if parsed["answer"] and not verified_citations:
        # The model claimed a sufficient, citeable answer, but every
        # citation it gave was invalid. Treat this as a failure to
        # ground, not a free pass to show an uncited answer.
        return GenerationResult(
            sufficient=False,
            answer="",
            citations=[],
            retrieved=retrieved,
            decline_reason="model's answer had no valid citations after verification",
        )

    return GenerationResult(
        sufficient=True,
        answer=parsed["answer"],
        citations=verified_citations,
        retrieved=retrieved,
    )


def _print_result(query: str, result: GenerationResult) -> None:
    print(f"\nQuery: {query}\n{'=' * 70}")
    if not result.sufficient:
        print(f"\n[DECLINED] {result.decline_reason}")
        print("\"I don't have enough information in this corpus to answer that.\"")
        return

    print(f"\n{result.answer}\n")
    print("Sources:")
    seen_indices = []
    for cite in result.citations:
        if cite["chunk_index"] not in seen_indices:
            seen_indices.append(cite["chunk_index"])
            c = cite["chunk"]
            label = c["section_label"] or f"page {c['page']}"
            print(f"\n  [{cite['chunk_index']}] {c['doc_id']} \u2014 {label}")
        print(f"      \"{cite['claim']}\"")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python src/generate.py "your question"')
        sys.exit(1)
    q = sys.argv[1]
    _print_result(q, generate(q))