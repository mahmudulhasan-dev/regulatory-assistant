"""
Runs the hand-written Q&A set against the pipeline and scores performance.

Evaluates two core metrics:
  1. Retrieval hit-rate: Checks if the expected document appear anywhere in
     the top-k retrieved chunks? (doc-level).
  2. Decision correctness: Evaluates whether the system's sufficient/insufficient
     determination matches expected outcome to test constraint handling.

Notes:
    A question marked known_miss reports its result but is excluded from the
    pass/fail summary - it's a documented, already-understood limitation.
"""

import json
from pathlib import Path

import generate
import retrieve

QUESTIONS_PATH = Path("data/eval/questions.jsonl")
TOP_K = 5


def load_questions() -> list[dict]:
    with QUESTIONS_PATH.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def evaluate_one(q: dict) -> dict:
    retrieved = retrieve.search(q["question"], TOP_K)
    retrieved_doc_ids = {c["doc_id"] for c in retrieved}
    hit = q["doc_id"] in retrieved_doc_ids if q["doc_id"] else None  # N/A for unanswerable

    result = generate.generate(q["question"], TOP_K)
    decision_correct = result.sufficient == q["answerable"]

    cited_doc_ids = {c["chunk"]["doc_id"] for c in result.citations} if result.sufficient else set()
    cited_right_doc = (q["doc_id"] in cited_doc_ids) if (q["answerable"] and result.sufficient) else None

    return {
        "id": q["id"],
        "question": q["question"],
        "expected_doc_id": q["doc_id"],
        "answerable": q["answerable"],
        "known_miss": q.get("known_miss", False),
        "retrieval_hit": hit,
        "decision_correct": decision_correct,
        "sufficient": result.sufficient,
        "cited_right_doc": cited_right_doc,
        "answer": result.answer,
    }


def main() -> None:
    questions = load_questions()
    results = [evaluate_one(q) for q in questions]

    scored = [r for r in results if not r["known_miss"]]
    known_misses = [r for r in results if r["known_miss"]]

    answerable = [r for r in scored if r["answerable"]]
    hit_rate = sum(1 for r in answerable if r["retrieval_hit"]) / len(answerable) if answerable else 0
    decision_acc = sum(1 for r in scored if r["decision_correct"]) / len(scored) if scored else 0
    cite_acc = (
        sum(1 for r in answerable if r["cited_right_doc"]) / len(answerable) if answerable else 0
    )

    print(f"{'=' * 70}\nEVAL SUMMARY ({len(scored)} scored questions, "
          f"{len(known_misses)} documented known-miss excluded)\n{'=' * 70}")
    print(f"Retrieval hit-rate (answerable questions): {hit_rate:.1%}")
    print(f"Decision correctness (answer vs decline):   {decision_acc:.1%}")
    print(f"Citation doc correctness (answerable, where sufficient): {cite_acc:.1%}")

    print(f"\n{'-' * 70}\nPer-question detail:\n{'-' * 70}")
    for r in results:
        tag = " [KNOWN MISS]" if r["known_miss"] else ""
        status = "OK" if (r["known_miss"] or r["decision_correct"]) else "FAIL"
        print(f"[{status}]{tag} {r['id']}: hit={r['retrieval_hit']} "
              f"decision_correct={r['decision_correct']} "
              f"sufficient={r['sufficient']} cited_right_doc={r['cited_right_doc']}")
        print(f"    Q: {r['question']}")

    out_path = Path("data/eval/results.json")
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nFull results written to {out_path}")


if __name__ == "__main__":
    main()