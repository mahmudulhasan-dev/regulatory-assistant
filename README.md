# Regulatory Assistant

A retrieval-augmented generation (RAG) system that answers questions over public AI 
regulation and standards documents, citing the specific section, article, or page each 
claim comes from — and declining to answer when the corpus doesn't support a claim.

Built as an exploration into advanced RAG architecture, focusing on strict grounding and verifiable source attribution.

## Status

**Ingestion, retrieval, generation, and eval — complete.** UI not yet started.

## Corpus

Five public regulatory/policy documents, chosen for permissive reuse terms
and structural variety (numbered frameworks, legal articles, prose-only
policy papers):

| Document | Source | License basis |
|---|---|---|
| NIST AI RMF 1.0 | [nist.gov](https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.100-1.pdf) | US federal work, public domain |
| NIST Generative AI Profile (600-1) | [nist.gov](https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.600-1.pdf) | US federal work, public domain |
| EU AI Act (Regulation (EU) 2024/1689) | [EUR-Lex](https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:32024R1689) | EU document reuse policy, free reuse with attribution |
| UK AI white paper ("A pro-innovation approach to AI regulation") | [gov.uk](https://www.gov.uk/government/publications/ai-regulation-a-pro-innovation-approach/white-paper) | Open Government Licence v3.0 |
| Blueprint for an AI Bill of Rights | [whitehouse.gov archive](https://bidenwhitehouse.archives.gov/wp-content/uploads/2022/10/Blueprint-for-an-AI-Bill-of-Rights.pdf) | US federal work, public domain |

ISO/IEC 42001 was deliberately excluded — it's a paywalled ISO standard,
not freely redistributable, which would undermine the "verify the citation
yourself" premise of the project.

## Setup

```powershell
python -m venv venv
venv\Scripts\activate
pip install pymupdf sentence-transformers faiss-cpu
```

Place the five source PDFs in `data/raw/` (see table above for links),
named:
```
nist_ai_rmf_1.0.pdf
nist_genai_profile_600-1.pdf
eu_ai_act_2024_1689.pdf
uk_ai_whitepaper_2023.pdf
us_ai_bill_of_rights_2022.pdf
```

## Reproducing the chunked corpus

```powershell
python src/ingest.py
```

Reads every PDF in `data/raw/`, chunks it, and writes
`data/processed/chunks.jsonl` (one JSON object per line — `doc_id`,
`section_label`, `page`, `chunk_index`, `text`). This output is gitignored
and fully regenerable from the raw PDFs plus this command.

```powershell
python src/peek_chunks.py
```

Diagnostic: reports words-per-chunk stats and section-label coverage per
document, with a spread sample of chunk previews. Used to sanity-check
`ingest.py`'s output without reading `chunks.jsonl` by hand.

```powershell
python src/inspect_pdf.py <pdf-path> [num_pages]
python src/inspect_pdf.py <pdf-path> --page <0-based-index>
```

Diagnostic: prints raw PyMuPDF text extraction for a PDF, either the first
N pages or one page by index. Used throughout development to check how a
document's headings, tables, and figures actually extract before writing
chunking rules for them — see "Design decisions" below.

## Building the retrieval index

```powershell
python src/embed.py
```

Embeds every chunk in `chunks.jsonl` with `sentence-transformers`
(`all-MiniLM-L6-v2`, 384-dim) and builds a FAISS `IndexFlatIP` — exact
search over L2-normalized vectors, which makes inner product equivalent to
cosine similarity. Writes `data/processed/faiss.index` and
`chunk_metadata.jsonl` (the chunk records, same row order as the index, so
a FAISS result position maps back to `doc_id` / `section_label` / `page` /
`text`). Both outputs are gitignored and regenerable via this one command —
first run downloads the ~80MB model from Hugging Face.

**Why exact search instead of an approximate FAISS index (IVF, HNSW,
etc.):** approximate indexing trades a little retrieval accuracy for speed
at scale — typically justified from hundreds of thousands of vectors
upward. This corpus is ~1,100 chunks; brute-force exact search runs near
instantly at that size, so approximate indexing would add complexity to
solve a problem the project doesn't have.

## Retrieval quality notes

Manual testing with `retrieve.py` against a handful of queries showed a
clear, usable gap between genuinely topical and nonsense queries:

| Query type | Top-5 score range |
|---|---|
| Directly answerable (e.g. "what is a prohibited AI practice under the EU AI Act") | 0.70–0.73 |
| Real topic, wrong jurisdiction (e.g. asking about China in an EU Act query, or Japan in a UK white paper query) | 0.56–0.61 |
| Genuine nonsense (e.g. "what is the capital of France") | 0.16–0.21 |

The gap between nonsense and anything topical (roughly 0.3–0.4) is a
usable signal for a score-based decline threshold. However, a score floor
alone cannot catch the harder case of a *plausible but unsupported*
question — the wrong-jurisdiction queries above score nearly in the same range
as a genuine match (0.56–0.61), because the retrieved chunks are
topically related even though they don't actually answer the question
asked. Closing that gap will need the generation step itself to judge
whether retrieved chunks address the specific question, not just whether
retrieval found *something* similar enough.

**Known retrieval limitation:** a direct query about the EU Act's
Article 5 ("prohibited AI practices") does not surface Article 5 in the
top 5 results — cross-referencing Articles that mention "artificial
intelligence systems" and "safety components" in unrelated amendment
clauses score higher. This persisted after the ToC and short-chunk fixes
above, confirming it's a real limit of a small, general-purpose sentence
embedding model matching plain-English questions against dense,
cross-referencing legal text — not a pipeline bug.

## Generation

```powershell
pip install google-genai python-dotenv
```

```
# .env
GEMINI_API_KEY=your_key_here
```

```powershell
python src/generate.py "your question"
```

Retrieves the top-k chunks, then applies two layers of "don't answer what
isn't supported," in order:

1. **A hard score floor (0.3)** on the top retrieved result, set directly
   from the score-gap data above — below it, the system declines without
   calling the LLM at all.
2. **The model itself judges sufficiency** from the retrieved text. This
   catches the harder case the score floor can't: a topically-related but
   wrong-jurisdiction question scores in the same range as a genuine
   match (see "Retrieval quality notes"), so only reading the actual
   content — not just the similarity score — can catch it.

**Citations are verified, not trusted.** Gemini returns structured JSON
(`response_schema`) with citations referencing retrieved chunk indices,
not free-text with inline citations it could fabricate. Every returned
`chunk_index` is checked against what was actually retrieved before being
shown; an answer whose citations are all invalid after verification is
treated as insufficient rather than shown uncited. This is the actual
enforcement mechanism for the project's core constraint — no hallucinated
citations — not a prompt instruction alone.

Tested against four cases: a directly answerable question (correctly
answered and grounded), two topically-similar-but-wrong-jurisdiction
questions (both correctly declined — by the model's sufficiency judgment,
not the score floor, since both scored in the "real match" range), and
one nonsense question (declined by the score floor, no API call made).

## Evaluation

```bash
python src/eval.py
```

Runs a hand-written set of 20 Q&A pairs (`data/eval/questions.jsonl`) — 16
answerable, covering all five documents, and 4 deliberately unanswerable
(two real topics grafted onto the wrong jurisdiction, one pure nonsense
question, and one trap question about ISO/IEC 42001, a document
explicitly excluded from the corpus) — against the full retrieve +
generate pipeline, and reports:

- **Retrieval hit-rate** — did the expected document appear in the top-k?
- **Decision correctness** — did the system's answer/decline choice match
  what it should have done? This is the direct measure of the project's
  core constraint.

One question (`q7`) is flagged `known_miss` — it reproduces the EU Act
Article 5 limitation documented above — and is reported individually but
excluded from the summary, since it's an understood limitation rather
than a regression to chase on every rerun.

**Latest run (19 scored questions):**

| Metric | Result |
|---|---|
| Retrieval hit-rate (answerable questions) | 100% |
| Decision correctness (answer vs. decline) | 63.2% |

**The 63.2% figure undersells what's actually happening — every failure
was investigated individually, and all seven are the same root cause,
not seven different bugs.** For each failing question, the expected
document was retrieved (hit-rate is 100%), but the specific chunk
containing the correct answer did not score in the top 5, so the model
correctly declined rather than answering from unrelated or insufficient
context. Confirmed directly by searching the indexed corpus for the
exact expected content in every case:

- **Short, parenthetical definitions embedded in longer prose** (e.g.
  NIST's definition of "residual risk," one sentence inside a denser
  paragraph) consistently rank below less-relevant chunks for
  definitional queries — the embedding model weights the whole chunk's
  meaning, diluting a short specific answer.
- **A defining section competing against the same keyword used casually
  throughout the document** (e.g. the UK white paper's formal definition
  of "proportionate" as a named principle, versus 20+ other chunks that
  use "proportionate" as an ordinary adjective) has a harder ranking
  problem than topic-level retrieval alone can solve.

**In every one of these cases, the system behaved correctly given what
it was shown: it declined rather than fabricating an answer.** No
hallucinated citation or confidently wrong answer occurred anywhere in
this eval run. The citation-verification mechanism and the sufficiency
check both held up under real, unplanned failure conditions — not just
the handful of cases they were originally designed against.

**Implication for future work:** the fix for this class of failure is a
retrieval improvement, not a generation one — most plausibly a hybrid
approach combining semantic search with exact keyword matching (e.g.
BM25), since several of the failing queries ask for a specific defined
term that keyword search would likely surface directly.

## Design decisions

**Chunking is structure-aware per document, not one-size-fits-all.** Each
source cites itself differently, so the chunker detects each document's own
heading style via regex and uses the most specific citable unit available,
falling back to fixed-size chunking (300 words, 50-word overlap) wherever no
structure is detected:

| Document | Citation granularity |
|---|---|
| NIST AI RMF 1.0 | Section / subsection number (e.g. `3.1 Valid and Reliable`) |
| NIST GenAI Profile | Category heading or individual Action ID (e.g. `MP-2.1-001`) |
| EU AI Act | Article / Chapter / Annex (e.g. `Article 5 Prohibited AI practices`) |
| UK white paper | Part / Annex / subsection, or individual numbered paragraph (e.g. `46.`) |
| Blueprint for an AI Bill of Rights | Section heading (e.g. `SAFE AND EFFECTIVE SYSTEMS`, or the more specific `WHY THIS PRINCIPLE IS IMPORTANT` where applicable) |

**Headings are matched on whole lines, not searched for within a joined
block of text.** An early version joined each page's lines into one string
before running heading regexes, which caused the EU Act's cross-references
(e.g. "...as referred to in Article 42(1)...") to be wrongly detected as
Article headings. Anchoring every rule to `fullmatch` against one line —
never a substring search — rules this out structurally, since a
cross-reference embedded mid-sentence never *is* the entirety of its own
line.

**A Table of Contents page is structurally indistinguishable from real
body headings, once page-number stripping removes the one thing that
would tell them apart.** A ToC entry (`3.7` alone on a line, its title on
the next line) matches the same shape as a genuine section heading — the
only difference is the trailing page number, which boilerplate stripping
already removes before heading detection runs. This caused one section
(NIST's `3.7 Fair – with Harmful Bias Managed`) to silently absorb the
Table of Contents' *next* entry title as its own body text, surfaced via
an anomalously high-scoring but 5-word chunk at retrieval time. Fixed by
excluding each document's known ToC page(s) from heading detection
entirely (`skip_pages` in `DOCS`), rather than trying to distinguish the
two shapes by pattern alone.

**Header/footer stripping is frequency-based, not hardcoded per document.**
A line that repeats across a large fraction of a document's own pages is
treated as running-header/footer boilerplate and stripped before chunking.
This is what lets one function handle NIST's `NIST AI 100-1 / AI RMF 1.0`
header and the EU Act's unrelated `EN / OJ L, 12.7.2024` header without any
per-document special-casing.

**Figures and diagrams are not extractable as text — captions are used as
their citable proxy.** Confirmed by inspecting NIST's circular lifecycle
diagrams: none of their internal labels extract, while the figure's caption
("Fig. 2. Lifecycle and Key Dimensions of an AI System...") extracts
cleanly as ordinary body text. This is a limitation of the system. It
cannot cite content that only exists inside an image.

**A detected section with fewer than 5 words of body text is dropped
rather than emitted as its own chunk.** This mainly catches Table of
Contents pages, where heading-shaped lines appear back to back with
almost no real content between them — without the guard, a ToC entry
like "ENDNOTES" would produce its own near-empty, citable-looking chunk.

**Known open gaps, not yet addressed:**
- **EU AI Act recitals are not structurally detected.** The ~180 numbered
  "whereas" clauses before Article 1 (e.g. `(4) AI is a fast evolving
  family of technologies...`) fall into the fixed-size fallback and cite
  by page only, instead of getting their own `Recital N` label.

  Investigated and deliberately deferred, not overlooked: a recital's
  heading shape — a bare `(N)` alone on its own line, body text on the
  next line — is structurally identical to a footnote definition (e.g.
  `(7) Regulation (EC) No 765/2008 of the European Parliament...`), and
  the two are interleaved on the same pages. Both numbering sequences
  also climb monotonically through the whole document, so the
  strictly-increasing-number check that disambiguates the UK white
  paper's restarting paragraph lists (see `split_inline_markers` in
  `src/ingest.py`) doesn't help here — both a recital and a footnote
  satisfy "greater than the last one."

  The one workable signal is content, not shape: footnote bodies read
  as citations (`Regulation (EU) 2016/679 of...`, `OJ C 517,
  22.12.2021, p. 56.`), while recital bodies read as prose (`The
  purpose of this Regulation is to...`). A fix would need to: scope
  detection to only fire between the `Whereas:` line and
  `HAVE ADOPTED THIS REGULATION:` (recitals only exist in that span),
  keep the strictly-increasing check as a second filter, and reject a
  `(N)` match whose following line starts with a citation-like pattern
  (a capitalised proper-noun-heavy opener, "OJ", "Regulation", a
  leading number-slash-year, etc.). Meaningfully more involved than
  the other heading rules in this project — closer to a small
  classifier than a regex — which is why it's deferred rather than
  built: the eval set is unlikely to need Recital-level citations, and
  the four already-solved documents demonstrate the core
  structure-detection pattern well on their own.
- UK white paper: a heading whose title word-wraps across two lines in
  the source PDF (e.g. `Annex C: How to respond to this`) only captures
  the first line's text, since heading detection reads one line at a time.
- Blueprint for an AI Bill of Rights: the cover page's two-line title
  ("BLUEPRINT FOR AN AI BILL OF / RIGHTS") gets split the same way —
  cosmetic, since it's the cover page, not real content.
- Section-final chunks in the GenAI Profile's Action ID tables can pick up
  a trailing `AI Actor Tasks: ...` tag from the next row's boilerplate.
  Minor noise, not incorrect information.

## Repo layout

```
data/raw/         source PDFs (committed - all permissively licensed)
data/processed/   chunked output (gitignored - regenerate via ingest.py)
src/              ingestion + retrieval pipeline and diagnostic scripts
```
