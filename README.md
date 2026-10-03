# Regulatory Assistant

A retrieval-augmented generation (RAG) system that answers questions over public AI 
regulation and standards documents, citing the specific section, article, or page each 
claim comes from — and declining to answer when the corpus doesn't support a claim.

Built as an exploration into advanced RAG architecture, focusing on strict grounding and verifiable source attribution.

## Status

**Ingestion (chunking) — in progress.** Retrieval, generation, eval, and UI
not yet started.

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
pip install pymupdf
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
src/              ingestion pipeline and diagnostic scripts
```
