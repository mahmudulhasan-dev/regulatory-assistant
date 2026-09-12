# Utility script to inspect raw text extraction from a single PDF.
# Use this to evaluate headers, page breaks, and column handling
# before finalizing the chunking strategy.

import sys
from pathlib import Path
import fitz

def inspect_pdf(pdf_path: Path, pages_to_show: int = 3) -> None:
    doc = fitz.open(pdf_path)
    print(f"File: {pdf_path.name}")
    print(f"Total pages: {len(doc)}")
    print("=" * 70)

    for page_num in range(min(pages_to_show, len(doc))):
        page = doc[page_num]
        text = page.get_text()
        print(f"\n--- PAGE {page_num + 1} (raw extracted text) ---\n")
        print(text)
        print(f"\n--- END PAGE {page_num + 1} ---")
        print("=" * 70)
    doc.close()


def show_single_page(pdf_path: Path, page_index: int) -> None:
    doc = fitz.open(pdf_path)
    page = doc[page_index]
    text = page.get_text()
    print(f"File: {pdf_path.name}  |  file position: {page_index} (0-based)")
    print("=" * 70)
    print(text)
    print("=" * 70)
    doc.close()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print(" python src/inspect_pdf.py <pdf> [num_pages]")
        print(" python src/inspect_pdf.py <pdf> --page <index>")
        sys.exit(1)

    pdf_path = Path(sys.argv[1])
    if not pdf_path.exists():
        print(f"File not found: {pdf_path}")
        sys.exit(1)

    if len(sys.argv) > 2 and sys.argv[2] == "--page":
        show_single_page(pdf_path, int(sys.argv[3]))
    else:
        num_pages = int(sys.argv[2]) if len(sys.argv) > 2 else 3
        inspect_pdf(pdf_path, num_pages)