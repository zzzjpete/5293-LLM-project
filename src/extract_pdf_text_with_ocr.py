#!/usr/bin/env python3
"""
Extract text from a PDF, with optional OCR fallback for scanned/image PDFs.

Why this exists:
    The project notebook currently uses PyPDF2/PyPDF-style extraction. That works
    for text-based PDFs, but not for scanned PDFs where text is only an image.

Basic usage:
    python extract_pdf_text_with_ocr.py input.pdf
    python extract_pdf_text_with_ocr.py input.pdf -o extracted_text.txt
    python extract_pdf_text_with_ocr.py input.pdf --force-ocr

OCR dependency setup:
    macOS:
        brew install poppler tesseract
        pip install pypdf pdf2image pytesseract pillow

    Google Colab:
        !apt-get update && apt-get install -y poppler-utils tesseract-ocr
        !pip install pypdf pdf2image pytesseract pillow
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


def find_executable(name: str) -> str | None:
    """Find an executable on PATH or in common Homebrew locations."""
    found = shutil.which(name)
    if found:
        return found

    for base in [Path("/opt/homebrew/bin"), Path("/usr/local/bin")]:
        candidate = base / name
        if candidate.exists():
            return str(candidate)

    return None


def extract_embedded_text(pdf_path: Path, max_pages: int | None = None) -> str:
    """Extract text embedded in a normal text-based PDF."""
    try:
        from pypdf import PdfReader
    except ImportError:
        try:
            from PyPDF2 import PdfReader
        except ImportError as exc:
            raise RuntimeError("Install pypdf or PyPDF2 for normal PDF text extraction.") from exc

    reader = PdfReader(str(pdf_path))
    pages = reader.pages[:max_pages] if max_pages else reader.pages
    parts: list[str] = []

    for page_num, page in enumerate(pages, 1):
        text = page.extract_text() or ""
        if text.strip():
            parts.append(f"--- Page {page_num} ---\n{text.strip()}")

    return "\n\n".join(parts)


def extract_text_with_ocr(pdf_path: Path, dpi: int = 250, max_pages: int | None = None) -> str:
    """OCR a scanned/image PDF using pdf2image + Tesseract."""
    tesseract_cmd = find_executable("tesseract")
    pdftoppm_cmd = find_executable("pdftoppm")
    poppler_path = str(Path(pdftoppm_cmd).parent) if pdftoppm_cmd else None

    if not tesseract_cmd:
        raise RuntimeError(
            "Tesseract OCR is not installed or not on PATH. "
            "Install it first: macOS `brew install tesseract`; "
            "Colab `!apt-get install -y tesseract-ocr`."
        )

    try:
        import pytesseract
        from pdf2image import convert_from_path
    except ImportError as exc:
        raise RuntimeError(
            "Missing OCR Python packages. Install with: "
            "`pip install pdf2image pytesseract pillow`."
        ) from exc

    pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    last_page = max_pages if max_pages else None
    try:
        images = convert_from_path(str(pdf_path), dpi=dpi, last_page=last_page, poppler_path=poppler_path)
    except Exception as exc:
        raise RuntimeError(
            "Could not render PDF pages for OCR. You may need Poppler: "
            "macOS `brew install poppler`; Colab `!apt-get install -y poppler-utils`."
        ) from exc

    parts: list[str] = []
    for page_num, image in enumerate(images, 1):
        text = pytesseract.image_to_string(image)
        if text.strip():
            parts.append(f"--- Page {page_num} OCR ---\n{text.strip()}")

    return "\n\n".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract PDF text with optional OCR fallback.")
    parser.add_argument("pdf", type=Path, help="Path to the PDF file.")
    parser.add_argument("-o", "--output", type=Path, help="Where to write extracted text.")
    parser.add_argument("--force-ocr", action="store_true", help="Skip embedded text extraction and run OCR.")
    parser.add_argument("--min-chars", type=int, default=80, help="Run OCR fallback if embedded text is shorter than this.")
    parser.add_argument("--dpi", type=int, default=250, help="OCR render DPI.")
    parser.add_argument("--max-pages", type=int, default=None, help="Limit extraction to the first N pages.")
    args = parser.parse_args()

    if not args.pdf.exists():
        print(f"PDF not found: {args.pdf}", file=sys.stderr)
        return 2

    text = ""
    method = "embedded text"

    if not args.force_ocr:
        text = extract_embedded_text(args.pdf, max_pages=args.max_pages)

    if args.force_ocr or len(text.strip()) < args.min_chars:
        method = "OCR"
        try:
            text = extract_text_with_ocr(args.pdf, dpi=args.dpi, max_pages=args.max_pages)
        except RuntimeError as exc:
            print(f"OCR fallback unavailable: {exc}", file=sys.stderr)
            if text.strip():
                method = "embedded text only"
            else:
                print("No extractable text found.", file=sys.stderr)
                return 1

    if not text.strip():
        print("No extractable text found.", file=sys.stderr)
        return 1

    output = args.output or args.pdf.with_suffix(".extracted.txt")
    output.write_text(text, encoding="utf-8")

    print(f"Extraction method: {method}")
    print(f"Characters extracted: {len(text)}")
    print(f"Wrote: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
