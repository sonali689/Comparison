"""
core/pdf_utils.py

PDF -> page images (for the vision/OCR extractors), plus a helper to
check for a real, selectable text layer via pdfplumber.

Checked against real Autoliv drawing PDFs during development: they have
NO selectable text layer at all (pdfplumber.extract_text() returns empty
on every page) -- they're fully rasterised. So OCR/vision reading isn't
optional for this kind of source file; has_text_layer()/extract_text_layer()
are here in case you work with a different, CAD-exported PDF that does
keep real text, but don't expect them to help on scanned drawings.
"""

import os
import fitz  # PyMuPDF
import pdfplumber

import config


def render_pdf_pages(pdf_path: str, out_dir: str, dpi: int = None) -> list:
    """Renders every page of pdf_path to a PNG in out_dir, in page order."""
    dpi = dpi or config.RENDER_DPI
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    zoom = dpi / 72
    mat = fitz.Matrix(zoom, zoom)
    doc = fitz.open(pdf_path)
    base = os.path.splitext(os.path.basename(pdf_path))[0]
    for i, page in enumerate(doc):
        pix = page.get_pixmap(matrix=mat)
        out_path = os.path.join(out_dir, f"{base}_p{i + 1}.png")
        pix.save(out_path)
        paths.append(out_path)
    doc.close()
    return paths


def has_text_layer(pdf_path: str, page_no: int, min_chars: int = 30) -> bool:
    """Quick check: does this page have a real, extractable text layer?"""
    with pdfplumber.open(pdf_path) as pdf:
        if page_no >= len(pdf.pages):
            return False
        text = pdf.pages[page_no].extract_text() or ""
        return len(text.strip()) >= min_chars


def extract_text_layer(pdf_path: str, page_no: int) -> str:
    with pdfplumber.open(pdf_path) as pdf:
        return pdf.pages[page_no].extract_text() or ""
