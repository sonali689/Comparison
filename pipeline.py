"""
pipeline.py

End-to-end: two PDF revisions + a purpose list in -> reconciliation
report out. Extraction backend chosen in config.py ("ollama" / "ocr" /
"region_detector"). All produce the same shape of extraction dict, so
structured_diff.py and core/reconciler.py don't need to know which ran.

drawing_number, if given, is used to look up a saved grid calibration
for the "ocr" backend (see core/calibration.py) -- comparing revisions
of the same drawing means the same calibration applies to both.
"""

import os
import tempfile

import config
from core.pdf_utils import render_pdf_pages
from core.structured_diff import diff_drawings
from core.reconciler import reconcile


def _extract_drawing(page_images: list, drawing_number: str = None) -> dict:
    if config.EXTRACTION_BACKEND == "ollama":
        from core.vision_extractor import extract_drawing
        return extract_drawing(page_images)
    elif config.EXTRACTION_BACKEND == "region_detector":
        from core.region_extractor import extract_drawing_regions
        return extract_drawing_regions(page_images, config.DETECTOR_WEIGHTS, conf=config.DETECTOR_CONF)
    else:
        from core.ocr_extractor import extract_drawing
        return extract_drawing(page_images, drawing_number=drawing_number)


def run_purpose_check(old_pdf_path: str, new_pdf_path: str, purpose_items: list,
                       drawing_number: str = None) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        old_pages = render_pdf_pages(old_pdf_path, os.path.join(tmp, "old"))
        new_pages = render_pdf_pages(new_pdf_path, os.path.join(tmp, "new"))

        old_extract = _extract_drawing(old_pages, drawing_number=drawing_number)
        new_extract = _extract_drawing(new_pages, drawing_number=drawing_number)

        diff = diff_drawings(old_extract, new_extract)
        report = reconcile(diff, purpose_items)

        return {"diff": diff, "report": report, "old_extract": old_extract, "new_extract": new_extract}
