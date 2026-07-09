"""
pipeline.py

End-to-end: two PDF revisions + a purpose list in -> reconciliation
report + visual highlight images out. Extraction backend chosen in
config.py ("ollama" / "ocr" / "region_detector").

Unlike earlier versions, page images are rendered into a persistent
`runs/<timestamp>/` folder instead of a temp dir that gets deleted --
the visual report needs those images to still exist after extraction
finishes, so you (and the Streamlit app) can look at the actual
highlighted regions, not just a text table.
"""

import os
import time

import config
from core.pdf_utils import render_pdf_pages
from core.structured_diff import diff_drawings
from core.reconciler import reconcile
from core.report_renderer import build_visual_report


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
                       drawing_number: str = None, run_dir: str = None) -> dict:
    run_dir = run_dir or os.path.join("runs", time.strftime("%Y%m%d_%H%M%S"))
    old_dir = os.path.join(run_dir, "old_pages")
    new_dir = os.path.join(run_dir, "new_pages")
    report_dir = os.path.join(run_dir, "report_images")

    old_pages = render_pdf_pages(old_pdf_path, old_dir)
    new_pages = render_pdf_pages(new_pdf_path, new_dir)

    old_extract = _extract_drawing(old_pages, drawing_number=drawing_number)
    new_extract = _extract_drawing(new_pages, drawing_number=drawing_number)

    diff = diff_drawings(old_extract, new_extract)
    report = reconcile(diff, purpose_items)
    visual_changes = build_visual_report(diff, old_pages, new_pages, report_dir)

    return {
        "diff": diff, "report": report, "visual_changes": visual_changes,
        "old_extract": old_extract, "new_extract": new_extract, "run_dir": run_dir,
    }
