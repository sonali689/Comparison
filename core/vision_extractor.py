"""
core/vision_extractor.py

Structured, ID-keyed extraction of BOM rows + annotations from a drawing
page image, using a local multimodal LLM via Ollama. Recommended backend:
it reads handwriting, red strike-throughs, and table structure the way a
person would, instead of relying on character-level OCR -- which, tested
against real drawings in this repo's development, failed badly on the
thin CAD-weight header font even after heavy preprocessing (see README).

Everything comes back keyed by its own ID (FN number, balloon/ref number)
instead of its (x, y) position, so a component that moved to a different
spot on the redrawn sheet is still matched correctly later in
core/structured_diff.py.
"""

import config
from core.ollama_client import chat_json

EXTRACTION_PROMPT = """You are reading one page of an engineering drawing \
(airbag cushion component drawing). Extract EVERYTHING you can read on \
this page into the JSON schema below. Read handwritten red annotations \
and struck-through (crossed out) rows too -- they matter.

Return ONLY valid JSON, no markdown fences, no commentary, matching this \
schema exactly:

{
  "bom_rows": [
    {
      "fn": "<FN number as string, e.g. '10'>",
      "qty": "<quantity>",
      "title": "<standard title>",
      "material": "<material code>",
      "material_spec": "<material specification / supplier text>",
      "usage": "<usage number>",
      "struck_through": true/false
    }
  ],
  "annotations": [
    {
      "ref_id": "<any number/letter this item is labelled with on the \
sheet -- balloon number, view number, seam ID, FN cross-reference, or \
'none' if unlabelled>",
      "kind": "<one of: dimension, note, handwritten_revision, \
section_title, seam_table_row, label, other>",
      "text": "<the exact text/value, e.g. '584', 'ADD STITCHES ON HEAT \
PATCH', 'STEP 4', '40:5'>",
      "color": "<black, red, or other -- red usually marks something \
new/changed in this revision>"
    }
  ]
}

If this page has no BOM table, return an empty list for "bom_rows". Be \
exhaustive with "annotations" -- err on the side of including too much \
rather than too little, especially any red handwriting."""


def extract_page(image_path: str) -> dict:
    return chat_json(EXTRACTION_PROMPT, image_paths=[image_path], model=config.OLLAMA_VISION_MODEL)


def extract_drawing(page_image_paths: list) -> dict:
    """
    Extracts every page and merges into one structured dict: bom_rows
    keyed by FN (later pages win if a FN appears twice), and a flat list
    of annotations tagged with which page they came from.
    """
    bom_rows, annotations = {}, []
    for i, path in enumerate(page_image_paths):
        page_data = extract_page(path)
        for row in page_data.get("bom_rows", []):
            row["_page"] = i + 1
            bom_rows[row["fn"]] = row
        for ann in page_data.get("annotations", []):
            ann["_page"] = i + 1
            annotations.append(ann)
    return {"bom_rows": bom_rows, "annotations": annotations}
