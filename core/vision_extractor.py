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

import re

import config
from core.ollama_client import chat_json

EXTRACTION_PROMPT = """You are reading one page of an engineering drawing \
(airbag cushion component drawing). Extract EVERYTHING you can read on \
this page into the JSON schema below. Read handwritten red annotations \
and struck-through (crossed out) rows too -- they matter.

The "fn" field is the FN column value from the BOM table specifically --
a short 1-3 digit part-position number (e.g. "10", "21", "53"). It is
NEVER a material code (6-7 digit numbers like "6231960"), NEVER the
drawing number (e.g. "X670001400A"), and NEVER a quantity or usage
value. If you cannot find a genuine 1-3 digit FN value for a table row,
do not guess -- omit "fn" or leave it as an empty string rather than
putting some other number from the row into that field.

Also estimate a bounding box for each item: bbox_normalized is
[x0, y0, x1, y1] as FRACTIONS of the page width/height (0.0 to 1.0, top-left
origin), roughly enclosing where that row/item is on the page. Approximate
is fine -- this is for drawing a highlight box on the image, not for
precise measurement.

Return ONLY valid JSON, no markdown fences, no commentary, matching this \
schema exactly:

{
  "bom_rows": [
    {
      "fn": "<FN number, 1-3 digits ONLY, e.g. '10' -- see rules above>",
      "qty": "<quantity>",
      "title": "<standard title>",
      "material": "<material code>",
      "material_spec": "<material specification / supplier text>",
      "usage": "<usage number>",
      "struck_through": true/false,
      "bbox_normalized": [0.0, 0.0, 0.0, 0.0]
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
new/changed in this revision>",
      "bbox_normalized": [0.0, 0.0, 0.0, 0.0]
    }
  ]
}

If this page has no BOM table, return an empty list for "bom_rows". Be \
exhaustive with "annotations" -- err on the side of including too much \
rather than too little, especially any red handwriting. This page may
contain multiple separate component drawings -- cover all of them, not
just the first one you notice."""

FN_RE = re.compile(r"^\d{1,3}$")


def _valid_fn(row: dict) -> bool:
    """
    The single check that was missing before: nothing gets treated as a
    real FN-keyed component unless it actually looks like one. This is
    what was letting material codes and the drawing number get merged
    into bom_rows as if they were real, distinct components.
    """
    fn = str(row.get("fn", "")).strip()
    return bool(FN_RE.match(fn))


def extract_page(image_path: str) -> dict:
    return chat_json(EXTRACTION_PROMPT, image_paths=[image_path], model=config.OLLAMA_VISION_MODEL)


def extract_drawing(page_image_paths: list) -> dict:
    """
    Extracts every page and merges into one structured dict: bom_rows
    keyed by FN (later pages win if a FN appears twice), and a flat list
    of annotations tagged with which page they came from. Rows whose "fn"
    doesn't pass _valid_fn are NOT silently included -- they're dropped
    into "flagged_rows" instead, so a bad model read never masquerades as
    a real component in the diff.
    """
    bom_rows, annotations, flagged_rows = {}, [], []
    for i, path in enumerate(page_image_paths):
        page_data = extract_page(path)
        for row in page_data.get("bom_rows", []):
            row["_page"] = i + 1
            if "bbox_normalized" in row:
                row["bbox"] = tuple(row.pop("bbox_normalized"))
            if _valid_fn(row):
                bom_rows[str(row["fn"]).strip()] = row
            else:
                flagged_rows.append(row)
        for ann in page_data.get("annotations", []):
            ann["_page"] = i + 1
            if "bbox_normalized" in ann:
                ann["bbox"] = tuple(ann.pop("bbox_normalized"))
            annotations.append(ann)
    return {"bom_rows": bom_rows, "annotations": annotations, "flagged_rows": flagged_rows}
