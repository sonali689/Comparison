"""
core/report_renderer.py

Turns a diff into a VISUAL report: for every change, draws a highlight
box on the actual page image where it happened, and saves a padded
close-up crop -- so you see not just "what changed" but "where," across
however many pages the PDF has.

Works with bbox data from:
  - core/ocr_extractor.py: pixel-exact (computed directly from the
    detected table/text grid).
  - core/vision_extractor.py: model-estimated (bbox_normalized in the
    extraction prompt) -- best-effort, not measurement-grade. Treat the
    highlight as "roughly here," not as a precise coordinate.
  - core/region_extractor.py: pixel-exact (from the trained detector's
    own bounding boxes), once that backend is trained.

Entries with no bbox at all (e.g. an older extraction, or a description
the backend couldn't localize) still show up in the text report, just
without an image -- build_visual_report never drops a change for lack
of a picture.
"""

import os
import cv2


def _bbox_to_pixels(bbox, img_shape):
    x0, y0, x1, y1 = bbox
    if max(x0, y0, x1, y1) <= 1.0:  # normalized (ollama backend)
        h, w = img_shape[:2]
        return int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h)
    return int(x0), int(y0), int(x1), int(y1)  # already pixel-exact (ocr/region_detector)


def highlight_region(image_path: str, bbox, out_path: str, pad: int = 60,
                      color=(0, 0, 255), thickness=5):
    """
    Saves two images: the full page with a box drawn around bbox (so you
    see where on the sheet it is), and a padded close-up crop (so you can
    actually read what's in it). Returns (full_page_path, close_up_path),
    either of which is None if the source image couldn't be read.
    """
    img = cv2.imread(image_path)
    if img is None:
        return None, None
    x0, y0, x1, y1 = _bbox_to_pixels(bbox, img.shape)
    h, w = img.shape[:2]
    x0, y0 = max(0, min(x0, w)), max(0, min(y0, h))
    x1, y1 = max(0, min(x1, w)), max(0, min(y1, h))

    full = img.copy()
    cv2.rectangle(full, (x0, y0), (x1, y1), color, thickness)
    full_path = out_path.replace(".png", "_page.png")
    cv2.imwrite(full_path, full)

    cx0, cy0 = max(0, x0 - pad), max(0, y0 - pad)
    cx1, cy1 = min(w, x1 + pad), min(h, y1 + pad)
    crop = img[cy0:cy1, cx0:cx1].copy()
    if crop.size > 0:
        cv2.rectangle(crop, (x0 - cx0, y0 - cy0), (x1 - cx0, y1 - cy0), color, max(2, thickness - 2))
    crop_path = out_path.replace(".png", "_crop.png")
    cv2.imwrite(crop_path, crop)

    return full_path, crop_path


def _describe(c) -> str:
    """Human-readable description -- never shows raw internal join-keys
    like 'dimension:21' to the user; uses the actual extracted content."""
    prefix = "[LOW CONFIDENCE -- verify by eye] " if getattr(c, "low_confidence", False) else ""
    if c.component_type == "bom_row":
        title = (c.old_value or c.new_value or {}).get("title", "").strip()
        label = f"FN{c.key}" + (f" ({title})" if title else "")
        if c.change_type == "removed":
            return f"{prefix}{label} removed from BOM"
        if c.change_type == "added":
            return f"{prefix}{label} added to BOM"
        parts = [f"{fc.field}: '{fc.old}' \u2192 '{fc.new}'" for fc in c.changes]
        return f"{prefix}{label}: " + "; ".join(parts)
    else:
        src = c.new_value or c.old_value or {}
        text, kind = src.get("text", "").strip(), src.get("kind", "item").replace("_", " ")
        if c.change_type == "removed":
            return f"{kind.title()} removed: \"{text}\""
        if c.change_type == "added":
            return f"{kind.title()} added: \"{text}\""
        parts = [f"{fc.field}: '{fc.old}' \u2192 '{fc.new}'" for fc in c.changes]
        return f"{kind.title()} changed: " + "; ".join(parts)


def build_visual_report(diff: dict, old_pages: list, new_pages: list, out_dir: str) -> list:
    """
    Returns a list of dicts, one per non-unchanged change:
    {description, change_type, low_confidence, page_side, page_no,
     full_image, crop_image}
    full_image/crop_image are None when no bbox was available or the
    page image couldn't be matched.
    """
    os.makedirs(out_dir, exist_ok=True)

    def page_path(side, page_no):
        pages = new_pages if side == "new" else old_pages
        idx = (page_no or 1) - 1
        return pages[idx] if 0 <= idx < len(pages) else None

    records = []
    all_changes = [c for c in diff["bom_changes"] if c.change_type != "unchanged"] + diff["annotation_changes"]
    for i, c in enumerate(all_changes):
        value = c.new_value or c.old_value or {}
        side = "new" if c.new_value else "old"
        bbox = value.get("bbox")
        page_no = value.get("_page", 1)

        record = {
            "description": _describe(c), "change_type": c.change_type,
            "low_confidence": getattr(c, "low_confidence", False),
            "page_side": side, "page_no": page_no,
            "full_image": None, "crop_image": None,
        }

        if bbox:
            img_path = page_path(side, page_no)
            if img_path:
                out_path = os.path.join(out_dir, f"change_{i}.png")
                full_p, crop_p = highlight_region(img_path, bbox, out_path)
                record["full_image"], record["crop_image"] = full_p, crop_p

        records.append(record)
    return records
