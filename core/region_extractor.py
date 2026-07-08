"""
core/region_extractor.py

Turns detected regions (from core/region_detector.py) into content-keyed
records: reads the text/number inside each crop, and where a balloon_ref
(circled number) sits near a component_outline or section_or_view_label
on the SAME page, uses that number as the entity's reference id. This
nearest-neighbour linking only ever happens within one page of one
revision -- it is never used to match across revisions. Matching across
revisions is always by content (structured_diff.py).
"""

import os
import tempfile
import cv2
import numpy as np
import pytesseract

import config
from core.ollama_client import chat_json

READ_PROMPT = """Read the text/number in this image crop from an \
engineering drawing. Return ONLY valid JSON, no markdown fences: \
{"text": "<exact text you see, or empty string if illegible>", \
"color": "<black, red, or other>"}"""

TABLE_CLASSES = {"bom_table", "seam_or_spec_table"}


def _is_red_crop(crop_bgr: np.ndarray, threshold: float = 0.1) -> bool:
    if crop_bgr.size == 0:
        return False
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    mask = (cv2.inRange(hsv, (0, 70, 50), (10, 255, 255)) |
            cv2.inRange(hsv, (170, 70, 50), (180, 255, 255)))
    return (mask > 0).mean() >= threshold


def _read_crop_text(crop_bgr: np.ndarray) -> dict:
    backend = getattr(config, "REGION_READ_BACKEND", "ocr")
    if backend == "ollama":
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            cv2.imwrite(f.name, crop_bgr)
            tmp_path = f.name
        try:
            result = chat_json(READ_PROMPT, image_paths=[tmp_path], model=config.OLLAMA_VISION_MODEL)
        finally:
            os.unlink(tmp_path)
        return {"text": result.get("text", ""), "color": result.get("color", "black")}
    text = pytesseract.image_to_string(crop_bgr).strip()
    return {"text": text, "color": "red" if _is_red_crop(crop_bgr) else "black"}


def _bbox_center(bbox):
    x0, y0, x1, y1 = bbox
    return ((x0 + x1) / 2, (y0 + y1) / 2)


def _link_balloons(regions: list) -> dict:
    balloons = [r for r in regions if r["class"] == "balloon_ref"]
    targets = [i for i, r in enumerate(regions)
               if r["class"] in ("component_outline", "section_or_view_label")]
    links = {}
    for b in balloons:
        if not targets:
            break
        bc = _bbox_center(b["bbox"])
        best_i, best_dist = None, float("inf")
        for i in targets:
            tc = _bbox_center(regions[i]["bbox"])
            dist = (bc[0] - tc[0]) ** 2 + (bc[1] - tc[1]) ** 2
            if dist < best_dist:
                best_i, best_dist = i, dist
        if best_i is not None:
            links[best_i] = b.get("_text", "")
    return links


def extract_entities_from_page(image_path: str, detector) -> tuple:
    regions = detector.detect(image_path)

    for r in regions:
        if r["class"] == "balloon_ref":
            r["_text"] = _read_crop_text(r["crop"]).get("text", "").strip()

    balloon_links = _link_balloons(regions)

    bom_rows = {}
    entities = []

    for idx, r in enumerate(regions):
        cls = r["class"]

        if cls in TABLE_CLASSES:
            from core.ocr_extractor import extract_bom_table
            rows = extract_bom_table(r["crop"])
            bom_rows.update(rows)
            continue

        if cls == "balloon_ref":
            continue

        read = _read_crop_text(r["crop"])
        text = read.get("text", "").strip()
        if not text:
            continue

        ref_id = balloon_links.get(idx) or "none"
        entities.append({
            "ref_id": ref_id, "kind": cls, "text": text,
            "color": read.get("color", "black"), "confidence": r["confidence"],
        })

    return bom_rows, entities


def extract_drawing_regions(page_image_paths: list, weights_path: str, conf: float = 0.25) -> dict:
    from core.region_detector import RegionDetector
    detector = RegionDetector(weights_path, conf=conf)

    bom_rows, annotations = {}, []
    for i, path in enumerate(page_image_paths):
        page_bom, page_entities = extract_entities_from_page(path, detector)
        for fn, row in page_bom.items():
            row["_page"] = i + 1
            bom_rows[fn] = row
        for ann in page_entities:
            ann["_page"] = i + 1
            annotations.append(ann)
    return {"bom_rows": bom_rows, "annotations": annotations}
