"""
core/ocr_extractor.py

Fallback extraction backend: Tesseract OCR + OpenCV, no LLM. Two things
were validated against a real drawing during development and drive the
design here:

1. Reading a whole row/page at once with a global (OTSU) threshold fails
   badly on these drawings -- they render as very low-contrast, thin
   CAD-weight text. Isolating each CELL individually and using adaptive
   thresholding fixed it (see _ocr_cell).
2. Auto-detecting the table's row/column grid from scratch, from a single
   page, is unreliable -- pages can have more than one grid-like region
   with different pitches, and "pick the longest consistent run" doesn't
   reliably pick the right one. Since you're always comparing revisions
   of the SAME drawing template, the fix is to calibrate the grid
   geometry ONCE per drawing number (see core/calibration.py) and reuse
   it -- not to re-detect it from scratch on every run. Auto-detection is
   kept as a fallback for uncalibrated drawings, clearly less reliable.

A third real issue: red strike-through marks corrupt the FN digits
themselves, not just the rest of the row -- several different struck
rows can OCR to the same garbled text. Silently keying a dict by that
text would let one row's data overwrite another's. Rows whose FN can't
be confidently parsed get a page/row-position fallback key instead
(`unread_row_p{page}_r{idx}`) so they never collide -- and since
revisions of the same drawing preserve row order (items get struck
through in place rather than deleted/reordered), matching those
fallback keys by page+row-index across revisions is a reasonable,
narrowly-scoped exception to "never match by position": it only ever
applies to cells OCR couldn't read at all, never to the primary
FN-based matching.
"""

import re
import cv2
import numpy as np
import pytesseract

import config

if config.TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = config.TESSERACT_CMD

RED_HSV_RANGES = [
    ((0, 70, 50), (10, 255, 255)),
    ((170, 70, 50), (180, 255, 255)),
]

BOM_COLUMNS = ["fn", "qty", "title", "material", "material_spec", "usage"]
FN_RE = re.compile(r"^\d{1,3}$")


def _adaptive_mask(gray: np.ndarray, block_size: int = 35, c: int = 10) -> np.ndarray:
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY_INV, block_size, c)


def _red_mask(img_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lo, hi in RED_HSV_RANGES:
        mask |= cv2.inRange(hsv, np.array(lo), np.array(hi))
    return mask


def _is_red_region(mask: np.ndarray, x0, y0, x1, y1, threshold=0.08) -> bool:
    region = mask[max(0, y0):y1, max(0, x0):x1]
    if region.size == 0:
        return False
    return (region > 0).mean() >= threshold


def _line_positions(projection: np.ndarray, thresh_frac: float = 0.3) -> list:
    if projection.max() == 0:
        return []
    thr = projection.max() * thresh_frac
    positions, in_line, start = [], False, 0
    for i, v in enumerate(projection):
        if v > thr and not in_line:
            in_line, start = True, i
        elif v <= thr and in_line:
            in_line = False
            positions.append((start + i) // 2)
    if in_line:
        positions.append((start + len(projection)) // 2)
    return positions


def _merge_close(positions: list, min_gap: int) -> list:
    if not positions:
        return []
    merged = [positions[0]]
    for p in positions[1:]:
        if p - merged[-1] < min_gap:
            merged[-1] = (merged[-1] + p) // 2
        else:
            merged.append(p)
    return merged


def _detect_grid_lines(mask: np.ndarray):
    h, w = mask.shape
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(20, int(w * 0.15)), 1))
    h_lines = cv2.morphologyEx(mask, cv2.MORPH_OPEN, h_kernel, iterations=1)
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(20, int(h * 0.01))))
    v_lines = cv2.morphologyEx(mask, cv2.MORPH_OPEN, v_kernel, iterations=1)
    return h_lines, v_lines


def _find_table_row_band(row_ys: list, min_rows: int = 5):
    """Best-effort fallback for uncalibrated drawings: longest run of
    near-equally-spaced lines. Less reliable than calibration -- see
    module docstring."""
    if len(row_ys) < min_rows + 1:
        return []
    diffs = [row_ys[i + 1] - row_ys[i] for i in range(len(row_ys) - 1)]
    best_run, cur_run = [], [0]
    for i in range(1, len(diffs)):
        if abs(diffs[i] - diffs[cur_run[-1]]) <= 0.25 * diffs[cur_run[-1]]:
            cur_run.append(i)
        else:
            if len(cur_run) > len(best_run):
                best_run = cur_run
            cur_run = [i]
    if len(cur_run) > len(best_run):
        best_run = cur_run
    if len(best_run) < min_rows:
        return []
    idxs = best_run + [best_run[-1] + 1]
    return [row_ys[i] for i in idxs]


def _ocr_cell(gray_cell: np.ndarray) -> str:
    if gray_cell.size == 0:
        return ""
    mask = cv2.adaptiveThreshold(gray_cell, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY_INV, 31, 10)
    inv = cv2.bitwise_not(mask)
    return pytesseract.image_to_string(inv, config="--psm 7").strip()


def _split_spec_and_usage(combined_text: str) -> tuple:
    m = re.search(r"([\d.]+)\s*$", combined_text.strip())
    if m:
        return combined_text[:m.start()].strip(" |"), m.group(1)
    return combined_text.strip(), ""


def _row_bands_from_calibration(cal: dict) -> list:
    row_top, pitch, n = cal["row_top"], cal["row_pitch"], cal["num_rows"]
    return [int(row_top + i * pitch) for i in range(n + 1)]


def _row_bands_auto(mask: np.ndarray):
    h_lines, v_lines = _detect_grid_lines(mask)
    row_ys = _merge_close(_line_positions(h_lines.sum(axis=1)), min_gap=15)
    col_xs = _merge_close(_line_positions(v_lines.sum(axis=0)), min_gap=15)
    table_rows = _find_table_row_band(row_ys)
    return table_rows, col_xs


def extract_bom_table(img_bgr: np.ndarray, calibration: dict = None, page_no: int = 1) -> dict:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    if calibration:
        row_bands = _row_bands_from_calibration(calibration)
        col_xs = [int(x) for x in calibration["col_xs"]]
        start_row_idx = 0  # calibration's row_top already points at the first DATA row
    else:
        mask = _adaptive_mask(gray)
        row_bands, col_xs = _row_bands_auto(mask)
        start_row_idx = 1  # first band in auto-detected output is the header row

    if len(row_bands) < 3 or len(col_xs) < 3:
        return {}  # no BOM-shaped grid found/calibrated for this page

    red_mask = _red_mask(img_bgr)
    strike_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN,
                                    cv2.getStructuringElement(cv2.MORPH_RECT,
                                                               (max(20, int(img_bgr.shape[1] * 0.03)), 1)))

    bom_rows = {}
    for r in range(start_row_idx, len(row_bands) - 1):
        y0, y1 = row_bands[r], row_bands[r + 1]
        cell_texts = [_ocr_cell(gray[y0:y1, col_xs[c]:col_xs[c + 1]])
                      for c in range(len(col_xs) - 1) if col_xs[c + 1] - col_xs[c] >= 15]
        if not cell_texts:
            continue

        struck = _is_red_region(strike_mask, col_xs[0], y0 - 2, col_xs[-1], y1 + 2)

        first_digits = re.sub(r"[^\d]", "", cell_texts[0])[:3] if cell_texts[0] else ""
        if cell_texts[0].strip().isdigit() and FN_RE.match(cell_texts[0].strip()):
            fn_key, fn_confidence = cell_texts[0].strip(), "high"
        else:
            # FN couldn't be confidently read -- very common on
            # struck-through rows, where the red line corrupts the
            # digits. Use a position-based fallback key instead of a
            # possibly-garbled/colliding OCR guess, so it never silently
            # overwrites another row's data. See module docstring.
            fn_key, fn_confidence = f"unread_row_p{page_no}_r{r}", "low"

        values = dict(zip(BOM_COLUMNS, cell_texts))
        if len(cell_texts) == 5:
            spec, usage = _split_spec_and_usage(cell_texts[4])
            values["material_spec"], values["usage"] = spec, usage

        bom_rows[fn_key] = {
            "fn": fn_key,
            "qty": values.get("qty", ""),
            "title": values.get("title", ""),
            "material": values.get("material", ""),
            "material_spec": values.get("material_spec", ""),
            "usage": values.get("usage", ""),
            "struck_through": bool(struck),
            "fn_confidence": fn_confidence,
        }
    return bom_rows


NUM_RE = re.compile(r"^[\d.]+$")


def extract_annotations(img_bgr: np.ndarray) -> list:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    mask = _adaptive_mask(gray)
    red_mask = _red_mask(img_bgr)

    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 5))
    dilated = cv2.dilate(mask, dilate_kernel, iterations=1)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    annotations = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w < 15 or h < 10 or w > img_bgr.shape[1] * 0.9:
            continue
        text = _ocr_cell(gray[y:y + h, x:x + w])
        if not text:
            continue
        is_red = _is_red_region(red_mask, x, y, x + w, y + h, threshold=0.2)

        flat = text.replace(":", "").replace(" ", "")
        if NUM_RE.match(flat):
            kind = "dimension"
        elif is_red and len(text) > 3:
            kind = "handwritten_revision"
        else:
            kind = "other"

        annotations.append({"ref_id": "none", "kind": kind, "text": text,
                             "color": "red" if is_red else "black"})
    return annotations


def extract_page_local(image_path: str, calibration: dict = None, page_no: int = 1) -> dict:
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"Could not read image: {image_path}")
    return {"bom_rows": extract_bom_table(img_bgr, calibration=calibration, page_no=page_no),
            "annotations": extract_annotations(img_bgr)}


def extract_drawing(page_image_paths: list, drawing_number: str = None) -> dict:
    calibration = None
    if drawing_number:
        from core.calibration import load_calibration
        calibration = load_calibration(drawing_number, current_dpi=config.RENDER_DPI)

    bom_rows, annotations = {}, []
    for i, path in enumerate(page_image_paths):
        page_data = extract_page_local(path, calibration=calibration, page_no=i + 1)
        for fn, row in page_data["bom_rows"].items():
            row["_page"] = i + 1
            bom_rows[fn] = row
        for ann in page_data["annotations"]:
            ann["_page"] = i + 1
            annotations.append(ann)
    return {"bom_rows": bom_rows, "annotations": annotations}
