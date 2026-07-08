"""
core/calibration.py

Grid-line auto-detection from scratch turned out to be unreliable on
real pages (multiple grid-like regions with different pitches confuse
"find the longest consistent run"). But you're always comparing
revisions of the SAME drawing template, and that template's table
geometry doesn't move between revisions -- only the red marks do. So:
calibrate the row/column pixel positions ONCE per drawing number, save
it, and reuse it for every revision of that drawing. Auto-detection is
kept as a best-effort fallback for uncalibrated drawings, clearly
labelled as less reliable.

Calibration file format (JSON), one per drawing number, e.g.
calibrations/X670001400A.json:
{
  "row_top": 555,       // y-pixel where the first DATA row starts (below header)
  "row_pitch": 113.5,   // pixel height per row
  "num_rows": 22,       // how many FN rows the table has
  "col_xs": [654, 807, 959, 3106, 3826, 6216],  // column boundary x-pixels
  "render_dpi": 300     // DPI these pixel values were measured at -- if you
                         // render at a different DPI, values get rescaled
}
All pixel values assume config.RENDER_DPI at calibration time; if the
running DPI differs, positions are scaled proportionally.
"""

import json
import os

CALIBRATION_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "calibrations")


def load_calibration(drawing_number: str, current_dpi: int) -> dict:
    """Returns a calibration dict scaled to current_dpi, or None if not calibrated."""
    path = os.path.join(CALIBRATION_DIR, f"{drawing_number}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        cal = json.load(f)
    scale = current_dpi / cal.get("render_dpi", current_dpi)
    if scale != 1.0:
        cal = dict(cal)
        cal["row_top"] = cal["row_top"] * scale
        cal["row_pitch"] = cal["row_pitch"] * scale
        cal["col_xs"] = [x * scale for x in cal["col_xs"]]
    return cal


def save_calibration(drawing_number: str, row_top: float, row_pitch: float,
                      num_rows: int, col_xs: list, render_dpi: int):
    os.makedirs(CALIBRATION_DIR, exist_ok=True)
    path = os.path.join(CALIBRATION_DIR, f"{drawing_number}.json")
    with open(path, "w") as f:
        json.dump({"row_top": row_top, "row_pitch": row_pitch, "num_rows": num_rows,
                   "col_xs": col_xs, "render_dpi": render_dpi}, f, indent=2)
    return path


def save_grid_debug_overlay(image_path: str, out_path: str):
    """
    Draws every auto-detected candidate row/column line on the page,
    labelled with its pixel position, so you can read off the right
    values for row_top/row_pitch/col_xs by eye instead of guessing.
    Run this once per new drawing template, look at the output image,
    then call save_calibration() with the values you read off it.
    """
    import cv2
    from core.ocr_extractor import _adaptive_mask, _detect_grid_lines, _line_positions, _merge_close

    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = _adaptive_mask(gray)
    h_lines, v_lines = _detect_grid_lines(mask)
    row_ys = _merge_close(_line_positions(h_lines.sum(axis=1)), min_gap=15)
    col_xs = _merge_close(_line_positions(v_lines.sum(axis=0)), min_gap=15)

    overlay = img.copy()
    for y in row_ys:
        cv2.line(overlay, (0, y), (overlay.shape[1], y), (0, 255, 0), 2)
        cv2.putText(overlay, str(y), (5, max(15, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    for x in col_xs:
        cv2.line(overlay, (x, 0), (x, overlay.shape[0]), (255, 0, 0), 2)
        cv2.putText(overlay, str(x), (x + 5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    cv2.imwrite(out_path, overlay)
    return {"row_ys": row_ys, "col_xs": col_xs}
