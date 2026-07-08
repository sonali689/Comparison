"""
calibrate.py

One-time helper: run this against a single clean page of a NEW drawing
template to get the pixel positions you need for a calibration file
(see core/calibration.py). You only do this once per drawing number --
every future revision of that drawing reuses it.

Usage:
    python calibrate.py path/to/some_revision.pdf --page 1

This renders the page, auto-detects candidate grid lines, and saves an
overlay image (calibration_debug.png) with every candidate row (green)
and column (blue) line labelled with its pixel position. Open that image,
read off:
  - row_top: the y-position where the first DATA row (below the header)
    starts
  - row_pitch: the y-distance between consecutive rows
  - num_rows: how many FN rows the table has
  - col_xs: the x-positions bounding each column (need one more value
    than the number of columns -- e.g. 6 values for a 5-boundary,
    6-column table... this template happens to merge material_spec+usage
    into one detected column, which gets auto-split at read time, so 6
    x-values for the 5 "columns" you see the boundary lines for is right)
Then save that into calibrations/<drawing_number>.json (see
core/calibration.py's docstring for the exact format), or call
core/calibration.save_calibration(...) directly.
"""

import argparse
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core.pdf_utils import render_pdf_pages
from core.calibration import save_grid_debug_overlay


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf_path")
    parser.add_argument("--page", type=int, default=1, help="1-indexed page number with the BOM table")
    parser.add_argument("--out", default="calibration_debug.png")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        pages = render_pdf_pages(args.pdf_path, tmp)
        if args.page > len(pages):
            print(f"PDF only has {len(pages)} page(s).")
            return
        result = save_grid_debug_overlay(pages[args.page - 1], args.out)

    print(f"Overlay saved to {args.out}")
    print(f"Candidate row y-positions: {result['row_ys']}")
    print(f"Candidate col x-positions: {result['col_xs']}")
    print("\nOpen the overlay image, identify which lines bound the BOM table,")
    print("and write calibrations/<drawing_number>.json using those values")
    print("(see core/calibration.py docstring for the format).")


if __name__ == "__main__":
    main()
