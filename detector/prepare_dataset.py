"""
detector/prepare_dataset.py

Renders every page of every PDF in --pdf_dir to a PNG in --out_dir, ready
to import into a labeling tool (X-AnyLabeling, CVAT, etc.).

Usage:
    python detector/prepare_dataset.py --pdf_dir /path/to/pdfs --out_dir detector/dataset/images/all
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.pdf_utils import render_pdf_pages


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    pdf_files = [f for f in os.listdir(args.pdf_dir) if f.lower().endswith(".pdf")]
    if not pdf_files:
        print(f"No PDFs found in {args.pdf_dir}")
        return

    total = 0
    for fname in sorted(pdf_files):
        pdf_path = os.path.join(args.pdf_dir, fname)
        paths = render_pdf_pages(pdf_path, args.out_dir, dpi=args.dpi)
        print(f"{fname}: {len(paths)} page(s) -> {args.out_dir}")
        total += len(paths)

    print(f"\nDone. {total} page images written to {args.out_dir}")
    print("Next: import this folder into X-AnyLabeling and start labeling (see detector/README.md).")


if __name__ == "__main__":
    main()
