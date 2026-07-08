"""
tests/sample_purpose_check.py

Quick CLI smoke test, no Streamlit needed:
    python tests/sample_purpose_check.py old.pdf new.pdf "Remove FN22" "Change seam table"
    python tests/sample_purpose_check.py old.pdf new.pdf --drawing X670001400A "Remove FN22"
"""
import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline import run_purpose_check

if __name__ == "__main__":
    args = sys.argv[1:]
    drawing_number = None
    if "--drawing" in args:
        i = args.index("--drawing")
        drawing_number = args[i + 1]
        args = args[:i] + args[i + 2:]

    if len(args) < 3:
        print("Usage: python tests/sample_purpose_check.py old.pdf new.pdf \"purpose 1\" "
              "[\"purpose 2\" ...] [--drawing X670001400A]")
        sys.exit(1)

    old_pdf, new_pdf, *purposes = args
    result = run_purpose_check(old_pdf, new_pdf, purposes, drawing_number=drawing_number)
    print(json.dumps(result["report"], indent=2))
