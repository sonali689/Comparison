"""
config.py — central place to tune the pipeline without touching code.
"""

# --- Extraction backend --------------------------------------------------
# "ollama":          local multimodal LLM reads each whole page. No
#                    training/tuning needed. 100% local, no cloud calls.
#                    Recommended if you have Ollama running -- it reads
#                    tables/handwriting contextually rather than relying
#                    on character-level OCR, which struggled badly on
#                    these drawings' thin CAD-weight fonts (see README).
# "ocr":             Tesseract + OpenCV, no LLM. Locates the BOM table by
#                    its grid lines (not by OCR-reading "FN"/"QTY", which
#                    failed in testing) and reads each cell individually
#                    with adaptive thresholding. Works without a GPU/Ollama.
# "region_detector": trained YOLO model crops regions first, then reads
#                    each crop. Most accurate once trained, but needs a
#                    labeled dataset + training run first -- detector/README.md.
EXTRACTION_BACKEND = "ollama"

# --- Reconciliation backend -----------------------------------------------
# "ollama": semantic reconciliation via a local text LLM (recommended).
# "rules":  keyword/fuzzy text matching only, no LLM.
RECONCILE_BACKEND = "ollama"

# --- Ollama settings -------------------------------------------------------
OLLAMA_HOST = "http://localhost:11434"
OLLAMA_VISION_MODEL = "qwen2.5vl:7b"    # or "llava:13b" -- whatever you've pulled
OLLAMA_TEXT_MODEL = "qwen2.5:7b"        # used for purpose reconciliation

# --- Rendering ---------------------------------------------------------
# 300+ recommended -- these drawings use thin, low-contrast CAD line-weight
# fonts that need real resolution to read reliably at all.
RENDER_DPI = 300

# --- Rule-based reconciler sensitivity (only used if RECONCILE_BACKEND == "rules")
RULES_MATCH_THRESHOLD = 0.28

# --- Tesseract path (only needed if EXTRACTION_BACKEND == "ocr", or if
# REGION_READ_BACKEND == "ocr")
# e.g. r"C:\Program Files\Tesseract-OCR\tesseract.exe" on Windows
TESSERACT_CMD = None

# --- Region detector settings (only used if EXTRACTION_BACKEND == "region_detector")
DETECTOR_WEIGHTS = "detector/runs/drawing_regions/weights/best.pt"
DETECTOR_CONF = 0.25
# How each cropped region's text/number gets read: "ocr" (Tesseract, fast,
# free) or "ollama" (local vision LLM, more robust on handwriting/red ink).
REGION_READ_BACKEND = "ocr"
