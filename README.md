# Drawing Revision Auditor

A local tool for auditing engineering drawing revisions. Compare two PDF revisions of the same drawing, provide the stated revision purpose, and get a report showing whether each stated purpose was met plus every detected BOM / annotation change.

This repo is designed to work entirely locally, with no cloud API calls. It supports multiple extraction and reconciliation approaches and includes a region-detector workflow for a stronger long-term extraction pipeline.

## Key features

- Compare two revisions of the same drawing and detect changes in:
  - BOM rows / FN table
  - annotations, dimensions, notes, handwritten changes
- Reconcile changes against the stated purpose of the revision
- Support for multiple extraction backends:
  - `ollama` vision model extraction
  - `ocr` fallback using Tesseract + OpenCV + calibration
  - `region_detector` using a trained YOLO region detector
- Support for multiple purpose reconciliation backends:
  - `ollama` semantic reconciliation
  - `rules` keyword/fuzzy matching fallback
- One-time calibration helper for stable BOM grid extraction on OCR backend
- Simple Streamlit UI plus CLI smoke test for quick runs

## Repository layout

- `app.py` — Streamlit front-end for file upload, purpose entry, and audit report
- `pipeline.py` — end-to-end orchestration of rendering, extraction, diffing, and reconciliation
- `config.py` — global backend selection and settings
- `calibrate.py` — one-time helper to produce calibration debug overlays
- `requirements.txt` — runtime dependencies
- `core/`
  - `vision_extractor.py` — Ollama-based page extraction
  - `ocr_extractor.py` — Tesseract/OpenCV BOM and annotation extraction
  - `region_extractor.py` — region detector + crop reading extraction
  - `structured_diff.py` — diff engine for BOM rows and annotations
  - `reconciler.py` — purpose reconciliation logic
  - `ollama_client.py` — local Ollama request wrapper
  - `pdf_utils.py` — PDF page rendering and text-layer helpers
  - `calibration.py` — calibration loading/saving and overlay helpers
  - `region_detector.py` — YOLO detector inference wrapper
- `calibrations/` — saved BOM grid calibration data for OCR backend
- `detector/` — dataset preparation, training, and evaluation scripts for region detector backend
- `tests/` — sample CLI smoke test

## How it works

### Pipeline overview

`pipeline.py` does the following:

1. Render both PDF revisions to images with `core/pdf_utils.py`
2. Extract structured content from each page using the configured backend
3. Diff the old and new extractions with `core/structured_diff.py`
4. Reconcile detected changes against the stated purpose using `core/reconciler.py`

### Extraction backends

#### 1. `ollama`

- Uses a local Ollama multimodal vision model via `core/vision_extractor.py`
- Reads full pages and extracts BOM rows plus annotations as JSON
- Recommended when Ollama is available because it handles handwriting, red ink, and thin CAD fonts better than OCR
- Requires `ollama serve` running and the chosen models pulled locally

#### 2. `ocr`

- Uses Tesseract + OpenCV in `core/ocr_extractor.py`
- Locates the BOM table by grid lines and reads each cell individually with adaptive thresholding
- Uses calibration for stable BOM extraction across revisions of the same drawing template
- No LLM required

#### 3. `region_detector`

- Uses a trained YOLO region detector in `core/region_extractor.py`
- Detects regions like BOM tables, balloons, dimensions, and handwritten markup
- Reads the text inside each crop using either Tesseract or Ollama (`REGION_READ_BACKEND`)
- Most accurate long-term once a labeled dataset and model are available

### Reconciliation backends

#### 1. `ollama`

- Uses a local text LLM via `core/reconciler.py`
- Performs semantic matching between stated purpose items and detected changes
- Returns verdicts for each purpose item plus unexpected changes

#### 2. `rules`

- Keyword- and fuzzy-text matching fallback
- No LLM required
- Uses token overlap, string similarity, and explicit FN references

## Setup

1. Create and activate a Python environment.
2. Install requirements:

```bash
pip install -r requirements.txt
```

3. If using the `ocr` backend on Windows, install Tesseract and set `config.TESSERACT_CMD` to the executable path.
4. If using `ollama`, install and run Ollama locally, and pull the required models:

```bash
ollama serve
ollama pull qwen2.5vl:7b
ollama pull qwen2.5:7b
```

5. Configure `config.py`:
- `EXTRACTION_BACKEND` = `ollama`, `ocr`, or `region_detector`
- `RECONCILE_BACKEND` = `ollama` or `rules`
- `OLLAMA_HOST`, `OLLAMA_VISION_MODEL`, `OLLAMA_TEXT_MODEL`
- `TESSERACT_CMD` if using OCR
- `DETECTOR_WEIGHTS` if using `region_detector`

## Running the app

```bash
streamlit run app.py
```

Upload the previous and new revision PDFs, enter the stated purpose items, and click `Run audit`.

## CLI smoke test

Use the sample purpose-check script to run a quick audit without Streamlit:

```bash
python tests/sample_purpose_check.py old.pdf new.pdf "Remove FN22" "Change seam table"
python tests/sample_purpose_check.py old.pdf new.pdf --drawing X670001400A "Remove FN22"
```

## Calibration for OCR backend

The OCR backend is designed for the same drawing template across revisions. For best accuracy, create a calibration file in `calibrations/<drawing_number>.json`.

1. Run the helper against a clean template page:

```bash
python calibrate.py path/to/revision.pdf --page 1 --out calibration_debug.png
```

2. Open `calibration_debug.png`, inspect the candidate row and column lines, and save a JSON file with:

- `row_top` — y-pixel of the first data row
- `row_pitch` — row height in pixels
- `num_rows` — number of BOM rows
- `col_xs` — x-pixel boundaries for table columns
- `render_dpi` — DPI used for rendering

3. Save it as `calibrations/<drawing_number>.json`.

The OCR extractor scales calibration values automatically when `config.RENDER_DPI` differs from the calibration DPI.

## Region detector workflow

The region detector backend is optional and requires dataset labeling and model training.

### Dataset preparation

```bash
python detector/prepare_dataset.py --pdf_dir /path/to/pdfs --out_dir detector/dataset/images/all
```

Label images with a tool like X-AnyLabeling using the classes defined in `detector/classes.yaml`.

### Training

```bash
python detector/train.py
```

### Evaluation

```bash
python detector/evaluate.py --weights runs/drawing_regions/weights/best.pt
```

### Enabling in the app

Set `config.EXTRACTION_BACKEND = "region_detector"` and update `config.DETECTOR_WEIGHTS`.

## Important notes and limitations

- The OCR backend is fragile on thin CAD-style fonts and handwritten red strike-throughs, which is why calibration and fallback keys exist.
- `ollama` backends require a local Ollama instance and model downloads.
- `region_detector` is a stronger long-term solution, but only after collecting and training on a quality labeled dataset.
- `core/structured_diff.py` matches BOM rows by FN or fallback positional keys, and annotations by kind/ref_id.
- Unexpected changes are surfaced separately from purpose validation, so review both the audit verdicts and the full diff.

## Suggested README additions

This repo still benefits from more complete coverage in future README updates:

- Example input/output screenshots from the Streamlit app
- Sample calibration JSON and debug overlay explanation
- A known-good `requirements-dev.txt` for training/evaluation dependencies
- More explicit troubleshooting for Ollama connection failures
- A short note on how to generate a new drawing calibration from scratch
- Any dataset conventions used for `region_detector` labels and data splits

## Dependencies

- `streamlit`
- `PyMuPDF`
- `pdfplumber`
- `pytesseract`
- `opencv-python`
- `numpy`
- `requests`
- `ultralytics` (optional, only for `region_detector`)

## Contact

For questions about the pipeline or to extend the tool, inspect:
- `core/vision_extractor.py`
- `core/ocr_extractor.py`
- `core/region_extractor.py`
- `core/structured_diff.py`
- `core/reconciler.py`

