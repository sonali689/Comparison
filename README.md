# Drawing Revision Auditor

Upload the old and new revision of an engineering drawing, state the
purpose of the revision (the ECO / handwritten notes), and get back:
- a verdict per purpose item (met / partially met / not met, with evidence)
- a list of unexpected changes that don't map to any stated purpose
- a full table of every BOM/FN field change
- a full table of every other annotation/dimension/text change

**Fully local. No cloud API calls anywhere.**

## Design principle
Never compare positions. BOM rows are keyed by FN number; everything
else is matched by content (a balloon number, or fuzzy text similarity
for unlabelled notes/dimensions). A component that moved to a different
spot on a redrawn sheet is still the same key/content, so it's still
matched correctly.

## This was tested against a real drawing, not just written and shipped
During development this was run against an actual Autoliv drawing
(X670001400A, revisions 00-63/00-65/00-67), and the results changed the
design more than once:

1. **No PDF text layer.** `pdfplumber.extract_text()` returned nothing on
   any of 8 real pages checked — these are fully rasterised, so OCR/vision
   reading isn't optional, there's no selectable-text shortcut available.
2. **Global OCR thresholding (OTSU) failed completely** on the BOM table
   header, even at high DPI with upscaling — these drawings render with
   very low-contrast, thin CAD-weight fonts. Isolating each table **cell**
   individually and using **adaptive thresholding** instead fixed it —
   confirmed by reading the header correctly (`FN`, `QTY`, `STANDARD
   TITLE`...) once cropped tightly.
3. **Auto-detecting the table grid from scratch was unreliable** — a page
   can have more than one grid-like region with different row pitches,
   and "pick the longest consistent run" picked the wrong one. Since
   you're always comparing revisions of the *same* drawing template, the
   fix was to calibrate the grid geometry once per drawing number
   (`core/calibration.py`, `calibrate.py`) and reuse it — not re-detect
   it every run.
4. **Red strike-through marks corrupt the FN digits themselves**, not
   just the row text — several different struck rows OCR'd to the same
   garbled text, which was silently overwriting each other in a
   dictionary keyed by that text. Fixed by falling back to a
   `unread_row_p{page}_r{idx}` key (instead of the unreliable OCR guess)
   whenever the FN can't be confidently parsed, and flagging those rows
   `fn_confidence: "low"` end-to-end so the report calls them out for
   manual review instead of silently guessing.
5. Clean, unstruck rows (FN 10, 21, 51, 200, 201, 800, 801, 24, 53, 54)
   read correctly with this approach. Struck-through rows are still the
   hardest case and are honestly flagged, not silently trusted.

A working calibration for X670001400A, from that actual test, ships in
`calibrations/X670001400A.json`.

## Visual highlighting — what it does, tested against real data
Every change now carries the pixel region it happened in through the
whole pipeline, and `core/report_renderer.py` draws a highlight box on
the actual page (plus a padded close-up crop) for each one — shown
inline in the Streamlit app. This works across however many pages a PDF
has; each change is tagged with its own page number, so a multi-page,
multi-drawing PDF highlights the right page for each item, not just page 1.

Two bugs fixed on the way to this (found from an earlier bad run, not
hypothetical): raw internal join-keys (`dimension:21`) were leaking into
the UI instead of being translated into readable descriptions — fixed in
`report_renderer._describe`. And the `ollama` backend's output wasn't
validated at all, so a misread material code or the drawing number could
end up masquerading as a real FN-keyed component — fixed with `_valid_fn`
in `vision_extractor.py`, which now rejects anything that isn't a
genuine 1-3 digit FN before it's ever added to `bom_rows`.

**Re-tested end to end against the real 00-63 → 00-65 pair after those
fixes**, using the `ocr` backend: the highlighting pipeline itself works
correctly (crops generated, correctly located, for every change). But
the diff also surfaced a lot of false positives — rows that should be
identical between these two revisions showing as "changed" because OCR
read the same printed text slightly differently on the two separate
runs (`'6231960 14A'` vs `'6231 960 14A'`, `'6050183 OIA'` vs `'6050183
O1A.'`). Whitespace was normalized away as a safe first pass, but the
dominant noise is character-level misreads (O/0 confusion, stray
punctuation), which weren't touched on purpose — silently "cleaning"
those risks masking a genuine material-code change, which is exactly
what this tool is supposed to catch.

**Practical takeaway: the `ocr` backend's false-positive rate on this
scan quality is high enough that its raw diff output needs a human
skim, not blind trust.** The `ollama` backend should handle this
failure mode fundamentally better, since it reads content semantically
rather than character-by-character — but that hasn't been confirmed
against real data yet (no Ollama instance was available in the sandbox
used to build this). Test `ollama` against the same 00-63/00-65 pair
before deciding which backend to rely on.

## Three extraction backends
Set in `config.py` (`EXTRACTION_BACKEND`):

| Backend | How it reads a page | Needs setup? |
|---|---|---|
| `ollama` | Local vision LLM reads the whole page contextually | Install Ollama, pull a model |
| `ocr` | Tesseract + calibrated grid detection, per-cell adaptive threshold | Tesseract install; calibrate new drawing templates once (`calibrate.py`) |
| `region_detector` | Trained YOLO model crops regions first, each read separately | Needs a labeled dataset + training run — `detector/README.md` |

`ollama` is the least fiddly if you have it running — it doesn't need
per-drawing-template calibration the way `ocr` does, since it reads
contextually rather than via a fixed pixel grid.

## Repo layout
```
config.py                      -- all tunables: backends, models, paths
pipeline.py                    -- orchestrator, picks backend per config
app.py                         -- Streamlit UI
calibrate.py                   -- one-time grid calibration helper for a new drawing template
calibrations/
  X670001400A.json              -- real, tested calibration for this drawing
core/
  pdf_utils.py                  -- render PDF pages to PNG; text-layer check
  ollama_client.py                -- thin local Ollama HTTP wrapper
  vision_extractor.py               -- "ollama" backend
  ocr_extractor.py                    -- "ocr" backend (calibrated grid + per-cell OCR)
  calibration.py                        -- load/save/visualize per-drawing grid calibration
  region_detector.py                      -- "region_detector" backend: YOLO inference
  region_extractor.py                       -- "region_detector" backend: reads each crop
  structured_diff.py                          -- ID-keyed diff, shared by all backends
  reconciler.py                                 -- purpose-vs-changes check (ollama or rules)
detector/                        -- training subproject for region_detector
  classes.yaml                    -- the 9-class taxonomy
  README.md                        -- labeling (X-AnyLabeling) + training guide
  prepare_dataset.py                 -- PDFs -> page images for labeling
  data.yaml                            -- YOLO dataset config
  train.py                               -- fine-tune YOLOv8
  evaluate.py                              -- per-class mAP + prediction overlays
tests/
  sample_purpose_check.py         -- CLI smoke test, no Streamlit needed
```

## Setup
```
pip install -r requirements.txt
```
- **`ollama` backend**: install [Ollama](https://ollama.com), run `ollama serve`,
  pull the models named in `config.py`.
- **`ocr` backend**: install Tesseract, set `config.TESSERACT_CMD` if not
  on PATH. For a NEW drawing template (not X670001400A), run
  `python calibrate.py some_revision.pdf --page 1` first and follow its
  instructions to save a calibration file.
- **`region_detector` backend**: follow `detector/README.md` end to end first.

No API key needed anywhere.

## Running it
```
streamlit run app.py
```
or:
```
python tests/sample_purpose_check.py old_rev.pdf new_rev.pdf --drawing X670001400A "Remove FN22" "Change seam table"
```

## Known gaps — the honest part
1. **Strike-through detection threshold needs tuning.** In testing it
   caught some struck rows but missed others — the red-pixel-fraction
   check (`core/ocr_extractor.py`, `_is_red_region` threshold) is checked
   against the full row width, which may be too strict when a strike mark
   only covers part of a row. Worth tuning against more real examples.
2. **Rows with `fn_confidence: "low"`** (mostly struck-through rows) have
   unreliable field values beyond the struck/not-struck flag — the
   report calls these out explicitly; treat them as "check this by eye,"
   not as a confirmed reading.
3. **`ocr` backend needs one-time calibration per drawing number.** No
   calibration on file → falls back to auto-detection, which is
   meaningfully less reliable (see point 3 in the testing section above).
4. **`region_detector` is not a drop-in win yet** — needs a real labeled
   dataset and at least one train/evaluate/correct loop before it should
   be trusted over `ollama`/`ocr`.
5. **`ollama` backend is the least tested of the three in this
   development pass** — no Ollama instance was available in the sandbox
   used to build this, so its real-world accuracy on these drawings is a
   prediction based on how badly character-level OCR struggled, not a
   confirmed result. Test it before relying on it, same as everything else here.

Test against a revision pair you already know the answer to before
trusting any backend on one you don't.
