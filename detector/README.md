# Region Detector — labeling & training guide

## 1. Get pages into image form
```
python detector/prepare_dataset.py --pdf_dir /path/to/your/pdfs --out_dir detector/dataset/images/all
```

## 2. Label with X-AnyLabeling
Repo: https://github.com/CVHub520/X-AnyLabeling

1. **Import images**: File → Image File Directory (Ctrl+U) → `detector/dataset/images/all`.
2. **Set your label list** to the 9 classes in `classes.yaml`.
3. **Load an AI-assist model** (Ctrl+A). For zero-shot pre-labeling with
   no training at all:
   - **Grounding DINO** — text-prompted. Prompt once per class, e.g.
     "closed hand-drawn outline shape" for `component_outline`, "table
     with rows and columns" for `bom_table`/`seam_or_spec_table`,
     "circled number" for `balloon_ref`, "red handwritten text" for
     `handwritten_markup`, "small boxed text label" for
     `section_or_view_label`.
   - **UPN** — generic "propose boxes for anything" mode, no prompt
     needed, useful as a first pass you then reclassify by hand.
4. **Run inference**, then **correct rather than create from scratch**
   — fix wrong boxes, add missed ones, fix misclassified labels.
5. **Export YOLO format** into `detector/dataset/`.

Target: **150-300 labeled pages**, mixing drawings that do and don't box
their components, before your first training run. Short on that many?
Start with 30-50 to validate the pipeline end-to-end (expect a weak
model), then grow the dataset once the approach is proven worth
investing further in.

## 3. Split into train/val
```
detector/dataset/
├── images/{train,val}/*.png
└── labels/{train,val}/*.txt
```
80/20 or 85/15 is reasonable at this size. Make sure boxed and unboxed
drawings both appear in both splits.

## 4. Train
```
pip install ultralytics
python detector/train.py
```
Default `yolov8n.pt` (nano — fastest, least data-hungry). Move to
`yolov8s.pt` once you have more data. `imgsz=1280` because several
classes (balloon numbers, dimension text) are small relative to the
page — don't drop this without checking recall on those classes.

## 5. Evaluate
```
python detector/evaluate.py --weights runs/drawing_regions/weights/best.pt
```
Per-class mAP + prediction overlays on your val set, so you can see
exactly which classes are weak instead of one aggregate number.

## 6. Wire it into the app
Set `config.EXTRACTION_BACKEND = "region_detector"` and point
`config.DETECTOR_WEIGHTS` at your trained `best.pt`.

## Reality check
Keep `ollama` or `ocr` as your working path while this is in progress —
this needs a real, checked dataset and at least one full
train/evaluate/correct loop before it should be trusted over the two
backends that work today. Run it in parallel, not as a blocker.
