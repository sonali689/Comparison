"""
core/region_detector.py

Inference wrapper around the trained region detector (see detector/).
Given a page image, returns every detected region as a class + bounding
box + cropped image. This is purely a "find and crop" step -- it does
NOT identify which specific component something is, and it does NOT
compare anything across revisions. That happens in region_extractor.py
(reads content from each crop) and structured_diff.py (matches by that
content), same as the other backends.
"""

import cv2
from ultralytics import YOLO


class RegionDetector:
    def __init__(self, weights_path: str, conf: float = 0.25):
        self.model = YOLO(weights_path)
        self.conf = conf

    def detect(self, image_path: str) -> list:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")

        results = self.model.predict(image_path, conf=self.conf, verbose=False)[0]
        regions = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            cls_name = self.model.names[cls_id]
            x0, y0, x1, y1 = map(int, box.xyxy[0].tolist())
            x0, y0 = max(0, x0), max(0, y0)
            crop = img[y0:y1, x0:x1]
            regions.append({
                "class": cls_name,
                "bbox": (x0, y0, x1, y1),
                "crop": crop,
                "confidence": float(box.conf[0]),
            })
        return regions
