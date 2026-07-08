"""
detector/evaluate.py

Usage:
    python detector/evaluate.py --weights runs/drawing_regions/weights/best.pt
"""

import argparse
from ultralytics import YOLO


def evaluate(weights, data_yaml="data.yaml", imgsz=1280):
    model = YOLO(weights)
    metrics = model.val(data=data_yaml, imgsz=imgsz)

    print("\nPer-class mAP50-95:")
    for i, name in model.names.items():
        try:
            print(f"  {name:25s} {metrics.box.maps[i]:.3f}")
        except (IndexError, AttributeError):
            print(f"  {name:25s} (no val instances)")

    print(f"\nOverall mAP50:    {metrics.box.map50:.3f}")
    print(f"Overall mAP50-95: {metrics.box.map:.3f}")
    print(f"\nPrediction overlays saved under: {metrics.save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data_yaml", default="data.yaml")
    parser.add_argument("--imgsz", type=int, default=1280)
    args = parser.parse_args()
    evaluate(args.weights, args.data_yaml, args.imgsz)
