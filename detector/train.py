"""
detector/train.py

Fine-tunes a YOLOv8 model on the labeled region dataset, starting from a
COCO-pretrained checkpoint so a modest labeled set (150-300 pages, per
detector/README.md) is enough to get a usable first model.

Usage:
    python detector/train.py
    python detector/train.py --base_model yolov8s.pt --epochs 150
"""

import argparse
from ultralytics import YOLO


def train(data_yaml="data.yaml", base_model="yolov8n.pt", epochs=100,
          imgsz=1280, batch=8, project="runs", name="drawing_regions"):
    model = YOLO(base_model)
    model.train(data=data_yaml, epochs=epochs, imgsz=imgsz, batch=batch,
                project=project, name=name, patience=20)
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_yaml", default="data.yaml")
    parser.add_argument("--base_model", default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--batch", type=int, default=8)
    args = parser.parse_args()
    train(data_yaml=args.data_yaml, base_model=args.base_model,
          epochs=args.epochs, imgsz=args.imgsz, batch=args.batch)
