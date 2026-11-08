#!/usr/bin/env python3
"""Run YOLO26x detection on all RAOS test slices in safe batches and save detections."""

import argparse
import csv
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ultralytics import YOLO
from lib.packaged_weights import detector_weight_path

DEFAULT_YOLO_WEIGHTS = ROOT / "weights/best.pt"
DEFAULT_MANIFEST = ROOT / "configs/raos_manifest.csv"
DEFAULT_OUT = ROOT / "outputs/yolo26_detections_test.json"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, default=DEFAULT_YOLO_WEIGHTS)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", default="4")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--batch", type=int, default=32)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with open(args.manifest, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        test_rows = [r for r in reader if r["split"] == "test"]

    print(f"Loaded {len(test_rows)} test slices from {args.manifest}")
    image_paths = [r["image_path"] for r in test_rows]

    print(f"Loading YOLO model from {args.weights} on device {args.device}...")
    with detector_weight_path(args.weights) as detector_path:
        model = YOLO(str(detector_path))

    t0 = time.time()
    detections_by_key = {}
    total_boxes = 0

    print(f"Running predictions in manual chunks of size {args.batch}...")
    n_total = len(image_paths)
    for start_idx in range(0, n_total, args.batch):
        end_idx = min(start_idx + args.batch, n_total)
        chunk = image_paths[start_idx:end_idx]

        results = model.predict(
            source=chunk,
            imgsz=512,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            verbose=False,
        )

        for i, r in enumerate(results):
            img_p = chunk[i]
            name = Path(img_p).name
            boxes = r.boxes
            xyxy = boxes.xyxy.cpu().numpy().tolist() if len(boxes) else []
            conf = boxes.conf.cpu().numpy().tolist() if len(boxes) else []
            cls_ids = boxes.cls.cpu().numpy().tolist() if len(boxes) else []

            dets = []
            for b, c, cl in zip(xyxy, conf, cls_ids):
                dets.append({
                    "box": [float(x) for x in b],
                    "conf": float(c),
                    "cls": int(cl),
                })
                total_boxes += 1

            detections_by_key[img_p] = dets
            detections_by_key[name] = dets

        elapsed = time.time() - t0
        speed = end_idx / max(elapsed, 1e-5)
        print(f"[{end_idx}/{n_total}] ({speed:.1f} img/s) - Total boxes: {total_boxes}")

    elapsed = time.time() - t0
    print(f"Finished detection in {elapsed:.1f}s. Total detected boxes: {total_boxes}")

    summary = {
        "weights": str(args.weights),
        "manifest": str(args.manifest),
        "total_images": len(image_paths),
        "total_boxes": total_boxes,
        "conf": args.conf,
        "iou": args.iou,
        "elapsed_seconds": elapsed,
        "detections": detections_by_key,
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(summary, f)

    print(f"Saved detection results to {args.output}")


if __name__ == "__main__":
    main()
