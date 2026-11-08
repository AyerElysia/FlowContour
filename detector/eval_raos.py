#!/usr/bin/env python3
"""Evaluate a trained YOLO checkpoint on RAOS splits (val / test).

Computes comprehensive metrics: Precision, Recall, mAP@0.5, mAP@0.5:0.95 across all 19 anatomical classes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import numpy as np
from ultralytics import YOLO

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "flowcontour"))
from lib.packaged_weights import detector_weight_path
DEFAULT_DATA = HERE / "raos_yolo26_512.yaml"
DEFAULT_WEIGHTS = HERE.parent / "flowcontour" / "weights" / "best.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate YOLO detector on RAOS")
    parser.add_argument("--weights", type=str, default=str(DEFAULT_WEIGHTS), help=f"Path to weights checkpoint (default: {DEFAULT_WEIGHTS})")
    parser.add_argument("--data", type=str, default=str(DEFAULT_DATA), help=f"Path to dataset YAML config (default: {DEFAULT_DATA})")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"], help="Dataset split to evaluate (default: test)")
    parser.add_argument("--batch", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--device", default="0", help="Logical CUDA device index after CUDA_VISIBLE_DEVICES")
    parser.add_argument("--workers", type=int, default=4, help="Dataloader workers (default: 4)")
    parser.add_argument("--output", type=str, default=str(HERE / "test_evaluation_report.json"), help="Path to save evaluation metrics JSON")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    weights_path = Path(args.weights).resolve()
    data_path = Path(args.data).resolve()

    if not weights_path.is_file():
        raise FileNotFoundError(f"Weights file not found: {weights_path}")
    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset config not found: {data_path}")

    print("=" * 70)
    print("FULL DATASET EVALUATION")
    print(f"Model Checkpoint: {weights_path}")
    print(f"Dataset Config:   {data_path}")
    print(f"Target Split:     {args.split}")
    print(f"Batch Size:       {args.batch}")
    print(f"Device:           CUDA:{args.device}")
    print("=" * 70)

    with detector_weight_path(weights_path) as detector_path:
        model = YOLO(str(detector_path))
    metrics = model.val(
        data=str(data_path),
        split=args.split,
        imgsz=512,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        plots=True,
        save_json=True,
    )

    box = metrics.box
    names = metrics.names

    overall_p = float(box.mp)
    overall_r = float(box.mr)
    overall_map50 = float(box.map50)
    overall_map = float(box.map)

    # Per-class metrics
    class_results = []
    ap50_arr = box.ap50
    map_arr = box.maps
    p_arr = box.p
    r_arr = box.r

    for cls_idx in range(len(names)):
        cls_name = names.get(cls_idx, f"Class {cls_idx}")
        p_val = float(p_arr[cls_idx]) if len(p_arr) > cls_idx else 0.0
        r_val = float(r_arr[cls_idx]) if len(r_arr) > cls_idx else 0.0
        ap50_val = float(ap50_arr[cls_idx]) if len(ap50_arr) > cls_idx else 0.0
        map_val = float(map_arr[cls_idx]) if len(map_arr) > cls_idx else 0.0

        class_results.append({
            "class_id": cls_idx,
            "class_name": cls_name,
            "precision": round(p_val, 4),
            "recall": round(r_val, 4),
            "mAP50": round(ap50_val, 4),
            "mAP50_95": round(map_val, 4),
        })

    print("\n" + "=" * 70)
    print("EVALUATION METRICS SUMMARY")
    print("=" * 70)
    print(f"Overall Precision (B):    {overall_p:.4f} ({overall_p * 100:.2f}%)")
    print(f"Overall Recall (B):       {overall_r:.4f} ({overall_r * 100:.2f}%)")
    print(f"Overall mAP@0.5 (B):      {overall_map50:.4f} ({overall_map50 * 100:.2f}%)")
    print(f"Overall mAP@0.5:0.95 (B): {overall_map:.4f} ({overall_map * 100:.2f}%)")
    print("=" * 70)

    print("\n" + "-" * 70)
    print(f"{'Class ID':<10}{'Class Name':<24}{'P':<10}{'R':<10}{'mAP@0.5':<12}{'mAP@0.5:0.95':<12}")
    print("-" * 70)
    for c in class_results:
        print(f"{c['class_id']:<10}{c['class_name']:<24}{c['precision']:<10.4f}{c['recall']:<10.4f}{c['mAP50']:<12.4f}{c['mAP50_95']:<12.4f}")
    print("-" * 70)

    report = {
        "model": str(weights_path),
        "dataset_yaml": str(data_path),
        "split": args.split,
        "summary": {
            "precision": round(overall_p, 4),
            "recall": round(overall_r, 4),
            "mAP50": round(overall_map50, 4),
            "mAP50_95": round(overall_map, 4),
        },
        "raw_results_dict": {k: float(v) if isinstance(v, (int, float, np.floating)) else str(v) for k, v in metrics.results_dict.items()},
        "per_class_metrics": class_results,
    }

    if args.output:
        out_path = Path(args.output).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"\n[Saved] Detailed report saved to: {out_path}")


if __name__ == "__main__":
    main()
