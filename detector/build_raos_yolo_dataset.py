#!/usr/bin/env python3
"""Build the 512px RAOS YOLO detection dataset from HU windowed slice data.

Images are referenced with symlinks. The source data are never modified.
Annotations are extracted from integer slice masks:
- Pixel value N in 1..19 -> YOLO class N-1 (0..18).
- Each 8-connected positive component generates one YOLO bounding box.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd
from PIL import Image

HERE = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = Path(__file__).resolve().parents[1] / "data"
DEFAULT_MANIFEST = DEFAULT_DATA_ROOT / "raos_slice_manifest.csv"
TARGET_SIZE = (512, 512)

CLASS_NAMES = [
    "Liver", "Spleen", "left kidney", "right kidney", "Stomach", "Gallbladder",
    "Esophagus", "Pancreas", "Duodenum", "Colon", "Intestine", "left adrenal",
    "right adrenal", "Rectum", "Bladder", "left head of femur",
    "right head of femur", "Prostate", "Seminal vesicles",
]

SPLIT_MAP = {
    "training": "train",
    "validation": "val",
    "test": "test",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build YOLO dataset for RAOS slice detection")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT, help="Source dataset root directory")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST, help="Path to raos_slice_manifest.csv")
    parser.add_argument("--output-dir", type=Path, default=HERE / "dataset", help="Output directory for YOLO dataset")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel workers")
    parser.add_argument("--force", action="store_true", help="Force overwrite existing dataset directory")
    return parser.parse_args()


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)


def extract_boxes_from_mask(mask_path: Path) -> Tuple[List[Tuple[int, float, float, float, float]], Dict[int, int]]:
    """Extract 8-connected components from multi-class mask."""
    mask = np.asarray(Image.open(mask_path))
    unique_vals = np.unique(mask)
    boxes = []
    class_box_counts: Dict[int, int] = {}
    
    for c in unique_vals:
        if c == 0:
            continue
        c_int = int(c)
        if c_int > len(CLASS_NAMES):
            continue
        cls_idx = c_int - 1
        binary = np.asarray(mask == c, dtype=np.uint8)
        count, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        comp_count = 0
        for comp in range(1, count):
            x, y, w, h, area = stats[comp]
            if area > 0 and w > 0 and h > 0:
                xc = (x + w / 2.0) / 512.0
                yc = (y + h / 2.0) / 512.0
                nw = w / 512.0
                nh = h / 512.0
                boxes.append((cls_idx, xc, yc, nw, nh))
                comp_count += 1
        if comp_count > 0:
            class_box_counts[cls_idx] = class_box_counts.get(cls_idx, 0) + comp_count
            
    return boxes, class_box_counts


def process_slice_batch(items: List[Tuple[str, str, str, str, str]]) -> List[Dict]:
    """Process a batch of slices: symlink image, extract mask boxes, write YOLO txt."""
    results = []
    for abs_img_str, abs_mask_str, img_link_str, label_file_str, yolo_split in items:
        abs_img = Path(abs_img_str)
        abs_mask = Path(abs_mask_str)
        img_link = Path(img_link_str)
        label_file = Path(label_file_str)
        
        # Create symlink
        if not img_link.exists() and not img_link.is_symlink():
            img_link.symlink_to(abs_img)
            
        boxes, class_box_counts = extract_boxes_from_mask(abs_mask)
        
        label_content = "".join(
            f"{cls} {xc:.8f} {yc:.8f} {w:.8f} {h:.8f}\n" for cls, xc, yc, w, h in boxes
        )
        label_file.write_text(label_content, encoding="ascii")
        
        results.append({
            "split": yolo_split,
            "has_boxes": len(boxes) > 0,
            "box_count": len(boxes),
            "class_box_counts": class_box_counts,
        })
    return results


def main() -> None:
    args = parse_args()
    
    if not args.manifest.is_file():
        raise FileNotFoundError(f"Manifest not found: {args.manifest}")
    if not args.data_root.is_dir():
        raise FileNotFoundError(f"Data root not found: {args.data_root}")
        
    print(f"Reading manifest from {args.manifest}...")
    df = pd.read_csv(args.manifest)
    total_slices = len(df)
    print(f"Found {total_slices} slices in manifest.")
    
    # Prepare directories
    for yolo_split in ["train", "val", "test"]:
        img_dir = args.output_dir / "images" / yolo_split
        lbl_dir = args.output_dir / "labels" / yolo_split
        if args.force:
            reset_dir(img_dir)
            reset_dir(lbl_dir)
        else:
            img_dir.mkdir(parents=True, exist_ok=True)
            lbl_dir.mkdir(parents=True, exist_ok=True)
            
    # Prepare tasks
    tasks_by_split: Dict[str, List] = {"train": [], "val": [], "test": []}
    for _, row in df.iterrows():
        raw_split = str(row["split"]).strip()
        yolo_split = SPLIT_MAP.get(raw_split)
        if not yolo_split:
            continue
            
        abs_img = (args.data_root / row["output_image"]).resolve()
        abs_mask = (args.data_root / row["output_mask"]).resolve()
        
        img_stem = Path(row["output_image"]).stem
        img_link = args.output_dir / "images" / yolo_split / f"{img_stem}.png"
        label_file = args.output_dir / "labels" / yolo_split / f"{img_stem}.txt"
        
        tasks_by_split[yolo_split].append((
            str(abs_img), str(abs_mask), str(img_link), str(label_file), yolo_split
        ))
        
    all_tasks = tasks_by_split["train"] + tasks_by_split["val"] + tasks_by_split["test"]
    print(f"Prepared {len(all_tasks)} items across splits:")
    for s, tasks in tasks_by_split.items():
        print(f"  - {s}: {len(tasks)} slices")
        
    # Chunk tasks for workers
    chunk_size = 100
    chunks = [all_tasks[i:i + chunk_size] for i in range(0, len(all_tasks), chunk_size)]
    print(f"Processing in {len(chunks)} chunks with {args.workers} workers...")
    
    t0 = time.time()
    processed_count = 0
    split_stats: Dict[str, Dict] = {
        s: {
            "images": 0,
            "images_with_boxes": 0,
            "empty_images": 0,
            "boxes": 0,
            "boxes_per_class": {name: 0 for name in CLASS_NAMES},
        }
        for s in ["train", "val", "test"]
    }
    
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for batch_results in executor.map(process_slice_batch, chunks):
            for res in batch_results:
                s = res["split"]
                split_stats[s]["images"] += 1
                if res["has_boxes"]:
                    split_stats[s]["images_with_boxes"] += 1
                else:
                    split_stats[s]["empty_images"] += 1
                split_stats[s]["boxes"] += res["box_count"]
                for cls_idx, count in res["class_box_counts"].items():
                    cls_name = CLASS_NAMES[cls_idx]
                    split_stats[s]["boxes_per_class"][cls_name] += count
            processed_count += len(batch_results)
            if processed_count % 1000 == 0 or processed_count == len(all_tasks):
                print(f"  Processed {processed_count}/{len(all_tasks)} slices ({processed_count / len(all_tasks) * 100:.1f}%)...")
                
    elapsed = time.time() - t0
    print(f"Dataset processing completed in {elapsed:.1f} s.")
    
    # Save dataset stats
    stats_output = {
        "source_dataset": str(args.data_root.resolve()),
        "manifest": str(args.manifest.resolve()),
        "output_dataset": str(args.output_dir.resolve()),
        "class_names": CLASS_NAMES,
        "yolo_class_id_rule": "mask pixel value N (1..19) -> YOLO class N-1 (0..18)",
        "annotation_policy": "one YOLO bbox per 8-connected positive component of each class mask",
        "image_size": list(TARGET_SIZE),
        "splits": split_stats,
    }
    stats_file = HERE / "dataset_stats.json"
    stats_file.write_text(json.dumps(stats_output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Saved dataset stats to {stats_file}")
    
    # Generate YAML configuration
    yaml_lines = [
        f"path: {args.output_dir.resolve()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        f"nc: {len(CLASS_NAMES)}",
        "names:",
    ]
    yaml_lines.extend(f"  {i}: {name}" for i, name in enumerate(CLASS_NAMES))
    yaml_file = HERE / "raos_yolo26_512.yaml"
    yaml_file.write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")
    print(f"Generated YAML config at {yaml_file}")
    
    # Summary report
    print("\n" + "=" * 60)
    print("DATASET BUILD SUMMARY")
    print("=" * 60)
    for s in ["train", "val", "test"]:
        info = split_stats[s]
        print(f"Split [{s}]: {info['images']} images ({info['images_with_boxes']} with boxes, {info['empty_images']} empty), {info['boxes']} total boxes")
    print("=" * 60)


if __name__ == "__main__":
    main()
