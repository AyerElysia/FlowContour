#!/usr/bin/env python3
"""Low-impact single-GPU training entry point for 512px RAOS YOLO detection.

Adapted for the RAOS HU window (WW 400 / WL 40) slice dataset.
Trained on component-level bounding boxes as initialization for FlowContour / active contour models.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
from pathlib import Path

import torch
import ultralytics
from ultralytics import YOLO

HERE = Path(__file__).resolve().parent
DEFAULT_CFG = HERE / "models" / "yolo26x_raos.yaml"
DEFAULT_WEIGHTS = HERE / "weights" / "best.pt"
DEFAULT_DATA = HERE / "raos_yolo26_512.yaml"
NUM_TRAIN_SAMPLES = 8683  # Total slices in RAOS training split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO detector on RAOS 512px slices")
    parser.add_argument(
        "--cfg",
        type=str,
        default=str(DEFAULT_CFG),
        help=f"Path to model architecture YAML (default: {DEFAULT_CFG})",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=str(DEFAULT_WEIGHTS),
        help=f"Path to initial pretrained weights (default: {DEFAULT_WEIGHTS})",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=str(DEFAULT_DATA),
        help=f"Path to dataset YAML config (default: {DEFAULT_DATA})",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs (default: 100, equals ~108,500 update steps with batch=8)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Optional: specify total training steps directly (e.g. 60000 or 100000). Overrides --epochs if set.",
    )
    parser.add_argument(
        "--save-period",
        type=int,
        default=10,
        help="Save checkpoint every N epochs (default: 10 epochs = 10,850 steps; best.pt and last.pt saved every epoch)",
    )
    parser.add_argument("--batch", type=int, default=8, help="Batch size (default: 8)")
    parser.add_argument("--workers", type=int, default=2, help="Dataloader worker processes (default: 2)")
    parser.add_argument("--device", default="0", help="Logical CUDA device index after CUDA_VISIBLE_DEVICES (default: 0)")
    parser.add_argument("--name", default="yolo26x_raos_512_e100", help="Experiment run directory name")
    parser.add_argument("--resume", action="store_true", help="Resume training from last.pt")
    parser.add_argument("--smoke", action="store_true", help="Run 1 epoch smoke test on 0.2 percent of train data")
    parser.add_argument("--seed", type=int, default=20261019, help="Random seed for reproducibility (default: 20261019)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    steps_per_epoch = math.ceil(NUM_TRAIN_SAMPLES / args.batch)
    if args.steps is not None and args.steps > 0:
        total_epochs = math.ceil(args.steps / steps_per_epoch)
        total_steps = total_epochs * steps_per_epoch
        print(f"[Steps -> Epochs] Target {args.steps:,} steps -> {total_epochs} epochs ({total_steps:,} total steps)")
    else:
        total_epochs = 1 if args.smoke else args.epochs
        total_steps = total_epochs * steps_per_epoch

    print("\n" + "=" * 65)
    print("TRAINING SCHEDULE & STEP MAPPING")
    print("=" * 65)
    print(f"Dataset Train Size:     {NUM_TRAIN_SAMPLES:,} slices")
    print(f"Batch Size:             {args.batch}")
    print(f"Steps per Epoch:        {steps_per_epoch:,} iterations / gradient updates")
    print(f"Total Epochs:           {total_epochs}")
    print(f"Total Update Steps:     {total_steps:,} iterations")
    print(f"Checkpoint Interval:    Every {args.save_period} epochs ({args.save_period * steps_per_epoch:,} steps)")
    print(f"Best & Last Checkpoint: Every epoch ({steps_per_epoch:,} steps)")
    print("=" * 65 + "\n")

    weights_path = Path(args.weights).resolve()
    if not args.resume and not weights_path.is_file():
        raise FileNotFoundError(f"Missing initial weights: {weights_path}")

    data_path = Path(args.data).resolve()
    if not data_path.is_file():
        raise FileNotFoundError(f"Missing dataset config: {data_path}")

    cfg_path = Path(args.cfg).resolve() if args.cfg else None

    run_root = HERE / "runs"
    run_dir = run_root / args.name
    run_dir.mkdir(parents=True, exist_ok=True)

    last_ckpt = run_dir / "weights" / "last.pt"
    if args.resume and not last_ckpt.is_file():
        raise FileNotFoundError(f"Cannot resume; checkpoint missing: {last_ckpt}")

    configuration = {
        "purpose": "RAOS 512x512 component-level detector for FlowContour / snake initialization",
        "model": "YOLO26x",
        "architecture_cfg": str(cfg_path) if cfg_path else "built-in",
        "initial_weights": str(weights_path),
        "data": str(data_path),
        "epochs": total_epochs,
        "total_steps": total_steps,
        "steps_per_epoch": steps_per_epoch,
        "save_period_epochs": args.save_period,
        "save_period_steps": args.save_period * steps_per_epoch,
        "batch": args.batch,
        "imgsz": 512,
        "workers": args.workers,
        "logical_device": args.device,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "ultralytics": ultralytics.__version__,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "python": platform.python_version(),
        "resource_policy": "one isolated physical GPU, nice +10, two dataloader workers, cache disabled",
        "augmentation": {
            "hsv_h": 0.0,
            "hsv_s": 0.0,
            "hsv_v": 0.12,
            "degrees": 3.0,
            "translate": 0.03,
            "scale": 0.15,
            "shear": 1.0,
            "perspective": 0.0,
            "fliplr": 0.0,
            "flipud": 0.0,
            "mosaic": 0.15,
            "close_mosaic": 15,
            "mixup": 0.0,
            "cutmix": 0.0,
            "copy_paste": 0.0,
        },
        "rationale": (
            "Medical CT domain adaptation: horizontal and vertical flips disabled because left/right "
            "organ anatomy carries distinct semantic labels. Color jitter disabled for grayscale CT with "
            "slight intensity jitter (hsv_v=0.12); mosaic is conservative (0.15) and retired in final 15 "
            "epochs to retain anatomical realism."
        ),
    }

    config_path = run_dir / "run_config.json"
    config_path.write_text(json.dumps(configuration, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[Config] Saved training configuration to {config_path}")

    # Build model: start from custom architecture YAML if available and load weights
    if args.resume:
        model = YOLO(str(last_ckpt))
    elif cfg_path and cfg_path.is_file():
        print(f"[Model] Initializing architecture from {cfg_path} with weights {weights_path}...")
        model = YOLO(str(cfg_path)).load(str(weights_path))
    else:
        model = YOLO(str(weights_path))

    # Launch training
    model.train(
        data=str(data_path),
        epochs=total_epochs,
        imgsz=512,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=str(run_root),
        name=args.name,
        exist_ok=True,
        resume=args.resume,
        pretrained=False if args.resume else str(weights_path),
        optimizer="auto",
        patience=30,
        save=True,
        save_period=args.save_period,
        cache=False,
        amp=True,
        cos_lr=True,
        warmup_epochs=3.0,
        close_mosaic=0 if args.smoke else 15,
        mosaic=0.0 if args.smoke else 0.15,
        mixup=0.0,
        degrees=3.0,
        translate=0.03,
        scale=0.15,
        shear=1.0,
        perspective=0.0,
        fliplr=0.0,
        flipud=0.0,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.12,
        seed=args.seed,
        deterministic=False,
        plots=True,
        verbose=True,
        fraction=0.002 if args.smoke else 1.0,
    )

    # Post-training evaluation on test split
    if not args.smoke:
        best_weights = run_dir / "weights" / "best.pt"
        if best_weights.exists():
            print(f"\n[Evaluation] Running validation on test split: {best_weights}")
            val_model = YOLO(str(best_weights))
            metrics = val_model.val(
                data=str(data_path),
                split="test",
                imgsz=512,
                batch=args.batch,
                device=args.device,
                workers=args.workers,
                plots=True,
            )
            metrics_path = run_dir / "test_metrics.json"
            metrics_path.write_text(json.dumps(metrics.results_dict, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            print(f"[Evaluation] Test metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
