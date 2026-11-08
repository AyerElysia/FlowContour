#!/usr/bin/env python3
"""Manually start GDPO preflight and training from a completed pretrain checkpoint."""
import argparse
import hashlib
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True,
                        help="RAOS dataset with training and validation splits")
    parser.add_argument("--slice-manifest", type=Path, required=True,
                        help="Full RAOS train/validation manifest")
    parser.add_argument("--output", type=Path, required=True,
                        help="New directory for formal GDPO output")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    import torch
    root = Path(__file__).resolve().parents[1]
    checkpoint = args.checkpoint.resolve()
    data_root = args.data_root.resolve()
    manifest = args.slice_manifest.resolve()
    output = args.output.resolve()
    if not data_root.is_dir() or not manifest.is_file():
        raise FileNotFoundError("GDPO requires the external RAOS train/validation dataset and manifest")
    if output.exists():
        raise FileExistsError(output)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("training_stage") != "pretrain" or payload.get("run_mode") != "train":
        raise ValueError("a completed formal pretraining checkpoint is required")
    step = int(payload["step"])
    del payload
    digest = hashlib.file_digest(checkpoint.open("rb"), "sha256").hexdigest()
    preflight = output.with_name(output.name + "_preflight")
    if preflight.exists():
        raise FileExistsError(preflight)
    common = [sys.executable, str(root/"tools/train.py"), "--stage", "gdpo",
              "--source-checkpoint", str(checkpoint), "--source-sha256", digest,
              "--source-step", str(step), "--gpu", str(args.gpu), "--allow-shared-gpu",
              "--data-root", str(data_root),
              "--slice-manifest", str(manifest)]
    subprocess.run(common+["--mode", "preflight", "--output", str(preflight)], cwd=root, check=True)
    subprocess.run(common+["--mode", "train", "--preflight-output", str(preflight),
                           "--output", str(output)], cwd=root, check=True)


if __name__ == "__main__":
    main()
