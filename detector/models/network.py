#!/usr/bin/env python3
"""YOLO26x RAOS Network Architecture Inspector and Builder.

Explicitly defines and inspects the 19-class component detector architecture for RAOS CT slices.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import torch
from ultralytics import YOLO

HERE = Path(__file__).resolve().parent
DEFAULT_CFG = HERE / "yolo26x_raos.yaml"


def inspect_model(cfg_path: Path, weights_path: Path | None = None) -> None:
    print(f"Loading architecture from {cfg_path}...")
    model = YOLO(str(cfg_path))
    if weights_path and weights_path.is_file():
        print(f"Loading weights from {weights_path}...")
        model.load(str(weights_path))

    net = model.model
    params_total = sum(p.numel() for p in net.parameters())
    params_trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)

    print("\n" + "=" * 65)
    print("YOLO26x RAOS ARCHITECTURE SUMMARY")
    print("=" * 65)
    print(f"Model Class:        {type(net).__name__}")
    print(f"Number of Classes:  {net.yaml.get('nc', 19)}")
    print(f"Total Parameters:   {params_total:,} ({params_total / 1e6:.2f} M)")
    print(f"Trainable Params:   {params_trainable:,}")
    print(f"Network Nodes:      {len(net.model)} layers")
    print("=" * 65)

    print("\nLayer-by-layer breakdown:")
    for idx, layer in enumerate(net.model):
        layer_params = sum(p.numel() for p in layer.parameters())
        print(f"  [{idx:02d}] {layer.__class__.__name__:<16} params: {layer_params:>10,}")

    # Forward pass test
    dummy_input = torch.zeros(1, 3, 512, 512)
    with torch.no_grad():
        output = net(dummy_input)
    if isinstance(output, (tuple, list)):
        print(f"\nForward pass successful! Output tensors count: {len(output)}")
        if isinstance(output[0], torch.Tensor):
            print(f"Pred output shape: {output[0].shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=Path, default=DEFAULT_CFG)
    parser.add_argument("--weights", type=Path, default=None)
    args = parser.parse_args()
    inspect_model(args.cfg, args.weights)
