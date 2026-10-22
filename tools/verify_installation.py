#!/usr/bin/env python3
"""Read-only CPU source, real-data and fixed-protocol verification."""

import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/pretrain.yaml"))
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--slice-manifest", required=True)
    parser.add_argument("--moonvit-cache", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--expected-step", type=int, required=True)
    parser.add_argument("--initialize-dense", action="store_true")
    args = parser.parse_args()
    os.chdir(ROOT)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["FLOWCONTOUR_DATA_ROOT"] = str(Path(args.data_root).resolve())
    os.environ["FLOWCONTOUR_SLICE_MANIFEST"] = str(Path(args.slice_manifest).resolve())
    sys.argv = [sys.argv[0], "--cfg_file", args.config]
    import torch
    from lib.config import cfg
    from lib.checkpoints import sha256_file
    from lib.networks import make_network
    from lib.datasets.make_dataset import make_dataset
    from lib.train.engine import load_weights, DENSE_SHA
    from lib.train.wrapper import FlowContourNetworkWrapper
    from lib.protocol import protocol_metadata
    from tools.infer import foreground_smoke_indices
    if sha256_file(args.checkpoint) != args.checkpoint_sha256:
        raise RuntimeError("source SHA mismatch")
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if payload.get("step") != args.expected_step:
        raise RuntimeError("source step mismatch")
    if args.initialize_dense and (args.checkpoint_sha256 != DENSE_SHA or args.expected_step != 19000):
        raise RuntimeError("not the signed Dense source")
    if not args.initialize_dense and payload.get("protocol") != protocol_metadata():
        raise RuntimeError("protocol mismatch")
    cfg.locate_feat_cache_root = args.moonvit_cache
    cfg.sagittal_moonvit_cache_root = args.moonvit_cache
    wrapper = FlowContourNetworkWrapper(make_network(cfg))
    audit = load_weights(wrapper, payload, args.initialize_dense)
    dataset = make_dataset(cfg, "VolMemTrain")
    index = foreground_smoke_indices(dataset.records, 1)[0]
    sample = dataset[index]  # exercises actual image/mask/cache metadata checks
    if "locate_feat" not in sample:
        raise RuntimeError("real MoonViT feature missing")
    print({"status": "PASS", "protocol": protocol_metadata(),
           "parameters": sum(p.numel() for p in wrapper.parameters()),
           "train_rows": len(dataset), "real_foreground_index": index,
           "strict_source_audit": audit})


if __name__ == "__main__":
    main()
