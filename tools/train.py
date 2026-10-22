#!/usr/bin/env python3
"""One audited CLI for all three FlowContour training stages."""

import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("pretrain", "online", "gdpo"), required=True)
    parser.add_argument("--mode", choices=("preflight", "train"), required=True)
    parser.add_argument("--config")
    parser.add_argument("--source-checkpoint", required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--source-step", type=int, required=True)
    parser.add_argument("--initialize-dense", action="store_true")
    parser.add_argument("--resume-checkpoint")
    parser.add_argument("--preflight-output")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--slice-manifest", required=True)
    parser.add_argument("--moonvit-cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--allow-shared-gpu", action="store_true",
                        help="explicit shared-GPU functional preflight; NOT formal approval")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--seed", type=int, default=20260731)
    args = parser.parse_args(argv)
    args.root = str(ROOT)
    if args.mode == "train" and (not args.preflight_output or args.allow_shared_gpu or args.device != "cuda"):
        parser.error("formal training requires an idle-CUDA preflight and --preflight-output")
    if args.allow_shared_gpu and args.mode != "preflight":
        parser.error("shared resource exception is restricted to functional preflight")
    if Path(args.output).exists():
        raise FileExistsError("output must be a new directory")
    os.chdir(ROOT)
    config = Path(args.config or ROOT / f"configs/{args.stage}.yaml").resolve()
    os.environ["CFG_FILE"] = str(config)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu) if args.device == "cuda" else ""
    os.environ["FLOWCONTOUR_DATA_ROOT"] = str(Path(args.data_root).resolve())
    os.environ["FLOWCONTOUR_SLICE_MANIFEST"] = str(Path(args.slice_manifest).resolve())
    sys.argv = [sys.argv[0], "--cfg_file", str(config)]
    from lib.config import cfg
    from lib.runtime.gpu import acquire_gpu
    from lib.train.engine import run
    if cfg.training_stage != args.stage:
        raise ValueError("CLI stage does not match the configuration")
    cfg.locate_feat_cache_root = str(Path(args.moonvit_cache).resolve())
    cfg.sagittal_moonvit_cache_root = cfg.locate_feat_cache_root
    checks = acquire_gpu(args.gpu, args.allow_shared_gpu) if args.device == "cuda" else []
    try:
        result = run(args, checks)
    except Exception as error:
        import json
        import traceback
        from lib.train.engine import dump_json
        path = Path(args.output) / "manifest.json"
        previous = json.loads(path.read_text()) if path.exists() else {}
        dump_json(path, {**previous, "status": "FAILED", "error": repr(error),
                        "traceback": traceback.format_exc()})
        raise
    print({"status": result["status"], "stage": args.stage,
           "updates": result["effective_updates"], "output": args.output})


if __name__ == "__main__":
    main()
