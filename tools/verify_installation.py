#!/usr/bin/env python3
"""CPU verification for the published MoonViT-cache + Flow mainline."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path


LOCKED_CASES = {"sub-verse010", "sub-verse011", "sub-verse013"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/stage2_rl.yaml")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--slice-manifest", required=True)
    parser.add_argument("--moonvit-cache", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-sha256", default="")
    parser.add_argument(
        "--expected-step",
        type=int,
        required=True,
        help="exact step stored in the selected CR-MoE checkpoint",
    )
    parser.add_argument("--max-scan", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    os.environ["DIFFUSIONSNAKE_DATA_ROOT"] = str(Path(args.data_root).resolve())
    os.environ["DIFFUSIONSNAKE_SLICE_MANIFEST"] = str(
        Path(args.slice_manifest).resolve()
    )

    # lib.config owns the shared command-line parser. Give it only the
    # selected production config, after this verifier has parsed its arguments.
    sys.argv = [sys.argv[0], "--cfg_file", str(Path(args.config))]

    import numpy as np
    import torch

    from lib.checkpoints import extract_state_dict, normalize_state_dict, sha256_file
    from lib.config import cfg
    from lib.datasets.sagittal_2d_fixed.snake import Dataset
    from lib.networks import make_network
    from lib.rl.fourier import (
        fourier_action_logprob,
        fourier_mean_kl,
        low_frequency_delta,
        stage_progress,
        standard_normal_logprob,
    )
    from lib.train.rewards.region_reward import (
        compute_delta_nsd_reward,
        compute_nsd_score,
    )
    from lib.train.trainers.diffusion_trainer import DiffusionPretrainNetworkWrapper

    cache_root = str(Path(args.moonvit_cache).resolve())
    cfg.locate_feat_cache_root = cache_root
    cfg.sagittal_moonvit_cache_root = cache_root
    if bool(getattr(cfg, "rl_use_delta_nsd_reward", False)) is not True:
        raise RuntimeError("stage-2 reward must be delta-NSD")
    nsd_delta_px = float(getattr(cfg, "rl_nsd_delta_px", -1.0))
    if abs(nsd_delta_px - 2.0) > 1e-8:
        raise RuntimeError(f"stage-2 NSD tolerance drift: {nsd_delta_px} != 2.0")
    random.seed(20260823)
    np.random.seed(20260823)
    torch.manual_seed(20260823)

    dataset = Dataset(
        ann_file=os.environ["DIFFUSIONSNAKE_SLICE_MANIFEST"],
        data_root=os.environ["DIFFUSIONSNAKE_DATA_ROOT"],
        split="train",
    )
    if len(dataset) != int(cfg.sagittal_expected_train_row_count):
        raise RuntimeError(
            f"Train72 row count mismatch: {len(dataset)} != "
            f"{int(cfg.sagittal_expected_train_row_count)}"
        )
    selected_cases = set(dataset.case_partition_audit["selected_case_ids"])
    if selected_cases.intersection(LOCKED_CASES):
        raise RuntimeError("locked cases leaked into Train72")

    sample = None
    sample_index = -1
    for index in range(min(len(dataset), max(1, int(args.max_scan)))):
        candidate = dataset[index]
        if int(candidate["meta"]["ct_num"]) > 0:
            sample = candidate
            sample_index = index
            break
    if sample is None:
        raise RuntimeError("no foreground training sample found in verification scan")
    feature = np.asarray(sample["locate_feat"])
    if feature.ndim != 3 or feature.shape[0] != int(cfg.locate_feat_dim):
        raise RuntimeError(f"unexpected MoonViT feature shape: {feature.shape}")
    if not np.isfinite(feature).all():
        raise FloatingPointError("MoonViT cache sample contains non-finite values")
    if not sample["i_it_py"] or not sample["i_gt_py"]:
        raise RuntimeError("foreground sample did not produce contour targets")

    network = make_network(cfg)
    wrapper = DiffusionPretrainNetworkWrapper(network)
    parameter_count = sum(parameter.numel() for parameter in wrapper.parameters())
    flow_parameter_count = sum(parameter.numel() for parameter in network.gcn.parameters())
    replacer_parameter_count = sum(
        parameter.numel() for parameter in network.locate_feat_replacer.parameters()
    )
    expected_parameters = int(cfg.pure2d_expected_parameter_count)
    if parameter_count != expected_parameters:
        raise RuntimeError(
            f"parameter gate failed: {parameter_count} != {expected_parameters}"
        )

    checkpoint_path = Path(args.checkpoint).resolve()
    checkpoint_sha = sha256_file(checkpoint_path)
    if args.checkpoint_sha256 and checkpoint_sha != args.checkpoint_sha256.lower():
        raise RuntimeError(
            f"checkpoint SHA256 mismatch: {checkpoint_sha} != "
            f"{args.checkpoint_sha256.lower()}"
        )
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = normalize_state_dict(extract_state_dict(payload))
    target = wrapper.state_dict()
    missing = sorted(set(target).difference(state))
    unexpected = sorted(set(state).difference(target))
    shape_mismatch = sorted(
        key
        for key in set(target).intersection(state)
        if tuple(target[key].shape) != tuple(state[key].shape)
    )
    if missing or unexpected or shape_mismatch:
        raise RuntimeError(
            "strict checkpoint mismatch: missing={} unexpected={} shape={}".format(
                missing[:8], unexpected[:8], shape_mismatch[:8]
            )
        )
    wrapper.load_state_dict(state, strict=True)
    checkpoint_step = int(payload.get("step", -1))
    if checkpoint_step != int(args.expected_step):
        raise RuntimeError(
            f"checkpoint step mismatch: {checkpoint_step} != {args.expected_step}"
        )
    if any(not torch.isfinite(value).all() for value in state.values()):
        raise FloatingPointError("checkpoint contains non-finite tensors")

    # The sampled coefficients must be exactly recoverable by the policy
    # log-probability projection, and KL must propagate to the Flow mean.
    angles = torch.arange(128, dtype=torch.float64) * (2.0 * torch.pi / 128.0)
    contour = torch.stack((angles.cos(), angles.sin()), dim=-1).unsqueeze(0)
    coefficients = torch.randn(1, 8, dtype=torch.float64)
    mean = torch.zeros_like(contour)
    action = mean + low_frequency_delta(contour, coefficients, sigma=0.8)
    projected_logprob = fourier_action_logprob(
        action, mean, contour, sigma=0.8, n_modes=8
    )
    direct_logprob = standard_normal_logprob(coefficients)
    if not torch.allclose(projected_logprob, direct_logprob, atol=1e-10, rtol=0.0):
        raise RuntimeError("Fourier action projection is not self-consistent")
    current_mean = torch.zeros_like(contour, requires_grad=True)
    reference_mean = torch.full_like(contour, 0.01)
    kl = fourier_mean_kl(
        current_mean, reference_mean, contour, sigma=0.8, n_modes=8
    ).mean()
    kl.backward()
    if current_mean.grad is None or not torch.isfinite(current_mean.grad).all():
        raise RuntimeError("Fourier KL gradient is missing or non-finite")
    progress = stage_progress([0.2, 0.25, 0.3333, 0.5, 1.0])
    expected_progress = [0.0, 0.2, 0.4, 0.59998, 0.79999]
    if any(abs(a - b) > 1e-8 for a, b in zip(progress, expected_progress)):
        raise RuntimeError(f"five-stage progress mismatch: {progress}")

    # Reward smoke test: the production contract is symmetric 2D NSD@2px.
    # Keep this beside the Fourier checks so a clean installation cannot pass
    # while silently falling back to the former composite region reward.
    target_poly = torch.tensor(
        [[[16.0, 16.0], [40.0, 16.0], [40.0, 40.0], [16.0, 40.0]]]
    )
    shifted_poly = target_poly + torch.tensor([[[1.0, 0.0]]])
    distant_poly = target_poly + torch.tensor([[[32.0, 32.0]]])

    def nsd(first: torch.Tensor, second: torch.Tensor) -> float:
        return float(
            compute_nsd_score(
                first,
                second,
                H=96,
                W=96,
                delta_px=nsd_delta_px,
            ).item()
        )

    nsd_identical = nsd(target_poly, target_poly)
    nsd_shifted = nsd(shifted_poly, target_poly)
    nsd_shifted_reverse = nsd(target_poly, shifted_poly)
    nsd_distant = nsd(distant_poly, target_poly)
    if abs(nsd_identical - 1.0) > 1e-7:
        raise RuntimeError(f"identical NSD smoke test failed: {nsd_identical}")
    if abs(nsd_shifted - nsd_shifted_reverse) > 1e-7:
        raise RuntimeError("NSD symmetry smoke test failed")
    if not nsd_shifted > nsd_distant:
        raise RuntimeError("NSD ordering smoke test failed")
    reward_probe = compute_delta_nsd_reward(
        torch.tensor([0.80]),
        torch.tensor([0.75]),
        torch.tensor([0.50]),
        burr_weight=0.06,
    ).item()
    if abs(reward_probe - 0.02) > 1e-7:
        raise RuntimeError(f"delta-NSD reward smoke test failed: {reward_probe}")

    report = {
        "status": "PASS",
        "config": str(Path(args.config)),
        "train_rows": len(dataset),
        "train_cases": len(selected_cases),
        "locked_cases_selected": [],
        "sample_index": sample_index,
        "sample_case": sample["meta"]["case_id"],
        "sample_slice": int(sample["meta"]["slice_idx"]),
        "sample_contours": int(sample["meta"]["ct_num"]),
        "moonvit_feature_shape": list(feature.shape),
        "checkpoint_step": checkpoint_step,
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_tensors": len(state),
        "parameters": parameter_count,
        "flow_parameters": flow_parameter_count,
        "feature_replacer_parameters": replacer_parameter_count,
        "five_stage_progress": progress,
        "reward_contract": {
            "mode": "delta_nsd",
            "nsd_delta_px": nsd_delta_px,
            "coordinate_space": "2d_image_pixels",
            "identical_score": nsd_identical,
            "shifted_score": nsd_shifted,
            "distant_score": nsd_distant,
            "reward_probe": reward_probe,
        },
        "fourier_projection_max_abs": float(
            (projected_logprob - direct_logprob).abs().max().item()
        ),
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
