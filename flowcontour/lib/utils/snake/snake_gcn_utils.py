"""Route-B contour geometry and feature sampling used by the Flow mainline."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from lib.config import cfg
from lib.utils.snake import snake_config, snake_decode


def collect_training(poly: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Flatten valid contours while retaining batch-major order."""
    return torch.cat(
        [poly[index][valid[index]] for index in range(valid.shape[0])], dim=0
    )


def img_poly_to_can_poly(image_poly: torch.Tensor) -> torch.Tensor:
    """Translate each contour so its top-left bound is the origin."""
    if image_poly.numel() == 0:
        return torch.zeros_like(image_poly)
    minimum = image_poly.amin(dim=-2, keepdim=True)
    return image_poly - minimum


def uniform_upsample(poly: torch.Tensor, point_count: int) -> torch.Tensor:
    """Arc-length-resample closed polygons to an exact fixed point count."""
    if poly.ndim != 4 or poly.shape[-1] != 2:
        raise ValueError(f"poly must have shape [B,N,V,2], got {tuple(poly.shape)}")
    point_count = int(point_count)
    vertex_count = int(poly.shape[2])
    if point_count < vertex_count:
        raise ValueError(
            f"point_count ({point_count}) must be >= vertices ({vertex_count})"
        )
    if poly.shape[1] == 0:
        return poly.new_zeros((poly.shape[0], 0, point_count, 2))

    batches: list[torch.Tensor] = []
    for batch_polys in poly:
        contours: list[torch.Tensor] = []
        for points in batch_polys:
            next_points = torch.roll(points, -1, dims=0)
            edge_length = (next_points - points).square().sum(dim=1).sqrt()
            total = edge_length.sum().clamp_min(1e-6)
            per_edge = torch.clamp(
                torch.round(edge_length / total * float(point_count)), min=1
            ).to(dtype=torch.int64)

            residual = point_count - int(per_edge.sum().item())
            order = torch.argsort(edge_length, descending=True)
            cursor = 0
            while residual:
                edge_index = int(order[cursor % vertex_count].item())
                if residual > 0:
                    per_edge[edge_index] += 1
                    residual -= 1
                elif per_edge[edge_index] > 1:
                    per_edge[edge_index] -= 1
                    residual += 1
                cursor += 1

            samples: list[torch.Tensor] = []
            for edge_index in range(vertex_count):
                count = int(per_edge[edge_index].item())
                fraction = torch.linspace(
                    0.0,
                    1.0,
                    steps=count + 1,
                    device=poly.device,
                    dtype=poly.dtype,
                )[:-1]
                start = points[edge_index]
                end = next_points[edge_index]
                samples.append(
                    start.unsqueeze(0) * (1.0 - fraction[:, None])
                    + end.unsqueeze(0) * fraction[:, None]
                )
            contour = torch.cat(samples, dim=0)
            if contour.shape[0] != point_count:
                raise RuntimeError(
                    f"resampling produced {contour.shape[0]} points, expected {point_count}"
                )
            contours.append(contour)
        batches.append(torch.stack(contours, dim=0))
    return torch.stack(batches, dim=0)


def _box_to_octagon_init(box: torch.Tensor, point_count: int) -> torch.Tensor:
    """Convert ``[N,4]`` or ``[B,N,4]`` xyxy boxes to fixed-point octagons."""
    if box.numel() == 0:
        if box.ndim == 3:
            return box.new_zeros((box.shape[0], 0, int(point_count), 2))
        return box.new_zeros((0, int(point_count), 2))
    if box.ndim == 2:
        batched = box.unsqueeze(0)
        squeeze = True
    elif box.ndim == 3:
        batched = box
        squeeze = False
    else:
        raise ValueError(f"box must be [N,4] or [B,N,4], got {tuple(box.shape)}")
    polygon = uniform_upsample(snake_decode.get_init(batched), int(point_count))
    return polygon[0] if squeeze else polygon


def resolve_routeb_box_jitter_config(config) -> dict:
    """Validate and normalize the four-level Route-B box augmentation."""
    probabilities = [
        float(value)
        for value in getattr(
            config, "routeb_box_jitter_probabilities", [1.0, 0.0, 0.0, 0.0]
        )
    ]
    shift = [
        float(value)
        for value in getattr(
            config, "routeb_box_jitter_shift_fractions", [0.0, 0.05, 0.10, 0.15]
        )
    ]
    log_scale = [
        float(value)
        for value in getattr(
            config,
            "routeb_box_jitter_log_scale_fractions",
            [0.0, 0.10, 0.20, 0.30],
        )
    ]
    edge = [
        float(value)
        for value in getattr(
            config, "routeb_box_jitter_edge_fractions", [0.0, 0.03, 0.08, 0.15]
        )
    ]
    if len({len(probabilities), len(shift), len(log_scale), len(edge)}) != 1:
        raise ValueError("Route-B jitter lists must have equal length")
    if not probabilities:
        raise ValueError("Route-B jitter needs at least one severity")
    values = probabilities + shift + log_scale + edge
    if any(not math.isfinite(value) or value < 0.0 for value in values):
        raise ValueError("Route-B jitter values must be finite and non-negative")
    probability_sum = sum(probabilities)
    if probability_sum <= 0.0:
        raise ValueError("Route-B jitter probabilities must have positive sum")
    if any(abs(values[0]) > 1e-12 for values in (shift, log_scale, edge)):
        raise ValueError("Route-B severity zero must be the exact clean-box route")
    min_iou = float(getattr(config, "routeb_box_jitter_min_iou", 0.20))
    if not math.isfinite(min_iou) or not 0.0 <= min_iou <= 1.0:
        raise ValueError("routeb_box_jitter_min_iou must be in [0,1]")
    return {
        "enabled": bool(getattr(config, "routeb_box_jitter_enabled", False)),
        "probabilities": [value / probability_sum for value in probabilities],
        "shift_fractions": shift,
        "log_scale_fractions": log_scale,
        "edge_fractions": edge,
        "min_iou": min_iou,
    }


def _aligned_box_iou(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    inter_min = torch.maximum(first[:, :2], second[:, :2])
    inter_max = torch.minimum(first[:, 2:], second[:, 2:])
    inter_wh = (inter_max - inter_min).clamp_min(0.0)
    intersection = inter_wh[:, 0] * inter_wh[:, 1]
    first_wh = (first[:, 2:] - first[:, :2]).clamp_min(0.0)
    second_wh = (second[:, 2:] - second[:, :2]).clamp_min(0.0)
    union = (
        first_wh[:, 0] * first_wh[:, 1]
        + second_wh[:, 0] * second_wh[:, 1]
        - intersection
    )
    return intersection / union.clamp_min(1e-6)


def jitter_routeb_boxes_xyxy(
    box: torch.Tensor, jitter: dict, image_hw=None
) -> tuple[torch.Tensor, dict]:
    """Apply bounded Route-B box jitter while enforcing the configured IoU floor."""
    if box.ndim != 2 or box.shape[-1] != 4:
        raise ValueError(f"box must have shape [N,4], got {tuple(box.shape)}")
    if not torch.is_floating_point(box):
        raise TypeError("Route-B box jitter requires floating-point boxes")
    instance_count = int(box.shape[0])
    severity_count = len(jitter["probabilities"])
    if not instance_count:
        zero = box.new_tensor(0.0)
        return box.clone(), {
            "routeb_box_jitter_count": zero,
            "routeb_box_jitter_clean_count": zero,
            "routeb_box_jitter_mean_iou": zero,
            "routeb_box_jitter_min_iou": zero,
            "routeb_box_jitter_severity_counts": box.new_zeros(severity_count),
        }

    clean_only = abs(float(jitter["probabilities"][0]) - 1.0) <= 1e-12 and all(
        abs(float(value)) <= 1e-12 for value in jitter["probabilities"][1:]
    )
    if not jitter.get("enabled", False) or clean_only:
        counts = box.new_zeros(severity_count)
        counts[0] = float(instance_count)
        return box.clone(), {
            "routeb_box_jitter_count": box.new_tensor(0.0),
            "routeb_box_jitter_clean_count": box.new_tensor(float(instance_count)),
            "routeb_box_jitter_mean_iou": box.new_tensor(1.0),
            "routeb_box_jitter_min_iou": box.new_tensor(1.0),
            "routeb_box_jitter_severity_counts": counts,
        }

    probabilities = torch.as_tensor(
        jitter["probabilities"], device=box.device, dtype=torch.float32
    )
    severity = torch.multinomial(probabilities, instance_count, replacement=True)
    amplitude = {
        name: torch.as_tensor(jitter[name], device=box.device, dtype=box.dtype)[severity]
        for name in ("shift_fractions", "log_scale_fractions", "edge_fractions")
    }
    gt_min, gt_max = box[:, :2], box[:, 2:]
    gt_wh = (gt_max - gt_min).clamp_min(1e-3)
    gt_center = (gt_min + gt_max) * 0.5
    center = gt_center + (torch.rand_like(gt_center) * 2.0 - 1.0) * amplitude[
        "shift_fractions"
    ][:, None] * gt_wh
    size = gt_wh * torch.exp(
        (torch.rand_like(gt_wh) * 2.0 - 1.0)
        * amplitude["log_scale_fractions"][:, None]
    )
    candidate = torch.cat((center - size * 0.5, center + size * 0.5), dim=1)
    candidate += (
        (torch.rand_like(candidate) * 2.0 - 1.0)
        * amplitude["edge_fractions"][:, None]
        * torch.cat((gt_wh, gt_wh), dim=1)
    )

    active = severity > 0
    candidate = torch.where(active[:, None], candidate, box)
    if image_hw is not None:
        image_h, image_w = map(int, image_hw)
        if image_h < 2 or image_w < 2:
            raise ValueError(f"image_hw must be at least 2x2, got {image_hw}")
        upper = box.new_tensor([image_w - 1.0, image_h - 1.0])
        low = candidate[:, :2].clamp_min(0.0)
        low = torch.minimum(low, upper)
        high = candidate[:, 2:].clamp_min(0.0)
        high = torch.minimum(high, upper)
        low, high = torch.minimum(low, high), torch.maximum(low, high)
        high = torch.minimum(torch.maximum(high, low + 1e-3), upper)
        low = torch.minimum(low, high - 1e-3).clamp_min(0.0)
        candidate = torch.cat((low, high), dim=1)
    candidate = torch.where(
        torch.isfinite(candidate).all(dim=1, keepdim=True), candidate, box
    )

    min_iou = float(jitter.get("min_iou", 0.0))
    for _ in range(12 if min_iou > 0.0 else 0):
        iou = _aligned_box_iou(candidate, box)
        invalid = active & ((iou < min_iou) | ~torch.isfinite(iou))
        candidate = torch.where(invalid[:, None], (candidate + box) * 0.5, candidate)
    if min_iou > 0.0:
        iou = _aligned_box_iou(candidate, box)
        invalid = active & ((iou < min_iou) | ~torch.isfinite(iou))
        candidate = torch.where(invalid[:, None], box, candidate)
    candidate = torch.where(active[:, None], candidate, box)

    iou = _aligned_box_iou(candidate, box)
    counts = torch.stack(
        [(severity == index).sum() for index in range(severity_count)]
    ).to(dtype=box.dtype)
    return candidate, {
        "routeb_box_jitter_count": active.sum().to(dtype=box.dtype),
        "routeb_box_jitter_clean_count": (~active).sum().to(dtype=box.dtype),
        "routeb_box_jitter_mean_iou": iou.mean(),
        "routeb_box_jitter_min_iou": iou.min(),
        "routeb_box_jitter_severity_counts": counts,
    }


def replace_training_init_with_gt_box_octagon(
    training: dict,
    jitter_config: dict | None = None,
    image_hw=None,
    return_jitter_stats: bool = False,
):
    """Replace annotation-derived initialization with the shared Route-B box."""
    target = training.get("i_gt_py")
    if not torch.is_tensor(target) or target.numel() == 0:
        return (training, {}) if return_jitter_stats else training
    gt_box = torch.cat((target.amin(dim=1), target.amax(dim=1)), dim=1)
    jitter_config = jitter_config or {
        "enabled": False,
        "probabilities": [1.0],
        "shift_fractions": [0.0],
        "log_scale_fractions": [0.0],
        "edge_fractions": [0.0],
        "min_iou": 0.0,
    }
    initial_box, statistics = jitter_routeb_boxes_xyxy(
        gt_box, jitter_config, image_hw=image_hw
    )
    result = dict(training)
    result["i_it_py"] = _box_to_octagon_init(initial_box, snake_config.poly_num)
    result["c_it_py"] = img_poly_to_can_poly(result["i_it_py"])
    result["i_it_4py"] = _box_to_octagon_init(
        initial_box, snake_config.init_poly_num
    )
    result["c_it_4py"] = img_poly_to_can_poly(result["i_it_4py"])
    return (result, statistics) if return_jitter_stats else result


def prepare_training(_output: dict, batch: dict) -> dict:
    """Flatten the fixed Route-B targets for Flow supervision."""
    valid = batch["ct_01"].bool()
    result = {
        name: collect_training(batch[name], valid)
        for name in (
            "i_it_4py",
            "c_it_4py",
            "i_gt_4py",
            "c_gt_4py",
            "i_it_py",
            "c_it_py",
            "i_gt_py",
            "c_gt_py",
        )
    }
    counts = batch["meta"]["ct_num"].to(device=valid.device, dtype=torch.long)
    owner = torch.arange(valid.shape[0], device=valid.device).repeat_interleave(counts)
    if owner.numel() != int(valid.sum().item()):
        raise RuntimeError("ct_num and ct_01 disagree")
    result["4py_ind"] = owner
    result["py_ind"] = owner
    return result


def prepare_testing(output: dict) -> dict:
    """Convert GT or externally supplied detections into Route-B contours."""
    detection = output["detection"]
    box = detection[..., :4]
    score = detection[..., 4]
    valid = score > 1e-4
    owner = valid.nonzero(as_tuple=False)[:, 0]
    if not bool(valid.any()):
        empty_40 = box.new_zeros((0, snake_config.init_poly_num, 2))
        empty_128 = box.new_zeros((0, snake_config.poly_num, 2))
        return {
            "i_it_4py": empty_40,
            "c_it_4py": empty_40.clone(),
            "ind": owner,
            "i_it_py": empty_128,
            "c_it_py": empty_128.clone(),
            "py_ind": owner,
        }

    valid_box = box[valid]
    full_scale = uniform_upsample(
        snake_decode.get_init(valid_box.unsqueeze(0)), snake_config.init_poly_num
    )[0]
    initial_40 = full_scale / float(snake_config.down_ratio)
    initial_128 = _box_to_octagon_init(
        valid_box / float(snake_config.down_ratio), snake_config.poly_num
    )
    return {
        "i_it_4py": initial_40,
        "c_it_4py": img_poly_to_can_poly(initial_40),
        "ind": owner,
        "i_it_py": initial_128,
        "c_it_py": img_poly_to_can_poly(initial_128),
        "py_ind": owner,
    }


def get_gcn_feature(
    feature: torch.Tensor,
    image_poly: torch.Tensor,
    owner: torch.Tensor,
    height: int,
    width: int,
) -> torch.Tensor:
    """Sample MoonViT-derived features at contour points using pixel centers."""
    if str(cfg.gcn_sample_mode) != "half_pixel":
        raise ValueError("the mainline requires gcn_sample_mode='half_pixel'")
    if str(cfg.gcn_sample_padding_mode) != "border":
        raise ValueError("the mainline requires gcn_sample_padding_mode='border'")
    height, width = int(height), int(width)
    if tuple(feature.shape[-2:]) != (height, width):
        raise ValueError("height/width must match the feature map")
    owner = owner.to(device=image_poly.device, dtype=torch.long)
    grid = torch.stack(
        (
            (image_poly[..., 0] + 0.5) * (2.0 / width) - 1.0,
            (image_poly[..., 1] + 0.5) * (2.0 / height) - 1.0,
        ),
        dim=-1,
    )
    sampled = feature.new_zeros(
        (image_poly.shape[0], feature.shape[1], image_poly.shape[1])
    )
    for batch_index in range(feature.shape[0]):
        selected = owner == batch_index
        if not bool(selected.any()):
            continue
        values = F.grid_sample(
            feature[batch_index : batch_index + 1],
            grid[selected].unsqueeze(0),
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )[0].permute(1, 0, 2)
        # CUDA autocast may promote ``grid_sample`` to float32 even when the
        # feature tensor (and therefore the destination cache) is float16.
        # Keep assignment explicit so mixed-precision inference follows the
        # same numerical path without a dtype mismatch.
        sampled[selected] = values.to(dtype=sampled.dtype)
    return sampled


def get_adj_ind(neighbor_count: int, node_count: int, device) -> torch.Tensor:
    """Return circular neighbor indices for each ordered contour point."""
    offsets = torch.tensor(
        [
            offset
            for offset in range(-int(neighbor_count) // 2, int(neighbor_count) // 2 + 1)
            if offset != 0
        ],
        dtype=torch.long,
        device=device,
    )
    nodes = torch.arange(int(node_count), device=device, dtype=torch.long)
    return (nodes[:, None] + offsets[None]) % int(node_count)
