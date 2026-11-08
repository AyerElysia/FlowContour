"""Batch collation for the single released MoonViT + Flow data path."""

from __future__ import annotations

import torch
from torch.utils.data.dataloader import default_collate

from lib.config import cfg
from lib.utils.snake import snake_config


_MOONVIT_META_FIELDS = (
    "locate_feat_grid_hw",
    "locate_feat_orig_hw",
    "locate_feat_resized_hw",
    "locate_feat_padded_hw",
    "locate_feat_pad",
    "locate_feat_scale",
    "locate_feat_patch_size",
)
_CONTOUR_FIELDS = (
    ("i_it_4py", snake_config.init_poly_num),
    ("c_it_4py", snake_config.init_poly_num),
    ("i_gt_4py", 4),
    ("c_gt_4py", 4),
    ("i_it_py", snake_config.poly_num),
    ("c_it_py", snake_config.poly_num),
    ("i_gt_py", snake_config.gt_poly_num),
    ("c_gt_py", snake_config.gt_poly_num),
    ("dense_gt", 512),
)


def _flatten(samples: list[dict], field: str) -> list:
    return [value for sample in samples for value in sample[field]]


def _pack_contours(
    samples: list[dict],
    valid: torch.Tensor,
    field: str,
    point_count: int,
) -> torch.Tensor:
    packed = torch.zeros(
        (len(samples), valid.shape[1], point_count, 2), dtype=torch.float32
    )
    values = _flatten(samples, field)
    if values:
        stacked = torch.stack(
            [torch.as_tensor(value, dtype=torch.float32) for value in values]
        )
        expected = (len(values), point_count, 2)
        if tuple(stacked.shape) != expected:
            raise ValueError(
                f"{field} must have fixed shape [N,{point_count},2], "
                f"got {tuple(stacked.shape)}"
            )
        packed[valid] = stacked
    return packed


def _collate_moonvit(batch: list[dict], result: dict) -> None:
    present = ["locate_feat" in sample for sample in batch]
    if not any(present):
        return
    if not all(present):
        missing = [
            sample.get("img_path", "<unknown>")
            for sample in batch
            if "locate_feat" not in sample
        ]
        raise KeyError(f"cached MoonViT feature is missing for: {missing}")
    result["locate_feat"] = [
        torch.as_tensor(sample["locate_feat"], dtype=torch.float16)
        for sample in batch
    ]
    for field in _MOONVIT_META_FIELDS:
        result[field] = default_collate([sample[field] for sample in batch])
    result["locate_feat_path"] = [sample["locate_feat_path"] for sample in batch]


def snake_collator(batch: list[dict]) -> dict:
    """Collate fixed 40/128-point Route-B targets and MoonViT cache metadata."""
    if not batch:
        raise ValueError("cannot collate an empty batch")

    result = {
        "inp": default_collate([sample["inp"] for sample in batch]),
        "orig_img": [sample["orig_img"] for sample in batch],
        "img_path": [sample["img_path"] for sample in batch],
    }
    _collate_moonvit(batch, result)

    max_ct_cfg = getattr(cfg.train, "max_ct_num", None)
    max_ct_num = int(max_ct_cfg) if max_ct_cfg is not None else None
    samples: list[dict] = []
    counts: list[int] = []
    fields = ("wh", "ct_cls", "ct_ind", "dense_weights",
              *[name for name, _ in _CONTOUR_FIELDS])
    for sample in batch:
        count = len(sample["wh"])
        # The old trainer randomly selected up to eight training contours.
        # Retaining only the first eight biases toward low organ class IDs.
        if (sample["meta"].get("is_train", False) and
                max_ct_num is not None and count > max_ct_num):
            indices = torch.randperm(count)[:max_ct_num].sort().values.tolist()
        else:
            indices = list(range(count))
        count = len(indices)
        counts.append(count)
        samples.append({field: [sample[field][i] for i in indices] for field in fields})

    meta = default_collate([sample["meta"] for sample in batch])
    meta["ct_num"] = torch.as_tensor(counts, dtype=torch.int64)
    result["meta"] = meta

    batch_size = len(batch)
    max_count = max(counts, default=0)
    valid = torch.zeros((batch_size, max_count), dtype=torch.bool)
    for batch_index, count in enumerate(counts):
        valid[batch_index, :count] = True

    wh = torch.zeros((batch_size, max_count, 2), dtype=torch.float32)
    ct_cls = torch.zeros((batch_size, max_count), dtype=torch.int64)
    ct_ind = torch.zeros((batch_size, max_count), dtype=torch.int64)
    flat_wh = _flatten(samples, "wh")
    if flat_wh:
        wh[valid] = torch.as_tensor(flat_wh, dtype=torch.float32)
        ct_cls[valid] = torch.as_tensor(
            _flatten(samples, "ct_cls"), dtype=torch.int64
        )
        ct_ind[valid] = torch.as_tensor(
            _flatten(samples, "ct_ind"), dtype=torch.int64
        )
    result.update(
        {
            "wh": wh,
            "ct_cls": ct_cls,
            "ct_ind": ct_ind,
            "ct_01": valid.to(dtype=torch.float32),
        }
    )
    for field, point_count in _CONTOUR_FIELDS:
        result[field] = _pack_contours(samples, valid, field, point_count)
    dense_weights = torch.zeros((batch_size, max_count, 512), dtype=torch.float32)
    weights = _flatten(samples, "dense_weights")
    if weights:
        dense_weights[valid] = torch.stack(
            [torch.as_tensor(w, dtype=torch.float32) for w in weights])
    result["dense_weights"] = dense_weights
    return result


def make_collator(config):
    if config.task != "snake":
        raise ValueError(
            f"the released data path only supports task='snake', got {config.task!r}"
        )
    return snake_collator
