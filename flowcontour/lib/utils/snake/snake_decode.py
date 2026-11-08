"""Torch geometry for converting xyxy boxes into Route-B octagons."""

from __future__ import annotations

import torch


def get_quadrangle(box: torch.Tensor) -> torch.Tensor:
    """Return top/left/bottom/right midpoints for ``[...,4]`` xyxy boxes."""
    if box.shape[-1] != 4:
        raise ValueError(f"box must end in four xyxy values, got {tuple(box.shape)}")
    x_min, y_min, x_max, y_max = box.unbind(dim=-1)
    return torch.stack(
        (
            (x_min + x_max) * 0.5,
            y_min,
            x_min,
            (y_min + y_max) * 0.5,
            (x_min + x_max) * 0.5,
            y_max,
            x_max,
            (y_min + y_max) * 0.5,
        ),
        dim=-1,
    ).reshape(*box.shape[:-1], 4, 2)


def get_octagon(extreme: torch.Tensor) -> torch.Tensor:
    """Expand T/L/B/R midpoints into the canonical 12-control-point octagon."""
    if extreme.shape[-2:] != (4, 2):
        raise ValueError(
            f"extreme points must end in [4,2], got {tuple(extreme.shape)}"
        )
    width = extreme[..., 3, 0] - extreme[..., 1, 0]
    height = extreme[..., 2, 1] - extreme[..., 0, 1]
    top = extreme[..., 0, 1]
    left = extreme[..., 1, 0]
    bottom = extreme[..., 2, 1]
    right = extreme[..., 3, 0]
    divisor = 8.0
    values = (
        extreme[..., 0, 0], extreme[..., 0, 1],
        torch.maximum(extreme[..., 0, 0] - width / divisor, left), extreme[..., 0, 1],
        extreme[..., 1, 0], torch.maximum(extreme[..., 1, 1] - height / divisor, top),
        extreme[..., 1, 0], extreme[..., 1, 1],
        extreme[..., 1, 0], torch.minimum(extreme[..., 1, 1] + height / divisor, bottom),
        torch.maximum(extreme[..., 2, 0] - width / divisor, left), extreme[..., 2, 1],
        extreme[..., 2, 0], extreme[..., 2, 1],
        torch.minimum(extreme[..., 2, 0] + width / divisor, right), extreme[..., 2, 1],
        extreme[..., 3, 0], torch.minimum(extreme[..., 3, 1] + height / divisor, bottom),
        extreme[..., 3, 0], extreme[..., 3, 1],
        extreme[..., 3, 0], torch.maximum(extreme[..., 3, 1] - height / divisor, top),
        torch.minimum(extreme[..., 0, 0] + width / divisor, right), extreme[..., 0, 1],
    )
    return torch.stack(values, dim=-1).reshape(*extreme.shape[:-2], 12, 2)


def get_init(box: torch.Tensor) -> torch.Tensor:
    """The released mainline has one initialization: bbox -> octagon."""
    return get_octagon(get_quadrangle(box))
