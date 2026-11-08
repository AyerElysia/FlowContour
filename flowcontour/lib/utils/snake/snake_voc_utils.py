"""NumPy image augmentation and contour geometry for the VerSe slice dataset."""

from __future__ import annotations

import os

import cv2
import numpy as np
from shapely.geometry import Polygon

from lib.utils import data_utils
from lib.utils.snake import snake_config


def augment(
    image,
    split,
    data_rng,
    eig_val,
    eig_vec,
    mean,
    std,
    polys=None,
    color_aug=True,
    lr_flip=True,
    random_crop=True,
):
    """Apply the one released 512px affine/photometric augmentation path."""
    del polys
    disable_aug = os.environ.get("SNAKE_DISABLE_AUG", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    disable_flip = os.environ.get("SNAKE_DISABLE_LR_FLIP", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if disable_aug and split == "train":
        split = "val"

    height, width = image.shape[:2]
    center = np.asarray([width / 2.0, height / 2.0], dtype=np.float32)
    extent = float(max(height, width))
    scale = np.asarray([extent, extent], dtype=np.float32)
    flipped = False
    if split == "train":
        if random_crop:
            scale *= np.random.uniform(0.6, 1.4)
            center_x, center_y = center
            width_border = data_utils.get_border(width / 4, scale[0]) + 1
            height_border = data_utils.get_border(height / 4, scale[0]) + 1
            center[0] = np.random.randint(
                low=max(center_x - width_border, 0),
                high=min(center_x + width_border, width - 1),
            )
            center[1] = np.random.randint(
                low=max(center_y - height_border, 0),
                high=min(center_y + height_border, height - 1),
            )
        if lr_flip and not disable_flip and np.random.random() < 0.5:
            flipped = True
            image = image[:, ::-1, :]
            center[0] = width - center[0] - 1
    else:
        center = np.asarray([width / 2.0, height / 2.0], dtype=np.float32)
        scale = np.asarray([extent, extent], dtype=np.float32)

    input_h = snake_config.voc_input_h
    input_w = snake_config.voc_input_w
    transform_input = data_utils.get_affine_transform(
        center, scale, 0, [input_w, input_h]
    )
    network_image = cv2.warpAffine(
        image, transform_input, (input_w, input_h), flags=cv2.INTER_LINEAR
    )
    original_image = network_image.copy()
    network_image = network_image.astype(np.float32) / 255.0
    if split == "train" and color_aug:
        data_utils.color_aug(data_rng, network_image, eig_val, eig_vec)
    network_image = ((network_image - mean) / std).transpose(2, 0, 1)

    output_h = input_h // snake_config.down_ratio
    output_w = input_w // snake_config.down_ratio
    transform_output = data_utils.get_affine_transform(
        center, scale, 0, [output_w, output_h]
    )
    return (
        original_image,
        network_image,
        transform_input,
        transform_output,
        flipped,
        center,
        scale,
        (input_h, input_w, output_h, output_w),
    )


def _clip_at_boundary(poly, axis, boundary, outside):
    if len(poly) == 0 or len(poly[outside(poly[:, axis], boundary)]) == len(poly):
        return []
    crossings = np.argwhere(
        outside(poly[:-1, axis], boundary)
        != outside(poly[1:, axis], boundary)
    ).ravel()
    if len(crossings) == 0:
        return poly

    segments = []
    if not outside(poly[crossings[0], axis], boundary):
        segments.append(poly[: crossings[0]])
    for index, crossing in enumerate(crossings):
        current = poly[crossing]
        following = poly[crossing + 1]
        intersection = current + (following - current) * (
            (boundary - current[axis]) / (following[axis] - current[axis])
        )
        if outside(current[axis], boundary):
            if intersection[axis] != following[axis]:
                segments.append([intersection])
            end = len(poly) if index == len(crossings) - 1 else crossings[index + 1]
            segments.append(poly[crossing + 1 : end])
        else:
            segments.append([current])
            if intersection[axis] != current[axis]:
                segments.append([intersection])
    if outside(poly[-1, axis], boundary) != outside(poly[0, axis], boundary):
        current, following = poly[-1], poly[0]
        intersection = current + (following - current) * (
            (boundary - current[axis]) / (following[axis] - current[axis])
        )
        segments.append([intersection])
    return np.concatenate(segments)


def transform_polys(polys, transform, output_h, output_w):
    """Affine-transform and clip polygons to the stride-four output plane."""
    transformed = []
    for polygon in polys:
        polygon = data_utils.affine_transform(polygon, transform)
        polygon = _clip_at_boundary(polygon, 0, 0, lambda value, edge: value < edge)
        polygon = _clip_at_boundary(
            polygon, 0, output_w, lambda value, edge: value >= edge
        )
        polygon = _clip_at_boundary(polygon, 1, 0, lambda value, edge: value < edge)
        polygon = _clip_at_boundary(
            polygon, 1, output_h, lambda value, edge: value >= edge
        )
        if len(polygon) and len(np.unique(polygon, axis=0)) > 2:
            transformed.append(polygon)
    return transformed


def filter_tiny_polys(polys):
    return [poly for poly in polys if Polygon(poly).area > 5]


def get_cw_polys(polys):
    return [poly[::-1] if Polygon(poly).exterior.is_ccw else poly for poly in polys]


def get_extreme_points(points):
    """Find stable top/left/bottom/right points on a closed polygon."""
    left, top = points[:, 0].min(), points[:, 1].min()
    right, bottom = points[:, 0].max(), points[:, 1].max()
    threshold = 0.02
    width = right - left + 1
    height = bottom - top + 1

    def plateau(seed, coordinate, maximum, tolerance):
        indices = [seed]
        cursor = (seed + 1) % len(points)
        while cursor != seed and abs(points[cursor, coordinate] - maximum) <= tolerance:
            indices.append(cursor)
            cursor = (cursor + 1) % len(points)
        cursor = (seed - 1) % len(points)
        while cursor != seed and abs(points[cursor, coordinate] - maximum) <= tolerance:
            indices.append(cursor)
            cursor = (cursor - 1) % len(points)
        return indices

    top_indices = plateau(int(np.argmin(points[:, 1])), 1, top, threshold * height)
    bottom_indices = plateau(
        int(np.argmax(points[:, 1])), 1, bottom, threshold * height
    )
    left_indices = plateau(int(np.argmin(points[:, 0])), 0, left, threshold * width)
    right_indices = plateau(
        int(np.argmax(points[:, 0])), 0, right, threshold * width
    )
    return np.asarray(
        [
            [(points[top_indices, 0].max() + points[top_indices, 0].min()) / 2, top],
            [left, (points[left_indices, 1].max() + points[left_indices, 1].min()) / 2],
            [(points[bottom_indices, 0].max() + points[bottom_indices, 0].min()) / 2, bottom],
            [right, (points[right_indices, 1].max() + points[right_indices, 1].min()) / 2],
        ]
    )


def get_quadrangle(box):
    x_min, y_min, x_max, y_max = box
    return np.asarray(
        [
            [(x_min + x_max) / 2.0, y_min],
            [x_min, (y_min + y_max) / 2.0],
            [(x_min + x_max) / 2.0, y_max],
            [x_max, (y_min + y_max) / 2.0],
        ]
    )


def get_octagon(extreme):
    width = extreme[3, 0] - extreme[1, 0]
    height = extreme[2, 1] - extreme[0, 1]
    top, left = extreme[0, 1], extreme[1, 0]
    bottom, right = extreme[2, 1], extreme[3, 0]
    divisor = 8.0
    return np.asarray(
        [
            extreme[0, 0], extreme[0, 1],
            max(extreme[0, 0] - width / divisor, left), extreme[0, 1],
            extreme[1, 0], max(extreme[1, 1] - height / divisor, top),
            extreme[1, 0], extreme[1, 1],
            extreme[1, 0], min(extreme[1, 1] + height / divisor, bottom),
            max(extreme[2, 0] - width / divisor, left), extreme[2, 1],
            extreme[2, 0], extreme[2, 1],
            min(extreme[2, 0] + width / divisor, right), extreme[2, 1],
            extreme[3, 0], min(extreme[3, 1] + height / divisor, bottom),
            extreme[3, 0], extreme[3, 1],
            extreme[3, 0], max(extreme[3, 1] - height / divisor, top),
            min(extreme[0, 0] + width / divisor, right), extreme[0, 1],
        ]
    ).reshape(-1, 2)


def get_init(box):
    return get_octagon(get_quadrangle(box))


def get_evolution_init(extreme_point, box):
    del box
    return get_octagon(extreme_point)


def uniformsample(points, new_point_count):
    """Arc-length-resample a closed NumPy polygon to an exact point count."""
    point_count, coordinate_count = points.shape
    if coordinate_count != 2:
        raise ValueError(f"polygon must be [N,2], got {points.shape}")
    next_points = points[(np.arange(point_count, dtype=np.int32) + 1) % point_count]
    edge_length = np.sqrt(np.sum((next_points - points) ** 2, axis=1))
    edge_order = np.argsort(edge_length)
    if point_count > new_point_count:
        keep = np.sort(edge_order[point_count - new_point_count :])
        return points[keep]

    total = np.sum(edge_length)
    if total <= 1e-6:
        raise ValueError("cannot resample a zero-length polygon")
    per_edge = np.round(edge_length * new_point_count / total).astype(np.int32)
    per_edge[per_edge == 0] = 1
    assigned = int(np.sum(per_edge))
    if assigned > new_point_count:
        cursor = -1
        remaining = assigned - new_point_count
        while remaining > 0:
            edge_index = edge_order[cursor]
            removable = int(per_edge[edge_index] - 1)
            take = min(removable, remaining)
            per_edge[edge_index] -= take
            remaining -= take
            cursor -= 1
    elif assigned < new_point_count:
        per_edge[edge_order[-1]] += new_point_count - assigned
    if int(np.sum(per_edge)) != int(new_point_count):
        raise RuntimeError("polygon resampling failed to allocate the requested points")

    samples = []
    for index, count in enumerate(per_edge):
        fraction = np.arange(count, dtype=np.float32).reshape(-1, 1) / count
        samples.append(
            points[index : index + 1] * (1.0 - fraction)
            + next_points[index : index + 1] * fraction
        )
    return np.concatenate(samples, axis=0)


def img_poly_to_can_poly(image_poly, x_min, y_min, x_max, y_max):
    del x_min, y_min, x_max, y_max
    return image_poly - np.min(image_poly, axis=0)
