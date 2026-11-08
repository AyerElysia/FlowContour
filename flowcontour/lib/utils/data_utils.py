"""Small image/geometry helpers shared by the released data and model paths."""

from __future__ import annotations

import random

import cv2
import numpy as np
import torch


def _third_point(first, second):
    direction = first - second
    return second + np.asarray([-direction[1], direction[0]], dtype=np.float32)


def _rotate_direction(point, radians):
    sine, cosine = np.sin(radians), np.cos(radians)
    return [
        point[0] * cosine - point[1] * sine,
        point[0] * sine + point[1] * cosine,
    ]


def get_affine_transform(
    center,
    scale,
    rotation,
    output_size,
    shift=np.asarray([0, 0], dtype=np.float32),
    inv=0,
):
    """Construct the CenterNet-style affine transform used by the dataset."""
    if not isinstance(scale, (np.ndarray, list)):
        scale = np.asarray([scale, scale], dtype=np.float32)
    scale = np.asarray(scale, dtype=np.float32)
    source_width = scale[0]
    destination_width, destination_height = output_size
    source_direction = _rotate_direction(
        [0, source_width * -0.5], np.pi * rotation / 180
    )
    destination_direction = np.asarray(
        [0, destination_width * -0.5], dtype=np.float32
    )
    source = np.zeros((3, 2), dtype=np.float32)
    destination = np.zeros((3, 2), dtype=np.float32)
    source[0] = center + scale * shift
    source[1] = center + source_direction + scale * shift
    destination[0] = [destination_width * 0.5, destination_height * 0.5]
    destination[1] = destination[0] + destination_direction
    source[2] = _third_point(source[0], source[1])
    destination[2] = _third_point(destination[0], destination[1])
    if inv:
        return cv2.getAffineTransform(destination, source)
    return cv2.getAffineTransform(source, destination)


def affine_transform(points, transform):
    """Apply a 2x3 affine matrix to an ``[N,2]`` point array."""
    return np.dot(np.asarray(points), transform[:, :2].T) + transform[:, 2]


def get_border(border, size):
    divisor = 1
    while np.any(size - border // divisor <= border // divisor):
        divisor *= 2
    return border // divisor


def _grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _blend(alpha, first, second):
    first *= alpha
    second *= 1 - alpha
    first += second


def _brightness(data_rng, image, _gray, _mean, variance):
    image *= 1.0 + data_rng.uniform(low=-variance, high=variance)


def _contrast(data_rng, image, _gray, gray_mean, variance):
    _blend(1.0 + data_rng.uniform(low=-variance, high=variance), image, gray_mean)


def _saturation(data_rng, image, gray, _mean, variance):
    _blend(
        1.0 + data_rng.uniform(low=-variance, high=variance),
        image,
        gray[:, :, None],
    )


def color_aug(data_rng, image, eig_val, eig_vec):
    """Apply the original brightness/contrast/saturation/PCA augmentation."""
    transforms = [_brightness, _contrast, _saturation]
    random.shuffle(transforms)
    gray = _grayscale(image)
    gray_mean = gray.mean()
    for transform in transforms:
        transform(data_rng, image, gray, gray_mean, 0.4)
    alpha = data_rng.normal(scale=0.1, size=(3,))
    image += np.dot(eig_vec, eig_val * alpha)


def clip_to_image(box, height, width):
    """Clip xyxy tensors in place to image bounds."""
    box[..., :2] = torch.clamp(box[..., :2], min=0)
    box[..., 2] = torch.clamp(box[..., 2], max=width - 1)
    box[..., 3] = torch.clamp(box[..., 3], max=height - 1)
    return box
