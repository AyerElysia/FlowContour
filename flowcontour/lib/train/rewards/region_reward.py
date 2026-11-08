"""绝对硬多尺度 NSD 奖励；Dice/IoU 只作报告，不进入 GDPO 目标。"""

import cv2
import numpy as np
import torch
from lib.protocol import NSD_SCALES_PX


def _mask(poly, height, width):
    """把闭合多边形栅格化到网络输入分辨率。"""
    mask = np.zeros((height, width), np.uint8)
    if not np.isfinite(poly).all():
        raise ValueError("nonfinite contour")
    if poly.shape[0] >= 3:
        cv2.fillPoly(mask, [np.round(poly).astype(np.int32)], 1)
    return mask


def _boundary(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result = np.zeros_like(mask)
    if contours:
        cv2.drawContours(result, contours, -1, 1, thickness=1)
    return result.astype(bool)


def boundary_distances(pred, target):
    """预测边界到 GT 边界、以及反向的 L2 距离。空边界返回 None（NSD=0）。"""
    a, b = _boundary(pred), _boundary(target)
    if not a.any() or not b.any():
        return None
    # Retain the released OpenCV L2/mask3 distance convention.
    to_b = cv2.distanceTransform((~b).astype(np.uint8), cv2.DIST_L2, 3)[a]
    to_a = cv2.distanceTransform((~a).astype(np.uint8), cv2.DIST_L2, 3)[b]
    return to_b, to_a


def contour_metrics(poly, target, height, width, coord_scale=4.):
    """在网络输入像素上打分，不是特征网格像素，也不是 CT 毫米。

    默认 coord_scale=4：轮廓存在 stride-4 特征坐标里，乘 down_ratio 才到输入像素。
    quality = NSD@1/2/4/8px 等权平均，这才是 GDPO 奖励。
    """
    pred_np = poly.detach().float().cpu().numpy() * coord_scale
    gt_np = target.detach().float().cpu().numpy() * coord_scale
    if pred_np.shape != gt_np.shape:
        raise ValueError("prediction and target contour shapes differ")
    rows = []
    for a, b in zip(pred_np, gt_np):
        pred, gt = _mask(a, height, width), _mask(b, height, width)
        distances = boundary_distances(pred, gt)
        scales = [0.] * len(NSD_SCALES_PX) if distances is None else [
            .5 * np.mean(distances[0] <= tau) + .5 * np.mean(distances[1] <= tau)
            for tau in NSD_SCALES_PX]
        intersection = np.logical_and(pred, gt).sum()
        rows.append(scales + [2 * intersection / max(int(pred.sum() + gt.sum()), 1),
                              intersection / max(int(np.logical_or(pred, gt).sum()), 1)])
    values = torch.as_tensor(rows, device=poly.device, dtype=torch.float32).reshape(-1, 6)
    result = {f"nsd_{int(tau)}px": values[:, index] for index, tau in enumerate(NSD_SCALES_PX)}
    result.update(quality=values[:, :4].mean(-1), dice=values[:, 4], iou=values[:, 5])
    return result
