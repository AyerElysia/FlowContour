"""Non-differentiable contour rewards used by supervised evaluation and GDPO."""

import cv2
import numpy as np
import torch


def _poly_to_mask_np(poly: np.ndarray, H: int, W: int) -> np.ndarray:
    """将多边形轮廓转换为二值掩码。"""
    m = np.zeros((H, W), dtype=np.uint8)
    p = np.round(poly).astype(np.int32)
    if p.ndim == 2 and p.shape[0] > 2:
        cv2.fillPoly(m, [p.reshape(-1, 1, 2)], 1)
    return m


def _extract_contour(mask: np.ndarray, tolerance: int = 1) -> np.ndarray:
    """从二值掩码中提取边界（通过描绘轮廓实现，可调厚度）"""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_mask = np.zeros_like(mask, dtype=np.uint8)
    if len(contours) > 0:
        cv2.drawContours(contour_mask, contours, -1, 1, thickness=tolerance)
    return contour_mask.astype(np.float32)


def _boundary_dice(pred_mask: np.ndarray, gt_mask: np.ndarray, tolerance: int) -> float:
    pred_contour = _extract_contour(pred_mask, tolerance)
    gt_contour = _extract_contour(gt_mask, tolerance)
    intersection = (pred_contour * gt_contour).sum()
    union = pred_contour.sum() + gt_contour.sum()
    return float(2.0 * intersection / union) if union > 0 else 0.0


def _calc_mboundf(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """复用 test_medical 中逻辑：多容忍度下的边界Dice平均"""
    tolerances = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    vals = []
    for tol in tolerances:
        vals.append(_boundary_dice(pred_mask, gt_mask, tol))
    return float(np.mean(vals)) if len(vals) > 0 else 0.0


def _calc_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    inter = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    return float(inter / union) if union > 0 else 0.0


def _calc_dice(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    inter = np.logical_and(pred_mask, gt_mask).sum()
    denom = pred_mask.sum() + gt_mask.sum()
    return float(2.0 * inter / denom) if denom > 0 else 0.0


def _calc_boundary_distance_score(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    max_dist: float = 8.0,
    quantile: float = 95.0,
    quantile_weight: float = 0.5,
) -> float:
    """双向边界距离分数；越贴近 GT 越接近 1。"""
    pred_contour = _extract_contour(pred_mask, 1) > 0
    gt_contour = _extract_contour(gt_mask, 1) > 0
    if pred_contour.sum() == 0 or gt_contour.sum() == 0:
        return 0.0

    max_dist = max(float(max_dist), 1e-6)
    quantile = min(max(float(quantile), 0.0), 100.0)
    quantile_weight = max(float(quantile_weight), 0.0)

    gt_dt = cv2.distanceTransform((~gt_contour).astype(np.uint8), cv2.DIST_L2, 3)
    pred_dt = cv2.distanceTransform((~pred_contour).astype(np.uint8), cv2.DIST_L2, 3)
    d_pred_to_gt = gt_dt[pred_contour]
    d_gt_to_pred = pred_dt[gt_contour]
    if d_pred_to_gt.size == 0 or d_gt_to_pred.size == 0:
        return 0.0

    mean_dist = 0.5 * (float(d_pred_to_gt.mean()) + float(d_gt_to_pred.mean()))
    q_dist = 0.5 * (
        float(np.percentile(d_pred_to_gt, quantile)) +
        float(np.percentile(d_gt_to_pred, quantile))
    )
    dist = (mean_dist + quantile_weight * q_dist) / (1.0 + quantile_weight)
    return float(1.0 - np.clip(dist / max_dist, 0.0, 1.0))


def _calc_nsd(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    delta_px: float = 2.0,
) -> float:
    """Symmetric normalized surface Dice at a 2D pixel tolerance.

    Half of the score measures the fraction of predicted boundary pixels that
    lie within ``delta_px`` of the GT boundary.  The other half measures the
    reverse direction. This is the released contour-RL reward and is
    intentionally expressed in image pixels, not the millimetre-space 3D NSD
    reported by the VerSe evaluator.
    """

    pred_contour = _extract_contour(pred_mask, 1) > 0
    gt_contour = _extract_contour(gt_mask, 1) > 0
    n_pred = int(pred_contour.sum())
    n_gt = int(gt_contour.sum())
    if n_pred == 0 or n_gt == 0:
        return 0.0

    delta = max(float(delta_px), 0.5)
    gt_distance = cv2.distanceTransform(
        (~gt_contour).astype(np.uint8), cv2.DIST_L2, 3
    )
    pred_distance = cv2.distanceTransform(
        (~pred_contour).astype(np.uint8), cv2.DIST_L2, 3
    )
    pred_covered = float((gt_distance[pred_contour] <= delta).sum()) / n_pred
    gt_covered = float((pred_distance[gt_contour] <= delta).sum()) / n_gt
    return 0.5 * pred_covered + 0.5 * gt_covered


def compute_region_score(poly_py: torch.Tensor,
                         gt_py: torch.Tensor,
                         H: int,
                         W: int,
                         w_boundary: float = 1.0,
                         w_dice: float = 0.0,
                         w_iou: float = 0.0,
                         w_dist: float = 0.0,
                         dist_max_px: float = 8.0,
                         dist_quantile: float = 95.0,
                         dist_quantile_weight: float = 0.5,
                         coord_scale: float = 1.0) -> torch.Tensor:
    """Compute absolute contour quality from boundary F-score, Dice, and optional IoU."""
    device = poly_py.device
    pred_np = poly_py.detach().float().cpu().numpy() * float(coord_scale)
    gt_np = gt_py.detach().float().cpu().numpy() * float(coord_scale)

    scores = []
    for i in range(pred_np.shape[0]):
        m_pred = _poly_to_mask_np(pred_np[i], H, W)
        m_gt = _poly_to_mask_np(gt_np[i], H, W)
        score = 0.0
        if w_boundary != 0:
            score += float(w_boundary) * _calc_mboundf(m_pred, m_gt)
        if w_dice != 0:
            score += float(w_dice) * _calc_dice(m_pred, m_gt)
        if w_iou != 0:
            score += float(w_iou) * _calc_iou(m_pred, m_gt)
        if w_dist != 0:
            score += float(w_dist) * _calc_boundary_distance_score(
                m_pred,
                m_gt,
                max_dist=dist_max_px,
                quantile=dist_quantile,
                quantile_weight=dist_quantile_weight,
            )
        scores.append(score)

    return torch.tensor(scores, device=device, dtype=torch.float32)


def compute_nsd_score(
    poly_py: torch.Tensor,
    gt_py: torch.Tensor,
    H: int,
    W: int,
    delta_px: float = 2.0,
    coord_scale: float = 1.0,
) -> torch.Tensor:
    """Return per-contour 2D NSD scores in ``[0, 1]``."""

    device = poly_py.device
    pred_np = poly_py.detach().float().cpu().numpy() * float(coord_scale)
    gt_np = gt_py.detach().float().cpu().numpy() * float(coord_scale)
    scores = []
    for index in range(pred_np.shape[0]):
        pred_mask = _poly_to_mask_np(pred_np[index], H, W)
        gt_mask = _poly_to_mask_np(gt_np[index], H, W)
        scores.append(_calc_nsd(pred_mask, gt_mask, delta_px=delta_px))
    return torch.tensor(scores, device=device, dtype=torch.float32)


def compute_delta_nsd_reward(
    sampled_nsd: torch.Tensor,
    deterministic_nsd: torch.Tensor,
    sampled_burr: torch.Tensor,
    burr_weight: float = 0.06,
) -> torch.Tensor:
    """Return the released five-stage Fourier GDPO reward.

    The deterministic term is a same-stage, same-latent counterfactual.  The
    burr regularizer is absolute on the sampled endpoint; it is deliberately
    not a sampled-minus-deterministic burr difference.
    """

    return (
        sampled_nsd
        - deterministic_nsd
        - float(burr_weight) * sampled_burr
    )
