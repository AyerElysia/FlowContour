#!/usr/bin/env python3
"""TEAMS-style 2D metrics for FlowContour semantic label masks.

This module is an independent, dependency-light adaptation of the formulas in
``third_party/TEAMS/tools/test_medical.py``.  It does not import TEAMS because
that repository's test entry point depends on an externally distributed
``lib/`` tree and an old environment.  Metrics here remain 2D pixel-domain
diagnostics; they do not replace the VerSe 3D physical-space protocol.
"""

from __future__ import print_function

import hashlib
import json
import os
import pathlib

import cv2
import numpy as np


TOLERANCES = tuple(range(1, 11))
PQ_IOU_THRESHOLD = 0.5
MIN_INSTANCE_AREA = 10


def _sha256(path):
    digest = hashlib.sha256()
    with open(os.fspath(path), "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dump_json(path, payload):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _mean(values):
    return float(np.mean(values)) if values else 0.0


def _binary_iou(pred, gt):
    pred = np.asarray(pred, dtype=bool)
    gt = np.asarray(gt, dtype=bool)
    intersection = int(np.logical_and(pred, gt).sum())
    union = int(np.logical_or(pred, gt).sum())
    return float(intersection / union) if union > 0 else 0.0


def _dice_from_iou(iou):
    return float(2.0 * iou / (iou + 1.0))


def _contour_mask(mask, tolerance):
    binary = (np.asarray(mask) > 0).astype(np.uint8)
    found = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = found[1] if len(found) == 3 else found[0]
    result = np.zeros_like(binary)
    cv2.drawContours(result, contours, -1, 1, thickness=int(tolerance))
    return result.astype(np.float32)


def _boundary_dice(pred, gt, tolerance):
    pred_boundary = _contour_mask(pred, tolerance)
    gt_boundary = _contour_mask(gt, tolerance)
    numerator = 2.0 * float((pred_boundary * gt_boundary).sum())
    denominator = float(pred_boundary.sum() + gt_boundary.sum())
    return numerator / denominator if denominator > 0 else 0.0


def _mboundf(pred, gt):
    return _mean([_boundary_dice(pred, gt, value) for value in TOLERANCES])


def _contours(mask):
    binary = (np.asarray(mask) > 0).astype(np.uint8) * 255
    if not binary.any():
        return []
    found = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = found[1] if len(found) == 3 else found[0]
    return [item.reshape(-1, 2).astype(np.float32) for item in contours if len(item) > 2]


def _nearest_distances(source, target, chunk_size=1024):
    source = np.asarray(source, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    values = []
    for start in range(0, len(source), int(chunk_size)):
        block = source[start:start + int(chunk_size)]
        squared = np.sum((block[:, None, :] - target[None, :, :]) ** 2, axis=2)
        values.append(np.sqrt(np.min(squared, axis=1)))
    return np.concatenate(values) if values else np.zeros((0,), dtype=np.float32)


def _contour_distances(first, second):
    return np.concatenate([
        _nearest_distances(first, second),
        _nearest_distances(second, first),
    ])


def _teams_hausdorff(pred, gt):
    """Return TEAMS' max-over-contour-pairs HD/HD95 and missing flag."""
    pred_contours = _contours(pred)
    gt_contours = _contours(gt)
    if not pred_contours or not gt_contours:
        return None, None, True
    hd = 0.0
    hd95 = 0.0
    for pred_contour in pred_contours:
        for gt_contour in gt_contours:
            distances = _contour_distances(pred_contour, gt_contour)
            if distances.size:
                hd = max(hd, float(np.max(distances)))
                hd95 = max(hd95, float(np.percentile(distances, 95)))
    return float(hd), float(hd95), False


def _instances(label_mask, min_area=MIN_INSTANCE_AREA):
    masks = []
    classes = []
    for class_id in sorted(int(x) for x in np.unique(label_mask) if int(x) > 0):
        binary = (np.asarray(label_mask) == class_id).astype(np.uint8)
        count, components = cv2.connectedComponents(binary)
        for component_id in range(1, int(count)):
            instance = (components == component_id).astype(np.uint8)
            if int(instance.sum()) >= int(min_area):
                masks.append(instance)
                classes.append(class_id)
    return masks, classes


def _pq(pred_masks, pred_classes, gt_masks, gt_classes):
    class_ids = sorted(set(pred_classes + gt_classes))
    per_class = {}
    for class_id in class_ids:
        pred_indices = [index for index, value in enumerate(pred_classes) if value == class_id]
        gt_indices = [index for index, value in enumerate(gt_classes) if value == class_id]
        tp = 0
        fp = len(pred_indices)
        fn = len(gt_indices)
        sum_iou = 0.0
        if pred_indices and gt_indices:
            matrix = np.zeros((len(pred_indices), len(gt_indices)), dtype=np.float32)
            for pred_row, pred_index in enumerate(pred_indices):
                for gt_row, gt_index in enumerate(gt_indices):
                    matrix[pred_row, gt_row] = _binary_iou(
                        pred_masks[pred_index], gt_masks[gt_index]
                    )
            pairs = np.argwhere(matrix >= float(PQ_IOU_THRESHOLD))
            if pairs.size:
                order = np.argsort(-matrix[pairs[:, 0], pairs[:, 1]])
                used_pred = set()
                used_gt = set()
                for index in order:
                    pred_row, gt_row = [int(x) for x in pairs[index]]
                    if pred_row in used_pred or gt_row in used_gt:
                        continue
                    used_pred.add(pred_row)
                    used_gt.add(gt_row)
                    tp += 1
                    sum_iou += float(matrix[pred_row, gt_row])
            fp -= tp
            fn -= tp
        if tp:
            sq = float(sum_iou / tp)
            rq = float(tp / (tp + 0.5 * fp + 0.5 * fn))
            pq = float(sq * rq)
        else:
            pq = sq = rq = 0.0
        per_class[str(class_id)] = {
            "pq": pq,
            "sq": sq,
            "rq": rq,
            "tp": int(tp),
            "fp": int(max(fp, 0)),
            "fn": int(max(fn, 0)),
        }
    return per_class, _mean([item["pq"] for item in per_class.values()])


def _gray_bgr(image):
    image = np.asarray(image)
    if image.ndim not in (2, 3):
        raise ValueError("visualization image must be HxW or HxWxC, got {}".format(image.shape))
    finite = image[np.isfinite(image)]
    lo, hi = np.percentile(finite, [1.0, 99.0]) if finite.size else (0.0, 1.0)
    if hi <= lo:
        hi = lo + 1.0
    normalized = np.clip((image.astype(np.float32) - lo) / (hi - lo), 0.0, 1.0)
    gray = np.rint(normalized * 255.0).astype(np.uint8)
    if gray.ndim == 2:
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    if gray.shape[2] == 1:
        return cv2.cvtColor(gray[:, :, 0], cv2.COLOR_GRAY2BGR)
    if gray.shape[2] == 3:
        return np.ascontiguousarray(gray)
    if gray.shape[2] == 4:
        return cv2.cvtColor(gray, cv2.COLOR_BGRA2BGR)
    raise ValueError("unsupported visualization channel count {}".format(gray.shape[2]))


def _colors():
    colors = np.zeros((256, 3), dtype=np.uint8)
    for label in range(1, 256):
        hsv = np.uint8([[[int((label * 37) % 180), 210, 245]]])
        colors[label] = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return colors


def _overlay(base, labels, colors):
    result = base.copy()
    foreground = labels > 0
    if foreground.any():
        color = colors[np.asarray(labels, dtype=np.uint8)]
        result[foreground] = np.rint(
            0.54 * result[foreground].astype(np.float32)
            + 0.46 * color[foreground].astype(np.float32)
        ).astype(np.uint8)
    return result


def _title(panel, text):
    panel = cv2.copyMakeBorder(panel, 36, 0, 0, 0, cv2.BORDER_CONSTANT, value=(18, 18, 18))
    cv2.putText(panel, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (245, 245, 245), 2, cv2.LINE_AA)
    return panel


class TEAMSStyleCollector(object):
    """Streaming collector for TEAMS-compatible and diagnostic metrics."""

    def __init__(self, result_dir, upstream_root=None):
        self.result_dir = pathlib.Path(result_dir).resolve()
        self.prediction_dir = self.result_dir / "predictions"
        self.prediction_dir.mkdir(parents=True, exist_ok=True)
        self.upstream_root = pathlib.Path(upstream_root).resolve() if upstream_root else None
        self.rows = []
        self._class_rows = {}
        self._class_rows_failure_inclusive = {}
        self._pq_class_rows = {}

    def add(self, case_id, slice_idx, image_path, mask_path, pred, gt):
        pred = np.asarray(pred, dtype=np.uint16)
        gt = np.asarray(gt, dtype=np.uint16)
        if pred.shape != gt.shape:
            raise ValueError("TEAMS mask shape mismatch: {} != {}".format(pred.shape, gt.shape))
        case_dir = self.prediction_dir / str(case_id)
        case_dir.mkdir(parents=True, exist_ok=True)
        pred_path = case_dir / "slice_{:04d}.png".format(int(slice_idx))
        if not cv2.imwrite(os.fspath(pred_path), pred):
            raise RuntimeError("failed to write {}".format(pred_path))

        pred_fg = pred > 0
        gt_fg = gt > 0
        iou = _binary_iou(pred_fg, gt_fg)
        dice = _dice_from_iou(iou)
        mboundf = _mboundf(pred_fg, gt_fg)
        gt_labels = sorted(int(x) for x in np.unique(gt) if int(x) > 0)
        pred_labels = sorted(int(x) for x in np.unique(pred) if int(x) > 0)
        class_metrics = {}
        class_boundaries = []
        for class_id in gt_labels:
            class_pred = pred == class_id
            class_gt = gt == class_id
            class_iou = _binary_iou(class_pred, class_gt)
            class_dice = _dice_from_iou(class_iou)
            class_mboundf = _mboundf(class_pred, class_gt)
            hd, hd95, missing = _teams_hausdorff(class_pred, class_gt)
            # TEAMS converts missing-contour infinity to 0.0; retain that
            # compatibility value and expose the missing flag explicitly.
            item = {
                "iou": class_iou,
                "dice": class_dice,
                "mboundf_1_to_10px": class_mboundf,
                "hd_pixels_teams_compat": 0.0 if missing else hd,
                "hd95_pixels_teams_compat": 0.0 if missing else hd95,
                "missing_contour": bool(missing),
            }
            class_metrics[str(class_id)] = item
            class_boundaries.append(class_mboundf)
            self._class_rows_failure_inclusive.setdefault(str(class_id), []).append(item)
            if pred_labels:
                self._class_rows.setdefault(str(class_id), []).append(item)

        pred_instances, pred_instance_classes = _instances(pred)
        gt_instances, gt_instance_classes = _instances(gt)
        if pred_instances or gt_instances:
            pq_per_class, pq_mean = _pq(
                pred_instances, pred_instance_classes, gt_instances, gt_instance_classes
            )
            for class_id, item in pq_per_class.items():
                self._pq_class_rows.setdefault(class_id, []).append(item)
        else:
            pq_per_class, pq_mean = {}, None

        row = {
            "case_id": str(case_id),
            "slice_idx": int(slice_idx),
            "image_path": os.path.abspath(os.fspath(image_path)),
            "mask_path": os.path.abspath(os.fspath(mask_path)),
            "prediction_path": str(pred_path),
            "gt_foreground_pixels": int(gt_fg.sum()),
            "pred_foreground_pixels": int(pred_fg.sum()),
            "class_agnostic": {
                "iou": iou,
                "dice_from_iou": dice,
                "mboundf_1_to_10px": mboundf,
            },
            "class_aware_gt_present": {
                "mean_iou": _mean([item["iou"] for item in class_metrics.values()]),
                "mean_dice": _mean([item["dice"] for item in class_metrics.values()]),
                "mean_mboundf_1_to_10px": _mean(class_boundaries),
                "per_class": class_metrics,
            },
            "instances_area_ge_10px": {
                "prediction_count": len(pred_instances),
                "ground_truth_count": len(gt_instances),
                "pq_iou_ge_0_5": pq_mean,
                "per_class": pq_per_class,
            },
        }
        self.rows.append(row)
        return row

    @staticmethod
    def _aggregate_classes(source):
        result = {}
        for class_id, rows in sorted(source.items(), key=lambda item: int(item[0])):
            result[class_id] = {
                "image_count": len(rows),
                "mean_iou": _mean([row["iou"] for row in rows]),
                "mean_dice": _mean([row["dice"] for row in rows]),
                "mean_mboundf_1_to_10px": _mean([row["mboundf_1_to_10px"] for row in rows]),
                "mean_hd_pixels_teams_compat": _mean([row["hd_pixels_teams_compat"] for row in rows]),
                "mean_hd95_pixels_teams_compat": _mean([row["hd95_pixels_teams_compat"] for row in rows]),
                "missing_contour_count": sum(int(row["missing_contour"]) for row in rows),
            }
        return result

    def _visualize(self, path):
        foreground = sorted(
            [row for row in self.rows if row["gt_foreground_pixels"] > 0],
            key=lambda row: row["class_agnostic"]["dice_from_iou"],
        )
        if not foreground:
            raise RuntimeError("no foreground slices for TEAMS visualization")
        percentiles = (0.05, 0.20, 0.40, 0.60, 0.80, 0.95)
        selected = []
        used = set()
        for percentile in percentiles:
            index = int(round(percentile * (len(foreground) - 1)))
            while index in used and index + 1 < len(foreground):
                index += 1
            used.add(index)
            selected.append((percentile, foreground[index]))
        colors = _colors()
        canvases = []
        selected_rows = []
        for percentile, row in selected:
            image = cv2.imread(row["image_path"], cv2.IMREAD_UNCHANGED)
            gt = cv2.imread(row["mask_path"], cv2.IMREAD_UNCHANGED)
            pred = cv2.imread(row["prediction_path"], cv2.IMREAD_UNCHANGED)
            if image is None or gt is None or pred is None:
                raise FileNotFoundError("TEAMS visualization input missing")
            base = _gray_bgr(image)
            error = base.copy()
            gt_fg = gt > 0
            pred_fg = pred > 0
            correct = gt_fg & pred_fg
            false_positive = (~gt_fg) & pred_fg
            false_negative = gt_fg & (~pred_fg)
            wrong_class = correct & (gt != pred)
            error[correct] = np.asarray([0, 205, 0], dtype=np.uint8)
            error[false_positive] = np.asarray([20, 20, 245], dtype=np.uint8)
            error[false_negative] = np.asarray([245, 80, 20], dtype=np.uint8)
            error[wrong_class] = np.asarray([215, 35, 215], dtype=np.uint8)
            panels = [base, _overlay(base, gt, colors), _overlay(base, pred, colors), error]
            titles = ["Input", "Ground truth", "Prediction", "TP green / FP red / FN blue / class magenta"]
            panels = [cv2.resize(item, (340, 340), interpolation=cv2.INTER_AREA) for item in panels]
            panels = [_title(item, title) for item, title in zip(panels, titles)]
            canvas = np.concatenate(panels, axis=1)
            footer = np.full((40, canvas.shape[1], 3), 18, dtype=np.uint8)
            caption = "P{:02d}  {} x{:04d}  Dice={:.4f}  IoU={:.4f}  mBoundF={:.4f}".format(
                int(round(100 * percentile)), row["case_id"], row["slice_idx"],
                row["class_agnostic"]["dice_from_iou"],
                row["class_agnostic"]["iou"],
                row["class_agnostic"]["mboundf_1_to_10px"],
            )
            cv2.putText(footer, caption, (10, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (245, 245, 245), 2, cv2.LINE_AA)
            canvases.append(np.concatenate([canvas, footer], axis=0))
            selected_rows.append({
                "percentile": percentile,
                "case_id": row["case_id"],
                "slice_idx": row["slice_idx"],
                "dice": row["class_agnostic"]["dice_from_iou"],
                "iou": row["class_agnostic"]["iou"],
                "mboundf": row["class_agnostic"]["mboundf_1_to_10px"],
            })
        output = np.concatenate(canvases, axis=0)
        if not cv2.imwrite(os.fspath(path), output):
            raise RuntimeError("failed to write {}".format(path))
        return selected_rows

    def finalize(self):
        if not self.rows:
            raise RuntimeError("no TEAMS-style rows collected")
        foreground = [row for row in self.rows if row["gt_foreground_pixels"] > 0]
        pq_rows = [row for row in self.rows if row["instances_area_ge_10px"]["pq_iou_ge_0_5"] is not None]
        class_compat = self._aggregate_classes(self._class_rows)
        class_failure = self._aggregate_classes(self._class_rows_failure_inclusive)
        pq_classes = {}
        for class_id, rows in sorted(self._pq_class_rows.items(), key=lambda item: int(item[0])):
            pq_classes[class_id] = {
                "image_count": len(rows),
                "mean_pq": _mean([row["pq"] for row in rows]),
                "mean_sq": _mean([row["sq"] for row in rows]),
                "mean_rq": _mean([row["rq"] for row in rows]),
                "tp_sum": sum(row["tp"] for row in rows),
                "fp_sum": sum(row["fp"] for row in rows),
                "fn_sum": sum(row["fn"] for row in rows),
            }
        visualization_path = self.result_dir / "teams_style_2d_visualization_6slice.png"
        selected = self._visualize(visualization_path)
        upstream = {
            "repository": "https://github.com/Richard-Zhang-AI/TEAMS",
            "adapted_file": "tools/test_medical.py",
        }
        if self.upstream_root:
            test_file = self.upstream_root / "tools" / "test_medical.py"
            readme_file = self.upstream_root / "README.md"
            upstream.update({
                "local_root": str(self.upstream_root),
                "test_medical_sha256": _sha256(test_file) if test_file.is_file() else None,
                "readme_sha256": _sha256(readme_file) if readme_file.is_file() else None,
            })
        all_metrics = {
            "image_count": len(self.rows),
            "mean_iou": _mean([row["class_agnostic"]["iou"] for row in self.rows]),
            "mean_dice_from_iou": _mean([row["class_agnostic"]["dice_from_iou"] for row in self.rows]),
            "mean_mboundf_1_to_10px": _mean([row["class_agnostic"]["mboundf_1_to_10px"] for row in self.rows]),
        }
        foreground_metrics = {
            "image_count": len(foreground),
            "mean_iou": _mean([row["class_agnostic"]["iou"] for row in foreground]),
            "mean_dice_from_iou": _mean([row["class_agnostic"]["dice_from_iou"] for row in foreground]),
            "mean_mboundf_1_to_10px": _mean([row["class_agnostic"]["mboundf_1_to_10px"] for row in foreground]),
        }
        summary = {
            "schema": "diffusionsnake.teams_style_2d_metrics.v1",
            "status": "PASS_TEAMS_STYLE_SUPPLEMENTARY_2D",
            "scope_warning": "2D pixel-domain diagnostic; not VerSe official 3D physical-space evaluation",
            "upstream": upstream,
            "definitions": {
                "empty_empty_iou": 0.0,
                "dice": "2*IoU/(1+IoU)",
                "mboundf_tolerances_pixels": list(TOLERANCES),
                "instance_min_area_pixels": MIN_INSTANCE_AREA,
                "pq_match_iou_threshold": PQ_IOU_THRESHOLD,
                "hd_units": "2D pixels",
                "missing_contour_hd_teams_compat": 0.0,
                "aggregation": "image-equal, then class-equal where applicable",
            },
            "class_agnostic_all_images_upstream_compat": all_metrics,
            "class_agnostic_foreground_images_extension": foreground_metrics,
            "class_aware_upstream_outer_gate_compat": {
                "per_class": class_compat,
                "macro_mean_iou": _mean([row["mean_iou"] for row in class_compat.values()]),
                "macro_mean_dice": _mean([row["mean_dice"] for row in class_compat.values()]),
                "macro_mean_mboundf_1_to_10px": _mean([row["mean_mboundf_1_to_10px"] for row in class_compat.values()]),
            },
            "class_aware_failure_inclusive_extension": {
                "per_class": class_failure,
                "macro_mean_iou": _mean([row["mean_iou"] for row in class_failure.values()]),
                "macro_mean_dice": _mean([row["mean_dice"] for row in class_failure.values()]),
                "macro_mean_mboundf_1_to_10px": _mean([row["mean_mboundf_1_to_10px"] for row in class_failure.values()]),
            },
            "instance_pq_upstream_compat": {
                "evaluated_image_count": len(pq_rows),
                "image_equal_mean_pq": _mean([
                    row["instances_area_ge_10px"]["pq_iou_ge_0_5"] for row in pq_rows
                ]),
                "per_class": pq_classes,
            },
            "visualization": {
                "path": str(visualization_path),
                "sha256": _sha256(visualization_path),
                "selected_slices": selected,
            },
        }
        _dump_json(self.result_dir / "per_image.json", self.rows)
        _dump_json(self.result_dir / "summary.json", summary)
        return summary
