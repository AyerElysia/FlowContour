"""Actual candidate overlays with a GT-independent shared local crop."""

import cv2
import numpy as np


def save_candidates(path, image, target, trajectories, quality):
    image = image[0, 0].detach().float().cpu().numpy()
    image = np.uint8(np.clip(image * .5 + .5, 0, 1) * 255)
    base = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    finals = np.stack([row["py"][0].detach().cpu().numpy() * 4 for row in trajectories])
    gt = target[0].detach().cpu().numpy() * 4
    vertex = np.argmax(finals.std(0).sum(-1))
    center = finals[:, vertex].mean(0).astype(int)
    height, width = image.shape
    x = int(np.clip(center[0] - 40, 0, max(width - 80, 0)))
    y = int(np.clip(center[1] - 40, 0, max(height - 80, 0)))
    panels = []
    for index, contour in enumerate(finals):
        panel = base.copy()
        cv2.polylines(panel, [np.round(gt).astype(np.int32)], True, (0, 200, 0), 1)
        cv2.polylines(panel, [np.round(contour).astype(np.int32)], True, (0, 170, 255), 1)
        cv2.rectangle(panel, (x, y), (x + 80, y + 80), (255, 200, 0), 1)
        crop = panel[y:y + 80, x:x + 80]
        full = cv2.resize(panel, (256, 256))
        zoom = cv2.resize(crop, (256, 256), interpolation=cv2.INTER_NEAREST)
        tile = np.concatenate((full, zoom), axis=0)
        cv2.putText(tile, f"candidate {index + 1} Q={float(quality[index, -1, 0]):.4f}",
                    (4, 18), cv2.FONT_HERSHEY_SIMPLEX, .42, (255, 255, 255), 1)
        panels.append(tile)
    grid = np.concatenate([np.concatenate(panels[:4], 1), np.concatenate(panels[4:], 1)], 0)
    if not cv2.imwrite(str(path), grid):
        raise IOError(f"could not save {path}")
