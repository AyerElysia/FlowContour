"""Fixed geometry constants for the released MoonViT + Flow mainline."""

from __future__ import annotations

import numpy as np


mean = np.asarray([0.40789654, 0.44719302, 0.47026115], dtype=np.float32).reshape(1, 1, 3)
std = np.asarray([0.28863828, 0.27408164, 0.27809835], dtype=np.float32).reshape(1, 1, 3)

data_rng = np.random.RandomState(123)
eig_val = np.asarray([0.2141788, 0.01817699, 0.00341571], dtype=np.float32)
eig_vec = np.asarray(
    [
        [-0.58752847, -0.69563484, 0.41340352],
        [-0.5832747, 0.00994535, -0.81221408],
        [-0.56089297, 0.71832671, 0.41158938],
    ],
    dtype=np.float32,
)

# Images and contours are represented on the stride-four feature plane.
down_ratio = 4
voc_input_h = 512
voc_input_w = 512

# Route-B always starts from a bbox octagon and evolves 128 ordered points.
init_poly_num = 40
poly_num = 128
gt_poly_num = 128
adj_num = 4
