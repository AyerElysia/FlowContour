"""Small numerical helpers shared by the mainline GDPO trainer."""
from __future__ import annotations
import torch


class EMA:
    """Scalar exponential moving average."""
    def __init__(self, decay: float = 0.99):
        self.decay = float(decay)
        self.value: float = 0.0
        self.initialized = False

    def update(self, x: float) -> float:
        x = float(x)
        if not self.initialized:
            self.value = x
            self.initialized = True
        else:
            self.value = self.decay * self.value + (1.0 - self.decay) * x
        return self.value


def freeze_bn_running_stats(model: torch.nn.Module) -> int:
    """Set all BN layers to eval() so running mean/var are frozen during RL/GDPO.

    Returns count of BN layers frozen. Keeps params trainable but stats frozen.
    Crucial for RL fine-tuning to avoid policy drift from BN stat updates.
    """
    n = 0
    for m in model.modules():
        if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d,
                          torch.nn.BatchNorm3d, torch.nn.SyncBatchNorm)):
            m.eval()
            n += 1
    return n


def percentiles(xs: torch.Tensor) -> dict:
    """Return p10/p50/p90 of a 1-D tensor (or empty if too small)."""
    if xs.numel() == 0:
        return {'p10': 0.0, 'p50': 0.0, 'p90': 0.0}
    q = torch.quantile(xs.float(), torch.tensor([0.10, 0.50, 0.90], device=xs.device))
    return {'p10': float(q[0]), 'p50': float(q[1]), 'p90': float(q[2])}
