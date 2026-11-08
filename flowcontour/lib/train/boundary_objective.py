"""Dense, curvature-weighted boundary supervision from the tuned RAOS trainer."""
import torch
from torch.nn import functional as F
from lib.networks.diffusion.ordered_geometry import geometry_losses

def point_to_segments(points, polygon):
    start=polygon[:,None,:,:]
    edge=(torch.roll(polygon,-1,1)-polygon)[:,None,:,:]
    delta=points[:,:,None,:]-start
    fraction=(delta*edge).sum(-1)/edge.square().sum(-1).clamp_min(1e-8)
    nearest=start+fraction.clamp(0,1)[...,None]*edge
    return (points[:,:,None,:]-nearest).square().sum(-1).clamp_min(1e-8).sqrt().amin(-1)

def boundary_loss(pred, target, dense_target=None, dense_weights=None):
    """Bidirectional Huber fitting; distances/radii are in 512-input pixels.

    pred and targets arrive in stride-four feature coordinates. Restoring pixel
    units here reproduces the original delta=2, radius>=4 objective; using
    delta=.5 in feature coordinates without this conversion reduces its strength
    by a factor of four.
    """
    pred = pred * 4.0
    target = (target if dense_target is None else dense_target) * 4.0
    forward = point_to_segments(pred, target)
    reverse = point_to_segments(target, pred)
    radius = (target - target.mean(1, keepdim=True)).norm(dim=-1).mean(1).clamp_min(4)
    huber = lambda d: F.huber_loss(d, torch.zeros_like(d), delta=2.0, reduction='none')
    fit = huber(forward).mean(1) / radius
    coverage = huber(reverse)
    if dense_weights is not None:
        if dense_weights.shape != coverage.shape:
            raise ValueError('Dense boundary weights must match dense target points')
        coverage = coverage * dense_weights
    return (0.5 * (fit + coverage.mean(1) / radius)).mean()

__all__=['boundary_loss','geometry_losses']
