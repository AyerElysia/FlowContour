"""Order-aware geometry without GT-dependent inference or output smoothing."""
import math

import torch
from torch.nn import functional as F


def correlated_noise(like):
    white = torch.randn_like(like)
    smooth = sum(weight * torch.roll(white, shift, 1)
                 for weight, shift in zip((1, 4, 6, 4, 1), (-2, -1, 0, 1, 2))) / 16
    smooth = smooth / math.sqrt(70 / 256)
    # Keep full-rank support and unit marginal variance, not a truncated output.
    return math.sqrt(0.96) * smooth + 0.2 * torch.randn_like(like)


def resample_closed(poly):
    n, p, _ = poly.shape
    edges = torch.roll(poly, -1, 1) - poly
    length = edges.norm(dim=-1).clamp_min(1e-7)
    arc = F.pad(length.cumsum(1), (1, 0))
    positions = arc[:, -1:] * torch.arange(p, device=poly.device, dtype=poly.dtype)[None] / p
    idx = torch.searchsorted(arc[:, 1:].contiguous(), positions.contiguous(), right=True).clamp_max(p - 1)
    starts = poly.gather(1, idx[..., None].expand(n, p, 2))
    directions = edges.gather(1, idx[..., None].expand(n, p, 2))
    alpha = (positions - arc.gather(1, idx)) / length.gather(1, idx)
    return starts + alpha[..., None] * directions


def signed_area(poly):
    q = torch.roll(poly, -1, 1)
    return 0.5 * (poly[..., 0] * q[..., 1] - poly[..., 1] * q[..., 0]).sum(1)


def align_target(pred, target):
    with torch.no_grad():
        mismatch = (signed_area(pred) >= 0) != (signed_area(target) >= 0)
        target = torch.where(mismatch[:, None, None], target.flip(1), target)
        p = target.size(1)
        shifts = torch.arange(p, device=pred.device)
        indices = (shifts[:, None] + shifts[None]) % p
        candidates = target[:, indices]
        cost = (pred.detach()[:, None] - candidates).square().sum((2, 3))
        best = cost.argmin(1)
        idx = (best[:, None] + shifts) % p
        return target.gather(1, idx[..., None].expand(-1, -1, 2))


def crossing_penalty(poly):
    a = poly[:, :, None]
    b = torch.roll(poly, -1, 1)[:, :, None]
    c, d = a.transpose(1, 2), b.transpose(1, 2)
    cross = lambda u, v: u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]
    ab, cd = b - a, d - c
    u1, u2 = cross(ab, c - a), cross(ab, d - a)
    v1, v2 = cross(cd, a - c), cross(cd, b - c)
    p = poly.size(1)
    allowed = torch.triu(torch.ones(p, p, device=poly.device, dtype=torch.bool), diagonal=2)
    allowed[0, -1] = False
    hit = ((u1.detach() * u2.detach() < -1e-9)
           & (v1.detach() * v2.detach() < -1e-9) & allowed[None])
    la, lc = ab.norm(dim=-1).clamp_min(1e-6), cd.norm(dim=-1).clamp_min(1e-6)
    depth = torch.stack((u1.abs() / la, u2.abs() / la,
                         v1.abs() / lc, v2.abs() / lc)).amin(0)
    edge_scale = (la + lc).detach() * 0.5
    return ((depth / edge_scale) * hit).sum((1, 2)).div(p).mean()


def geometry_losses(pred, target, align=False):
    if align:
        target = align_target(pred, target)
    e, g = torch.roll(pred, -1, 1) - pred, torch.roll(target, -1, 1) - target
    scale = g.norm(dim=-1).mean(1, keepdim=True).clamp_min(0.05)[..., None]
    edge = F.huber_loss(e / scale, g / scale, delta=1.0)
    bend = F.huber_loss((e - torch.roll(e, 1, 1)) / scale,
                        (g - torch.roll(g, 1, 1)) / scale, delta=1.0)
    ratio = e.norm(dim=-1).sum(1) / g.norm(dim=-1).sum(1).clamp_min(1e-6)
    excess = F.relu(ratio - 1.03)
    length = F.huber_loss(excess, torch.zeros_like(excess), delta=0.5)
    crossing = crossing_penalty(pred)
    total = 0.10 * edge + 0.05 * bend + 0.10 * length + 0.10 * crossing
    return total, {'edge': edge, 'bend': bend, 'length': length, 'crossing': crossing}
