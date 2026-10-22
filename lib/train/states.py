"""Mixed artificial states and terminal-quality credit, with no legacy paths."""

import torch
from lib.rl.geometry import contour_normals, fourier_basis


def mixed_states(initial, target, probabilities=(.5, .2, .3), refine_sigma_px=3., refine_modes=8):
    probabilities = initial.new_tensor(probabilities)
    if probabilities.numel() != 3 or (probabilities < 0).any() or abs(float(probabilities.sum()) - 1) > 1e-6:
        raise ValueError("state probabilities must be three nonnegative values summing to one")
    count = initial.shape[0]
    kind = torch.multinomial(probabilities, count, True)
    alpha = torch.rand(count, 1, 1, device=initial.device, dtype=initial.dtype)
    middle = initial + alpha * (target - initial)
    basis = fourier_basis(target.shape[1], refine_modes, target.device, target.dtype)
    coefficients = torch.randn(count, refine_modes, device=target.device, dtype=target.dtype)
    scalar = coefficients @ basis.T * (refine_sigma_px / 4)
    refined = target + scalar[..., None] * contour_normals(target)
    current = torch.where((kind == 1)[:, None, None], middle, initial)
    current = torch.where((kind == 2)[:, None, None], refined, current)
    # Phase is an action identifier, not interpolation progress. Intermediate
    # states cover BOTH phase IDs; initial=update, GT-near=refinement.
    phase = torch.randint(2, (count,), device=initial.device).to(initial.dtype)
    phase = torch.where(kind == 0, torch.zeros_like(phase), phase)
    phase = torch.where(kind == 2, torch.ones_like(phase), phase)
    return current, phase, kind


@torch.no_grad()
def terminal_credit(rewards, stage_quality, std_floor=.1, clip=2.):
    if std_floor <= 0 or clip <= 0:
        raise ValueError("advantage floor and clip must be positive")
    if rewards.ndim != 2 or stage_quality.shape != (rewards.shape[0], 2, rewards.shape[1]):
        raise ValueError("expected terminal [K,N], stage quality [K,2,N]")
    if not torch.isfinite(stage_quality).all() or not torch.isfinite(rewards).all():
        raise ValueError("nonfinite quality")
    if (stage_quality < 0).any() or (stage_quality > 1).any():
        raise ValueError("stage quality must be absolute quality in [0,1]")
    advantage = ((rewards - rewards.mean(0)) /
                 rewards.std(0, unbiased=False).clamp_min(std_floor)).clamp(-clip, clip)
    total = stage_quality.sum(1, keepdim=True)
    weights = torch.where(total > 0, 2 * stage_quality / total.clamp_min(1e-12),
                          torch.ones_like(stage_quality))
    allocated = advantage[:, None] * weights
    if (allocated.mean(1) - advantage).abs().amax() > 1e-6:
        raise RuntimeError("terminal credit conservation failed")
    return advantage, weights, allocated
