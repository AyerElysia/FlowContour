"""Geometric candidates and their conditional projected-Gaussian surrogate.

The component (low-frequency or local bump) is sampled independently of Flow.
Its stored dictionary is held fixed when replaying the policy. This is NOT
the exact density of the full high-dimensional Flow action.
"""

import math
import torch


def contour_normals(poly):
    tangent = torch.roll(poly, -1, 1) - torch.roll(poly, 1, 1)
    # Repeated vertices: seek the closest nondegenerate cyclic tangent. A fully
    # collapsed contour gets a fixed unit normal, never a zero inverse column.
    for offset in range(2, poly.shape[1]):
        bad = tangent.norm(dim=-1) < 1e-8
        if not bad.any():
            break
        alternative = torch.roll(poly, -offset, 1) - poly
        tangent = torch.where(bad[..., None], alternative, tangent)
    normal = torch.stack((-tangent[..., 1], tangent[..., 0]), -1)
    normal = torch.where((normal.norm(dim=-1) < 1e-8)[..., None],
                         normal.new_tensor([1.0, 0.0]), normal)
    return normal / normal.norm(dim=-1, keepdim=True)


def fourier_basis(points, modes, device, dtype):
    if not 1 <= modes < points:
        raise ValueError("Fourier modes must lie in [1, point_count)")
    theta = torch.arange(points, device=device, dtype=dtype) * (2 * math.pi / points)
    columns = [torch.ones_like(theta) / math.sqrt(points)]
    frequency = 1
    while len(columns) < modes:
        columns.append(math.sqrt(2 / points) * torch.cos(frequency * theta))
        if len(columns) < modes:
            columns.append(math.sqrt(2 / points) * torch.sin(frequency * theta))
        frequency += 1
    return torch.stack(columns, -1)


def geometry_component(state, sigma, modes=8, local=False,
                       widths=(8, 12), width_probabilities=(.3, .7)):
    """Construct a fixed, per-contour linear chart in feature-grid coordinates."""
    n, p, _ = state.shape
    base = fourier_basis(p, modes, state.device, torch.float64)
    chart = (base * float(sigma)).expand(n, -1, -1).clone()
    offset = state.new_zeros((n, p), dtype=torch.float64)
    info = {"local": bool(local), "sigma_feature": float(sigma)}
    if local:
        center = torch.randint(p, (n,), device=state.device)
        choice = torch.multinomial(torch.tensor(width_probabilities, device=state.device), n, True)
        width = torch.tensor(widths, device=state.device)[choice].double()
        edges = (torch.roll(state.double(), -1, 1) - state.double()).norm(dim=-1)
        perimeter = edges.sum(1, keepdim=True)
        if (perimeter <= 1e-12).any():
            raise ValueError("a local bump needs a noncollapsed contour")
        position = (edges.cumsum(1) - edges) / perimeter
        center_position = position.gather(1, center[:, None])
        fraction = (position - center_position).abs()
        distance = torch.minimum(fraction, 1 - fraction) * 128
        bump = .5 * (1 + torch.cos(math.pi * (distance / (width[:, None] / 2)).clamp(max=1)))
        sign = torch.where(torch.rand(n, device=state.device) < .5, -1., 1.)
        height_index = torch.multinomial(state.new_tensor([.7, .25, .05]), n, True)
        height_px = state.new_tensor([2., 3., 4.])[height_index] * sign
        # Input pixels -> feature pixels. Gaussian height std is 1 input px.
        offset = bump * (height_px.double() / 4)[:, None]
        chart = torch.cat((chart, (bump / 4)[..., None]), -1)
        info.update(center=center.tolist(), chosen_width_reference=width.tolist(),
                    height_center_px=height_px.tolist())
    normals = contour_normals(state.double())
    design = (chart[..., None] * normals[:, :, None, :]).permute(0, 1, 3, 2).flatten(1, 2)
    inverse = torch.linalg.pinv(design, rtol=1e-12)
    identity_error = (inverse @ design - torch.eye(chart.shape[-1], device=state.device)).abs().amax()
    if not torch.isfinite(inverse).all() or identity_error > 1e-6:
        raise RuntimeError(f"degenerate geometry chart: {float(identity_error)}")
    return {"design": design.detach(), "inverse": inverse.detach(),
            "offset": (offset[..., None] * normals).detach(), "info": info}


def chart_logprob(action, mean, component):
    residual = (action.double() - mean.double() - component["offset"]).flatten(1)
    coefficients = torch.einsum("nmp,np->nm", component["inverse"], residual)
    return (-.5 * coefficients.square() - .5 * math.log(2 * math.pi)).mean(-1)


def chart_kl(mean, reference_mean, component):
    shift = (mean.double() - reference_mean.detach().double()).flatten(1)
    coordinates = torch.einsum("nmp,np->nm", component["inverse"], shift)
    return .5 * coordinates.square().mean(-1)


def sample_geometry(mean, state, sigma, *, local=False, modes=8):
    component = geometry_component(state, sigma, modes=modes, local=local)
    z = torch.randn(state.shape[0], component["design"].shape[-1],
                    device=state.device, dtype=torch.float64)
    perturbation = torch.einsum("npm,nm->np", component["design"], z).view_as(state)
    action = (mean.double() + perturbation + component["offset"]).to(state.dtype)
    expected = (-.5 * z.square() - .5 * math.log(2 * math.pi)).mean(-1)
    replay = chart_logprob(action, mean, component)
    error = (replay - expected).abs().amax()
    if not torch.isfinite(error) or error >= .002:
        raise RuntimeError(f"geometry probability replay failed: {float(error)}")
    component["replay_error"] = float(error)
    return action.detach(), replay.detach(), component
