"""One Flow implementation: mixed-state FM, online imitation and full2 AB2."""

import json
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.config import cfg
from lib.protocol import ODE_STEPS, STAGE_VALUES
from lib.train.states import mixed_states
from lib.utils.snake import snake_gcn_utils as geometry
from .mainline_denoiser import MainlineFlowDenoiser


class FlowMatchingEvolution(nn.Module):
    def __init__(self):
        super().__init__()
        actual = (cfg.snake_feature_dim, cfg.dit_state_dim, cfg.dit_num_layers,
                  cfg.dit_num_heads, cfg.poly_num, cfg.gt_poly_num)
        if actual != (256, 256, 6, 8, 128, 128):
            raise ValueError(f"only L6/F256/H8/P128 is supported, got {actual}")
        if cfg.flow_ode_steps != 4 or cfg.flow_ode_solver != "ab2":
            raise ValueError("the mainline requires exactly four true AB2 evaluations")
        if cfg.down_ratio != 4 or cfg.flow_train_noise_scale != 1.:
            raise ValueError("the mainline requires stride 4 and unit Gaussian FM noise")
        if cfg.infer_noise_scale != 1. or cfg.infer_avg_samples != 1 or not cfg.flow_2d_s_conditioning:
            raise ValueError("latent scale 1, one sample, phase conditioning are required")
        self.num_points = 128
        self.ode_steps = ODE_STEPS
        self.denoiser = MainlineFlowDenoiser()
        self.routeb_box_jitter_config = geometry.resolve_routeb_box_jitter_config(cfg)
        self._disp_norm_enabled = bool(cfg.diffusion_disp_norm)
        if self._disp_norm_enabled:
            with Path(cfg.diffusion_disp_stats).open(encoding="utf-8") as handle:
                stats = json.load(handle)
            minimum = torch.tensor([stats["dx_min"], stats["dy_min"]]).float().view(1, 1, 2)
            maximum = torch.tensor([stats["dx_max"], stats["dy_max"]]).float().view(1, 1, 2)
            if not torch.isfinite(minimum).all() or not torch.isfinite(maximum).all() or not (maximum > minimum).all():
                raise ValueError("invalid displacement statistics")
        else:
            minimum = maximum = None
        self.register_buffer("_disp_min", minimum)
        self.register_buffer("_disp_max", maximum)

    def normalize_disp(self, displacement):
        if not self._disp_norm_enabled:
            return displacement
        return (displacement - self._disp_min.to(displacement)) * (
            2 / (self._disp_max - self._disp_min).to(displacement)) - 1

    def denormalize_disp(self, displacement):
        if not self._disp_norm_enabled:
            return displacement
        return (displacement + 1) * .5 * (self._disp_max - self._disp_min).to(displacement) + self._disp_min.to(displacement)

    @staticmethod
    def _align_ground_truth(initial, target):
        def signed_area(poly):
            following = torch.roll(poly, -1, 1)
            return (poly[..., 0] * following[..., 1] - poly[..., 1] * following[..., 0]).sum(1)
        reverse = (signed_area(initial) >= 0) != (signed_area(target) >= 0)
        target = torch.where(reverse[:, None, None], target.flip(1), target)
        shift = (initial[:, :1] - target).square().sum(-1).argmin(1)
        indices = (torch.arange(target.shape[1], device=target.device)[None] + shift[:, None]) % target.shape[1]
        return target.gather(1, indices[..., None].expand(-1, -1, 2))

    def prepare_training(self, output, feature, batch, augment=False):
        result = geometry.prepare_training(output, batch)
        if not augment:
            deployed = geometry.prepare_testing(output)
            if not torch.equal(deployed["py_ind"], result["py_ind"]):
                raise RuntimeError("GT prompt and supervised target ordering disagree")
            for key in ("i_it_py", "c_it_py", "i_it_4py", "c_it_4py"):
                result[key] = deployed[key]
        if augment:
            result, jitter = geometry.replace_training_init_with_gt_box_octagon(
                result, jitter_config=self.routeb_box_jitter_config,
                image_hw=feature.shape[-2:], return_jitter_stats=True)
            result.update(jitter)
        if result["i_it_py"].numel():
            result["i_gt_py"] = self._align_ground_truth(result["i_it_py"], result["i_gt_py"])
        return result

    def predict_velocity(self, feature, sampled, x, t, owner, phase):
        return self.denoiser(feature, sampled, x, t * 1000, py_ind=owner, s=phase)

    def action_mean(self, feature, current, owner, stage, latent=None):
        """The ONLY AB2 integrator; exactly four calls and one full action."""
        if stage not in (0, 1):
            raise ValueError("full2 has only update stage 0 and refinement stage 1")
        if current.numel() == 0:
            return torch.zeros_like(current)
        sampled = geometry.get_gcn_feature(feature, current, owner,
                                            int(feature.shape[-2]), int(feature.shape[-1]))
        x = torch.randn_like(current) if latent is None else latent
        if x.shape != current.shape:
            raise ValueError("latent shape must match the joint contour action")
        phase = current.new_full((current.shape[0],), STAGE_VALUES[stage])
        global_context = self.denoiser.global_compressor(feature)[owner]
        local_context = self.denoiser.local_proj(sampled.transpose(1, 2))
        for index, layer in enumerate(self.denoiser.dit_layers):
            layer.set_kv_cache(global_context if index % 2 == 0 else local_context)
        try:
            previous = None
            for index in range(ODE_STEPS):
                time = current.new_full((current.shape[0],), index / ODE_STEPS)
                velocity, _ = self.predict_velocity(feature, sampled, x, time, owner, phase)
                effective = velocity if previous is None else 1.5 * velocity - .5 * previous
                x = x + effective / ODE_STEPS
                previous = velocity
            return self.denormalize_disp(x)
        finally:
            for layer in self.denoiser.dit_layers:
                layer.clear_kv_cache()

    def rollout(self, feature, initial, owner, latents=None, sampler=None):
        if latents is not None and len(latents) != 2:
            raise ValueError("two stage latents are required")
        current = initial
        polygons, actions = [current], []
        for stage in range(2):
            mean = self.action_mean(feature, current, owner, stage,
                                    None if latents is None else latents[stage])
            action = mean if sampler is None else sampler(stage, current, mean)
            current = current + action
            actions.append(action)
            polygons.append(current)
        return {"polys": polygons, "actions": actions, "py": current, "disp": current - initial}

    def supervised_loss(self, feature, current, target, owner, phase):
        if not current.numel():
            raise ValueError("empty contour batch must be skipped")
        normalized_target = self.normalize_disp(target - current)
        time = torch.rand(current.shape[0], 1, 1, device=current.device)
        noise = torch.randn_like(normalized_target) * cfg.flow_train_noise_scale
        noisy = (1 - time) * noise + time * normalized_target
        sampled = geometry.get_gcn_feature(feature, current, owner,
                                            int(feature.shape[-2]), int(feature.shape[-1]))
        predicted, regularization = self.predict_velocity(feature, sampled, noisy, time.flatten(), owner, phase)
        data_loss = F.mse_loss(predicted, normalized_target - noise)
        prediction = current + self.denormalize_disp(noisy + (1 - time) * predicted)
        return {"diff_loss": data_loss + regularization,
                "diff_loss_data": data_loss.detach(),
                "diff_loss_moe_regularization": regularization.detach(),
                "py": prediction, "phase_mean": phase.mean().detach()}

    def forward(self, output, feature, batch):
        if not self.training:
            result = geometry.prepare_testing(output)
            with torch.no_grad():
                result.update(self.rollout(feature, result["i_it_py"], result["py_ind"]))
            return result
        if cfg.training_stage == "gdpo":
            raise RuntimeError("GDPO must use the policy objective, not supervised forward")
        online = cfg.training_stage == "online"
        result = self.prepare_training(output, feature, batch, augment=not online)
        initial, target, owner = result["i_it_py"], result["i_gt_py"], result["py_ind"]
        if initial.numel() == 0:
            result.update(diff_loss=feature.sum() * 0, empty_batch=True)
            return result
        if online:
            with torch.no_grad():
                generated = self.rollout(feature.detach(), initial, owner)
                phase = torch.randint(2, (initial.shape[0],), device=initial.device)
                current = torch.stack(generated["polys"][:2], 1)[
                    torch.arange(initial.shape[0], device=initial.device), phase].detach()
            kind = phase
            phase = phase.to(initial.dtype)
        else:
            current, phase, kind = mixed_states(initial, target, cfg.train_state_probabilities,
                                               cfg.refine_sigma_px, cfg.refine_modes)
        result.update(self.supervised_loss(feature, current, target, owner, phase))
        result["state_counts"] = torch.bincount(kind, minlength=2 if online else 3)
        for key, value in self.denoiser.cr_moe_diagnostics().items():
            result[f"moe_{key.replace('.', '_')}"] = value
        return result
