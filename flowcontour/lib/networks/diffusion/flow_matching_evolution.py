"""YOLO RAOS Flow evolution with the effective original training semantics.

Displacements are divided by each current contour's maximum bounding-box span
before flow matching, then multiplied by that same span after ODE integration.
This normalization is independent of the unused dataset-wide min/max scaling.
The six DiT blocks and contour-routed experts remain the paper architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.config import cfg
from lib.protocol import ODE_STEPS, STAGE_VALUES
from lib.train.states import mixed_states
from lib.utils.snake import snake_gcn_utils as geometry
from .mainline_denoiser import MainlineFlowDenoiser
from .ordered_geometry import correlated_noise, resample_closed


class FlowMatchingEvolution(nn.Module):
    def __init__(self):
        super().__init__()
        actual = (cfg.snake_feature_dim, cfg.dit_state_dim, cfg.dit_num_layers,
                  cfg.dit_num_heads, cfg.poly_num, cfg.gt_poly_num)
        if actual != (256, 256, 6, 8, 128, 128):
            raise ValueError(f"only L6/F256/H8/P128 is supported, got {actual}")
        if cfg.flow_ode_steps != 4 or cfg.flow_ode_solver != "ab2":
            raise ValueError("RAOS requires exactly four AB2 evaluations per outer stage")
        if cfg.down_ratio != 4 or cfg.flow_train_noise_scale != 1.:
            raise ValueError("RAOS requires stride 4 and unit marginal FM noise")
        if cfg.infer_noise_scale != 1. or cfg.infer_avg_samples != 1 or not cfg.flow_2d_s_conditioning:
            raise ValueError("unit latent, one sample and cumulative stage conditioning required")
        if not bool(getattr(cfg, "v3_7_use_contour_norm", True)):
            raise ValueError("the effective original RAOS training requires contour normalization")
        if bool(cfg.diffusion_disp_norm):
            raise ValueError("dataset min/max normalization cannot replace contour-span normalization")
        self.num_points = 128
        self.ode_steps = ODE_STEPS
        self.denoiser = MainlineFlowDenoiser()
        self.class_embedding = nn.Embedding(int(cfg.anatomical_class_count), 256)
        nn.init.normal_(self.class_embedding.weight, mean=0., std=.02)
        self.routeb_box_jitter_config = geometry.resolve_routeb_box_jitter_config(cfg)

    @staticmethod
    def compute_contour_scale(poly):
        span = poly.amax(1) - poly.amin(1)
        return span.amax(-1).clamp_min(1.).view(-1, 1, 1)

    @staticmethod
    def normalize_target_disp(displacement, contour_scale):
        return displacement / contour_scale.to(displacement)

    @staticmethod
    def denormalize_pred_disp(displacement, contour_scale):
        return displacement * contour_scale.to(displacement)

    @staticmethod
    def _align_ground_truth(initial, target):
        """Match winding and minimize total point MSE over every cyclic shift."""
        def area(poly):
            following = torch.roll(poly, -1, 1)
            return (poly[..., 0] * following[..., 1] - poly[..., 1] * following[..., 0]).sum(1)
        reverse = (area(initial) >= 0) != (area(target) >= 0)
        target = torch.where(reverse[:, None, None], target.flip(1), target)
        count = target.shape[1]
        base = torch.arange(count, device=target.device)
        indices = (base[None, :] + base[:, None]) % count
        candidates = target[:, indices, :]
        shift = (candidates - initial[:, None]).square().sum((-1, -2)).argmin(1)
        return target.gather(1, ((base[None] + shift[:, None]) % count)[..., None].expand(-1, -1, 2))

    def _validate_classes(self, class_ids, count, device):
        if class_ids is None:
            raise ValueError("explicit zero-based class_ids are required for RAOS Flow")
        ids = class_ids.to(device=device, dtype=torch.long).view(-1)
        if ids.numel() != count:
            raise ValueError(f"class/contour count differs: {ids.numel()} vs {count}")
        if bool(((ids < 0) | (ids >= self.class_embedding.num_embeddings)).any()):
            raise ValueError("class_ids must be in [0, anatomical_class_count)")
        return ids

    def prepare_training(self, output, feature, batch, augment=False):
        result = geometry.prepare_training(output, batch)
        result["class_ids"] = self._validate_classes(
            batch["ct_cls"][batch["ct_01"].bool()] - int(cfg.gt_detection_class_offset),
            result["i_it_py"].shape[0], feature.device)
        if not augment:
            deployed = geometry.prepare_testing(output)
            if not torch.equal(deployed["py_ind"], result["py_ind"]):
                raise RuntimeError("GT prompt and supervised target ordering disagree")
            for key in ("i_it_py", "c_it_py", "i_it_4py", "c_it_4py"):
                result[key] = deployed[key]
        else:
            result, jitter = geometry.replace_training_init_with_gt_box_octagon(
                result, jitter_config=self.routeb_box_jitter_config,
                image_hw=feature.shape[-2:], return_jitter_stats=True)
            result.update(jitter)
        if result["i_it_py"].numel():
            result["i_gt_py"] = self._align_ground_truth(result["i_it_py"], result["i_gt_py"])
        return result

    @staticmethod
    def sample_detail_features(feature, current, owner, sampled):
        """Original six YOLO feature differences at 0.5/1.0 feature pixels."""
        h, w = feature.shape[-2:]
        tangent = torch.roll(current, -1, 1) - torch.roll(current, 1, 1)
        tangent = tangent / tangent.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        normal = torch.stack((-tangent[..., 1], tangent[..., 0]), -1)
        def sample(offset):
            return geometry.get_gcn_feature(feature, current + offset, owner, h, w)
        n1p, n1m = sample(normal * .5), sample(-normal * .5)
        n2p, n2m = sample(normal), sample(-normal)
        t1p, t1m = sample(tangent * .5), sample(-tangent * .5)
        t2p, t2m = sample(tangent), sample(-tangent)
        return torch.cat((n1p - n1m, n2p - n2m, .5 * (n1p + n1m) - sampled,
                          t1p - t1m, t2p - t2m, .5 * (t1p + t1m) - sampled), 1)

    def visual_context(self, feature, current, owner, class_ids):
        ids = self._validate_classes(class_ids, current.shape[0], current.device)
        sampled = geometry.get_gcn_feature(feature, current, owner, *feature.shape[-2:])
        detail = self.sample_detail_features(feature, current, owner, sampled)
        sampled = sampled + self.class_embedding(ids).to(sampled)[..., None]
        return sampled, detail

    def predict_velocity(self, feature, sampled, x, t, owner, phase, detail_feat):
        return self.denoiser(feature, sampled, x, t * 1000, py_ind=owner,
                             s=phase, detail_feat=detail_feat)

    def action_mean(self, feature, current, owner, stage, latent=None, class_ids=None):
        """Integrate normalized joint displacement, then restore current-contour scale."""
        if stage not in (0, 1):
            raise ValueError("only two outer stages are supported")
        if not current.numel():
            return torch.zeros_like(current)
        sampled, detail = self.visual_context(feature, current, owner, class_ids)
        scale = self.compute_contour_scale(current)
        x = correlated_noise(current) if latent is None else latent
        if x.shape != current.shape:
            raise ValueError("latent shape must match joint contour action")
        phase = current.new_full((current.shape[0],), STAGE_VALUES[stage])
        # Disable cache in differentiable rollout: its graph must include visual
        # projection and detail branches on every ODE evaluation.
        use_cache = not torch.is_grad_enabled()
        if use_cache:
            global_context = self.denoiser.global_compressor(feature)[owner]
            local_context = (self.denoiser.local_proj(sampled.transpose(1, 2))
                             + self.denoiser.detail_local_proj(detail.transpose(1, 2)))
            for index, layer in enumerate(self.denoiser.dit_layers):
                layer.set_kv_cache(global_context if index % 2 == 0 else local_context)
        try:
            previous = None
            for index in range(ODE_STEPS):
                time = current.new_full((current.shape[0],), index / ODE_STEPS)
                velocity, _ = self.predict_velocity(feature, sampled, x, time, owner, phase, detail)
                effective = velocity if previous is None else 1.5 * velocity - .5 * previous
                x = x + effective / ODE_STEPS
                previous = velocity
            return self.denormalize_pred_disp(x, scale)
        finally:
            for layer in self.denoiser.dit_layers:
                layer.clear_kv_cache()

    def rollout(self, feature, initial, owner, latents=None, sampler=None, class_ids=None):
        if latents is not None and len(latents) != 2:
            raise ValueError("two stage latents are required")
        current = initial
        polygons, actions = [current], []
        for stage in range(2):
            mean = self.action_mean(feature, current, owner, stage,
                                    None if latents is None else latents[stage], class_ids)
            action = mean if sampler is None else sampler(stage, current, mean)
            current = current + action * (0.6667, 1.)[stage]
            if stage == 0:
                current = resample_closed(current)
            actions.append(action)
            polygons.append(current)
        return {"polys": polygons, "actions": actions, "py": current, "disp": current - initial}

    @staticmethod
    def compute_curvature_weights(poly):
        smoothed = F.avg_pool1d(F.pad(poly.transpose(1, 2), (2, 2), mode="circular"),
                                kernel_size=5, stride=1).transpose(1, 2)
        left, right = smoothed - torch.roll(smoothed, 2, 1), torch.roll(smoothed, -2, 1) - smoothed
        cosine = F.cosine_similarity(left, right, dim=-1, eps=1e-6)
        angle = torch.acos(cosine.clamp(-1 + 1e-6, 1 - 1e-6))
        weights = 1 + 2 * (angle / 1.0471975512).clamp(max=1)
        return weights / weights.mean(1, keepdim=True)

    @staticmethod
    def soft_chamfer(prediction, target, tau=.05):
        distance = torch.cdist(prediction.float(), target.float())
        forward = (distance * torch.softmax(-distance / tau, dim=2)).sum(2)
        backward = (distance * torch.softmax(-distance / tau, dim=1)).sum(1)
        return .5 * (forward.mean() + backward.mean())

    def supervised_loss(self, feature, current, target, owner, phase, class_ids=None):
        if not current.numel():
            raise ValueError("empty contour batch must be skipped")
        scale = self.compute_contour_scale(current)
        normalized_target = self.normalize_target_disp(target - current, scale)
        time = torch.rand(current.shape[0], 1, 1, device=current.device)
        noise = correlated_noise(normalized_target)
        noisy = (1 - time) * noise + time * normalized_target
        sampled, detail = self.visual_context(feature, current, owner, class_ids)
        predicted, regularization = self.predict_velocity(
            feature, sampled, noisy, time.flatten(), owner, phase, detail)
        weights = self.compute_curvature_weights(target).to(predicted)
        data_loss = ((predicted - (normalized_target - noise)).square().mean(-1) * weights).mean()
        endpoint = noisy + (1 - time) * predicted
        endpoint_loss = F.mse_loss(endpoint, normalized_target)
        prediction = current + self.denormalize_pred_disp(endpoint, scale)
        chamfer = self.soft_chamfer(prediction / scale, target / scale)
        # Dense raw-mask boundary and ordered geometry supervision are added by
        # the trainer, which owns those dense targets and the warmup schedule.
        full_loss = data_loss + endpoint_loss + chamfer + regularization
        return {"diff_loss": full_loss, "diff_loss_data": data_loss.detach(),
                "diff_loss_endpoint": endpoint_loss.detach(), "diff_loss_chamfer": chamfer.detach(),
                "diff_loss_moe_regularization": regularization.detach(),
                "py": prediction, "phase_mean": phase.mean().detach()}

    def forward(self, output, feature, batch):
        if not self.training:
            result = geometry.prepare_testing(output)
            detection = output["detection"]
            result["class_ids"] = self._validate_classes(
                detection[..., 5][detection[..., 4] > 1e-4],
                result["i_it_py"].shape[0], feature.device)
            with torch.no_grad():
                result.update(self.rollout(feature, result["i_it_py"], result["py_ind"],
                                           class_ids=result["class_ids"]))
            return result
        if cfg.training_stage == "gdpo":
            raise RuntimeError("GDPO must use the policy objective")
        result = self.prepare_training(output, feature, batch, augment=True)
        initial, target, owner = result["i_it_py"], result["i_gt_py"], result["py_ind"]
        if initial.numel() == 0:
            result.update(diff_loss=feature.sum() * 0, empty_batch=True)
            return result
        current, phase, kind = mixed_states(initial, target, cfg.train_state_probabilities,
                                           cfg.refine_sigma_px, cfg.refine_modes)
        result.update(self.supervised_loss(feature, current, target, owner, phase, result["class_ids"]))
        result["state_counts"] = torch.bincount(kind, minlength=3)
        for key, value in self.denoiser.cr_moe_diagnostics().items():
            result[f"moe_{key.replace('.', '_')}"] = value
        return result
