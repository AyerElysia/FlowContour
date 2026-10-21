"""Flow-matching contour evolution for the released FlowContour mainline.

There is deliberately one implementation in this package:

* Route-B bbox-octagon initialization;
* a six-layer Flow denoiser with one contour-level CR-MoE route path;
* continuous residual-progress sampling for supervised training;
* Adams--Bashforth-2 with four evaluations per outer stage; and
* a fixed two-stage deployment schedule.

The five-stage Fourier RL trajectory calls the same public solver with an
explicit training schedule. Alternative architecture switches, detail-token
branches, latent policies, geometry bridges, displacement gates and internal
RL implementations are not part of this module.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.config import cfg as global_cfg
from lib.utils.snake import snake_config
import lib.utils.snake.snake_gcn_utils as snake_gcn_utils

from .mainline_denoiser import MainlineDiTBlock, MainlineFlowDenoiser


_DEPLOYMENT_FRACTIONS = (0.6667, 1.0)
_ODE_STEPS = 4


class FlowMatchingEvolution(nn.Module):
    """Checkpoint-compatible Flow model shared by supervised, RL and inference."""

    def __init__(self) -> None:
        super().__init__()
        architecture = {
            "feature_dim": int(global_cfg.snake_feature_dim),
            "state_dim": int(global_cfg.dit_state_dim),
            "num_layers": int(global_cfg.dit_num_layers),
            "num_heads": int(global_cfg.dit_num_heads),
            "num_points": int(global_cfg.poly_num),
        }
        expected = {
            "feature_dim": 256,
            "state_dim": 256,
            "num_layers": 6,
            "num_heads": 8,
            "num_points": 128,
        }
        if architecture != expected:
            raise ValueError(
                "the released mainline has one checkpoint-compatible architecture: "
                f"{expected}; got {architecture}"
            )
        if int(global_cfg.flow_ode_steps) != _ODE_STEPS:
            raise ValueError("the mainline requires exactly four AB2 evaluations")
        if not bool(getattr(global_cfg, "flow_2d_s_conditioning", False)):
            raise ValueError("the mainline requires flow_2d_s_conditioning=true")
        if str(getattr(global_cfg, "flow_ode_solver", "ab2")).lower() != "ab2":
            raise ValueError("the mainline requires flow_ode_solver='ab2'")
        if int(getattr(global_cfg, "infer_avg_samples", 1)) != 1:
            raise ValueError("the mainline requires infer_avg_samples=1")

        deployment_fractions = tuple(
            float(value)
            for value in getattr(
                global_cfg, "iterative_fractions", _DEPLOYMENT_FRACTIONS
            )
        )
        if deployment_fractions != _DEPLOYMENT_FRACTIONS:
            raise ValueError(
                "deployment fractions must be exactly [0.6667, 1.0], got "
                f"{deployment_fractions}"
            )
        if int(getattr(global_cfg, "iterative_ode_steps", _ODE_STEPS)) != _ODE_STEPS:
            raise ValueError("deployment requires four AB2 evaluations per stage")

        init_source = str(
            getattr(global_cfg, "diffusion_init_source", "bbox_octagon")
        ).strip().lower()
        if init_source not in {
            "gt_box_octagon",
            "gt_bbox_octagon",
            "box_octagon",
            "bbox_octagon",
        }:
            raise ValueError("the mainline requires bbox-octagon initialization")

        self.num_points = 128
        self.ode_steps = _ODE_STEPS
        self.deployment_fractions = _DEPLOYMENT_FRACTIONS
        self.routeb_box_jitter_config = (
            snake_gcn_utils.resolve_routeb_box_jitter_config(global_cfg)
        )

        self.denoiser = MainlineFlowDenoiser(
            state_dim=256,
            feature_dim=256,
            num_layers=6,
            num_heads=8,
            num_points=128,
            dense_residual_hidden_dim=1024,
            use_s_conditioning=True,
        )

        # These explicit attributes are read by the RL launcher's fail-closed
        # deployment audit.  They are constants, not dormant feature switches.
        self._ode_solver = "ab2"
        self._ode_smooth_k = 0
        self._use_s_cond = True
        self._use_self_conditioning = False
        self._use_latent_policy = False
        self._geom_bridge = False
        self._resample_feat_at_xt = False
        self._use_disp_gate = False
        self._disp_gate_apply_inference = False
        self._infer_avg_samples = 1
        self._infer_noise_scale = float(
            getattr(global_cfg, "infer_noise_scale", 1.0)
        )
        self._flow_train_noise_scale = float(
            getattr(global_cfg, "flow_train_noise_scale", 1.0)
        )
        if self._infer_noise_scale != 1.0:
            raise ValueError("the deployed initial latent scale must be 1.0")

        self._train_progress_sigma = max(
            float(getattr(global_cfg, "train_progress_sigma", 0.05)), 1e-6
        )
        self._train_progress_uniform_prob = min(
            max(
                float(
                    getattr(global_cfg, "train_progress_uniform_prob", 0.15)
                ),
                0.0,
            ),
            1.0,
        )
        self._train_progress_centers = tuple(
            float(value)
            for value in getattr(
                global_cfg,
                "train_progress_centers",
                (0.0, 0.3333, 0.5, 0.80, 0.97),
            )
        )
        self._train_progress_weights = tuple(
            float(value)
            for value in getattr(
                global_cfg,
                "train_progress_weights",
                (0.2933, 0.1767, 0.27, 0.179, 0.081),
            )
        )
        self._validate_progress_distribution()

        self._disp_norm_enabled = bool(
            getattr(global_cfg, "diffusion_disp_norm", False)
        )
        self._register_displacement_statistics(
            str(getattr(global_cfg, "diffusion_disp_stats", ""))
        )

        print(
            "[FlowMatchingEvolution] mainline "
            "layers=6 dim=256 CR-MoE=blocks2/4/6,E4K2 "
            "output=dense deployment=2x4-AB2",
            flush=True,
        )

    def _validate_progress_distribution(self) -> None:
        if not self._train_progress_centers:
            raise ValueError("train_progress_centers must not be empty")
        if len(self._train_progress_centers) != len(self._train_progress_weights):
            raise ValueError(
                "train_progress_centers and train_progress_weights must match"
            )
        if any(
            not 0.0 <= value <= 0.999
            for value in self._train_progress_centers
        ):
            raise ValueError("training progress centers must lie in [0, 0.999]")
        if any(value < 0.0 for value in self._train_progress_weights):
            raise ValueError("training progress weights must be non-negative")
        if sum(self._train_progress_weights) <= 0.0:
            raise ValueError("training progress weights must have positive sum")

    def _register_displacement_statistics(self, stats_path: str) -> None:
        if not self._disp_norm_enabled:
            self.register_buffer("_disp_min", None)
            self.register_buffer("_disp_max", None)
            return
        path = Path(stats_path)
        if not stats_path or not path.is_file():
            raise FileNotFoundError(
                "diffusion_disp_norm is enabled but statistics are missing: "
                f"{stats_path!r}"
            )
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        disp_min = torch.tensor(
            [float(payload["dx_min"]), float(payload["dy_min"])],
            dtype=torch.float32,
        ).view(1, 1, 2)
        disp_max = torch.tensor(
            [float(payload["dx_max"]), float(payload["dy_max"])],
            dtype=torch.float32,
        ).view(1, 1, 2)
        if not torch.isfinite(disp_min).all() or not torch.isfinite(disp_max).all():
            raise ValueError("displacement statistics contain non-finite values")
        if not torch.all(disp_max > disp_min):
            raise ValueError("each displacement maximum must exceed its minimum")
        self.register_buffer("_disp_min", disp_min)
        self.register_buffer("_disp_max", disp_max)

    def _has_disp_stats(self) -> bool:
        return (
            self._disp_norm_enabled
            and isinstance(self._disp_min, torch.Tensor)
            and isinstance(self._disp_max, torch.Tensor)
        )

    def normalize_disp(self, displacement: torch.Tensor) -> torch.Tensor:
        if not self._has_disp_stats():
            return displacement
        minimum = self._disp_min.to(displacement)
        span = (self._disp_max - self._disp_min).clamp_min(1e-12).to(
            displacement
        )
        return (displacement - minimum) * (2.0 / span) - 1.0

    def denormalize_disp(self, normalized: torch.Tensor) -> torch.Tensor:
        if not self._has_disp_stats():
            return normalized
        minimum = self._disp_min.to(normalized)
        span = (self._disp_max - self._disp_min).clamp_min(1e-12).to(
            normalized
        )
        return (normalized + 1.0) * 0.5 * span + minimum

    def normalize_target_disp(
        self,
        displacement: torch.Tensor,
        contour_scale: torch.Tensor,
    ) -> torch.Tensor:
        del contour_scale
        return self.normalize_disp(displacement)

    def denormalize_pred_disp(
        self,
        displacement: torch.Tensor,
        contour_scale: torch.Tensor,
    ) -> torch.Tensor:
        del contour_scale
        return self.denormalize_disp(displacement)

    @staticmethod
    def compute_contour_scale(polygons: torch.Tensor) -> torch.Tensor:
        span_x = polygons[..., 0].amax(dim=1) - polygons[..., 0].amin(dim=1)
        span_y = polygons[..., 1].amax(dim=1) - polygons[..., 1].amin(dim=1)
        return torch.maximum(span_x, span_y).clamp_min(1.0).view(-1, 1, 1)

    @staticmethod
    def clamp_pred_disp(
        displacement: torch.Tensor,
        init_polygon: torch.Tensor,
    ) -> torch.Tensor:
        del init_polygon
        return displacement

    @staticmethod
    def fourier_smooth(displacement: torch.Tensor, modes: int) -> torch.Tensor:
        """Low-pass helper retained for the external Fourier action module."""

        modes = int(modes)
        if modes <= 0:
            return displacement
        point_count = int(displacement.shape[1])
        frequency = torch.fft.rfft(displacement, dim=1)
        keep = torch.zeros(
            frequency.shape[1], device=displacement.device, dtype=torch.bool
        )
        keep[: modes + 1] = True
        frequency[:, ~keep, :] = 0
        return torch.fft.irfft(frequency, n=point_count, dim=1)

    def sample_train_t(
        self,
        count: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        return torch.rand(count, 1, 1, device=device, dtype=dtype)

    def sample_train_x0(self, target: torch.Tensor) -> torch.Tensor:
        return torch.randn_like(target) * self._flow_train_noise_scale

    def _sample_training_progress(
        self,
        count: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Sample the released mixture-of-Gaussians plus uniform distribution."""

        centers = torch.tensor(
            self._train_progress_centers, device=device, dtype=dtype
        )
        weights = torch.tensor(
            self._train_progress_weights, device=device, dtype=dtype
        )
        weights = weights / weights.sum()
        cdf = weights.cumsum(0)
        draw = torch.rand(count, device=device, dtype=dtype)
        index = (draw.unsqueeze(1) >= cdf.unsqueeze(0)).sum(1)
        index = index.clamp_(0, centers.numel() - 1)
        gaussian = (
            centers[index]
            + torch.randn(count, device=device, dtype=dtype)
            * self._train_progress_sigma
        ).clamp_(0.0, 0.999)
        use_uniform = (
            torch.rand(count, device=device) < self._train_progress_uniform_prob
        )
        uniform = torch.rand(count, device=device, dtype=dtype) * 0.999
        return torch.where(use_uniform, uniform, gaussian).view(count, 1, 1)

    def _set_denoiser_kv_cache(
        self,
        cnn_feature: torch.Tensor,
        sampled_feat: torch.Tensor,
        py_ind: Optional[torch.Tensor],
    ) -> None:
        parameter_dtype = next(self.denoiser.parameters()).dtype
        image_feature = cnn_feature.to(parameter_dtype)
        point_feature = sampled_feat.to(parameter_dtype)
        if image_feature.ndim == 3:
            image_feature = image_feature.unsqueeze(0)
        global_context = self.denoiser.global_compressor(image_feature)
        if py_ind is not None:
            global_context = global_context[py_ind]
        elif global_context.shape[0] != point_feature.shape[0]:
            if global_context.shape[0] != 1:
                raise ValueError("image/contour batch mismatch")
            global_context = global_context.expand(
                point_feature.shape[0], -1, -1
            )
        local_context = self.denoiser.local_proj(
            point_feature.transpose(1, 2)
        )
        for index, layer in enumerate(self.denoiser.dit_layers):
            if not isinstance(layer, MainlineDiTBlock):
                raise TypeError("unexpected denoiser layer in the mainline")
            layer.set_kv_cache(
                global_context if index % 2 == 0 else local_context
            )

    def _clear_denoiser_kv_cache(self) -> None:
        for layer in self.denoiser.dit_layers:
            if isinstance(layer, MainlineDiTBlock):
                layer.clear_kv_cache()

    def predict_velocity(
        self,
        cnn_feature: torch.Tensor,
        i_it_py: torch.Tensor,
        c_it_py: torch.Tensor,
        sampled_feat: torch.Tensor,
        detail_feat: Optional[torch.Tensor],
        py_ind: torch.Tensor,
        x_t: torch.Tensor,
        t_continuous: torch.Tensor,
        contour_scale: Optional[torch.Tensor] = None,
        x_self_cond: Optional[torch.Tensor] = None,
        locate_context: Optional[dict[str, torch.Tensor]] = None,
        s: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate the single released Flow velocity field."""

        if detail_feat is not None:
            raise ValueError("detail features are not part of the mainline")
        if x_self_cond is not None:
            raise ValueError("self-conditioning is not part of the mainline")
        if locate_context:
            raise ValueError("LocateToken conditioning is not part of the mainline")
        if s is None:
            raise ValueError("stage progress s is required")
        adjacency = snake_gcn_utils.get_adj_ind(
            snake_config.adj_num,
            i_it_py.size(1),
            i_it_py.device,
        )
        return self.denoiser(
            cnn_feature,
            sampled_feat,
            x_t,
            t_continuous * 1000.0,
            adjacency,
            polys=i_it_py,
            py_ind=py_ind,
            contour_scale=contour_scale,
            detail_feat=None,
            x_self_cond=None,
            s=s,
        )

    def prepare_sampling_context(
        self,
        cnn_feature: torch.Tensor,
        i_it_py: torch.Tensor,
        py_ind: torch.Tensor,
        batch: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        del batch
        height, width = cnn_feature.shape[-2:]
        sampled = snake_gcn_utils.get_gcn_feature(
            cnn_feature,
            i_it_py,
            py_ind,
            int(height),
            int(width),
        )
        return {
            "sampled_feat": sampled,
            "detail_feat": None,
            "contour_scale": self.compute_contour_scale(i_it_py),
            "h": int(height),
            "w": int(width),
        }

    def _integrate_ab2(
        self,
        cnn_feature: torch.Tensor,
        i_it_py: torch.Tensor,
        c_it_py: torch.Tensor,
        py_ind: torch.Tensor,
        sampled_feat: torch.Tensor,
        contour_scale: torch.Tensor,
        latent: torch.Tensor,
        steps: int,
        stage_progress: float,
    ) -> torch.Tensor:
        steps = int(steps)
        if steps != _ODE_STEPS:
            raise ValueError("the mainline solver requires exactly four AB2 steps")
        x_t = latent
        previous_velocity: Optional[torch.Tensor] = None
        dt = 1.0 / float(steps)
        s_tensor = torch.full(
            (x_t.size(0),),
            float(stage_progress),
            device=x_t.device,
            dtype=x_t.dtype,
        )
        for index in range(steps):
            t_tensor = torch.full(
                (x_t.size(0),),
                float(index) * dt,
                device=x_t.device,
                dtype=x_t.dtype,
            )
            velocity, _ = self.predict_velocity(
                cnn_feature,
                i_it_py,
                c_it_py,
                sampled_feat,
                None,
                py_ind,
                x_t,
                t_tensor,
                contour_scale=contour_scale.view(-1).to(x_t),
                s=s_tensor,
            )
            if index == 0:
                effective_velocity = velocity
            else:
                if previous_velocity is None:
                    raise RuntimeError("AB2 history is missing")
                effective_velocity = (
                    1.5 * velocity - 0.5 * previous_velocity
                )
            x_t = x_t + dt * effective_velocity
            previous_velocity = velocity.detach()
        return x_t

    def _sample_disp_from_context(
        self,
        cnn_feature: torch.Tensor,
        i_it_py: torch.Tensor,
        c_it_py: torch.Tensor,
        py_ind: torch.Tensor,
        context: dict[str, Any],
        steps: int,
        stage_progress: float,
        latent: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if latent is None:
            latent = torch.randn_like(i_it_py) * self._infer_noise_scale

        cache_enabled = os.environ.get(
            "FLOW_DISABLE_KV_CACHE", ""
        ).strip().lower() not in {"1", "true", "yes"}
        if cache_enabled:
            self._set_denoiser_kv_cache(
                cnn_feature, context["sampled_feat"], py_ind
            )
        try:
            normalized = self._integrate_ab2(
                cnn_feature,
                i_it_py,
                c_it_py,
                py_ind,
                context["sampled_feat"],
                context["contour_scale"],
                latent,
                steps,
                stage_progress,
            )
        finally:
            if cache_enabled:
                self._clear_denoiser_kv_cache()
        displacement = self.denormalize_pred_disp(
            normalized, context["contour_scale"]
        )
        return self.clamp_pred_disp(displacement, i_it_py)

    def sample_disp(
        self,
        cnn_feature: torch.Tensor,
        i_it_py: torch.Tensor,
        c_it_py: torch.Tensor,
        py_ind: torch.Tensor,
        steps: Optional[int] = None,
        batch: Optional[dict[str, Any]] = None,
    ) -> torch.Tensor:
        """One full residual stage; kept for a small public compatibility surface."""

        del batch
        if i_it_py.numel() == 0:
            return torch.zeros_like(i_it_py)
        context = self.prepare_sampling_context(
            cnn_feature, i_it_py, py_ind
        )
        return self._sample_disp_from_context(
            cnn_feature,
            i_it_py,
            c_it_py,
            py_ind,
            context,
            self.ode_steps if steps is None else int(steps),
            0.0,
        )

    def sample_disp_iterative(
        self,
        cnn_feature: torch.Tensor,
        i_it_py: torch.Tensor,
        c_it_py: torch.Tensor,
        py_ind: torch.Tensor,
        num_iter_steps: int = 2,
        fractions: Optional[Sequence[float]] = None,
        ode_steps: Optional[int] = None,
        batch: Optional[dict[str, Any]] = None,
    ) -> torch.Tensor:
        """Run deployment or RL outer stages through the same AB2 solver."""

        del batch
        schedule = (
            self.deployment_fractions
            if fractions is None
            else tuple(float(value) for value in fractions)
        )
        if int(num_iter_steps) != len(schedule):
            raise ValueError(
                "num_iter_steps must equal the number of residual fractions"
            )
        if any(not 0.0 < value <= 1.0 for value in schedule):
            raise ValueError("every residual fraction must lie in (0, 1]")
        steps = self.ode_steps if ode_steps is None else int(ode_steps)
        if steps != _ODE_STEPS:
            raise ValueError("every outer stage requires four AB2 steps")
        if i_it_py.numel() == 0:
            return torch.zeros_like(i_it_py)

        current = i_it_py.clone()
        total = torch.zeros_like(current)
        progress = 0.0
        for fraction in schedule:
            canonical = snake_gcn_utils.img_poly_to_can_poly(current)
            context = self.prepare_sampling_context(
                cnn_feature, current, py_ind
            )
            stage_displacement = self._sample_disp_from_context(
                cnn_feature,
                current,
                canonical,
                py_ind,
                context,
                steps,
                progress,
            )
            applied = stage_displacement * float(fraction)
            current = current + applied
            total = total + applied
            progress += (1.0 - progress) * float(fraction)
        return total

    @staticmethod
    def _align_ground_truth(
        initial: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Match orientation and cyclic start without changing target geometry."""

        def signed_area(polygon: torch.Tensor) -> torch.Tensor:
            x, y = polygon[..., 0], polygon[..., 1]
            next_x = torch.roll(x, -1, 1)
            next_y = torch.roll(y, -1, 1)
            return 0.5 * torch.sum(x * next_y - next_x * y, dim=1)

        mismatch = (signed_area(initial) >= 0) ^ (signed_area(target) >= 0)
        if mismatch.any():
            target = torch.where(
                mismatch.view(-1, 1, 1),
                torch.flip(target, dims=[1]),
                target,
            )
        distance = (initial[:, :1] - target).square().sum(-1)
        shift = distance.argmin(dim=1)
        point_count = int(target.size(1))
        index = (
            torch.arange(point_count, device=target.device).unsqueeze(0)
            + shift.unsqueeze(1)
        ) % point_count
        return torch.gather(
            target,
            1,
            index.unsqueeze(-1).expand(-1, -1, target.size(2)),
        )

    def _supervised_forward(
        self,
        output: dict[str, Any],
        cnn_feature: torch.Tensor,
        batch: dict[str, Any],
    ) -> dict[str, Any]:
        train_dict = snake_gcn_utils.prepare_training(output, batch)
        train_dict, jitter_stats = (
            snake_gcn_utils.replace_training_init_with_gt_box_octagon(
                train_dict,
                jitter_config=self.routeb_box_jitter_config,
                image_hw=cnn_feature.shape[-2:],
                return_jitter_stats=True,
            )
        )
        result: dict[str, Any] = dict(train_dict)
        for severity_index, severity_count in enumerate(
            jitter_stats.pop(
                "routeb_box_jitter_severity_counts",
                cnn_feature.new_zeros(0),
            )
        ):
            jitter_stats[
                f"routeb_box_jitter_severity_{severity_index}_count"
            ] = severity_count
        result.update(jitter_stats)
        result["diffusion_init_source"] = "bbox_octagon"

        initial = train_dict["i_it_py"]
        if initial.numel() == 0:
            zero = cnn_feature.sum() * 0.0
            result.update(
                {
                    "diff_loss": zero,
                    "diff_lossA1": zero,
                    "diff_loss_total": zero,
                    "diff_loss1": zero,
                    "py_pred": [initial],
                }
            )
            return result

        target = self._align_ground_truth(
            initial, train_dict["i_gt_py"].clone()
        )
        full_displacement = target - initial
        count = int(initial.size(0))
        progress = self._sample_training_progress(
            count, initial.device, initial.dtype
        )
        current = initial + full_displacement * progress
        residual = full_displacement * (1.0 - progress)
        canonical = snake_gcn_utils.img_poly_to_can_poly(current)
        contour_scale = self.compute_contour_scale(current)
        normalized_target = self.normalize_target_disp(
            residual, contour_scale
        )

        t = self.sample_train_t(
            count, initial.device, normalized_target.dtype
        )
        x0 = self.sample_train_x0(normalized_target)
        x_t = (1.0 - t) * x0 + t * normalized_target
        target_velocity = normalized_target - x0
        context = self.prepare_sampling_context(
            cnn_feature, current, train_dict["py_ind"]
        )
        predicted_velocity, regularization = self.predict_velocity(
            cnn_feature,
            current,
            canonical,
            context["sampled_feat"],
            None,
            train_dict["py_ind"],
            x_t,
            t.view(-1),
            contour_scale=contour_scale.view(-1),
            s=progress.view(-1),
        )
        data_loss = F.mse_loss(
            predicted_velocity, target_velocity, reduction="mean"
        )
        loss = data_loss + regularization

        endpoint = x_t + (1.0 - t) * predicted_velocity
        predicted_displacement = self.denormalize_pred_disp(
            endpoint, contour_scale
        )
        prediction = current + predicted_displacement
        zero = loss.detach() * 0.0
        result.update(
            {
                "diff_loss": loss,
                "diff_loss_data": data_loss.detach(),
                "diff_loss_moe_regularization": regularization.detach(),
                "diff_lossA1": loss,
                "diff_loss_total": loss,
                "diff_loss1": loss,
                "diff_loss_chamfer": zero,
                "diff_loss_gate": zero,
                "point_mask": train_dict.get("point_mask"),
                "py_pred": [current],
                "py": prediction,
                "pred_contours": prediction,
                "pred_disp": predicted_displacement,
                "v_pred": predicted_velocity.mean(),
                "train_progress_mean": progress.mean().detach(),
            }
        )
        for key, value in self.denoiser.moe_diagnostics().items():
            result[f"moe_{key.replace('.', '_')}"] = value
        return result

    def _inference_forward(
        self,
        output: dict[str, Any],
        cnn_feature: torch.Tensor,
    ) -> dict[str, Any]:
        result = snake_gcn_utils.prepare_testing(output)
        initial = result["i_it_py"]
        if initial.numel() == 0:
            displacement = torch.zeros_like(initial)
        else:
            displacement = self.sample_disp_iterative(
                cnn_feature,
                initial,
                result["c_it_py"],
                result["py_ind"],
                num_iter_steps=len(self.deployment_fractions),
                fractions=self.deployment_fractions,
                ode_steps=self.ode_steps,
            )
        result.update(
            {
                "disp": displacement,
                "py": initial + displacement,
            }
        )
        return result

    def forward(
        self,
        output: dict[str, Any],
        cnn_feature: torch.Tensor,
        batch: dict[str, Any],
    ) -> dict[str, Any]:
        if self.training:
            return self._supervised_forward(output, cnn_feature, batch)
        with torch.no_grad():
            return self._inference_forward(output, cnn_feature)


__all__ = ("FlowMatchingEvolution",)
