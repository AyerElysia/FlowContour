"""MoonViT-cache + Flow network used by the official VerSe mainline.

The checkpoint-facing attribute names are preserved:

``locate_feat_replacer``
    Projects cached MoonViT layer-18 features to the 256-channel contour grid.

``gcn``
    The inherited Flow-matching contour evolution module.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.config import cfg
from lib.utils import data_utils


class MoonViTFeatureReplacer(nn.Module):
    """Project cached MoonViT tokens and align them to the contour grid.

    The layer order intentionally matches the signed source checkpoint's
    ``LocateFeatReplacer`` so official checkpoints load with ``strict=True``.
    """

    def __init__(
        self,
        in_channels: int = 1152,
        hidden_channels: int = 512,
        out_channels: int = 256,
        upscale: int = 2,
        input_layers: int = 1,
    ) -> None:
        super().__init__()
        in_channels = int(in_channels)
        hidden_channels = int(hidden_channels)
        out_channels = int(out_channels)
        upscale = int(upscale)
        input_layers = int(input_layers)
        if (in_channels, hidden_channels, out_channels, upscale, input_layers) != (1152, 512, 256, 2, 1):
            raise ValueError("only the signed single-layer MoonViT replacer is supported")
        self.upscale = upscale

        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False),
            nn.GroupNorm(16, hidden_channels),
            nn.GELU(),
            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(16, hidden_channels),
            nn.GELU(),
        ]
        layers.extend([nn.PixelShuffle(2), nn.Conv2d(128, 256, 3, padding=1)])
        self.proj = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.proj(features)


class Network(nn.Module):
    """Official detector-free MoonViT-cache + Flow model."""

    def __init__(
        self,
        down_ratio: int = 4,
    ) -> None:
        super().__init__()
        self.detector_backend = str(cfg.detector_backend).strip().lower()
        if self.detector_backend != "flow_box_only":
            raise ValueError(
                "The mainline package only supports detector_backend='flow_box_only'; "
                f"got {self.detector_backend!r}."
            )
        if not bool(cfg.use_gt_det) or bool(cfg.use_gt_det_train_only):
            raise ValueError("the oracle mainline requires GT prompts for both training and inference")
        if not bool(cfg.locate_feat_replace) or bool(cfg.locate_feat_inject):
            raise ValueError(
                "The mainline requires locate_feat_replace=true and "
                "locate_feat_inject=false."
            )
        if not bool(cfg.use_diffusion_evolution) or not bool(cfg.use_flow_matching):
            raise ValueError("The mainline requires Flow-matching evolution.")

        self.down_ratio = float(down_ratio)
        self.detector_num_classes = int(cfg.anatomical_class_count)
        if self.detector_num_classes != 26:
            raise ValueError("the VerSe mainline requires 26 anatomical classes")
        self.locate_feat_replace = True
        self.locate_feat_replace_upscale = int(cfg.locate_feat_replace_upscale)
        self.locate_feat_replacer = MoonViTFeatureReplacer(
            in_channels=int(cfg.locate_feat_dim),
            hidden_channels=int(cfg.locate_feat_replace_hidden_dim),
            out_channels=int(cfg.locate_feat_replace_out_channels),
            upscale=self.locate_feat_replace_upscale,
            input_layers=int(cfg.locate_feat_input_layers),
        )

        from lib.networks.diffusion import FlowMatchingEvolution

        self.gcn = FlowMatchingEvolution()

        replacer_parameters = sum(
            parameter.numel() for parameter in self.locate_feat_replacer.parameters()
        )
        print(
            "[Mainline] backend=flow_box_only MoonViT=layer_18 "
            f"replacer={replacer_parameters:,} Flow={sum(p.numel() for p in self.gcn.parameters()):,}",
            flush=True,
        )

    @staticmethod
    def _batch_meta_tensor(
        batch: dict | None,
        key: str,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        if batch is None or "meta" not in batch or key not in batch["meta"]:
            return None
        value = batch["meta"][key]
        return torch.as_tensor(value, device=device, dtype=dtype)

    @staticmethod
    def _batch_tensor(
        batch: dict | None,
        key: str,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        if batch is None or key not in batch:
            return None
        return torch.as_tensor(batch[key], device=device, dtype=dtype)

    def _build_feature_grid(
        self,
        batch: dict,
        target_h: int,
        target_w: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        inv_trans = self._batch_meta_tensor(batch, "inv_trans_input", device, dtype)
        orig_hw = self._batch_meta_tensor(batch, "orig_hw", device, dtype)
        flipped = self._batch_meta_tensor(batch, "flipped", device, dtype)
        scale = self._batch_tensor(batch, "locate_feat_scale", device, dtype)
        grid_hw = self._batch_tensor(batch, "locate_feat_grid_hw", device, dtype)
        patch_size = self._batch_tensor(
            batch, "locate_feat_patch_size", device, dtype
        )
        padding = self._batch_tensor(batch, "locate_feat_pad", device, dtype)
        if inv_trans is None or orig_hw is None or scale is None or grid_hw is None:
            raise KeyError(
                "MoonViT features require meta.inv_trans_input, meta.orig_hw, "
                "locate_feat_scale and locate_feat_grid_hw."
            )
        if inv_trans.ndim != 3 or tuple(inv_trans.shape[1:]) != (2, 3):
            raise ValueError(
                f"meta.inv_trans_input must be [B,2,3], got {tuple(inv_trans.shape)}"
            )

        batch_size = int(inv_trans.shape[0])
        if patch_size is None:
            patch_size = torch.full(
                (batch_size, 1), 14.0, device=device, dtype=dtype
            )
        if padding is None:
            padding = torch.zeros((batch_size, 4), device=device, dtype=dtype)
        if scale.ndim == 1:
            scale = scale[:, None]
        if patch_size.ndim == 1:
            patch_size = patch_size[:, None]
        if flipped is None:
            flipped = torch.zeros((batch_size, 1), device=device, dtype=dtype)
        if flipped.ndim == 1:
            flipped = flipped[:, None]

        ys = (
            torch.arange(target_h, device=device, dtype=dtype) + 0.5
        ) * self.down_ratio - 0.5
        xs = (
            torch.arange(target_w, device=device, dtype=dtype) + 0.5
        ) * self.down_ratio - 0.5
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        xy1 = torch.stack((xx, yy, torch.ones_like(xx)), dim=0).view(3, -1)
        source_xy = torch.bmm(
            inv_trans, xy1.unsqueeze(0).expand(batch_size, -1, -1)
        ).view(batch_size, 2, target_h, target_w)
        source_x, source_y = source_xy[:, 0], source_xy[:, 1]
        original_w = orig_hw[:, 1].view(batch_size, 1, 1)
        source_x = torch.where(
            flipped.view(batch_size, 1, 1) > 0.5,
            original_w - source_x - 1.0,
            source_x,
        )

        source_scale = float(self.locate_feat_replace_upscale)
        effective_patch = (
            patch_size.view(batch_size, 1, 1) / source_scale
        ).clamp(min=1e-6)
        feature_x = (
            source_x * scale.view(batch_size, 1, 1)
            + padding[:, 0].view(batch_size, 1, 1)
        ) / effective_patch - 0.5
        feature_y = (
            source_y * scale.view(batch_size, 1, 1)
            + padding[:, 1].view(batch_size, 1, 1)
        ) / effective_patch - 0.5
        grid_h = (grid_hw[:, 0].view(batch_size, 1, 1) * source_scale).clamp(
            min=1.0
        )
        grid_w = (grid_hw[:, 1].view(batch_size, 1, 1) * source_scale).clamp(
            min=1.0
        )
        norm_x = torch.where(
            grid_w > 1.0,
            feature_x / (grid_w - 1.0) * 2.0 - 1.0,
            torch.zeros_like(feature_x),
        )
        norm_y = torch.where(
            grid_h > 1.0,
            feature_y / (grid_h - 1.0) * 2.0 - 1.0,
            torch.zeros_like(feature_y),
        )
        return torch.stack((norm_x, norm_y), dim=-1)

    def _replace_features(
        self,
        placeholder: torch.Tensor,
        batch: dict,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if "locate_feat" not in batch:
            raise KeyError("batch is missing cached MoonViT field 'locate_feat'")
        features = batch["locate_feat"]
        device, dtype = placeholder.device, placeholder.dtype
        batch_size = int(placeholder.shape[0])
        grid = self._build_feature_grid(
            batch,
            int(placeholder.shape[-2]),
            int(placeholder.shape[-1]),
            device,
            dtype,
        )
        padding_mode = str(cfg.locate_feat_resample_padding_mode).lower()
        if padding_mode not in {"zeros", "border", "reflection"}:
            raise ValueError(f"invalid locate feature padding mode: {padding_mode}")

        if isinstance(features, (list, tuple)):
            if len(features) != batch_size:
                raise ValueError("locate_feat list length must match batch size")
            outputs = []
            absmax = placeholder.new_tensor(0.0)
            for index, feature in enumerate(features):
                feature = torch.as_tensor(feature, device=device, dtype=dtype)
                if feature.ndim == 3:
                    feature = feature.unsqueeze(0)
                replaced = self.locate_feat_replacer(feature)
                replaced = F.grid_sample(
                    replaced,
                    grid[index : index + 1],
                    mode="bilinear",
                    padding_mode=padding_mode,
                    align_corners=True,
                )
                outputs.append(replaced)
                absmax = torch.maximum(absmax, replaced.detach().abs().max())
            return torch.cat(outputs, dim=0), {
                "locate_feat_replace_absmax": absmax
            }

        features = torch.as_tensor(features, device=device, dtype=dtype)
        replaced = self.locate_feat_replacer(features)
        replaced = F.grid_sample(
            replaced,
            grid,
            mode="bilinear",
            padding_mode=padding_mode,
            align_corners=True,
        )
        return replaced, {
            "locate_feat_replace_absmax": replaced.detach().abs().max()
        }

    def apply_locate_feature_replacement(
        self,
        placeholder: torch.Tensor,
        batch: dict,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Expose the checkpoint-compatible MoonViT replacement interface.

        RL builds the frozen visual context without running the full detector
        forward pass.  Keeping that code on this public method also gives the
        supervised and RL paths one implementation of the geometry mapping.
        """

        return self._replace_features(placeholder, batch)

    def _use_gt_detection(self, output: dict, batch: dict) -> None:
        valid = batch["ct_01"].bool()
        batch_size = int(valid.shape[0])
        feature_h, feature_w = output["feat_hw"]
        counts = valid.long().sum(dim=1)
        max_count = int(counts.max().item()) if counts.numel() else 0
        device, dtype = batch["inp"].device, batch["inp"].dtype
        output["ct"] = torch.zeros(
            (batch_size, max_count, 2), device=device, dtype=dtype
        )
        output["detection"] = torch.zeros(
            (batch_size, max_count, 6), device=device, dtype=dtype
        )
        for batch_index in range(batch_size):
            keep = valid[batch_index]
            count = int(keep.sum().item())
            if not count:
                continue
            indices = batch["ct_ind"][batch_index, keep].to(device=device)
            x = (indices % int(feature_w)).to(dtype=dtype)
            y = (indices // int(feature_w)).to(dtype=dtype)
            centers = torch.stack((x, y), dim=1)
            sizes = batch["wh"][batch_index, keep].to(device=device, dtype=dtype)
            boxes = torch.cat(
                (
                    x[:, None] - sizes[..., 0:1] / 2,
                    y[:, None] - sizes[..., 1:2] / 2,
                    x[:, None] + sizes[..., 0:1] / 2,
                    y[:, None] + sizes[..., 1:2] / 2,
                ),
                dim=1,
            ) * self.down_ratio
            boxes = data_utils.clip_to_image(
                boxes,
                int(round(feature_h * self.down_ratio)),
                int(round(feature_w * self.down_ratio)),
            )
            classes = batch["ct_cls"][batch_index, keep].to(
                device=device, dtype=dtype
            ).view(count, 1)
            classes -= float(getattr(cfg, "gt_detection_class_offset", 0))
            scores = torch.ones((count, 1), device=device, dtype=dtype)
            output["ct"][batch_index, :count] = centers * self.down_ratio
            output["detection"][batch_index, :count] = torch.cat(
                (boxes, scores, classes), dim=1
            )

    def encode_batch(self, image: torch.Tensor, batch: dict | None = None) -> dict:
        if batch is None:
            raise ValueError("the mainline forward pass requires a batch dictionary")
        if batch.get("external_detection") is not None:
            raise ValueError("external detections are outside the oracle mainline")
        stride = max(int(round(self.down_ratio)), 1)
        feature_h = (int(image.shape[-2]) + stride - 1) // stride
        feature_w = (int(image.shape[-1]) + stride - 1) // stride
        feature_dtype = (
            torch.get_autocast_gpu_dtype()
            if torch.is_autocast_enabled()
            else image.dtype
        )
        placeholder = image.new_zeros(
            (image.shape[0], 1, feature_h, feature_w), dtype=feature_dtype
        )
        features, feature_stats = self.apply_locate_feature_replacement(
            placeholder, batch
        )
        output = {
            "ct": image.new_zeros((image.shape[0], 0, 2)),
            "detection": image.new_zeros((image.shape[0], 0, 6)),
            "feat_hw": (feature_h, feature_w),
            "cnn_feature": features,
            **feature_stats,
        }
        self._use_gt_detection(output, batch)
        return output

    def forward(self, image: torch.Tensor, batch: dict | None = None) -> dict:
        output = self.encode_batch(image, batch)
        features = output["cnn_feature"]
        feature_h, feature_w = output["feat_hw"]
        feature_stats = {key: value for key, value in output.items() if key.startswith("locate_")}

        detection_context = {
            key: output[key] for key in ("ct", "detection")
        }
        output = self.gcn(output, features, batch)
        output.update(detection_context)
        output["feat_hw"] = (feature_h, feature_w)
        output["cnn_feature"] = features
        output.update(feature_stats)
        return output
