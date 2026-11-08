"""YOLO26 P2/P3 context, without a detection head or MoonViT branch."""

import torch
from torch import nn
from torch.nn import functional as F


def align_stride_map(feature, output_hw, source_stride, target_stride=4):
    # Padded strided convolutions have centers 0, stride, 2*stride, ... .
    h, w = output_hw
    y = torch.arange(h, device=feature.device, dtype=feature.dtype)
    x = torch.arange(w, device=feature.device, dtype=feature.dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    ratio = target_stride / source_stride
    grid = torch.stack(((xx * ratio + 0.5) * 2 / feature.size(3) - 1,
                        (yy * ratio + 0.5) * 2 / feature.size(2) - 1), -1)
    return F.grid_sample(feature, grid[None].expand(feature.size(0), -1, -1, -1),
                         align_corners=False, padding_mode="border")


class Pure2DYOLO26FPN(nn.Module):
    def __init__(self, pretrained_path, out_channels=256, pyramid_channels=128):
        super().__init__()
        from ultralytics import YOLO
        from lib.utils.snake import snake_config

        from lib.packaged_weights import detector_weight_path

        with detector_weight_path(pretrained_path) as detector_path:
            model = YOLO(str(detector_path)).model
        if "26" not in str(model.yaml.get("yaml_file", "")):
            raise ValueError("The checkpoint is not identified as YOLO26")
        self.backbone = nn.ModuleList(list(model.model.children())[:5])
        self.backbone.eval().requires_grad_(False)
        self.register_buffer("source_mean", torch.as_tensor(snake_config.mean).view(1, 3, 1, 1))
        self.register_buffer("source_std", torch.as_tensor(snake_config.std).view(1, 3, 1, 1))
        channels = []
        with torch.no_grad():
            x = torch.zeros(1, 3, 64, 64)
            for i, block in enumerate(self.backbone):
                if block.f != -1:
                    raise ValueError("YOLO shallow blocks must form a sequential backbone")
                x = block(x)
                if i in (2, 4):
                    channels.append(x.size(1))
        self.channels = channels
        self.projections = nn.ModuleList(nn.Sequential(
            nn.Conv2d(c, pyramid_channels, 1, bias=False),
            nn.GroupNorm(8, pyramid_channels), nn.GELU()) for c in channels)
        self.fuse = nn.Sequential(
            nn.Conv2d(pyramid_channels * 2, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(16, out_channels), nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 1), nn.GroupNorm(16, out_channels))

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def rgb_input(self, normalized_bgr):
        bgr = normalized_bgr * self.source_std + self.source_mean
        return bgr[:, [2, 1, 0]].clamp(0, 1)

    def forward(self, image):
        with torch.no_grad():
            x = self.rgb_input(image)
            maps = []
            for i, block in enumerate(self.backbone):
                x = block(x)
                if i in (2, 4):
                    maps.append(x)
        p2, p3 = [project(m.detach()) for project, m in zip(self.projections, maps)]
        p3 = align_stride_map(p3, p2.shape[-2:], source_stride=8)
        return self.fuse(torch.cat((p2, p3), dim=1))
