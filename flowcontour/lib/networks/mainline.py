"""Frozen YOLO26x P2/P3 backbone, trainable FPN and CR-MoE contour evolution."""
import torch
from torch import nn

from lib.config import cfg
from lib.networks.pure2d_yolo26_context import Pure2DYOLO26FPN
from lib.networks.diffusion import FlowMatchingEvolution
from lib.utils import data_utils


class Network(nn.Module):
    def __init__(self, down_ratio=4):
        super().__init__()
        if str(cfg.detector_backend) != "flow_box_only" or not cfg.use_gt_det:
            raise ValueError("RAOS contour experiments require GT boxes and classes")
        if int(cfg.anatomical_class_count) != 19 or int(down_ratio) != 4:
            raise ValueError("RAOS requires 19 classes and stride four")
        self.down_ratio = float(down_ratio)
        self.detector_backend = "flow_box_only"
        self.detector_num_classes = 19
        # Keep the module name consistent between training, GDPO and visualization.
        self.locate_feat_replacer = Pure2DYOLO26FPN(
            pretrained_path=str(cfg.yolo26_pretrained_path),
            out_channels=256, pyramid_channels=int(cfg.yolo26_pyramid_channels))
        self.gcn = FlowMatchingEvolution()
        print("[RAOS3] YOLO26x_P2P3 + Dense6/CR-MoE | contour-normalized actions | "
              f"FPN/backbone={sum(p.numel() for p in self.locate_feat_replacer.parameters()):,} "
              f"Flow={sum(p.numel() for p in self.gcn.parameters()):,}", flush=True)

    def _use_gt_detection(self, output, batch):
        valid = batch["ct_01"].bool()
        batch_size = valid.shape[0]
        max_count = int(valid.sum(1).max()) if valid.numel() else 0
        image = batch["inp"]
        output["ct"] = image.new_zeros((batch_size, max_count, 2))
        output["detection"] = image.new_zeros((batch_size, max_count, 6))
        for index in range(batch_size):
            keep = valid[index]
            count = int(keep.sum())
            if not count:
                continue
            # Continuous GT prompt boxes, as in the effective original evaluator.
            target = batch["i_gt_py"][index, keep]
            boxes = torch.cat((target.amin(1), target.amax(1)), -1) * self.down_ratio
            boxes = data_utils.clip_to_image(boxes, image.shape[-2], image.shape[-1])
            classes = batch["ct_cls"][index, keep].to(image.dtype)
            classes = classes - float(cfg.gt_detection_class_offset)
            output["ct"][index, :count] = (boxes[:, :2] + boxes[:, 2:]) / 2
            output["detection"][index, :count] = torch.cat(
                (boxes, torch.ones_like(classes[:, None]), classes[:, None]), -1)

    def encode_batch(self, image, batch=None):
        if batch is None:
            raise ValueError("a batch with GT box/class prompts is required")
        if batch.get("external_detection") is not None:
            raise ValueError("this experiment evaluates GT prompts")
        features = self.locate_feat_replacer(image)
        output = {"cnn_feature": features, "feat_hw": tuple(features.shape[-2:])}
        self._use_gt_detection(output, batch)
        return output

    def forward(self, image, batch=None):
        encoded = self.encode_batch(image, batch)
        result = self.gcn(encoded, encoded["cnn_feature"], batch)
        result.update({key: encoded[key] for key in ("ct", "detection", "feat_hw", "cnn_feature")})
        return result
