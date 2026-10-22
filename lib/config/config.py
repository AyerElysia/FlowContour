"""Configuration bootstrap for the FlowContour full2.v2 pipeline.

common.yaml is merged once, then pretrain.yaml / online.yaml / gdpo.yaml.

This module contains only runtime defaults needed before a YAML file is
merged. The local CfgNode implementation accepts keys declared by those YAML
files, so experimental switches do not live here.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from .yacs import CfgNode as CN


cfg = CN()

# Values required before and around YAML loading.
cfg.model = "moonvit_flow"
cfg.model_dir = "data/outputs"
cfg.task = "snake"
cfg.gpus = [0]
cfg.train_or_test = "train"
cfg.anatomical_class_count = 26

# Shared runtime state used by the launchers and trainer.
cfg.detector_backend = "flow_box_only"
cfg.use_gt_det = True
cfg.use_gt_det_train_only = True
cfg.gt_detection_class_offset = 1
cfg.use_diffusion_evolution = True
cfg.use_flow_matching = True
cfg.use_gdpo = False
cfg.training_stage = "pretrain"

# Fixed contour and data geometry.
cfg.down_ratio = 4
cfg.poly_num = 128
cfg.gt_poly_num = 128
cfg.min_poly_area_output = 0.5

# The mainline is single-frame. These names replace old pseudo-3D terminology.
cfg.image_mean = 0.5
cfg.image_std = 0.5
cfg.image_color_aug = False
cfg.image_lr_flip = False
cfg.image_random_crop = True

cfg.sagittal_moonvit_cache_root = ""
cfg.sagittal_train_exclude_case_ids = []
cfg.sagittal_eval_include_case_ids = []
cfg.sagittal_forbidden_case_ids = []
cfg.sagittal_expected_train_case_count = 0
cfg.sagittal_expected_train_row_count = 0
cfg.sagittal_expected_eval_case_count = 0
cfg.sagittal_expected_eval_row_count = 0
cfg.sagittal_component_mode = "significant"
cfg.sagittal_max_components_per_class = 4
cfg.sagittal_min_component_area_raw = 2.0
cfg.sagittal_max_instances_per_slice = 32

cfg.locate_feat_inject = False
cfg.locate_feat_replace = True
cfg.locate_feat_cache_root = "data/sagittal_moonvit_cache"
cfg.locate_feat_keys = ["layer_18"]
cfg.locate_feat_dim = 1152
cfg.locate_feat_input_layers = 1
cfg.locate_feat_fusion_mode = "center_only"
cfg.locate_feat_replace_hidden_dim = 512
cfg.locate_feat_replace_out_channels = 256
cfg.locate_feat_replace_upscale = 2
cfg.locate_feat_resample_padding_mode = "border"
cfg.snake_feature_dim = 256
cfg.gcn_sample_mode = "half_pixel"
cfg.gcn_sample_padding_mode = "border"

# One Flow architecture and one deployment solver.
cfg.flow_2d_s_conditioning = True
cfg.flow_ode_steps = 4
cfg.flow_ode_solver = "ab2"
cfg.flow_train_noise_scale = 1.0
cfg.infer_noise_scale = 1.0
cfg.infer_avg_samples = 1
cfg.diffusion_disp_norm = True
cfg.diffusion_disp_stats = "data/stats/volmem_sagittal_disp_stats.json"
cfg.dit_num_layers = 6
cfg.dit_num_heads = 8
cfg.dit_state_dim = 256
cfg.train_state_probabilities = [.5, .2, .3]
cfg.refine_sigma_px = 3.0
cfg.refine_modes = 8
cfg.diffusion_init_source = "bbox_octagon"
cfg.contour_init_method = "octagon"

# Route-B box augmentation.
cfg.routeb_box_jitter_enabled = True
cfg.routeb_box_jitter_probabilities = [0.35, 0.40, 0.20, 0.05]
cfg.routeb_box_jitter_shift_fractions = [0.0, 0.05, 0.10, 0.15]
cfg.routeb_box_jitter_log_scale_fractions = [0.0, 0.10, 0.20, 0.30]
cfg.routeb_box_jitter_edge_fractions = [0.0, 0.03, 0.08, 0.15]
cfg.routeb_box_jitter_min_iou = 0.20

# Data-loader and accelerator behavior.
cfg.dataloader_persistent_workers = True
cfg.dataloader_prefetch_factor = 4

cfg.train = CN()
cfg.train.dataset = "VolMemTrain"
cfg.train.per_contour = False
cfg.train.lr = 1e-5
cfg.train.batch_size = 1
cfg.train.gradient_clip = 1.0
cfg.train.weight_decay = 0.01
cfg.train.warmup_steps = 0
cfg.train.drop_last = True
cfg.train.max_steps = 0
cfg.train.step_checkpoint_every = 0
cfg.train.num_workers = 0
cfg.train.max_ct_num = 32

cfg.test = CN()
cfg.test.dataset = "VolMemDev8"
cfg.test.batch_size = 1

cfg.record_dir = "data/records"
cfg.result_dir = "data/results"


def _infer_default_cfg(parsed_args: argparse.Namespace) -> str:
    if getattr(parsed_args, "cfg_file", ""):
        return str(parsed_args.cfg_file)
    environment_path = os.environ.get("CFG_FILE", "").strip()
    if environment_path:
        return environment_path
    return "configs/pretrain.yaml"


def _parse_runtime(config: CN, parsed_args: argparse.Namespace) -> None:
    if not str(config.task):
        raise ValueError("task must be specified")
    gpu_override = os.environ.get("FLOWCONTOUR_GPU", "").strip()
    if gpu_override:
        config.gpus = [int(gpu_override)]
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
            str(gpu) for gpu in config.gpus
        )
    config.record_dir = os.path.join(
        str(config.record_dir), str(config.task), str(config.model)
    )
    config.result_dir = os.path.join(
        str(config.result_dir), str(config.task), str(config.model)
    )


def make_cfg(parsed_args: argparse.Namespace) -> CN:
    config_path = _infer_default_cfg(parsed_args)
    cfg.merge_from_file(str(Path(__file__).resolve().parents[2] / "configs/common.yaml"))
    cfg.merge_from_file(config_path)
    cfg.merge_from_list(parsed_args.opts or [])
    _parse_runtime(cfg, parsed_args)
    return cfg


parser = argparse.ArgumentParser()
parser.add_argument("--cfg_file", default="", type=str)
parser.add_argument("-f", default="", type=str)
parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
# Importing a library must not consume unittest/notebook/application flags.
# Public entry points pass an explicit --cfg_file before importing this module.
args = parser.parse_args() if "--cfg_file" in sys.argv else parser.parse_args([])
cfg = make_cfg(args)


__all__ = ("args", "cfg", "make_cfg")
