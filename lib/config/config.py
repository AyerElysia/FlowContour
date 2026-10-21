"""Configuration bootstrap for the single released DiffusionSnake pipeline.

The two human-readable YAML files are the source of truth:

* configs/stage1.yaml for supervised MoonViT-replacer + Flow training;
* configs/stage2_rl.yaml for five-action Fourier delta-NSD GDPO.

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

cfg.resume = False
cfg.resume_weights_only = False
cfg.resume_path = ""
cfg.resume_exclude_prefixes = []
cfg.resume_allowed_missing_prefixes = []
cfg.resume_allow_partial_copy = False
cfg.resume_source_format = ""

# Shared runtime state used by the launchers and trainer.
cfg.detector_backend = "flow_box_only"
cfg.use_gt_det = True
cfg.use_gt_det_train_only = True
cfg.gt_detection_class_offset = 1
cfg.use_diffusion_evolution = True
cfg.use_flow_matching = True
cfg.use_diffusion_trainer = True
cfg.use_gdpo = False
cfg.use_grpo = False  # Backward-compatible alias for GDPO

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
cfg.diffusion_loss_weight = 1.0
cfg.diffusion_disp_norm = True
cfg.diffusion_disp_stats = "data/stats/volmem_sagittal_disp_stats.json"
cfg.dit_num_layers = 6
cfg.dit_num_heads = 8
cfg.dit_state_dim = 256
cfg.iterative_fractions = [0.6667, 1.0]
cfg.iterative_ode_steps = 4
cfg.train_progress_sigma = 0.05
cfg.train_progress_uniform_prob = 0.15
cfg.train_progress_centers = [0.0, 0.3333, 0.5, 0.80, 0.97]
cfg.train_progress_weights = [0.2933, 0.1767, 0.27, 0.179, 0.081]
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
cfg.ddp_find_unused_parameters = True
cfg.ddp_gradient_as_bucket_view = True
cfg.ddp_bucket_cap_mb = 25
cfg.use_amp = False
cfg.amp_dtype = "float16"
cfg.enable_tf32 = True
cfg.cudnn_benchmark = True
cfg.cuda_empty_cache_interval = 0

cfg.train = CN()
cfg.train.dataset = "VolMemTrain"
cfg.train.per_contour = False
cfg.train.optim = "adamw"
cfg.train.lr = 1e-5
cfg.train.locate_lr_multiplier = 1.0
cfg.train.milestones = [80, 120]
cfg.train.gamma = 0.5
cfg.train.batch_size = 1
cfg.train.gradient_accumulation_steps = 1
cfg.train.gradient_clip = 1.0
cfg.train.weight_decay = 0.01
cfg.train.warmup_steps = 0
cfg.train.drop_last = True
cfg.train.epoch = 1
cfg.train.max_steps = 0
cfg.train.step_checkpoint_every = 0
cfg.train.step_checkpoint_keep = 12
cfg.train.step_checkpoint_milestones = []
cfg.train.num_workers = 0
cfg.train.save_ep = 0
cfg.train.max_ct_num = 32

cfg.test = CN()
cfg.test.dataset = "VolMemDev8"
cfg.test.batch_size = 1
cfg.test.epoch = -1

cfg.record_dir = "data/records"
cfg.result_dir = "data/results"
cfg.save_ep = 0
cfg.eval_ep = 0
cfg.skip_eval = False


def _infer_default_cfg(parsed_args: argparse.Namespace) -> str:
    if getattr(parsed_args, "cfg_file", ""):
        return str(parsed_args.cfg_file)
    environment_path = os.environ.get("CFG_FILE", "").strip()
    if environment_path:
        return environment_path
    script_name = Path(sys.argv[0]).stem.lower()
    return (
        "configs/stage2_rl.yaml"
        if "train_rl" in script_name
        else "configs/stage1.yaml"
    )


def _parse_runtime(config: CN, parsed_args: argparse.Namespace) -> None:
    if not str(config.task):
        raise ValueError("task must be specified")
    gpu_override = os.environ.get("DIFFUSIONSNAKE_GPU", "").strip()
    if gpu_override:
        config.gpus = [int(gpu_override)]
    if not os.environ.get("CUDA_VISIBLE_DEVICES", "").strip():
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
    cfg.merge_from_file(config_path)
    cfg.merge_from_list(parsed_args.opts or [])
    _parse_runtime(cfg, parsed_args)
    return cfg


parser = argparse.ArgumentParser()
parser.add_argument("--cfg_file", default="", type=str)
parser.add_argument("-f", default="", type=str)
parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
args = parser.parse_args()
cfg = make_cfg(args)


__all__ = ("args", "cfg", "make_cfg")
