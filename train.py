"""Supervised stage-1 trainer for the MoonViT-cache + Flow mainline."""

import os
import torch
import time
import copy
import sys
import json
import hashlib
import math
import random
import traceback
import uuid
import logging
from datetime import datetime
from pathlib import Path
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

TIME_SCALE_FACTOR = 1000.0
CHECKPOINT_FORMAT_VERSION = 2
DEFAULT_STEP_CHECKPOINT_EVERY = 500
DEFAULT_STEP_CHECKPOINT_KEEP = 3


def _is_finite_value(value):
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value.detach()).all().item())
    if isinstance(value, dict):
        return all(_is_finite_value(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_is_finite_value(item) for item in value)
    if isinstance(value, (float, int)):
        return math.isfinite(float(value))
    return True


def _scalar_float(value, field_name):
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(f'{field_name} must be scalar, got shape={tuple(value.shape)}')
        value = value.detach().item()
    value = float(value)
    if not math.isfinite(value):
        raise FloatingPointError(f'{field_name} is non-finite: {value!r}')
    return value


def _loss_stats_to_floats(loss_stats):
    if loss_stats is None:
        return {}
    if not isinstance(loss_stats, dict):
        raise TypeError(f'loss_stats must be a dict, got {type(loss_stats).__name__}')
    if not _is_finite_value(loss_stats):
        raise FloatingPointError('loss_stats contains a non-finite value')

    result = {}
    for key, value in loss_stats.items():
        field_name = f'loss_stats[{key!r}]'
        if isinstance(value, torch.Tensor) and value.numel() != 1:
            value = value.detach().float().mean()
        result[str(key)] = _scalar_float(value, field_name)
    return result


_CRASH_CONTEXT = {}


def _write_crash_event(exc):
    path = os.environ.get('ONE_SAMPLE_CRASH_PATH', '').strip()
    if not path:
        path = os.path.join(_CRASH_CONTEXT.get('out_dir', 'data/outputs'), 'crash.jsonl')
    event = {
        'event': 'crash',
        'run_id': _CRASH_CONTEXT.get('run_id'),
        'cfg_hash': _CRASH_CONTEXT.get('cfg_hash'),
        'timestamp': datetime.utcnow().isoformat(timespec='milliseconds') + 'Z',
        'rank': _CRASH_CONTEXT.get('rank'),
        'phase': _CRASH_CONTEXT.get('phase'),
        'step': _CRASH_CONTEXT.get('step'),
        'epoch': _CRASH_CONTEXT.get('epoch'),
        'step_in_epoch': _CRASH_CONTEXT.get('step_in_epoch'),
        'loss': _CRASH_CONTEXT.get('loss'),
        'exception_type': type(exc).__name__,
        'error': str(exc),
        'traceback': traceback.format_exc(),
    }
    try:
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'a', encoding='utf-8') as handle:
            handle.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + '\n')
            handle.flush()
    except Exception as write_exc:
        logger.error('Failed to write crash JSONL: %s', write_exc)


def _config_hash(config):
    normalized = copy.deepcopy(config)
    for key, value in (
        ('resume', False),
        ('resume_path', ''),
        ('resume_weights_only', False),
    ):
        if key in normalized:
            normalized[key] = value
    return hashlib.sha256(normalized.dump().encode('utf-8')).hexdigest()


def _validate_formal_resume_cfg_hash(checkpoint_cfg_hash, current_cfg_hash):
    """Validate strict-resume identity, including an explicit audited transition.

    A scientific configuration change normally invalidates a formal strict
    resume. The only exception is a launcher-authorized transition tied to the
    exact source checkpoint hash. Model, optimizer, scheduler, progress and RNG
    restoration remain strict; this merely records that the input curriculum
    intentionally changed at the resume boundary.
    """
    checkpoint_cfg_hash = str(checkpoint_cfg_hash)
    current_cfg_hash = str(current_cfg_hash)
    if checkpoint_cfg_hash == current_cfg_hash:
        return None

    expected_source = os.environ.get(
        'FORMAL_RESUME_EXPECTED_SOURCE_CFG_HASH', ''
    ).strip()
    transition_id = os.environ.get(
        'FORMAL_RESUME_CONFIG_TRANSITION_ID', ''
    ).strip()
    if not expected_source or not transition_id:
        raise RuntimeError(
            'formal resume cfg hash mismatch: '
            f'checkpoint={checkpoint_cfg_hash} current={current_cfg_hash}'
        )
    if checkpoint_cfg_hash != expected_source:
        raise RuntimeError(
            'formal resume source cfg hash does not match the authorized '
            f'transition: checkpoint={checkpoint_cfg_hash} '
            f'expected_source={expected_source} current={current_cfg_hash}'
        )
    if len(transition_id) > 128 or any(
        character not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-'
        for character in transition_id
    ):
        raise RuntimeError(
            'FORMAL_RESUME_CONFIG_TRANSITION_ID must be a non-empty safe token '
            'of at most 128 characters'
        )
    return {
        'transition_id': transition_id,
        'source_cfg_hash': checkpoint_cfg_hash,
        'target_cfg_hash': current_cfg_hash,
    }


def _step_checkpoint_paths_to_remove(step_paths, keep_recent, milestone_steps):
    """Return non-recent, non-milestone step checkpoints for pruning."""
    keep_recent = int(keep_recent)
    if keep_recent <= 0:
        raise ValueError('keep_recent must be positive')
    milestones = {int(step) for step in milestone_steps}
    ordered = sorted(
        step_paths,
        key=lambda item: int(item.stem.split('_', 1)[1]),
    )
    recent = set(ordered[-keep_recent:])
    return [
        path
        for path in ordered
        if path not in recent
        and int(path.stem.split('_', 1)[1]) not in milestones
    ]


def _capture_rng_state():
    state = {
        'python': random.getstate(),
        'torch': torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state['cuda'] = torch.cuda.get_rng_state_all()
    try:
        import numpy as np
        state['numpy'] = np.random.get_state()
    except ImportError:
        pass
    return state


def _restore_rng_state(state):
    if not isinstance(state, dict):
        raise ValueError('checkpoint RNG state is missing or invalid')
    random.setstate(state['python'])
    torch.set_rng_state(state['torch'])
    if 'cuda' in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['cuda'])
    if 'numpy' in state:
        import numpy as np
        np.random.set_state(state['numpy'])


def _adapter_parameters(module):
    return [
        (name, parameter)
        for name, parameter in module.named_parameters()
        if 'locate_feat_adapter' in name.lower() and parameter.requires_grad
    ]


def _adapter_metrics(module, before=None):
    grad_sq = 0.0
    grad_max = 0.0
    weight_absmax = 0.0
    update_sq = 0.0
    update_max = 0.0
    current = {}
    grad_parameter_count = 0
    adapter_parameters = _adapter_parameters(module)
    for name, parameter in adapter_parameters:
        weight = parameter.detach()
        if not bool(torch.isfinite(weight).all().item()):
            raise FloatingPointError(f'non-finite adapter weight: {name}')
        weight_absmax = max(weight_absmax, float(weight.abs().max().item()))
        current[name] = weight.clone()
        if parameter.grad is not None:
            grad_parameter_count += 1
            grad = parameter.grad.detach()
            if not bool(torch.isfinite(grad).all().item()):
                raise FloatingPointError(f'non-finite adapter gradient: {name}')
            grad_sq += float(torch.sum(grad * grad).item())
            grad_max = max(grad_max, float(grad.abs().max().item()))
        if before is not None and name in before:
            delta = weight - before[name]
            update_sq += float(torch.sum(delta * delta).item())
            update_max = max(update_max, float(delta.abs().max().item()))
    return {
        'adapter_parameter_count': int(len(adapter_parameters)),
        'adapter_grad_parameter_count': int(grad_parameter_count),
        'adapter_grad_l2': float(grad_sq ** 0.5),
        'adapter_grad_max': float(grad_max),
        'adapter_weight_absmax': float(weight_absmax),
        'adapter_update_l2': float(update_sq ** 0.5),
        'adapter_update_max': float(update_max),
    }, current


def _batch_health_metrics(batch):
    foreground_count = 0
    contour_count = 0
    foreground_ratio = 0.0
    if isinstance(batch, dict):
        mask = batch.get('ct_01')
        if isinstance(mask, torch.Tensor):
            valid = mask > 0.5
            contour_count = int(valid.sum().item())
            if valid.ndim >= 2 and valid.size(0) > 0:
                foreground_samples = valid.any(dim=1)
                foreground_count = int(foreground_samples.sum().item())
                foreground_ratio = float(foreground_samples.float().mean().item())
            elif valid.numel() > 0:
                foreground_count = int(valid.any().item())
                foreground_ratio = float(valid.any().item())
        elif 'meta' in batch and isinstance(batch['meta'], dict):
            count = batch['meta'].get('ct_num')
            if isinstance(count, torch.Tensor):
                count = count.detach().reshape(-1)
                foreground_samples = count > 0
                foreground_count = int(foreground_samples.sum().item())
                contour_count = int(count.sum().item())
                foreground_ratio = float(foreground_samples.float().mean().item()) if count.numel() else 0.0
    return {
        'foreground_count': int(foreground_count),
        'foreground_ratio': float(foreground_ratio),
        'contour_count': int(contour_count),
    }


def _cuda_health_metrics():
    if not torch.cuda.is_available():
        return {
            'cuda_allocated_bytes': 0,
            'cuda_reserved_bytes': 0,
            'cuda_max_allocated_bytes': 0,
        }
    return {
        'cuda_allocated_bytes': int(torch.cuda.memory_allocated()),
        'cuda_reserved_bytes': int(torch.cuda.memory_reserved()),
        'cuda_max_allocated_bytes': int(torch.cuda.max_memory_allocated()),
    }

'''
尝试导入一些可选工具。
如果电脑里没有安装，也不要让程序直接崩溃，而是换一种备用方案继续运行。

'''
try:
    import wandb#wandb 是一个训练可视化平台，全名一般叫 Weights & Biases。
except ImportError:
    wandb = None
    logger.warning("wandb not available, logging will be local only")
'''
这里尝试从 diffusers 库里面导入学习率调度器。
diffusers 通常是 HuggingFace 里和扩散模型相关的库。这个项目是 DiffusionSnake，
所以作者可能原本想用 diffusers 里面的 scheduler。
get_scheduler 的作用是创建学习率调度器。
SchedulerType 是学习率调度类型，比如：
constant：学习率不变
linear：学习率线性下降
cosine：学习率按余弦曲线下降
没有 diffusers 的调度器，那就改用 PyTorch 自带的学习率调度器。
'''
try:
    from diffusers.optimization import get_scheduler, SchedulerType
except ImportError:
    get_scheduler = None
    logger.info("diffusers.optimization not available, using PyTorch schedulers")
    class SchedulerType:
        CONSTANT = "constant"
        CONSTANT_WITH_WARMUP = "constant_with_warmup"
        LINEAR = "linear"
        COSINE = "cosine"
        COSINE_WITH_RESTARTS = "cosine_with_restarts"
        POLYNOMIAL = "polynomial"


# IMPORTANT: lib.config 会在 import 时就解析 argv/环境变量并加载 cfg 文件。
# 这里不能强行覆盖用户通过环境变量/命令行指定的 cfg。
# 仅当用户没有设置 CFG_FILE 且没有传 --cfg_file 时，才回退到默认 diffusion 配置。
'''
__file__ 表示当前这个 Python 文件的路径
os.path.dirname(__file__) 表示当前文件所在的文件夹路径。
sys.argv 是命令行参数列表
比如你运行：python diffusion_train.py --cfg_file configs/my.yaml
那么 sys.argv 大概是：
[
    "diffusion_train.py",
    "--cfg_file",
    "configs/my.yaml"
]
.lower()：这句代码把里面所有参数都变成小写，方便后面判断。

优先级：
1. 命令行 --cfg_file
2. 环境变量 CFG_FILE
3. 环境变量 DIFFUSION_CFG_FILE
4. 默认 configs/diffusion_snake.yaml
'''
_THIS_DIR = os.path.dirname(__file__)
#这句是在拼接默认配置文件路径。
_DEFAULT_CFG = os.path.join(_THIS_DIR, 'configs', 'stage1.yaml')
_argv_lower = [a.lower() for a in sys.argv]
#用户有没有在命令行里手动指定配置文件？它同时支持两种写法：--cfg_file和：--cfg-file
_has_cli_cfg = ('--cfg_file' in _argv_lower) or ('--cfg-file' in _argv_lower)
#如果用户没有通过命令行指定配置文件，
# 并且用户也没有通过环境变量 CFG_FILE 指定配置文件，
# 那就进入下面的默认设置。
# 如果用户设置了 DIFFUSION_CFG_FILE，就用 DIFFUSION_CFG_FILE；
# 如果用户没设置 DIFFUSION_CFG_FILE，就用默认的 diffusion_snake.yaml。
if (not _has_cli_cfg) and (not os.environ.get('CFG_FILE')):
    os.environ['CFG_FILE'] = os.environ.get('DIFFUSION_CFG_FILE', _DEFAULT_CFG)

from lib.config import cfg, args
from lib.networks import make_network
from lib.train.trainers import make_trainer
from lib.train.optimizer import make_optimizer
from lib.datasets import make_data_loader
from lib.datasets.dataset_catalog import DatasetCatalog
from lib.recorder import JsonLogger, NullLogger
from lib.visualizers.diffusion_one_sample import save_affine_visualization


def safe_barrier(is_distributed: bool) -> None:
    """Safely execute distributed barrier with error handling."""
    if is_distributed:
        try:
            dist.barrier()
        except RuntimeError as e:
            logger.warning(f"Distributed barrier failed: {e}")


def main():
    is_distributed = False
    rank = 0
    world_size = 1
    local_rank = 0
    try:
        world_size = int(os.environ.get('WORLD_SIZE', '1'))
        rank = int(os.environ.get('RANK', '0'))
        local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    except (ValueError, TypeError) as e:
        logger.warning(f"Failed to parse distributed environment variables: {e}")
        world_size, rank, local_rank = 1, 0, 0

    is_distributed = world_size > 1
    if is_distributed:
        torch.cuda.set_device(local_rank)
        if not dist.is_initialized():
            dist.init_process_group(backend='nccl')
    is_main_process = (rank == 0)

    # Enable Diffusion policy + trainer (遵循配置，其它开关不在此处强行覆盖)
    cfg.use_diffusion_evolution = True
    cfg.use_diffusion_trainer = True

    try:
        _cfg_file_used = getattr(args, 'cfg_file', '') or os.environ.get('CFG_FILE', '')
        _cfg_stem = Path(str(_cfg_file_used)).stem if _cfg_file_used else 'default'
    except (AttributeError, TypeError) as e:
        logger.warning(f"Failed to get config file stem: {e}")
        _cfg_stem = 'default'

    out_dir_override = os.environ.get('ONE_SAMPLE_OUT_DIR', '').strip()
    out_dir = out_dir_override or os.path.join(os.path.dirname(__file__), 'data', 'outputs', _cfg_stem)
    os.makedirs(out_dir, exist_ok=True)
    run_id = os.environ.get('TRAIN_RUN_ID', '').strip() or uuid.uuid4().hex
    if is_distributed:
        run_id_object = [run_id if is_main_process else None]
        dist.broadcast_object_list(
            run_id_object,
            src=0,
            device=torch.device('cuda', local_rank),
        )
        run_id = str(run_id_object[0])
    cfg_hash = _config_hash(cfg)
    _CRASH_CONTEXT.update({
        'run_id': run_id,
        'cfg_hash': cfg_hash,
        'out_dir': out_dir,
        'rank': rank,
    })
    os.environ['ONE_SAMPLE_CRASH_PATH'] = os.path.join(out_dir, 'crash.jsonl')

    accumulation_steps = int(getattr(cfg.train, 'gradient_accumulation_steps', 1))
    if accumulation_steps != 1:
        raise ValueError(
            'gradient_accumulation_steps must be exactly 1 for health-safe training, '
            f'got {accumulation_steps}'
        )
    configured_clip = getattr(cfg.train, 'gradient_clip', 1.0)
    gradient_clip = float(configured_clip)
    if not math.isfinite(gradient_clip) or gradient_clip <= 0.0:
        raise ValueError(f'cfg.train.gradient_clip must be finite and > 0, got {gradient_clip!r}')

    try:
        train_name = getattr(cfg.train, 'dataset', None)
        test_name = getattr(cfg.test, 'dataset', None)
        logger.info(f"cfg_file={os.environ.get('CFG_FILE','')}")
        logger.info(f"train.dataset={train_name} test.dataset={test_name}")
        if train_name in DatasetCatalog.dataset_attrs:
            a = DatasetCatalog.get(train_name)
            logger.info(f"train.data_root={a.get('data_root')} ann_file={a.get('ann_file')}")
        if test_name in DatasetCatalog.dataset_attrs:
            a = DatasetCatalog.get(test_name)
            logger.info(f"test.data_root={a.get('data_root')} ann_file={a.get('ann_file')}")
    except (AttributeError, KeyError) as e:
        logger.warning(f"Failed to log dataset info: {e}")

    # cfg.train.num_workers = 0  # 已移除：使用 yaml 中配置的 num_workers (默认4)

    # Set optimizer to adamw if not already configured
    if not hasattr(cfg.train, 'optim') or cfg.train.optim is None:
        cfg.train.optim = 'adamw'
        logger.info("Set default optimizer to adamw")

    # Build
    logger.info("Building network")
    network = make_network(cfg)
    logger.info("Building trainer")
    trainer = make_trainer(cfg, network)

    # The pure-2D mainline is a physically slim model, not a MemFlow model run
    # with memory disabled.  Keep this as a hard construction-time contract so
    # a future config change cannot silently reintroduce Memory or the internal
    # detector before a long training run.
    if bool(getattr(cfg, 'pure2d_memory_free_required', False)):
        named_parameters = list(trainer.network.named_parameters())
        parameter_names = [name for name, _ in named_parameters]
        forbidden_memory_tokens = (
            'memory_encoder',
            'memflow_controller',
            'memory_reader',
            'memory_writer',
            'memory_bank',
        )
        forbidden_memory_names = [
            name for name in parameter_names
            if any(token in name.lower() for token in forbidden_memory_tokens)
        ]
        if forbidden_memory_names:
            raise RuntimeError(
                'pure2d Memory-free contract failed; forbidden parameters: '
                + ', '.join(forbidden_memory_names[:20])
            )

        if bool(getattr(cfg, 'pure2d_internal_detector_free_required', False)):
            detector_names = [
                name for name in parameter_names if 'heatmap_detector' in name.lower()
            ]
            if detector_names:
                raise RuntimeError(
                    'pure2d detector-free contract failed; internal detector parameters: '
                    + ', '.join(detector_names[:20])
                )

        observed_parameters = sum(parameter.numel() for _, parameter in named_parameters)
        expected_parameters = int(
            getattr(cfg, 'pure2d_expected_parameter_count', observed_parameters)
        )
        if observed_parameters != expected_parameters:
            raise RuntimeError(
                'pure2d parameter-count contract failed: '
                f'observed={observed_parameters} expected={expected_parameters}'
            )
        logger.info(
            'Pure2D physical-slim contract PASS: '
            f'parameters={observed_parameters}, memory_parameters=0, '
            'internal_detector_parameters=0'
        )
    if torch.cuda.is_available():
        trainer.network.cuda()

    if is_distributed:
        _ddp_find_unused = os.environ.get('DDP_FIND_UNUSED_PARAMETERS', '1')
        _ddp_find_unused = str(_ddp_find_unused).strip().lower() in ('1', 'true', 'yes', 'y', 'on')
        trainer.network = DDP(
            trainer.network,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
            find_unused_parameters=bool(_ddp_find_unused),
        )
    optimizer = make_optimizer(cfg, trainer.network)

    if is_main_process:
        logger.info(f"use_gdpo={getattr(cfg, 'use_gdpo', getattr(cfg, 'use_grpo', None))}")

    data_loader = make_data_loader(
        cfg, is_train=True, is_distributed=is_distributed
    )
    logger.info("Train72-only training contract PASS")
    logger.info(f"Train samples: {len(data_loader.dataset)}")
    logger.info(f"Batches per epoch: {len(data_loader)}")

    def move_batch_to_cuda(batch):
        for k in list(batch.keys()):
            if k in ('meta', 'orig_img', 'img_path') or k == 'locate_feat' or str(k).startswith('locate_feat_'):
                continue
            if isinstance(batch[k], torch.Tensor):
                batch[k] = batch[k].cuda(non_blocking=True)
        return batch

    # 取一个批次用于可视化，训练循环内重新取 batch
    if is_main_process:
        try:
            viz_batch = move_batch_to_cuda(copy.deepcopy(next(iter(data_loader))))
        except StopIteration:
            viz_batch = None
    else:
        viz_batch = None

    # Helper: run inference and save visualization on affine-input coordinates
    def _visualize_one_sample(tag: str):
        if not is_main_process:
            return
        batch = viz_batch
        trainer.network.eval()
        wrapped_network = (
            trainer.network.module
            if hasattr(trainer.network, 'module')
            else trainer.network
        )
        inference_network = getattr(wrapped_network, 'net', wrapped_network)
        restore_train_only = bool(getattr(cfg, 'use_gt_det_train_only', False))
        use_visualization_gt = (
            str(getattr(cfg, 'detector_backend', '')).strip()
            in ('flow_box_only', 'flow_box_cnn')
            and restore_train_only
        )
        if use_visualization_gt:
            cfg.use_gt_det_train_only = False
        try:
            with torch.no_grad():
                output = inference_network(batch['inp'], batch)
        finally:
            if use_visualization_gt:
                cfg.use_gt_det_train_only = restore_train_only
        save_dir = os.path.join(out_dir, 'visual', 'diffusion_one_sample')
        save_affine_visualization(output=output, batch=batch, tag=str(tag), save_dir=save_dir)

    infer_and_save = _visualize_one_sample

    # json logging setup
    out_dir_override = os.environ.get('ONE_SAMPLE_OUT_DIR', '').strip()
    if out_dir_override:
        out_dir = out_dir_override
    else:
        out_dir = os.path.join(os.path.dirname(__file__), 'data', 'outputs', _cfg_stem)
    log_path = os.path.join(out_dir, 'logs.jsonl')
    ckpt_dir = os.path.join(out_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)

    save_ep = int(getattr(cfg.train, 'save_ep', 0))
    step_checkpoint_every = int(
        os.environ.get(
            'TRAIN_STEP_CHECKPOINT_EVERY',
            getattr(cfg.train, 'step_checkpoint_every', DEFAULT_STEP_CHECKPOINT_EVERY),
        )
    )
    step_checkpoint_keep = int(
        os.environ.get(
            'TRAIN_STEP_CHECKPOINT_KEEP',
            getattr(cfg.train, 'step_checkpoint_keep', DEFAULT_STEP_CHECKPOINT_KEEP),
        )
    )
    if step_checkpoint_every < 0 or step_checkpoint_keep <= 0:
        raise ValueError('step checkpoint interval must be >= 0 and retention must be > 0')
    try:
        step_checkpoint_milestones = {
            int(step)
            for step in getattr(cfg.train, 'step_checkpoint_milestones', ())
        }
    except (TypeError, ValueError) as error:
        raise ValueError('step checkpoint milestones must be integer steps') from error
    if any(step <= 0 for step in step_checkpoint_milestones):
        raise ValueError('step checkpoint milestones must be positive')
    resume_step = 0
    resume_epoch = 0
    resume_step_in_epoch = 0
    resume_json_pos = None
    resume_checkpoint = None
    strict_resume = False
    resume_cfg_transition = None
    resume_path = os.environ.get('ONE_SAMPLE_RESUME_PATH', '').strip()
    if not resume_path:
        resume_path = str(getattr(cfg, 'resume_path', '') or '').strip()

    def _safe_load_optimizer_state(optimizer_obj, opt_state_dict) -> bool:
        """Load optimizer state_dict while dropping states whose tensor shapes mismatch current params.

        This is important when initializing from the signed source checkpoint.
        """
        if not isinstance(opt_state_dict, dict):
            return False
        try:
            # Build mapping from parameter index used in optimizer state_dict to actual parameter.
            param_list = []
            for group in optimizer_obj.param_groups:
                param_list.extend(list(group.get('params', [])))
            if not param_list:
                return False

            state = opt_state_dict.get('state', {})
            if not isinstance(state, dict):
                return False

            filtered_state = {}
            for pid, st in state.items():
                if not isinstance(st, dict):
                    continue
                try:
                    pid_int = int(pid)
                except Exception:
                    continue
                if pid_int < 0 or pid_int >= len(param_list):
                    continue
                p = param_list[pid_int]
                if p is None or (not hasattr(p, 'shape')):
                    continue

                new_st = dict(st)
                ok = True
                for k in ('exp_avg', 'exp_avg_sq', 'max_exp_avg_sq'):
                    v = new_st.get(k, None)
                    if v is None:
                        continue
                    if isinstance(v, torch.Tensor) and tuple(v.shape) != tuple(p.shape):
                        ok = False
                        break
                if ok:
                    filtered_state[pid_int] = new_st

            filtered = dict(opt_state_dict)
            filtered['state'] = filtered_state
            optimizer_obj.load_state_dict(filtered)
            return True
        except Exception:
            return False

    def _safe_load_model_state(model_obj, source_state_dict):
        """Load checkpoint tensors conservatively, reusing exact matches and overlapping slices when possible."""
        if not isinstance(source_state_dict, dict):
            return None

        raw_exclude_prefixes = getattr(cfg, 'resume_exclude_prefixes', [])
        if isinstance(raw_exclude_prefixes, str):
            exclude_prefixes = [item.strip() for item in raw_exclude_prefixes.split(',') if item.strip()]
        elif isinstance(raw_exclude_prefixes, (list, tuple)):
            exclude_prefixes = [str(item).strip() for item in raw_exclude_prefixes if str(item).strip()]
        else:
            raise TypeError('resume_exclude_prefixes must be a string, list, or tuple')
        excluded_keys = []
        if exclude_prefixes:
            filtered_state_dict = {}
            for key, value in source_state_dict.items():
                if any(str(key).startswith(prefix) for prefix in exclude_prefixes):
                    excluded_keys.append(str(key))
                else:
                    filtered_state_dict[key] = value
            source_state_dict = filtered_state_dict

        target_state_dict = model_obj.state_dict()
        matched_state_dict = {}
        partially_copied = []
        skipped_shape = []
        unexpected = []
        exact_match_params = 0
        partial_copy_params = 0
        skipped_params = 0

        def _build_overlap_tensor(target_tensor, source_tensor):
            if (not isinstance(target_tensor, torch.Tensor)) or (not isinstance(source_tensor, torch.Tensor)):
                return None, 0
            if target_tensor.ndim != source_tensor.ndim or target_tensor.ndim == 0:
                return None, 0

            overlap_shape = tuple(min(int(s), int(t)) for s, t in zip(source_tensor.shape, target_tensor.shape))
            if any(dim <= 0 for dim in overlap_shape):
                return None, 0

            try:
                patched = target_tensor.detach().cpu().clone()
                source_cpu = source_tensor.detach().cpu().to(dtype=patched.dtype)
                overlap_slices = tuple(slice(0, dim) for dim in overlap_shape)
                patched[overlap_slices] = source_cpu[overlap_slices]
                overlap_numel = 1
                for dim in overlap_shape:
                    overlap_numel *= dim
                return patched, int(overlap_numel)
            except Exception:
                return None, 0

        for key, value in source_state_dict.items():
            if key not in target_state_dict:
                unexpected.append(key)
                continue

            target_value = target_state_dict[key]
            if not isinstance(value, torch.Tensor) or not isinstance(target_value, torch.Tensor):
                matched_state_dict[key] = value
                continue

            if tuple(value.shape) == tuple(target_value.shape):
                matched_state_dict[key] = value
                exact_match_params += int(value.numel())
            else:
                patched_value, overlap_numel = _build_overlap_tensor(target_value, value)
                if patched_value is not None and overlap_numel > 0:
                    matched_state_dict[key] = patched_value
                    partially_copied.append((key, tuple(value.shape), tuple(target_value.shape), overlap_numel))
                    partial_copy_params += int(overlap_numel)
                else:
                    skipped_shape.append((key, tuple(value.shape), tuple(target_value.shape)))
                    skipped_params += int(target_value.numel())

        incompatible = model_obj.load_state_dict(matched_state_dict, strict=False)
        return {
            'matched_keys': len(matched_state_dict),
            'exact_match_keys': len(matched_state_dict) - len(partially_copied),
            'exact_match_params': exact_match_params,
            'partial_copy_keys': len(partially_copied),
            'partial_copy_params': partial_copy_params,
            'matched_params': exact_match_params + partial_copy_params,
            'partially_copied': partially_copied,
            'skipped_shape': skipped_shape,
            'skipped_shape_keys': len(skipped_shape),
            'skipped_params': skipped_params,
            'excluded_keys': excluded_keys,
            'missing_keys': list(getattr(incompatible, 'missing_keys', [])),
            'unexpected_keys': list(getattr(incompatible, 'unexpected_keys', [])),
            'unexpected_ckpt_keys': unexpected,
        }

    resume_weights_only = bool(getattr(cfg, 'resume_weights_only', False))

    if getattr(cfg, 'resume', False):
        candidate = resume_path if resume_path else os.path.join(ckpt_dir, 'latest.pt')
        if not candidate or not os.path.exists(candidate):
            message = f'formal resume checkpoint was not found: {candidate or "<empty path>"}'
            if not resume_weights_only:
                raise RuntimeError(message)
            logger.error(message)
        if candidate and os.path.exists(candidate):
            try:
                resume_checkpoint = torch.load(candidate, map_location='cpu')
                logger.info(f"Loaded checkpoint from {candidate}")
            except (RuntimeError, FileNotFoundError) as e:
                if not resume_weights_only:
                    raise RuntimeError(f"Failed to load formal resume checkpoint {candidate}: {e}") from e
                logger.error(f"Failed to load weights-only checkpoint from {candidate}: {e}")
                resume_checkpoint = None

            if resume_checkpoint is not None and not isinstance(resume_checkpoint, dict):
                message = (
                    'resume checkpoint root must be a dict, '
                    f'got {type(resume_checkpoint).__name__}'
                )
                if not resume_weights_only:
                    raise RuntimeError(message)
                logger.error(message)
                resume_checkpoint = None

            if isinstance(resume_checkpoint, dict):
                format_version = int(resume_checkpoint.get('format_version', 0) or 0)
                strict_resume = (not resume_weights_only) and format_version >= CHECKPOINT_FORMAT_VERSION
                if strict_resume:
                    required_fields = (
                        'format_version', 'run_id', 'cfg_hash', 'saved_at',
                        'epoch', 'step', 'step_in_epoch', 'rng', 'state_dict',
                        'optimizer', 'scheduler',
                    )
                    missing_fields = [key for key in required_fields if key not in resume_checkpoint]
                    if missing_fields:
                        raise RuntimeError(
                            f"formal resume checkpoint is missing required fields: {missing_fields}"
                        )
                    inherited_transition = resume_checkpoint.get('resume_cfg_transition')
                    if isinstance(inherited_transition, dict):
                        resume_cfg_transition = dict(inherited_transition)
                    approved_transition = _validate_formal_resume_cfg_hash(
                        resume_checkpoint['cfg_hash'], cfg_hash
                    )
                    if approved_transition is not None:
                        resume_cfg_transition = approved_transition
                        logger.warning(
                            'Authorized strict-resume config transition: '
                            f"id={approved_transition['transition_id']} "
                            f"source={approved_transition['source_cfg_hash']} "
                            f"target={approved_transition['target_cfg_hash']}"
                        )
                    checkpoint_run_id = str(resume_checkpoint['run_id'])
                    configured_run_id = os.environ.get('TRAIN_RUN_ID', '').strip()
                    if not checkpoint_run_id:
                        raise RuntimeError('formal resume checkpoint has an empty run_id')
                    if configured_run_id and configured_run_id != checkpoint_run_id:
                        raise RuntimeError(
                            'formal resume run_id mismatch: '
                            f'checkpoint={checkpoint_run_id} configured={configured_run_id}'
                        )
                    run_id = checkpoint_run_id
                    _CRASH_CONTEXT['run_id'] = run_id

                state_dict = resume_checkpoint.get('state_dict')
                if state_dict is None:
                    state_dict = resume_checkpoint.get('model')
                if state_dict is None:
                    state_dict = resume_checkpoint

                from lib.checkpoints import normalize_state_dict
                state_dict = normalize_state_dict(state_dict)

                model_to_load = trainer.network.module if hasattr(trainer.network, 'module') else trainer.network
                if strict_resume:
                    if not isinstance(state_dict, dict):
                        raise RuntimeError('formal resume state_dict is not a dict')
                    model_to_load.load_state_dict(state_dict, strict=True)
                    logger.info('Strictly restored model state from formal checkpoint.')
                else:
                    try:
                        load_report = _safe_load_model_state(model_to_load, state_dict)
                        if load_report is None:
                            raise RuntimeError("checkpoint state_dict is not a dict")

                        logger.info(
                            "Partial checkpoint init: "
                            f"matched_keys={load_report['matched_keys']} "
                            f"exact_match_keys={load_report['exact_match_keys']} "
                            f"partial_copy_keys={load_report['partial_copy_keys']} "
                            f"matched_params={load_report['matched_params']} "
                            f"exact_match_params={load_report['exact_match_params']} "
                            f"partial_copy_params={load_report['partial_copy_params']} "
                            f"skipped_shape_keys={load_report['skipped_shape_keys']} "
                            f"skipped_params={load_report['skipped_params']} "
                            f"excluded_keys={len(load_report['excluded_keys'])} "
                            f"missing_after_load={len(load_report['missing_keys'])} "
                            f"unexpected_ckpt_keys={len(load_report['unexpected_ckpt_keys'])}"
                        )

                        # Architecture migrations may intentionally add a new
                        # module, but an explicitly fail-closed weights-only
                        # resume must never silently copy overlapping tensor
                        # slices or leave unrelated model keys uninitialized.
                        if (
                            resume_weights_only
                            and not bool(getattr(cfg, 'resume_allow_partial_copy', True))
                        ):
                            raw_allowed_missing = getattr(
                                cfg, 'resume_allowed_missing_prefixes', []
                            )
                            if isinstance(raw_allowed_missing, str):
                                allowed_missing_prefixes = [
                                    item.strip()
                                    for item in raw_allowed_missing.split(',')
                                    if item.strip()
                                ]
                            elif isinstance(raw_allowed_missing, (list, tuple)):
                                allowed_missing_prefixes = [
                                    str(item).strip()
                                    for item in raw_allowed_missing
                                    if str(item).strip()
                                ]
                            else:
                                raise TypeError(
                                    'resume_allowed_missing_prefixes must be a string, list, or tuple'
                                )
                            disallowed_missing = [
                                key for key in load_report['missing_keys']
                                if not any(
                                    str(key).startswith(prefix)
                                    for prefix in allowed_missing_prefixes
                                )
                            ]
                            migration_errors = []
                            if load_report['partial_copy_keys']:
                                migration_errors.append(
                                    f"partial_copy_keys={load_report['partial_copy_keys']}"
                                )
                            if load_report['skipped_shape_keys']:
                                migration_errors.append(
                                    f"skipped_shape_keys={load_report['skipped_shape_keys']}"
                                )
                            if load_report['unexpected_ckpt_keys']:
                                migration_errors.append(
                                    f"unexpected_ckpt_keys={len(load_report['unexpected_ckpt_keys'])}"
                                )
                            if load_report['unexpected_keys']:
                                migration_errors.append(
                                    f"unexpected_loaded_keys={len(load_report['unexpected_keys'])}"
                                )
                            if disallowed_missing:
                                migration_errors.append(
                                    f"disallowed_missing_keys={len(disallowed_missing)} "
                                    f"examples={disallowed_missing[:8]}"
                                )
                            if migration_errors:
                                raise RuntimeError(
                                    'Fail-closed weights-only migration rejected: '
                                    + '; '.join(migration_errors)
                                )
                            logger.info(
                                'Fail-closed weights-only migration PASS: '
                                f"allowed_missing={len(load_report['missing_keys'])} "
                                f"excluded_source={len(load_report['excluded_keys'])} "
                                'partial_copy=0 skipped_shape=0 unexpected=0'
                            )

                        critical_missing = [
                            k for k in load_report['missing_keys']
                            if any(x in k for x in ['gcn', 'locate_feat_replacer'])
                        ]
                        if load_report['matched_keys'] == 0:
                            logger.error("Checkpoint init matched zero parameter tensors.")
                        if critical_missing:
                            logger.warning(
                                f"Critical modules still partially uninitialized ({len(critical_missing)} keys): "
                                f"{critical_missing[:10]}..."
                            )
                        if load_report['skipped_shape_keys']:
                            logger.warning(
                                "Shape-mismatched checkpoint tensors were skipped: "
                                f"{load_report['skipped_shape'][:5]}..."
                            )
                        if load_report['partial_copy_keys']:
                            logger.info(
                                "Partially copied checkpoint tensors into larger target shapes: "
                                f"{load_report['partially_copied'][:5]}..."
                            )
                        if load_report['unexpected_ckpt_keys']:
                            logger.warning(
                                f"Unexpected checkpoint keys ({len(load_report['unexpected_ckpt_keys'])} total): "
                                f"{load_report['unexpected_ckpt_keys'][:5]}..."
                            )
                        if load_report['excluded_keys']:
                            logger.info(
                                f"Excluded checkpoint tensors ({len(load_report['excluded_keys'])} total): "
                                f"{load_report['excluded_keys'][:5]}..."
                            )
                    except RuntimeError as e:
                        logger.error(f"Failed to load source model state dict: {e}")

                if not resume_weights_only:
                    if 'optimizer' not in resume_checkpoint:
                        logger.warning('Source checkpoint has no optimizer state; using fresh optimizer.')
                    elif strict_resume:
                        optimizer.load_state_dict(resume_checkpoint['optimizer'])
                    elif not _safe_load_optimizer_state(optimizer, resume_checkpoint['optimizer']):
                        logger.warning('Source optimizer state was not compatible; using fresh optimizer.')

                if resume_weights_only:
                    resume_step = 0
                    resume_epoch = 0
                    resume_step_in_epoch = 0
                    logger.info("Weights-only resume enabled: optimizer, scheduler, and step state were reset.")
                else:
                    resume_step = int(resume_checkpoint.get('step', 0))
                    resume_epoch = int(resume_checkpoint.get('epoch', 0))
                    resume_step_in_epoch = int(resume_checkpoint.get('step_in_epoch', 0))
                    if resume_step < 0 or resume_epoch < 0 or resume_step_in_epoch < 0:
                        raise RuntimeError('resume checkpoint has negative progress metadata')
                    if strict_resume:
                        _restore_rng_state(resume_checkpoint['rng'])
                    logger.info(
                        f"Resuming from epoch={resume_epoch}, step={resume_step}, "
                        f"step_in_epoch={resume_step_in_epoch}, strict={strict_resume}"
                    )

    def _find_jsonl_truncate_pos(jsonl_path: str, keep_step: int):
        """Return byte position to truncate jsonl so that all remaining lines satisfy step <= keep_step."""
        try:
            keep_step = int(keep_step)
        except (ValueError, TypeError):
            return None
        if keep_step <= 0:
            return None
        if not jsonl_path or (not os.path.exists(jsonl_path)):
            return None
        pos = 0
        truncate_pos = None
        try:
            with open(jsonl_path, 'rb') as f:
                while True:
                    line_start = pos
                    line = f.readline()
                    if not line:
                        break
                    pos = f.tell()
                    try:
                        obj = json.loads(line.decode('utf-8'))
                    except (json.JSONDecodeError, UnicodeDecodeError) as e:
                        # Log corrupted lines for debugging
                        logger.debug(f"Skipping corrupted log line at pos {line_start}: {e}")
                        continue
                    try:
                        step_val = int(obj.get('step', -1))
                    except (ValueError, TypeError):
                        step_val = -1
                    if step_val > keep_step:
                        truncate_pos = line_start
                        break
        except IOError as e:
            logger.warning(f"Failed to read log file for truncation: {e}")
            return None
        return truncate_pos

    # If resuming, truncate logs.jsonl so new records overwrite the future part.
    # Default enabled unless ONE_SAMPLE_TRUNCATE_LOG=0
    truncate_log = str(os.environ.get('ONE_SAMPLE_TRUNCATE_LOG', '1')).strip().lower() not in ('0', 'false', 'no', 'off')

    safe_barrier(is_distributed)

    if is_main_process and truncate_log and int(resume_step) > 0:
        resume_json_pos = _find_jsonl_truncate_pos(log_path, int(resume_step))
        if resume_json_pos is not None:
            try:
                with open(log_path, 'rb+') as f:
                    f.truncate(int(resume_json_pos))
                logger.info(f"Truncated log file at position {resume_json_pos}")
            except IOError as e:
                logger.warning(f"Failed to truncate log file: {e}")

    safe_barrier(is_distributed)

    if is_main_process:
        json_logger = JsonLogger(log_path)
    else:
        json_logger = NullLogger()

    wandb_run = None
    if is_main_process and wandb is not None:
        try:
            wandb_project = os.environ.get('WANDB_PROJECT', 'DiffusionSnake')
            wandb_entity = os.environ.get('WANDB_ENTITY', None)
            wandb_name = os.environ.get('WANDB_NAME', None)
            wandb_dir = os.environ.get('WANDB_DIR', out_dir)
            wandb_run = wandb.init(
                project=wandb_project,
                entity=wandb_entity,
                name=wandb_name,
                dir=wandb_dir,
                resume='allow',
            )
            logger.info("WandB initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize WandB: {e}")
            wandb_run = None

    def _healthy_optimizer_step(
        network_obj,
        optimizer_obj,
        scheduler_obj,
        batch_obj,
        loss_obj,
        loss_stats_obj,
        phase_name,
        epoch_index,
        step_index,
    ):
        loss_obj = loss_obj.mean() if isinstance(loss_obj, torch.Tensor) else loss_obj
        loss_value = _scalar_float(loss_obj, 'total loss')
        stats_values = _loss_stats_to_floats(loss_stats_obj)
        if not _is_finite_value(stats_values):
            raise FloatingPointError('loss_stats contains a non-finite value')
        if not isinstance(loss_obj, torch.Tensor) or not loss_obj.requires_grad:
            raise RuntimeError('total loss does not require gradients')

        _CRASH_CONTEXT.update({
            'phase': str(phase_name),
            'epoch': int(epoch_index),
            'step_in_epoch': int(step_index),
        })
        optimizer_obj.zero_grad(set_to_none=True)
        loss_obj.backward()

        trainable_parameters = [
            parameter for parameter in network_obj.parameters() if parameter.requires_grad
        ]
        total_l2_sq = 0.0
        grad_max = 0.0
        for parameter in trainable_parameters:
            if parameter.grad is None:
                continue
            gradient = parameter.grad.detach()
            gradient_values = gradient.coalesce().values() if gradient.is_sparse else gradient
            if not bool(torch.isfinite(gradient_values).all().item()):
                raise FloatingPointError('gradient contains a non-finite value')
            total_l2_sq += float(torch.sum(gradient_values * gradient_values).item())
            grad_max = max(grad_max, float(gradient_values.abs().max().item()))
        preclip_l2 = float(total_l2_sq ** 0.5)
        adapter_before_metrics, adapter_before = _adapter_metrics(network_obj)
        clipped_norm = torch.nn.utils.clip_grad_norm_(
            trainable_parameters,
            max_norm=gradient_clip,
            error_if_nonfinite=True,
        )
        preclip_l2 = _scalar_float(clipped_norm, 'preclip gradient norm')
        clip_coefficient = 1.0 if preclip_l2 <= gradient_clip or preclip_l2 == 0.0 else gradient_clip / preclip_l2
        optimizer_obj.step()
        if scheduler_obj is not None:
            scheduler_obj.step()

        adapter_after_metrics, _ = _adapter_metrics(network_obj, before=adapter_before)
        lr_values = [
            _scalar_float(group.get('lr'), 'learning rate')
            for group in optimizer_obj.param_groups
        ]
        adapter_after_metrics['adapter_grad_l2'] = adapter_before_metrics['adapter_grad_l2']
        adapter_after_metrics['adapter_grad_max'] = adapter_before_metrics['adapter_grad_max']
        adapter_after_metrics['moonvit_adapter_grad_l2'] = adapter_before_metrics['adapter_grad_l2']
        adapter_after_metrics['moonvit_adapter_grad_max'] = adapter_before_metrics['adapter_grad_max']
        adapter_after_metrics['moonvit_adapter_weight_absmax'] = adapter_after_metrics['adapter_weight_absmax']
        adapter_after_metrics['moonvit_adapter_update_l2'] = adapter_after_metrics['adapter_update_l2']
        adapter_after_metrics['moonvit_adapter_update_max'] = adapter_after_metrics['adapter_update_max']
        health = {
            'loss': loss_value,
            'loss_stats': stats_values,
            'grad_l2': preclip_l2,
            'grad_max': float(grad_max),
            'preclip_grad_l2': preclip_l2,
            'preclip_grad_max': float(grad_max),
            'grad_clip_coefficient': float(clip_coefficient),
            'lr_min': min(lr_values) if lr_values else 0.0,
            'lr_max': max(lr_values) if lr_values else 0.0,
            'scheduler_last_epoch': int(scheduler_obj.last_epoch) if scheduler_obj is not None else -1,
            **adapter_after_metrics,
            **_batch_health_metrics(batch_obj),
            **_cuda_health_metrics(),
        }
        health['cuda_allocated'] = health['cuda_allocated_bytes']
        health['cuda_reserved'] = health['cuda_reserved_bytes']
        health['cuda_max_allocated'] = health['cuda_max_allocated_bytes']
        return health

    def _atomic_torch_save(payload, path):
        path = os.path.abspath(path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp_path = f'{path}.tmp.{os.getpid()}'
        try:
            with open(tmp_path, 'wb') as handle:
                torch.save(payload, handle)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_path, path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def _save_checkpoint(
        step: int,
        scheduler_obj,
        total_steps: int = None,
        epoch: int = None,
        step_in_epoch: int = 0,
        step_file: bool = False,
    ):
        if not is_main_process:
            return
        model_to_save = trainer.network.module if hasattr(trainer.network, 'module') else trainer.network
        ckpt = {
            'format_version': CHECKPOINT_FORMAT_VERSION,
            'run_id': str(run_id),
            'cfg_hash': str(cfg_hash),
            'saved_at': datetime.utcnow().isoformat(timespec='milliseconds') + 'Z',
            'state_dict': model_to_save.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler_obj.state_dict() if scheduler_obj is not None else None,
            'step': int(step),
            'step_in_epoch': int(step_in_epoch),
            'rng': _capture_rng_state(),
        }
        if resume_cfg_transition is not None:
            ckpt['resume_cfg_transition'] = dict(resume_cfg_transition)
        if epoch is not None:
            ckpt['epoch'] = int(epoch)
        if total_steps is not None:
            ckpt['total_steps'] = int(total_steps)

        if step_file:
            ckpt_path = os.path.join(ckpt_dir, f'step_{int(step)}.pt')
            _atomic_torch_save(ckpt, ckpt_path)
            step_paths = list(Path(ckpt_dir).glob('step_*.pt'))
            for old_path in _step_checkpoint_paths_to_remove(
                step_paths,
                step_checkpoint_keep,
                step_checkpoint_milestones,
            ):
                old_path.unlink()
        elif epoch is not None:
            ckpt_path = os.path.join(
                ckpt_dir,
                f'ep{int(epoch)}_st{int(step)}_{time.strftime("%m%d_%H%M")}.pt',
            )
            _atomic_torch_save(ckpt, ckpt_path)
        _atomic_torch_save(ckpt, os.path.join(ckpt_dir, 'latest.pt'))


    # 单阶段联合训练（余弦学习率调度器）。
    if True:
        logger.info("Starting epoch-based training loop")
        trainer.network.train()
        steps_per_epoch = 0
        try:
            steps_per_epoch = int(len(data_loader))
        except Exception:
            steps_per_epoch = 0
        if steps_per_epoch <= 0:
            raise RuntimeError('DataLoader is empty or has no __len__; cannot run epoch-based training.')

        try:
            num_epochs = int(getattr(cfg.train, 'epoch', 0))
        except Exception:
            num_epochs = 0
        if num_epochs <= 0:
            raise RuntimeError('cfg.train.epoch must be > 0 for epoch-based training.')

        total_steps = int(num_epochs * steps_per_epoch)
        max_steps_cfg = getattr(cfg.train, 'max_steps', 0)
        try:
            max_train_steps = int(max_steps_cfg)
        except Exception:
            max_train_steps = 0
        if max_train_steps > 0:
            total_steps = min(total_steps, max_train_steps)
            logger.info(f"Limiting training to absolute max_steps={max_train_steps}; stop_step={total_steps}")
        global_step = int(resume_step)
        if strict_resume:
            expected_step = int(resume_epoch * steps_per_epoch + resume_step_in_epoch)
            if global_step != expected_step:
                raise RuntimeError(
                    'formal resume progress is inconsistent: '
                    f'step={global_step}, epoch={resume_epoch}, '
                    f'step_in_epoch={resume_step_in_epoch}, '
                    f'steps_per_epoch={steps_per_epoch}'
                )
            start_epoch = int(resume_epoch)
            start_step_in_epoch = int(resume_step_in_epoch)
            if start_step_in_epoch >= steps_per_epoch:
                start_epoch += int(start_step_in_epoch // steps_per_epoch)
                start_step_in_epoch = int(start_step_in_epoch % steps_per_epoch)
        else:
            start_epoch = int(global_step // steps_per_epoch)
            start_step_in_epoch = int(global_step % steps_per_epoch)
        if global_step >= total_steps:
            logger.info(
                f"Checkpoint already reached stop_step={total_steps}; "
                "no optimization steps are required."
            )
            start_epoch = num_epochs
            start_step_in_epoch = 0

        try:
            eta_min = float(getattr(cfg.train, 'eta_min', 0.0))
        except Exception:
            eta_min = 0.0
        scheduler = None
        disable_scheduler = bool(getattr(cfg.train, 'disable_scheduler', False))
        if disable_scheduler:
            logger.info("LR scheduler disabled; using optimizer checkpoint/current LR.")
        scheduler_resume_state = None
        if (not resume_weights_only) and isinstance(resume_checkpoint, dict):
            scheduler_resume_state = resume_checkpoint.get('scheduler')
            if strict_resume and not disable_scheduler and scheduler_resume_state is None:
                raise RuntimeError('formal resume checkpoint has no scheduler state')

        scheduler_base_lrs = None
        if isinstance(scheduler_resume_state, dict):
            scheduler_base_lrs = scheduler_resume_state.get('base_lrs')
            if scheduler_base_lrs is None:
                nested_schedulers = scheduler_resume_state.get('_schedulers')
                if nested_schedulers and isinstance(nested_schedulers[0], dict):
                    scheduler_base_lrs = nested_schedulers[0].get('base_lrs')

        if global_step > 0:
            missing_initial_lrs = 0
            for index, param_group in enumerate(optimizer.param_groups):
                if 'initial_lr' in param_group:
                    continue
                missing_initial_lrs += 1
                fallback_lr = param_group.get('lr')
                if isinstance(scheduler_base_lrs, (list, tuple)) and index < len(scheduler_base_lrs):
                    fallback_lr = scheduler_base_lrs[index]
                param_group['initial_lr'] = _scalar_float(
                    fallback_lr,
                    f'initial learning rate for parameter group {index}',
                )
            if missing_initial_lrs:
                logger.warning(
                    'Resume optimizer checkpoint omitted initial_lr for '
                    f'{missing_initial_lrs} parameter groups; restored scheduler base_lrs when available.'
                )

        if total_steps > 0 and not disable_scheduler:
            warmup_steps = int(getattr(cfg.train, 'warmup_steps', 0))
            if warmup_steps > 0:
                # 1. 线性预热阶段：从基准 LR 的 1e-3 开始爬升
                warmup_sched = LinearLR(
                    optimizer, 
                    start_factor=1.0e-3, 
                    end_factor=1.0, 
                    total_iters=warmup_steps
                )
                # 2. 余弦退火主阶段：处理剩余的所有步数
                main_sched = CosineAnnealingLR(
                    optimizer,
                    T_max=max(1, int(total_steps - warmup_steps)),
                    eta_min=float(eta_min)
                )
                # 3. 组合通过 SequentialLR 实现无缝切换
                scheduler = SequentialLR(
                    optimizer, 
                    schedulers=[warmup_sched, main_sched], 
                    milestones=[warmup_steps],
                    last_epoch=int(global_step - 1)
                )
            else:
                # 若无预热步数，则维持纯余弦退火
                scheduler = CosineAnnealingLR(
                    optimizer,
                    T_max=max(1, int(total_steps)),
                    eta_min=float(eta_min),
                    last_epoch=int(global_step - 1),
                )
            if (not resume_weights_only) and isinstance(resume_checkpoint, dict):
                scheduler_state = resume_checkpoint.get('scheduler')
                if strict_resume and scheduler_state is None:
                    raise RuntimeError('formal resume checkpoint has no scheduler state')
                if scheduler_state is not None:
                    try:
                        scheduler.load_state_dict(scheduler_state)
                        restored_lrs = scheduler.get_last_lr()
                        if len(restored_lrs) != len(optimizer.param_groups):
                            raise RuntimeError(
                                'scheduler/optimizer parameter-group count mismatch: '
                                f'{len(restored_lrs)} != {len(optimizer.param_groups)}'
                            )
                        for param_group, lr in zip(optimizer.param_groups, restored_lrs):
                            param_group['lr'] = _scalar_float(lr, 'restored learning rate')
                        logger.info(f"Restored scheduler LR: {restored_lrs[0]:.8g}")
                    except Exception:
                        if strict_resume:
                            raise
                        logger.warning('Historical scheduler state was not compatible; using current scheduler.')

        for epoch in range(start_epoch, num_epochs):
            # distributed sampler epoch
            try:
                if hasattr(getattr(data_loader, 'batch_sampler', None), 'sampler') and hasattr(data_loader.batch_sampler.sampler, 'set_epoch'):
                    data_loader.batch_sampler.sampler.set_epoch(epoch)
            except Exception:
                pass

            # skip batches if resuming mid-epoch
            data_iter = iter(data_loader)
            if epoch == start_epoch and start_step_in_epoch > 0:
                for _ in range(start_step_in_epoch):
                    try:
                        next(data_iter)
                    except StopIteration:
                        data_iter = iter(data_loader)
                        next(data_iter)

            for step_in_epoch in range(start_step_in_epoch if epoch == start_epoch else 0, steps_per_epoch):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(data_loader)
                    batch = next(data_iter)
                batch = move_batch_to_cuda(batch)
                t0 = time.time()
                _CRASH_CONTEXT.update({
                    'phase': 'one_stage',
                    'step': int(global_step + 1),
                    'epoch': int(epoch),
                    'step_in_epoch': int(step_in_epoch + 1),
                    'loss': None,
                })

                output, loss, loss_stats, _ = trainer.network(batch)
                health = _healthy_optimizer_step(
                    trainer.network,
                    optimizer,
                    scheduler,
                    batch,
                    loss,
                    loss_stats,
                    'one_stage',
                    epoch,
                    step_in_epoch + 1,
                )

                global_step += 1
                _CRASH_CONTEXT.update({
                    'step': int(global_step),
                    'epoch': int(epoch),
                    'step_in_epoch': int(step_in_epoch + 1),
                    'loss': health['loss'],
                })
                dt = time.time() - t0
                stop_after_step = max_train_steps > 0 and global_step >= total_steps

                # periodic inference visualization every N steps
                viz_interval = int(os.environ.get('ONE_SAMPLE_VIZ_INTERVAL', '0'))
                do_viz = (viz_interval > 0 and (global_step % viz_interval == 0))
                if do_viz:
                    safe_barrier(is_distributed)
                    if is_main_process:
                        if viz_batch is not None:
                            infer_and_save(f"one_stage_step{global_step}")
                            trainer.network.train()
                    safe_barrier(is_distributed)

                entry = {
                    'run_id': str(run_id),
                    'timestamp': datetime.utcnow().isoformat(timespec='milliseconds') + 'Z',
                    'phase': 'one_stage',
                    'epoch': int(epoch),
                    'step_in_epoch': int(step_in_epoch + 1),
                    'steps_per_epoch': int(steps_per_epoch),
                    'step': int(global_step),
                    'total_steps': int(total_steps),
                    'time_ms': float(dt * TIME_SCALE_FACTOR),
                    **health,
                }
                json_logger.log(entry)

                if wandb_run is not None:
                    try:
                        wandb_payload = dict(entry)
                        loss_stats_dict = wandb_payload.pop('loss_stats', None)
                        if isinstance(loss_stats_dict, dict):
                            for k, v in loss_stats_dict.items():
                                wandb_payload[f"loss_stats/{k}"] = v
                        wandb.log(wandb_payload, step=int(global_step))
                    except Exception:
                        pass

                if step_checkpoint_every > 0 and global_step % step_checkpoint_every == 0:
                    safe_barrier(is_distributed)
                    if is_main_process:
                        _save_checkpoint(
                            global_step,
                            scheduler,
                            total_steps=total_steps,
                            epoch=epoch,
                            step_in_epoch=step_in_epoch + 1,
                            step_file=True,
                        )
                    safe_barrier(is_distributed)

                del output, loss, loss_stats, batch
                torch.cuda.empty_cache()
                if stop_after_step:
                    break

            # epoch-based checkpoint saving
            reached_max_steps = max_train_steps > 0 and global_step >= total_steps
            do_save_epoch = (
                reached_max_steps
                or (save_ep > 0 and ((epoch + 1) % save_ep == 0))
                or (epoch == (num_epochs - 1))
            )
            if do_save_epoch:
                safe_barrier(is_distributed)
                if is_main_process:
                    _save_checkpoint(
                        global_step,
                        scheduler,
                        total_steps=total_steps,
                        epoch=epoch,
                        step_in_epoch=step_in_epoch + 1,
                    )
                safe_barrier(is_distributed)

            start_step_in_epoch = 0
            if reached_max_steps:
                logger.info(f"Reached configured max_steps; stopping at step {global_step}.")
                break

    # Final inference visualization on affine-input coords
    if is_main_process:
        infer_and_save('final')

    # close logger
    if hasattr(json_logger, 'close'):
        json_logger.close()

    if is_main_process and wandb_run is not None:
        try:
            wandb.finish()
            logger.info("WandB run finished")
        except Exception as e:
            logger.warning(f"Failed to finish WandB run: {e}")

    safe_barrier(is_distributed)

    if is_distributed:
        try:
            dist.destroy_process_group()
        except Exception as e:
            logger.warning(f"Failed to destroy process group: {e}")


if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        _write_crash_event(exc)
        raise
