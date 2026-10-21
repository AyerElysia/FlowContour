"""Construct the supervised MoonViT + Flow trainer."""

from __future__ import annotations

import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from .diffusion_trainer import DiffusionPretrainNetworkWrapper
from .trainer import Trainer


def _wrapper_factory(cfg, network):
    """Wrap a mainline network with its supervised Flow objective."""
    if not bool(getattr(cfg, "use_diffusion_trainer", False)):
        raise ValueError("the mainline package requires use_diffusion_trainer=true")
    if bool(getattr(cfg, "use_gdpo", False)) or bool(getattr(cfg, "use_grpo", False)):
        raise ValueError("GDPO uses the dedicated RL entry point")
    return DiffusionPretrainNetworkWrapper(network)


def make_trainer(cfg, network, distributed=False, local_rank=None):
    wrapped = _wrapper_factory(cfg, network)
    if distributed:
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            raise RuntimeError("distributed=True requires an initialized process group")
        if local_rank is None:
            local_rank = torch.distributed.get_rank()
        wrapped = DDP(
            wrapped,
            device_ids=[int(local_rank)],
            output_device=int(local_rank),
            broadcast_buffers=True,
            bucket_cap_mb=int(getattr(cfg, "ddp_bucket_cap_mb", 25)),
            find_unused_parameters=bool(getattr(cfg, "ddp_find_unused_parameters", True)),
            gradient_as_bucket_view=bool(getattr(cfg, "ddp_gradient_as_bucket_view", True)),
        )
    return Trainer(wrapped)
