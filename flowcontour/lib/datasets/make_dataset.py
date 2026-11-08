"""Data-loader construction for the single VerSe sagittal data path."""

from __future__ import annotations

import torch

from .collate_batch import make_collator
from .dataset_catalog import DatasetCatalog
from .sagittal_2d_fixed.snake import Dataset


torch.multiprocessing.set_sharing_strategy("file_system")


def make_dataset(cfg, dataset_name: str, is_train: bool = True) -> Dataset:
    del is_train
    attributes = DatasetCatalog.get(dataset_name)
    data_source = attributes.pop("id")
    if data_source != "sagittal_2d_fixed" or cfg.task != "snake":
        raise ValueError("the mainline only supports sagittal_2d_fixed/snake")
    return Dataset(**attributes)


def make_data_loader(cfg, is_train: bool = True, is_distributed: bool = False):
    dataset_name = cfg.train.dataset if is_train else cfg.test.dataset
    dataset = make_dataset(cfg, dataset_name, is_train=is_train)
    shuffle = bool(is_train)

    if is_distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, shuffle=shuffle
        )
    elif shuffle:
        sampler = torch.utils.data.RandomSampler(dataset)
    else:
        sampler = torch.utils.data.SequentialSampler(dataset)

    batch_size = int(
        cfg.train.batch_size if is_train else cfg.test.batch_size
    )
    drop_last = bool(cfg.train.drop_last) if is_train else False
    batch_sampler = torch.utils.data.BatchSampler(
        sampler, batch_size=batch_size, drop_last=drop_last
    )

    num_workers = int(cfg.train.num_workers)
    loader_options = {
        "batch_sampler": batch_sampler,
        "num_workers": num_workers,
        "collate_fn": make_collator(cfg),
        "pin_memory": True,
    }
    if num_workers > 0:
        loader_options["persistent_workers"] = bool(
            cfg.dataloader_persistent_workers
        )
        prefetch_factor = int(cfg.dataloader_prefetch_factor)
        if prefetch_factor > 0:
            loader_options["prefetch_factor"] = prefetch_factor

    return torch.utils.data.DataLoader(dataset, **loader_options)
