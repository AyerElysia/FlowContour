"""Checkpoint identity and compatibility helpers shared by all stages."""

from __future__ import annotations

import hashlib
import re
from collections import OrderedDict
from pathlib import Path


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def extract_state_dict(checkpoint: dict) -> dict:
    """Return a model state from any supported source envelope."""
    for key in ("state_dict", "model", "net", "network"):
        candidate = checkpoint.get(key)
        if isinstance(candidate, dict) and candidate:
            return candidate
    if checkpoint and all(hasattr(value, "shape") for value in checkpoint.values()):
        return checkpoint
    raise ValueError("checkpoint does not contain a recognized model state")


def normalize_state_dict(state_dict: dict) -> OrderedDict:
    """Remove DDP prefixes and apply the signed-source time-key rename."""
    source_time = re.compile(r"(\.?)time_emb_(\d)(\..*)")
    normalized = OrderedDict()
    for raw_key, value in state_dict.items():
        key = str(raw_key)
        while key.startswith("module."):
            key = key[len("module.") :]
        match = source_time.search(key)
        if match:
            key = source_time.sub(r"\1time_emb_net.\2\3", key)
        if key in normalized:
            raise ValueError(f"checkpoint key collision after normalization: {key}")
        normalized[key] = value
    return normalized
