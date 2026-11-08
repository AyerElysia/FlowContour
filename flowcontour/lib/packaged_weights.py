"""Load the detector stored inside the single FlowContour release checkpoint."""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
from pathlib import Path
import tempfile


@contextmanager
def detector_weight_path(checkpoint: str | Path):
    """Yield a YOLO-compatible temporary path from the bundled checkpoint.

    A standalone Ultralytics checkpoint is also accepted for training from a
    separately supplied detector. The temporary file is removed after YOLO
    has loaded its model into memory.
    """
    source = Path(checkpoint).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)

    import torch

    payload = torch.load(source, map_location="cpu", weights_only=False)
    detector_bytes = payload.get("detector_weight_bytes") if isinstance(payload, dict) else None
    if detector_bytes is None:
        yield source
        return
    expected = payload.get("detector_sha256")
    actual = hashlib.sha256(detector_bytes).hexdigest()
    if actual != expected:
        raise ValueError("Bundled YOLO26 detector checksum mismatch")
    del payload
    with tempfile.TemporaryDirectory(prefix="flowcontour_detector_") as temp_dir:
        extracted = Path(temp_dir) / "yolo26_best.pt"
        extracted.write_bytes(detector_bytes)
        yield extracted
