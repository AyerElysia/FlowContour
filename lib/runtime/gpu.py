"""Fail-closed physical-GPU availability checks."""

from __future__ import annotations

from datetime import datetime, timezone
import subprocess
import time


def gpu_sample(gpu: int) -> dict:
    gpu = int(gpu)
    row = subprocess.check_output(
        [
            "nvidia-smi",
            f"--id={gpu}",
            "--query-gpu=index,memory.used,utilization.gpu,uuid",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    parts = [value.strip() for value in row.split(",")]
    if len(parts) != 4 or int(parts[0]) != gpu:
        raise RuntimeError(f"unexpected nvidia-smi row: {row!r}")
    uuid = parts[3]
    applications = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    compute_pids = []
    for line in applications.splitlines():
        fields = [value.strip() for value in line.split(",")]
        if len(fields) == 2 and fields[0] == uuid:
            compute_pids.append(int(fields[1]))
    return {
        "gpu": gpu,
        "memory_mib": int(parts[1]),
        "util_percent": int(parts[2]),
        "uuid": uuid,
        "compute_pids": compute_pids,
        "time": datetime.now(timezone.utc).isoformat(),
    }


def require_idle_gpu(
    gpu: int,
    *,
    wait_seconds: float = 15.0,
    max_memory_mib: int = 20,
) -> list[dict]:
    """Require two clean samples from one physical GPU before doing any work."""

    samples = [gpu_sample(gpu)]
    time.sleep(float(wait_seconds))
    samples.append(gpu_sample(gpu))
    for sample in samples:
        if (
            sample["memory_mib"] > int(max_memory_mib)
            or sample["util_percent"] != 0
            or sample["compute_pids"]
        ):
            raise RuntimeError(f"GPU{int(gpu)} is not strictly idle: {samples}")
    return samples


def acquire_gpu(gpu, allow_shared=False):
    """Keep the strict formal gate; shared headroom is an explicit smoke-only choice."""
    if not allow_shared:
        return require_idle_gpu(gpu)
    samples = [gpu_sample(gpu)]
    time.sleep(15)
    samples.append(gpu_sample(gpu))
    if any(s["memory_mib"] > 30000 or s["util_percent"] > 50 for s in samples):
        raise RuntimeError(f"GPU{gpu} lacks safe smoke headroom: {samples}")
    return samples
