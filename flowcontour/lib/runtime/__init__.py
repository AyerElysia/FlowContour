"""Small runtime guards shared by launchers and inference."""

from .gpu import gpu_sample, require_idle_gpu

__all__ = ["gpu_sample", "require_idle_gpu"]
