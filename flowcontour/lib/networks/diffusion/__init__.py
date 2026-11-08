"""Flow-matching contour evolution used by the official mainline.

Keep the public evolution class lazy so architecture-only imports do not
eagerly parse the training command line through :mod:`lib.config`.
"""

from __future__ import annotations

from typing import Any


def __getattr__(name: str) -> Any:
    if name != "FlowMatchingEvolution":
        raise AttributeError(name)
    from .flow_matching_evolution import FlowMatchingEvolution

    return FlowMatchingEvolution


__all__ = ("FlowMatchingEvolution",)
