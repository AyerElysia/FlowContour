"""Dataset registry for the VerSe sagittal mainline."""

from __future__ import annotations

import os
from pathlib import Path


def _dataset_paths() -> tuple[str, str]:
    """Resolve data from environment variables, with repository-local defaults."""
    root = Path(os.environ.get("FLOWCONTOUR_DATA_ROOT", "data/verse_sagittal"))
    manifest = Path(
        os.environ.get(
            "FLOWCONTOUR_SLICE_MANIFEST",
            str(root / "manifests" / "slice_manifest.csv"),
        )
    )
    return str(root), str(manifest)


def _entry(split: str) -> dict[str, str]:
    root, manifest = _dataset_paths()
    return {
        "id": "sagittal_2d_fixed",
        "data_root": root,
        "ann_file": manifest,
        "split": split,
    }


class DatasetCatalog:
    """The four sequence-safe VerSe splits used by training and evaluation."""

    dataset_attrs = {
        "RAOSTrain": _entry("train"),
        "RAOSVal": _entry("val"),
        "RAOSTest": _entry("test"),
    }

    @classmethod
    def refresh_paths(cls) -> None:
        """Apply environment overrides set after module import."""
        for name, split in (
            ("RAOSTrain", "train"),
            ("RAOSVal", "val"),
            ("RAOSTest", "test"),
        ):
            cls.dataset_attrs[name] = _entry(split)

    @classmethod
    def get(cls, name: str) -> dict:
        if name not in cls.dataset_attrs:
            raise KeyError(
                f"unknown mainline dataset {name!r}; "
                f"available={sorted(cls.dataset_attrs)}"
            )
        return cls.dataset_attrs[name].copy()
