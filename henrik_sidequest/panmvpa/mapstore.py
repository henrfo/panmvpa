"""Persist individualised parcellations so raw scans can be deleted after pass 1.

A map is 132k int16 labels (~260 KB), so the whole cohort -- 10 subjects x 5 data levels
x n_seeds -- is a few hundred MB even at 10 seeds. That is what makes the streaming
workflow possible: raw BOLD is transient, maps are the durable artefact, and pass 2 can
score any scan against every subject long after their scans are gone.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import config


def maps_dir(root: Path | None = None) -> Path:
    return Path(root) if root is not None else config.DERIV_ROOT / "maps"


def map_path(subject: str, minutes: float, seed: int, root: Path | None = None) -> Path:
    sid = config.sub_id(subject)
    return maps_dir(root) / f"sub-{sid}_min-{config.level_key(minutes)}_seed-{seed:02d}.npy"


def save_map(labels: np.ndarray, subject: str, minutes: float, seed: int,
             root: Path | None = None) -> Path:
    path = map_path(subject, minutes, seed, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, labels.astype(np.int16))
    return path


def load_map(subject: str, minutes: float, seed: int, root: Path | None = None) -> np.ndarray:
    return np.load(map_path(subject, minutes, seed, root)).astype(np.int64)


def has_map(subject: str, minutes: float, seed: int, root: Path | None = None) -> bool:
    return map_path(subject, minutes, seed, root).exists()


def available_subjects(minutes: float, seed: int, root: Path | None = None) -> list[str]:
    """Subjects with a stored map at this level/seed (i.e. that finished pass 1)."""
    return sorted(
        s for s in config.SUBJECTS if has_map(s, minutes, seed, root)
    )


def load_cohort(minutes: float, seed: int, root: Path | None = None) -> dict[str, np.ndarray]:
    """{subject: labels} for every subject with a stored map at this level/seed."""
    return {s: load_map(s, minutes, seed, root) for s in available_subjects(minutes, seed, root)}
