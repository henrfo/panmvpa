"""Find scans on disk and load them as timeseries over the analysis domain.

Rest runs build the maps and are split into four equal quarters. Task scans are the
held-out test set for identification -- they are never used to build a map at any level,
so the test set is identical at every point on the x-axis.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import nibabel as nib

from . import config

_SES_RUN = re.compile(r"_ses-(\d+)_.*_run-(\d+)_")


@dataclass(frozen=True)
class Scan:
    path: Path
    session: int
    run: int


def _present(path: Path) -> bool:
    """True if the file's content is actually on disk (annex symlinks may be broken)."""
    try:
        return path.resolve(strict=True).is_file() and path.stat().st_size > 0
    except (FileNotFoundError, OSError):
        return False


def rest_runs(subject: str) -> list[Scan]:
    """A subject's resting-state runs present on disk, ordered by (session, run)."""
    sid = config.sub_id(subject)
    pattern = (f"sub-{sid}/ses-*/func/sub-{sid}_ses-*_task-{config.REST_TASK}_run-*"
               f"_space-{config.SPACE}_desc-preproc_bold.nii.gz")
    out = []
    for p in sorted(config.DATA_ROOT.glob(pattern)):
        m = _SES_RUN.search(p.name)
        if m and _present(p):
            out.append(Scan(p, int(m.group(1)), int(m.group(2))))
    return sorted(out, key=lambda s: (s.session, s.run))


def task_scans(subject: str) -> list[Path]:
    """A subject's non-rest runs present on disk -- the held-out identification test set."""
    sid = config.sub_id(subject)
    pattern = (f"sub-{sid}/ses-*/func/sub-{sid}_ses-*_task-*"
               f"_space-{config.SPACE}_desc-preproc_bold.nii.gz")
    return [p for p in sorted(config.DATA_ROOT.glob(pattern))
            if f"_task-{config.REST_TASK}_" not in p.name and _present(p)]


def task_name(path: Path) -> str:
    """'..._task-msit_space-...' -> 'msit'."""
    for part in path.name.split("_"):
        if part.startswith("task-"):
            return part[len("task-"):]
    return "unknown"


@lru_cache(maxsize=64)
def run_minutes(path_str: str) -> float:
    """Duration in minutes, read from the header only."""
    return nib.load(path_str).shape[-1] * config.TR / 60.0


def quarters(subject: str) -> list[list[Scan]]:
    """Split a subject's rest runs, in order, into four near-equal groups."""
    runs = rest_runs(subject)
    if len(runs) < config.N_QUARTERS:
        raise ValueError(
            f"{config.sub_id(subject)}: only {len(runs)} rest runs on disk, "
            f"need at least {config.N_QUARTERS} to form quarters."
        )
    return [list(chunk) for chunk in np.array_split(np.array(runs, dtype=object),
                                                   config.N_QUARTERS)]


def runs_for(subject: str, quarter_ids: tuple[int, ...]) -> list[Scan]:
    """The runs making up a given combination of quarters."""
    qs = quarters(subject)
    return [run for q in quarter_ids for run in qs[q]]


def minutes_for(subject: str, quarter_ids: tuple[int, ...]) -> float:
    return sum(run_minutes(str(r.path)) for r in runs_for(subject, quarter_ids))


@lru_cache(maxsize=24)
def _load_masked(path_str: str) -> np.ndarray:
    """(n_voxels, n_time) z-scored timeseries over the fixed analysis domain."""
    from .parcellation import analysis_domain

    idx = analysis_domain()
    # dataobj as float32 rather than get_fdata()'s float64: half the transient memory.
    data = np.asarray(nib.load(path_str).dataobj, dtype=np.float32)
    ts = data[idx[0], idx[1], idx[2], :]
    del data
    mu = ts.mean(axis=1, keepdims=True)
    sd = ts.std(axis=1, keepdims=True)
    return np.divide(ts - mu, sd, out=np.zeros_like(ts), where=sd > 0)


def timeseries(runs: list[Scan]) -> np.ndarray:
    """Concatenate z-scored, domain-masked timeseries across runs -> (n_voxels, n_time).

    Each run is standardised before concatenation so run-level offsets and scale don't
    leak into the correlations. Returns a fresh array the caller may modify.
    """
    if not runs:
        raise ValueError("no runs to load")
    return np.concatenate([_load_masked(str(r.path)) for r in runs], axis=1)


def load_scan(path: Path) -> np.ndarray:
    """One scan as (n_voxels, n_time) z-scored over the analysis domain."""
    return _load_masked(str(path))


def clear_cache() -> None:
    """Drop cached timeseries (call between subjects to bound peak memory)."""
    _load_masked.cache_clear()
