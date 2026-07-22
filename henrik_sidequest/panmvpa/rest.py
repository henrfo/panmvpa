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


@lru_cache(maxsize=16)
def timeline(subject: str) -> tuple[tuple[Scan, int, int], ...]:
    """Lay the runs end to end: (run, global_start, global_end) in timepoints."""
    out, t = [], 0
    for r in rest_runs(subject):
        n = int(round(run_minutes(str(r.path)) * 60.0 / config.TR))
        out.append((r, t, t + n))
        t += n
    return tuple(out)


def chunk_bounds(subject: str) -> list[tuple[int, int]]:
    """N_CHUNKS segments of *equal duration* over the concatenated rest timeline.

    Splitting by timepoints rather than by run count means every chunk holds exactly the
    same amount of data, so a data level means the same thing at every point on the
    x-axis and across subjects. The cost is that a chunk may straddle a run boundary,
    which is harmless for correlation-based estimates. Any remainder timepoints beyond
    the last whole chunk are dropped.
    """
    tl = timeline(subject)
    if not tl:
        raise ValueError(f"{config.sub_id(subject)}: no rest runs on disk.")
    total = tl[-1][2]
    per = total // config.N_CHUNKS
    if per < 1:
        raise ValueError(
            f"{config.sub_id(subject)}: only {total} rest timepoints, too few for "
            f"{config.N_CHUNKS} chunks."
        )
    return [(i * per, (i + 1) * per) for i in range(config.N_CHUNKS)]


def span_for(subject: str, spec: tuple[int, int]) -> tuple[int, int]:
    """Global timepoint range [start, end) covered by a contiguous block of chunks."""
    start, size = spec
    bounds = chunk_bounds(subject)
    return bounds[start][0], bounds[start + size - 1][1]


def minutes_for(subject: str, spec: tuple[int, int]) -> float:
    t0, t1 = span_for(subject, spec)
    return (t1 - t0) * config.TR / 60.0


def runs_for(subject: str, spec: tuple[int, int]) -> list[Scan]:
    """The runs that overlap a block — what actually has to be read off disk."""
    t0, t1 = span_for(subject, spec)
    return [r for r, s, e in timeline(subject) if s < t1 and e > t0]


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
    """Concatenate z-scored, domain-masked timeseries across whole runs.

    Each run is standardised before concatenation so run-level offsets and scale don't
    leak into the correlations. Returns a fresh array the caller may modify.
    """
    if not runs:
        raise ValueError("no runs to load")
    return np.concatenate([_load_masked(str(r.path)) for r in runs], axis=1)


def timeseries_for(subject: str, spec: tuple[int, int]) -> np.ndarray:
    """Exactly the timepoints of a chunk block -> (n_voxels, n_time).

    Only the runs overlapping the block are read, and each is sliced to its overlapping
    portion, so a block that straddles a run boundary still yields exactly the requested
    duration. Returns a fresh array the caller may modify.
    """
    t0, t1 = span_for(subject, spec)
    parts = []
    for run, start, end in timeline(subject):
        if start >= t1 or end <= t0:
            continue
        ts = _load_masked(str(run.path))
        lo, hi = max(t0, start) - start, min(t1, end) - start
        parts.append(ts[:, lo:hi])
    if not parts:
        raise ValueError(f"{config.sub_id(subject)}: no data for block {spec}")
    return np.concatenate(parts, axis=1)


def load_scan(path: Path) -> np.ndarray:
    """One scan as (n_voxels, n_time) z-scored over the analysis domain."""
    return _load_masked(str(path))


def clear_cache() -> None:
    """Drop cached timeseries (call between subjects to bound peak memory)."""
    _load_masked.cache_clear()
