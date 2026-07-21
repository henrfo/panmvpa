"""Enumerate and load resting-state runs, concatenated to a target amount of data.

The parcellation x-axis is *minutes of rest*. We concatenate whole runs (each ~5 min)
in a deterministic session/run order until the target is reached. Each run is masked to
the cortical analysis domain (the group Schaefer coverage) and z-scored per run before
concatenation, so run-level offsets/scale don't leak into the connectivity estimates.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import nibabel as nib

from . import config

_SES_RUN = re.compile(r"_ses-(\d+)_.*_run-(\d+)_")

# Each cached run is ~n_domain_voxels * n_timepoints * 4 B (~120 MB at 132k voxels /
# 222 TRs), and the concatenated timeseries at the top data level costs about as much
# again. Caching a whole 20-run working set therefore roughly doubles peak memory, so
# the default trades some re-reading for headroom on a 15 GB box. Raise it if you have
# RAM to spare, lower it (e.g. 8) if you are still tight.
REST_CACHE_RUNS = int(os.environ.get("PANMVPA_REST_CACHE", "10"))


@dataclass(frozen=True)
class RestRun:
    session: int
    run: int
    path: Path

    @property
    def key(self) -> tuple[int, int]:
        return (self.session, self.run)


def rest_runs(subject: str, fetched_only: bool = False) -> list[RestRun]:
    """All rest runs for a subject, sorted by (session, run).

    With ``fetched_only`` we keep only runs whose BOLD content is actually on disk
    (datalad annex symlinks that haven't been ``get``-ed are dropped).
    """
    runs: list[RestRun] = []
    for p in sorted(config.DATA_ROOT.glob(config.rest_glob(subject))):
        m = _SES_RUN.search(p.name)
        if not m:
            continue
        if fetched_only and not _is_content_present(p):
            continue
        runs.append(RestRun(int(m.group(1)), int(m.group(2)), p))
    return sorted(runs, key=lambda r: r.key)


def _is_content_present(path: Path) -> bool:
    """True if the annexed BOLD content is present (symlink resolves to a real file)."""
    try:
        return path.resolve(strict=True).is_file() and path.stat().st_size > 0
    except (FileNotFoundError, OSError):
        return False


def run_minutes(run: RestRun) -> float:
    """Duration of a run in minutes (n_volumes * TR)."""
    n = nib.load(str(run.path)).shape[-1]
    return n * config.TR / 60.0


def select_runs(subject: str, minutes: float, runs: list[RestRun] | None = None) -> list[RestRun]:
    """Fewest leading runs whose cumulative duration reaches ``minutes``.

    Raises if the subject doesn't have enough fetched rest data for the target.
    """
    runs = runs if runs is not None else rest_runs(subject, fetched_only=True)
    chosen: list[RestRun] = []
    total = 0.0
    for r in runs:
        if total >= minutes:
            break
        chosen.append(r)
        total += run_minutes(r)
    if total < minutes - 1e-6:
        raise ValueError(
            f"{config.sub_id(subject)}: only {total:.1f} min of fetched rest available, "
            f"need {minutes:.0f} min ({len(runs)} runs on disk)."
        )
    return chosen


@lru_cache(maxsize=REST_CACHE_RUNS)
def _load_masked_run(path_str: str, domain_hash: int) -> np.ndarray:
    """(n_voxels, n_timepoints) z-scored timeseries for one run over the analysis domain.

    Cached per path so repeated parcellation builds reload each run at most once per
    process. ``domain_hash`` keys the cache to the domain mask actually in use.
    """
    from .parcellation import analysis_domain  # local import to avoid import cycle

    idx = analysis_domain()  # (3, n_voxels) voxel coordinates, fixed group domain
    # dataobj + explicit float32 keeps the transient whole-volume array half the size
    # get_fdata()'s float64 would be, which matters on a 15 GB box.
    data = np.asarray(nib.load(path_str).dataobj, dtype=np.float32)
    ts = data[idx[0], idx[1], idx[2], :]  # (n_voxels, n_time)
    del data
    # z-score each voxel over time; flat voxels (std 0) -> 0 so they never dominate.
    mu = ts.mean(axis=1, keepdims=True)
    sd = ts.std(axis=1, keepdims=True)
    ts = np.divide(ts - mu, sd, out=np.zeros_like(ts), where=sd > 0)
    return ts


def clear_cache() -> None:
    """Drop cached run timeseries (call between subjects to bound peak memory)."""
    _load_masked_run.cache_clear()


def masked_timeseries(runs: list[RestRun]) -> np.ndarray:
    """Concatenate z-scored, domain-masked timeseries across runs -> (n_voxels, n_time)."""
    from .parcellation import analysis_domain

    dh = hash(analysis_domain().tobytes())
    parts = [_load_masked_run(str(r.path), dh) for r in runs]
    return np.concatenate(parts, axis=1)
