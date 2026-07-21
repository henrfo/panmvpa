"""Enumerate and load resting-state runs, concatenated to a target amount of data.

The parcellation x-axis is *minutes of rest*. We concatenate whole runs (each ~5 min)
in a deterministic session/run order until the target is reached. Each run is masked to
the cortical analysis domain (the group Schaefer coverage) and z-scored per run before
concatenation, so run-level offsets/scale don't leak into the connectivity estimates.
"""
from __future__ import annotations

import os
import re
import zlib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import nibabel as nib

from . import config

_SES_RUN = re.compile(r"_ses-(\d+)_.*_run-(\d+)_")

# Each cached run is ~n_domain_voxels * n_timepoints * 4 B (~120 MB at 132k voxels /
# 222 TRs). With seed-based sampling every seed draws a *different* subset of the same
# per-subject pool, so a cache smaller than the pool is defeated: each build re-reads
# ~20 runs off disk and the run becomes I/O bound. Sizing the cache to a subject's whole
# pool (21-33 runs) instead makes every build after the first pure arithmetic. Trade-off
# is ~120 MB per cached run; lower it if the hub is tight and accept slower runs.
REST_CACHE_RUNS = int(os.environ.get("PANMVPA_REST_CACHE", "24"))


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


@lru_cache(maxsize=1024)
def _run_minutes_cached(path_str: str) -> float:
    return nib.load(path_str).shape[-1] * config.TR / 60.0


def run_minutes(run: RestRun) -> float:
    """Duration of a run in minutes (n_volumes * TR). Header-only, cached."""
    return _run_minutes_cached(str(run.path))


def _take_until(runs: list[RestRun], order, minutes: float, subject: str) -> list[RestRun]:
    chosen: list[RestRun] = []
    total = 0.0
    for i in order:
        if total >= minutes:
            break
        chosen.append(runs[i])
        total += run_minutes(runs[i])
    if total < minutes - 1e-6:
        raise ValueError(
            f"{config.sub_id(subject)}: only {total:.1f} min of fetched rest available, "
            f"need {minutes:.0f} min ({len(runs)} runs on disk)."
        )
    return chosen


def select_runs(subject: str, minutes: float, runs: list[RestRun] | None = None) -> list[RestRun]:
    """Fewest *leading* runs whose cumulative duration reaches ``minutes``."""
    runs = runs if runs is not None else rest_runs(subject, fetched_only=True)
    return _take_until(runs, range(len(runs)), minutes, subject)


def sample_runs(
    subject: str,
    minutes: float,
    seed: int,
    runs: list[RestRun] | None = None,
) -> list[RestRun]:
    """A random subset of runs totalling ``minutes``, reproducible from ``seed``.

    Different seeds give slightly different parcellations, which is what turns a single
    point estimate into a per-subject distribution. The draw is decorrelated across
    subject and data level (not just seed) so level 20 seed 0 and level 40 seed 0 are not
    nested draws of the same shuffle.

    Caveat: when a subject barely has enough rest for the target, every subset is nearly
    the same set of runs, so seed-to-seed spread shrinks for reasons that have nothing to
    do with the estimate being more stable. See ``sampling_headroom``.
    """
    runs = runs if runs is not None else rest_runs(subject, fetched_only=True)
    rng = np.random.default_rng([zlib.crc32(config.sub_id(subject).encode()),
                                 int(round(minutes)), int(seed)])
    return _take_until(runs, rng.permutation(len(runs)), minutes, subject)


def runs_for(
    subject: str,
    minutes: float,
    seed: int | None = None,
    runs: list[RestRun] | None = None,
) -> list[RestRun]:
    """Leading runs when ``seed`` is None, otherwise a random subset for that seed."""
    if seed is None:
        return select_runs(subject, minutes, runs=runs)
    return sample_runs(subject, minutes, seed, runs=runs)


def sampling_headroom(subject: str, minutes: float) -> dict:
    """How much freedom the random draw actually has at this data level.

    ``spare_runs`` is how many runs beyond the required number exist; at 0 every seed
    draws essentially the same subset and per-subject error bars collapse.
    """
    runs = rest_runs(subject, fetched_only=True)
    total = sum(run_minutes(r) for r in runs)
    needed = len(select_runs(subject, minutes, runs=runs)) if total >= minutes else None
    return {
        "n_runs": len(runs),
        "total_minutes": total,
        "runs_needed": needed,
        "spare_runs": (len(runs) - needed) if needed is not None else None,
    }


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
