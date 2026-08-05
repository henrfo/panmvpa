"""Build a personal map: winner-take-all assignment to the Yeo-17 group networks.

The group atlas is a fixed spatial anchor. It is regridded once to the BOLD grid and
collapsed to 17 network regions, and those regions never change -- not across data levels,
not across subjects. What changes with data is the *signal* averaged within each region,
and therefore which voxels defect to which network.

For one set of rest runs:

1. Load them, keep the analysis-domain voxels, z-score each voxel per run, concatenate.
2. Average the timeseries within each of the 17 fixed group regions -> 17 references.
3. Correlate every voxel against all 17 references and assign it to the best match.

Maps are ~260 KB, so they persist while raw BOLD streams through and is deleted.
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import nibabel as nib
from nilearn.image import resample_to_img

from . import config, rest

_NET = re.compile(r"17Networks_(?:LH|RH)_([A-Za-z]+)")


# --------------------------------------------------------------------- group anchor
@lru_cache(maxsize=1)
def network_order() -> tuple[str, ...]:
    """The 17 network names, in first-appearance order in the atlas lookup table."""
    seen: list[str] = []
    for name in _parcel_names().values():
        net = _net_of(name)
        if net and net not in seen:
            seen.append(net)
    return tuple(seen)


@lru_cache(maxsize=1)
def _parcel_names() -> dict[int, str]:
    names: dict[int, str] = {}
    for line in config.ATLAS_ORDER.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0].isdigit():
            names[int(parts[0])] = parts[1]
    return names


def _net_of(parcel_name: str) -> str | None:
    m = _NET.search(parcel_name)
    return m.group(1) if m else None


def _reference_bold() -> str:
    """Any preprocessed BOLD, used only to define the target grid (all share it)."""
    for sub in config.SUBJECTS:
        runs = rest.rest_runs(sub)
        if runs:
            return str(runs[0].path)
        scans = rest.task_scans(sub)
        if scans:
            return str(scans[0])
    raise FileNotFoundError("No BOLD on disk to define the resampling grid.")


@lru_cache(maxsize=1)
def grid() -> tuple[tuple[int, int, int], np.ndarray]:
    """(shape, affine) of the analysis grid, without needing any BOLD on disk.

    The geometry is fixed and known (config.GRID_SHAPE / GRID_AFFINE), so every stage
    after `maps` works from the atlas alone -- which matters because `--cleanup` deletes
    the BOLD. A BOLD header is consulted only to cross-check when one happens to be
    present, never as a requirement.
    """
    return tuple(config.GRID_SHAPE), np.array(config.GRID_AFFINE, dtype=float)


def grid_template() -> nib.Nifti1Image:
    """An empty image carrying the analysis grid, as a resampling target."""
    shape, affine = grid()
    return nib.Nifti1Image(np.zeros(shape, dtype=np.int16), affine)


@lru_cache(maxsize=1)
def group_networks() -> np.ndarray:
    """3D array on the analysis grid: 0 outside cortex, 1..17 group network id.

    Loaded from the cache when it exists, otherwise built from the atlas and cached.
    """
    if config.GROUP_MAP_CACHE.exists() and config.DOMAIN_CACHE.exists():
        vol = np.zeros(tuple(config.GRID_SHAPE), dtype=np.int16)
        idx = np.load(config.DOMAIN_CACHE)
        vol[idx[0], idx[1], idx[2]] = np.load(config.GROUP_MAP_CACHE).astype(np.int16)
        return vol
    return build_group_networks()


def build_group_networks() -> np.ndarray:
    """Regrid the atlas onto the analysis grid and collapse it to 17 networks."""
    atlas = resample_to_img(nib.load(str(config.ATLAS_IMAGE)), grid_template(),
                            interpolation="nearest", force_resample=True,
                            copy_header=True)
    parcels = np.asarray(atlas.get_fdata()).astype(np.int32)
    out = np.zeros(parcels.shape, dtype=np.int16)
    order = network_order()
    for pid, pname in _parcel_names().items():
        net = _net_of(pname)
        if net:
            out[parcels == pid] = order.index(net) + 1
    return out


def write_grid_cache() -> tuple[Path, Path]:
    """Persist the analysis domain and group map so later stages need no BOLD."""
    vol = build_group_networks()
    idx = np.array(np.nonzero(vol > 0)).astype(np.int64)
    labels = vol[idx[0], idx[1], idx[2]].astype(np.int16)
    config.DOMAIN_CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.save(config.DOMAIN_CACHE, idx)
    np.save(config.GROUP_MAP_CACHE, labels)
    return config.DOMAIN_CACHE, config.GROUP_MAP_CACHE


@lru_cache(maxsize=1)
def analysis_domain() -> np.ndarray:
    """(3, n_voxels) coordinates of the cortical voxels we model. Fixed for all analyses."""
    if config.DOMAIN_CACHE.exists():
        return np.load(config.DOMAIN_CACHE).astype(np.int64)
    return np.array(np.nonzero(group_networks() > 0)).astype(np.int64)


@lru_cache(maxsize=1)
def _domain_group_labels() -> np.ndarray:
    """(n_voxels,) the fixed group network id of each domain voxel."""
    if config.GROUP_MAP_CACHE.exists():
        return np.load(config.GROUP_MAP_CACHE).astype(np.int64)
    idx = analysis_domain()
    return group_networks()[idx[0], idx[1], idx[2]].astype(np.int64)


# --------------------------------------------------------------------- map building
def _standardize_inplace(x: np.ndarray) -> np.ndarray:
    """Z-score rows without allocating a second copy (the matrix is multi-GB)."""
    mu = x.mean(axis=1, keepdims=True)
    sd = x.std(axis=1, keepdims=True)
    np.subtract(x, mu, out=x)
    np.divide(x, sd, out=x, where=sd > 0)
    x[(sd <= 0).ravel()] = 0.0
    return x


def winner_take_all(ts: np.ndarray) -> np.ndarray:
    """Assign each domain voxel to its most-correlated group reference -> labels 1..17.

    ``ts`` is standardised in place to keep peak memory down, so pass an array you own
    (``rest.timeseries`` always returns a fresh concatenation).
    """
    group = _domain_group_labels()
    n_time = ts.shape[1]

    references = np.zeros((config.N_NETWORKS, n_time), dtype=np.float32)
    for k in range(1, config.N_NETWORKS + 1):
        members = group == k
        if members.any():
            references[k - 1] = ts[members].mean(axis=0)

    z = _standardize_inplace(ts)
    zr = references - references.mean(axis=1, keepdims=True)
    sd = references.std(axis=1, keepdims=True)
    zr = np.divide(zr, sd, out=np.zeros_like(zr), where=sd > 0)

    corr = (z @ zr.T) / n_time            # (n_voxels, 17)
    return corr.argmax(axis=1).astype(np.int16) + 1


def build_map(subject: str, spec: tuple[int, int]) -> np.ndarray:
    """The personal map from a contiguous block of a subject's rest chunks.

    Chunks are equal-duration slices of the concatenated timeline, so every block holds
    exactly (size/N_CHUNKS) of the subject's rest regardless of how runs divide up.
    """
    return winner_take_all(rest.timeseries_for(subject, spec))


# --------------------------------------------------------------------- persistence
def map_path(subject: str, spec: tuple[int, int]) -> Path:
    return config.MAPS_DIR / f"sub-{config.sub_id(subject)}_{config.map_key(spec)}.npy"


def save_map(labels: np.ndarray, subject: str, spec: tuple[int, int]) -> Path:
    path = map_path(subject, spec)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, labels.astype(np.int16))
    return path


def load_map(subject: str, spec: tuple[int, int]) -> np.ndarray:
    return np.load(map_path(subject, spec)).astype(np.int64)


def has_map(subject: str, spec: tuple[int, int]) -> bool:
    return map_path(subject, spec).exists()


def cohort(spec: tuple[int, int]) -> dict[str, np.ndarray]:
    """{subject: map} for every subject with a stored map at this level."""
    return {s: load_map(s, spec) for s in config.SUBJECTS if has_map(s, spec)}


# --------------------------------------------------------------------- comparison
def dice_per_network(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Dice for each of the 17 networks separately -> array of length 17.

    Reported alongside the mean because Dice runs systematically lower for small
    networks, so a rising mean could in principle be driven by the large ones alone.
    NaN where a network is absent from both maps.
    """
    out = np.full(config.N_NETWORKS, np.nan)
    for k in range(1, config.N_NETWORKS + 1):
        ak, bk = (a == k), (b == k)
        denom = ak.sum() + bk.sum()
        if denom:
            out[k - 1] = 2.0 * np.logical_and(ak, bk).sum() / denom
    return out


def map_dice(a: np.ndarray, b: np.ndarray) -> float:
    """Agreement between two whole maps: mean Dice over the 17 networks."""
    per = dice_per_network(a, b)
    return float(np.nanmean(per)) if np.isfinite(per).any() else float("nan")


def group_map() -> np.ndarray:
    """The fixed group parcellation over the analysis domain, as a comparable 'map'.

    Used as the reference for similarity-to-group: a noisy individual map has barely
    moved away from this, so early on it scores high and should fall as real individual
    structure emerges.
    """
    return _domain_group_labels()


@lru_cache(maxsize=1)
def group_network_sizes() -> dict[str, int]:
    """Voxel count of each network in the fixed group atlas, for size-vs-Dice checks."""
    labels = _domain_group_labels()
    order = network_order()
    return {order[k - 1]: int((labels == k).sum()) for k in range(1, config.N_NETWORKS + 1)}


def example_networks() -> list[str]:
    """A small, a medium and a large network, to check all sizes rise with data."""
    ranked = sorted(group_network_sizes().items(), key=lambda kv: kv[1])
    return [ranked[0][0], ranked[len(ranked) // 2][0], ranked[-1][0]]


def to_image(labels: np.ndarray) -> nib.Nifti1Image:
    """Scatter domain labels back into a 3D NIfTI, for viewing a map. Needs no BOLD."""
    shape, affine = grid()
    vol = np.zeros(shape, dtype=np.int16)
    idx = analysis_domain()
    vol[idx[0], idx[1], idx[2]] = labels.astype(np.int16)
    return nib.Nifti1Image(vol, affine)
