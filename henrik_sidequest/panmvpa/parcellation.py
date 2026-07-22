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
def group_networks() -> np.ndarray:
    """3D array on the BOLD grid: 0 outside cortex, 1..17 group network id."""
    ref = nib.load(_reference_bold())
    atlas = resample_to_img(nib.load(str(config.ATLAS_IMAGE)), ref,
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


@lru_cache(maxsize=1)
def analysis_domain() -> np.ndarray:
    """(3, n_voxels) coordinates of the cortical voxels we model. Fixed for all analyses."""
    return np.array(np.nonzero(group_networks() > 0)).astype(np.int64)


@lru_cache(maxsize=1)
def _domain_group_labels() -> np.ndarray:
    """(n_voxels,) the fixed group network id of each domain voxel."""
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


def build_map(subject: str, quarter_ids: tuple[int, ...]) -> np.ndarray:
    """The personal map from a given combination of a subject's rest quarters."""
    return winner_take_all(rest.timeseries(rest.runs_for(subject, quarter_ids)))


# --------------------------------------------------------------------- persistence
def map_path(subject: str, quarter_ids: tuple[int, ...]) -> Path:
    return config.MAPS_DIR / f"sub-{config.sub_id(subject)}_{config.map_key(quarter_ids)}.npy"


def save_map(labels: np.ndarray, subject: str, quarter_ids: tuple[int, ...]) -> Path:
    path = map_path(subject, quarter_ids)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, labels.astype(np.int16))
    return path


def load_map(subject: str, quarter_ids: tuple[int, ...]) -> np.ndarray:
    return np.load(map_path(subject, quarter_ids)).astype(np.int64)


def has_map(subject: str, quarter_ids: tuple[int, ...]) -> bool:
    return map_path(subject, quarter_ids).exists()


def cohort(quarter_ids: tuple[int, ...]) -> dict[str, np.ndarray]:
    """{subject: map} for every subject with a stored map at this level."""
    return {s: load_map(s, quarter_ids) for s in config.SUBJECTS if has_map(s, quarter_ids)}


# --------------------------------------------------------------------- comparison
def map_dice(a: np.ndarray, b: np.ndarray) -> float:
    """Agreement between two whole maps: mean Dice over the 17 networks.

    Dice per network is 2|A n B| / (|A| + |B|); networks absent from both maps are
    skipped. 1.0 means the two maps carve the cortex identically.
    """
    scores = []
    for k in range(1, config.N_NETWORKS + 1):
        ak, bk = (a == k), (b == k)
        denom = ak.sum() + bk.sum()
        if denom:
            scores.append(2.0 * np.logical_and(ak, bk).sum() / denom)
    return float(np.mean(scores)) if scores else float("nan")


def to_image(labels: np.ndarray) -> nib.Nifti1Image:
    """Scatter domain labels back into a 3D NIfTI, for viewing a map."""
    ref = nib.load(_reference_bold())
    vol = np.zeros(group_networks().shape, dtype=np.int16)
    idx = analysis_domain()
    vol[idx[0], idx[1], idx[2]] = labels.astype(np.int16)
    return nib.Nifti1Image(vol, ref.affine, ref.header)
