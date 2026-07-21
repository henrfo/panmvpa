"""Individualised volumetric parcellation by winner-take-all to the group Yeo-17.

Method (anchored to a FIXED group atlas, per the study design):

1. The group reference is Schaefer-400 grouped into its 17 Yeo-Krienen networks,
   resampled once to the BOLD grid. These 17 *seed regions* never change with data
   amount and are never re-derived from the individual.
2. At a given data level we form 17 reference timeseries = the mean signal within each
   fixed group region, computed from THAT level's concatenated rest data. (Reference
   *signals* are recomputed per level because correlation needs shared timepoints; their
   *definition* is frozen to the group atlas.)
3. Every cortical voxel is reassigned to whichever of the 17 references it correlates
   with most (winner-take-all). Only allegiance moves — that is the individualisation.

DN-A is the DefaultC network; ``dn_a_mask`` returns its binary volume.
"""
from __future__ import annotations

import re
from functools import lru_cache

import numpy as np
import nibabel as nib
from nilearn.image import resample_to_img

from . import config, rest

_NET = re.compile(r"17Networks_(?:LH|RH)_([A-Za-z]+)")


@lru_cache(maxsize=1)
def network_order() -> tuple[str, ...]:
    """The 17 network names in a stable order (first appearance in the Schaefer LUT)."""
    seen: list[str] = []
    for name in _parcel_names().values():
        net = _net_of(name)
        if net and net not in seen:
            seen.append(net)
    return tuple(seen)


def network_id(name: str) -> int:
    """1-based id of a network within :func:`network_order`."""
    return network_order().index(name) + 1


@lru_cache(maxsize=1)
def _parcel_names() -> dict[int, str]:
    names: dict[int, str] = {}
    for ln in config.SCHAEFER_ORDER.read_text().splitlines():
        p = ln.split()
        if len(p) >= 2 and p[0].isdigit():
            names[int(p[0])] = p[1]
    return names


def _net_of(parcel_name: str) -> str | None:
    m = _NET.search(parcel_name)
    return m.group(1) if m else None


def _reference_bold() -> str:
    """A representative preproc BOLD path, used only to define the target grid.

    Every ds006598 preproc BOLD shares the MNI152NLin6Asym 2mm grid, so any fetched
    run works. We look across subjects for the first one present on disk.
    """
    for sub in config.SUBJECTS:
        runs = rest.rest_runs(sub, fetched_only=True)
        if runs:
            return str(runs[0].path)
    raise FileNotFoundError("No fetched rest BOLD found to define the resampling grid.")


@lru_cache(maxsize=1)
def group_network_volume() -> np.ndarray:
    """3D int array on the BOLD grid: 0 = outside cortex, 1..17 = group network id.

    Schaefer-400 is resampled (nearest) to the BOLD grid, then each parcel is collapsed
    to its 17-network id. This is the fixed group anchor.
    """
    ref = nib.load(_reference_bold())
    atlas = nib.load(str(config.SCHAEFER_ATLAS))
    atlas_r = resample_to_img(
        atlas, ref, interpolation="nearest", force_resample=True, copy_header=True
    )
    parcels = np.asarray(atlas_r.get_fdata()).astype(np.int32)  # 0..400

    net_vol = np.zeros(parcels.shape, dtype=np.int16)
    order = network_order()
    names = _parcel_names()
    for pid, pname in names.items():
        net = _net_of(pname)
        if net is None:
            continue
        net_vol[parcels == pid] = order.index(net) + 1
    return net_vol


@lru_cache(maxsize=1)
def analysis_domain() -> np.ndarray:
    """(3, n_voxels) integer voxel coordinates of the cortical domain (group label > 0)."""
    coords = np.array(np.nonzero(group_network_volume() > 0))
    return coords.astype(np.int64)


@lru_cache(maxsize=1)
def _domain_group_labels() -> np.ndarray:
    """(n_voxels,) fixed group network id for each domain voxel (1..17)."""
    idx = analysis_domain()
    return group_network_volume()[idx[0], idx[1], idx[2]].astype(np.int64)


def _standardize(x: np.ndarray, axis: int) -> np.ndarray:
    mu = x.mean(axis=axis, keepdims=True)
    sd = x.std(axis=axis, keepdims=True)
    return np.divide(x - mu, sd, out=np.zeros_like(x), where=sd > 0)


def _standardize_inplace(x: np.ndarray, axis: int) -> np.ndarray:
    """Z-score along ``axis`` without allocating a second copy of ``x``.

    At 120 min the domain timeseries is ~3 GB, so a copy here would roughly double
    peak memory. ``x`` must be an array we own (a fresh concatenation).
    """
    mu = x.mean(axis=axis, keepdims=True)
    sd = x.std(axis=axis, keepdims=True)
    np.subtract(x, mu, out=x)
    np.divide(x, sd, out=x, where=sd > 0)
    # A zero-variance row is already all-zero after centring; clamp any FP residue.
    if axis == 1:
        x[(sd <= 0).ravel()] = 0.0
    return x


def winner_take_all(ts: np.ndarray) -> np.ndarray:
    """Assign each domain voxel to its most-correlated group reference.

    ``ts``: (n_voxels, n_time) domain timeseries (rows aligned to ``analysis_domain``).
    Returns (n_voxels,) individualised network labels in 1..17.

    NOTE: ``ts`` is standardised **in place** to keep peak memory down, so pass an array
    you own. ``rest.masked_timeseries`` always returns a fresh concatenation, which is
    the intended caller; never pass a cached per-run array directly.
    """
    group = _domain_group_labels()
    n_time = ts.shape[1]
    # Fixed group regions -> reference signals from THIS data.
    refs = np.zeros((config.N_NETWORKS, n_time), dtype=np.float32)
    for k in range(1, config.N_NETWORKS + 1):
        members = group == k
        if members.any():
            refs[k - 1] = ts[members].mean(axis=0)
    # Standardise in place: ``ts`` is a fresh concatenation we own, and at 120 min a
    # copy would add ~3 GB. refs is tiny, so the copying version is fine there.
    zt = _standardize_inplace(ts, axis=1)
    zr = _standardize(refs, axis=1)
    corr = (zt @ zr.T) / n_time  # (n_voxels, 17)
    return corr.argmax(axis=1).astype(np.int64) + 1


def build_parcellation(
    subject: str,
    minutes: float | None = None,
    runs: list[rest.RestRun] | None = None,
) -> np.ndarray:
    """Individualised labels (n_voxels,) for a subject at a data level.

    Pass ``minutes`` to take the fewest leading runs reaching that amount, or pass an
    explicit ``runs`` list (used for split-half reliability).
    """
    if runs is None:
        if minutes is None:
            raise ValueError("Provide either minutes or an explicit runs list.")
        runs = rest.select_runs(subject, minutes)
    ts = rest.masked_timeseries(runs)
    return winner_take_all(ts)


def dn_a_mask(labels: np.ndarray) -> np.ndarray:
    """Boolean (n_voxels,) mask of DN-A (DefaultC) voxels for an individualised labelling."""
    return labels == network_id(config.DN_A_NETWORK)


@lru_cache(maxsize=1)
def domain_mask_img() -> nib.Nifti1Image:
    """Cortical-domain mask (group coverage > 0) as a NIfTI on the BOLD grid.

    Used as the first-level GLM analysis mask so beta maps align exactly with
    :func:`analysis_domain`.
    """
    ref = nib.load(_reference_bold())
    vol = (group_network_volume() > 0).astype(np.int16)
    return nib.Nifti1Image(vol, ref.affine, ref.header)


def labels_to_img(labels: np.ndarray) -> nib.Nifti1Image:
    """Scatter domain labels back into a 3D NIfTI on the BOLD grid (for inspection/saving)."""
    ref = nib.load(_reference_bold())
    vol = np.zeros(group_network_volume().shape, dtype=np.int16)
    idx = analysis_domain()
    vol[idx[0], idx[1], idx[2]] = labels.astype(np.int16)
    return nib.Nifti1Image(vol, ref.affine, ref.header)
