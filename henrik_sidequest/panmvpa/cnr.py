"""Plot 2 — DN-A contrast-to-noise vs amount of rest data.

How much does DN-A stand out from the rest of cortex during episodic projection?

    CNR = mean(Z inside DN-A) - mean(Z in all other network voxels)

A better individualised mask should concentrate epiproj signal, raising the contrast.
No classifier, no cross-validation, and no session-overlap requirement -- so unlike the
4-class decoding this works for all 10 subjects, every one of which has epiproj runs.

The epiproj Z-maps do not depend on rest data, so they are computed once per subject and
cached; only the DN-A mask changes with the amount of rest.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from . import config, events, glm, parcellation


@lru_cache(maxsize=16)
def epiproj_zmaps(subject: str) -> dict:
    """{session: z_vector} over the analysis domain for each fetched epiproj session."""
    out: dict[int, np.ndarray] = {}
    for ses in events.epiproj_sessions(subject):
        if glm.subtask_present(subject, ses, config.TASK):
            out[ses] = glm.task_zmap(subject, ses, config.TASK)
    if not out:
        raise ValueError(f"{config.sub_id(subject)}: no fetched epiproj runs.")
    return out


def cnr_from_mask(zmap: np.ndarray, dn_mask: np.ndarray) -> dict:
    """Mean Z inside DN-A minus mean Z in all other domain (cortical) voxels."""
    other = ~dn_mask
    if not dn_mask.any() or not other.any():
        return {"cnr": float("nan"), "mean_in": float("nan"), "mean_out": float("nan")}
    mean_in = float(zmap[dn_mask].mean())
    mean_out = float(zmap[other].mean())
    return {"cnr": mean_in - mean_out, "mean_in": mean_in, "mean_out": mean_out}


def cnr_at(
    subject: str, minutes: float, zmaps: dict | None = None, seed: int | None = None
) -> dict:
    """DN-A contrast-to-noise at one data level for one draw, averaged over sessions."""
    zmaps = zmaps if zmaps is not None else epiproj_zmaps(subject)
    labels = parcellation.build_parcellation(subject, minutes=minutes, seed=seed)
    dn_mask = parcellation.dn_a_mask(labels)

    per_session = [cnr_from_mask(z, dn_mask) for z in zmaps.values()]
    return {
        "minutes": minutes,
        "seed": seed,
        "cnr": float(np.mean([r["cnr"] for r in per_session])),
        "cnr_sd_across_sessions": float(np.std([r["cnr"] for r in per_session])),
        "mean_in": float(np.mean([r["mean_in"] for r in per_session])),
        "mean_out": float(np.mean([r["mean_out"] for r in per_session])),
        "n_dna_voxels": int(dn_mask.sum()),
        "n_other_voxels": int((~dn_mask).sum()),
        "n_sessions": len(per_session),
    }


def cnr_at_seeds(
    subject: str, minutes: float, n_seeds: int, zmaps: dict | None = None
) -> dict:
    """Aggregate CNR over ``n_seeds`` random subsets at one data level."""
    zmaps = zmaps if zmaps is not None else epiproj_zmaps(subject)
    from . import rest
    effective = 1 if rest.sampling_headroom(subject, minutes).get("spare_runs") == 0 else n_seeds
    per_seed = [cnr_at(subject, minutes, zmaps=zmaps, seed=s) for s in range(effective)]
    vals = np.array([r["cnr"] for r in per_seed], dtype=float)
    return {
        "minutes": minutes,
        "cnr": float(vals.mean()),            # subject's mean across seeds
        "cnr_std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
        "cnr_seeds": vals.tolist(),
        "n_seeds": len(per_seed),
        "mean_in": float(np.mean([r["mean_in"] for r in per_seed])),
        "mean_out": float(np.mean([r["mean_out"] for r in per_seed])),
        "n_dna_voxels": int(np.mean([r["n_dna_voxels"] for r in per_seed])),
        "n_sessions": per_seed[0]["n_sessions"],
    }


def cnr_curve(
    subject: str, minute_levels: list[float] | None = None, n_seeds: int = 1
) -> list[dict]:
    """DN-A contrast-to-noise across the standard minute levels (Z-maps fit once)."""
    levels = minute_levels if minute_levels is not None else config.MINUTE_LEVELS
    zmaps = epiproj_zmaps(subject)
    return [cnr_at_seeds(subject, m, n_seeds, zmaps=zmaps) for m in levels]
