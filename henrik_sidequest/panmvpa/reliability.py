"""Plot 1 — parcellation reliability vs amount of rest data.

At each data level we take that many minutes of rest, split it in half, build a
parcellation from each half independently, and Dice-overlap the two DN-A (DefaultC)
masks. As data grows the two halves should converge (Dice -> 1).

Repeating this over ``n_seeds`` random subsets of runs turns a single point estimate into
a per-subject distribution, so a noisy subject can be told apart from a flat effect.
Halves are interleaved within a subset so each is balanced across sessions and time
rather than early-vs-late.
"""
from __future__ import annotations

import numpy as np

from . import config, parcellation, rest


def dice(a: np.ndarray, b: np.ndarray) -> float:
    """Sorensen-Dice overlap of two boolean masks. 0 if both empty."""
    a = a.astype(bool)
    b = b.astype(bool)
    denom = a.sum() + b.sum()
    if denom == 0:
        return 0.0
    return float(2.0 * np.logical_and(a, b).sum() / denom)


def split_half_runs(
    subject: str, minutes: float, seed: int | None = None
) -> tuple[list[rest.RestRun], list[rest.RestRun]]:
    """Interleave the runs for ``minutes`` of rest into two balanced halves.

    Split-half Dice needs both halves non-empty, i.e. at least two runs (~10 min). At the
    smallest levels a subject may not have that, so callers must check.
    """
    chosen = rest.runs_for(subject, minutes, seed=seed)
    return chosen[0::2], chosen[1::2]


def reliability_at(subject: str, minutes: float, seed: int | None = None) -> dict:
    """DN-A split-half Dice at one data level for one draw of runs.

    Returns NaN Dice (not a crash) when there are fewer than two runs to split -- e.g. the
    5 min level, which is a single run. Plot 1 simply has no point there for that subject.
    """
    half_a, half_b = split_half_runs(subject, minutes, seed=seed)
    if not half_a or not half_b:
        return {"minutes": minutes, "seed": seed, "dice": float("nan"),
                "n_voxels_a": 0, "n_voxels_b": 0, "runs_per_half": len(half_a),
                "too_few_runs": True}
    mask_a = parcellation.dn_a_mask(parcellation.build_parcellation(subject, runs=half_a))
    mask_b = parcellation.dn_a_mask(parcellation.build_parcellation(subject, runs=half_b))
    return {
        "minutes": minutes,
        "seed": seed,
        "dice": dice(mask_a, mask_b),
        "n_voxels_a": int(mask_a.sum()),
        "n_voxels_b": int(mask_b.sum()),
        "runs_per_half": len(half_a),
    }


def reliability_at_seeds(subject: str, minutes: float, n_seeds: int) -> dict:
    """Aggregate Dice over ``n_seeds`` random subsets at one data level.

    With no spare runs (notably the FULL level) every seed draws the identical subset, so
    we run a single seed instead of repeating the same expensive computation n times.
    """
    head = rest.sampling_headroom(subject, minutes)
    effective = 1 if head.get("spare_runs") == 0 else n_seeds
    per_seed = [reliability_at(subject, minutes, seed=s) for s in range(effective)]
    dices = np.array([r["dice"] for r in per_seed], dtype=float)
    finite = dices[np.isfinite(dices)]
    return {
        "minutes": minutes,
        "dice": float(finite.mean()) if finite.size else float("nan"),
        "dice_std": float(finite.std(ddof=1)) if finite.size > 1 else 0.0,
        "dice_seeds": dices.tolist(),
        "n_seeds": len(per_seed),
        "n_voxels_a": int(np.mean([r["n_voxels_a"] for r in per_seed])),
        "n_voxels_b": int(np.mean([r["n_voxels_b"] for r in per_seed])),
        "runs_per_half": per_seed[0]["runs_per_half"],
        "spare_runs": head["spare_runs"],     # 0 => seeds are identical draws
        "too_few_runs": bool(per_seed[0].get("too_few_runs", False)),
    }


def reliability_curve(
    subject: str, minute_levels: list[float] | None = None, n_seeds: int = 1
) -> list[dict]:
    """DN-A reliability across the standard minute levels."""
    levels = minute_levels if minute_levels is not None else config.MINUTE_LEVELS
    return [reliability_at_seeds(subject, m, n_seeds) for m in levels]
