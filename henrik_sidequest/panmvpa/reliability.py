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
    """Interleave the runs for ``minutes`` of rest into two balanced halves."""
    chosen = rest.runs_for(subject, minutes, seed=seed)
    return chosen[0::2], chosen[1::2]


def reliability_at(subject: str, minutes: float, seed: int | None = None) -> dict:
    """DN-A split-half Dice at one data level for one draw of runs."""
    half_a, half_b = split_half_runs(subject, minutes, seed=seed)
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
    """Aggregate Dice over ``n_seeds`` random subsets at one data level."""
    per_seed = [reliability_at(subject, minutes, seed=s) for s in range(n_seeds)]
    dices = np.array([r["dice"] for r in per_seed], dtype=float)
    head = rest.sampling_headroom(subject, minutes)
    return {
        "minutes": minutes,
        "dice": float(dices.mean()),          # subject's mean across seeds
        "dice_std": float(dices.std(ddof=1)) if len(dices) > 1 else 0.0,
        "dice_seeds": dices.tolist(),
        "n_seeds": len(per_seed),
        "n_voxels_a": int(np.mean([r["n_voxels_a"] for r in per_seed])),
        "n_voxels_b": int(np.mean([r["n_voxels_b"] for r in per_seed])),
        "runs_per_half": per_seed[0]["runs_per_half"],
        "spare_runs": head["spare_runs"],     # 0 => seeds are near-identical draws
    }


def reliability_curve(
    subject: str, minute_levels: list[float] | None = None, n_seeds: int = 1
) -> list[dict]:
    """DN-A reliability across the standard minute levels."""
    levels = minute_levels if minute_levels is not None else config.MINUTE_LEVELS
    return [reliability_at_seeds(subject, m, n_seeds) for m in levels]
