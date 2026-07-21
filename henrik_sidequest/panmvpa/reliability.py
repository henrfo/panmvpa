"""Plot 1 — parcellation reliability vs amount of rest data.

At each data level we split that amount of rest in half, build a parcellation from each
half independently, and Dice-overlap the two DN-A (DefaultC) masks. As data grows the two
halves should converge (Dice -> 1). Halves are interleaved (even/odd runs) so each is
balanced across sessions and time rather than early-vs-late.
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
    subject: str, minutes: float
) -> tuple[list[rest.RestRun], list[rest.RestRun]]:
    """Interleave the runs for ``minutes`` of rest into two balanced halves."""
    chosen = rest.select_runs(subject, minutes)
    half_a = chosen[0::2]
    half_b = chosen[1::2]
    return half_a, half_b


def reliability_at(subject: str, minutes: float) -> dict:
    """DN-A split-half Dice at one data level (plus mask sizes for context)."""
    half_a, half_b = split_half_runs(subject, minutes)
    labels_a = parcellation.build_parcellation(subject, runs=half_a)
    labels_b = parcellation.build_parcellation(subject, runs=half_b)
    mask_a = parcellation.dn_a_mask(labels_a)
    mask_b = parcellation.dn_a_mask(labels_b)
    return {
        "minutes": minutes,
        "dice": dice(mask_a, mask_b),
        "n_voxels_a": int(mask_a.sum()),
        "n_voxels_b": int(mask_b.sum()),
        "runs_per_half": len(half_a),
    }


def reliability_curve(
    subject: str, minute_levels: list[float] | None = None
) -> list[dict]:
    """DN-A reliability across the standard minute levels."""
    levels = minute_levels if minute_levels is not None else config.MINUTE_LEVELS
    return [reliability_at(subject, m) for m in levels]
