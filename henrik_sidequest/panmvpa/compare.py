"""Three matched map comparisons, all computed from maps already on disk.

within    same subject, two disjoint chunks of their own rest data
between   two different subjects, the SAME chunk position and the same amount of data
to_group  one subject's map against the fixed Yeo-17 group parcellation

Why the between-person null matters: a rising within-person curve on its own is
ambiguous, because "more data makes every map converge toward the group" would produce
the same rise. The null rules that out. The *gap* between within and between is
individuation.

Similarity-to-group falls as real individual structure emerges. The crossover -- where
within-person agreement first overtakes similarity-to-group -- is the amount of data at
which a map resembles that person's own other half more than it resembles the average
brain. That single number is the practical answer to "how long do I need to scan?".

Everything here is Dice on maps that already exist; no BOLD is read.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np

from . import config, parcellation


def _nanmean_rows(rows: list[np.ndarray]) -> np.ndarray:
    """Mean over comparisons, per network, ignoring absent networks."""
    if not rows:
        return np.full(config.N_NETWORKS, np.nan)
    with np.errstate(invalid="ignore"):
        return np.nanmean(np.array(rows), axis=0)


def within_subject(subject: str, block: int) -> np.ndarray | None:
    """Per-network Dice between disjoint equal-sized maps of one subject."""
    rows = []
    for a, b in config.stability_pairs(block):
        if parcellation.has_map(subject, a) and parcellation.has_map(subject, b):
            rows.append(parcellation.dice_per_network(parcellation.load_map(subject, a),
                                                      parcellation.load_map(subject, b)))
    return _nanmean_rows(rows) if rows else None


def between_subjects(subjects: list[str], block: int) -> np.ndarray | None:
    """Per-network Dice between different subjects at matched chunk and data amount.

    Matched means the same (start, size) block for both people, so any difference cannot
    come from one map having more data or covering a different part of the session.
    """
    starts = list(range(0, config.N_CHUNKS, block))
    rows = []
    for start in starts:
        spec = (start, block)
        have = [s for s in subjects if parcellation.has_map(s, spec)]
        for a, b in combinations(have, 2):
            rows.append(parcellation.dice_per_network(parcellation.load_map(a, spec),
                                                      parcellation.load_map(b, spec)))
    return _nanmean_rows(rows) if rows else None


def to_group(subject: str, block: int) -> np.ndarray | None:
    """Per-network Dice between a subject's maps at this level and the group atlas."""
    group = parcellation.group_map()
    rows = []
    for start in range(0, config.N_CHUNKS, block):
        spec = (start, block)
        if parcellation.has_map(subject, spec):
            rows.append(parcellation.dice_per_network(parcellation.load_map(subject, spec),
                                                      group))
    return _nanmean_rows(rows) if rows else None


def family_mean(per_network: np.ndarray, family: str | None = None) -> float:
    """Mean Dice over all 17 networks, or over one family of them."""
    if per_network is None:
        return float("nan")
    names = parcellation.network_order()
    if family is None:
        values = per_network
    else:
        keep = set(config.NETWORK_FAMILIES[family])
        values = np.array([v for n, v in zip(names, per_network) if n in keep])
    with np.errstate(invalid="ignore"):
        return float(np.nanmean(values)) if values.size else float("nan")


def crossover(levels: list[str], within: list[float], group: list[float]) -> str | None:
    """First level where within-person agreement overtakes similarity-to-group.

    Returns the level label, or None if it never crosses within the measured range.
    """
    for level, w, g in zip(levels, within, group):
        if np.isfinite(w) and np.isfinite(g) and w > g:
            return level
    return None


def compare_all(subjects: list[str]) -> dict:
    """Every comparison at every level, per network and per family.

    Returns {"per_subject": {sub: [row, ...]}, "between": [row, ...]} where each row
    covers one data level.
    """
    names = list(parcellation.network_order())
    out: dict = {"networks": names, "per_subject": {}, "between": []}

    for block in config.STABILITY_BLOCKS:
        level = config.level_name(block)
        btw = between_subjects(subjects, block)
        out["between"].append({
            "level": level,
            "dice": family_mean(btw),
            "association": family_mean(btw, "association"),
            "sensorimotor": family_mean(btw, "sensorimotor"),
            "per_network": None if btw is None else
                           {n: float(v) for n, v in zip(names, btw)},
        })

    for subject in subjects:
        rows = []
        for block in config.STABILITY_BLOCKS:
            level = config.level_name(block)
            win, grp = within_subject(subject, block), to_group(subject, block)
            rows.append({
                "level": level,
                "within": family_mean(win),
                "to_group": family_mean(grp),
                "within_association": family_mean(win, "association"),
                "within_sensorimotor": family_mean(win, "sensorimotor"),
                "to_group_association": family_mean(grp, "association"),
                "to_group_sensorimotor": family_mean(grp, "sensorimotor"),
                "within_per_network": None if win is None else
                                      {n: float(v) for n, v in zip(names, win)},
                "to_group_per_network": None if grp is None else
                                        {n: float(v) for n, v in zip(names, grp)},
            })
        if rows:
            rows_levels = [r["level"] for r in rows]
            out["per_subject"][subject] = {
                "levels": rows,
                "crossover": crossover(rows_levels,
                                       [r["within"] for r in rows],
                                       [r["to_group"] for r in rows]),
            }
    return out
