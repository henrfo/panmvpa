"""Score a held-out scan against candidate maps and predict whose brain it is.

Fit is within-network homogeneity: for a candidate map, take the voxels it assigns to a
network, measure the average correlation between those voxels *in this scan*, and average
over the 17 networks. A map that carves this brain correctly groups voxels that really do
fluctuate together, so its homogeneity is high. The best-fitting map is the prediction.

The sum trick keeps this cheap: for z-scored rows, the sum of all pairwise correlations in
a set of n voxels is ``||sum of rows||^2 / T - n``, so each network costs O(n*T) and no
voxel-by-voxel correlation matrix is ever formed.
"""
from __future__ import annotations

import numpy as np

from . import config


def network_homogeneity(scan: np.ndarray, labels: np.ndarray) -> float:
    """Mean within-network voxel-voxel correlation, averaged over the 17 networks.

    ``scan``: (n_voxels, n_time), z-scored rows, aligned to the analysis domain.
    ``labels``: a candidate map's network assignment for those same voxels.
    """
    n_time = scan.shape[1]
    per_network = []
    for k in range(1, config.N_NETWORKS + 1):
        idx = np.flatnonzero(labels == k)
        if idx.size < 2:
            continue
        s = scan[idx].sum(axis=0)
        sum_pairs = float(s @ s) / n_time - idx.size
        per_network.append(sum_pairs / (idx.size * (idx.size - 1)))
    return float(np.mean(per_network)) if per_network else float("nan")


def identify(scan: np.ndarray, maps: dict[str, np.ndarray], truth: str) -> dict:
    """Score a scan against every candidate map; the best fit is the prediction.

    ``margin`` is the continuous version of the answer:

        margin = homogeneity(correct subject's map) - best homogeneity among the rest

    Positive means correct, and larger means more confident. Accuracy saturates at 1.0
    once 10 brains are easy to tell apart, but the margin keeps growing after that, so it
    is the measure that can still show improvement at the top of the data range.
    """
    scores = {sub: network_homogeneity(scan, lab) for sub, lab in maps.items()}
    predicted = max(scores, key=scores.get)
    others = [v for sub, v in scores.items() if sub != truth]
    margin = (scores[truth] - max(others)) if (truth in scores and others) else float("nan")
    return {
        "true": truth,
        "predicted": predicted,
        "correct": predicted == truth,
        "margin": float(margin),
        "score_true": float(scores.get(truth, float("nan"))),
        "scores": {s: float(v) for s, v in scores.items()},
    }


def accuracy(records: list[dict]) -> float:
    """Fraction of held-out scans assigned to the right subject."""
    return float(np.mean([r["correct"] for r in records])) if records else float("nan")


def mean_margin(records: list[dict]) -> float:
    """Mean signed margin. Keeps discriminating after accuracy hits its ceiling."""
    vals = [r["margin"] for r in records if r.get("margin") is not None]
    vals = [v for v in vals if v == v]  # drop NaN
    return float(np.mean(vals)) if vals else float("nan")


def summarize(records: list[dict], n_subjects: int) -> dict:
    by_subject: dict[str, list[bool]] = {}
    for r in records:
        by_subject.setdefault(r["true"], []).append(r["correct"])
    return {
        "accuracy": accuracy(records),
        "chance": 1.0 / n_subjects if n_subjects else float("nan"),
        "n_scans": len(records),
        "per_subject_recall": {s: float(np.mean(v)) for s, v in sorted(by_subject.items())},
    }
