"""Plot 2 — subject identification from individualised maps.

Are the maps actually *individual*? Take a held-out scan, score it against all 10
subjects' maps, and see whether its own subject's map fits best. More rest data should
make maps more distinctive and identification easier.

Fit is within-network homogeneity: the mean correlation between voxels assigned to the
same network, averaged over the 17 networks. A map that carves this brain correctly
groups voxels that actually covary, so homogeneity is high.

Two design choices worth knowing:

* **Held-out scans are task runs only.** Task runs are never used to build maps at any
  data level, so the test set is *identical* across the whole x-axis. Leftover rest runs
  would change with the level and seed, confounding "better maps" with "different test
  data".
* **Homogeneity rises as networks get smaller**, so a map that happens to carve small
  networks could win for the wrong reason. ``size_matched=True`` subsamples every
  network to a common voxel count across the candidate maps before scoring; we report
  both and expect them to agree.
"""
from __future__ import annotations

import numpy as np

from . import config, parcellation


def _standardize_rows(x: np.ndarray) -> np.ndarray:
    mu = x.mean(axis=1, keepdims=True)
    sd = x.std(axis=1, keepdims=True)
    return np.divide(x - mu, sd, out=np.zeros_like(x), where=sd > 0)


def network_homogeneity(
    ts: np.ndarray,
    labels: np.ndarray,
    member_counts: dict[int, int] | None = None,
    rng: np.random.Generator | None = None,
    standardized: bool = False,
) -> float:
    """Mean within-network voxel-voxel correlation, averaged over the 17 networks.

    ``ts``: (n_voxels, n_time) domain timeseries for one scan, rows aligned to
    ``parcellation.analysis_domain()``. ``labels``: that candidate map's network ids.

    Computed without ever forming a voxel x voxel matrix. For z-scored rows,
    ``sum_{i!=j} r_ij = ||sum_i z_i||^2 / T - n``, so each network costs O(n*T).

    ``member_counts`` caps each network to that many voxels (size-matched scoring).
    """
    z = ts if standardized else _standardize_rows(ts)
    n_time = z.shape[1]
    per_network: list[float] = []

    for k in range(1, config.N_NETWORKS + 1):
        idx = np.flatnonzero(labels == k)
        if member_counts is not None:
            cap = member_counts.get(k, 0)
            if cap < 2 or idx.size < 2:
                continue
            if idx.size > cap:
                rng = rng if rng is not None else np.random.default_rng(0)
                idx = rng.choice(idx, size=cap, replace=False)
        if idx.size < 2:
            continue
        s = z[idx].sum(axis=0)
        sum_pairs = float(s @ s) / n_time - idx.size
        per_network.append(sum_pairs / (idx.size * (idx.size - 1)))

    return float(np.mean(per_network)) if per_network else float("nan")


def _common_counts(maps: dict[str, np.ndarray]) -> dict[int, int]:
    """Smallest size of each network across candidate maps (for size-matched scoring)."""
    counts: dict[int, int] = {}
    for k in range(1, config.N_NETWORKS + 1):
        counts[k] = min(int((lab == k).sum()) for lab in maps.values())
    return counts


def score_scan(
    ts: np.ndarray,
    maps: dict[str, np.ndarray],
    size_matched: bool = False,
    seed: int = 0,
) -> dict[str, float]:
    """Homogeneity of one scan under each candidate subject's map."""
    z = _standardize_rows(ts)
    counts = _common_counts(maps) if size_matched else None
    return {
        sub: network_homogeneity(
            z, lab,
            member_counts=counts,
            rng=np.random.default_rng([seed, abs(hash(sub)) % (2**31)]),
            standardized=True,
        )
        for sub, lab in maps.items()
    }


def identify_scan(
    ts: np.ndarray,
    maps: dict[str, np.ndarray],
    true_subject: str,
    size_matched: bool = False,
    seed: int = 0,
) -> dict:
    """Predict which subject a scan belongs to; the best-fitting map wins."""
    scores = score_scan(ts, maps, size_matched=size_matched, seed=seed)
    predicted = max(scores, key=scores.get)
    ordered = sorted(scores.values(), reverse=True)
    margin = (ordered[0] - ordered[1]) if len(ordered) > 1 else float("nan")
    return {
        "true": true_subject,
        "predicted": predicted,
        "correct": predicted == true_subject,
        "margin": float(margin),
        "scores": scores,
    }


def accuracy(records: list[dict]) -> float:
    """Fraction of held-out scans assigned to the right subject."""
    if not records:
        return float("nan")
    return float(np.mean([r["correct"] for r in records]))


def summarize(records: list[dict], n_subjects: int) -> dict:
    """Identification accuracy plus chance and per-subject recall."""
    by_subject: dict[str, list[bool]] = {}
    for r in records:
        by_subject.setdefault(r["true"], []).append(r["correct"])
    return {
        "accuracy": accuracy(records),
        "chance": 1.0 / n_subjects if n_subjects else float("nan"),
        "n_scans": len(records),
        "mean_margin": float(np.mean([r["margin"] for r in records])) if records else float("nan"),
        "per_subject_recall": {s: float(np.mean(v)) for s, v in sorted(by_subject.items())},
    }
