"""The figure: stability and usefulness against amount of rest data.

Top    pairwise map agreement (Dice) between maps built from the same amount of data.
       Higher = lower estimation variance = more stable. Only the quarter and half levels
       have two or more equal-sized maps, so only those levels have a point.
Bottom subject-identification accuracy from the cumulative map at each level.
       Higher = the map says more about who this person is.

Faint per-subject lines, a bold group mean, and a shaded +/-1 SEM band.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import config

STABLE_COLOR = "#2c7fb8"
SIGNAL_COLOR = "#d95f0e"


def _fmt(x) -> str:
    if x is None or (isinstance(x, float) and x != x):
        return ""
    return f"{x:.4f}"


def _mean_sem(rows: list[list[float]]) -> tuple[np.ndarray, np.ndarray]:
    """Column-wise mean and standard error across subjects, ignoring missing values."""
    arr = np.asarray(rows, dtype=float)
    if arr.size == 0:
        return np.array([]), np.array([])
    counts = np.sum(~np.isnan(arr), axis=0)
    with np.errstate(invalid="ignore"):
        mean = np.nanmean(arr, axis=0)
        sd = np.nanstd(arr, axis=0, ddof=1) if arr.shape[0] > 1 else np.zeros_like(mean)
    sem = np.divide(sd, np.sqrt(counts), out=np.zeros_like(sd), where=counts > 0)
    return mean, sem


def _column(results: dict, section: str, field: str) -> list[list[float]]:
    """Per-subject rows of one field, one column per level (NaN where absent)."""
    rows = []
    for res in results["subjects"].values():
        by_level = {r["level"]: r.get(field) for r in res.get(section, [])}
        rows.append([
            float(by_level[lv]) if by_level.get(lv) is not None else float("nan")
            for lv in config.LEVELS
        ])
    return rows


def save_csv(results: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["subject", "level", "minutes", "map_agreement_dice", "n_pairs"])
        for sub, res in results["subjects"].items():
            for row in res.get("stability", []):
                w.writerow([sub, row["level"], _fmt(row.get("minutes")),
                            _fmt(row.get("dice")), row.get("n_pairs", "")])

        w.writerow([])
        w.writerow(["level", "identification_accuracy", "chance", "n_scans",
                    "n_subjects_tested", "n_candidate_maps"])
        for row in results.get("identification", []):
            w.writerow([row["level"], _fmt(row.get("accuracy")),
                        _fmt(results.get("chance")), row.get("n_scans", ""),
                        row.get("n_subjects", ""), row.get("n_candidates", "")])

        w.writerow([])
        w.writerow(["level", "group_mean_dice", "sem_dice"])
        mean, sem = _mean_sem(_column(results, "stability", "dice"))
        for i, lv in enumerate(config.LEVELS):
            if mean.size and np.isfinite(mean[i]):
                w.writerow([lv, _fmt(mean[i]), _fmt(sem[i])])


def plot(results: dict, path: Path) -> None:
    x = np.arange(len(config.LEVELS), dtype=float)
    subs = results["subjects"]
    n = len(subs)

    dice_rows = _column(results, "stability", "dice")
    dice_mean, dice_sem = _mean_sem(dice_rows)

    ident = {r["level"]: r for r in results.get("identification", [])}
    acc = np.array([ident[lv]["accuracy"] if lv in ident else np.nan
                    for lv in config.LEVELS], dtype=float)
    chance = results.get("chance", float("nan"))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 6.6), sharex=True)

    for row in dice_rows:
        ax1.plot(x, row, color=STABLE_COLOR, alpha=0.2, lw=1)
    if dice_mean.size:
        ax1.fill_between(x, dice_mean - dice_sem, dice_mean + dice_sem,
                         color=STABLE_COLOR, alpha=0.3, lw=0)
        ax1.plot(x, dice_mean, "o-", color=STABLE_COLOR, lw=2.5,
                 label=f"group mean, n={n} (±1 SEM)")
    ax1.set_ylabel("Map agreement (Dice)")
    ax1.set_ylim(0, 1.02)
    ax1.set_title("Stability: do equal-sized maps agree?")
    ax1.legend(loc="lower right", fontsize=8, framealpha=0.9)
    ax1.grid(alpha=0.25)

    ax2.plot(x, acc, "s-", color=SIGNAL_COLOR, lw=2.5, label="identification accuracy")
    if np.isfinite(chance):
        ax2.axhline(chance, ls="--", lw=1, color="gray", label=f"chance ({chance:.2f})")
    ax2.set_ylabel("Identification accuracy")
    ax2.set_xlabel("Resting-state data used (fraction of each subject's total)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(config.LEVELS)
    ax2.set_ylim(0, 1.02)
    ax2.set_title("Usefulness: can the map pick out whose brain it is?")
    ax2.legend(loc="lower right", fontsize=8, framealpha=0.9)
    ax2.grid(alpha=0.25)

    fig.suptitle(f"Personal brain maps vs amount of rest data (n={n})", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
