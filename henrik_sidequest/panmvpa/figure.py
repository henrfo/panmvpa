"""The paper figure: two panels over a shared minutes-of-rest x-axis.

Top: DN-A split-half parcellation reliability (Dice).
Bottom: DN-A contrast-to-noise during episodic projection.

Group form = mean across subjects with a shaded +/-1 SEM band, plus faint per-subject
traces so the spread behind the mean stays visible.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DICE_COLOR = "#2c7fb8"
CNR_COLOR = "#d95f0e"


def _mean_sem(rows: list[list[float]]) -> tuple[np.ndarray, np.ndarray]:
    """Column-wise mean and standard error over subjects (NaNs ignored)."""
    arr = np.asarray(rows, dtype=float)
    n = np.sum(~np.isnan(arr), axis=0)
    mean = np.nanmean(arr, axis=0)
    sd = np.nanstd(arr, axis=0, ddof=1) if arr.shape[0] > 1 else np.zeros_like(mean)
    sem = np.divide(sd, np.sqrt(n), out=np.zeros_like(sd), where=n > 0)
    return mean, sem


def save_group_csv(results: dict, path: Path) -> None:
    """One row per subject per data level, plus the group mean/SEM rows."""
    minutes = results["minutes"]
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["subject", "minutes", "dice", "dna_voxels_a", "dna_voxels_b",
                    "cnr", "mean_z_in", "mean_z_out", "n_dna_voxels", "n_sessions"])
        for sub, res in results["subjects"].items():
            for rel, c in zip(res["reliability"], res["cnr"]):
                w.writerow([sub, rel["minutes"], f"{rel['dice']:.4f}",
                            rel["n_voxels_a"], rel["n_voxels_b"],
                            f"{c['cnr']:.4f}", f"{c['mean_in']:.4f}",
                            f"{c['mean_out']:.4f}", c["n_dna_voxels"], c["n_sessions"]])
        dice_mean, dice_sem = _mean_sem(
            [[r["dice"] for r in res["reliability"]] for res in results["subjects"].values()]
        )
        cnr_mean, cnr_sem = _mean_sem(
            [[c["cnr"] for c in res["cnr"]] for res in results["subjects"].values()]
        )
        for i, m in enumerate(minutes):
            w.writerow(["GROUP_MEAN", m, f"{dice_mean[i]:.4f}", "", "",
                        f"{cnr_mean[i]:.4f}", "", "", "", len(results["subjects"])])
        for i, m in enumerate(minutes):
            w.writerow(["GROUP_SEM", m, f"{dice_sem[i]:.4f}", "", "",
                        f"{cnr_sem[i]:.4f}", "", "", "", len(results["subjects"])])


def plot_group(results: dict, path: Path) -> None:
    minutes = np.asarray(results["minutes"], dtype=float)
    subs = results["subjects"]
    dice_rows = [[r["dice"] for r in res["reliability"]] for res in subs.values()]
    cnr_rows = [[c["cnr"] for c in res["cnr"]] for res in subs.values()]
    dice_mean, dice_sem = _mean_sem(dice_rows)
    cnr_mean, cnr_sem = _mean_sem(cnr_rows)
    n = len(subs)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 6.8), sharex=True)

    for row in dice_rows:
        ax1.plot(minutes, row, color=DICE_COLOR, alpha=0.18, lw=1)
    ax1.fill_between(minutes, dice_mean - dice_sem, dice_mean + dice_sem,
                     color=DICE_COLOR, alpha=0.25, lw=0)
    ax1.plot(minutes, dice_mean, "o-", color=DICE_COLOR, lw=2,
             label=f"mean of {n} subjects (±1 SEM)")
    ax1.set_ylabel("DN-A split-half Dice")
    ax1.set_ylim(0, 1.02)
    ax1.set_title("DN-A parcellation reliability")
    ax1.legend(loc="lower right", fontsize=8, framealpha=0.9)
    ax1.grid(alpha=0.25)

    for row in cnr_rows:
        ax2.plot(minutes, row, color=CNR_COLOR, alpha=0.18, lw=1)
    ax2.fill_between(minutes, cnr_mean - cnr_sem, cnr_mean + cnr_sem,
                     color=CNR_COLOR, alpha=0.25, lw=0)
    ax2.plot(minutes, cnr_mean, "s-", color=CNR_COLOR, lw=2,
             label=f"mean of {n} subjects (±1 SEM)")
    ax2.axhline(0, ls="--", lw=1, color="gray", label="no contrast (0)")
    ax2.set_ylabel("DN-A contrast-to-noise (Z in − Z out)")
    ax2.set_xlabel("Resting-state data used (minutes)")
    ax2.set_xticks(minutes)
    ax2.set_title("Episodic-projection signal in the DN-A mask")
    ax2.legend(loc="lower right", fontsize=8, framealpha=0.9)
    ax2.grid(alpha=0.25)

    fig.suptitle(f"Precision fMRI: individualised DN-A vs amount of rest data (n={n})",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
