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
    """Per-subject Dice rows, group Dice mean/SEM rows, and the identification curve."""
    minutes = results["minutes"]
    subs = results["subjects"]
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["subject", "minutes", "dice", "dice_std", "n_seeds", "spare_runs",
                    "dna_voxels_a", "dna_voxels_b"])
        for sub, res in subs.items():
            for rel in res["reliability"]:
                w.writerow([sub, rel["minutes"], f"{rel['dice']:.4f}",
                            f"{rel.get('dice_std', 0.0):.4f}", rel.get("n_seeds", 1),
                            rel.get("spare_runs", ""),
                            rel["n_voxels_a"], rel["n_voxels_b"]])
        dice_mean, dice_sem = _mean_sem(
            [[r["dice"] for r in res["reliability"]] for res in subs.values()]
        )
        for i, m in enumerate(minutes):
            w.writerow(["GROUP_MEAN", m, f"{dice_mean[i]:.4f}", f"{dice_sem[i]:.4f}",
                        len(subs), "", "", ""])

        w.writerow([])
        w.writerow(["minutes", "identification_accuracy", "accuracy_std",
                    "accuracy_size_matched", "chance", "n_scans", "n_seeds"])
        for row in results.get("identification", []):
            w.writerow([row["minutes"], f"{row['accuracy']:.4f}",
                        f"{row['accuracy_std']:.4f}",
                        f"{row['accuracy_size_matched']:.4f}",
                        f"{results.get('chance', float('nan')):.4f}",
                        row["n_scans"], row["n_seeds"]])


def plot_group(results: dict, path: Path) -> None:
    minutes = np.asarray(results["minutes"], dtype=float)
    subs = results["subjects"]
    dice_rows = [[r["dice"] for r in res["reliability"]] for res in subs.values()]
    dice_mean, dice_sem = _mean_sem(dice_rows)
    n = len(subs)

    ident = results.get("identification", [])
    acc = np.array([row["accuracy"] for row in ident], dtype=float)
    acc_sd = np.array([row["accuracy_std"] for row in ident], dtype=float)
    acc_sm = np.array([row["accuracy_size_matched"] for row in ident], dtype=float)
    chance = results.get("chance", float("nan"))

    # Per-subject seed spread, so a noisy subject is distinguishable from a flat effect.
    dice_sd = [[r.get("dice_std", 0.0) for r in res["reliability"]] for res in subs.values()]
    n_seeds = results.get("n_seeds", 1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 6.8), sharex=True)

    for row, sd in zip(dice_rows, dice_sd):
        ax1.plot(minutes, row, color=DICE_COLOR, alpha=0.2, lw=1)
        if n_seeds > 1:
            ax1.errorbar(minutes, row, yerr=sd, fmt="none", ecolor=DICE_COLOR,
                         alpha=0.18, elinewidth=1, capsize=2)
    ax1.fill_between(minutes, dice_mean - dice_sem, dice_mean + dice_sem,
                     color=DICE_COLOR, alpha=0.3, lw=0)
    ax1.plot(minutes, dice_mean, "o-", color=DICE_COLOR, lw=2.5,
             label=f"group mean, n={n} (±1 SEM)")
    ax1.set_ylabel("DN-A split-half Dice")
    ax1.set_ylim(0, 1.02)
    ax1.set_title("DN-A parcellation reliability")
    ax1.legend(loc="lower right", fontsize=8, framealpha=0.9)
    ax1.grid(alpha=0.25)

    if len(acc):
        ax2.fill_between(minutes, acc - acc_sd, acc + acc_sd,
                         color=CNR_COLOR, alpha=0.3, lw=0)
        ax2.plot(minutes, acc, "s-", color=CNR_COLOR, lw=2.5,
                 label=f"identification accuracy (±1 SD over {n_seeds} seeds)")
        if np.isfinite(acc_sm).any():
            ax2.plot(minutes, acc_sm, "^--", color="#7b3294", lw=1.5, alpha=0.85,
                     label="size-matched control")
    ax2.axhline(chance, ls="--", lw=1, color="gray", label=f"chance ({chance:.2f})")
    ax2.set_ylabel("Subject identification accuracy")
    ax2.set_xlabel("Resting-state data used (minutes)")
    ax2.set_xticks(minutes)
    ax2.set_ylim(0, 1.02)
    ax2.set_title("Identifying the subject from a held-out scan")
    ax2.legend(loc="lower right", fontsize=8, framealpha=0.9)
    ax2.grid(alpha=0.25)

    seed_note = f", {n_seeds} random run-subsets/subject" if n_seeds > 1 else ""
    fig.suptitle(
        f"Precision fMRI: individualised maps vs amount of rest data "
        f"(n={n}{seed_note})",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
