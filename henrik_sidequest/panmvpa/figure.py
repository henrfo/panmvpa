"""The paper figure: two panels over a shared minutes-of-rest x-axis.

Top: DN-A split-half parcellation reliability (Dice).
Bottom: 4-class task-decoding accuracy (leave-one-session-out SVM), with the 25% chance line.
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save_csv(reliability: list[dict], decoding: list[dict], path: Path) -> None:
    """Raw numbers behind the figure, one row per data level."""
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            ["minutes", "dice", "dna_voxels_a", "dna_voxels_b",
             "accuracy", "chance", "n_features", "n_folds", "n_samples"]
        )
        for r, d in zip(reliability, decoding):
            w.writerow([
                r["minutes"], f"{r['dice']:.4f}", r["n_voxels_a"], r["n_voxels_b"],
                f"{d['accuracy']:.4f}", d["chance"], d["n_features"],
                d["n_folds"], d["n_samples"],
            ])


def plot(subject: str, reliability: list[dict], decoding: list[dict], path: Path) -> None:
    minutes = [r["minutes"] for r in reliability]
    dice = [r["dice"] for r in reliability]
    acc = [d["accuracy"] for d in decoding]
    chance = decoding[0]["chance"] if decoding else 0.25

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 6.5), sharex=True)

    ax1.plot(minutes, dice, "o-", color="#2c7fb8")
    ax1.set_ylabel("DN-A split-half Dice")
    ax1.set_ylim(0, 1.02)
    ax1.set_title(f"{subject}: DN-A parcellation reliability")
    ax1.grid(alpha=0.25)

    ax2.plot(minutes, acc, "s-", color="#d95f0e", label="4-class accuracy")
    ax2.axhline(chance, ls="--", lw=1, color="gray", label=f"chance ({chance:.2f})")
    ax2.set_ylabel("Task-decoding accuracy")
    ax2.set_xlabel("Resting-state data used (minutes)")
    ax2.set_ylim(0, 1.02)
    ax2.set_xticks(minutes)
    ax2.set_title("Task decoding from the DN-A pattern")
    ax2.legend(loc="lower right", fontsize=8, framealpha=0.9)
    ax2.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
