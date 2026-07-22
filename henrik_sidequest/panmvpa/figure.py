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


def _example_networks(results: dict) -> list[str]:
    """A small, medium and large network — to check the mean isn't driven by big ones."""
    try:
        from . import parcellation
        return parcellation.example_networks()
    except Exception:
        # Fall back to whatever names the results happen to carry.
        for res in results["subjects"].values():
            for row in res.get("stability", []):
                names = sorted(row.get("dice_per_network", {}))
                if names:
                    return [names[0], names[len(names) // 2], names[-1]]
        return []


def _network_column(results: dict, network: str) -> np.ndarray:
    """Group-mean Dice for one network across levels."""
    rows = []
    for res in results["subjects"].values():
        by_level = {r["level"]: r.get("dice_per_network", {}).get(network)
                    for r in res.get("stability", [])}
        rows.append([float(by_level[lv]) if by_level.get(lv) is not None else float("nan")
                     for lv in config.LEVELS])
    mean, _ = _mean_sem(rows)
    return mean


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
        from . import parcellation
        try:
            net_names = list(parcellation.network_order())
        except Exception:                       # atlas unavailable (e.g. no BOLD on disk)
            net_names = []

        w.writerow(["subject", "level", "minutes", "map_agreement_dice", "n_pairs"]
                   + [f"dice_{n}" for n in net_names])
        for sub, res in results["subjects"].items():
            for row in res.get("stability", []):
                per = row.get("dice_per_network", {})
                w.writerow([sub, row["level"], _fmt(row.get("minutes")),
                            _fmt(row.get("dice")), row.get("n_pairs", "")]
                           + [_fmt(per.get(n)) for n in net_names])

        w.writerow([])
        w.writerow(["level", "mean_margin", "identification_accuracy", "chance",
                    "n_scans", "n_subjects_tested", "n_candidate_maps"])
        for row in results.get("identification", []):
            w.writerow([row["level"], _fmt(row.get("margin")), _fmt(row.get("accuracy")),
                        _fmt(results.get("chance")), row.get("n_scans", ""),
                        row.get("n_subjects", ""), row.get("n_candidates", "")])

        w.writerow([])
        w.writerow(["level", "group_mean_dice", "sem_dice"])
        mean, sem = _mean_sem(_column(results, "stability", "dice"))
        for i, lv in enumerate(config.LEVELS):
            if mean.size and np.isfinite(mean[i]):
                w.writerow([lv, _fmt(mean[i]), _fmt(sem[i])])


def _comparison_series(results: dict, field: str) -> tuple[np.ndarray, np.ndarray, list]:
    """Group mean, SEM and per-subject rows of one comparison field across levels."""
    comp = results.get("comparisons", {})
    per_sub = comp.get("per_subject", {})
    levels = [config.level_name(b) for b in config.STABILITY_BLOCKS]
    rows = []
    for res in per_sub.values():
        by_level = {r["level"]: r.get(field) for r in res.get("levels", [])}
        rows.append([float(by_level[lv]) if by_level.get(lv) is not None else np.nan
                     for lv in levels])
    mean, sem = _mean_sem(rows)
    return mean, sem, rows


def _between_series(results: dict, field: str = "dice") -> np.ndarray:
    comp = results.get("comparisons", {})
    by_level = {r["level"]: r.get(field) for r in comp.get("between", [])}
    return np.array([by_level.get(config.level_name(b), np.nan)
                     if by_level.get(config.level_name(b)) is not None else np.nan
                     for b in config.STABILITY_BLOCKS], dtype=float)


def save_comparison_csv(results: dict, path: Path) -> None:
    """Everything: per subject, per level, per network, for all three comparisons."""
    comp = results.get("comparisons", {})
    if not comp:
        return
    from . import parcellation
    families = {n: fam for fam, names in config.NETWORK_FAMILIES.items() for n in names}
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["subject", "level", "network", "family", "comparison", "dice"])

        for row in comp.get("between", []):
            for net, val in (row.get("per_network") or {}).items():
                w.writerow(["GROUP", row["level"], net, families.get(net, "?"),
                            "between", _fmt(val)])

        for sub, res in comp.get("per_subject", {}).items():
            for row in res.get("levels", []):
                for key, tag in (("within_per_network", "within"),
                                 ("to_group_per_network", "to_group")):
                    for net, val in (row.get(key) or {}).items():
                        w.writerow([sub, row["level"], net, families.get(net, "?"),
                                    tag, _fmt(val)])

        w.writerow([])
        w.writerow(["subject", "crossover_level"])
        for sub, res in comp.get("per_subject", {}).items():
            w.writerow([sub, res.get("crossover") or "never"])


def plot_comparisons(results: dict, path: Path) -> None:
    """Three panels: within/between/to-group, the family split, and the margin curve."""
    comp = results.get("comparisons", {})
    if not comp.get("per_subject"):
        return
    levels = [config.level_name(b) for b in config.STABILITY_BLOCKS]
    x = np.arange(len(levels), dtype=float)

    within, within_sem, within_rows = _comparison_series(results, "within")
    group, group_sem, group_rows = _comparison_series(results, "to_group")
    between = _between_series(results)
    n = len(within_rows)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    # --- A: the three curves, and where within overtakes to-group -------------
    ax = axes[0]
    for row in within_rows:
        ax.plot(x, row, color=STABLE_COLOR, alpha=0.2, lw=1)
    for row in group_rows:
        ax.plot(x, row, color="#7b3294", alpha=0.15, lw=1)
    ax.fill_between(x, within - within_sem, within + within_sem,
                    color=STABLE_COLOR, alpha=0.3, lw=0)
    ax.plot(x, within, "o-", color=STABLE_COLOR, lw=2.5, label="within-person")
    ax.plot(x, between, "s-", color="#999999", lw=2, label="between-person (null)")
    ax.fill_between(x, group - group_sem, group + group_sem, color="#7b3294",
                    alpha=0.2, lw=0)
    ax.plot(x, group, "^-", color="#7b3294", lw=2, label="similarity to group")

    cross = next((i for i in range(len(x))
                  if np.isfinite(within[i]) and np.isfinite(group[i])
                  and within[i] > group[i]), None)
    if cross is not None:
        ax.axvline(x[cross], ls=":", color="black", lw=1.5)
        ax.annotate(f"crossover\n{levels[cross]}", (x[cross], 0.06),
                    fontsize=8, ha="center",
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", alpha=0.85))
    ax.set_xticks(x); ax.set_xticklabels(levels)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Dice"); ax.set_xlabel("rest data used")
    ax.set_title(f"A. Individuation (n={n})\nwithin vs between vs group")
    ax.legend(fontsize=7, loc="upper left"); ax.grid(alpha=0.25)

    # --- B: association vs sensorimotor ---------------------------------------
    ax = axes[1]
    for family, ls in (("association", "-"), ("sensorimotor", "--")):
        w, _, _ = _comparison_series(results, f"within_{family}")
        g, _, _ = _comparison_series(results, f"to_group_{family}")
        b = _between_series(results, family)
        ax.plot(x, w, ls, marker="o", color=STABLE_COLOR, lw=2, label=f"within · {family}")
        ax.plot(x, b, ls, marker="s", color="#999999", lw=1.5, label=f"between · {family}")
        ax.plot(x, g, ls, marker="^", color="#7b3294", lw=1.5, label=f"group · {family}")
    ax.set_xticks(x); ax.set_xticklabels(levels)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Dice"); ax.set_xlabel("rest data used")
    ax.set_title("B. Association vs sensorimotor\n(solid = association, dashed = sensory)")
    ax.legend(fontsize=6, loc="upper left", ncol=2); ax.grid(alpha=0.25)

    # --- C: identification margin ---------------------------------------------
    ax = axes[2]
    ident = {r["level"]: r for r in results.get("identification", [])}
    xc = np.arange(len(config.LEVELS), dtype=float)
    margin = np.array([ident[lv].get("margin", np.nan) if lv in ident else np.nan
                       for lv in config.LEVELS], dtype=float)
    for res in results["subjects"].values():
        row = [next((m["margin"] for m in res.get("margin", []) if m["level"] == lv),
                    np.nan) for lv in config.LEVELS]
        ax.plot(xc, row, color=SIGNAL_COLOR, alpha=0.2, lw=1)
    ax.plot(xc, margin, "s-", color=SIGNAL_COLOR, lw=2.5, label="mean margin")
    ax.axhline(0, ls="--", lw=1, color="gray", label="0 = misidentified")
    ax.set_xticks(xc); ax.set_xticklabels(config.LEVELS)
    ax.set_ylabel("margin (correct − best wrong)"); ax.set_xlabel("rest data used")
    ax.set_title("C. Identification margin")
    ax.legend(fontsize=7, loc="upper left"); ax.grid(alpha=0.25)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot(results: dict, path: Path) -> None:
    x = np.arange(len(config.LEVELS), dtype=float)
    subs = results["subjects"]
    n = len(subs)

    dice_rows = _column(results, "stability", "dice")
    dice_mean, dice_sem = _mean_sem(dice_rows)

    ident = {r["level"]: r for r in results.get("identification", [])}
    acc = np.array([ident[lv]["accuracy"] if lv in ident else np.nan
                    for lv in config.LEVELS], dtype=float)
    margin = np.array([ident[lv].get("margin", np.nan) if lv in ident else np.nan
                       for lv in config.LEVELS], dtype=float)
    chance = results.get("chance", float("nan"))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.5, 7.2), sharex=True)

    # --- top: stability, mean plus three example networks spanning the size range ---
    for row in dice_rows:
        ax1.plot(x, row, color=STABLE_COLOR, alpha=0.2, lw=1)
    if dice_mean.size:
        ax1.fill_between(x, dice_mean - dice_sem, dice_mean + dice_sem,
                         color=STABLE_COLOR, alpha=0.3, lw=0)
        ax1.plot(x, dice_mean, "o-", color=STABLE_COLOR, lw=2.5,
                 label=f"mean of 17 networks, n={n} (±1 SEM)")

    for name, style in zip(_example_networks(results), [":", "-.", "--"]):
        vals = _network_column(results, name)
        if np.isfinite(vals).any():
            ax1.plot(x, vals, style, lw=1.4, alpha=0.9, label=f"{name}")
    ax1.set_ylabel("Map agreement (Dice)")
    ax1.set_ylim(0, 1.02)
    ax1.set_title("Stability: do equal-sized maps agree?")
    ax1.legend(loc="upper left", fontsize=7, framealpha=0.9, ncol=2)
    ax1.grid(alpha=0.25)

    # --- bottom: margin is primary (accuracy ceilings), accuracy on a twin axis ---
    for res in subs.values():
        row = [next((m["margin"] for m in res.get("margin", []) if m["level"] == lv),
                    np.nan) for lv in config.LEVELS]
        ax2.plot(x, row, color=SIGNAL_COLOR, alpha=0.2, lw=1)
    ax2.plot(x, margin, "s-", color=SIGNAL_COLOR, lw=2.5, label="mean margin (left)")
    ax2.axhline(0, ls="--", lw=1, color="gray", label="margin 0 = wrong ID")
    ax2.set_ylabel("Identification margin\n(correct − best wrong)")
    ax2.set_xlabel("Resting-state data used (fraction of each subject's total)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(config.LEVELS)
    ax2.set_title("Usefulness: how far ahead is the right map?")
    ax2.grid(alpha=0.25)

    ax2b = ax2.twinx()
    ax2b.plot(x, acc, "^--", color="#7b3294", lw=1.5, alpha=0.9, label="accuracy (right)")
    if np.isfinite(chance):
        ax2b.axhline(chance, ls=":", lw=1, color="#7b3294", alpha=0.7)
    ax2b.set_ylabel("Identification accuracy", color="#7b3294")
    ax2b.set_ylim(0, 1.05)
    ax2b.tick_params(axis="y", labelcolor="#7b3294")

    lines = ax2.get_lines()[-2:] + ax2b.get_lines()[:1]
    ax2.legend(lines, [ln.get_label() for ln in lines], loc="lower right", fontsize=8,
               framealpha=0.9)

    fig.suptitle(f"Personal brain maps vs amount of rest data (n={n})", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
