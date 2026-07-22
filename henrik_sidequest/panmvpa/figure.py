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
import matplotlib.ticker as mticker

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
        w.writerow(["subject", "level", "minutes", "network", "family",
                    "comparison", "dice"])

        for row in comp.get("between", []):
            for net, val in (row.get("per_network") or {}).items():
                w.writerow(["GROUP", row["level"], "", net, families.get(net, "?"),
                            "between", _fmt(val)])

        for sub, res in comp.get("per_subject", {}).items():
            for row in res.get("levels", []):
                for key, tag in (("within_per_network", "within"),
                                 ("to_group_per_network", "to_group")):
                    for net, val in (row.get(key) or {}).items():
                        w.writerow([sub, row["level"], _fmt(row.get("minutes")), net,
                                    families.get(net, "?"), tag, _fmt(val)])

        w.writerow([])
        w.writerow(["subject", "crossover_level"])
        for sub, res in comp.get("per_subject", {}).items():
            w.writerow([sub, res.get("crossover") or "never"])


def _log_x(ax) -> None:
    """Log-scale the x-axis, but only if something positive was actually plotted.

    Matplotlib raises if every x is <= 0, which happens when durations are missing (an
    older results file, or maps built before minutes were recorded).
    """
    xs = np.concatenate([c.get_offsets()[:, 0] for c in ax.collections
                         if len(c.get_offsets())]) if ax.collections else np.array([])
    if not (xs.size and np.nanmax(xs) > 0):
        return
    ax.set_xscale("log")
    # Plain minute labels; the default log formatter renders these as 3x10^0 etc.
    lo, hi = np.nanmin(xs[xs > 0]), np.nanmax(xs)
    ticks = [t for t in (1, 2, 5, 10, 20, 30, 60, 90, 120, 180) if lo * 0.9 <= t <= hi * 1.1]
    if len(ticks) >= 2:
        ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())


def _logfit(x: np.ndarray, y: np.ndarray, n_points: int = 100):
    """Least-squares fit of y against log(minutes), evaluated on a smooth grid.

    A straight line in log-minutes is the simplest curve that can express "gains shrink
    as data accumulates", which is what these measures do. Subjects sit at different
    durations, so we fit the pooled raw points rather than averaging over unequal bins --
    a bin containing one subject's 53 min and another's 83 min would report a mean at
    neither. Returns (grid_x, grid_y, slope) or None if there is too little to fit.
    """
    ok = np.isfinite(x) & np.isfinite(y) & (x > 0)
    if ok.sum() < 3 or len(np.unique(x[ok])) < 2:
        return None
    lx = np.log(x[ok])
    slope, intercept = np.polyfit(lx, y[ok], 1)
    gx = np.linspace(x[ok].min(), x[ok].max(), n_points)
    return gx, intercept + slope * np.log(gx), slope


def _scatter_fit(ax, x, y, color, label, marker="o"):
    """Raw points plus a log-minutes trend line through them."""
    ax.scatter(x, y, s=22, color=color, alpha=0.55, marker=marker,
               edgecolors="none", label=label)
    fit = _logfit(np.asarray(x, float), np.asarray(y, float))
    if fit is not None:
        gx, gy, _ = fit
        ax.plot(gx, gy, color=color, lw=2.2, alpha=0.95)
    return fit


def _points(comp: dict, field: str) -> tuple[np.ndarray, np.ndarray]:
    """(minutes, value) for every subject x level with a recorded duration."""
    xs, ys = [], []
    for res in comp.get("per_subject", {}).values():
        for row in res.get("levels", []):
            if row.get("minutes") and row.get(field) is not None:
                xs.append(row["minutes"]); ys.append(row[field])
    return np.array(xs, float), np.array(ys, float)


def _between_points(comp: dict, field: str = "dice") -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for rec in comp.get("between_pairs", []):
        if rec.get("minutes") and rec.get(field) is not None:
            xs.append(rec["minutes"]); ys.append(rec[field])
    return np.array(xs, float), np.array(ys, float)


def _subject_curves(comp: dict, field: str):
    """[(minutes, values)] per subject for a per-subject field."""
    out = []
    for res in comp.get("per_subject", {}).values():
        rows = [r for r in res.get("levels", []) if r.get("minutes")
                and r.get(field) is not None]
        if len(rows) >= 2:
            out.append((np.array([r["minutes"] for r in rows], float),
                        np.array([r[field] for r in rows], float)))
    return out


def _network_curves(comp: dict, key: str):
    """{network: (minutes, value)} — one smooth curve per network, averaged over subjects.

    Each subject contributes their own curve on their own durations; those are then
    interpolated onto a shared grid and averaged. Pooling the raw points from several
    subjects into one sorted series instead produces a sawtooth, because subjects sit at
    interleaved x with different values.
    """
    per_net: dict[str, list] = {}
    for res in comp.get("per_subject", {}).values():
        rows = [r for r in res.get("levels", []) if r.get("minutes")]
        by_net: dict[str, list] = {}
        for row in rows:
            for net, val in (row.get(key) or {}).items():
                if val is not None and np.isfinite(val):
                    by_net.setdefault(net, []).append((row["minutes"], val))
        for net, pts in by_net.items():
            if len(pts) >= 2:
                per_net.setdefault(net, []).append(
                    (np.array([p[0] for p in pts], float),
                     np.array([p[1] for p in pts], float)))

    out = {}
    for net, curves in per_net.items():
        got = _grid_mean(curves)
        if got:
            gx, gy, _ = got
            out[net] = (gx, gy)
    return out


def _grid_mean(curves, n_points: int = 40):
    """Average curves that sit on different x by interpolating onto a shared grid.

    Restricted to the range every curve actually covers, so the mean is never an
    extrapolation of any subject.
    """
    curves = [(x, y) for x, y in curves if len(x) >= 2]
    if not curves:
        return None
    lo = max(float(np.min(x)) for x, _ in curves)
    hi = min(float(np.max(x)) for x, _ in curves)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return None
    grid = np.linspace(lo, hi, n_points)
    stack = []
    for x, y in curves:
        order = np.argsort(x)
        stack.append(np.interp(grid, x[order], y[order]))
    arr = np.array(stack)
    mean = arr.mean(axis=0)
    if len(arr) > 1:
        sem = arr.std(axis=0, ddof=1) / np.sqrt(len(arr))
    else:
        sem = np.zeros_like(grid)
    return grid, mean, sem


def _faint_then_mean(ax, curves, color, label, lw=2.6):
    """Every curve at alpha 0.2, the interpolated mean bold on top."""
    for x, y in curves:
        order = np.argsort(x)
        ax.plot(x[order], y[order], color=color, alpha=0.2, lw=1)
    got = _grid_mean(curves)
    if got:
        gx, gy, gsem = got
        ax.fill_between(gx, gy - gsem, gy + gsem, color=color, alpha=0.25, lw=0)
        ax.plot(gx, gy, color=color, lw=lw, label=label)
    return got


def _fd_series(comp: dict, field: str):
    """Per-subject finite differences dy/dlog(x): {mid_minutes: [slopes across subjects]}."""
    from . import compare as _cmp
    buckets: dict[float, list[float]] = {}
    for res in comp.get("per_subject", {}).values():
        rows = [r for r in res.get("levels", []) if r.get("minutes")
                and r.get(field) is not None]
        if len(rows) < 2:
            continue
        fds = _cmp.finite_differences([r["minutes"] for r in rows],
                                      [r[field] for r in rows])
        for f in fds:
            buckets.setdefault(round(f["mid_minutes"], 3), []).append(f["slope"])
    return buckets


def _fd_between(comp: dict):
    """Finite differences of the between-person null, pooled per level."""
    from . import compare as _cmp
    by_level: dict[str, list] = {}
    mins: dict[str, list] = {}
    for rec in comp.get("between_pairs", []):
        if rec.get("minutes") and rec.get("dice") is not None:
            by_level.setdefault(rec["level"], []).append(rec["dice"])
            mins.setdefault(rec["level"], []).append(rec["minutes"])
    levels = [config.level_name(b) for b in config.STABILITY_BLOCKS if b and
              config.level_name(b) in by_level]
    if len(levels) < 2:
        return {}
    x = [float(np.mean(mins[lv])) for lv in levels]
    y = [float(np.mean(by_level[lv])) for lv in levels]
    return {round(f["mid_minutes"], 3): [f["slope"]]
            for f in _cmp.finite_differences(x, y)}


def _plot_fd(ax, buckets, color, label, marker="o"):
    """Mean +/- SEM of the finite differences across subjects, at each midpoint."""
    if not buckets:
        return None
    xs = sorted(buckets)
    means = [float(np.mean(buckets[x])) for x in xs]
    sems = [float(np.std(buckets[x], ddof=1) / np.sqrt(len(buckets[x])))
            if len(buckets[x]) > 1 else 0.0 for x in xs]
    ax.errorbar(xs, means, yerr=sems, marker=marker, color=color, lw=2, capsize=3,
                label=label)
    return xs, means


def plot_comparisons(results: dict, path: Path) -> None:
    """Five panels on a linear minutes axis.

    Everything is shown -- every subject, every network -- faintly, with the mean bold.
    No network is selected for display, so the mean cannot be flattered by the choice.
    """
    comp = results.get("comparisons", {})
    if not comp.get("per_subject"):
        return
    n = len(comp["per_subject"])
    names = comp.get("networks", [])

    fig, axes = plt.subplots(2, 4, figsize=(22, 9))
    axes = axes.ravel()

    # --- A: individuation, every subject faint --------------------------------
    ax = axes[0]
    _faint_then_mean(ax, _subject_curves(comp, "within"), STABLE_COLOR, "within-person")
    _faint_then_mean(ax, _subject_curves(comp, "to_group"), "#7b3294",
                     "similarity to group")
    btw = {}
    for rec in comp.get("between_pairs", []):
        if rec.get("minutes") and rec.get("dice") is not None:
            btw.setdefault(tuple(rec["subjects"]), []).append((rec["minutes"], rec["dice"]))
    bcurves = [(np.array([p[0] for p in v]), np.array([p[1] for p in v]))
               for v in btw.values()]
    _faint_then_mean(ax, bcurves, "#999999", "between-person (null)")
    ax.set_ylabel("Dice"); ax.set_xlabel("minutes of rest per map")
    ax.set_ylim(0, 1.02)
    ax.set_title(f"A. Individuation (n={n})")
    ax.legend(fontsize=8, loc="upper left"); ax.grid(alpha=0.25)

    # --- B: within-person, all 17 networks ------------------------------------
    ax = axes[1]
    nets = _network_curves(comp, "within_per_network")
    for net, (x, y) in nets.items():
        order = np.argsort(x)
        ax.plot(x[order], y[order], color=STABLE_COLOR, alpha=0.2, lw=1)
    _grid = _grid_mean(list(nets.values()))
    if _grid:
        gx, gy, _ = _grid
        ax.plot(gx, gy, color=STABLE_COLOR, lw=2.8, label="mean of 17 networks")
    ax.set_ylabel("within-person Dice"); ax.set_xlabel("minutes of rest per map")
    ax.set_ylim(0, 1.02)
    ax.set_title("B. Every network (within-person)")
    ax.legend(fontsize=8, loc="upper left"); ax.grid(alpha=0.25)

    # --- C: network consistency across subjects (the between-person null) -----
    ax = axes[2]
    bnet: dict[str, list] = {}
    for rec in comp.get("between_pairs", []):
        pass  # per-network between lives on the aggregated rows, below
    for row in comp.get("between", []):
        per = row.get("per_network") or {}
        # x is the mean duration of that level across subjects
        mins = [r["minutes"] for res in comp["per_subject"].values()
                for r in res["levels"] if r["level"] == row["level"] and r.get("minutes")]
        if not mins:
            continue
        for net, val in per.items():
            if val is not None and np.isfinite(val):
                bnet.setdefault(net, []).append((float(np.mean(mins)), val))
    bcur = {n_: (np.array([p[0] for p in v]), np.array([p[1] for p in v]))
            for n_, v in bnet.items()}
    for net, (x, y) in bcur.items():
        order = np.argsort(x)
        ax.plot(x[order], y[order], color="#999999", alpha=0.25, lw=1)
    g = _grid_mean(list(bcur.values()))
    if g:
        gx, gy, _ = g
        ax.plot(gx, gy, color="#555555", lw=2.8, label="mean of 17 networks")
    ax.set_ylabel("between-person Dice"); ax.set_xlabel("minutes of rest per map")
    ax.set_ylim(0, 1.02)
    ax.set_title("C. Network consistency across subjects\n(high = network looks alike in everyone)")
    ax.legend(fontsize=8, loc="upper left"); ax.grid(alpha=0.25)

    # --- D: within minus between, per network = individuation per network -----
    ax = axes[3]
    gaps = {}
    for net, (wx, wy) in nets.items():
        if net not in bcur:
            continue
        bx, by = bcur[net]
        order = np.argsort(wx)
        gx_, gy_ = wx[order], wy[order]
        interp = np.interp(gx_, bx[np.argsort(bx)], by[np.argsort(bx)])
        gaps[net] = (gx_, gy_ - interp)
    for net, (x, y) in gaps.items():
        ax.plot(x, y, color="#41ab5d", alpha=0.25, lw=1)
    g = _grid_mean(list(gaps.values()))
    if g:
        gx, gy, _ = g
        ax.plot(gx, gy, color="#238443", lw=2.8, label="mean of 17 networks")
    ax.axhline(0, ls="--", lw=1, color="gray")
    ax.set_ylabel("within − between (Dice)"); ax.set_xlabel("minutes of rest per map")
    ax.set_title("D. Individuation per network\n(how far above the null)")
    ax.legend(fontsize=8, loc="upper left"); ax.grid(alpha=0.25)

    # --- E: ranked individuation gap at the most data ------------------------
    ax = axes[4]
    ranked = sorted(((net, y[-1]) for net, (x, y) in gaps.items()),
                    key=lambda kv: kv[1])
    if ranked:
        fam = {n_: f for f, ns in config.NETWORK_FAMILIES.items() for n_ in ns}
        colors = ["#41ab5d" if fam.get(k) == "association" else "#2c7fb8"
                  for k, _ in ranked]
        ax.barh([k for k, _ in ranked], [v for _, v in ranked], color=colors)
        ax.set_xlabel("within − between at the most data")
        ax.tick_params(axis="y", labelsize=7)
        ax.set_title("E. Which networks individuate most\n(green = association, blue = sensorimotor)")
        ax.grid(alpha=0.25, axis="x")

    # --- F: THE derivative panel ---------------------------------------------
    # If within-person finite differences stay flat while similarity-to-group's decline,
    # the two are not offset versions of one curve -- they are different shapes. Group
    # similarity saturates; individuation does not. That is the core claim.
    ax = axes[5]
    fd_w = _fd_series(comp, "within")
    fd_g = _fd_series(comp, "to_group")
    fd_b = _fd_between(comp)
    _plot_fd(ax, fd_w, STABLE_COLOR, "within-person", "o")
    _plot_fd(ax, fd_b, "#999999", "between-person", "s")
    _plot_fd(ax, fd_g, "#7b3294", "similarity to group", "^")
    ax.axhline(0, ls="--", lw=1, color="gray")
    ax.set_ylabel("d(Dice) / d log(minutes)")
    ax.set_xlabel("minutes of rest per map (midpoint)")
    ax.set_title("F. Are the gains slowing?\nflat = still improving, falling = saturating")
    ax.legend(fontsize=8, loc="upper right"); ax.grid(alpha=0.25)

    # Say plainly whether the individual signal is exhausted at the most data we have.
    if fd_w:
        last_x = max(fd_w)
        last = float(np.mean(fd_w[last_x]))
        first = float(np.mean(fd_w[min(fd_w)]))
        verdict = ("still rising at the most data\n(individual signal NOT exhausted)"
                   if last > 0.5 * first and last > 0 else "flattening")
        ax.annotate(f"within-person: {verdict}", (0.03, 0.04), xycoords="axes fraction",
                    fontsize=7.5, bbox=dict(boxstyle="round,pad=0.3", fc="white",
                                            alpha=0.85))

    # --- G: identification margin --------------------------------------------
    ax = axes[6]
    total = comp.get("total_minutes", {})
    curves = []
    for sub, res in results["subjects"].items():
        pts = [(total[sub] * int(r["level"].split("/")[0]) / config.N_CHUNKS, r["margin"])
               for r in res.get("margin", [])
               if sub in total and r.get("margin") is not None]
        if len(pts) >= 2:
            curves.append((np.array([p[0] for p in pts]), np.array([p[1] for p in pts])))
    if curves:
        _faint_then_mean(ax, curves, SIGNAL_COLOR, "mean margin")
    ax.axhline(0, ls="--", lw=1, color="gray")
    ax.set_ylabel("margin (correct − best wrong)")
    ax.set_xlabel("minutes of rest per map")
    ax.set_title("G. Identification margin")
    ax.legend(fontsize=8, loc="upper left"); ax.grid(alpha=0.25)

    axes[7].axis("off")
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
