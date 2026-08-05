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


def between_subjects(subjects: list[str], block: int,
                     minutes: dict[str, float] | None = None
                     ) -> tuple[np.ndarray | None, list[dict]]:
    """Per-network Dice between different subjects at matched chunk and data amount.

    Matched means the same (start, size) block for both people, so a difference cannot
    come from one map having more data *within* the block. Two subjects still differ in
    absolute duration, so each pair also carries the mean of their two durations -- that
    is the x-coordinate when the axis is minutes.

    Returns (mean over pairs per network, per-pair records).
    """
    rows, pairs = [], []
    for start in range(0, config.N_CHUNKS, block):
        spec = (start, block)
        have = [s for s in subjects if parcellation.has_map(s, spec)]
        for a, b in combinations(have, 2):
            per = parcellation.dice_per_network(parcellation.load_map(a, spec),
                                                parcellation.load_map(b, spec))
            rows.append(per)
            rec = {"level": config.level_name(block), "subjects": [a, b],
                   "dice": family_mean(per),
                   "association": family_mean(per, "association"),
                   "sensorimotor": family_mean(per, "sensorimotor")}
            if minutes and a in minutes and b in minutes:
                frac = block / config.N_CHUNKS
                rec["minutes"] = 0.5 * (minutes[a] + minutes[b]) * frac
            pairs.append(rec)
    return (_nanmean_rows(rows) if rows else None), pairs


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


def compare_all(subjects: list[str], minutes: dict[str, float] | None = None) -> dict:
    """Every comparison at every level, per network and per family.

    ``minutes`` maps subject -> total minutes of rest, so each point can carry its real
    duration. Fractions are not comparable across subjects (one person's 8/16 may be
    53 min and another's 83 min), so minutes is the interpretable x-axis.
    """
    names = list(parcellation.network_order())
    out: dict = {"networks": names, "per_subject": {}, "between": [],
                 "between_pairs": [], "total_minutes": dict(minutes or {})}

    for block in config.STABILITY_BLOCKS:
        level = config.level_name(block)
        btw, pairs = between_subjects(subjects, block, minutes)
        out["between_pairs"].extend(pairs)
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
                "minutes": (minutes[subject] * block / config.N_CHUNKS
                            if minutes and subject in minutes else None),
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


# ------------------------------------------------------------------ slopes / saturation
def log_slope(minutes, values) -> dict:
    """Least-squares fit of y against log(minutes): y = a + b*log(x).

    ``b`` is the gain per e-fold of scan time; ``per_doubling`` = b*ln2 is the more
    readable "how much do I gain by scanning twice as long".
    """
    x = np.asarray(minutes, float)
    y = np.asarray(values, float)
    ok = np.isfinite(x) & np.isfinite(y) & (x > 0)
    if ok.sum() < 2 or len(np.unique(x[ok])) < 2:
        return {"b": float("nan"), "intercept": float("nan"), "r2": float("nan"),
                "per_doubling": float("nan"), "n": int(ok.sum())}
    lx, yy = np.log(x[ok]), y[ok]
    b, a = np.polyfit(lx, yy, 1)
    resid = yy - (a + b * lx)
    ss_tot = float(((yy - yy.mean()) ** 2).sum())
    r2 = 1.0 - float((resid ** 2).sum()) / ss_tot if ss_tot > 0 else float("nan")
    return {"b": float(b), "intercept": float(a), "r2": float(r2),
            "per_doubling": float(b * np.log(2)), "n": int(ok.sum())}


def finite_differences(minutes, values) -> list[dict]:
    """Slope between adjacent levels: dy / dlog(x).

    If the curve really is log-linear these are constant; if they decline, the single
    fitted slope is hiding saturation and the log-linear form is the wrong model.
    """
    order = np.argsort(np.asarray(minutes, float))
    x = np.asarray(minutes, float)[order]
    y = np.asarray(values, float)[order]
    out = []
    for i in range(len(x) - 1):
        if not (np.isfinite(x[i]) and np.isfinite(x[i + 1]) and x[i] > 0
                and np.isfinite(y[i]) and np.isfinite(y[i + 1])):
            continue
        dlog = np.log(x[i + 1]) - np.log(x[i])
        if dlog <= 0:
            continue
        out.append({"from_minutes": float(x[i]), "to_minutes": float(x[i + 1]),
                    "mid_minutes": float(np.sqrt(x[i] * x[i + 1])),
                    "slope": float((y[i + 1] - y[i]) / dlog),
                    "per_doubling": float((y[i + 1] - y[i]) / dlog * np.log(2))})
    return out


def declining(fds: list[dict], tol: float = 0.0) -> bool:
    """True if the finite differences trend downward -- i.e. the curve is saturating."""
    if len(fds) < 3:
        return False
    x = np.log([f["mid_minutes"] for f in fds])
    y = np.array([f["slope"] for f in fds])
    return bool(np.polyfit(x, y, 1)[0] < -tol)


def saturating_fit(minutes, values, threshold: float = 0.01) -> dict | None:
    """Fit y = ymax * (1 - exp(-x/tau)) and find where the gains go flat.

    ``enough_minutes`` is where scanning twice as long buys less than ``threshold`` Dice.
    That is the defensible "enough data" number when the curve saturates -- a log-linear
    fit can never produce one, because its gain per doubling is constant by construction.
    """
    from scipy.optimize import curve_fit

    x = np.asarray(minutes, float)
    y = np.asarray(values, float)
    ok = np.isfinite(x) & np.isfinite(y) & (x > 0)
    if ok.sum() < 3:
        return None
    x, y = x[ok], y[ok]

    def model(t, ymax, tau):
        return ymax * (1.0 - np.exp(-t / tau))

    try:
        p, _ = curve_fit(model, x, y, p0=[max(float(y.max()), 1e-3), float(np.median(x))],
                         bounds=([0, 1e-3], [1.5, 1e5]), maxfev=20000)
    except Exception:
        return None
    ymax, tau = float(p[0]), float(p[1])
    resid = y - model(x, ymax, tau)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float((resid ** 2).sum()) / ss_tot if ss_tot > 0 else float("nan")

    grid = np.linspace(x.min(), max(x.max() * 6, x.max() + 1), 4000)
    gain = model(2 * grid, ymax, tau) - model(grid, ymax, tau)
    below = np.flatnonzero(gain < threshold)
    enough = float(grid[below[0]]) if below.size else None
    return {"ymax": ymax, "tau": tau, "r2": r2, "threshold": threshold,
            "enough_minutes": enough,
            "extrapolated": bool(enough is not None and enough > float(x.max()))}


def slope_report(comp: dict, threshold: float = 0.01) -> dict:
    """Slopes, finite differences and (if the curve saturates) an 'enough data' point.

    Per subject and pooled, for within / between / to_group. The ratio
    b_within / b_between says how much faster individuation accrues than the baseline
    drift that affects everyone's maps equally.
    """
    per_subject: dict[str, dict] = {}
    pooled: dict[str, list[tuple[float, float]]] = {"within": [], "to_group": [],
                                                    "between": []}

    for sub, res in comp.get("per_subject", {}).items():
        rows = [r for r in res.get("levels", []) if r.get("minutes")]
        if len(rows) < 2:
            continue
        x = [r["minutes"] for r in rows]
        entry = {}
        for field in ("within", "to_group"):
            y = [r.get(field) for r in rows]
            entry[field] = {"slope": log_slope(x, y),
                            "finite_differences": finite_differences(x, y)}
            pooled[field].extend((xi, yi) for xi, yi in zip(x, y)
                                 if yi is not None and np.isfinite(yi))
        bw = entry["within"]["slope"]["b"]
        entry["crossover_level"] = res.get("crossover")
        per_subject[sub] = entry
        entry["b_within"] = bw

    for rec in comp.get("between_pairs", []):
        if rec.get("minutes") and rec.get("dice") is not None:
            pooled["between"].append((rec["minutes"], rec["dice"]))

    group: dict[str, dict] = {}
    for field, pts in pooled.items():
        if not pts:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        # Finite differences must be taken WITHIN a subject and then averaged. Pooling
        # subjects into one sorted series and differencing consecutive entries hops
        # between people at interleaved durations and produces alternating nonsense.
        fds = _mean_within_subject_fds(per_subject, field) if field != "between" \
            else finite_differences(*_bin_by_x(xs, ys))
        entry = {"slope": log_slope(xs, ys), "finite_differences": fds,
                 "finite_differences_decline": declining(fds)}
        if entry["finite_differences_decline"]:
            entry["saturating"] = saturating_fit(xs, ys, threshold)
        group[field] = entry

    bw = group.get("within", {}).get("slope", {}).get("b", float("nan"))
    bb = group.get("between", {}).get("slope", {}).get("b", float("nan"))
    ratio = float(bw / bb) if (np.isfinite(bw) and np.isfinite(bb) and bb != 0) else float("nan")

    subj_b = [v["b_within"] for v in per_subject.values() if np.isfinite(v.get("b_within", np.nan))]
    return {
        "threshold": threshold,
        "per_subject": per_subject,
        "group": group,
        "b_within_over_b_between": ratio,
        "b_within_subject_mean": float(np.mean(subj_b)) if subj_b else float("nan"),
        "b_within_subject_sd": float(np.std(subj_b, ddof=1)) if len(subj_b) > 1 else 0.0,
    }


def _bin_by_x(xs, ys, decimals: int = 3):
    """Average duplicate x positions so finite differences step through distinct levels."""
    x = np.round(np.asarray(xs, float), decimals)
    y = np.asarray(ys, float)
    uniq = np.unique(x)
    return uniq.tolist(), [float(np.nanmean(y[x == u])) for u in uniq]


def _mean_within_subject_fds(per_subject: dict, field: str) -> list[dict]:
    """Average each subject's own finite differences, matched by data level.

    Subjects sit at different durations, so we key by level index rather than by minutes
    and report the mean midpoint. This keeps every difference within one subject.
    """
    by_index: dict[int, list[dict]] = {}
    for entry in per_subject.values():
        for i, fd in enumerate(entry.get(field, {}).get("finite_differences", [])):
            by_index.setdefault(i, []).append(fd)
    out = []
    for i in sorted(by_index):
        group = by_index[i]
        slopes = [f["slope"] for f in group]
        out.append({
            "mid_minutes": float(np.mean([f["mid_minutes"] for f in group])),
            "slope": float(np.mean(slopes)),
            "sem": float(np.std(slopes, ddof=1) / np.sqrt(len(slopes)))
            if len(slopes) > 1 else 0.0,
            "per_doubling": float(np.mean([f["per_doubling"] for f in group])),
            "n_subjects": len(group),
        })
    return out
