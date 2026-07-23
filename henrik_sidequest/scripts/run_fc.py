"""How much rest data does a personal covariance need — to be stable, and to be identifying?

Volumetric ds006598. The download is 180 GB and the hub has 15, so the only real machinery
is a loop that pulls one run, shrinks it, deletes it. Everything downstream runs on the
few-MB reductions.

Per run we keep three arrays (nilearn does the work):
    parcels  (T, 400)  Schaefer-400 parcel means, RAW  (NiftiLabelsMasker)
    gs       (T,)      whole-brain mean signal          (NiftiMasker on the brain mask)
    dvars    (T,)      frame-to-frame RMS change        (the motion-spike proxy)
Raw on purpose: cleaning is nilearn.signal.clean at analysis time (global-signal
regression + band-pass + z-score), so any recipe re-runs in seconds on the small files.

    python scripts/run_fc.py reduce  --subjects PAN01            # reduce, keep BOLD
    python scripts/run_fc.py reduce  --subjects PAN01 --cleanup  # reduce, then delete BOLD
    python scripts/run_fc.py inspect --subjects PAN01            # one run, look, delete nothing
    python scripts/run_fc.py analyze                             # the two curves + control

Deletion is the only irreversible step: `inspect` first, and `--cleanup` skips any run whose
sanity check fails.
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
import warnings
from contextlib import contextmanager
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from panmvpa import config, rest  # noqa: E402

TR = config.TR
BANDPASS = dict(low_pass=0.08, high_pass=0.009, t_r=TR)


@contextmanager
def _quiet_nilearn():
    """Silence nilearn's per-call warnings for the one wrapped call only.

    nilearn prints a confound-standardization deprecation while cleaning (a future version
    divides by a slightly different number -- harmless, results unaffected), plus the masker
    standardize/resampling notices. Matching by class is brittle: our nilearn raises it as
    FutureWarning, others as DeprecationWarning. So we don't match by class or message -- we
    scope-ignore warnings around a numerically-trusted nilearn call and restore immediately
    after. Not a global filter; genuine errors still raise.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


# ---------------------------------------------------------------- reduce one run
def brain_mask_path() -> Path:
    p = config.ASSET_DIR / "brain_mask_NLin6Asym_2mm.nii.gz"
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(config.BRAIN_MASK_URL, p)
    return p


_maskers: dict = {}
def _get_maskers():
    """Fit the two maskers once; both grids match the BOLD so nothing is resampled."""
    if not _maskers:
        from nilearn.maskers import NiftiLabelsMasker, NiftiMasker
        _maskers["labels"] = NiftiLabelsMasker(
            str(config.ATLAS_IMAGE), standardize=False, detrend=False).fit()
        _maskers["brain"] = NiftiMasker(
            mask_img=str(brain_mask_path()), standardize=False, detrend=False).fit()
    return _maskers["labels"], _maskers["brain"]


def reduce_run(bold_path: str | Path) -> dict:
    labels, brain = _get_maskers()
    with _quiet_nilearn():
        parcels = labels.transform(str(bold_path)).astype(np.float32)   # (T, 400)
        b = brain.transform(str(bold_path)).astype(np.float32)          # (T, n_brain)
    gs = b.mean(axis=1)
    dvars = np.full(b.shape[0], np.nan, dtype=np.float32)   # frame 0 undefined
    dvars[1:] = np.sqrt((np.diff(b, axis=0) ** 2).mean(axis=1))
    return {"parcels": parcels, "gs": gs, "dvars": dvars, "tr": np.float32(TR)}


def sanity(rec: dict) -> tuple[bool, str]:
    """Cheap checks before any deletion. Returns (ok, one-line summary)."""
    p, gs, dv = rec["parcels"], rec["gs"], rec["dvars"]
    dead = int((p.std(axis=0) == 0).sum() + (~np.isfinite(p).all(axis=0)).sum())
    drift = float(np.ptp(gs) / (np.std(np.diff(gs)) + 1e-9))       # wanders > it jitters?
    spikes = int((dv[1:] > dv[1:].mean() + 3 * dv[1:].std()).sum())
    ok = dead == 0 and np.isfinite(gs).all()
    return ok, (f"T={p.shape[0]} dead={dead} gs_drift={drift:.1f} "
                f"dvars_spikes={spikes} {'OK' if ok else 'CHECK'}")


# ---------------------------------------------------------------- persistence
def reduced_path(sid: str, ses: int, run: int) -> Path:
    return config.REDUCED_DIR / f"sub-{sid}_ses-{ses:02d}_run-{run:02d}.npz"


def load_reduced(path) -> dict:
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


# ---------------------------------------------------------------- stages
def stage_reduce(subjects, cleanup: bool, inspect_only: bool) -> None:
    config.REDUCED_DIR.mkdir(parents=True, exist_ok=True)
    for subject in subjects:
        sid = config.sub_id(subject)
        runs = rest.rest_runs(subject)
        if not runs:
            print(f"{sid}: no rest runs on disk.")
            continue
        print(f"\n=== {sid}: {len(runs)} rest runs ===")
        freed = 0.0
        for scan in runs:
            out = reduced_path(sid, scan.session, scan.run)
            if out.exists() and not inspect_only:
                if cleanup:
                    freed += _drop(scan.path)
                continue
            rec = reduce_run(scan.path)
            ok, msg = sanity(rec)
            print(f"  ses-{scan.session} run-{scan.run}: {msg}")
            if inspect_only:
                _diagnostic_figure(rec, sid, scan.session, scan.run)
                return
            np.savez_compressed(out, **rec)
            if cleanup and ok:
                freed += _drop(scan.path)
            elif cleanup:
                print("    kept BOLD — sanity failed.")
        if cleanup:
            print(f"  freed {freed:.1f} GB")


def _drop(path: Path) -> float:
    """Delete one BOLD (annex drop or unlink). Returns GB freed."""
    import shutil, subprocess
    root = config.DATA_ROOT
    try:
        size = path.resolve(strict=True).stat().st_size
    except (FileNotFoundError, OSError):
        return 0.0
    if (root / ".git" / "annex").exists() and shutil.which("git-annex"):
        subprocess.run(["git", "-C", str(root), "annex", "drop", "--force",
                        str(path.relative_to(root))], capture_output=True)
    else:
        try:
            path.unlink()
        except (FileNotFoundError, OSError):
            return 0.0
    return size / 1e9


def _diagnostic_figure(rec, sid, ses, run) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = np.arange(rec["parcels"].shape[0]) * TR / 60.0
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.4))
    ax[0].plot(t, rec["gs"], lw=0.8); ax[0].set_title("whole-brain mean (wanders slowly)")
    ax[1].plot(t, rec["dvars"], lw=0.8); ax[1].set_title("DVARS (flat + spikes)")
    p = rec["parcels"].T
    p = p - p.mean(axis=1, keepdims=True)   # demean each parcel so fluctuations show
    ax[2].imshow(p, aspect="auto", cmap="coolwarm",
                 vmin=np.percentile(p, 2), vmax=np.percentile(p, 98),
                 extent=[0, t[-1], 400, 0])
    ax[2].set_title("400 parcels, demeaned (no dead rows)")
    for a in ax[:2]:
        a.set_xlabel("minutes")
    fig.suptitle(f"{sid} ses-{ses} run-{run} — reduction sanity")
    fig.tight_layout()
    config.FC_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = config.FC_RESULTS_DIR / f"inspect_{sid}_ses-{ses:02d}_run-{run:02d}.png"
    fig.savefig(out, dpi=110); plt.close(fig)
    print(f"  figure -> {out}")
    return out


# ---------------------------------------------------------------- analysis (tiny files)
def clean_run(rec: dict, gsr: bool = True) -> np.ndarray:
    """Global-signal regression (optional) + band-pass + z-score the 400 timelines."""
    from nilearn import signal
    conf = rec["gs"][:, None] if gsr else None
    with _quiet_nilearn():   # confound-standardization deprecation, fires per run
        return signal.clean(rec["parcels"], confounds=conf, detrend=True,
                            standardize="zscore_sample", **BANDPASS)


def edges(fc: np.ndarray) -> np.ndarray:
    iu = np.triu_indices_from(fc, k=1)
    return fc[iu]


def fc_edges(ts: np.ndarray) -> np.ndarray:
    """Upper-triangle of the parcel correlation matrix from (T, 400)."""
    return edges(np.corrcoef(ts.T))


# Overlap ladder: trains nothing, so it runs the full cohort range. A subject with too
# little rest to fill the first half at a given minute simply contributes NaN there and
# drops out of that point's average.
LADDER = np.array([1, 2, 3, 4, 5, 7.5, 10, 15, 20, 30, 45, 60, 80], dtype=float)

# Minimum timepoints for a covariance worth correlating. With 400 parcels the matrix is
# rank-deficient below ~400 TR, but every edge is still defined -- that low-data noise is
# exactly the convergence we are measuring -- so the floor is just "enough for a Pearson r".
MIN_TP = 20


def _first_minutes(ts: np.ndarray, minutes: float) -> np.ndarray:
    return ts[: int(round(minutes * 60.0 / TR))]


def _sessions() -> dict[str, dict[int, list[dict]]]:
    """{sid: {session: [run records, in run order]}} from the reduced .npz files."""
    import re
    out: dict[str, dict[int, list[dict]]] = {}
    pat = re.compile(r"sub-(\w+)_ses-(\d+)_run-(\d+)\.npz")
    for f in sorted(config.REDUCED_DIR.glob("sub-*_ses-*_run-*.npz")):
        m = pat.search(f.name)
        if not m:
            continue
        sid, ses = m.group(1), int(m.group(2))
        out.setdefault(sid, {}).setdefault(ses, []).append(load_reduced(f))
    return out


def _session_ts(runs: list[dict], gsr: bool) -> np.ndarray:
    """Clean each run, then concatenate within the session -> (T, 400)."""
    return np.concatenate([clean_run(r, gsr) for r in runs], axis=0)


def _corr(u: np.ndarray, v: np.ndarray) -> float:
    return float(np.corrcoef(u, v)[0, 1])


def _colmean(mat: np.ndarray) -> np.ndarray:
    """Column mean over the rows that are finite, NaN where a column is entirely NaN --
    without np.nanmean's empty-slice warning."""
    counts = np.sum(~np.isnan(mat), axis=0)
    sums = np.nansum(mat, axis=0)
    return np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)


def identity_curves(sess: dict[str, dict[int, list[dict]]], gsr: bool) -> dict:
    """Within- and between-person FC agreement, both grown on the same minute ladder.

    The raw within-person curve is meaningless alone: two halves of ONE person already agree
    ~0.9, two DIFFERENT people ~0.6 -- most of a connectivity table is just "human cortex,"
    and the between-person floor itself rises with data. So we grow both from the identical
    A-side estimate and change only the reference:

        within_A(X)  = corr( FC(A first-half, first X min),  FC(A second-half, full) )
        between_A(X) = mean_B!=A corr( FC(A first-half, first X min),  FC(B second-half, full) )
        gap_A(X)     = within_A(X) - between_A(X)          # per subject, THEN averaged

    The gap is formed within each subject before averaging -- never mean(within)-mean(between),
    which would difference one subject's within against a different mix and compare A to B
    rather than A to itself.
    """
    subs = sorted(sess)
    first_half, ref_edges = {}, {}
    for sid in subs:
        runs = [r for ses in sorted(sess[sid]) for r in sess[sid][ses]]
        X = np.concatenate([clean_run(r, gsr) for r in runs], axis=0)
        h = X.shape[0] // 2
        first_half[sid] = X[:h]
        ref_edges[sid] = fc_edges(X[h:])

    within = {s: np.full(len(LADDER), np.nan) for s in subs}
    between = {s: np.full(len(LADDER), np.nan) for s in subs}
    for sid in subs:
        a = first_half[sid]
        for i, m in enumerate(LADDER):
            n = int(round(m * 60.0 / TR))
            if not (MIN_TP <= n <= a.shape[0]):      # only points the subject truly has
                continue
            grow = fc_edges(a[:n])
            within[sid][i] = _corr(grow, ref_edges[sid])
            others = [_corr(grow, ref_edges[b]) for b in subs if b != sid]
            if others:
                between[sid][i] = float(np.mean(others))
    gap = {s: within[s] - between[s] for s in subs}

    stack = lambda d: np.vstack([d[s] for s in subs])
    W = stack(within)
    return {
        "minutes": LADDER, "subs": subs,
        "within": within, "between": between, "gap": gap,
        "within_mean": _colmean(W), "between_mean": _colmean(stack(between)),
        "gap_mean": _colmean(stack(gap)),
        "n": np.sum(~np.isnan(W), axis=0),
    }


def _slope(minutes: np.ndarray, values: np.ndarray, lo: float, hi: float) -> float:
    """Linear-axis slope (per minute) of a curve over [lo, hi]. The plot is linear, so the
    slope is too -- an earlier ratio computed on a log axis under a linear plot was wrong."""
    m = np.isfinite(values) & (minutes >= lo) & (minutes <= hi)
    return float(np.polyfit(minutes[m], values[m], 1)[0]) if m.sum() >= 2 else float("nan")


def _reaches(minutes: np.ndarray, values: np.ndarray, target: float) -> float:
    """First minute at which `values` rises through `target`, linearly interpolated between
    the bracketing rungs. No fitted asymptote, no assumed functional form."""
    for i in range(1, len(minutes)):
        a, b = values[i - 1], values[i]
        if np.isfinite(a) and np.isfinite(b) and a < target <= b:
            frac = (target - a) / (b - a) if b != a else 0.0
            return float(minutes[i - 1] + frac * (minutes[i] - minutes[i - 1]))
    return float("nan")


def _final(minutes: np.ndarray, values: np.ndarray) -> tuple[float, float]:
    """(value, minute) at the largest rung that has data -- each curve's own maximum-data
    value, used to normalise the 90% times (rather than a fitted asymptote)."""
    fin = np.where(np.isfinite(values))[0]
    return (float(values[fin[-1]]), float(minutes[fin[-1]])) if len(fin) else (float("nan"), float("nan"))


# SVM ladder is separate from the overlap ladder and capped: an example eats X minutes, so
# larger X means fewer examples. At X=20 a session is one example (~8/subject); by X=40 an
# example spans two sessions (~4/subject); past that there is nothing to train on. The curve
# stops where any subject has fewer than two examples -- stated, not hidden. The overlap line
# has no such limit because it trains nothing.
SVM_LADDER = np.array([5, 10, 15, 20, 30, 40, 60, 80], dtype=float)


def _bundles(by_ses, minutes: float, gsr: bool) -> list[np.ndarray]:
    """Tile a subject's sessions into examples of `minutes` each, made of WHOLE sessions.

    Sessions are grouped in order until they reach the target, then truncated to exactly
    `minutes`; the trailing partial group is dropped. Because every session belongs to one
    example, leaving an example out leaves whole sessions out -- no two-minutes-from-the-
    same-scan leakage across train and test.
    """
    out, cur, cur_min = [], [], 0.0
    for ses in sorted(by_ses):
        ts = _session_ts(by_ses[ses], gsr)
        cur.append(ts); cur_min += ts.shape[0] * TR / 60.0
        if cur_min >= minutes:
            out.append(_first_minutes(np.concatenate(cur, axis=0), minutes))
            cur, cur_min = [], 0.0
    return out


def _score(feats, y, groups):
    """Leave-one-example-out (== whole-session) linear SVM. Returns (accuracy, mean margin).

    margin = decision score of the true subject minus the best-scoring other subject,
    averaged over held-out examples. Once ten subjects are trivially separable accuracy
    pins at 1.0, but the margin keeps rising -- so it is the line still saying something.

    Works for any number of classes. With exactly two (the sparse high-X rungs, where only
    a couple of subjects have enough data) sklearn's decision_function returns one signed
    score per sample instead of a column per class; that single hyperplane gives antisymmetric
    class scores (+s, -s), so it expands to the same two-column form and margin means exactly
    what it does in the multi-class case: true-class score minus the runner-up's.
    """
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import LinearSVC
    from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
    y = np.asarray(y)
    clf = make_pipeline(StandardScaler(), LinearSVC(dual="auto", C=1.0))
    df = cross_val_predict(clf, np.asarray(feats), y, groups=np.asarray(groups),
                           cv=LeaveOneGroupOut(), method="decision_function")
    if df.ndim == 1:                       # binary -> columns [class_0, class_1] = [-s, +s]
        df = np.column_stack([-df, df])
    classes = np.unique(y)
    ti = np.searchsorted(classes, y)
    acc = float((classes[df.argmax(1)] == y).mean())
    true_score = df[np.arange(len(y)), ti]
    other = df.copy(); other[np.arange(len(y)), ti] = -np.inf
    return acc, float((true_score - other.max(1)).mean())


def _sessions_used(by_ses, minutes: float) -> int:
    """How many sessions the emitted bundles actually consume (the trailing partial group
    is dropped). Mirrors _bundles' reset logic on cheap session lengths, no cleaning."""
    used_total, cur_min, cur_n = 0, 0.0, 0
    for ses in sorted(by_ses):
        cur_min += sum(r["parcels"].shape[0] for r in by_ses[ses]) * TR / 60.0
        cur_n += 1
        if cur_min >= minutes:
            used_total += cur_n
            cur_min, cur_n = 0.0, 0
    return used_total


def identify_table(sess, gsr: bool) -> list[dict]:
    """For each X: build X-minute examples, score FC and the structural control.

    ``used_frac`` is sessions actually behind this rung / sessions available -- it dips
    below 1.0 when X doesn't divide a subject's session count, so a noisier rung isn't
    misread as signal.
    """
    rows = []
    total_sessions = sum(len(v) for v in sess.values())
    for X in SVM_LADDER:
        feats_fc, feats_st, subs, groups, g = [], [], [], [], 0
        per_sub: dict[str, int] = {}
        used = 0
        for sid, by_ses in sorted(sess.items()):
            bs = _bundles(by_ses, X, gsr)
            per_sub[sid] = len(bs)
            used += _sessions_used(by_ses, X)
            for ts in bs:
                feats_fc.append(fc_edges(ts))
                feats_st.append(np.concatenate([ts.mean(0), ts.std(0)]))  # no connectivity
                subs.append(sid); groups.append(g); g += 1
        min_per_sub = min(per_sub.values()) if per_sub else 0
        row = {"min": X, "n": len(subs), "per_sub": min_per_sub,
               "used_frac": used / total_sessions if total_sessions else 0.0}
        if len(set(subs)) >= 2 and min_per_sub >= 2:
            row["acc_fc"], row["margin_fc"] = _score(feats_fc, subs, groups)
            row["acc_st"], row["margin_st"] = _score(feats_st, subs, groups)
        rows.append(row)
    return rows


def stage_analyze(subjects) -> None:
    sess = _sessions()
    if not sess:
        raise SystemExit(f"no reduced files in {config.REDUCED_DIR} — run `reduce` first.")
    n_sub = len(sess)
    print(f"analyze: {n_sub} subjects, "
          f"{sum(len(v) for v in sess.values())} sessions reduced.")

    cur = identity_curves(sess, gsr=True)
    mins, wmn, bmn, gmn, npr = (cur["minutes"], cur["within_mean"],
                                cur["between_mean"], cur["gap_mean"], cur["n"])
    print("\nindividuality: within-person vs between-person FC, same ladder (the main result)")
    print("  min   n   within  between   gap")
    for i, m in enumerate(mins):
        if npr[i] == 0:
            continue
        b = f"{bmn[i]:.3f}" if np.isfinite(bmn[i]) else "  -- "
        g = f"{gmn[i]:+.3f}" if np.isfinite(gmn[i]) else "  -- "
        print(f"  {m:4.1f}  {npr[i]:2d}   {wmn[i]:.3f}   {b}   {g}")

    # Gap only means something against a growing floor, so report both component slopes and
    # their ratio over a stated linear range (the range where both curves have data).
    both = np.isfinite(wmn) & np.isfinite(bmn)
    if both.sum() >= 2:
        lo, hi = float(mins[both].min()), float(mins[both].max())
        sw, sb = _slope(mins, wmn, lo, hi), _slope(mins, bmn, lo, hi)
        ratio = sw / sb if np.isfinite(sb) and sb != 0 else float("nan")
        print(f"\nslope over [{lo:.0f}, {hi:.0f}] min (per minute, linear axis):")
        print(f"  within {sw:+.4f}   between {sb:+.4f}   ratio {ratio:.2f}x "
              f"(individuality accrues {ratio:.1f}x the shared floor)")

    table = identify_table(sess, gsr=True)
    print(f"\nsubject ID vs minutes (chance={1.0/n_sub:.2f}; whole-session holdout):")
    print("  min  examples  used   FC acc / margin    structural acc / margin (control)")
    for r in table:
        used = f"{r['used_frac']:.0%}"
        if "acc_fc" in r:
            print(f"  {r['min']:4.1f}  {r['n']:3d} (min {r['per_sub']}/subj)  {used:>4}  "
                  f"{r['acc_fc']:.2f} / {r['margin_fc']:+.2f}     "
                  f"{r['acc_st']:.2f} / {r['margin_st']:+.2f}")
        else:
            why = "need >=2 subjects" if n_sub < 2 else f"only {r['per_sub']} example(s)/subj — too few"
            print(f"  {r['min']:4.1f}  {r['n']:3d}          {used:>4}  ({why})")

    # Three headline numbers, each normalised to its own curve's maximum-data value.
    scored = [r for r in table if "acc_fc" in r]
    mmin = np.array([r["min"] for r in scored], dtype=float)
    mval = np.array([r["margin_fc"] for r in scored], dtype=float)
    # Crossover: minutes of your OWN data until self-similarity beats a stranger's *stable*
    # map -- within(X) rising through the between-person floor (between at max data). At low X
    # a noisy self-estimate matches a stranger's well-estimated map better than your own, so
    # this can sit well above 1 min. (Same-rung within-vs-between shares the noisy estimate and
    # would cross at rung 1, saying nothing.) Floor taken at the largest rung backed by >=2.
    stable = np.isfinite(bmn) & (npr >= 2)
    bfloor = float(bmn[np.where(stable)[0][-1]]) if stable.any() else float("nan")
    crossover = _reaches(mins, wmn, bfloor) if np.isfinite(bfloor) else float("nan")
    gfin, _ = _final(mins, gmn)
    t90_gap = _reaches(mins, gmn, 0.9 * gfin) if np.isfinite(gfin) else float("nan")
    mfin, _ = _final(mmin, mval) if len(mmin) else (float("nan"), float("nan"))
    t90_margin = _reaches(mmin, mval, 0.9 * mfin) if np.isfinite(mfin) else float("nan")
    fmt = lambda x: f"{x:.1f} min" if np.isfinite(x) else "n/a (need more subjects/data)"
    print("\nheadline numbers:")
    print(f"  crossover (within beats between-person floor {bfloor:.3f}): {fmt(crossover)}"
          if np.isfinite(bfloor) else "  crossover: n/a (need >=2 subjects)")
    print(f"  to 90% of final gap ({gfin:+.3f}): {fmt(t90_gap)}" if np.isfinite(gfin)
          else "  to 90% of final gap: n/a")
    print(f"  to 90% of final margin ({mfin:+.2f}): {fmt(t90_margin)}" if np.isfinite(mfin)
          else "  to 90% of final margin: n/a")

    _analysis_figure(cur, table)


def _analysis_figure(cur, table) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    mins = cur["minutes"]
    wmn, bmn, npr = cur["within_mean"], cur["between_mean"], cur["n"]
    fig, ax = plt.subplots(2, 1, figsize=(9, 8.5), sharex=True)

    # Top: within, between, gap shaded; a thin line per subject behind each bold mean.
    for sid in cur["subs"]:
        ax[0].plot(mins, cur["within"][sid], color="C0", lw=0.5, alpha=0.3)
        ax[0].plot(mins, cur["between"][sid], color="C1", lw=0.5, alpha=0.3)
    both = np.isfinite(wmn) & np.isfinite(bmn)
    ax[0].fill_between(mins, bmn, wmn, where=both, color="C2", alpha=0.15,
                       label="gap = individuality")
    ax[0].plot(mins, wmn, "o-", color="C0", lw=2, label="within (own other half)")
    ax[0].plot(mins, bmn, "s-", color="C1", lw=2, label="between (a stranger)")
    ymin = np.nanmin([np.nanmin(bmn[both]) if both.any() else np.nan, 0.5])
    for i, m in enumerate(mins):          # n per rung, so 2-subject points read differently
        if npr[i] > 0:
            ax[0].annotate(str(npr[i]), (m, ymin), fontsize=6, ha="center", color="gray")
    ax[0].set(ylabel="FC edge correlation (r)",
              title="Stable & individual: within- vs between-person, same ladder")
    ax[0].legend(fontsize=8, loc="lower right")

    # Bottom: margin only (accuracy is pinned at 1.0 with this cohort and says nothing) + control.
    ok = [r for r in table if "acc_fc" in r]
    if ok:
        mm = [r["min"] for r in ok]
        ax[1].plot(mm, [r["margin_fc"] for r in ok], "o-", label="FC (covariance)")
        ax[1].plot(mm, [r["margin_st"] for r in ok], "s--", label="structural (control)")
        ax[1].legend(fontsize=8, loc="upper left")
    ax[1].set(xlabel="minutes of rest (linear)", ylabel="SVM margin (true − runner-up)",
              title="Identifying: margin grows with data (accuracy pins at 1.0, dropped)")
    fig.tight_layout()
    config.FC_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = config.FC_RESULTS_DIR / "curves.png"
    fig.savefig(out, dpi=120); plt.close(fig)
    print(f"\nfigure -> {out}")


# ---------------------------------------------------------------- entry
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("stage", choices=["reduce", "inspect", "analyze"])
    ap.add_argument("--subjects", nargs="+", default=config.SUBJECTS)
    ap.add_argument("--cleanup", action="store_true",
                    help="reduce only: delete each run's BOLD after a passing reduction")
    args = ap.parse_args()
    if args.stage == "analyze":
        stage_analyze(args.subjects)
    else:
        if args.cleanup:
            print("cleanup: ON — BOLD deleted after each passing reduction\n")
        stage_reduce(args.subjects, args.cleanup, inspect_only=(args.stage == "inspect"))


if __name__ == "__main__":
    main()
