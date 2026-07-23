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
from pathlib import Path

import numpy as np

# nilearn emits these two on every masker transform -- hundreds of lines per run that bury
# the numbers. Silence exactly these, by message, not the whole warning system: the raw
# (standardize=False) parcels are deliberate, and the atlas is knowingly resampled to the
# BOLD grid (a no-op here since they share it).
warnings.filterwarnings("ignore", category=FutureWarning,
                        message=r"boolean values for 'standardize'.*")
warnings.filterwarnings("ignore", category=UserWarning,
                        message=r"Resampling labels at transform time.*")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from panmvpa import config, rest  # noqa: E402

TR = config.TR
BANDPASS = dict(low_pass=0.08, high_pass=0.009, t_r=TR)


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
    parcels = labels.transform(str(bold_path)).astype(np.float32)       # (T, 400)
    b = brain.transform(str(bold_path)).astype(np.float32)              # (T, n_brain)
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


def overlap_curve(sess: dict[str, dict[int, list[dict]]], gsr: bool) -> np.ndarray:
    """Per subject: split all rest in half by time, grow the first half minute-by-minute,
    correlate its FC edges against the independent second half's full FC. Mean over subjects.
    """
    curves = []
    for sid, by_ses in sess.items():
        runs = [r for ses in sorted(by_ses) for r in by_ses[ses]]
        X = np.concatenate([clean_run(r, gsr) for r in runs], axis=0)
        half = X.shape[0] // 2
        ref = fc_edges(X[half:])
        a = X[:half]
        # NaN (not the plateau value) once the ask exceeds the half's real duration --
        # a[:n] with n past the end silently returns the whole half, which would fake a
        # flat tail. Each subject only contributes points it genuinely has the data for.
        row = []
        for m in LADDER:
            n = int(round(m * 60.0 / TR))
            row.append(np.corrcoef(fc_edges(a[:n]), ref)[0, 1]
                       if MIN_TP <= n <= a.shape[0] else np.nan)
        curves.append(row)
    # Average each minute over the subjects that reached it. A column no subject reached
    # (a rung past everyone's half) averages to NaN without np.nanmean's empty-slice warning.
    arr = np.array(curves)
    counts = np.sum(~np.isnan(arr), axis=0)
    sums = np.nansum(arr, axis=0)
    return np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)


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

    ov = overlap_curve(sess, gsr=True)
    print("\noverlap r vs minutes (split-half FC convergence, no training):")
    for m, r in zip(LADDER, ov):
        print(f"  {m:4.1f} min  r={r:.3f}" if r == r else f"  {m:4.1f} min  (n/a)")

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

    _analysis_figure(ov, table, n_sub)


def _analysis_figure(ov, table, n_sub) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ok = [r for r in table if "acc_fc" in r]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].plot(LADDER, ov, "o-")
    ax[0].set(xlabel="minutes of rest", ylabel="FC overlap r",
              title="Stable: covariance converges", ylim=(0, 1))
    if ok:
        mm = [r["min"] for r in ok]
        ax[1].plot(mm, [r["acc_fc"] for r in ok], "o-", label="FC (covariance)")
        ax[1].plot(mm, [r["acc_st"] for r in ok], "s--", label="structural (control)")
        ax[1].axhline(1.0 / n_sub, ls=":", c="k", label="chance")
        ax[2].plot(mm, [r["margin_fc"] for r in ok], "o-", label="FC")
        ax[2].plot(mm, [r["margin_st"] for r in ok], "s--", label="structural")
    ax[1].set(xlabel="minutes of rest", ylabel="accuracy", ylim=(0, 1.02),
              title="Identifying: accuracy (pins at 1.0 early)")
    ax[2].set(xlabel="minutes of rest", ylabel="margin (true − runner-up)",
              title="Identifying: margin (still rising)")
    if ok:  # only once the SVM panels have lines (skipped when <2 subjects are reduced)
        ax[1].legend(fontsize=8); ax[2].legend(fontsize=8)
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
