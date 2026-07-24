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
from panmvpa import plotstyle as ps  # noqa: E402

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
    ps.apply()
    t = np.arange(rec["parcels"].shape[0]) * TR / 60.0
    fig, ax = ps.plt.subplots(1, 3, figsize=(ps.FULL, 2.4))
    ax[0].plot(t, rec["gs"], lw=0.9, color=ps.ACCENT)
    ax[1].plot(t, rec["dvars"], lw=0.9, color=ps.ACCENT)
    p = rec["parcels"].T
    p = p - p.mean(axis=1, keepdims=True)   # demean each parcel so fluctuations show
    lim = float(np.percentile(np.abs(p), 98))
    ax[2].imshow(p, aspect="auto", cmap=ps.SUNSET_DIV, vmin=-lim, vmax=lim,
                 extent=[0, t[-1], 400, 0])
    for j, name in enumerate(("whole-brain mean", "DVARS", "400 parcels, demeaned")):
        ps.panel(ax[j], j, name)
    for a in ax[:2]:
        a.set_xlabel("minutes"); ps.style_ax(a)
    ax[2].tick_params(length=0)
    ps.titles(fig, f"{sid} ses-{ses} run-{run} — reduction sanity", top=0.80)
    config.FC_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = ps.save(fig, config.FC_RESULTS_DIR / f"inspect_{sid}_ses-{ses:02d}_run-{run:02d}")
    print(f"  figure -> {out}")
    return Path(out)


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


def _regress_out(x: np.ndarray, g: np.ndarray) -> np.ndarray:
    """Residual of x after removing its projection onto g (mean-centred): x_c − β·g_c. Unlike
    x − g, this discards the whole along-g direction, so a uniform rescaling of g leaves ~0 —
    global amplitude differences don't survive as individuality."""
    xc, gc = x - x.mean(), g - g.mean()
    d = float(gc @ gc)
    return xc - (float(xc @ gc) / d) * gc if d > 0 else xc


def _colmean(mat: np.ndarray) -> np.ndarray:
    """Column mean over the rows that are finite, NaN where a column is entirely NaN --
    without np.nanmean's empty-slice warning."""
    counts = np.sum(~np.isnan(mat), axis=0)
    sums = np.nansum(mat, axis=0)
    return np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)


def identity_curves(sess: dict[str, dict[int, list[dict]]], gsr: bool) -> dict:
    """Nearest-neighbour identification, grown on the full minute ladder (no held-out
    examples, so unlike the SVM it runs to 80 min alongside the convergence curve).

    For each subject A, grow the first half minute-by-minute and correlate its FC against
    every subject's reference (second) half:

        r_self(X)  = corr( FC(A first-half, first X min),  FC(A second-half, full) )
        r_other    = { corr(same grow, FC(B second-half, full)) : B != A }
        near(X)    = max r_other   -- the nearest impostor, the identification competitor
        floor(X)   = mean r_other  -- the group floor (this is the earlier between-person mean)
        signal(X)  = r_self - near -- the individual signal; the headline, reported directly
                     (not as headroom = signal/(1-near): that denominator moves, so headroom
                     can rise while signal falls -- misleading)
        hit(X)     = r_self > near -- A's own half is the top match => identified (will ceiling)

    Because r_self is dominated by shared "this is a human cortex" structure (two strangers
    already agree ~0.6), we also residualise both sides against the group before correlating --
    the reliability of the DEVIATION from the group, which is what precision fMRI cares about:

        g1 = leave-one-out mean of the OTHER subjects' FULL FIRST halves
        g2 = leave-one-out mean of the OTHER subjects' FULL SECOND halves   (independent of g1,
             so its estimation error is not shared across the two sides and cannot inflate r)
        r_resid_sub(X) = corr( A(X) - g1,             ref(s) - g2 )          # plain subtraction
        r_resid_reg(X) = corr( regress_out(A(X), g1), regress_out(ref(s), g2) )  # projection out

    Subtraction leaves global amplitude in (uniformly stronger connectivity reads as
    individuality); regression removes it. If the two differ, some "individuality" is scaling.
    """
    subs = sorted(sess)
    first_half, first_edges, ref_edges = {}, {}, {}
    for sid in subs:
        runs = [r for ses in sorted(sess[sid]) for r in sess[sid][ses]]
        X = np.concatenate([clean_run(r, gsr) for r in runs], axis=0)
        h = X.shape[0] // 2
        first_half[sid] = X[:h]
        first_edges[sid] = fc_edges(X[:h])          # full first half, for the group template g1
        ref_edges[sid] = fc_edges(X[h:])

    # Leave-one-out group templates: g1[s] excludes s, from FIRST halves; g2[s] from SECOND.
    n = len(subs)
    sum1 = np.sum([first_edges[s] for s in subs], axis=0) if n else None
    sum2 = np.sum([ref_edges[s] for s in subs], axis=0) if n else None
    g1 = {s: (sum1 - first_edges[s]) / (n - 1) for s in subs} if n >= 2 else {}
    g2 = {s: (sum2 - ref_edges[s]) / (n - 1) for s in subs} if n >= 2 else {}

    keys = ("r_self", "near", "floor", "signal", "hit", "r_resid_sub", "r_resid_reg")
    per = {k: {s: np.full(len(LADDER), np.nan) for s in subs} for k in keys}
    for sid in subs:
        a = first_half[sid]
        for i, m in enumerate(LADDER):
            npt = int(round(m * 60.0 / TR))
            if not (MIN_TP <= npt <= a.shape[0]):    # only points the subject truly has
                continue
            grow = fc_edges(a[:npt])
            rs = _corr(grow, ref_edges[sid])
            per["r_self"][sid][i] = rs
            cross = [_corr(grow, ref_edges[b]) for b in subs if b != sid]
            if cross:
                near = max(cross)
                per["near"][sid][i] = near
                per["floor"][sid][i] = float(np.mean(cross))
                per["signal"][sid][i] = rs - near
                per["hit"][sid][i] = float(rs > near)
            if sid in g1:
                per["r_resid_sub"][sid][i] = _corr(grow - g1[sid], ref_edges[sid] - g2[sid])
                per["r_resid_reg"][sid][i] = _corr(_regress_out(grow, g1[sid]),
                                                   _regress_out(ref_edges[sid], g2[sid]))

    stack = lambda k: np.vstack([per[k][s] for s in subs])
    out = {"minutes": LADDER, "subs": subs, "per": per,
           "first_half": first_half, "first_edges": first_edges, "ref_edges": ref_edges,
           "g1": g1, "g2": g2,                                  # reused by the network breakdown
           "n": np.sum(~np.isnan(stack("r_self")), axis=0)}
    for k in keys:
        out[k + "_mean"] = _colmean(stack(k))   # hit_mean == hit rate (mean of 0/1)
    return out


def _parcel_networks() -> tuple[np.ndarray, list[str]]:
    """(net_of_parcel, network_names): the Yeo-17 network index (0..16) of each Schaefer
    parcel column, from the atlas metadata we have been ignoring. Column j of the parcel
    matrix is label j+1 (labels are 1..400 contiguous), matching the atlas order file."""
    from panmvpa import parcellation
    order = list(parcellation.network_order())
    nets = np.full(config.N_PARCELS, -1, dtype=int)
    for pid, name in parcellation._parcel_names().items():
        net = parcellation._net_of(name)
        if net in order:
            nets[pid - 1] = order.index(net)
    return nets, order


def network_breakdown(cur: dict, minutes_list) -> dict:
    """Split the FC edge vector by Yeo-17 network pair and compute the subject-mean r_self /
    nearest / signal per block at every rung, plus the group-residual reliability per block
    (r_resid_sub / r_resid_reg) -- the same leave-one-out residualisation as the main curves,
    which unconfounds the per-network ranking from how much group structure each block carries.
    Reuses the cached first/second halves and group templates from identity_curves.
    """
    subs, first_half, first_edges, ref_edges = (cur["subs"], cur["first_half"],
                                                cur["first_edges"], cur["ref_edges"])
    g1, g2 = cur["g1"], cur["g2"]
    nets, names = _parcel_networks()
    K = len(names)
    iu = np.triu_indices(config.N_PARCELS, k=1)
    lo, hi = np.minimum(nets[iu[0]], nets[iu[1]]), np.maximum(nets[iu[0]], nets[iu[1]])
    block_idx = {(p, q): np.where((lo == p) & (hi == q))[0]
                 for p in range(K) for q in range(p, K)}
    block_idx = {k: v for k, v in block_idx.items() if v.size >= 3}

    rows = []
    for m in minutes_list:
        n = int(round(m * 60.0 / TR))
        grow = {s: fc_edges(first_half[s][:n]) for s in subs
                if MIN_TP <= n <= first_half[s].shape[0]}
        valid = list(grow)
        if len(valid) < 2:                       # need >=2 for a nearest impostor
            continue
        for (p, q), idx in block_idx.items():
            rs_l, nr_l, sg_l, sub_l, reg_l = [], [], [], [], []
            for s in valid:
                gb, sb = grow[s][idx], ref_edges[s][idx]
                rs = _corr(gb, sb)
                near = max(_corr(gb, ref_edges[b][idx]) for b in valid if b != s)
                rs_l.append(rs); nr_l.append(near); sg_l.append(rs - near)
                if s in g1:                       # block-restricted group residual, both variants
                    g1b, g2b = g1[s][idx], g2[s][idx]
                    sub_l.append(_corr(gb - g1b, sb - g2b))
                    reg_l.append(_corr(_regress_out(gb, g1b), _regress_out(sb, g2b)))
            mean = lambda v: float(np.mean(v)) if v else float("nan")
            rows.append({"a": names[p], "b": names[q], "minutes": float(m), "n": len(valid),
                         "r_self": mean(rs_l), "nearest": mean(nr_l), "signal": mean(sg_l),
                         "r_resid_sub": mean(sub_l), "r_resid_reg": mean(reg_l)})
    return {"rows": rows, "names": names, "nets": nets}


def _write_curves_csv(cur, n_sub, path) -> None:
    """Long format, one row per (subject, minutes): the per-subject curves behind the means."""
    import csv
    mins, npr = cur["minutes"], cur["n"]
    cell = lambda x: f"{x:.6f}" if np.isfinite(x) else ""
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["subject", "minutes", "n", "r_self", "nearest", "floor", "signal",
                    "r_resid_sub", "r_resid_reg"])
        for s in cur["subs"]:
            for i, m in enumerate(mins):
                rself = cur["per"]["r_self"][s][i]
                if not np.isfinite(rself):
                    continue
                p = cur["per"]
                w.writerow([s, m, int(npr[i]), cell(rself), cell(p["near"][s][i]),
                            cell(p["floor"][s][i]), cell(p["signal"][s][i]),
                            cell(p["r_resid_sub"][s][i]), cell(p["r_resid_reg"][s][i])])


def _write_network_csv(rows, path) -> None:
    import csv
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["network_a", "network_b", "minutes", "r_self", "nearest", "signal",
                    "r_resid_sub", "r_resid_reg"])
        cell = lambda x: f"{x:.6f}" if np.isfinite(x) else ""
        for r in rows:
            w.writerow([r["a"], r["b"], r["minutes"], cell(r["r_self"]), cell(r["nearest"]),
                        cell(r["signal"]), cell(r["r_resid_sub"]), cell(r["r_resid_reg"])])


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


def stage_analyze(subjects, run_svm: bool = False) -> None:
    sess = _sessions()
    if not sess:
        raise SystemExit(f"no reduced files in {config.REDUCED_DIR} — run `reduce` first.")
    n_sub = len(sess)
    print(f"analyze: {n_sub} subjects, "
          f"{sum(len(v) for v in sess.values())} sessions reduced.")

    cur = identity_curves(sess, gsr=True)
    mins, npr = cur["minutes"], cur["n"]
    rs, near, floor = cur["r_self_mean"], cur["near_mean"], cur["floor_mean"]
    sig, hit = cur["signal_mean"], cur["hit_mean"]

    # The full-cohort range is the rungs every subject reaches; past it the mean is carried by
    # a shrinking, self-selected few (the 80-min point can be one person), so the figures cap
    # the x-axis there.
    full = npr == n_sub
    fi = np.where(full)[0]
    hi_full = float(mins[fi[-1]]) if len(fi) else float("nan")

    print("\nr_self / nearest / floor / signal, mean over subjects:")
    print("  min   n   r_self  nearest   floor   signal   hit%")
    for i, m in enumerate(mins):
        if npr[i] == 0:
            continue
        tail = "" if full[i] else "  n<%d" % n_sub
        if np.isfinite(near[i]):
            print(f"  {m:4.1f}  {npr[i]:2d}   {rs[i]:.3f}   {near[i]:.3f}   {floor[i]:.3f}   "
                  f"{sig[i]:+.3f}   {hit[i]*100:3.0f}%{tail}")
        else:
            print(f"  {m:4.1f}  {npr[i]:2d}   {rs[i]:.3f}     --       --       --      --{tail}")

    # The SVM is the only slow part (20+ min); off by default, run only with --svm.
    if run_svm:
        table = identify_table(sess, gsr=True)
        print(f"\nSVM leave-one-session-out (chance={1.0/n_sub:.2f}):")
        print("  min  examples  used   FC acc / margin    structural acc / margin")
        for r in table:
            used = f"{r['used_frac']:.0%}"
            if "acc_fc" in r:
                print(f"  {r['min']:4.1f}  {r['n']:3d} (min {r['per_sub']}/subj)  {used:>4}  "
                      f"{r['acc_fc']:.2f} / {r['margin_fc']:+.2f}     "
                      f"{r['acc_st']:.2f} / {r['margin_st']:+.2f}")
            else:
                print(f"  {r['min']:4.1f}  {r['n']:3d}          {used:>4}  ({r['per_sub']} example/subj)")

    # Write data, not conclusions: long-format CSVs + figures to interpret in a notebook.
    csv_dir = config.FC_RESULTS_DIR / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    curves_csv = csv_dir / "curves.csv"
    _write_curves_csv(cur, n_sub, curves_csv)
    print(f"\ncsv -> {curves_csv}")
    _analysis_figure(cur, n_sub)
    _sampling_figure(cur, n_sub)          # x-axis check: first vs random X min
    if n_sub >= 2:
        _residual_figure(cur, n_sub)      # the main analysis: reliability of the group residual

    if n_sub >= 2:
        nb = network_breakdown(cur, mins)
        net_csv = csv_dir / "networks.csv"
        _write_network_csv(nb["rows"], net_csv)
        print(f"csv -> {net_csv}")
        if len(fi):
            _network_figure(cur, nb, hi_full)
            _nettraj_figure(cur, nb, n_sub)


def _fig_range(cur, n_sub):
    """(mins, full-mask, hi, seg) -- the full-cohort range and a segment helper the figures share."""
    mins, npr = cur["minutes"], cur["n"]
    full = npr == n_sub                        # rungs every subject reaches (contiguous from 1)
    fi = np.where(full)[0]
    hi = float(mins[fi[-1]]) if len(fi) else float(mins[-1])
    seg = lambda y: (np.where(full, mins, np.nan), np.where(full, y, np.nan))
    return mins, full, hi, seg


def _analysis_figure(cur, n_sub) -> None:
    """One panel: r_self and the nearest impostor, the individual signal shaded between them."""
    ps.apply()
    mins, full, hi, seg = _fig_range(cur, n_sub)
    rs, near = cur["r_self_mean"], cur["near_mean"]
    c_self, c_near = ps.SUNSET(0.30), ps.ACCENT   # darkest on the top (r_self) line

    fig, ax = ps.plt.subplots(figsize=(ps.HALF, 2.9))
    for sid in cur["subs"]:
        ax.plot(mins, cur["per"]["r_self"][sid], color=c_self, lw=0.4, alpha=0.15)
        ax.plot(mins, cur["per"]["near"][sid], color=c_near, lw=0.4, alpha=0.15)
    ax.fill_between(mins, near, rs, where=full & np.isfinite(rs) & np.isfinite(near),
                    color=ps.ACCENT, alpha=0.12, lw=0, label="individual signal (own − nearest)")
    ax.plot(*seg(rs), "o-", color=c_self, lw=1.4, ms=3, label="vs own other half")
    ax.plot(*seg(near), "s-", color=c_near, lw=1.4, ms=3, label="nearest stranger")
    ax.set(xlabel="minutes of rest", ylabel="correlation between maps", xlim=(0, hi))
    ps.style_ax(ax)
    ps.legend(ax, loc="lower right")
    ps.titles(fig, "Own half vs the nearest stranger",
              f"$N$ = {n_sub} people  |  1–{hi:.0f} min  |  global signal removed, 0.008–0.08 Hz")
    config.FC_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"figure -> {ps.save(fig, config.FC_RESULTS_DIR / 'curves')}")


def _residual_figure(cur, n_sub) -> None:
    """The main analysis: r_self (dominated by shared 'human cortex' structure) against r_resid,
    the reliability of the deviation from the group (regression variant; subtraction is
    numerically identical here). Thin line per subject behind each; x capped at n=n_sub."""
    ps.apply()
    mins, full, hi, seg = _fig_range(cur, n_sub)
    cmean = lambda k: _colmean(np.vstack([cur["per"][k][s] for s in cur["subs"]]))
    c_self, c_res = ps.SUNSET(0.35), ps.ACCENT

    fig, ax = ps.plt.subplots(figsize=(ps.HALF, 2.9))
    for s in cur["subs"]:
        ax.plot(mins, cur["per"]["r_self"][s], color=c_self, lw=0.4, alpha=0.15)
        ax.plot(mins, cur["per"]["r_resid_reg"][s], color=c_res, lw=0.4, alpha=0.15)
    ax.plot(*seg(cmean("r_self")), "o-", color=c_self, lw=1.4, ms=3, label="vs own other half")
    ax.plot(*seg(cmean("r_resid_reg")), "o-", color=c_res, lw=1.4, ms=3,
            label="vs own other half (group pattern removed)")
    ax.set(xlabel="minutes of rest", ylabel="correlation between maps", xlim=(0, hi))
    ps.style_ax(ax)
    ps.legend(ax, loc="lower right")
    ps.titles(fig, "Matching your own map, before and after removing the group",
              f"$N$ = {n_sub} people  |  1–{hi:.0f} min  |  group = average of the other {n_sub - 1}")
    print(f"figure -> {ps.save(fig, config.FC_RESULTS_DIR / 'residual')}")


def _draw_scatter(T: int, n: int, rng) -> np.ndarray:
    """n individual timepoints scattered across the half. Breaks temporal autocorrelation, so
    with the 0.08 Hz low-pass each is ~independent -- inflates effective DOF."""
    return rng.choice(T, n, replace=False)


def _draw_block(T: int, n: int, rng, block: int = 0) -> np.ndarray:
    """n timepoints as contiguous ~1-min chunks from random (non-overlapping) positions across
    the half. Same session spread as scatter, but autocorrelation is preserved WITHIN each chunk
    -- so effective DOF matches the first-X-min curve. If the gap survives this, it's sessions."""
    block = block or max(1, int(round(60.0 / TR)))
    n_tiles = T // block
    if n_tiles < 1:
        return _draw_scatter(T, n, rng)
    starts = np.arange(n_tiles) * block
    rng.shuffle(starts)
    idx: list[int] = []
    for s in starts:
        idx.extend(range(int(s), int(s) + block))
        if len(idx) >= n:
            break
    return np.array(idx[:n], dtype=int)


def _resampled_curves(cur, draw, seed: int = 0) -> dict:
    """Per-subject r_self / nearest / signal when the growing X minutes are drawn by `draw`
    instead of taken from the front. Same reference. Seeded, so the output is reproducible."""
    subs, first_half, ref_edges = cur["subs"], cur["first_half"], cur["ref_edges"]
    rng = np.random.default_rng(seed)
    per = {k: {s: np.full(len(LADDER), np.nan) for s in subs} for k in ("r_self", "near", "signal")}
    for sid in subs:
        a = first_half[sid]
        for i, m in enumerate(LADDER):
            n = int(round(m * 60.0 / TR))
            if not (MIN_TP <= n <= a.shape[0]):
                continue
            grow = fc_edges(a[draw(a.shape[0], n, rng)])
            rs = _corr(grow, ref_edges[sid])
            per["r_self"][sid][i] = rs
            cross = [_corr(grow, ref_edges[b]) for b in subs if b != sid]
            if cross:
                near = max(cross)
                per["near"][sid][i] = near
                per["signal"][sid][i] = rs - near
    return per


def _sampling_figure(cur, n_sub) -> None:
    """x-axis check, three ways of drawing X minutes from the first half:
        first   -- the opening X min (one session early, many late) -- what you can collect
        scatter -- X individual timepoints across the half (breaks autocorrelation -> high DOF)
        block   -- X min as ~1-min contiguous chunks from random positions (session spread with
                   autocorrelation preserved)
    scatter vs block separates effective-DOF from session diversity: if the scatter gap collapses
    onto block it was DOF; if block still sits well above first, it's sessions."""
    ps.apply()
    # Legend order (top -> bottom): first minutes, 1-min chunks, scattered timepoints.
    modes = {"first": cur["per"],
             "block": _resampled_curves(cur, _draw_block),
             "scatter": _resampled_curves(cur, _draw_scatter)}
    mins, full, hi, seg = _fig_range(cur, n_sub)
    cmean = lambda per, k: _colmean(np.vstack([per[k][s] for s in cur["subs"]]))
    style = {"first": (ps.SUNSET(0.30), "o-"), "scatter": (ps.SUNSET(0.55), "^--"),
             "block": (ps.ACCENT, "s-.")}
    label = {"first": "first minutes", "scatter": "scattered timepoints",
             "block": "1-min chunks from across all sessions"}

    fig, axes = ps.plt.subplots(1, 2, figsize=(ps.FULL, 3.2), sharex=True)
    for j, metric in enumerate(("r_self", "signal")):
        ax = axes[j]
        for name, per in modes.items():
            c, ls = style[name]
            ax.plot(*seg(cmean(per, metric)), ls, color=c, lw=1.4, ms=3, label=label[name])
        ax.set(xlabel="minutes of rest", xlim=(0, hi))
        ps.style_ax(ax)
        ps.panel(ax, j, "vs own other half" if metric == "r_self" else "individual signal")
    axes[0].set_ylabel("correlation between maps")
    ps.legend(axes[0], loc="lower right")

    # Factual block result: chunks share scattered's session spread and first's continuity, so
    # where they land relative to the two says which factor is acting. They coincide with first
    # at 1 min by construction, so only rungs >=3 min are informative.
    mid = full & (np.array(mins) >= 3) & (np.array(mins) <= hi)
    b, s, f = (cmean(modes["block"], "r_self"), cmean(modes["scatter"], "r_self"),
               cmean(modes["first"], "r_self"))
    lo, up = np.minimum(f, s), np.maximum(f, s)
    frac_between = float(np.nanmean(((b >= lo - 1e-9) & (b <= up + 1e-9))[mid])) if mid.any() else 0.0
    if frac_between >= 0.6:
        l1 = "1-min chunks fall between first minutes and scattered at every rung"
        l2 = "so both session variety and number of independent samples contribute"
    elif np.nanmean(np.abs(b - s)[mid]) < np.nanmean(np.abs(b - f)[mid]):
        l1 = "1-min chunks track scattered timepoints, not first minutes"
        l2 = "session variety dominates over number of independent samples"
    else:
        l1 = "1-min chunks track first minutes, not scattered"
        l2 = "number of independent samples dominates over session variety"
    subtitle = f"{l1}\n{l2}  (chunks meet first minutes at 1 min; compare from 3 min)  |  $N$ = {n_sub}"
    ps.titles(fig, "Which minutes you use, and how they are spread across sessions",
              subtitle, top=0.78)
    print(f"figure -> {ps.save(fig, config.FC_RESULTS_DIR / 'sampling')}")

    import csv
    npr = cur["n"]
    cell = lambda x: f"{x:.6f}" if np.isfinite(x) else ""
    path = config.FC_RESULTS_DIR / "csv" / "sampling.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["subject", "minutes", "sample", "n", "r_self", "nearest", "signal"])
        for name, per in modes.items():
            for s in cur["subs"]:
                for i, m in enumerate(mins):
                    if not np.isfinite(per["r_self"][s][i]):
                        continue
                    w.writerow([s, m, name, int(npr[i]), cell(per["r_self"][s][i]),
                                cell(per["near"][s][i]), cell(per["signal"][s][i])])
    print(f"csv -> {path}")


def _network_figure(cur, nb, target_min) -> None:
    """Two panels: the mean reference FC with the 400 parcels sorted by Yeo-17 network, and the
    17x17 group-residual-per-network-pair matrix at `target_min`. They are different quantities,
    so different colormaps: (a) is signed (anticorrelation is real structure -- e.g. default vs
    dorsal-attention -- and must not merge with zero) so it is diverging with white at 0; (b) is
    one-signed so it is sequential, dark = more. The figure is sized large so the 17 network
    labels fit on the 400-parcel panel without colliding."""
    ps.apply()
    nets, names, K = nb["nets"], nb["names"], len(nb["names"])
    full_names = [config.network_label(nm) for nm in names]
    order = np.argsort(nets, kind="stable")           # parcels grouped by network
    sizes = [int(np.sum(nets[order] == k)) for k in range(K)]
    bounds = np.cumsum(sizes)
    centers = bounds - np.array(sizes) / 2.0

    mean_edges = np.mean(np.vstack([cur["ref_edges"][s] for s in cur["subs"]]), axis=0)
    M = np.zeros((config.N_PARCELS, config.N_PARCELS))
    iu = np.triu_indices(config.N_PARCELS, k=1)
    M[iu] = mean_edges
    M = M + M.T
    np.fill_diagonal(M, 1.0)
    Ms = M[np.ix_(order, order)]

    idx = {name: k for k, name in enumerate(names)}
    S = np.full((K, K), np.nan)
    for r in nb["rows"]:
        if r["minutes"] == target_min:
            p, q = idx[r["a"]], idx[r["b"]]
            S[p, q] = S[q, p] = r["r_resid_reg"]

    fig, ax = ps.plt.subplots(1, 2, figsize=(11.5, 6.2))
    vmax = float(np.nanmax(np.abs(Ms)))
    im0 = ax[0].imshow(Ms, cmap=ps.SUNSET_DIV, vmin=-vmax, vmax=vmax)   # signed -> diverging, white at 0
    for b in bounds[:-1]:
        ax[0].axhline(b - 0.5, color="#888888", lw=0.3); ax[0].axvline(b - 0.5, color="#888888", lw=0.3)
    ax[0].set_xticks(centers); ax[0].set_xticklabels(full_names, rotation=90, fontsize=ps.FS["tick"])
    ax[0].set_yticks(centers); ax[0].set_yticklabels(full_names, fontsize=ps.FS["tick"])
    ps.panel(ax[0], 0, "average connectivity, parcels sorted by network")
    ps.colorbar(fig, im0, ax[0])

    im1 = ax[1].imshow(S, cmap=ps.SUNSET_HI)   # dark = more
    ax[1].set_xticks(range(K)); ax[1].set_xticklabels(full_names, rotation=90, fontsize=ps.FS["tick"])
    ax[1].set_yticks(range(K)); ax[1].set_yticklabels(full_names, fontsize=ps.FS["tick"])
    ps.panel(ax[1], 1, f"match to own map, group removed ({target_min:.0f} min)")
    ps.colorbar(fig, im1, ax[1])
    for a in ax:
        a.tick_params(length=0)
    ps.titles(fig, "Connectivity by network, and where individuality survives",
              f"$N$ = {len(cur['subs'])} people  |  Yeo-17 networks  |  at {target_min:.0f} min",
              top=0.82)
    print(f"figure -> {ps.save(fig, config.FC_RESULTS_DIR / 'networks')}")


def _nettraj_figure(cur, nb, n_sub) -> None:
    """Per-network group-residual reliability vs data. The data set the order (highest final
    value first); the legend lists the networks in exactly that order and the colour runs along
    it top-to-bottom, so the legend is a direct key to the stacked lines."""
    ps.apply()
    _, _, hi, _ = _fig_range(cur, n_sub)
    names = nb["names"]
    ms = sorted({r["minutes"] for r in nb["rows"] if r["minutes"] <= hi})
    acc = {nm: {m: [] for m in ms} for nm in names}
    for r in nb["rows"]:
        if r["minutes"] > hi or not np.isfinite(r["r_resid_reg"]):
            continue
        for nm in ({r["a"], r["b"]}):          # every block counts toward both its networks
            acc[nm][r["minutes"]].append(r["r_resid_reg"])
    traj = {nm: np.array([np.mean(acc[nm][m]) if acc[nm][m] else np.nan for m in ms]) for nm in names}
    final = {nm: (traj[nm][-1] if len(ms) else np.nan) for nm in names}
    order = sorted(names, key=lambda nm: (-final[nm] if np.isfinite(final[nm]) else np.inf))
    col = dict(zip(order, list(reversed(ps.sunset_colors(len(order))))))   # dark = high, light = low

    from matplotlib.lines import Line2D
    fig, ax = ps.plt.subplots(figsize=(ps.FULL * 0.66, 3.4))
    for nm in order:                            # every line coloured by its rank
        ax.plot(ms, traj[nm], color=col[nm], lw=1.4)
    handles = [Line2D([], [], color=col[nm], lw=1.8, label=config.network_label(nm)) for nm in order]
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.02, 0.5),
              frameon=False, fontsize=ps.FS["legend"], labelcolor=ps.TICKINK,
              title="networks, high → low", title_fontsize=ps.FS["legend"])
    ax.set(xlabel="minutes of rest", ylabel="match to own map (group removed)", xlim=(0, hi))
    ps.style_ax(ax)
    ps.titles(fig, "Individuality by network",
              f"$N$ = {n_sub} people  |  coloured by rank, high → low  |  to {hi:.0f} min")
    print(f"figure -> {ps.save(fig, config.FC_RESULTS_DIR / 'nettraj')}")


# ---------------------------------------------------------------- entry
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("stage", choices=["reduce", "inspect", "analyze"])
    ap.add_argument("--subjects", nargs="+", default=config.SUBJECTS)
    ap.add_argument("--cleanup", action="store_true",
                    help="reduce only: delete each run's BOLD after a passing reduction")
    ap.add_argument("--svm", action="store_true",
                    help="analyze only: also run the leave-one-session-out SVM (20+ min; off by default)")
    args = ap.parse_args()
    if args.stage == "analyze":
        stage_analyze(args.subjects, run_svm=args.svm)
    else:
        if args.cleanup:
            print("cleanup: ON — BOLD deleted after each passing reduction\n")
        stage_reduce(args.subjects, args.cleanup, inspect_only=(args.stage == "inspect"))


if __name__ == "__main__":
    main()
