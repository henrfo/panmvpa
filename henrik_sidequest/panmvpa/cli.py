"""End-to-end driver: map stability + subject identification -> group figure + CSV.

  Plot 1  DN-A split-half parcellation reliability (Dice)
  Plot 2  subject identification accuracy from individualised maps (chance = 1/n)

Runs in stages so raw scans never all have to be on disk at once:

  maps      per subject: Dice curve, then build+save every (level, seed) map.
            Maps are ~260 KB each and persist, so raw rest can be deleted after.
  identify  per subject: score each held-out task scan against ALL subjects' stored
            maps. Needs pass 1 finished for every subject -- that is why this cannot be
            folded into a single pass with cleanup.
  figure    aggregate the stored results and plot.

    panmvpa-run --stage maps --cleanup
    panmvpa-run --stage identify --cleanup
    panmvpa-run --stage figure
    panmvpa-run                         # all three in order

``--cleanup`` deletes a subject's raw BOLD once their results are computed, so disk holds
one subject at a time (~60 GB) rather than the 0.64 TB cohort.
"""
from __future__ import annotations

import argparse
import json
import resource
import traceback
from pathlib import Path

import numpy as np
import nibabel as nib

from . import (
    bold,
    clear_caches,
    config,
    figure,
    identify,
    mapstore,
    parcellation,
    reliability,
)


def _peak_gb() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 1e9 if peak > 1e7 else peak / 1e6


def _write(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))


def _read(path: Path, default):
    return json.loads(path.read_text()) if path.exists() else default


# --------------------------------------------------------------------------- pass 1
def stage_maps(subjects, minutes, n_seeds, outdir: Path, cleanup: bool) -> None:
    store = _read(outdir / "reliability.json", {})
    for subject in subjects:
        sid = config.sub_id(subject)
        print(f"\n=== [maps] {sid} ===", flush=True)
        try:
            rel = reliability.reliability_curve(sid, minutes, n_seeds=n_seeds)
            for r in rel:
                spare = r.get("spare_runs")
                warn = (f"  <- only {spare} spare run(s), seeds near-identical"
                        if spare is not None and spare <= 1 and n_seeds > 1 else "")
                print(f"    {r['minutes']:>4.0f} min | Dice {r['dice']:.3f}"
                      f"±{r['dice_std']:.3f} | {r['n_seeds']} seeds{warn}", flush=True)
            for m in minutes:
                for seed in range(n_seeds):
                    if mapstore.has_map(sid, m, seed):
                        continue
                    labels = parcellation.build_parcellation(sid, minutes=m, seed=seed)
                    mapstore.save_map(labels, sid, m, seed)
            store[sid] = rel
            _write(outdir / "reliability.json", store)
            print(f"    saved {len(minutes) * n_seeds} maps", flush=True)
            if cleanup:
                freed = cleanup_subject(sid, kind="rest")
                print(f"    cleaned {freed:.1f} GB of rest", flush=True)
        except Exception as exc:
            print(f"    SKIPPED -- {type(exc).__name__}: {exc}", flush=True)
            traceback.print_exc(limit=1)
        finally:
            clear_caches()
            print(f"    [peak RSS {_peak_gb():.1f} GB]", flush=True)


# --------------------------------------------------------------------------- pass 2
def stage_identify(subjects, minutes, n_seeds, outdir: Path, cleanup: bool,
                   size_matched: bool) -> None:
    # Preload every stored map once: ~260 KB each, so the whole cohort is a few hundred MB
    # and each held-out scan is read from disk exactly once, not once per level.
    cohorts: dict[tuple[float, int], dict[str, np.ndarray]] = {}
    for m in minutes:
        for seed in range(n_seeds):
            c = mapstore.load_cohort(m, seed)
            if len(c) >= 2:
                cohorts[(m, seed)] = c
    if not cohorts:
        raise SystemExit("No stored maps found -- run `--stage maps` first.")
    n_sub = max(len(c) for c in cohorts.values())
    print(f"loaded {len(cohorts)} cohort maps sets, up to {n_sub} subjects each")

    records = _read(outdir / "identification.json", {})
    idx = parcellation.analysis_domain()

    for subject in subjects:
        sid = config.sub_id(subject)
        scans = bold.task_scans(sid)
        print(f"\n=== [identify] {sid} — {len(scans)} held-out task scans ===", flush=True)
        if not scans:
            print("    none present, skipping", flush=True)
            continue
        try:
            for path in scans:
                arr = np.asarray(nib.load(str(path)).dataobj, dtype=np.float32)
                ts = arr[idx[0], idx[1], idx[2], :]
                del arr
                z = identify._standardize_rows(ts)
                del ts
                for (m, seed), maps in cohorts.items():
                    res = identify.identify_scan(z, maps, sid,
                                                 size_matched=False, seed=seed)
                    key = f"{m}|{seed}"
                    records.setdefault(key, []).append({
                        "true": res["true"], "predicted": res["predicted"],
                        "correct": res["correct"], "margin": res["margin"],
                        "scan": path.name, "task": bold.scan_task_name(path),
                    })
                    if size_matched:
                        rm = identify.identify_scan(z, maps, sid,
                                                    size_matched=True, seed=seed)
                        records.setdefault(key + "|sm", []).append({
                            "true": rm["true"], "predicted": rm["predicted"],
                            "correct": rm["correct"], "margin": rm["margin"],
                            "scan": path.name, "task": bold.scan_task_name(path),
                        })
                del z
            _write(outdir / "identification.json", records)
            hit = np.mean([r["correct"] for k, v in records.items()
                           if not k.endswith("|sm") for r in v if r["true"] == sid])
            print(f"    recall so far for {sid}: {hit:.2f}", flush=True)
            if cleanup:
                freed = cleanup_subject(sid, kind="task")
                print(f"    cleaned {freed:.1f} GB of task scans", flush=True)
        except Exception as exc:
            print(f"    SKIPPED -- {type(exc).__name__}: {exc}", flush=True)
            traceback.print_exc(limit=1)
        finally:
            clear_caches()
            print(f"    [peak RSS {_peak_gb():.1f} GB]", flush=True)


def cleanup_subject(subject: str, kind: str) -> float:
    """Delete a subject's raw BOLD after their results are computed. Returns GB freed."""
    sid = config.sub_id(subject)
    pattern = f"sub-{sid}/ses-*/func/*_desc-preproc_bold.nii.gz"
    freed = 0
    for p in config.DATA_ROOT.glob(pattern):
        is_rest = "_task-rest_" in p.name
        if (kind == "rest" and not is_rest) or (kind == "task" and is_rest):
            continue
        try:
            target = p.resolve(strict=True)
            size = target.stat().st_size
            target.unlink()          # annex object
            p.unlink(missing_ok=True)  # and the symlink/file itself
            freed += size
        except (FileNotFoundError, OSError):
            continue
    return freed / 1e9


# --------------------------------------------------------------------------- figure
def stage_figure(minutes, n_seeds, outdir: Path) -> None:
    rel = _read(outdir / "reliability.json", {})
    ident = _read(outdir / "identification.json", {})
    if not rel:
        raise SystemExit("No reliability results -- run `--stage maps` first.")

    results = {"minutes": minutes, "n_seeds": n_seeds,
               "subjects": {s: {"reliability": v} for s, v in rel.items()}}

    ident_curve = []
    for m in minutes:
        accs, sms = [], []
        for seed in range(n_seeds):
            recs = ident.get(f"{m}|{seed}")
            if recs:
                accs.append(identify.accuracy(recs))
            sm = ident.get(f"{m}|{seed}|sm")
            if sm:
                sms.append(identify.accuracy(sm))
        n_scans = len(ident.get(f"{m}|0", []))
        ident_curve.append({
            "minutes": m,
            "accuracy": float(np.mean(accs)) if accs else float("nan"),
            "accuracy_std": float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0,
            "accuracy_size_matched": float(np.mean(sms)) if sms else float("nan"),
            "n_scans": n_scans,
            "n_seeds": len(accs),
        })
    results["identification"] = ident_curve
    n_sub = len({r["true"] for v in ident.values() for r in v}) if ident else len(rel)
    results["chance"] = 1.0 / max(n_sub, 1)

    figure.save_group_csv(results, outdir / "group_curves.csv")
    figure.plot_group(results, outdir / "group_figure.png")
    _write(outdir / "group_results.json", results)
    print(f"\nfigure -> {outdir / 'group_figure.png'}")
    print(f"csv    -> {outdir / 'group_curves.csv'}")
    for row in ident_curve:
        print(f"  {row['minutes']:>4.0f} min | identification "
              f"{row['accuracy']:.3f}±{row['accuracy_std']:.3f} "
              f"(size-matched {row['accuracy_size_matched']:.3f}) "
              f"| {row['n_scans']} scans, chance {results['chance']:.2f}")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--subjects", nargs="+", default=config.SUBJECTS)
    ap.add_argument("--minutes", type=float, nargs="+", default=config.MINUTE_LEVELS)
    ap.add_argument("--outdir", default=str(config.RESULTS_DIR))
    ap.add_argument("--n-seeds", type=int, default=10,
                    help="random run-subsets per subject per level (default 10)")
    ap.add_argument("--stage", choices=["maps", "identify", "figure", "all"],
                    default="all")
    ap.add_argument("--cleanup", action="store_true",
                    help="delete a subject's raw BOLD once their results are computed")
    ap.add_argument("--no-size-matched", action="store_true",
                    help="skip the size-matched identification control (halves pass-2 cost)")
    return ap


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    minutes = list(args.minutes)
    subjects = [config.sub_id(s) for s in args.subjects]

    print(f"data:    {config.DATA_ROOT}")
    print(f"results: {outdir}")
    print(f"levels:  {minutes} min | seeds: {args.n_seeds} | stage: {args.stage}")
    if args.cleanup:
        print("cleanup: ON -- raw BOLD is deleted after each subject")

    if args.stage in ("maps", "all"):
        stage_maps(subjects, minutes, args.n_seeds, outdir, args.cleanup)
    if args.stage in ("identify", "all"):
        stage_identify(subjects, minutes, args.n_seeds, outdir, args.cleanup,
                       size_matched=not args.no_size_matched)
    if args.stage in ("figure", "all"):
        stage_figure(minutes, args.n_seeds, outdir)


if __name__ == "__main__":
    main()
