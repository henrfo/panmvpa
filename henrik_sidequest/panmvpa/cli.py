"""Run the pipeline in stages so raw BOLD never all has to be on disk at once.

  maps      per subject: build the 8 quarter-combination maps, save them (~260 KB each),
            and measure stability (pairwise agreement between equal-sized maps).
  identify  per subject: score every held-out task scan against ALL subjects' cumulative
            maps. Needs `maps` finished for every subject first -- that is why this cannot
            be folded into one pass with cleanup.
  figure    read the saved results and plot.

    panmvpa-run --stage maps --cleanup
    panmvpa-run --stage identify --cleanup
    panmvpa-run --stage figure

``--cleanup`` deletes a subject's raw BOLD once their numbers are computed, so disk holds
one subject at a time. Everything deleted is re-downloadable from OpenNeuro.
"""
from __future__ import annotations

import argparse
import json
import resource
import shutil
import subprocess
import traceback
from itertools import combinations
from pathlib import Path

import numpy as np

from . import config, figure, identify, parcellation, rest


def _peak_gb() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 1e9 if peak > 1e7 else peak / 1e6


def _write(path: Path, obj) -> None:
    def clean(o):
        if isinstance(o, float) and (o != o or o in (float("inf"), float("-inf"))):
            return None
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, list):
            return [clean(v) for v in o]
        return o

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(obj), indent=2))


def _read(path: Path, default):
    return json.loads(path.read_text()) if path.exists() else default


# ------------------------------------------------------------------ pass 1: maps
def stage_maps(subjects, outdir: Path, cleanup: bool) -> None:
    store = _read(outdir / "stability.json", {})
    for subject in subjects:
        sid = config.sub_id(subject)
        print(f"\n=== [maps] {sid} ===", flush=True)
        try:
            built: dict[tuple[int, ...], np.ndarray] = {}
            for quarters in config.MAP_KEYS:
                if parcellation.has_map(sid, quarters):
                    built[quarters] = parcellation.load_map(sid, quarters)
                else:
                    labels = parcellation.build_map(sid, quarters)
                    parcellation.save_map(labels, sid, quarters)
                    built[quarters] = labels
            print(f"    built {len(built)} maps "
                  f"({rest.minutes_for(sid, (0,1,2,3)):.0f} min total rest)", flush=True)

            curve = []
            for level, keys in config.VARIANCE_GROUPS.items():
                pairs = [parcellation.map_dice(built[a], built[b])
                         for a, b in combinations(keys, 2)]
                row = {"level": level, "dice": float(np.mean(pairs)),
                       "n_pairs": len(pairs),
                       "minutes": rest.minutes_for(sid, keys[0])}
                curve.append(row)
                print(f"    {level} | agreement {row['dice']:.3f} "
                      f"({row['n_pairs']} pairs, {row['minutes']:.0f} min/map)", flush=True)
            store[sid] = curve
            _write(outdir / "stability.json", store)

            if cleanup:
                print(f"    cleaned {cleanup_subject(sid, 'rest'):.1f} GB of rest",
                      flush=True)
        except Exception as exc:
            print(f"    SKIPPED -- {type(exc).__name__}: {exc}", flush=True)
            traceback.print_exc(limit=1)
        finally:
            rest.clear_cache()
            print(f"    [peak RSS {_peak_gb():.1f} GB]", flush=True)


# -------------------------------------------------------------- pass 2: identify
def stage_identify(subjects, outdir: Path, cleanup: bool) -> None:
    # Every cumulative map for the whole cohort, loaded once (~260 KB each), so each
    # held-out scan is read from disk exactly once rather than once per level.
    cohorts = {lv: parcellation.cohort(q) for lv, q in config.CUMULATIVE.items()}
    cohorts = {lv: c for lv, c in cohorts.items() if len(c) >= 2}
    if not cohorts:
        raise SystemExit("Fewer than two subjects have maps -- run `--stage maps` first.")
    for lv, c in cohorts.items():
        print(f"level {lv}: {len(c)} subjects in the lineup")

    records = _read(outdir / "identification.json", {})
    for subject in subjects:
        sid = config.sub_id(subject)
        scans = rest.task_scans(sid)
        print(f"\n=== [identify] {sid} — {len(scans)} held-out scans ===", flush=True)
        if not scans:
            print("    none on disk, skipping", flush=True)
            continue
        try:
            for path in scans:
                ts = rest.load_scan(path)
                for lv, maps in cohorts.items():
                    res = identify.identify(ts, maps, sid)
                    # Chance depends on how many maps were in the lineup, not on how many
                    # subjects happen to contribute test scans -- record it per scan.
                    res.update({"scan": path.name, "task": rest.task_name(path),
                                "n_candidates": len(maps)})
                    records.setdefault(lv, []).append(res)
                rest.clear_cache()  # one scan at a time; do not accumulate
            _write(outdir / "identification.json", records)
            for lv in cohorts:
                mine = [r for r in records[lv] if r["true"] == sid]
                print(f"    {lv}: recall {identify.accuracy(mine):.2f} "
                      f"over {len(mine)} scans", flush=True)
            if cleanup:
                print(f"    cleaned {cleanup_subject(sid, 'task'):.1f} GB of task data",
                      flush=True)
        except Exception as exc:
            print(f"    SKIPPED -- {type(exc).__name__}: {exc}", flush=True)
            traceback.print_exc(limit=1)
        finally:
            rest.clear_cache()
            print(f"    [peak RSS {_peak_gb():.1f} GB]", flush=True)


def cleanup_subject(subject: str, kind: str) -> float:
    """Delete a subject's raw BOLD once their numbers are in. Returns GB freed.

    A datalad/git-annex clone keeps content as read-only objects behind symlinks, so a
    plain unlink fails there and we must `git annex drop`. A plain download (the hub) is
    just files. We detect which and take the matching path.
    """
    sid = config.sub_id(subject)
    root = config.DATA_ROOT
    paths = [p for p in root.glob(f"sub-{sid}/ses-*/func/*_desc-preproc_bold.nii.gz")
             if (f"_task-{config.REST_TASK}_" in p.name) == (kind == "rest")]
    if not paths:
        return 0.0

    freed = 0.0
    if (root / ".git" / "annex").exists() and shutil.which("git-annex"):
        rels = []
        for p in paths:
            try:
                freed += p.resolve(strict=True).stat().st_size
                rels.append(str(p.relative_to(root)))
            except (FileNotFoundError, OSError):
                continue
        if rels:
            subprocess.run(["git", "-C", str(root), "annex", "drop", "--force", *rels],
                           capture_output=True)
    else:
        for p in paths:
            try:
                freed += p.stat().st_size
                p.unlink()
            except (FileNotFoundError, OSError):
                continue
    return freed / 1e9


# ----------------------------------------------------------------- stage: figure
def stage_figure(outdir: Path) -> None:
    stability = _read(outdir / "stability.json", {})
    ident = _read(outdir / "identification.json", {})
    if not stability:
        raise SystemExit("No stability results -- run `--stage maps` first.")

    results = {"subjects": {s: {"stability": v} for s, v in stability.items()}}
    # Chance = 1 / number of candidate maps the scan was scored against.
    lineup = max((r.get("n_candidates", 0) for v in ident.values() for r in v), default=0)
    results["chance"] = 1.0 / lineup if lineup else float("nan")
    results["identification"] = [
        {"level": lv, "accuracy": identify.accuracy(ident[lv]),
         "n_scans": len(ident[lv]),
         "n_subjects": len({r["true"] for r in ident[lv]}),
         "n_candidates": max((r.get("n_candidates", 0) for r in ident[lv]), default=0)}
        for lv in config.LEVELS if ident.get(lv)
    ]

    figure.save_csv(results, outdir / "curves.csv")
    figure.plot(results, outdir / "figure.png")
    _write(outdir / "results.json", results)
    print(f"\nfigure -> {outdir / 'figure.png'}")
    print(f"csv    -> {outdir / 'curves.csv'}")
    for row in results["identification"]:
        print(f"  {row['level']} | identification {row['accuracy']:.3f} "
              f"({row['n_scans']} scans, chance {results['chance']:.2f})")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--subjects", nargs="+", default=config.SUBJECTS)
    ap.add_argument("--outdir", default=str(config.RESULTS_DIR))
    ap.add_argument("--stage", choices=["maps", "identify", "figure", "all"], default="all")
    ap.add_argument("--cleanup", action="store_true",
                    help="delete a subject's raw BOLD once their numbers are computed")
    return ap


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    subjects = [config.sub_id(s) for s in args.subjects]

    print(f"data:    {config.DATA_ROOT}")
    print(f"maps:    {config.MAPS_DIR}")
    print(f"results: {outdir}")
    if args.cleanup:
        print("cleanup: ON -- raw BOLD is deleted after each subject")

    if args.stage in ("maps", "all"):
        stage_maps(subjects, outdir, args.cleanup)
    if args.stage in ("identify", "all"):
        stage_identify(subjects, outdir, args.cleanup)
    if args.stage in ("figure", "all"):
        stage_figure(outdir)


if __name__ == "__main__":
    main()
