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

from . import compare, config, figure, identify, parcellation, rest


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
    # Persist the grid now, while BOLD is still around, so `compare`/`identify`/`figure`
    # never need it -- `--cleanup` will have deleted every scan by the time they run.
    if not (config.DOMAIN_CACHE.exists() and config.GROUP_MAP_CACHE.exists()):
        d, g = parcellation.write_grid_cache()
        print(f"wrote grid cache: {d.name}, {g.name}", flush=True)

    store = _read(outdir / "stability.json", {})
    for subject in subjects:
        sid = config.sub_id(subject)
        print(f"\n=== [maps] {sid} ===", flush=True)
        try:
            built: dict[tuple[int, int], np.ndarray] = {}
            for spec in config.map_specs():
                if parcellation.has_map(sid, spec):
                    built[spec] = parcellation.load_map(sid, spec)
                else:
                    labels = parcellation.build_map(sid, spec)
                    parcellation.save_map(labels, sid, spec)
                    built[spec] = labels
            total_min = rest.minutes_for(sid, (0, config.N_CHUNKS))
            print(f"    built {len(built)} maps ({total_min:.0f} min total rest)",
                  flush=True)

            names = parcellation.network_order()
            curve = []
            for block in config.STABILITY_BLOCKS:
                pairs = config.stability_pairs(block)
                per_net = np.array([parcellation.dice_per_network(built[a], built[b])
                                    for a, b in pairs])            # (n_pairs, 17)
                mean_per_net = np.nanmean(per_net, axis=0)
                # Runs don't always divide evenly into 16 chunks, so the two maps in a
                # pair can hold slightly different amounts of data. Record the spread so
                # an imbalanced comparison is visible rather than silently averaged in.
                mins = [rest.minutes_for(sid, m) for pair in pairs for m in pair]
                row = {
                    "level": config.level_name(block),
                    "dice": float(np.nanmean(mean_per_net)),
                    "dice_per_network": {names[i]: float(mean_per_net[i])
                                         for i in range(config.N_NETWORKS)},
                    "n_pairs": len(pairs),
                    "minutes": float(np.mean(mins)),
                    "minutes_min": float(np.min(mins)),
                    "minutes_max": float(np.max(mins)),
                }
                curve.append(row)
                spread = ("" if row["minutes_max"] - row["minutes_min"] < 0.1
                          else f"  <- maps span {row['minutes_min']:.0f}-"
                               f"{row['minutes_max']:.0f} min, not equal")
                print(f"    {row['level']:>5} | agreement {row['dice']:.3f} "
                      f"({row['n_pairs']} pairs, {row['minutes']:.0f} min/map){spread}",
                      flush=True)
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
    cohorts = {config.level_name(b): parcellation.cohort((0, b))
               for b in config.CUMULATIVE_BLOCKS}
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
                print(f"    {lv:>5}: recall {identify.accuracy(mine):.2f} | "
                      f"mean margin {identify.mean_margin(mine):+.4f} "
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


# ---------------------------------------------------------------- stage: compare
def stage_compare(subjects, outdir: Path) -> None:
    """Within / between / to-group Dice, from maps already on disk. Reads no BOLD."""
    have = [s for s in subjects if parcellation.has_map(s, (0, config.N_CHUNKS))]
    if len(have) < 2:
        raise SystemExit("Need maps for >=2 subjects -- run `--stage maps` first.")
    print(f"comparing {len(have)} subjects: {have}", flush=True)

    # Durations come from stability.json, written by `maps` while BOLD was still on disk.
    # Chunks are equal-duration, so total = minutes(block) * N_CHUNKS / block.
    minutes = {}
    for sub, rows in _read(outdir / "stability.json", {}).items():
        for row in rows:
            block = int(row["level"].split("/")[0])
            if row.get("minutes"):
                minutes[sub] = row["minutes"] * config.N_CHUNKS / block
                break
    missing = [s for s in have if s not in minutes]
    if missing:
        print(f"  ! no duration recorded for {missing}; those points will lack a "
              f"minutes coordinate (re-run `--stage maps` with BOLD present to fix)",
              flush=True)
    else:
        print("  total rest per subject: "
              + ", ".join(f"{s} {minutes[s]:.0f}m" for s in have), flush=True)

    result = compare.compare_all(have, minutes)
    _write(outdir / "comparisons.json", result)

    print(f"\n{'level':>6} {'within':>8} {'between':>8} {'to-group':>9}   "
          f"{'assoc(w)':>9} {'sensori(w)':>10}")
    btw = {r["level"]: r for r in result["between"]}
    for i, level in enumerate(config.level_name(b) for b in config.STABILITY_BLOCKS):
        w = np.nanmean([result["per_subject"][s]["levels"][i]["within"] for s in have])
        g = np.nanmean([result["per_subject"][s]["levels"][i]["to_group"] for s in have])
        wa = np.nanmean([result["per_subject"][s]["levels"][i]["within_association"]
                         for s in have])
        ws = np.nanmean([result["per_subject"][s]["levels"][i]["within_sensorimotor"]
                         for s in have])
        print(f"{level:>6} {w:>8.3f} {btw[level]['dice']:>8.3f} {g:>9.3f}   "
              f"{wa:>9.3f} {ws:>10.3f}")

    # --- slopes, finite differences, saturation -------------------------------
    slopes = compare.slope_report(result)
    result["slopes"] = slopes
    _write(outdir / "comparisons.json", result)

    print("\nslope in log(minutes)   b = gain per e-fold; per-doubling = b*ln2")
    for field in ("within", "between", "to_group"):
        g = slopes["group"].get(field)
        if not g:
            continue
        s = g["slope"]
        print(f"  {field:9} b={s['b']:+.4f}  per doubling={s['per_doubling']:+.4f}  "
              f"r2={s['r2']:.3f}  (n={s['n']})")
    ratio = slopes["b_within_over_b_between"]
    if np.isfinite(ratio):
        print(f"  b_within / b_between = {ratio:.2f}x "
              f"({'individuation outpaces the baseline' if ratio > 1 else 'NO faster than baseline'})")
    print(f"  per-subject b_within: mean {slopes['b_within_subject_mean']:+.4f} "
          f"(sd {slopes['b_within_subject_sd']:.4f})")

    print("\nfinite differences (dy/dlog x between adjacent levels)")
    for field in ("within", "between", "to_group"):
        g = slopes["group"].get(field)
        if not g:
            continue
        fds = " ".join(f"{f['mid_minutes']:.0f}m:{f['slope']:+.3f}"
                       for f in g["finite_differences"])
        flag = "DECLINING -> saturating" if g["finite_differences_decline"] else "~constant"
        print(f"  {field:9} {fds}   [{flag}]")
        sat = g.get("saturating")
        if sat:
            enough = sat["enough_minutes"]
            where = ("not reached in the measured range" if enough is None else
                     f"{enough:.0f} min" + (" (EXTRAPOLATED beyond the data)"
                                            if sat["extrapolated"] else ""))
            print(f"            saturating fit: ymax={sat['ymax']:.3f} "
                  f"tau={sat['tau']:.1f}m r2={sat['r2']:.3f}")
            print(f"            'enough data' (<{sat['threshold']} Dice per doubling): {where}")

    crossings = {s: result["per_subject"][s]["crossover"] for s in have}
    print("\ncrossover (within-person first exceeds similarity-to-group):")
    for s, c in crossings.items():
        print(f"  {s}: {c if c else 'not within the measured range'}")
    # Report the distribution rather than a single "most common" level: with few subjects
    # ties are common, and picking a winner out of a set is order-dependent.
    order = [config.level_name(b) for b in config.STABILITY_BLOCKS]
    reached = [c for c in crossings.values() if c]
    counts = {lv: reached.count(lv) for lv in order if reached.count(lv)}
    print(f"  -> {len(reached)}/{len(have)} subjects cross"
          + (f"; by level: {counts}" if counts else ""))
    if reached:
        median = sorted(reached, key=order.index)[len(reached) // 2]
        print(f"  -> median crossover level: {median}")


# ----------------------------------------------------------------- stage: figure
def stage_figure(outdir: Path) -> None:
    stability = _read(outdir / "stability.json", {})
    ident = _read(outdir / "identification.json", {})
    comparisons = _read(outdir / "comparisons.json", {})
    if not stability:
        raise SystemExit("No stability results -- run `--stage maps` first.")

    results = {"subjects": {s: {"stability": v} for s, v in stability.items()},
               "comparisons": comparisons}
    # Chance = 1 / number of candidate maps the scan was scored against.
    lineup = max((r.get("n_candidates", 0) for v in ident.values() for r in v), default=0)
    results["chance"] = 1.0 / lineup if lineup else float("nan")
    results["identification"] = [
        {"level": lv,
         "accuracy": identify.accuracy(ident[lv]),
         "margin": identify.mean_margin(ident[lv]),
         "n_scans": len(ident[lv]),
         "n_subjects": len({r["true"] for r in ident[lv]}),
         "n_candidates": max((r.get("n_candidates", 0) for r in ident[lv]), default=0)}
        for lv in config.LEVELS if ident.get(lv)
    ]
    # Per-subject mean margin, so the bottom panel can show individual traces too.
    for sub, res in results["subjects"].items():
        res["margin"] = [
            {"level": lv,
             "margin": identify.mean_margin([r for r in ident[lv] if r["true"] == sub])}
            for lv in config.LEVELS if ident.get(lv)
        ]

    figure.save_csv(results, outdir / "curves.csv")
    figure.plot(results, outdir / "figure.png")
    if comparisons.get("per_subject"):
        figure.plot_comparisons(results, outdir / "figure_comparisons.png")
        figure.save_comparison_csv(results, outdir / "comparisons.csv")
        print(f"panels -> {outdir / 'figure_comparisons.png'}")
    _write(outdir / "results.json", results)
    print(f"\nfigure -> {outdir / 'figure.png'}")
    print(f"csv    -> {outdir / 'curves.csv'}")
    for row in results["identification"]:
        print(f"  {row['level']:>5} | margin {row['margin']:+.4f} "
              f"| accuracy {row['accuracy']:.3f} "
              f"({row['n_scans']} scans, chance {results['chance']:.2f})")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--subjects", nargs="+", default=config.SUBJECTS)
    ap.add_argument("--outdir", default=str(config.RESULTS_DIR))
    ap.add_argument("--stage",
                    choices=["maps", "compare", "identify", "figure", "all"],
                    default="all")
    ap.add_argument("--cleanup", action="store_true",
                    help="delete a subject's raw BOLD once their numbers are computed")
    return ap


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    subjects = [config.sub_id(s) for s in args.subjects]

    print(f"data:    {config.DATA_ROOT}")
    print(f"maps:    {config.MAPS_DIR}   (version {config.MAP_VERSION})")
    print(f"results: {outdir}")

    # Maps used to live unversioned in derivatives/maps. If a previous run's maps are
    # sitting there, say so rather than silently building a fresh empty version.
    versioned = list(config.MAPS_DIR.glob("*.npy")) if config.MAPS_DIR.exists() else []
    legacy = list(config.legacy_maps_dir().glob("*.npy"))
    if legacy and not versioned:
        print(f"\n!! {len(legacy)} unversioned maps found in {config.legacy_maps_dir()}")
        print(f"   Move them into the versioned folder to use them:")
        print(f"     mkdir -p {config.MAPS_DIR} && "
              f"mv {config.legacy_maps_dir()}/*.npy {config.MAPS_DIR}/")
        print(f"   (or point PANMVPA_MAPS at them)\n")
    if args.cleanup:
        print("cleanup: ON -- raw BOLD is deleted after each subject")

    if args.stage in ("maps", "all"):
        stage_maps(subjects, outdir, args.cleanup)
    if args.stage in ("compare", "all"):
        stage_compare(subjects, outdir)
    if args.stage in ("identify", "all"):
        stage_identify(subjects, outdir, args.cleanup)
    if args.stage in ("figure", "all"):
        stage_figure(outdir)


if __name__ == "__main__":
    main()
