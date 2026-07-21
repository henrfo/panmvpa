"""End-to-end driver: both curves across subjects -> group figure + CSV.

  Plot 1  DN-A split-half parcellation reliability (Dice)
  Plot 2  DN-A contrast-to-noise during episodic projection (mean Z in − mean Z out)

Installed as the ``panmvpa-run`` console script, so it works from any working directory
(the data path comes from $DATA_DIR, not from where you happen to be standing):

    panmvpa-run                          # all 10 subjects
    panmvpa-run --subjects PAN01 PAN02
    python henrik_sidequest/scripts/run_all.py    # equivalent

Subjects are processed strictly one at a time and every per-subject cache is dropped
between them, so peak memory tracks a single subject rather than the cohort.
"""
from __future__ import annotations

import argparse
import json
import resource
import traceback
from pathlib import Path

from . import cnr, config, figure, reliability
from . import clear_caches


def _peak_gb() -> float:
    """Peak RSS in GB (ru_maxrss is bytes on macOS, kB on Linux)."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 1e9 if peak > 1e7 else peak / 1e6


def run_subject(subject: str, minutes: list[float]) -> dict:
    rel = reliability.reliability_curve(subject, minutes)
    c = cnr.cnr_curve(subject, minutes)
    for r, cc in zip(rel, c):
        print(
            f"    {r['minutes']:>4.0f} min | Dice {r['dice']:.3f} "
            f"| CNR {cc['cnr']:+.3f} (Z in {cc['mean_in']:+.2f} / out {cc['mean_out']:+.2f}) "
            f"| DN-A {cc['n_dna_voxels']} vox | {cc['n_sessions']} epiproj ses",
            flush=True,
        )
    return {"reliability": rel, "cnr": c}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--subjects", nargs="+", default=config.SUBJECTS)
    ap.add_argument("--minutes", type=float, nargs="+", default=config.MINUTE_LEVELS)
    ap.add_argument(
        "--outdir",
        default=str(config.RESULTS_DIR),
        help="where the figure/CSV go (default: repo results/, which is tracked in git)",
    )
    return ap


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    minutes = list(args.minutes)

    print(f"data:    {config.DATA_ROOT}")
    print(f"results: {outdir}")
    print(f"levels:  {minutes} min")

    results: dict = {"minutes": minutes, "subjects": {}}
    skipped: dict[str, str] = {}

    for subject in args.subjects:
        sid = config.sub_id(subject)
        print(f"\n=== {sid} ===", flush=True)
        try:
            results["subjects"][sid] = run_subject(sid, minutes)
        except Exception as exc:  # one thin subject shouldn't kill the cohort run
            skipped[sid] = f"{type(exc).__name__}: {exc}"
            print(f"    SKIPPED -- {skipped[sid]}", flush=True)
            traceback.print_exc(limit=1)
        finally:
            clear_caches()  # bound peak memory to a single subject
            print(f"    [peak RSS {_peak_gb():.1f} GB]", flush=True)

    if not results["subjects"]:
        raise SystemExit("No subjects completed; nothing to plot.")

    figure.save_group_csv(results, outdir / "group_curves.csv")
    figure.plot_group(results, outdir / "group_figure.png")
    (outdir / "group_results.json").write_text(json.dumps(results, indent=2))

    n = len(results["subjects"])
    print(f"\nfigure -> {outdir / 'group_figure.png'}  ({n} subjects)")
    print(f"csv    -> {outdir / 'group_curves.csv'}")
    if skipped:
        print("\nskipped subjects:")
        for sid, why in skipped.items():
            print(f"  {sid}: {why}")


if __name__ == "__main__":
    main()
