"""End-to-end PAN-MVPA group figure.

Loops over subjects and rest-data levels, builds both curves on a shared x-axis:

  Plot 1  DN-A split-half parcellation reliability (Dice)
  Plot 2  DN-A contrast-to-noise during episodic projection (mean Z in − mean Z out)

Writes:
  <outdir>/group_figure.png   two panels, mean across subjects ± 1 SEM
  <outdir>/group_curves.csv   per-subject rows + GROUP_MEAN / GROUP_SEM rows

    python scripts/run_all.py                     # all 10 subjects
    python scripts/run_all.py --subjects PAN01 PAN02

Subjects that lack enough rest or any fetched epiproj run are skipped with a warning
rather than aborting the run.
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

# Make `panmvpa` importable when run as `python scripts/run_all.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from panmvpa import cnr, config, figure, reliability  # noqa: E402


def run_subject(subject: str, minutes: list[float]) -> dict | None:
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", nargs="+", default=config.SUBJECTS)
    ap.add_argument("--minutes", type=float, nargs="+", default=config.MINUTE_LEVELS)
    ap.add_argument("--outdir", default=str(config.DERIV_ROOT / "figures"))
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    minutes = list(args.minutes)

    results: dict = {"minutes": minutes, "subjects": {}}
    skipped: dict[str, str] = {}

    for subject in args.subjects:
        sid = config.sub_id(subject)
        print(f"\n=== {sid} ===", flush=True)
        try:
            res = run_subject(sid, minutes)
        except Exception as exc:  # keep going; one thin subject shouldn't kill the run
            skipped[sid] = f"{type(exc).__name__}: {exc}"
            print(f"    SKIPPED -- {skipped[sid]}", flush=True)
            traceback.print_exc(limit=1)
            continue
        results["subjects"][sid] = res

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
