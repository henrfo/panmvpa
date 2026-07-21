"""End-to-end PAN-MVPA figure for one subject.

Loops over rest-data levels, builds both curves on a shared x-axis, and writes:
  <outdir>/<SID>_figure.png   two-panel figure (Dice top, accuracy bottom)
  <outdir>/<SID>_curves.csv   raw numbers

    python scripts/run_all.py --subject PAN01

Multi-subject is just a loop over --subject; nothing here is PAN01-specific.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make `panmvpa` importable when run as `python scripts/run_all.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from panmvpa import config, figure, mvpa, reliability  # noqa: E402
from panmvpa.mvpa import complete_sessions  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", default="PAN01")
    ap.add_argument("--minutes", type=float, nargs="+", default=config.MINUTE_LEVELS)
    ap.add_argument("--outdir", default=str(config.DERIV_ROOT / "figures"))
    args = ap.parse_args()

    sid = config.sub_id(args.subject)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    sessions = complete_sessions(sid)
    print(f"{sid}: complete sessions (all 4 classes) = {sessions}  ->  {len(sessions)} CV folds")
    if len(sessions) < 2:
        raise SystemExit("Need >=2 complete sessions for leave-one-session-out CV.")

    minutes = list(args.minutes)
    print(f"Computing reliability curve over {minutes} min ...")
    rel = reliability.reliability_curve(sid, minutes)
    print(f"Computing decoding curve over {minutes} min ...")
    dec = mvpa.decoding_curve(sid, minutes)

    figure.save_csv(rel, dec, outdir / f"{sid}_curves.csv")
    figure.plot(sid, rel, dec, outdir / f"{sid}_figure.png")
    print(f"figure -> {outdir / f'{sid}_figure.png'}")
    print(f"csv    -> {outdir / f'{sid}_curves.csv'}")

    print("\n=== summary ===")
    for r, d in zip(rel, dec):
        print(
            f"{r['minutes']:>4.0f} min | Dice {r['dice']:.3f} "
            f"(DN-A {r['n_voxels_a']}/{r['n_voxels_b']} vox) | "
            f"acc {d['accuracy']:.3f} / chance {d['chance']:.2f} "
            f"({d['n_features']} feat, {d['n_samples']} samples, {d['n_folds']} folds)"
        )


if __name__ == "__main__":
    main()
