"""datalad get helper — pull only the epiproj slice for a subject.

Clone first (once):
    datalad clone https://github.com/OpenNeuroDatasets/ds006598.git data/ds006598

Then:
    python fetch_data.py --subject PAN01              # all epiproj sessions
    python fetch_data.py --subject PAN01 --sessions 1 3
"""
from __future__ import annotations

import argparse
import subprocess

from panmvpa import config
from panmvpa.events import epiproj_sessions


def datalad_get(paths: list, dry_run: bool) -> None:
    existing = [str(p) for p in paths if p.exists()]  # annex symlink present in clone
    for p in paths:
        if not p.exists():
            print(f"  [skip] not in clone: {p.name}")
    if not existing:
        print("Nothing to get.")
        return
    cmd = ["datalad", "get", *existing]
    if dry_run:
        print(f"Would `datalad get` {len(existing)} paths.")
        return
    subprocess.run(cmd, cwd=config.DATA_ROOT, check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subject", required=True, help="e.g. PAN01")
    ap.add_argument("--sessions", type=int, nargs="*", help="default: all epiproj sessions")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    sessions = args.sessions or epiproj_sessions(args.subject)
    if not sessions:
        raise SystemExit(
            f"No epiproj sessions for {config.sub_id(args.subject)}. "
            f"Is the dataset cloned at {config.DATA_ROOT}?"
        )
    print(f"{config.sub_id(args.subject)}: epiproj sessions {sessions}")

    paths = []
    for ses in sessions:
        paths.append(config.bold_path(args.subject, ses))
        for cond in config.CONDITIONS:
            paths.append(config.afni_timing_path(args.subject, ses, cond))

    datalad_get(paths, args.dry_run)


if __name__ == "__main__":
    main()
