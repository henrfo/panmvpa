"""datalad get helper — pull only the epiproj slice for a subject.

Clone first (once):
    datalad clone https://github.com/OpenNeuroDatasets/ds006598.git data/ds006598

Then:
    python fetch_data.py --subject PAN01              # all sessions
    python fetch_data.py --subject PAN01 --sessions 1 2 3
"""
from __future__ import annotations

import argparse
import subprocess

import config


def find_sessions(subject: str) -> list[int]:
    """Sessions that actually have an epiproj BOLD on disk (post-clone, pre-get)."""
    sid = config.sub_id(subject)
    subdir = config.DATA_ROOT / f"sub-{sid}"
    if not subdir.exists():
        raise SystemExit(f"{subdir} not found. Clone the dataset first (see module docstring).")
    sessions = []
    for ses in sorted(subdir.glob("ses-*")):
        n = int(ses.name.removeprefix("ses-"))
        if config.bold_path(subject, n).exists():  # annex symlink exists even before get
            sessions.append(n)
    return sessions


def datalad_get(paths: list, dry_run: bool) -> None:
    existing = [str(p) for p in paths if p.exists()]
    missing = [p for p in paths if not p.exists()]
    for p in missing:
        print(f"  [skip] not in clone: {p}")
    if not existing:
        print("Nothing to get.")
        return
    cmd = ["datalad", "get", *existing]
    if dry_run:
        print("Would run:", " ".join(cmd[:3]), f"... ({len(existing)} paths)")
        return
    subprocess.run(cmd, cwd=config.DATA_ROOT, check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subject", required=True, help="e.g. PAN01")
    ap.add_argument("--sessions", type=int, nargs="*", help="default: all with epiproj")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    sessions = args.sessions or find_sessions(args.subject)
    print(f"{config.sub_id(args.subject)}: sessions {sessions}")

    paths = []
    for ses in sessions:
        paths.append(config.bold_path(args.subject, ses))
        for cond in config.CONDITIONS:
            paths.append(config.afni_timing_path(args.subject, ses, cond))

    datalad_get(paths, args.dry_run)


if __name__ == "__main__":
    main()
