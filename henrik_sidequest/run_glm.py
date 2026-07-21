"""Run Step-1 first-level GLM for one subject.

    python run_glm.py --subject PAN01                 # all sessions with data on disk
    python run_glm.py --subject PAN01 --sessions 1 2
"""
from __future__ import annotations

import argparse

import config
import glm


def sessions_on_disk(subject: str) -> list[int]:
    sid = config.sub_id(subject)
    subdir = config.DATA_ROOT / f"sub-{sid}"
    found = []
    for ses in sorted(subdir.glob("ses-*")):
        n = int(ses.name.removeprefix("ses-"))
        # need the actual BOLD fetched, not just the annex symlink
        if config.bold_path(subject, n).is_file():
            found.append(n)
    return found


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subject", required=True)
    ap.add_argument("--sessions", type=int, nargs="*")
    args = ap.parse_args()

    sessions = args.sessions or sessions_on_disk(args.subject)
    if not sessions:
        raise SystemExit(
            f"No epiproj BOLD found for {config.sub_id(args.subject)}. "
            f"Run: python fetch_data.py --subject {args.subject}"
        )
    glm.run_subject(args.subject, sessions)


if __name__ == "__main__":
    main()
