"""Fetch the rest + epiproj slice of ds006598 straight from OpenNeuro's public S3.

Deliberately dependency-free: stdlib only, no datalad and no git-annex, because both are
awkward to install on a JupyterHub. Lists the public bucket over HTTPS and downloads the
files the pipeline needs.

    python scripts/fetch_hub.py --dest /home/jovyan/data/ds006598
    python scripts/fetch_hub.py --dest $DATA_DIR --subjects PAN01 PAN02

Then point the pipeline at it:

    export DATA_DIR=/home/jovyan/data/ds006598
    panmvpa-run

Downloads are resumable in the sense that files already present with the expected byte
size are skipped, so re-running after an interruption only fetches what's missing.
"""
from __future__ import annotations

import argparse
import sys
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

BUCKET = "https://s3.amazonaws.com/openneuro.org"
DATASET = "ds006598"
SPACE = "space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz"
NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}

# One PAN rest run is 222 volumes at TR 1.355 s ~= 5.01 min.
RUN_MINUTES = 5.01
SUBJECTS = [f"PAN{i:02d}" for i in range(1, 11)]


def list_keys(prefix: str) -> list[tuple[str, int]]:
    """(key, size) for every object under a prefix, following continuation tokens."""
    out: list[tuple[str, int]] = []
    token = None
    while True:
        q = {"list-type": "2", "prefix": prefix, "max-keys": "1000"}
        if token:
            q["continuation-token"] = token
        with urllib.request.urlopen(f"{BUCKET}?{urllib.parse.urlencode(q)}") as r:
            root = ET.fromstring(r.read())
        for c in root.findall("s3:Contents", NS):
            out.append((c.findtext("s3:Key", "", NS), int(c.findtext("s3:Size", "0", NS))))
        if root.findtext("s3:IsTruncated", "false", NS) != "true":
            return out
        token = root.findtext("s3:NextContinuationToken", None, NS)


def _ses_run(key: str) -> tuple[int, int]:
    """Sort key: (session, run) parsed from a BIDS filename."""
    name = key.rsplit("/", 1)[-1]

    def grab(tag: str) -> int:
        for part in name.split("_"):
            if part.startswith(tag):
                return int(part[len(tag):])
        return 0

    return grab("ses-"), grab("run-")


def select(subject: str, minutes: float, kind: str = "all",
           max_tasks: int | None = None) -> list[tuple[str, int]]:
    """Files for one subject.

    ``kind='rest'``  rest runs covering ``minutes`` (+ timing) -- what pass 1 needs.
    ``kind='task'``  the held-out non-rest scans -- what pass 2 needs.
    ``kind='all'``   both.

    Splitting by kind is what lets the two passes each hold only one subject's worth of
    raw data on disk at a time.
    """
    keys = list_keys(f"{DATASET}/sub-{subject}/")
    keys += list_keys(f"{DATASET}/derivatives/afni_timing/{subject}/")

    timing = [(k, s) for k, s in keys if k.endswith(".1D")]
    bold = [kv for kv in keys if kv[0].endswith(SPACE)]
    rest = sorted((kv for kv in bold if "_task-rest_" in kv[0]), key=lambda kv: _ses_run(kv[0]))
    tasks = sorted((kv for kv in bold if "_task-rest_" not in kv[0]),
                   key=lambda kv: (_task_of(kv[0]), _ses_run(kv[0])))

    n_needed = int(-(-minutes // RUN_MINUTES))  # ceil
    chosen_rest = rest[:n_needed]
    if len(chosen_rest) < n_needed:
        print(
            f"  ! {subject}: only {len(rest)} rest runs (~{len(rest)*RUN_MINUTES:.0f} min) "
            f"available, wanted {n_needed} for {minutes:.0f} min",
            file=sys.stderr,
        )
    if max_tasks is not None:
        tasks = _spread_across_tasks(tasks, max_tasks)

    if kind == "rest":
        return timing + chosen_rest
    if kind == "task":
        return timing + tasks
    return timing + chosen_rest + tasks


def _task_of(key: str) -> str:
    for part in key.rsplit("/", 1)[-1].split("_"):
        if part.startswith("task-"):
            return part[len("task-"):]
    return "unknown"


def _spread_across_tasks(items: list[tuple[str, int]], cap: int) -> list[tuple[str, int]]:
    """Take up to ``cap`` runs, round-robin across task families so no task dominates."""
    by_task: dict[str, list[tuple[str, int]]] = {}
    for kv in items:
        by_task.setdefault(_task_of(kv[0]), []).append(kv)
    out: list[tuple[str, int]] = []
    while len(out) < cap and any(by_task.values()):
        for t in sorted(by_task):
            if by_task[t] and len(out) < cap:
                out.append(by_task[t].pop(0))
    return out


def download(key: str, size: int, dest_root: Path) -> bool:
    """Download one key, preserving its path under the dataset root. True if fetched."""
    rel = key[len(DATASET) + 1:]
    out = dest_root / rel
    if out.exists() and out.stat().st_size == size:
        return False
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".part")
    url = f"{BUCKET}/{urllib.parse.quote(key)}"
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as fh:
        while chunk := r.read(1 << 20):
            fh.write(chunk)
    tmp.replace(out)
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dest", required=True, help="dataset root, e.g. $DATA_DIR")
    ap.add_argument("--subjects", nargs="+", default=SUBJECTS)
    ap.add_argument("--minutes", type=float, default=100.0,
                    help="rest minutes to cover per subject (default 100)")
    ap.add_argument("--kind", choices=["rest", "task", "all"], default="all",
                    help="rest = what `--stage maps` needs; task = what `--stage identify` "
                         "needs. Fetch one kind at a time to keep disk use low.")
    ap.add_argument("--max-tasks", type=int, default=None,
                    help="cap held-out task runs per subject, spread across task families "
                         "(default: all of them)")
    args = ap.parse_args()

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    grand_bytes = 0

    for subject in args.subjects:
        print(f"\n=== {subject} ({args.kind}) ===", flush=True)
        items = select(subject, args.minutes, kind=args.kind, max_tasks=args.max_tasks)
        total = sum(s for _, s in items)
        print(f"  {len(items)} files, {total/1e9:.1f} GB", flush=True)
        got = 0
        for i, (key, size) in enumerate(items, 1):
            if download(key, size, dest):
                got += size
            if i % 5 == 0 or i == len(items):
                print(f"  [{i}/{len(items)}] {got/1e9:.1f} GB fetched", flush=True)
        grand_bytes += got

    print(f"\nDone. Downloaded {grand_bytes/1e9:.1f} GB into {dest}")
    print(f"Now:  export DATA_DIR={dest}  &&  panmvpa-run")


if __name__ == "__main__":
    main()
