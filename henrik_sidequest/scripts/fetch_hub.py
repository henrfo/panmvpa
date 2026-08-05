"""Stream ds006598 scans from OpenNeuro's public S3, one subject at a time.

Stdlib only -- no datalad, no git-annex, nothing to install on a JupyterHub.

    python scripts/fetch_hub.py --dest $DATA_DIR --subjects PAN01 --kind rest
    python scripts/fetch_hub.py --dest $DATA_DIR --subjects PAN01 --kind task

``--kind rest`` fetches what pass 1 needs, ``--kind task`` what pass 2 needs. Splitting
them is what keeps disk to roughly one subject's worth at a time. Files already present at
the right size are skipped, so an interrupted fetch resumes where it stopped.
"""
from __future__ import annotations

import argparse
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

BUCKET = "https://s3.amazonaws.com/openneuro.org"
DATASET = "ds006598"
SUFFIX = "space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz"
NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
SUBJECTS = [f"PAN{i:02d}" for i in range(1, 11)]


def list_keys(prefix: str) -> list[tuple[str, int]]:
    """(key, size) for every object under a prefix, following continuation tokens."""
    out, token = [], None
    while True:
        query = {"list-type": "2", "prefix": prefix, "max-keys": "1000"}
        if token:
            query["continuation-token"] = token
        with urllib.request.urlopen(f"{BUCKET}?{urllib.parse.urlencode(query)}") as r:
            root = ET.fromstring(r.read())
        for c in root.findall("s3:Contents", NS):
            out.append((c.findtext("s3:Key", "", NS), int(c.findtext("s3:Size", "0", NS))))
        if root.findtext("s3:IsTruncated", "false", NS) != "true":
            return out
        token = root.findtext("s3:NextContinuationToken", None, NS)


def select(subject: str, kind: str) -> list[tuple[str, int]]:
    """The subject's rest runs, or their non-rest (held-out) runs."""
    keys = [kv for kv in list_keys(f"{DATASET}/sub-{subject}/") if kv[0].endswith(SUFFIX)]
    is_rest = lambda k: "_task-rest_" in k  # noqa: E731
    chosen = [kv for kv in keys if is_rest(kv[0]) == (kind == "rest")]
    return sorted(chosen)


def download(key: str, size: int, dest_root: Path) -> bool:
    """Fetch one object, preserving its path under the dataset root. True if downloaded."""
    out = dest_root / key[len(DATASET) + 1:]
    if out.exists() and out.stat().st_size == size:
        return False
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".part")
    with urllib.request.urlopen(f"{BUCKET}/{urllib.parse.quote(key)}") as r, \
            open(tmp, "wb") as fh:
        while chunk := r.read(1 << 20):
            fh.write(chunk)
    tmp.replace(out)
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dest", required=True, help="dataset root, e.g. $DATA_DIR")
    ap.add_argument("--subjects", nargs="+", default=SUBJECTS)
    ap.add_argument("--kind", choices=["rest", "task"], default="rest")
    args = ap.parse_args()

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    total = 0

    for subject in args.subjects:
        items = select(subject, args.kind)
        size = sum(s for _, s in items)
        print(f"\n=== {subject} ({args.kind}) — {len(items)} runs, {size/1e9:.1f} GB ===",
              flush=True)
        got = 0
        for i, (key, sz) in enumerate(items, 1):
            if download(key, sz, dest):
                got += sz
            if i % 5 == 0 or i == len(items):
                print(f"  [{i}/{len(items)}] {got/1e9:.1f} GB fetched", flush=True)
        total += got

    print(f"\nDone. Downloaded {total/1e9:.1f} GB into {dest}")


if __name__ == "__main__":
    main()
