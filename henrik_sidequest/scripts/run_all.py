"""Thin wrapper so `python scripts/run_all.py` works without installing the package.

The real driver is `panmvpa.cli`, also exposed as the `panmvpa-run` console script.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make `panmvpa` importable when run as `python scripts/run_all.py` (uninstalled).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from panmvpa.cli import main  # noqa: E402

if __name__ == "__main__":
    main()
