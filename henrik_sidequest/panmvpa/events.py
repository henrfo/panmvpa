"""Parse AFNI .1D timing files into nilearn-compatible events tables.

.1D format (ds006598): whitespace-separated onset times in seconds, one run per row.
epiproj has a single run per session, so a single row. An AFNI '*' marks an empty run.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import config


def parse_1d_file(path: str | Path) -> np.ndarray:
    """Return sorted onset times (seconds) from a .1D file. Empty if none/absent."""
    path = Path(path)
    if not path.exists():
        return np.array([])
    text = path.read_text().strip()
    if not text:
        return np.array([])
    onsets = []
    for tok in text.split():
        try:
            onsets.append(float(tok))
        except ValueError:
            continue  # AFNI '*' placeholder or similar
    return np.array(sorted(onsets))


def epiproj_sessions(subject: str) -> list[int]:
    """Sessions for which this subject actually has epiproj timing files.

    Works off the datalad clone's symlinks (present before `datalad get`).
    """
    sid = config.sub_id(subject)
    timing_dir = config.DATA_ROOT / "derivatives" / "afni_timing" / sid
    if not timing_dir.exists():
        return []
    sessions = set()
    for f in timing_dir.glob(f"sub-{sid}_ses-*_task-{config.TASK}_*.1D"):
        # sub-PANxx_ses-{N}_task-epiproj_{cond}.1D
        ses = f.name.split("_ses-")[1].split("_")[0]
        sessions.add(int(ses))
    return sorted(sessions)


def build_events(
    subject: str,
    session: int,
    conditions: list[str] | None = None,
    duration: float = config.EVENT_DURATION,
) -> pd.DataFrame:
    """nilearn events frame (onset, duration, trial_type) for one epiproj session."""
    conditions = conditions or config.CONDITIONS
    rows = []
    for cond in conditions:
        path = config.afni_timing_path(subject, session, cond)
        for onset in parse_1d_file(path):
            rows.append({"onset": onset, "duration": duration, "trial_type": cond})
    if not rows:
        raise ValueError(
            f"no epiproj events for {config.sub_id(subject)} ses-{session} "
            f"(is this an epiproj session? see epiproj_sessions())"
        )
    return pd.DataFrame(rows).sort_values("onset").reset_index(drop=True)
