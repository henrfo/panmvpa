"""Paths and constants for the PAN-MVPA sidequest.

Dataset facts verified against the ds006598 S3 listing (2026-07-20). Durations confirmed
from the paper's STAR Methods.
"""
from __future__ import annotations

import os
from pathlib import Path

# --- Locations -------------------------------------------------------------
PKG_DIR = Path(__file__).resolve().parent
REPO_DIR = PKG_DIR.parent  # henrik_sidequest/

# Point PANMVPA_DATA at your datalad clone of ds006598.
DATA_ROOT = Path(os.environ.get("PANMVPA_DATA", REPO_DIR / "data" / "ds006598"))

ATLAS_DIR = REPO_DIR / "atlases"       # downloaded template atlases live here
DERIV_ROOT = REPO_DIR / "derivatives"  # any derived outputs

# --- Subjects --------------------------------------------------------------
SUBJECTS = [f"PAN{n:02d}" for n in range(1, 11)]  # PAN01 .. PAN10

# --- Acquisition -----------------------------------------------------------
TR = 1.355  # seconds, from *_bold.json RepetitionTime
SPACE = "MNI152NLin6Asym_res-2"

# --- Episodic Projection task ---------------------------------------------
TASK = "epiproj"

CONDITIONS = [
    "pastself",
    "presentself",
    "futureself",
    "pastnonself",
    "presentnonself",
    "futurenonself",
]

# STAR Methods: each block is 20 s (5 s fixation + 10 s trial + 5 s fixation).
# We model the 10 s active trial period, not the full block.
EVENT_DURATION = 10.0

# Only ~6 epiproj runs exist per subject, spread across sessions. Never assume a
# session has epiproj — discover it (see events.epiproj_sessions).

CONTRASTS = {
    "retrospection": ("pastself", "presentself"),   # past vs present self
    "prospection": ("futureself", "presentself"),   # future vs present self
}


# --- Path helpers ----------------------------------------------------------
def sub_id(subject: str) -> str:
    """Normalise 'PAN01', 'sub-PAN01', or '01' -> 'PAN01'."""
    s = subject.removeprefix("sub-").removeprefix("PAN")
    return f"PAN{s}"


def bold_path(subject: str, session: int, task: str = TASK) -> Path:
    """Preprocessed BOLD for one subject/session/task."""
    sid = sub_id(subject)
    fname = f"sub-{sid}_ses-{session}_task-{task}_space-{SPACE}_desc-preproc_bold.nii.gz"
    return DATA_ROOT / f"sub-{sid}" / f"ses-{session}" / "func" / fname


def afni_timing_path(subject: str, session: int, condition: str, task: str = TASK) -> Path:
    """AFNI .1D onset file for one subject/session/condition."""
    sid = sub_id(subject)
    fname = f"sub-{sid}_ses-{session}_task-{task}_{condition}.1D"
    return DATA_ROOT / "derivatives" / "afni_timing" / sid / fname
