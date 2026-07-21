"""Paths and constants for the PAN-MVPA sidequest.

All dataset facts here were verified against the ds006598 S3 listing on 2026-07-20.
"""
from __future__ import annotations

import os
from pathlib import Path

# --- Data location ---------------------------------------------------------
# Point PANMVPA_DATA at your datalad clone of ds006598.
DATA_ROOT = Path(os.environ.get("PANMVPA_DATA", Path(__file__).parent / "data" / "ds006598"))

# Where derived outputs (beta maps, etc.) go.
DERIV_ROOT = Path(__file__).parent / "derivatives"

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

# The .1D files hold onset times only; the paradigm's block/trial duration is NOT
# encoded there. CONFIRM this against the task design before trusting GLM betas.
# TODO: set to the true epiproj block/trial length (seconds).
EVENT_DURATION = 4.0

# Contrasts of interest for the MVPA (Step 3).
CONTRASTS = {
    "retrospection": ("pastself", "presentself"),   # past vs present self
    "prospection": ("futureself", "presentself"),   # future vs present self
}


def sub_id(subject: str) -> str:
    """Normalise 'PAN01', 'sub-PAN01', or '01' -> 'PAN01'."""
    s = subject.removeprefix("sub-").removeprefix("PAN")
    return f"PAN{s}"


def bold_path(subject: str, session: int) -> Path:
    """Preprocessed epiproj BOLD for one subject/session."""
    sid = sub_id(subject)
    fname = f"sub-{sid}_ses-{session}_task-{TASK}_space-{SPACE}_desc-preproc_bold.nii.gz"
    return DATA_ROOT / f"sub-{sid}" / f"ses-{session}" / "func" / fname


def afni_timing_path(subject: str, session: int, condition: str) -> Path:
    """AFNI .1D onset file for one subject/session/condition."""
    sid = sub_id(subject)
    fname = f"sub-{sid}_ses-{session}_task-{TASK}_{condition}.1D"
    return DATA_ROOT / "derivatives" / "afni_timing" / sid / fname
