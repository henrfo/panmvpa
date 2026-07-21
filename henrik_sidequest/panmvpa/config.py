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

# Dataset location. Set DATA_DIR (or PANMVPA_DATA) to your clone of ds006598 so the same
# code runs locally and on a JupyterHub without editing anything. Falls back to the
# in-repo clone used during local development.
DATA_ROOT = Path(
    os.environ.get("DATA_DIR")
    or os.environ.get("PANMVPA_DATA")
    or REPO_DIR / "data" / "ds006598"
)

ATLAS_DIR = REPO_DIR / "atlases"       # downloaded template atlases live here
DERIV_ROOT = REPO_DIR / "derivatives"  # scratch/derived outputs (gitignored)

# Figures + CSVs that we DO want in git, so hub results can be pulled back locally.
RESULTS_DIR = Path(os.environ.get("PANMVPA_RESULTS") or REPO_DIR.parent / "results")

# --- Subjects --------------------------------------------------------------
SUBJECTS = [f"PAN{n:02d}" for n in range(1, 11)]  # PAN01 .. PAN10

# --- Acquisition -----------------------------------------------------------
TR = 1.355  # seconds, from *_bold.json RepetitionTime
SPACE = "MNI152NLin6Asym_res-2"

# --- Resting state (parcellation input) -----------------------------------
REST_TASK = "rest"
# Each PAN rest run is 222 volumes ~= 5.01 min. The reliability/decoding x-axis
# is minutes of rest; we concatenate whole runs up to each target.
# Capped at 100 min so every subject contributes at every level: PAN03 and PAN05 have
# only ~105 min of rest and PAN07 ~115, so a 120 min level would silently drop them.
MINUTE_LEVELS = [20, 40, 60, 80, 100]

# --- Group reference parcellation -----------------------------------------
# The Yeo-Krienen 17-network taxonomy, realised via Schaefer-400 (its parcels ARE
# labelled by the 17 networks) in FSL-MNI152 2mm. This is the FIXED group anchor:
# it defines the 17 seed regions used to derive reference timeseries at every data
# level. Only voxel allegiance is individualised, never these region definitions.
SCHAEFER_ATLAS = ATLAS_DIR / "schaefer_2018" / (
    "Schaefer2018_400Parcels_17Networks_order_FSLMNI152_2mm.nii.gz"
)
SCHAEFER_ORDER = ATLAS_DIR / "schaefer_2018" / "Schaefer2018_400Parcels_17Networks_order.txt"
N_NETWORKS = 17
# DN-A := Yeo-17 DefaultC (retrosplenial / parahippocampal / dorsal PCC).
DN_A_NETWORK = "DefaultC"

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

# --- Task-decoding (Plot 2) -----------------------------------------------
# 4-class problem: which task is being performed, decoded from the DN-A pattern.
# Each class beta = mean of its sub-task betas that are present in a session
# (sub-tasks are averaged when both exist, otherwise the one present is used).
# Block durations (s) are from the paper's STAR Methods.
TASK_DURATIONS = {
    "langlocaud": 18.0,
    "langlocvis": 18.0,
    "tomfalse": 15.0,   # 10 s story + 5 s response
    "tompain": 15.0,
    "epiproj": 10.0,
    "msit": 42.0,
    "spatialwm": 34.0,
}

# Class label = insertion order (language=0, tom=1, epiproj=2, control=3).
DECODING_FAMILIES = {
    "language": ["langlocaud", "langlocvis"],
    "tom": ["tomfalse", "tompain"],
    "epiproj": ["epiproj"],
    "control": ["msit"],
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


def rest_glob(subject: str) -> str:
    """Glob (relative to DATA_ROOT) matching every preproc rest BOLD run for a subject."""
    sid = sub_id(subject)
    return (
        f"sub-{sid}/ses-*/func/"
        f"sub-{sid}_ses-*_task-{REST_TASK}_run-*_space-{SPACE}_desc-preproc_bold.nii.gz"
    )


def afni_timing_path(subject: str, session: int, condition: str, task: str = TASK) -> Path:
    """AFNI .1D onset file for one subject/session/condition."""
    sid = sub_id(subject)
    fname = f"sub-{sid}_ses-{session}_task-{task}_{condition}.1D"
    return DATA_ROOT / "derivatives" / "afni_timing" / sid / fname


def task_timing_files(subject: str, session: int, task: str) -> list[Path]:
    """All AFNI .1D condition files for one subject/session/task (any condition)."""
    sid = sub_id(subject)
    timing_dir = DATA_ROOT / "derivatives" / "afni_timing" / sid
    return sorted(timing_dir.glob(f"sub-{sid}_ses-{session}_task-{task}_*.1D"))
