"""Paths, subjects, and the deterministic data-level design.

How much resting-state data do you need before a personal brain map is stable (gives the
same answer every time) and useful (identifies whose brain it is)?

Data levels are deterministic -- no random seeds. Each subject's rest runs are taken in
order and split into four equal quarters, and we build eight maps:

    Q1, Q2, Q3, Q4          four independent maps from a quarter of the data each
    Q1+Q2, Q3+Q4            two independent maps from half the data each
    Q1+Q2+Q3                three quarters
    Q1+Q2+Q3+Q4             everything

VARIANCE (plot 1) compares maps built from the *same* amount of data: the four quarter
maps (6 pairs) and the two half maps (1 pair). Three-quarter and full are single maps, so
they have no pair and no variance point.

SIGNAL (plot 2) uses the *cumulative* maps -- Q1, Q1+Q2, Q1+Q2+Q3, all -- at all four
levels.
"""
from __future__ import annotations

import os
from pathlib import Path

# --- Locations -------------------------------------------------------------
PKG_DIR = Path(__file__).resolve().parent
REPO_DIR = PKG_DIR.parent  # henrik_sidequest/

# Set DATA_DIR to the ds006598 root so the same code runs locally and on a hub.
DATA_ROOT = Path(
    os.environ.get("DATA_DIR") or os.environ.get("PANMVPA_DATA")
    or REPO_DIR / "data" / "ds006598"
)
ATLAS_DIR = REPO_DIR / "atlases"
MAPS_DIR = Path(os.environ.get("PANMVPA_MAPS") or REPO_DIR / "derivatives" / "maps")
RESULTS_DIR = Path(os.environ.get("PANMVPA_RESULTS") or REPO_DIR.parent / "results")

# --- Dataset ---------------------------------------------------------------
SUBJECTS = [f"PAN{n:02d}" for n in range(1, 11)]
TR = 1.355                       # seconds, from the BOLD sidecars
SPACE = "MNI152NLin6Asym_res-2"  # preprocessed fMRIPrep output space
REST_TASK = "rest"

# --- Group atlas -----------------------------------------------------------
# The Yeo-17 network taxonomy, realised via Schaefer-400 (its parcels are labelled by the
# Yeo-Krienen 17 networks) because that ships already in FSL-MNI152 2mm -- the same space
# family as the BOLD -- so it only needs a nearest-neighbour regrid, not a cross-space
# resample. Collapsing its 400 parcels by network label gives the 17 group regions.
ATLAS_IMAGE = ATLAS_DIR / "schaefer_2018" / (
    "Schaefer2018_400Parcels_17Networks_order_FSLMNI152_2mm.nii.gz"
)
ATLAS_ORDER = ATLAS_DIR / "schaefer_2018" / "Schaefer2018_400Parcels_17Networks_order.txt"
N_NETWORKS = 17

# --- Data levels -----------------------------------------------------------
N_QUARTERS = 4

# Maps built per subject, keyed by the quarters they use (0-indexed).
MAP_KEYS: list[tuple[int, ...]] = [
    (0,), (1,), (2,), (3,),      # quarter-sized
    (0, 1), (2, 3),              # half-sized
    (0, 1, 2),                   # three quarters
    (0, 1, 2, 3),                # everything
]

# Plot 1: groups of equal-sized maps to compare against each other.
VARIANCE_GROUPS: dict[str, list[tuple[int, ...]]] = {
    "1/4": [(0,), (1,), (2,), (3,)],
    "2/4": [(0, 1), (2, 3)],
}

# Plot 2: the cumulative map at each level.
CUMULATIVE: dict[str, tuple[int, ...]] = {
    "1/4": (0,),
    "2/4": (0, 1),
    "3/4": (0, 1, 2),
    "4/4": (0, 1, 2, 3),
}

LEVELS = ["1/4", "2/4", "3/4", "4/4"]  # shared x-axis


def map_key(quarters: tuple[int, ...]) -> str:
    """(0,1) -> 'q01'  — filename-safe id for a map."""
    return "q" + "".join(str(q) for q in quarters)


def sub_id(subject: str) -> str:
    """Normalise 'PAN01', 'sub-PAN01', or '01' -> 'PAN01'."""
    s = subject.removeprefix("sub-").removeprefix("PAN")
    return f"PAN{s}"
