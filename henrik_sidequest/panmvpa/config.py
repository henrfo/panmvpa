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

# Association cortex is expected to need more data to individuate than sensory/motor
# cortex. If so, "how long to scan" depends on which system you care about.
SENSORIMOTOR = ["VisCent", "VisPeri", "SomMotA", "SomMotB"]
ASSOCIATION = ["DefaultA", "DefaultB", "DefaultC", "ContA", "ContB", "ContC",
               "SalVentAttnA", "SalVentAttnB", "DorsAttnA", "DorsAttnB",
               "LimbicA", "LimbicB", "TempPar"]
NETWORK_FAMILIES = {"association": ASSOCIATION, "sensorimotor": SENSORIMOTOR}

# --- Data levels -----------------------------------------------------------
# Rest runs are split into 16 equal chunks. A map is identified by (start chunk, number
# of chunks), always a contiguous block, so every map at a given size is disjoint from
# its partner and the amount of data per map is exactly block/16 of the subject's rest.
N_CHUNKS = 16

# Block sizes used for stability. Size b gives 16/b disjoint maps -> 8/b disjoint pairs.
STABILITY_BLOCKS = [1, 2, 4, 8]        # 8, 4, 2, 1 pairs respectively
CUMULATIVE_BLOCKS = [1, 2, 4, 8, 16]   # the growing map used for identification

LEVELS = [f"{b}/{N_CHUNKS}" for b in CUMULATIVE_BLOCKS]  # shared x-axis


def level_name(block: int) -> str:
    return f"{block}/{N_CHUNKS}"


def stability_pairs(block: int) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    """Disjoint map pairs of a given block size, as ((start,size),(start,size))."""
    starts = list(range(0, N_CHUNKS, block))
    return [((starts[i], block), (starts[i + 1], block))
            for i in range(0, len(starts) - 1, 2)]


def map_specs() -> list[tuple[int, int]]:
    """Every (start, size) map a subject needs, deduplicated."""
    specs = {(0, b) for b in CUMULATIVE_BLOCKS}
    for b in STABILITY_BLOCKS:
        specs.update((s, b) for s in range(0, N_CHUNKS, b))
    return sorted(specs)


def map_key(spec: tuple[int, int]) -> str:
    """(0, 2) -> 's00n02' — filename-safe id (two digits, so 1 and 10 never collide)."""
    start, size = spec
    return f"s{start:02d}n{size:02d}"


def sub_id(subject: str) -> str:
    """Normalise 'PAN01', 'sub-PAN01', or '01' -> 'PAN01'."""
    s = subject.removeprefix("sub-").removeprefix("PAN")
    return f"PAN{s}"
