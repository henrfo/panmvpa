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
DERIV_DIR = REPO_DIR / "derivatives"

# Maps and results are versioned. A design change should write to a NEW folder rather
# than silently overwrite the last run, so old and new can be compared instead of one
# being lost. Bump this (or set PANMVPA_VERSION) whenever the map design changes --
# chunking, atlas, WTA, anything that alters what a map means.
MAP_VERSION = os.environ.get("PANMVPA_VERSION", "v1")

MAPS_DIR = Path(os.environ.get("PANMVPA_MAPS") or DERIV_DIR / "maps" / MAP_VERSION)
RESULTS_DIR = Path(os.environ.get("PANMVPA_RESULTS") or REPO_DIR / "results" / MAP_VERSION)

# The grid is a property of the atlas, not of a map design, so it is shared across
# versions rather than rebuilt per version.
CACHE_DIR = Path(os.environ.get("PANMVPA_CACHE") or DERIV_DIR)


def legacy_maps_dir() -> Path:
    """Where unversioned maps from before this change would sit."""
    return DERIV_DIR / "maps"
DOMAIN_CACHE = CACHE_DIR / "domain.npy"
GROUP_MAP_CACHE = CACHE_DIR / "group_map.npy"

# The fMRIPrep output grid (MNI152NLin6Asym 2mm). Hardcoded so the domain can be rebuilt
# from the atlas alone; every ds006598 preproc BOLD carries exactly this geometry, and
# the stored maps were built against it.
GRID_SHAPE = (91, 109, 91)
GRID_AFFINE = (
    (2.0, 0.0, 0.0, -90.0),
    (0.0, 2.0, 0.0, -126.0),
    (0.0, 0.0, 2.0, -72.0),
    (0.0, 0.0, 0.0, 1.0),
)

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

# Full display names for the Yeo-17 systems -- used everywhere a network is shown to a reader
# (figure labels, axis ticks) so nothing reads as an abbreviation.
NETWORK_LABELS = {
    "VisCent": "Visual central", "VisPeri": "Visual peripheral",
    "SomMotA": "Somatomotor A", "SomMotB": "Somatomotor B",
    "DorsAttnA": "Dorsal attention A", "DorsAttnB": "Dorsal attention B",
    "SalVentAttnA": "Salience / ventral attention A",
    "SalVentAttnB": "Salience / ventral attention B",
    "LimbicA": "Limbic A", "LimbicB": "Limbic B",
    "ContA": "Control A", "ContB": "Control B", "ContC": "Control C",
    "DefaultA": "Default mode A", "DefaultB": "Default mode B", "DefaultC": "Default mode C",
    "TempPar": "Temporal parietal",
}


def network_label(abbr: str) -> str:
    return NETWORK_LABELS.get(abbr, abbr)

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


# --- FC sidequest ----------------------------------------------------------
# A leaner analysis alongside the WTA maps. Reduce each rest run to three arrays and delete
# the 730 MB BOLD: the 400 Schaefer parcel timeseries (raw), the whole-brain mean signal,
# and frame-to-frame change (DVARS). From those few-MB files:
#
#   overlap  -- how fast a subject's parcel-covariance converges to its own stable value
#   identify -- how much rest a linear SVM needs to tell the 10 subjects apart
#
# Parcels are saved raw so cleaning is a cheap analysis-time knob: nilearn.signal.clean
# regresses out the global signal, band-passes and z-scores the 400 timelines in one call.
# (z-scoring is the one step that must come after parcel-averaging, so it never happens
# here.) The whole-brain mean and DVARS both fall out of the brain mask, which is published
# in NLin6Asym -- our exact BOLD space -- so nothing needs resampling.
FC_VERSION = os.environ.get("PANMVPA_FC_VERSION", "v1")
REDUCED_DIR = Path(os.environ.get("PANMVPA_REDUCED") or DERIV_DIR / "reduced" / FC_VERSION)
FC_RESULTS_DIR = Path(os.environ.get("PANMVPA_FC_RESULTS") or REPO_DIR / "results" / "fc" / FC_VERSION)
ASSET_DIR = Path(os.environ.get("PANMVPA_ASSETS") or DERIV_DIR / "assets")
N_PARCELS = 400
BRAIN_MASK_URL = ("https://templateflow.s3.amazonaws.com/tpl-MNI152NLin6Asym/"
                  "tpl-MNI152NLin6Asym_res-02_desc-brain_mask.nii.gz")
