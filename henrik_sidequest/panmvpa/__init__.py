"""panmvpa — how much rest data does a personal brain map need?

Stable = equal-sized maps agree with each other. Useful = the map identifies whose brain
a held-out scan came from.

    import panmvpa
    panmvpa.build_map("PAN01", (0, 1))      # map from the first half of the rest data
    panmvpa.map_dice(map_a, map_b)          # agreement between two maps
"""
from __future__ import annotations

from . import config, figure, identify, parcellation, rest
from .config import LEVELS, SUBJECTS
from .identify import identify as identify_scan
from .identify import network_homogeneity
from .parcellation import (
    build_map,
    dice_per_network,
    load_map,
    map_dice,
    save_map,
)
from .rest import chunk_bounds, rest_runs, task_scans

__all__ = [
    "config",
    "rest",
    "parcellation",
    "identify",
    "figure",
    "SUBJECTS",
    "LEVELS",
    "rest_runs",
    "task_scans",
    "chunk_bounds",
    "build_map",
    "save_map",
    "load_map",
    "map_dice",
    "dice_per_network",
    "network_homogeneity",
    "identify_scan",
]
