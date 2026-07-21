"""panmvpa — plumbing for the PAN precision-fMRI MVPA sidequest.

    import panmvpa
    sessions = panmvpa.epiproj_sessions("PAN01")
    events = panmvpa.build_events("PAN01", sessions[0])
    img = panmvpa.load_bold("PAN01", sessions[0])
"""
from __future__ import annotations

from . import (
    atlases,
    bold,
    config,
    events,
    figure,
    glm,
    mvpa,
    parcellation,
    reliability,
    rest,
)
from .bold import find_bold, is_fetched, load_bold
from .config import CONDITIONS, CONTRASTS, MINUTE_LEVELS, SUBJECTS, TASK, TR
from .events import build_events, epiproj_sessions, parse_1d_file
from .mvpa import complete_sessions, decoding_curve
from .parcellation import build_parcellation, dn_a_mask
from .reliability import reliability_curve
from .rest import rest_runs, select_runs

__all__ = [
    "config",
    "events",
    "bold",
    "atlases",
    "rest",
    "parcellation",
    "reliability",
    "glm",
    "mvpa",
    "figure",
    "SUBJECTS",
    "TASK",
    "TR",
    "CONDITIONS",
    "CONTRASTS",
    "MINUTE_LEVELS",
    "parse_1d_file",
    "build_events",
    "epiproj_sessions",
    "find_bold",
    "load_bold",
    "is_fetched",
    "rest_runs",
    "select_runs",
    "build_parcellation",
    "dn_a_mask",
    "reliability_curve",
    "decoding_curve",
    "complete_sessions",
]
