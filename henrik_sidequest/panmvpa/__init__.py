"""panmvpa — plumbing for the PAN precision-fMRI MVPA sidequest.

    import panmvpa
    sessions = panmvpa.epiproj_sessions("PAN01")
    events = panmvpa.build_events("PAN01", sessions[0])
    img = panmvpa.load_bold("PAN01", sessions[0])
"""
from __future__ import annotations

from . import atlases, bold, config, events
from .bold import find_bold, is_fetched, load_bold
from .config import CONDITIONS, CONTRASTS, SUBJECTS, TASK, TR
from .events import build_events, epiproj_sessions, parse_1d_file

__all__ = [
    "config",
    "events",
    "bold",
    "atlases",
    "SUBJECTS",
    "TASK",
    "TR",
    "CONDITIONS",
    "CONTRASTS",
    "parse_1d_file",
    "build_events",
    "epiproj_sessions",
    "find_bold",
    "load_bold",
    "is_fetched",
]
