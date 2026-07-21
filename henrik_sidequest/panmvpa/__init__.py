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
    cnr,
    identify,
    mapstore,
    config,
    events,
    figure,
    glm,
    mvpa,
    parcellation,
    reliability,
    rest,
)
from .cnr import cnr_curve
from .bold import find_bold, is_fetched, load_bold
from .config import CONDITIONS, CONTRASTS, MINUTE_LEVELS, SUBJECTS, TASK, TR
from .events import build_events, epiproj_sessions, parse_1d_file
from .mvpa import complete_sessions, decoding_curve
from .parcellation import build_parcellation, dn_a_mask
from .reliability import reliability_curve
from .rest import rest_runs, select_runs


def clear_caches() -> None:
    """Free every per-subject cache (rest timeseries, GLM betas/z-maps).

    ``run_all.py`` calls this between subjects so peak memory tracks one subject rather
    than the whole cohort -- the 15 GB hub cannot hold 10 subjects' timeseries at once.
    """
    import gc

    rest.clear_cache()
    glm.task_beta.cache_clear()
    glm.task_zmap.cache_clear()
    cnr.epiproj_zmaps.cache_clear()
    mvpa.decoding_betas.cache_clear()
    gc.collect()  # cache_clear only drops references; collect actually reclaims

__all__ = [
    "config",
    "events",
    "bold",
    "atlases",
    "rest",
    "parcellation",
    "reliability",
    "glm",
    "cnr",
    "identify",
    "mapstore",
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
    "cnr_curve",
    "decoding_curve",
    "complete_sessions",
    "clear_caches",
]
