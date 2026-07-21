"""Plot 2 — 4-class task decoding vs amount of rest data.

Classes: language / theory-of-mind / episodic-projection / cognitive-control (chance 25%).
Each class beta for a session is the mean of its sub-task task-vs-baseline betas that are
present that session (sub-tasks averaged when both exist, else the one present is used).
The DN-A (DefaultC) mask from the individualised parcellation selects the features; a
linear SVM classifies with leave-one-session-out CV over sessions that contain all 4
classes.

Betas don't depend on rest data, so they're computed once and cached; only the DN-A mask
changes with the amount of rest, which is what moves the decoding curve.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from sklearn.model_selection import LeaveOneGroupOut, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from . import config, events, glm, parcellation

# Class label = order in config.DECODING_FAMILIES.
FAMILIES = list(config.DECODING_FAMILIES)  # ["language","tom","epiproj","control"]


def present_subtasks(subject: str, session: int, family: str) -> list[str]:
    """Sub-tasks of a family whose BOLD+timing are present for this session."""
    return [
        t
        for t in config.DECODING_FAMILIES[family]
        if glm.subtask_present(subject, session, t)
    ]


def family_available(subject: str, session: int, family: str) -> bool:
    return len(present_subtasks(subject, session, family)) > 0


def complete_sessions(subject: str) -> list[int]:
    """Sessions in which all 4 classes have at least one present sub-task."""
    # Candidate sessions = any session with an epiproj timing entry, unioned with the
    # sessions the other families appear in; simplest is to scan the subject's sessions.
    candidate = set()
    for fam in FAMILIES:
        for t in config.DECODING_FAMILIES[fam]:
            for f in _task_sessions(subject, t):
                candidate.add(f)
    return sorted(
        s for s in candidate if all(family_available(subject, s, fam) for fam in FAMILIES)
    )


def _task_sessions(subject: str, task: str) -> list[int]:
    sid = config.sub_id(subject)
    root = config.DATA_ROOT / "derivatives" / "afni_timing" / sid
    out = set()
    for f in root.glob(f"sub-{sid}_ses-*_task-{task}_*.1D"):
        out.add(int(f.name.split("_ses-")[1].split("_")[0]))
    return sorted(out)


def family_beta(subject: str, session: int, family: str) -> np.ndarray:
    """Mean of present sub-task betas for a class in one session (over the domain)."""
    subtasks = present_subtasks(subject, session, family)
    betas = [glm.task_beta(subject, session, t) for t in subtasks]
    return np.mean(betas, axis=0)


@lru_cache(maxsize=8)
def decoding_betas(subject: str) -> dict:
    """{session: {family: beta_vector}} for every complete session (cached)."""
    out: dict[int, dict[str, np.ndarray]] = {}
    for ses in complete_sessions(subject):
        out[ses] = {fam: family_beta(subject, ses, fam) for fam in FAMILIES}
    return out


def _design_matrix(betas: dict, dn_mask: np.ndarray):
    X, y, groups = [], [], []
    for ses, fams in sorted(betas.items()):
        for label, fam in enumerate(FAMILIES):
            X.append(fams[fam][dn_mask])
            y.append(label)
            groups.append(ses)
    return np.asarray(X), np.asarray(y), np.asarray(groups)


def decoding_at(subject: str, minutes: float, betas: dict | None = None) -> dict:
    """Leave-one-session-out 4-class accuracy using the DN-A mask built at ``minutes``."""
    betas = betas if betas is not None else decoding_betas(subject)
    labels = parcellation.build_parcellation(subject, minutes=minutes)
    dn_mask = parcellation.dn_a_mask(labels)
    X, y, groups = _design_matrix(betas, dn_mask)
    clf = make_pipeline(StandardScaler(), LinearSVC(C=1.0, dual="auto"))
    scores = cross_val_score(clf, X, y, groups=groups, cv=LeaveOneGroupOut())
    return {
        "minutes": minutes,
        "accuracy": float(scores.mean()),
        "n_features": int(dn_mask.sum()),
        "n_folds": len(scores),
        "n_samples": len(y),
        "chance": 1.0 / len(FAMILIES),
    }


def decoding_curve(
    subject: str, minute_levels: list[float] | None = None
) -> list[dict]:
    """4-class decoding accuracy across the standard minute levels (betas fit once)."""
    levels = minute_levels if minute_levels is not None else config.MINUTE_LEVELS
    betas = decoding_betas(subject)
    return [decoding_at(subject, m, betas=betas) for m in levels]
