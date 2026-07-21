"""First-level GLM for any task -> one task-vs-baseline beta map per session.

For a task-identity decoding problem we don't care which condition within a task was
run, only the task's activation pattern. So every block of the task (all conditions
pooled) is modelled as a single regressor with the task's block duration, and the
task-vs-baseline effect map is the per-session beta.

Only a high-pass cosine drift model is used (no confounds are deposited in this dataset).
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd
import nibabel as nib
from nilearn.glm.first_level import FirstLevelModel

from . import config, events, parcellation


def task_events(subject: str, session: int, task: str) -> pd.DataFrame:
    """Single-regressor events (onset, duration, trial_type=task) pooling all conditions."""
    duration = config.TASK_DURATIONS[task]
    onsets: list[float] = []
    for f in config.task_timing_files(subject, session, task):
        onsets.extend(events.parse_1d_file(f).tolist())
    if not onsets:
        raise ValueError(f"no timing onsets for {config.sub_id(subject)} ses-{session} {task}")
    onsets.sort()
    return pd.DataFrame(
        {"onset": onsets, "duration": duration, "trial_type": [task] * len(onsets)}
    )


def _bold_present(subject: str, session: int, task: str) -> bool:
    """True if this task's preproc BOLD content is on disk for the session."""
    p = config.bold_path(subject, session, task)
    try:
        return p.resolve(strict=True).is_file() and p.stat().st_size > 0
    except (FileNotFoundError, OSError):
        return False


def subtask_present(subject: str, session: int, task: str) -> bool:
    """True if both the BOLD and at least one timing file exist for a sub-task."""
    return _bold_present(subject, session, task) and bool(
        config.task_timing_files(subject, session, task)
    )


def _fit(subject: str, session: int, task: str) -> FirstLevelModel:
    img = nib.load(str(config.bold_path(subject, session, task)))
    ev = task_events(subject, session, task)
    model = FirstLevelModel(
        t_r=config.TR,
        slice_time_ref=0.0,           # slice timing NOT corrected in this dataset
        hrf_model="glover",
        drift_model="cosine",
        high_pass=0.01,
        mask_img=parcellation.domain_mask_img(),
        minimize_memory=True,
        standardize=False,
    )
    model.fit(img, events=ev)
    return model


def _sample_domain(img) -> np.ndarray:
    arr = np.asarray(img.get_fdata(), dtype=np.float32)
    idx = parcellation.analysis_domain()
    return arr[idx[0], idx[1], idx[2]]


@lru_cache(maxsize=256)
def task_beta(subject: str, session: int, task: str) -> np.ndarray:
    """Task-vs-baseline effect-size beta over the analysis domain (cached)."""
    model = _fit(subject, session, task)
    return _sample_domain(model.compute_contrast(task, output_type="effect_size"))


@lru_cache(maxsize=256)
def task_zmap(subject: str, session: int, task: str) -> np.ndarray:
    """Task-vs-baseline z-statistic over the analysis domain (cached).

    Z rather than beta so the contrast-to-noise metric is on a unitless, noise-normalised
    scale that is comparable across sessions and subjects.
    """
    model = _fit(subject, session, task)
    return _sample_domain(model.compute_contrast(task, output_type="z_score"))
