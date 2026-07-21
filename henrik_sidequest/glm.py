"""Step 1 — first-level GLM for the Episodic Projection task.

One epiproj run per session, so we fit one FirstLevelModel per session and write one
beta (effect-size) map per condition per session. Those per-session maps are the samples
the Step-3 MVPA cross-validates across.

No fMRIPrep confounds are deposited with ds006598, so drift is handled by the model's
cosine high-pass only. Add motion regressors here if you later compute them.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from nibabel import load as nib_load
from nilearn.glm.first_level import FirstLevelModel

import config


def read_1d_onsets(path) -> np.ndarray:
    """AFNI .1D: whitespace-separated onset times (seconds), one run per row.

    ds006598 epiproj has a single run per session -> a single row. '*' marks an empty
    run in AFNI; we treat any non-numeric token as 'no trials'.
    """
    text = path.read_text().strip()
    if not text:
        return np.array([])
    onsets = []
    for tok in text.split():
        try:
            onsets.append(float(tok))
        except ValueError:
            continue  # AFNI '*' placeholder etc.
    return np.array(sorted(onsets))


def build_events(subject: str, session: int) -> pd.DataFrame:
    """Assemble a nilearn events frame (onset, duration, trial_type) for one session."""
    rows = []
    for cond in config.CONDITIONS:
        path = config.afni_timing_path(subject, session, cond)
        if not path.exists():
            raise FileNotFoundError(f"missing timing file: {path}")
        for onset in read_1d_onsets(path):
            rows.append({"onset": onset, "duration": config.EVENT_DURATION, "trial_type": cond})
    if not rows:
        raise ValueError(f"no events found for {subject} ses-{session}")
    return pd.DataFrame(rows).sort_values("onset").reset_index(drop=True)


def fit_session(subject: str, session: int) -> tuple[FirstLevelModel, pd.DataFrame]:
    """Fit the GLM for one epiproj session."""
    bold = config.bold_path(subject, session)
    if not bold.exists():
        raise FileNotFoundError(f"missing BOLD (datalad get it first): {bold}")
    events = build_events(subject, session)

    model = FirstLevelModel(
        t_r=config.TR,
        hrf_model="spm",
        drift_model="cosine",
        high_pass=1 / 128,
        smoothing_fwhm=None,          # keep patterns intact for MVPA
        signal_scaling=(0, 1),
        minimize_memory=True,
    )
    model.fit(str(bold), events=events)
    return model, events


def save_condition_betas(subject: str, session: int, model: FirstLevelModel) -> list:
    """Write one effect-size map per condition. Returns the output paths."""
    sid = config.sub_id(subject)
    outdir = config.DERIV_ROOT / "first_level" / f"sub-{sid}" / f"ses-{session}"
    outdir.mkdir(parents=True, exist_ok=True)

    written = []
    for cond in config.CONDITIONS:
        zmap = model.compute_contrast(cond, output_type="effect_size")
        out = outdir / f"sub-{sid}_ses-{session}_task-{config.TASK}_cond-{cond}_beta.nii.gz"
        zmap.to_filename(str(out))
        written.append(out)
    return written


def run_subject(subject: str, sessions: list[int]) -> None:
    for ses in sessions:
        print(f"[{config.sub_id(subject)} ses-{ses}] fitting GLM...")
        model, events = fit_session(subject, ses)
        print(f"  {len(events)} events across {events.trial_type.nunique()} conditions")
        paths = save_condition_betas(subject, ses, model)
        print(f"  wrote {len(paths)} beta maps -> {paths[0].parent}")
