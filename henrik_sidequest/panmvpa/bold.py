"""Find and load preprocessed BOLD files."""
from __future__ import annotations

from pathlib import Path

import nibabel as nib

from . import config


def find_bold(subject: str, session: int, task: str = config.TASK) -> Path:
    """Path to the preproc BOLD. Raises if it isn't present in the clone."""
    path = config.bold_path(subject, session, task)
    if not path.exists():
        raise FileNotFoundError(
            f"BOLD not found: {path}\n"
            f"Clone the dataset and `datalad get` it (see fetch_data.py)."
        )
    return path


def is_fetched(subject: str, session: int, task: str = config.TASK) -> bool:
    """True if the BOLD content is actually downloaded (not just an annex symlink)."""
    return config.bold_path(subject, session, task).is_file()


def load_bold(subject: str, session: int, task: str = config.TASK) -> nib.Nifti1Image:
    """Load the preproc BOLD as a nibabel image (lazy; call .get_fdata() for the array)."""
    return nib.load(str(find_bold(subject, session, task)))
