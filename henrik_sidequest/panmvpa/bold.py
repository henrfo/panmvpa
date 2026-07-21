"""Find and load preprocessed BOLD files."""
from __future__ import annotations

from pathlib import Path

import nibabel as nib

from . import config  # noqa: F401  (Path used in annotations below)


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


def task_scans(subject: str) -> list[Path]:
    """Every non-rest preproc BOLD present on disk for a subject.

    These are the held-out scans for subject identification: task runs are never used to
    build maps at any data level, so this test set stays identical across the whole
    x-axis.
    """
    sid = config.sub_id(subject)
    pattern = f"sub-{sid}/ses-*/func/sub-{sid}_ses-*_task-*_space-{config.SPACE}_desc-preproc_bold.nii.gz"
    out = []
    for p in sorted(config.DATA_ROOT.glob(pattern)):
        if "_task-rest_" in p.name:
            continue
        try:
            if p.resolve(strict=True).is_file() and p.stat().st_size > 0:
                out.append(p)
        except (FileNotFoundError, OSError):
            continue
    return out


def scan_task_name(path: Path) -> str:
    """'...task-msit_space-...' -> 'msit'."""
    for part in path.name.split("_"):
        if part.startswith("task-"):
            return part[len("task-"):]
    return "unknown"
