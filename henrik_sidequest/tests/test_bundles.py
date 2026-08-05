"""The SVM's examples must be session-disjoint. If two examples ever share a session, one
can land in training and the other in test, and the classifier has already seen that data --
fingerprinting then looks better than it is, with no error to warn you, and the inflation
grows exactly as X grows (more sessions per bundle). Cheap insurance against a silent regress.
"""
import importlib.util as u
from pathlib import Path

import numpy as np

_spec = u.spec_from_file_location("run_fc", Path(__file__).resolve().parent.parent
                                  / "scripts" / "run_fc.py")
run_fc = u.module_from_spec(_spec)
_spec.loader.exec_module(run_fc)


def test_bundles_are_session_disjoint(monkeypatch):
    # Stub cleaning: each session becomes ~20 min of a constant equal to its id, so the
    # sessions inside a bundle are recoverable from the values.
    frames = int(round(20 * 60 / run_fc.TR))
    monkeypatch.setattr(run_fc, "_session_ts",
                        lambda runs, gsr: np.full((frames, 400), float(runs[0]), np.float32))
    by_ses = {s: [s] for s in range(1, 10)}  # 9 sessions, ids 1..9

    for X in (5, 10, 20, 40, 60, 80):
        seen: set[int] = set()
        for bundle in run_fc._bundles(by_ses, X, gsr=True):
            ids = set(np.unique(bundle).astype(int).tolist())
            assert not (seen & ids), f"session reused across bundles at X={X}: {seen & ids}"
            seen |= ids
