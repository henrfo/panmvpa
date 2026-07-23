"""_score must work for any class count. sklearn's decision_function returns one signed
score per sample in the BINARY case (shape (n,)) and a column-per-class otherwise (n, K) --
a shape difference that only surfaces on real data at the sparse high-X rungs, at the end of
a long run. This guards both: accuracy and a margin that means the same thing (true-class
score minus the runner-up's) in the two-class and multi-class cases.
"""
import importlib.util as u
from pathlib import Path

import numpy as np

_spec = u.spec_from_file_location("run_fc", Path(__file__).resolve().parent.parent
                                  / "scripts" / "run_fc.py")
run_fc = u.module_from_spec(_spec)
_spec.loader.exec_module(run_fc)


def _synthetic(n_classes, per_class=3, dim=8, sep=6.0, seed=0):
    """Well-separated classes: `per_class` examples each, one group per example so
    leave-one-group-out always leaves >=2 of every class in training."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(n_classes, dim)) * sep
    feats, y, groups, g = [], [], [], 0
    for c in range(n_classes):
        for _ in range(per_class):
            feats.append(centers[c] + rng.normal(size=dim))
            y.append(f"S{c}"); groups.append(g); g += 1
    return feats, y, groups


def test_score_binary_and_multiclass_match_in_meaning():
    for k in (2, 3, 5):
        acc, margin = run_fc._score(*_synthetic(k))
        assert 0.0 <= acc <= 1.0, f"k={k}: accuracy out of range ({acc})"
        assert acc >= 0.8, f"k={k}: separable data should classify well, got {acc}"
        # margin is the point of this arm: positive means the true subject beats the
        # runner-up on average, and it must be finite in every class count (incl. binary).
        assert np.isfinite(margin), f"k={k}: margin not finite ({margin})"
        assert margin > 0, f"k={k}: separable data should give a positive margin ({margin})"
