"""Sanity check for Part 1: your metrics must match scikit-learn.

Run from the project root with:  pytest
This test WILL FAIL until you complete src/metrics.py.
"""

import numpy as np
from sklearn import metrics as skm

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import metrics as mine


def test_metrics_match_sklearn():
    rng = np.random.default_rng(42)
    y_true = rng.integers(0, 2, size=200)
    y_pred = rng.integers(0, 2, size=200)

    assert abs(mine.accuracy(y_true, y_pred) - skm.accuracy_score(y_true, y_pred)) < 1e-9
    assert abs(mine.precision(y_true, y_pred) - skm.precision_score(y_true, y_pred, zero_division=0)) < 1e-9
    assert abs(mine.recall(y_true, y_pred) - skm.recall_score(y_true, y_pred, zero_division=0)) < 1e-9
    assert abs(mine.f1_score(y_true, y_pred) - skm.f1_score(y_true, y_pred, zero_division=0)) < 1e-9
