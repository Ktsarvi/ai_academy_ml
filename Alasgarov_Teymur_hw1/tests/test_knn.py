"""Sanity check for Part 1: your KNN must agree with scikit-learn.

Run from the project root with:  pytest
This test WILL FAIL until you complete src/knn.py.
"""

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from knn import KNN


def _data():
    X, y = load_breast_cancer(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    scaler = StandardScaler().fit(Xtr)
    return scaler.transform(Xtr), scaler.transform(Xte), ytr, yte


def test_matches_sklearn():
    Xtr, Xte, ytr, yte = _data()
    k = 5
    mine = KNN(k=k).fit(Xtr, ytr).predict(Xte)
    ref = KNeighborsClassifier(n_neighbors=k).fit(Xtr, ytr).predict(Xte)
    assert np.array_equal(np.asarray(mine), np.asarray(ref))
