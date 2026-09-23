"""Part 1.2 -- Classification metrics from scratch.

Implement each metric from the confusion-matrix counts (TP, FP, TN, FN).
Do NOT call sklearn.metrics here. Return 0.0 for any zero-denominator
edge case. `positive_label` is the class treated as "positive".
"""

import numpy as np


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # TODO: fraction of correct predictions.
    raise NotImplementedError


def precision(y_true: np.ndarray, y_pred: np.ndarray, positive_label: int = 1) -> float:
    # TODO: TP / (TP + FP), or 0.0 if the denominator is 0.
    raise NotImplementedError


def recall(y_true: np.ndarray, y_pred: np.ndarray, positive_label: int = 1) -> float:
    # TODO: TP / (TP + FN), or 0.0 if the denominator is 0.
    raise NotImplementedError


def f1_score(y_true: np.ndarray, y_pred: np.ndarray, positive_label: int = 1) -> float:
    # TODO: harmonic mean of precision and recall, or 0.0 if both are 0.
    raise NotImplementedError
