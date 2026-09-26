"""Part 1.2 -- Classification metrics from scratch.

Implement each metric from the confusion-matrix counts (TP, FP, TN, FN).
Do NOT call sklearn.metrics here. Return 0.0 for any zero-denominator
edge case. `positive_label` is the class treated as "positive".
"""

import numpy as np


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred))


def precision(y_true: np.ndarray, y_pred: np.ndarray, positive_label: int = 1) -> float:
    tp = np.sum((y_pred == positive_label) & (y_true == positive_label))
    fp = np.sum((y_pred == positive_label) & (y_true != positive_label))

    if tp + fp == 0:
        return 0.0
    return tp / (tp + fp)


def recall(y_true: np.ndarray, y_pred: np.ndarray, positive_label: int = 1) -> float:
    tp = np.sum((y_pred == positive_label) & (y_true == positive_label))
    fn = np.sum((y_pred != positive_label) & (y_true == positive_label))

    if tp + fn == 0:
        return 0.0
    return tp / (tp + fn)


def f1_score(y_true: np.ndarray, y_pred: np.ndarray, positive_label: int = 1) -> float:
    tp = np.sum((y_pred == positive_label) & (y_true == positive_label))
    fp = np.sum((y_pred == positive_label) & (y_true != positive_label))
    fn = np.sum((y_pred != positive_label) & (y_true == positive_label))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)
