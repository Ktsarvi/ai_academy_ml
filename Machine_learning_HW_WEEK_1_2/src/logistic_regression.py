"""Part 2.2 -- Logistic regression from scratch (gradient descent).

Train by gradient descent on the binary cross-entropy loss -- no sklearn
for the core fit. Standardize your features first (fit on train only).

BONUS (optional): add L1 and/or L2 regularization via `penalty` and
`lam`. Do NOT regularize the bias term.
"""

import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    # TODO: numerically stable 1 / (1 + exp(-z)).
    raise NotImplementedError


class LogisticRegression:
    def __init__(
        self,
        lr: float = 0.1,
        epochs: int = 1000,
        penalty: str | None = None,   # None | "l1" | "l2"  (bonus)
        lam: float = 0.0,             # regularization strength (bonus)
    ) -> None:
        self.lr = lr
        self.epochs = epochs
        self.penalty = penalty
        self.lam = lam
        self.w: np.ndarray | None = None
        self.b: float = 0.0
        self.loss_history: list[float] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegression":
        """Fit w, b with gradient descent on cross-entropy.
        Append the loss each epoch to self.loss_history."""
        # TODO: initialize w, b; loop epochs; compute p = sigmoid(Xw+b),
        # the cross-entropy loss and gradients (add the penalty term for
        # the bonus), and update w and b.
        raise NotImplementedError

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        # TODO: return sigmoid(X @ w + b).
        raise NotImplementedError

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        # TODO: threshold predict_proba at 0.5.
        raise NotImplementedError
