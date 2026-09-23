"""Part 2.1 -- Linear regression from scratch (batch gradient descent).

Train by gradient descent only -- no normal equation, no sklearn for the
core fit. Standardize your features first (fit the scaler on train only).

BONUS (optional): add L1 and/or L2 regularization via `penalty` and
`lam`. Do NOT regularize the bias term.
"""

import numpy as np


class LinearRegression:
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

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegression":
        """Fit w, b with batch gradient descent on the MSE objective.
        Append the loss each epoch to self.loss_history."""
        # TODO: initialize w, b; loop epochs; compute predictions,
        # gradients (add the penalty term for the bonus), update w and b.
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        # TODO: return X @ w + b.
        raise NotImplementedError
