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
        penalty: str | None = None,  # None | "l1" | "l2"  (bonus)
        lam: float = 0.0,  # regularization strength (bonus)
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
        np.random.seed(42)
        N, p = X.shape
        self.w = np.zeros(p)
        self.b = 0.0
        for _ in range(self.epochs):
            y_hat = X @ self.w + self.b
            loss = np.mean((y_hat - y) ** 2)
            self.loss_history.append(loss)
            grad_w = (2 / N) * X.T @ (y_hat - y)
            grad_b = (2 / N) * np.sum(y_hat - y)

            if self.penalty == "l1":
                grad_w += self.lam * np.sign(self.w)
            elif self.penalty == "l2":
                grad_w += 2 * self.lam * self.w
            self.w -= self.lr * grad_w
            self.b -= self.lr * grad_b
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self.w + self.b
