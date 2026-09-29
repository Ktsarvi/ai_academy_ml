"""Part 2.2 -- Logistic regression from scratch (gradient descent).

Train by gradient descent on the binary cross-entropy loss -- no sklearn
for the core fit. Standardize your features first (fit on train only).

BONUS (optional): add L1 and/or L2 regularization via `penalty` and
`lam`. Do NOT regularize the bias term.
"""

import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    return np.where(z >= 0, 1 / (1 + np.exp(-z)), np.exp(z) / (1 + np.exp(z)))


class LogisticRegression:
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

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegression":
        """Fit w, b with gradient descent on cross-entropy.
        Append the loss each epoch to self.loss_history."""
        np.random.seed(42)
        N, p = X.shape
        self.w = np.zeros(p)
        self.b = 0.0
        for _ in range(self.epochs):
            p_hat = sigmoid(X @ self.w + self.b)
            loss = -np.mean(
                y * np.log(p_hat + 1e-15) + (1 - y) * np.log(1 - p_hat + 1e-15)
            )
            self.loss_history.append(loss)
            grad_w = (1 / N) * X.T @ (p_hat - y)
            grad_b = (1 / N) * np.sum(p_hat - y)

            if self.penalty == "l1":
                grad_w += self.lam * np.sign(self.w)
            elif self.penalty == "l2":
                grad_w += 2 * self.lam * self.w
            self.w -= self.lr * grad_w
            self.b -= self.lr * grad_b
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return sigmoid(X @ self.w + self.b)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)
