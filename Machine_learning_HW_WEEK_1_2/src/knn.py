"""Part 1.1 -- k-Nearest Neighbors from scratch.

Fill in the methods below. Use vectorized NumPy for the distance
computation: NO Python `for` loop over the training points.
"""

import numpy as np


class KNN:
    def __init__(self, k: int = 5) -> None:
        self.k = k
        self.X_train: np.ndarray | None = None
        self.y_train: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KNN":
        """Memorize the training data. Returns self."""
        self.X_train = X.astype(float)
        self.y_train = y.astype(int)
        return self

    def _distances(self, X: np.ndarray) -> np.ndarray:
        """Return the (n_queries, n_train) matrix of Euclidean distances.

        Hint: ||x - z||^2 = ||x||^2 - 2 x.z + ||z||^2 lets you build the
        whole matrix with matrix multiplication and broadcasting.
        """
        sq_X = np.sum(X**2, axis=1, keepdims=True)
        sq_XT = np.sum(self.X_train**2, axis=1, keepdims=True).T
        cross_prods = X @ self.X_train.T
        dist = sq_X - 2 * cross_prods + sq_XT
        return np.sqrt(np.maximum(dist, 0))

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict a label for each row of X by majority vote of the k
        nearest training points."""
        D = self._distances(X)
        k_indices = np.argpartition(D, self.k, axis=1)[:, : self.k]
        k_labels = self.y_train[k_indices]
        return np.array([np.bincount(row).argmax() for row in k_labels])
