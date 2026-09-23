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
        # TODO: store X and y as float / int arrays.
        raise NotImplementedError

    def _distances(self, X: np.ndarray) -> np.ndarray:
        """Return the (n_queries, n_train) matrix of Euclidean distances.

        Hint: ||x - z||^2 = ||x||^2 - 2 x.z + ||z||^2 lets you build the
        whole matrix with matrix multiplication and broadcasting.
        """
        # TODO: implement the vectorized pairwise distance.
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict a label for each row of X by majority vote of the k
        nearest training points."""
        # TODO: use self._distances, np.argpartition/np.argsort, and a
        # majority vote over the k nearest labels.
        raise NotImplementedError
