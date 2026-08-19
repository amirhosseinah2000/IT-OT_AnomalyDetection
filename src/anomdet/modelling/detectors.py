"""Small detector adapters with one consistent anomaly-score contract."""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA


class PCAAutoencoder:
    """A lightweight linear autoencoder using PCA reconstruction error as anomaly score."""

    def __init__(self, explained_variance: float = 0.95) -> None:
        self.explained_variance = explained_variance
        self.model = PCA(n_components=explained_variance, svd_solver="full")

    def fit(self, values: np.ndarray) -> "PCAAutoencoder":
        """Fit the compact reconstruction basis on presumed normal training data."""
        self.model.fit(values)
        return self

    def score_samples(self, values: np.ndarray) -> np.ndarray:
        """Return reconstruction MSE; larger values consistently mean more anomalous."""
        reconstructed = self.model.inverse_transform(self.model.transform(values))
        return np.mean(np.square(values - reconstructed), axis=1)
