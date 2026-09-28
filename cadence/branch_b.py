"""
Branch B: Distributional Interval Expert
Part of the CADENCE Classifier Suite.

Combines three complementary representations (total D_B = 1,851 features):
1. Competing dilated kernels (Hydra, 16 groups x 8 kernels = 128 features).
2. Thread-safe dyadic Cornish-Fisher intervals across raw signal, velocity,
   and acceleration (1,701 features).
3. Real FFT spectral energy bands, centroid, spread, and power quantiles (22 features).
Fitted with an ExtraTrees ensemble (100 trees, 10% feature subsampling, entropy criterion).
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import ExtraTreesClassifier
from aeon.transformations.collection.convolution_based import HydraTransformer

from .moment_quant import extract_moment_quant_features
from .spectral import extract_spectral_features


class DistributionalIntervalExpert(BaseEstimator, ClassifierMixin):
    """
    Distributional Interval Expert (Branch B).

    Parameters
    ----------
    hydra_groups : int, default=16
        Number of competing kernel groups in Hydra.
    quant_depth : int, default=5
        Maximum depth for dyadic interval Cornish-Fisher moment extraction.
    n_fft_bands : int, default=16
        Number of log-spaced frequency energy bands.
    n_estimators : int, default=100
        Number of decision trees in ExtraTrees ensemble.
    max_features : float, default=0.1
        Fraction of features randomly evaluated at each tree split.
    criterion : str, default="entropy"
        Splitting impurity criterion (Shannon entropy).
    n_jobs : int, default=4
        Number of parallel CPU worker threads.
    random_state : int, default=42
        Seed for reproducible tree splits and Hydra kernel sampling.
    """

    def __init__(
        self,
        hydra_groups=16,
        quant_depth=5,
        n_fft_bands=16,
        n_estimators=100,
        max_features=0.1,
        criterion="entropy",
        n_jobs=4,
        random_state=42
    ):
        self.hydra_groups = hydra_groups
        self.quant_depth = quant_depth
        self.n_fft_bands = n_fft_bands
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.criterion = criterion
        self.n_jobs = n_jobs
        self.random_state = random_state

    def fit_transform_features(self, X):
        """Fit Hydra transformer and extract combined interval and spectral features."""
        self.hydra_ = HydraTransformer(
            n_kernels=8,
            n_groups=self.hydra_groups,
            n_jobs=self.n_jobs,
            random_state=self.random_state
        )
        F_h = self.hydra_.fit_transform(X)
        F_mq = extract_moment_quant_features(X, max_depth=self.quant_depth)
        F_fft = extract_spectral_features(X, n_bands=self.n_fft_bands)
        return np.hstack([F_h, F_mq, F_fft])

    def transform_features(self, X):
        """Transform new series using fitted Hydra and extract features."""
        F_h = self.hydra_.transform(X)
        F_mq = extract_moment_quant_features(X, max_depth=self.quant_depth)
        F_fft = extract_spectral_features(X, n_bands=self.n_fft_bands)
        return np.hstack([F_h, F_mq, F_fft])

    def fit(self, X, y, precomputed_features=None, n_estimators=None):
        """
        Fit Branch B on training series X and class labels y.
        Optionally accepts precomputed features and custom tree count.
        """
        self.classes_ = np.unique(y)
        if precomputed_features is not None:
            F = precomputed_features
        else:
            F = self.fit_transform_features(X)

        n_trees = n_estimators if n_estimators is not None else self.n_estimators
        self.tree_ = ExtraTreesClassifier(
            n_estimators=n_trees,
            max_features=self.max_features,
            criterion=self.criterion,
            random_state=self.random_state,
            n_jobs=self.n_jobs
        )
        self.tree_.fit(F, y)
        return self

    def predict_proba(self, X, precomputed_features=None):
        """Compute calibrated class posterior probabilities."""
        if precomputed_features is not None:
            F = precomputed_features
        else:
            F = self.transform_features(X)
        return self.tree_.predict_proba(F)

    def predict(self, X, precomputed_features=None):
        """Predict class labels."""
        if precomputed_features is not None:
            F = precomputed_features
        else:
            F = self.transform_features(X)
        return self.tree_.predict(F)
