"""
Branch A: Convolutional Linear Expert
Part of the CADENCE Classifier Suite.

Extracts D_A = 10,000 dilated convolutional features via MiniRocket
with Proportion of Positive Values (PPV) pooling, standardized with
StandardScaler, and fitted via closed-form L2-regularized Woodbury ridge classification.
Calibrated posterior probabilities are derived via softmax over decision values.
"""

import numpy as np
from scipy.special import softmax
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeClassifierCV
from aeon.transformations.collection.convolution_based import MiniRocket


class ConvolutionalLinearExpert(BaseEstimator, ClassifierMixin):
    """
    Convolutional Linear Expert (Branch A).

    Parameters
    ----------
    alphas : array-like, default=np.logspace(-3, 4, 15)
        Candidate L2 regularization penalties.
    n_jobs : int, default=4
        Number of parallel CPU worker threads.
    random_state : int, default=42
        Seed for reproducible kernel dilation sampling.
    """

    def __init__(
        self,
        alphas=None,
        n_jobs=4,
        random_state=42
    ):
        self.alphas = alphas if alphas is not None else np.logspace(-3, 4, 15)
        self.n_jobs = n_jobs
        self.random_state = random_state

    def fit_transform_features(self, X):
        """Fit MiniRocket transformer and StandardScaler, returning standardized features."""
        self.mr_ = MiniRocket(random_state=self.random_state, n_jobs=self.n_jobs)
        F = self.mr_.fit_transform(X)
        self.scaler_ = StandardScaler()
        return self.scaler_.fit_transform(F)

    def transform_features(self, X):
        """Transform new series using fitted MiniRocket and StandardScaler."""
        F = self.mr_.transform(X)
        return self.scaler_.transform(F)

    def fit(self, X, y, precomputed_features=None):
        """
        Fit Branch A on training series X and class labels y.
        Optionally accepts precomputed features to avoid redundant transforms.
        """
        self.classes_ = np.unique(y)
        if precomputed_features is not None:
            F = precomputed_features
        else:
            F = self.fit_transform_features(X)

        self.ridge_ = RidgeClassifierCV(alphas=self.alphas)
        self.ridge_.fit(F, y)
        return self

    def decision_function(self, X, precomputed_features=None):
        """Compute decision function values."""
        if precomputed_features is not None:
            F = precomputed_features
        else:
            F = self.transform_features(X)
        df = self.ridge_.decision_function(F)
        if df.ndim == 1:
            df = np.column_stack([-df, df])
        return df

    def predict_proba(self, X, precomputed_features=None):
        """Compute calibrated class posterior probabilities via softmax."""
        df = self.decision_function(X, precomputed_features=precomputed_features)
        return softmax(df, axis=1)

    def predict(self, X, precomputed_features=None):
        """Predict class labels."""
        if precomputed_features is not None:
            F = precomputed_features
        else:
            F = self.transform_features(X)
        return self.ridge_.predict(F)
