"""
CADENCE: Confidence-Adaptive Dual-Expert Network
for Fast and Accurate Time Series Classification

A unified, CPU-native dual-expert architecture decoupling representation
learning into:
1. Convolutional Linear Expert (Branch A: MiniRocket + Ridge Woodbury)
2. Distributional Interval Expert (Branch B: Hydra + Cornish-Fisher dyadic intervals + FFT + ExtraTrees)
3. Confidence-Adaptive Meta-Router (Internal validation routing & soft blending)
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import RidgeClassifierCV
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score

from .branch_a import ConvolutionalLinearExpert
from .branch_b import DistributionalIntervalExpert
from .router import AdaptiveMetaRouter, safe_validation_split


class CADENCEClassifier(BaseEstimator, ClassifierMixin):
    """
    CADENCE Classifier: Confidence-Adaptive Dual-Expert Network.

    Parameters
    ----------
    mode : str, default="adaptive"
        Operating mode:
        - "adaptive": Automatic confidence-aware routing and soft blending.
        - "pure_a" or "conv": Fixed Branch A (Convolutional Linear Expert alone).
        - "pure_b" or "interval": Fixed Branch B (Distributional Interval Expert alone).
        - "fixed_blend": Fixed 50/50 soft blend without adaptive validation.
    router_threshold : float, default=0.08
        Regime transition threshold (tau). When |val_A - val_B| <= threshold,
        CADENCE operates in Regime II (Competitive Soft Blending).
    quant_depth : int, default=5
        Maximum depth for dyadic interval Cornish-Fisher moment approximations.
    n_fft_bands : int, default=16
        Number of log-spaced FFT frequency energy bands.
    hydra_groups : int, default=16
        Number of competing kernel groups in Hydra (Branch B).
    n_jobs : int, default=4
        Number of parallel CPU worker threads.
    random_state : int, default=42
        Random seed for reproducible kernel dilations, tree splits, and splits.

    Attributes
    ----------
    classes_ : np.ndarray
        Unique class labels encountered during fit.
    decision_ : str
        Selected routing regime: 'pure_a', 'pure_b', or 'blend'.
    w_a_ : float
        Dynamic blend weight assigned to Branch A during prediction.
    routing_info_ : dict
        Internal validation metrics: {'val_A', 'val_B', 'diff', 'decision', 'w_a'}.
    """

    def __init__(
        self,
        mode="adaptive",
        router_threshold=0.08,
        quant_depth=5,
        n_fft_bands=16,
        hydra_groups=16,
        n_jobs=4,
        random_state=42
    ):
        self.mode = mode
        self.router_threshold = router_threshold
        self.quant_depth = quant_depth
        self.n_fft_bands = n_fft_bands
        self.hydra_groups = hydra_groups
        self.n_jobs = n_jobs
        self.random_state = random_state

    def fit(self, X, y):
        """
        Fit CADENCE on training series X and class labels y.

        Parameters
        ----------
        X : np.ndarray
            Input time series of shape (n_cases, length) or (n_cases, 1, length).
        y : np.ndarray
            Target class labels of shape (n_cases,).

        Returns
        -------
        self : CADENCEClassifier
            Fitted classifier instance.
        """
        self.classes_ = np.unique(y)

        # Initialize expert branches
        self.branch_a_ = ConvolutionalLinearExpert(
            n_jobs=self.n_jobs,
            random_state=self.random_state
        )
        self.branch_b_ = DistributionalIntervalExpert(
            hydra_groups=self.hydra_groups,
            quant_depth=self.quant_depth,
            n_fft_bands=self.n_fft_bands,
            n_estimators=100,
            n_jobs=self.n_jobs,
            random_state=self.random_state
        )

        if self.mode == "adaptive":
            router = AdaptiveMetaRouter(
                threshold=self.router_threshold,
                val_fraction=0.30,
                random_state=self.random_state
            )

            # Extract features for both branches
            F_A_all = self.branch_a_.fit_transform_features(X)
            F_B_all = self.branch_b_.fit_transform_features(X)

            # Internal class-aware validation split
            tr_sub, val_sub = safe_validation_split(y, val_fraction=0.30, random_state=self.random_state)

            if len(val_sub) > 0:
                # Fast validation on Branch A (Ridge Woodbury)
                r_val = RidgeClassifierCV(alphas=np.logspace(-3, 4, 15))
                r_val.fit(F_A_all[tr_sub], y[tr_sub])
                val_a = float(np.mean(r_val.predict(F_A_all[val_sub]) == y[val_sub]))

                # Fast validation on Branch B (50 trees for rapid evaluation)
                et_val = ExtraTreesClassifier(
                    n_estimators=50,
                    max_features=0.1,
                    criterion="entropy",
                    random_state=self.random_state,
                    n_jobs=self.n_jobs
                )
                et_val.fit(F_B_all[tr_sub], y[tr_sub])
                val_b = float(np.mean(et_val.predict(F_B_all[val_sub]) == y[val_sub]))
            else:
                val_a, val_b = 0.5, 0.5

            decision, diff, w_a = router.route(val_a, val_b)
            self.decision_ = decision
            self.w_a_ = w_a
            self.routing_info_ = {
                "val_A": val_a,
                "val_B": val_b,
                "diff": diff,
                "decision": decision,
                "w_a": w_a
            }

            # Full refit on 100% training data based on selected regime
            if self.decision_ == "pure_a":
                self.branch_a_.fit(X, y, precomputed_features=F_A_all)
                self.branch_b_ = None  # Prune unselected branch for memory & inference efficiency
            elif self.decision_ == "pure_b":
                self.branch_b_.fit(X, y, precomputed_features=F_B_all, n_estimators=100)
                self.branch_a_ = None  # Prune unselected branch
            else:
                # Regime II: Competitive Soft Blending
                self.branch_a_.fit(X, y, precomputed_features=F_A_all)
                self.branch_b_.fit(X, y, precomputed_features=F_B_all, n_estimators=100)

        elif self.mode in ["pure_a", "conv"]:
            self.decision_ = "pure_a"
            self.w_a_ = 1.0
            self.routing_info_ = {"decision": "pure_a"}
            self.branch_a_.fit(X, y)
            self.branch_b_ = None

        elif self.mode in ["pure_b", "interval"]:
            self.decision_ = "pure_b"
            self.w_a_ = 0.0
            self.routing_info_ = {"decision": "pure_b"}
            self.branch_b_.fit(X, y)
            self.branch_a_ = None

        elif self.mode == "fixed_blend":
            self.decision_ = "blend"
            self.w_a_ = 0.50
            self.routing_info_ = {"decision": "blend", "w_a": 0.50}
            self.branch_a_.fit(X, y)
            self.branch_b_.fit(X, y)

        else:
            raise ValueError(f"Unknown CADENCE mode: {self.mode}")

        return self

    def predict_proba(self, X):
        """
        Compute calibrated posterior class probabilities.
        Automatically executes inference pruning during dominance regimes.
        """
        if self.decision_ == "pure_a":
            return self.branch_a_.predict_proba(X)
        elif self.decision_ == "pure_b":
            return self.branch_b_.predict_proba(X)
        else:
            # Regime II: Soft confidence blending
            prob_a = self.branch_a_.predict_proba(X)
            prob_b = self.branch_b_.predict_proba(X)
            return self.w_a_ * prob_a + (1.0 - self.w_a_) * prob_b

    def predict(self, X):
        """
        Predict class labels for test series X.
        """
        if self.decision_ == "pure_a":
            return self.branch_a_.predict(X)
        elif self.decision_ == "pure_b":
            return self.branch_b_.predict(X)
        else:
            prob = self.predict_proba(X)
            return self.classes_[np.argmax(prob, axis=1)]

    def score(self, X, y):
        """Returns accuracy score on test series X and labels y."""
        return accuracy_score(y, self.predict(X))

    def get_routing_info(self):
        """Returns internal routing metrics dictionary."""
        return getattr(self, "routing_info_", {})
