"""
Confidence-Adaptive Meta-Router
Part of the CADENCE Classifier Suite.

Partitions training data into an internal 30% held-out validation set
while safely preserving singleton classes. Determines whether a dataset
benefits from:
- Regime I: Pure Branch B Dominance (Δ < -0.08)
- Regime II: Smooth Confidence Blending (|Δ| <= 0.08)
- Regime III: Pure Branch A Dominance (Δ > 0.08)
"""

import numpy as np


def safe_validation_split(y, val_fraction=0.30, random_state=42):
    """
    Safely partitions classes into train and validation sets, even if
    certain classes have only 1 training instance.
    Classes with count == 1 are placed into train; classes with count >= 2
    are stratified.

    Parameters
    ----------
    y : np.ndarray
        Class labels array of shape (n_cases,).
    val_fraction : float, default=0.30
        Fraction of samples to allocate to internal validation.
    random_state : int, default=42
        Random seed for reproducible permutation.

    Returns
    -------
    train_indices : np.ndarray
        Indices allocated to internal sub-training.
    val_indices : np.ndarray
        Indices allocated to internal validation.
    """
    classes, counts = np.unique(y, return_counts=True)
    rng = np.random.RandomState(random_state)
    val_indices = []
    train_indices = []

    for c, cnt in zip(classes, counts):
        idx = np.where(y == c)[0]
        if cnt < 2:
            train_indices.extend(idx)
        else:
            n_val = max(1, int(round(cnt * val_fraction)))
            if n_val >= cnt:
                n_val = cnt - 1
            perm = rng.permutation(idx)
            val_indices.extend(perm[:n_val])
            train_indices.extend(perm[n_val:])

    return np.array(train_indices, dtype=int), np.array(val_indices, dtype=int)


class AdaptiveMetaRouter:
    """
    Meta-Router for CADENCE Dual-Expert Architecture.

    Parameters
    ----------
    threshold : float, default=0.08
        Regime transition margin threshold (tau).
    val_fraction : float, default=0.30
        Internal validation set fraction.
    min_blend_weight : float, default=0.15
        Lower bound for Branch A blend weight.
    max_blend_weight : float, default=0.85
        Upper bound for Branch A blend weight.
    blend_slope : float, default=2.0
        Slope factor for dynamic blend weight adjustment.
    random_state : int, default=42
        Seed for reproducible validation splitting.
    """

    def __init__(
        self,
        threshold=0.08,
        val_fraction=0.30,
        min_blend_weight=0.15,
        max_blend_weight=0.85,
        blend_slope=2.0,
        random_state=42
    ):
        self.threshold = threshold
        self.val_fraction = val_fraction
        self.min_blend_weight = min_blend_weight
        self.max_blend_weight = max_blend_weight
        self.blend_slope = blend_slope
        self.random_state = random_state

    def route(self, val_a, val_b):
        """
        Computes routing regime and blending weight based on validation accuracies.

        Parameters
        ----------
        val_a : float
            Internal validation accuracy of Branch A.
        val_b : float
            Internal validation accuracy of Branch B.

        Returns
        -------
        decision : str
            One of 'pure_a', 'pure_b', or 'blend'.
        diff : float
            Margin diff = val_a - val_b.
        w_a : float or None
            Dynamic blend weight for Branch A (None if dominance regime).
        """
        diff = float(val_a - val_b)

        if diff > self.threshold:
            decision = "pure_a"
            w_a = 1.0
        elif diff < -self.threshold:
            decision = "pure_b"
            w_a = 0.0
        else:
            decision = "blend"
            raw_w = 0.5 + self.blend_slope * diff
            w_a = float(np.clip(raw_w, self.min_blend_weight, self.max_blend_weight))

        return decision, diff, w_a
