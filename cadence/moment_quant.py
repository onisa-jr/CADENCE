"""
Cornish-Fisher Dyadic Interval Feature Extraction (MomentQuant)
Part of the CADENCE Classifier Suite.

Extracts thread-safe, linear-time O(L) statistical moments and
analytical quantiles via the Cornish-Fisher asymptotic expansion across
dyadic intervals for raw signal (X), velocity (ΔX), and acceleration (Δ²X).
"""

import numpy as np
from numba import njit, prange

# Standard normal quantiles corresponding to probabilities: 5%, 25%, 50%, 75%, 95%
Z_QUANTILES = np.array(
    [-1.6448536269514722, -0.6744897501960817, 0.0, 0.6744897501960817, 1.6448536269514722],
    dtype=np.float64
)


@njit(fastmath=True)
def _interval_moments_and_cf(segment, z_vals):
    """
    Computes sample mean, standard deviation, skewness, excess kurtosis,
    and 5 Cornish-Fisher asymptotic quantiles for a 1D time-series segment.
    """
    n = len(segment)
    if n < 4:
        return np.zeros(9, dtype=np.float64)

    m1 = 0.0
    for i in range(n):
        m1 += segment[i]
    mean = m1 / n

    m2 = 0.0
    m3 = 0.0
    m4 = 0.0
    for i in range(n):
        diff = segment[i] - mean
        d2 = diff * diff
        m2 += d2
        m3 += d2 * diff
        m4 += d2 * d2

    var = m2 / n
    std = np.sqrt(var + 1e-8)

    denom3 = var * std + 1e-8
    denom4 = var * var + 1e-8
    skew = (m3 / n) / denom3
    kurt = ((m4 / n) / denom4) - 3.0

    # Robust bounds on sample skewness and kurtosis
    if skew > 4.0:
        skew = 4.0
    elif skew < -4.0:
        skew = -4.0

    if kurt > 8.0:
        kurt = 8.0
    elif kurt < -2.0:
        kurt = -2.0

    out = np.empty(9, dtype=np.float64)
    out[0] = mean
    out[1] = std
    out[2] = skew
    out[3] = kurt

    skew_term1 = skew / 6.0
    kurt_term = kurt / 24.0
    skew_term2 = (skew * skew) / 36.0

    for idx in range(5):
        z = z_vals[idx]
        z2 = z * z
        z3 = z2 * z
        expansion = (
            z
            + skew_term1 * (z2 - 1.0)
            + kurt_term * (z3 - 3.0 * z)
            - skew_term2 * (2.0 * z3 - 5.0 * z)
        )
        out[4 + idx] = mean + std * expansion

    return out


@njit(parallel=True, fastmath=True)
def _extract_moment_quant_single_representation(X_rep, max_depth, z_vals):
    """
    Extracts dyadic interval moments across binary tree partitions up to max_depth.
    """
    n_cases, length = X_rep.shape
    total_intervals = (1 << (max_depth + 1)) - 1
    n_features = total_intervals * 9

    features = np.zeros((n_cases, n_features), dtype=np.float64)

    for i in prange(n_cases):
        series = X_rep[i]
        for depth in range(max_depth + 1):
            n_splits = 1 << depth
            seg_len = length // n_splits
            if seg_len < 4:
                continue
            depth_offset = ((1 << depth) - 1) * 9
            for s in range(n_splits):
                start = s * seg_len
                end = length if s == n_splits - 1 else (s + 1) * seg_len
                segment = series[start:end]
                moments_cf = _interval_moments_and_cf(segment, z_vals)
                col_start = depth_offset + s * 9
                for f in range(9):
                    features[i, col_start + f] = moments_cf[f]

    return features


def extract_moment_quant_features(X, max_depth=5):
    """
    Extracts dyadic interval statistical moments and Cornish-Fisher quantiles
    across three signal kinematic representations:
    1. Raw signal: X
    2. Velocity (1st order backward difference): ΔX
    3. Acceleration (2nd order backward difference): Δ²X

    Parameters
    ----------
    X : np.ndarray
        Input series of shape (n_cases, length) or (n_cases, 1, length).
    max_depth : int, default=5
        Maximum depth of dyadic binary tree interval partitioning.

    Returns
    -------
    np.ndarray
        Concatenated features of shape (n_cases, 3 * (2^(max_depth+1) - 1) * 9).
        At max_depth=5, yields 3 * 63 * 9 = 1,701 features.
    """
    if X.ndim == 3:
        X_2d = X[:, 0, :]
    else:
        X_2d = X

    f_raw = _extract_moment_quant_single_representation(X_2d, max_depth, Z_QUANTILES)
    dX = np.diff(X_2d, axis=1)
    f_diff1 = _extract_moment_quant_single_representation(dX, max_depth, Z_QUANTILES)
    d2X = np.diff(dX, axis=1)
    f_diff2 = _extract_moment_quant_single_representation(d2X, max_depth, Z_QUANTILES)

    return np.hstack([f_raw, f_diff1, f_diff2])
