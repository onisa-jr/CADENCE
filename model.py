"""
Unified Time Series Classifier (FastUnifiedClassifier)

Architecture:
1. Compact Hydra: Random dilated competing kernels across the time domain.
2. MomentQuant: Thread-safe, linear-time O(L) Cornish-Fisher dyadic intervals 
   (mean, variance, skewness, kurtosis, and 5 analytical quantiles) across:
   - Raw signal (X)
   - First difference / Velocity (ΔX)
   - Second difference / Acceleration (Δ²X)
3. FFT Spectral Features: Real FFT magnitude with log-spaced frequency bands,
   spectral centroid, spectral spread, and cumulative harmonic power quantiles.
4. Decision Head: Robust tree-based or regularized linear estimator.
"""

import numpy as np
from numba import njit, prange
from scipy.fft import rfft
from scipy.special import softmax
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeClassifierCV
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score
from aeon.transformations.collection.convolution_based import HydraTransformer

# Standard normal quantiles: 5%, 25%, 50%, 75%, 95%
Z_QUANTILES = np.array(
    [-1.6448536269514722, -0.6744897501960817, 0.0, 0.6744897501960817, 1.6448536269514722],
    dtype=np.float64
)

# ─────────────────────────────────────────────────────────────
# 1. THREAD-SAFE MOMENTQUANT (CORNISH-FISHER DYADIC INTERVALS)
# ─────────────────────────────────────────────────────────────

@njit(fastmath=True)
def _interval_moments_and_cf(segment, z_vals):
    """
    Computes 4 sample moments (mean, std, skewness, excess kurtosis)
    and 5 Cornish-Fisher approximate quantiles in O(L) time without sorting.
    """
    n = len(segment)
    if n < 4:
        return np.zeros(9, dtype=np.float64)

    # 1. Mean
    m1 = 0.0
    for i in range(n):
        m1 += segment[i]
    mean = m1 / n

    # 2. Central moments m2, m3, m4
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
    
    # Skewness & Excess Kurtosis
    denom3 = var * std + 1e-8
    denom4 = var * var + 1e-8
    skew = (m3 / n) / denom3
    kurt = ((m4 / n) / denom4) - 3.0

    # Bounded regularization for extreme moments
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
    kurt_term  = kurt / 24.0
    skew_term2 = (skew * skew) / 36.0

    for idx in range(5):
        z = z_vals[idx]
        z2 = z * z
        z3 = z2 * z
        expansion = z + skew_term1 * (z2 - 1.0) + kurt_term * (z3 - 3.0 * z) - skew_term2 * (2.0 * z3 - 5.0 * z)
        out[4 + idx] = mean + std * expansion

    return out


@njit(parallel=True, fastmath=True)
def _extract_moment_quant_single_representation(X_rep, max_depth, z_vals):
    """
    Extracts Cornish-Fisher interval features over dyadic recursive intervals.
    Deterministic and thread-safe across parallel workers.
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
    Extracts MomentQuant features across 3 views:
    1. Raw signal (X)
    2. Velocity / First difference (dX)
    3. Acceleration / Second difference (d2X)
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


# ─────────────────────────────────────────────────────────────
# 2. FFT SPECTRAL ENERGY & HARMONIC QUANTILES
# ─────────────────────────────────────────────────────────────

def extract_spectral_features(X, n_bands=16):
    """
    Extracts frequency-domain features via Real FFT:
    - Spectral energy across logarithmic frequency bands
    - Spectral centroid and spread
    - Cumulative power quantiles (25%, 50%, 75%, 90%)
    """
    if X.ndim == 3:
        X_2d = X[:, 0, :]
    else:
        X_2d = X
        
    n_cases, length = X_2d.shape
    fft_vals = np.abs(rfft(X_2d, axis=1))
    fft_len = fft_vals.shape[1]
    
    fft_norm = fft_vals / (np.sum(fft_vals, axis=1, keepdims=True) + 1e-8)
    
    features = []
    
    # 1. Log-spaced frequency bands
    band_edges = np.logspace(0, np.log10(fft_len), n_bands + 1, dtype=int)
    band_edges[0] = 0
    band_edges = np.unique(band_edges)
    
    band_energies = []
    for b in range(len(band_edges) - 1):
        s_idx, e_idx = band_edges[b], band_edges[b+1]
        if e_idx > s_idx:
            energy = np.sum(fft_norm[:, s_idx:e_idx], axis=1, keepdims=True)
            band_energies.append(energy)
    if band_energies:
        features.append(np.hstack(band_energies))
        
    # 2. Spectral Centroid and Spread
    freq_indices = np.arange(fft_len)
    centroid = np.sum(fft_norm * freq_indices, axis=1, keepdims=True)
    spread = np.sqrt(np.sum(fft_norm * ((freq_indices - centroid) ** 2), axis=1, keepdims=True))
    features.append(centroid / fft_len)
    features.append(spread / fft_len)
    
    # 3. Spectral quantiles
    cum_power = np.cumsum(fft_norm, axis=1)
    quant_indices = []
    for q in [0.25, 0.50, 0.75, 0.90]:
        idx = np.argmax(cum_power >= q, axis=1, keepdims=True) / fft_len
        quant_indices.append(idx)
    features.append(np.hstack(quant_indices))
    
    return np.hstack(features)


# ─────────────────────────────────────────────────────────────
# 3. FAST UNIFIED CLASSIFIER
# ─────────────────────────────────────────────────────────────

class FastUnifiedClassifier(BaseEstimator, ClassifierMixin):
    """
    Self-contained, fast, and multi-domain Time Series Classifier.
    """
    def __init__(
        self,
        hydra_kernels=8,
        hydra_groups=16,
        quant_depth=5,
        n_fft_bands=16,
        classifier_type="tree",  # 'tree', 'ridge', or 'ensemble'
        n_jobs=4,
        random_state=42
    ):
        self.hydra_kernels = hydra_kernels
        self.hydra_groups = hydra_groups
        self.quant_depth = quant_depth
        self.n_fft_bands = n_fft_bands
        self.classifier_type = classifier_type
        self.n_jobs = n_jobs
        self.random_state = random_state

    def _extract_all(self, X, is_fit=False):
        if is_fit:
            self.hydra_raw_ = HydraTransformer(
                n_kernels=self.hydra_kernels,
                n_groups=self.hydra_groups,
                n_jobs=self.n_jobs,
                random_state=self.random_state
            )
            F_hydra = self.hydra_raw_.fit_transform(X)
        else:
            F_hydra = self.hydra_raw_.transform(X)
            
        F_quant = extract_moment_quant_features(X, max_depth=self.quant_depth)
        F_fft = extract_spectral_features(X, n_bands=self.n_fft_bands)
        
        return F_hydra, F_quant, F_fft

    def fit(self, X, y):
        F_hydra, F_quant, F_fft = self._extract_all(X, is_fit=True)
        F_combined = np.hstack([F_hydra, F_quant, F_fft])
        
        self.classes_ = np.unique(y)
        
        if self.classifier_type in ["ridge", "ensemble"]:
            self.scaler_ = StandardScaler()
            F_scaled = self.scaler_.fit_transform(F_combined)
            self.ridge_ = RidgeClassifierCV(alphas=np.logspace(-3, 4, 15))
            self.ridge_.fit(F_scaled, y)
            
        if self.classifier_type in ["tree", "ensemble"]:
            self.tree_ = ExtraTreesClassifier(
                n_estimators=100,
                max_features=0.1,
                criterion="entropy",
                random_state=self.random_state,
                n_jobs=self.n_jobs
            )
            self.tree_.fit(F_combined, y)
            
        return self

    def predict_proba(self, X):
        F_hydra, F_quant, F_fft = self._extract_all(X, is_fit=False)
        F_combined = np.hstack([F_hydra, F_quant, F_fft])
        
        if self.classifier_type == "tree":
            return self.tree_.predict_proba(F_combined)
            
        F_scaled = self.scaler_.transform(F_combined)
        df = self.ridge_.decision_function(F_scaled)
        if df.ndim == 1:
            df = np.column_stack([-df, df])
        probs_ridge = softmax(df, axis=1)
        
        if self.classifier_type == "ridge":
            return probs_ridge
            
        # 50/50 Soft Voting Ensemble
        probs_tree = self.tree_.predict_proba(F_combined)
        return 0.5 * probs_ridge + 0.5 * probs_tree

    def predict(self, X):
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]

    def score(self, X, y):
        preds = self.predict(X)
        return accuracy_score(y, preds)
