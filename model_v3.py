"""
Adaptive Expert Time Series Classifier V3 (AdaptiveExpertClassifierV3)

Architecture:
1. Branch A (Convolutional Linear Expert):
   - MiniRocket (10,000 multi-dilation kernels)
   - Standard Scaler normalization
   - RidgeClassifierCV (closed-form Woodbury L2 regularized solver across 15 log-spaced alphas)
   - Calibrated class probabilities via softmax(decision_function)
   - Optimal for continuous waveforms, geometric outlines, and high-class counts.

2. Branch B (Distributional Interval Expert):
   - Compact Hydra: Competing dilated kernels capturing local shape occurrences.
   - MomentQuant: Thread-safe, linear-time O(L) Cornish-Fisher dyadic intervals 
     across raw signal (X), velocity (ΔX), and acceleration (Δ²X).
   - Real FFT Spectral Quantiles: Energy bands, spectral centroid, spectral spread.
   - Fitted with ExtraTreesClassifier (100 trees, 10% max_features, entropy criterion).
   - Calibrated class probabilities via predict_proba.
   - Optimal for phase-free sensors, motion kinematics, acoustic formants, and bone outlines.

3. Confidence-Aware Adaptive Meta-Router & Blending:
   - Uses a class-aware validation split on X_train (handles single-case classes safely).
   - Computes internal validation scores: val_A vs val_B.
   - Routing & Blending Rules:
     * Clear Dominance by A (val_A - val_B > 0.08): Executes Pure Branch A (e.g. PigAirwayPressure, ShapesAll).
     * Clear Dominance by B (val_B - val_A > 0.08): Executes Pure Branch B (e.g. InlineSkate, Semg).
     * Competitive Regime (|val_A - val_B| <= 0.08): Smoothly confidence-weights both experts 
       proportional to validation accuracy (e.g. Phoneme, ArrowHead, LargeKitchenAppliances).
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
from aeon.transformations.collection.convolution_based import MiniRocket, HydraTransformer

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
    if X.ndim == 3:
        X_2d = X[:, 0, :]
    else:
        X_2d = X
        
    n_cases, length = X_2d.shape
    fft_vals = np.abs(rfft(X_2d, axis=1))
    fft_len = fft_vals.shape[1]
    
    fft_norm = fft_vals / (np.sum(fft_vals, axis=1, keepdims=True) + 1e-8)
    
    features = []
    
    # Log-spaced frequency bands
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
        
    # Spectral Centroid and Spread
    freq_indices = np.arange(fft_len)
    centroid = np.sum(fft_norm * freq_indices, axis=1, keepdims=True)
    spread = np.sqrt(np.sum(fft_norm * ((freq_indices - centroid) ** 2), axis=1, keepdims=True))
    features.append(centroid / fft_len)
    features.append(spread / fft_len)
    
    # Cumulative power quantiles
    cum_power = np.cumsum(fft_norm, axis=1)
    quant_indices = []
    for q in [0.25, 0.50, 0.75, 0.90]:
        idx = np.argmax(cum_power >= q, axis=1, keepdims=True) / fft_len
        quant_indices.append(idx)
    features.append(np.hstack(quant_indices))
    
    return np.hstack(features)


# ─────────────────────────────────────────────────────────────
# 3. CLASS-AWARE VALIDATION SPLIT
# ─────────────────────────────────────────────────────────────

def _safe_validation_split(y, val_fraction=0.30, random_state=42):
    """
    Safely partitions classes even if some classes only have 1 training sample.
    Classes with count == 1 are placed into train, classes >= 2 are stratified.
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
    return np.array(train_indices), np.array(val_indices)


# ─────────────────────────────────────────────────────────────
# 4. ADAPTIVE EXPERT CLASSIFIER V3
# ─────────────────────────────────────────────────────────────

class AdaptiveExpertClassifierV3(BaseEstimator, ClassifierMixin):
    """
    Version 3 Adaptive Expert Classifier with Confidence-Aware Routing and Blending.
    """
    def __init__(
        self,
        mode="adaptive",
        quant_depth=5,
        n_fft_bands=16,
        hydra_groups=16,
        n_jobs=4,
        random_state=42
    ):
        self.mode = mode
        self.quant_depth = quant_depth
        self.n_fft_bands = n_fft_bands
        self.hydra_groups = hydra_groups
        self.n_jobs = n_jobs
        self.random_state = random_state

    def _extract_branch_a(self, X, fit=False):
        if fit:
            self.mr_ = MiniRocket(random_state=self.random_state, n_jobs=self.n_jobs)
            F = self.mr_.fit_transform(X)
            self.scaler_ = StandardScaler()
            return self.scaler_.fit_transform(F)
        else:
            F = self.mr_.transform(X)
            return self.scaler_.transform(F)

    def _extract_branch_b(self, X, fit=False):
        if fit:
            self.hydra_ = HydraTransformer(
                n_kernels=8,
                n_groups=self.hydra_groups,
                n_jobs=self.n_jobs,
                random_state=self.random_state
            )
            F_h = self.hydra_.fit_transform(X)
        else:
            F_h = self.hydra_.transform(X)
            
        F_mq = extract_moment_quant_features(X, max_depth=self.quant_depth)
        F_fft = extract_spectral_features(X, n_bands=self.n_fft_bands)
        return np.hstack([F_h, F_mq, F_fft])

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        K = len(self.classes_)
        
        if self.mode == "adaptive":
            # Extract features for both branches
            F_A_all = self._extract_branch_a(X, fit=True)
            F_B_all = self._extract_branch_b(X, fit=True)
            
            # Internal class-aware validation
            tr_sub, val_sub = _safe_validation_split(y, val_fraction=0.30, random_state=self.random_state)
            
            if len(val_sub) > 0:
                # Validation evaluation on Branch A (Ridge)
                r_val = RidgeClassifierCV(alphas=np.logspace(-3, 4, 15))
                r_val.fit(F_A_all[tr_sub], y[tr_sub])
                val_a = float(np.mean(r_val.predict(F_A_all[val_sub]) == y[val_sub]))
                
                # Validation evaluation on Branch B (fast 50 trees)
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
                
            diff = val_a - val_b
            self.val_scores_ = {"val_A": val_a, "val_B": val_b, "diff": diff}
            
            # Decision boundary:
            # - Clear dominance by A: Pure A
            # - Clear dominance by B: Pure B
            # - Close / competitive: Confidence-weighted soft blend
            if diff > 0.08:
                self.decision_ = "pure_a"
                self.ridge_ = RidgeClassifierCV(alphas=np.logspace(-3, 4, 15)).fit(F_A_all, y)
                self.tree_ = None
            elif diff < -0.08:
                self.decision_ = "pure_b"
                self.tree_ = ExtraTreesClassifier(
                    n_estimators=100,
                    max_features=0.1,
                    criterion="entropy",
                    random_state=self.random_state,
                    n_jobs=self.n_jobs
                ).fit(F_B_all, y)
                self.ridge_ = None
            else:
                self.decision_ = "blend"
                self.w_a_ = float(max(0.15, min(0.85, 0.5 + 2.0 * diff)))
                self.ridge_ = RidgeClassifierCV(alphas=np.logspace(-3, 4, 15)).fit(F_A_all, y)
                self.tree_ = ExtraTreesClassifier(
                    n_estimators=100,
                    max_features=0.1,
                    criterion="entropy",
                    random_state=self.random_state,
                    n_jobs=self.n_jobs
                ).fit(F_B_all, y)

        elif self.mode == "conv":
            self.decision_ = "pure_a"
            F_A = self._extract_branch_a(X, fit=True)
            self.ridge_ = RidgeClassifierCV(alphas=np.logspace(-3, 4, 15)).fit(F_A, y)
            self.tree_ = None
        else:
            self.decision_ = "pure_b"
            F_B = self._extract_branch_b(X, fit=True)
            self.tree_ = ExtraTreesClassifier(
                n_estimators=100,
                max_features=0.1,
                criterion="entropy",
                random_state=self.random_state,
                n_jobs=self.n_jobs
            ).fit(F_B, y)
            self.ridge_ = None

        return self

    def predict_proba(self, X):
        if self.decision_ == "pure_a":
            F_A = self._extract_branch_a(X, fit=False)
            df = self.ridge_.decision_function(F_A)
            if df.ndim == 1:
                df = np.column_stack([-df, df])
            return softmax(df, axis=1)
        elif self.decision_ == "pure_b":
            F_B = self._extract_branch_b(X, fit=False)
            return self.tree_.predict_proba(F_B)
        else:
            # Blend
            F_A = self._extract_branch_a(X, fit=False)
            df = self.ridge_.decision_function(F_A)
            if df.ndim == 1:
                df = np.column_stack([-df, df])
            prob_a = softmax(df, axis=1)
            
            F_B = self._extract_branch_b(X, fit=False)
            prob_b = self.tree_.predict_proba(F_B)
            
            return self.w_a_ * prob_a + (1.0 - self.w_a_) * prob_b

    def predict(self, X):
        if self.decision_ == "pure_a":
            F_A = self._extract_branch_a(X, fit=False)
            return self.ridge_.predict(F_A)
        elif self.decision_ == "pure_b":
            F_B = self._extract_branch_b(X, fit=False)
            return self.tree_.predict(F_B)
        else:
            prob = self.predict_proba(X)
            return self.classes_[np.argmax(prob, axis=1)]

    def score(self, X, y):
        preds = self.predict(X)
        return accuracy_score(y, preds)
