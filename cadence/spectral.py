"""
Fast Real FFT Spectral Feature Extraction
Part of the CADENCE Classifier Suite.

Extracts log-spaced harmonic frequency energy bands, spectral centroid,
spectral spread, and cumulative power quantiles.
"""

import numpy as np
from scipy.fft import rfft


def extract_spectral_features(X, n_bands=16):
    """
    Extracts frequency-domain features from univariate time series:
    1. Log-spaced frequency energy bands (default 16 bands).
    2. Spectral centroid (center of mass of spectrum).
    3. Spectral spread (variance/dispersion around centroid).
    4. Cumulative harmonic power quantiles (25%, 50%, 75%, 90%).

    Parameters
    ----------
    X : np.ndarray
        Input series of shape (n_cases, length) or (n_cases, 1, length).
    n_bands : int, default=16
        Number of candidate log-spaced frequency bands.

    Returns
    -------
    np.ndarray
        Array of shape (n_cases, n_features) (typically 22 features).
    """
    if X.ndim == 3:
        X_2d = X[:, 0, :]
    else:
        X_2d = X

    fft_vals = np.abs(rfft(X_2d, axis=1))
    fft_len = fft_vals.shape[1]

    # Normalize spectral power per series
    fft_norm = fft_vals / (np.sum(fft_vals, axis=1, keepdims=True) + 1e-8)

    features = []

    # 1. Log-spaced frequency bands
    band_edges = np.logspace(0, np.log10(fft_len), n_bands + 1, dtype=int)
    band_edges[0] = 0
    band_edges = np.unique(band_edges)

    band_energies = []
    for b in range(len(band_edges) - 1):
        s_idx, e_idx = band_edges[b], band_edges[b + 1]
        if e_idx > s_idx:
            energy = np.sum(fft_norm[:, s_idx:e_idx], axis=1, keepdims=True)
            band_energies.append(energy)
    if band_energies:
        features.append(np.hstack(band_energies))

    # 2. Spectral Centroid and Spread
    freq_indices = np.arange(fft_len)
    centroid = np.sum(fft_norm * freq_indices, axis=1, keepdims=True)
    spread = np.sqrt(
        np.sum(fft_norm * ((freq_indices - centroid) ** 2), axis=1, keepdims=True)
    )
    features.append(centroid / fft_len)
    features.append(spread / fft_len)

    # 3. Cumulative power harmonic quantiles
    cum_power = np.cumsum(fft_norm, axis=1)
    quant_indices = []
    for q in [0.25, 0.50, 0.75, 0.90]:
        idx = np.argmax(cum_power >= q, axis=1, keepdims=True) / fft_len
        quant_indices.append(idx)
    features.append(np.hstack(quant_indices))

    return np.hstack(features)
