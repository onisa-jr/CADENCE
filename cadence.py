"""
CADENCE: Confidence-Adaptive Dual-Expert Network
for Fast and Accurate Time Series Classification.

Convenience root module re-exporting the CADENCE package API.
"""

from cadence.classifier import CADENCEClassifier
from cadence.branch_a import ConvolutionalLinearExpert
from cadence.branch_b import DistributionalIntervalExpert
from cadence.router import AdaptiveMetaRouter, safe_validation_split
from cadence.moment_quant import extract_moment_quant_features
from cadence.spectral import extract_spectral_features

__version__ = "1.0.0"
__author__ = "Onisa Mapunda"

__all__ = [
    "CADENCEClassifier",
    "ConvolutionalLinearExpert",
    "DistributionalIntervalExpert",
    "AdaptiveMetaRouter",
    "safe_validation_split",
    "extract_moment_quant_features",
    "extract_spectral_features",
]
