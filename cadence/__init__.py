"""
CADENCE: A Confidence-Adaptive Dual-Expert Network
for Fast and Accurate Time Series Classification.
"""

from .classifier import CADENCEClassifier
from .branch_a import ConvolutionalLinearExpert
from .branch_b import DistributionalIntervalExpert
from .router import AdaptiveMetaRouter, safe_validation_split
from .moment_quant import extract_moment_quant_features
from .spectral import extract_spectral_features

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
