"""
CADENCE Model Interface (model.py)
Re-exports CADENCEClassifier and provides backward-compatibility aliases.
"""

from cadence.classifier import CADENCEClassifier
from cadence.branch_a import ConvolutionalLinearExpert
from cadence.branch_b import DistributionalIntervalExpert
from cadence.router import AdaptiveMetaRouter, safe_validation_split
from cadence.moment_quant import extract_moment_quant_features
from cadence.spectral import extract_spectral_features

# Backward-compatibility aliases for legacy scripts
FastUnifiedClassifier = DistributionalIntervalExpert
DualExpertClassifierV2 = CADENCEClassifier
AdaptiveExpertClassifierV3 = CADENCEClassifier

__all__ = [
    "CADENCEClassifier",
    "ConvolutionalLinearExpert",
    "DistributionalIntervalExpert",
    "AdaptiveMetaRouter",
    "safe_validation_split",
    "extract_moment_quant_features",
    "extract_spectral_features",
    "FastUnifiedClassifier",
    "DualExpertClassifierV2",
    "AdaptiveExpertClassifierV3",
]
