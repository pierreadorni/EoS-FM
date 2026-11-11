"""EosFM Tasks module."""

from .change_detection import ChangeDetectionTask
from .regression import PixelwiseRegressionTask
from .segmentation import SemanticSegmentationTask

__all__ = ["ChangeDetectionTask", "PixelwiseRegressionTask", "SemanticSegmentationTask"]
