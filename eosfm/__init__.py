"""EosFM package initialization."""

# Import model factory to register it
from .model_factory import ChangeDetectionEncoderDecoderFactory

__all__ = ["ChangeDetectionEncoderDecoderFactory"]
