"""EoS-FM inference with optional, explicit TerraTorch training integration."""

from .encoder import EosFM, SmartBandAdapter

__all__ = ["EosFM", "SmartBandAdapter", "register_terratorch"]


def register_terratorch() -> None:
    """Register the backbone and custom factories for TerraTorch training.

    Install ``eosfm[training]`` and call this before building a configured
    TerraTorch model. Basic inference never imports the training stack.
    """
    from terratorch.registry import TERRATORCH_BACKBONE_REGISTRY
    from . import model_factory  # factory decorators perform registration

    if "EosFM" not in TERRATORCH_BACKBONE_REGISTRY:
        TERRATORCH_BACKBONE_REGISTRY.register(EosFM)
