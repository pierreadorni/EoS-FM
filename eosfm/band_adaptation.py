import torch

ADAPTATIONS_REGISTRY = {}


def register_strategy(cls):
    """Decorator to register a band adaptation strategy."""
    key = (cls.required_bands, cls.available_bands)
    if key in ADAPTATIONS_REGISTRY:
        ADAPTATIONS_REGISTRY[key].append(cls)
    else:
        ADAPTATIONS_REGISTRY[key] = [cls]
    return cls


class BandAdaptationStrategy:
    """Base class for band adaptation strategies."""

    required_bands: int
    available_bands: int

    def adapt(self, features: torch.Tensor) -> torch.Tensor:
        """Adapt the features to the target domain."""
        raise NotImplementedError("This method should be overridden by subclasses.")


@register_strategy
class S212toRGB(BandAdaptationStrategy):
    required_bands = 3
    available_bands = 12

    def adapt(self, features: torch.Tensor) -> torch.Tensor:
        # pick Sentinel-2 bands 4,3,2
        return features[:, [3, 2, 1], :, :]


@register_strategy
class S213ToRGB(BandAdaptationStrategy):
    required_bands = 3
    available_bands = 13

    def adapt(self, features: torch.Tensor) -> torch.Tensor:
        # pick Sentinel-2 bands 4,3,2 out of Sentinel-2&3 composite
        return features[:, [3, 2, 1], :, :]


@register_strategy
class SARtoRGB(BandAdaptationStrategy):
    required_bands = 3
    available_bands = 2

    def adapt(self, features: torch.Tensor) -> torch.Tensor:
        # create a composite from SAR bands (VV, VH, VH)
        return torch.cat([features, features[:, [1], :, :]], dim=1)


@register_strategy
class S12ToS212(BandAdaptationStrategy):
    required_bands = 12
    available_bands = 14

    def adapt(self, features: torch.Tensor) -> torch.Tensor:
        # pick Sentinel-2 bands out of Sentinel-1&2 composite (usually bands 2-14)
        return features[:, 2:14, :, :]


@register_strategy
class S12ToSAR(BandAdaptationStrategy):
    required_bands = 2
    available_bands = 14

    def adapt(self, features: torch.Tensor) -> torch.Tensor:
        # select Sentinel-1 bands out of Sentinel-1&2 composite (usually bands 0-1)
        return features[:, :2, :, :]


@register_strategy
class S12ToRGB(BandAdaptationStrategy):
    required_bands = 3
    available_bands = 14

    def adapt(self, features: torch.Tensor) -> torch.Tensor:
        # pick Sentinel-2 bands 4,3,2 out of Sentinel-1&2 composite
        return features[:, [5, 4, 3], :, :]
