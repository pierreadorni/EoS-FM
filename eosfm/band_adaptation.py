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


@register_strategy
class S213ToRGIR(BandAdaptationStrategy):
    required_bands = 3
    available_bands = 13

    def adapt(self, features: torch.Tensor) -> torch.Tensor:
        # pick Sentinel-2 bands B8,B4,B3
        return features[:, [7, 3, 2], :, :]


@register_strategy
class RGBIRToIRRG(BandAdaptationStrategy):
    required_bands = 3
    available_bands = 4

    def adapt(self, features: torch.Tensor) -> torch.Tensor:
        # create a IR-R-G composite from the R-G-B-IR
        return features[:, [3, 0, 1], :, :]


@register_strategy
class RGBIRToRGB(BandAdaptationStrategy):
    """
    Given four channels organized like [B2, B3, B4, B8], extracts the RGB channels.
    """

    required_bands = 3
    available_bands = 4

    def adapt(self, features: torch.Tensor) -> torch.Tensor:
        # extract only the first three channels
        return features[:, [0, 1, 2], :, :]


@register_strategy
class RGBAndSARToRGB(BandAdaptationStrategy):
    """
    Given five channels organized like [B2, B3, B4, VV, VH], extracts the RGB channels.
    """

    required_bands = 3
    available_bands = 5

    def adapt(self, features: torch.Tensor) -> torch.Tensor:
        # extract only the first three channels
        return features[:, [0, 1, 2], :, :]


@register_strategy
class RGBAndSARToSAR(BandAdaptationStrategy):
    """
    Given five channels organized like [B2, B3, B4, VV, VH], extracts the SAR channels as a 3-channel composite.
    """

    required_bands = 3
    available_bands = 5

    def adapt(self, features: torch.Tensor) -> torch.Tensor:
        # extract only the last two channels and create composite
        return torch.cat([features[:, [3, 4], :, :], features[:, [4], :, :]], dim=1)


@register_strategy
class RGBToS212(BandAdaptationStrategy):
    """
    Given three channels organized like [B2, B3, B4], creates a 12-channel composite by
    repeating the RGB channels to fill the 12 channels.
    """

    required_bands = 12
    available_bands = 3

    def adapt(self, features: torch.Tensor) -> torch.Tensor:
        # repeat the RGB channels to create a 12-channel composite
        repeats = 12 // 3
        remainder = 12 % 3
        repeated = features.repeat(1, repeats, 1, 1)
        if remainder > 0:
            repeated = torch.cat([repeated, features[:, :remainder, :, :]], dim=1)
        return repeated


@register_strategy
class RGBToS213(BandAdaptationStrategy):
    """
    Given three channels organized like [B2, B3, B4], creates a 13-channel composite by
    repeating the RGB channels to fill the 13 channels.
    """

    required_bands = 13
    available_bands = 3

    def adapt(self, features: torch.Tensor) -> torch.Tensor:
        # repeat the RGB channels to create a 13-channel composite
        repeats = 13 // 3
        remainder = 13 % 3
        repeated = features.repeat(1, repeats, 1, 1)
        if remainder > 0:
            repeated = torch.cat([repeated, features[:, :remainder, :, :]], dim=1)
        return repeated


@register_strategy
class RGBToS12(BandAdaptationStrategy):
    """
    Given three channels organized like [B2, B3, B4], creates a 14-channel composite by
    repeating the RGB channels to fill the 14 channels.
    """

    required_bands = 14
    available_bands = 3

    def adapt(self, features: torch.Tensor) -> torch.Tensor:
        # repeat the RGB channels to create a 14-channel composite
        repeats = 14 // 3
        remainder = 14 % 3
        repeated = features.repeat(1, repeats, 1, 1)
        if remainder > 0:
            repeated = torch.cat([repeated, features[:, :remainder, :, :]], dim=1)
        return repeated


@register_strategy
class RGBIRToS212(BandAdaptationStrategy):
    """
    Given four channels organized like [B2, B3, B4, B8], creates a 12-channel composite by
    repeating the RGBIR channels to fill the 12 channels.
    """

    required_bands = 12
    available_bands = 4

    def adapt(self, features: torch.Tensor) -> torch.Tensor:
        # repeat the RGBIR channels to create a 12-channel composite
        repeats = 12 // 4
        remainder = 12 % 4
        repeated = features.repeat(1, repeats, 1, 1)
        if remainder > 0:
            repeated = torch.cat([repeated, features[:, :remainder, :, :]], dim=1)
        return repeated


@register_strategy
class RGBIRToS213(BandAdaptationStrategy):
    """
    Given four channels organized like [B2, B3, B4, B8], creates a 13-channel composite by
    repeating the RGBIR channels to fill the 13 channels.
    """

    required_bands = 13
    available_bands = 4

    def adapt(self, features: torch.Tensor) -> torch.Tensor:
        # repeat the RGBIR channels to create a 13-channel composite
        repeats = 13 // 4
        remainder = 13 % 4
        repeated = features.repeat(1, repeats, 1, 1)
        if remainder > 0:
            repeated = torch.cat([repeated, features[:, :remainder, :, :]], dim=1)
        return repeated


@register_strategy
class RGBIRToS12(BandAdaptationStrategy):
    """
    Given four channels organized like [B2, B3, B4, B8], creates a 14-channel composite by
    repeating the RGBIR channels to fill the 14 channels.
    """

    required_bands = 14
    available_bands = 4

    def adapt(self, features: torch.Tensor) -> torch.Tensor:
        # repeat the RGBIR channels to create a 14-channel composite
        repeats = 14 // 4
        remainder = 14 % 4
        repeated = features.repeat(1, repeats, 1, 1)
        if remainder > 0:
            repeated = torch.cat([repeated, features[:, :remainder, :, :]], dim=1)
        return repeated


@register_strategy
class SARToS212(BandAdaptationStrategy):
    """
    Given two channels organized like [VV, VH], creates a 12-channel composite by
    repeating the SAR channels to fill the 12 channels.
    """

    required_bands = 12
    available_bands = 2

    def adapt(self, features: torch.Tensor) -> torch.Tensor:
        # repeat the SAR channels to create a 12-channel composite
        repeats = 12 // 2
        remainder = 12 % 2
        repeated = features.repeat(1, repeats, 1, 1)
        if remainder > 0:
            repeated = torch.cat([repeated, features[:, :remainder, :, :]], dim=1)
        return repeated


@register_strategy
class SARToS213(BandAdaptationStrategy):
    """
    Given two channels organized like [VV, VH], creates a 13-channel composite by
    repeating the SAR channels to fill the 13 channels.
    """

    required_bands = 13
    available_bands = 2

    def adapt(self, features: torch.Tensor) -> torch.Tensor:
        # repeat the SAR channels to create a 13-channel composite
        repeats = 13 // 2
        remainder = 13 % 2
        repeated = features.repeat(1, repeats, 1, 1)
        if remainder > 0:
            repeated = torch.cat([repeated, features[:, :remainder, :, :]], dim=1)
        return repeated


@register_strategy
class SARToS12(BandAdaptationStrategy):
    """
    Given two channels organized like [VV, VH], creates a 14-channel composite by
    repeating the SAR channels to fill the 14 channels.
    """

    required_bands = 14
    available_bands = 2

    def adapt(self, features: torch.Tensor) -> torch.Tensor:
        # repeat the SAR channels to create a 14-channel composite
        repeats = 14 // 2
        remainder = 14 % 2
        repeated = features.repeat(1, repeats, 1, 1)
        if remainder > 0:
            repeated = torch.cat([repeated, features[:, :remainder, :, :]], dim=1)
        return repeated
