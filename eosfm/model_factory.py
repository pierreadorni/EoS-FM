"""Change Detection Model Factory for dual-temporal image inputs."""

import logging
import warnings
from torch import nn

from terratorch.models.model import (
    AuxiliaryHead,
    AuxiliaryHeadWithDecoderWithoutInstantiatedHead,
    Model,
    ModelFactory,
)
from terratorch.models.necks import Neck, build_neck_list, NeckSequential
from terratorch.models.peft_utils import get_peft_backbone
from terratorch.models.pixel_wise_model import PixelWiseModel
from terratorch.models.utils import extract_prefix_keys
from terratorch.registry import (
    BACKBONE_REGISTRY,
    DECODER_REGISTRY,
    MODEL_FACTORY_REGISTRY,
)

logger = logging.getLogger("terratorch")


def _get_backbone(backbone: str | nn.Module, **backbone_kwargs):
    """Get backbone from string or return as-is if already a module."""
    if isinstance(backbone, nn.Module):
        return backbone

    if backbone.startswith("timm_"):
        from timm import create_model

        timm_model_name = backbone.replace("timm_", "")
        pretrained = backbone_kwargs.pop("pretrained", False)
        model = create_model(
            timm_model_name,
            pretrained=pretrained,
            features_only=True,
            **backbone_kwargs,
        )
        # Add out_channels attribute if using feature_info
        if hasattr(model, "feature_info"):
            model.out_channels = model.feature_info.channels()
        return model
    else:
        return BACKBONE_REGISTRY.build(backbone, **backbone_kwargs)


def _get_decoder_and_head_kwargs(
    decoder: str | nn.Module,
    channel_list: list[int],
    decoder_kwargs: dict,
    head_kwargs: dict,
    num_classes: int | None = None,
) -> tuple[nn.Module, dict, bool]:
    """Build decoder and prepare head kwargs."""
    # if its already an nn Module, check if it includes a head
    if isinstance(decoder, nn.Module):
        if not getattr(decoder, "includes_head", False) and num_classes is not None:
            head_kwargs["num_classes"] = num_classes
        elif head_kwargs:
            msg = "Decoder already includes a head, but `head_` arguments were specified. These should be removed."
            raise ValueError(msg)
        return decoder, head_kwargs, False

    # if its not an nn module, check if the class includes a head
    try:
        decoder_includes_head = DECODER_REGISTRY.find_class(decoder).includes_head
    except AttributeError:
        msg = (
            f"Decoder {decoder} does not have an `includes_head` attribute. "
            "Falling back to the value of the registry."
        )
        logging.warning(msg)
        decoder_includes_head = DECODER_REGISTRY.find_registry(decoder).includes_head

    if num_classes is not None:
        if decoder_includes_head:
            decoder_kwargs["num_classes"] = num_classes
            if head_kwargs:
                msg = "Decoder already includes a head, but `head_` arguments were specified. These should be removed."
                raise ValueError(msg)
        else:
            head_kwargs["num_classes"] = num_classes

    return (
        DECODER_REGISTRY.build(decoder, channel_list, **decoder_kwargs),
        head_kwargs,
        decoder_includes_head,
    )


def _check_all_args_used(kwargs):
    if kwargs:
        msg = f"arguments {kwargs} were passed but not used."
        raise ValueError(msg)


@MODEL_FACTORY_REGISTRY.register
class ChangeDetectionEncoderDecoderFactory(ModelFactory):
    """Model factory for change detection tasks with dual-temporal inputs.

    This factory is similar to EncoderDecoderFactory but handles the fact that
    change detection passes two images through the encoder and concatenates their
    features before passing them to the decoder. Therefore, the decoder input channels
    are doubled compared to the backbone output channels.
    """

    def build_model(
        self,
        task: str,
        backbone: str | nn.Module,
        decoder: str | nn.Module,
        backbone_kwargs: dict | None = None,
        decoder_kwargs: dict | None = None,
        head_kwargs: dict | None = None,
        num_classes: int | None = None,
        necks: list[dict] | None = None,
        aux_decoders: list[AuxiliaryHead] | None = None,
        rescale: bool = True,
        peft_config: dict | None = None,
        **kwargs,
    ) -> Model:
        """Build a change detection model with encoder, decoder, and head.

        Args:
            task (str): Task to be performed. Should be "segmentation" for change detection.
            backbone (str | nn.Module): Backbone encoder to be used.
            decoder (str | nn.Module): Decoder to be used.
            backbone_kwargs (dict | None): Arguments for backbone instantiation.
            decoder_kwargs (dict | None): Arguments for decoder instantiation.
            head_kwargs (dict | None): Arguments for the segmentation head.
            num_classes (int | None): Number of classes for change detection.
            necks (list[dict] | None): Neck modules to apply to encoder features.
            aux_decoders (list[AuxiliaryHead] | None): Auxiliary decoder heads.
            rescale (bool): Whether to rescale output to match input size.
            peft_config (dict | None): PEFT configuration for parameter-efficient fine-tuning.
            **kwargs: Additional arguments with backbone_, decoder_, or head_ prefixes.

        Returns:
            Model: Complete change detection model.
        """
        task = task.lower()
        if task != "segmentation":
            msg = f"ChangeDetectionEncoderDecoderFactory only supports segmentation task, got {task}"
            raise ValueError(msg)

        # Extract prefixed kwargs
        if not backbone_kwargs:
            backbone_kwargs, kwargs = extract_prefix_keys(kwargs, "backbone_")
        if not decoder_kwargs:
            decoder_kwargs, kwargs = extract_prefix_keys(kwargs, "decoder_")
        if not head_kwargs:
            head_kwargs, kwargs = extract_prefix_keys(kwargs, "head_")

        # Build backbone
        backbone = _get_backbone(backbone, **backbone_kwargs)

        # Get patch size and padding
        patch_size = backbone_kwargs.get("patch_size", None)
        if patch_size is None:
            for module in backbone.modules():
                if hasattr(module, "patch_size"):
                    patch_size = module.patch_size
                    break
        padding = backbone_kwargs.get("padding", "reflect")

        # Apply PEFT if configured
        if peft_config is not None:
            if not backbone_kwargs.get("pretrained", False):
                msg = (
                    "Using PEFT without a pretrained backbone. "
                    "Check the backbone_pretrained parameter if training from scratch."
                )
                warnings.warn(msg, stacklevel=1)
            backbone = get_peft_backbone(peft_config, backbone)

        # Get backbone output channels
        try:
            out_channels = backbone.out_channels
        except AttributeError as e:
            msg = "backbone must have out_channels attribute"
            raise AttributeError(msg) from e

        # Build necks if specified
        if necks is None:
            necks = []
        neck_list, channel_list = build_neck_list(necks, out_channels)

        # DOUBLE the channels for change detection (concatenating two temporal features)
        doubled_channel_list = [c * 2 for c in channel_list]

        logger.info(f"Building change detection model:")
        logger.info(f"  Backbone: {backbone}")
        logger.info(f"  Backbone output channels: {out_channels}")
        logger.info(f"  Channels after neck: {channel_list}")
        logger.info(f"  Doubled channels for decoder: {doubled_channel_list}")
        logger.info(f"  Decoder: {decoder}")

        # Build decoder with doubled input channels
        decoder, head_kwargs, decoder_includes_head = _get_decoder_and_head_kwargs(
            decoder,
            doubled_channel_list,
            decoder_kwargs,
            head_kwargs,
            num_classes=num_classes,
        )

        # Build auxiliary decoders if specified
        if aux_decoders is None:
            _check_all_args_used(kwargs)
            return _build_change_detection_model(
                task,
                backbone,
                decoder,
                head_kwargs,
                patch_size=patch_size,
                padding=padding,
                necks=neck_list,
                decoder_includes_head=decoder_includes_head,
                rescale=rescale,
            )

        to_be_aux_decoders: list[AuxiliaryHeadWithDecoderWithoutInstantiatedHead] = []
        for aux_decoder in aux_decoders:
            args = aux_decoder.decoder_args if aux_decoder.decoder_args else {}
            aux_decoder_kwargs, args = extract_prefix_keys(args, "decoder_")
            aux_head_kwargs, args = extract_prefix_keys(args, "head_")
            aux_decoder_instance, aux_head_kwargs, aux_decoder_includes_head = (
                _get_decoder_and_head_kwargs(
                    aux_decoder.decoder,
                    doubled_channel_list,
                    aux_decoder_kwargs,
                    aux_head_kwargs,
                    num_classes=num_classes,
                )
            )
            to_be_aux_decoders.append(
                AuxiliaryHeadWithDecoderWithoutInstantiatedHead(
                    aux_decoder.name, aux_decoder_instance, aux_head_kwargs
                )
            )
            _check_all_args_used(args)

        _check_all_args_used(kwargs)

        return _build_change_detection_model(
            task,
            backbone,
            decoder,
            head_kwargs,
            patch_size=patch_size,
            padding=padding,
            necks=neck_list,
            decoder_includes_head=decoder_includes_head,
            rescale=rescale,
            auxiliary_heads=to_be_aux_decoders,
        )


def _build_change_detection_model(
    task: str,
    backbone: nn.Module,
    decoder: nn.Module,
    head_kwargs: dict,
    patch_size: int | list | None,
    padding: str,
    decoder_includes_head: bool = False,
    necks: list[Neck] | None = None,
    rescale: bool = True,
    auxiliary_heads: (
        list[AuxiliaryHeadWithDecoderWithoutInstantiatedHead] | None
    ) = None,
):
    """Build PixelWiseModel for change detection."""
    if necks:
        neck_module: nn.Module = NeckSequential(*necks)
    else:
        neck_module = None

    return PixelWiseModel(
        task,
        backbone,
        decoder,
        head_kwargs,
        patch_size=patch_size,
        padding=padding,
        decoder_includes_head=decoder_includes_head,
        neck=neck_module,
        rescale=rescale,
        auxiliary_heads=auxiliary_heads,
    )
