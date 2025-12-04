import os
from typing import List, Tuple, Optional
import torch
from pathlib import Path
import warnings

import numpy as np
from torch import nn
from terratorch.registry import TERRATORCH_BACKBONE_REGISTRY
import timm
from tqdm import tqdm

from .band_adaptation import ADAPTATIONS_REGISTRY


@TERRATORCH_BACKBONE_REGISTRY.register
class EosFM(nn.Module):
    """
    EosFM encoder for Earth observation data.

    This encoder is actually an aggregation of multiple backbone networks, each trained to solve a different task using a different set of input bands.

    Args:
        model_weights (str): Path to the model weights pth file.
        device (str): Device to run the model on (e.g., "cpu" or "cuda").
        in_chans (int): Number of input channels.
        freeze (bool): Whether to freeze the encoder weights during training.
        feature_fusion (str | None): The type of feature fusion to apply amongst [None, "conv1x1", "mlp", "addition", "multiplication"].
        fuse_to_mult (int): A multiple used to compute the number of output channels.
        normalize_features (bool): Whether to apply feature normalization before fusion.
        normalization_type (str): Type of normalization to use: "batch" or "layer".
        projection_layer (bool): Whether to add Conv2d 1x1 + LeakyReLU projection after normalization.
        max_encoders (int | None): If set, selects only the top-k encoders using an EncoderSelector module.
        scale_features (bool): Whether to apply feature scaling in the EncoderSelector.
        encoder_selection_mode (str): Selection mode for EncoderSelector: "topk" or "smooth".
        ablate_encoders (List[int]): Indices of encoders to ablate (0-indexed).
    """

    def __init__(
        self,
        model_weights,
        device,
        in_chans,
        freeze=False,
        feature_fusion: Optional[str] = None,
        fuse_to_mult: int = 1,
        normalize_features: bool = True,
        normalization_type: str = "batch",
        projection_layer: bool = False,
        max_encoders: Optional[int] = None,
        scale_features: bool = False,
        encoder_selection_mode: str = "topk",
        ablate_encoders: List[int] = [],
        *args,
        **kwargs,
    ) -> None:
        """Initialize the EosFM model with encoders from a specified folder."""
        super().__init__(*args, **kwargs)

        self.device = device
        self.encoders = nn.ModuleList()
        self.encoders_input_bands = []
        self.encoders_out_channels = []
        self.encoder_names = []
        self.encoder_configs = []
        self.ablate_encoders = ablate_encoders
        print(f"Ablating encoders with indices: {self.ablate_encoders}")

        self.load_weights(model_weights)

        print(f"Initializing EosFM with {in_chans} input channels")
        self.in_chans = in_chans
        self.freeze = freeze
        if freeze:
            self._apply_freeze()

        self.normalize_features = normalize_features
        self.normalization_type = normalization_type
        self.projection_layer = projection_layer

        self.out_channels = self._compute_out_channels()
        print(f"Output channels: {self.out_channels}")

        if not self.out_channels or all(od == 0 for od in self.out_channels):
            raise ValueError(
                "No encoders were loaded successfully, please make sure the input channels are compatible with EoS-FM encoders and band adaptation strategies."
            )

        # Add encoder selector if max_encoders is set
        self.encoder_selector = None
        self.max_encoders = max_encoders
        self.encoder_selection_mode = encoder_selection_mode
        if max_encoders is not None:
            print(
                f"Adding EncoderSelector with mode '{encoder_selection_mode}' to select top {max_encoders} encoders out of {len(self.encoders)} available."
            )
            self.encoder_selector = EncoderSelector(
                num_specialists=len(self.encoders),
                k=max_encoders,
                scale_features=scale_features,
                selection_mode=encoder_selection_mode,
            )

        # Add feature normalization layers per encoder
        if self.normalize_features:
            if self.normalization_type not in ["batch", "layer"]:
                raise ValueError(
                    f"Invalid normalization_type '{self.normalization_type}'. Must be 'batch' or 'layer'"
                )

            self.feature_normalizers = nn.ModuleList()
            for enc_idx in range(len(self.encoders)):
                encoder_normalizers = nn.ModuleList()
                for level_idx in range(len(self.encoders_out_channels[enc_idx])):
                    num_features = self.encoders_out_channels[enc_idx][level_idx]
                    if self.normalization_type == "batch":
                        normalizer = nn.BatchNorm2d(
                            num_features=num_features, affine=False
                        )
                    else:  # layer
                        normalizer = nn.GroupNorm(
                            num_groups=1, num_channels=num_features, affine=False
                        )
                    encoder_normalizers.append(normalizer)
                self.feature_normalizers.append(encoder_normalizers)
            print(
                f"Added {self.normalization_type}Norm feature normalization for {len(self.encoders)} encoders x {len(self.encoders_out_channels[0])} feature levels"
            )
        else:
            self.feature_normalizers = nn.ModuleList()
            for enc_idx in range(len(self.encoders)):
                encoder_normalizers = nn.ModuleList(
                    [
                        nn.Identity()
                        for _ in range(len(self.encoders_out_channels[enc_idx]))
                    ]
                )
                self.feature_normalizers.append(encoder_normalizers)

        # Add projection layers (Conv2d 1x1 + LeakyReLU) after normalization if requested
        if self.projection_layer:
            self.feature_projections = nn.ModuleList()
            for enc_idx in range(len(self.encoders)):
                encoder_projections = nn.ModuleList()
                for level_idx in range(len(self.encoders_out_channels[enc_idx])):
                    num_features = self.encoders_out_channels[enc_idx][level_idx]
                    projection = nn.Sequential(
                        nn.Conv2d(
                            in_channels=num_features,
                            out_channels=num_features,
                            kernel_size=1,
                        ),
                        nn.LeakyReLU(inplace=True),
                    )
                    encoder_projections.append(projection)
                self.feature_projections.append(encoder_projections)
            print(
                f"Added projection layers (Conv2d 1x1 + LeakyReLU) for {len(self.encoders)} encoders x {len(self.encoders_out_channels[0])} feature levels"
            )
        else:
            self.feature_projections = nn.ModuleList()
            for enc_idx in range(len(self.encoders)):
                encoder_projections = nn.ModuleList(
                    [
                        nn.Identity()
                        for _ in range(len(self.encoders_out_channels[enc_idx]))
                    ]
                )
                self.feature_projections.append(encoder_projections)

        # Add feature fusion layers
        self.feature_fusion_mode = feature_fusion
        self.fuse_to_mult = fuse_to_mult
        if self.feature_fusion_mode is None:
            self.feature_fusion = nn.ModuleList(
                [nn.Identity()] * len(self.out_channels)
            )
        elif self.feature_fusion_mode == "conv1x1":
            base_n_channels: List[int] = self.encoders_out_channels[0]
            target_n_channels = [int(bc * self.fuse_to_mult) for bc in base_n_channels]
            self.feature_fusion = nn.ModuleList()
            for i, in_channels in enumerate(self.out_channels):
                self.feature_fusion.append(
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=target_n_channels[i],
                        kernel_size=1,
                    )
                )
            self.out_channels = target_n_channels
        elif self.feature_fusion_mode == "mlp":
            base_n_channels: List[int] = self.encoders_out_channels[0]
            target_n_channels = [int(bc * self.fuse_to_mult) for bc in base_n_channels]
            self.feature_fusion = nn.ModuleList()
            for i, in_channels in enumerate(self.out_channels):
                hidden_channels = int(in_channels * 1.25)
                self.feature_fusion.append(
                    nn.Sequential(
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=hidden_channels,
                            kernel_size=1,
                        ),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(
                            in_channels=hidden_channels,
                            out_channels=target_n_channels[i],
                            kernel_size=1,
                        ),
                    )
                )
            self.out_channels = target_n_channels
        elif self.feature_fusion_mode == "addition":
            base_n_channels: List[int] = self.encoders_out_channels[0]
            for enc_channels in self.encoders_out_channels:
                if enc_channels != base_n_channels:
                    raise ValueError(
                        f"Addition fusion requires all encoders to have the same output dimensions. "
                        f"Expected {base_n_channels}, got {enc_channels}"
                    )
            self.out_channels = base_n_channels
            self.feature_fusion = nn.ModuleList(
                [nn.Identity()] * len(self.out_channels)
            )
        elif self.feature_fusion_mode == "multiplication":
            base_n_channels: List[int] = self.encoders_out_channels[0]
            for enc_channels in self.encoders_out_channels:
                if enc_channels != base_n_channels:
                    raise ValueError(
                        f"Multiplication fusion requires all encoders to have the same output dimensions. "
                        f"Expected {base_n_channels}, got {enc_channels}"
                    )
            self.out_channels = base_n_channels
            self.feature_fusion = nn.ModuleList(
                [nn.Identity()] * len(self.out_channels)
            )
        else:
            raise ValueError(
                f"Invalid feature_fusion type: {self.feature_fusion_mode}. Valid types are: [None, 'conv1x1', 'mlp', 'addition', 'multiplication']"
            )

    def load_weights(self, model_weights):
        """Load model weights from a specified path."""
        if not Path(model_weights).exists():
            raise FileNotFoundError(
                f"Model weights file {model_weights} does not exist."
            )

        checkpoint = torch.load(model_weights, map_location=self.device)
        for in_chans, state_dict, config in checkpoint:
            self.encoders_input_bands.append(in_chans)
            encoder: nn.Module = timm.create_model(
                config["backbone"].removeprefix("timm_"),
                in_chans=in_chans,
                features_only=True,
                pretrained=False,
            )  # type: ignore
            encoder.load_state_dict(state_dict)
            self.encoders.append(encoder)
            self.encoders_out_channels.append(
                self._compute_out_channels_encoder(encoder, in_chans)
            )
            self.encoder_names.append(
                config.get("encoder_name", f"encoder_{len(self.encoders)}")
            )
            self.encoder_configs.append(config)
            print(
                f"Loaded encoder: {self.encoder_names[-1]} with {in_chans} input bands"
            )
        print(
            f"Loaded encoders from pth file. Total: {len(self.encoders)} encoders loaded"
        )

    @staticmethod
    def _compute_out_channels_encoder(encoder: nn.Module, in_chans: int) -> List[int]:
        """Compute output channels for a single encoder."""
        dummy_input = torch.zeros(1, in_chans, 32, 32)
        with torch.no_grad():
            output = encoder(dummy_input)
        return [o.shape[1] for o in output]

    def encoders_info(self) -> List[dict]:
        """Get information about the loaded encoders."""
        info = []
        for i, enc in enumerate(self.encoders):
            enc_info = {
                "backbone": self.encoder_configs[i].get("backbone", "unknown"),
                "name": self.encoder_names[i],
                "input_bands": self.encoders_input_bands[i],
                "output_channels": enc.out_channels,
            }
            info.append(enc_info)
        return info

    def set_freeze(self, freeze: bool):
        """Dynamically (un)freeze encoders."""
        if self.freeze == freeze:
            return
        self.freeze = freeze
        self._apply_freeze()

    def _apply_freeze(self):
        """Apply requires_grad and mode according to freeze flag."""
        for enc in self.encoders:
            for p in enc.parameters():
                p.requires_grad = not self.freeze
            if self.freeze:
                enc.eval()
            else:
                enc.train()

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Forward pass through the model."""
        if not self.encoders:
            raise ValueError("No encoders loaded. Please check the encoders folder.")

        features: List[List[torch.Tensor]] = []
        features_encoders: List[int] = []

        n_ablated_encoders = 0
        for i, encoder in enumerate(self.encoders):
            encoder = encoder.to(self.device)
            required_bands = self.encoders_input_bands[i]
            available_bands = x.shape[1]

            x_adapted_list = [x]
            if required_bands != available_bands:
                if (required_bands, available_bands) not in ADAPTATIONS_REGISTRY:
                    continue
                band_adaptations = ADAPTATIONS_REGISTRY.get(
                    (required_bands, available_bands)
                )
                x_adapted_list = [band_adaptation().adapt(x) for band_adaptation in band_adaptations]  # type: ignore

            for x_adapted in x_adapted_list:
                if not self.freeze:
                    feature = encoder(x_adapted.to(self.device))
                else:
                    encoder.eval()
                    feature = encoder(x_adapted.to(self.device))

                if i in self.ablate_encoders:
                    n_ablated_encoders += 1
                    feature = [torch.zeros_like(f) for f in feature]

                features.append(feature)
                features_encoders.append(i)

        # Normalize features BEFORE selection
        normalized_features = []
        for feat_idx, enc_feats in enumerate(features):
            encoder_idx = features_encoders[feat_idx]
            # Apply normalization
            normalized_enc_feats = [
                self.feature_normalizers[encoder_idx][level_idx](enc_feats[level_idx])
                for level_idx in range(len(enc_feats))
            ]
            # Apply projection layer (Conv2d + LeakyReLU)
            projected_enc_feats = [
                self.feature_projections[encoder_idx][level_idx](
                    normalized_enc_feats[level_idx]
                )
                for level_idx in range(len(normalized_enc_feats))
            ]
            normalized_features.append(projected_enc_feats)
        features = normalized_features

        # Apply encoder selector after normalization
        if self.encoder_selector is not None:
            features = self.encoder_selector(features, features_encoders)

        # Apply feature fusion based on mode
        if self.feature_fusion_mode == "addition":
            features = [
                torch.sum(
                    torch.stack([features[e][i] for e in range(len(features))]), dim=0
                )
                for i in range(len(features[0]))
            ]
        elif self.feature_fusion_mode == "multiplication":
            features = [
                torch.prod(
                    torch.stack([features[e][i] for e in range(len(features))]), dim=0
                )
                for i in range(len(features[0]))
            ]
        else:
            # concatenate features from all encoders per level (default and conv1x1/mlp)
            features = [
                torch.cat([features[e][i] for e in range(len(features))], dim=1)
                for i in range(len(features[0]))
            ]
            # Apply feature fusion (conv1x1/mlp or identity)
            features = [self.feature_fusion[i](f) for i, f in enumerate(features)]

        return features  # type: ignore

    def _compute_out_channels(self) -> List[int]:
        """Compute the output channels for each feature level from all encoders."""
        if not self.encoders:
            raise ValueError("No encoders loaded. Please check the encoders folder.")

        out_channels = []
        n_different_inputs = 0
        valid_encoders_indices = []

        for i, encoder in enumerate(self.encoders):
            encoder = encoder.to(self.device)
            required_bands = self.encoders_input_bands[i]
            if required_bands == self.in_chans:
                n_different_inputs += 1
                out_channels += [self.encoders_out_channels[i]]
            else:
                if (required_bands, self.in_chans) not in ADAPTATIONS_REGISTRY:
                    warnings.warn(
                        f"Encoder {self.encoder_names[i]} requires {required_bands} bands but input has {self.in_chans}, and no adaptation strategy is defined. skipping encoder."
                    )
                    continue
                adaptation_strategies = ADAPTATIONS_REGISTRY.get(
                    (required_bands, self.in_chans), []
                )
                n_different_inputs += len(adaptation_strategies)
                out_channels += [self.encoders_out_channels[i]] * len(
                    adaptation_strategies
                )
            valid_encoders_indices.append(i)

        print(f"n different inputs: {n_different_inputs}")
        print(f"Valid encoder indices: {valid_encoders_indices}")

        total_out_channels = []
        num_feature_levels = max(len(channels) for channels in out_channels)
        for i in range(num_feature_levels):
            level_channels = 0
            for encoder_channels in out_channels:
                if i < len(encoder_channels):
                    level_channels += encoder_channels[i]
            total_out_channels.append(level_channels)

        return total_out_channels

    def get_encoder_weights(self) -> torch.Tensor | None:
        """
        Get encoder selection weights for regularization.

        Returns:
            Tensor of encoder weights if encoder_selector exists, None otherwise
        """
        if self.encoder_selector is not None:
            return self.encoder_selector.specialists_weights
        return None


class EncoderSelector(nn.Module):
    """Select which specialists to use for a given task.

    Instead of reordering features, this module returns all encoder features in their
    original positions, but zeros out non-selected encoders. This maintains:
    1. Positional stability - each encoder always maps to the same position
    2. Differentiability - gradients flow through the weight-based masking
    3. Simplicity - no complex reordering logic needed

    Supports two modes:
    - topk: Hard selection of top-k encoders (can cause training instability)
    - smooth: Soft weighting of all encoders (use with sparsity regularization)
    """

    def __init__(
        self,
        num_specialists: int,
        k: int,
        scale_features: bool = False,
        selection_mode: str = "topk",
    ):
        super().__init__()
        self.num_specialists = num_specialists
        self.k = k
        self.scale_features = scale_features
        self.selection_mode = selection_mode

        if selection_mode not in ["topk", "smooth"]:
            raise ValueError(
                f"Invalid selection_mode '{selection_mode}'. Must be 'topk' or 'smooth'"
            )

        self.specialists_weights = nn.Parameter(torch.ones(num_specialists))

    def forward(
        self, feature_maps: List[List[torch.Tensor]], features_encoders: List[int]
    ) -> List[List[torch.Tensor]]:
        """
        Apply selection mask to feature maps based on encoder weights.

        Args:
            feature_maps: List of feature pyramids, one per encoder input
            features_encoders: List mapping each feature map index to its source encoder index

        Returns:
            List of masked feature pyramids (same length as input, non-selected are zeroed)
        """
        weights = self.specialists_weights

        if self.selection_mode == "topk":
            topk_weights, topk_indices = torch.topk(weights, self.k)
            selection_mask = torch.zeros(
                self.num_specialists, device=weights.device, dtype=weights.dtype
            )
            selection_mask.scatter_(0, topk_indices, 1.0)

            if self.scale_features:
                weight_mask = torch.zeros_like(weights)
                weight_mask.scatter_(0, topk_indices, topk_weights)
            else:
                weight_mask = selection_mask

        else:  # selection_mode == "smooth"
            if self.scale_features:
                weight_mask = weights
            else:
                weight_mask = torch.sigmoid(weights)

        # Apply mask to all feature maps based on their source encoder
        adapted_feature_maps = []
        for i, features in enumerate(feature_maps):
            encoder_idx = features_encoders[i]
            mask_value = weight_mask[encoder_idx]
            masked_features = [f * mask_value for f in features]
            adapted_feature_maps.append(masked_features)

        return adapted_feature_maps

    def get_active_encoder_count(self, threshold: float = 0.1) -> int:
        """
        Get the number of effectively active encoders.

        Args:
            threshold: Minimum weight value to consider an encoder active

        Returns:
            Number of encoders with weights above threshold
        """
        with torch.no_grad():
            if self.selection_mode == "topk":
                return self.k
            else:
                if self.scale_features:
                    active_weights = torch.abs(self.specialists_weights)
                else:
                    active_weights = torch.sigmoid(self.specialists_weights)
                return int(torch.sum(active_weights > threshold).item())

    def freeze_specialists_weights(self):
        """Freeze the specialists weights to prevent further updates."""
        self.specialists_weights.requires_grad = False
