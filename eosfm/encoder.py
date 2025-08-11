import os
from typing import List, Tuple
import torch
from pathlib import Path
import warnings

import numpy as np
from torch import nn
from terratorch.registry import TERRATORCH_BACKBONE_REGISTRY, BACKBONE_REGISTRY
from tqdm import tqdm

from .band_adaptation import ADAPTATIONS_REGISTRY


@TERRATORCH_BACKBONE_REGISTRY.register
class EosFM(nn.Module):
    def __init__(self, encoders_folder, device, in_chans, *args, **kwargs) -> None:
        """Initialize the EosFM model with encoders from a specified folder."""
        super().__init__(*args, **kwargs)

        self.device = device
        self.encoders = nn.ModuleList()
        self.encoders_input_bands = []
        self.encoder_names = []
        self._load_encoders(encoders_folder)
        self.in_chans = in_chans
        self.out_channels = self._compute_out_channels()
        print(f"Output channels: {self.out_channels}")

    def forward(self, x: torch.Tensor) -> List[List[torch.Tensor]]:
        """Forward pass through the model."""
        if not self.encoders:
            raise ValueError("No encoders loaded. Please check the encoders folder.")

        # Assuming the input x is a batch of images
        features = []
        for i, encoder in enumerate(self.encoders):
            encoder = encoder.to(self.device)
            required_bands = self.encoders_input_bands[i]
            available_bands = x.shape[1]

            if required_bands != available_bands:
                if (required_bands, available_bands) not in ADAPTATIONS_REGISTRY:
                    warnings.warn(
                        f"Encoder {self.encoder_names[i]} requires {required_bands} bands but input has {available_bands}, and no adaptation strategy is defined. skipping encoder."
                    )
                    continue
                band_adaptation = ADAPTATIONS_REGISTRY.get(
                    (required_bands, available_bands)
                )
                x = band_adaptation[0]().adapt(x)  # type: ignore

            with torch.no_grad():
                feature = encoder(x.to(self.device))
            for i, new_feat in enumerate(feature):
                if len(features) <= i:
                    features.append(new_feat)
                else:
                    features[i] = torch.concat([features[i], new_feat], dim=1)

        return features

    def _load_encoders(self, encoders_folder: str) -> None:
        """Load encoder models from a specified folder."""
        encoders_folder_path = Path(encoders_folder)

        pbar = tqdm(
            desc="Loading encoders",
            total=len(list(encoders_folder_path.glob("*"))),
        )
        for foldername in encoders_folder_path.glob("*"):
            pbar.set_postfix({"encoder": foldername.name})
            pbar.update(1)

            versions = list(foldername.glob("*"))
            if not versions:
                warnings.warn(f"No versions found in {foldername}. Skipping.")
                continue
            last_version = sorted(versions, key=lambda x: int(x.name.split("_")[-1]))[
                -1
            ]

            checkpoints_path = last_version / "checkpoints"
            checkpoints = list(checkpoints_path.glob("*.ckpt"))
            if not checkpoints:
                warnings.warn(
                    f"No checkpoints found in {checkpoints_path}. Skipping {foldername.name}."
                )
                continue

            encoder_path = checkpoints[0]
            n_bands, encoder = self._load_encoder_from_ckpt(encoder_path)
            self.encoders.append(encoder)
            self.encoders_input_bands.append(n_bands)
            self.encoder_names.append(foldername.name)

        print(f"Loaded encoders: {self.encoder_names}")
        if not self.encoders:
            raise ValueError(f"No encoders found in {encoders_folder}")

    def _load_encoder_from_ckpt(self, encoder_path: Path) -> Tuple[int, nn.Module]:
        """Load a single encoder model from a .ckpt file."""
        if not encoder_path.exists():
            raise FileNotFoundError(f"Encoder file {encoder_path} does not exist.")

        object = torch.load(encoder_path, map_location=self.device)

        config = object.get("hyper_parameters", {})
        config_model = config["model_args"]

        state_dict = object["state_dict"]
        state_dict = {
            k: v for k, v in state_dict.items() if k.startswith("model.encoder.")
        }
        state_dict = {
            k.removeprefix("model.encoder."): v for k, v in state_dict.items()
        }

        model: torch.nn.Module = BACKBONE_REGISTRY.build(config_model["backbone"], in_chans=config_model.get("backbone_in_chans", 3), pretrained=False)  # type: ignore
        model.load_state_dict(state_dict)
        return config_model.get("backbone_in_chans", 3), model

    def _compute_out_channels(self) -> List[int]:
        """Compute the output channels for each feature level from all encoders."""
        if not self.encoders:
            raise ValueError("No encoders loaded. Please check the encoders folder.")

        out_channels = []
        valid_encoders = []
        for i, encoder in enumerate(self.encoders):
            encoder = encoder.to(self.device)
            # Check if the encoder should be used based on input channels
            required_bands = self.encoders_input_bands[i]
            if (
                required_bands != self.in_chans
                and (required_bands, self.in_chans) not in ADAPTATIONS_REGISTRY
            ):
                warnings.warn(
                    f"Encoder {self.encoder_names[i]} requires {required_bands} bands but input has {self.in_chans}, and no adaptation strategy is defined. skipping encoder."
                )
                continue
            valid_encoders.append(encoder)

            # Assuming the encoder has a method to return the number of output channels for each feature level
            if hasattr(encoder, "out_channels"):
                encoder_out_channels = encoder.out_channels
            elif hasattr(encoder, "output_shapes"):
                encoder_out_channels = [shape[1] for shape in encoder.output_shapes]
            else:
                raise AttributeError(
                    "Encoder must have 'out_channels' or 'output_shapes' attribute."
                )

            out_channels.append(encoder_out_channels)

        print(f"Valid encoders: {len(valid_encoders)}")
        print(f"Output channels: {out_channels}")

        # Sum the output channels for each feature level
        total_out_channels = []
        num_feature_levels = max(len(channels) for channels in out_channels)
        for i in range(num_feature_levels):
            level_channels = 0
            for encoder_channels in out_channels:
                if i < len(encoder_channels):
                    level_channels += encoder_channels[i]
            total_out_channels.append(level_channels)

        return total_out_channels
