#!/usr/bin/env python3
"""Export trusted specialist training runs to one standalone inference checkpoint."""

import argparse
from collections import OrderedDict
from pathlib import Path
import re

from omegaconf import OmegaConf
import torch


def plain(value):
    if OmegaConf.is_config(value):
        value = OmegaConf.to_container(value, resolve=True)
    if isinstance(value, dict):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    if type(value) in (str, int, float, bool, type(None)):
        return value
    raise TypeError(f"Unsupported checkpoint metadata: {type(value)}")


def load_specialist(path):
    """Extract a backbone without constructing a second model implementation."""
    # Training checkpoints are trusted local artifacts, unlike downloaded weights.
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = plain(checkpoint["hyper_parameters"]["model_args"])
    saved = OmegaConf.load(path.parent.parent / "config.yaml")
    bands = plain(saved["bands"])
    channels = config.get("backbone_in_chans", 3)
    if not isinstance(bands, list) or len(bands) != channels:
        raise ValueError(f"Expected {channels} band labels in {path.parent.parent}/config.yaml")
    config.update(
        encoder_name=path.parent.parent.parent.name,
        encoder_version=path.parent.parent.name,
        encoder_checkpoint=path.name,
        bands=bands,
    )
    if "normalization" in saved:
        normalization = plain(saved["normalization"])
        if any(len(normalization[key]) != channels for key in ("mean", "std")):
            raise ValueError(f"Normalization channels do not match: {path}")
        config["normalization"] = normalization
    prefix = "model.encoder."
    original = checkpoint["state_dict"]
    state = OrderedDict((key.removeprefix(prefix), value) for key, value in original.items()
                        if key.startswith(prefix))
    if not state:
        raise ValueError(f"No model.encoder weights found: {path}")
    if hasattr(original, "_metadata"):
        state._metadata = {
            key.removeprefix("model.encoder").removeprefix("."): plain(value)
            for key, value in original._metadata.items()
            if key == "model.encoder" or key.startswith(prefix)
        }
    return channels, state, config


def export(folder, output):
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    entries = []
    for specialist in sorted(folder.iterdir()):
        if not specialist.is_dir():
            continue
        versions = [path for path in specialist.iterdir()
                    if path.is_dir() and re.fullmatch(r"version_\d+", path.name)]
        if not versions:
            continue
        version = max(versions, key=lambda path: int(path.name.split("_")[-1]))
        candidates = []
        for path in (version / "checkpoints").glob("*.ckpt"):
            match = re.fullmatch(r"epoch=(\d+)-step=(\d+)", path.stem)
            if match:
                candidates.append((tuple(map(int, match.groups())), path))
        if not candidates:
            raise ValueError(f"No epoch/step checkpoints in {version}/checkpoints")
        path = max(candidates)[1]
        entries.append(load_specialist(path))
        print(f"{specialist.name}: {version.name}/{path.name}")
    if not entries:
        raise ValueError(f"No specialist runs found in {folder}")
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation protects existing release artifacts.
    with output.open("xb") as stream:
        torch.save(entries, stream, pickle_protocol=2)
    loaded = torch.load(output, map_location="cpu", weights_only=True)
    for (channels, state, config), (actual_channels, actual_state, actual_config) in zip(entries, loaded):
        if channels != actual_channels or config != actual_config or list(state) != list(actual_state):
            raise ValueError("Export metadata changed during serialization")
        for name, tensor in state.items():
            other = actual_state[name]
            if tensor.dtype != other.dtype or tensor.shape != other.shape or not torch.equal(
                tensor.contiguous().reshape(-1).view(torch.uint8),
                other.contiguous().reshape(-1).view(torch.uint8),
            ):
                raise ValueError(f"Exported tensor changed: {config['encoder_name']}/{name}")
    print(f"Exported {len(entries)} specialists to {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoders-folder", required=True, type=Path)
    parser.add_argument("--output-path", required=True, type=Path)
    args = parser.parse_args()
    export(args.encoders_folder, args.output_path)


if __name__ == "__main__":
    main()
