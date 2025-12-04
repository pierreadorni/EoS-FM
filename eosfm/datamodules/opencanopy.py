# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Open-Canopy datamodule."""

from typing import Any, Callable, Dict, Optional
import os
import json

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import v2
from datasets import load_dataset
import rasterio
from torchgeo.datasets.utils import Path
from torchgeo.datasets import NonGeoDataset

from .base import PinNonGeoDataModule


class OpenCanopy(NonGeoDataset):
    """Open-Canopy dataset from HuggingFace."""

    def __init__(
        self,
        root: Path = "data",
        split: str = "train",
        transforms: Optional[Callable[[Dict[str, Tensor]], Dict[str, Tensor]]] = None,
        root_dir: str = "data",
    ) -> None:
        """Initialize a new OpenCanopy dataset instance.

        Args:
            root: root directory where dataset can be cached (passed as cache_dir to HF)
            split: one of "train", "validation", or "test"
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
            cache_dir: directory to cache the HuggingFace dataset
        """
        self.root = root
        self.split = split
        self.transforms = transforms

        # Check whether the data exists at root/canopy_height
        data_path = os.path.join(root_dir, "canopy_height")
        if not os.path.exists(data_path):
            self.download(root_dir)

        # load metadata to filter by split
        with open("data/canopy_height/geometries.geojson", "r") as f:
            self.metadata = json.load(f)

        # delete all geometries that are not in the split
        self.metadata["features"] = [
            feature
            for feature in self.metadata["features"]
            if feature["properties"]["split"] == split
        ]

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        """Return an index within the dataset."""

        tile = self.metadata["features"][index]
        filename = tile["properties"]["image_name"]
        year = filename.split("_")[-1][:4]
        spot_folder = f"data/canopy_height/{year}/spot/"
        lidar_folder = f"data/canopy_height/{year}/lidar/"
        spot_path = os.path.join(spot_folder, filename)
        lidar_path = os.path.join(
            lidar_folder, "compressed_lidar_" + filename.split("_")[-1]
        )

        coords = tile["geometry"]["coordinates"]

        with rasterio.open(spot_path) as src:
            window = rasterio.windows.from_bounds(
                coords[0][0][0],
                coords[0][0][1],
                coords[0][2][0],
                coords[0][2][1],
                transform=src.transform,
            )
            rgbir = torch.Tensor(src.read(window=window))
            irrg = rgbir[[3, 0, 1], :, :]

        with rasterio.open(lidar_path) as src:
            canopy_height = torch.Tensor(src.read(1, window=window))

        sample = {"image": irrg, "mask": canopy_height}

        if self.transforms:
            sample = self.transforms(sample)

        return sample

    def __len__(self) -> int:
        """Return the number of data points in the dataset."""
        return len(self.metadata["features"])

    @staticmethod
    def download(root_dir: str) -> None:
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id="AI4Forest/Open-Canopy",
            repo_type="dataset",
            local_dir=root_dir,
        )

    def plot(
        self,
        sample: Dict[str, Tensor],
        show_titles: bool = True,
        suptitle: Optional[str] = None,
    ) -> plt.Figure:
        """Plot a sample from the dataset.

        Args:
            sample: a sample returned by :meth:`__getitem__`
            show_titles: flag indicating whether to show titles above each panel
            suptitle: optional string to use as a suptitle

        Returns:
            a matplotlib Figure with the rendered sample
        """
        image = sample["image"].numpy()
        mask = sample["mask"].numpy()

        # Assuming image is CHW and we want to display RGB
        if image.shape[0] == 3:
            image = np.moveaxis(image, 0, -1)

        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].imshow(image.astype(np.uint8))
        axs[0].axis("off")
        axs[1].imshow(mask, cmap="viridis")
        axs[1].axis("off")

        if show_titles:
            axs[0].set_title("Image")
            axs[1].set_title("Mask")

        if suptitle:
            plt.suptitle(suptitle)

        return fig


class OpenCanopyDataModule(PinNonGeoDataModule):
    """LightningDataModule implementation for the Open-Canopy dataset."""

    def __init__(
        self,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialize a new OpenCanopyDataModule instance.

        Args:
            batch_size: Size of each mini-batch.
            num_workers: Number of workers for parallel data loading.
            pin_memory: Whether to pin memory for faster GPU transfer.
            **kwargs: Additional keyword arguments passed to
                :class:`~OpenCanopy`.
        """
        super().__init__(OpenCanopy, batch_size, num_workers, pin_memory, **kwargs)

    def setup(self, stage: str) -> None:
        """Set up datasets.

        Args:
            stage: Either 'fit', 'validate', 'test', or 'predict'.
        """
        if stage in ["fit", "validate"]:
            self.train_dataset = OpenCanopy(split="train", **self.kwargs)
            self.val_dataset = OpenCanopy(split="val", **self.kwargs)
            print(
                f"Train size: {len(self.train_dataset)}, Val size: {len(self.val_dataset)}"
            )
        if stage == "test":
            self.test_dataset = OpenCanopy(split="test", **self.kwargs)
        if stage == "predict":
            # Use test split for prediction if no predict split exists
            self.predict_dataset = OpenCanopy(split="test", **self.kwargs)
