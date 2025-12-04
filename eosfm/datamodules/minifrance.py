# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""MiniFrance datamodule."""

from typing import Any, Callable, Dict, List, Optional, Tuple
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
from torch import Tensor
from torch.utils.data import random_split
from torchvision.transforms import v2
from rasterio.windows import Window

from .base import PinNonGeoDataModule
from torchgeo.datasets import NonGeoDataset
from torchgeo.datasets.utils import Path


class MiniFrance(NonGeoDataset):
    """MiniFrance dataset."""

    def __init__(
        self,
        root: Path = "data",
        split: str = "train",
        transforms: Optional[Callable[[Dict[str, Tensor]], Dict[str, Tensor]]] = None,
        tile_size: int = 512,
        stride: int = 256,
    ) -> None:
        """Initialize a new MiniFrance dataset instance.
        Args:
            root: root directory where dataset can be found
            split: one of "train", "val", or "test"
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
            tile_size: size of the tiles to extract from the images
            stride: stride of the tiling
        """
        self.root = pathlib.Path(root)
        self.split = split
        self.transforms = transforms
        self.tile_size = tile_size
        self.stride = stride
        self.files = self._load_files()
        self.tiles = self._create_tiles()

    def _load_files(self) -> List[Dict[str, Path]]:
        """Return the paths of the files in the dataset."""
        image_dir = self.root / "labeled_training" / "labeled"
        label_dir = self.root / "labels" / "labels"

        image_files = [p for p in image_dir.glob("*/*.tif")]
        label_files = [p for p in label_dir.glob("*/*.tif")]

        get_id_from_image_path = lambda image_path: image_path.stem.split(".")[0]
        get_id_from_mask_path = lambda mask_path: mask_path.stem.split("_")[0]

        image_ids = [get_id_from_image_path(p) for p in image_files]
        mask_ids = [get_id_from_mask_path(p) for p in label_files]

        images_removed = []
        masks_removed = []

        for image_file in image_files:
            image_id = get_id_from_image_path(image_file)
            if image_id not in mask_ids:
                images_removed.append(image_file)

        for mask_file in label_files:
            mask_id = get_id_from_mask_path(mask_file)
            if mask_id not in image_ids:
                masks_removed.append(mask_file)

        images_clean = sorted([p for p in image_files if p not in images_removed])
        masks_clean = sorted([p for p in label_files if p not in masks_removed])

        files = [
            {"image": img, "mask": lbl} for img, lbl in zip(images_clean, masks_clean)
        ]
        return files

    def _create_tiles(self) -> List[Dict[str, Any]]:
        """Create tiles from the images."""
        tiles = []
        for i, file_info in enumerate(self.files):
            with rasterio.open(file_info["image"]) as src:
                width, height = src.width, src.height

            for y in range(0, height - self.tile_size + 1, self.stride):
                for x in range(0, width - self.tile_size + 1, self.stride):
                    tiles.append({"file_index": i, "x": x, "y": y})
        return tiles

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        """Return an index within the dataset."""
        tile_info = self.tiles[index]
        file_info = self.files[tile_info["file_index"]]

        window = Window(tile_info["x"], tile_info["y"], self.tile_size, self.tile_size)

        with rasterio.open(file_info["image"]) as src:
            image = src.read(window=window)

        with rasterio.open(file_info["mask"]) as src:
            mask = src.read(1, window=window)

        image = torch.from_numpy(image).float()
        mask = torch.from_numpy(mask).long()

        sample = {"image": image, "mask": mask}

        if self.transforms:
            sample = self.transforms(sample)

        return sample

    def __len__(self) -> int:
        """Return the number of data points in the dataset."""
        return len(self.tiles)

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
        axs[1].imshow(mask.astype(np.uint8), cmap="viridis")
        axs[1].axis("off")

        if show_titles:
            axs[0].set_title("Image")
            axs[1].set_title("Mask")

        if suptitle:
            plt.suptitle(suptitle)

        return fig


class MiniFranceDataModule(PinNonGeoDataModule):
    """LightningDataModule implementation for the MiniFrance dataset."""

    def __init__(
        self,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialize a new MiniFranceDataModule instance.
        Args:
            batch_size: Size of each mini-batch.
            num_workers: Number of workers for parallel data loading.
            **kwargs: Additional keyword arguments passed to
                :class:`~MiniFrance`.
        """
        super().__init__(MiniFrance, batch_size, num_workers, pin_memory, **kwargs)

    def setup(self, stage: str) -> None:
        """Set up datasets.
        Args:
            stage: Either 'fit', 'validate', 'test', or 'predict'.
        """
        if stage in ["fit", "validate"]:
            dataset = MiniFrance(split="train", **self.kwargs)
            train_size = int(0.8 * len(dataset))
            print(f"Train size: {train_size}, Val size: {len(dataset) - train_size}")
            val_size = len(dataset) - train_size
            self.train_dataset, self.val_dataset = random_split(
                dataset, [train_size, val_size]
            )
        if stage == "test":
            self.test_dataset = MiniFrance(split="test", **self.kwargs)
        if stage == "predict":
            self.predict_dataset = MiniFrance(split="predict", **self.kwargs)
