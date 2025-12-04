# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Inria Aerial Image Labeling datamodule."""

from typing import Any, Callable, Dict, List, Optional
import pathlib
import os
import warnings
import logging
import re
import glob

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import rasterio
import torch
from torch import Tensor
from torch.utils.data import random_split
from rasterio.windows import Window

from .base import PinNonGeoDataModule
from torchgeo.datasets import NonGeoDataset

# Suppress noisy GDAL/rasterio warnings
logging.getLogger("rasterio._env").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)


class InriaAerial(NonGeoDataset):
    """Inria Aerial Image Labeling Dataset.

    This implementation properly handles the large 5000x5000 images by creating tiles
    in the __getitem__ method rather than as augmentations, avoiding batch size issues.

    Dataset format:
    * images are 3-channel RGB geotiffs (5000x5000)
    * masks are single-channel binary geotiffs (5000x5000): 0=not building, 1=building

    Dataset classes:
    0. Not building
    1. Building

    Train cities: Austin, Chicago, Kitsap, West Tyrol, Vienna (180 images total)
    Test cities: Bellingham, Bloomington, Innsbruck, San Francisco, East Tyrol (180 images)

    Validation split: First 5 images of each training city (idx 1-5)
    Training split: Remaining images of each training city (idx > 5)
    """

    directory = "AerialImageDataset"

    def __init__(
        self,
        root: str = "data",
        split: str = "train",
        transforms: Optional[Callable[[Dict[str, Tensor]], Dict[str, Tensor]]] = None,
        tile_size: int = 512,
        stride: Optional[int] = None,
        use_tiling: bool = True,
        bands: Optional[List[int]] = None,
    ) -> None:
        """Initialize a new InriaAerial dataset instance.

        Args:
            root: root directory where dataset can be found
            split: one of "train", "val", or "test"
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
            tile_size: size of the tiles to extract from the images
            stride: stride of the tiling. If None, uses tile_size (no overlap).
                For training with random crops, this is ignored.
            use_tiling: if True, create tiles for systematic coverage.
                If False, return random crops (for training only).
            bands: list of band indices to use (0=R, 1=G, 2=B).
                If None, uses all bands [0, 1, 2] (RGB).
                Examples: [0, 1, 2] for RGB, [2, 1, 0] for BGR.
        """
        assert split in [
            "train",
            "val",
            "test",
        ], f"Split must be one of ['train', 'val', 'test']"

        self.root = pathlib.Path(root)
        self.split = split
        self.transforms = transforms
        self.tile_size = tile_size
        self.stride = stride if stride is not None else tile_size
        self.use_tiling = use_tiling
        self.bands = bands if bands is not None else [0, 1, 2]  # Default: RGB

        # Validate bands
        assert all(
            0 <= b < 3 for b in self.bands
        ), "Band indices must be in range [0, 2]"

        self.files = self._load_files()

        if self.use_tiling:
            self.tiles = self._create_tiles()
        else:
            # For random cropping, we just need the file list
            self.tiles = None

    def _load_files(self) -> List[Dict[str, pathlib.Path]]:
        """Return the paths of the files in the dataset."""
        files = []
        split_dir = "train" if self.split in ["train", "val"] else "test"
        root_dir = self.root / self.directory / split_dir

        pattern = re.compile(r"([A-Za-z\-]+)(\d+)")

        images = sorted(glob.glob(str(root_dir / "images" / "*.tif")))

        if split_dir == "train":
            labels = sorted(glob.glob(str(root_dir / "gt" / "*.tif")))

            for img, lbl in zip(images, labels):
                if match := pattern.search(img):
                    idx = int(match.group(2))
                    # For validation, use the first 5 images of every location
                    if self.split == "train" and idx > 5:
                        files.append(
                            {
                                "image": pathlib.Path(img),
                                "mask": pathlib.Path(lbl),
                                "name": pathlib.Path(img).stem,
                            }
                        )
                    elif self.split == "val" and idx <= 5:
                        files.append(
                            {
                                "image": pathlib.Path(img),
                                "mask": pathlib.Path(lbl),
                                "name": pathlib.Path(img).stem,
                            }
                        )
        else:
            # Test split has no labels in the official dataset
            for img in images:
                files.append(
                    {
                        "image": pathlib.Path(img),
                        "mask": None,
                        "name": pathlib.Path(img).stem,
                    }
                )

        if len(files) == 0:
            raise FileNotFoundError(
                f"No files found in {root_dir}. Make sure the dataset is properly extracted."
            )

        return files

    def _create_tiles(self) -> List[Dict[str, Any]]:
        """Create tiles from the images for systematic coverage."""
        tiles = []
        for i, file_info in enumerate(self.files):
            with rasterio.open(file_info["image"]) as src:
                width, height = src.width, src.height

            for y in range(0, height - self.tile_size + 1, self.stride):
                for x in range(0, width - self.tile_size + 1, self.stride):
                    tiles.append(
                        {"file_index": i, "x": x, "y": y, "name": file_info["name"]}
                    )
        return tiles

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        """Return an index within the dataset."""
        if self.use_tiling:
            # Systematic tiling (for validation/testing)
            assert self.tiles is not None
            tile_info = self.tiles[index]
            file_info = self.files[tile_info["file_index"]]
            window = Window(tile_info["x"], tile_info["y"], self.tile_size, self.tile_size)  # type: ignore
        else:
            # Random cropping (for training)
            file_info = self.files[index % len(self.files)]

            # Get image dimensions
            with rasterio.open(file_info["image"]) as src:
                img_width, img_height = src.width, src.height

            # Random crop position
            max_x = max(0, img_width - self.tile_size)
            max_y = max(0, img_height - self.tile_size)
            x = torch.randint(0, max_x + 1, (1,)).item() if max_x > 0 else 0
            y = torch.randint(0, max_y + 1, (1,)).item() if max_y > 0 else 0

            window = Window(x, y, self.tile_size, self.tile_size)  # type: ignore

        # Load image
        with rasterio.open(file_info["image"]) as src:
            image = src.read(window=window)
            # Select bands
            image = image[self.bands, :, :]

        # Load mask if available
        if file_info["mask"] is not None:
            with rasterio.open(file_info["mask"]) as src:
                mask = src.read(1, window=window)  # Single band
                # Ensure binary: 0 or 1
                mask = np.clip(mask, 0, 1)
        else:
            # For test set without labels, create dummy mask
            mask = np.zeros((self.tile_size, self.tile_size), dtype=np.int64)

        image = torch.from_numpy(image).float()
        mask = torch.from_numpy(mask).long()

        sample = {"image": image, "mask": mask}

        if self.transforms:
            sample = self.transforms(sample)

        return sample

    def __len__(self) -> int:
        """Return the number of data points in the dataset."""
        if self.use_tiling:
            assert self.tiles is not None
            return len(self.tiles)
        else:
            # For random cropping, we can define an epoch size
            # Let's use 100 crops per image as a reasonable epoch size
            return len(self.files) * 100

    def plot(
        self,
        sample: Dict[str, Tensor],
        show_titles: bool = True,
        suptitle: Optional[str] = None,
        alpha: float = 0.5,
    ) -> Figure:
        """Plot a sample from the dataset.

        Args:
            sample: a sample returned by :meth:`__getitem__`
            show_titles: flag indicating whether to show titles above each panel
            suptitle: optional string to use as a suptitle
            alpha: opacity with which to render masks
        Returns:
            a matplotlib Figure with the rendered sample
        """
        image = sample["image"].numpy()
        mask = sample["mask"].numpy()

        # Handle different numbers of bands
        if image.shape[0] >= 3:
            # Use first 3 bands for RGB display
            image_display = image[:3]
        else:
            # If less than 3 bands, replicate to make grayscale RGB
            image_display = np.repeat(image[:1], 3, axis=0)

        # Convert from CHW to HWC for display
        image_display = np.transpose(image_display, (1, 2, 0))

        # Normalize image to 0-255 range for display
        image_display = (
            (image_display - image_display.min())
            / (image_display.max() - image_display.min() + 1e-8)
            * 255
        )
        image_display = image_display.astype(np.uint8)

        ncols = 2
        fig, axs = plt.subplots(1, ncols, figsize=(ncols * 5, 5))

        axs[0].imshow(image_display)
        axs[0].axis("off")
        axs[1].imshow(mask, cmap="gray", vmin=0, vmax=1)
        axs[1].axis("off")

        if show_titles:
            band_names = {0: "R", 1: "G", 2: "B"}
            band_str = "".join([band_names.get(b, str(b)) for b in self.bands])
            axs[0].set_title(f"Image ({band_str})")
            axs[1].set_title("Mask (Building)")

        if suptitle:
            plt.suptitle(suptitle)

        return fig


class InriaAerialDataModule(PinNonGeoDataModule):
    """LightningDataModule implementation for the InriaAerial dataset.

    This implementation properly handles tiling:
    - Training: random crops from full images
    - Validation: systematic tiling with optional overlap (first 5 images per city)
    - Testing: systematic tiling with optional overlap
    """

    def __init__(
        self,
        batch_size: int = 64,
        tile_size: int = 512,
        stride: Optional[int] = None,
        num_workers: int = 0,
        pin_memory: bool = True,
        bands: Optional[List[int]] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize a new InriaAerialDataModule instance.

        Args:
            batch_size: Size of each mini-batch.
            tile_size: Size of each tile to extract.
            stride: Stride for validation/test tiling. If None, uses tile_size.
            num_workers: Number of workers for parallel data loading.
            pin_memory: Whether to pin memory for faster GPU transfer.
            bands: list of band indices to use (0=R, 1=G, 2=B).
                If None, uses all bands [0, 1, 2] (RGB).
            **kwargs: Additional keyword arguments passed to InriaAerial.
        """
        self.tile_size = tile_size
        self.stride = stride if stride is not None else tile_size
        self.bands = bands

        super().__init__(
            InriaAerial,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            bands=bands,
            **kwargs,
        )

    def setup(self, stage: str) -> None:
        """Set up datasets.

        Args:
            stage: Either 'fit', 'validate', 'test', or 'predict'.
        """
        if stage in ["fit", "validate"]:
            # Training dataset: random crops, no tiling
            self.train_dataset = InriaAerial(
                split="train",
                tile_size=self.tile_size,
                use_tiling=False,  # Random crops for training
                **self.kwargs,
            )

            # Validation: systematic tiling
            self.val_dataset = InriaAerial(
                split="val",
                tile_size=self.tile_size,
                stride=self.stride,
                use_tiling=True,  # Systematic tiling
                **self.kwargs,
            )

        if stage == "test":
            # Test dataset: systematic tiling
            self.test_dataset = InriaAerial(
                split="test",
                tile_size=self.tile_size,
                stride=self.stride,
                use_tiling=True,  # Systematic tiling
                **self.kwargs,
            )

        if stage == "predict":
            # Predict dataset: systematic tiling
            self.predict_dataset = InriaAerial(
                split="test",
                tile_size=self.tile_size,
                stride=self.stride,
                use_tiling=True,  # Systematic tiling
                **self.kwargs,
            )
