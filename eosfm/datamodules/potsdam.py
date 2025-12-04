# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Potsdam datamodule."""

from typing import Any, Callable, Dict, List, Optional
import pathlib
import os
import warnings
import logging

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import rasterio
import torch
from torch import Tensor
from torch.utils.data import random_split
from rasterio.windows import Window
from PIL import Image

from .base import PinNonGeoDataModule
from torchgeo.datasets import NonGeoDataset
from torchgeo.datasets.utils import Path, rgb_to_mask

# Suppress noisy GDAL/rasterio warnings for Potsdam TIFFs
# - Metadata inconsistencies (CPLE_AppDefined, TIFFReadDirectory)
# - Not georeferenced warnings (expected for NonGeoDataset usage)
logging.getLogger("rasterio._env").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)


class Potsdam2D(NonGeoDataset):
    """Potsdam 2D Semantic Segmentation dataset.

    This implementation properly handles the large 6000x6000 images by creating tiles
    in the __getitem__ method rather than as augmentations, avoiding batch size issues.

    Dataset format:
    * images are 4-channel geotiffs (RGBIR)
    * masks are 3-channel geotiffs with unique RGB values representing the class

    Dataset classes:
    0. Clutter/background
    1. Impervious surfaces
    2. Building
    3. Low Vegetation
    4. Tree
    5. Car
    """

    image_root = "4_Ortho_RGBIR"
    masks_root = "5_Labels_all"

    splits = {
        "train": [
            "top_potsdam_2_10",
            "top_potsdam_2_11",
            "top_potsdam_2_12",
            "top_potsdam_3_10",
            "top_potsdam_3_11",
            "top_potsdam_3_12",
            "top_potsdam_4_10",
            "top_potsdam_4_11",
            "top_potsdam_4_12",
            "top_potsdam_5_10",
            "top_potsdam_5_11",
            "top_potsdam_5_12",
            "top_potsdam_6_10",
            "top_potsdam_6_11",
            "top_potsdam_6_12",
            "top_potsdam_6_7",
            "top_potsdam_6_8",
            "top_potsdam_6_9",
            "top_potsdam_7_10",
            "top_potsdam_7_11",
            "top_potsdam_7_12",
            "top_potsdam_7_7",
            "top_potsdam_7_8",
            "top_potsdam_7_9",
        ],
        "test": [
            "top_potsdam_5_15",
            "top_potsdam_6_15",
            "top_potsdam_6_13",
            "top_potsdam_3_13",
            "top_potsdam_4_14",
            "top_potsdam_6_14",
            "top_potsdam_5_14",
            "top_potsdam_2_13",
            "top_potsdam_4_15",
            "top_potsdam_2_14",
            "top_potsdam_5_13",
            "top_potsdam_4_13",
            "top_potsdam_3_14",
            "top_potsdam_7_13",
        ],
    }

    classes = (
        "Clutter/background",
        "Impervious surfaces",
        "Building",
        "Low Vegetation",
        "Tree",
        "Car",
    )

    colormap = (
        (255, 0, 0),
        (255, 255, 255),
        (0, 0, 255),
        (0, 255, 255),
        (0, 255, 0),
        (255, 255, 0),
    )

    def __init__(
        self,
        root: Path = "data",
        split: str = "train",
        transforms: Optional[Callable[[Dict[str, Tensor]], Dict[str, Tensor]]] = None,
        tile_size: int = 512,
        stride: Optional[int] = None,
        use_tiling: bool = True,
        bands: Optional[List[int]] = None,
    ) -> None:
        """Initialize a new Potsdam2D dataset instance.

        Args:
            root: root directory where dataset can be found
            split: one of "train" or "test"
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
            tile_size: size of the tiles to extract from the images
            stride: stride of the tiling. If None, uses tile_size (no overlap).
                For training with random crops, this is ignored.
            use_tiling: if True, create tiles for systematic coverage.
                If False, return random crops (for training only).
            bands: list of band indices to use (0=R, 1=G, 2=B, 3=IR).
                If None, uses all bands [0, 1, 2, 3] (RGBIR).
                Examples: [0, 1, 2] for RGB, [3, 0, 1] for IRRG, [2, 1, 0] for BGR.
        """
        assert split in self.splits, f"Split must be one of {list(self.splits.keys())}"

        self.root = pathlib.Path(root)
        self.split = split
        self.transforms = transforms
        self.tile_size = tile_size
        self.stride = stride if stride is not None else tile_size
        self.use_tiling = use_tiling
        self.bands = bands if bands is not None else [0, 1, 2, 3]  # Default: RGBIR

        # Validate bands
        assert all(
            0 <= b < 4 for b in self.bands
        ), "Band indices must be in range [0, 3]"

        self.files = self._load_files()

        if self.use_tiling:
            self.tiles = self._create_tiles()
        else:
            # For random cropping, we just need the file list
            self.tiles = None

    def _load_files(self) -> List[Dict[str, Path]]:
        """Return the paths of the files in the dataset."""
        files = []
        for name in self.splits[self.split]:
            image = self.root / self.image_root / f"{name}_RGBIR.tif"
            mask = self.root / self.masks_root / f"{name}_label.tif"
            if image.exists() and mask.exists():
                files.append({"image": image, "mask": mask, "name": name})

        if len(files) == 0:
            raise FileNotFoundError(
                f"No files found in {self.root}. Make sure the dataset is properly extracted."
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
        # Suppress GDAL/rasterio warnings about TIFF metadata inconsistencies
        # The Potsdam TIFFs have inconsistent Photometric/ExtraSamples metadata
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*CPLE_AppDefined.*")
            warnings.filterwarnings("ignore", message=".*TIFFReadDirectory.*")
            with rasterio.open(file_info["image"]) as src:
                image = src.read(window=window)
                # Select bands
                image = image[self.bands, :, :]

        # Load mask
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*CPLE_AppDefined.*")
            warnings.filterwarnings("ignore", message=".*TIFFReadDirectory.*")
            with rasterio.open(file_info["mask"]) as src:
                mask_rgb = src.read(window=window)
                # Convert from CHW to HWC for rgb_to_mask
                mask_rgb = np.transpose(mask_rgb, (1, 2, 0))
                mask = rgb_to_mask(mask_rgb, self.colormap)

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
        axs[1].imshow(mask, cmap="tab10", vmin=0, vmax=5)
        axs[1].axis("off")

        if show_titles:
            band_names = {0: "R", 1: "G", 2: "B", 3: "IR"}
            band_str = "".join([band_names.get(b, str(b)) for b in self.bands])
            axs[0].set_title(f"Image ({band_str})")
            axs[1].set_title("Mask")

        if suptitle:
            plt.suptitle(suptitle)

        return fig


class Potsdam2DDataModule(PinNonGeoDataModule):
    """LightningDataModule implementation for the Potsdam2D dataset.

    This implementation properly handles tiling:
    - Training: random crops from full images
    - Validation/Testing: systematic tiling with optional overlap
    """

    def __init__(
        self,
        batch_size: int = 64,
        tile_size: int = 512,
        stride: Optional[int] = None,
        val_split_pct: float = 0.2,
        num_workers: int = 0,
        pin_memory: bool = True,
        bands: Optional[List[int]] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize a new Potsdam2DDataModule instance.

        Args:
            batch_size: Size of each mini-batch.
            tile_size: Size of each tile to extract.
            stride: Stride for validation/test tiling. If None, uses tile_size.
            val_split_pct: Percentage of train set to use as validation.
            num_workers: Number of workers for parallel data loading.
            pin_memory: Whether to pin memory for faster GPU transfer.
            bands: list of band indices to use (0=R, 1=G, 2=B, 3=IR).
                If None, uses all bands [0, 1, 2, 3] (RGBIR).
                Examples: [0, 1, 2] for RGB, [3, 0, 1] for IRRG.
            **kwargs: Additional keyword arguments passed to Potsdam2D.
        """
        self.tile_size = tile_size
        self.stride = stride if stride is not None else tile_size
        self.val_split_pct = val_split_pct
        self.bands = bands

        super().__init__(
            Potsdam2D,
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
            train_dataset_full = Potsdam2D(
                split="train",
                tile_size=self.tile_size,
                use_tiling=False,  # Random crops for training
                **self.kwargs,
            )

            # Split into train and val
            generator = torch.Generator().manual_seed(0)
            train_size = int((1 - self.val_split_pct) * len(Potsdam2D.splits["train"]))
            val_size = len(Potsdam2D.splits["train"]) - train_size

            # Create separate datasets for train and val
            train_files = Potsdam2D.splits["train"][:train_size]
            val_files = Potsdam2D.splits["train"][train_size:]

            # Training: random crops
            self.train_dataset = Potsdam2D(
                split="train",
                tile_size=self.tile_size,
                use_tiling=False,  # Random crops
                **self.kwargs,
            )
            # Filter to only train files
            self.train_dataset.files = [
                f for f in self.train_dataset.files if f["name"] in train_files
            ]

            # Validation: systematic tiling
            self.val_dataset = Potsdam2D(
                split="train",
                tile_size=self.tile_size,
                stride=self.stride,
                use_tiling=True,  # Systematic tiling
                **self.kwargs,
            )
            # Filter to only val files and recreate tiles
            self.val_dataset.files = [
                f for f in self.val_dataset.files if f["name"] in val_files
            ]
            self.val_dataset.tiles = self.val_dataset._create_tiles()

        if stage == "test":
            # Test dataset: systematic tiling
            self.test_dataset = Potsdam2D(
                split="test",
                tile_size=self.tile_size,
                stride=self.stride,
                use_tiling=True,  # Systematic tiling
                **self.kwargs,
            )

        if stage == "predict":
            # Predict dataset: systematic tiling
            self.predict_dataset = Potsdam2D(
                split="test",
                tile_size=self.tile_size,
                stride=self.stride,
                use_tiling=True,  # Systematic tiling
                **self.kwargs,
            )
