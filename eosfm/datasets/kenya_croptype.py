# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""CV4A Kenya Crop Type dataset."""

import os
import shutil
import subprocess
from collections.abc import Callable, Sequence
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
from matplotlib.figure import Figure
from rasterio.windows import Window
from torch import Tensor

from torchgeo.datasets import DatasetNotFoundError, RGBBandsMissingError, NonGeoDataset

Path = Union[str, os.PathLike[str]]


class CV4AKenyaCropType(NonGeoDataset):
    """CV4A Kenya Crop Type Competition dataset.

    The `CV4A Kenya Crop Type Competition
    <https://beta.source.coop/repositories/radiantearth/african-crops-kenya-02/>`__
    dataset was produced as part of the Crop Type Detection competition at the
    Computer Vision for Agriculture (CV4A) Workshop at the ICLR 2020 conference.
    The objective of the competition was to create a machine learning model to
    classify fields by crop type from images collected during the growing season
    by the Sentinel-2 satellites.

    See the `dataset documentation
    <https://data.source.coop/radiantearth/african-crops-kenya-02/Documentation.pdf>`__
    for details.

    Consists of 4 tiles of Sentinel 2 imagery from 13 different points in time.

    Each tile has:

    * 13 multi-band observations throughout the growing season. Each observation
      includes 12 bands from Sentinel-2 L2A product, and a cloud probability layer.
      The twelve bands are [B01, B02, B03, B04, B05, B06, B07, B08, B8A,
      B09, B11, B12] (refer to Sentinel-2 documentation for more information about
      the bands). The cloud probability layer is a product of the
      Sentinel-2 atmospheric correction algorithm (Sen2Cor) and provides an estimated
      cloud probability (0-100%) per pixel. All of the bands are mapped to a common
      10 m spatial resolution grid.
    * A raster layer indicating the crop ID for the fields in the training set.
    * A raster layer indicating field IDs for the fields (both training and test sets).
      Fields with a crop ID 0 are the test fields.

    There are 3,286 fields in the train set and 1,402 fields in the test set.

    If you use this dataset in your research, please cite the following paper:

    * https://doi.org/10.34911/RDNT.DW605X

    .. note::

       This dataset requires the following additional library to be installed:

       * `azcopy <https://github.com/Azure/azure-storage-azcopy>`_: to download the
         dataset from Source Cooperative.
    """

    url = "https://radiantearth.blob.core.windows.net/mlhub/kenya-crop-challenge"
    tiles = tuple(map(str, range(4)))
    dates = (
        "20190606",
        "20190701",
        "20190706",
        "20190711",
        "20190721",
        "20190805",
        "20190815",
        "20190825",
        "20190909",
        "20190919",
        "20190924",
        "20191004",
        "20191103",
    )
    all_bands = (
        "B01",
        "B02",
        "B03",
        "B04",
        "B05",
        "B06",
        "B07",
        "B08",
        "B8A",
        "B09",
        "B11",
        "B12",
        "CLD",
    )
    rgb_bands = ("B04", "B03", "B02")

    # Same for all tiles
    tile_height = 3035
    tile_width = 2016

    def __init__(
        self,
        root: Path = "data",
        chip_size: int = 256,
        stride: int = 128,
        bands: Sequence[str] = all_bands,
        transforms: Callable[[dict[str, Tensor]], dict[str, Tensor]] | None = None,
        download: bool = False,
    ) -> None:
        """Initialize a new CV4A Kenya Crop Type Dataset instance.

        Args:
            root: root directory where dataset can be found
            chip_size: size of chips
            stride: spacing between chips, if less than chip_size, then there
                will be overlap between chips
            bands: the subset of bands to load
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
            download: if True, download dataset and store it in the root directory

        Raises:
            AssertionError: If *bands* are invalid.
            DatasetNotFoundError: If dataset is not found and *download* is False.
        """
        assert set(bands) <= set(self.all_bands)

        self.root = root
        self.chip_size = chip_size
        self.stride = stride
        self.bands = bands
        self.transforms = transforms
        self.download = download

        self._verify()

        # Calculate the indices that we will use over all tiles
        self.chips_metadata = []
        for tile_index in range(len(self.tiles)):
            for y in [
                *list(range(0, self.tile_height - self.chip_size, stride)),
                self.tile_height - self.chip_size,
            ]:
                for x in [
                    *list(range(0, self.tile_width - self.chip_size, stride)),
                    self.tile_width - self.chip_size,
                ]:
                    self.chips_metadata.append((tile_index, y, x))

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            data, labels, field ids, and metadata at that index
        """
        tile_index, y, x = self.chips_metadata[index]
        tile = self.tiles[tile_index]

        img = self._load_image_chip(tile, y, x)
        labels, field_ids = self._load_label_chip(tile, y, x)

        sample = {
            "image": img,
            "mask": labels,
            "field_ids": field_ids,
            "tile_index": torch.tensor(tile_index),
            "x": torch.tensor(x),
            "y": torch.tensor(y),
        }

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def __len__(self) -> int:
        """Return the number of chips in the dataset.

        Returns:
            length of the dataset
        """
        return len(self.chips_metadata)

    def _load_label_chip(self, tile: str, y: int, x: int) -> tuple[Tensor, Tensor]:
        """Load label and field_id patches for a single tile at position (y, x).

        Args:
            tile: name of tile to load
            y: row offset of the chip
            x: column offset of the chip

        Returns:
            tuple of labels and field ids tensors of shape (chip_size, chip_size)
        """
        directory = os.path.join(self.root, "data", tile)
        window = Window(x, y, self.chip_size, self.chip_size)

        with rasterio.open(os.path.join(directory, f"{tile}_label.tif")) as src:
            array: np.typing.NDArray[np.int_] = src.read(1, window=window)
            labels = torch.from_numpy(array.copy())

        with rasterio.open(os.path.join(directory, f"{tile}_field_id.tif")) as src:
            array = src.read(1, window=window)
            field_ids = torch.from_numpy(array.copy())

        return labels, field_ids

    def _load_image_chip(self, tile: str, y: int, x: int) -> Tensor:
        """Load all dates and selected bands for a single chip at position (y, x).

        Args:
            tile: name of tile to load
            y: row offset of the chip
            x: column offset of the chip

        Returns:
            imagery of shape (num_dates, num_bands, chip_size, chip_size)
        """
        img = torch.zeros(
            len(self.dates),
            len(self.bands),
            self.chip_size,
            self.chip_size,
            dtype=torch.float32,
        )
        window = Window(x, y, self.chip_size, self.chip_size)

        for date_index, date in enumerate(self.dates):
            directory = os.path.join(self.root, "data", tile, date)
            for band_index, band_name in enumerate(self.bands):
                filepath = os.path.join(directory, f"{tile}_{band_name}_{date}.tif")
                with rasterio.open(filepath) as src:
                    data = src.read(1, window=window)
                    img[date_index, band_index] = torch.from_numpy(
                        data.astype(np.float32)
                    )

        return img

    def _verify(self) -> None:
        """Verify the integrity of the dataset."""
        # Check if the files already exist
        if os.path.exists(os.path.join(self.root, "FieldIds.csv")):
            return

        # Check if the user requested to download the dataset
        if not self.download:
            raise DatasetNotFoundError(self)

        # Download the dataset
        self._download()

    def _download(self) -> None:
        """Download the dataset."""
        os.makedirs(self.root, exist_ok=True)
        azcopy = shutil.which("azcopy")
        if azcopy is None:
            raise RuntimeError("azcopy executable not found in PATH")
        subprocess.run(
            [azcopy, "sync", self.url, str(self.root), "--recursive=true"], check=True
        )

    def plot(
        self,
        sample: dict[str, Tensor],
        show_titles: bool = True,
        time_step: int = 0,
        suptitle: str | None = None,
    ) -> Figure:
        """Plot a sample from the dataset.

        Args:
            sample: a sample returned by :meth:`__getitem__`
            show_titles: flag indicating whether to show titles above each panel
            time_step: time step at which to access image, beginning with 0
            suptitle: optional suptitle to use for figure

        Returns:
            a matplotlib Figure with the rendered sample

        Raises:
            RGBBandsMissingError: If *bands* does not include all RGB bands.

        .. versionadded:: 0.2
        """
        rgb_indices = []
        for band in self.rgb_bands:
            if band in self.bands:
                rgb_indices.append(self.bands.index(band))
            else:
                raise RGBBandsMissingError()

        if "prediction" in sample:
            prediction = sample["prediction"]
            n_cols = 3
        else:
            n_cols = 2

        image, mask = sample["image"], sample["mask"]

        image = image[time_step, rgb_indices].float()
        # Min-max normalisation per channel
        img_np = image.permute(1, 2, 0).numpy()
        img_min = img_np.min(axis=(0, 1), keepdims=True)
        img_max = img_np.max(axis=(0, 1), keepdims=True)
        img_np = (img_np - img_min) / np.clip(img_max - img_min, 1e-6, None)

        fig, axs = plt.subplots(nrows=1, ncols=n_cols, figsize=(10, n_cols * 5))

        axs[0].imshow(img_np)
        axs[0].axis("off")
        axs[1].imshow(mask)
        axs[1].axis("off")

        if "prediction" in sample:
            axs[2].imshow(prediction)
            axs[2].axis("off")
            if show_titles:
                axs[2].set_title("Prediction")

        if show_titles:
            axs[0].set_title("Image")
            axs[1].set_title("Mask")

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig
