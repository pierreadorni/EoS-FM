# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""SEN12MS datamodule."""

from typing import Any

import torch
from torch import Tensor
from torch.utils.data import Subset

from .base import PinNonGeoDataModule
from torchgeo.datamodules.utils import group_shuffle_split


import os
from collections.abc import Callable, Sequence
from typing import ClassVar

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
from matplotlib.figure import Figure
from torch import Tensor

from torchgeo.datasets.errors import DatasetNotFoundError, RGBBandsMissingError
from torchgeo.datasets.geo import NonGeoDataset
from torchgeo.datasets.utils import Path, check_integrity, percentile_normalization


class SEN12MS(NonGeoDataset):
    """SEN12MS dataset.

    The `SEN12MS <https://doi.org/10.14459/2019mp1474000>`__ dataset contains
    180,662 patch triplets of corresponding Sentinel-1 dual-pol SAR data,
    Sentinel-2 multi-spectral images, and MODIS-derived land cover maps.
    The patches are distributed across the land masses of the Earth and
    spread over all four meteorological seasons. This is reflected by the
    dataset structure. All patches are provided in the form of 16-bit GeoTiffs
    containing the following specific information:

    * Sentinel-1 SAR: 2 channels corresponding to sigma nought backscatter
      values in dB scale for VV and VH polarization.
    * Sentinel-2 Multi-Spectral: 13 channels corresponding to the 13 spectral bands
      (B1, B2, B3, B4, B5, B6, B7, B8, B8a, B9, B10, B11, B12).
    * MODIS Land Cover: 4 channels corresponding to IGBP, LCCS Land Cover,
      LCCS Land Use, and LCCS Surface Hydrology layers.

    If you use this dataset in your research, please cite the following paper:

    * https://doi.org/10.5194/isprs-annals-IV-2-W7-153-2019

    .. note::

       This dataset can be automatically downloaded using the following bash script:

       .. code-block:: bash

          for season in 1158_spring 1868_summer 1970_fall 2017_winter
          do
              for source in lc s1 s2
              do
                  wget "ftp://m1474000:m1474000@dataserv.ub.tum.de/ROIs${season}_${source}.tar.gz"
                  tar xvzf "ROIs${season}_${source}.tar.gz"
              done
          done

          for split in train test
          do
              wget "https://raw.githubusercontent.com/schmitt-muc/SEN12MS/3a41236a28d08d253ebe2fa1a081e5e32aa7eab4/splits/${split}_list.txt"
          done

       or manually downloaded from https://dataserv.ub.tum.de/s/m1474000
       and https://github.com/schmitt-muc/SEN12MS/tree/master/splits.
       This download will likely take several hours.
    """

    BAND_SETS: ClassVar[dict[str, tuple[str, ...]]] = {
        'all': (
            'VV',
            'VH',
            'B01',
            'B02',
            'B03',
            'B04',
            'B05',
            'B06',
            'B07',
            'B08',
            'B8A',
            'B09',
            'B10',
            'B11',
            'B12',
        ),
        's1': ('VV', 'VH'),
        's2-all': (
            'B01',
            'B02',
            'B03',
            'B04',
            'B05',
            'B06',
            'B07',
            'B08',
            'B8A',
            'B09',
            'B10',
            'B11',
            'B12',
        ),
        's2-reduced': ('B02', 'B03', 'B04', 'B08', 'B10', 'B11'),
        'rgb': ('B04', 'B03', 'B02'),
    }

    band_names = (
        'VV',
        'VH',
        'B01',
        'B02',
        'B03',
        'B04',
        'B05',
        'B06',
        'B07',
        'B08',
        'B8A',
        'B09',
        'B10',
        'B11',
        'B12',
    )

    filenames = (
        'ROIs1158_spring_lc.tar.gz',
        'ROIs1158_spring_s1.tar.gz',
        'ROIs1158_spring_s2.tar.gz',
        'ROIs1868_summer_lc.tar.gz',
        'ROIs1868_summer_s1.tar.gz',
        'ROIs1868_summer_s2.tar.gz',
        'ROIs1970_fall_lc.tar.gz',
        'ROIs1970_fall_s1.tar.gz',
        'ROIs1970_fall_s2.tar.gz',
        'ROIs2017_winter_lc.tar.gz',
        'ROIs2017_winter_s1.tar.gz',
        'ROIs2017_winter_s2.tar.gz',
        'train_list.txt',
        'test_list.txt',
    )
    light_filenames = (
        'ROIs1158_spring',
        'ROIs1868_summer',
        'ROIs1970_fall',
        'ROIs2017_winter',
        'train_list.txt',
        'test_list.txt',
    )
    md5s = (
        '6e2e8fa8b8cba77ddab49fd20ff5c37b',
        'fba019bb27a08c1db96b31f718c34d79',
        'd58af2c15a16f376eb3308dc9b685af2',
        '2c5bd80244440b6f9d54957c6b1f23d4',
        '01044b7f58d33570c6b57fec28a3d449',
        '4dbaf72ecb704a4794036fe691427ff3',
        '9b126a68b0e3af260071b3139cb57cee',
        '19132e0aab9d4d6862fd42e8e6760847',
        'b8f117818878da86b5f5e06400eb1866',
        '0fa0420ef7bcfe4387c7e6fe226dc728',
        'bb8cbfc16b95a4f054a3d5380e0130ed',
        '3807545661288dcca312c9c538537b63',
        '0a68d4e1eb24f128fccdb930000b2546',
        'c7faad064001e646445c4c634169484d',
    )

    def __init__(
        self,
        root: Path = 'data',
        split: str = 'train',
        bands: Sequence[str] = BAND_SETS['all'],
        transforms: Callable[[dict[str, Tensor]], dict[str, Tensor]] | None = None,
        checksum: bool = False,
    ) -> None:
        """Initialize a new SEN12MS dataset instance.

        The ``bands`` argument allows for the subsetting of bands returned by the
        dataset. Integers in ``bands`` index into a stack of Sentinel 1 and Sentinel 2
        imagery. Indices 0 and 1 correspond to the Sentinel 1 imagery where indices 2
        through 14 correspond to the Sentinel 2 imagery.

        Args:
            root: root directory where dataset can be found
            split: one of "train" or "test"
            bands: a sequence of band indices to use where the indices correspond to the
                array index of combined Sentinel 1 and Sentinel 2
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
            checksum: if True, check the MD5 of the downloaded files (may be slow)

        Raises:
            AssertionError: if ``split`` argument is invalid
            DatasetNotFoundError: If dataset is not found.
        """
        assert split in ['train', 'test']

        self._validate_bands(bands)
        self.band_indices = torch.tensor(
            [self.band_names.index(b) for b in bands]
        ).long()
        self.bands = bands

        self.root = root
        self.split = split
        self.transforms = transforms
        self.checksum = checksum

        if (
            checksum and not self._check_integrity()
        ) or not self._check_integrity_light():
            raise DatasetNotFoundError(self)

        with open(os.path.join(self.root, split + '_list.txt')) as f:
            self.ids = [line.rstrip() for line in f.readlines()]

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            data and label at that index
        """
        filename = self.ids[index]

        lc = self._load_raster(filename, 'lc').long()
        s1 = self._load_raster(filename, 's1')
        s2 = self._load_raster(filename, 's2')

        image = torch.cat(tensors=[s1, s2], dim=0)
        image = torch.index_select(image, dim=0, index=self.band_indices)

        sample: dict[str, Tensor] = {'image': image, 'mask': lc[0]}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def __len__(self) -> int:
        """Return the number of data points in the dataset.

        Returns:
            length of the dataset
        """
        return len(self.ids)

    def _load_raster(self, filename: str, source: str) -> Tensor:
        """Load a single raster image or target.

        Args:
            filename: name of the file to load
            source: one of "lc", "s1", or "s2"

        Returns:
            the raster image or target
        """
        parts = filename.split('_')
        parts[2] = source

        with rasterio.open(
            os.path.join(
                self.root,
                '{}_{}'.format(*parts),
                '{2}_{3}'.format(*parts),
                '{}_{}_{}_{}_{}'.format(*parts),
            )
        ) as f:
            array = f.read()
            if array.dtype == np.uint16:
                array = array.astype(np.int32)
            tensor = torch.from_numpy(array)
            return tensor

    def _validate_bands(self, bands: Sequence[str]) -> None:
        """Validate list of bands.

        Args:
            bands: user-provided sequence of bands to load

        Raises:
            ValueError: if an invalid band name is provided
        """
        for band in bands:
            if band not in self.band_names:
                raise ValueError(f"'{band}' is an invalid band name.")

    def _check_integrity_light(self) -> bool:
        """Checks the integrity of the dataset structure.

        Returns:
            True if the dataset directories and split files are found, else False
        """
        for filename in self.light_filenames:
            filepath = os.path.join(self.root, filename)
            if not os.path.exists(filepath):
                return False
        return True

    def _check_integrity(self) -> bool:
        """Check integrity of dataset.

        Returns:
            True if dataset files are found and/or MD5s match, else False
        """
        for filename, md5 in zip(self.filenames, self.md5s):
            filepath = os.path.join(self.root, filename)
            if not check_integrity(filepath, md5 if self.checksum else None):
                return False
        return True

    def plot(
        self,
        sample: dict[str, Tensor],
        show_titles: bool = True,
        suptitle: str | None = None,
    ) -> Figure:
        """Plot a sample from the dataset.

        Args:
            sample: a sample returned by :meth:`__getitem__`
            show_titles: flag indicating whether to show titles above each panel
            suptitle: optional suptitle to use for figure

        Returns:
            a matplotlib Figure with the rendered sample

        Raises:
            RGBBandsMissingError: If *bands* does not include all RGB bands.

        .. versionadded:: 0.2
        """
        rgb_indices = []
        for band in self.BAND_SETS['rgb']:
            if band in self.bands:
                rgb_indices.append(self.bands.index(band))
            else:
                raise RGBBandsMissingError()

        image, mask = sample['image'][rgb_indices].numpy(), sample['mask']
        image = percentile_normalization(image)
        ncols = 2

        showing_predictions = 'prediction' in sample
        if showing_predictions:
            prediction = sample['prediction']
            ncols += 1

        fig, axs = plt.subplots(nrows=1, ncols=ncols, figsize=(10, ncols * 5))

        axs[0].imshow(np.transpose(image, (1, 2, 0)))
        axs[0].axis('off')
        axs[1].imshow(mask)
        axs[1].axis('off')

        if showing_predictions:
            axs[2].imshow(prediction)
            axs[2].axis('off')

        if show_titles:
            axs[0].set_title('Image')
            axs[1].set_title('Mask')
            if showing_predictions:
                axs[2].set_title('Prediction')

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig


class SEN12MSDataModule(PinNonGeoDataModule):
    """LightningDataModule implementation for the SEN12MS dataset.

    Implements 80/20 geographic train/val splits and uses the test split from the
    classification dataset definitions.

    Uses the Simplified IGBP scheme defined in the 2020 Data Fusion Competition. See
    https://arxiv.org/abs/2002.08254.
    """

    #: Mapping from the IGBP class definitions to the DFC2020, taken from the dataloader
    #: here: https://github.com/lukasliebel/dfc2020_baseline.
    DFC2020_CLASS_MAPPING = torch.tensor(
        [0, 1, 1, 1, 1, 1, 2, 2, 3, 3, 4, 5, 6, 7, 6, 8, 9, 10]
    )

    std = torch.tensor(
        [-25, -25, 1e4, 1e4, 1e4, 1e4, 1e4, 1e4, 1e4, 1e4, 1e4, 1e4, 1e4, 1e4, 1e4]
    )

    def __init__(
        self,
        batch_size: int = 64,
        num_workers: int = 0,
        band_set: str = 'all',
        pin_memory: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialize a new SEN12MSDataModule instance.

        Args:
            batch_size: Size of each mini-batch.
            num_workers: Number of workers for parallel data loading.
            band_set: Subset of S1/S2 bands to use. Options are: "all",
                "s1", "s2-all", "rgb", and "s2-reduced" where the "s2-reduced" set includes:
                B2, B3, B4, B8, B11, and B12.
            **kwargs: Additional keyword arguments passed to
                :class:`~torchgeo.datasets.SEN12MS`.
        """
        kwargs['bands'] = SEN12MS.BAND_SETS[band_set]

        if band_set == 's1':
            self.std = self.std[:2]
            self.std = torch.tensor([self.std[0], self.std[1], self.std[1]])  # S1 bands are VV and VH but we make a composite RGB
        elif band_set == 's2-all':
            self.std = self.std[2:]
        elif band_set == 's2-reduced':
            self.std = self.std[torch.tensor([3, 4, 5, 9, 12, 13])]
        elif band_set == 'rgb':
            self.std = self.std[torch.tensor([2, 3, 4])]

        super().__init__(SEN12MS, batch_size, num_workers, pin_memory, **kwargs)


    def setup(self, stage: str) -> None:
        """Set up datasets.

        Args:
            stage: Either 'fit', 'validate', 'test', or 'predict'.
        """
        if stage in ['fit', 'validate']:
            season_to_int = {'winter': 0, 'spring': 1000, 'summer': 2000, 'fall': 3000}

            self.dataset = SEN12MS(split='train', **self.kwargs)

            # A patch is a filename like:
            #     "ROIs{num}_{season}_s2_{scene_id}_p{patch_id}.tif"
            # This patch will belong to the scene that is uniquely identified by its
            # (season, scene_id) tuple. Because the largest scene_id is 149, we can
            # simply give each season a large number and representing a unique_scene_id
            # as (season_id + scene_id).
            scenes = []
            for scene_fn in self.dataset.ids:
                parts = scene_fn.split('_')
                season_id = season_to_int[parts[1]]
                scene_id = int(parts[3])
                scenes.append(season_id + scene_id)

            train_indices, val_indices = group_shuffle_split(
                scenes, test_size=0.2, random_state=0
            )

            self.train_dataset = Subset(self.dataset, train_indices)
            self.val_dataset = Subset(self.dataset, val_indices)
        if stage in ['test']:
            self.test_dataset = SEN12MS(split='test', **self.kwargs)


    def on_before_batch_transfer(self, batch: Any, dataloader_idx: int) -> Any:
        batch['mask'] = torch.take(self.DFC2020_CLASS_MAPPING, batch['mask'])

        if 'image' in batch and batch['image'].shape[1] == 2:  # S1 data has 2 channels (VV, VH)
            vv_channel = batch['image'][:, 0:1, :, :]  # VV channel
            vh_channel = batch['image'][:, 1:2, :, :]  # VH channel
            # Create RGB composite: (VV, VH, VH)
            batch['image'] = torch.cat([vv_channel, vh_channel, vh_channel], dim=1)
        return super().on_before_batch_transfer(batch, dataloader_idx)
