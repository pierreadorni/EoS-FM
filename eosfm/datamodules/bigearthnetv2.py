# # Copyright (c) Microsoft Corporation. All rights reserved.
# # Licensed under the MIT License.

# """BigEarthNet dataset."""

# import glob
# import json
# import os
# import textwrap
# from collections.abc import Callable
# from typing import ClassVar

# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# import rasterio
# import torch
# from matplotlib.colors import BoundaryNorm, ListedColormap
# from matplotlib.figure import Figure
# from matplotlib.patches import Rectangle
# from rasterio.enums import Resampling
# from torch import Tensor

# from .errors import DatasetNotFoundError
# from .geo import NonGeoDataset
# from .utils import Path, download_url, extract_archive, sort_sentinel2_bands

from torchgeo.datamodules import NonGeoDataModule
from typing import Any, List

import glob
import json
import os
import textwrap
from collections.abc import Callable
from typing import ClassVar

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import torch
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from rasterio.enums import Resampling
from torch import Tensor

from torchgeo.datasets.errors import DatasetNotFoundError
from torchgeo.datasets.geo import NonGeoDataset
from torchgeo.datasets.utils import Path, download_url, extract_archive, sort_sentinel2_bands


class BigEarthNetV2DataModule(NonGeoDataModule):
    """LightningDataModule implementation for the BigEarthNetV2 dataset.

    .. versionadded:: 0.7
    """

    def __init__(
        self,
        batch_size: int = 64,
        num_workers: int = 0,
        bands: str = 'all',
        **kwargs: Any,
    ) -> None:
        """Initialize a new BigEarthNetV2DataModule instance.

        Args:
            batch_size: Size of each mini-batch.
            num_workers: Number of workers for parallel data loading.
            bands: Bands to load. One of {s1, s2, all}.
            **kwargs: Additional keyword arguments passed to
                :class:`~torchgeo.datasets.BigEarthNetV2`.
        """
        super().__init__(BigEarthNetV2, batch_size, num_workers, **kwargs)
        self.bands = bands

    def setup(self, stage: str) -> None:
        """Set up datasets.

        Args:
            stage: Either 'fit', 'validate', 'test', or 'predict'.
        """
        if stage in ['fit']:
            self.train_dataset = BigEarthNetV2(
                split='train', bands=self.bands, **self.kwargs
            )
        if stage in ['fit', 'validate']:
            self.val_dataset = BigEarthNetV2(
                split='validation', bands=self.bands, **self.kwargs
            )
        if stage in ['test']:
            self.test_dataset = BigEarthNetV2(
                split='test', bands=self.bands, **self.kwargs
            )

    def on_after_batch_transfer(self, batch, dataloader_idx):

        if self.bands == 'all':
            batch["image"] = torch.cat(
                [batch["image_s1"], batch["image_s2"]], dim=1
            )

            del batch["image_s1"]
            del batch["image_s2"]
            

        if self.bands == 's1':
            # create 3-channel composite from Sentinel-1 bands

            batch["image"] = torch.stack(
                [batch["image"][:, 0], batch["image"][:, 1], batch["image"][:, 1]], dim=1
            )


        del batch["mask"]  # Remove mask bc we do classification, not segmentation
        batch['label'] = batch['label'].float()  # Ensure label is float for BCE loss
        return super().on_after_batch_transfer(batch, dataloader_idx)


class BigEarthNetV2(NonGeoDataset):
    """BigEarthNetV2 dataset.

    The `BigEarthNet V2 <https://bigearth.net/>`__ dataset contains improved labels, improved
    geospatial data splits and additionally pixel-level labels from CORINE Land
    Cover (CLC) map of 2018. Additionally, some problematic patches from V1 have been removed.

    If you use this dataset in your research, please cite the following paper:

    * https://arxiv.org/abs/2407.03653

    .. versionadded:: 0.7
    """

    class_set: List[str] = [
        'Urban fabric',
        'Industrial or commercial units',
        'Arable land',
        'Permanent crops',
        'Pastures',
        'Complex cultivation patterns',
        'Land principally occupied by agriculture, with significant areas of'
        ' natural vegetation',
        'Agro-forestry areas',
        'Broad-leaved forest',
        'Coniferous forest',
        'Mixed forest',
        'Natural grassland and sparsely vegetated areas',
        'Moors, heathland and sclerophyllous vegetation',
        'Transitional woodland, shrub',
        'Beaches, dunes, sands',
        'Inland wetlands',
        'Coastal wetlands',
        'Inland waters',
        'Marine waters',
    ]

    image_size = (120, 120)

    url = 'https://hf.co/datasets/torchgeo/bigearthnet/resolve/3cf3a5910a5302d449fdb8e570e5b78de24fe07f/V2/{}'

    metadata_locs: ClassVar[dict[str, dict[str, dict[str, str]]]] = {
        's1': {
            'files': {
                'BigEarthNet-S1.tar.gzaa': '039b9ce305fc6582b2c3d60d1573f5b7',
                'BigEarthNet-S1.tar.gzab': 'e94f0ea165d04992ca91d8e58e82ec6d',
            }
        },
        's2': {
            'files': {
                'BigEarthNet-S2.tar.gzaa': '94e8ed32065234d3ab46353d814778d1',
                'BigEarthNet-S2.tar.gzab': '24c223d9e36166136c13b24a27debe34',
            }
        },
        'maps': {
            'files': {'Reference_Maps.tar.gzaa': 'b0cd1f0a31b49fcbfd61d80f963e759d'}
        },
        'metadata': {'files': {'metadata.parquet': '55687065e77b6d0b0f1ff604a6e7b49c'}},
    }

    dir_file_names: ClassVar[dict[str, str]] = {
        's1': 'BigEarthNet-S1',
        's2': 'BigEarthNet-S2',
        'maps': 'Reference_Maps',
        'metadata': 'metadata.parquet',
    }

    # https://collections.sentinel-hub.com/corine-land-cover/readme.html
    # Table 1 of https://bigearth.net/static/documents/Description_BigEarthNet_v2.pdf
    clc_colors: ClassVar[dict[str, str]] = {
        'Urban fabric': '#e6004d',
        'Industrial or commercial units': '#cc4df2',
        'Arable land': '#ffffa8',
        'Permanent crops': '#e68000',
        'Pastures': '#e6e64d',
        'Complex cultivation patterns': '#ffe64d',
        'Land principally occupied by agriculture, with significant areas of natural vegetation': '#e6cc4d',
        'Agro-forestry areas': '#f2cca6',
        'Broad-leaved forest': '#80ff00',
        'Coniferous forest': '#00a600',
        'Mixed forest': '#4dff00',
        'Natural grassland and sparsely vegetated areas': '#ccf24d',
        'Moors, heathland and sclerophyllous vegetation': '#a6ff80',
        'Transitional woodland, shrub': '#a6f200',
        'Beaches, dunes, sands': '#e6e6e6',
        'Inland wetlands': '#a6a6ff',
        'Coastal wetlands': '#ccccff',
        'Inland waters': '#80f2e6',
        'Marine waters': '#e6f2ff',
    }

    clc_codes: ClassVar[dict[int, int]] = {
        111: 0,  # Continuous Urban fabric
        112: 0,  # Discontinuous Urban fabric
        121: 1,  # Industrial or commercial units
        211: 2,  # Non-irrigated arable land
        212: 2,  # Permanently irrigated land
        213: 2,  # Rice fields
        221: 3,  # Vineyards
        222: 3,  # Fruit trees and berry plantations
        223: 3,  # Olive groves
        231: 4,  # Pastures
        241: 3,  # Annual crops with permanent crops
        242: 5,  # Complex cultivation patterns
        243: 6,  # Land principally occupied by agriculture...
        244: 7,  # Agro-forestry areas
        311: 8,  # Broad-leaved forest
        312: 9,  # Coniferous forest
        313: 10,  # Mixed forest
        321: 11,  # Natural grassland
        322: 12,  # Moors and heathland
        323: 12,  # Sclerophyllous vegetation
        324: 13,  # Transitional woodland/shrub
        331: 14,  # Beaches, dunes, sands
        333: 11,  # Sparsely vegetated areas
        411: 15,  # Inland marshes
        412: 15,  # Peatbogs
        421: 16,  # Salt marshes
        422: 16,  # Salines
        511: 17,  # Water courses
        512: 17,  # Water bodies
        521: 18,  # Coastal lagoons
        522: 18,  # Estuaries
        523: 18,  # Sea and ocean
    }

    valid_splits = ('train', 'validation', 'test')

    def __init__(
        self,
        root: Path = 'data',
        split: str = 'train',
        bands: str = 'all',
        transforms: Callable[[dict[str, Tensor]], dict[str, Tensor]] | None = None,
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        """Initialize a new BigEarthNet V2 dataset instance.

        Args:
            root: root directory where dataset can be found
            split: train/val/test split to load
            bands: load Sentinel-1 bands, Sentinel-2, or both. one of {s1, s2, all}
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
            download: if True, download dataset and store it in the root directory
            checksum: if True, check the MD5 of the downloaded files (may be slow)

        Raises:
            DatasetNotFoundError: If dataset is not found and *download* is False.
            AssertionError: If *split*, or *bands*, are not valid.
        """
        assert split in self.valid_splits, f'split must be one of {self.valid_splits}'
        assert bands in ['s1', 's2', 'all']
        self.root = root
        self.split = split
        self.bands = bands
        self.transforms = transforms
        self.num_classes = 19
        self.download = download
        self.checksum = checksum

        self.class2idx = {c: i for i, c in enumerate(self.class_set)}
        self._verify()

        self.metadata_df = pd.read_parquet(os.path.join(self.root, 'metadata.parquet'))
        self.metadata_df = self.metadata_df[
            self.metadata_df['split'] == self.split
        ].reset_index(drop=True)

        # Map chosen classes to ordinal numbers, all others mapped to background class
        self.ordinal_map = torch.zeros(19)
        for corine, ordinal in self.clc_codes.items():
            self.ordinal_map[ordinal] = corine

    def __len__(self) -> int:
        """Return the number of data points in the dataset.

        Returns:
            length of the dataset
        """
        return len(self.metadata_df)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            data and label at that index
        """
        sample: dict[str, Tensor] = {}

        match self.bands:
            case 's1':
                sample['image'] = self._load_image(index, 's1')
            case 's2':
                sample['image'] = self._load_image(index, 's2')
            case 'all':
                sample['image_s1'] = self._load_image(index, 's1')
                sample['image_s2'] = self._load_image(index, 's2')

        sample['mask'] = self._load_map(index)
        sample['label'] = self._load_target(index)

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def _load_image(self, index: int, sensor: str) -> Tensor:
        """Generic image loader for both S1 and S2.

        Args:
            index: index to return
            sensor: 's1' or 's2'

        Returns:
            the sensor image
        """
        row = self.metadata_df.loc[index]
        id_field = 's1_name' if sensor == 's1' else 'patch_id'
        patch_id = row[id_field]
        if sensor == 's2':
            patch_dir = '_'.join(patch_id.split('_')[0:-2])
        else:
            patch_dir = '_'.join(patch_id.split('_')[0:-3])

        paths = glob.glob(
            os.path.join(
                self.root, self.dir_file_names[sensor], patch_dir, patch_id, '*.tif'
            )
        )

        if sensor == 's2':
            paths = sorted(paths, key=sort_sentinel2_bands)
        else:
            paths = sorted(paths)

        images = []
        for path in paths:
            with rasterio.open(path) as dataset:
                array = dataset.read(
                    indexes=1,
                    out_shape=self.image_size,
                    out_dtype='int32',
                    resampling=Resampling.bilinear,
                )
                images.append(array)

        return torch.from_numpy(np.stack(images, axis=0)).float()

    def _load_map(self, index: int) -> Tensor:
        """Load a single image.

        Args:
            index: index to return

        Returns:
            the Corine Land Cover map
        """
        row = self.metadata_df.loc[index]
        patch_id = row['patch_id']
        patch_dir = '_'.join(patch_id.split('_')[0:-2])
        path = os.path.join(
            self.root,
            self.dir_file_names['maps'],
            patch_dir,
            patch_id,
            patch_id + '_reference_map.tif',
        )
        with rasterio.open(path) as dataset:
            map = dataset.read(out_dtype='int32')

        tensor = torch.from_numpy(map)
        # remap to ordinal values
        for corine, ordinal in self.clc_codes.items():
            tensor[tensor == corine] = ordinal
        return tensor.long()

    def _load_target(self, index: int) -> Tensor:
        """Load the target mask for a single image.

        Args:
            index: index to return

        Returns:
            the target label
        """
        label_names = self.metadata_df.iloc[index]['labels']

        indices = [self.class2idx[label_names] for label_names in label_names]

        image_target = torch.zeros(self.num_classes, dtype=torch.long)
        image_target[indices] = 1
        return image_target

    def _verify(self) -> None:
        """Verify the integrity of the dataset."""
        exists = []
        for key, metadata in self.metadata_locs.items():
            exists.append(
                os.path.exists(os.path.join(self.root, self.dir_file_names[key]))
            )

        if all(exists):
            return

        # check if compressed files already exist
        exists = []
        for key, metadata in self.metadata_locs.items():
            if key == 'metadata':
                exists.append(
                    os.path.exists(os.path.join(self.root, self.dir_file_names[key]))
                )
            else:
                for fname in metadata['files']:
                    fpath = os.path.join(self.root, fname)
                    exists.append(os.path.exists(fpath))

        if all(exists):
            return

        if not self.download:
            raise DatasetNotFoundError(self)

        self._download()
        self._extract()

    def _download(self) -> None:
        """Download the required tarball parts using the URL template and md5 sums."""
        for key, meta in self.metadata_locs.items():
            for fname, md5 in meta['files'].items():
                target_path = os.path.join(self.root, fname)
                if not os.path.exists(target_path):
                    download_url(self.url.format(fname), self.root, md5)

    def _extract(self) -> None:
        """Extract the tarball parts.

        For each modality (s1, s2, maps), its parts are concatenated together and then extracted.
        """
        chunk_size = 2**15  # same as used in torchvision and ssl4eo
        for key, meta in self.metadata_locs.items():
            if key == 'metadata':
                continue
            parts = [os.path.join(self.root, f) for f in meta['files'].keys()]
            concat_path = os.path.join(self.root, self.dir_file_names[key] + '.tar.gz')
            with open(concat_path, 'wb') as outfile:
                for part in parts:
                    with open(part, 'rb') as g:
                        while chunk := g.read(chunk_size):
                            outfile.write(chunk)
            extract_archive(concat_path, self.root)

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
            suptitle: optional string to use as a suptitle

        Returns:
            a matplotlib Figure with the rendered sample
        """
        fig, axes = plt.subplots(1, 2 if self.bands != 'all' else 3, figsize=(12, 4))

        if self.bands in ['s2', 'all']:
            s2_img = sample['image_s2' if self.bands == 'all' else 'image']
            rgb = np.rollaxis(s2_img[[3, 2, 1]].numpy(), 0, 3)
            axes[0].imshow(np.clip(rgb / 2000, 0, 1))
            if show_titles:
                axes[0].set_title('Sentinel-2 RGB')
            axes[0].axis('off')

        if self.bands in ['s1', 'all']:
            idx = 0 if self.bands == 's1' else 1
            s1_img = sample['image_s1' if self.bands == 'all' else 'image']
            axes[idx].imshow(s1_img[0].numpy())
            if show_titles:
                axes[idx].set_title('Sentinel-1 VV')
            axes[idx].axis('off')

        # Handle mask plotting
        mask_idx = 1 if self.bands != 'all' else 2
        mask = sample['mask'][0].numpy()

        # Get unique ordinal labels from mask
        unique_labels = sorted(np.unique(mask))

        # Map ordinal labels to class names and colors directly
        colors = []
        class_names = []
        for label in unique_labels:
            name = self.class_set[label]  # Get class name from ordinal index
            colors.append(self.clc_colors[name])  # Get color for class name
            class_names.append(name)

        # Create custom colormap
        cmap = ListedColormap(colors)
        bounds = [*unique_labels, unique_labels[-1] + 1]
        norm = BoundaryNorm(bounds, len(colors))

        axes[mask_idx].imshow(mask, cmap=cmap, norm=norm)

        # Add legend with class names
        legend_elements = [Rectangle((0, 0), 1, 1, facecolor=color) for color in colors]
        wrapped_names = [textwrap.fill(name, width=25) for name in class_names]
        axes[mask_idx].legend(
            legend_elements,
            wrapped_names,
            loc='center left',
            bbox_to_anchor=(1, 0.5),
            fontsize='x-small',
        )
        axes[mask_idx].axis('off')

        if show_titles:
            axes[mask_idx].set_title('Land Cover Map')

        if 'label' in sample:
            label_indices = sample['label'].nonzero().squeeze(1).tolist()
            label_names = [self.class_set[idx] for idx in label_indices]
            if suptitle:
                suptitle = f'{suptitle}\nLabels: {", ".join(label_names)}'
            else:
                suptitle = f'Labels: {", ".join(label_names)}'

        if suptitle:
            plt.suptitle(suptitle)

        # Adjust layout to prevent overlap
        plt.tight_layout()

        return fig
