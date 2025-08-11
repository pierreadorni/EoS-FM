from torchgeo.datamodules import NonGeoDataModule
from torchgeo.datasets import CV4AKenyaCropType
from typing import Any
import torch
from torch.utils.data import random_split


class CV4AKenyaCropTypeDataModule(NonGeoDataModule):
    """LightningDataModule implementation for the CV4AKenyaCropType dataset."""

    def __init__(
        self,
        batch_size: int = 64,
        num_workers: int = 0,
        **kwargs: Any,
    ) -> None:
        """Initialize a new CV4AKenyaCropTypeDataModule instance.
        Args:
            batch_size: Size of each mini-batch.
            num_workers: Number of workers for parallel data loading.
            **kwargs: Additional keyword arguments passed to
                :class:`~torchgeo.datasets.CV4AKenyaCropType`.
        """
        super().__init__(CV4AKenyaCropType, batch_size, num_workers, **kwargs)

    def setup(self, stage: str) -> None:
        """Set up datasets.
        Args:
            stage: Either 'fit', 'validate', 'test', or 'predict'.
        """
        dataset = CV4AKenyaCropType(**self.kwargs)

        train_indices = []
        test_indices = []

        for i, (tile_index, y, x) in enumerate(dataset.chips_metadata):
            tile = dataset.tiles[tile_index]
            labels, _ = dataset._load_label_tile(tile)
            chip_labels = labels[y : y + dataset.chip_size, x : x + dataset.chip_size]

            if torch.any(chip_labels != 0):
                train_indices.append(i)
            else:
                test_indices.append(i)

        train_dataset = torch.utils.data.Subset(dataset, train_indices)
        self.test_dataset = torch.utils.data.Subset(dataset, test_indices)

        generator = torch.Generator().manual_seed(0)
        self.train_dataset, self.val_dataset = random_split(
            train_dataset, [0.8, 0.2], generator
        )

        if stage in ["predict"]:
            self.predict_dataset = self.test_dataset

