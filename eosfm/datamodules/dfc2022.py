from torchgeo.datasets.utils import Path
from torch import Tensor
from torchgeo.datamodules import NonGeoDataModule
from torchgeo.datasets import DFC2022
from torch.utils.data import DataLoader, random_split
from torchgeo.datasets.utils import Path


class DFC2022DataModule(NonGeoDataModule):
    """LightningDataModule implementation for the DFC2022 dataset.

    .. versionadded:: 0.3
    """

    def __init__(
        self,
        batch_size: int = 64,
        num_workers: int = 0,
        val_split_pct: float = 0.2,
        **kwargs: any,
    ) -> None:
        """Initialize a new DFC2022DataModule instance.

        Args:
            batch_size: The batch size to use in all created DataLoaders.
            num_workers: The number of workers to use in all created DataLoaders.
            val_split_pct: What percentage of the training set to use as a validation
                set.
            **kwargs: Additional keyword arguments passed to
                :class:`~torchgeo.datasets.DFC2022`.
        """
        super().__init__(DFC2022, batch_size, num_workers, **kwargs)
        self.val_split_pct = val_split_pct

    def setup(self, stage: str | None = None) -> None:
        """Set up datasets.

        Args:
            stage: Either 'fit', 'validate', 'test', or 'predict'.
        """
        if stage in ('fit', 'validate'):
            # The 'train' split is the only one with labels.
            dataset = DFC2022(split='train', **self.kwargs)
            val_len = int(len(dataset) * self.val_split_pct)
            train_len = len(dataset) - val_len
            self.train_dataset, self.val_dataset = random_split(
                dataset, [train_len, val_len]
            )
        if stage == 'test':
            # The 'val' split is the official competition test set (unlabeled).
            self.test_dataset = DFC2022(split='val', **self.kwargs)
        if stage == 'predict':
            # The 'val' split is the official competition test set (unlabeled).
            self.predict_dataset = DFC2022(split='val', **self.kwargs)

    def train_dataloader(self) -> DataLoader[dict[str, Tensor]]:
        """Return a DataLoader for training.

        Returns:
            A DataLoader for training.
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[dict[str, Tensor]]:
        """Return a DataLoader for validation.

        Returns:
            A DataLoader for validation.
        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[dict[str, Tensor]]:
        """Return a DataLoader for testing.

        Returns:
            A DataLoader for testing.
        """
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def predict_dataloader(self) -> DataLoader[dict[str, Tensor]]:
        """Return a DataLoader for prediction.

        Returns:
            A DataLoader for prediction.
        """
        return DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )