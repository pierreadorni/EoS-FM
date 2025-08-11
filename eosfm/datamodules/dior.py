from torchgeo.datamodules import NonGeoDataModule
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from torchgeo.datasets import DIOR
from typing import Optional, Callable

from datasets import load_dataset
class DIORDataModule(NonGeoDataModule):
    """LightningDataModule implementation for the DIOR dataset.

    .. versionadded:: 0.7
    """

    def __init__(
        self,
        root_dir: str,
        batch_size: int = 64,
        num_workers: int = 0,
        download: bool = False,
        checksum: bool = False,
        **kwargs,
    ) -> None:
        """Initialize a new DIORDataModule instance.

        Args:
            root_dir: The root directory of the dataset.
            batch_size: The batch size to use in Python Dataloaders.
            num_workers: The number of workers to use in Python Dataloaders.
            train_transforms: A function/transform that takes input sample and its target as
                entry and returns a transformed version for the train split.
            val_transforms: A function/transform that takes input sample and its target as
                entry and returns a transformed version for the validation split.
            test_transforms: A function/transform that takes input sample and its target as
                entry and returns a transformed version for the test split.
            download: If True, download dataset and store it in the root directory.
            checksum: If True, check the MD5 of the downloaded files (may be slow).
        """
        super().__init__(DIOR, batch_size, num_workers, **kwargs)
        self.root_dir = root_dir

        self.train_transforms = v2.Compose([
            v2.RandomHorizontalFlip(),
            v2.RandomVerticalFlip(),
            v2.ToTensor(),
        ])
        self.val_transforms = v2.Compose([
            v2.ToTensor(),
        ])
        self.test_transforms = self.val_transforms
        self.download = download
        self.checksum = checksum


    def setup(self, stage: Optional[str] = None) -> None:
        """Set up the DIOR dataset.

        Args:
            stage: Either "fit", "validate", "test", or "predict".
                   If stage is None, set up all splits.
        """
        if stage == "fit" or stage is None:
            self.train_dataset = DIOR(
                self.root_dir,
                split="train",
                transforms=self.train_transforms,
                download=self.download,
                checksum=self.checksum,
            )
        if stage == "fit" or stage == "validate" or stage is None:
            self.val_dataset = DIOR(
                self.root_dir,
                split="val",
                transforms=self.val_transforms,
                download=self.download,
                checksum=self.checksum,
            )
        if stage == "test" or stage is None:
            self.test_dataset = DIOR(
                self.root_dir,
                split="test",
                transforms=self.test_transforms,
                download=self.download,
                checksum=self.checksum,
            )

    def train_dataloader(self) -> DataLoader:
        """Return a DataLoader for the train split."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        """Return a DataLoader for the validation split."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader:
        """Return a DataLoader for the test split."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )
