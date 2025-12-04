from torchgeo.datamodules import BaseDataModule
from torchgeo.datasets import NonGeoDataset
from typing import Any, Callable, Dict, List, Optional
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch import Tensor


class PinNonGeoDataModule(BaseDataModule):
    """Base class for data modules lacking geospatial information.

    .. versionadded:: 0.4
    """

    def __init__(
        self,
        dataset_class: type[NonGeoDataset],
        batch_size: int = 1,
        num_workers: int = 0,
        pin_memory: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize a new NonGeoDataModule instance.

        Args:
            dataset_class: Class used to instantiate a new dataset.
            batch_size: Size of each mini-batch.
            num_workers: Number of workers for parallel data loading.
            **kwargs: Additional keyword arguments passed to ``dataset_class``
        """
        super().__init__(dataset_class, batch_size, num_workers, **kwargs)

        # Collation
        self.collate_fn = default_collate

        self.pin_memory = pin_memory

    def setup(self, stage: str) -> None:
        """Set up datasets.

        Called at the beginning of fit, validate, test, or predict. During distributed
        training, this method is called from every process across all the nodes. Setting
        state here is recommended.

        Args:
            stage: Either 'fit', 'validate', 'test', or 'predict'.
        """
        if stage in ["fit"]:
            self.train_dataset = self.dataset_class(  # type: ignore[call-arg]
                split="train", **self.kwargs
            )
        if stage in ["fit", "validate"]:
            self.val_dataset = self.dataset_class(  # type: ignore[call-arg]
                split="val", **self.kwargs
            )
        if stage in ["test"]:
            self.test_dataset = self.dataset_class(  # type: ignore[call-arg]
                split="test", **self.kwargs
            )

    def _dataloader_factory(self, split: str) -> DataLoader[dict[str, Tensor]]:
        """Implement one or more PyTorch DataLoaders.

        Args:
            split: Either 'train', 'val', 'test', or 'predict'.

        Returns:
            A collection of data loaders specifying samples.

        Raises:
            MisconfigurationException: If :meth:`setup` does not define a
                dataset or sampler, or if the dataset or sampler has length 0.
        """
        dataset = self._valid_attribute(f"{split}_dataset", "dataset")
        batch_size = self._valid_attribute(f"{split}_batch_size", "batch_size")

        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=split == "train",
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            persistent_workers=self.num_workers > 0,
            pin_memory=self.pin_memory,
            prefetch_factor=1 if self.num_workers > 0 else None,
            timeout=30 if self.num_workers > 0 else 0,
        )

    def train_dataloader(self) -> DataLoader[dict[str, Tensor]]:
        """Implement one or more PyTorch DataLoaders for training.

        Returns:
            A collection of data loaders specifying training samples.

        Raises:
            MisconfigurationException: If :meth:`setup` does not define a
                dataset, or if the dataset has length 0.
        """
        return self._dataloader_factory("train")

    def val_dataloader(self) -> DataLoader[dict[str, Tensor]]:
        """Implement one or more PyTorch DataLoaders for validation.

        Returns:
            A collection of data loaders specifying validation samples.

        Raises:
            MisconfigurationException: If :meth:`setup` does not define a
                dataset, or if the dataset has length 0.
        """
        return self._dataloader_factory("val")

    def test_dataloader(self) -> DataLoader[dict[str, Tensor]]:
        """Implement one or more PyTorch DataLoaders for testing.

        Returns:
            A collection of data loaders specifying testing samples.

        Raises:
            MisconfigurationException: If :meth:`setup` does not define a
                dataset, or if the dataset has length 0.
        """
        return self._dataloader_factory("test")

    def predict_dataloader(self) -> DataLoader[dict[str, Tensor]]:
        """Implement one or more PyTorch DataLoaders for prediction.

        Returns:
            A collection of data loaders specifying prediction samples.

        Raises:
            MisconfigurationException: If :meth:`setup` does not define a
                dataset, or if the dataset has length 0.
        """
        return self._dataloader_factory("predict")
