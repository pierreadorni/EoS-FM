"""Cloud Cover Detection datamodule."""

from typing import Any, List, Optional
import torch
from torch import Tensor
from torchgeo.datamodules import NonGeoDataModule
from torchgeo.datasets import CloudCoverDetection


class CloudCoverDetectionDataModule(NonGeoDataModule):
    """LightningDataModule for the Cloud Cover Detection dataset.

    Uses Sentinel-2 imagery with 4 bands (B02, B03, B04, B08) for cloud segmentation.
    By default, selects IRRG (indices [3, 2, 1]) for 3-channel input.

    Band mapping:
    - B02 (index 0): Blue
    - B03 (index 1): Green
    - B04 (index 2): Red
    - B08 (index 3): NIR/Infrared

    Note: This dataset only has 'train' and 'test' splits, no validation split.
    For validation, we use the test split.

    .. versionadded:: 0.4
    """

    def __init__(
        self,
        batch_size: int = 8,
        num_workers: int = 0,
        band_indices: Optional[List[int]] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize a new CloudCoverDetectionDataModule instance.

        Args:
            batch_size: Size of each mini-batch.
            num_workers: Number of workers for parallel data loading.
            band_indices: List of band indices to select. Default [3, 2, 1] for IRRG.
            **kwargs: Additional keyword arguments passed to CloudCoverDetection.
        """
        self.band_indices = (
            band_indices if band_indices is not None else [3, 2, 1]
        )  # IRRG
        super().__init__(CloudCoverDetection, batch_size, num_workers, **kwargs)

    def setup(self, stage: str) -> None:
        """Set up datasets.

        Args:
            stage: Either 'fit', 'validate', 'test', or 'predict'.
        """
        if stage in ["fit"]:
            self.train_dataset = CloudCoverDetection(split="train", **self.kwargs)
        if stage in ["fit", "validate"]:
            # Use test split for validation since no val split exists
            self.val_dataset = CloudCoverDetection(split="test", **self.kwargs)
        if stage in ["test"]:
            self.test_dataset = CloudCoverDetection(split="test", **self.kwargs)

    def on_after_batch_transfer(
        self, batch: dict[str, Tensor], dataloader_idx: int
    ) -> dict[str, Tensor]:
        """Apply band selection after batch is transferred to device.

        Args:
            batch: A batch of data.
            dataloader_idx: The index of the dataloader.

        Returns:
            A batch of data with selected bands.
        """
        # Select bands: IRRG (indices [3, 2, 1]) by default
        if "image" in batch:
            batch["image"] = batch["image"][:, self.band_indices, :, :]

        return super().on_after_batch_transfer(batch, dataloader_idx)
