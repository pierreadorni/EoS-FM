from torchgeo.datamodules import NonGeoDataModule
from eosfm.datasets import CV4AKenyaCropType
from typing import Any, Optional
import torch
from torch.utils.data import random_split
from .base import PinNonGeoDataModule


class CV4AKenyaCropTypeDataModule(PinNonGeoDataModule):
    """LightningDataModule implementation for the CV4AKenyaCropType dataset.

    Keeps the native [B, T, C, H, W] shape expected by TemporalEncoderDecoderFactory.
    Normalisation is applied per-band (13 values, broadcast over T and spatial dims).
    """

    def __init__(
        self,
        batch_size: int = 64,
        num_workers: int = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__(CV4AKenyaCropType, batch_size, num_workers, **kwargs)

    def on_after_batch_transfer(self, batch: Any, dataloader_idx: int) -> Any:
        """Normalise [B, T, C, H, W] images without passing them through kornia.

        Lightning calls transfer_batch_to_device before this hook, so the batch
        is already on the correct device here. For 5-D temporal tensors we bypass
        kornia entirely (it only accepts 2-D to 4-D inputs) and apply normalisation
        manually using the mean/std stored in the augmentation pipeline.
        For non-5-D tensors the default NonGeoDataModule behaviour is used.
        """
        image = batch.get("image")

        if image is None or image.ndim != 5:
            # Non-temporal batch — use the standard kornia pipeline
            return super().on_after_batch_transfer(batch, dataloader_idx)

        # 5-D temporal case: skip kornia, apply normalisation manually
        mean, std = self._get_norm_params(image.device)
        if mean is not None:
            # mean/std: [C] -> [1, 1, C, 1, 1] to broadcast over [B, T, C, H, W]
            mean = mean.view(1, 1, -1, 1, 1)
            std = std.view(1, 1, -1, 1, 1)
            batch["image"] = (image.float() - mean) / std
        else:
            batch["image"] = image.float()

        return batch

    def _get_norm_params(
        self, device: torch.device
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Extract mean and std from the kornia augmentation pipeline."""
        if not hasattr(self, "aug") or self.aug is None:
            return None, None
        try:
            import kornia.augmentation as K

            for transform in self.aug.children():
                if isinstance(transform, K.Normalize):
                    return transform.mean.to(device), transform.std.to(device)
            # AugmentationSequential may nest them
            for transform in self.aug.modules():
                if isinstance(transform, K.Normalize):
                    return transform.mean.to(device), transform.std.to(device)
        except Exception:
            pass
        return None, None

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
            chip_labels, _ = dataset._load_label_chip(tile, y, x)

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
