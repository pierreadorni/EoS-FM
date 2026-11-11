from typing import Any, Dict, Optional
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
from torch import Tensor
from torchgeo.datamodules import NonGeoDataModule
from .base import PinNonGeoDataModule
from torchgeo.datasets import CaFFe
import torch
import kornia.augmentation as K


class CaFFeDataModule(PinNonGeoDataModule):
    """LightningDataModule implementation for the CaFFe dataset.

    Implements the default splits that come with the dataset.

    .. versionadded:: 0.7
    """

    mean = torch.Tensor([0.5517])
    std = torch.Tensor([11.8478])

    def __init__(
        self, batch_size: int = 64, num_workers: int = 0, size: int = 512, **kwargs: Any
    ) -> None:
        """Initialize a new CaFFeDataModule instance.

        Args:
            batch_size: Size of each mini-batch.
            num_workers: Number of workers for parallel data loading.
            size: resize images of input size 512x512 to size x size
            **kwargs: Additional keyword arguments passed to
                :class:`~torchgeo.datasets.CaFFe`.
        """
        super().__init__(CaFFe, batch_size, num_workers, **kwargs)

        self.size = size

        self.train_aug = K.AugmentationSequential(
            K.Normalize(mean=self.mean, std=self.std),
            K.Resize((size, size)),
            K.RandomHorizontalFlip(p=0.5),
            K.RandomVerticalFlip(p=0.5),
            data_keys=None,
            keepdim=True,
        )

        self.aug = K.AugmentationSequential(
            K.Normalize(mean=self.mean, std=self.std),
            K.Resize((size, size)),
            data_keys=None,
            keepdim=True,
        )

    def on_after_batch_transfer(self, batch, dataloader_idx):
        """Apply augmentation to the batch after it has been transferred to the device.
        Args:
            batch: The batch of data.
            dataloader_idx: The index of the DataLoader.
        """
        # transform SAR 1 channel to composite RGB
        batch["image"] = batch["image"].repeat(1, 3, 1, 1)
        # Add mask key for terratorch.
        if "mask_zones" in batch:
            batch["mask"] = batch["mask_zones"]
        elif "mask_front" in batch:
            batch["mask"] = batch["mask_front"]

        # Remove torchgeo-specific keys to avoid passing unexpected kwargs
        # (e.g. mask_front/mask_zones) to model.forward via terratorch.
        if "mask_zones" in batch:
            del batch["mask_zones"]
        if "mask_front" in batch:
            del batch["mask_front"]

        return super().on_after_batch_transfer(batch, dataloader_idx)

    def plot(
        self,
        sample: Dict[str, Tensor],
        show_titles: bool = True,
        suptitle: Optional[str] = None,
    ) -> Figure:
        """Plot a sample from the dataset.

        Args:
            sample: a sample returned by the dataloader (with transformed keys)
            show_titles: flag indicating whether to show titles above each panel
            suptitle: optional string to use as a suptitle

        Returns:
            a matplotlib Figure with the rendered sample
        """
        image = sample["image"].cpu().numpy()
        if "mask" in sample:
            mask = sample["mask"].cpu().numpy()
        elif "mask_zones" in sample:
            mask = sample["mask_zones"].cpu().numpy()
        else:
            mask = sample["mask_front"].cpu().numpy()

        # Denormalize image for visualization (broadcasting handles channel dimension)
        image = image * self.std.numpy() + self.mean.numpy()
        image = np.clip(image, 0, 255)

        # Convert from CHW to HWC for display
        if image.shape[0] == 3:
            image = np.moveaxis(image, 0, -1)

        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].imshow(
            image.astype(np.uint8), cmap="gray" if image.shape[-1] == 1 else None
        )
        axs[0].axis("off")
        axs[1].imshow(mask, cmap="viridis")
        axs[1].axis("off")

        if show_titles:
            axs[0].set_title("SAR Image (VV-VH composite)")
            axs[1].set_title("Calving Front Zones")

        if suptitle:
            plt.suptitle(suptitle)

        return fig
