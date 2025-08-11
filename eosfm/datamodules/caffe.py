from torchgeo.datamodules import NonGeoDataModule
from torchgeo.datasets import CaFFe
import torch
import kornia.augmentation as K

class CaFFeDataModule(NonGeoDataModule):
    """LightningDataModule implementation for the CaFFe dataset.

    Implements the default splits that come with the dataset.

    .. versionadded:: 0.7
    """

    mean = torch.Tensor([0.5517])
    std = torch.Tensor([11.8478])

    def __init__(
        self, batch_size: int = 64, num_workers: int = 0, size: int = 512, **kwargs: any
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
        batch['image'] = batch['image'].repeat(1, 3, 1, 1)
        batch['mask'] = batch['mask_zones']
        del batch['mask_zones']
        del batch['mask_front']
  
        return super().on_after_batch_transfer(batch, dataloader_idx)