"""Custom semantic segmentation task with robust validation plotting."""

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from torch import Tensor
from torchgeo.datasets.utils import unbind_samples

from terratorch.tasks import (
    SemanticSegmentationTask as TerraTorchSemanticSegmentationTask,
)
from terratorch.tasks.segmentation_tasks import to_segmentation_prediction


class SemanticSegmentationTask(TerraTorchSemanticSegmentationTask):
    """Semantic segmentation task with safer batch handling.

    This task avoids forwarding dataset-specific metadata keys (e.g. ``mask_front``)
    to the model and makes validation plotting robust when ``filename`` is missing.
    """

    def _plot_sample_with_prediction(self, sample: dict[str, Any]):
        """Create a generic image/target/prediction visualization."""

        image = sample["image"]
        if isinstance(image, Tensor):
            image = image.detach().cpu().numpy()

        # For time series [T, C, H, W], take the first timestep
        if image.ndim == 4:
            image = image[0]

        if image.ndim == 3 and image.shape[0] in (1, 3):
            image = np.moveaxis(image, 0, -1)

        if image.ndim == 3 and image.shape[-1] >= 3:
            image = image[..., :3].astype(np.float32)
            img_min, img_max = float(image.min()), float(image.max())
            if img_max > img_min:
                image = (image - img_min) / (img_max - img_min)
        elif image.ndim == 3 and image.shape[-1] == 1:
            image = np.squeeze(image, axis=-1)

        mask = sample["mask"]
        if isinstance(mask, Tensor):
            mask = mask.detach().cpu().numpy()
        if mask.ndim == 3:
            mask = np.squeeze(mask)

        prediction = sample["prediction"]
        if isinstance(prediction, Tensor):
            prediction = prediction.detach().cpu().numpy()
        if prediction.ndim == 3:
            prediction = np.squeeze(prediction)

        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        axs[0].imshow(image, cmap="gray" if image.ndim == 2 else None)
        axs[0].set_title("Image")
        axs[0].axis("off")

        axs[1].imshow(mask, cmap="viridis")
        axs[1].set_title("Target")
        axs[1].axis("off")

        axs[2].imshow(prediction, cmap="viridis")
        axs[2].set_title("Prediction")
        axs[2].axis("off")

        return fig

    def validation_step(
        self, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        """Compute validation loss/metrics and optionally plot a sample."""
        x = batch["image"]
        y = self.squeeze_ground_truth(batch["mask"])

        model_output = self.handle_full_or_tiled_inference(
            x, self.tiled_inference_on_validation
        )

        loss = self.val_loss_handler.compute_loss(
            model_output, y, self.criterion, self.aux_loss
        )
        self.val_loss_handler.log_loss(self.log, loss_dict=loss, batch_size=y.shape[0])

        y_hat_hard = to_segmentation_prediction(model_output)
        self.val_metrics.update(y_hat_hard, y)

        if self._do_plot_samples(batch_idx):
            try:
                datamodule = self.trainer.datamodule

                image = batch["image"]
                if isinstance(image, dict):
                    rgb_modality = (
                        getattr(datamodule, "rgb_modality", None)
                        or list(image.keys())[0]
                    )
                    image = image[rgb_modality]

                plot_batch: dict[str, Any] = {
                    "image": image,
                    "mask": batch["mask"],
                    "prediction": y_hat_hard,
                }

                if "filename" in batch:
                    plot_batch["filename"] = batch["filename"]
                else:
                    batch_size = int(y_hat_hard.shape[0])
                    plot_batch["filename"] = [
                        f"sample_{batch_idx}_{i}" for i in range(batch_size)
                    ]

                for key in ["image", "mask", "prediction"]:
                    value = plot_batch[key]
                    if isinstance(value, Tensor):
                        plot_batch[key] = value.cpu()

                sample = unbind_samples(plot_batch)[0]

                fig = self._plot_sample_with_prediction(sample)

                if fig is None and hasattr(datamodule, "plot"):
                    try:
                        fig = datamodule.plot(sample, "val")
                    except TypeError:
                        fig = datamodule.plot(sample)

                if (
                    fig is None
                    and hasattr(datamodule, "val_dataset")
                    and hasattr(datamodule.val_dataset, "plot")
                ):
                    fig = datamodule.val_dataset.plot(sample)

                if fig:
                    summary_writer = self.logger.experiment
                    if hasattr(summary_writer, "add_figure"):
                        summary_writer.add_figure(
                            f"image/{batch_idx}", fig, global_step=self.global_step
                        )
                    elif hasattr(summary_writer, "log_figure"):
                        summary_writer.log_figure(
                            self.logger.run_id,
                            fig,
                            f"epoch_{self.current_epoch}_{batch_idx}.png",
                        )
            except (ValueError, AttributeError, KeyError):
                pass
            finally:
                plt.close()
