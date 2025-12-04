"""Change Detection Task for dual-temporal image inputs."""

import logging
import warnings
from functools import partial
from typing import Any

import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torchgeo.datasets.utils import unbind_samples
from torchmetrics import ClasswiseWrapper, MetricCollection, Metric
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassF1Score,
    MulticlassJaccardIndex,
)

from terratorch.models.model import AuxiliaryHead, ModelOutput
from terratorch.registry import MODEL_FACTORY_REGISTRY
from terratorch.tasks.base_task import TerraTorchTask
from terratorch.tasks.loss_handler import LossHandler
from terratorch.tasks.tiled_inference import tiled_inference

BATCH_IDX_FOR_VALIDATION_PLOTTING = 10

logger = logging.getLogger("terratorch")


class CombinedLoss(nn.modules.loss._WeightedLoss):
    """Combined loss that aggregates multiple loss functions."""

    def __init__(self, losses: dict[str, nn.Module], weight: list[float] | None = None):
        super().__init__()
        self.losses = nn.ModuleDict(losses)
        if weight is not None:
            self.weight = torch.Tensor(weight)
        else:
            self.weight = None

    def forward(self, y_hat: Tensor, ground_truth: Tensor) -> dict:
        loss = {
            name: criterion(y_hat, ground_truth)
            for name, criterion in self.losses.items()
        }

        if self.weight is not None:
            # Apply provided loss weights
            loss["loss"] = (torch.stack(list(loss.values())) * self.weight).sum()
        else:
            # Sum up all losses without weighting
            loss["loss"] = torch.stack(list(loss.values())).sum()

        return loss


class BoundaryMeanIoU(Metric):
    """Boundary mIoU for multiclass segmentation.

    Computes IoU on n-pixel-wide boundary bands of prediction and target for each class,
    then aggregates (macro/micro). `ignore_index` is ignored in both pred/target.

    Metric based on https://arxiv.org/abs/2103.16562:
    Cheng, B., Girshick, R., Dollár, P., Berg, A. C., & Kirillov, A. (2021). Boundary IoU: Improving object-centric
    image segmentation evaluation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition.
    """

    full_state_update = False

    def __init__(
        self,
        num_classes: int,
        thickness: int = 2,  # boundary band half-width in pixels
        ignore_index: int | None = None,
        average: str = "macro",  # "macro" or "micro"
        include_background: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if average not in {"macro", "micro"}:
            raise ValueError("average must be 'macro' or 'micro'")

        self.num_classes = num_classes
        self.thickness = thickness
        self.ignore_index = ignore_index
        self.average = average
        self.include_background = include_background

        # accumulators across batches
        self.add_state(
            "intersections",
            default=torch.zeros(num_classes, dtype=torch.int),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "unions",
            default=torch.zeros(num_classes, dtype=torch.int),
            dist_reduce_fx="sum",
        )

    @torch.no_grad()
    def update(self, preds: Tensor, target: Tensor) -> None:
        """
        preds: (N, C, H, W) logits/probs OR (N, H, W) class indices
        target: (N, H, W) class indices
        """
        if preds.dim() == 4:
            # logits -> hard labels
            preds_idx = preds.argmax(dim=1)
        elif preds.dim() == 3:
            preds_idx = preds
        else:
            raise ValueError("preds must be (N,C,H,W) or (N,H,W)")

        if target.dim() != 3:
            raise ValueError("target must be (N,H,W)")

        k = 2 * self.thickness + 1

        # mask out ignore_index everywhere
        ignore_mask = torch.zeros_like(target, dtype=torch.bool)
        if self.ignore_index is not None:
            ignore_mask = target == self.ignore_index

        # optionally skip background class
        start_cls = 0 if self.include_background else 1

        for c in range(start_cls, self.num_classes):
            if self.ignore_index is not None and c == self.ignore_index:
                continue

            # binary masks for class c
            pred_c = (preds_idx == c).float().unsqueeze(1)  # (N,1,H,W)
            target_c = (target == c).float().unsqueeze(1)

            # compute boundary bands via morphological gradient on binary maps
            dil_pred = F.max_pool2d(
                pred_c, kernel_size=k, stride=1, padding=self.thickness
            )
            ero_pred = 1.0 - F.max_pool2d(
                1.0 - pred_c, kernel_size=k, stride=1, padding=self.thickness
            )
            bnd_pred = (dil_pred - ero_pred).clamp_min(0.0) > 0.5  # (N,1,H,W) -> bool

            dil_target = F.max_pool2d(
                target_c, kernel_size=k, stride=1, padding=self.thickness
            )
            ero_target = 1.0 - F.max_pool2d(
                1.0 - target_c, kernel_size=k, stride=1, padding=self.thickness
            )
            bnd_target = (dil_target - ero_target).clamp_min(0.0) > 0.5

            # Apply ignore mask
            bnd_pred = bnd_pred & ~ignore_mask
            bnd_target = bnd_target & ~ignore_mask

            # IoU on boundary bands
            inter = (bnd_pred & bnd_target).sum()
            union = (bnd_pred | bnd_target).sum()

            # Accumulate
            self.intersections[c] += inter
            self.unions[c] += union

    def compute(self) -> Tensor:
        eps = 1e-9
        valid = self.unions > 0  # classes that had any boundary pixels at all

        # exclude classes with no boundary (union==0) from macro average
        if self.average == "macro":
            denom = valid.sum()
            iou_per_class = self.intersections[valid] / (self.unions[valid] + eps)
            return iou_per_class.mean() if denom > 0 else torch.tensor(0.0)
        else:  # micro
            inter_sum = self.intersections[valid].sum()
            union_sum = self.unions[valid].sum() + eps
            return inter_sum / union_sum


def to_segmentation_prediction(y: ModelOutput) -> Tensor:
    y_hat = y.output
    return y_hat.argmax(dim=1)


def init_loss(
    loss: str, ignore_index: int = None, class_weights: list = None
) -> nn.Module:
    if loss == "ce":
        return nn.CrossEntropyLoss(ignore_index=ignore_index, weight=class_weights)
    elif loss == "jaccard":
        if ignore_index is not None:
            raise RuntimeError(
                f"Jaccard loss does not support ignore_index, but found non-None value of {ignore_index}."
            )
        return smp.losses.JaccardLoss(mode="multiclass")
    elif loss == "focal":
        return smp.losses.FocalLoss(
            "multiclass", ignore_index=ignore_index, normalized=True
        )
    elif loss == "dice":
        return smp.losses.DiceLoss("multiclass", ignore_index=ignore_index)
    elif loss == "lovasz":
        return smp.losses.LovaszLoss(mode="multiclass", ignore_index=ignore_index)
    else:
        raise ValueError(
            f"Loss type '{loss}' is not valid. Currently, supports 'ce', 'jaccard', 'dice', 'focal', or 'lovasz' loss."
        )


class ChangeDetectionTask(TerraTorchTask):
    """Change Detection Task for dual-temporal images.

    This task handles inputs with two images (image1, image2) and a mask.
    It passes both images through the encoder separately, concatenates their features,
    and passes the concatenated features through the decoder.

    This class is adapted from SemanticSegmentationTask to handle the dual-image input format
    required for change detection datasets like LEVIR-CD+.
    """

    def __init__(
        self,
        model_args: dict,
        model_factory: str | None = None,
        model: torch.nn.Module | None = None,
        loss: str | list[str] | dict[str, float] = "ce",
        aux_heads: list[AuxiliaryHead] | None = None,
        aux_loss: dict[str, float] | None = None,
        class_weights: list[float] | None = None,
        ignore_index: int | None = None,
        lr: float = 0.001,
        optimizer: str | None = None,
        optimizer_hparams: dict | None = None,
        scheduler: str | None = None,
        scheduler_hparams: dict | None = None,
        freeze_backbone: bool = False,
        freeze_decoder: bool = False,
        freeze_head: bool = False,
        plot_on_val: bool | int = 10,
        class_names: list[str] | None = None,
        tiled_inference_parameters: dict = None,
        test_dataloaders_names: list[str] | None = None,
        lr_overrides: dict[str, float] | None = None,
        output_on_inference: str | list[str] = "prediction",
        output_most_probable: bool = True,
        path_to_record_metrics: str = None,
        tiled_inference_on_testing: bool = False,
        tiled_inference_on_validation: bool = False,
    ) -> None:
        """Constructor for Change Detection Task.

        Args are identical to SemanticSegmentationTask - see that class for detailed documentation.
        """
        self.tiled_inference_parameters = tiled_inference_parameters
        self.aux_loss = aux_loss
        self.aux_heads = aux_heads

        if model is not None and model_factory is not None:
            logger.warning(
                "A model_factory and a model was provided. The model_factory is ignored."
            )
        if model is None and model_factory is None:
            raise ValueError(
                "A model_factory or a model (torch.nn.Module) must be provided."
            )

        if model_factory and model is None:
            self.model_factory = MODEL_FACTORY_REGISTRY.build(model_factory)

        super().__init__(
            task="segmentation",
            tiled_inference_on_testing=tiled_inference_on_testing,
            tiled_inference_on_validation=tiled_inference_on_validation,
            path_to_record_metrics=path_to_record_metrics,
        )

        if model is not None:
            # Custom model
            self.model = model

        self.train_loss_handler = LossHandler(self.train_metrics.prefix)
        self.test_loss_handler: list[LossHandler] = []
        for metrics in self.test_metrics:
            self.test_loss_handler.append(LossHandler(metrics.prefix))
        self.val_loss_handler = LossHandler(self.val_metrics.prefix)
        self.monitor = f"{self.val_metrics.prefix}loss"
        self.plot_on_val = int(plot_on_val)
        self.output_on_inference = output_on_inference

        # When the user decides to use `output_most_probable` as `False` in
        # order to output the probabilities instead of the prediction.
        if not output_most_probable:
            warnings.warn(
                "The argument `output_most_probable` is deprecated and will be replaced with `output_on_inference='probabilities'`.",
                stacklevel=1,
            )
            output_on_inference = "probabilities"

        # Processing the `output_on_inference` argument.
        self.output_prediction = lambda y: (y.argmax(dim=1), "pred")
        self.output_logits = lambda y: (y, "logits")
        self.output_probabilities = lambda y: (torch.nn.Softmax()(y), "probabilities")

        # The possible methods to define outputs.
        self.operation_map = {
            "prediction": self.output_prediction,
            "logits": self.output_logits,
            "probabilities": self.output_probabilities,
        }

        # `output_on_inference` can be a list or a string.
        if isinstance(output_on_inference, list):
            list_of_selectors = ()
            for var in output_on_inference:
                if var in self.operation_map:
                    list_of_selectors += (self.operation_map[var],)
                else:
                    raise ValueError(
                        f"Option {var} is not supported. It must be in ['prediction', 'logits', 'probabilities']"
                    )

            if not len(list_of_selectors):
                raise ValueError(
                    "The list of selectors for the output is empty, please, provide a valid value for `output_on_inference`"
                )

            self.select_classes = lambda y: [op(y) for op in list_of_selectors]
        elif isinstance(output_on_inference, str):
            self.select_classes = self.operation_map[output_on_inference]
        else:
            raise ValueError(
                f"The value {output_on_inference} isn't supported for `output_on_inference`."
            )

    def squeeze_ground_truth(self, x):
        return torch.squeeze(x, 1)

    def forward_change_detection(
        self, image1: Tensor, image2: Tensor, **kwargs
    ) -> ModelOutput:
        """Forward pass for change detection with dual images.

        Args:
            image1: First temporal image (T1)
            image2: Second temporal image (T2)
            **kwargs: Additional arguments to pass to the model

        Returns:
            ModelOutput with change detection predictions
        """
        # Get the encoder, decoder, head from the PixelWiseModel
        encoder = self.model.encoder
        decoder = self.model.decoder
        head = self.model.head

        # Pass both images through the encoder
        features1 = encoder(image1)
        features2 = encoder(image2)

        # Concatenate features at each pyramid level
        # features1 and features2 are lists of tensors (one per pyramid level)
        concatenated_features = []
        for f1, f2 in zip(features1, features2):
            # Concatenate along channel dimension
            concatenated_features.append(torch.cat([f1, f2], dim=1))

        # Apply neck if present (PixelWiseModel creates a lambda that requires image_size)
        if hasattr(self.model, "neck") and self.model.neck is not None:
            # The neck lambda expects (features, image_size)
            concatenated_features = self.model.neck(
                concatenated_features, image1.shape[-2:]
            )

        # Pass concatenated features through decoder
        decoder_output = decoder(concatenated_features)

        # Pass through segmentation head
        output = head(decoder_output)

        # Rescale output to match input size if needed
        if hasattr(self.model, "rescale") and self.model.rescale:
            if output.shape[-2:] != image1.shape[-2:]:
                output = torch.nn.functional.interpolate(
                    output, size=image1.shape[-2:], mode="bilinear", align_corners=False
                )

        # Return in ModelOutput format
        return ModelOutput(output=output)

    def configure_losses(self) -> None:
        """Initialize the loss criterion.

        Raises:
            ValueError: If *loss* is invalid.
        """
        loss = self.hparams["loss"]
        ignore_index = self.hparams["ignore_index"]

        class_weights = (
            torch.Tensor(self.hparams["class_weights"])
            if self.hparams["class_weights"] is not None
            else None
        )

        if isinstance(loss, str):
            # Single loss
            self.criterion = init_loss(
                loss, ignore_index=ignore_index, class_weights=class_weights
            )
        elif isinstance(loss, nn.Module):
            # Custom loss
            self.criterion = loss
        elif isinstance(loss, list):
            # List of losses with equal weights
            losses = {
                l: init_loss(l, ignore_index=ignore_index, class_weights=class_weights)
                for l in loss
            }
            self.criterion = CombinedLoss(losses=losses)
        elif isinstance(loss, dict):
            # Weighted losses
            loss_names, weights = list(loss.keys()), list(loss.values())
            losses = {
                l: init_loss(l, ignore_index=ignore_index, class_weights=class_weights)
                for l in loss_names
            }
            self.criterion = CombinedLoss(losses=losses, weight=weights)
        else:
            raise ValueError(
                f"The loss type {loss} isn't supported. Provide loss as string, list, or "
                f"dict[name, weights]."
            )

    def configure_metrics(self) -> None:
        """Initialize the performance metrics."""
        num_classes: int = self.hparams["model_args"]["num_classes"]
        ignore_index: int = self.hparams["ignore_index"]
        class_names = self.hparams["class_names"]
        metrics = MetricCollection(
            {
                "mIoU": MulticlassJaccardIndex(
                    num_classes=num_classes,
                    ignore_index=ignore_index,
                    average="macro",
                ),
                "mIoU_Micro": MulticlassJaccardIndex(
                    num_classes=num_classes,
                    ignore_index=ignore_index,
                    average="micro",
                ),
                "F1_Score": MulticlassF1Score(
                    num_classes=num_classes,
                    ignore_index=ignore_index,
                    average="macro",
                ),
                "Accuracy": MulticlassAccuracy(
                    num_classes=num_classes,
                    ignore_index=ignore_index,
                    average="macro",
                ),
                "Pixel_Accuracy": MulticlassAccuracy(
                    num_classes=num_classes,
                    ignore_index=ignore_index,
                    average="micro",
                ),
                "Boundary_mIoU": BoundaryMeanIoU(
                    num_classes=num_classes,
                    thickness=2,
                    ignore_index=ignore_index,
                    average="macro",
                    include_background=False,
                ),
                "IoU": ClasswiseWrapper(
                    MulticlassJaccardIndex(
                        num_classes=num_classes, ignore_index=ignore_index, average=None
                    ),
                    labels=class_names,
                    prefix="IoU_",
                ),
                "Class_Accuracy": ClasswiseWrapper(
                    MulticlassAccuracy(
                        num_classes=num_classes,
                        ignore_index=ignore_index,
                        average=None,
                    ),
                    labels=class_names,
                    prefix="Class_Accuracy_",
                ),
            }
        )
        self.train_metrics = metrics.clone(prefix="train/")
        self.val_metrics = metrics.clone(prefix="val/")
        if self.hparams["test_dataloaders_names"] is not None:
            self.test_metrics = nn.ModuleList(
                [
                    metrics.clone(prefix=f"test/{dl_name}/")
                    for dl_name in self.hparams["test_dataloaders_names"]
                ]
            )
        else:
            self.test_metrics = nn.ModuleList([metrics.clone(prefix="test/")])

    def training_step(
        self, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> Tensor:
        """Compute the train loss and additional metrics.

        Args:
            batch: The output of your DataLoader with image1, image2, and mask.
            batch_idx: Integer displaying index of this batch.
            dataloader_idx: Index of the current dataloader.
        """
        image1 = batch["image1"]
        image2 = batch["image2"]
        y = self.squeeze_ground_truth(batch["mask"])

        # Forward pass with both images
        model_output: ModelOutput = self.forward_change_detection(image1, image2)

        # Compute loss
        loss = self.train_loss_handler.compute_loss(
            model_output, y, self.criterion, self.aux_loss
        )
        self.train_loss_handler.log_loss(
            self.log, loss_dict=loss, batch_size=y.shape[0]
        )

        # Compute predictions and update metrics
        y_hat_hard = to_segmentation_prediction(model_output)
        self.train_metrics.update(y_hat_hard, y)

        return loss["loss"]

    def validation_step(
        self, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        """Compute the validation loss and additional metrics.

        Args:
            batch: The output of your DataLoader with image1, image2, and mask.
            batch_idx: Integer displaying index of this batch.
            dataloader_idx: Index of the current dataloader.
        """
        image1 = batch["image1"]
        image2 = batch["image2"]
        y = self.squeeze_ground_truth(batch["mask"])

        # Forward pass with both images
        model_output = self.forward_change_detection(image1, image2)

        # Compute loss
        loss = self.val_loss_handler.compute_loss(
            model_output, y, self.criterion, self.aux_loss
        )
        self.val_loss_handler.log_loss(self.log, loss_dict=loss, batch_size=y.shape[0])

        # Compute predictions and update metrics
        y_hat_hard = to_segmentation_prediction(model_output)
        self.val_metrics.update(y_hat_hard, y)

        # Plotting (optional - simplified for change detection)
        if self._do_plot_samples(batch_idx):
            try:
                datamodule = self.trainer.datamodule
                batch["prediction"] = y_hat_hard

                # Use image2 for visualization (or could concatenate both)
                batch["image"] = image2

                for key in ["image", "mask", "prediction"]:
                    if key in batch:
                        batch[key] = batch[key].cpu()

                sample = unbind_samples(batch)[0]
                fig = (
                    datamodule.val_dataset.plot(sample)
                    if hasattr(datamodule.val_dataset, "plot")
                    else datamodule.plot(sample, "val")
                )

                if fig:
                    summary_writer = self.logger.experiment
                    summary_writer.add_figure(
                        f"val/sample_{batch_idx}",
                        fig,
                        global_step=self.current_epoch,
                    )
            except (ValueError, AttributeError) as e:
                logger.debug(f"Could not plot validation samples: {e}")
            finally:
                plt.close()

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        """Compute the test loss and additional metrics.

        Args:
            batch: The output of your DataLoader with image1, image2, and mask.
            batch_idx: Integer displaying index of this batch.
            dataloader_idx: Index of the current dataloader.
        """
        image1 = batch["image1"]
        image2 = batch["image2"]
        y = self.squeeze_ground_truth(batch["mask"])

        # Forward pass with both images
        model_output = self.forward_change_detection(image1, image2)

        if dataloader_idx >= len(self.test_loss_handler):
            msg = "You are returning more than one test dataloader but not defining enough test_dataloaders_names."
            raise ValueError(msg)

        # Compute loss
        loss = self.test_loss_handler[dataloader_idx].compute_loss(
            model_output, y, self.criterion, self.aux_loss
        )
        self.test_loss_handler[dataloader_idx].log_loss(
            partial(self.log, add_dataloader_idx=False),
            loss_dict=loss,
            batch_size=y.shape[0],
        )

        # Compute predictions and update metrics
        y_hat_hard = to_segmentation_prediction(model_output)
        self.test_metrics[dataloader_idx].update(y_hat_hard, y)

        self.record_metrics(dataloader_idx, y_hat_hard, y)

    def predict_step(
        self, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> Tensor:
        """Compute the predicted class probabilities.

        Args:
            batch: The output of your DataLoader with image1, image2.
            batch_idx: Integer displaying index of this batch.
            dataloader_idx: Index of the current dataloader.

        Returns:
            Output predicted probabilities.
        """
        image1 = batch["image1"]
        image2 = batch["image2"]
        file_names = batch["filename"] if "filename" in batch else None

        # Forward pass with both images
        model_output = self.forward_change_detection(image1, image2)
        y_hat = model_output.output

        y_hat_ = self.select_classes(y_hat)

        return y_hat_, file_names
