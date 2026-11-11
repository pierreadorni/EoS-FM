import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import lightning.pytorch as pl
import kornia.augmentation as K
from codecarbon import track_emissions
from hydra.utils import instantiate
from pathlib import Path

# Keep training integrations explicit so basic EoS-FM inference imports do not
# require the training framework and its optional dependencies.
from eosfm import register_terratorch

register_terratorch()

torch.set_float32_matmul_precision("medium")


def _patch_normalize_in_augmentation(
    augmentation, mean: torch.Tensor, std: torch.Tensor
) -> bool:
    """Patch all Kornia Normalize modules found in an augmentation pipeline."""
    if augmentation is None:
        return False

    patched = False
    for module in augmentation.modules():
        if isinstance(module, K.Normalize):
            module.flags["mean"] = mean
            module.flags["std"] = std
            patched = True

    return patched


def _apply_normalization_overrides(cfg: DictConfig, datamodule) -> None:
    """Apply per-experiment normalization stats from config to the DataModule."""
    normalization_cfg = cfg.get("normalization")
    if normalization_cfg is None:
        return

    mean = torch.tensor(normalization_cfg.mean, dtype=torch.float32)
    std = torch.tensor(normalization_cfg.std, dtype=torch.float32)

    datamodule.mean = mean
    datamodule.std = std

    patched_any = False
    for aug_name in ["aug", "train_aug", "val_aug", "test_aug", "predict_aug"]:
        patched_any = (
            _patch_normalize_in_augmentation(
                getattr(datamodule, aug_name, None), mean, std
            )
            or patched_any
        )

    if not patched_any:
        datamodule.aug = K.AugmentationSequential(
            K.Normalize(mean=mean, std=std),
            data_keys=None,
            keepdim=True,
        )


def _save_config(cfg: DictConfig, trainer: pl.Trainer) -> None:
    """Save the fully resolved Hydra config to the experiment version directory.

    Mirrors what LightningCLI used to do: each version folder gets a ``config.yaml``
    with the complete config that was used for the run.
    """
    logger = trainer.logger
    if logger is None:
        return
    # TensorBoardLogger exposes log_dir pointing to the versioned directory
    log_dir = getattr(logger, "log_dir", None)
    if log_dir is None:
        return
    config_path = Path(log_dir) / "config.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, config_path)


@track_emissions(log_level="warning")
@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    torch.set_float32_matmul_precision(cfg.get("float32_matmul_precision", "medium"))

    if cfg.get("seed"):
        pl.seed_everything(cfg.seed, workers=True)

    # Instantiate DataModule
    print("Instantiating DataModule...")
    datamodule = instantiate(cfg.data)
    _apply_normalization_overrides(cfg, datamodule)

    # Instantiate Model
    print("Instantiating Model...")
    model = instantiate(cfg.model)

    # Instantiate Trainer
    print("Instantiating Trainer...")
    trainer = instantiate(cfg.trainer)

    # Save the fully resolved config to the experiment version directory
    _save_config(cfg, trainer)

    # Train
    if cfg.get("train"):
        print("Starting training...")
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))

    # Test
    if cfg.get("test"):
        print("Starting testing...")
        trainer.test(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))


if __name__ == "__main__":
    main()
