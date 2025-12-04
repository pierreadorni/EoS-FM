import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import lightning.pytorch as pl
from codecarbon import track_emissions
from hydra.utils import instantiate

# import the EosFM to force the registration of the EosFM model in the TERRATORCH_BACKBONE_REGISTRY
from eosfm.encoder import EosFM

torch.set_float32_matmul_precision("medium")


@track_emissions(log_level="warning")
@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    if cfg.get("seed"):
        pl.seed_everything(cfg.seed, workers=True)

    # Instantiate DataModule
    print("Instantiating DataModule...")
    datamodule = instantiate(cfg.data)

    # Instantiate Model
    print("Instantiating Model...")
    model = instantiate(cfg.model)

    # Instantiate Trainer
    print("Instantiating Trainer...")
    trainer = instantiate(cfg.trainer)

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
