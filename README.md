<div align="center">

# EoS-FM: Can an Ensemble of Specialist Models Act as a Generalist Feature Extractor?

[![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa] [![arXiv](https://img.shields.io/badge/arXiv-2511.21523-b31b1b.svg)](https://arxiv.org/abs/2511.21523)
</div>

<p align='center'>
  <img src="https://github.com/user-attachments/assets/8a4e9f7f-6fdf-42f3-a4b7-879343024962" width="80%"/>
</p>

An ensemble foundation model for Earth Observation that aggregates features from multiple specialist backbone encoders to achieve generalist capabilities across diverse remote sensing tasks.

## Overview

EoS-FM addresses a fundamental question in remote sensing foundation models: rather than training a single large generalist model on massive amounts of data, can we ensemble multiple specialist models trained on different tasks and band combinations to achieve comparable or better generalist performance?

### Key Innovation

The `EosFM` encoder implements a **late fusion ensemble** approach:
- **Specialist Models**: Multiple pre-trained backbone networks, each trained on different datasets and band combinations (RGB, multispectral, SAR)
- **Band Adaptation Layer**: Automatically adapts input bands to match each specialist's requirements using rule-based strategies
- **Feature Concatenation**: Combines specialist features channel-wise at each pyramid level for downstream decoders
- **Encoder Selection Layer**: Selects an optimal subset of encoders to automatically prune the encoder at training time

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .

# Install additional dependencies (adjust based on your needs)
pip install terratorch torchgeo lightning rasterio torch torchvision
```

## Quick Start

### Training a Model

```bash
# Set up environment
source venv/bin/activate
export PYTHONPATH=$(pwd):$PYTHONPATH

# Train on a dataset (e.g., Cloud Cover)
python run.py experiment=train/cloud_cover

# List all available experiments
python run.py --help | grep "experiment"

# Available experiments: ben_rgb, ben_s1, ben_all, caffe, cloud_cover, deepglobe_lcc, 
# dfc2022, dior, etci2021, eurosat_rgb, eurosat_s2, firerisk, inria_aerial, 
# kenya_croptype, landcoverai, levircdplus, loveda, minifrance, opencanopy, oscd, 
# potsdam, sen12ms_rgb, sen12ms_s1, sen12ms_s2, sen12ms_all
```

### Dry Run (Verify Configuration)

```bash
# Check that configuration is valid without training
python run.py experiment=train/potsdam train=false
```

### Using EosFM Ensemble Backbone

```yaml
# Example: in conf/experiment/my_experiment.yaml
defaults:
  - override /model: segmentation

data:
  _target_: eosfm.datamodules.MyDataModule
  batch_size: 32
  num_workers: 8
  root: /path/to/data

trainer:
  max_epochs: 100
  accelerator: auto
  devices: auto
  precision: bf16-mixed

logger:
  name: my_experiment

model:
  model_args:
    backbone: EosFM
    backbone_in_chans: 3
    model_weights: models/eosfm.pth
    freeze: true  # Freeze encoders during training (recommended)
    num_classes: 6
  loss: dice
```

#### Advanced Feature Fusion Examples

**1. Basic Concatenation (Default)**
```yaml
model:
  model_args:
    backbone: EosFM
    num_classes: 6
```

**2. Encoder Selection with Top-K**
```yaml
model:
  model_args:
    backbone: EosFM
    max_encoders: 5  # Select only 5 most relevant encoders
    num_classes: 6
```

**3. Feature Normalization**
```yaml
model:
  model_args:
    backbone: EosFM
    normalize_features: true
    normalization_type: "batch"  # "batch" or "layer"
    num_classes: 6
```

### Creating Ensemble .pth Files

The ensemble file format is a list of tuples: `[(in_chans, state_dict, config), ...]`

The `utils.py` script provides utilities to load and export ensemble models from a folder of trained Lightning checkpoints:

```bash
# Export all trained models from an experiments folder to an ensemble .pth file
python utils.py export \
  --encoders-folder experiments/train \
  --output-path eosfm_ensemble.pth
```

This command:
- Recursively loads all Lightning checkpoints from `experiments/train`
- Extracts encoder weights and configurations from each checkpoint
- Automatically detects the input channels and backbone architecture
- Saves them in the correct ensemble format


## Supported Datasets

Custom datamodules in `eosfm/datamodules/`:
- **Potsdam2D**: Urban semantic segmentation (RGBIR, 6000×6000 images)
- **MiniFrance**: Land cover classification
- **SEN12MS**: Multi-modal Sentinel-1/2 with land cover labels
- **BigEarthNetV2**: Multi-label classification
- **ETCI2021**: Flood detection
- And more...

Also supports TorchGeo datasets: EuroSAT, DeepGlobe, FireRisk, OSCD, etc.

## Creating Custom Datasets

```python
class MyDataModule(PinNonGeoDataModule):
    def setup(self, stage):
        if stage in ['fit', 'validate']:
            self.train_dataset = MyDataset(use_tiling=False)  # Random crops
            self.val_dataset = MyDataset(use_tiling=True)     # Systematic tiling
```

## Configuration

Configurations use **Hydra** for composable, modular setup. Each experiment is self-contained with data, model, and trainer configurations.

### Example Experiment Configuration

Each experiment is a minimal YAML file that specifies data, model, and training settings:

```yaml
# config/experiment/train/cloud_cover.yaml
# @package _global_
defaults:
  - override /model: segmentation

data:
  _target_: eosfm.datamodules.CloudCoverDetectionDataModule
  batch_size: 32
  num_workers: 4
  root: data/cloud_cover

trainer:
  max_epochs: 20
  check_val_every_n_epoch: 1

logger:
  name: cloud_cover

model:
  model_args:
    num_classes: 2
  loss: ce
```

### Creating Custom Experiments

1. Create a new YAML file in `config/experiment/train/`:

```yaml
# config/experiment/train/my_dataset.yaml
# @package _global_
defaults:
  - override /model: segmentation

data:
  _target_: my.custom.DataModule
  batch_size: 32
  num_workers: 8
  root: /path/to/data

trainer:
  max_epochs: 100

logger:
  name: my_dataset

model:
  model_args:
    num_classes: 10
    backbone: timm_convnextv2_atto
  loss: dice
```

2. Run the experiment:

```bash
python run.py experiment=train/my_dataset

# Or override specific settings from CLI:
python run.py experiment=train/my_dataset trainer.max_epochs=50 data.batch_size=64
```

### Command Line Overrides

Override any configuration value from the command line:

```bash
# Override epochs, batch size, learning rate
python run.py experiment=train/potsdam trainer.max_epochs=200 data.batch_size=16

# Disable training, only validate
python run.py experiment=train/potsdam train=false

# Run a dry run to verify configuration
python run.py experiment=train/potsdam train=false test=false --cfg job
```

## Carbon Tracking

Training automatically tracks CO₂ emissions via [CodeCarbon](https://github.com/mlco2/codecarbon). The energy consumption and estimated CO₂eq for this project are detailed below. Note that these **are not the values for a single training of the model**, but rather for all of the trial and error runs during the project, up to the final full training run.

| Total emissions   	        | Total energy consumed      	|
|-----------------------------	|-----------------------------	|
| 4.15 kg CO₂eq                	| 76.05 kWh                   	|

## License

This work is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg

## Contributing

Contributions welcome! Please:
1. Follow the existing code structure
3. Update configs and documentation

## Citing this work

```
@article{adorni2025eos,
  title={EoS-FM: Can an Ensemble of Specialist Models act as a Generalist Feature Extractor?},
  author={Adorni, Pierre and Pham, Minh-Tan and May, St{\'e}phane and Lef{\`e}vre, S{\'e}bastien},
  journal={arXiv preprint arXiv:2511.21523},
  year={2025}
}
```
