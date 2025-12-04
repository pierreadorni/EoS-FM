# EoS-FM: Can an Ensemble of Specialist Models Act as a Generalist Feature Extractor?

[![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa] [![arXiv](https://img.shields.io/badge/arXiv-2511.21523-b31b1b.svg)](https://arxiv.org/abs/2511.21523)

**Paper**: [arXiv:2511.21523v1](https://arxiv.org/abs/2511.21523)
**License**: CC BY-NC-SA 4.0

<p align='center'>
  <img src="https://github.com/user-attachments/assets/8a4e9f7f-6fdf-42f3-a4b7-879343024962" width="80%"/>
</p>

An ensemble foundation model for Earth Observation that aggregates features from multiple specialist backbone encoders to achieve generalist capabilities across diverse remote sensing tasks.

## Overview

EoS-FM addresses a fundamental question in remote sensing foundation models: rather than training a single large generalist model on massive amounts of data, can we ensemble multiple specialist models trained on different tasks and band combinations to achieve comparable or better generalist performance?

### Key Innovation

The `EosFM` encoder implements a **late fusion ensemble** approach:
- **Specialist Models**: Multiple pre-trained backbone networks, each trained on different datasets and band combinations (RGB, multispectral, SAR)
- **Band Adaptation Layer**: Automatically adapts input bands to match each specialist's requirements using learned or rule-based strategies
- **Feature Concatenation**: Combines specialist features channel-wise at each pyramid level for downstream decoders
- **Mixed-Modality Support**: Seamlessly handles optical, SAR, and multispectral inputs

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

#### Using utils.py (Recommended)

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

#### Manual Creation

Alternatively, you can create the ensemble file manually:

```python
import torch

# Extract from trained Lightning checkpoints
specialists = []
for ckpt_path in specialist_checkpoints:
    checkpoint = torch.load(ckpt_path)
    config = checkpoint['hyper_parameters']['model_args']
    
    # Extract encoder state dict
    state_dict = {
        k.removeprefix('model.encoder.'): v 
        for k, v in checkpoint['state_dict'].items() 
        if k.startswith('model.encoder.')
    }
    
    specialists.append((
        config.get('backbone_in_chans', 3),  # Number of input bands
        state_dict,
        {'backbone': config['backbone'], 'name': 'specialist_name'}
    ))

# Save ensemble
torch.save(specialists, 'eosfm_ensemble.pth')
```

## Project Structure

```
ensemble/
├── eosfm/                      # Main package
│   ├── encoder.py              # EosFM ensemble encoder
│   ├── band_adaptation.py      # Band adaptation strategies
│   └── datamodules/            # Custom dataset implementations
│       ├── base.py             # Base datamodule with pin_memory
│       ├── potsdam.py          # Efficient tiling for large images
│       ├── minifrance.py       # Land cover segmentation
│       ├── sen12ms.py          # Multi-modal Sentinel-1/2
│       └── ...
├── configs/                    # Training configurations
│   ├── potsdam.yaml            # Potsdam semantic segmentation
│   ├── sen12ms-all.yaml        # SEN12MS all bands
│   └── ...
├── experiments/                # Training outputs (checkpoints, logs)
├── data/                       # Dataset storage
└── terratorch_co2.py          # Training CLI with carbon tracking
```

## Key Features

### 1. Band Adaptation System

Automatically adapts between different band combinations:

```python
from eosfm.band_adaptation import register_strategy, BandAdaptationStrategy

@register_strategy
class CustomAdaptation(BandAdaptationStrategy):
    required_bands = 3  # What encoder expects
    available_bands = 13  # What your data has
    
    def adapt(self, features: torch.Tensor) -> torch.Tensor:
        # Select/transform bands as needed
        return features[:, [3, 2, 1], :, :]
```

Built-in adaptations:
- Sentinel-2 → RGB (S212toRGB, S213ToRGB)
- SAR → RGB composite (SARtoRGB)
- Sentinel-1&2 → Sentinel-2 (S12ToS212)
- And more...

### 2. Multi-Modal Support

Datasets can provide different modalities, handled via batch transfer hooks:

```python
def on_before_batch_transfer(self, batch, dataloader_idx):
    # Combine Sentinel-1 and Sentinel-2
    if 'image_s1' in batch and 'image_s2' in batch:
        batch['image'] = torch.cat([batch['image_s1'], batch['image_s2']], dim=1)
    
    # Create SAR RGB composite
    if batch['image'].shape[1] == 2:  # VV, VH channels
        vv, vh = batch['image'][:, 0:1], batch['image'][:, 1:2]
        batch['image'] = torch.cat([vv, vh, vh], dim=1)
    
    return batch
```

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

### For Large Images (>1000×1000)

Follow the `Potsdam2D` pattern:

```python
class MyLargeImageDataset(NonGeoDataset):
    def __init__(self, root, split, tile_size=512, stride=None, use_tiling=True):
        self.tile_size = tile_size
        self.stride = stride or tile_size
        self.use_tiling = use_tiling
        
        if use_tiling:
            self.tiles = self._create_tiles()  # Pre-compute tile positions
        else:
            self.tiles = None  # Random cropping
    
    def _create_tiles(self):
        tiles = []
        for file_idx, file_info in enumerate(self.files):
            with rasterio.open(file_info['image']) as src:
                width, height = src.width, src.height
            for y in range(0, height - self.tile_size + 1, self.stride):
                for x in range(0, width - self.tile_size + 1, self.stride):
                    tiles.append({'file_index': file_idx, 'x': x, 'y': y})
        return tiles
    
    def __getitem__(self, index):
        if self.use_tiling:
            # Validation: systematic tiling
            tile = self.tiles[index]
            window = Window(tile['x'], tile['y'], self.tile_size, self.tile_size)
        else:
            # Training: random crop
            file = self.files[index % len(self.files)]
            # ... random x, y ...
            window = Window(x, y, self.tile_size, self.tile_size)
        
        with rasterio.open(file['image']) as src:
            image = src.read(window=window)
        # ... process and return
```

### DataModule

```python
class MyDataModule(PinNonGeoDataModule):
    def setup(self, stage):
        if stage in ['fit', 'validate']:
            self.train_dataset = MyDataset(use_tiling=False)  # Random crops
            self.val_dataset = MyDataset(use_tiling=True)     # Systematic tiling
```

## Configuration

Configurations use **Hydra** for composable, modular setup. Each experiment is self-contained with data, model, and trainer configurations.

### Directory Structure

```
config/
├── config.yaml                 # Base configuration
├── model/                      # Model configurations (segmentation, classification, etc.)
│   ├── segmentation.yaml
│   ├── multilabel_classification.yaml
│   ├── classification.yaml
│   ├── object_detection.yaml
│   ├── change_detection.yaml
│   └── pixelwise_regression.yaml
├── trainer/                    # Trainer settings
│   └── default.yaml
├── logger/                     # Logger configurations
│   └── tensorboard.yaml
├── callbacks/                  # Callback configurations
│   └── default.yaml
└── experiment/                 # Experiment configs (combine data + model + trainer)
    ├── train/
    │   ├── ben_rgb.yaml
    │   ├── caffe.yaml
    │   ├── cloud_cover.yaml
    │   ├── potsdam.yaml
    │   ├── sen12ms_rgb.yaml
    │   └── ... (24 total experiments)
    └── test/
        └── eurosat.yaml
```

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