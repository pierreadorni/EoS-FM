# EoS-FM: Can an Ensemble of Specialist Models Act as a Generalist Feature Extractor?

[![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa] [![arXiv](https://img.shields.io/badge/arXiv-2511.21523-b31b1b.svg)](https://arxiv.org/abs/2511.21523)

**Paper**: [arXiv:2511.21523v1](https://arxiv.org/abs/2511.21523)
**License**: CC BY-NC-SA 4.0

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

# Train on a dataset (e.g., Potsdam)
python terratorch_co2.py fit --config configs/potsdam.yaml
```

### Using EosFM Ensemble Backbone

```yaml
# In your config YAML
model:
  class_path: terratorch.tasks.SemanticSegmentationTask
  init_args:
    model_factory: EncoderDecoderFactory
    model_args:
      backbone: EosFM
      backbone_in_chans: 3
      model_weights: path/to/eosfm_ensemble.pth
      freeze: true  # Freeze encoders during training (recommended)
    
      normalize_features: true  # Apply feature normalization before fusion
      normalization_type: "batch"  # "batch" for BatchNorm2d, "layer" for LayerNorm
      projection_layer: false  # Add Conv2d 1x1 + LeakyReLU after normalization
      
      feature_fusion: null  # Options: null, "conv1x1", "mlp", "addition", "multiplication"
      fuse_to_mult: 1  # Multiplier for output channels (used with conv1x1/mlp)
      
      max_encoders: null  # Set to k to select only top-k encoders (e.g., 5)
      encoder_selection_mode: "topk"  # "topk" for hard selection, "smooth" for soft weighting
      scale_features: false  # Whether to scale features by selection weights
      
      ablate_encoders: []  # List of encoder indices to disable (e.g., [0, 3, 7])
      
      decoder: UperNetDecoder
      num_classes: 6
```

#### Advanced Feature Fusion Examples

**1. Basic Concatenation (Default)**
```yaml
model_args:
  backbone: EosFM
  feature_fusion: null  # Simply concatenates all encoder features
  normalize_features: true  # Recommended for stable training
```

**2. Dimensionality Reduction with Conv1x1**
```yaml
model_args:
  backbone: EosFM
  feature_fusion: "conv1x1"  # Reduces concatenated features
  fuse_to_mult: 2  # Output = 2 × single_encoder_channels
  normalize_features: true
```

**3. Non-linear Fusion with MLP**
```yaml
model_args:
  backbone: EosFM
  feature_fusion: "mlp"  # Two-layer MLP with ReLU
  fuse_to_mult: 1  # Match single encoder dimensions
  normalize_features: true
  projection_layer: true  # Additional learnable projection per encoder
```

**4. Addition/Multiplication Fusion** (requires all encoders to have same output dimensions)
```yaml
model_args:
  backbone: EosFM
  feature_fusion: "addition"  # or "multiplication"
  normalize_features: true  # Critical for these modes
```

**5. Encoder Selection with Top-K**
```yaml
model_args:
  backbone: EosFM
  max_encoders: 5  # Select only 5 most relevant encoders
  encoder_selection_mode: "topk"  # Hard selection
  scale_features: false  # Binary mask (selected=1, others=0)
  normalize_features: true
```

**6. Smooth Encoder Selection** (use with sparsity regularization)
```yaml
model_args:
  backbone: EosFM
  max_encoders: 5  # Target number of encoders
  encoder_selection_mode: "smooth"  # Soft weighting with sigmoid
  scale_features: true  # Weight features by learned importance
  normalize_features: true
```

**7. Ablation Study Configuration**
```yaml
model_args:
  backbone: EosFM
  ablate_encoders: [0, 3, 7]  # Disable encoders 0, 3, and 7
  # Useful for analyzing individual encoder contributions
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

Training configs use PyTorch Lightning CLI format (`configs/*.yaml`):

```yaml
seed_everything: 0

trainer:
  accelerator: auto
  devices: auto
  precision: bf16-mixed
  max_epochs: 10
  logger:
    class_path: TensorBoardLogger
    init_args:
      save_dir: experiments
      name: my_experiment

data:
  class_path: eosfm.datamodules.MyDataModule
  init_args:
    batch_size: 8
    tile_size: 512
    num_workers: 4
  dict_kwargs:
    root: /path/to/data

model:
  class_path: terratorch.tasks.SemanticSegmentationTask
  init_args:
    model_args:
      backbone: EosFM  # or timm_convnextv2_atto, etc.
      backbone_in_chans: 3
      model_weights: models/eosfm.pth
      decoder: UperNetDecoder
      num_classes: 6
    loss: dice
    freeze_backbone: false

optimizer:
  class_path: torch.optim.AdamW
  init_args:
    lr: 1.e-4
    weight_decay: 0.1
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