<div align="center">

# EoS-FM

### Can an Ensemble of Specialist Models Act as a Generalist Feature Extractor?

[![ECCV 2026](https://img.shields.io/badge/ECCV-2026-4b44ce.svg)](https://doi.org/10.1007/978-3-032-37624-4_28)
[![arXiv](https://img.shields.io/badge/arXiv-2511.21523-b31b1b.svg)](https://arxiv.org/abs/2511.21523)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97_Hugging_Face-Weights-ffd21e.svg)](https://huggingface.co/pierreadorni/eos-fm)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776ab?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6%2B-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]

**Pierre Adorni · Minh-Tan Pham · Stéphane May · Sébastien Lefèvre**

IRISA, Université Bretagne Sud · CNES · UiT The Arctic University of Norway

**Published at [ECCV 2026](https://doi.org/10.1007/978-3-032-37624-4_28)**

<p>
  <a href="https://doi.org/10.1007/978-3-032-37624-4_28"><b>📄 Paper</b></a> &nbsp;·&nbsp;
  <a href="https://huggingface.co/pierreadorni/eos-fm"><b>🤗 Weights</b></a> &nbsp;·&nbsp;
  <a href="#use-the-checkpoint-in-your-framework"><b>⚡ Inference</b></a> &nbsp;·&nbsp;
  <a href="#train-specialists"><b>🛠 Training</b></a>
</p>

</div>

<p align="center">
  <img src="https://github.com/user-attachments/assets/f62362c9-edd7-440c-b160-28c92d731ae8" alt="EoS-FM performance and model size on the Pangaea benchmark" width="80%"/>
  <br>
  <sub>An ensemble of 22 specialists for Earth observation.</sub>
</p>

EoS-FM combines specialist encoders trained on Earth observation datasets. This
repository provides specialist training and a standalone PyTorch encoder for
integration into other frameworks.

## Install

Inference requires Python 3.10+, PyTorch and timm:

```bash
pip install -e .
```

For specialist training and checkpoint export:

```bash
pip install -e ".[training]"
```

## Use the checkpoint in your framework

Download [`eosfm.pth`](https://huggingface.co/pierreadorni/eos-fm/resolve/main/eosfm.pth)
from [Hugging Face](https://huggingface.co/pierreadorni/eos-fm) and save it as
`release/eosfm.pth`. The 22-specialist checkpoint has SHA-256:

```text
51d15c6ac0a39914780d78df3f9f953b4482f189c27c57398ae4cb75c0d850b0
```

The weights are hosted separately from the code. The
[release manifest](release/eosfm.manifest.json) records their specialist
inventory, band order and normalization statistics.

```python
import torch
from eosfm import EosFM

bands = ["B2", "B3", "B4", "B8", "B11", "B12"]
model = EosFM(
    input_bands={"optical": bands},
    input_size=64,
    encoder_weights="release/eosfm.pth",
    normalize_features=False,
    feature_fusion=None,
    projection_layer=False,
    max_encoders=None,
    torch_compile=False,
).eval()

# images: float32 [batch, len(bands), height, width], already standardized
# with statistics fitted on your downstream training split.
images = torch.randn(2, len(bands), 64, 64)
with torch.no_grad():
    feature_maps = model(images)
    embeddings = feature_maps[-1].mean(dim=(-2, -1))
```

The example returns four concatenated feature stages. With all 22 specialists,
they have 880, 1760, 3520 and 7040 channels; final-stage global average pooling
produces a 7040-dimensional embedding. FP32 is the reference inference precision.

For mixed inputs, set `input_bands={"optical": [...], "sar": ["VV", "VH"]}`.
A tensor must contain optical channels first, then SAR, in the listed order.
Alternatively pass `{"optical": optical_tensor, "sar": sar_tensor}` with each
value shaped `[batch, channels, height, width]`.

Band adaptation uses exact names, wavelength matching, band expressions and
fallbacks for missing bands. Expressions operate on the supplied standardized
channels. The encoder does not reapply specialist training statistics. Missing
Kenya `CLD` uses the optical fallback. Fit downstream input statistics on training
data only.

`normalize_features=True` enables spatial feature BatchNorm. Its running
statistics must be fitted on downstream training data and frozen for evaluation.
Learned fusion, projection and selection require downstream training; their
weights are not included in the specialist checkpoint. The example above uses
raw frozen features.

For direct integration without installing this package, copy
[`eosfm/encoder.py`](eosfm/encoder.py) into your framework and import `EosFM` from
that file. It requires only PyTorch, timm and Python's standard library. This is
the same implementation used by the package; TerraTorch is only needed for training.

## Train specialists

Training uses `run.py` and the recipes in `config/experiment/train/`. Set each
dataset's local root and download settings before launching a run:

```bash
python run.py experiment=train/ben_s1 data.root=/path/to/bigearthnet
python run.py experiment=train/ben_all data.root=/path/to/bigearthnet
python run.py experiment=train/cloud_cover data.root=/path/to/cloud_cover
```

Each recipe specifies the task, input bands and normalization, model settings,
optimizer and training schedule. These are current recipes; source references in
comments do not certify every historical training detail. Run logs, checkpoints
and the resolved configuration are saved under `experiments/train/`.

To inspect a recipe without training:

```bash
python run.py experiment=train/ben_s1 --cfg job
```

Export newly trained specialists to the inference checkpoint format:

```bash
python scripts/export_checkpoint.py \
  --encoders-folder experiments/train \
  --output-path release/my-eosfm.pth
```

The exporter takes the latest numbered run and latest epoch/step checkpoint per
specialist. Point it at a directory containing only the specialist runs you intend
to release. It preserves the saved band labels and normalization; historical
configs with incorrect labels must be corrected before export. Only local,
trusted training checkpoints should be used as export inputs. The exported
artifact contains backbone weights and plain metadata and loads with
`torch.load(..., weights_only=True)`.

A frozen downstream classifier example is available via:

```bash
python run.py experiment=test/test_eurosat \
  data.root=/path/to/eurosat \
  model.model_args.backbone_encoder_weights=release/eosfm.pth
```

This trains a classifier over raw frozen features. EuroSAT and BigEarthNet overlap
specialist supervision, so their results are not independent transfer evaluations.

`run.py` registers EoS-FM with TerraTorch automatically. If you build a TerraTorch
model in your own code, call `from eosfm import register_terratorch` followed by
`register_terratorch()` before constructing it.

## Checks

```bash
python -m unittest discover -s tests
```

The default checks use synthetic specialists. Full-checkpoint and TerraTorch
integration checks are opt-in using `EOSFM_RELEASE_CHECKPOINT` and
`EOSFM_TEST_TRAINING`, respectively; see the test module docstrings.

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
2. Update configs and documentation

## Citing this work

```bibtex
@inproceedings{adorni2026eos,
  title={EoS-FM: Can an Ensemble of Specialist Models Act as a Generalist Feature Extractor?},
  author={Adorni, Pierre and Pham, Minh-Tan and May, St{\'e}phane and Lef{\`e}vre, S{\'e}bastien},
  booktitle={Computer Vision -- ECCV 2026},
  series={Lecture Notes in Computer Science},
  volume={17028},
  pages={464--482},
  publisher={Springer Nature Switzerland},
  year={2026},
  doi={10.1007/978-3-032-37624-4_28}
}
```
