# Deep Learning–Based Bone Mineral Density Prediction Using Pediatric Chest Radiographs

Official repository for the paper “Deep Learning–Based Bone Mineral Density Prediction Using Pediatric Chest Radiographs: A Multicenter Feasibility Study” (under review).

- Entry point: [train.py](train.py)
- Key configs: [config/train.yaml](config/train.yaml), [config/experiment/](config/experiment/), [config/data/](config/data/), [config/paths/](config/paths/)
- Data module: [data/datamodule.py](data/datamodule.py), [data/dataset.py](data/dataset.py), [data/utils.py](data/utils.py)
- Models: [model/bmd.py](model/bmd.py), [model/regression.py](model/regression.py)
- Networks: [networks/classifier.py](networks/classifier.py), [networks/backbone_tvm.py](networks/backbone_tvm.py)
- Scripts: [scripts/create_sample_data.py](scripts/create_sample_data.py)
- Outputs: [runs/](runs/)
- License: [LICENSE](LICENSE)

## Installation (uv)
This project uses [uv](https://github.com/astral-sh/uv) for environment management and dependency resolution.

Tested versions:
- PyTorch: `torch==2.2.2`
- MONAI: `monai==1.3.2`

Setup steps:

```bash
# Ensure uv is installed
curl -Ls https://astral.sh/uv/install.sh | sh

# From project root, sync dependencies
uv sync

# Optionally verify installed packages
uv run python -m pip list | grep -E "monai|torch|torchvision"
```

Note: If you require a specific CUDA build for PyTorch, follow the official PyTorch instructions to install the matching wheel before running `uv sync`.

## Training
Main Hydra-style command:

```bash
python train.py experiment=exp_bmdcxr data/dataset=bmdcxr_dav1 +model/loss@model.loss_cls=mae +model/metrics=[mae,pearson,icc]
```

- `experiment=exp_bmdcxr`: selects the experiment in [config/experiment/](config/experiment/)
- `data/dataset=bmdcxr_dav1`: chooses the dataset config in [config/data/](config/data/)
- `+model/loss@model.loss_cls=mae`: sets the model loss to MAE
- `+model/metrics=[mae,pearson,icc]`: enables MAE, Pearson, and ICC metrics

Outputs (configs, checkpoints, logs) are written under [runs/](runs/).

## Pre-trained Weights
The pre-trained weights have not been deposited in the public repository to protect institutional intellectual property and ongoing commercialization plans.

## Dummy Data and Format
Generate a minimal dummy dataset to inspect the expected data structure and run a quick sanity check:

```bash
# Generate 25 cases by default
uv run python scripts/create_sample_data.py

# Or specify the number of cases via -n / --number
uv run python scripts/create_sample_data.py -n 10
```

This creates:
- Images: `temp/sample_data/images/case_XXXX.npy` (512×512 float16 arrays)
- Metadata + splits: `temp/sample_data/ds.pkl`

Point your dataset configuration or environment variable (e.g., `DATA_ROOT=temp/sample_data`) to the generated folder and use the main training command to verify end-to-end loading.

## Reproducibility
- Seeds: see [config/seed/](config/seed/)
- Training configuration: see [config/train.yaml](config/train.yaml)

## Citation
To be updated upon publication.

## License
This project is released under the MIT License. See [LICENSE](LICENSE).
