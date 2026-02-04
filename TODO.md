# Public Release TODO

Use this checklist to prepare the repo for a public release linked to the publication.

## Repository Hygiene
- [ ] Remove nonessential artifacts:
  - [ ] Delete `temp/` and clean up intermediate outputs
  - [ ] Remove `__pycache__/` folders
  - [ ] Remove notebook checkpoints (if any)
- [ ] Add/update `.gitignore` to ensure above paths are ignored

## Licensing & Attribution
- [ ] Confirm license text and year/author in [LICENSE](LICENSE)
- [ ] Add third‑party license notices (e.g., vendored local libs like "mislight")
- [ ] Add `CITATION.cff` with publication DOI

## Dependency Management (uv)
- [ ] Initialize uv project:
  - [ ] `uv init` and create `pyproject.toml` with Python version pin
- [ ] Discover and add dependencies:
  - [ ] Scan imports across codebase and run `uv add <pkg>` for each
    - Likely in [train.py](train.py), [data/datamodule.py](data/datamodule.py), [data/dataset.py](data/dataset.py), [model/bmd.py](model/bmd.py), [model/regression.py](model/regression.py), [networks/backbone_tvm.py](networks/backbone_tvm.py), [networks/classifier.py](networks/classifier.py), [src/test.py](src/test.py), [utils/bmd.py](utils/bmd.py)
- [ ] Lock and reproducibility:
  - [ ] `uv lock` and commit lockfile
  - [ ] Add `uv run` scripts for train/eval in `tool.uv.scripts`

## Internalize Local‑Only Library (e.g., mislight)
- [ ] Vendor code under `src/mislight/` or `utils/mislight/`
- [ ] Add `__init__.py` and minimal tests
- [ ] Document origin and license in `src/mislight/README.md`
- [ ] Replace imports to reference vendored module

## Paths & Configuration
- [ ] Replace hard‑coded local paths with config variables:
  - [ ] Centralize in [config/train.yaml](config/train.yaml) and/or [config/paths/](config/paths/)
  - [ ] Add environment variable overrides (e.g., `DATA_ROOT`, `OUTPUT_DIR`)
- [ ] Update code to read from config:
  - [ ] [train.py](train.py): inject paths from config/env
  - [ ] [data/datamodule.py](data/datamodule.py) and [data/dataset.py](data/dataset.py): remove hard‑coded directories
- [ ] Document all required variables in README

## Remove Unused Code/Config
- [ ] Audit and prune:
  - [ ] [config/callbacks/](config/callbacks/)
  - [ ] [config/scheduler/](config/scheduler/)
  - [ ] [config/optimizer/](config/optimizer/)
  - [ ] Legacy scripts/tests like [src/test.py](src/test.py)
  - [ ] Experimental or duplicate models in [model/](model/) and [networks/](networks/)
- [ ] Keep only what reproduces the final experiment

## Documentation
- [ ] Write README.md:
  - [ ] Project overview and publication link
  - [ ] Environment setup with uv
  - [ ] Data preparation and expected layout
  - [ ] Configuration reference (paths, hyperparams)
  - [ ] Training/evaluation commands
  - [ ] Citation
- [ ] Add example configs in [config/](config/) with explanations

## Reproducibility
- [ ] Ensure deterministic seeds (see [config/seed/](config/seed/))
- [ ] Record exact training config used in the paper ([config/train.yaml](config/train.yaml))
- [ ] Save and document checkpoints and metrics

## Data Handling
- [ ] Provide scripts or instructions for data download/prep
- [ ] Anonymize or exclude sensitive data
- [ ] Validate data paths via [data/utils.py](data/utils.py)

## Testing & CI
- [ ] Add minimal unit tests:
  - [ ] Dataset/datamodule loading
  - [ ] Model forward pass (e.g., [model/bmd.py](model/bmd.py), [networks/classifier.py](networks/classifier.py))
- [ ] GitHub Actions:
  - [ ] CI with `uv sync` and `uv run pytest`
  - [ ] Lint/type checks (if applicable)

## Packaging & Release
- [ ] Ensure importable package layout under `src/` or top‑level package
- [ ] Add versioning and changelog
- [ ] Tag `v1.0.0` and create GitHub release
- [ ] Final sanity run of training with public config
