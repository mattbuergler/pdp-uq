# Reproducibility

## Environment

The project supports CPython 3.11 through 3.14. `uv.lock` pins the complete
cross-platform dependency graph; the model metadata records the exact packages
used to serialize release artifacts.

```bash
uv sync --locked --all-extras
uv run python --version
uv run pip check
```

The `pipeline` extra installs DVC. No DVC remote is required because the
canonical raw artifact is fetched from its ETH DOI and every downstream
artifact can be regenerated.

## Pipeline targets

| Stage | Inputs | Outputs | Typical purpose |
| --- | --- | --- | --- |
| `fetch` | DOI manifest | raw CSV | immutable acquisition and checksum |
| `prepare` | raw CSV, config | prepared CSV, validation metrics | schema and cohort definition |
| `split` | prepared CSV, seed | frozen simulation IDs | leakage control |
| `tune` | training IDs, search space | best parameters | full 300-trial search |
| `evaluate` | frozen split, fixed parameters | metrics and plot | untouched holdout |
| `train` | all eligible rows, fixed parameters | release models and metadata | deployment artifacts |

Generate the routine evaluated release:

```bash
uv run dvc repro evaluate train
uv run dvc metrics show
uv run dvc plots show
```

Repeat the paper-scale hyperparameter search:

```bash
uv run dvc repro dvc-tune.yaml
```

This fits 300 parameter samples for each target across five folds and can take
hours on a workstation. A fast end-to-end development configuration is
available separately:

```bash
uv run pdp-uq data prepare data/simulation_results.csv \
  --output data/processed/smoke.csv \
  --report reports/generated/smoke-data.json \
  --config configs/smoke.toml
```

The smoke configuration is exercised on generated fixtures in the automated
test suite; it must not be presented as a scientific model release.

## Determinism

- dataset content is identified by SHA-256;
- the split is based on stable simulation IDs and seed 0;
- estimator and search random states are fixed at 0;
- the split artifact is hashed into evaluation metrics;
- code and dependency hashes are captured by `dvc.lock`;
- model files and their Python package versions are captured by
  `data/model_metadata.json`.

Parallel tree construction and platform-specific floating-point behavior can
still produce non-identical binary joblib files across operating systems.
Compare evaluation metrics and tolerances, not model-file hashes, across
platforms. Within one locked environment, a model digest is used as an
integrity and provenance check.

## Reproducing the paper versus the maintained release

The paper describes an 80/20 split, five-fold cross-validation, and 300
randomized-search iterations. The maintained pipeline implements those
controls without exposing the holdout during tuning. The default evaluated
release uses the recorded fixed hyperparameters so normal reproduction does
not implicitly launch a multi-hour search.

To evaluate tuned parameters explicitly:

```bash
uv run pdp-uq evaluate data/processed/training.csv \
  --split artifacts/splits.json \
  --tuning artifacts/tuning.json \
  --metrics reports/generated/tuned-metrics.json \
  --figure reports/generated/tuned-validation.png
```

To train tuned deployment artifacts, use a separate directory so the fixed
release remains intact:

```bash
uv run pdp-uq train data/processed/training.csv \
  --tuning artifacts/tuning.json \
  --model-dir artifacts/models/tuned \
  --metadata artifacts/models/tuned/model_metadata.json
```

## Verification

Run all engineering gates locally:

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy src
uv run pytest
uv build
uv run dvc status
```

Tests use small generated fixtures and do not require network access or the
published dataset.
