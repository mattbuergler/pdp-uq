"""Typed loading of TOML model configuration."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pdp_uq.constants import DEFAULT_QUANTILES
from pdp_uq.exceptions import ConfigurationError


@dataclass(frozen=True)
class DataConfig:
    """Training-data preparation settings."""

    minimum_awcc_windows: int = 100
    outlier_policy: str = "flag"


@dataclass(frozen=True)
class SplitConfig:
    """Holdout and cross-validation settings."""

    random_seed: int = 0
    test_fraction: float = 0.2
    cross_validation_folds: int = 5


@dataclass(frozen=True)
class SearchConfig:
    """Randomized hyperparameter-search settings."""

    iterations: int = 300
    n_jobs: int = -1


@dataclass(frozen=True)
class ModelConfig:
    """Quantile random-forest settings."""

    random_seed: int = 0
    n_estimators: int = 100
    max_depth: int | None = 12
    max_features: float = 1.0
    n_jobs: int = -1


@dataclass(frozen=True)
class ProjectConfig:
    """Complete reproducibility configuration."""

    data: DataConfig
    split: SplitConfig
    search: SearchConfig
    model: ModelConfig
    quantiles: tuple[float, ...]


def _section(raw: dict[str, Any], name: str) -> dict[str, Any]:
    value = raw.get(name, {})
    if not isinstance(value, dict):
        raise ConfigurationError(f"Configuration section [{name}] must be a table.")
    return value


def load_config(path: Path) -> ProjectConfig:
    """Load and validate a project TOML configuration."""
    try:
        with path.open("rb") as stream:
            raw = tomllib.load(stream)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ConfigurationError(f"Unable to read configuration {path}: {exc}") from exc

    data = DataConfig(**_section(raw, "data"))
    split = SplitConfig(**_section(raw, "split"))
    search = SearchConfig(**_section(raw, "search"))
    model = ModelConfig(**_section(raw, "model"))
    prediction = _section(raw, "prediction")
    quantiles = tuple(float(item) for item in prediction.get("quantiles", DEFAULT_QUANTILES))

    if data.outlier_policy not in {"flag", "exclude"}:
        raise ConfigurationError("data.outlier_policy must be 'flag' or 'exclude'.")
    if not 0.0 < split.test_fraction < 1.0:
        raise ConfigurationError("split.test_fraction must be between 0 and 1.")
    if split.cross_validation_folds < 2:
        raise ConfigurationError("split.cross_validation_folds must be at least 2.")
    if search.iterations < 1:
        raise ConfigurationError("search.iterations must be positive.")
    if not quantiles or any(not 0.0 < item < 1.0 for item in quantiles):
        raise ConfigurationError("prediction.quantiles must contain values between 0 and 1.")
    if tuple(sorted(set(quantiles))) != quantiles:
        raise ConfigurationError("prediction.quantiles must be sorted and unique.")

    return ProjectConfig(data, split, search, model, quantiles)
