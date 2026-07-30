"""Public, testable inference API."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

import joblib
import numpy as np
import numpy.typing as npt
import pandas as pd

from pdp_uq.constants import (
    DEFAULT_QUANTILES,
    MODEL_FEATURE_COLUMNS,
    MODEL_FILENAMES,
)
from pdp_uq.exceptions import ArtifactError
from pdp_uq.io import read_json, sha256_file
from pdp_uq.schema import application_domain, prepare_measurement_features


class QuantileModel(Protocol):
    """Structural type required from a fitted quantile model."""

    def predict(
        self,
        features: npt.NDArray[np.float64],
        *,
        quantiles: list[float],
    ) -> npt.NDArray[np.float64]:
        """Predict one or more conditional quantiles."""


@dataclass(frozen=True)
class PredictionConfig:
    """Probe geometry and prediction settings."""

    delta_x: float
    delta_y: float
    particles_per_window: int
    quantiles: tuple[float, ...] = DEFAULT_QUANTILES


def _expected_model_hash(model_path: Path, metadata_path: Path | None) -> str | None:
    if metadata_path is None or not metadata_path.exists():
        return None
    metadata = read_json(metadata_path)
    models = metadata.get("models", {})
    if not isinstance(models, dict):
        return None
    record = models.get(model_path.name, {})
    if not isinstance(record, dict):
        return None
    digest = record.get("sha256")
    return str(digest) if digest else None


def load_quantile_model(
    path: Path,
    *,
    metadata_path: Path | None = None,
    require_checksum: bool = True,
) -> QuantileModel:
    """Load a trusted model artifact after verifying its recorded checksum."""
    if not path.is_file():
        raise ArtifactError(
            f"Model artifact not found: {path}. Run `pdp-uq train` or `dvc repro train`."
        )
    expected = _expected_model_hash(path, metadata_path)
    if require_checksum and expected is None:
        raise ArtifactError(f"No checksum metadata is available for model artifact {path}.")
    if expected is not None:
        actual = sha256_file(path)
        if actual != expected:
            raise ArtifactError(
                f"Model checksum mismatch for {path}: expected {expected}, received {actual}."
            )
    # joblib is pickle-based. Loading is intentionally restricted to a locally
    # checksum-verified artifact whose provenance is recorded by this project.
    return cast(QuantileModel, joblib.load(path))


def _prediction_matrix(
    model: QuantileModel,
    features: pd.DataFrame,
    valid: pd.Series,
    quantiles: tuple[float, ...],
) -> npt.NDArray[np.float64]:
    predictions = np.full((len(features), len(quantiles)), np.nan, dtype=float)
    if valid.any():
        selected = features.loc[valid, MODEL_FEATURE_COLUMNS].to_numpy(dtype=float)
        values = np.asarray(model.predict(selected, quantiles=list(quantiles)), dtype=float)
        predictions[np.flatnonzero(valid.to_numpy()), :] = values.reshape(
            int(valid.sum()), len(quantiles)
        )
    return predictions


def predict_dataframe(
    measurements: pd.DataFrame,
    config: PredictionConfig,
    *,
    velocity_model: QuantileModel,
    turbulence_model: QuantileModel,
) -> pd.DataFrame:
    """Correct measurements and append predictive quantiles and domain flags."""
    if not config.quantiles or tuple(sorted(set(config.quantiles))) != config.quantiles:
        raise ValueError("Quantiles must be sorted and unique.")
    features = prepare_measurement_features(
        measurements,
        delta_x=config.delta_x,
        delta_y=config.delta_y,
        particles_per_window=config.particles_per_window,
    )
    valid, reasons = application_domain(features)
    velocity = _prediction_matrix(velocity_model, features, valid, config.quantiles)
    turbulence = _prediction_matrix(turbulence_model, features, valid, config.quantiles)

    result = features.copy()
    result["within_model_domain"] = valid
    result["model_domain_warning"] = reasons
    for index, quantile in enumerate(config.quantiles):
        label = f"{quantile:g}"
        result[f"u_corrected_q{label} [m/s]"] = velocity[:, index]
        result[f"T_u_corrected_q{label} [-]"] = turbulence[:, index]
    return result


def predict_file(
    input_path: Path,
    config: PredictionConfig,
    *,
    model_directory: Path,
    output_path: Path | None = None,
    metadata_path: Path | None = None,
) -> Path:
    """Apply released models to a CSV file and atomically write the result."""
    metadata = metadata_path or model_directory / "model_metadata.json"
    velocity_model = load_quantile_model(
        model_directory / MODEL_FILENAMES["mean_velocity"], metadata_path=metadata
    )
    turbulence_model = load_quantile_model(
        model_directory / MODEL_FILENAMES["turbulence_intensity"], metadata_path=metadata
    )
    measurements = pd.read_csv(input_path)
    result = predict_dataframe(
        measurements,
        config,
        velocity_model=velocity_model,
        turbulence_model=turbulence_model,
    )
    destination = output_path or input_path.with_name(f"{input_path.stem}_uq.csv")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.part")
    try:
        result.to_csv(temporary, index=False, na_rep="nan")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination
