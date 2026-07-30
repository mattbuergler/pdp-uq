"""Reproducible preparation, splitting, tuning, and model training."""

from __future__ import annotations

import importlib.metadata
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import numpy.typing as npt
import pandas as pd
from quantile_forest import RandomForestQuantileRegressor
from scipy.stats import randint, uniform
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, RandomizedSearchCV, train_test_split

from pdp_uq.config import ProjectConfig
from pdp_uq.constants import (
    APPLICATION_RANGES,
    MODEL_FEATURE_COLUMNS,
    MODEL_FILENAMES,
    TARGET_COLUMNS,
    TRAINING_COLUMN_MAP,
)
from pdp_uq.exceptions import DataValidationError
from pdp_uq.io import read_json, sha256_file, write_json
from pdp_uq.schema import application_domain, numeric_frame, require_columns

TRAINING_SOURCE_COLUMNS = (
    "id [-]",
    "n_awcc [-]",
    *TRAINING_COLUMN_MAP,
    "delta_x [m]",
    "delta_y [m]",
    "N_p [-]",
    *TARGET_COLUMNS.values(),
)


def prepare_training_data(
    source_path: Path,
    output_path: Path,
    report_path: Path,
    config: ProjectConfig,
) -> dict[str, Any]:
    """Validate and normalize the published simulation dataset."""
    source = pd.read_csv(source_path)
    require_columns(source, TRAINING_SOURCE_COLUMNS, context="Published training data")
    frame = numeric_frame(
        source,
        TRAINING_SOURCE_COLUMNS,
        context="Published training data",
        allow_missing=True,
    )
    total_rows = len(frame)

    enough_windows = frame["n_awcc [-]"] > config.data.minimum_awcc_windows
    complete = pd.Series(
        np.isfinite(frame.loc[:, TRAINING_SOURCE_COLUMNS].to_numpy(dtype=float)).all(axis=1),
        index=frame.index,
        dtype=bool,
    )
    eligible = enough_windows & complete
    prepared = frame.loc[eligible, TRAINING_SOURCE_COLUMNS].copy()
    prepared = prepared.rename(columns=TRAINING_COLUMN_MAP)

    within_domain, reasons = application_domain(prepared)
    prepared["within_application_domain"] = within_domain
    prepared["application_domain_warning"] = reasons
    excluded_outliers = 0
    if config.data.outlier_policy == "exclude":
        excluded_outliers = int((~within_domain).sum())
        prepared = prepared.loc[within_domain].copy()

    prepared = prepared.sort_values("id [-]").reset_index(drop=True)
    prepared["id [-]"] = prepared["id [-]"].astype(int)
    prepared["N_p [-]"] = prepared["N_p [-]"].astype(int)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prepared.to_csv(output_path, index=False)

    report: dict[str, Any] = {
        "source": str(source_path),
        "source_sha256": sha256_file(source_path),
        "total_rows": total_rows,
        "eligible_rows": int(eligible.sum()),
        "removed_low_awcc_window_count": int((~enough_windows).sum()),
        "removed_missing_values": int((enough_windows & ~complete).sum()),
        "outside_application_domain": int((~within_domain).sum()),
        "excluded_by_outlier_policy": excluded_outliers,
        "outlier_policy": config.data.outlier_policy,
        "application_ranges": {
            name: {"minimum": limits[0], "maximum": limits[1]}
            for name, limits in APPLICATION_RANGES.items()
        },
        "feature_summary": {
            name: {
                "minimum": float(prepared[name].min()),
                "median": float(prepared[name].median()),
                "maximum": float(prepared[name].max()),
            }
            for name in MODEL_FEATURE_COLUMNS
        },
    }
    write_json(report_path, report)
    return report


def create_split(
    prepared_path: Path,
    output_path: Path,
    config: ProjectConfig,
) -> dict[str, Any]:
    """Create and persist a deterministic untouched holdout split."""
    frame = pd.read_csv(prepared_path)
    require_columns(frame, ("id [-]",), context="Prepared training data")
    identifiers = frame["id [-]"].astype(int).to_numpy()
    train_ids, test_ids = train_test_split(
        identifiers,
        test_size=config.split.test_fraction,
        random_state=config.split.random_seed,
        shuffle=True,
    )
    split = {
        "dataset_sha256": sha256_file(prepared_path),
        "random_seed": config.split.random_seed,
        "test_fraction": config.split.test_fraction,
        "train_ids": sorted(int(item) for item in train_ids),
        "test_ids": sorted(int(item) for item in test_ids),
    }
    write_json(output_path, split)
    return split


def _split_frames(frame: pd.DataFrame, split_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    split = read_json(split_path)
    train_ids = {int(item) for item in split["train_ids"]}
    test_ids = {int(item) for item in split["test_ids"]}
    identifiers = set(frame["id [-]"].astype(int))
    if train_ids & test_ids:
        raise DataValidationError("Training and test identifiers overlap.")
    if train_ids | test_ids != identifiers:
        raise DataValidationError("Persisted split identifiers do not match the prepared dataset.")
    train = frame.loc[frame["id [-]"].isin(train_ids)].copy()
    test = frame.loc[frame["id [-]"].isin(test_ids)].copy()
    return train, test


def _negative_median_rmse(
    estimator: RandomForestQuantileRegressor,
    features: npt.NDArray[np.float64],
    target: npt.NDArray[np.float64],
) -> float:
    prediction = np.asarray(estimator.predict(features, quantiles=[0.5])).reshape(-1)
    return -float(mean_squared_error(target, prediction) ** 0.5)


def tune_models(
    prepared_path: Path,
    split_path: Path,
    output_path: Path,
    config: ProjectConfig,
) -> dict[str, Any]:
    """Tune both targets using only the training partition and five-fold CV."""
    frame = pd.read_csv(prepared_path)
    train, _ = _split_frames(frame, split_path)
    features = train.loc[:, MODEL_FEATURE_COLUMNS].to_numpy(dtype=float)
    cross_validation = KFold(
        n_splits=config.split.cross_validation_folds,
        shuffle=True,
        random_state=config.split.random_seed,
    )
    distribution: dict[str, Any] = {
        "n_estimators": randint(50, 501),
        "max_depth": [6, 8, 10, 12, 16, 20, 30, None],
        "max_features": uniform(0.4, 0.6),
    }
    results: dict[str, Any] = {
        "method": "RandomizedSearchCV",
        "iterations": config.search.iterations,
        "cross_validation_folds": config.split.cross_validation_folds,
        "random_seed": config.split.random_seed,
        "training_rows": len(train),
        "targets": {},
    }

    for target_key, target_column in TARGET_COLUMNS.items():
        estimator = RandomForestQuantileRegressor(
            random_state=config.model.random_seed,
            n_jobs=config.model.n_jobs,
        )
        search = RandomizedSearchCV(
            estimator,
            param_distributions=distribution,
            n_iter=config.search.iterations,
            scoring=_negative_median_rmse,
            cv=cross_validation,
            random_state=config.split.random_seed,
            n_jobs=config.search.n_jobs,
            refit=True,
            return_train_score=True,
        )
        search.fit(features, train[target_column].to_numpy(dtype=float))
        best = {
            name: item.item() if isinstance(item, np.generic) else item
            for name, item in search.best_params_.items()
        }
        results["targets"][target_key] = {
            "target_column": target_column,
            "best_parameters": best,
            "best_cv_rmse": -float(search.best_score_),
        }

    write_json(output_path, results)
    return results


def parameters_for_target(
    target_key: str,
    config: ProjectConfig,
    tuning_path: Path | None,
) -> dict[str, Any]:
    """Resolve estimator parameters from tuned output or fixed configuration."""
    parameters: dict[str, Any] = {
        "random_state": config.model.random_seed,
        "n_estimators": config.model.n_estimators,
        "max_depth": config.model.max_depth,
        "max_features": config.model.max_features,
        "n_jobs": config.model.n_jobs,
    }
    if tuning_path is not None and tuning_path.exists():
        tuning = read_json(tuning_path)
        target_record = tuning.get("targets", {}).get(target_key, {})
        if isinstance(target_record, dict):
            best = target_record.get("best_parameters", {})
            if isinstance(best, dict):
                parameters.update(best)
    return parameters


def train_release_models(
    prepared_path: Path,
    output_directory: Path,
    metadata_path: Path,
    config: ProjectConfig,
    *,
    tuning_path: Path | None = None,
) -> dict[str, Any]:
    """Train deployment models on all eligible data and record provenance."""
    frame = pd.read_csv(prepared_path)
    features = frame.loc[:, MODEL_FEATURE_COLUMNS].to_numpy(dtype=float)
    output_directory.mkdir(parents=True, exist_ok=True)
    model_records: dict[str, Any] = {}

    for target_key, target_column in TARGET_COLUMNS.items():
        parameters = parameters_for_target(target_key, config, tuning_path)
        model = RandomForestQuantileRegressor(**parameters)
        model.fit(features, frame[target_column].to_numpy(dtype=float))
        model_path = output_directory / MODEL_FILENAMES[target_key]
        joblib.dump(model, model_path)
        model_records[model_path.name] = {
            "sha256": sha256_file(model_path),
            "target": target_column,
            "parameters": parameters,
        }

    packages = [
        "joblib",
        "numpy",
        "pandas",
        "quantile-forest",
        "scikit-learn",
        "scipy",
    ]
    metadata: dict[str, Any] = {
        "schema_version": 1,
        "source_commit": os.environ.get("PDP_UQ_COMMIT", "recorded-by-dvc"),
        "training_data": {
            "path": str(prepared_path),
            "sha256": sha256_file(prepared_path),
            "rows": len(frame),
        },
        "features": list(MODEL_FEATURE_COLUMNS),
        "models": model_records,
        "environment": {package: importlib.metadata.version(package) for package in packages},
    }
    write_json(metadata_path, metadata)
    return metadata


def record_existing_models(
    model_directory: Path,
    output_path: Path,
    *,
    training_data_path: Path,
) -> dict[str, Any]:
    """Create provenance checksums for trusted released artifacts without loading them."""
    models: dict[str, Any] = {}
    for target_key, filename in MODEL_FILENAMES.items():
        model_path = model_directory / filename
        if not model_path.is_file():
            raise FileNotFoundError(model_path)
        models[filename] = {
            "sha256": sha256_file(model_path),
            "target": TARGET_COLUMNS[target_key],
            "provenance": "released with upstream commit d788999",
        }
    metadata = {
        "schema_version": 1,
        "source_commit": "d788999",
        "training_data": {
            "path": str(training_data_path),
            "sha256": sha256_file(training_data_path),
        },
        "features": list(MODEL_FEATURE_COLUMNS),
        "models": models,
        "environment": {
            "scikit-learn": "1.6.0",
            "quantile-forest": "1.3.11",
        },
    }
    write_json(output_path, metadata)
    return metadata
