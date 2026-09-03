"""Point and probabilistic evaluation for released quantile models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import numpy.typing as npt
import pandas as pd
from quantile_forest import RandomForestQuantileRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_pinball_loss,
    mean_squared_error,
    r2_score,
)

from pdp_uq.config import ProjectConfig
from pdp_uq.constants import (
    BASELINE_COLUMNS,
    MODEL_FEATURE_COLUMNS,
    TARGET_COLUMNS,
)
from pdp_uq.io import sha256_file, write_json
from pdp_uq.training import _split_frames, parameters_for_target

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _point_metrics(
    target: npt.NDArray[np.float64],
    prediction: npt.NDArray[np.float64],
) -> dict[str, float]:
    return {
        "rmse": float(mean_squared_error(target, prediction) ** 0.5),
        "mae": float(mean_absolute_error(target, prediction)),
        "r2": float(r2_score(target, prediction)),
    }


def _probabilistic_metrics(
    target: npt.NDArray[np.float64],
    predictions: npt.NDArray[np.float64],
    quantiles: tuple[float, ...],
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "mean_pinball_loss": {
            f"{quantile:g}": float(mean_pinball_loss(target, predictions[:, index], alpha=quantile))
            for index, quantile in enumerate(quantiles)
        },
        "quantile_crossing_rate": float(np.mean(np.any(np.diff(predictions, axis=1) < 0, axis=1))),
        "intervals": {},
    }
    index_by_quantile = {round(item, 10): index for index, item in enumerate(quantiles)}
    for lower in quantiles:
        upper = round(1.0 - lower, 10)
        if lower >= 0.5 or upper not in index_by_quantile:
            continue
        lower_values = predictions[:, index_by_quantile[round(lower, 10)]]
        upper_values = predictions[:, index_by_quantile[upper]]
        nominal = upper - lower
        result["intervals"][f"{nominal:.3f}"] = {
            "lower_quantile": lower,
            "upper_quantile": upper,
            "empirical_coverage": float(
                np.mean((target >= lower_values) & (target <= upper_values))
            ),
            "mean_width": float(np.mean(upper_values - lower_values)),
        }
    return result


def _plot_validation(
    values: dict[str, dict[str, npt.NDArray[np.float64]]],
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    specifications = {
        "mean_velocity": {
            "limits": (0.0, 50.0),
            "label": r"Mean velocity [m s$^{-1}$]",
            "title": "Mean velocity",
        },
        "turbulence_intensity": {
            "limits": (0.0, 0.4),
            "label": "Turbulence intensity [-]",
            "title": "Turbulence intensity",
        },
    }
    for axis, target_key in zip(axes, TARGET_COLUMNS, strict=True):
        record = values[target_key]
        spec = specifications[target_key]
        axis.scatter(
            record["target"],
            record["baseline"],
            alpha=0.18,
            s=8,
            color="#d95f5f",
            edgecolors="none",
            label="AWCC",
        )
        axis.scatter(
            record["target"],
            record["prediction"],
            alpha=0.22,
            s=8,
            color="#3567c8",
            edgecolors="none",
            label="Quantile forest median",
        )
        minimum, maximum = spec["limits"]
        axis.plot([minimum, maximum], [minimum, maximum], color="black", linewidth=1.2)
        axis.set(xlim=spec["limits"], ylim=spec["limits"], title=spec["title"])
        axis.set_xlabel(f"True {spec['label']}")
        axis.set_ylabel(f"Estimated {spec['label']}")
        axis.grid(alpha=0.25)
    axes[1].legend(frameon=False, loc="upper left")
    figure.suptitle("Untouched 20% holdout evaluation")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def evaluate_models(
    prepared_path: Path,
    split_path: Path,
    metrics_path: Path,
    figure_path: Path,
    config: ProjectConfig,
    *,
    tuning_path: Path | None = None,
) -> dict[str, Any]:
    """Fit on the training split and evaluate once on the untouched holdout."""
    frame = pd.read_csv(prepared_path)
    train, test = _split_frames(frame, split_path)
    train_features = train.loc[:, MODEL_FEATURE_COLUMNS].to_numpy(dtype=float)
    test_features = test.loc[:, MODEL_FEATURE_COLUMNS].to_numpy(dtype=float)
    quantiles = config.quantiles
    if 0.5 not in quantiles:
        raise ValueError("Evaluation requires the median quantile 0.5.")
    median_index = quantiles.index(0.5)

    metrics: dict[str, Any] = {
        "schema_version": 1,
        "dataset_sha256": sha256_file(prepared_path),
        "training_rows": len(train),
        "test_rows": len(test),
        "split_sha256": sha256_file(split_path),
        "quantiles": list(quantiles),
        "targets": {},
    }
    plot_values: dict[str, dict[str, npt.NDArray[np.float64]]] = {}

    for target_key, target_column in TARGET_COLUMNS.items():
        parameters = parameters_for_target(target_key, config, tuning_path)
        model = RandomForestQuantileRegressor(**parameters)
        model.fit(train_features, train[target_column].to_numpy(dtype=float))
        predictions = np.asarray(
            model.predict(test_features, quantiles=list(quantiles)), dtype=float
        ).reshape(len(test), len(quantiles))
        target = test[target_column].to_numpy(dtype=float)
        baseline = test[BASELINE_COLUMNS[target_key]].to_numpy(dtype=float)
        median = predictions[:, median_index]
        model_point = _point_metrics(target, median)
        baseline_point = _point_metrics(target, baseline)
        metrics["targets"][target_key] = {
            "target_column": target_column,
            "parameters": parameters,
            "model": model_point,
            "awcc_baseline": baseline_point,
            "rmse_reduction_fraction": 1.0 - model_point["rmse"] / baseline_point["rmse"],
            "probabilistic": _probabilistic_metrics(target, predictions, quantiles),
        }
        plot_values[target_key] = {
            "target": target,
            "baseline": baseline,
            "prediction": median,
        }

    write_json(metrics_path, metrics)
    _plot_validation(plot_values, figure_path)
    return metrics
