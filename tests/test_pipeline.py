from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from pdp_uq.config import load_config
from pdp_uq.evaluation import evaluate_models
from pdp_uq.inference import PredictionConfig, predict_file
from pdp_uq.plotting import plot_profile_file
from pdp_uq.training import (
    create_split,
    prepare_training_data,
    record_existing_models,
    train_release_models,
    tune_models,
)


def test_complete_smoke_pipeline(
    tmp_path: Path,
    tiny_training_frame: pd.DataFrame,
    measurements: pd.DataFrame,
) -> None:
    config = load_config(Path("configs/smoke.toml"))
    source = tmp_path / "source.csv"
    prepared = tmp_path / "prepared.csv"
    data_report = tmp_path / "data_report.json"
    split = tmp_path / "split.json"
    tuning = tmp_path / "tuning.json"
    metrics = tmp_path / "metrics.json"
    validation_figure = tmp_path / "validation.png"
    model_directory = tmp_path / "models"
    metadata = model_directory / "model_metadata.json"

    tiny_training_frame.to_csv(source, index=False)
    prepare_training_data(source, prepared, data_report, config)
    create_split(prepared, split, config)
    tuning_result = tune_models(prepared, split, tuning, config)
    assert set(tuning_result["targets"]) == {"mean_velocity", "turbulence_intensity"}

    evaluation = evaluate_models(
        prepared,
        split,
        metrics,
        validation_figure,
        config,
        tuning_path=tuning,
    )
    assert validation_figure.stat().st_size > 0
    assert evaluation["test_rows"] == 6
    assert (
        0.0
        <= evaluation["targets"]["mean_velocity"]["probabilistic"]["quantile_crossing_rate"]
        <= 1.0
    )

    training = train_release_models(
        prepared,
        model_directory,
        metadata,
        config,
        tuning_path=tuning,
    )
    assert set(training["models"]) == {
        "qrf_model_u_x.joblib",
        "qrf_model_T_ux.joblib",
    }

    released_metadata = tmp_path / "released_metadata.json"
    record_existing_models(
        model_directory,
        released_metadata,
        training_data_path=prepared,
    )
    assert json.loads(released_metadata.read_text(encoding="utf-8"))["models"]

    input_path = tmp_path / "measurements.csv"
    output_path = tmp_path / "measurements_uq.csv"
    measurements.to_csv(input_path, index=False)
    predict_file(
        input_path,
        PredictionConfig(0.004, 0.001, 10),
        model_directory=model_directory,
        output_path=output_path,
        metadata_path=metadata,
    )
    corrected = pd.read_csv(output_path)
    assert corrected["within_model_domain"].all()

    profile = plot_profile_file(output_path)
    assert profile.stat().st_size > 0
