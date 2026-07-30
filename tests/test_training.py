from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from pdp_uq.config import load_config
from pdp_uq.training import create_split, prepare_training_data


def test_prepare_and_split_are_deterministic(
    tmp_path: Path,
    tiny_training_frame: pd.DataFrame,
) -> None:
    source = tmp_path / "source.csv"
    prepared = tmp_path / "prepared.csv"
    report = tmp_path / "report.json"
    split = tmp_path / "split.json"
    tiny_training_frame.to_csv(source, index=False)
    config = load_config(Path("configs/smoke.toml"))

    result = prepare_training_data(source, prepared, report, config)
    assert result["eligible_rows"] == len(tiny_training_frame)
    normalized = pd.read_csv(prepared)
    assert "u [m/s]" in normalized
    assert normalized["within_application_domain"].all()

    create_split(prepared, split, config)
    first = json.loads(split.read_text(encoding="utf-8"))
    create_split(prepared, split, config)
    second = json.loads(split.read_text(encoding="utf-8"))
    assert first == second
    assert len(first["train_ids"]) == 24
    assert len(first["test_ids"]) == 6
