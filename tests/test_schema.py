from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pdp_uq.exceptions import DataValidationError
from pdp_uq.schema import application_domain, prepare_measurement_features


def test_prepares_features_and_domain(measurements: pd.DataFrame) -> None:
    prepared = prepare_measurement_features(
        measurements,
        delta_x=0.004,
        delta_y=0.001,
        particles_per_window=10,
    )
    assert prepared["T_u [-]"].tolist() == pytest.approx([0.1, 0.2])
    valid, reasons = application_domain(prepared)
    assert valid.all()
    assert not reasons.any()


def test_zero_velocity_is_outside_domain(measurements: pd.DataFrame) -> None:
    measurements.loc[0, "u [m/s]"] = 0.0
    prepared = prepare_measurement_features(
        measurements,
        delta_x=0.004,
        delta_y=0.001,
        particles_per_window=10,
    )
    assert np.isnan(prepared.loc[0, "T_u [-]"])
    valid, reasons = application_domain(prepared)
    assert not valid.iloc[0]
    assert "u [m/s]" in reasons.iloc[0]


def test_rejects_missing_or_non_numeric_columns(measurements: pd.DataFrame) -> None:
    with pytest.raises(DataValidationError, match="missing required"):
        prepare_measurement_features(
            measurements.drop(columns="c [-]"),
            delta_x=0.004,
            delta_y=0.001,
            particles_per_window=10,
        )

    measurements["c [-]"] = measurements["c [-]"].astype(object)
    measurements.loc[0, "c [-]"] = "invalid"
    with pytest.raises(DataValidationError, match="non-numeric"):
        prepare_measurement_features(
            measurements,
            delta_x=0.004,
            delta_y=0.001,
            particles_per_window=10,
        )
