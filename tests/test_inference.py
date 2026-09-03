from __future__ import annotations

import numpy as np
import pandas as pd

from pdp_uq.inference import PredictionConfig, predict_dataframe


class FeatureModel:
    def __init__(self, feature_index: int) -> None:
        self.feature_index = feature_index

    def predict(self, features: np.ndarray, *, quantiles: list[float]) -> np.ndarray:
        center = features[:, self.feature_index]
        offsets = np.asarray(quantiles) - 0.5
        return center[:, None] + offsets[None, :]


def test_predict_dataframe_appends_quantiles_and_flags(measurements: pd.DataFrame) -> None:
    result = predict_dataframe(
        measurements,
        PredictionConfig(0.004, 0.001, 10, (0.05, 0.5, 0.95)),
        velocity_model=FeatureModel(0),
        turbulence_model=FeatureModel(1),
    )
    assert result["within_model_domain"].all()
    assert result["u_corrected_q0.5 [m/s]"].tolist() == [5.0, 10.0]
    assert result["T_u_corrected_q0.5 [-]"].tolist() == [0.1, 0.2]


def test_predict_dataframe_does_not_send_invalid_rows_to_model(
    measurements: pd.DataFrame,
) -> None:
    measurements.loc[0, "u [m/s]"] = 100.0
    result = predict_dataframe(
        measurements,
        PredictionConfig(0.004, 0.001, 10, (0.05, 0.5, 0.95)),
        velocity_model=FeatureModel(0),
        turbulence_model=FeatureModel(1),
    )
    assert not result.loc[0, "within_model_domain"]
    assert np.isnan(result.loc[0, "u_corrected_q0.5 [m/s]"])
    assert result.loc[1, "u_corrected_q0.5 [m/s]"] == 10.0
