"""Names, units, and scientific application limits used by pdp-uq."""

from __future__ import annotations

from typing import Final

INPUT_COLUMNS: Final[tuple[str, ...]] = (
    "u [m/s]",
    "u_rms [m/s]",
    "c [-]",
    "d_32a [m]",
)

MODEL_FEATURE_COLUMNS: Final[tuple[str, ...]] = (
    "u [m/s]",
    "T_u [-]",
    "c [-]",
    "d_32a [m]",
    "delta_x [m]",
    "delta_y [m]",
    "N_p [-]",
)

TRAINING_COLUMN_MAP: Final[dict[str, str]] = {
    "u_x_awcc [m/s]": "u [m/s]",
    "T_ux_awcc [-]": "T_u [-]",
    "c_real [-]": "c [-]",
    "d_bx_real [m]": "d_32a [m]",
}

TARGET_COLUMNS: Final[dict[str, str]] = {
    "mean_velocity": "u_x_real [m/s]",
    "turbulence_intensity": "T_ux_real [-]",
}

BASELINE_COLUMNS: Final[dict[str, str]] = {
    "mean_velocity": "u [m/s]",
    "turbulence_intensity": "T_u [-]",
}

MODEL_FILENAMES: Final[dict[str, str]] = {
    "mean_velocity": "qrf_model_u_x.joblib",
    "turbulence_intensity": "qrf_model_T_ux.joblib",
}

APPLICATION_RANGES: Final[dict[str, tuple[float, float]]] = {
    "u [m/s]": (1.0, 50.0),
    "T_u [-]": (0.01, 0.35),
    "c [-]": (0.005, 0.4),
    "d_32a [m]": (0.0005, 0.02),
    "delta_x [m]": (0.0005, 0.01),
    "delta_y [m]": (0.0, 0.002),
    "N_p [-]": (5.0, 20.0),
}

DEFAULT_QUANTILES: Final[tuple[float, ...]] = (
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    0.75,
    0.9,
    0.95,
    0.975,
)
