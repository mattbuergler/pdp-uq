"""DataFrame schema and physical-domain checks."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from pdp_uq.constants import APPLICATION_RANGES, INPUT_COLUMNS, MODEL_FEATURE_COLUMNS
from pdp_uq.exceptions import DataValidationError


def require_columns(frame: pd.DataFrame, columns: Iterable[str], *, context: str) -> None:
    """Require a collection of named DataFrame columns."""
    missing = [name for name in columns if name not in frame.columns]
    if missing:
        raise DataValidationError(f"{context} is missing required columns: {', '.join(missing)}.")


def numeric_frame(
    frame: pd.DataFrame,
    columns: Iterable[str],
    *,
    context: str,
    allow_missing: bool = True,
) -> pd.DataFrame:
    """Return a copy with required columns validated as numeric."""
    names = tuple(columns)
    require_columns(frame, names, context=context)
    result = frame.copy()
    for name in names:
        converted = pd.to_numeric(result[name], errors="coerce")
        introduced = converted.isna() & result[name].notna()
        if introduced.any():
            rows = ", ".join(str(item) for item in result.index[introduced][:5])
            raise DataValidationError(
                f"{context} column {name!r} contains non-numeric values at rows {rows}."
            )
        result[name] = converted

    values = result.loc[:, names].to_numpy(dtype=float)
    if np.isinf(values).any():
        raise DataValidationError(f"{context} contains infinite values.")
    if not allow_missing and np.isnan(values).any():
        raise DataValidationError(f"{context} contains missing numeric values.")
    return result


def prepare_measurement_features(
    frame: pd.DataFrame,
    *,
    delta_x: float,
    delta_y: float,
    particles_per_window: int,
) -> pd.DataFrame:
    """Validate measurements and derive the model feature table."""
    result = numeric_frame(frame, INPUT_COLUMNS, context="Measurement data")
    velocity = result["u [m/s]"].to_numpy(dtype=float)
    root_mean_square = result["u_rms [m/s]"].to_numpy(dtype=float)
    turbulence = np.divide(
        root_mean_square,
        velocity,
        out=np.full_like(root_mean_square, np.nan),
        where=velocity != 0.0,
    )
    result["T_u [-]"] = turbulence
    result["delta_x [m]"] = float(delta_x)
    result["delta_y [m]"] = float(delta_y)
    result["N_p [-]"] = int(particles_per_window)
    return result


def application_domain(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Return a validity mask and semicolon-separated domain failure reasons."""
    require_columns(frame, MODEL_FEATURE_COLUMNS, context="Model features")
    valid = pd.Series(True, index=frame.index, dtype=bool)
    reasons = pd.Series("", index=frame.index, dtype="object")

    for column, (minimum, maximum) in APPLICATION_RANGES.items():
        values = pd.to_numeric(frame[column], errors="coerce")
        failed = values.isna() | ~values.between(minimum, maximum, inclusive="both")
        valid &= ~failed
        reason = f"{column} outside [{minimum:g}, {maximum:g}]"
        current = reasons.loc[failed].astype(str)
        reasons.loc[failed] = np.where(current == "", reason, current + "; " + reason)

    return valid, reasons
