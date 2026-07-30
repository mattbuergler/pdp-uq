from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def measurements() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "z [m]": [0.01, 0.02],
            "u [m/s]": [5.0, 10.0],
            "u_rms [m/s]": [0.5, 2.0],
            "c [-]": [0.1, 0.2],
            "d_32a [m]": [0.005, 0.01],
        }
    )


@pytest.fixture
def tiny_training_frame() -> pd.DataFrame:
    rows = 30
    velocity = np.linspace(2.0, 45.0, rows)
    turbulence = np.linspace(0.02, 0.3, rows)
    return pd.DataFrame(
        {
            "id [-]": np.arange(rows),
            "n_awcc [-]": np.full(rows, 500),
            "u_x_awcc [m/s]": velocity * 1.05,
            "T_ux_awcc [-]": turbulence * 0.85,
            "c_real [-]": np.linspace(0.02, 0.35, rows),
            "d_bx_real [m]": np.linspace(0.001, 0.015, rows),
            "delta_x [m]": np.linspace(0.001, 0.008, rows),
            "delta_y [m]": np.linspace(0.0001, 0.0015, rows),
            "N_p [-]": np.resize(np.arange(5, 20), rows),
            "u_x_real [m/s]": velocity,
            "T_ux_real [-]": turbulence,
        }
    )
