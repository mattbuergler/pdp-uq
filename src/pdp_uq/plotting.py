"""Portfolio-quality plots for corrected measurement profiles."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

from pdp_uq.schema import require_columns

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROFILE_COLUMNS = (
    "z [m]",
    "u [m/s]",
    "T_u [-]",
    "u_corrected_q0.05 [m/s]",
    "u_corrected_q0.25 [m/s]",
    "u_corrected_q0.5 [m/s]",
    "u_corrected_q0.75 [m/s]",
    "u_corrected_q0.95 [m/s]",
    "T_u_corrected_q0.05 [-]",
    "T_u_corrected_q0.25 [-]",
    "T_u_corrected_q0.5 [-]",
    "T_u_corrected_q0.75 [-]",
    "T_u_corrected_q0.95 [-]",
)


def plot_profiles(frame: pd.DataFrame, output_path: Path) -> Path:
    """Plot corrected medians and predictive intervals over a profile."""
    require_columns(frame, PROFILE_COLUMNS, context="Corrected profile data")
    figure, axes = plt.subplots(1, 2, figsize=(9.5, 4.0), sharey=True)
    height = frame["z [m]"].to_numpy(dtype=float)
    definitions = (
        {
            "axis": axes[0],
            "baseline": "u [m/s]",
            "median": "u_corrected_q0.5 [m/s]",
            "lower_50": "u_corrected_q0.25 [m/s]",
            "upper_50": "u_corrected_q0.75 [m/s]",
            "lower_90": "u_corrected_q0.05 [m/s]",
            "upper_90": "u_corrected_q0.95 [m/s]",
            "label": r"Mean velocity [m s$^{-1}$]",
        },
        {
            "axis": axes[1],
            "baseline": "T_u [-]",
            "median": "T_u_corrected_q0.5 [-]",
            "lower_50": "T_u_corrected_q0.25 [-]",
            "upper_50": "T_u_corrected_q0.75 [-]",
            "lower_90": "T_u_corrected_q0.05 [-]",
            "upper_90": "T_u_corrected_q0.95 [-]",
            "label": "Turbulence intensity [-]",
        },
    )
    for definition in definitions:
        axis = definition["axis"]
        assert hasattr(axis, "plot")
        axis.fill_betweenx(
            height,
            frame[str(definition["lower_90"])],
            frame[str(definition["upper_90"])],
            color="#5d80d6",
            alpha=0.16,
            label="90% predictive interval",
        )
        axis.fill_betweenx(
            height,
            frame[str(definition["lower_50"])],
            frame[str(definition["upper_50"])],
            color="#5d80d6",
            alpha=0.32,
            label="50% predictive interval",
        )
        axis.plot(
            frame[str(definition["baseline"])],
            height,
            color="#2b2b2b",
            marker="o",
            markersize=3,
            label="AWCC",
        )
        axis.plot(
            frame[str(definition["median"])],
            height,
            color="#2457bd",
            linestyle="--",
            marker="o",
            markersize=3,
            label="Corrected median",
        )
        axis.set_xlabel(str(definition["label"]))
        axis.grid(alpha=0.25)
        axis.set_xlim(left=0)
        axis.set_ylim(0, 1.05 * float(np.nanmax(height)))

    axes[0].set_ylabel("Invert-normal distance [m]")
    axes[1].legend(frameon=False, loc="best")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return output_path


def plot_profile_file(input_path: Path, output_path: Path | None = None) -> Path:
    """Load corrected CSV data and save the standard profile figure."""
    destination = output_path or input_path.with_name("profiles_uq.png")
    return plot_profiles(pd.read_csv(input_path), destination)
