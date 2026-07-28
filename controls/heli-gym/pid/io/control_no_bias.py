"""Load and save control_effects_no_bias.csv."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from pid.constants import (
    ACCEL_NAMES,
    ANGULAR_ACCEL_NAMES,
    CONTROL_REGRESSORS,
)
from pid.models.angular import AngularDynamicsModel
from pid.models.full_control import FullControlEffectsModel
from pid.models.vertical import VerticalControlEffects


def load_full_control_effects(csv_path: Path, mass: float) -> FullControlEffectsModel:
    """Load full B matrix (controls x outputs) for AV = B.T @ u inversion."""
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = {row["term"]: row for row in csv.DictReader(handle) if row.get("term")}

    missing = [name for name in CONTROL_REGRESSORS if name not in rows]
    if missing:
        raise ValueError(f"Missing regressors in {csv_path}: {missing}")

    B = np.array(
        [
            [float(rows[regressor][name]) for name in ACCEL_NAMES]
            for regressor in CONTROL_REGRESSORS
        ],
        dtype=np.float64,
    )
    return FullControlEffectsModel(B=B, mass=mass)


def load_angular_model(csv_path: Path) -> AngularDynamicsModel:
    """Load control matrix B_ctrl from control_effects_no_bias.csv (zero bias)."""
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = {row["term"]: row for row in csv.DictReader(handle) if row.get("term")}

    missing = [name for name in CONTROL_REGRESSORS if name not in rows]
    if missing:
        raise ValueError(f"Missing regressors in {csv_path}: {missing}")

    B_ctrl = np.array(
        [
            [float(rows[regressor][name]) for name in ANGULAR_ACCEL_NAMES]
            for regressor in CONTROL_REGRESSORS
        ],
        dtype=np.float64,
    )
    return AngularDynamicsModel(b=np.zeros(len(ANGULAR_ACCEL_NAMES)), B_ctrl=B_ctrl)


def load_attitude_inversion_model(csv_path: Path) -> AngularDynamicsModel:
    """Load lon/lat/pedal control effects for attitude-rate inversion."""
    full_model = load_angular_model(csv_path)
    return AngularDynamicsModel(b=full_model.b, B_ctrl=full_model.B_ctrl[1:, :])


def load_vertical_effects(csv_path: Path, mass: float) -> VerticalControlEffects:
    """Load vertical force couplings from no-bias control effects CSV."""
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = {row["term"]: row for row in csv.DictReader(handle) if row.get("term")}
    missing = [name for name in CONTROL_REGRESSORS if name not in rows]
    if missing:
        raise ValueError(f"Missing regressors in {csv_path}: {missing}")
    if "vertical_acceleration" not in rows["col_t"]:
        raise ValueError(
            "control_effects_no_bias CSV is missing 'vertical_acceleration'. "
            "Re-run: python -m pid.control_effects_no_bias_estimation"
        )
    return VerticalControlEffects(
        mass=mass,
        bias=0.0,
        k_col=float(rows["col_t"]["vertical_acceleration"]),
        k_lon=float(rows["cyc_lon_t"]["vertical_acceleration"]),
        k_lat=float(rows["cyc_lat_t"]["vertical_acceleration"]),
        k_pedal=float(rows["pedal_t"]["vertical_acceleration"]),
    )


def save_control_effects_no_bias_csv(B: np.ndarray, r2: np.ndarray, csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["term", *ACCEL_NAMES])
        for regressor_name, row in zip(CONTROL_REGRESSORS, B):
            writer.writerow([regressor_name, *[float(value) for value in row]])
        writer.writerow([])
        writer.writerow(["metric", *ACCEL_NAMES])
        writer.writerow(["r2", *[float(value) for value in r2]])

