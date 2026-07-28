"""Shared constants for the pid package."""

from __future__ import annotations

from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent / "results"
CONTROL_EFFECTS_CSV = RESULTS_DIR / "control_effects.csv"
CONTROL_EFFECTS_NO_BIAS_CSV = RESULTS_DIR / "control_effects_no_bias.csv"
PREDICTIVE_DEMOS_PATH = RESULTS_DIR / "demos_predictive.npz"
PID_SWEEP_CSV = RESULTS_DIR / "pid_sweep.csv"
POSITION_SWEEP_CSV = RESULTS_DIR / "position_sweep.csv"
POSITION_OPTUNA_DB = RESULTS_DIR / "position_optuna.db"
POSITION_OPTUNA_BEST_JSON = RESULTS_DIR / "position_optuna_best.json"

BIAS_NAME = "bias"
CONTROL_REGRESSORS = ["col_t", "cyc_lon_t", "cyc_lat_t", "pedal_t"]
CONTROL_EFFECTS_REGRESSORS = [BIAS_NAME, *CONTROL_REGRESSORS]
ANGULAR_ACCEL_NAMES = [
    "roll_acceleration",
    "pitch_acceleration",
    "yaw_acceleration",
]
ACCEL_NAMES = [*ANGULAR_ACCEL_NAMES, "vertical_acceleration"]

DEFAULT_N_SWEEP = 50
DEFAULT_EFFECT_SWEEP_LIMIT = 0.2
RATE_PID_OUTPUT_LIMITS = (-10.0, 10.0)

DEFAULT_KP_VALUES = [0.1, 0.3, 0.5]
DEFAULT_KI_VALUES = [0.0, 0.01, 0.05]
DEFAULT_KD_VALUES = [0.0, 0.05, 0.1]

DEFAULT_POS_KP_VALUES = [0.01, 0.05, 0.1]
DEFAULT_POS_KI_VALUES = [0.0, 0.01]
DEFAULT_POS_KD_VALUES = [0.0, 0.5, 1.0]
DEFAULT_VEL_KP_VALUES = [0.1, 0.3, 0.5]
DEFAULT_VEL_KI_VALUES = [0.0, 0.01]
DEFAULT_VEL_KD_VALUES = [0.0, 0.1, 0.5]

VEL_OUTPUT_LIMITS = (-20.0, 20.0)
ACCEL_OUTPUT_LIMITS = (-10.0, 10.0)
DEFAULT_INVERT_DT_STEPS = 5
