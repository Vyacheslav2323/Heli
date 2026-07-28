"""Wind configuration helpers for pid simulation scripts."""

from __future__ import annotations

import argparse
import math

import numpy as np

from skills.training.envs import HeliStabilization


def add_wind_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--wind-spd",
        type=float,
        default=None,
        help="Mean wind speed [ft/s]. Default: value from aw109.yaml.",
    )
    parser.add_argument(
        "--wind-dir",
        type=float,
        default=None,
        help="Mean wind direction [deg]. Default: value from aw109.yaml.",
    )
    parser.add_argument(
        "--turb-lvl",
        type=int,
        default=None,
        help="Turbulence level 1-7. Default: value from aw109.yaml.",
    )


def configure_wind(
    env: HeliStabilization,
    *,
    wind_spd: float | None = None,
    wind_dir: float | None = None,
    turb_lvl: int | None = None,
) -> None:
    """Apply wind overrides before the first env.reset()."""
    wind_dyn = env.wind_dyn
    if wind_spd is not None:
        wind_dyn.wind_speed = float(wind_spd)
    if wind_dir is not None:
        wind_dyn.wind_dir = math.radians(float(wind_dir))
    if turb_lvl is not None:
        wind_dyn.turbulence_level = int(turb_lvl)

    wind_dyn.wind_mean_ned = wind_dyn.wind_speed * np.array(
        [np.cos(wind_dyn.wind_dir), np.sin(wind_dyn.wind_dir), 0.0],
        dtype=np.float32,
    )
    env.heli_dyn.set_wind(wind_dyn.wind_mean_ned)


def describe_wind(env: HeliStabilization) -> str:
    wind_dyn = env.wind_dyn
    return (
        f"wind_spd={wind_dyn.wind_speed:.1f} ft/s, "
        f"wind_dir={math.degrees(wind_dyn.wind_dir):.1f} deg, "
        f"turb_lvl={wind_dyn.turbulence_level}"
    )
