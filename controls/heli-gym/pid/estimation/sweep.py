"""Sweep controls and collect acceleration responses for OLS fitting."""

from __future__ import annotations

import numpy as np

from heligym.envs.dynamics.kinematic import euler_to_rotmat
from heligym.envs.dynamics.utils import cross_product
from pid.constants import CONTROL_REGRESSORS
from skills.training.envs import HeliStabilization


def _current_up_accel(env: HeliStabilization) -> float:
    """Upward acceleration in ft/s^2 from dynamics state derivatives."""
    state = env.heli_dyn.state
    dots = env.heli_dyn.state_dots
    body_acc = dots["uvw"] + cross_product(state["pqr"], state["uvw"])
    earth_acc = euler_to_rotmat(state["euler"]).T @ body_acc
    return float(-earth_acc[2])


def collect_control_effects_data(
    env: HeliStabilization,
    *,
    n_sweep: int,
    action_limit: float,
    effect_sweep_limit: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Sweep each control individually and record local accel responses around trim."""
    sweep_values = np.linspace(-effect_sweep_limit, effect_sweep_limit, n_sweep)
    rows_x: list[np.ndarray] = []
    rows_y: list[np.ndarray] = []
    mass = float(env.heli_dyn.HELI["M"])

    for control_idx, _control_name in enumerate(CONTROL_REGRESSORS):
        for sweep_idx, value in enumerate(sweep_values):
            sample_seed = seed + control_idx * n_sweep + sweep_idx

            env.reset(seed=sample_seed)
            trim_action = env.heli_dyn.action.astype(np.float32).copy()
            env.step(trim_action)
            a_up_trim = _current_up_accel(env)

            env.reset(seed=sample_seed)
            trim_action = env.heli_dyn.action.astype(np.float32).copy()
            requested_delta = np.zeros(4, dtype=np.float32)
            requested_delta[control_idx] = float(value)
            action = np.clip(trim_action + requested_delta, -action_limit, action_limit)
            applied_delta = (action - trim_action).astype(np.float64)

            env.step(action)
            ang_accel = env.heli_dyn.state_dots["pqr"].astype(np.float64)
            a_up_delta = _current_up_accel(env)
            delta_a_up = a_up_delta - a_up_trim
            delta_f_up = delta_a_up * mass
            accel = np.concatenate([ang_accel, np.array([delta_f_up], dtype=np.float64)])
            rows_x.append(applied_delta)
            rows_y.append(accel)

    return np.vstack(rows_x), np.vstack(rows_y)
